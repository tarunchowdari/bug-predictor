# bug-predictor inference API
# Endpoints: POST /predict, POST /webhook/github, GET /predictions/recent,
#            GET /feed, GET /model/info, GET /analytics, GET /health

from __future__ import annotations

import hashlib
import hmac
import io
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api.kamei_computer import KameiComputer, get_repo_cache

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger("forge.api")

# Feature order must match training exactly
KAMEI_FEATURES = [
    "NS", "ND", "NF", "Entropy", "LA", "LD", "LT",
    "FIX", "NOD", "NUC", "AGE", "EXP", "REXP", "SEXP",
]
AST_FEATURES = [
    "n_nodes_added", "n_nodes_deleted", "n_control_flow_changes",
    "n_function_changes", "max_depth_before", "max_depth_after",
    "depth_delta", "unique_node_types_added",
]
ALL_FEATURES = KAMEI_FEATURES + AST_FEATURES  # 22-dim

EXPERIMENTS_DIR    = Path("experiments")
BASELINE_DIR       = EXPERIMENTS_DIR / "baseline"
SHAP_VALUES_PATH   = EXPERIMENTS_DIR / "shap_values.npy"
FEATURE_NAMES_PATH = EXPERIMENTS_DIR / "feature_names.json"

GITHUB_WEBHOOK_SECRET = os.getenv("GITHUB_WEBHOOK_SECRET", "")
EMBED_MODEL_NAME      = "sentence-transformers/all-MiniLM-L6-v2"

MAX_PREDICTIONS = 1000  # drop oldest when full
_predictions: list[dict[str, Any]] = []

_embed_model: Any = None


def _get_embed_model() -> Any:
    global _embed_model
    if _embed_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model %s …", EMBED_MODEL_NAME)
            _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
            logger.info("Embedding model ready.")
        except Exception as exc:
            logger.warning("Could not load embedding model: %s — embeddings disabled", exc)
            _embed_model = False
    return _embed_model if _embed_model is not False else None


class _ModelBundle:
    stage1_model:    Any
    stage2_model:    Any
    threshold:       float
    features:        list[str]
    metrics:         dict[str, Any]
    model_timestamp: str
    stage1_path:     Path
    stage2_path:     Path


_bundle: Optional[_ModelBundle] = None


def _load_latest_bundle() -> _ModelBundle:
    stage1_files = sorted(BASELINE_DIR.glob("stage1_*.pkl"))
    if not stage1_files:
        raise RuntimeError(f"No stage1 model in {BASELINE_DIR} — run train.py first")
    stage1_path = stage1_files[-1]

    stage2_files = sorted(BASELINE_DIR.glob("stage2_*.pkl"))
    stage2_path  = stage2_files[-1] if stage2_files else None

    metrics: dict[str, Any] = {}
    metrics_files = sorted(BASELINE_DIR.glob("metrics_*.json"))
    if metrics_files:
        with open(metrics_files[-1], encoding="utf-8") as fh:
            metrics = json.load(fh)

    s1 = joblib.load(str(stage1_path))
    s2 = joblib.load(str(stage2_path)) if stage2_path else None

    b = _ModelBundle()
    b.stage1_model    = s1["model"]
    b.threshold       = s1["threshold"]
    b.features        = s1["features"]
    b.stage2_model    = s2["model"] if s2 else None
    b.metrics         = metrics
    b.model_timestamp = stage1_path.stem.replace("stage1_", "")
    b.stage1_path     = stage1_path
    b.stage2_path     = stage2_path
    logger.info("Loaded Stage 1: %s  threshold=%.4f", stage1_path.name, b.threshold)
    if s2:
        logger.info("Loaded Stage 2: %s", stage2_path.name)
    return b


def _get_bundle() -> _ModelBundle:
    global _bundle
    if _bundle is None:
        _bundle = _load_latest_bundle()
    return _bundle


# Pydantic schemas

class CommitFeatures(BaseModel):
    # Kamei 14
    NS:      float = Field(0.0, ge=0)
    ND:      float = Field(0.0, ge=0)
    NF:      float = Field(0.0, ge=0)
    Entropy: float = Field(0.0, ge=0)
    LA:      float = Field(0.0, ge=0)
    LD:      float = Field(0.0, ge=0)
    LT:      float = Field(0.0, ge=0)
    FIX:     float = Field(0.0, ge=0, le=1)
    NOD:     float = Field(0.0, ge=0)
    NUC:     float = Field(0.0, ge=0)
    AGE:     float = Field(0.0, ge=0)
    EXP:     float = Field(0.0, ge=0)
    REXP:    float = Field(0.0, ge=0)
    SEXP:    float = Field(0.0, ge=0)
    # AST 8
    n_nodes_added:           float = Field(0.0, ge=0)
    n_nodes_deleted:         float = Field(0.0, ge=0)
    n_control_flow_changes:  float = Field(0.0, ge=0)
    n_function_changes:      float = Field(0.0, ge=0)
    max_depth_before:        float = Field(0.0, ge=0)
    max_depth_after:         float = Field(0.0, ge=0)
    depth_delta:             float = 0.0
    unique_node_types_added: float = Field(0.0, ge=0)
    # Metadata (not used for prediction)
    commit_sha:     Optional[str] = None
    repo:           Optional[str] = None
    author:         Optional[str] = None
    commit_message: Optional[str] = None


class PredictionResponse(BaseModel):
    commit_sha:     Optional[str]
    repo:           Optional[str]
    author:         Optional[str]
    commit_message: Optional[str]
    bug_prob:       float
    is_buggy:       bool
    risk_level:     str        # "High" | "Medium" | "Low"
    severity_score: Optional[float]
    threshold:      float
    timestamp:      str
    feature_shap:   Optional[dict[str, float]] = None


class ModelInfoResponse(BaseModel):
    model_timestamp:  str
    stage1_path:      str
    stage2_available: bool
    threshold:        float
    features:         list[str]
    metrics:          dict[str, Any]
    gates:            dict[str, Any]


def _features_to_array(cf: CommitFeatures) -> np.ndarray:
    return np.array(
        [getattr(cf, f, 0.0) for f in ALL_FEATURES],
        dtype=np.float32,
    ).reshape(1, -1)


def _risk_level(prob: float, threshold: float) -> str:
    if prob >= min(threshold + 0.25, 0.85):
        return "High"
    if prob >= threshold:
        return "Medium"
    return "Low"


def _run_prediction(
    cf: CommitFeatures,
    kamei_metrics: Optional[dict] = None,
) -> PredictionResponse:
    bundle   = _get_bundle()
    X        = _features_to_array(cf)
    bug_prob = float(bundle.stage1_model.predict_proba(X)[0, 1])
    is_buggy = bug_prob >= bundle.threshold
    risk     = _risk_level(bug_prob, bundle.threshold)

    severity: Optional[float] = None
    if is_buggy and bundle.stage2_model is not None:
        severity = float(np.clip(bundle.stage2_model.predict(X)[0], 0.0, 1.0))

    shap_dict: Optional[dict[str, float]] = None
    try:
        import shap
        explainer = shap.TreeExplainer(bundle.stage1_model)
        sv = explainer.shap_values(X)
        if isinstance(sv, list):
            sv = sv[1]
        shap_dict = {f: float(sv[0, i]) for i, f in enumerate(ALL_FEATURES)}
    except Exception:
        pass

    now  = datetime.now(timezone.utc).isoformat()
    resp = PredictionResponse(
        commit_sha     = cf.commit_sha,
        repo           = cf.repo,
        author         = cf.author,
        commit_message = cf.commit_message,
        bug_prob       = round(bug_prob, 4),
        is_buggy       = bool(is_buggy),
        risk_level     = risk,
        severity_score = round(severity, 4) if severity is not None else None,
        threshold      = round(bundle.threshold, 4),
        timestamp      = now,
        feature_shap   = {k: round(v, 5) for k, v in shap_dict.items()} if shap_dict else None,
    )

    record = resp.model_dump()
    if kamei_metrics is not None:
        record["kamei_metrics"] = kamei_metrics
    _predictions.append(record)
    if len(_predictions) > MAX_PREDICTIONS:
        del _predictions[:-MAX_PREDICTIONS]

    return resp


# App setup

app = FastAPI(
    title="bug-predictor API",
    description="XGBoost two-stage bug recurrence prediction service.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _startup():
    logger.info("Loading models …")
    _get_bundle()
    _get_embed_model()
    get_repo_cache()
    logger.info("bug-predictor API ready.")


# Routes

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(cf: CommitFeatures):
    """Predict bug recurrence risk for a single commit."""
    try:
        return _run_prediction(cf)
    except Exception as exc:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    """Return active model metadata and evaluation metrics."""
    bundle = _get_bundle()
    return ModelInfoResponse(
        model_timestamp  = bundle.model_timestamp,
        stage1_path      = str(bundle.stage1_path),
        stage2_available = bundle.stage2_model is not None,
        threshold        = round(bundle.threshold, 4),
        features         = bundle.features,
        metrics          = bundle.metrics.get("metrics", {}),
        gates            = bundle.metrics.get("gates", {}),
    )


@app.post("/webhook/github", tags=["Webhook"])
async def github_webhook(request: Request):
    """
    Receive a GitHub push event, compute real Kamei metrics from git history,
    and run predictions for every pushed commit.
    Set GITHUB_WEBHOOK_SECRET to enable HMAC validation.
    """
    body = await request.body()

    if GITHUB_WEBHOOK_SECRET:
        sig_header = request.headers.get("X-Hub-Signature-256", "")
        expected   = "sha256=" + hmac.new(
            GITHUB_WEBHOOK_SECRET.encode(), body, hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(sig_header, expected):
            raise HTTPException(status_code=401, detail="Invalid webhook signature")

    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON") from exc

    event = request.headers.get("X-GitHub-Event", "push")
    if event != "push":
        return {"status": "ignored", "event": event}

    repo_info  = payload.get("repository", {})
    clone_url  = repo_info.get("clone_url") or repo_info.get("url", "")
    repo_name  = repo_info.get("full_name", clone_url)
    commits    = payload.get("commits", [])

    git_repo       = None
    clone_warning: Optional[str] = None
    if clone_url:
        try:
            cache    = get_repo_cache()
            git_repo = cache.get_repo(clone_url)
        except Exception as exc:
            clone_warning = f"repo_unavailable: {exc}"
            logger.warning("Webhook: could not acquire repo %s: %s", clone_url, exc)

    results = []
    for commit in commits:
        commit_hash    = commit.get("id", "")
        commit_message = commit.get("message", "")[:300]
        author_name    = commit.get("author", {}).get("name")

        metrics_warning: Optional[str] = clone_warning
        if git_repo is not None and commit_hash:
            try:
                computer      = KameiComputer(git_repo)
                kamei         = computer.compute(commit_hash)
                compute_error = kamei.pop("_compute_error", None)
                if compute_error and not metrics_warning:
                    metrics_warning = f"metrics_unavailable: {compute_error}"
            except Exception as exc:
                kamei           = {k: 0.0 for k in ("NS","ND","NF","Entropy","LA","LD","LT","FIX","NOD","NUC","AGE","EXP","REXP","SEXP")}
                metrics_warning = f"metrics_unavailable: {exc}"
                logger.warning("Webhook: metric computation failed for %s: %s", commit_hash[:8], exc)
        else:
            # Best-effort fallback when we have no local clone
            added    = len(commit.get("added",    []))
            removed  = len(commit.get("removed",  []))
            modified = len(commit.get("modified", []))
            import re as _re
            is_fix = bool(_re.search(
                r"(fix|bug|defect|patch|error|fault|repair|correct|resolv)",
                commit_message, _re.IGNORECASE,
            ))
            kamei = {
                "NS": 0.0, "ND": 0.0, "NF": float(added + removed + modified),
                "Entropy": 0.0,
                "LA": float(added), "LD": float(removed), "LT": 0.0,
                "FIX": 1.0 if is_fix else 0.0,
                "NOD": 0.0, "NUC": 0.0, "AGE": 0.0,
                "EXP": 0.0, "REXP": 0.0, "SEXP": 0.0,
            }
            if not metrics_warning:
                metrics_warning = "metrics_unavailable: no clone_url"

        cf   = CommitFeatures(
            **kamei,
            commit_sha     = commit_hash or None,
            repo           = repo_name,
            author         = author_name,
            commit_message = commit_message,
        )
        pred   = _run_prediction(cf, kamei_metrics=kamei)
        result = pred.model_dump()
        result["kamei_metrics"] = kamei
        if metrics_warning:
            result["warning"] = metrics_warning
        results.append(result)

    return {"processed": len(results), "predictions": results}


@app.get("/predictions/recent", tags=["Feed"])
async def get_predictions_recent(limit: int = 500):
    """Return up to `limit` most recent predictions, newest first."""
    recent = list(reversed(_predictions))
    return {"count": len(recent), "predictions": recent[:limit]}


@app.get("/feed", tags=["Feed"])
async def get_feed(limit: int = 50):
    """Return recent predictions sorted by bug probability descending (legacy endpoint)."""
    feed = sorted(_predictions, key=lambda x: x["bug_prob"], reverse=True)
    return {"count": len(feed), "predictions": feed[:limit]}


@app.get("/feed/{commit_sha}", tags=["Feed"])
async def get_commit_detail(commit_sha: str):
    """Return full prediction detail for a specific commit SHA."""
    matches = [p for p in _predictions if p.get("commit_sha") == commit_sha]
    if not matches:
        raise HTTPException(status_code=404, detail=f"Commit {commit_sha!r} not found")
    return matches[-1]


@app.get("/analytics", tags=["Analytics"])
async def get_analytics():
    """Aggregated bug rate stats over all stored predictions."""
    if not _predictions:
        return {"total": 0, "buggy": 0, "bug_rate": 0.0, "by_risk": {}, "top_authors": []}

    df    = pd.DataFrame(_predictions)
    total = len(df)
    buggy = int(df["is_buggy"].sum())

    by_risk = df["risk_level"].value_counts().to_dict()

    top_authors: list[dict] = []
    if "author" in df.columns:
        author_df = (
            df[df["author"].notna()]
            .groupby("author")
            .agg(total=("bug_prob", "count"), bug_rate=("is_buggy", "mean"))
            .reset_index()
            .sort_values("bug_rate", ascending=False)
            .head(10)
        )
        top_authors = author_df.to_dict(orient="records")

    trends: list[dict] = []
    if "timestamp" in df.columns:
        df["ts"]   = pd.to_datetime(df["timestamp"])
        df["hour"] = df["ts"].dt.floor("h").dt.strftime("%Y-%m-%dT%H:00")
        trend_df   = (
            df.groupby("hour")
            .agg(total=("is_buggy", "count"), buggy=("is_buggy", "sum"))
            .reset_index()
        )
        trend_df["bug_rate"] = trend_df["buggy"] / trend_df["total"]
        trends = trend_df.to_dict(orient="records")

    return {
        "total":       total,
        "buggy":       buggy,
        "bug_rate":    round(buggy / total, 4) if total else 0.0,
        "by_risk":     by_risk,
        "top_authors": top_authors,
        "trends":      trends,
    }


@app.get("/shap/features", tags=["SHAP"])
async def get_shap_features():
    """Mean |SHAP| per feature from the last training run."""
    if not SHAP_VALUES_PATH.exists():
        raise HTTPException(status_code=404, detail=f"Run train.py first — {SHAP_VALUES_PATH} not found")
    sv    = np.load(str(SHAP_VALUES_PATH))
    names = ALL_FEATURES
    if FEATURE_NAMES_PATH.exists():
        with open(FEATURE_NAMES_PATH, encoding="utf-8") as fh:
            names = json.load(fh)
    mean_abs = np.abs(sv).mean(axis=0)
    ranked   = sorted(
        [{"feature": n, "importance": round(float(v), 5)} for n, v in zip(names, mean_abs)],
        key=lambda x: x["importance"], reverse=True,
    )
    return {"features": ranked}


@app.get("/health", tags=["Meta"])
async def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}
