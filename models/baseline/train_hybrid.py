"""
models/baseline/train_hybrid.py

Hybrid XGBoost: 22 tabular features + 768-dim BERT embeddings = 790 features.

The key idea: sentence-transformers already produce L2-normalised 768-dim
vectors that encode both the commit message and diff text semantics.
Feeding them directly into XGBoost (alongside the existing Kamei+AST
features) costs nothing architecturally and may lift recall on borderline
commits where the diff text is the clearest signal.

Feature vector layout:
  [0:22]   tabular (StandardScaler-normalised)
  [22:790] BERT sentence-transformer embedding (already normalised by ST)

Stage 2 (severity) uses the same 790-dim vector — the regressor may find
useful signal in the embedding dims too.

SHAP:
  Computed on a 1000-row test sample to keep runtime reasonable.
  Only the top 22 tabular features are visualised — embedding dim names
  are omitted from the waterfall (too many to show meaningfully).

Promotion gates vs XGBoost tabular-only baseline (test = kamei/platform):
  AUC-ROC >= 0.780
  F1      >= 0.640
  Recall  >= 0.900

Usage:
  python models/baseline/train_hybrid.py
  python models/baseline/train_hybrid.py --n-trials 5   # smoke test
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import pickle
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_recall_curve,
    precision_score, recall_score, mean_absolute_error, mean_squared_error,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore", category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# -- Logging -----------------------------------------------------------------

def _configure_logging(level: str) -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        level=getattr(logging, level.upper(), logging.INFO),
        stream=sys.stdout,
    )


logger = logging.getLogger("forge.hybrid")

# -- Constants ---------------------------------------------------------------

SEED = 42
ROOT = Path(__file__).resolve().parents[2]
EMB_DIR = ROOT / "data" / "processed" / "embeddings"

KAMEI_FEATURES: list[str] = [
    "NS", "ND", "NF", "Entropy", "LA", "LD", "LT",
    "FIX", "NOD", "NUC", "AGE", "EXP", "REXP", "SEXP",
]
AST_FEATURES: list[str] = [
    "n_nodes_added", "n_nodes_deleted", "n_control_flow_changes",
    "n_function_changes", "max_depth_before", "max_depth_after",
    "depth_delta", "unique_node_types_added",
]
TABULAR_FEATURES: list[str] = KAMEI_FEATURES + AST_FEATURES   # 22-dim
EMBED_DIM = 768
N_FEATURES = len(TABULAR_FEATURES) + EMBED_DIM   # 790

# XGBoost tabular-only baseline numbers (kamei/platform test set)
BASELINE = {
    "auc":       0.763,
    "f1":        0.628,
    "precision": 0.451,
    "recall":    0.932,
}

# Promotion gates
PROMOTION_GATES = {
    "auc":    0.780,
    "f1":     0.640,
    "recall": 0.900,
}

MLFLOW_EXPERIMENT = "hybrid_v1"

# -- Embedding loader --------------------------------------------------------

def _load_embeddings(hashes: list[str]) -> np.ndarray:
    """
    Load per-commit embeddings. Missing files get a zero vector.
    Returns (N, 768) float32.
    """
    out = np.zeros((len(hashes), EMBED_DIM), dtype=np.float32)
    n_missing = 0
    for i, h in enumerate(hashes):
        npy = EMB_DIR / f"{h}.npy"
        try:
            out[i] = np.load(str(npy)).astype(np.float32)
        except Exception:
            n_missing += 1
    if n_missing:
        logger.warning("Missing embeddings: %d / %d (filled with zeros)", n_missing, len(hashes))
    return out


# -- Data loading ------------------------------------------------------------

def _load_split(
    path: Path,
    name: str,
    scaler: StandardScaler | None = None,
    fit_scaler: bool = False,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, StandardScaler]:
    """
    Load parquet split, build 790-dim hybrid feature matrix.

    Args:
        scaler:      pre-fit StandardScaler (pass None when fit_scaler=True)
        fit_scaler:  if True, fit a new scaler on this split's tabular cols

    Returns: X (N, 790), y (N,), df, scaler
    """
    logger.info("Loading %s split: %s", name, path)
    df = pd.read_parquet(path, engine="pyarrow")
    logger.info("  %s: %d rows, label dist: %s",
                name, len(df),
                dict(df["label"].value_counts().sort_index()))

    # --- Tabular block (22-dim) ---
    tab_parts: list[np.ndarray] = []
    for col in TABULAR_FEATURES:
        if col in df.columns:
            tab_parts.append(
                pd.to_numeric(df[col], errors="coerce").fillna(0).values.reshape(-1, 1)
            )
        else:
            logger.debug("Feature '%s' missing in %s -- filling with 0.", col, name)
            tab_parts.append(np.zeros((len(df), 1), dtype=np.float32))

    X_tab = np.hstack(tab_parts).astype(np.float32)

    # Fit or apply scaler (fit on train only)
    if fit_scaler:
        scaler = StandardScaler()
        scaler.fit(X_tab)
        logger.info("Tabular scaler fitted on %s split.", name)
    X_tab_scaled = scaler.transform(X_tab).astype(np.float32)

    # --- Embedding block (768-dim) ---
    hashes = df["commit_hash"].tolist()
    logger.info("Loading %d embeddings for %s ...", len(hashes), name)
    X_emb = _load_embeddings(hashes)

    # Concatenate: [tab_scaled | embeddings]
    X = np.hstack([X_tab_scaled, X_emb]).astype(np.float32)
    y = df["label"].astype(np.int8).values

    logger.info("  %s feature matrix: %s  (%.1f MB)", name, X.shape,
                X.nbytes / 1024 / 1024)
    return X, y, df, scaler


# -- Severity target (same formula as baseline) ------------------------------

def _build_severity_target(df: pd.DataFrame) -> np.ndarray:
    def _get_col(names: list[str]) -> np.ndarray:
        for name in names:
            if name in df.columns:
                return pd.to_numeric(df[name], errors="coerce").fillna(0).values.astype(np.float64)
        return np.zeros(len(df), dtype=np.float64)

    days  = _get_col(["days_to_fix", "AGE"])
    files = _get_col(["files_changed", "NF"])
    lines = _get_col(["lines_added", "LA"])

    norm_days  = np.clip(np.log1p(days)  / np.log1p(365.0),   0.0, 1.0)
    norm_files = np.clip(np.log1p(files) / np.log1p(50.0),    0.0, 1.0)
    norm_lines = np.clip(np.log1p(lines) / np.log1p(1000.0),  0.0, 1.0)

    score = 0.4 * norm_files + 0.4 * norm_lines + 0.2 * norm_days
    return np.clip(score, 0.0, 1.0).astype(np.float32)


# -- Optuna objective --------------------------------------------------------

def _make_objective(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    scale_pos_weight: float,
):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth":          trial.suggest_int("max_depth", 3, 8),
            "learning_rate":      trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators":       trial.suggest_int("n_estimators", 100, 800),
            "subsample":          trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "colsample_bylevel":  trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            "min_child_weight":   trial.suggest_int("min_child_weight", 1, 10),
            "gamma":              trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha":          trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":         trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "scale_pos_weight":   scale_pos_weight,
            "tree_method":        "hist",
            "device":             "cuda",
            "random_state":       SEED,
            "eval_metric":        "logloss",
            "verbosity":          0,
        }
        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        y_prob = model.predict_proba(X_val)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        return f1_score(y_val, y_pred, zero_division=0)

    return objective


# -- Threshold tuning --------------------------------------------------------

def _tune_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = np.where(
        (precision + recall) > 0,
        2 * precision * recall / (precision + recall),
        0.0,
    )
    min_threshold = 0.30
    valid_idx = [i for i, t in enumerate(thresholds) if t >= min_threshold]
    best_idx = (valid_idx[np.argmax(f1_scores[valid_idx])]
                if valid_idx else np.argmax(f1_scores[:-1]))
    best_threshold = float(thresholds[best_idx])
    logger.info(
        "Optimal threshold: %.4f  (val F1=%.4f, P=%.4f, R=%.4f)",
        best_threshold, f1_scores[best_idx], precision[best_idx], recall[best_idx],
    )
    return best_threshold


# -- SHAP analysis -----------------------------------------------------------

def _compute_shap(
    model: XGBClassifier,
    X_sample: np.ndarray,       # 1000-row subsample
    tabular_feature_names: list[str],
    artifact_dir: Path,
) -> None:
    """
    Compute SHAP on a subsample, save tabular-only SHAP values and plot.
    Embedding dims are excluded from the waterfall for readability.
    """
    artifact_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Computing SHAP values on %d-row sample ...", len(X_sample))

    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X_sample)   # (N, 790)

    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    # Keep only tabular dims for interpretability
    n_tab = len(tabular_feature_names)
    shap_tab = shap_vals[:, :n_tab]                # (N, 22)

    # Save tabular SHAP values
    shap_path = artifact_dir / "shap_values.npy"
    np.save(str(shap_path), shap_tab.astype(np.float32))
    logger.info("SHAP values saved -> %s  shape=%s", shap_path, shap_tab.shape)

    feat_path = artifact_dir / "feature_names.json"
    with open(feat_path, "w", encoding="utf-8") as fh:
        json.dump(tabular_feature_names, fh, indent=2)

    # Also log total embedding contribution vs tabular contribution
    tab_contrib  = np.abs(shap_vals[:, :n_tab]).sum(axis=1).mean()
    emb_contrib  = np.abs(shap_vals[:, n_tab:]).sum(axis=1).mean()
    logger.info(
        "Mean |SHAP| contribution -- Tabular: %.4f  Embedding: %.4f  "
        "Embedding share: %.1f%%",
        tab_contrib, emb_contrib, 100 * emb_contrib / (tab_contrib + emb_contrib + 1e-9),
    )

    # Waterfall: top 20 tabular features by mean |SHAP|
    mean_abs = np.abs(shap_tab).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[::-1][:20]
    top_names = [tabular_feature_names[i] for i in top_idx]
    top_means = mean_abs[top_idx]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(range(len(top_names)), top_means[::-1], color="#4C9BE8")
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Top 20 Tabular Features by SHAP (Hybrid Stage 1 -- Test Sample)")
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    plt.tight_layout()

    plot_path = artifact_dir / "waterfall_top20.png"
    fig.savefig(str(plot_path), dpi=150)
    plt.close(fig)
    logger.info("SHAP waterfall saved -> %s", plot_path)


# -- Comparison table and promotion decision ---------------------------------

def _compare_and_promote(
    test_f1: float, test_auc: float, test_p: float, test_r: float,
    timestamp: str, model_path: Path, artifact_dir: Path,
) -> bool:
    sep = "-" * 66

    def winner(hybrid, base):
        return "HYBRID" if hybrid > base else ("TIE" if hybrid == base else "XGBOOST")

    print(f"\n{sep}")
    print("  COMPARISON TABLE  (test = kamei/platform)")
    print(sep)
    print(f"  {'METRIC':<14} {'XGBOOST':>9} {'HYBRID':>9} {'DELTA':>9}  WINNER")
    print(f"  {'------':<14} {'-------':>9} {'------':>9} {'-----':>9}  ------")
    for label, base_v, hyb_v in [
        ("AUC-ROC",   BASELINE["auc"],       test_auc),
        ("F1",        BASELINE["f1"],        test_f1),
        ("Precision", BASELINE["precision"], test_p),
        ("Recall",    BASELINE["recall"],    test_r),
    ]:
        delta = hyb_v - base_v
        print(f"  {label:<14} {base_v:>9.4f} {hyb_v:>9.4f} {delta:>+9.4f}  {winner(hyb_v, base_v)}")
    print(sep)

    # Gate evaluation
    gate_results = {
        "AUC-ROC": (test_auc >= PROMOTION_GATES["auc"],  test_auc, PROMOTION_GATES["auc"]),
        "F1":      (test_f1  >= PROMOTION_GATES["f1"],   test_f1,  PROMOTION_GATES["f1"]),
        "Recall":  (test_r   >= PROMOTION_GATES["recall"], test_r, PROMOTION_GATES["recall"]),
    }

    print("\n  PROMOTION GATES")
    all_passed = True
    failed: list[str] = []
    for gate_name, (passed, actual, required) in gate_results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}]   {gate_name:<10}  {actual:.4f}  (required >= {required:.3f})")
        if not passed:
            all_passed = False
            failed.append(f"{gate_name}: {actual:.4f} < {required:.3f} (gap {actual-required:+.4f})")
    print(sep)

    if all_passed:
        flag = artifact_dir / "PROMOTED"
        flag.write_text(f"Promoted: {timestamp}\nModel: {model_path}\n", encoding="utf-8")
        print(f"\n  *** HYBRID MODEL PROMOTED ***")
        print(f"  Flag written: {flag}")
    else:
        print(f"\n  *** HYBRID MODEL NOT PROMOTED -- XGBoost remains active ***")
        print("  Failed gates:")
        for fg in failed:
            print(f"    - {fg}")
    print()

    _update_comparison_doc(test_auc, test_f1, test_p, test_r,
                           timestamp, all_passed, failed)
    return all_passed


def _update_comparison_doc(
    test_auc: float, test_f1: float, test_p: float, test_r: float,
    timestamp: str, promoted: bool, failed: list[str],
) -> None:
    doc_path = ROOT / "experiments" / "model_comparison.md"
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    promo_str = "PROMOTED" if promoted else "NOT PROMOTED"
    date_str  = datetime.now().strftime("%Y-%m-%d")

    section = f"""
## Phase 3 -- Hybrid XGBoost (Tabular + BERT embeddings)
Date: {date_str}
Features: 22 tabular (StandardScaler-normalised) + 768-dim BERT sentence-transformer = 790 total
Test set: kamei/platform (25,959 rows, 36.4% buggy)

| METRIC    | XGBOOST | HYBRID  | DELTA   | WINNER  |
|-----------|---------|---------|---------|---------|
| AUC-ROC   | {BASELINE['auc']:.4f}  | {test_auc:.4f}  | {test_auc-BASELINE['auc']:+.4f}  | {'HYBRID' if test_auc > BASELINE['auc'] else 'XGBOOST'} |
| F1        | {BASELINE['f1']:.4f}  | {test_f1:.4f}  | {test_f1-BASELINE['f1']:+.4f}  | {'HYBRID' if test_f1 > BASELINE['f1'] else 'XGBOOST'} |
| Precision | {BASELINE['precision']:.4f}  | {test_p:.4f}  | {test_p-BASELINE['precision']:+.4f}  | {'HYBRID' if test_p > BASELINE['precision'] else 'XGBOOST'} |
| Recall    | {BASELINE['recall']:.4f}  | {test_r:.4f}  | {test_r-BASELINE['recall']:+.4f}  | {'HYBRID' if test_r > BASELINE['recall'] else 'XGBOOST'} |

Promotion result: **{promo_str}**
"""
    if promoted:
        section += "All promotion gates passed (AUC >= 0.780, F1 >= 0.640, Recall >= 0.900).\n"
    else:
        section += "Failed gates:\n"
        for fg in failed:
            section += f"- {fg}\n"

    mode = "a" if doc_path.exists() else "w"
    with open(doc_path, mode, encoding="utf-8") as f:
        if mode == "w":
            f.write("# Model Comparison\n")
        f.write(section)
    logger.info("model_comparison.md updated: %s", doc_path)


# -- Main training pipeline --------------------------------------------------

def train_hybrid(
    train_path: Path,
    val_path:   Path,
    test_path:  Path,
    n_trials:   int  = 50,
    artifact_dir: Path = ROOT / "experiments",
) -> bool:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    hybrid_dir = artifact_dir / "hybrid"
    hybrid_dir.mkdir(parents=True, exist_ok=True)

    # -- Load splits ----------------------------------------------------------
    X_train, y_train, df_train, scaler = _load_split(train_path, "train", fit_scaler=True)
    X_val,   y_val,   df_val,   _      = _load_split(val_path,   "val",   scaler=scaler)
    X_test,  y_test,  df_test,  _      = _load_split(test_path,  "test",  scaler=scaler)

    # Save the scaler for API use
    scaler_path = hybrid_dir / f"scaler_{timestamp}.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info("Scaler saved -> %s", scaler_path)

    n_buggy = int(y_train.sum())
    n_clean = len(y_train) - n_buggy
    scale_pos_weight = max(n_clean / max(n_buggy, 1), 1.0)
    logger.info("Class ratio -- buggy=%d clean=%d scale_pos_weight=%.2f",
                n_buggy, n_clean, scale_pos_weight)

    # -- MLflow ---------------------------------------------------------------
    mlflow_uri = (artifact_dir / "mlruns").as_uri()
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # =========================================================================
    # STAGE 1 -- Recurrence Classifier
    # =========================================================================
    with mlflow.start_run(run_name=f"hybrid_stage1_{timestamp}") as run:
        logger.info("=" * 60)
        logger.info("STAGE 1 -- Hybrid Recurrence Classifier (Optuna %d trials)", n_trials)
        logger.info("=" * 60)

        mlflow.log_params({
            "n_trials": n_trials, "seed": SEED,
            "n_features": N_FEATURES, "embed_dim": EMBED_DIM,
            "scale_pos_weight": round(scale_pos_weight, 4),
            "train_rows": len(X_train), "val_rows": len(X_val), "test_rows": len(X_test),
        })

        # Optuna search
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=SEED),
            study_name="hybrid_stage1",
        )
        objective = _make_objective(X_train, y_train, X_val, y_val, scale_pos_weight)
        logger.info("Running Optuna ...")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_params
        logger.info("Best val F1: %.4f", study.best_value)
        logger.info("Best params: %s", best_params)

        for k, v in best_params.items():
            mlflow.log_param(f"best_{k}", v)

        # Retrain with best params
        logger.info("Retraining with best hyperparameters ...")
        best_params.update({
            "scale_pos_weight": scale_pos_weight,
            "tree_method": "hist",
            "device": "cuda",
            "random_state": SEED,
            "eval_metric": "logloss",
            "verbosity": 0,
        })
        stage1 = XGBClassifier(**best_params)
        stage1.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        # Threshold tuning on val
        val_prob  = stage1.predict_proba(X_val)[:, 1]
        threshold = _tune_threshold(y_val, val_prob)
        val_pred  = (val_prob >= threshold).astype(int)

        val_f1  = f1_score(y_val, val_pred, zero_division=0)
        val_auc = roc_auc_score(y_val, val_prob)
        val_p   = precision_score(y_val, val_pred, zero_division=0)
        val_r   = recall_score(y_val, val_pred, zero_division=0)
        logger.info("VAL  -- F1=%.4f  AUC=%.4f  P=%.4f  R=%.4f  thresh=%.4f",
                    val_f1, val_auc, val_p, val_r, threshold)

        mlflow.log_metrics({
            "val_f1": val_f1, "val_auc": val_auc,
            "val_precision": val_p, "val_recall": val_r, "val_threshold": threshold,
        })

        # Test evaluation
        test_prob = stage1.predict_proba(X_test)[:, 1]
        test_pred = (test_prob >= threshold).astype(int)

        test_f1  = f1_score(y_test, test_pred, zero_division=0)
        test_auc = roc_auc_score(y_test, test_prob)
        test_p   = precision_score(y_test, test_pred, zero_division=0)
        test_r   = recall_score(y_test, test_pred, zero_division=0)
        logger.info("TEST -- F1=%.4f  AUC=%.4f  P=%.4f  R=%.4f",
                    test_f1, test_auc, test_p, test_r)

        mlflow.log_metrics({
            "test_f1": test_f1, "test_auc": test_auc,
            "test_precision": test_p, "test_recall": test_r,
        })

        # Save Stage 1
        model_path = hybrid_dir / f"stage1_{timestamp}.pkl"
        joblib.dump(
            {"model": stage1, "threshold": threshold,
             "tabular_features": TABULAR_FEATURES, "scaler": scaler},
            str(model_path),
        )
        logger.info("Stage 1 model saved -> %s", model_path)

        # SHAP on 1000-row test subsample
        logger.info("Computing SHAP ...")
        try:
            rng = np.random.default_rng(SEED)
            sample_idx = rng.choice(len(X_test), size=min(1000, len(X_test)), replace=False)
            _compute_shap(stage1, X_test[sample_idx], TABULAR_FEATURES, hybrid_dir)
        except Exception as exc:
            logger.warning("SHAP computation failed (non-fatal): %s", exc)

    # =========================================================================
    # STAGE 2 -- Severity Estimator (same as baseline, just 790-dim input)
    # =========================================================================
    logger.info("=" * 60)
    logger.info("STAGE 2 -- Severity Estimator (XGBRegressor)")
    logger.info("=" * 60)

    train_pred_full = (stage1.predict_proba(X_train)[:, 1] >= threshold).astype(bool)
    val_pred_full   = (val_prob  >= threshold).astype(bool)
    test_pred_bool  = (test_prob >= threshold).astype(bool)

    X_train_buggy = X_train[train_pred_full]
    X_val_buggy   = X_val[val_pred_full]
    X_test_buggy  = X_test[test_pred_bool]
    y_sev_train   = _build_severity_target(df_train[train_pred_full])
    y_sev_val     = _build_severity_target(df_val[val_pred_full])
    y_sev_test    = _build_severity_target(df_test[test_pred_bool])

    logger.info("Stage 2 train/val/test rows (predicted buggy): %d / %d / %d",
                len(X_train_buggy), len(X_val_buggy), len(X_test_buggy))

    if len(X_train_buggy) >= 10:
        stage2 = XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=1.0, reg_lambda=5.0,
            tree_method="hist", device="cuda",
            random_state=SEED, verbosity=0,
        )
        stage2.fit(
            X_train_buggy, y_sev_train,
            eval_set=[(X_val_buggy, y_sev_val)] if len(X_val_buggy) > 0 else None,
            verbose=False,
        )
        if len(X_test_buggy) > 0:
            test_sev_pred = stage2.predict(X_test_buggy).clip(0, 1)
            test_mae  = mean_absolute_error(y_sev_test, test_sev_pred)
            test_rmse = float(np.sqrt(mean_squared_error(y_sev_test, test_sev_pred)))
            logger.info("Stage 2 TEST -- MAE=%.4f  RMSE=%.4f", test_mae, test_rmse)

        s2_path = hybrid_dir / f"stage2_{timestamp}.pkl"
        joblib.dump({"model": stage2, "tabular_features": TABULAR_FEATURES, "scaler": scaler},
                    str(s2_path))
        logger.info("Stage 2 model saved -> %s", s2_path)
    else:
        logger.warning("Too few buggy training rows for Stage 2 -- skipping.")

    # -- Save metrics ---------------------------------------------------------
    metrics = {
        "timestamp": timestamp,
        "n_features": N_FEATURES,
        "val":  {"f1": val_f1, "auc": val_auc, "precision": val_p, "recall": val_r,
                 "threshold": threshold},
        "test": {"f1": test_f1, "auc": test_auc, "precision": test_p, "recall": test_r},
        "baseline": BASELINE,
        "promotion_gates": PROMOTION_GATES,
    }
    metrics_path = hybrid_dir / f"metrics_{timestamp}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved -> %s", metrics_path)

    # -- Comparison and promotion decision ------------------------------------
    promoted = _compare_and_promote(
        test_f1, test_auc, test_p, test_r, timestamp, model_path, hybrid_dir
    )
    return promoted


# -- CLI ---------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train_hybrid.py",
        description="Hybrid XGBoost: 22 tabular + 768-dim BERT = 790 features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train",        type=Path, default=ROOT / "data/splits/train.parquet")
    parser.add_argument("--val",          type=Path, default=ROOT / "data/splits/val.parquet")
    parser.add_argument("--test",         type=Path, default=ROOT / "data/splits/test.parquet")
    parser.add_argument("--n-trials",     type=int,  default=50)
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "experiments")
    parser.add_argument("--log-level",    default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    _configure_logging(args.log_level)
    logger.info("forge-bug-predictor | hybrid XGBoost -- TRAIN")
    logger.info("Train : %s", args.train)
    logger.info("Val   : %s", args.val)
    logger.info("Test  : %s", args.test)
    logger.info("Trials: %d", args.n_trials)
    logger.info("Features: %d tabular + %d embedding = %d total",
                len(TABULAR_FEATURES), EMBED_DIM, N_FEATURES)

    promoted = train_hybrid(
        train_path=args.train,
        val_path=args.val,
        test_path=args.test,
        n_trials=args.n_trials,
        artifact_dir=args.artifact_dir,
    )
    sys.exit(0 if promoted else 1)


if __name__ == "__main__":
    main()
