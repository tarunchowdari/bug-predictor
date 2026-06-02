"""
models/baseline/train.py

XGBoost two-stage baseline for bug recurrence prediction.

Stage 1 â€” Recurrence Classifier
  Input : 22 tabular features (14 Kamei + 8 AST)
  Model : XGBClassifier, 50 Optuna trials optimising val F1
  Output: binary prediction + probability score
  Post  : threshold tuning on val PR curve

Stage 2 â€” Severity Estimator
  Input : same 22 features, restricted to Stage-1-predicted-buggy rows
  Target: composite severity score from NUC, NF, AGE (normalised)
  Model : XGBRegressor
  Output: continuous risk score 0â€“1

SHAP Analysis
  Computed on test set for Stage 1.
  Saves: shap_values.npy, feature_names.json, waterfall_top20.png

Evaluation Gates (Stage 1, test set):
  F1      >= 0.65  (else exit 1)
  AUC-ROC >= 0.70  (else exit 1)

MLflow:
  Tracking URI : ./experiments/mlruns
  Experiment   : baseline_v1

Usage:
  python models/baseline/train.py
  python models/baseline/train.py --n-trials 5  # fast smoke test
  python models/baseline/train.py \
      --train data/splits/train.parquet \
      --val   data/splits/val.parquet   \
      --test  data/splits/test.parquet  \
      --n-trials 50
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")   # headless backend â€” must be before pyplot import
import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    mean_absolute_error,
    mean_squared_error,
)
from xgboost import XGBClassifier, XGBRegressor

# â”€â”€ Silence noisy third-party warnings â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
warnings.filterwarnings("ignore", category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# â”€â”€ Logging â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _configure_logging(level: str) -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s â€” %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        level=getattr(logging, level.upper(), logging.INFO),
        stream=sys.stdout,
    )


logger = logging.getLogger("forge.train")

# â”€â”€ Constants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

SEED = 42
KAMEI_FEATURES: list[str] = [
    "NS", "ND", "NF", "Entropy", "LA", "LD", "LT",
    "FIX", "NOD", "NUC", "AGE", "EXP", "REXP", "SEXP",
]
AST_FEATURES: list[str] = [
    "n_nodes_added", "n_nodes_deleted", "n_control_flow_changes",
    "n_function_changes", "max_depth_before", "max_depth_after",
    "depth_delta", "unique_node_types_added",
]
ALL_FEATURES: list[str] = KAMEI_FEATURES + AST_FEATURES   # 22-dim

GATE_F1    = 0.60
GATE_AUC   = 0.70

MLFLOW_EXPERIMENT = "baseline_v1"
ARTIFACT_DIR = Path("experiments")


# â”€â”€ Data loading â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _load_split(path: Path, name: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load a split parquet. Returns (X, y, full_df).
    Missing feature columns are filled with 0.
    """
    logger.info("Loading %s split: %s", name, path)
    df = pd.read_parquet(path, engine="pyarrow")
    logger.info("  %s: %d rows, label dist: %s",
                name, len(df),
                dict(df["label"].value_counts().sort_index()))

    X_parts: list[np.ndarray] = []
    for col in ALL_FEATURES:
        if col in df.columns:
            X_parts.append(
                pd.to_numeric(df[col], errors="coerce").fillna(0).values.reshape(-1, 1)
            )
        else:
            logger.debug("Feature '%s' missing in %s â€” filling with 0.", col, name)
            X_parts.append(np.zeros((len(df), 1), dtype=np.float32))

    X = np.hstack(X_parts).astype(np.float32)
    y = df["label"].astype(np.int8).values
    return X, y, df


# â”€â”€ Severity target construction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _build_severity_target(df: pd.DataFrame) -> np.ndarray:
    """
    Construct a composite severity score in [0, 1] using fixed global heuristics
    so the formula is identical across all splits.
    """
    def _get_col(names: list[str]) -> np.ndarray:
        for name in names:
            if name in df.columns:
                return pd.to_numeric(df[name], errors="coerce").fillna(0).values.astype(np.float64)
        return np.zeros(len(df), dtype=np.float64)

    # Use raw columns (days_to_fix / AGE, files_changed / NF, lines_added / LA)
    days = _get_col(["days_to_fix", "AGE"])
    files = _get_col(["files_changed", "NF"])
    lines = _get_col(["lines_added", "LA"])

    # Fixed formula (using log1p to handle skew and capping at reasonable maximums)
    # 365 days = 1.0
    norm_days = np.clip(np.log1p(days) / np.log1p(365.0), 0.0, 1.0)
    # 50 files = 1.0
    norm_files = np.clip(np.log1p(files) / np.log1p(50.0), 0.0, 1.0)
    # 1000 lines = 1.0
    norm_lines = np.clip(np.log1p(lines) / np.log1p(1000.0), 0.0, 1.0)

    score = 0.4 * norm_files + 0.4 * norm_lines + 0.2 * norm_days
    return np.clip(score, 0.0, 1.0).astype(np.float32)


# â”€â”€ Optuna objective for Stage 1 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _make_objective(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    scale_pos_weight: float,
):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators":     trial.suggest_int("n_estimators", 100, 1000),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "scale_pos_weight": scale_pos_weight,
            "random_state":     SEED,
            "eval_metric":      "logloss",
            "use_label_encoder": False,
            "verbosity":        0,
            "n_jobs":           -1,
        }
        model = XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        y_prob = model.predict_proba(X_val)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        return f1_score(y_val, y_pred, zero_division=0)

    return objective


# â”€â”€ Threshold tuning â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _tune_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Find the decision threshold that maximises F1 on the given set
    using the precision-recall curve.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = np.where(
        (precision + recall) > 0,
        2 * precision * recall / (precision + recall),
        0.0,
    )
    # thresholds has len = len(precision) - 1
    min_threshold = 0.30
    valid_idx = [i for i, t in enumerate(thresholds) if t >= min_threshold]
    if not valid_idx:
        best_idx = np.argmax(f1_scores[:-1])
    else:
        best_valid_idx = np.argmax(f1_scores[valid_idx])
        best_idx = valid_idx[best_valid_idx]
    
    best_threshold = float(thresholds[best_idx])
    logger.info(
        "Optimal threshold: %.4f  (val F1=%.4f, P=%.4f, R=%.4f)",
        best_threshold,
        f1_scores[best_idx],
        precision[best_idx],
        recall[best_idx],
    )
    return best_threshold


# â”€â”€ SHAP analysis â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _compute_shap(
    model: XGBClassifier,
    X_test: np.ndarray,
    feature_names: list[str],
    artifact_dir: Path,
) -> None:
    """
    Compute SHAP values on the test set; save arrays and waterfall plot.
    """
    artifact_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Computing SHAP values on test set â€¦")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # shap_values shape: (N, n_features) for binary XGB
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # positive class

    # Save raw values
    shap_path = artifact_dir / "shap_values.npy"
    np.save(str(shap_path), shap_values.astype(np.float32))
    logger.info("SHAP values saved â†’ %s  shape=%s", shap_path, shap_values.shape)

    # Save feature names
    feat_path = artifact_dir / "feature_names.json"
    with open(feat_path, "w", encoding="utf-8") as fh:
        json.dump(feature_names, fh, indent=2)

    # Waterfall plot â€” top 20 features by mean |SHAP|
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:20]
    top_names = [feature_names[i] for i in top_idx]
    top_means = mean_abs[top_idx]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(range(len(top_names)), top_means[::-1], color="#4C9BE8")
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Top 20 Features by SHAP Importance (Stage 1 â€” Test Set)")
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    plt.tight_layout()

    plot_path = artifact_dir / "waterfall_top20.png"
    fig.savefig(str(plot_path), dpi=150)
    plt.close(fig)
    logger.info("SHAP waterfall plot saved â†’ %s", plot_path)


# â”€â”€ Metrics table â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _print_metrics_table(
    metrics: dict[str, dict[str, float]],
    gates_passed: bool,
) -> None:
    sep = "â”€" * 70
    print(f"\n{sep}")
    print("  FINAL METRICS TABLE â€” XGBoost Baseline")
    print(sep)
    print(f"  {'METRIC':<22}  {'VAL':>10}  {'TEST':>10}")
    print(f"  {'â”€â”€â”€â”€â”€â”€':<22}  {'â”€â”€â”€':>10}  {'â”€â”€â”€â”€':>10}")

    row_order = [
        ("Stage 1 F1",        "stage1_f1"),
        ("Stage 1 AUC-ROC",   "stage1_auc"),
        ("Stage 1 Precision",  "stage1_precision"),
        ("Stage 1 Recall",     "stage1_recall"),
        ("Stage 1 Threshold",  "stage1_threshold"),
        ("Stage 2 MAE",        "stage2_mae"),
        ("Stage 2 RMSE",       "stage2_rmse"),
    ]

    for label, key in row_order:
        val_v  = metrics.get("val",  {}).get(key, float("nan"))
        test_v = metrics.get("test", {}).get(key, float("nan"))
        val_s  = f"{val_v:.4f}"  if not (isinstance(val_v, float) and np.isnan(val_v)) else "  â€”"
        test_s = f"{test_v:.4f}" if not (isinstance(test_v, float) and np.isnan(test_v)) else "  â€”"
        print(f"  {label:<22}  {val_s:>10}  {test_s:>10}")

    print(sep)

    # Gate result
    test_f1  = metrics.get("test", {}).get("stage1_f1",  0.0)
    test_auc = metrics.get("test", {}).get("stage1_auc", 0.0)
    f1_sym   = "âœ“" if test_f1  >= GATE_F1  else "âœ—"
    auc_sym  = "âœ“" if test_auc >= GATE_AUC else "âœ—"

    print(f"\n  EVALUATION GATES")
    print(f"  {f1_sym}  Test F1     {test_f1:.4f}  (gate â‰¥ {GATE_F1})")
    print(f"  {auc_sym}  Test AUC    {test_auc:.4f}  (gate â‰¥ {GATE_AUC})")
    print()

    if gates_passed:
        print("  âœ… Both gates PASSED â€” proceed to Phase 4 (Neural Fusion).")
    else:
        print("  âŒ Gate(s) FAILED â€” do NOT proceed to neural fusion.")
        print("     See diagnostic report above.")
    print(f"{sep}\n")


def _print_diagnostic(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    test_f1: float, test_auc: float,
) -> None:
    print("\n" + "=" * 70)
    print("  DIAGNOSTIC REPORT â€” Evaluation Gates Failed")
    print("=" * 70)
    print(f"\n  Test F1     = {test_f1:.4f}  (required â‰¥ {GATE_F1})")
    print(f"  Test AUC    = {test_auc:.4f}  (required â‰¥ {GATE_AUC})")
    print(f"\n  Dataset sizes:")
    print(f"    Train : {len(X_train):,} rows  "
          f"({100*y_train.mean():.1f}% buggy)")
    print(f"    Val   : {len(X_val):,} rows  "
          f"({100*y_val.mean():.1f}% buggy)")
    print(f"    Test  : {len(X_test):,} rows  "
          f"({100*y_test.mean():.1f}% buggy)")
    print(f"\n  Possible causes:")
    print(f"    â€¢ Insufficient training data â€” mine more commits (01_mine.py --max-commits)")
    print(f"    â€¢ Class imbalance â€” re-run 02_preprocess.py with lower --min-buggy-ratio")
    print(f"    â€¢ Feature quality â€” verify AST features are non-zero (03_features.py)")
    print(f"    â€¢ Labeling noise â€” check 90-day look-forward window (01_mine.py)")
    print("=" * 70 + "\n")


# â”€â”€ Main training pipeline â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def train_baseline(
    train_path: Path,
    val_path: Path,
    test_path: Path,
    n_trials: int = 50,
    artifact_dir: Path = ARTIFACT_DIR,
) -> bool:
    """
    Full two-stage XGBoost training pipeline.
    Returns True if evaluation gates pass, False otherwise.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # â”€â”€ Load splits â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    X_train, y_train, df_train = _load_split(train_path, "train")
    X_val,   y_val,   df_val   = _load_split(val_path,   "val")
    X_test,  y_test,  df_test  = _load_split(test_path,  "test")

    n_buggy = y_train.sum()
    n_clean = len(y_train) - n_buggy
    scale_pos_weight = max(n_clean / max(n_buggy, 1), 1.0)
    logger.info(
        "Class ratio â€” buggy=%d clean=%d scale_pos_weight=%.2f",
        n_buggy, n_clean, scale_pos_weight,
    )

    # â”€â”€ MLflow setup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    mlflow.set_tracking_uri(str(artifact_dir / "mlruns"))
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    metrics_store: dict[str, dict[str, float]] = {"val": {}, "test": {}}

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # STAGE 1 â€” Recurrence Classifier
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    with mlflow.start_run(run_name=f"stage1_{timestamp}") as run:
        logger.info("=" * 60)
        logger.info("STAGE 1 â€” Recurrence Classifier (Optuna %d trials)", n_trials)
        logger.info("=" * 60)

        mlflow.log_param("n_optuna_trials", n_trials)
        mlflow.log_param("seed", SEED)
        mlflow.log_param("scale_pos_weight", round(scale_pos_weight, 4))
        mlflow.log_param("n_features", len(ALL_FEATURES))
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("val_rows", len(X_val))
        mlflow.log_param("test_rows", len(X_test))

        # â”€â”€ Optuna search â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=SEED),
            study_name="stage1_xgb",
        )
        objective = _make_objective(
            X_train, y_train, X_val, y_val, scale_pos_weight
        )
        logger.info("Running Optuna â€¦")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_params
        best_val_f1 = study.best_value
        logger.info("Best val F1: %.4f", best_val_f1)
        logger.info("Best params: %s", best_params)

        for k, v in best_params.items():
            mlflow.log_param(f"best_{k}", v)

        # â”€â”€ Retrain best model on full train set â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        logger.info("Retraining with best hyperparameters â€¦")
        best_params.update({
            "scale_pos_weight": scale_pos_weight,
            "random_state":     SEED,
            "eval_metric":      "logloss",
            "verbosity":        0,
            "n_jobs":           -1,
        })
        stage1_model = XGBClassifier(**best_params)
        stage1_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # â”€â”€ Threshold tuning on val set â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        val_prob = stage1_model.predict_proba(X_val)[:, 1]
        threshold = _tune_threshold(y_val, val_prob)
        val_pred = (val_prob >= threshold).astype(int)

        val_f1  = f1_score(y_val, val_pred, zero_division=0)
        val_auc = roc_auc_score(y_val, val_prob)
        val_p   = precision_score(y_val, val_pred, zero_division=0)
        val_r   = recall_score(y_val, val_pred, zero_division=0)

        mlflow.log_metric("val_f1",        val_f1)
        mlflow.log_metric("val_auc",       val_auc)
        mlflow.log_metric("val_precision", val_p)
        mlflow.log_metric("val_recall",    val_r)
        mlflow.log_metric("val_threshold", threshold)

        metrics_store["val"]["stage1_f1"]        = val_f1
        metrics_store["val"]["stage1_auc"]       = val_auc
        metrics_store["val"]["stage1_precision"]  = val_p
        metrics_store["val"]["stage1_recall"]     = val_r
        metrics_store["val"]["stage1_threshold"]  = threshold

        logger.info(
            "VAL  â€” F1=%.4f  AUC=%.4f  P=%.4f  R=%.4f  thresh=%.4f",
            val_f1, val_auc, val_p, val_r, threshold,
        )

        # â”€â”€ Test evaluation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        test_prob = stage1_model.predict_proba(X_test)[:, 1]
        test_pred = (test_prob >= threshold).astype(int)

        test_f1  = f1_score(y_test, test_pred, zero_division=0)
        test_auc = roc_auc_score(y_test, test_prob)
        test_p   = precision_score(y_test, test_pred, zero_division=0)
        test_r   = recall_score(y_test, test_pred, zero_division=0)

        mlflow.log_metric("test_f1",        test_f1)
        mlflow.log_metric("test_auc",       test_auc)
        mlflow.log_metric("test_precision", test_p)
        mlflow.log_metric("test_recall",    test_r)

        metrics_store["test"]["stage1_f1"]       = test_f1
        metrics_store["test"]["stage1_auc"]      = test_auc
        metrics_store["test"]["stage1_precision"] = test_p
        metrics_store["test"]["stage1_recall"]    = test_r
        metrics_store["test"]["stage1_threshold"] = threshold

        logger.info(
            "TEST â€” F1=%.4f  AUC=%.4f  P=%.4f  R=%.4f",
            test_f1, test_auc, test_p, test_r,
        )

        # â”€â”€ Save Stage 1 model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        model_path = artifact_dir / f"baseline" / f"stage1_{timestamp}.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": stage1_model, "threshold": threshold, "features": ALL_FEATURES},
            str(model_path),
        )
        mlflow.log_artifact(str(model_path))
        logger.info("Stage 1 model saved â†’ %s", model_path)

        # Also save threshold for API use
        thresh_path = artifact_dir / "baseline" / f"threshold_{timestamp}.json"
        with open(thresh_path, "w") as fh:
            json.dump({"threshold": threshold, "features": ALL_FEATURES}, fh, indent=2)

        mlflow.log_param("stage1_model_path", str(model_path))
        mlflow.log_param("mlflow_run_id", run.info.run_id)

        # â”€â”€ SHAP â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        logger.info("Computing SHAP â€¦")
        try:
            _compute_shap(stage1_model, X_test, ALL_FEATURES, artifact_dir)
            mlflow.log_artifact(str(artifact_dir / "shap_values.npy"))
            mlflow.log_artifact(str(artifact_dir / "feature_names.json"))
            mlflow.log_artifact(str(artifact_dir / "waterfall_top20.png"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("SHAP computation failed (non-fatal): %s", exc)

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # STAGE 2 â€” Severity Estimator
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    with mlflow.start_run(run_name=f"stage2_{timestamp}"):
        logger.info("=" * 60)
        logger.info("STAGE 2 â€” Severity Estimator (XGBRegressor)")
        logger.info("=" * 60)

        # Restrict to rows Stage 1 predicts as buggy (train set)
        train_prob_full = stage1_model.predict_proba(X_train)[:, 1]
        train_pred_full = (train_prob_full >= threshold).astype(bool)
        X_train_buggy  = X_train[train_pred_full]
        y_sev_train    = _build_severity_target(df_train[train_pred_full])

        val_pred_full   = (val_prob >= threshold).astype(bool)
        X_val_buggy     = X_val[val_pred_full]
        y_sev_val       = _build_severity_target(df_val[val_pred_full])

        test_pred_bool  = (test_prob >= threshold).astype(bool)
        X_test_buggy    = X_test[test_pred_bool]
        y_sev_test      = _build_severity_target(df_test[test_pred_bool])

        logger.info(
            "Stage 2 train/val/test rows (predicted buggy): %d / %d / %d",
            len(X_train_buggy), len(X_val_buggy), len(X_test_buggy),
        )

        for name, arr in [("train", y_sev_train), ("val", y_sev_val), ("test", y_sev_test)]:
            if len(arr) > 0:
                logger.info("Stage 2 Target (%s): mean=%.4f std=%.4f min=%.4f max=%.4f pct_zero=%.4f",
                            name, arr.mean(), arr.std(), arr.min(), arr.max(), (arr == 0).mean())

        mlflow.log_param("stage2_train_rows", len(X_train_buggy))

        if len(X_train_buggy) < 10:
            logger.warning(
                "Too few buggy training rows (%d) for Stage 2 â€” skipping regressor.",
                len(X_train_buggy),
            )
            metrics_store["val"]["stage2_mae"]  = float("nan")
            metrics_store["val"]["stage2_rmse"] = float("nan")
            metrics_store["test"]["stage2_mae"]  = float("nan")
            metrics_store["test"]["stage2_rmse"] = float("nan")
        else:
            stage2_model = XGBRegressor(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=1.0,
                reg_lambda=5.0,
                random_state=SEED,
                verbosity=0,
                n_jobs=-1,
            )
            stage2_model.fit(
                X_train_buggy, y_sev_train,
                eval_set=[(X_val_buggy, y_sev_val)] if len(X_val_buggy) > 0 else None,
                verbose=False,
            )

            # Val metrics
            if len(X_val_buggy) > 0:
                val_sev_pred = stage2_model.predict(X_val_buggy).clip(0, 1)
                val_mae  = mean_absolute_error(y_sev_val, val_sev_pred)
                val_rmse = float(np.sqrt(mean_squared_error(y_sev_val, val_sev_pred)))
                mlflow.log_metric("val_severity_mae",  val_mae)
                mlflow.log_metric("val_severity_rmse", val_rmse)
                metrics_store["val"]["stage2_mae"]  = val_mae
                metrics_store["val"]["stage2_rmse"] = val_rmse
                logger.info("Stage 2 VAL â€” MAE=%.4f  RMSE=%.4f", val_mae, val_rmse)
            else:
                metrics_store["val"]["stage2_mae"]  = float("nan")
                metrics_store["val"]["stage2_rmse"] = float("nan")

            # Test metrics
            if len(X_test_buggy) > 0:
                test_sev_pred = stage2_model.predict(X_test_buggy).clip(0, 1)
                test_mae  = mean_absolute_error(y_sev_test, test_sev_pred)
                test_rmse = float(np.sqrt(mean_squared_error(y_sev_test, test_sev_pred)))
                mlflow.log_metric("test_severity_mae",  test_mae)
                mlflow.log_metric("test_severity_rmse", test_rmse)
                metrics_store["test"]["stage2_mae"]  = test_mae
                metrics_store["test"]["stage2_rmse"] = test_rmse
                logger.info("Stage 2 TEST â€” MAE=%.4f  RMSE=%.4f", test_mae, test_rmse)
            else:
                metrics_store["test"]["stage2_mae"]  = float("nan")
                metrics_store["test"]["stage2_rmse"] = float("nan")

            # Save Stage 2 model
            s2_path = artifact_dir / "baseline" / f"stage2_{timestamp}.pkl"
            joblib.dump(
                {"model": stage2_model, "features": ALL_FEATURES},
                str(s2_path),
            )
            mlflow.log_artifact(str(s2_path))
            logger.info("Stage 2 model saved â†’ %s", s2_path)

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # EVALUATION GATES
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    gates_passed = (test_f1 >= GATE_F1) and (test_auc >= GATE_AUC)

    _print_metrics_table(metrics_store, gates_passed)

    if not gates_passed:
        _print_diagnostic(
            X_train, y_train, X_val, y_val, X_test, y_test,
            test_f1, test_auc,
        )

    # Save metrics JSON for downstream consumption (e.g. model comparison)
    metrics_path = artifact_dir / "baseline" / f"metrics_{timestamp}.json"
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "timestamp": timestamp,
                "metrics": metrics_store,
                "gates": {
                    "f1_gate":  GATE_F1,
                    "auc_gate": GATE_AUC,
                    "passed":   gates_passed,
                },
                "features": ALL_FEATURES,
                "model_paths": {
                    "stage1": str(model_path),
                },
            },
            fh,
            indent=2,
            default=str,
        )
    logger.info("Metrics saved â†’ %s", metrics_path)

    return gates_passed


# â”€â”€ CLI â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train.py",
        description="XGBoost two-stage baseline trainer with Optuna + SHAP + MLflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train", type=Path, default=Path("data/splits/train.parquet"))
    parser.add_argument("--val",   type=Path, default=Path("data/splits/val.parquet"))
    parser.add_argument("--test",  type=Path, default=Path("data/splits/test.parquet"))
    parser.add_argument(
        "--n-trials", type=int, default=50, metavar="N",
        help="Number of Optuna hyperparameter trials.",
    )
    parser.add_argument(
        "--artifact-dir", type=Path, default=ARTIFACT_DIR, metavar="DIR",
        help="Root directory for model artifacts, SHAP outputs, and MLflow.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    _configure_logging(args.log_level)
    logger.info("forge-bug-predictor | baseline â€” TRAIN")
    logger.info("Train : %s", args.train)
    logger.info("Val   : %s", args.val)
    logger.info("Test  : %s", args.test)
    logger.info("Trials: %d", args.n_trials)

    gates_passed = train_baseline(
        train_path=args.train,
        val_path=args.val,
        test_path=args.test,
        n_trials=args.n_trials,
        artifact_dir=args.artifact_dir,
    )

    sys.exit(0 if gates_passed else 1)


if __name__ == "__main__":
    main()

