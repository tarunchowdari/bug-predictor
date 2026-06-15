"""
pipeline/02_preprocess.py
Merge Kamei PROMISE benchmark + PyDriller-mined commits, clean, balance,
and produce the canonical dataset that all downstream feature steps consume.

Input:
  data/raw/mined_commits.parquet         — produced by 01_mine.py
  data/raw/kamei_promise/                — CSVs downloaded from PROMISE repo
      (qt.csv, mozilla.csv, jdt.csv, platform.csv, postgres.csv, safeftp.csv)

Output:
  data/processed/commits_clean.parquet   — cleaned, merged, balanced
  data/processed/schema.json             — column schema for documentation

Processing steps:
  1. Load Kamei CSVs and normalise column names to match mined schema.
  2. Load PyDriller output.
  3. Merge on commit_hash (outer join, dedup, Kamei labels take precedence).
  4. Drop rows where label == –1 (unlabeled mined commits that couldn't be
     labeled because they had no subsequent touching commit).
  5. Text cleaning: strip null bytes, control chars, normalise whitespace.
  6. Numeric sanity: clamp negative line counts to 0, fill NaN metrics with 0.
  7. Class balancing: undersample clean class so buggy ≥ 30% of total.
  8. Save parquet.

Usage:
  python pipeline/02_preprocess.py
  python pipeline/02_preprocess.py --mined data/raw/mined_commits.parquet \
      --kamei-dir data/raw/kamei_promise --output-dir data/processed \
      --min-buggy-ratio 0.30 --log-level DEBUG
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.utils import resample

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# Constants

# Regex to strip control characters except newlines/tabs
_CTRL_RE = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]")

# Kamei 14 change metrics column names (as they appear after rename)
KAMEI_METRICS: list[str] = [
    "NS", "ND", "NF", "Entropy", "LA", "LD", "LT",
    "FIX", "NOD", "NUC", "AGE", "EXP", "REXP", "SEXP",
]

# Projects shipped in the PROMISE Kamei dataset
KAMEI_PROJECTS: list[str] = [
    "qt", "mozilla", "jdt", "platform", "postgres", "safeftp",
]

# Expected columns after cleaning (union of mined + Kamei columns)
REQUIRED_OUTPUT_COLUMNS: list[str] = [
    "commit_hash", "repo", "source",
    "author_name", "author_email", "author_date",
    "commit_message", "files_changed", "lines_added", "lines_deleted",
    "modified_files", "is_merge",
    *KAMEI_METRICS,
    "label",
    "diff_text",   # empty string for PROMISE rows; real diff for PyDriller rows
]

# Logging

def _configure_logging(level: str) -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        level=getattr(logging, level.upper(), logging.INFO),
        stream=sys.stdout,
    )


logger = logging.getLogger("forge.preprocess")

# Kamei loader

# Column aliases for common PROMISE CSV header variants
_KAMEI_COL_MAP: dict[str, str] = {
    # raw CSV name          → canonical name
    "commit_id":            "commit_hash",
    "commitid":             "commit_hash",
    "rev":                  "commit_hash",
    "hash":                 "commit_hash",
    "author":               "author_name",
    "date":                 "author_date",
    "fix":                  "FIX",
    "ns":                   "NS",
    "nd":                   "ND",
    "nf":                   "NF",
    "entropy":              "Entropy",
    "entrophy":             "Entropy",   # common typo in Kamei PROMISE CSVs
    "la":                   "LA",
    "ld":                   "LD",
    "lt":                   "LT",
    "nod":                  "NOD",
    "ndev":                 "NOD",       # original Kamei CSVs use 'ndev'
    "nuc":                  "NUC",
    "npt":                  "NUC",       # 'number of prior touches' alias
    "age":                  "AGE",
    "exp":                  "EXP",
    "rexp":                 "REXP",
    "sexp":                 "SEXP",
    "buggy":                "label",
    "bug":                  "label",
    "defective":            "label",
}

# Fallback alias map: if a Kamei target column is all-zeros after primary rename,
# try copying from these unmapped source columns (order = preference).
_KAMEI_ALIAS_FALLBACK: dict[str, list[str]] = {
    "NOD":     ["ndev"],
    "NUC":     ["npt"],
    "Entropy": ["entrophy"],
    "ND":      ["ndiff", "n_dirs"],
    "AGE":     ["avg_age", "fileage", "file_age"],
}


def _load_kamei_csv(path: Path, project_name: str) -> pd.DataFrame:
    """Load one Kamei PROMISE CSV and normalise its schema."""
    logger.debug("  Loading Kamei CSV: %s", path)

    df = pd.read_csv(path, low_memory=False)

    # Lower-case all header names, then rename via alias map
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={k: v for k, v in _KAMEI_COL_MAP.items() if k in df.columns})

    # Ensure commit_hash column exists; if not, generate a surrogate
    if "commit_hash" not in df.columns:
        logger.warning(
            "    No commit_hash column in %s — generating surrogate IDs (kamei_%s_N).",
            path.name, project_name,
        )
        df["commit_hash"] = [f"kamei_{project_name}_{i}" for i in range(len(df))]

    # Source tag
    df["repo"] = f"kamei/{project_name}"
    df["source"] = "kamei"

    # Normalise label to int8
    if "label" in df.columns:
        df["label"] = (
            df["label"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": 1, "false": 0, "yes": 1, "no": 0, "1": 1, "0": 0})
            .fillna(0)
            .astype("int8")
        )
    else:
        logger.warning("    No label column in %s — defaulting all to 0.", path.name)
        df["label"] = np.int8(0)

    return df


def load_kamei_dataset(kamei_dir: Path) -> pd.DataFrame | None:
    """Load all available Kamei PROMISE CSVs from *kamei_dir*."""
    if not kamei_dir.exists():
        logger.warning(
            "Kamei directory not found: %s  — skipping Kamei data. "
            "Download CSVs from https://zenodo.org/record/322455 and place them here.",
            kamei_dir,
        )
        return None

    parts: list[pd.DataFrame] = []
    for project in KAMEI_PROJECTS:
        for ext in (".csv", ".CSV"):
            csv_path = kamei_dir / f"{project}{ext}"
            if csv_path.exists():
                try:
                    parts.append(_load_kamei_csv(csv_path, project))
                except Exception as exc:  # noqa: BLE001
                    logger.error("Failed to load %s: %s", csv_path, exc)
                break
        else:
            logger.debug("  Kamei CSV not found for project '%s' — skipping.", project)

    if not parts:
        logger.warning("No Kamei CSVs found in %s.", kamei_dir)
        return None

    df = pd.concat(parts, ignore_index=True)
    logger.info("Loaded %d rows from %d Kamei CSV(s).", len(df), len(parts))
    return df


# Text cleaning

def _clean_text(series: pd.Series) -> pd.Series:
    """Strip null bytes and control characters; normalise whitespace."""
    return (
        series
        .fillna("")
        .astype(str)
        .str.replace(_CTRL_RE, "", regex=True)
        .str.strip()
    )


# Merge & clean

def merge_and_clean(
    mined_df: pd.DataFrame | None,
    kamei_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Merge Kamei and mined dataframes. Kamei labels take precedence on overlap.
    """
    if mined_df is None and kamei_df is None:
        raise RuntimeError("Both mined_df and kamei_df are None — nothing to process.")

    parts: list[pd.DataFrame] = []

    if mined_df is not None:
        mined_df = mined_df.copy()
        mined_df["source"] = "pydriller"
        parts.append(mined_df)

    if kamei_df is not None:
        parts.append(kamei_df)

    df = pd.concat(parts, ignore_index=True)
    logger.info("Combined raw rows (before dedup): %d", len(df))

    # Deduplication — Kamei first so its labels win
    # Sort so kamei rows come first
    df["_sort"] = df["source"].map({"kamei": 0, "pydriller": 1}).fillna(2)
    df = df.sort_values("_sort").drop(columns=["_sort"])
    before = len(df)
    df = df.drop_duplicates(subset=["commit_hash"], keep="first")
    logger.info("Deduplication: %d -> %d rows (removed %d duplicates)",
                before, len(df), before - len(df))

    # Drop unlabeled rows
    before = len(df)
    df = df[df["label"] != -1].copy()
    logger.info("Dropped %d unlabeled rows (label == –1); remaining: %d",
                before - len(df), len(df))

    # Ensure all expected columns exist
    for col in REQUIRED_OUTPUT_COLUMNS:
        if col not in df.columns:
            if col in KAMEI_METRICS:
                df[col] = np.float32(0)
            elif col == "modified_files":
                df[col] = None   # will be fixed in the repair step below
            elif col == "is_merge":
                df[col] = False
            else:
                df[col] = ""

    # Repair modified_files
    # After concat, PROMISE rows have NaN for modified_files (the column
    # existed only in mined_df). Convert NaN → [] and ensure every cell is
    # a plain Python list (handles pyarrow scalar types from parquet reads).
    def _to_list(v: Any) -> list[str]:
        if v is None:
            return []
        try:
            if isinstance(v, float) and np.isnan(v):
                return []
        except (TypeError, ValueError):
            pass
        if isinstance(v, list):
            return [str(x) for x in v if x is not None and str(x).strip()]
        # PyArrow scalar: has .as_py() method
        if hasattr(v, "as_py"):
            py = v.as_py()
            return [str(x) for x in py if x is not None and str(x).strip()] \
                if isinstance(py, list) else []
        # Generic iterable (numpy array, etc.)
        try:
            lst = list(v)
            return [str(x) for x in lst if x is not None and str(x).strip()]
        except (TypeError, ValueError):
            pass
        if isinstance(v, str):
            import json as _json
            try:
                r = _json.loads(v)
                return [str(x) for x in r if x is not None] \
                    if isinstance(r, list) else []
            except Exception:  # noqa: BLE001
                pass
        return []

    df["modified_files"] = df["modified_files"].apply(_to_list)

    # Numeric sanity
    for col in ("files_changed", "lines_added", "lines_deleted"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0)

    for col in KAMEI_METRICS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("float32")

    # ── Alias rescue: fill zero Kamei columns from unmapped alias columns ──
    # This handles CSVs where 'ndev' was loaded but not yet mapped to NOD, etc.
    kamei_lower = {c.lower(): c for c in KAMEI_METRICS}
    for target, aliases in _KAMEI_ALIAS_FALLBACK.items():
        if target not in df.columns:
            continue
        # Only rescue if the column is entirely zero/null (i.e. mapping missed it)
        col_vals = pd.to_numeric(df[target], errors="coerce")
        if col_vals.fillna(0).eq(0).all():
            for alias in aliases:
                if alias in df.columns:
                    rescued = pd.to_numeric(df[alias], errors="coerce").fillna(0)
                    if rescued.gt(0).any():
                        logger.info(
                            "Alias rescue: copied '%s' -> '%s' (%d non-zero rows).",
                            alias, target, rescued.gt(0).sum(),
                        )
                        df[target] = rescued.astype("float32")
                        break

    # Text cleaning
    for col in ("commit_message", "author_name", "author_email", "repo"):
        df[col] = _clean_text(df[col])

    # Type coercion
    df["files_changed"] = df["files_changed"].astype("int32")
    df["lines_added"] = df["lines_added"].astype("int32")
    df["lines_deleted"] = df["lines_deleted"].astype("int32")
    df["is_merge"] = df["is_merge"].fillna(False).astype(bool)
    df["label"] = df["label"].astype("int8")

    # Normalise author_date
    df["author_date"] = pd.to_datetime(df["author_date"], utc=True, errors="coerce")

    return df.reset_index(drop=True)


# Class balancing

def balance_dataset(df: pd.DataFrame, min_buggy_ratio: float = 0.30) -> pd.DataFrame:
    """
    Undersample the clean class until buggy fraction ≥ min_buggy_ratio.
    Uses random_state=42 for reproducibility.
    """
    buggy = df[df["label"] == 1]
    clean = df[df["label"] == 0]
    n_buggy = len(buggy)
    n_clean = len(clean)

    current_ratio = n_buggy / max(n_buggy + n_clean, 1)
    logger.info(
        "Before balancing: buggy=%d, clean=%d, ratio=%.3f",
        n_buggy, n_clean, current_ratio,
    )

    if current_ratio >= min_buggy_ratio:
        logger.info("Buggy ratio already meets threshold (%.3f ≥ %.3f). No undersampling.",
                    current_ratio, min_buggy_ratio)
        return df

    # Target: buggy / (buggy + clean_target) == min_buggy_ratio
    # → clean_target = buggy * (1 - min_buggy_ratio) / min_buggy_ratio
    target_clean = int(n_buggy * (1 - min_buggy_ratio) / min_buggy_ratio)
    target_clean = max(target_clean, 1)

    if target_clean >= n_clean:
        logger.info("Not enough clean samples to undersample. Keeping all.")
        return df

    clean_downsampled = resample(
        clean,
        replace=False,
        n_samples=target_clean,
        random_state=42,
    )
    df_balanced = pd.concat([buggy, clean_downsampled], ignore_index=True)
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    n_final_buggy = (df_balanced["label"] == 1).sum()
    n_final_total = len(df_balanced)
    logger.info(
        "After balancing: buggy=%d, clean=%d, ratio=%.3f",
        n_final_buggy, n_final_total - n_final_buggy, n_final_buggy / n_final_total,
    )
    return df_balanced


# Schema export

def export_schema(df: pd.DataFrame, output_path: Path) -> None:
    """Write a human-readable JSON schema file for downstream documentation."""
    schema: dict[str, Any] = {}
    for col in df.columns:
        dtype = str(df[col].dtype)
        sample_vals: list[Any] = []
        try:
            sample_vals = df[col].dropna().head(3).tolist()
            # convert non-JSON-serialisable types
            sample_vals = [
                v if isinstance(v, (str, int, float, bool, type(None))) else str(v)
                for v in sample_vals
            ]
        except Exception:  # noqa: BLE001
            sample_vals = []

        schema[col] = {
            "dtype": dtype,
            "non_null": int(df[col].notna().sum()),
            "sample_values": sample_vals,
        }

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(schema, fh, indent=2, default=str)

    logger.info("Schema written -> %s", output_path)


# Orchestrator

def preprocess(
    mined_parquet: Path | None,
    kamei_dir: Path,
    output_dir: Path,
    min_buggy_ratio: float,
) -> Path:
    """
    Full preprocessing pipeline.
    Returns path to the output parquet.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_parquet = output_dir / "commits_clean.parquet"
    output_schema  = output_dir / "schema.json"

    # Load mined commits — prefer the diff-enriched version if it exists
    mined_df: pd.DataFrame | None = None
    if mined_parquet is not None:
        enriched = mined_parquet.parent / "mined_commits_with_diffs.parquet"
        if enriched.exists():
            logger.info(
                "Found enriched mined parquet — loading: %s", enriched
            )
            mined_df = pd.read_parquet(enriched, engine="pyarrow")
        elif mined_parquet.exists():
            logger.info("Loading mined commits: %s", mined_parquet)
            mined_df = pd.read_parquet(mined_parquet, engine="pyarrow")
        else:
            logger.warning("Mined parquet not found: %s", mined_parquet)

        if mined_df is not None:
            logger.info("  Loaded %d mined rows.", len(mined_df))
            if "diff_text" not in mined_df.columns:
                # Enriched file expected but column missing — fill with empty
                logger.warning("  diff_text column missing from mined parquet — filling with empty string.")
                mined_df["diff_text"] = ""

    kamei_df = load_kamei_dataset(kamei_dir)

    # Merge + clean
    df = merge_and_clean(mined_df, kamei_df)

    # PROMISE rows have no source code — give them an empty diff_text
    if "diff_text" in df.columns:
        df["diff_text"] = df["diff_text"].fillna("").astype(str)
        promise_mask = df["source"] == "kamei"
        df.loc[promise_mask, "diff_text"] = ""
        logger.info(
            "diff_text: %d non-empty (PyDriller), %d empty (PROMISE + fallback)",
            (df["diff_text"] != "").sum(),
            (df["diff_text"] == "").sum(),
        )
    else:
        # No diff text available at all — add empty column so downstream doesn't break
        logger.info("No diff_text available — adding empty column (run 01b_extract_diffs.py to populate).")
        df["diff_text"] = ""

    # Balance
    df = balance_dataset(df, min_buggy_ratio)

    # Final column ordering
    ordered_cols = REQUIRED_OUTPUT_COLUMNS + [
        c for c in df.columns if c not in REQUIRED_OUTPUT_COLUMNS
    ]
    df = df[[c for c in ordered_cols if c in df.columns]]

    # Save
    df.to_parquet(output_parquet, index=False, engine="pyarrow")
    logger.info(
        "Saved cleaned dataset -> %s  (%d rows x %d cols)",
        output_parquet, len(df), len(df.columns),
    )

    export_schema(df, output_schema)

    # Print schema summary to stdout
    _print_schema_summary(df)

    return output_parquet


def _print_schema_summary(df: pd.DataFrame) -> None:
    """Pretty-print the output parquet schema and stats."""
    sep = "-" * 72
    print(f"\n{sep}")
    print("  OUTPUT PARQUET SCHEMA -- data/processed/commits_clean.parquet")
    print(sep)
    print(f"  Rows    : {len(df):,}")
    print(f"  Columns : {len(df.columns)}")
    buggy = (df['label'] == 1).sum()
    clean = (df['label'] == 0).sum()
    print(f"  Label   : buggy={buggy:,}  clean={clean:,}  "
          f"ratio={buggy/(buggy+clean):.2%}")
    print(sep)
    print(f"  {'COLUMN':<30}  {'DTYPE':<22}  {'NON-NULL':>8}  {'NULL%':>6}")
    print(f"  {'------':<30}  {'-----':<22}  {'--------':>8}  {'-----':>6}")
    for col in df.columns:
        non_null = df[col].notna().sum()
        null_pct = 100 * (1 - non_null / max(len(df), 1))
        print(f"  {col:<30}  {str(df[col].dtype):<22}  {non_null:>8,}  {null_pct:>5.1f}%")
    print(sep)
    print()


# CLI entry point

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="02_preprocess.py",
        description="Merge Kamei + PyDriller data, clean, balance, and save parquet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mined",
        type=Path,
        default=Path("data/raw/mined_commits.parquet"),
        metavar="PARQUET",
        help="Path to the mined_commits.parquet produced by 01_mine.py.",
    )
    parser.add_argument(
        "--kamei-dir",
        type=Path,
        default=Path("data/raw/kamei_promise"),
        metavar="DIR",
        help=(
            "Directory containing Kamei PROMISE CSV files "
            "(qt.csv, mozilla.csv, jdt.csv, platform.csv, postgres.csv, safeftp.csv)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        metavar="DIR",
        help="Directory where commits_clean.parquet and schema.json will be written.",
    )
    parser.add_argument(
        "--min-buggy-ratio",
        type=float,
        default=0.30,
        metavar="RATIO",
        help="Minimum fraction of buggy samples after undersampling (0–1).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    _configure_logging(args.log_level)
    logger.info("forge-bug-predictor | pipeline step 02 — PREPROCESS")

    output_path = preprocess(
        mined_parquet=args.mined,
        kamei_dir=args.kamei_dir,
        output_dir=args.output_dir,
        min_buggy_ratio=args.min_buggy_ratio,
    )
    logger.info("Step 02 complete. Output: %s", output_path)


if __name__ == "__main__":
    main()
