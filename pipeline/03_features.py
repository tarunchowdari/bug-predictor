"""
pipeline/03_features.py

Compute all feature groups and produce the canonical feature matrix.

Input:  data/processed/commits_clean.parquet  (from 02_preprocess.py)
Output: data/processed/features.parquet       (all features + label)

Feature Groups:
  Group A — Kamei 14 change metrics (Kamei et al. 2013)
    NS, ND, NF, Entropy, LA, LD, LT, FIX, NOD, NUC, AGE, EXP, REXP, SEXP
    PROMISE rows: validated and used as-is.
    PyDriller rows: re-computed via a single ascending-date scan per repo.

  Group C — AST-derived features (Tree-sitter, Python + JS)
    n_nodes_added, n_nodes_deleted, n_control_flow_changes,
    n_function_changes, max_depth_before, max_depth_after,
    depth_delta, unique_node_types_added
    Reads from --source-cache-dir; defaults to 0 if cache is absent.

Usage:
  python pipeline/03_features.py
  python pipeline/03_features.py --input data/processed/commits_clean.parquet \
      --output data/processed/features.parquet \
      --source-cache-dir data/raw/source_cache
"""

from __future__ import annotations

import argparse
import logging
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Logging

def _configure_logging(level: str) -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        level=getattr(logging, level.upper(), logging.INFO),
        stream=sys.stdout,
    )


logger = logging.getLogger("forge.features")

# Constants

KAMEI_METRICS: list[str] = [
    "NS", "ND", "NF", "Entropy", "LA", "LD", "LT",
    "FIX", "NOD", "NUC", "AGE", "EXP", "REXP", "SEXP",
]

AST_METRICS: list[str] = [
    "n_nodes_added", "n_nodes_deleted", "n_control_flow_changes",
    "n_function_changes", "max_depth_before", "max_depth_after",
    "depth_delta", "unique_node_types_added",
]

BUG_RE = re.compile(r"\b(fix|bug|defect|patch|error|fault)\b", re.IGNORECASE)

# Kamei metric expected ranges for validation
KAMEI_RANGES: dict[str, tuple[float, float]] = {
    "NS":      (0, 1e6),
    "ND":      (0, 1e6),
    "NF":      (0, 1e6),
    "Entropy": (0, 100),    # log2(NF) is bounded; large values are suspicious
    "LA":      (0, 1e7),
    "LD":      (0, 1e7),
    "LT":      (0, 1e8),
    "FIX":     (0, 1),
    "NOD":     (0, 1e5),
    "NUC":     (0, 1e6),
    "AGE":     (0, 5000),   # days; ~14 years max
    "EXP":     (0, 1e5),
    "REXP":    (0, 1e4),
    "SEXP":    (0, 1e4),
}

# Tree-sitter node type sets — used for control-flow and function counting
CONTROL_FLOW_NODES: frozenset[str] = frozenset({
    "if_statement", "for_statement", "while_statement", "switch_statement",
    "try_statement", "with_statement", "except_clause",
    "if", "for", "while", "switch", "do_statement",
})
FUNCTION_NODES: frozenset[str] = frozenset({
    "function_definition", "method_definition", "function_declaration",
    "method_declaration", "arrow_function", "lambda",
})

# REXP decay: each prior author commit contributes 1/(1 + weeks_ago)
REXP_DECAY_UNIT = 7  # days per "unit" for recency decay

# Tree-sitter loader

_ts_parsers: dict[str, Any] = {}
_ts_available: bool | None = None  # None = not yet checked


def _get_ts_parser(lang: str) -> Any | None:
    """
    Lazily load a Tree-sitter parser. Returns None if Tree-sitter is
    unavailable or the grammar cannot be loaded — callers must handle None.
    """
    global _ts_available

    if _ts_available is False:
        return None

    if lang in _ts_parsers:
        return _ts_parsers[lang]

    try:
        import tree_sitter_python as tspython
        import tree_sitter_javascript as tsjavascript
        from tree_sitter import Language, Parser

        lang_map = {
            "python": Language(tspython.language()),
            "javascript": Language(tsjavascript.language()),
        }

        if lang not in lang_map:
            return None

        parser = Parser(lang_map[lang])
        _ts_parsers[lang] = parser
        _ts_available = True
        logger.debug("Tree-sitter parser loaded for: %s", lang)
        return parser

    except Exception as exc:  # noqa: BLE001
        if _ts_available is None:
            logger.warning(
                "Tree-sitter unavailable (%s). AST features will be 0. "
                "Install: pip install tree-sitter tree-sitter-python tree-sitter-javascript",
                exc,
            )
        _ts_available = False
        return None


def _detect_lang(filepath: str) -> str | None:
    """Infer language from file extension."""
    ext = Path(filepath).suffix.lower()
    if ext == ".py":
        return "python"
    if ext in (".js", ".jsx", ".ts", ".tsx", ".mjs"):
        return "javascript"
    return None


# AST helpers

def _iter_nodes(node: Any):
    """DFS iterator over Tree-sitter nodes."""
    yield node
    for child in node.children:
        yield from _iter_nodes(child)


def _tree_depth(node: Any) -> int:
    """Maximum depth of a Tree-sitter tree (1-indexed at root)."""
    if not node.children:
        return 1
    return 1 + max(_tree_depth(c) for c in node.children)


def _parse_source(source: str, lang: str) -> Any | None:
    """Parse *source* with Tree-sitter. Returns root node or None."""
    parser = _get_ts_parser(lang)
    if parser is None:
        return None
    try:
        tree = parser.parse(bytes(source, "utf-8"))
        return tree.root_node
    except Exception as exc:  # noqa: BLE001
        logger.debug("Tree-sitter parse error (%s): %s", lang, exc)
        return None


def _ast_features_for_file(
    source_before: str,
    source_after: str,
    lang: str,
) -> dict[str, float]:
    """
    Compute AST delta features for a single (before, after) file pair.
    Returns zero-filled dict on any parse failure.
    """
    zero = {k: 0.0 for k in AST_METRICS}

    root_before = _parse_source(source_before, lang)
    root_after  = _parse_source(source_after,  lang)

    if root_before is None or root_after is None:
        return zero

    try:
        # Node type counters
        types_before: Counter[str] = Counter(
            n.type for n in _iter_nodes(root_before)
        )
        types_after: Counter[str] = Counter(
            n.type for n in _iter_nodes(root_after)
        )

        n_before = sum(types_before.values())
        n_after  = sum(types_after.values())
        n_nodes_added   = max(0, n_after  - n_before)
        n_nodes_deleted = max(0, n_before - n_after)

        # Control-flow delta
        cf_before = sum(types_before[t] for t in CONTROL_FLOW_NODES)
        cf_after  = sum(types_after[t]  for t in CONTROL_FLOW_NODES)
        n_control_flow_changes = abs(cf_after - cf_before)

        # Function definition delta
        fn_before = sum(types_before[t] for t in FUNCTION_NODES)
        fn_after  = sum(types_after[t]  for t in FUNCTION_NODES)
        n_function_changes = abs(fn_after - fn_before)

        # Depth
        depth_before = _tree_depth(root_before)
        depth_after  = _tree_depth(root_after)

        # Unique node types added
        new_types = set(types_after.keys()) - set(types_before.keys())
        unique_node_types_added = float(len(new_types))

        return {
            "n_nodes_added":          float(n_nodes_added),
            "n_nodes_deleted":        float(n_nodes_deleted),
            "n_control_flow_changes": float(n_control_flow_changes),
            "n_function_changes":     float(n_function_changes),
            "max_depth_before":       float(depth_before),
            "max_depth_after":        float(depth_after),
            "depth_delta":            float(depth_after - depth_before),
            "unique_node_types_added": unique_node_types_added,
        }

    except Exception as exc:  # noqa: BLE001
        logger.debug("AST feature extraction error: %s", exc)
        return zero


def compute_ast_features_for_commit(
    commit_hash: str,
    modified_files: list[str],
    source_cache_dir: Path | None,
) -> dict[str, float]:
    """
    Average AST features across all modified files in a commit.
    Reads pre/post source from source_cache_dir/{commit_hash}/{filename}_before.txt
    and ..._after.txt. Returns zeros if cache is absent or files are missing.
    """
    zero = {k: 0.0 for k in AST_METRICS}

    if source_cache_dir is None or not source_cache_dir.exists():
        return zero

    commit_dir = source_cache_dir / commit_hash
    if not commit_dir.exists():
        return zero

    file_results: list[dict[str, float]] = []

    for filepath in modified_files:
        if not filepath:
            continue

        lang = _detect_lang(filepath)
        if lang is None:
            continue  # unsupported language → skip (not treated as failure)

        safe_name = filepath.replace("/", "__").replace("\\", "__")
        before_path = commit_dir / f"{safe_name}_before.txt"
        after_path  = commit_dir / f"{safe_name}_after.txt"

        try:
            source_before = before_path.read_text(encoding="utf-8", errors="replace") \
                if before_path.exists() else ""
            source_after  = after_path.read_text(encoding="utf-8", errors="replace") \
                if after_path.exists() else ""

            feats = _ast_features_for_file(source_before, source_after, lang)
            file_results.append(feats)

        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "AST compute error for %s in commit %s: %s",
                filepath, commit_hash[:8], exc,
            )
            file_results.append(zero)

    if not file_results:
        return zero

    # Average across all modified files
    averaged: dict[str, float] = {}
    for key in AST_METRICS:
        averaged[key] = float(np.mean([r[key] for r in file_results]))
    return averaged


# Kamei metric re-computation
# All 14 metrics come from a single ascending-date scan per repo.
# Rolling state dicts track file age, developer sets, line counts,
# and per-author experience — all updated AFTER each commit is processed.

def _extract_subsystem(filepath: str) -> str:
    """Top-level directory of a file path (the 'subsystem')."""
    parts = Path(filepath).parts
    return parts[0] if parts else ""


def _extract_directory(filepath: str) -> str:
    """Immediate parent directory of a file path."""
    return str(Path(filepath).parent)


def _compute_entropy(file_deltas: list[float]) -> float:
    """
    Shannon entropy H = -Σ p_i * log2(p_i) over file-level change proportions.
    p_i = |delta_i| / Σ|delta_j|.
    If all deltas are 0 or there is only one file, returns 0.
    """
    total = sum(abs(d) for d in file_deltas)
    if total == 0 or len(file_deltas) <= 1:
        return 0.0
    entropy = 0.0
    for d in file_deltas:
        p = abs(d) / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def _compute_rexp(
    author_commit_dates: list[datetime],
    current_date: datetime,
) -> float:
    """
    Recent experience: Σ_{c in past_commits} 1 / (1 + days_since(c) / UNIT)
    Each prior commit by the same author contributes a decaying weight.
    """
    rexp = 0.0
    for past_date in author_commit_dates:
        days_ago = max(0.0, (current_date - past_date).total_seconds() / 86400)
        rexp += 1.0 / (1.0 + days_ago / REXP_DECAY_UNIT)
    return rexp


def recompute_kamei_for_repo(
    repo_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Re-compute all 14 Kamei metrics for a single repo's commits
    using a single ascending-date sweep. Returns a copy with Kamei columns
    overwritten.

    The input df must be sorted by author_date ascending and contain columns:
      commit_hash, author_email, author_date, commit_message,
      files_changed, lines_added, lines_deleted, modified_files,
      [LA, LD already present from PROMISE for kamei rows]
    """
    df = repo_df.copy().sort_values("author_date").reset_index(drop=True)

    # ── Rolling state initialisation ──────────────────────────────────────────
    # file → date of most recent prior commit touching it
    file_last_date: dict[str, datetime] = {}
    # file → list of commit hashes (chronological)
    file_commits: dict[str, list[str]] = defaultdict(list)
    # file → set of developer emails (all time)
    file_developers: dict[str, set[str]] = defaultdict(set)
    # file → estimated current line count
    file_lines: dict[str, float] = defaultdict(float)

    # author → Counter of {subsystem: n_commits}
    author_subsystem_count: dict[str, Counter] = defaultdict(Counter)
    # author → sorted list of past commit datetimes (all subsystems)
    author_commit_dates: dict[str, list[datetime]] = defaultdict(list)
    # author → set of distinct subsystems ever touched
    author_subsystems_seen: dict[str, set[str]] = defaultdict(set)

    # ── Output arrays ─────────────────────────────────────────────────────────
    n = len(df)
    out_NS      = np.zeros(n, dtype=np.float32)
    out_ND      = np.zeros(n, dtype=np.float32)
    out_NF      = np.zeros(n, dtype=np.float32)
    out_Entropy = np.zeros(n, dtype=np.float32)
    out_LA      = np.zeros(n, dtype=np.float32)
    out_LD      = np.zeros(n, dtype=np.float32)
    out_LT      = np.zeros(n, dtype=np.float32)
    out_FIX     = np.zeros(n, dtype=np.float32)
    out_NOD     = np.zeros(n, dtype=np.float32)
    out_NUC     = np.zeros(n, dtype=np.float32)
    out_AGE     = np.zeros(n, dtype=np.float32)
    out_EXP     = np.zeros(n, dtype=np.float32)
    out_REXP    = np.zeros(n, dtype=np.float32)
    out_SEXP    = np.zeros(n, dtype=np.float32)

    for i, row in df.iterrows():
        files: list[str] = row["modified_files"] \
            if isinstance(row["modified_files"], list) else []
        files = [f for f in files if f]  # drop empty strings

        author: str = str(row["author_email"] or row["author_name"] or "unknown")
        cur_date: datetime = row["author_date"]
        if cur_date is pd.NaT or cur_date is None:
            cur_date = datetime(2000, 1, 1, tzinfo=timezone.utc)
        # Ensure tz-aware
        if cur_date.tzinfo is None:
            cur_date = cur_date.replace(tzinfo=timezone.utc)

        total_la = float(row.get("lines_added", 0) or 0)
        total_ld = float(row.get("lines_deleted", 0) or 0)
        nf = max(len(files), int(row.get("files_changed", 0) or 0))

        # NS — number of unique top-level subsystems touched
        subsystems: set[str] = {_extract_subsystem(f) for f in files}
        subsystems.discard("")
        ns = float(len(subsystems))

        # ND — number of unique directories touched
        directories: set[str] = {_extract_directory(f) for f in files}
        directories.discard(".")
        nd = float(len(directories))

        # NF
        out_NF[i] = float(nf)
        out_NS[i] = ns
        out_ND[i] = nd

        # Entropy — proportional to per-file change (uniform if no per-file data)
        if nf > 1:
            # Distribute total delta uniformly across files as best approximation
            per_file_delta = (total_la + total_ld) / nf
            file_deltas = [per_file_delta] * nf
            out_Entropy[i] = float(_compute_entropy(file_deltas))
        else:
            out_Entropy[i] = 0.0

        # LA, LD — directly from parquet
        out_LA[i] = total_la
        out_LD[i] = total_ld

        # LT — sum of estimated pre-commit line counts across modified files
        lt = sum(file_lines.get(f, 0.0) for f in files)
        out_LT[i] = float(lt)

        # FIX — bug keyword in message
        msg = str(row.get("commit_message", "") or "")
        out_FIX[i] = 1.0 if BUG_RE.search(msg) else 0.0

        # NOD — unique developers who touched these files BEFORE this commit
        nod_devs: set[str] = set()
        for f in files:
            nod_devs.update(file_developers.get(f, set()))
        out_NOD[i] = float(len(nod_devs))

        # NUC — total number of prior commits touching these files
        nuc = sum(len(file_commits.get(f, [])) for f in files)
        out_NUC[i] = float(nuc)

        # AGE — mean days since last modification across modified files
        age_values: list[float] = []
        for f in files:
            if f in file_last_date:
                delta = (cur_date - file_last_date[f]).total_seconds() / 86400
                age_values.append(max(0.0, delta))
        out_AGE[i] = float(np.mean(age_values)) if age_values else 0.0

        # EXP — author's total prior commits to any of the touched subsystems
        exp = sum(
            author_subsystem_count[author].get(ss, 0)
            for ss in subsystems
        )
        out_EXP[i] = float(exp)

        # REXP — recent experience weighted by recency
        rexp = _compute_rexp(author_commit_dates[author], cur_date)
        out_REXP[i] = float(rexp)

        # SEXP — number of distinct subsystems the author has touched before
        sexp = len(author_subsystems_seen[author])
        out_SEXP[i] = float(sexp)

        # ── Update rolling state AFTER computing features for this commit ─────
        commit_hash = str(row["commit_hash"])
        per_file_net = (total_la - total_ld) / max(nf, 1)

        for f in files:
            file_last_date[f] = cur_date
            file_commits[f].append(commit_hash)
            file_developers[f].add(author)
            file_lines[f] = max(0.0, file_lines.get(f, 0.0) + per_file_net)

        for ss in subsystems:
            author_subsystem_count[author][ss] += 1
            author_subsystems_seen[author].add(ss)

        author_commit_dates[author].append(cur_date)

    # Write back
    df["NS"]      = out_NS
    df["ND"]      = out_ND
    df["NF"]      = out_NF
    df["Entropy"] = out_Entropy
    df["LA"]      = out_LA
    df["LD"]      = out_LD
    df["LT"]      = out_LT
    df["FIX"]     = out_FIX
    df["NOD"]     = out_NOD
    df["NUC"]     = out_NUC
    df["AGE"]     = out_AGE
    df["EXP"]     = out_EXP
    df["REXP"]    = out_REXP
    df["SEXP"]    = out_SEXP

    return df


# PROMISE row validation

def validate_kamei_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """
    For PROMISE-sourced rows, validate Kamei columns against expected ranges.
    Clamp out-of-range values and log warnings (never silently corrupt data).
    """
    df = df.copy()
    for col, (lo, hi) in KAMEI_RANGES.items():
        if col not in df.columns:
            continue
        mask = (df["source"] == "kamei")
        suspect = mask & ((df[col] < lo) | (df[col] > hi))
        n_suspect = suspect.sum()
        if n_suspect > 0:
            logger.warning(
                "PROMISE validation: %d rows have %s outside [%.2f, %.2f]. "
                "Clamping.",
                n_suspect, col, lo, hi,
            )
            df.loc[mask, col] = df.loc[mask, col].clip(lower=lo, upper=hi)
    return df


# Kamei dispatcher

def compute_kamei_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dispatch Kamei computation:
      - PROMISE rows: validate + use pre-computed values
      - PyDriller rows: re-compute via rolling scan, per-repo
    Returns df with all 14 Kamei columns populated.
    """
    # Ensure Kamei columns exist with float32 dtype
    for col in KAMEI_METRICS:
        if col not in df.columns:
            df[col] = np.float32(0)
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("float32")

    # ── Validate PROMISE rows ─────────────────────────────────────────────────
    promise_mask = df["source"] == "kamei"
    n_promise = promise_mask.sum()
    logger.info("PROMISE rows to validate: %d", n_promise)
    if n_promise > 0:
        df = validate_kamei_ranges(df)

    # ── Re-compute for PyDriller rows, per-repo ───────────────────────────────
    pydriller_mask = df["source"] == "pydriller"
    n_pydriller = pydriller_mask.sum()
    logger.info("PyDriller rows to re-compute: %d", n_pydriller)

    if n_pydriller == 0:
        return df

    # Ensure author_date is tz-aware UTC
    df["author_date"] = pd.to_datetime(df["author_date"], utc=True, errors="coerce")

    pd_parts: list[pd.DataFrame] = []
    promise_parts: list[pd.DataFrame] = []

    for repo_name, repo_group in df.groupby("repo"):
        pd_rows = repo_group[repo_group["source"] == "pydriller"]
        pr_rows = repo_group[repo_group["source"] == "kamei"]

        if len(pd_rows) > 0:
            logger.info(
                "  Re-computing Kamei for %s (%d PyDriller commits) …",
                repo_name, len(pd_rows),
            )
            pd_computed = recompute_kamei_for_repo(pd_rows)
            pd_parts.append(pd_computed)

        if len(pr_rows) > 0:
            promise_parts.append(pr_rows)

    # Reassemble — preserve original order
    all_parts = pd_parts + promise_parts
    if all_parts:
        result = pd.concat(all_parts, ignore_index=True)
        # Restore original row order (by original index or date)
        result = result.sort_values(["repo", "author_date"]).reset_index(drop=True)
    else:
        result = df

    return result


# AST dispatcher

def compute_ast_features(
    df: pd.DataFrame,
    source_cache_dir: Path | None,
) -> pd.DataFrame:
    """
    Compute Group C AST features for every row. Adds 8 columns to df.
    Fills with 0 on any failure — never raises.
    """
    df = df.copy()
    for col in AST_METRICS:
        df[col] = np.float32(0)

    if source_cache_dir is None or not source_cache_dir.exists():
        logger.warning(
            "Source cache dir not found: %s. "
            "AST features will be 0 for all rows. "
            "To enable AST features, populate the cache with "
            "{commit_hash}/{file}_before.txt and ..._after.txt files.",
            source_cache_dir,
        )
        return df

    logger.info("Computing AST features from source cache: %s", source_cache_dir)
    n = len(df)
    ast_records: list[dict[str, float]] = []

    for i, row in df.iterrows():
        commit_hash = str(row["commit_hash"])
        mf_raw = row.get("modified_files", [])
        # Inline robust conversion (same logic as _ensure_list above)
        if mf_raw is None:
            modified_files: list[str] = []
        elif isinstance(mf_raw, list):
            modified_files = [str(x) for x in mf_raw if x is not None and str(x).strip()]
        elif hasattr(mf_raw, "as_py"):
            py = mf_raw.as_py()
            modified_files = [str(x) for x in py if x is not None and str(x).strip()] \
                if isinstance(py, list) else []
        else:
            try:
                modified_files = [str(x) for x in list(mf_raw) if x is not None and str(x).strip()]
            except (TypeError, ValueError):
                modified_files = []

        feats = compute_ast_features_for_commit(
            commit_hash, modified_files, source_cache_dir
        )
        ast_records.append(feats)

        if (i + 1) % 1000 == 0:
            logger.info("  AST: processed %d / %d commits …", i + 1, n)

    ast_df = pd.DataFrame(ast_records, index=df.index)
    for col in AST_METRICS:
        df[col] = ast_df[col].astype("float32")

    n_nonzero = (df[AST_METRICS].sum(axis=1) > 0).sum()
    logger.info(
        "AST features computed. Non-zero rows: %d / %d (%.1f%%)",
        n_nonzero, n, 100 * n_nonzero / max(n, 1),
    )
    return df


# Feature stats printer

def print_feature_stats(df: pd.DataFrame, feature_cols: list[str]) -> None:
    """
    Print a formatted statistics table for all feature columns:
    mean, std, min, max, null%.
    """
    sep = "─" * 86
    print(f"\n{sep}")
    print("  FEATURE STATISTICS TABLE — data/processed/features.parquet")
    print(sep)
    print(
        f"  {'FEATURE':<30}  {'MEAN':>12}  {'STD':>12}  "
        f"{'MIN':>12}  {'MAX':>12}  {'NULL%':>6}"
    )
    print(f"  {'───────':<30}  {'────':>12}  {'───':>12}  "
          f"{'───':>12}  {'───':>12}  {'─────':>6}")

    for col in feature_cols:
        if col not in df.columns:
            print(f"  {col:<30}  {'(missing)'}")
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        null_pct = 100 * s.isna().mean()
        s_clean = s.dropna()
        if len(s_clean) == 0:
            print(f"  {col:<30}  {'(all null)':>12}")
            continue
        mean_v = s_clean.mean()
        std_v  = s_clean.std()
        min_v  = s_clean.min()
        max_v  = s_clean.max()

        def _fmt(v: float) -> str:
            if abs(v) >= 1e6:
                return f"{v:.3e}"
            if abs(v) >= 100:
                return f"{v:>12.1f}"
            return f"{v:>12.4f}"

        print(
            f"  {col:<30}  {_fmt(mean_v)}  {_fmt(std_v)}  "
            f"{_fmt(min_v)}  {_fmt(max_v)}  {null_pct:>5.1f}%"
        )

    print(sep)
    # Label distribution
    if "label" in df.columns:
        buggy = (df["label"] == 1).sum()
        clean = (df["label"] == 0).sum()
        total = len(df)
        print(f"\n  Label distribution: total={total:,}  "
              f"buggy={buggy:,} ({100*buggy/total:.1f}%)  "
              f"clean={clean:,} ({100*clean/total:.1f}%)")
    print()


# Orchestrator

def engineer_features(
    input_parquet: Path,
    output_parquet: Path,
    source_cache_dir: Path | None = None,
) -> Path:
    """
    Full feature engineering pipeline.
    Returns path to the output features.parquet.
    """
    if not input_parquet.exists():
        raise RuntimeError(
            f"Input parquet not found: {input_parquet}\n"
            "Run pipeline/02_preprocess.py first."
        )

    logger.info("Loading: %s", input_parquet)
    df = pd.read_parquet(input_parquet, engine="pyarrow")
    logger.info("Loaded %d rows × %d cols", len(df), len(df.columns))

    # Ensure modified_files is always a plain Python list of non-empty strings.
    # Handles: Python list, PyArrow scalar (.as_py()), numpy array, JSON string, NaN.
    if "modified_files" in df.columns:
        def _ensure_list(v: Any) -> list[str]:
            if v is None:
                return []
            try:
                if isinstance(v, float) and np.isnan(v):
                    return []
            except (TypeError, ValueError):
                pass
            if isinstance(v, list):
                return [str(x) for x in v if x is not None and str(x).strip()]
            # PyArrow scalar exposes .as_py() — most common parquet read case
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
        df["modified_files"] = df["modified_files"].apply(_ensure_list)
        # Log a sample to confirm the column is populated correctly
        pd_mask = df.get("source", pd.Series("")) == "pydriller"
        n_pd = pd_mask.sum()
        if n_pd > 0:
            sample_len = df.loc[pd_mask, "modified_files"].apply(len).describe()
            logger.info(
                "modified_files length for PyDriller rows: "
                "mean=%.1f  max=%.0f  pct_empty=%.1f%%",
                sample_len["mean"], sample_len["max"],
                100 * (df.loc[pd_mask, "modified_files"].apply(len) == 0).mean(),
            )

    # Ensure source column exists (for PROMISE vs PyDriller dispatch)
    if "source" not in df.columns:
        logger.warning("'source' column missing — defaulting all rows to 'pydriller'.")
        df["source"] = "pydriller"

    # ── Group A: Kamei 14 metrics ─────────────────────────────────────────────
    logger.info("Computing Group A: Kamei 14 change metrics …")
    df = compute_kamei_features(df)
    logger.info("Group A complete.")

    # ── Group C: AST features ─────────────────────────────────────────────────
    logger.info("Computing Group C: AST-derived features …")
    df = compute_ast_features(df, source_cache_dir)
    logger.info("Group C complete.")

    # ── Select & order feature columns ────────────────────────────────────────
    feature_cols = KAMEI_METRICS + AST_METRICS
    meta_cols = [
        "commit_hash", "repo", "source", "author_name", "author_email",
        "author_date", "commit_message", "modified_files",
        "files_changed", "lines_added", "lines_deleted", "is_merge",
        "label",
    ]
    all_cols = meta_cols + [c for c in feature_cols if c not in meta_cols]
    available = [c for c in all_cols if c in df.columns]
    df = df[available]

    # ── Cast feature columns to float32 ──────────────────────────────────────
    for col in feature_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("float32")

    # ── Save ─────────────────────────────────────────────────────────────────
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_parquet, index=False, engine="pyarrow")
    logger.info(
        "Saved features → %s  (%d rows × %d cols)",
        output_parquet, len(df), len(df.columns),
    )

    # ── Print stats table ─────────────────────────────────────────────────────
    print_feature_stats(df, feature_cols)

    return output_parquet


# CLI

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="03_features.py",
        description=(
            "Compute Kamei 14 change metrics and AST-derived features "
            "from the cleaned commit parquet."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/commits_clean.parquet"),
        metavar="PARQUET",
        help="Input parquet produced by 02_preprocess.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/features.parquet"),
        metavar="PARQUET",
        help="Output path for the feature matrix parquet.",
    )
    parser.add_argument(
        "--source-cache-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Optional directory containing pre-extracted source files for AST parsing. "
            "Structure: {dir}/{commit_hash}/{filepath}_before.txt and ..._after.txt. "
            "If absent, AST features default to 0."
        ),
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
    logger.info("forge-bug-predictor | pipeline step 03 — FEATURES")
    logger.info("Input          : %s", args.input)
    logger.info("Output         : %s", args.output)
    logger.info("Source cache   : %s", args.source_cache_dir or "(none — AST will be 0)")

    output_path = engineer_features(
        input_parquet=args.input,
        output_parquet=args.output,
        source_cache_dir=args.source_cache_dir,
    )
    logger.info("Step 03 complete. Output: %s", output_path)


if __name__ == "__main__":
    main()
