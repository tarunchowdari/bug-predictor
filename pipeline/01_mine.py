"""
pipeline/01_mine.py
Mine commits from public GitHub repositories using PyDriller.

Labeling strategy:
  A commit is labeled buggy (1) if:
    1. Its message matches the bug-fix regex  (fix|bug|defect|patch|error|fault)
    2. At least one subsequent commit within 90 days touches the same file(s).
  Otherwise the commit is labeled clean (0).

Output:
  data/raw/mined_commits.parquet  — one row per commit, schema documented below

Usage:
  python pipeline/01_mine.py --output-dir data/raw --max-commits 5000
  python pipeline/01_mine.py --repos pytorch/pytorch django/django --log-level DEBUG
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from pydriller import Repository

# Constants

BUG_REGEX = re.compile(
    r"\b(fix|bug|defect|patch|error|fault)\b", re.IGNORECASE
)

LOOKFORWARD_DAYS: int = 90
DEFAULT_MAX_COMMITS: int = 5_000

DEFAULT_REPOS: list[str] = [
    "https://github.com/pytorch/pytorch",
    "https://github.com/tensorflow/tensorflow",
    "https://github.com/django/django",
    "https://github.com/apache/kafka",
]

# Columns produced by this script (schema contract for downstream steps)
SCHEMA: dict[str, str] = {
    "commit_hash": "string",
    "repo": "string",
    "author_name": "string",
    "author_email": "string",
    "author_date": "datetime64[us, UTC]",
    "commit_message": "string",
    "files_changed": "int32",
    "lines_added": "int32",
    "lines_deleted": "int32",
    "modified_files": "object",      # list[str] — serialised as JSON string
    "dmm_unit_size": "float32",
    "dmm_unit_complexity": "float32",
    "dmm_unit_interfacing": "float32",
    "is_merge": "bool",
    "label": "int8",                 # 1 = buggy, 0 = clean (–1 = unlabeled)
}

# Logging setup

def _configure_logging(level: str) -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        level=getattr(logging, level.upper(), logging.INFO),
        stream=sys.stdout,
    )


logger = logging.getLogger("forge.mine")

# Core mining logic

def _normalise_repo_url(repo: str) -> str:
    """Accept 'owner/name' short form or full HTTPS URL."""
    if repo.startswith("http"):
        return repo
    return f"https://github.com/{repo}"


def _mine_single_repo(
    repo_url: str,
    max_commits: int,
) -> list[dict[str, Any]]:
    """
    Mine up to *max_commits* commits from *repo_url*.
    Returns a list of raw commit records (no labeling yet).
    """
    repo_name = repo_url.rstrip("/").split("/")[-2] + "/" + repo_url.rstrip("/").split("/")[-1]
    logger.info("Mining %s (cap=%d) …", repo_name, max_commits)

    records: list[dict[str, Any]] = []

    try:
        traversal = Repository(repo_url, order="reverse")  # newest first → faster cap
        for i, commit in enumerate(traversal.traverse_commits()):
            if i >= max_commits:
                logger.info("  Reached cap of %d commits for %s", max_commits, repo_name)
                break

            try:
                modified_files: list[str] = [
                    mf.new_path or mf.old_path or ""
                    for mf in commit.modified_files
                ]
                records.append(
                    {
                        "commit_hash": commit.hash,
                        "repo": repo_name,
                        "author_name": commit.author.name or "",
                        "author_email": commit.author.email or "",
                        "author_date": commit.author_date,
                        "commit_message": (commit.msg or "").strip(),
                        "files_changed": len(commit.modified_files),
                        "lines_added": commit.insertions,
                        "lines_deleted": commit.deletions,
                        "modified_files": modified_files,
                        "dmm_unit_size": commit.dmm_unit_size or 0.0,
                        "dmm_unit_complexity": commit.dmm_unit_complexity or 0.0,
                        "dmm_unit_interfacing": commit.dmm_unit_interfacing or 0.0,
                        "is_merge": commit.merge,
                        "label": -1,  # filled in during labeling pass
                    }
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("  Skipping commit %s: %s", commit.hash[:8], exc)
                continue

            if (i + 1) % 500 == 0:
                logger.info("  … %d commits collected from %s", i + 1, repo_name)

    except Exception as exc:
        logger.error("Failed to mine %s: %s", repo_url, exc)
        raise RuntimeError(f"Mining failed for {repo_url}: {exc}") from exc

    logger.info("  Collected %d raw commits from %s", len(records), repo_name)
    return records


# Labeling pass

def _apply_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorised labeling across an entire repo's commits.

    A commit is buggy (1) if:
      a) its message matches BUG_REGEX, AND
      b) at least one later commit (within LOOKFORWARD_DAYS) touches the same files.

    This runs per-repo so that the look-forward window is scoped correctly.
    """
    # Ensure author_date is tz-aware UTC for comparison
    df = df.copy()
    df["author_date"] = pd.to_datetime(df["author_date"], utc=True)
    df = df.sort_values("author_date").reset_index(drop=True)

    # Pre-compute bug-regex match flag
    has_bug_keyword = df["commit_message"].str.contains(BUG_REGEX, na=False)

    # Build per-commit file sets
    file_sets: list[set[str]] = [
        set(files) for files in df["modified_files"].tolist()
    ]

    labels = pd.Series(0, index=df.index, dtype="int8")
    lookforward = timedelta(days=LOOKFORWARD_DAYS)

    for idx in df.index:
        if not has_bug_keyword.iloc[idx]:
            labels.iloc[idx] = 0
            continue

        commit_date: datetime = df["author_date"].iloc[idx]
        commit_files: set[str] = file_sets[idx]

        if not commit_files:
            labels.iloc[idx] = 0
            continue

        # Search forward in time for any commit that touches the same files
        found_recurrence = False
        for jdx in range(idx + 1, len(df)):
            later_date: datetime = df["author_date"].iloc[jdx]
            if later_date - commit_date > lookforward:
                break  # sorted by date, so no point continuing
            if commit_files & file_sets[jdx]:  # non-empty intersection
                found_recurrence = True
                break

        labels.iloc[idx] = 1 if found_recurrence else 0

    df["label"] = labels.values.astype("int8")
    return df


# Orchestrator

def mine_repos(
    repos: list[str],
    output_dir: Path,
    max_commits: int = DEFAULT_MAX_COMMITS,
) -> Path:
    """
    Mine all repos, apply labeling, deduplicate, and save to parquet.
    Returns the path to the output parquet file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "mined_commits.parquet"

    all_records: list[dict[str, Any]] = []

    for repo in repos:
        url = _normalise_repo_url(repo)
        try:
            records = _mine_single_repo(url, max_commits)
        except RuntimeError as exc:
            logger.error("Skipping repo after failure: %s", exc)
            continue
        all_records.extend(records)

    if not all_records:
        raise RuntimeError(
            "No commits were collected from any repository. "
            "Check network connectivity and repo URLs."
        )

    logger.info("Total raw commits collected: %d", len(all_records))

    df = pd.DataFrame(all_records)

    # Deduplication on commit hash
    before = len(df)
    df = df.drop_duplicates(subset=["commit_hash"], keep="first")
    logger.info("Deduplication: %d → %d rows (removed %d duplicates)",
                before, len(df), before - len(df))

    # Per-repo labeling
    labeled_parts: list[pd.DataFrame] = []
    for repo_name, group in df.groupby("repo"):
        logger.info("Labeling %s (%d commits) …", repo_name, len(group))
        labeled_parts.append(_apply_labels(group))

    df = pd.concat(labeled_parts, ignore_index=True)

    # Type coercion to declared schema
    df["files_changed"] = df["files_changed"].astype("int32")
    df["lines_added"] = df["lines_added"].astype("int32")
    df["lines_deleted"] = df["lines_deleted"].astype("int32")
    df["dmm_unit_size"] = df["dmm_unit_size"].astype("float32")
    df["dmm_unit_complexity"] = df["dmm_unit_complexity"].astype("float32")
    df["dmm_unit_interfacing"] = df["dmm_unit_interfacing"].astype("float32")
    df["is_merge"] = df["is_merge"].astype(bool)
    df["modified_files"] = df["modified_files"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    # Summary statistics
    total = len(df)
    buggy = (df["label"] == 1).sum()
    clean = (df["label"] == 0).sum()
    logger.info(
        "Label distribution: total=%d  buggy=%d (%.1f%%)  clean=%d (%.1f%%)",
        total, buggy, 100 * buggy / total, clean, 100 * clean / total,
    )

    per_repo = df.groupby("repo")["label"].value_counts().unstack(fill_value=0)
    logger.info("Per-repo label counts:\n%s", per_repo.to_string())

    # Save
    df.to_parquet(output_path, index=False, engine="pyarrow")
    logger.info("Saved mined commits → %s  (%d rows, %d cols)",
                output_path, len(df), len(df.columns))

    return output_path


# CLI entry point

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="01_mine.py",
        description="Mine GitHub commits via PyDriller and produce a labeled parquet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--repos",
        nargs="+",
        default=DEFAULT_REPOS,
        metavar="REPO",
        help=(
            "Repository URLs or 'owner/name' shorthand. "
            "Defaults to pytorch, tensorflow, django, kafka."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        metavar="DIR",
        help="Directory where mined_commits.parquet will be written.",
    )
    parser.add_argument(
        "--max-commits",
        type=int,
        default=DEFAULT_MAX_COMMITS,
        metavar="N",
        help="Maximum commits to mine per repository.",
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
    logger.info("forge-bug-predictor | pipeline step 01 — MINE")
    logger.info("Repositories : %s", args.repos)
    logger.info("Max commits  : %d per repo", args.max_commits)
    logger.info("Output dir   : %s", args.output_dir)

    output_path = mine_repos(
        repos=args.repos,
        output_dir=args.output_dir,
        max_commits=args.max_commits,
    )
    logger.info("Step 01 complete. Output: %s", output_path)


if __name__ == "__main__":
    main()
