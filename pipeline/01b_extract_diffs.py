"""
pipeline/01b_extract_diffs.py

Add diff_text to the existing mined commits without re-mining.
Reads the already-cloned repos from data/repos/ and uses GitPython
to extract per-commit diffs. Saves an enriched parquet that
02_preprocess.py will prefer over the plain mined_commits.parquet.

Input:  data/raw/mined_commits.parquet  (from 01_mine.py)
        data/repos/{owner}__{name}/     (cloned by 01_mine.py)

Output: data/raw/mined_commits_with_diffs.parquet
        — same schema as input plus a new `diff_text` column

Diff rules:
  - Only .py and .js file diffs are included (skip binary/config/docs)
  - Total diff text per commit is capped at 2000 characters (truncated)
  - Initial commits (no parents) → empty string
  - Any extraction error → empty string + warning logged

Usage:
  python pipeline/01b_extract_diffs.py
  python pipeline/01b_extract_diffs.py \\
      --input  data/raw/mined_commits.parquet \\
      --output data/raw/mined_commits_with_diffs.parquet \\
      --repos-dir data/repos \\
      --max-diff-chars 2000 \\
      --log-level INFO
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
from pathlib import Path
from typing import Any

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

import pandas as pd
from tqdm import tqdm

# Constants

DIFF_CAP = 2000  # characters per commit, hard truncate

# Only these extensions carry signal we care about
SOURCE_EXTENSIONS = {".py", ".js"}

# Maps repo full_name (as stored in mined parquet) → folder name under data/repos/
REPO_DIR_MAP: dict[str, str] = {
    "apache/kafka":                  "apache__kafka",
    "django/django":                 "django__django",
    "pallets/flask":                 "pallets__flask",
    "psf/black":                     "psf__black",
    "requests/requests":             "requests__requests",
    "scikit-learn/scikit-learn":     "scikit-learn__scikit-learn",
}

# Logging

def _configure_logging(level: str) -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        level=getattr(logging, level.upper(), logging.INFO),
        stream=sys.stdout,
    )


logger = logging.getLogger("forge.diffs")

# Git repo cache — one gitpython Repo object per folder, reused across commits

def _open_repos(repos_dir: Path, df: pd.DataFrame) -> dict[str, Any]:
    """Open every unique repo referenced in df. Missing repos get a warning."""
    try:
        import git  # gitpython
    except ImportError as exc:
        raise RuntimeError(
            "GitPython is not installed. Run: pip install gitpython"
        ) from exc

    opened: dict[str, Any] = {}
    for repo_name in df["repo"].unique():
        folder = REPO_DIR_MAP.get(repo_name)
        if folder is None:
            logger.warning("No folder mapping for repo '%s' — will skip those rows.", repo_name)
            continue
        repo_path = repos_dir / folder
        if not repo_path.exists():
            logger.warning("Repo path not found: %s — will skip '%s'.", repo_path, repo_name)
            continue
        try:
            opened[repo_name] = git.Repo(str(repo_path))
            logger.info("  Opened: %s -> %s", repo_name, repo_path)
        except Exception as exc:
            logger.warning("  Failed to open %s: %s", repo_path, exc)

    return opened


# Diff extraction

def _extract_diff(git_repo: Any, commit_hash: str, max_chars: int) -> str:
    """
    Return concatenated unified-diff text for a commit, capped at max_chars.
    Only .py/.js files are included. Returns "" on any error.

    Must use parent.diff(commit, create_patch=True) — that's the only form
    GitPython uses to populate the raw patch bytes in diff_item.diff.
    """
    try:
        commit = git_repo.commit(commit_hash)
    except Exception as exc:
        logger.warning("  Cannot resolve %s: %s", commit_hash[:10], exc)
        return ""

    # Initial commit — no parent to diff against
    if not commit.parents:
        return ""

    parent = commit.parents[0]
    try:
        # parent.diff(commit, create_patch=True) gives us parent->commit changes
        # with the unified diff text populated in each diff_item.diff
        diffs = parent.diff(commit, create_patch=True)
    except Exception as exc:
        logger.warning("  diff() failed for %s: %s", commit_hash[:10], exc)
        return ""

    parts: list[str] = []
    total = 0

    for diff_item in diffs:
        # Filter to source files only — check both sides of the rename/add/delete
        b_path = diff_item.b_path or ""
        a_path = diff_item.a_path or ""
        ext_b = Path(b_path).suffix.lower()
        ext_a = Path(a_path).suffix.lower()
        if ext_b not in SOURCE_EXTENSIONS and ext_a not in SOURCE_EXTENSIONS:
            continue

        try:
            raw = diff_item.diff
            if not raw:
                continue
            if isinstance(raw, bytes):
                text = raw.decode("utf-8", errors="replace")
            else:
                text = str(raw)
        except Exception:
            continue

        remaining = max_chars - total
        if remaining <= 0:
            break

        chunk = text[:remaining]
        parts.append(chunk)
        total += len(chunk)

        if total >= max_chars:
            break

    return "".join(parts)


# Main pipeline

def extract_diffs(
    input_parquet: Path,
    output_parquet: Path,
    repos_dir: Path,
    max_diff_chars: int,
) -> None:
    logger.info("Loading: %s", input_parquet)
    df = pd.read_parquet(input_parquet, engine="pyarrow")
    logger.info("  %d rows, repos: %s", len(df), df["repo"].unique().tolist())

    logger.info("Opening repos from %s …", repos_dir)
    git_repos = _open_repos(repos_dir, df)
    logger.info("  %d repo(s) opened.", len(git_repos))

    diff_texts: list[str] = []
    errors = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting diffs", unit="commit"):
        repo_name   = row["repo"]
        commit_hash = row["commit_hash"]

        git_repo = git_repos.get(repo_name)
        if git_repo is None:
            diff_texts.append("")
            continue

        text = _extract_diff(git_repo, commit_hash, max_diff_chars)
        if not text:
            errors += 1
        diff_texts.append(text)

    df["diff_text"] = diff_texts

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_parquet, index=False, engine="pyarrow")
    logger.info("Saved -> %s", output_parquet)

    # Summary
    non_empty  = sum(1 for t in diff_texts if t)
    mean_len   = sum(len(t) for t in diff_texts) / max(non_empty, 1)
    max_len    = max((len(t) for t in diff_texts), default=0)
    pct        = 100 * non_empty / max(len(diff_texts), 1)

    print()
    print("-- Diff Extraction Summary --------------------------------")
    print(f"  Total commits       : {len(df):,}")
    print(f"  With non-empty diff : {non_empty:,}  ({pct:.1f}%)")
    print(f"  Empty / failed      : {errors:,}")
    print(f"  Mean diff length    : {mean_len:,.0f} chars")
    print(f"  Max diff length     : {max_len:,} chars  (cap={max_diff_chars})")
    print(f"  Output              : {output_parquet}")
    print()


# CLI

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="01b_extract_diffs.py",
        description="Add diff_text column to mined_commits.parquet using cloned repos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/mined_commits.parquet"),
        metavar="PARQUET",
        help="Input mined commits parquet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/mined_commits_with_diffs.parquet"),
        metavar="PARQUET",
        help="Output path for enriched parquet with diff_text column.",
    )
    parser.add_argument(
        "--repos-dir",
        type=Path,
        default=Path("data/repos"),
        metavar="DIR",
        help="Root directory containing the cloned repos (one subdir per repo).",
    )
    parser.add_argument(
        "--max-diff-chars",
        type=int,
        default=DIFF_CAP,
        metavar="N",
        help="Hard character cap per commit diff (excess is truncated).",
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
    logger.info("forge-bug-predictor | pipeline step 01b — EXTRACT DIFFS")

    extract_diffs(
        input_parquet=args.input,
        output_parquet=args.output,
        repos_dir=args.repos_dir,
        max_diff_chars=args.max_diff_chars,
    )
    logger.info("Step 01b complete.")


if __name__ == "__main__":
    main()
