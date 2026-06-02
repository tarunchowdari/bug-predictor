"""
pipeline/03b_build_source_cache.py
Build the source-code cache needed for AST feature extraction in 03_features.py.

For every PyDriller commit in commits_clean.parquet, and for each .py / .js
file it modified, this script writes two files:

  data/source_cache/{commit_hash}/{safe_filepath}_before.txt
  data/source_cache/{commit_hash}/{safe_filepath}_after.txt

Key design:
  • Blob reading uses GitPython's tree / path OPERATOR (not tree[path]) which
    correctly performs recursive tree traversal for nested file paths.
  • Path separators are normalised to forward-slashes before tree lookup.
  • A file pair is only written when at least ONE side has ≥ MIN_FILE_BYTES.
  • Progress is logged every 500 commits showing file_pairs / commits_done.
  • Idempotent: commits whose cache directory already contains files are
    skipped unless --force is passed.

Filtering rules:
  • Only process source == 'pydriller' rows
  • Only .py and .js/.jsx/.ts/.tsx/.mjs files
  • Skip binary blobs (null-byte in first 8 KB)
  • Skip blobs larger than 500 KB
  • Cap at --max-files-per-commit (default 50) per commit

Note on Java repos (e.g. apache/kafka):
  These are mined as PyDriller commits but .java files are skipped by the
  extension filter. Kafka will contribute 0 file pairs — this is expected
  and will result in zero AST features for those commits (filled with 0).

Usage:
  python pipeline/03b_build_source_cache.py
  python pipeline/03b_build_source_cache.py \
      --input data/processed/commits_clean.parquet \
      --output-dir data/source_cache \
      --repos-dir data/repos \
      --max-files-per-commit 50 \
      --log-level INFO
  python pipeline/03b_build_source_cache.py --force   # re-cache everything
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

# Logging

def _configure_logging(level: str) -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        level=getattr(logging, level.upper(), logging.INFO),
        stream=sys.stdout,
    )


logger = logging.getLogger("forge.source_cache")

# Constants

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".py", ".js", ".jsx", ".ts", ".tsx", ".mjs"}
)
BINARY_SNIFF_BYTES = 8192       # bytes to scan for null-byte binary detection
MAX_FILE_SIZE_BYTES = 512_000   # 500 KB per file — skip larger blobs
MIN_FILE_BYTES = 10             # minimum content bytes for a pair to count
PROGRESS_EVERY = 500            # log a summary line every N commits

# Repo URL resolver

_REPO_URL_MAP: dict[str, str] = {
    "apache/kafka":              "https://github.com/apache/kafka.git",
    "django/django":             "https://github.com/django/django.git",
    "pallets/flask":             "https://github.com/pallets/flask.git",
    "psf/black":                 "https://github.com/psf/black.git",
    "requests/requests":         "https://github.com/psf/requests.git",
    "scikit-learn/scikit-learn": "https://github.com/scikit-learn/scikit-learn.git",
    "pytorch/pytorch":           "https://github.com/pytorch/pytorch.git",
    "tensorflow/tensorflow":     "https://github.com/tensorflow/tensorflow.git",
}


def _clone_url_for(repo_name: str) -> str | None:
    if repo_name in _REPO_URL_MAP:
        return _REPO_URL_MAP[repo_name]
    parts = repo_name.strip("/").split("/")
    if len(parts) == 2:
        url = f"https://github.com/{parts[0]}/{parts[1]}.git"
        logger.info("No explicit URL for '%s' — guessing: %s", repo_name, url)
        return url
    return None


# Git helpers

def _ensure_repo(repo_name: str, repos_dir: Path) -> Any | None:
    """Return a GitPython Repo, cloning if needed."""
    try:
        import git as gitmodule
    except ImportError:
        logger.error("GitPython not installed. Run: pip install gitpython")
        return None

    safe_name = repo_name.strip("/").replace("/", "__")
    local_path = repos_dir / safe_name

    if local_path.exists() and (local_path / ".git").exists():
        logger.debug("Reusing local clone: %s", local_path)
        try:
            return gitmodule.Repo(str(local_path))
        except gitmodule.exc.InvalidGitRepositoryError:
            logger.warning("Invalid repo at %s — re-cloning.", local_path)

    clone_url = _clone_url_for(repo_name)
    if clone_url is None:
        logger.error(
            "Cannot resolve clone URL for '%s'. Add to _REPO_URL_MAP.", repo_name
        )
        return None

    logger.info("Cloning %s → %s …", clone_url, local_path)
    local_path.mkdir(parents=True, exist_ok=True)
    try:
        import git as gitmodule
        repo = gitmodule.Repo.clone_from(
            clone_url, str(local_path), no_single_branch=True
        )
        logger.info("Clone complete: %s", local_path)
        return repo
    except Exception as exc:  # noqa: BLE001
        logger.error("Clone failed for %s: %s", clone_url, exc)
        return None


def _is_binary(data: bytes) -> bool:
    return b"\x00" in data[:BINARY_SNIFF_BYTES]


def _get_blob(tree: Any, filepath: str) -> Any | None:
    """
    Robustly retrieve a blob from a git Tree by path.

    Uses the GitPython `/` operator (Tree.__truediv__) which performs
    RECURSIVE tree traversal — unlike tree[key] which only looks at the
    root level and fails for all nested paths.

    filepath is normalised to forward-slashes before lookup.
    Returns None on any error.
    """
    clean = filepath.replace("\\", "/").strip("/")
    if not clean:
        return None
    try:
        blob = tree / clean
        # Confirm it's actually a blob (not a subtree)
        if hasattr(blob, "data_stream"):
            return blob
        return None
    except (KeyError, AttributeError):
        return None
    except Exception:  # noqa: BLE001
        return None


def _read_blob(blob: Any) -> str | None:
    """
    Decode a GitPython Blob to a string.
    Returns None if the blob is binary, too large, or unreadable.
    """
    if blob is None:
        return None
    try:
        size = blob.size
        if size > MAX_FILE_SIZE_BYTES:
            return None
        data = blob.data_stream.read()
        if _is_binary(data):
            return None
        return data.decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return None


def _safe_filepath(filepath: str) -> str:
    return (
        filepath
        .replace("/", "__")
        .replace("\\", "__")
        .replace(":", "_")
    )


# Core caching logic

def _process_commit(
    repo: Any,
    commit_hash: str,
    modified_files: list[str],
    cache_dir: Path,
    max_files: int,
    err_counts: dict[str, int],
) -> int:
    """
    Extract before/after source for supported files in a commit.
    Returns the number of file-pairs actually written (both sides ≥ MIN_FILE_BYTES).
    """
    import git as gitmodule

    # Resolve commit object
    try:
        commit = repo.commit(commit_hash)
    except (gitmodule.exc.BadName, ValueError, Exception) as exc:  # noqa: BLE001
        err_counts["commit_not_found"] += 1
        logger.debug("Commit not found (%s): %s", commit_hash[:8], exc)
        return 0

    parent = commit.parents[0] if commit.parents else None

    commit_cache = cache_dir / commit_hash
    commit_cache.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_skipped_ext = 0
    n_skipped_empty = 0

    for filepath in modified_files[:max_files]:
        ext = Path(filepath).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            n_skipped_ext += 1
            continue

        # After blob (this commit's version of the file)
        after_blob = _get_blob(commit.tree, filepath)
        after_text = _read_blob(after_blob) or ""

        # Before blob (parent commit's version)
        before_text = ""
        if parent is not None:
            before_blob = _get_blob(parent.tree, filepath)
            before_text = _read_blob(before_blob) or ""

        # Only write pairs where at least one side meets the minimum size
        has_before = len(before_text.encode("utf-8", errors="replace")) >= MIN_FILE_BYTES
        has_after  = len(after_text.encode("utf-8",  errors="replace")) >= MIN_FILE_BYTES

        if has_before or has_after:
            safe = _safe_filepath(filepath)
            (commit_cache / f"{safe}_before.txt").write_text(
                before_text, encoding="utf-8"
            )
            (commit_cache / f"{safe}_after.txt").write_text(
                after_text, encoding="utf-8"
            )
            n_written += 1
        else:
            n_skipped_empty += 1

    if n_skipped_ext > 0:
        err_counts["skipped_unsupported_ext"] += n_skipped_ext
    if n_skipped_empty > 0:
        err_counts["skipped_empty_pair"] += n_skipped_empty

    return n_written


# Modified-files parser

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
    if hasattr(v, "as_py"):
        py = v.as_py()
        return [str(x) for x in py if x is not None and str(x).strip()] \
            if isinstance(py, list) else []
    try:
        return [str(x) for x in list(v) if x is not None and str(x).strip()]
    except (TypeError, ValueError):
        pass
    if isinstance(v, str):
        import json
        try:
            r = json.loads(v)
            return [str(x) for x in r if x is not None] if isinstance(r, list) else []
        except Exception:  # noqa: BLE001
            pass
    return []


# Orchestrator

def build_source_cache(
    input_parquet: Path,
    output_dir: Path,
    repos_dir: Path,
    max_files_per_commit: int = 50,
    force: bool = False,
) -> None:
    if not input_parquet.exists():
        raise RuntimeError(
            f"Input parquet not found: {input_parquet}\n"
            "Run pipeline/02_preprocess.py first."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    repos_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading: %s", input_parquet)
    df = pd.read_parquet(input_parquet, engine="pyarrow")
    logger.info("Loaded %d rows total.", len(df))

    pd_df = df[df["source"] == "pydriller"].copy()
    pd_df["modified_files"] = pd_df["modified_files"].apply(_to_list)
    logger.info(
        "PyDriller rows: %d across %d repos.", len(pd_df), pd_df["repo"].nunique()
    )

    # Identify already-cached commits
    if force:
        already_done: set[str] = set()
        logger.info("--force: re-caching all commits.")
    else:
        already_done = {
            d.name
            for d in output_dir.iterdir()
            if d.is_dir() and any(d.iterdir())  # skip empty dirs
        } if output_dir.exists() else set()
        if already_done:
            logger.info("Skipping %d already-cached commits.", len(already_done))

    todo_df = pd_df[~pd_df["commit_hash"].isin(already_done)].reset_index(drop=True)
    logger.info("Commits to process: %d", len(todo_df))

    if len(todo_df) == 0:
        logger.info("Nothing to do. Use --force to re-cache all commits.")
        return

    # Process per repo
    total_pairs = 0
    total_commits_done = 0
    failed_repos: list[str] = []
    err_counts: dict[str, int] = {
        "commit_not_found":       0,
        "skipped_unsupported_ext": 0,
        "skipped_empty_pair":      0,
    }

    for repo_name, repo_group in todo_df.groupby("repo"):
        n_repo = len(repo_group)
        logger.info("Repo: %s  (%d commits)", repo_name, n_repo)

        git_repo = _ensure_repo(repo_name, repos_dir)
        if git_repo is None:
            logger.error("  Skipping '%s' — could not obtain git clone.", repo_name)
            failed_repos.append(repo_name)
            continue

        repo_pairs = 0
        repo_done  = 0

        for _, row in tqdm(
            repo_group.iterrows(),
            total=n_repo,
            desc=f"  {repo_name.split('/')[-1]:<14}",
            unit="commit",
        ):
            commit_hash = str(row["commit_hash"])
            modified_files = _to_list(row["modified_files"])

            n = _process_commit(
                git_repo,
                commit_hash,
                modified_files,
                output_dir,
                max_files_per_commit,
                err_counts,
            )
            repo_pairs += n
            repo_done  += 1
            total_pairs += n
            total_commits_done += 1

            if repo_done % PROGRESS_EVERY == 0:
                logger.info(
                    "  [%s] %d / %d commits — file pairs so far: %d",
                    repo_name, repo_done, n_repo, repo_pairs,
                )

        logger.info(
            "  Finished %s: %d commits → %d file pairs written.",
            repo_name, repo_done, repo_pairs,
        )

    # Summary
    sep = "─" * 66
    print(f"\n{sep}")
    print(f"  SOURCE CACHE BUILD COMPLETE")
    print(sep)
    print(f"  Commits processed         : {total_commits_done:,}")
    print(f"  File pairs written        : {total_pairs:,}")
    print(f"  Pairs per commit (avg)    : "
          f"{total_pairs/max(total_commits_done,1):.2f}")
    print(f"  Skipped (bad extension)   : {err_counts['skipped_unsupported_ext']:,}")
    print(f"  Skipped (empty content)   : {err_counts['skipped_empty_pair']:,}")
    print(f"  Commits not in git        : {err_counts['commit_not_found']:,}")
    print(f"  Cache directory           : {output_dir}")
    if failed_repos:
        print(f"  Failed repos              : {', '.join(failed_repos)}")
    print(f"\n  Next step:")
    print(f"    python pipeline/03_features.py --source-cache-dir {output_dir}")
    print(f"{sep}\n")


# CLI

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="03b_build_source_cache.py",
        description="Extract before/after source blobs for AST feature computation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/commits_clean.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/source_cache"),
        help="Root dir for {commit_hash}/*_{before,after}.txt files.",
    )
    parser.add_argument(
        "--repos-dir",
        type=Path,
        default=Path("data/repos"),
        help="Directory where git repos are cloned/cached.",
    )
    parser.add_argument(
        "--max-files-per-commit",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-cache all commits, ignoring existing cache dirs.",
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
    logger.info("forge-bug-predictor | pipeline step 03b — SOURCE CACHE")
    logger.info("Input    : %s", args.input)
    logger.info("Cache dir: %s", args.output_dir)
    logger.info("Repos dir: %s", args.repos_dir)
    logger.info("Max files: %d per commit", args.max_files_per_commit)

    build_source_cache(
        input_parquet=args.input,
        output_dir=args.output_dir,
        repos_dir=args.repos_dir,
        max_files_per_commit=args.max_files_per_commit,
        force=args.force,
    )
    logger.info("Step 03b complete.")


if __name__ == "__main__":
    main()
