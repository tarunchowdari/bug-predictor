"""
pipeline/05_split.py
Project-stratified 70/15/15 train/val/test split.

Split strategy:
  - The atomic unit is a REPO, not a row.
  - Repos are sorted by row count descending and greedily assigned to the
    split with the most room remaining until target proportions are met.
  - No repo ever straddles two splits (eliminates data leakage across projects).
  - After repo assignment the splits are further shuffled (seed=42).

Quality gates (fail loudly):
  - Each split must have >= 100 rows.
  - Each split must have >= 15% buggy class.
  - Every repo in test must be absent from train.

Outputs:
  data/splits/train.parquet
  data/splits/val.parquet
  data/splits/test.parquet
  data/splits/split_manifest.json  — repo→split assignment for auditability

Usage:
  python pipeline/05_split.py
  python pipeline/05_split.py \
      --features data/processed/features.parquet \
      --manifest data/processed/embeddings_manifest.parquet \
      --output-dir data/splits \
      --train-ratio 0.70 --val-ratio 0.15 \
      --log-level DEBUG
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

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


logger = logging.getLogger("forge.split")

# Constants

SEED = 42
MIN_SPLIT_ROWS = 100
MIN_BUGGY_RATIO = 0.15


# Repo-stratified assignment

def _assign_repos_to_splits(
    repo_sizes: dict[str, int],
    train_ratio: float,
    val_ratio: float,
) -> dict[str, str]:
    """
    Greedy repo-to-split assignment with optional forced overrides.

    Sorts repos by descending size (largest first to reduce fragmentation),
    then assigns each repo to whichever split has the most remaining capacity.
    Repos listed in FORCED_ASSIGNMENTS bypass the greedy logic entirely.

    Returns dict: repo_name → split_name ("train" | "val" | "test").
    """
    total = sum(repo_sizes.values())
    targets = {
        "train": total * train_ratio,
        "val":   total * val_ratio,
        "test":  total * (1 - train_ratio - val_ratio),
    }
    allocated: dict[str, float] = {"train": 0.0, "val": 0.0, "test": 0.0}
    assignment: dict[str, str] = {}

    # Largest repos first — greedy approach minimises imbalance
    sorted_repos = sorted(repo_sizes.items(), key=lambda x: x[1], reverse=True)

    # Forced overrides — bypass greedy logic for specific repos
    FORCED_ASSIGNMENTS = {
        "kamei/jdt": "train",
        "kamei/platform": "test",
    }

    for repo, size in sorted_repos:
        if repo in FORCED_ASSIGNMENTS:
            chosen = FORCED_ASSIGNMENTS[repo]
        else:
            gaps = {s: targets[s] - allocated[s] for s in targets}
            chosen = max(gaps, key=gaps.get)
        assignment[repo] = chosen
        allocated[chosen] += size
        logger.debug("  Repo %-40s → %-5s  (size=%d)", repo, chosen, size)

    return assignment


# Quality gates

def _validate_splits(
    splits: dict[str, pd.DataFrame],
    train_repos: set[str],
    test_repos: set[str],
) -> None:
    """
    Raise RuntimeError (loud failure) if any quality gate is violated.
    """
    errors: list[str] = []

    for split_name, df in splits.items():
        n = len(df)
        if n < MIN_SPLIT_ROWS:
            errors.append(
                f"Split '{split_name}' has only {n} rows (minimum {MIN_SPLIT_ROWS})."
            )
        if n > 0:
            buggy_ratio = (df["label"] == 1).mean()
            if buggy_ratio < MIN_BUGGY_RATIO:
                errors.append(
                    f"Split '{split_name}' has buggy ratio {buggy_ratio:.3f} "
                    f"(minimum {MIN_BUGGY_RATIO:.2f})."
                )

    # No repo leakage between train and test
    leaking = train_repos & test_repos
    if leaking:
        errors.append(
            f"DATA LEAKAGE: repos appear in both train and test: {leaking}"
        )

    if errors:
        msg = "\n".join(f"  ✗ {e}" for e in errors)
        raise RuntimeError(
            f"Split quality gate(s) FAILED:\n{msg}\n"
            "Adjust --train-ratio / --val-ratio or check class balance in the dataset."
        )

    logger.info("All quality gates passed.")


# Stats printer

def _print_split_stats(splits: dict[str, pd.DataFrame]) -> None:
    sep = "─" * 74
    print(f"\n{sep}")
    print("  SPLIT STATISTICS")
    print(sep)
    print(f"  {'SPLIT':<8}  {'ROWS':>7}  {'BUGGY%':>7}  REPOS")
    print(f"  {'─────':<8}  {'────':>7}  {'──────':>7}  ─────")

    for name, df in splits.items():
        n = len(df)
        buggy_pct = 100 * (df["label"] == 1).mean() if n > 0 else 0
        repos = sorted(df["repo"].unique()) if "repo" in df.columns else []
        repos_str = ", ".join(repos[:6])
        if len(repos) > 6:
            repos_str += f" … (+{len(repos) - 6} more)"
        print(f"  {name:<8}  {n:>7,}  {buggy_pct:>6.1f}%  {repos_str}")

    print(sep)

    # Per-split repo breakdown
    print("\n  REPO DISTRIBUTION PER SPLIT")
    print(sep)
    for name, df in splits.items():
        print(f"\n  [{name.upper()}]")
        if "repo" not in df.columns:
            continue
        for repo, grp in df.groupby("repo"):
            buggy = (grp["label"] == 1).sum()
            total = len(grp)
            print(f"    {repo:<45}  {total:>5,} rows  {100*buggy/total:>5.1f}% buggy")

    print(sep + "\n")


# Orchestrator

def create_splits(
    features_parquet: Path,
    manifest_parquet: Path | None,
    output_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> dict[str, Path]:
    """
    Main split pipeline. Returns dict of split_name → output path.
    """
    if not features_parquet.exists():
        raise RuntimeError(
            f"Features parquet not found: {features_parquet}\n"
            "Run pipeline/03_features.py first."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load features
    logger.info("Loading features: %s", features_parquet)
    df = pd.read_parquet(features_parquet, engine="pyarrow")
    logger.info("Loaded %d rows × %d cols", len(df), len(df.columns))

    # Optionally filter to commits that have embeddings
    if manifest_parquet is not None and manifest_parquet.exists():
        logger.info("Loading embeddings manifest: %s", manifest_parquet)
        manifest = pd.read_parquet(manifest_parquet, columns=["commit_hash"],
                                   engine="pyarrow")
        embedded_hashes = set(manifest["commit_hash"].tolist())
        before = len(df)
        df = df[df["commit_hash"].isin(embedded_hashes)].reset_index(drop=True)
        logger.info(
            "Filtered to commits with embeddings: %d → %d rows",
            before, len(df),
        )
    else:
        logger.info(
            "No embeddings manifest found — splitting full feature set "
            "(embeddings will be loaded at training time by hash lookup)."
        )

    if len(df) == 0:
        raise RuntimeError("No rows remain after filtering. Check pipeline outputs.")

    # Ensure required columns
    for col in ("commit_hash", "repo", "label"):
        if col not in df.columns:
            raise RuntimeError(
                f"Required column '{col}' missing from features parquet."
            )

    # Repo-level assignment
    repo_sizes: dict[str, int] = df.groupby("repo").size().to_dict()
    logger.info("Repos in dataset: %d", len(repo_sizes))
    for r, s in sorted(repo_sizes.items(), key=lambda x: x[1], reverse=True):
        buggy = (df[df["repo"] == r]["label"] == 1).sum()
        logger.info("  %-45s  %5d rows  %5.1f%% buggy", r, s, 100 * buggy / s)

    assignment = _assign_repos_to_splits(repo_sizes, train_ratio, val_ratio)

    # Log assignment
    logger.info("Repo → split assignment:")
    for repo, split in sorted(assignment.items()):
        logger.info("  %-45s → %s", repo, split)

    # Build split DataFrames
    rng = np.random.default_rng(SEED)

    split_dfs: dict[str, pd.DataFrame] = {}
    for split_name in ("train", "val", "test"):
        repos_in_split = [r for r, s in assignment.items() if s == split_name]
        part = df[df["repo"].isin(repos_in_split)].copy()
        # Shuffle rows within each split
        part = part.sample(frac=1, random_state=SEED).reset_index(drop=True)
        split_dfs[split_name] = part

    # Quality gate
    train_repos = set(r for r, s in assignment.items() if s == "train")
    test_repos  = set(r for r, s in assignment.items() if s == "test")
    _validate_splits(split_dfs, train_repos, test_repos)

    # Print statistics
    _print_split_stats(split_dfs)

    # Save parquets
    output_paths: dict[str, Path] = {}
    for split_name, part in split_dfs.items():
        out_path = output_dir / f"{split_name}.parquet"
        part.to_parquet(out_path, index=False, engine="pyarrow")
        logger.info(
            "Saved %s → %s  (%d rows)", split_name, out_path, len(part)
        )
        output_paths[split_name] = out_path

    # Save audit manifest
    manifest_out = output_dir / "split_manifest.json"
    with open(manifest_out, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "repo_assignment": assignment,
                "split_sizes": {k: len(v) for k, v in split_dfs.items()},
                "train_ratio": train_ratio,
                "val_ratio": val_ratio,
                "seed": SEED,
            },
            fh,
            indent=2,
        )
    logger.info("Split manifest saved → %s", manifest_out)

    return output_paths


# CLI

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="05_split.py",
        description="Project-stratified 70/15/15 train/val/test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/processed/features.parquet"),
        metavar="PARQUET",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/embeddings_manifest.parquet"),
        metavar="PARQUET",
        help="Embeddings manifest; only commits present here are split. "
             "Pass --no-manifest to skip this filter.",
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        default=False,
        help="Ignore embeddings manifest; split all feature rows.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/splits"),
        metavar="DIR",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio",   type=float, default=0.15)
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
    logger.info("forge-bug-predictor | pipeline step 05 — SPLIT")

    manifest = None if args.no_manifest else args.manifest

    create_splits(
        features_parquet=args.features,
        manifest_parquet=manifest,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    logger.info("Step 05 complete.")


if __name__ == "__main__":
    main()