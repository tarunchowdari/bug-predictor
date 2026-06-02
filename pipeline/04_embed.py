"""
pipeline/04_embed.py
Generate 768-dim semantic embeddings for every commit using
sentence-transformers/all-MiniLM-L6-v2.

Embedding strategy per commit:
  message_vec  = embed(commit_message)           → 384-dim
  diff_vec     = embed(diff_text[:512 tokens])   → 384-dim
  final_vec    = concat(message_vec, diff_vec)   → 768-dim

If diff_text is absent or empty the diff branch is embedded as "".
Embeddings are saved as float32 .npy files, one per commit hash.

Output:
  data/processed/embeddings/{commit_hash}.npy   — 768-dim float32 array
  data/processed/embeddings_manifest.parquet    — index of all embeddings
    columns: commit_hash (str), npy_path (str), embed_dim (int)

Idempotent: commits whose .npy already exists are skipped unless --force.

Usage:
  python pipeline/04_embed.py
  python pipeline/04_embed.py --input data/processed/features.parquet \
      --output-dir data/processed/embeddings \
      --batch-size 64 --device cpu --log-level INFO
  python pipeline/04_embed.py --device cuda --force
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Windows: register CUDA DLL directories BEFORE any torch import
# torch._load_dll_libraries() needs cudart64_NNN.dll and friends.
# The CUDA Toolkit installer adds these to the system PATH, but the current
# terminal session may have been opened before the install.  Calling
# os.add_dll_directory() here makes Windows find the DLLs unconditionally.
if sys.platform == "win32":
    _CUDA_BIN_CANDIDATES = [
        # Standard CUDA Toolkit install locations — newest first
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin",
    ]
    # Also honour CUDA_PATH env var (set by the installer)
    _cuda_env = os.environ.get("CUDA_PATH", "")
    if _cuda_env:
        _CUDA_BIN_CANDIDATES.insert(0, os.path.join(_cuda_env, "bin"))

    for _cuda_bin in _CUDA_BIN_CANDIDATES:
        if os.path.isdir(_cuda_bin):
            try:
                os.add_dll_directory(_cuda_bin)
            except (OSError, AttributeError):
                pass

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


logger = logging.getLogger("forge.embed")

# Constants

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MSG_DIM = 384
DIFF_DIM = 384
TOTAL_DIM = MSG_DIM + DIFF_DIM          # 768
MAX_DIFF_TOKENS = 512                   # truncation limit for diff text
DEFAULT_BATCH_SIZE = 64


# Model loader

def _load_model(device: str):
    """Load sentence-transformer model onto *device*."""
    logger.info("Loading model: %s  (device=%s)", MODEL_NAME, device)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME, device=device)
    logger.info("Model loaded. Max sequence length: %d", model.max_seq_length)
    return model


# Text preparation

def _prepare_text(text: str | None, max_chars: int = 2000) -> str:
    """
    Clean and truncate text for embedding.
    - Replaces null/control bytes
    - Limits to max_chars to stay within the model's token budget
    """
    if not text or not isinstance(text, str):
        return ""
    text = text.replace("\x00", " ").strip()
    return text[:max_chars]


def _build_texts(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Return parallel lists of (message_texts, diff_texts) for all rows.
    Uses diff_text column if present; falls back to empty string.
    """
    messages = [
        _prepare_text(str(row.get("commit_message", "") or ""))
        for _, row in df.iterrows()
    ]
    has_diff = "diff_text" in df.columns
    diffs = [
        _prepare_text(str(row.get("diff_text", "") or ""), max_chars=MAX_DIFF_TOKENS * 4)
        if has_diff else ""
        for _, row in df.iterrows()
    ]
    return messages, diffs


# Batch embedding

def _embed_batch(
    model,
    texts: list[str],
    batch_size: int,
    desc: str,
) -> np.ndarray:
    """
    Embed a list of texts in batches. Returns array of shape (N, 384).
    """
    all_vecs: list[np.ndarray] = []
    for start in tqdm(range(0, len(texts), batch_size), desc=desc, unit="batch"):
        chunk = texts[start : start + batch_size]
        vecs = model.encode(
            chunk,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        all_vecs.append(vecs.astype(np.float32))
    return np.vstack(all_vecs) if all_vecs else np.zeros((0, 384), dtype=np.float32)


# Save helpers

def _npy_path(output_dir: Path, commit_hash: str) -> Path:
    """Canonical path for a commit's .npy file."""
    return output_dir / f"{commit_hash}.npy"


def _save_embedding(path: Path, vec: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), vec)


# Manifest I/O

def _load_manifest(manifest_path: Path) -> set[str]:
    """Return set of commit_hashes already in the manifest."""
    if not manifest_path.exists():
        return set()
    try:
        df = pd.read_parquet(manifest_path, columns=["commit_hash"], engine="pyarrow")
        return set(df["commit_hash"].tolist())
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read existing manifest (%s) — rebuilding.", exc)
        return set()


def _save_manifest(manifest_path: Path, records: list[dict]) -> None:
    """
    Append *records* to the manifest parquet (or create it).
    Records: list of {commit_hash, npy_path, embed_dim}.
    """
    new_df = pd.DataFrame(records)
    if manifest_path.exists():
        try:
            existing = pd.read_parquet(manifest_path, engine="pyarrow")
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["commit_hash"], keep="last")
        except Exception:  # noqa: BLE001
            combined = new_df
    else:
        combined = new_df

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(manifest_path, index=False, engine="pyarrow")
    logger.info("Manifest saved: %d total entries → %s", len(combined), manifest_path)


# Orchestrator

def generate_embeddings(
    input_parquet: Path,
    output_dir: Path,
    manifest_path: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str = "cpu",
    force: bool = False,
) -> Path:
    """
    Main embedding pipeline. Returns path to manifest parquet.
    """
    if not input_parquet.exists():
        raise RuntimeError(
            f"Input parquet not found: {input_parquet}\n"
            "Run pipeline/03_features.py first."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading: %s", input_parquet)
    df = pd.read_parquet(input_parquet, engine="pyarrow")
    logger.info("Loaded %d rows.", len(df))

    if "commit_hash" not in df.columns:
        raise RuntimeError("'commit_hash' column missing from input parquet.")

    if "diff_text" not in df.columns:
        logger.warning(
            "Column 'diff_text' not found in parquet. "
            "Diff branch will embed empty strings (768-dim vectors will have "
            "zeros in the upper 384 dims). Add diff_text to the parquet to "
            "enable full semantic embeddings."
        )

    # Identify commits to process
    already_done: set[str] = set() if force else _load_manifest(manifest_path)
    if already_done:
        logger.info("Skipping %d already-embedded commits.", len(already_done))

    todo_df = df[~df["commit_hash"].isin(already_done)].reset_index(drop=True)
    n_todo = len(todo_df)

    if n_todo == 0:
        logger.info("All commits already embedded. Nothing to do. Use --force to re-run.")
        return manifest_path

    logger.info("Commits to embed: %d (skipping %d)", n_todo, len(already_done))

    # Load model
    model = _load_model(device)

    # Build text inputs
    logger.info("Preparing text inputs …")
    messages, diffs = _build_texts(todo_df)

    # Embed messages
    logger.info("Embedding commit messages (%d texts) …", len(messages))
    msg_vecs = _embed_batch(model, messages, batch_size, desc="messages")   # (N, 384)

    # Embed diffs
    logger.info("Embedding diff texts (%d texts) …", len(diffs))
    diff_vecs = _embed_batch(model, diffs, batch_size, desc="diffs")        # (N, 384)

    # Concatenate and save
    logger.info("Saving .npy files …")
    new_records: list[dict] = []
    n_saved = 0

    for i, (_, row) in enumerate(
        tqdm(todo_df.iterrows(), total=n_todo, desc="saving", unit="commit")
    ):
        commit_hash = str(row["commit_hash"])
        vec = np.concatenate([msg_vecs[i], diff_vecs[i]]).astype(np.float32)  # (768,)

        assert vec.shape == (TOTAL_DIM,), (
            f"Unexpected embedding shape {vec.shape} for commit {commit_hash}"
        )

        npy = _npy_path(output_dir, commit_hash)
        _save_embedding(npy, vec)
        new_records.append(
            {
                "commit_hash": commit_hash,
                "npy_path": str(npy),
                "embed_dim": TOTAL_DIM,
            }
        )
        n_saved += 1

    logger.info("Saved %d new embedding files.", n_saved)

    # Update manifest
    _save_manifest(manifest_path, new_records)

    # Summary
    total_manifest = len(_load_manifest(manifest_path))
    logger.info(
        "Embedding complete. Total in manifest: %d / %d commits.",
        total_manifest, len(df),
    )
    print(f"\n  Embeddings : {n_saved} new files saved in {output_dir}")
    print(f"  Manifest   : {manifest_path}  ({total_manifest} total entries)")
    print(f"  Shape      : ({TOTAL_DIM},) float32 per commit\n")

    return manifest_path


# CLI

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="04_embed.py",
        description="Generate 768-dim sentence-transformer embeddings per commit.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/features.parquet"),
        metavar="PARQUET",
        help="Input features parquet (from 03_features.py).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/embeddings"),
        metavar="DIR",
        help="Directory to write per-commit .npy files.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/embeddings_manifest.parquet"),
        metavar="PARQUET",
        help="Path for the embeddings manifest parquet.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        metavar="N",
        help="Encoding batch size.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Torch device for inference.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-embed even if .npy already exists.",
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
    logger.info("forge-bug-predictor | pipeline step 04 — EMBED")
    logger.info("Input    : %s", args.input)
    logger.info("Out dir  : %s", args.output_dir)
    logger.info("Device   : %s", args.device)
    logger.info("Batch    : %d", args.batch_size)

    generate_embeddings(
        input_parquet=args.input,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        batch_size=args.batch_size,
        device=args.device,
        force=args.force,
    )
    logger.info("Step 04 complete.")


if __name__ == "__main__":
    main()
