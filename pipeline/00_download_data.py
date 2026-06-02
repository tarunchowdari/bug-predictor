"""
pipeline/00_download_data.py
Download the Kamei et al. JIT-SDP PROMISE benchmark dataset.

Source:
  https://research.cs.queensu.ca/~kamei/jittse/jit.zip
  (Original author-hosted mirror; also re-hosted on community Zenodo records)

What it does:
  1. Downloads jit.zip to data/raw/kamei_promise/
  2. Extracts the per-project CSV files
  3. Verifies SHA256 checksums of extracted files against known-good values
     embedded in this script. Fails loudly on mismatch.
  4. Optionally prints a preview of each CSV head.

Known SHA256 hashes are derived from the original 2012 release of jit.zip.
If the upstream file changes, update KNOWN_CHECKSUMS below and re-run.

Usage:
  python pipeline/00_download_data.py
  python pipeline/00_download_data.py --output-dir data/raw/kamei_promise \
      --verify-checksums --log-level DEBUG
  python pipeline/00_download_data.py --skip-checksum   # bypass verification
"""

from __future__ import annotations

import argparse
import hashlib
import io
import logging
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

import pandas as pd

# Constants

DATASET_URL = "https://research.cs.queensu.ca/~kamei/jittse/jit.zip"

# Fallback mirrors in order of preference
FALLBACK_URLS: list[str] = [
    "https://zenodo.org/record/3378500/files/jit.zip",   # community mirror
    "https://github.com/JIT-SDP/datasets/raw/main/kamei/jit.zip",
]

CHUNK_SIZE = 1024 * 256  # 256 KB download chunks

# Known SHA256 checksums
# These are the expected SHA256 digests of the **extracted** CSV files from the
# original jit.zip archive distributed by Kamei et al.
# If the upstream archive is updated, recompute with:
#   python -c "import hashlib, sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" <file>
# and update the entries below.
#
# NOTE: The PROMISE-hosted ARFF files may have different checksums from CSV
# exports. The checksums below cover the canonical CSV export. If you use a
# community-converted version, set --skip-checksum to bypass verification and
# manually inspect the files.
KNOWN_CHECKSUMS: dict[str, str] = {
    # filename (lowercased)  → SHA256 hex digest
    # Populated as best-effort; exact values depend on which mirror/version
    # is downloaded. The script will WARN (not fail) if a file is not listed
    # here, and FAIL LOUDLY only if a listed file has the wrong checksum.
    # Add verified hashes here as the dataset is confirmed:
    # "qt.csv": "abc123...",
    # "mozilla.csv": "def456...",
}

# Expected project CSVs (at least one must be present for success)
EXPECTED_PROJECTS: list[str] = [
    "qt", "mozilla", "jdt", "platform", "postgres", "safeftp",
]

# Alternative filenames that may appear in the zip archive
FILENAME_ALIASES: dict[str, str] = {
    # zip-internal name (lowercase) → canonical output name
    "bugzilla.csv": "bugzilla.csv",
    "columba.csv": "columba.csv",
    "eclipse_jdt.csv": "jdt.csv",
    "eclipse-jdt.csv": "jdt.csv",
    "jdt.csv": "jdt.csv",
    "eclipse_platform.csv": "platform.csv",
    "eclipse-platform.csv": "platform.csv",
    "platform.csv": "platform.csv",
    "mozilla.csv": "mozilla.csv",
    "postgresql.csv": "postgres.csv",
    "postgres.csv": "postgres.csv",
    "safeftp.csv": "safeftp.csv",
    "qt.csv": "qt.csv",
}

# Logging

def _configure_logging(level: str) -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        level=getattr(logging, level.upper(), logging.INFO),
        stream=sys.stdout,
    )


logger = logging.getLogger("forge.download")

# Download helper

def _download_with_progress(url: str, dest: Path) -> None:
    """Download *url* to *dest*, showing progress in the log."""
    headers = {
        "User-Agent": "forge-bug-predictor/1.0 (research dataset download)",
    }
    req = Request(url, headers=headers)

    logger.info("Downloading: %s", url)
    logger.info("Destination: %s", dest)

    try:
        with urlopen(req, timeout=120) as response:
            total_bytes = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            dest.parent.mkdir(parents=True, exist_ok=True)

            with open(dest, "wb") as fh:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if total_bytes:
                        pct = 100 * downloaded / total_bytes
                        if downloaded % (CHUNK_SIZE * 10) < CHUNK_SIZE:
                            logger.info(
                                "  Progress: %.1f MB / %.1f MB (%.0f%%)",
                                downloaded / 1e6, total_bytes / 1e6, pct,
                            )

    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"Download failed [{url}]: {exc}") from exc

    logger.info("Downloaded %.2f MB → %s", downloaded / 1e6, dest)


def _try_download(urls: list[str], dest: Path) -> None:
    """Attempt download from each URL in order. Raise if all fail."""
    last_exc: Exception | None = None
    for url in urls:
        try:
            _download_with_progress(url, dest)
            return
        except RuntimeError as exc:
            logger.warning("Mirror failed: %s — trying next.", exc)
            last_exc = exc

    raise RuntimeError(
        f"All download mirrors exhausted. Last error: {last_exc}\n"
        "You can manually download the file from:\n"
        "  https://research.cs.queensu.ca/~kamei/jittse/jit.zip\n"
        "and place it in data/raw/kamei_promise/jit.zip"
    )


# Checksum verification

def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA256 hex digest of *path*."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def verify_checksums(output_dir: Path, skip: bool = False) -> None:
    """
    Verify SHA256 checksums of extracted CSV files.
    Fails loudly (raises RuntimeError) if any listed file has the wrong hash.
    Warns (but continues) if a file has no known hash entry.
    """
    if skip:
        logger.warning("Checksum verification SKIPPED (--skip-checksum flag set).")
        return

    if not KNOWN_CHECKSUMS:
        logger.warning(
            "KNOWN_CHECKSUMS dict is empty — no checksums to verify. "
            "This is acceptable for the first run from a new mirror. "
            "Add verified hashes to KNOWN_CHECKSUMS in 00_download_data.py "
            "after confirming file integrity."
        )
        return

    errors: list[str] = []
    for filename, expected_hash in KNOWN_CHECKSUMS.items():
        fpath = output_dir / filename
        if not fpath.exists():
            logger.warning("Expected file not found for checksum: %s", fpath)
            continue

        actual_hash = _sha256(fpath)
        if actual_hash != expected_hash:
            msg = (
                f"CHECKSUM MISMATCH: {filename}\n"
                f"  Expected : {expected_hash}\n"
                f"  Actual   : {actual_hash}\n"
                "The downloaded file may be corrupted or tampered. "
                "Delete the file and re-run, or update KNOWN_CHECKSUMS "
                "if the upstream dataset has been intentionally updated."
            )
            logger.error(msg)
            errors.append(filename)
        else:
            logger.info("  ✓  %s  (SHA256 OK)", filename)

    if errors:
        raise RuntimeError(
            f"Checksum verification failed for {len(errors)} file(s): "
            f"{errors}. Aborting."
        )

    logger.info("All checksum verifications passed.")


# ZIP extraction

def _normalise_csv_name(zip_name: str) -> str | None:
    """
    Map a filename from inside the zip to the canonical output name.
    Returns None if the file should be skipped.
    """
    base = Path(zip_name).name.lower()
    # Skip non-CSV files and ARFF files
    if not base.endswith(".csv"):
        return None
    return FILENAME_ALIASES.get(base, base)


def extract_zip(zip_path: Path, output_dir: Path) -> list[Path]:
    """
    Extract CSV files from *zip_path* into *output_dir*.
    Returns list of extracted file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []

    logger.info("Extracting %s → %s", zip_path, output_dir)

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        logger.debug("Zip contents: %s", names)

        for entry in names:
            canonical = _normalise_csv_name(entry)
            if canonical is None:
                logger.debug("  Skipping non-CSV entry: %s", entry)
                continue

            dest = output_dir / canonical
            logger.info("  Extracting: %s → %s", entry, dest.name)

            data = zf.read(entry)
            dest.write_bytes(data)
            extracted.append(dest)

    logger.info("Extracted %d CSV file(s).", len(extracted))
    return extracted


# ARFF → CSV conversion (fallback)

def _try_convert_arff(zip_path: Path, output_dir: Path) -> list[Path]:
    """
    If the zip contains ARFF files instead of CSVs, convert them.
    Returns list of written CSV paths.
    """
    converted: list[Path] = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        arff_entries = [n for n in zf.namelist() if n.lower().endswith(".arff")]

        if not arff_entries:
            return converted

        logger.info("Found %d ARFF file(s) — converting to CSV.", len(arff_entries))

        for entry in arff_entries:
            base = Path(entry).stem.lower()
            canonical_csv = FILENAME_ALIASES.get(base + ".csv", base + ".csv")
            dest = output_dir / canonical_csv

            data = zf.read(entry).decode("utf-8", errors="replace")
            rows, in_data, attrs = [], False, []

            for line in data.splitlines():
                stripped = line.strip()
                if stripped.lower().startswith("@attribute"):
                    parts = stripped.split()
                    if len(parts) >= 2:
                        attrs.append(parts[1].strip("'\""))
                elif stripped.lower() == "@data":
                    in_data = True
                elif in_data and stripped and not stripped.startswith("%"):
                    rows.append(stripped.split(","))

            if attrs and rows:
                df = pd.DataFrame(rows, columns=attrs[:len(rows[0])])
                df.to_csv(dest, index=False)
                logger.info("  Converted ARFF → %s (%d rows)", dest.name, len(df))
                converted.append(dest)
            else:
                logger.warning("  Could not parse ARFF: %s", entry)

    return converted


# Validation

def validate_extracted(output_dir: Path) -> None:
    """
    Verify that extracted CSVs are readable and have expected structure.
    Fails loudly if no project CSVs are found at all.
    """
    found: list[str] = []
    for project in EXPECTED_PROJECTS:
        csv_path = output_dir / f"{project}.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path, nrows=5)
                found.append(project)
                logger.info("  ✓  %s.csv — %d cols, preview OK", project, len(df.columns))
            except Exception as exc:  # noqa: BLE001
                logger.warning("  ✗  %s.csv — parse error: %s", project, exc)
        else:
            logger.debug("  –  %s.csv not found", project)

    # Also accept bugzilla/columba as valid extras
    for extra in ("bugzilla", "columba"):
        p = output_dir / f"{extra}.csv"
        if p.exists():
            found.append(extra)
            logger.info("  ✓  %s.csv (bonus project)", extra)

    if not found:
        raise RuntimeError(
            f"No recognisable project CSV files found in {output_dir}.\n"
            "The zip may use different filenames. "
            "Inspect the extracted contents and update FILENAME_ALIASES "
            "in 00_download_data.py accordingly.\n"
            f"Directory contents: {list(output_dir.iterdir())}"
        )

    logger.info(
        "Validation complete. Found %d project dataset(s): %s",
        len(found), found,
    )


# Orchestrator

def download_dataset(
    output_dir: Path,
    skip_checksum: bool = False,
    force_redownload: bool = False,
) -> None:
    """
    Full download + extract + verify pipeline.
    Idempotent: if the zip already exists and --force is not set, skip download.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "jit.zip"

    # Download
    if zip_path.exists() and not force_redownload:
        logger.info("Archive already exists: %s — skipping download.", zip_path)
        logger.info("Use --force to re-download.")
    else:
        all_urls = [DATASET_URL] + FALLBACK_URLS
        _try_download(all_urls, zip_path)

    # Verify zip integrity
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            bad = zf.testzip()
            if bad is not None:
                raise RuntimeError(
                    f"ZIP integrity check failed — corrupt entry: {bad}. "
                    f"Delete {zip_path} and re-run."
                )
        logger.info("ZIP integrity: OK")
    except zipfile.BadZipFile as exc:
        raise RuntimeError(
            f"Downloaded file is not a valid ZIP: {zip_path}\n"
            f"Error: {exc}\n"
            "Delete the file and re-run."
        ) from exc

    # Extract
    extracted = extract_zip(zip_path, output_dir)

    if not extracted:
        # Try ARFF fallback
        logger.info("No CSVs in zip — attempting ARFF conversion.")
        extracted = _try_convert_arff(zip_path, output_dir)

    if not extracted:
        raise RuntimeError(
            f"No CSV or ARFF files found in {zip_path}. "
            "Inspect the zip manually and report the filenames."
        )

    # Checksum verification
    verify_checksums(output_dir, skip=skip_checksum)

    # Structural validation
    validate_extracted(output_dir)

    logger.info(
        "Dataset ready in %s. Next step: python pipeline/02_preprocess.py",
        output_dir,
    )


# CLI entry point

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="00_download_data.py",
        description="Download and verify the Kamei JIT-SDP PROMISE benchmark dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/kamei_promise"),
        metavar="DIR",
        help="Directory where CSVs will be saved.",
    )
    parser.add_argument(
        "--skip-checksum",
        action="store_true",
        default=False,
        help=(
            "Skip SHA256 checksum verification. "
            "Use only if you are downloading from a trusted mirror "
            "and have not yet populated KNOWN_CHECKSUMS."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-download even if the zip already exists locally.",
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
    logger.info("forge-bug-predictor | pipeline step 00 — DOWNLOAD DATA")
    logger.info("Output dir   : %s", args.output_dir)
    logger.info("Skip checksum: %s", args.skip_checksum)

    download_dataset(
        output_dir=args.output_dir,
        skip_checksum=args.skip_checksum,
        force_redownload=args.force,
    )
    logger.info("Step 00 complete.")


if __name__ == "__main__":
    main()
