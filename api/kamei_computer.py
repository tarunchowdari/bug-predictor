# Computes all 14 Kamei et al. (2013) change metrics from a live git repo.
# Called by the webhook endpoint at inference time, not during training.

from __future__ import annotations

import collections
import logging
import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import git

logger = logging.getLogger("forge.kamei")

REPO_CACHE_DIR  = Path("data/webhook_repos")
MAX_CACHED_REPOS = 5
HISTORY_LOOKBACK = 200   # commits to scan for NOD/NUC/AGE/EXP
COMPUTE_TIMEOUT  = 60.0  # give up and return zeros after this many seconds

FIX_REGEX = re.compile(
    r"(fix|bug|defect|patch|error|fault|repair|correct|resolv)", re.IGNORECASE
)


class RepoCache:
    """Thread-safe LRU cache of locally cloned git repos (max 5)."""

    def __init__(self, cache_dir: Path = REPO_CACHE_DIR, max_size: int = MAX_CACHED_REPOS):
        self._cache_dir = cache_dir
        self._max_size  = max_size
        self._lru: "collections.OrderedDict[str, str]" = collections.OrderedDict()
        self._global_lock = threading.Lock()
        self._repo_locks:  dict[str, threading.Lock] = {}
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _url_to_dir_name(self, repo_url: str) -> str:
        parsed = urlparse(repo_url)
        path = parsed.path.strip("/").rstrip(".git")
        return path.replace("/", "__")

    def _evict_lru(self) -> None:
        oldest_url, _ = self._lru.popitem(last=False)
        self._repo_locks.pop(oldest_url, None)
        logger.info("RepoCache: evicted %s", oldest_url)

    def get_repo(self, repo_url: str) -> git.Repo:
        """Clone or fetch repo_url, returning a git.Repo. Falls back to stale cache on fetch error."""
        dir_name   = self._url_to_dir_name(repo_url)
        local_path = self._cache_dir / dir_name

        with self._global_lock:
            if repo_url not in self._repo_locks:
                self._repo_locks[repo_url] = threading.Lock()
            repo_lock = self._repo_locks[repo_url]

        with repo_lock:
            if local_path.exists() and (local_path / ".git").exists():
                try:
                    repo = git.Repo(str(local_path))
                    logger.info("RepoCache: fetching %s …", repo_url)
                    repo.remotes.origin.fetch(depth=500)
                    logger.info("RepoCache: fetch complete for %s", repo_url)
                except Exception as exc:
                    logger.warning("RepoCache: fetch failed (%s) — using stale cache", exc)
                    repo = git.Repo(str(local_path))
            else:
                if self._max_size and len(self._lru) >= self._max_size:
                    with self._global_lock:
                        self._evict_lru()

                logger.info("RepoCache: cloning %s → %s …", repo_url, local_path)
                try:
                    repo = git.Repo.clone_from(
                        repo_url, str(local_path),
                        depth=500,
                        no_single_branch=True,
                    )
                    logger.info("RepoCache: clone complete for %s", repo_url)
                except Exception as exc:
                    raise RuntimeError(f"Clone failed for {repo_url}: {exc}") from exc

            with self._global_lock:
                self._lru.pop(repo_url, None)
                self._lru[repo_url] = str(local_path)

        return repo


class KameiComputer:
    """Computes all 14 Kamei metrics for a single commit from git history."""

    def __init__(self, repo: git.Repo) -> None:
        self._repo = repo

    def compute(self, commit_hash: str) -> dict:
        """Run metric computation with a hard timeout. Returns zeros + _compute_error key on failure."""
        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(self._compute_inner, commit_hash)
                return fut.result(timeout=COMPUTE_TIMEOUT)
        except FuturesTimeout:
            msg = f"timeout after {COMPUTE_TIMEOUT}s"
            logger.warning("KameiComputer: %s for %s — using zeros", msg, commit_hash[:8])
            return {**self._zero_metrics(), "_compute_error": msg}
        except Exception as exc:
            msg = str(exc)
            logger.warning("KameiComputer: error for %s: %s — using zeros", commit_hash[:8], msg)
            return {**self._zero_metrics(), "_compute_error": msg}

    @staticmethod
    def _zero_metrics() -> dict:
        return {
            "NS": 0.0, "ND": 0.0, "NF": 0.0, "Entropy": 0.0,
            "LA": 0.0, "LD": 0.0, "LT": 0.0,  "FIX": 0.0,
            "NOD": 0.0, "NUC": 0.0, "AGE": 0.0,
            "EXP": 0.0, "REXP": 0.0, "SEXP": 0.0,
        }

    def _compute_inner(self, commit_hash: str) -> dict:
        repo    = self._repo
        commit  = repo.commit(commit_hash)
        parents = commit.parents

        # Which files changed?
        if parents:
            diff = parents[0].diff(commit)
            modified_files: list[str] = [
                d.b_path or d.a_path or "" for d in diff if (d.b_path or d.a_path)
            ]
        else:
            # Root commit — every file is "added"
            modified_files = list(commit.stats.files.keys())

        if not modified_files:
            return self._zero_metrics()

        # NS / ND — how spread out are the changes?
        subsystems  = set()
        directories = set()
        for f in modified_files:
            parts = f.replace("\\", "/").split("/")
            subsystems.add(parts[0])
            directories.add("/".join(parts[:-1]) if len(parts) > 1 else ".")

        NS = float(len(subsystems))
        ND = float(len(directories))
        NF = float(len(modified_files))

        # LA / LD — raw churn
        stats = commit.stats.files
        LA = float(sum(stats.get(f, {}).get("insertions", 0) for f in modified_files))
        LD = float(sum(stats.get(f, {}).get("deletions",  0) for f in modified_files))

        # Entropy — scattered changes across many files = higher entropy = more risk
        # Single-file changes always return 0
        total_churn = LA + LD
        entropy = 0.0
        if total_churn > 0 and len(modified_files) > 1:
            for f in modified_files:
                fc = stats.get(f, {}).get("insertions", 0) + stats.get(f, {}).get("deletions", 0)
                p  = fc / total_churn
                if p > 0:
                    entropy -= p * math.log2(p)
        Entropy = entropy

        # LT — total pre-existing lines in the touched files before this commit
        LT = 0.0
        if parents:
            parent_commit = parents[0]
            for f in modified_files:
                try:
                    blob    = parent_commit.tree / f
                    content = blob.data_stream.read().decode("utf-8", errors="replace")
                    LT += content.count("\n") + (1 if content and not content.endswith("\n") else 0)
                except Exception:
                    pass  # file didn't exist in parent (newly added)

        FIX = 1.0 if FIX_REGEX.search(commit.message or "") else 0.0

        # History-based metrics: one pass through prior commits collects
        # NOD, NUC, AGE, EXP, REXP, SEXP — avoids two separate git subprocess loops.
        commit_dt    = datetime.fromtimestamp(commit.committed_date, tz=timezone.utc)
        file_set     = set(modified_files)
        author_email = (commit.author.email or "").lower()
        author_name  = (commit.author.name  or "").lower()

        prior_devs:    set[str] = set()
        prior_commits: set[str] = set()
        file_last_date: dict[str, Optional[datetime]] = {f: None for f in modified_files}
        exp_count   = 0
        rexp_weight = 0.0
        sexp_systems: set[str] = set()

        try:
            history = list(repo.iter_commits(commit_hash + "~1", max_count=HISTORY_LOOKBACK))
        except Exception:
            history = []

        for hc in history:
            hc_files = set(hc.stats.files.keys())
            hc_dt    = datetime.fromtimestamp(hc.committed_date, tz=timezone.utc)

            touched = file_set & hc_files
            if touched:
                prior_devs.add(hc.author.email or hc.author.name or "unknown")
                prior_commits.add(hc.hexsha)
                for f in touched:
                    if file_last_date[f] is None:
                        file_last_date[f] = hc_dt

            hc_email = (hc.author.email or "").lower()
            hc_name  = (hc.author.name  or "").lower()
            if hc_email == author_email or (author_email == "" and hc_name == author_name):
                exp_count  += 1
                days_ago    = max(0.0, (commit_dt - hc_dt).total_seconds() / 86400.0)
                rexp_weight += 1.0 / (1.0 + days_ago / 7.0)
                for f in hc_files:
                    parts = f.replace("\\", "/").split("/")
                    sexp_systems.add(parts[0])

        NOD = float(len(prior_devs))
        NUC = float(len(prior_commits))

        # AGE — mean days since each file was last touched
        age_days = [
            (commit_dt - last_dt).total_seconds() / 86400.0 if last_dt else 0.0
            for last_dt in file_last_date.values()
        ]
        AGE  = float(sum(age_days) / len(age_days)) if age_days else 0.0
        EXP  = float(exp_count)
        REXP = float(rexp_weight)
        SEXP = float(len(sexp_systems))

        return {
            "NS": NS, "ND": ND, "NF": NF, "Entropy": Entropy,
            "LA": LA, "LD": LD, "LT": LT, "FIX": FIX,
            "NOD": NOD, "NUC": NUC, "AGE": AGE,
            "EXP": EXP, "REXP": REXP, "SEXP": SEXP,
        }


# Module-level singleton — one cache shared across all webhook requests
_repo_cache: Optional[RepoCache] = None


def get_repo_cache() -> RepoCache:
    global _repo_cache
    if _repo_cache is None:
        _repo_cache = RepoCache()
    return _repo_cache
