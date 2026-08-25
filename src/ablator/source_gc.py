"""Lifecycle management for immutable Git-SHA worktree caches."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from . import source_checkout


@dataclass(frozen=True)
class GCEntry:
    checkout: str
    sidecar: str
    source_repo_path: str | None
    last_used_at: float


@dataclass(frozen=True)
class GCResult:
    removed: tuple[str, ...]
    candidates: tuple[str, ...]
    protected: tuple[str, ...]
    retained: tuple[str, ...]
    errors: tuple[str, ...]


def _read_sidecar(path: Path) -> GCEntry | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError, TypeError):
        return None
    checkout = data.get("checkout")
    if not isinstance(checkout, str) or not checkout:
        return None
    last = data.get("last_used_at")
    try:
        last_used_at = float(last)
    except (TypeError, ValueError):
        try:
            last_used_at = path.stat().st_mtime
        except OSError:
            last_used_at = 0.0
    repo = data.get("source_repo_path")
    return GCEntry(
        checkout=os.path.abspath(os.path.expanduser(checkout)),
        sidecar=str(path),
        source_repo_path=(str(repo) if repo else None),
        last_used_at=last_used_at,
    )


def scan_entries(root: str) -> list[GCEntry]:
    """Read valid Ablator worktree sidecars under one cache root."""
    path = Path(os.path.abspath(os.path.expanduser(root))).resolve()
    if not path.exists():
        return []
    entries: list[GCEntry] = []
    for sidecar in path.rglob("*.ablator.json"):
        item = _read_sidecar(sidecar)
        if item is None:
            continue
        checkout = Path(item.checkout).resolve()
        try:
            checkout.relative_to(path)
        except ValueError:
            continue
        expected_sidecar = Path(f"{checkout}.ablator.json").resolve()
        if sidecar.resolve() != expected_sidecar:
            continue
        entries.append(item)
    return entries


def active_checkouts(jobs: list[dict]) -> set[str]:
    """Checkouts leased by jobs that may still be executing."""
    out: set[str] = set()
    for job in jobs:
        if job.get("status") != "running":
            continue
        checkout = job.get("source_checkout")
        if isinstance(checkout, str) and checkout:
            out.add(os.path.abspath(os.path.expanduser(checkout)))
    return out


def _run_git(repo: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", repo, *args], capture_output=True, text=True,
        timeout=60.0, check=False,
    )


def _remove_entry(entry: GCEntry) -> str | None:
    """Remove one worktree and its Git metadata. Return an error string."""
    checkout = entry.checkout
    repo = entry.source_repo_path

    if repo and os.path.isdir(repo):
        # `git worktree remove` removes both files and the owning repository's
        # worktree administration entry. It also handles a checkout that has
        # already disappeared less cleanly than `rm -rf`, so prune afterwards.
        if os.path.exists(checkout):
            result = _run_git(repo, "worktree", "remove", "--force", checkout)
            if result.returncode != 0:
                detail = (result.stderr or result.stdout).strip()
                return f"{checkout}: git worktree remove failed: {detail[:400]}"
        _run_git(repo, "worktree", "prune")
    elif os.path.exists(checkout):
        # The owning repository can itself have been removed. In that orphan
        # case there is no Git administration directory left to protect; clear
        # the now-unmanageable cache tree and its sidecar.
        try:
            shutil.rmtree(checkout)
        except OSError as exc:
            return f"{checkout}: orphan removal failed: {exc}"

    try:
        os.remove(entry.sidecar)
    except FileNotFoundError:
        pass
    except OSError as exc:
        return f"{checkout}: removed checkout but not sidecar: {exc}"
    return None


def gc_worktrees(
    cfg: dict,
    machine: str,
    jobs: list[dict],
    *,
    dry_run: bool = False,
    max_age_days: float | None = None,
    now: float | None = None,
) -> GCResult:
    """Collect stale SHA worktrees for the current execution machine.

    Running jobs lease their `source_checkout` unconditionally. All other
    entries become candidates once their last-use age exceeds the configured
    threshold. Dry-run performs the same classification without mutation.
    """
    root = source_checkout.cache_root(cfg, machine)
    if max_age_days is None:
        configured = (cfg.get("git") or {}).get("gc_max_age_days", 30)
        try:
            max_age_days = float(configured)
        except (TypeError, ValueError):
            max_age_days = 30.0
    if max_age_days < 0:
        raise ValueError("max_age_days must be >= 0")
    now = time.time() if now is None else float(now)
    cutoff = now - max_age_days * 86400.0
    active = active_checkouts(jobs)

    removed: list[str] = []
    candidates: list[str] = []
    protected: list[str] = []
    retained: list[str] = []
    errors: list[str] = []

    for entry in scan_entries(root):
        if entry.checkout in active:
            protected.append(entry.checkout)
            continue
        if entry.last_used_at > cutoff:
            retained.append(entry.checkout)
            continue
        candidates.append(entry.checkout)
        if dry_run:
            continue
        error = _remove_entry(entry)
        if error:
            errors.append(error)
        else:
            removed.append(entry.checkout)

    return GCResult(
        removed=tuple(removed),
        candidates=tuple(candidates),
        protected=tuple(protected),
        retained=tuple(retained),
        errors=tuple(errors),
    )
