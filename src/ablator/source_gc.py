"""Lifecycle management for immutable Git-SHA worktree caches."""
from __future__ import annotations

import json
import os
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
    source_common_dir: str | None
    repo_identity: str | None
    lock_path: str
    active: bool
    lease_id: str | None
    last_used_at: float


@dataclass(frozen=True)
class GCResult:
    removed: tuple[str, ...]
    candidates: tuple[str, ...]
    protected: tuple[str, ...]
    retained: tuple[str, ...]
    errors: tuple[str, ...]


@dataclass(frozen=True)
class TrustedRepository:
    repo_path: str
    common_dir: str
    lock_path: str


def _inside(path: Path, root: Path) -> bool:
    """Return whether *path* resolves strictly below the managed root."""
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return path != root


def _read_sidecar(path: Path, root: Path) -> GCEntry:
    """Parse and validate one sidecar before trusting any filesystem path."""
    resolved_sidecar = path.resolve(strict=False)
    if not _inside(resolved_sidecar, root):
        raise ValueError(f"{path}: sidecar is outside managed root {root}")
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"{path}: could not read sidecar: {exc}") from exc
    checkout = data.get("checkout")
    if not isinstance(checkout, str) or not checkout:
        raise ValueError(f"{path}: sidecar has no checkout path")
    resolved_checkout = Path(checkout).expanduser().resolve(strict=False)
    if not _inside(resolved_checkout, root):
        raise ValueError(
            f"{path}: checkout {resolved_checkout} is outside managed root {root}"
        )
    expected_sidecar = Path(f"{resolved_checkout}.ablator.json").resolve(strict=False)
    if resolved_sidecar != expected_sidecar:
        raise ValueError(
            f"{path}: sidecar is not adjacent to claimed checkout {resolved_checkout}"
        )

    active = data.get("active", False)
    if not isinstance(active, bool):
        raise ValueError(f"{path}: active lease marker must be boolean")
    lease_id = data.get("lease_id")
    if lease_id is not None and (not isinstance(lease_id, str) or not lease_id):
        raise ValueError(f"{path}: lease_id must be a non-empty string")

    lock_value = data.get("lock_path")
    if lock_value is None:
        if active or data.get("schema") == "ablator.source-lease/v1":
            raise ValueError(f"{path}: active source lease has no repository lock")
        # Historical inactive cache records predate per-repository leases. A
        # root-local legacy lock is sufficient because they cannot reactivate.
        lock_path = root / "_locks" / "legacy-gc.lock"
    elif not isinstance(lock_value, str) or not lock_value:
        raise ValueError(f"{path}: lock_path must be a non-empty string")
    else:
        lock_path = Path(lock_value).expanduser().resolve(strict=False)
        if not _inside(lock_path, root):
            raise ValueError(
                f"{path}: repository lock {lock_path} is outside managed root {root}"
            )

    last = data.get("last_used_at")
    try:
        last_used_at = float(last)
    except (TypeError, ValueError):
        try:
            last_used_at = path.stat().st_mtime
        except OSError:
            last_used_at = 0.0
    repo = data.get("source_repo_path")
    common_dir = data.get("source_common_dir")
    repo_identity = data.get("repo")
    return GCEntry(
        checkout=str(resolved_checkout),
        sidecar=str(resolved_sidecar),
        source_repo_path=(str(repo) if repo else None),
        source_common_dir=(str(common_dir) if common_dir else None),
        repo_identity=(str(repo_identity) if repo_identity else None),
        lock_path=str(lock_path),
        active=active,
        lease_id=lease_id,
        last_used_at=last_used_at,
    )


def _scan_entries(root: str) -> tuple[list[GCEntry], list[str]]:
    path = Path(os.path.abspath(os.path.expanduser(root))).resolve(strict=False)
    if not path.exists():
        return [], []
    entries: list[GCEntry] = []
    errors: list[str] = []
    for sidecar in path.rglob("*.ablator.json"):
        try:
            entries.append(_read_sidecar(sidecar, path))
        except ValueError as exc:
            errors.append(str(exc))
    return entries, errors


def scan_entries(root: str) -> list[GCEntry]:
    """Read valid Ablator worktree sidecars under one cache root."""
    entries, _errors = _scan_entries(root)
    return entries


def active_checkouts(jobs: list[dict]) -> set[str]:
    """Checkouts leased by jobs that may still be executing."""
    out: set[str] = set()
    for job in jobs:
        if job.get("status") != "running":
            continue
        checkout = job.get("source_checkout")
        if isinstance(checkout, str) and checkout:
            out.add(str(Path(checkout).expanduser().resolve(strict=False)))
    return out


def _run_git(repo: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", repo, *args], capture_output=True, text=True,
        timeout=60.0, check=False,
    )


def _listed_worktrees(repo: str) -> set[str]:
    result = _run_git(repo, "worktree", "list", "--porcelain")
    if result.returncode != 0:
        return set()
    return {
        str(Path(line.removeprefix("worktree ")).resolve(strict=False))
        for line in result.stdout.splitlines()
        if line.startswith("worktree ")
    }


def _trusted_repository(root: Path, entry: GCEntry) -> TrustedRepository:
    """Derive repository ownership from Git and cross-check all sidecar hints."""
    if not entry.source_repo_path or not os.path.isdir(entry.source_repo_path):
        raise ValueError(f"{entry.checkout}: cannot prove owning Git repository")
    repo_path = os.path.realpath(entry.source_repo_path)
    try:
        repo_common = source_checkout.git_common_dir(repo_path)
    except source_checkout.SourcePreparationError as exc:
        raise ValueError(
            f"{entry.checkout}: cannot prove owning Git repository: {exc}"
        ) from exc

    if os.path.exists(entry.checkout):
        try:
            checkout_common = source_checkout.git_common_dir(entry.checkout)
        except source_checkout.SourcePreparationError as exc:
            raise ValueError(
                f"{entry.checkout}: cannot prove checkout is a Git worktree: {exc}"
            ) from exc
        if checkout_common != repo_common:
            raise ValueError(
                f"{entry.checkout}: sidecar source_repo_path does not own checkout"
            )
    elif entry.checkout not in _listed_worktrees(repo_path):
        raise ValueError(
            f"{entry.checkout}: cannot prove missing checkout belongs to repository"
        )

    if (not entry.source_common_dir
            or os.path.realpath(entry.source_common_dir) != repo_common):
        raise ValueError(
            f"{entry.checkout}: source common directory is missing or inconsistent"
        )
    if not entry.repo_identity:
        raise ValueError(f"{entry.checkout}: repository identity is missing")
    try:
        source_checkout.validate_repo_identity(repo_path, entry.repo_identity)
    except source_checkout.SourcePreparationError as exc:
        raise ValueError(f"{entry.checkout}: repository identity mismatch: {exc}") from exc

    expected_lock = str(
        Path(source_checkout.repository_lock_path(str(root), repo_path)).resolve(strict=False)
    )
    if entry.lock_path != expected_lock:
        raise ValueError(
            f"{entry.checkout}: repository lock does not match verified common directory"
        )
    return TrustedRepository(
        repo_path=repo_path,
        common_dir=repo_common,
        lock_path=expected_lock,
    )


def _remove_entry(
    root: Path, entry: GCEntry, trusted: TrustedRepository,
) -> tuple[str, str | None]:
    """Remove one inactive worktree under its repository lease lock.

    Returns ``(outcome, error)`` where outcome is ``removed``, ``protected``,
    or ``error``. The sidecar is deliberately retained until both worktree
    removal and Git administrative pruning succeed.
    """
    try:
        with source_checkout._locked(trusted.lock_path):
            try:
                current = _read_sidecar(Path(entry.sidecar), root)
            except ValueError as exc:
                return "error", str(exc)
            except FileNotFoundError:
                return "error", f"{entry.checkout}: source lease disappeared during cleanup"

            if current.checkout != entry.checkout or current.lease_id != entry.lease_id:
                return "error", f"{entry.checkout}: source lease identity changed during cleanup"
            if current.active:
                return "protected", None

            try:
                current_trusted = _trusted_repository(root, current)
            except ValueError as exc:
                return "error", str(exc)
            if current_trusted != trusted:
                return "error", f"{entry.checkout}: repository trust changed during cleanup"

            checkout = current.checkout
            repo = trusted.repo_path
            if os.path.exists(checkout):
                result = _run_git(repo, "worktree", "remove", "--force", checkout)
                if result.returncode != 0:
                    detail = (result.stderr or result.stdout).strip()
                    return (
                        "error",
                        f"{checkout}: git worktree remove failed: {detail[:400]}",
                    )
            pruned = _run_git(repo, "worktree", "prune")
            if pruned.returncode != 0:
                detail = (pruned.stderr or pruned.stdout).strip()
                return "error", f"{checkout}: git worktree prune failed: {detail[:400]}"

            try:
                os.remove(current.sidecar)
            except FileNotFoundError:
                pass
            except OSError as exc:
                return "error", f"{checkout}: removed checkout but not sidecar: {exc}"
            return "removed", None
    except source_checkout.SourcePreparationError as exc:
        return "error", f"{entry.checkout}: could not acquire cleanup lease: {exc}"


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
    resolved_root = Path(root).resolve(strict=False)
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
    entries, errors = _scan_entries(root)

    for entry in entries:
        if entry.active or entry.checkout in active:
            protected.append(entry.checkout)
            continue
        if entry.last_used_at > cutoff:
            retained.append(entry.checkout)
            continue
        try:
            trusted = _trusted_repository(resolved_root, entry)
        except ValueError as exc:
            retained.append(entry.checkout)
            errors.append(str(exc))
            continue
        candidates.append(entry.checkout)
        if dry_run:
            continue
        outcome, error = _remove_entry(resolved_root, entry, trusted)
        if outcome == "protected":
            protected.append(entry.checkout)
        elif error:
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
