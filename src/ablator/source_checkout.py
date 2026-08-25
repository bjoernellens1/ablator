"""Immutable source preparation for Git-SHA-pinned jobs.

A pinned job must not execute from the operator's mutable development checkout.
This module turns ``requested_git_sha`` into a deterministic detached Git
worktree on the machine that will execute the job, rewrites the job type config
to use that worktree, and provides the hard requested-vs-executed provenance
check used immediately before workload launch.

The implementation is intentionally stdlib + git only.  Existing mutable jobs
remain a no-op and keep their historical behavior.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

try:  # Linux is the production target; keep import failure explicit on others.
    import fcntl
except ImportError:  # pragma: no cover - CI/production are POSIX
    fcntl = None  # type: ignore[assignment]


class SourcePreparationError(RuntimeError):
    """A pinned source could not be prepared or proven safe to execute."""


@dataclass(frozen=True)
class PreparedSource:
    type_config: dict
    checkout_path: str | None = None
    requested_git_sha: str | None = None
    source_repo: str | None = None


_CONTAINER_RUNTIMES = {"docker", "podman"}
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _run(cmd: list[str], *, cwd: str | None = None, timeout: float = 90.0
         ) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True,
            timeout=timeout, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SourcePreparationError(f"command failed to start/finish: {cmd!r}: {exc}") from exc


def _git(repo: str, *args: str, timeout: float = 90.0
         ) -> subprocess.CompletedProcess[str]:
    return _run(["git", "-C", repo, *args], timeout=timeout)


def _require_ok(result: subprocess.CompletedProcess[str], what: str) -> str:
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise SourcePreparationError(f"{what} failed (rc={result.returncode}): {detail[:600]}")
    return result.stdout.strip()


def _is_git_repo(path: str | None) -> bool:
    if not path or not os.path.isdir(path):
        return False
    try:
        return _git(path, "rev-parse", "--git-dir", timeout=10).returncode == 0
    except SourcePreparationError:
        return False


def _origin_url(repo: str) -> str | None:
    result = _git(repo, "remote", "get-url", "origin", timeout=10)
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _cache_root(cfg: dict, machine: str) -> str:
    machine_cfg = (cfg.get("machines") or {}).get(machine) or {}
    configured = (
        machine_cfg.get("git_worktree_root")
        or (cfg.get("git") or {}).get("worktree_root")
        or os.environ.get("ABLATOR_GIT_WORKTREE_ROOT")
    )
    return os.path.abspath(os.path.expanduser(
        configured or "~/.cache/ablator/worktrees"
    ))


def _repo_key(identity: str) -> str:
    stripped = identity.rstrip("/")
    stem = os.path.basename(stripped).removesuffix(".git") or "repo"
    stem = _SAFE_NAME_RE.sub("-", stem).strip("-._") or "repo"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    return f"{stem}-{digest}"


@contextmanager
def _locked(path: str) -> Iterator[None]:
    """Cross-process lock for one repository cache namespace."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a+") as handle:
        if fcntl is None:
            raise SourcePreparationError("Git worktree materialization requires POSIX file locking")
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _ensure_source_repo(source_cwd: str | None, git_repo: str | None,
                        cache_root: str) -> tuple[str, str]:
    """Return ``(git repo path, stable identity)``.

    Prefer the type's already-configured repository so existing SSH/local
    authentication continues to work.  If that checkout does not exist, a
    structured ``git.repo`` declaration can bootstrap a local bare mirror.
    """
    if _is_git_repo(source_cwd):
        assert source_cwd is not None
        source = os.path.realpath(source_cwd)
        identity = git_repo or _origin_url(source) or source
        return source, identity

    if not git_repo:
        raise SourcePreparationError(
            "pinned job has no usable type cwd Git repository and no git.repo fallback"
        )

    identity = git_repo
    mirror_root = os.path.join(cache_root, "_repos")
    mirror = os.path.join(mirror_root, f"{_repo_key(identity)}.git")
    lock = os.path.join(cache_root, "_locks", f"mirror-{_repo_key(identity)}.lock")
    with _locked(lock):
        if not _is_git_repo(mirror):
            os.makedirs(mirror_root, exist_ok=True)
            result = _run(["git", "clone", "--mirror", git_repo, mirror], timeout=180)
            _require_ok(result, f"clone mirror {git_repo}")
        else:
            result = _git(mirror, "remote", "set-url", "origin", git_repo, timeout=15)
            _require_ok(result, f"refresh mirror remote {git_repo}")
    return mirror, identity


def _commit_available(repo: str, sha: str) -> bool:
    return _git(repo, "cat-file", "-e", f"{sha}^{{commit}}", timeout=15).returncode == 0


def _ensure_commit(repo: str, sha: str) -> None:
    if _commit_available(repo, sha):
        return
    fetch = _git(repo, "fetch", "--no-tags", "origin", sha, timeout=180)
    _require_ok(fetch, f"fetch requested Git SHA {sha}")
    if not _commit_available(repo, sha):
        raise SourcePreparationError(
            f"requested Git SHA {sha} is still unavailable after successful fetch"
        )


def _checkout_state(path: str) -> tuple[str, str]:
    head = _require_ok(_git(path, "rev-parse", "HEAD", timeout=15),
                       f"read checkout HEAD at {path}")
    status = _require_ok(_git(path, "status", "--porcelain", timeout=15),
                         f"read checkout status at {path}")
    return head, status


def _materialize(repo: str, identity: str, sha: str, cache_root: str) -> str:
    repo_key = _repo_key(identity)
    checkout = os.path.join(cache_root, repo_key, sha)
    lock = os.path.join(cache_root, "_locks", f"{repo_key}-{sha}.lock")

    with _locked(lock):
        _ensure_commit(repo, sha)
        if os.path.exists(checkout):
            if not _is_git_repo(checkout):
                raise SourcePreparationError(
                    f"cached checkout path exists but is not a Git worktree: {checkout}"
                )
            head, status = _checkout_state(checkout)
            if head != sha:
                raise SourcePreparationError(
                    f"cached checkout {checkout} is at {head}, expected {sha}"
                )
            if status:
                raise SourcePreparationError(
                    f"cached immutable checkout {checkout} is dirty; refusing reuse"
                )
        else:
            os.makedirs(os.path.dirname(checkout), exist_ok=True)
            result = _git(repo, "worktree", "add", "--detach", checkout, sha, timeout=90)
            _require_ok(result, f"materialize worktree for {sha}")
            head, status = _checkout_state(checkout)
            if head != sha or status:
                raise SourcePreparationError(
                    f"new worktree verification failed: head={head!r} dirty={bool(status)}"
                )

        # Sidecar metadata lives outside the checkout so it cannot make the
        # immutable worktree dirty.  #31 can build cache GC/leases on this.
        sidecar = f"{checkout}.ablator.json"
        try:
            with open(sidecar, "w") as handle:
                json.dump({"sha": sha, "repo": identity, "checkout": checkout}, handle)
        except OSError:
            pass  # provenance safety does not depend on cache metadata

    return checkout


def _replace_checkout(value, old: str | None, new: str):
    if isinstance(value, str):
        out = value.replace("{repo_cwd}", new)
        if old:
            out = out.replace(old, new)
        return out
    if isinstance(value, list):
        return [_replace_checkout(item, old, new) for item in value]
    if isinstance(value, dict):
        return {key: _replace_checkout(item, old, new) for key, item in value.items()}
    return value


def _container_reaches_checkout(tcfg: dict, source_cwd: str | None) -> bool:
    command = tcfg.get("command") or []
    if not command or command[0] not in _CONTAINER_RUNTIMES or "run" not in command[:2]:
        return True
    for token in command:
        if isinstance(token, str) and (
            "{repo_cwd}" in token or (source_cwd and source_cwd in token)
        ):
            return True
    return False


def prepare_job_source(cfg: dict, job: dict, machine: str, tcfg: dict) -> PreparedSource:
    """Prepare and wire the immutable checkout for a pinned job.

    Unpinned jobs are returned byte-for-byte as a deep copy and perform no Git
    operations.  Pinned jobs are materialized on *this* runner machine; this is
    what makes the same mechanism work for main and independently-running SSH
    satellite runners without coordinator-side checkout manipulation.
    """
    requested = job.get("requested_git_sha")
    if not requested:
        return PreparedSource(type_config=copy.deepcopy(tcfg))

    source_cwd = tcfg.get("cwd")
    if not _container_reaches_checkout(tcfg, source_cwd):
        raise SourcePreparationError(
            "pinned container job does not expose its source checkout to the container; "
            "mount the type cwd or use {repo_cwd} in the command template"
        )

    root = _cache_root(cfg, machine)
    repo, identity = _ensure_source_repo(source_cwd, job.get("git_repo"), root)
    checkout = _materialize(repo, identity, requested, root)
    rewritten = _replace_checkout(copy.deepcopy(tcfg), source_cwd, checkout)
    rewritten["cwd"] = checkout
    return PreparedSource(
        type_config=rewritten,
        checkout_path=checkout,
        requested_git_sha=requested,
        source_repo=identity,
    )


def verify_executed_provenance(job: dict, state: dict) -> str | None:
    """Hard execution contract for a pinned job.

    Returns the executed SHA for pinned jobs and ``None`` for legacy jobs.
    Any uncertainty is a failure: an immutable source claim is only useful if
    we refuse to spend workload time when it cannot be proven exactly.
    """
    requested = job.get("requested_git_sha")
    if not requested:
        return None

    executed = state.get("commit")
    if executed != requested:
        raise SourcePreparationError(
            f"source provenance mismatch: requested {requested}, executed {executed or 'unknown'}"
        )
    if state.get("dirty") is not False:
        raise SourcePreparationError(
            "source provenance cannot be trusted: pinned execution checkout is dirty or unreadable"
        )
    return executed
