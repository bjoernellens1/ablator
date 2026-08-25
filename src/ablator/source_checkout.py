"""Immutable source preparation for Git-SHA-pinned jobs.

A pinned job must not execute from the operator's mutable development checkout.
This module turns ``requested_git_sha`` into a deterministic detached Git
worktree on the machine that will execute the job, rewrites the job type config
to use that worktree, and provides the hard requested-vs-executed provenance
check used immediately before workload launch.

The implementation is intentionally stdlib + git only. Existing mutable jobs
remain a no-op and keep their historical behavior.

Pinned-job policy deliberately distinguishes two concepts that the historical
mutable-checkout gate conflated:

* explicit ``[[urgent_fixes.fixes]]`` SHAs are safety requirements and MUST be
  ancestors of the requested revision;
* ``auto_sync_ref`` is a freshness policy for a mutable shared checkout. It is
  not applied to an explicitly pinned revision, because doing so would turn a
  user-requested immutable PR/commit validation into an implicit request to run
  a different, newer commit.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import subprocess
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator
from urllib.parse import unquote, urlsplit

try:  # Linux is the production target; keep import failure explicit on others.
    import fcntl
except ImportError:  # pragma: no cover - CI/production are POSIX
    fcntl = None  # type: ignore[assignment]


class SourcePreparationError(RuntimeError):
    """A pinned source could not be prepared or proven safe to execute."""


@dataclass(frozen=True)
class SourceLease:
    checkout: str
    sidecar: str
    lock_path: str
    lease_id: str


@dataclass(frozen=True)
class PreparedSource:
    type_config: dict
    checkout_path: str | None = None
    requested_git_sha: str | None = None
    source_repo: str | None = None
    source_repo_path: str | None = None
    lease: SourceLease | None = None
    state: dict | None = None


_CONTAINER_RUNTIMES = {"docker", "podman"}
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_FULL_GIT_SHA = re.compile(r"^[0-9a-fA-F]{40}$")


def normalize_git_target(
    sha: object,
    repo: object = None,
    *,
    required: bool = False,
    where: str = "job",
) -> tuple[str, str | None] | None:
    """Validate and normalize one immutable Git target.

    Queue producers and runners share this check so a non-plan submission
    cannot bypass the full-SHA contract established by ``spec.expand_spec``.
    """
    if sha is None and repo is None:
        if required:
            raise SourcePreparationError(
                f"{where} requires an immutable Git target (full commit SHA)"
            )
        return None
    if not isinstance(sha, str) or _FULL_GIT_SHA.fullmatch(sha) is None:
        raise SourcePreparationError(
            f"{where}: Git target must be a full 40-character hexadecimal commit SHA"
        )
    if repo is not None and (not isinstance(repo, str) or not repo.strip()):
        raise SourcePreparationError(f"{where}: git.repo must be a non-empty string")
    return sha.lower(), (repo.strip() if isinstance(repo, str) else None)


def job_git_target(
    job: dict, *, required: bool = False, where: str | None = None,
) -> tuple[str, str | None] | None:
    """Return a queue job's normalized registered source identity."""
    return normalize_git_target(
        job.get("requested_git_sha"),
        job.get("git_repo"),
        required=required,
        where=where or f"job {job.get('id')!r}",
    )


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


def git_common_dir(repo: str) -> str:
    """Return the canonical Git common directory for a repository/worktree."""
    value = _require_ok(
        _git(repo, "rev-parse", "--path-format=absolute", "--git-common-dir", timeout=15),
        f"resolve Git common directory at {repo}",
    )
    if not os.path.isabs(value):
        value = os.path.join(repo, value)
    return os.path.realpath(value)


def repository_lock_path(cache_root: str, repo: str) -> str:
    """Derive the repository lock solely from its verified Git common dir."""
    common_dir = git_common_dir(repo)
    digest = hashlib.sha256(common_dir.encode("utf-8")).hexdigest()[:24]
    return os.path.join(cache_root, "_locks", f"common-{digest}.lock")


def _local_repo_from_identity(value: str, *, base: str) -> str | None:
    parsed = urlsplit(value)
    if parsed.scheme == "file":
        return os.path.realpath(os.path.expanduser(unquote(parsed.path)))
    if parsed.scheme or re.match(r"^[^/@:]+@[^:]+:", value):
        return None
    candidate = os.path.expanduser(value)
    if not os.path.isabs(candidate):
        candidate = os.path.join(base, candidate)
    return os.path.realpath(candidate)


def _canonical_remote_identity(value: str, *, base: str) -> str:
    """Normalize transport spellings without retaining embedded credentials."""
    local = _local_repo_from_identity(value, base=base)
    if local is not None:
        return f"file:{local}"
    scp = re.match(r"^(?:[^/@:]+@)?([^:]+):(.+)$", value)
    if scp and "://" not in value:
        host, path = scp.groups()
    else:
        parsed = urlsplit(value)
        host = parsed.hostname or ""
        path = parsed.path
    normalized_path = path.strip("/")
    if normalized_path.endswith(".git"):
        normalized_path = normalized_path[:-4]
    return f"remote:{host.lower()}/{normalized_path}"


def validate_repo_identity(repo: str, declared: str) -> None:
    """Require an explicit git.repo to identify the configured source repo."""
    declared_local = _local_repo_from_identity(declared, base=repo)
    if declared_local is not None and _is_git_repo(declared_local):
        if git_common_dir(declared_local) != git_common_dir(repo):
            raise SourcePreparationError(
                f"git.repo {declared!r} does not match configured checkout common directory"
            )
        return
    origin = _origin_url(repo)
    if not origin:
        raise SourcePreparationError(
            f"git.repo {declared!r} cannot be verified: configured checkout has no origin"
        )
    if _canonical_remote_identity(origin, base=repo) != _canonical_remote_identity(
        declared, base=repo
    ):
        raise SourcePreparationError(
            f"git.repo {declared!r} does not match configured checkout origin {origin!r}"
        )


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


def cache_root(cfg: dict, machine: str) -> str:
    """Public cache-root resolver shared by materialization and GC."""
    return _cache_root(cfg, machine)


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
    authentication continues to work. If that checkout does not exist, a
    structured ``git.repo`` declaration can bootstrap a local bare mirror.
    """
    if _is_git_repo(source_cwd):
        assert source_cwd is not None
        source = os.path.realpath(source_cwd)
        if git_repo:
            validate_repo_identity(source, git_repo)
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


def _initialize_submodules(checkout: str) -> None:
    """Materialize exactly the recursive submodule commits in the superproject."""
    if not os.path.isfile(os.path.join(checkout, ".gitmodules")):
        return
    sync = _git(checkout, "submodule", "sync", "--recursive", timeout=90)
    _require_ok(sync, f"synchronize submodules at {checkout}")
    update = _git(
        checkout,
        "-c", "protocol.file.allow=always",
        "submodule", "update", "--init", "--recursive", "--checkout",
        timeout=300,
    )
    _require_ok(update, f"initialize submodules at {checkout}")


def _submodule_state(checkout: str, *, require_clean: bool = True) -> list[dict]:
    result = _git(checkout, "submodule", "status", "--recursive", timeout=60)
    if result.returncode != 0:
        _require_ok(result, f"read recursive submodule state at {checkout}")
    # The first character is semantic (' ', '-', '+', or 'U'), so do not use
    # _require_ok()'s whitespace-stripped successful value here.
    output = result.stdout.rstrip()
    state: list[dict] = []
    for line in output.splitlines():
        if not line:
            continue
        marker = line[0]
        fields = line[1:].split()
        if len(fields) < 2:
            raise SourcePreparationError(
                f"could not parse submodule state at {checkout}: {line!r}"
            )
        sha, path = fields[:2]
        if marker != " ":
            meaning = {
                "-": "uninitialized",
                "+": "at a different commit",
                "U": "conflicted",
            }.get(marker, f"unexpected marker {marker!r}")
            if require_clean:
                raise SourcePreparationError(f"submodule {path} is {meaning}")
        submodule_path = os.path.join(checkout, path)
        dirty_status = _require_ok(
            _git(
                submodule_path,
                "status", "--porcelain", "--untracked-files=all",
                timeout=30,
            ),
            f"read submodule status at {submodule_path}",
        )
        if dirty_status and require_clean:
            raise SourcePreparationError(f"submodule {path} is dirty")
        state.append({"path": path, "sha": sha, "dirty": bool(dirty_status)})
    return state


def inspect_checkout_state(path: str) -> dict:
    """Capture complete source state without turning drift into an exception."""
    head = _require_ok(
        _git(path, "rev-parse", "HEAD", timeout=15),
        f"read checkout HEAD at {path}",
    )
    symbolic = _git(path, "symbolic-ref", "--quiet", "--short", "HEAD", timeout=15)
    ref = symbolic.stdout.strip() if symbolic.returncode == 0 else "DETACHED"
    submodules = _submodule_state(path, require_clean=False)
    status = _require_ok(
        _git(path, "status", "--porcelain", "--untracked-files=all", timeout=15),
        f"read checkout status at {path}",
    )
    return {
        "commit": head,
        "ref": ref,
        "dirty": bool(status) or any(item["dirty"] for item in submodules),
        "submodules": submodules,
    }


def capture_checkout_state(path: str) -> dict:
    """Capture and require the clean, detached source state of a worktree."""
    state = inspect_checkout_state(path)
    dirty_submodule = next(
        (item["path"] for item in state["submodules"] if item["dirty"]), None
    )
    if dirty_submodule is not None:
        raise SourcePreparationError(f"submodule {dirty_submodule} is dirty")
    if state["dirty"]:
        raise SourcePreparationError(f"immutable checkout {path} is dirty")
    return state


def _atomic_json_write(path: str, payload: dict) -> None:
    temporary = f"{path}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        with open(temporary, "x") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.remove(temporary)
        except FileNotFoundError:
            pass


def read_source_lease(lease: SourceLease | None) -> dict:
    if lease is None:
        raise SourcePreparationError("source lease is missing")
    try:
        with open(lease.sidecar) as handle:
            data = json.load(handle)
    except (OSError, ValueError, TypeError) as exc:
        raise SourcePreparationError(
            f"could not read source lease {lease.sidecar}: {exc}"
        ) from exc
    if data.get("lease_id") != lease.lease_id:
        raise SourcePreparationError(
            f"source lease identity changed at {lease.sidecar}"
        )
    return data


def release_source(prepared: PreparedSource) -> None:
    """Mark one execution worktree inactive after final attestation."""
    lease = prepared.lease
    if lease is None:
        return
    with _locked(lease.lock_path):
        data = read_source_lease(lease)
        data["active"] = False
        data["released_at"] = time.time()
        data["last_used_at"] = data["released_at"]
        _atomic_json_write(lease.sidecar, data)


def _materialize(
    repo: str, identity: str, sha: str, cache_root: str, job_id: str,
) -> tuple[SourceLease, dict]:
    repo_key = _repo_key(identity)
    lease_id = uuid.uuid4().hex
    safe_job_id = _SAFE_NAME_RE.sub("-", str(job_id)).strip("-._")[:80] or "job"
    checkout = os.path.join(
        cache_root, repo_key, sha, f"{safe_job_id}-{lease_id}"
    )
    sidecar = f"{checkout}.ablator.json"
    lock = repository_lock_path(cache_root, repo)
    lease = SourceLease(
        checkout=checkout,
        sidecar=sidecar,
        lock_path=lock,
        lease_id=lease_id,
    )

    with _locked(lock):
        os.makedirs(os.path.dirname(checkout), exist_ok=True)
        try:
            _ensure_commit(repo, sha)
            if os.path.exists(checkout):
                raise SourcePreparationError(
                    f"unique execution checkout path already exists: {checkout}"
                )
            result = _git(repo, "worktree", "add", "--detach", checkout, sha, timeout=90)
            _require_ok(result, f"materialize worktree for {sha}")
            _initialize_submodules(checkout)
            state = capture_checkout_state(checkout)
            if state["commit"] != sha or state["ref"] != "DETACHED":
                raise SourcePreparationError(
                    "new worktree verification failed: "
                    f"head={state['commit']!r} ref={state['ref']!r}"
                )
            _atomic_json_write(sidecar, {
                "schema": "ablator.source-lease/v1",
                "lease_id": lease_id,
                "job_id": str(job_id),
                "active": True,
                "sha": sha,
                "repo": identity,
                "checkout": checkout,
                "source_repo_path": repo,
                "source_common_dir": git_common_dir(repo),
                "lock_path": lock,
                "created_at": time.time(),
                "last_used_at": time.time(),
                "materialization_state": "ready",
            })
        except Exception as exc:
            safe_error = re.sub(r"(://)[^/@\s]+@", r"\1<redacted>@", str(exc))
            try:
                _atomic_json_write(sidecar, {
                    "schema": "ablator.source-lease/v1",
                    "lease_id": lease_id,
                    "job_id": str(job_id),
                    "active": False,
                    "sha": sha,
                    "repo": identity,
                    "checkout": checkout,
                    "source_repo_path": repo,
                    "source_common_dir": git_common_dir(repo),
                    "lock_path": lock,
                    "created_at": time.time(),
                    "last_used_at": time.time(),
                    "materialization_state": "failed",
                    "materialization_error": safe_error[:600],
                })
            except Exception as evidence_exc:
                raise SourcePreparationError(
                    f"{exc}; failed to retain materialization evidence at "
                    f"{sidecar}: {evidence_exc}"
                ) from exc
            raise SourcePreparationError(
                f"{exc}; failed materialization evidence retained at {sidecar}"
            ) from exc

    return lease, state


def _required_fix_shas(cfg: dict) -> list[tuple[str, str]]:
    """Explicit mandatory fixes for every pinned job.

    ``auto_sync_ref`` is intentionally absent here: it governs freshness of
    the shared mutable checkout, whereas a pin is an explicit request for one
    exact historical/review revision. Explicit fix entries remain hard safety
    requirements for both modes.
    """
    out: list[tuple[str, str]] = []
    for fix in (cfg.get("urgent_fixes") or {}).get("fixes") or []:
        sha = fix.get("sha")
        if isinstance(sha, str) and sha:
            out.append((sha, str(fix.get("subject") or "mandatory urgent fix")))
    return out


def _validate_required_fixes(cfg: dict, repo: str, checkout: str,
                             requested: str) -> None:
    for required, subject in _required_fix_shas(cfg):
        _ensure_commit(repo, required)
        result = _git(checkout, "merge-base", "--is-ancestor", required, "HEAD", timeout=20)
        if result.returncode != 0:
            raise SourcePreparationError(
                f"requested Git SHA {requested} omits mandatory urgent fix "
                f"{required} ({subject}); refusing this job without pausing the machine"
            )


def validate_requested_revision_policy(
    cfg: dict, job: dict, machine: str, source_cwd: str | None,
) -> str | None:
    """Validate a pinned revision without creating a host worktree.

    Used by backends such as Kubernetes whose init container materializes the
    checkout itself. The dispatch host still verifies that the requested
    object exists and contains every explicit mandatory urgent fix before any
    workload object is submitted. Returns the stable repository identity for
    pinned jobs and ``None`` for legacy jobs.
    """
    requested = job.get("requested_git_sha")
    if not requested:
        return None
    root = _cache_root(cfg, machine)
    repo, identity = _ensure_source_repo(source_cwd, job.get("git_repo"), root)
    _ensure_commit(repo, requested)
    for required, subject in _required_fix_shas(cfg):
        _ensure_commit(repo, required)
        result = _git(repo, "merge-base", "--is-ancestor", required, requested, timeout=20)
        if result.returncode != 0:
            raise SourcePreparationError(
                f"requested Git SHA {requested} omits mandatory urgent fix "
                f"{required} ({subject}); refusing this job without pausing the machine"
            )
    return identity


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
    if (not command or os.path.basename(str(command[0])) not in _CONTAINER_RUNTIMES
            or "run" not in command[:2]):
        return True
    for token in command:
        if isinstance(token, str) and (
            "{repo_cwd}" in token or (source_cwd and source_cwd in token)
        ):
            return True
    return False


def _path_is_within(path: str, parent: str) -> bool:
    try:
        return os.path.commonpath((path, parent)) == parent
    except ValueError:
        return False


def _checkout_bind(source: str, checkout: str) -> bool:
    lexical_source = os.path.abspath(os.path.expanduser(source))
    lexical_checkout = os.path.abspath(os.path.expanduser(checkout))
    resolved_source = os.path.realpath(lexical_source)
    resolved_checkout = os.path.realpath(lexical_checkout)
    lexical_descendant = _path_is_within(lexical_source, lexical_checkout)
    resolved_descendant = _path_is_within(resolved_source, resolved_checkout)
    if lexical_descendant and not resolved_descendant:
        raise SourcePreparationError(
            f"container bind {source!r} escapes immutable checkout through a symlink"
        )
    return resolved_descendant


def _read_only_volume(spec: str, checkout: str) -> tuple[str, bool]:
    parts = spec.split(":")
    if len(parts) < 2 or not _checkout_bind(parts[0], checkout):
        return spec, False
    options = [item for item in ",".join(parts[2:]).split(",") if item]
    if "ro" not in options and "readonly" not in options:
        options.append("ro")
    suffix = f":{','.join(options)}" if options else ""
    return f"{parts[0]}:{parts[1]}{suffix}", True


def _read_only_long_mount(spec: str, checkout: str) -> tuple[str, bool]:
    items = spec.split(",")
    fields = dict(item.split("=", 1) for item in items if "=" in item)
    source = fields.get("src") or fields.get("source")
    if fields.get("type", "bind") != "bind" or not source:
        return spec, False
    if not _checkout_bind(source, checkout):
        return spec, False
    if not any(item in {"readonly", "ro"} or item.startswith(("readonly=", "ro="))
               for item in items):
        items.append("readonly")
    return ",".join(items), True


def _make_checkout_mount_read_only(tcfg: dict, checkout: str) -> dict:
    command = list(tcfg.get("command") or [])
    if not command or os.path.basename(str(command[0])) not in _CONTAINER_RUNTIMES:
        return tcfg
    if "run" not in command:
        return tcfg

    found = False
    rewritten: list = []
    index = 0
    while index < len(command):
        token = command[index]
        if token in ("-v", "--volume", "--mount") and index + 1 < len(command):
            value = str(command[index + 1])
            if token == "--mount":
                value, matched = _read_only_long_mount(value, checkout)
            else:
                value, matched = _read_only_volume(value, checkout)
            found = found or matched
            rewritten.extend([token, value])
            index += 2
            continue
        if isinstance(token, str) and token.startswith("--volume="):
            value, matched = _read_only_volume(token.split("=", 1)[1], checkout)
            rewritten.append(f"--volume={value}")
            found = found or matched
            index += 1
            continue
        if isinstance(token, str) and token.startswith("--mount="):
            value, matched = _read_only_long_mount(token.split("=", 1)[1], checkout)
            rewritten.append(f"--mount={value}")
            found = found or matched
            index += 1
            continue
        if (isinstance(token, str) and token.startswith("-v")
                and not token.startswith("--") and len(token) > 2):
            value, matched = _read_only_volume(token[2:].removeprefix("="), checkout)
            rewritten.append(f"-v{value}")
            found = found or matched
            index += 1
            continue
        rewritten.append(token)
        index += 1

    if not found:
        raise SourcePreparationError(
            "pinned container job does not bind its prepared source checkout; "
            "use {repo_cwd} in a -v/--volume/--mount source"
        )
    updated = copy.deepcopy(tcfg)
    updated["command"] = rewritten
    return updated


def prepare_job_source(cfg: dict, job: dict, machine: str, tcfg: dict) -> PreparedSource:
    """Prepare and wire the immutable checkout for a pinned job.

    Unpinned jobs are returned byte-for-byte as a deep copy and perform no Git
    operations. Pinned jobs are materialized on *this* runner machine; this is
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
    lease, state = _materialize(repo, identity, requested, root, str(job.get("id") or "job"))
    checkout = lease.checkout
    partial = PreparedSource(
        type_config={},
        checkout_path=checkout,
        requested_git_sha=requested,
        source_repo=identity,
        source_repo_path=repo,
        lease=lease,
        state=state,
    )
    try:
        _validate_required_fixes(cfg, repo, checkout, requested)
        rewritten = _replace_checkout(copy.deepcopy(tcfg), source_cwd, checkout)
        rewritten["cwd"] = checkout
        rewritten = _make_checkout_mount_read_only(rewritten, checkout)
        rewritten.setdefault("env", {})["PYTHONDONTWRITEBYTECODE"] = "1"
    except Exception as exc:
        # Materialization is already durable at this point. Mark the lease
        # inactive when later policy/config wiring rejects the launch so a
        # never-executed checkout is not retained as an active orphan forever.
        # Keep the checkout and sidecar as evidence for normal age-based GC.
        try:
            release_source(partial)
        except SourcePreparationError as release_exc:
            raise SourcePreparationError(
                f"{exc}; source preparation cleanup also failed: {release_exc}"
            ) from exc
        raise
    return PreparedSource(
        type_config=rewritten,
        checkout_path=checkout,
        requested_git_sha=requested,
        source_repo=identity,
        source_repo_path=repo,
        lease=lease,
        state=state,
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
