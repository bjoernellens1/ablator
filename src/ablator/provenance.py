"""Code-provenance capture: "what commit/branch, dirty or clean, executed
this job" — asked identically for bare-metal (local git state) and k8s
(baked image label) dispatch paths.

Why this exists: ablation results have repeatedly been invalidated by code
silently drifting between what was intended and what actually executed —
a stale k8s image layer despite imagePullPolicy, a stale COPY'd Dockerfile
context, and (most subtly) a bare-metal job on a remote machine (r9700)
running whatever that machine's OWN independent git checkout happened to
have at launch time, with zero visibility. This module makes that state
loud and durable instead of invisible.
"""
from __future__ import annotations

import subprocess


def _run_git(args: list[str], cwd: str, timeout: float = 10.0) -> str | None:
    try:
        r = subprocess.run(["git", *args], cwd=cwd, capture_output=True,
                           text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    return r.stdout.strip()


def capture_local_git_state(cwd: str, host: str | None = None) -> dict:
    """Capture the git state of the ACTUAL executing host's checkout at
    `cwd` (the job type's configured cwd, e.g. /home/bjoern/git/splatograph).

    Returns a dict with keys: host, cwd, commit, branch, dirty (bool or
    None if undeterminable), error (set + all-None fields on failure — a
    missing git binary or non-repo cwd must never crash job dispatch).
    """
    import socket as _socket
    host = host or _socket.gethostname()
    commit = _run_git(["rev-parse", "HEAD"], cwd)
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd)
    status = _run_git(["status", "--porcelain"], cwd)
    dirty = None if status is None else bool(status)
    out = {
        "host": host,
        "cwd": cwd,
        "commit": commit,
        "branch": branch,
        "dirty": dirty,
    }
    if commit is None:
        out["error"] = f"could not read git state at {cwd!r} (no git repo / git not found?)"
    return out


def check_remote_drift(ssh: str, cwd: str, local_commit: str | None,
                       timeout: float = 15.0) -> dict:
    """Cheap SSH-executed `git rev-parse HEAD` on a remote bare-metal
    machine (e.g. r9700), compared against the dispatching host's own
    current commit. Returns a dict: remote_commit, remote_branch,
    remote_dirty, drift (bool | None), warning (str | None).

    Never raises — an unreachable remote / no-git-there is reported as
    drift=None (undeterminable) with a warning, not a crash; it must never
    block dispatch of a job that would otherwise run fine.
    """
    try:
        r = subprocess.run(
            ["ssh", ssh, f"cd {cwd} && git rev-parse HEAD && "
                        f"git rev-parse --abbrev-ref HEAD && "
                        f"git status --porcelain | head -c1"],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"remote_commit": None, "remote_branch": None,
                "remote_dirty": None, "drift": None,
                "warning": f"could not check r9700 git state via ssh: {e!r}"}
    if r.returncode != 0:
        return {"remote_commit": None, "remote_branch": None,
                "remote_dirty": None, "drift": None,
                "warning": f"remote git check failed (rc={r.returncode}): "
                           f"{r.stderr.strip()[:200]}"}
    lines = r.stdout.splitlines()
    remote_commit = lines[0].strip() if len(lines) > 0 else None
    remote_branch = lines[1].strip() if len(lines) > 1 else None
    remote_dirty = bool(lines[2]) if len(lines) > 2 else False
    drift = None
    warning = None
    if remote_commit and local_commit:
        drift = remote_commit != local_commit
        if drift:
            warning = (f"CODE PROVENANCE DRIFT: dispatching host commit "
                       f"{local_commit[:12]} != remote {ssh} commit "
                       f"{remote_commit[:12]} — this job will run DIFFERENT "
                       f"code than what's checked out here. If intentional "
                       f"(e.g. testing a branch only on this remote), ignore; "
                       f"otherwise sync the checkouts before trusting results.")
    return {"remote_commit": remote_commit, "remote_branch": remote_branch,
            "remote_dirty": remote_dirty, "drift": drift, "warning": warning}


def capture_image_commit_label(image: str, runtime: str = "podman",
                               timeout: float = 20.0) -> dict:
    """Read the org.opencontainers.image.revision LABEL baked into a k8s
    job's container image, via `skopeo inspect` (registry, no local pull
    needed) falling back to `podman inspect` / `docker inspect` (local
    image cache) if skopeo is unavailable.

    Returns dict: image, baked_commit (str | None), source ("skopeo" |
    "podman" | "docker" | None), error (str | None).
    """
    import json as _json
    # Prefer skopeo: cheapest (no image pull), works against the registry
    # directly, so it reflects exactly what a fresh k8s pull would fetch.
    try:
        r = subprocess.run(["skopeo", "inspect", f"docker://{image}"],
                           capture_output=True, text=True, timeout=timeout,
                           check=False)
        if r.returncode == 0:
            data = _json.loads(r.stdout)
            labels = data.get("Labels") or {}
            return {"image": image,
                    "baked_commit": labels.get("org.opencontainers.image.revision"),
                    "source": "skopeo", "error": None}
    except (OSError, subprocess.TimeoutExpired, _json.JSONDecodeError) as e:
        skopeo_err = repr(e)
    else:
        skopeo_err = f"skopeo inspect rc={r.returncode}: {r.stderr.strip()[:200]}"

    for rt in (runtime, "podman", "docker"):
        try:
            r = subprocess.run([rt, "inspect", image], capture_output=True,
                               text=True, timeout=timeout, check=False)
            if r.returncode == 0:
                data = _json.loads(r.stdout)
                labels = (data[0].get("Config", {}).get("Labels") or {}) if data else {}
                return {"image": image,
                        "baked_commit": labels.get("org.opencontainers.image.revision"),
                        "source": rt, "error": None}
        except (OSError, subprocess.TimeoutExpired, _json.JSONDecodeError, IndexError):
            continue

    return {"image": image, "baked_commit": None, "source": None,
            "error": f"could not inspect image {image!r} via skopeo or podman/docker "
                     f"(skopeo: {skopeo_err})"}


def check_image_drift(image: str, local_commit: str | None,
                      runtime: str = "podman") -> dict:
    """Compare a k8s job's baked image commit label against the
    dispatching host's own current commit. Returns capture_image_commit_label's
    dict plus drift (bool | None) and warning (str | None)."""
    info = capture_image_commit_label(image, runtime=runtime)
    drift = None
    warning = None
    baked = info.get("baked_commit")
    if baked and baked != "unknown" and local_commit:
        drift = baked != local_commit
        if drift:
            warning = (f"CODE PROVENANCE DRIFT: k8s image {image!r} was built "
                       f"from commit {baked[:12]} but the dispatching host is "
                       f"currently at {local_commit[:12]} — this job runs "
                       f"OLDER/DIFFERENT code than the current checkout. If "
                       f"intentional (testing a validated older image against "
                       f"unrelated newer changes), ignore; otherwise rebuild "
                       f"and push the image before trusting results.")
    elif not baked:
        warning = (f"CODE PROVENANCE UNKNOWN: k8s image {image!r} has no baked "
                   f"org.opencontainers.image.revision label (built before this "
                   f"feature, or --build-arg GIT_COMMIT was omitted) — cannot "
                   f"verify what code this job actually runs.")
    return {**info, "drift": drift, "warning": warning}


def format_banner(kind: str, prov: dict) -> str:
    """Loud, unmissable single/multi-line banner mirroring splatograph's
    train_streaming.py `[RUN START] ... host=... pid=... model_path=...`
    convention, for durable per-job artifact logging."""
    lines = ["=" * 88, f"[PROVENANCE] kind={kind}"]
    for k, v in prov.items():
        lines.append(f"  {k}={v}")
    lines.append("=" * 88)
    return "\n".join(lines)
