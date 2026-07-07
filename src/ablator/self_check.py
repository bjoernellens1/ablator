"""Self-drift detection: does THIS running ablator installation's own
codebase match origin/main, or has it silently fallen behind?

Different problem from urgent_fixes.py (which tracks drift in the TARGET
repo, splatograph, that ablator dispatches jobs to) and provenance.py
(which tracks what code a given JOB actually ran). This module checks
ablator's OWN source tree — wherever `import ablator` actually resolves to
on this host (the editable-install checkout, e.g. /home/bjoern/git/ablator)
— against upstream.

Caught live 2026-07-07: r9700's ablator checkout was found 9 commits
behind main's (69caa7b vs fb33d3a), silently, missing essentially every
safety feature built that day (GPU memory guard, provenance tracking,
ghost-running-entry fixes, urgent-fix propagation, the k8s dataset-mount
fix, and the "any"-job fairness fix whose own deployment is what surfaced
the gap). Nobody knew until a human manually SSH'd in and ran `git log`.
This module makes that check automatic and loud instead.

Deliberately does NOT auto-`git pull` ablator's own running code — unlike
urgent_fixes.py's narrow, carefully-gated auto-pull of the TARGET repo,
having ablator rewrite its OWN source out from under its own running
process is a much riskier self-modification (a mid-import/mid-function
pull could leave the process running a mix of old bytecode and new
on-disk source, or crash mid-loop-iteration). A loud warning that a human
should manually `git pull` + restart is the right scope here.
"""
from __future__ import annotations

import os
import subprocess
import time


def _run_git(args: list[str], cwd: str, timeout: float = 10.0) -> str | None:
    try:
        r = subprocess.run(["git", *args], cwd=cwd, capture_output=True,
                           text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    return r.stdout.strip()


def ablator_source_root() -> str | None:
    """Directory containing THIS running installation's `.git` (the
    editable-install checkout root, e.g. /home/bjoern/git/ablator), derived
    from `ablator.__file__` rather than assumed — verifies rather than
    hardcodes the path, since a non-editable / different-layout install
    would resolve differently.
    """
    import ablator as _ablator
    pkg_dir = os.path.dirname(os.path.abspath(_ablator.__file__))
    # editable install layout: <root>/src/ablator/__init__.py
    candidate = os.path.dirname(os.path.dirname(pkg_dir))
    if os.path.isdir(os.path.join(candidate, ".git")):
        return candidate
    # fall back: walk up looking for .git, in case layout ever changes
    d = pkg_dir
    for _ in range(5):
        if os.path.isdir(os.path.join(d, ".git")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


def check_self_drift(repo_root: str | None = None, remote: str = "origin",
                     branch: str = "main", do_fetch: bool = True,
                     fetch_timeout: float = 15.0) -> dict:
    """Compare this host's own ablator checkout HEAD against
    <remote>/<branch>. Never raises — a missing repo / offline host / no
    git binary is reported as undeterminable (behind=None), not a crash;
    this check must never block or crash a runner's startup or loop.

    Returns dict: repo_root, local_commit, remote_commit, behind (int |
    None), ahead (int | None), fetch_ok (bool), error (str | None).
    """
    repo_root = repo_root or ablator_source_root()
    if repo_root is None:
        return {"repo_root": None, "local_commit": None,
                "remote_commit": None, "behind": None, "ahead": None,
                "fetch_ok": False,
                "error": "could not locate ablator's own git checkout "
                         "(not an editable install / no .git found)"}

    local_commit = _run_git(["rev-parse", "HEAD"], repo_root)
    fetch_ok = True
    if do_fetch:
        try:
            fetch_out = subprocess.run(
                ["git", "fetch", "--quiet", remote, branch],
                cwd=repo_root, capture_output=True, text=True,
                timeout=fetch_timeout, check=False,
            )
            fetch_ok = fetch_out.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            fetch_ok = False
    remote_commit = _run_git(["rev-parse", f"{remote}/{branch}"], repo_root)

    behind = ahead = None
    error = None
    if local_commit and remote_commit:
        # Count commits reachable from remote but not local (behind), and
        # vice versa (ahead, e.g. this host has unpushed/local-only work).
        behind_out = _run_git(
            ["rev-list", "--count", f"{local_commit}..{remote_commit}"],
            repo_root)
        ahead_out = _run_git(
            ["rev-list", "--count", f"{remote_commit}..{local_commit}"],
            repo_root)
        behind = int(behind_out) if behind_out and behind_out.isdigit() else None
        ahead = int(ahead_out) if ahead_out and ahead_out.isdigit() else None
    else:
        error = ("could not determine local/remote commit "
                 f"(local={local_commit!r}, remote={remote_commit!r}, "
                 f"fetch_ok={fetch_ok}) — offline host or fetch failure")

    return {"repo_root": repo_root, "local_commit": local_commit,
            "remote_commit": remote_commit, "behind": behind,
            "ahead": ahead, "fetch_ok": fetch_ok, "error": error}


def format_drift_banner(machine: str, drift: dict) -> str | None:
    """Loud, unmissable banner — mirrors this project's `[RUN START]` /
    `[GPU MEMORY DANGER]` / `CODE PROVENANCE UNKNOWN` convention — when
    this host's ablator checkout is behind upstream. Returns None if not
    behind (or undeterminable), so callers can log-and-skip cleanly.
    """
    behind = drift.get("behind")
    if not behind:
        return None
    local = (drift.get("local_commit") or "?")[:12]
    remote = (drift.get("remote_commit") or "?")[:12]
    lines = [
        "!" * 88,
        f"[ABLATOR SELF-DRIFT] {machine}'s ablator installation is "
        f"{behind} commit(s) BEHIND origin/main",
        f"  local  HEAD = {local}  ({drift.get('repo_root')})",
        f"  origin/main  = {remote}",
        "  This runner may be missing recent safety fixes (GPU memory "
        "guard, provenance tracking, urgent-fix propagation, etc).",
        f"  ACTION REQUIRED: manually `git -C {drift.get('repo_root')} "
        f"pull --ff-only origin main` then restart this runner. "
        f"ablator does NOT auto-update its own code.",
        "!" * 88,
    ]
    return "\n".join(lines)


def write_self_version_file(cfg, machine: str, drift: dict) -> None:
    """Write a small, shared, cross-machine-visible status file next to the
    existing heartbeat_<machine>.txt files in the queue dir, so `ablator
    status` (or a human) can see EVERY machine's ablator currency without
    SSHing into each one individually — the whole point of an automatic
    check is not needing a human to manually inspect per-host. Never
    raises (same contract as write_heartbeat)."""
    try:
        from . import config as cfgmod
        qdir = os.path.dirname(cfgmod.queue_path(cfg))
        path = os.path.join(qdir, f"ablator_version_{machine}.txt")
        with open(path, "w") as f:
            f.write(
                f"{machine} {time.strftime('%Y-%m-%dT%H:%M:%S')} "
                f"epoch={time.time():.0f} "
                f"local={drift.get('local_commit')} "
                f"remote={drift.get('remote_commit')} "
                f"behind={drift.get('behind')} "
                f"ahead={drift.get('ahead')} "
                f"error={drift.get('error')}\n"
            )
    except Exception as e:
        print(f"[ablator] self-version write failed: {e!r}", flush=True)


def run_self_check(cfg: dict, machine: str) -> dict:
    """Convenience entry point for run_loop(): checks drift, logs loudly if
    behind, writes the shared status file regardless of outcome (so a
    healthy/current machine also has an up-to-date file — absence of a
    recent file is itself a signal something's stuck). Returns the raw
    drift dict for callers that want it (e.g. periodic re-check timing)."""
    drift = check_self_drift()
    banner = format_drift_banner(machine, drift)
    if banner:
        print(banner, flush=True)
    elif drift.get("error"):
        print(f"[ablator] self-drift check undeterminable on {machine}: "
              f"{drift['error']}", flush=True)
    else:
        print(f"[ablator] self-drift check OK on {machine}: "
              f"up to date with origin/main ({(drift.get('local_commit') or '?')[:12]})",
              flush=True)
    write_self_version_file(cfg, machine, drift)
    return drift
