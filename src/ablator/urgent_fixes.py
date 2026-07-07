"""Pre-dispatch "urgent fix" currency gate.

Motivating incident (2026-07-07): two ScanNet++ bare-metal jobs
(spp8b5ccap_ctrl, sppb20acap_ctrl) hit near-total GPU memory exhaustion
because they executed on a checkout that did not yet have commit 00df186
(the camera-image-offload memory-leak fix) on disk. Diagnosis (see
docs/ or the session's advisor consult): this was NOT a stale-command-line
problem -- render_command()/_job_vars() already render extra_args at
DISPATCH time, so a new CLI flag's default is picked up automatically for
any job dispatched after the fix lands, with zero ablator changes needed.
The actual gap is narrower and different: bare-metal (and k8s git-sync,
which pins to the dispatching host's HEAD SHA) jobs run whatever code is
literally on disk in the dispatching host's checkout at the moment
`python train.py` starts -- and that checkout can lag behind a fix that
has already landed upstream if nobody has pulled it on this particular
host yet.

This module closes that gap: before any dispatch (bare-metal or k8s) on a
loop tick, verify the dispatcher's own local checkout (`repo_cwd`) is a
descendant of every registered "urgent fix" commit. If not:
  - host is idle (caller only reaches this after the existing
    resources.machine_busy() check passed) and the tree is clean and a
    fast-forward pull closes the gap -> auto-pull, log loudly, proceed.
  - anything unsafe (dirty tree, fetch failure, non-fast-forward,
    detached HEAD, sha unknown even after fetch) -> PAUSE this
    dispatcher's own machine identity via the existing pause-flag
    mechanism (queue.write_pause_flag/is_paused) and do not dispatch this
    tick. A human clears it (`ablator resume`) once satisfied.

Deliberately does NOT touch any job's `extra_args` or ledger entry: this
gate is about code currency, not experimental configuration, so a job
that deliberately pins old behavior (e.g.
`--no-streaming_offload_camera_images`) is completely unaffected --
nothing here can silently corrupt an ablation arm's intended comparison.
"""
from __future__ import annotations

import subprocess

from .provenance import _run_git


def load_urgent_fixes(cfg: dict) -> tuple[str | None, list[dict]]:
    """Returns (repo_cwd, fixes). Empty/missing config -> (None, [])."""
    uf = cfg.get("urgent_fixes") or {}
    repo_cwd = uf.get("repo_cwd")
    fixes = uf.get("fixes") or []
    if not repo_cwd or not fixes:
        return None, []
    return repo_cwd, fixes


def _is_ancestor(repo_cwd: str, sha: str) -> bool | None:
    """True if `sha` is an ancestor of (or equal to) HEAD. None on a git
    error (missing repo, unreadable sha) -- treated as "not current" by
    the caller, never as "safe to proceed"."""
    try:
        r = subprocess.run(
            ["git", "merge-base", "--is-ancestor", sha, "HEAD"],
            cwd=repo_cwd, capture_output=True, text=True, timeout=10.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode == 0:
        return True
    if r.returncode == 1:
        return False
    return None  # sha unknown to this checkout, or other git error


def missing_fixes(repo_cwd: str, fixes: list[dict]) -> list[dict]:
    """Fixes whose sha is NOT (yet) an ancestor of HEAD in repo_cwd."""
    missing = []
    for fx in fixes:
        sha = fx.get("sha")
        if not sha:
            continue
        if _is_ancestor(repo_cwd, sha) is not True:
            missing.append(fx)
    return missing


def _clean_tree(repo_cwd: str) -> bool | None:
    status = _run_git(["status", "--porcelain"], repo_cwd)
    if status is None:
        return None
    return status == ""


def _fetch(repo_cwd: str) -> bool:
    try:
        r = subprocess.run(["git", "fetch"], cwd=repo_cwd,
                           capture_output=True, text=True, timeout=60.0,
                           check=False)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0


def _ff_pull(repo_cwd: str) -> tuple[bool, str]:
    try:
        r = subprocess.run(["git", "pull", "--ff-only"], cwd=repo_cwd,
                           capture_output=True, text=True, timeout=60.0,
                           check=False)
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, f"git pull --ff-only crashed: {e!r}"
    return r.returncode == 0, (r.stdout + r.stderr).strip()


def enforce_urgent_fixes(cfg: dict, machine: str, q) -> bool:
    """Call once per run_loop tick, AFTER resources.machine_busy(cfg,
    machine) has already confirmed this host is idle (never auto-pull a
    live bind-mounted checkout out from under a running job) and BEFORE
    either the k8s-claim loop or the bare-metal claim_next() call --
    git-sync k8s dispatch pins to this same dispatching host's HEAD, so
    the gate covers both paths identically.

    Returns True if it is safe to proceed with dispatch this tick, False
    if dispatch was paused (caller should skip dispatch and continue the
    loop; the existing pause-flag mechanism already makes claim_next()
    for `machine` refuse new claims, so returning False here is mostly
    about also skipping the k8s-claim loop, which does not consult this
    machine's own pause flag).
    """
    from . import config as cfgmod
    from .queue import write_pause_flag, is_paused

    queue_path = cfgmod.queue_path(cfg)
    if is_paused(queue_path, machine):
        # Already paused (this gate or something else) -- nothing new to
        # do; caller's existing claim_next() calls already no-op.
        return False

    repo_cwd, fixes = load_urgent_fixes(cfg)
    if repo_cwd is None:
        return True  # feature not configured -- pure no-op

    missing = missing_fixes(repo_cwd, fixes)
    if not missing:
        return True

    shas = ", ".join(fx["sha"][:12] for fx in fixes if fx.get("sha"))
    print(f"[ablator] urgent-fix gate: {machine}'s checkout at {repo_cwd} "
          f"is missing {len(missing)} registered urgent fix(es) "
          f"({shas}) -- checking whether a safe fast-forward sync closes "
          f"the gap before dispatching anything", flush=True)

    clean = _clean_tree(repo_cwd)
    if clean is not True:
        evidence = (f"dirty or unreadable working tree at {repo_cwd}; "
                    f"missing fixes: {[fx.get('sha') for fx in missing]}")
        path = write_pause_flag(queue_path, machine, "urgent_fix_unsynced",
                                evidence)
        print(f"[ablator] PAUSING {machine} — urgent_fix_unsynced (dirty "
              f"tree, cannot auto-pull safely): {evidence!r} (flag: {path})",
              flush=True)
        return False

    if not _fetch(repo_cwd):
        evidence = f"git fetch failed at {repo_cwd}"
        path = write_pause_flag(queue_path, machine, "urgent_fix_unsynced",
                                evidence)
        print(f"[ablator] PAUSING {machine} — urgent_fix_unsynced (fetch "
              f"failed): {evidence!r} (flag: {path})", flush=True)
        return False

    # Re-check after fetch: the sha may now be reachable without a pull
    # (e.g. HEAD already includes it once remote-tracking refs update).
    missing = missing_fixes(repo_cwd, fixes)
    if not missing:
        print(f"[ablator] urgent-fix gate: {repo_cwd} already current "
              f"after fetch (no pull needed)", flush=True)
        return True

    ok, out = _ff_pull(repo_cwd)
    missing_after = missing_fixes(repo_cwd, fixes)
    if ok and not missing_after:
        print(f"[ablator] urgent-fix gate: auto fast-forward-pulled "
              f"{repo_cwd} to sync {len(missing)} registered urgent "
              f"fix(es) before dispatch -- {out!r}", flush=True)
        return True

    evidence = (f"git pull --ff-only did not close the gap (ok={ok}, "
                f"still missing: {[fx.get('sha') for fx in missing_after]}); "
                f"output: {out!r}")
    path = write_pause_flag(queue_path, machine, "urgent_fix_unsynced",
                            evidence)
    print(f"[ablator] PAUSING {machine} — urgent_fix_unsynced (auto-pull "
          f"did not resolve): {evidence!r} (flag: {path})", flush=True)
    return False
