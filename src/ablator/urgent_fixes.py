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

Second incident (2026-08-12): a bare-metal job was claimed by a machine
whose checkout predated a merged PR that added a new CLI flag entirely
(`train.py: error: unrecognized arguments`), because nobody had added
that commit's SHA to `[[urgent_fixes.fixes]]` -- the pinned-SHA list only
protects against drift a human remembered to register, and in practice
that registration was made once (2026-07-07) and never kept current.
`auto_sync_ref` (optional, alongside or instead of `fixes`) generalizes
the same safe fetch/fast-forward/pause machinery to "this checkout must
not be behind <ref>" (e.g. `origin/main`), so every future commit is
covered automatically with no per-fix bookkeeping. Same safety contract
as the pinned-SHA path exactly: only ever fast-forwards on an idle,
clean tree; anything else pauses the machine for a human, never forces.

Third incident (splatograph issue #629, 2026-08-14/15): a pause set here
(category "urgent_fix_unsynced") has no re-check or expiry once written,
so a transient `git fetch` failure paused a machine for ~9.5 hours after
the transport problem had already cleared. `_revalidate_urgent_fix_unsynced`
below (registered with pause_revalidation.py) closes that gap by
re-running THIS gate's own missing-fixes check on every idle loop tick
while paused, and auto-clearing only once it genuinely passes -- see
pause_revalidation.py's module docstring for why a blind TTL, or clearing
on "the fetch command now succeeds" alone, are both wrong.
"""
from __future__ import annotations

import subprocess

from .provenance import _run_git
from .pause_revalidation import register_auto_revalidator


def load_urgent_fixes(
    cfg: dict, machine: str | None = None,
) -> tuple[str | None, list[dict], str | None]:
    """Returns (repo_cwd, fixes, auto_sync_ref).
    Empty/missing config -> (None, [], None).

    `repo_cwd` and `auto_sync_ref` may be overridden per-machine via
    `[urgent_fixes.machines.<machine>]` (e.g. a different user's home
    directory, or a dedicated worktree that machine dispatches jobs
    from) -- found 2026-08-15: rtx3090 runs as user `bjoern1`
    (`/home/bjoern1/...`), not `bjoern`, so a single global `repo_cwd`
    silently pointed at a path that does not exist on that host,
    permanently no-op'ing the sync gate there. `fixes` (the pinned-SHA
    list) intentionally has no per-machine override -- the same set of
    commits must be present everywhere regardless of checkout path."""
    uf = cfg.get("urgent_fixes") or {}
    repo_cwd = uf.get("repo_cwd")
    fixes = uf.get("fixes") or []
    auto_sync_ref = uf.get("auto_sync_ref") or None
    if machine:
        override = (uf.get("machines") or {}).get(machine) or {}
        repo_cwd = override.get("repo_cwd") or repo_cwd
        auto_sync_ref = override.get("auto_sync_ref") or auto_sync_ref
    if not repo_cwd or (not fixes and not auto_sync_ref):
        return None, [], None
    return repo_cwd, fixes, auto_sync_ref


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


def _resolve_ref(repo_cwd: str, ref: str) -> str | None:
    """Resolves `ref` (e.g. "origin/main") to a commit sha in repo_cwd,
    using whatever remote-tracking data is currently on disk -- callers
    that need this to reflect the true remote state must `_fetch()`
    first. None on any git error/unknown ref."""
    try:
        r = subprocess.run(
            ["git", "rev-parse", ref], cwd=repo_cwd,
            capture_output=True, text=True, timeout=10.0, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    return r.stdout.strip() or None


def missing_fixes(
    repo_cwd: str, fixes: list[dict], auto_sync_ref: str | None = None,
) -> list[dict]:
    """Fixes whose sha is NOT (yet) an ancestor of HEAD in repo_cwd, plus
    (if `auto_sync_ref` given) a synthetic entry when HEAD is behind that
    ref's current commit -- same "missing" contract, so callers that
    already know how to fetch/pull/pause on a nonempty result need no
    changes to handle this generalized case."""
    missing = []
    for fx in fixes:
        sha = fx.get("sha")
        if not sha:
            continue
        if _is_ancestor(repo_cwd, sha) is not True:
            missing.append(fx)
    if auto_sync_ref:
        ref_sha = _resolve_ref(repo_cwd, auto_sync_ref)
        if ref_sha is None or _is_ancestor(repo_cwd, ref_sha) is not True:
            missing.append({
                "sha": ref_sha or auto_sync_ref,
                "subject": f"auto_sync_ref {auto_sync_ref} (checkout currency)",
            })
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


def _ls_remote_ok(repo_cwd: str) -> bool:
    """Genuinely read-only connectivity probe -- no ref mutation, unlike
    `git fetch` (which updates remote-tracking refs even when nothing
    else about it changes). Used only to decide whether transport is back
    before re-attempting the normal fetch/ff-pull path during
    re-validation; see the #629 incident note above for why this needs
    to be a separate step rather than just retrying `git fetch` and
    reading its exit code."""
    try:
        r = subprocess.run(
            ["git", "ls-remote", "origin", "HEAD"], cwd=repo_cwd,
            capture_output=True, text=True, timeout=15.0, check=False,
        )
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

    repo_cwd, fixes, auto_sync_ref = load_urgent_fixes(cfg, machine)
    if repo_cwd is None:
        return True  # feature not configured -- pure no-op

    # Fetch FIRST, unconditionally, before ever consulting missing_fixes().
    #
    # Found 2026-08-15 (issue splatograph#752): missing_fixes() with an
    # auto_sync_ref resolves that ref from whatever remote-tracking data is
    # ALREADY on disk (see its own docstring) -- it does not fetch. The old
    # code only called _fetch() *after* this first check already believed
    # something was missing. If a host's local `origin/main` ref is itself
    # stale and happens to already equal local HEAD (e.g. this host hasn't
    # dispatched in a while, so nothing has triggered a fetch), the first
    # check reports "nothing missing" and returns True immediately at line
    # ~231 (old numbering) without ever fetching -- permanently blind to a
    # real, growing gap. This is a self-reinforcing deadlock: fetching is
    # the ONLY thing that ever updates that stale ref, and the gate is the
    # only caller of fetch, but the gate only fetches once it already
    # (wrongly) suspects a gap. Confirmed live: r9700 fell 44 commits
    # behind and a full training run there silently landed inside the
    # (already-fixed-on-main) #646/#647 reportless window before being
    # caught. A fetch failure here does not by itself pause the machine --
    # only a genuine, post-fetch missing-fixes gap does, exactly as below.
    _fetch(repo_cwd)

    missing = missing_fixes(repo_cwd, fixes, auto_sync_ref)
    if not missing:
        return True

    shas = ", ".join(fx["sha"][:12] for fx in missing if fx.get("sha"))
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

    ok, out = _ff_pull(repo_cwd)
    missing_after = missing_fixes(repo_cwd, fixes, auto_sync_ref)
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


def _revalidate_urgent_fix_unsynced(cfg: dict, machine: str) -> tuple[bool, str]:
    """Re-checker for pause_revalidation.py, registered against category
    "urgent_fix_unsynced" below. Called on every idle loop tick while a
    machine is paused under this category.

    Re-runs the SAME guarded condition enforce_urgent_fixes() uses
    (missing_fixes' ancestor test), not a blind retry of whatever step
    happened to fail when the pause was set. Per the #629 incident: a
    pause evidenced by "git fetch failed" means the currency check could
    not run -- it is not itself evidence the checkout is behind. So the
    first, network-free thing this does is re-run the ancestor test
    against whatever is on disk right now; only if that still reports
    something missing does it fall through to a genuinely read-only
    connectivity probe (`git ls-remote`, no ref mutation) and then, if
    that succeeds, the identical safe fetch/ff-pull path
    enforce_urgent_fixes() itself uses -- so re-validation can never do
    anything enforce_urgent_fixes() wouldn't also have done on a fresh
    idle tick with no pre-existing pause.
    """
    repo_cwd, fixes, auto_sync_ref = load_urgent_fixes(cfg)
    if repo_cwd is None:
        # Feature has been unconfigured since the pause was set -- the
        # condition it was guarding no longer applies.
        return True, "urgent_fixes is no longer configured"

    missing = missing_fixes(repo_cwd, fixes, auto_sync_ref)
    if not missing:
        return True, (f"local checkout at {repo_cwd} already has every "
                       f"registered fix (ancestor test, no network needed)")

    if _clean_tree(repo_cwd) is not True:
        return False, f"working tree at {repo_cwd} is still dirty/unreadable"

    if not _ls_remote_ok(repo_cwd):
        return False, ("git ls-remote origin HEAD still fails -- transport "
                        "not yet restored")

    if not _fetch(repo_cwd):
        return False, "git fetch still fails despite ls-remote succeeding"

    missing = missing_fixes(repo_cwd, fixes, auto_sync_ref)
    if not missing:
        return True, "fetch closed the gap on re-check (no pull needed)"

    ok, out = _ff_pull(repo_cwd)
    missing_after = missing_fixes(repo_cwd, fixes, auto_sync_ref)
    if ok and not missing_after:
        return True, f"fast-forward pull closed the gap on re-check -- {out!r}"

    return False, (
        f"still missing after fetch/pull (ok={ok}): "
        f"{[fx.get('sha') for fx in missing_after]}; output: {out!r}")


register_auto_revalidator("urgent_fix_unsynced", _revalidate_urgent_fix_unsynced)
