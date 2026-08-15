"""Re-validation of machine-level pause flags.

Motivating incident (splatograph issue #629, 2026-08-14/15): the runner
auto-paused main's lane with `category=urgent_fix_unsynced`, evidence
`git fetch failed at /home/bjoern/git/splatograph`, at 2026-08-14T20:24:07.
The underlying `git fetch` failure was transient -- retried by hand hours
later, it exited 0 -- but the pause flag has no re-check and no expiry,
so the machine sat out of a running-dry queue for ~9.5 hours with nothing
actually protecting it.

Two wrong fixes were considered and rejected:

  - A blind TTL/expiry re-arms a genuinely-broken machine's pause only
    after some delay, and clears a genuinely-broken one just as readily
    as a transient one -- it tests nothing about the actual condition.
  - Clearing "urgent_fix_unsynced" on the strength of "the git command
    that failed now succeeds" is ALSO wrong: `git fetch failed` records
    that the currency check could not run, not that the checkout is
    current. Re-validation must re-run the SPECIFIC guarded condition
    (here: are the registered urgent fixes actually reachable from
    HEAD?), not just retry the transport step that happened to fail.

So this module re-runs the check that produced the pause, per category,
and clears the flag only if that check now passes:

  - Auto-set categories with a registered re-checker (currently just
    "urgent_fix_unsynced", see urgent_fixes.py) are re-validated on every
    idle loop tick and cleared automatically the moment the guarded
    condition is confirmed resolved.
  - "manual_pause" (an operator ran `ablator pause <machine>`) is NEVER
    auto-cleared -- a human asked for this, and only a human
    (`ablator unpause`) may lift it. There is deliberately no registered
    revalidator for it.
  - Any other category -- including a real hardware fault an operator
    paused for by hand under some other name, a future auto-pause
    category nobody has written a re-checker for yet, or a garbled/
    unrecognized flag -- is the bottom of the lattice: no revalidator
    registered means never auto-cleared. This is the same "unknown is
    not a permissive default" principle CLAUDE.md's experiment-validity
    section applies elsewhere in this project family; here it means an
    unrecognized pause category is treated as conservatively as a known
    non-auto-clearable one, not as "probably fine."

GPU-contention safety: this module is deliberately silent on WHEN it may
be called -- callers (runner.run_loop) are responsible for calling it
only from the same GPU-idle-confirmed position the urgent-fix gate itself
already occupies (after resources.machine_busy() has said this machine's
own GPU is idle). This module clears a *dispatch-eligibility* flag; it
never independently checks for a foreign/uncoordinated container on the
GPU, so auto-clearing a pause here does not (and must not) bypass the
runner's own separate GPU-contention guard -- clearing the flag only ever
makes claim_next() eligible to run again on the NEXT normal, already-
guarded loop tick.
"""
from __future__ import annotations

from typing import Callable

from .queue import clear_pause_flag, is_paused, read_pause_flag

# category -> fn(cfg, machine) -> (resolved: bool, detail: str)
# Populated by the modules that own each auto-pause category (see
# urgent_fixes.py's `register_auto_revalidator("urgent_fix_unsynced", ...)`
# call at import time). Never populate "manual_pause" here.
_AUTO_REVALIDATORS: dict[str, Callable[[dict, str], tuple[bool, str]]] = {}


def register_auto_revalidator(
    category: str, fn: Callable[[dict, str], tuple[bool, str]],
) -> None:
    """Registers `fn` as the re-check for pauses of `category`. `fn`
    receives (cfg, machine) and must return (resolved, detail) -- never
    raise (revalidate_pause already guards against a misbehaving fn, but
    a well-behaved one should not rely on that)."""
    if category == "manual_pause":
        raise ValueError(
            "manual_pause is a human-set category and must never have an "
            "auto-revalidator registered against it")
    _AUTO_REVALIDATORS[category] = fn


def revalidate_pause(cfg: dict, machine: str, q=None) -> bool:
    """Call once per idle loop tick, before any dispatch decision that
    depends on is_paused(machine). If `machine` is currently paused under
    a category with a registered auto-revalidator, re-runs that SPECIFIC
    check (never a blind timer) and clears the flag only if it now
    passes. Any pause without a registered revalidator for its category
    -- including every "manual_pause" -- is left untouched.

    Returns True if `machine` is NOT paused when this call returns
    (whether it was never paused or was just auto-cleared); False if it
    remains paused.
    """
    from . import config as cfgmod

    queue_path = cfgmod.queue_path(cfg)
    if not is_paused(queue_path, machine):
        return True

    info = read_pause_flag(queue_path, machine)
    if info is None:
        # Flag existed a moment ago (is_paused() above) but is gone or
        # unreadable now -- treat as "not paused" rather than guessing.
        return not is_paused(queue_path, machine)

    category = info.get("category", "")
    revalidator = _AUTO_REVALIDATORS.get(category)
    if revalidator is None:
        # manual_pause, or any category nobody has registered a checker
        # for -- bottom of the lattice, never auto-cleared.
        return False

    try:
        resolved, detail = revalidator(cfg, machine)
    except Exception as e:  # noqa: BLE001 - must never break the loop tick
        print(f"[ablator] pause re-validation for {machine} ({category}) "
              f"raised {e!r} -- leaving paused", flush=True)
        return False

    if not resolved:
        print(f"[ablator] pause re-validation for {machine} ({category}): "
              f"still failing -- {detail}", flush=True)
        return False

    cleared = clear_pause_flag(
        queue_path, machine, reason=f"auto_revalidate:{category}: {detail}")
    if cleared:
        print(f"[ablator] AUTO-CLEARED pause on {machine} ({category}) -- "
              f"re-check passed: {detail}", flush=True)
        return True
    print(f"[ablator] pause re-validation for {machine} ({category}) "
          f"passed ({detail}) but clearing the flag failed -- leaving "
          f"paused, will retry next tick", flush=True)
    return False
