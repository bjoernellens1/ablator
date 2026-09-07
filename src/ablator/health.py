"""Passive health probe for running jobs, derived from their artifacts.

Runs are standalone; this module observes their standard artifacts
(the progress log, result files, process liveness supplied by the caller)
and never injects anything into a run. The runner reads health and acts;
a job started by hand behaves identically and needs no runner at all.

Health dict:
  {"state": "starting"|"training"|"reporting"|"finishing"|"done"|"hung"|"crashed",
   "iter": int|None, "total": int|None, "log_age_s": float|None}

"finishing" means a result_glob/complete_marker/researchflow completion
artifact matched, but the caller-supplied `container_alive` says the job's
own container is still running -- see job_health()'s container_alive
param docstring. It is never "done" until the container has exited (or no
container runtime is available to check).

Configurable under [queue] (or per-type, or per-job -- see below):
  progress_log, progress_regex, progress_cap_regex   (as in progress.py)
  result_glob        success marker glob relative to model_path resolution
  hung_after_min     minutes without log writes before "hung" (default 20)
  crash_markers      list of substrings meaning "crashed"

`result_glob` resolution order is per-job `job["result_glob"]` (settable
directly on a queue job, or via a spec's `result_glob` at the base/arm/spec
level -- see ablator.spec.expand_spec) > the type's own `result_glob`
(layered into qcfg by runner._health_qcfg) > `[queue] result_glob` >
DEFAULT_RESULT_GLOB. This matters for multi-phase trainers whose final
artifact differs from their queue-wide default (e.g. splatograph's
causal_mapping trainer with `--streaming_post_mapping_refinement_steps N
> 0` writes an interim `comparison/mapping_endpoint/report.json` long
before its final `comparison/iter_<N>/report.json`; a job pinned to the
latter via `"result_glob": "comparison/iter_*/report.json"` is not
misread as done at the interim artifact).

`type: "researchflow"` external jobs (see docs/external-scheduler.md) have no
model_path and are never covered by the config keys above; see
researchflow_done_marker() for their own, independent completion evidence.
"""
from __future__ import annotations

import glob
import os
import re
import time

from . import progress as progmod

DEFAULT_HUNG_AFTER_MIN = 20.0
DEFAULT_RESULT_GLOB = "comparison/*/report.json"
# Splatograph's shared output-staging finalizer (train.py's
# _finalize_output_staging / splatograph.runtime.finalize) writes this
# zero-byte marker at the resolved model_path, for ANY trainer entry point
# (legacy train.py, train_streaming.py, the causal_mapping trainer), the
# instant a staged run's local scratch has been fully mirrored onto its
# canonical persist_model_path -- independent of whether that trainer also
# produces a `comparison/*/report.json`. Some trainer/run-type combinations
# never produce a report.json at all (e.g. the causal_mapping trainer with
# no evaluation holdout configured, `holdout_reserved: 0` in its own
# `causal_replay_summary.json`) while still completing entirely correctly
# (exit 0, real progress the whole way through). Treating result_glob as
# the ONLY completion evidence false-classified two such genuinely-complete
# smoke-check runs as "crashed" (found 2026-08-29, r9700
# fr3batchsmoke_smoke / scannetppsmoke_smoke -- both had `.COMPLETE` and a
# valid causal_replay_summary.json on disk, real train.log progress right
# up to a clean exit 0, and zero kernel/OOM/traceback evidence anywhere,
# yet were quarantined as `error_category: unknown` twice each). See
# docs/health.md.
DEFAULT_COMPLETE_MARKER = ".COMPLETE"
CRASH_TAIL_BYTES = 4096


def complete_markers(qcfg: dict) -> tuple[str, ...]:
    """Resolve the configured `complete_marker` into a tuple of candidate
    filenames, ANY of which existing at the resolved model_path counts as
    completion evidence.

    Accepts a single string (the common case, e.g. ".COMPLETE") or a
    list/tuple of strings. Needed because `.COMPLETE` is reliable evidence
    for splatograph's train.py/train_streaming.py (written once, at the very
    end of ANY trainer's output-staging finalization) but NOT for every
    trainer: confirmed live 2026-08-29 that the causal_mapping trainer's
    `.COMPLETE` is written on some attempts and not others (its own
    finalization does not go through the shared output-staging tail that
    guarantees this for the other two trainers) — one real job had a
    `.COMPLETE` mtime ~35 minutes stale across two back-to-back genuinely-
    successful reruns, while its `causal_replay_summary.json` (written near
    the end of that trainer's mapping phase, before any post-mapping
    refinement) was freshly rewritten on both. A site that runs
    causal_mapping jobs can list both:
        complete_marker = [".COMPLETE", "causal_replay_summary.json"]
    Caveat carried over into any such config: `causal_replay_summary.json`
    is written BEFORE that trainer's post-mapping refinement phase, so a job
    killed during refinement would read as done under this marker. This is
    a strictly narrower failure than every causal_mapping job with no
    refinement configured (the case in production today) being quarantined
    unconditionally -- but it is a real, known gap, not a full fix; the
    correct long-term fix is for causal_entry.py to route through the same
    shared output-staging completion tail (splatograph.runtime.finalize)
    that train.py/train_streaming.py already use.
    """
    v = qcfg.get("complete_marker", DEFAULT_COMPLETE_MARKER)
    if not v:
        return ()
    if isinstance(v, str):
        return (v,)
    return tuple(v)

DEFAULT_CRASH_MARKERS = [
    "Traceback (most recent call last)",
    "CUDA error",
    "HIP error",
    "std::exception",
    "Segmentation fault",
    "core dumped",
]


def researchflow_done_marker(job: dict) -> str | None:
    """Path to a ``type: "researchflow"`` external job's own completion marker.

    External jobs submitted via `external.py` (see docs/external-scheduler.md)
    carry no `model_path` at all -- everything above this point in the module
    (result_glob, complete_marker) resolves relative to `model_path` and is
    permanently unable to see them, so every such job was stuck reporting
    "no completion artifact" forever, however long it had actually already
    finished (confirmed live: varsplat x office0 stuck at status="running"
    well after ResearchFlow's own execute-job had already written its
    completion evidence).

    ResearchFlow's `execute_job` (src/researchflow/execution.py) writes
    `<state-dir>/jobs/<job_id>.done` -- containing the plan sha256, never
    zero-byte -- only once its own subprocess exits 0 AND its evidence
    contract (`<state-dir>/jobs/<job_id>.evidence.json`) validates; a
    process that fails, or one whose output fails evidence validation,
    never gets a `.done` file at all. So `.done` alone is authoritative:
    there is no case where checking it can produce a false "done" that
    checking both files together would have avoided.

    `<state-dir>` is not itself submitted as a job field, but the fixed
    relative layout is recoverable from two `external_metadata` fields
    ResearchFlow already stamps onto every job it submits: the sibling
    `experiment_origin.json` it writes at `<state-dir>/experiment_origin.json`
    (given here as `researchflow_origin_path`) and the job's own
    `researchflow_job_id`. Absent either field (e.g. a non-ResearchFlow
    caller reusing the `researchflow` type name directly), there is nothing
    to resolve and this returns None -- same as no marker configured.
    """
    if job.get("type") != "researchflow":
        return None
    metadata = job.get("external_metadata") or {}
    origin_path = metadata.get("researchflow_origin_path")
    job_id = metadata.get("researchflow_job_id")
    if not origin_path or not job_id:
        return None
    state_dir = os.path.dirname(str(origin_path))
    return os.path.join(state_dir, "jobs", f"{job_id}.done")


def parse_iter(tail: str, extra_args: str = "",
               counter_regex: str = progmod.DEFAULT_REGEX,
               cap_regex: str = progmod.DEFAULT_CAP_REGEX,
               ) -> tuple[int | None, int | None]:
    """Last 'cur/total' counter from a log tail -> (cur, total|None)."""
    matches = re.findall(counter_regex, tail)
    if not matches:
        return None, None
    cur, total = (int(x) for x in matches[-1])
    if total == progmod.TOTAL_SENTINEL:
        m = re.search(cap_regex, extra_args or "")
        total = int(m.group(1)) if m else None
    return cur, total


def resolve_model_path(model_path: str, base_dir: str) -> str:
    if not os.path.isabs(model_path):
        model_path = os.path.join(base_dir, model_path)
    return os.path.realpath(model_path)


def hung_after_s(qcfg: dict, job: dict | None = None) -> float:
    """Hung threshold in seconds: per-job override > [queue] > default."""
    v = (job or {}).get("hung_after_min", qcfg.get("hung_after_min"))
    try:
        return float(v) * 60.0
    except (TypeError, ValueError):
        return DEFAULT_HUNG_AFTER_MIN * 60.0


def job_health(job: dict, base_dir: str, qcfg: dict | None = None,
               process_alive: bool | None = None,
               now: float | None = None,
               container_alive: bool | None = None) -> dict:
    """Derive run health purely from the run's own artifacts.

    process_alive: caller-supplied liveness of the launching subprocess /
    container (None = unknown). A dead process without a success marker
    means the run died before finishing.

    container_alive: caller-supplied liveness of the job's own container
    (e.g. `docker/podman ps` for `splat_train_<job_id>`), independent of
    `process_alive` -- needed for callers (the daemon's stale-running
    reconciler in particular) that have no live subprocess handle at all
    because they are re-observing a job after a runner restart, yet a
    real container may still be executing it. None = unknown/unchecked
    (e.g. no container runtime available), in which case this behaves
    exactly as if it had never been passed.

    True is authoritative evidence the run has NOT finished yet, no
    matter what result_glob/complete_marker/researchflow marker say: a
    multi-phase trainer (e.g. splatograph's causal_mapping with
    `--streaming_post_mapping_refinement_steps N > 0`) writes its first
    `comparison/mapping_endpoint/report.json` minutes before its final
    `comparison/iter_<N>/report.json`, so a result_glob/complete_marker
    match alone is not sufficient evidence of completion while the
    container that would go on to write the final artifact is still
    up -- confirmed live 2026-09 on psnr26 causal_mapping jobs marked
    "done" (and re-dispatched by a second host) minutes before their
    container actually exited. When container_alive is True, it also
    supersedes `process_alive is False` in the crash heuristics below --
    the caller passing process_alive=False here only ever means "I have
    no live subprocess handle", not "the run is dead", and a live
    container is strictly better evidence than the absence of one.
    """
    qcfg = qcfg or {}
    now = time.time() if now is None else now
    # Effective process liveness for the crash heuristics below: a live
    # container is authoritative "not dead" evidence and overrides a
    # caller-supplied process_alive=False that only reflects "no local
    # subprocess handle" (see container_alive docstring above).
    effective_alive = True if container_alive else process_alive
    mp = resolve_model_path(job.get("model_path", ""), base_dir)
    log = os.path.join(mp, qcfg.get("progress_log", progmod.DEFAULT_LOG))
    markers = qcfg.get("crash_markers", DEFAULT_CRASH_MARKERS)
    # Per-job override > [queue]/type-merged qcfg (see runner._health_qcfg
    # for the type-level layering) > default.
    result_glob = job.get("result_glob", qcfg.get("result_glob", DEFAULT_RESULT_GLOB))
    # cli.py's `collect` documents (and configs in the wild use)
    # "{model_path}/comparison/*/report.json" — a template resolved via
    # str.format() against job vars, relative to the type's cwd. This
    # function's own docstring/DEFAULT_RESULT_GLOB documents the OTHER
    # convention: a bare pattern relative to `mp` (already resolved
    # above), no placeholder. The same config value must satisfy both
    # call sites, so normalize the {model_path}-prefixed form down to
    # the bare form here — this function never templates other
    # placeholders (scene/extra_args/...), only this one specific,
    # already-resolved prefix. Confirmed live: without this, every job
    # using the documented {model_path}/... convention was 100% falsely
    # classified as incomplete (glob'd for a literal, never-existing
    # "{model_path}" subdirectory), triggering pointless full retries.
    if result_glob:
        for prefix in ("{model_path}/", "{model_path}"):
            if result_glob.startswith(prefix):
                result_glob = result_glob[len(prefix):]
                break

    h: dict = {"state": "starting", "iter": None, "total": None, "log_age_s": None}

    if result_glob and glob.glob(os.path.join(mp, result_glob)):
        h["state"] = "done"

    if any(os.path.exists(os.path.join(mp, marker))
           for marker in complete_markers(qcfg)):
        # Trainer-agnostic completion evidence -- see DEFAULT_COMPLETE_MARKER
        # / complete_markers() docstrings above. This is intentionally an OR
        # with result_glob, not a replacement: a result_glob match stays the
        # richer, preferred signal (and is what `ablator collect`/
        # gradeability tooling reads), this is only a second, independent
        # way to reach "done" for run types whose completion contract never
        # produces that artifact.
        h["state"] = "done"

    researchflow_marker = researchflow_done_marker(job)
    if researchflow_marker and os.path.exists(researchflow_marker):
        # Independent OR, same rationale as complete_markers() above: a
        # `type: "researchflow"` external job's own `.done` file is the only
        # completion evidence it will ever produce at this layer (it has no
        # model_path), so this is not a fallback for the checks above -- for
        # this job type, it is the ONLY signal that can ever fire.
        h["state"] = "done"

    if h["state"] == "done" and container_alive:
        # Artifact-based completion evidence exists, but the job's own
        # container is still running -- see the container_alive param
        # docstring above. Downgrade off "done" here (before the early
        # returns below can lock it in) so the log tail is still
        # consulted for a more specific state (training/reporting/hung);
        # "finishing" is the fallback when the log gives no better signal.
        h["state"] = "finishing"

    try:
        h["log_age_s"] = round(now - os.path.getmtime(log), 1)
    except OSError:
        # No log yet: either just starting, or died before writing it.
        if h["state"] not in ("done", "finishing") and effective_alive is False:
            h["state"] = "crashed"
        return h

    tail = progmod.read_tail(log, CRASH_TAIL_BYTES)
    h["iter"], h["total"] = parse_iter(
        tail, job.get("extra_args", ""),
        counter_regex=qcfg.get("progress_regex", progmod.DEFAULT_REGEX),
        cap_regex=qcfg.get("progress_cap_regex", progmod.DEFAULT_CAP_REGEX))
    if h["state"] == "done":
        return h

    if any(m in tail for m in markers) or effective_alive is False:
        h["state"] = "crashed"
    elif h["log_age_s"] > hung_after_s(qcfg, job):
        h["state"] = "hung"
    elif h["iter"] is not None and h["total"] and h["iter"] >= h["total"]:
        h["state"] = "reporting"
    elif h["iter"] is not None:
        h["state"] = "training"
    return h
