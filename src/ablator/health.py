"""Passive health probe for running jobs, derived from their artifacts.

Runs are standalone; this module observes their standard artifacts
(the progress log, result files, process liveness supplied by the caller)
and never injects anything into a run. The runner reads health and acts;
a job started by hand behaves identically and needs no runner at all.

Health dict:
  {"state": "starting"|"training"|"reporting"|"done"|"hung"|"crashed",
   "iter": int|None, "total": int|None, "log_age_s": float|None}

Configurable under [queue]:
  progress_log, progress_regex, progress_cap_regex   (as in progress.py)
  result_glob        success marker glob relative to model_path resolution
  hung_after_min     minutes without log writes before "hung" (default 20)
  crash_markers      list of substrings meaning "crashed"
"""
from __future__ import annotations

import glob
import os
import re
import time

from . import progress as progmod

DEFAULT_HUNG_AFTER_MIN = 20.0
DEFAULT_RESULT_GLOB = "comparison/*/report.json"
CRASH_TAIL_BYTES = 4096

DEFAULT_CRASH_MARKERS = [
    "Traceback (most recent call last)",
    "CUDA error",
    "HIP error",
    "std::exception",
    "Segmentation fault",
    "core dumped",
]


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
               now: float | None = None) -> dict:
    """Derive run health purely from the run's own artifacts.

    process_alive: caller-supplied liveness of the launching subprocess /
    container (None = unknown). A dead process without a success marker
    means the run died before finishing.
    """
    qcfg = qcfg or {}
    now = time.time() if now is None else now
    mp = resolve_model_path(job.get("model_path", ""), base_dir)
    log = os.path.join(mp, qcfg.get("progress_log", progmod.DEFAULT_LOG))
    markers = qcfg.get("crash_markers", DEFAULT_CRASH_MARKERS)
    result_glob = qcfg.get("result_glob", DEFAULT_RESULT_GLOB)

    h: dict = {"state": "starting", "iter": None, "total": None, "log_age_s": None}

    if result_glob and glob.glob(os.path.join(mp, result_glob)):
        h["state"] = "done"

    try:
        h["log_age_s"] = round(now - os.path.getmtime(log), 1)
    except OSError:
        # No log yet: either just starting, or died before writing it.
        if h["state"] != "done" and process_alive is False:
            h["state"] = "crashed"
        return h

    tail = progmod.read_tail(log, CRASH_TAIL_BYTES)
    h["iter"], h["total"] = parse_iter(
        tail, job.get("extra_args", ""),
        counter_regex=qcfg.get("progress_regex", progmod.DEFAULT_REGEX),
        cap_regex=qcfg.get("progress_cap_regex", progmod.DEFAULT_CAP_REGEX))
    if h["state"] == "done":
        return h

    if any(m in tail for m in markers) or process_alive is False:
        h["state"] = "crashed"
    elif h["log_age_s"] > hung_after_s(qcfg, job):
        h["state"] = "hung"
    elif h["iter"] is not None and h["total"] and h["iter"] >= h["total"]:
        h["state"] = "reporting"
    elif h["iter"] is not None:
        h["state"] = "training"
    return h
