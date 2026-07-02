"""Live progress parsing from a running job's training log.

Reads the tail of ``<model_path>/<progress_log>`` and extracts the last
``cur/total`` iteration counter (tqdm-style). Both the log filename and
the counter regex are configurable under ``[queue]``:

  progress_log   = "train.log"          # file inside model_path
  progress_regex = "(\\d+)/(\\d+)"       # two capture groups: cur, total

A total equal to 2**31-1 (tqdm sentinel for unbounded runs) falls back
to ``--streaming_max_iterations N`` parsed from the job's extra_args
(pattern configurable via ``progress_cap_regex``).
"""
from __future__ import annotations

import os
import re

DEFAULT_LOG = "train.log"
DEFAULT_REGEX = r"(\d+)/(\d+)"
DEFAULT_CAP_REGEX = r"--streaming_max_iterations[= ](\d+)"
TOTAL_SENTINEL = 2**31 - 1
TAIL_BYTES = 2048


def read_tail(path: str, n: int = TAIL_BYTES) -> str:
    """Return the last n bytes of a file as text ('' if unreadable)."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - n))
            return f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def parse_progress(tail: str, extra_args: str = "",
                   counter_regex: str = DEFAULT_REGEX,
                   cap_regex: str = DEFAULT_CAP_REGEX) -> str:
    """Format 'iter cur/total (pct%)' from a log tail, or ''."""
    matches = re.findall(counter_regex, tail)
    if not matches:
        return ""
    cur, total = (int(x) for x in matches[-1])
    if total == TOTAL_SENTINEL:
        m = re.search(cap_regex, extra_args or "")
        if not m:
            return f"iter {cur}/?"
        total = int(m.group(1))
    pct = 100 * cur // total if total else 0
    return f"iter {cur}/{total} ({pct}%)"


def job_progress(job: dict, base_dir: str, qcfg: dict) -> str:
    """Live progress string for one job (empty if no log / no counter)."""
    mp = job.get("model_path", "")
    if not mp:
        return ""
    if not os.path.isabs(mp):
        mp = os.path.join(base_dir, mp)
    log = os.path.join(os.path.realpath(mp), qcfg.get("progress_log", DEFAULT_LOG))
    return parse_progress(read_tail(log), job.get("extra_args", ""),
                          counter_regex=qcfg.get("progress_regex", DEFAULT_REGEX),
                          cap_regex=qcfg.get("progress_cap_regex", DEFAULT_CAP_REGEX))
