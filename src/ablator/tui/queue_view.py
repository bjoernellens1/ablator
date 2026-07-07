"""Pure data-layer helpers for the TUI's queue/runs views: turn raw queue
job dicts into row tuples ready for display. No textual import here --
this is the unit-testable half of the queue/runs screens.
"""
from __future__ import annotations

from .. import config as cfgmod
from ..queue import Queue, job_lane


COLUMNS = ("id", "lane", "status", "machine", "claimed_by", "elapsed")


def _elapsed(job: dict) -> str:
    import time
    start = job.get("claimed_at")
    if not start:
        return ""
    try:
        t0 = time.mktime(time.strptime(start, "%Y-%m-%dT%H:%M:%S"))
        end = job.get("finished_at")
        t1 = time.mktime(time.strptime(end, "%Y-%m-%dT%H:%M:%S")) if end else time.time()
        m = int(t1 - t0) // 60
        return f"{m // 60}h{m % 60:02d}m"
    except ValueError:
        return ""


def load_jobs(cfg: dict) -> list[dict]:
    return Queue(cfgmod.queue_path(cfg)).read()


def job_row(job: dict) -> tuple:
    return (
        job.get("id", ""),
        str(job_lane(job)),
        job.get("status", ""),
        job.get("machine", ""),
        job.get("claimed_by") or "-",
        _elapsed(job),
    )


def queue_rows(jobs: list[dict], name: str | None = None) -> list[tuple]:
    """All jobs (optionally filtered to one ablation `name`), most
    recently claimed first."""
    if name:
        jobs = [j for j in jobs
               if j.get("ablation") == name or j.get("id", "").startswith(name + "_")]
    jobs = sorted(jobs, key=lambda j: j.get("claimed_at") or "", reverse=True)
    return [job_row(j) for j in jobs]


def running_rows(jobs: list[dict]) -> list[tuple]:
    """Just the currently-running jobs -- the "Runs" view."""
    return queue_rows([j for j in jobs if j.get("status") == "running"])


def k8s_job_name(job_id: str) -> str:
    """Same naming rule the runner uses to name the k8s Job object, so the
    TUI can shell out to `kubectl get pod -l job-name=<this>` for a
    selected job -- kept in one place (runner.py) and re-exported here so
    the TUI doesn't duplicate the RFC-1123 sanitization logic."""
    from .. import runner as runnermod
    return runnermod._k8s_job_name(job_id)
