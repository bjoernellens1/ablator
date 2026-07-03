"""Shared JSONL job queue with flock claiming.

One queue file on a shared filesystem (NFS is fine: flock over NFSv4
serializes claims across machines). Each line is one job dict:

  {"id": "...", "machine": "<name>"|"any", "type": "<config type>",
   "scene": "...", "model_path": "...", "extra_args": "...",
   "iterations": 30000, "status": "pending", ...}

Statuses: pending -> running -> done | failed -> (retry) -> quarantined;
plus cancelled. depends_on gates claiming on the named job being "done".
"""
from __future__ import annotations

import fcntl
import json
import os
import time
from typing import Callable

DEFAULT_FLOCK_TIMEOUT_S = 60.0


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def pause_flag_path(queue_path: str, machine: str) -> str:
    return os.path.join(os.path.dirname(queue_path), f"paused_{machine}.txt")


def is_paused(queue_path: str, machine: str) -> bool:
    return os.path.exists(pause_flag_path(queue_path, machine))


def write_pause_flag(queue_path: str, machine: str, category: str, evidence: str) -> str:
    path = pause_flag_path(queue_path, machine)
    try:
        with open(path, "w") as f:
            f.write(f"category={category}\n"
                     f"timestamp={_now()}\n"
                     f"evidence={evidence}\n")
    except OSError as e:
        print(f"[ablator] could not write pause flag {path}: {e}", flush=True)
    return path


def clear_pause_flag(queue_path: str, machine: str) -> bool:
    path = pause_flag_path(queue_path, machine)
    try:
        os.remove(path)
        return True
    except OSError:
        return False


def not_before_ok(job: dict, now: float | None = None) -> bool:
    """True if job has no future not_before (i.e. is claimable right now)."""
    nb = job.get("not_before")
    if nb is None:
        return True
    now = time.time() if now is None else now
    try:
        return float(nb) <= now
    except (TypeError, ValueError):
        return True


# --- priority lanes ---------------------------------------------------------
# lane 3 "urgent"     -> claimed first (failed-run redos, fast-answer runs)
# lane 2 "standard"   -> default; validation + direction-finding ablations
# lane 1 "background" -> fine-tuning around known-good configs; keeps the
#                        machine busy when nothing else is queued
# A job with no "lane" field is lane 2 (backward compatible). Within a lane,
# file order + depends_on/capability semantics are unchanged. A pending
# lane-3 job preempts a RUNNING lane-1 job on this machine; lane-2 jobs are
# never preempted. Guard: each lane-1 job is preempted at most once per
# PREEMPT_COOLDOWN_S (tracked via preempt_count / last_preempt_at).
LANES = (3, 2, 1)
PREEMPT_COOLDOWN_S = 30 * 60


def job_lane(job: dict) -> int:
    try:
        lane = int(job.get("lane", 2))
    except (TypeError, ValueError):
        return 2
    return lane if lane in (1, 2, 3) else 2


class Queue:
    def __init__(self, path: str):
        self.path = path

    # -- low-level IO (call only while holding the lock) ----------------
    @staticmethod
    def _load(f) -> list[dict]:
        f.seek(0)
        return [json.loads(l) for l in f if l.strip()]

    @staticmethod
    def _save(f, jobs: list[dict]) -> None:
        f.seek(0)
        f.truncate()
        for j in jobs:
            f.write(json.dumps(j) + "\n")
        f.flush()

    def _open_locked(self, timeout_s: float | None = None):
        """Open the queue file and acquire LOCK_EX with a deadline.

        The queue lives on NFS; a lost lock owner or a peer hung while
        holding the lock must not freeze this runner permanently. Raises
        TimeoutError instead of blocking forever.
        """
        if timeout_s is None:
            timeout_s = float(os.environ.get("ABLATOR_FLOCK_TIMEOUT_S",
                                             DEFAULT_FLOCK_TIMEOUT_S))
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        open(self.path, "a").close()
        f = open(self.path, "r+")
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return f
            except (BlockingIOError, PermissionError):
                if time.monotonic() >= deadline:
                    f.close()
                    raise TimeoutError(
                        f"queue flock not acquired within {timeout_s:.0f}s")
                time.sleep(0.5)

    # -- public API ------------------------------------------------------
    def read(self) -> list[dict]:
        """Lock-free snapshot for status displays."""
        if not os.path.exists(self.path):
            return []
        with open(self.path) as f:
            return [json.loads(l) for l in f if l.strip()]

    def append(self, new_jobs: list[dict]) -> None:
        """Atomically enqueue jobs; refuses duplicate ids."""
        with self._open_locked() as f:
            jobs = self._load(f)
            existing = {j.get("id") for j in jobs}
            dupes = [j["id"] for j in new_jobs if j["id"] in existing]
            if dupes:
                raise SystemExit(f"refusing to enqueue: duplicate job ids {dupes}")
            jobs.extend(new_jobs)
            self._save(f, jobs)

    def claim_next(self, machine: str,
                   can_run: Callable[[dict], bool] | None = None) -> dict | None:
        """Claim the first runnable pending job for this machine.

        can_run(job) is an optional capability predicate (e.g. required
        container images present); evaluated at most once per job type
        per scan by the caller if desired — here it is called per job.

        A machine-level pause flag (see pause_flag_path) blocks new claims
        without disturbing already-running jobs. A queue-flock timeout
        (contended/hung NFS lock) returns None rather than blocking
        forever — the caller retries on its next poll.
        """
        if is_paused(self.path, machine):
            return None
        try:
            f = self._open_locked()
        except TimeoutError:
            return None
        with f:
            jobs = self._load(f)
            by_id = {j.get("id"): j for j in jobs}
            for lane in LANES:  # urgent (3) first, then standard (2), background (1)
                for j in jobs:
                    if job_lane(j) != lane:
                        continue
                    if j.get("status") != "pending":
                        continue
                    if j.get("machine", "any") not in (machine, "any"):
                        continue
                    if not not_before_ok(j):
                        continue  # backoff window (e.g. gpu_busy/network_transient requeue)
                    dep = j.get("depends_on")
                    if dep and by_id.get(dep, {}).get("status") != "done":
                        continue
                    if can_run is not None and not can_run(j):
                        continue
                    j["status"] = "running"
                    j["claimed_by"] = machine
                    j["claimed_at"] = _now()
                    self._save(f, jobs)
                    return j
        return None

    def urgent_pending(self, machine: str) -> bool:
        """True if a lane-3 job is pending and claimable by this machine."""
        jobs = self.read()
        by_id = {j.get("id"): j for j in jobs}
        for j in jobs:
            if j.get("status") != "pending" or job_lane(j) != 3:
                continue
            if j.get("machine", "any") not in (machine, "any"):
                continue
            dep = j.get("depends_on")
            if dep and by_id.get(dep, {}).get("status") != "done":
                continue
            return True
        return False

    def preemption_due(self, job: dict, machine: str,
                       now: float | None = None) -> bool:
        """Should the currently running `job` yield to a pending lane-3 job?

        Only lane-1 (background) jobs are preemptable, at most once per
        PREEMPT_COOLDOWN_S per job (anti-thrash guard via last_preempt_at).
        """
        if job_lane(job) != 1:
            return False  # lane-2/3 jobs are never preempted
        now = time.time() if now is None else now
        last = job.get("last_preempt_at")
        try:
            if last is not None and now - float(last) < PREEMPT_COOLDOWN_S:
                return False
        except (TypeError, ValueError):
            pass
        return self.urgent_pending(machine)

    def finish(self, job_id: str, status: str, **extra) -> None:
        for attempt in range(5):
            try:
                with self._open_locked() as f:
                    jobs = self._load(f)
                    for j in jobs:
                        if j.get("id") == job_id:
                            j["status"] = status
                            j["finished_at"] = _now()
                            j.update(extra)
                    self._save(f, jobs)
                return
            except TimeoutError as e:
                print(f"[ablator] finish({job_id}) attempt {attempt + 1}: {e}",
                      flush=True)
        print(f"[ablator] ERROR: could not record {job_id} -> {status}; "
              f"job stays 'running' in the queue file", flush=True)

    def update(self, job_id: str, **fields) -> None:
        try:
            with self._open_locked() as f:
                jobs = self._load(f)
                for j in jobs:
                    if j.get("id") == job_id:
                        j.update(fields)
                self._save(f, jobs)
        except TimeoutError as e:
            print(f"[ablator] update({job_id}) skipped: {e}", flush=True)

    def cancel(self, predicate: Callable[[dict], bool]) -> int:
        """Mark matching pending jobs cancelled; returns count."""
        n = 0
        with self._open_locked() as f:
            jobs = self._load(f)
            for j in jobs:
                if j.get("status") == "pending" and predicate(j):
                    j["status"] = "cancelled"
                    n += 1
            self._save(f, jobs)
        return n
