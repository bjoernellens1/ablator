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


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


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

    def _open_locked(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        open(self.path, "a").close()
        f = open(self.path, "r+")
        fcntl.flock(f, fcntl.LOCK_EX)
        return f

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
        """
        with self._open_locked() as f:
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
        with self._open_locked() as f:
            jobs = self._load(f)
            for j in jobs:
                if j.get("id") == job_id:
                    j["status"] = status
                    j["finished_at"] = _now()
                    j.update(extra)
            self._save(f, jobs)

    def update(self, job_id: str, **fields) -> None:
        with self._open_locked() as f:
            jobs = self._load(f)
            for j in jobs:
                if j.get("id") == job_id:
                    j.update(fields)
            self._save(f, jobs)

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
