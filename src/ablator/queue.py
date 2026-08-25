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

from . import experiment_declaration as declarations
from . import source_checkout as sourcecheckout

DEFAULT_FLOCK_TIMEOUT_S = 60.0


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def pause_flag_path(queue_path: str, machine: str) -> str:
    return os.path.join(os.path.dirname(queue_path), f"paused_{machine}.txt")


def is_paused(queue_path: str, machine: str) -> bool:
    return os.path.exists(pause_flag_path(queue_path, machine))


def read_pause_flag(queue_path: str, machine: str) -> dict | None:
    """Parses an existing pause flag into {'category', 'timestamp',
    'evidence', ...}. None if there is no flag (or it is unreadable) --
    callers must treat None the same as "not paused", never as "paused,
    category unknown"."""
    path = pause_flag_path(queue_path, machine)
    info: dict[str, str] = {}
    try:
        with open(path) as f:
            for line in f:
                if "=" in line:
                    k, _, v = line.strip().partition("=")
                    info[k] = v
    except OSError:
        return None
    return info or None


def _pause_audit_path(queue_path: str) -> str:
    return os.path.join(os.path.dirname(queue_path), "pause_audit.log")


def _append_pause_audit(queue_path: str, line: str) -> None:
    """Best-effort durable audit trail of pause set/clear events, separate
    from the ephemeral flag file itself (which is deleted on clear) and
    from stdout (which is not always captured/searchable). Never raises --
    an audit-logging failure must not affect pause/clear/dispatch."""
    path = _pause_audit_path(queue_path)
    try:
        with open(path, "a") as f:
            f.write(f"{_now()} {line}\n")
    except OSError as e:
        print(f"[ablator] could not append pause audit {path}: {e}", flush=True)


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
    _append_pause_audit(
        queue_path,
        f"SET machine={machine} category={category} evidence={evidence!r}")
    return path


def clear_pause_flag(queue_path: str, machine: str, reason: str | None = None) -> bool:
    """Removes the pause flag for `machine`. `reason` (e.g.
    "manual:ablator unpause" or "auto_revalidate:urgent_fix_unsynced: ...")
    is recorded in the durable audit log alongside the category/evidence
    the flag carried, so a cleared pause stays auditable after the flag
    file itself is gone -- never a silent clear."""
    path = pause_flag_path(queue_path, machine)
    info = read_pause_flag(queue_path, machine)
    try:
        os.remove(path)
    except OSError:
        return False
    was_category = (info or {}).get("category", "?")
    was_evidence = (info or {}).get("evidence", "?")
    _append_pause_audit(
        queue_path,
        f"CLEAR machine={machine} was_category={was_category} "
        f"was_evidence={was_evidence!r} reason={reason or 'unspecified'}")
    return True


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
        """Write the queue's full contents.

        NOT atomic against the caller's own truncate-then-write of `f`
        (that's still needed since `f` is the flock'd fd itself), but
        callers must go through Queue.update()/append()/etc., which build
        the full serialized payload in memory FIRST via _serialize()
        before ever touching `f` -- so a mid-write failure (e.g. disk
        full) can still leave `f` truncated with a partial write, but
        never with a payload that was silently empty because upstream
        blew up before content existed. See _atomic_write_str() for the
        real fix: on ANY IOError/OSError during the write loop, restore
        the pre-truncate content instead of leaving the file empty --
        found live (2026-08-11) when a disk-full OSError during a normal
        update() call left queue.jsonl at 0 bytes, silently discarding
        the entire multi-day job history because truncate() had already
        run before the write raised.
        """
        payload = "".join(json.dumps(j) + "\n" for j in jobs)
        f.seek(0)
        original = f.read()
        f.seek(0)
        try:
            f.truncate()
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        except OSError:
            # Best-effort restore: put back exactly what was there before,
            # so a disk-full (or any other write failure) degrades to "the
            # update was lost" rather than "the entire queue was lost".
            try:
                f.seek(0)
                f.truncate()
                f.write(original)
                f.flush()
            except OSError:
                pass  # truly out of space even for the restore; nothing more we can do here
            raise

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

    @staticmethod
    def _validate_enqueue_job(job: dict, *, require_pinned_git: bool = False) -> None:
        try:
            declarations.validate_frozen_job(job)
            declarations.validate_external_submission(job)
            sourcecheckout.job_git_target(
                job,
                required=(
                    require_pinned_git
                    or job.get("gradeability") == "GRADEABLE_DECLARED"
                ),
            )
        except (
            declarations.ExperimentDeclarationError,
            sourcecheckout.SourcePreparationError,
        ) as exc:
            raise SystemExit(
                f"refusing to enqueue job {job.get('id')!r}: {exc}"
            ) from exc

    @staticmethod
    def _validate_dependency_edge(
        job: dict, by_id: dict[object, dict], *, require_dependency: bool = False,
    ) -> None:
        dependency_id = job.get("depends_on")
        if not dependency_id:
            return
        dependency = by_id.get(dependency_id)
        if dependency is None:
            if require_dependency:
                raise SystemExit(
                    f"refusing to enqueue job {job.get('id')!r}: "
                    f"dependency {dependency_id!r} is not in the queue"
                )
            return
        try:
            dependency_target = sourcecheckout.job_git_target(dependency)
            job_target = sourcecheckout.job_git_target(job)
        except sourcecheckout.SourcePreparationError as exc:
            raise SystemExit(
                f"refusing to enqueue job {job.get('id')!r}: {exc}"
            ) from exc
        if dependency_target != job_target:
            raise SystemExit(
                "refusing to enqueue: dependency chain changes Git target "
                f"between {dependency_id!r} and {job.get('id')!r}: "
                f"{dependency_target!r} -> {job_target!r}"
            )

    def append(self, new_jobs: list[dict]) -> None:
        """Atomically enqueue jobs; refuses duplicate ids."""
        # _open_locked() creates the queue file. Reject malformed input before
        # that observable side effect, then repeat validation under the lock so
        # the transaction remains authoritative if caller-owned data changes.
        for job in new_jobs:
            self._validate_enqueue_job(job)
        with self._open_locked() as f:
            jobs = self._load(f)
            for job in new_jobs:
                self._validate_enqueue_job(job)
            existing = {j.get("id") for j in jobs}
            dupes = [j["id"] for j in new_jobs if j["id"] in existing]
            if dupes:
                raise SystemExit(f"refusing to enqueue: duplicate job ids {dupes}")
            combined = [*jobs, *new_jobs]
            by_id = {job.get("id"): job for job in combined}
            for job in new_jobs:
                self._validate_dependency_edge(job, by_id)
            jobs.extend(new_jobs)
            self._save(f, jobs)

    def enqueue_idempotent(
        self,
        job: dict,
        *,
        fingerprint_field: str,
        require_pinned_git: bool = False,
    ) -> tuple[dict, bool]:
        """Validate and idempotently enqueue one job in one locked transaction.

        External producers use this public API instead of reaching into queue
        lock/file internals. Validation, duplicate comparison, dependency-pin
        checks, and persistence all observe the same locked queue snapshot.
        """
        self._validate_enqueue_job(job, require_pinned_git=require_pinned_git)
        with self._open_locked() as handle:
            jobs = self._load(handle)
            self._validate_enqueue_job(
                job, require_pinned_git=require_pinned_git
            )
            existing = next(
                (item for item in jobs if item.get("id") == job.get("id")), None
            )
            if existing is not None:
                self._validate_enqueue_job(
                    existing, require_pinned_git=require_pinned_git
                )
                if existing.get(fingerprint_field) == job.get(fingerprint_field):
                    return existing, False
                raise SystemExit(
                    f"job id {job.get('id')!r} already exists with a different specification"
                )
            by_id = {item.get("id"): item for item in jobs}
            by_id[job.get("id")] = job
            self._validate_dependency_edge(job, by_id, require_dependency=True)
            jobs.append(job)
            self._save(handle, jobs)
            return job, True

    def claim_next(self, machine: str,
                   can_run: Callable[[dict], bool] | None = None,
                   only_pinned: bool = False,
                   allow_pinned_git_while_paused: bool = False) -> dict | None:
        """Claim the first runnable pending job for this machine.

        can_run(job) is an optional capability predicate (e.g. required
        container images present); evaluated at most once per job type
        per scan by the caller if desired — here it is called per job.

        only_pinned=True restricts claiming to jobs explicitly pinned to
        `machine` (job["machine"] == machine), skipping jobs with
        machine="any" entirely. Used by run_loop's k8s-fill path to defer
        "any" jobs to an idle bare-metal machine for a tick (see
        _other_idle_baremetal) without touching pinned-job claiming at
        all.

        A machine-level pause flag (see pause_flag_path) normally blocks new
        claims without disturbing already-running jobs. When
        ``allow_pinned_git_while_paused`` is true, the one auto-generated
        ``urgent_fix_unsynced`` category becomes a pinned-Git-only filter:
        legacy mutable jobs remain blocked, while immutable jobs may be
        claimed and validate their requested revision independently before
        launch. Manual and unknown pause categories remain absolute.

        A queue-flock timeout
        (contended/hung NFS lock) returns None rather than blocking
        forever — the caller retries on its next poll.
        """
        paused_pinned_only = False
        if is_paused(self.path, machine):
            pause_info = read_pause_flag(self.path, machine) or {}
            if (allow_pinned_git_while_paused
                    and pause_info.get("category") == "urgent_fix_unsynced"):
                paused_pinned_only = True
            else:
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
                    if paused_pinned_only and not j.get("requested_git_sha"):
                        continue
                    jm = j.get("machine", "any")
                    if only_pinned:
                        if jm != machine:
                            continue
                    elif jm not in (machine, "any"):
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
                            try:
                                declarations.validate_immutable_update(j, extra)
                            except declarations.ExperimentDeclarationError as exc:
                                raise SystemExit(str(exc)) from exc
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
        # A resumable checkpoint records progress under the job's OLD
        # scene/extra_args -- if the caller is changing what the job
        # actually runs (a config/path fix, not just a status/claim
        # bookkeeping update) without also explicitly saying what to do
        # with the resume pointer, the safe default is to drop it rather
        # than silently resume a differently-configured run from a
        # checkpoint that may no longer even match (wrong scene, wrong
        # flags). Found live (2026-08-11): a scene-path-only fix left a
        # stale resume_checkpoint in place, so the "fixed" job silently
        # resumed from an old checkpoint's mid-training state instead of
        # training cleanly from scratch under the corrected config --
        # produced a real, materially wrong PSNR that looked plausible.
        if ("scene" in fields or "extra_args" in fields) and "resume_checkpoint" not in fields:
            fields = {**fields, "resume_checkpoint": None, "last_resumed_iter": None}
        try:
            with self._open_locked() as f:
                jobs = self._load(f)
                for j in jobs:
                    if j.get("id") == job_id:
                        try:
                            declarations.validate_immutable_update(j, fields)
                        except declarations.ExperimentDeclarationError as exc:
                            raise SystemExit(str(exc)) from exc
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
