# Queue semantics

The queue is a single JSONL file (one job dict per line) on a shared
filesystem. All mutations happen under `fcntl.flock` — NFSv4 serializes
this across machines, so multiple runners can safely poll and claim
from the same file concurrently.

## Status lifecycle

```
pending -> running -> done
                    -> failed -> (one retry) -> quarantined
pending -> cancelled   (via `ablator cancel`)
running -> failed      (via `ablator stop`, no retry)
running -> cancelled   (via `ablator skip`)
running -> pending     (via `ablator requeue`, kills + re-queues)
```

## Claiming

A runner claims the **first runnable pending job for its machine**,
scanning lanes in priority order (see below) and, within a lane, file
order. A job is claimable when **all** of:

- `status == "pending"`.
- `job["machine"]` is `"any"` or equals this runner's machine name.
- `not_before` (if set — e.g. a network/GPU-busy backoff window) is in
  the past.
- `depends_on` (if set) points at a job whose status is `"done"`. A
  failed or quarantined dependency permanently blocks the chain (it
  never becomes `"done"`).
- The job's `type` is defined in this machine's local config.
- Any `require_images` probe for the resolved type/machine passes (i.e.
  the required container images are present locally — this never pulls
  or builds, it only checks).

A runner only attempts to claim at all when its machine is **idle**:
GPU utilization below `[resources] gpu_busy_pct` in *both* of two
samples ~`sample_gap_s` apart (debounces a momentary spike), **and** no
configured `busy_guard` command reports the machine busy.

A machine-level pause flag (`ablator pause <machine>`) blocks new claims
without disturbing jobs already running — see
`pause_<machine>.txt` next to the queue file.

A contended/hung queue flock (e.g. a flaky NFS lock) makes `claim_next`
return `None` rather than blocking forever; the caller just retries on
its next poll.

## Priority lanes

```
lane 3  "urgent"      claimed first  — failed-run redos, fast-answer runs
lane 2  "standard"    default        — validation + direction-finding ablations
lane 1  "background"  claimed last   — fine-tuning around known-good configs,
                                        keeps a machine busy when nothing else is queued
```

A job with no `lane` field is treated as lane 2 (backward compatible).
Within a lane, ordering follows file order plus the `depends_on`/
capability rules above.

**Preemption:** a pending lane-3 job can preempt a **running lane-1**
job on the same machine. Lane-2 and lane-3 jobs are never preempted.
Each job is preempted at most once per 30 minutes
(`PREEMPT_COOLDOWN_S`), tracked via `last_preempt_at`, to prevent
thrashing.

## Retries and quarantine

A `failed` job gets exactly one automatic retry (back to `pending`)
before it's marked `quarantined` and stops being reclaimed. Retry/
backoff behavior for specific failure categories (e.g. a GPU-busy
conflict backs off 5 minutes via `not_before`, a transient network
error backs off 2 minutes) comes from
[failure classification](health.md) — see `SUGGESTED_ACTION` there.

## Logs

Per-job stdout/stderr lands at `<log_dir>/<job id>.log`
(`log_dir` defaults to `dirname(queue path)`, overridable via
`[queue] log_dir`).

## CLI surface over the queue

| Command | Effect |
|---|---|
| `ablator status [name]` | Print queue state table. |
| `ablator watch [name] [--interval N]` | Loop `status`; also mirrors to `queue_status.txt`. |
| `ablator errors [name]` | List failed/quarantined/paused jobs with classification. |
| `ablator health [job_id]` | Artifact-derived job health (progress, staleness). |
| `ablator promote job_id lane` | Move a **pending** job to another lane. |
| `ablator rerun job_id [lane]` | Reset a **terminal** (done/failed/quarantined/cancelled) job back to pending, optionally into a new lane. |
| `ablator stop job_id` | Kill a **running** job → `failed`, no retry. |
| `ablator skip job_id` | Kill a **running** job → `cancelled`. |
| `ablator requeue job_id` | Kill a **running** job and put it back to `pending`. |
| `ablator cancel name` | Cancel all still-**pending** jobs of an ablation. |
| `ablator pause machine` / `unpause machine` | Block/unblock new claims on a machine (running jobs unaffected). |
