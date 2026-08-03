# Health & error classification

Two independent, passive mechanisms observe jobs without ever injecting
anything into a run: **health** (is a still-running job actually making
progress?) and **failure classification** (why did a finished job
fail?). A job started by hand outside `ablator` produces the same
artifacts and is diagnosable identically — neither mechanism depends on
the runner being involved.

## `ablator health [job_id]`

Derived purely from a running job's own artifacts (progress log,
result-file glob, and caller-supplied process liveness):

```json
{"state": "starting"|"training"|"reporting"|"done"|"hung"|"crashed",
 "iter": 1234, "total": 30000, "log_age_s": 12.4}
```

| State | Meaning |
|---|---|
| `starting` | No log written yet (or nothing parseable). |
| `training` | Log is progressing; an iteration counter was parsed. |
| `reporting` | Iteration counter reached its total — post-training report phase. |
| `done` | The configured `result_glob` matched a file under the job's `model_path`. |
| `hung` | Log hasn't been written to in longer than `hung_after_min` (default 20, per-job or `[queue]`-level override). |
| `crashed` | A crash marker (`Traceback...`, `CUDA error`, `HIP error`, `std::exception`, `Segmentation fault`, `core dumped`, or config-overridden via `[queue] crash_markers`) appeared in the log tail, or the caller reported the process/container as no longer alive with no success marker present. |

Relevant `[queue]` config knobs: `progress_log`, `progress_regex`,
`progress_cap_regex` (see `ablator.progress`), `result_glob`,
`hung_after_min`, `crash_markers`.

`result_glob` accepts either a bare pattern relative to the resolved
`model_path` (e.g. `comparison/*/report.json`) or the
`{model_path}/...`-prefixed form documented for `ablator collect` — both
resolve to the same thing here.

## `ablator errors [name]`

Classifies **terminal** (failed/quarantined) jobs and machine-level
pause flags from the job's log tail, exit code, and machine context.
Pure diagnosis — the runner (not this module) decides what to actually
do (requeue with backoff / quarantine / pause the machine).

| Category | Suggested action |
|---|---|
| `disk_full` | `pause_queue_alert` |
| `image_missing` | `skip_permanently_this_machine` |
| `gpu_busy_conflict` | `requeue_backoff_5min` |
| `gpu_memory_exhaustion` | `quarantine_no_retry` (runner's own in-flight GPU-memory guard killed it — bypasses log-tail heuristics entirely, since the guard already knows why) |
| `oom_killed` | `requeue_once_needs_review` |
| `scene_missing` | `quarantine_no_retry` |
| `network_transient` | `requeue_backoff_2min` |
| `code_error` | `quarantine_code_fix_needed` |
| `unknown` | `retry_once_then_quarantine` |

Classification checks categories roughly in this priority order (first
match wins): `disk_full` → `image_missing` → `gpu_busy_conflict` (only
if the job was flagged busy-at-claim-time) → `oom_killed` (exit 137
without a GPU-OOM log signature) → `scene_missing` → `network_transient`
→ `gpu_busy_conflict` (fallback, lower confidence) → `code_error`
(Python traceback) → `unknown`.

Marker lists are config-driven — override any category (wholesale, not
merged) via `[error_patterns]` in the host config:

```toml
[error_patterns]
image_missing = ["pull access denied", "manifest unknown", "custom marker"]
```

Categories not mentioned keep their built-in defaults.
