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
{"state": "starting"|"training"|"reporting"|"finishing"|"done"|"hung"|"crashed",
 "iter": 1234, "total": 30000, "log_age_s": 12.4}
```

| State | Meaning |
|---|---|
| `starting` | No log written yet (or nothing parseable). |
| `training` | Log is progressing; an iteration counter was parsed. |
| `reporting` | Iteration counter reached its total — post-training report phase. |
| `finishing` | A `result_glob`/`complete_marker`/researchflow completion artifact matched, but the caller reported (via `container_alive=True`) that the job's own container is still running — see below. Never `done` while this holds. |
| `done` | The configured `result_glob` matched a file under the job's `model_path`, OR the `complete_marker` file (default `.COMPLETE`) exists there — AND the job's container, if checked, is no longer running. |
| `hung` | Log hasn't been written to in longer than `hung_after_min` (default 20, per-job or `[queue]`-level override). |
| `crashed` | A crash marker (`Traceback...`, `CUDA error`, `HIP error`, `std::exception`, `Segmentation fault`, `core dumped`, or config-overridden via `[queue] crash_markers`) appeared in the log tail, or the caller reported the process/container as no longer alive with no success marker present. |

Relevant `[queue]` config knobs: `progress_log`, `progress_regex`,
`progress_cap_regex` (see `ablator.progress`), `result_glob`,
`complete_marker`, `hung_after_min`, `crash_markers`.

### `result_glob` resolution order

Per-job `job["result_glob"]` (settable directly on a queue job, or via a
spec's `result_glob` at the arm/base/spec level — see
[spec-reference.md](spec-reference.md), arm > base > spec precedence) >
the job type's own `result_glob` (`[types.<type>] result_glob`, layered
into the merged qcfg by `runner._health_qcfg`) > `[queue] result_glob` >
the module default (`comparison/*/report.json`).

This matters for multi-phase trainers whose FINAL artifact differs from
what the rest of that job type normally produces. Splatograph's
`causal_mapping` trainer run with `--streaming_post_mapping_refinement_steps
N > 0` writes an interim `comparison/mapping_endpoint/report.json` minutes
before its final `comparison/iter_<N>/report.json` — a job (or spec arm)
that needs to be graded on the final refined result, not the interim
mapping-only one, sets `"result_glob": "comparison/iter_*/report.json"`
on itself without having to change the type's or queue's default for
every other job of that type.

### `container_alive` and the `finishing` state

`job_health(..., container_alive=...)` accepts a third, independent
liveness signal alongside `process_alive`: whether the job's own
container (`splat_train_<job_id>`, see `runner.expected_container_name`)
is currently up, typically from a `docker/podman ps --filter name=...`
call the caller makes (`runner.container_running`).

This exists because a `result_glob`/`complete_marker` match is not
sufficient completion evidence on its own for a multi-phase trainer: the
interim artifact above is real and matches `result_glob` the instant it's
written, well before the container that will go on to write the FINAL
artifact has exited. Before this existed, the daemon's stale-running
reconciler (`runner.reconcile_stale_running`, which has no live
subprocess handle to consult after a runner restart) read that interim
match as `done` and requeued/re-dispatched the job while its container
was still training — confirmed live 2026-09 on psnr26 `causal_mapping`
jobs. `container_alive=True` downgrades an otherwise-`done` verdict to
`finishing` (or a more specific state parsed from the still-live log,
e.g. `training`/`reporting`/`hung`) and is never overridden back to
`crashed` by a caller-supplied `process_alive=False` (which, from a
restarted daemon with no subprocess handle, means "unknown", not "dead").
`container_alive=False` (the container has genuinely exited) or `None`
(unknown — e.g. no container runtime available to check) behave exactly
as before: `done` stands once the artifact/marker evidence says so.

`result_glob` accepts either a bare pattern relative to the resolved
`model_path` (e.g. `comparison/*/report.json`) or the
`{model_path}/...`-prefixed form documented for `ablator collect` — both
resolve to the same thing here.

`complete_marker` (default `.COMPLETE`) is a second, independent way to
reach `done`, checked with plain `os.path.exists` (not a glob) against the
resolved `model_path`. It exists because `result_glob` alone assumes every
job type eventually produces a artifact matching that pattern (e.g.
`comparison/*/report.json`) — true for splatograph's `train.py`/
`train_streaming.py` when a full report is generated, but NOT true for
every trainer/configuration: e.g. its `causal_mapping` trainer with no
evaluation holdout configured never writes a `report.json` at all, even on
a fully successful run (confirmed live 2026-08-29 on two genuinely-complete
smoke-check jobs that were false-classified as `crashed` with
`error_category: unknown` for exactly this reason). Splatograph's shared
output-staging finalizer writes `.COMPLETE` at the resolved `model_path`
for ANY trainer the instant a staged run's local scratch has been fully
mirrored to its canonical path — independent of what richer artifacts that
trainer does or doesn't also produce — making it a safe, trainer-agnostic
completion signal to OR against `result_glob` rather than replace it with.
Set `complete_marker = ""` under `[queue]` (or a per-type override) to
disable this check entirely and fall back to `result_glob`-only behavior.

`complete_marker` also accepts a list of names (ANY existing counts as
`done`), e.g. `complete_marker = [".COMPLETE", "causal_replay_summary.json"]`.
This is needed because `.COMPLETE` itself is not equally reliable across
every trainer: confirmed live 2026-08-29 that splatograph's `causal_mapping`
trainer does not go through the same shared output-staging finalization
path as `train.py`/`train_streaming.py` and does not reliably rewrite
`.COMPLETE` on every attempt (one job's `.COMPLETE` was ~35 minutes stale
across two back-to-back genuinely-successful reruns), while its own
`causal_replay_summary.json` was rewritten fresh on both. **Caveat**:
`causal_replay_summary.json` is written by that trainer BEFORE its
post-mapping refinement phase runs, so a job configured with a non-zero
refinement budget that gets killed during refinement would read as `done`
under this marker — a real, narrower gap, not a full fix. The correct
long-term fix is for `causal_entry.py` to route through the same shared
completion tail (`splatograph.runtime.finalize`) the other two trainers use;
until then, only add `causal_replay_summary.json` to `complete_marker` for
job types/specs known not to configure real post-mapping refinement.

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
