# Config reference

The host config (TOML or JSON) defines everything machine/workload
specific: queue path, machine identities, busy guards, resource
thresholds, and job-type command templates. `ablator` itself has zero
built-in knowledge of any particular workload — swap the config, not
the code.

Resolution order: `--config PATH` flag > `$ABLATOR_CONFIG` env var >
`~/.config/ablator/config.toml`.

TOML parsing uses stdlib `tomllib`, which requires Python >= 3.11. On
3.10, use a `.json` config instead (same schema, JSON syntax).

## `[queue]`

```toml
[queue]
path = "/mnt/shared/queue/queue.jsonl"   # required; shared filesystem
log_dir = "/mnt/shared/queue"            # default: dirname(path)
model_path_template = "output/scratch/{name}_{arm}"
result_glob = "{model_path}/comparison/*/report.json"  # for `collect`
```

`$ABLATOR_QUEUE_FILE` overrides `path` at runtime without touching the
config file.

## `[machines.<name>]`

```toml
[machines.<name>]
hostname_patterns = ["*r9700*"]   # glob vs lowercased hostname; ["*"]/absent = fallback
ssh = "user@host"                 # remotes started by `ablator start`
runner_command = "ablator run"    # remote runner invocation override
backend = "k8s"                   # optional; see kubernetes.md — omit for local podman/docker
```

Machine identity resolves per-host: each runner process figures out
which `[machines.*]` entry it is by matching its own (lowercased)
hostname against `hostname_patterns` (first match wins, dict order). An
entry with no patterns, or `["*"]`, is the fallback used when nothing
else matches. If truly nothing matches, the machine resolves to
`"unknown"`.

### `[[machines.<name>.busy_guards]]`

Generic "is something else already using this box?" check, run before
claiming a job:

```toml
[[machines.<name>.busy_guards]]
command = ["podman", "ps", "--format", "{{.Names}}"]
contains = "splat_train"          # omit -> busy if any non-empty output
```

## `[resources]`

```toml
[resources]
gpu_busy_pct = 20     # busy iff BOTH samples > threshold (debounced spike filter)
sample_gap_s = 3.0    # env ABLATOR_GPU_BUSY_PCT overrides the threshold
cpu_max_concurrent = 1  # requires_gpu=false jobs run alongside the GPU job, at most this many
```

A runner only claims a job when the machine is idle: GPU utilization
below `gpu_busy_pct` in **both** of two samples ~`sample_gap_s` apart
(debounces momentary spikes) **and** no `busy_guard` fires.

### CPU-only job types (`requires_gpu = false`)

A type declared `requires_gpu = false` (a unit-test suite, a report or
plot step, a CPU sandbox simulation) is claimed by this machine's runner
**even while its GPU job is running** and executed on a background
thread, bounded by `[resources] cpu_max_concurrent` (default 1). The
GPU-util sampler, `busy_guards` and the GPU-memory guard are not
consulted for it; the machine pause flag, the per-type capability probe
(`require_images`) and every `[types.<type>]` field still apply, and the
job goes through the same retry/quarantine/receipt bookkeeping. Without
this, a 10-second CPU job pinned to a machine with a deep GPU queue waits
for the GPU to go idle -- on a busy workstation, hours.

## `[types.<type>]`

One entry per job "type" (referenced from a spec's `arm.type` /
`base.type`):

```toml
[types.<type>]
cwd = "/path/to/repo"
command = ["podman", "run", "...", "{scene}", "-m", "{model_path}",
           "--iterations", "{iterations}", "{extra_args}"]
env = { SCENE_SOURCE = "{scene}" }
result_glob = "..."               # per-type override for `collect` AND the
                                  # done/failed verdict (health.job_health)
complete_marker = ["..."]         # per-type override, same two consumers
requires_gpu = false              # CPU-only type: claimed even while this
                                  # machine's GPU job runs, executed on a
                                  # background thread (default true)
require_pinned_git = true         # reject jobs without a full registered SHA

[types.<type>.machines.<name>]    # shallow per-machine override
command = ["docker", "run", "..."]        # replaces base command
env = { CONTAINER_RUNTIME = "docker" }    # MERGES into base env
require_images = ["img:a", "img:b"]       # claim only if all present locally
image_probe_runtime = "docker"            # never pulls/builds
```

Per-machine overrides are **shallow**: any key present under
`[types.<type>.machines.<name>]` other than `env` replaces the base
value outright; `env` is merged on top of the base `env` dict instead
of replacing it.

`ABLATOR_EXPERIMENT_DECLARATION_JSON`,
`ABLATOR_EXPERIMENT_DECLARATION_SHA256`, `ABLATOR_JOB_ID`,
`ABLATOR_JOB_JSON`, and `ABLATOR_SUBMISSION_JSON` are reserved.
The runner derives them only from a validated immutable declaration in the
queue; ambient and `[types.*.env]` values with those names are scrubbed. Direct
Docker/Podman and Kubernetes jobs receive them automatically, so command
templates must not add their own copies.

### Template variables

Available in every `command` token and `env` value:
`{scene}` `{model_path}` `{extra_args}` `{iterations}` `{id}` `{machine}`.

Pinned type commands may also use `{repo_cwd}` for host source paths. The
runner replaces it with the unique per-attempt checkout, rewrites `cwd`, and
makes Docker/Podman binds of that checkout read-only. Keep datasets, scratch,
and result directories on separate mounts.

## `[git]`

```toml
[git]
worktree_root = "~/.cache/ablator/worktrees"
gc_max_age_days = 30
```

`machines.<name>.git_worktree_root` overrides the root for one runner.
`ABLATOR_GIT_WORKTREE_ROOT` is the fallback process override. See
[Immutable Git worktree cache](worktree-cache.md) for lease and cleanup rules.

A token that is *exactly* `"{extra_args}"` expands to zero or more
whitespace-split argv items (so an empty `extra_args` cleanly disappears
from argv instead of becoming an empty string arg).

`examples/splatograph.toml` reproduces a real two-machine setup (podman
+ a gfx1151 image on `main`, docker + a gfx1201 image on `r9700`, `bag`
jobs via a launch script fed through `TRAIN_EXTRA_ARGS`/`SCENE_SOURCE`).
`examples/pytorch-generic.toml` is a plain, non-splatting PyTorch
example, including a k8s-backend machine.

## `[error_patterns]`

Optional overrides for failure-classification marker lists (see
[health & error classification](health.md)):

```toml
[error_patterns]
image_missing = ["pull access denied", "manifest unknown", "custom marker"]
```

A category present here **replaces** its built-in marker list wholesale;
categories not mentioned keep their defaults.

## Kubernetes-specific machine fields

See [Kubernetes dispatch](kubernetes.md) for the full set of
`[machines.<name>]` fields available when `backend = "k8s"` (namespace,
PVCs, image, scheduler, git-sync, MPS, `shm_size_gb`, and more).
