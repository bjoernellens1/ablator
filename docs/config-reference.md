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
```

A runner only claims a job when the machine is idle: GPU utilization
below `gpu_busy_pct` in **both** of two samples ~`sample_gap_s` apart
(debounces momentary spikes) **and** no `busy_guard` fires.

## `[types.<type>]`

One entry per job "type" (referenced from a spec's `arm.type` /
`base.type`):

```toml
[types.<type>]
cwd = "/path/to/repo"
command = ["podman", "run", "...", "{scene}", "-m", "{model_path}",
           "--iterations", "{iterations}", "{extra_args}"]
env = { SCENE_SOURCE = "{scene}" }
result_glob = "..."               # per-type override for `collect`

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

### Template variables

Available in every `command` token and `env` value:
`{scene}` `{model_path}` `{extra_args}` `{iterations}` `{id}` `{machine}`.

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
