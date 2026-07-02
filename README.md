# ablator

Cross-machine ablation/experiment queue orchestrator. One shared flock'd
JSONL queue on any shared filesystem (NFS works), one host runner per
machine, and **containerized workloads defined entirely by command
templates in a host config file** — the package itself knows nothing
about any particular training system.

- **Stdlib-only** (`json`, `fcntl`, `subprocess`, `socket`, `argparse`):
  the runner must work on any host `python3` with zero installs, because
  it runs *outside* the containers it launches. Python >= 3.10
  (TOML configs need 3.11's `tomllib`; use a `.json` config on 3.10).
- **Host runner, containerized workloads** (design invariant): nothing
  about a job executes outside a container except the flock'd queue-file
  bookkeeping and the `podman/docker run` launch itself. Running the
  orchestrator inside a container would require nested container-socket
  forwarding (fragile with rootless podman/SELinux) and buys nothing —
  no GPU work happens in the runner process.

## Quickstart

```bash
pip install -e .          # or just: PYTHONPATH=src python3 -m ablator.cli
mkdir -p ~/.config/ablator
cp examples/splatograph.toml ~/.config/ablator/config.toml  # then edit

ablator plan spec.json          # expand arms -> queue jobs (refuses dup ids)
ablator status [name]           # queue state table
ablator run [--once]            # runner loop (claim + execute jobs)
ablator start                   # session-proof launch: local + ssh remotes
ablator collect <name>          # print result files (result_glob) of done jobs
ablator cancel <name>           # cancel that ablation's pending jobs
```

Config file resolution: `--config PATH` > `$ABLATOR_CONFIG` >
`~/.config/ablator/config.toml`.

## Spec reference

```json
{
  "name": "consol_sweep",
  "parallel": true,
  "base": {"type": "replay", "scene": "/mnt/data/fr3", "iterations": 30000,
           "machine": "any", "base_args": "--opacity_reg 0.001"},
  "arms": [
    {"id": "ctrl",   "extra_args": ""},
    {"id": "consol", "extra_args": "--foo bar",
     "machine": "main", "type": "bag", "iterations": 60000}
  ]
}
```

- Job ids are `<name>_<arm_id>`; `plan` refuses duplicates already queued.
- `parallel: false` chains arms with `depends_on` (each waits for the
  previous arm to finish **successfully**; failed/quarantined deps block).
- Per-arm overrides: `machine`, `type`, `iterations`, `scene`.
- `extra_args` = `base.base_args` + arm `extra_args`.
- `model_path` comes from `[queue] model_path_template`
  (default `output/scratch/{name}_{arm}`; variables `{name}` `{arm}` `{id}`).

## Config reference (TOML or JSON)

```toml
[queue]
path = "/mnt/shared/queue/queue.jsonl"   # required; shared filesystem
log_dir = "/mnt/shared/queue"            # default: dirname(path)
model_path_template = "output/scratch/{name}_{arm}"
result_glob = "{model_path}/comparison/*/report.json"  # for `collect`

[machines.<name>]
hostname_patterns = ["*r9700*"]   # glob vs lowercased hostname; ["*"]/absent = fallback
ssh = "user@host"                 # remotes started by `ablator start`
runner_command = "ablator run"    # remote runner invocation override

[[machines.<name>.busy_guards]]   # generic "is something using this box?"
command = ["podman", "ps", "--format", "{{.Names}}"]
contains = "splat_train"          # omit -> busy if any non-empty output

[resources]
gpu_busy_pct = 20     # busy iff BOTH samples > threshold (debounced spike filter)
sample_gap_s = 3.0    # env ABLATOR_GPU_BUSY_PCT overrides the threshold

[types.<type>]                    # one entry per job "type"
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

Template variables in `command` tokens and `env` values:
`{scene} {model_path} {extra_args} {iterations} {id} {machine}`.
A token that is exactly `"{extra_args}"` expands to zero or more
whitespace-split argv items.

`examples/splatograph.toml` reproduces the splatograph two-machine setup
(podman + gfx1151 image on `main`, docker + gfx1201 on `r9700`, `bag`
jobs via a launch script fed through `TRAIN_EXTRA_ARGS`/`SCENE_SOURCE`).

## Queue semantics

JSONL, one job per line; all mutations under `fcntl.flock` (NFSv4
serializes this across machines). Statuses:
`pending → running → done | failed → (one retry) → quarantined`, plus
`cancelled`. A job is claimable when: status pending, `machine` matches
this runner (or `any`), `depends_on` (if set) is `done`, its type is
defined in the local config, and any `require_images` probe passes.
Runner claims only when the machine is idle: GPU utilization below
threshold in at least one of two samples ~3 s apart AND no busy_guard
fires. Per-job logs land in `<log_dir>/<job id>.log`.

## Multi-machine setup

1. Same config file on every machine (identity resolves per-host from
   `hostname_patterns`); queue path on a filesystem all machines mount.
2. Install ablator on each machine (or set `runner_command` to a venv
   path / `PYTHONPATH=... python3 -m ablator.cli run`).
3. Passwordless ssh from the machine where you run `ablator start` to
   every `[machines.*] ssh` address.
4. `ablator start` — launches a `setsid nohup` runner locally and on
   each remote, skipping machines where one is already running. Logs:
   `<log_dir>/runner_<machine>.log`.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python3 -m pytest tests/ -q
```
