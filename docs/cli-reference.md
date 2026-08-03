# CLI reference

```
ablator [--config CONFIG] <subcommand> ...
```

`--config` overrides the config file for this invocation (see
[config reference](config-reference.md) for resolution order).

## Planning & inspecting

| Command | Args | Effect |
|---|---|---|
| `ablator plan` | `spec [--dry-run]` | Expand a spec JSON into queue jobs. `--dry-run` prints the expansion without writing. Refuses duplicate job ids already queued. |
| `ablator status` | `[name]` | Print queue state table, optionally filtered to one ablation. |
| `ablator watch` | `[name] [--interval N]` | Loop `status`; also mirrors output to `queue_status.txt`. |
| `ablator collect` | `name` | Print result files (`result_glob`) of that ablation's done jobs. |
| `ablator health` | `[job_id]` | Artifact-derived job health — see [health & error classification](health.md). |
| `ablator errors` | `[name]` | List failed/quarantined/paused jobs with their failure classification. |

## Mutating the queue

| Command | Args | Effect |
|---|---|---|
| `ablator cancel` | `name` | Cancel that ablation's still-pending jobs. |
| `ablator promote` | `job_id lane` | Move a pending job to another lane (1/2/3). |
| `ablator rerun` | `job_id [lane]` | Reset a terminal job back to pending, optionally into a new lane. |
| `ablator stop` | `job_id` | Kill a running job → `failed`, no retry. |
| `ablator skip` | `job_id` | Kill a running job → `cancelled`. |
| `ablator requeue` | `job_id` | Kill a running job and re-queue it as `pending`. |
| `ablator pause` | `machine` | Set a machine-level pause flag (blocks new claims; running jobs unaffected). |
| `ablator unpause` | `machine` | Clear a machine-level pause flag. |

## Running

| Command | Args | Effect |
|---|---|---|
| `ablator run` | `[--once]` | Runner loop on this machine: claim + execute jobs. `--once` claims/runs at most one job then exits. |
| `ablator start` | — | Session-proof launch: local runner + ssh remotes (see [multi-machine setup](multi-machine.md)). |

## TUI

| Command | Args | Effect |
|---|---|---|
| `ablator tui` | — | Launch the k9s-style TUI (guided setup on first run). Needs `pip install ablator[tui]`. Bare `ablator` on a TTY also launches it. |

Run any subcommand with `--help` for its exact flags, e.g.
`ablator promote --help`.
