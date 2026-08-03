# ablator

Cross-machine ablation/experiment queue orchestrator. One shared,
`flock`'d JSONL queue file on any shared filesystem (NFS works), one
host runner process per machine, and **containerized workloads defined
entirely by command templates in a host config file** — the `ablator`
package itself knows nothing about any particular training system.

## Design invariants

- **Stdlib-only runner.** The host runner (everything under
  `ablator.runner`, `ablator.queue`, `ablator.cli`) uses only `json`,
  `fcntl`, `subprocess`, `socket`, `argparse` and friends — no
  third-party dependency. It must work on any host `python3` with zero
  installs, because it runs *outside* the containers it launches.
  Python >= 3.10 is required; TOML configs need 3.11's `tomllib` (use a
  `.json` config on 3.10 if you're stuck there).
- **Host runner, containerized workloads.** Nothing about a job executes
  outside a container except the `flock`'d queue-file bookkeeping and
  the `podman run` / `docker run` / `kubectl apply` launch itself.
  Running the orchestrator *inside* a container would need nested
  container-socket forwarding (fragile with rootless podman + SELinux)
  and buys nothing, since no GPU work happens in the runner process
  either way.
- **Config-driven, workload-agnostic.** `ablator` has no idea what
  Gaussian splatting, PyTorch, or any other training system is. Every
  job type is a command template (`[types.<type>]` in the host config)
  with `{scene}`, `{model_path}`, `{extra_args}`, `{iterations}`, `{id}`,
  `{machine}` substitutions. Swapping workloads means swapping the
  config, not the code.

## Where to go next

| If you want to... | Read |
|---|---|
| Get a queue running in five minutes | [Quickstart](quickstart.md) |
| Understand the ablation spec JSON format | [Spec reference](spec-reference.md) |
| Understand the host config TOML/JSON format | [Config reference](config-reference.md) |
| Understand job lifecycle, claiming, retries | [Queue semantics](queue-semantics.md) |
| Run runners across several machines via ssh | [Multi-machine setup](multi-machine.md) |
| Dispatch jobs to a Kubernetes cluster instead of local containers | [Kubernetes dispatch](kubernetes.md) |
| Follow a from-scratch walkthrough against the CPS GPU cluster | [CPS cluster walkthrough](cluster-setup.md) |
| Look up a specific subcommand's flags | [CLI reference](cli-reference.md) |
| Understand how failures get classified and job health is judged | [Health & error classification](health.md) |

## Source

<https://github.com/bjoernellens1/ablator>
