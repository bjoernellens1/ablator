# Kubernetes dispatch

A `[machines.<name>]` entry with `backend = "k8s"` dispatches jobs as
Kubernetes `Job` resources (via `kubectl apply` / `kubectl get` polling)
instead of local `podman run` / `docker run` — same command-template
mechanism, same queue semantics as any other machine, just a different
execution backend. This is entirely config-driven and workload-agnostic
(nothing here assumes Gaussian splatting or any specific training code).

See `examples/pytorch-generic.toml` for a plain, non-splatting PyTorch
config, and the [CPS cluster walkthrough](cluster-setup.md) for a full
from-scratch guide (Rancher access, kubeconfig, kubectl, running a job)
against one real cluster.

## Required fields

```toml
[machines.a100cluster]
backend = "k8s"
namespace = "cps-users"
kai_queue = "your-kai-queue"
priority_class = "kai-batch-low"
image = "ghcr.io/you/your-image:tag"
```

k8s-backend machines don't need `hostname_patterns` — they're not
resolved by hostname matching, they're dispatch targets picked
explicitly (job `machine` field or config wiring), and multiple
k8s-backend machines can coexist with bare-metal ones in the same
config.

## Optional fields

| Field | Default | Meaning |
|---|---|---|
| `scheduler_name` | `"kai-scheduler"` | `schedulerName` on the pod spec — point at any cluster scheduler. |
| `image_pull_secret` | — (omitted) | Set only if `image` needs auth; a public image needs none. |
| `pvc_persistent` / `pvc_scratch` | — (omitted) | Optional PVC claim names. If neither is set, no dataset/scratch volumes are mounted at all. |
| `persistent_mount_root` | `/mnt/cps_persistent1_shared` | Host-side path prefix used to compute the `subPath` into `pvc_persistent`. |
| `scratch_mount_root` | `/mnt/cps_scratch1_tmp` | Same, for `pvc_scratch`. |
| `dataset_mount_path` | `/data/scene` | In-container mount path for the dataset volume. |
| `extra_volumes` | `[]` | List of `{name, claim_name, mount_path, read_only}` for any additional PVC (e.g. shared checkpoints). |
| `cpu_request` / `memory_request` / `cpu_limit` / `memory_limit` | `4` / `16Gi` / `8` / `32Gi` | Pod resource requests/limits. |
| `shm_size_gb` | — (omitted) | Mounts a `/dev/shm` `emptyDir` (`Memory` medium) sized in GiB — needed for training that relies on shared-memory IPC (e.g. PyTorch DataLoader workers) beyond Kubernetes' tiny default `/dev/shm`. |
| `mps` | `false` | Wires the trainer container as an MPS client of the cluster's per-node MPS control daemon — see [MPS](#mps-multi-process-service) below. |
| `active_deadline_s` | `86400` (24h) | Hard `activeDeadlineSeconds` safety net independent of ablator's own polling — bounds worst-case orphaned-Job cost if the coordinator process dies mid-job. |
| `termination_grace_period_s` | `150` | `terminationGracePeriodSeconds`. Anchor this above your workload's own graceful-shutdown/checkpoint-save time if it has one, since a plain SIGKILL after this window can truncate an in-progress checkpoint write. |

## Per-job-type image override

```toml
[types.bag.machines.a100cluster]
image = "ghcr.io/you/ros2-cuda-image:tag"
```

`mcfg["image"]` holds exactly one image per machine; this per-type
override (fed through `image_override` in the manifest builder) is the
one escape hatch for a job type that needs a different image than the
machine's default (e.g. a ROS2+CUDA bag-training image vs. a plain
CUDA-only image used by other types on the same machine).

## Git-sync (opt-in)

Gated entirely on `git_sync_repo_url` being set — absent by default, so
this is a no-op for every machine that hasn't configured it.

```toml
[machines.a100cluster]
git_sync_repo_url = "https://github.com/you/repo.git"
# one of:
git_sync_secret_name = "repo-deploy-key"        # SSH deploy key, mounted into the init container only
git_sync_http_secret_name = "repo-pat"          # HTTPS PAT secret with a single "token" key
git_sync_image = "alpine/git:2.45.2"            # default shown
```

Instead of trusting whatever source got baked into the image at build
time (which silently drifts from the dispatching host's actual
checkout), an `alpine/git`-based init container clones the repo at the
**exact commit SHA** the dispatching host was at when the job was
submitted — not just a branch head, which could itself move before the
cluster schedules the pod — into a shared `emptyDir`, and the trainer
container mounts that same `emptyDir` *over* the baked source path,
overlaying fresh source on top of (not beside) the image's own copy.

For a pinned job, the init container additionally initializes recursive
submodules, verifies the exact requested commit, detached HEAD, and a fully
clean source tree, and writes that proof to its termination message. The
trainer mounts the source `emptyDir` read-only and receives
`PYTHONDONTWRITEBYTECODE=1`. The runner captures the proof plus the actual pod,
node, image, and image ID before deleting the Job; a missing or inconsistent
proof changes a nominal Kubernetes success to failure.

Set `require_pinned_git = true` on scientific job types. With that setting,
the runner rejects a job without a full registered SHA and rejects a pinned job
when `git_sync_repo_url` is absent. The historical unpinned git-sync behavior
remains available only to non-strict legacy/debug types.

`git_sync_http_secret_name` takes precedence if both secret fields are
somehow set; it rewrites the remote URL to embed the token as an
`x-access-token` credential rather than relying on `GIT_SSH_COMMAND`
(which needs an `ssh://`/`git@` URL). A PAT already provisioned for
`image_pull_secret`/registry push can often be reused here instead of a
separate deploy key.

!!! warning "Known limitation"
    Overlaying source without rebuilding the image decouples "source
    freshness" from "environment freshness" by design (rebuilding heavy
    compiled deps like `gsplat`/`fused-ssim`/CUDA toolchains per job
    would be far too slow) — a freshly-pulled commit that needs a newer
    dependency than what's baked into `image` will fail or misbehave at
    runtime with no compatibility check performed.

## MPS (Multi-Process Service)

```toml
[machines.a100cluster]
mps = true
```

Needed on GPU nodes left in NVIDIA `Exclusive_Process` compute mode with
no per-pod permission to change it, where a plain `nvidia.com/gpu: 1`
request can otherwise fail its first CUDA call with `CUDA-capable
device(s) is/are busy or unavailable` even on a fully idle GPU. When
enabled, the manifest:

1. Mounts a hostPath volume at `/run/nvidia/mps` (host) → `/mps`
   (container) — this makes the pod a **client** of the cluster's own
   already-running per-node MPS control daemon, not a second
   self-hosted daemon inside the pod (running a second daemon was tried
   elsewhere and found to non-deterministically race the real one).
2. Sets `CUDA_MPS_PIPE_DIRECTORY=/mps/nvidia.com/gpu/pipe` and
   `CUDA_MPS_LOG_DIRECTORY=/mps/nvidia.com/gpu/log` on the trainer
   container, so any CUDA-using process becomes an MPS client
   automatically — no application code changes needed.
3. Adds soft (preferred, not required) anti-affinity against other
   `app: ablator-job` pods on the same node, since the MPS server's
   startup grabs every physical GPU on a node at once and can
   crash-loop if one is legitimately busy from an unrelated job.

A job's first CUDA call can still transiently fail even with this
wiring correct, because the control daemon spawns its real server
process lazily on a client's first connection — workloads should retry
the first CUDA call a few times; this is workload-side, not something
the manifest can fix.

## Stalled-pod detection

A pod stuck forever at `Init`/`Pending` (e.g. a stuck image pull, or a
node that can never satisfy the pod's scheduling constraints) is
detected independently of ablator's own control-file polling loop: if a
k8s job's log file hasn't grown in longer than the configured stall
threshold, the runner treats it as stalled, kills the underlying k8s
`Job`, and lets normal failure/retry handling take over — rather than
leaving it running forever with zero runner-side detection (this was
found live: a pod sat at `Init:0/1` for nearly 5 hours with nothing
noticing).

## Concurrency

k8s dispatch has its own per-machine concurrency cap, separate from
bare-metal "one job at a time" behavior — going over it just means some
submitted Jobs sit `Pending` in `kubectl` until a GPU frees up, not that
`ablator` blocks.
