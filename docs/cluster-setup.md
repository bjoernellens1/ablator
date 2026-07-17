# Using ablator against the CPS GPU cluster

This is a walkthrough for a chair colleague with basic Linux CLI comfort
but no prior Kubernetes/Rancher experience: get cluster access, install
the tools, and run your first arbitrary PyTorch job through `ablator`'s
Kubernetes dispatch path. It does not assume you know anything about
KAI Scheduler, Fleet, or the cluster's internals beyond what's below.

If anything here about the Rancher UI has drifted (buttons move between
Rancher versions), the underlying goal of each step is described too, so
you can find the equivalent control even if a screenshot would look
different today.

## 1. Get Rancher access

The cluster is managed through Rancher at:

```
https://rancher.dshl.unileoben.ac.at
```

Log in with the **"Sign in with CPS Authentik"** button (this cluster's
Rancher authenticates via Authentik OIDC, not a local Rancher account).

Cluster access is granted by **Authentik group membership**, not
per-user grants: you need to be a member of the `CPS` Authentik group.
A `cps-users` Kubernetes namespace with `edit`-role access already exists
for members of that group. **If you log in and can't see the cluster, or
`kubectl` later reports "Forbidden", the most likely cause is that you
aren't in the `CPS` Authentik group yet** — contact the cluster admin to
be added.

## 2. Download your kubeconfig from Rancher

Once logged in and viewing the cluster in Rancher's UI, look for a
**kubeconfig download control** — typically an icon or button near the
cluster name on the cluster's overview/detail page, or in the top-right
user/account menu (exact wording has been "Download KubeConfig" in
recent Rancher versions). Downloading it gives you a `.yaml` file
containing a `context` for this cluster plus a short-lived or
Rancher-proxied credential.

Save it somewhere you'll remember, e.g. `~/Downloads/rancher-cluster.yaml`.

## 3. Install kubectl

Official instructions (most reliable source, since exact package/version
steps change over time): https://kubernetes.io/docs/tasks/tools/

Quick reference:

- **Linux (apt, Debian/Ubuntu)**:
  ```bash
  sudo apt-get update && sudo apt-get install -y kubectl
  ```
  (If that package isn't available on your distro's repos, use the
  official binary download method from the link above.)
- **Linux (dnf, Fedora/RHEL)**:
  ```bash
  sudo dnf install -y kubectl
  ```
- **Linux (official binary, any distro)**:
  ```bash
  curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
  chmod +x kubectl
  sudo mv kubectl /usr/local/bin/
  ```
- **macOS**: `brew install kubectl`
- **Windows**: `winget install -e --id Kubernetes.kubectl` or
  `choco install kubernetes-cli`

Verify: `kubectl version --client`.

## 4. Set up your kubeconfig

If you don't already use `kubectl` for anything else, the simplest path
is to just use the downloaded file directly:

```bash
mkdir -p ~/.kube
cp ~/Downloads/rancher-cluster.yaml ~/.kube/config
```

If you **already have** a `~/.kube/config` (e.g. from another cluster),
merge the new one in instead of overwriting it:

```bash
KUBECONFIG=~/.kube/config:~/Downloads/rancher-cluster.yaml \
  kubectl config view --flatten > /tmp/merged-kubeconfig
mv /tmp/merged-kubeconfig ~/.kube/config
```

Verify:

```bash
kubectl config current-context
kubectl get pods -n cps-users
```

The second command should print `No resources found in cps-users
namespace.` (or a list of your own pods) — not a `Forbidden` error. See
the Troubleshooting section below if it fails.

## 5. Install ablator

`ablator` is not published on PyPI (yet) — install from a clone of the
repo. To also get the optional TUI (recommended for a first-time setup —
see below), install the `tui` extra:

```bash
git clone https://github.com/bjoernellens1/ablator.git
cd ablator
pip install -e ".[tui]"     # or just `pip install -e .` to skip the TUI
```

### Recommended: guided setup via the TUI

Run `ablator` with no arguments (or `ablator tui` explicitly) from a
terminal. If no config exists yet at `~/.config/ablator/config.toml`, it
walks you through creating one interactively — prompting for namespace,
KAI queue, priority class, your training image, GPU count, and (if you
have one) a shared dataset/checkpoint PVC name — then drops you into a
k9s-style full-screen view: `1` for the job queue, `2` for currently
running jobs (with live pod status/log tail for k8s-dispatched jobs),
`3` to review your config, `4` to list/switch kubeconfig contexts (useful
if you have more than one cluster in your kubeconfig and need to pick
this one), `q` to quit. This never runs unless you're at an interactive
terminal — scripted/cron use of `ablator run`/`plan`/etc. is completely
unaffected.

### Fallback / scripting path: hand-edit the config

If you'd rather not use the TUI (or you're setting this up non-
interactively), copy the example config directly:

```bash
mkdir -p ~/.config/ablator
cp examples/pytorch-generic.toml ~/.config/ablator/config.toml
```

Edit `~/.config/ablator/config.toml`:

- `[queue] path` — any path on **your own machine** that you can read/
  write (this is ablator's local bookkeeping file; it does not need to be
  reachable from inside the cluster — only `kubectl` needs to reach the
  cluster, from this same machine, when ablator submits/polls jobs).
- `[machines.a100cluster]`:
  - `namespace = "cps-users"` (already set — this is your namespace).
  - `kai_queue = "batch"` and `priority_class = "kai-batch-low"` (already
    set — this is the cluster's general-purpose, lowest-priority,
    preemptible batch queue; sensible defaults for chair-wide use. Ask
    the cluster admin if your project has been given a dedicated
    higher-priority queue/class instead).
  - `image` — your own training image (any image the cluster's nodes can
    pull; a public image on Docker Hub/GHCR works with no extra config,
    a private registry needs `image_pull_secret` set to a Kubernetes
    Secret name the admin has created for you).
  - `gpu_count` — how many GPUs (of the cluster's 8x A100 40GB) your job's
    single pod needs.
  - `extra_volumes` — optional, for a shared PVC (datasets/checkpoints).
    Ask the admin for a PVC name in `cps-users`, or create your own
    backed by the `longhorn` StorageClass (`longhorn-fast` /
    `longhorn-overcommit` also exist for different performance/
    overcommit tradeoffs; `local-path` is node-local and **not** shared
    across pods, so don't use it for anything you expect to survive a
    pod restart or read from your own machine).

## 6. Worked example: submitting a real PyTorch job

`examples/pytorch-generic.toml` (copied to your config in step 5) defines
one job type, `train`, that runs `python train.py --epochs {iterations}
--output-dir {model_path} {extra_args}` inside your configured image —
replace `train.py` and the args with your own script's actual CLI.

Write a spec file, `myjob.json`:

```json
{
  "name": "myjob",
  "parallel": true,
  "base": {"type": "train", "iterations": 20, "machine": "a100cluster",
           "base_args": "--lr 0.001"},
  "arms": [
    {"id": "ctrl", "extra_args": ""},
    {"id": "bigger_lr", "extra_args": "--lr 0.01"}
  ]
}
```

Plan (enqueue) it:

```bash
$ ablator plan myjob.json
[plan] enqueued 2 jobs for ablation 'myjob':
  myjob_ctrl                              machine=a100cluster type=train  -> output/myjob_ctrl
                                          scene=
  myjob_bigger_lr                         machine=a100cluster type=train  -> output/myjob_bigger_lr
                                          scene=
```

Run the dispatcher (this host needs `kubectl` on PATH and your kubeconfig
from step 4 configured; it submits each job as a Kubernetes `Job` in
`cps-users` and polls it to completion):

```bash
ablator run
```

Check status from another terminal (or after `run` exits) — columns are
`id lane status machine claimed_by elapsed depends_on progress`:

```bash
$ ablator status myjob
id                                       lane status       machine  claimed_by elapsed  depends_on   progress
myjob_ctrl                               2    done         a100cl.. a100cluster 0h04m    -
myjob_bigger_lr                          2    running      a100cl.. a100cluster 0h01m    -

totals: done=1, running=1
```

(exact column widths/truncation depend on your terminal and job id
length — this is `ablator status`'s real column layout, not just its
column names.)

Once jobs finish, collect result files (per `result_glob` in your config,
`{model_path}/report.json` by default in the generic example — adjust to
whatever your own `train.py` actually writes):

```bash
$ ablator collect myjob
== myjob_ctrl (output/myjob_ctrl)
   output/myjob_ctrl/report.json
   {"final_loss": 0.041, ...}
```

(`ablator collect` only lists jobs whose status is `done`; a job still
`running` or `pending` won't show up yet.)

Cancel any still-pending jobs of this ablation if needed:

```bash
ablator cancel myjob
```

## 7. Troubleshooting

**`kubectl` says `Forbidden`** (e.g. `Error from server (Forbidden):
pods is forbidden: User "..." cannot list resource "pods" in API group
"" in the namespace "cps-users"`):
- Most likely cause: you aren't (yet) a member of the `CPS` Authentik
  group. Contact the cluster admin.
- Second most likely cause: a namespace typo. It's `cps-users`, not
  `cps-user` or your own username.

**Pods stuck `Pending`**:
- This cluster uses KAI Scheduler with three priority tiers: `courses`
  (highest, reclaims from others), `phd-interactive`, and `batch`
  (lowest — what chair users dispatch to by default via
  `priority_class = "kai-batch-low"`). `batch` has **zero guaranteed
  quota** — it only runs in currently-idle capacity, and can be preempted
  at any time by higher-priority workloads.
- Check why: `kubectl describe pod -n cps-users <pod-name>` (look at the
  `Events` section for scheduling/preemption reasons) and
  `kubectl get podgroup -n cps-users` (KAI's own gang-scheduling object;
  its status/conditions often explain a stuck placement KAI is still
  trying to satisfy).
- **Your job must checkpoint.** A `batch`-queue pod can be evicted for
  cluster consolidation or preempted by a higher-priority workload at any
  time; Kubernetes will restart the pod (per `restartPolicy: Never`,
  ablator's dispatch treats this as a job failure, not an automatic
  resume) rather than migrate it live. Write checkpoints periodically to
  a shared volume (see `extra_volumes` in step 5), and have your training
  script resume from the latest checkpoint on startup.

**Image pull failures** (`ErrImagePull` / `ImagePullBackOff` in
`kubectl describe pod`):
- The cluster's nodes need network access to wherever your image lives.
  A public image on Docker Hub or GHCR works with no extra config.
- A private registry needs a `imagePullSecret` — ask the cluster admin
  to create a pull-secret in the `cps-users` namespace, then set
  `image_pull_secret = "<secret-name>"` in your `[machines.a100cluster]`
  config block.

**`RuntimeError: CUDA error: CUDA-capable device(s) is/are busy or
unavailable`, even though `nvidia-smi` shows the GPU completely idle**
(0% util, no processes):
- These GPU nodes run in NVIDIA `Exclusive_Process` compute mode with no
  per-pod permission to change it, and a plain `nvidia.com/gpu: 1`
  request gets no MPS (Multi-Process Service) arbitration by default. Add
  `mps = true` to your `[machines.a100cluster]` block (see
  `examples/pytorch-generic.toml`) — this wires the pod as a client of
  the cluster's own already-running per-node MPS control daemon.
- Even with `mps = true`, the FIRST CUDA call in a job can still
  transiently fail: the control daemon spawns its real server process
  lazily, on a client's first connection, and that very first connection
  can itself lose the race while the server is still starting. Wrap your
  job's first CUDA-touching call in a small retry loop (a few attempts,
  a few seconds apart) rather than treating one failure as fatal.
