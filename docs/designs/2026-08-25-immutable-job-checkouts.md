# Immutable Per-Job Checkout Hardening

## Status

Approved on 2026-08-25 for Splatograph issue #259 and Ablator issue #7.
This design builds on Ablator PRs #35 and #37 through #40 without rewriting
their published history. The follow-up branch is intentionally stacked and
will be retargeted after those dependencies merge.

## Problem

A queued scientific job currently identifies requested parameters, but the
runner can still execute whatever happens to be present in a shared mutable
checkout when the job is claimed. `auto_sync_ref` makes that worse for a
sequential family: a merge between arms can silently change the code under the
same registered experiment. A pre-launch dirty/SHA observation detects only
some failures; it does not make the source immutable for the lifetime of the
job or attest what remained after execution.

PRs #35 and #37 through #40 establish a full-SHA schema, detached worktree
materialization, pin-aware policy, visibility, and age-based cleanup. The
hardening below closes the remaining scientific-execution gaps.

## Binding invariants

1. A `GRADEABLE_DECLARED` job and any job type configured with
   `require_pinned_git = true` must carry a full 40-character requested commit
   SHA. Missing or ambiguous intent fails before enqueue or launch.
2. Every dependency edge must preserve the same requested repository and SHA.
   Sequential plan arms with mixed pins are rejected during expansion; queue
   append also validates edges so alternate producers cannot bypass the rule.
3. Every bare-metal launch attempt gets its own detached worktree. Worktrees
   are never shared by concurrent jobs or retries. The shared cache contains
   only Git object storage and inactive completed worktrees.
4. Tracked submodules are synchronized, initialized recursively at the commits
   recorded by the superproject, and verified clean before launch. Missing,
   conflicted, or drifted submodules fail closed.
5. A containerized pinned job must bind the prepared checkout into the
   container. Every bind of that checkout is made read-only; writable data and
   result mounts remain distinct. Direct host jobs receive
   `PYTHONDONTWRITEBYTECODE=1` and are checked again after execution.
6. Before launch, Ablator records a canonical execution receipt containing the
   requested and executed source identities, detached state, submodule state,
   machine and runner/config identities, rendered cwd, container runtime,
   image, and normalized bind mounts. The exact queue envelope containing this
   receipt is injected into the workload. After the process or pod terminates,
   a final attestation records commit/ref/dirty/submodule state and changes a
   nominal success to failure if immutability cannot be proven.
7. A lease is created atomically with worktree materialization and remains
   active until post-run attestation finishes. Materialization and garbage
   collection share the repository lock. GC never follows sidecar paths outside
   the configured cache root, never removes active leases, never touches result
   directories, and reports cleanup/prune failures instead of claiming removal.
8. Kubernetes jobs use one emptyDir checkout per pod, fetch the exact registered
   SHA, recursively initialize submodules, verify HEAD and cleanliness in the
   init container, mount source read-only in the trainer, and carry the same
   receipt contract. Deployment proof remains a separate gate from synthetic
   manifest tests.

## Data contract

Queue records gain these auditable fields:

- `requested_git_sha`, `git_repo`: immutable registered intent;
- `source_checkout`, `source_lease`: per-attempt host materialization identity;
- `execution_receipt`: canonical pre-launch receipt (`ablator.execution/v1`);
- `execution_attestation`: post-run source state and verdict;
- `source_prepare_error`: fail-closed preparation or attestation reason.

The pre-launch receipt is included in protected `ABLATOR_JOB_JSON`. It never
contains arbitrary environment values or credentials. Runtime identity records
only the runtime executable, image reference, normalized bind source/target and
options, cwd, and SHA-256 of the merged type configuration and rendered argv.

## Lifecycle

1. Authoring resolves a full SHA into each queue job and rejects chain drift.
2. Claiming validates the configured pin requirement.
3. Preparation takes the cache lock, creates a unique worktree and active
   sidecar lease, initializes submodules, rewrites paths, and makes source binds
   read-only.
4. Pre-launch verification captures source state, builds the receipt, persists
   it, then renders the protected child environment from that updated job.
5. Supervision runs the workload.
6. Final attestation runs before lease release and before a success is accepted.
7. The lease becomes inactive. Later GC may remove only an old inactive entry
   proven to be contained by the cache root.

Retries repeat steps 3 through 7 with a new worktree and lease. A runner crash
can leave an active lease retained; safety takes precedence over automatic
space recovery. Such an orphan is visible to GC and requires a later explicit
reconciliation path rather than unsafe age-only deletion.

## Compatibility and rollout

Undeclared legacy/debug jobs remain runnable from mutable source only when the
type does not require a pin. They remain non-gradeable and cannot acquire a
scientific declaration after enqueue. Splatograph config will require pins for
its scientific Ablator job types and use `{repo_cwd}` in host paths so isolated
checkouts work on every runner.

Unit and integration tests prove contracts with local temporary Git repositories
and synthetic process/container manifests. They do not prove scientific
validity or production deployment. Live-runner rollout requires the queue to be
idle, merged releases installed on each runner, and a fresh pinned diagnostic
whose queue receipt, workload artifacts, and final attestation agree.

## Rejected alternatives

- Merging the existing stack unchanged leaves submodule, runtime immutability,
  receipt, chain, and GC races unresolved.
- Rebuilding the stack from `main` duplicates active work and destroys the
  review history. The follow-up instead composes with it.
- Reusing one worktree per SHA makes concurrent jobs and retries share a
  writable filesystem. Content-addressed object storage is shared; execution
  worktrees are not.
