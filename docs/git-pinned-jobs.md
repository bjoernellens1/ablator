# Git-pinned jobs

Ablator can run an experiment against one immutable Git commit instead of whatever happens to be checked out in a machine's mutable development tree.

## Spec

Use either the compact form:

```json
{
  "name": "pr965gpu",
  "git_sha": "0123456789abcdef0123456789abcdef01234567",
  "base": {
    "machine": "main",
    "type": "replay"
  },
  "arms": [
    {"id": "default", "extra_args": ""},
    {"id": "legacy", "extra_args": "--no-streaming_optimization_engine"}
  ]
}
```

or the structured form when the repository must be explicit:

```json
"git": {
  "repo": "https://github.com/example/project.git",
  "sha": "0123456789abcdef0123456789abcdef01234567"
}
```

Only full 40-character commit SHAs are accepted. Ablator deliberately does not put a floating branch, tag, or PR ref into the queue. Resolve those to an immutable SHA before planning.

The Git target can be declared at spec, `base`, or arm level. Nearest scope
wins: arm > base > spec. A sequential (`parallel: false`) plan must resolve to
the same repository and SHA for every dependency edge. Mixed revisions are
rejected before enqueue, and the queue repeats that check for non-plan
producers.

Gradeable declarations always require a Git target. Operators can extend the
same fail-closed rule to any configured type:

```toml
[types.replay]
require_pinned_git = true
```

## What the runner does

For a pinned bare-metal job, the runner that actually claims the job:

1. verifies that the requested commit is available, fetching it from `origin` when necessary;
2. creates a unique detached worktree for that job attempt under the machine's
   Ablator cache;
3. recursively initializes every submodule at the commit recorded by the
   superproject and verifies the complete tree is clean;
4. rewrites the configured job `cwd` and source bind mount to that worktree;
5. makes every container bind of the checkout read-only (direct processes get
   `PYTHONDONTWRITEBYTECODE=1`);
6. records the resolved launch and source identity in an execution receipt;
7. requires `executed_git_sha == requested_git_sha`, detached HEAD, and clean
   source before launch and checks the same state again after the process exits.

A mismatch or unprovable checkout fails before the expensive workload starts.
Post-run drift changes an otherwise successful exit into a failed job. Attempts
never share a writable worktree, even when they request the same SHA.

The operator's normal development checkout is not switched, rebased, reset, or pulled as part of pinned-job preparation.

## Remote machines

There is no coordinator-side checkout handoff. `ablator start` launches independent `ablator run` processes on configured SSH machines, and the target runner materializes the SHA locally when it claims the job.

This means `main`, `r9700`, `rtx3090`, and future workers can each have different home directories and cache locations while still executing the same immutable revision.

Per-machine cache roots are optional:

```toml
[git]
worktree_root = "~/.cache/ablator/worktrees"

[machines.rtx3090]
git_worktree_root = "/home/bjoern1/.cache/ablator/worktrees"
```

`ABLATOR_GIT_WORKTREE_ROOT` is also supported as a process-level override.

A type whose source is bind-mounted into a Docker/Podman container must expose the configured source checkout in its command. Existing literal `cwd` mounts are rewritten automatically. New templates can use `{repo_cwd}` explicitly, for example:

```toml
[types.replay]
cwd = "/home/user/git/project"
command = [
  "docker", "run", "--rm",
  "-v", "{repo_cwd}:/workspace/project",
  "-w", "/workspace/project",
  "image:tag", "python", "train.py"
]
```

A pinned container job that does not expose its source checkout is rejected
rather than pretending the pin controls code baked into an unrelated image.
Writable datasets, scratch paths, and results must use separate mounts; the
source bind itself is always rewritten with `ro`/`readonly`.

## Kubernetes

Pinned Kubernetes jobs require `git_sync_repo_url` on the target machine
configuration. The dispatcher validates the requested revision policy first;
the pod's git-sync init container then fetches the exact requested SHA,
initializes recursive submodules, verifies detached/clean state, and records a
termination-message proof. The trainer sees that `emptyDir` at its configured
working directory as read-only. Final queue attestation also records the actual
pod, node, image reference, and image ID. A pinned k8s job is rejected when
git-sync is not configured or when its source proof is absent or inconsistent.

## Interaction with `urgent_fix_unsynced`

Ablator separates two policies:

- `auto_sync_ref` is a freshness rule for mutable shared checkouts;
- explicit `[[urgent_fixes.fixes]]` SHAs are mandatory safety requirements.

A SHA-pinned job is not silently rewritten to `origin/main` merely because `auto_sync_ref = "origin/main"`. Its exact requested SHA is preserved. However, every explicitly registered urgent-fix SHA must be an ancestor of the requested revision or that job is rejected before launch.

If a machine is paused specifically with `category=urgent_fix_unsynced`, legacy mutable jobs remain blocked, but a SHA-pinned job may still be claimed and evaluated against its own immutable revision. `manual_pause` and unknown pause categories remain absolute and block pinned jobs too.

This avoids the previous failure mode where a legitimate detached PR checkout caused `git pull --ff-only` to fail, repeatedly pausing the machine even though the job intended to validate that exact PR revision.

## Provenance fields

Pinned jobs persist source information in the queue as it becomes available:

- `requested_git_sha`
- `git_repo` when declared in the spec
- `source_repo`
- `source_checkout`
- `source_lease`
- `executed_git_sha`
- `execution_receipt` (pre-launch source, runner, config, argv, runtime, image,
  and normalized mount identity)
- `execution_attestation` (post-run source verdict and, for Kubernetes, actual
  pod/node/image identity)
- `source_prepare_error` on pre-launch preparation/provenance failure

The protected `ABLATOR_JOB_JSON` passed to the workload is rendered only after
the pre-launch receipt is persisted. Receipts intentionally hash the rendered
argv and merged type configuration rather than copying arbitrary environment
values or credentials.

These contracts are execution evidence, not proof that a scientific result is
valid. Production rollout still requires idle runners, the merged release on
every execution machine, and a fresh real-path pinned diagnostic whose queue,
workload artifacts, and final attestation agree.
