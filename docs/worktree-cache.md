# Immutable Git worktree cache

SHA-pinned jobs materialize detached Git worktrees on the machine that actually
executes the job. Git object storage is shared, but every job attempt gets a
unique execution worktree. Concurrent jobs and retries therefore never share a
writable filesystem, even when they request the same repository and SHA.

## Cache location

Default:

```text
~/.cache/ablator/worktrees/
```

Global override:

```toml
[git]
worktree_root = "/fast/local/ablator/worktrees"
```

Machine-specific override:

```toml
[machines.rtx3090]
git_worktree_root = "/home/bjoern1/.cache/ablator/worktrees"
```

The process environment variable `ABLATOR_GIT_WORKTREE_ROOT` is also supported when no config override is present.

Each materialized checkout has an adjacent sidecar metadata file outside the
worktree. It records the requested SHA, repository identity, owning repository
path, unique lease ID, repository lock, active state, and timestamps. The lease
becomes inactive only after post-run source attestation. Keeping the sidecar
outside the worktree ensures cache metadata never makes the immutable checkout
dirty.

## Garbage collection

The default retention age is 30 days. Configure it with:

```toml
[git]
gc_max_age_days = 30
```

Preview what would be removed:

```bash
ablator --config ~/.config/ablator/config.toml gc --dry-run
```

Run cleanup:

```bash
ablator --config ~/.config/ablator/config.toml gc
```

Override the age for one invocation:

```bash
ablator --config ~/.config/ablator/config.toml gc --dry-run --max-age-days 7
ablator --config ~/.config/ablator/config.toml gc --max-age-days 7
```

`gc` acts on the current machine's cache only. Remote machines manage their own local caches, which keeps source preparation and cleanup aligned with where the files actually live.

## Safety rules

- An active sidecar lease is protected regardless of age or queue-update lag.
- A worktree referenced by a currently `running` queue job is also protected;
  this preserves compatibility with older sidecars.
- Dry-run performs the same classification as a real cleanup but does not mutate anything.
- Cleanup accepts only an adjacent sidecar and checkout whose resolved paths
  are strictly below the configured cache root. Invalid records are reported;
  their claimed paths are never removed.
- Materialization, lease release, and cleanup share the repository lock. GC
  re-reads the lease under that lock and refuses it if it is active or changed.
- Managed worktrees are removed with `git worktree remove --force` followed by
  a successful `git worktree prune`, so the owning repository does not retain
  stale administration entries.
- An orphaned cache whose owning repository has already disappeared can be removed directly because no Git administration tree remains to update.
- Cleanup/prune errors are reported and make `ablator gc` exit non-zero. The
  sidecar is retained as evidence instead of treating a partial cleanup as
  success.
- A runner crash can leave an active lease. GC deliberately retains it; an
  operator must reconcile that orphan explicitly rather than weakening the
  active-lease safety rule with an age-only deletion.

The cache shares Git objects only. It does not weaken the requested-vs-executed
SHA contract or remove job result directories.
