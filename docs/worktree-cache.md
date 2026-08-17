# Immutable Git worktree cache

SHA-pinned jobs materialize detached Git worktrees on the machine that actually executes the job. Repeated jobs for the same repository and SHA reuse the same verified clean worktree rather than creating a fresh clone every time.

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

Each materialized checkout has a sidecar metadata file outside the worktree. It records the requested SHA, repository identity, owning repository path and last-use timestamp. Keeping the sidecar outside the worktree ensures cache metadata never makes the immutable checkout dirty.

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

- A worktree referenced by a currently `running` queue job is protected regardless of age.
- Dry-run performs the same classification as a real cleanup but does not mutate anything.
- Managed worktrees are removed with `git worktree remove --force` followed by `git worktree prune`, so the owning repository does not retain stale worktree administration entries.
- An orphaned cache whose owning repository has already disappeared can be removed directly because no Git administration tree remains to update.
- Cleanup errors are reported and make `ablator gc` exit non-zero; they are not silently treated as successful removal.
- Reusing a cached checkout remains subject to the normal pinning checks: exact requested HEAD and a clean worktree are required before execution.

The cache is an optimization only. It does not weaken the requested-vs-executed SHA contract.
