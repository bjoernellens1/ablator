# Multi-machine setup

1. **Same config file on every machine.** Machine identity resolves
   per-host from `hostname_patterns` (see
   [config reference](config-reference.md#machinesname)), so one shared
   config works everywhere — the queue path must point at a filesystem
   every machine mounts (NFS works; `flock` over NFSv4 serializes claims
   across hosts).
2. **Install ablator on each machine** (or set `runner_command` to a
   venv path / `PYTHONPATH=... python3 -m ablator.cli run` if you'd
   rather not `pip install` system-wide).
3. **Passwordless ssh** from the machine you'll run `ablator start` on
   to every `[machines.*] ssh` address.
4. Launch:

   ```bash
   ablator start
   ```

   This launches a `setsid nohup ablator run` locally **and** over ssh
   on each configured remote, skipping any machine where a runner is
   already alive (so it's safe to re-run `ablator start` after adding a
   new machine). Runner logs land at `<log_dir>/runner_<machine>.log`.

## Watching the whole fleet

```bash
ablator status          # all ablations, all machines
ablator watch --interval 30
ablator errors           # failed/quarantined jobs across the fleet, classified
```

Because the queue file is shared, `status`/`watch`/`errors`/`health`
give you a fleet-wide view regardless of which machine you run them
from — you don't need to ssh around to check on jobs claimed by another
runner.

## Pausing a machine

```bash
ablator pause r9700     # blocks new claims on r9700; running jobs unaffected
ablator unpause r9700
```

Useful when a machine needs manual attention (driver issue, disk full)
without disturbing whatever it's currently running.
