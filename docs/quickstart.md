# Quickstart

## Install

```bash
pip install -e .
# or, with zero installs, straight from the checkout:
PYTHONPATH=src python3 -m ablator.cli --help
```

The optional TUI (`ablator tui`, or bare `ablator` on a TTY) needs
[Textual](https://github.com/Textualize/textual):

```bash
pip install ablator[tui]
```

## Configure

```bash
mkdir -p ~/.config/ablator
cp examples/splatograph.toml ~/.config/ablator/config.toml
# edit paths, machine names, hostname_patterns, image names, etc.
```

Config resolution order: `--config PATH` flag > `$ABLATOR_CONFIG` env var
> `~/.config/ablator/config.toml`. See the [config reference](config-reference.md)
for every field.

## Run the loop

```bash
ablator plan spec.json      # expand a spec into queued jobs (refuses duplicate ids)
ablator status [name]       # queue state table
ablator run [--once]        # runner loop: claim + execute jobs on THIS machine
ablator start                # session-proof launch: local runner + ssh remotes
ablator collect <name>       # print result files (result_glob) of done jobs
ablator cancel <name>        # cancel that ablation's still-pending jobs
```

`ablator run` is the foreground/single-machine loop; `ablator start` is
what you use once you have more than one machine (it launches a
`setsid nohup` runner locally and over ssh on every configured remote —
see [multi-machine setup](multi-machine.md)).

## A minimal spec

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

```bash
ablator plan spec.json --dry-run   # preview the expanded jobs, no queue writes
ablator plan spec.json             # actually queue them
```

See the full [spec reference](spec-reference.md) for every field and
how arm overrides / `depends_on` chaining work.

## Watching it work

```bash
ablator watch [name] [--interval N]   # loops status; also mirrors to queue_status.txt
ablator errors [name]                 # failed/quarantined/paused jobs + classification
ablator health [job_id]               # artifact-derived job health (iteration progress, staleness)
```

Per-job logs land at `<log_dir>/<job id>.log` (default `log_dir`:
`dirname(queue path)`).
