"""ablator CLI: plan / status / collect / cancel / run / start."""
from __future__ import annotations

import argparse
import glob
import os
import time

from . import config as cfgmod
from . import runner, spec as specmod
from .queue import Queue


def _queue(cfg: dict) -> Queue:
    return Queue(cfgmod.queue_path(cfg))


# ------------------------------------------------------------------ plan

def cmd_plan(cfg: dict, spec_path: str, dry_run: bool = False) -> None:
    sp = specmod.load_spec(spec_path)
    tmpl = cfg["queue"].get("model_path_template",
                            specmod.DEFAULT_MODEL_PATH_TEMPLATE)
    jobs = specmod.expand_spec(sp, model_path_template=tmpl)
    for j in jobs:  # validate types exist before enqueueing anything
        if j["type"] not in cfg.get("types", {}):
            raise SystemExit(f"spec job {j['id']} uses type '{j['type']}' "
                             f"not defined in config {cfg['_path']}")
    if not dry_run:
        _queue(cfg).append(jobs)
    verb = "would enqueue" if dry_run else "enqueued"
    print(f"[plan] {verb} {len(jobs)} jobs for ablation '{sp['name']}':")
    for j in jobs:
        dep = f"  (after {j['depends_on']})" if j.get("depends_on") else ""
        print(f"  {j['id']:<40} machine={j['machine']:<6} type={j['type']:<8} "
              f"-> {j['model_path']}{dep}")


# ---------------------------------------------------------------- status

def _elapsed(j: dict) -> str:
    start = j.get("claimed_at")
    if not start:
        return ""
    try:
        t0 = time.mktime(time.strptime(start, "%Y-%m-%dT%H:%M:%S"))
        end = j.get("finished_at")
        t1 = time.mktime(time.strptime(end, "%Y-%m-%dT%H:%M:%S")) if end else time.time()
        m = int(t1 - t0) // 60
        return f"{m // 60}h{m % 60:02d}m"
    except ValueError:
        return ""


def _match(jobs: list[dict], name: str | None) -> list[dict]:
    if not name:
        return jobs
    return [j for j in jobs
            if j.get("ablation") == name or j.get("id", "").startswith(name + "_")]


def cmd_status(cfg: dict, name: str | None) -> None:
    jobs = _match(_queue(cfg).read(), name)
    if not jobs:
        print("no matching jobs in queue")
        return
    print(f"{'id':<40} {'status':<12} {'machine':<8} {'claimed_by':<10} "
          f"{'elapsed':<8} depends_on")
    for j in jobs:
        print(f"{j.get('id',''):<40} {j.get('status',''):<12} "
              f"{j.get('machine',''):<8} {j.get('claimed_by','-'):<10} "
              f"{_elapsed(j):<8} {j.get('depends_on','-')}")
    counts: dict[str, int] = {}
    for j in jobs:
        counts[j.get("status", "?")] = counts.get(j.get("status", "?"), 0) + 1
    print("\ntotals: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))


# --------------------------------------------------------------- collect

def cmd_collect(cfg: dict, name: str) -> None:
    """Generic collect: print per-job result files matched by result_glob.

    result_glob comes from the job's type config (or [queue] result_glob
    as a fallback) and may use the same template variables as commands,
    e.g. "{model_path}/comparison/*/report.json". Paths are resolved
    relative to the type's cwd if set.
    """
    jobs = [j for j in _match(_queue(cfg).read(), name) if j.get("status") == "done"]
    if not jobs:
        print(f"no done jobs for ablation '{name}' yet")
        return
    for j in jobs:
        machine = j.get("claimed_by", "any")
        try:
            tcfg = cfgmod.type_cfg(cfg, j.get("type", ""), machine)
        except KeyError:
            tcfg = {}
        pattern = tcfg.get("result_glob") or cfg["queue"].get("result_glob")
        print(f"== {j['id']} ({j.get('model_path','')})")
        if not pattern:
            print("   (no result_glob configured — nothing to collect)")
            continue
        pat = pattern.format(**runner._job_vars(j, machine))
        base = tcfg.get("cwd") or os.getcwd()
        if not os.path.isabs(pat):
            pat = os.path.join(base, pat)
        hits = sorted(glob.glob(pat))
        if not hits:
            print(f"   no files match {pat}")
        for h in hits:
            print(f"   {h}")
            if h.endswith(".json"):
                try:
                    with open(h) as f:
                        text = f.read()
                    print("   " + text[:2000].replace("\n", "\n   "))
                except OSError:
                    pass


# ---------------------------------------------------------------- cancel

def cmd_cancel(cfg: dict, name: str) -> None:
    n = _queue(cfg).cancel(
        lambda j: j.get("ablation") == name or j.get("id", "").startswith(name + "_"))
    print(f"[cancel] cancelled {n} pending jobs of ablation '{name}'")


# ------------------------------------------------------------------ main

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="ablator",
        description="Cross-machine ablation queue orchestrator (stdlib-only).")
    p.add_argument("--config", help="config file (TOML/JSON); default "
                   "$ABLATOR_CONFIG or ~/.config/ablator/config.toml")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("plan", help="expand a spec JSON into queue jobs")
    sp.add_argument("spec")
    sp.add_argument("--dry-run", action="store_true")
    sp = sub.add_parser("status", help="print queue state")
    sp.add_argument("name", nargs="?")
    sp = sub.add_parser("collect", help="print result files of done jobs")
    sp.add_argument("name")
    sp = sub.add_parser("cancel", help="cancel pending jobs of an ablation")
    sp.add_argument("name")
    sp = sub.add_parser("run", help="runner loop: claim and execute jobs")
    sp.add_argument("--once", action="store_true",
                    help="claim/run at most one job, then exit")
    sub.add_parser("start", help="launch runners locally and on ssh remotes")

    a = p.parse_args(argv)
    cfg = cfgmod.load_config(a.config)
    if a.cmd == "plan":
        cmd_plan(cfg, a.spec, dry_run=a.dry_run)
    elif a.cmd == "status":
        cmd_status(cfg, a.name)
    elif a.cmd == "collect":
        cmd_collect(cfg, a.name)
    elif a.cmd == "cancel":
        cmd_cancel(cfg, a.name)
    elif a.cmd == "run":
        runner.run_loop(cfg, once=a.once)
    elif a.cmd == "start":
        runner.start_runners(cfg)


if __name__ == "__main__":
    main()
