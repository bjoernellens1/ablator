"""ablator CLI: plan / status / watch / collect / cancel / health /
stop / skip / requeue / run / start."""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

from . import config as cfgmod
from . import health as healthmod
from . import progress as progmod
from . import runner, spec as specmod
from .queue import Queue, job_lane


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
        print(f"  {'':<40} scene={j['scene']}")


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


def _progress(cfg: dict, j: dict) -> str:
    """Live progress for a running job; model_path resolves against type cwd."""
    if j.get("status") != "running":
        return ""
    try:
        tcfg = cfgmod.type_cfg(cfg, j.get("type", ""), j.get("claimed_by", "any"))
    except KeyError:
        tcfg = {}
    base = tcfg.get("cwd") or os.getcwd()
    return progmod.job_progress(j, base, cfg.get("queue", {}))


def _health_note(j: dict) -> str:
    """Short health tag from the queue record written by the runner."""
    if j.get("status") != "running":
        return ""
    h = j.get("health")
    if not isinstance(h, dict):
        return ""
    age = h.get("log_age_s")
    return f"[{h.get('state', '?')}" + (f" {int(age)}s]" if age is not None else "]")


_STATE_RANK = {"running": 0, "pending": 1}


def _display_sort(jobs: list[dict]) -> list[dict]:
    """Lane desc (urgent first), then state (running, pending, rest); stable."""
    return sorted(jobs, key=lambda j: (-job_lane(j), _STATE_RANK.get(j.get("status"), 2)))


def _status_lines(cfg: dict, jobs: list[dict]) -> list[str]:
    lines = [f"{'id':<40} {'lane':<4} {'status':<12} {'machine':<8} {'claimed_by':<10} "
             f"{'elapsed':<8} {'depends_on':<12} progress"]
    for j in _display_sort(jobs):
        prog = " ".join(x for x in (_progress(cfg, j), _health_note(j)) if x)
        lines.append(f"{j.get('id',''):<40} {job_lane(j):<4} {j.get('status',''):<12} "
                     f"{j.get('machine',''):<8} {j.get('claimed_by','-'):<10} "
                     f"{_elapsed(j):<8} {j.get('depends_on','-'):<12} "
                     f"{prog}")
    counts: dict[str, int] = {}
    for j in jobs:
        counts[j.get("status", "?")] = counts.get(j.get("status", "?"), 0) + 1
    lines.append("")
    lines.append("totals: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return lines


def _lane1_restock_warning(all_jobs: list[dict]) -> str | None:
    """Cheap restock hint: WARN when the background lane is running dry."""
    n = sum(1 for j in all_jobs
            if job_lane(j) == 1 and j.get("status") == "pending")
    if n < 2:
        return (f"WARNING: background lane running dry — {n} pending lane-1 "
                f"job(s); queue more fine-tuning ablations so idle time "
                f"is never wasted")
    return None


def cmd_status(cfg: dict, name: str | None) -> None:
    all_jobs = _queue(cfg).read()
    jobs = _match(all_jobs, name)
    if not jobs:
        print("no matching jobs in queue")
        return
    print("\n".join(_status_lines(cfg, jobs)))
    warn = _lane1_restock_warning(all_jobs)
    if warn:
        print(warn)


def cmd_watch(cfg: dict, name: str | None, interval: int = 60) -> None:
    """Loop status every `interval` s; mirror to queue_status.txt beside the queue."""
    status_path = os.path.join(
        os.path.dirname(cfgmod.queue_path(cfg)), "queue_status.txt")
    while True:
        all_jobs = _queue(cfg).read()
        jobs = _match(all_jobs, name)
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        lines = [f"# queue status @ {stamp}"]
        lines += _status_lines(cfg, jobs) if jobs else ["no matching jobs in queue"]
        warn = _lane1_restock_warning(all_jobs)
        if warn:
            lines.append(warn)
        text = "\n".join(lines) + "\n"
        try:
            tmp = status_path + ".tmp"
            with open(tmp, "w") as f:
                f.write(text)
            os.replace(tmp, status_path)
        except OSError as e:
            print(f"[watch] WARNING: could not write {status_path}: {e}")
        print(text, flush=True)
        time.sleep(interval)


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


# ---------------------------------------------------------- health/control

def cmd_health(cfg: dict, job_id: str | None) -> None:
    """Print artifact-derived health of running (or the named) job(s)."""
    jobs = _queue(cfg).read()
    if job_id:
        jobs = [j for j in jobs if j.get("id") == job_id]
        if not jobs:
            raise SystemExit(f"no job '{job_id}' in queue")
    else:
        jobs = [j for j in jobs if j.get("status") == "running"]
        if not jobs:
            print("no running jobs")
            return
    for j in jobs:
        try:
            tcfg = cfgmod.type_cfg(cfg, j.get("type", ""),
                                   j.get("claimed_by", "any"))
        except KeyError:
            tcfg = {}
        base = tcfg.get("cwd") or os.getcwd()
        h = healthmod.job_health(j, base, cfg.get("queue", {}))
        print(f"{j['id']}: {json.dumps(h)}")


def cmd_control(cfg: dict, action: str, job_id: str) -> None:
    """Write a control file the supervising runner honors on its next poll.

    stop    -> kill, mark failed (no retry)
    skip    -> kill, mark cancelled
    requeue -> kill, mark pending (re-runs from scratch)
    """
    jobs = {j.get("id"): j for j in _queue(cfg).read()}
    j = jobs.get(job_id)
    if j is None:
        raise SystemExit(f"no job '{job_id}' in queue")
    if j.get("status") != "running":
        raise SystemExit(f"job '{job_id}' is {j.get('status')}, not running — "
                         f"use 'cancel' for pending jobs")
    path = runner.control_path(cfg, job_id)
    with open(path, "w") as f:
        f.write(action + "\n")
    print(f"[{action}] wrote {path} — the runner acts on its next "
          f"supervision poll (~{runner.HEALTH_POLL_S}s)")


# ---------------------------------------------------------------- promote

def cmd_promote(cfg: dict, job_id: str, lane_str: str) -> None:
    """Move a pending job to another lane (1=background, 2=standard, 3=urgent)."""
    try:
        lane = int(lane_str)
    except ValueError:
        raise SystemExit(f"lane must be 1, 2 or 3 (got {lane_str!r})")
    if lane not in (1, 2, 3):
        raise SystemExit(f"lane must be 1, 2 or 3 (got {lane})")
    q = _queue(cfg)
    with q._open_locked() as f:
        jobs = q._load(f)
        for j in jobs:
            if j.get("id") == job_id:
                if j.get("status") != "pending":
                    raise SystemExit(f"job '{job_id}' is {j.get('status')}, "
                                     f"not pending — only pending jobs can "
                                     f"change lanes")
                old = job_lane(j)
                j["lane"] = lane
                q._save(f, jobs)
                print(f"[promote] {job_id}: lane {old} -> {lane}")
                return
    raise SystemExit(f"no job '{job_id}' in queue")


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
    sp = sub.add_parser("watch", help="loop status; mirror to queue_status.txt")
    sp.add_argument("name", nargs="?")
    sp.add_argument("--interval", type=int, default=60)
    sp = sub.add_parser("collect", help="print result files of done jobs")
    sp.add_argument("name")
    sp = sub.add_parser("cancel", help="cancel pending jobs of an ablation")
    sp.add_argument("name")
    sp = sub.add_parser("promote", help="move a pending job to another lane")
    sp.add_argument("job_id")
    sp.add_argument("lane")
    sp = sub.add_parser("health", help="print artifact-derived job health")
    sp.add_argument("job_id", nargs="?")
    for action, hlp in (("stop", "kill a running job (failed, no retry)"),
                        ("skip", "kill a running job (cancelled)"),
                        ("requeue", "kill a running job and re-queue it")):
        sp = sub.add_parser(action, help=hlp)
        sp.add_argument("job_id")
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
    elif a.cmd == "watch":
        cmd_watch(cfg, a.name, interval=a.interval)
    elif a.cmd == "collect":
        cmd_collect(cfg, a.name)
    elif a.cmd == "cancel":
        cmd_cancel(cfg, a.name)
    elif a.cmd == "promote":
        cmd_promote(cfg, a.job_id, a.lane)
    elif a.cmd == "health":
        cmd_health(cfg, a.job_id)
    elif a.cmd in ("stop", "skip", "requeue"):
        cmd_control(cfg, a.cmd, a.job_id)
    elif a.cmd == "run":
        runner.run_loop(cfg, once=a.once)
    elif a.cmd == "start":
        runner.start_runners(cfg)


if __name__ == "__main__":
    main()
