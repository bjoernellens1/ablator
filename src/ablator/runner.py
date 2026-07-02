"""Job execution: command-template rendering and the runner loop.

Jobs carry no execution knowledge. Each job "type" maps, in the host
config, to a command template plus env, cwd, and optional per-machine
overrides. Template variables available in command tokens and env
values:

  {scene} {model_path} {extra_args} {iterations} {id} {machine}

A command token that is exactly "{extra_args}" expands to zero or more
argv items (whitespace-split); everywhere else substitution is plain
string formatting. Everything about a job executes via the configured
command (normally a podman/docker run) — the runner itself only does
queue bookkeeping and process launch.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import time

from . import config as cfgmod
from . import resources
from .queue import Queue

IDLE_POLL_S = 180
BUSY_POLL_S = 120


class TemplateError(SystemExit):
    pass


def _job_vars(job: dict, machine: str) -> dict:
    return {
        "scene": job.get("scene", ""),
        "model_path": job.get("model_path", ""),
        "extra_args": job.get("extra_args", ""),
        "iterations": str(job.get("iterations", "")),
        "id": job.get("id", ""),
        "machine": machine,
    }


def _fmt(s: str, vars: dict) -> str:
    try:
        return s.format(**vars)
    except (KeyError, IndexError) as e:
        raise TemplateError(f"unknown template variable in {s!r}: {e}")


def render_command(tcfg: dict, job: dict, machine: str) -> tuple[list[str], dict, str | None]:
    """Render (argv, env, cwd) for a job from its merged type config."""
    vars = _job_vars(job, machine)
    argv: list[str] = []
    for tok in tcfg.get("command", []):
        if tok == "{extra_args}":
            argv.extend(shlex.split(vars["extra_args"]))
        else:
            argv.append(_fmt(tok, vars))
    if not argv:
        raise TemplateError(f"job type for {job.get('id')} has empty command template")
    env = os.environ.copy()
    for k, v in (tcfg.get("env") or {}).items():
        env[k] = _fmt(str(v), vars)
    cwd = tcfg.get("cwd")
    return argv, env, (_fmt(cwd, vars) if cwd else None)


def type_capable(tcfg: dict, runtime_default: str = "docker") -> bool:
    """Capability probe: required images must already exist locally."""
    images = tcfg.get("require_images")
    if not images:
        return True
    runtime = tcfg.get("image_probe_runtime", runtime_default)
    return resources.images_present(runtime, images)


def make_can_run(cfg: dict, machine: str):
    """Per-scan claim predicate: job type defined here + capability probe.

    Capability results are cached for the lifetime of the predicate
    (one claim scan) so image probes run at most once per type.
    """
    cache: dict[str, bool] = {}

    def can_run(job: dict) -> bool:
        jt = job.get("type", "")
        if jt not in cache:
            try:
                tcfg = cfgmod.type_cfg(cfg, jt, machine)
            except KeyError:
                print(f"[ablator] skipping {job.get('id')}: type '{jt}' not in config",
                      flush=True)
                cache[jt] = False
            else:
                ok = type_capable(tcfg)
                if not ok:
                    print(f"[ablator] type '{jt}' not capable on {machine} "
                          f"(missing images)", flush=True)
                cache[jt] = ok
        return cache[jt]

    return can_run


def run_job(cfg: dict, job: dict, machine: str) -> str:
    """Execute one job; returns 'done' or 'failed'. Logs to <log_dir>/<id>.log."""
    log_path = os.path.join(cfgmod.log_dir(cfg), f"{job['id']}.log")
    try:
        tcfg = cfgmod.type_cfg(cfg, job.get("type", ""), machine)
        argv, env, cwd = render_command(tcfg, job, machine)
    except (KeyError, TemplateError) as e:
        print(f"[ablator] {job['id']} unrunnable: {e}", flush=True)
        return "failed"
    print(f"[ablator] running {job['id']} -> {job.get('model_path')} (log {log_path})",
          flush=True)
    try:
        with open(log_path, "w") as lf:
            lf.write(f"# {time.strftime('%Y-%m-%dT%H:%M:%S')} {job['id']}\n"
                     f"# cwd={cwd or os.getcwd()}\n# {shlex.join(argv)}\n")
            lf.flush()
            rc = subprocess.run(argv, env=env, cwd=cwd,
                                stdout=lf, stderr=subprocess.STDOUT).returncode
        return "done" if rc == 0 else "failed"
    except Exception as e:
        print(f"[ablator] {job['id']} crashed: {e}", flush=True)
        return "failed"


def run_loop(cfg: dict, once: bool = False) -> None:
    machine = cfgmod.machine_name(cfg)
    q = Queue(cfgmod.queue_path(cfg))
    print(f"[ablator] runner on {machine} watching {q.path}", flush=True)
    while True:
        if resources.machine_busy(cfg, machine):
            if once:
                return
            time.sleep(BUSY_POLL_S)
            continue
        job = q.claim_next(machine, can_run=make_can_run(cfg, machine))
        if job is None:
            if once:
                return
            time.sleep(IDLE_POLL_S)
            continue
        status = run_job(cfg, job, machine)
        # one retry on failure, then quarantine
        if status == "failed" and not job.get("retried"):
            q.update(job["id"], retried=True)
            print(f"[ablator] retrying {job['id']} once", flush=True)
            job["retried"] = True
            status = run_job(cfg, job, machine)
            if status == "failed":
                status = "quarantined"
        q.finish(job["id"], status)
        print(f"[ablator] {job['id']} -> {status}", flush=True)
        if once:
            return


# ------------------------------------------------------------------ start

def start_runners(cfg: dict) -> None:
    """Session-proof launch of the runner locally and on ssh remotes.

    Local: setsid nohup ablator run. Remotes: every [machines.<name>]
    with an `ssh` address that is not this machine. The remote runner
    command defaults to "ablator run" (override per machine with
    runner_command, e.g. a venv path); its config resolves on the
    remote via ~/.config/ablator/config.toml or ABLATOR_CONFIG.
    """
    me = cfgmod.machine_name(cfg)
    ldir = cfgmod.log_dir(cfg)
    os.makedirs(ldir, exist_ok=True)

    probe = subprocess.run(["pgrep", "-f", "[a]blator run"],
                           capture_output=True, text=True)
    if probe.stdout.strip():
        print(f"[start] runner already running on {me} — skipping local start")
    else:
        log = os.path.join(ldir, f"runner_{me}.log")
        cmd = (f"setsid nohup ablator run --config {shlex.quote(cfg['_path'])} "
               f"</dev/null > {shlex.quote(log)} 2>&1 &")
        subprocess.run(["bash", "-c", cmd], check=True)
        print(f"[start] launched runner on {me} (log {log})")

    for name, m in cfg.get("machines", {}).items():
        if name == me or not m.get("ssh"):
            continue
        runner_cmd = m.get("runner_command", "ablator run")
        log = os.path.join(ldir, f"runner_{name}.log")
        remote = (f"if pgrep -f '[a]blator run' >/dev/null; then "
                  f"echo '[start] runner already running on {name}'; else "
                  f"mkdir -p {shlex.quote(ldir)} && "
                  f"setsid nohup {runner_cmd} </dev/null > {shlex.quote(log)} 2>&1 & "
                  f"echo '[start] launched runner on {name}'; fi")
        r = subprocess.run(["ssh", m["ssh"], remote])
        if r.returncode != 0:
            print(f"[start] WARNING: could not reach {m['ssh']} — "
                  f"runner on {name} not started")
    print(f"[start] done. Watch: tail -f {ldir}/runner_*.log")
