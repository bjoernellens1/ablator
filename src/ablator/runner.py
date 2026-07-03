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
import signal
import subprocess
import time

from . import config as cfgmod
from . import health as healthmod
from . import resources
from .queue import Queue

IDLE_POLL_S = 30
BUSY_POLL_S = 30
HEALTH_POLL_S = 60

# Manual control action -> status returned by supervise(). "requeue" is
# turned back into a fresh pending job by run_loop.
CONTROL_STATUS = {"stop": "failed_no_retry", "skip": "cancelled",
                  "requeue": "requeue"}


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


def control_path(cfg: dict, job_id: str) -> str:
    return os.path.join(os.path.dirname(cfgmod.queue_path(cfg)),
                        f"control_{job_id}")


def read_control(cfg: dict, job_id: str) -> str | None:
    """Read and consume a manual control file: 'stop'|'skip'|'requeue'."""
    path = control_path(cfg, job_id)
    try:
        with open(path) as f:
            action = f.read().strip().lower()
    except OSError:
        return None
    try:
        os.remove(path)
    except OSError:
        pass
    return action if action in CONTROL_STATUS else None


def kill_job(proc: subprocess.Popen, job: dict) -> None:
    """Kill the job's process group (SIGTERM, then SIGKILL)."""
    print(f"[ablator] killing {job['id']} (pid {proc.pid})", flush=True)
    for sig, wait_s in ((signal.SIGTERM, 30), (signal.SIGKILL, 10)):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(timeout=wait_s)
            return
        except subprocess.TimeoutExpired:
            continue


def supervise(cfg: dict, job: dict, proc: subprocess.Popen, base_dir: str,
              q: Queue | None = None,
              poll_s: float = HEALTH_POLL_S,
              sleep=None,
              health_fn=None,
              kill=None,
              record=None,
              control=None) -> str | None:
    """Watch a running job until its process exits or intervention is needed.

    Returns None when the process exited on its own (caller reads
    returncode), or an override status: 'failed_no_retry' (manual stop),
    'cancelled' (manual skip), 'requeue' (manual requeue), 'failed'
    (hung/crashed — normal retry->quarantine path applies).

    Health comes ONLY from the run's own artifacts (health module); the run
    itself stays fully standalone and never talks to the runner.
    """
    if sleep is None:
        def sleep(s):  # returns as soon as the process exits
            try:
                proc.wait(timeout=s)
            except subprocess.TimeoutExpired:
                pass
    qcfg = cfg.get("queue", {})
    health_fn = health_fn or (lambda alive: healthmod.job_health(
        job, base_dir, qcfg, process_alive=alive))
    kill = kill or (lambda: kill_job(proc, job))
    record = record or (lambda h: q and q.update(job["id"], health=h))
    control = control or (lambda: read_control(cfg, job["id"]))
    while True:
        if proc.poll() is not None:
            record(health_fn(None))  # final snapshot; exit code judges result
            return None
        sleep(poll_s)
        alive = proc.poll() is None
        h = health_fn(alive)
        record(h)
        action = control()
        if action:
            print(f"[ablator] control '{action}' for {job['id']}", flush=True)
            kill()
            return CONTROL_STATUS[action]
        if not alive:
            continue  # exited during the sleep; loop back to reap returncode
        if h["state"] in ("hung", "crashed"):
            print(f"[ablator] {job['id']} unhealthy ({h['state']}, "
                  f"log_age={h.get('log_age_s')}s) — killing", flush=True)
            kill()
            return "failed"


def run_job(cfg: dict, job: dict, machine: str, q: Queue | None = None) -> str:
    """Execute one job; returns its status. Logs to <log_dir>/<id>.log.

    While the subprocess runs, a supervision loop mirrors artifact-derived
    health into the queue record, honors control files, and kills
    hung/crashed jobs.
    """
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
            proc = subprocess.Popen(argv, env=env, cwd=cwd,
                                    stdout=lf, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            override = supervise(cfg, job, proc, cwd or os.getcwd(), q)
        if override is not None:
            return override
        return "done" if proc.returncode == 0 else "failed"
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
        status = run_job(cfg, job, machine, q)
        # one retry on failure, then quarantine
        if status == "failed" and not job.get("retried"):
            q.update(job["id"], retried=True)
            print(f"[ablator] retrying {job['id']} once", flush=True)
            job["retried"] = True
            status = run_job(cfg, job, machine, q)
            if status == "failed":
                status = "quarantined"
        if status == "requeue":
            # manual requeue: back to pending, clear claim/health so any
            # runner (including this one) can retake it fresh
            q.update(job["id"], status="pending", health=None,
                     claimed_by=None, claimed_at=None)
        else:
            if status == "failed_no_retry":  # manual stop: never retried
                status = "failed"
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
