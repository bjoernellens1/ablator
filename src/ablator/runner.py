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
from . import error as errormod
from . import health as healthmod
from . import resources
from .queue import Queue, pause_flag_path, write_pause_flag

IDLE_POLL_S = 30
BUSY_POLL_S = 30
HEALTH_POLL_S = 60
STALL_WARN_S = 600  # loudly log if one loop iteration took longer than this

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
              control=None,
              preempt=None) -> str | None:
    """Watch a running job until its process exits or intervention is needed.

    Returns None when the process exited on its own (caller reads
    returncode), or an override status: 'failed_no_retry' (manual stop),
    'cancelled' (manual skip), 'requeue' (manual requeue), 'preempted'
    (lane-1 job yielding to a pending lane-3 job), 'failed'
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
    preempt = preempt or (lambda: q.preemption_due(job, cfgmod.machine_name(cfg))
                          if q is not None else False)
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
        if preempt():
            print(f"[ablator] preempting lane-1 job {job['id']} for a pending "
                  f"lane-3 job", flush=True)
            kill()
            return "preempted"
        if not alive:
            continue  # exited during the sleep; loop back to reap returncode
        if h["state"] in ("hung", "crashed"):
            print(f"[ablator] {job['id']} unhealthy ({h['state']}, "
                  f"log_age={h.get('log_age_s')}s) — killing", flush=True)
            kill()
            return "failed"


def write_heartbeat(cfg: dict, machine: str, state: str) -> None:
    """One line per loop iteration so 'stuck vs sleeping' is diagnosable at a
    glance: <queue dir>/heartbeat_<machine>.txt. Never raises."""
    try:
        path = os.path.join(os.path.dirname(cfgmod.queue_path(cfg)),
                            f"heartbeat_{machine}.txt")
        with open(path, "w") as f:
            f.write(f"{machine} {time.strftime('%Y-%m-%dT%H:%M:%S')} "
                    f"epoch={time.time():.0f} state={state}\n")
    except Exception as e:
        print(f"[ablator] heartbeat write failed: {e!r}", flush=True)


def _disk_free_bytes(path: str) -> int | None:
    try:
        import shutil
        d = path if os.path.isdir(path) else os.path.dirname(path) or "."
        return shutil.disk_usage(d).free
    except OSError:
        return None


def _docker_storage_free_bytes() -> int | None:
    for candidate in ("/var/lib/docker", "/var/lib/containers"):
        if os.path.isdir(candidate):
            return _disk_free_bytes(candidate)
    return None


def machine_context_snapshot(job: dict, base_dir: str) -> dict:
    """Best-effort, read-only machine signals for error.classify_failure()."""
    mp = healthmod.resolve_model_path(job.get("model_path", ""), base_dir)
    ctx: dict = {
        "disk_free_bytes": _disk_free_bytes(mp),
        "docker_storage_free_bytes": _docker_storage_free_bytes(),
    }
    for dmesg_path in ("/var/log/messages",):
        try:
            from . import progress as progmod
            ctx["dmesg_tail"] = progmod.read_tail(dmesg_path, 4096)
            break
        except OSError:
            continue
    return ctx


def _job_log_tail(cfg: dict, job: dict) -> str:
    from . import progress as progmod
    log = os.path.join(cfgmod.log_dir(cfg), f"{job['id']}.log")
    return progmod.read_tail(log, healthmod.CRASH_TAIL_BYTES)


def classify_and_record(cfg: dict, job: dict, exit_code: int | None,
                        base_dir: str, q: Queue | None = None) -> dict:
    """Classify a failed job's log/exit-code and persist error_* fields."""
    tail = _job_log_tail(cfg, job)
    ctx = machine_context_snapshot(job, base_dir)
    patterns = errormod.patterns_from_config(cfg)
    result = errormod.classify_failure(job, tail, exit_code, ctx, patterns=patterns)
    if q is not None:
        q.update(job["id"],
                error_category=result["category"],
                error_evidence=result["evidence_snippet"],
                error_confidence=result["confidence"],
                suggested_action=result["suggested_action"])
    job["error_category"] = result["category"]
    job["error_evidence"] = result["evidence_snippet"]
    job["error_confidence"] = result["confidence"]
    job["suggested_action"] = result["suggested_action"]
    return result


def handle_failure(cfg: dict, job: dict, exit_code: int | None, machine: str,
                   base_dir: str, q: Queue) -> str:
    """Classify a failure and decide the job's terminal/backoff disposition.

    Returns one of: "paused_disk_full", "quarantined", "pending" (requeued
    with not_before/needs_review bookkeeping already applied via q.update),
    or "retry" (caller should retry once, existing uniform behavior for
    'unknown').
    """
    result = classify_and_record(cfg, job, exit_code, base_dir, q)
    category = result["category"]
    action = result["suggested_action"]

    if action == "pause_queue_alert":
        path = write_pause_flag(cfgmod.queue_path(cfg), machine, category,
                                result["evidence_snippet"])
        print(f"[ablator] PAUSING {machine} — {category}: "
              f"{result['evidence_snippet']!r} (flag: {path})", flush=True)
        q.update(job["id"], status="paused_disk_full")
        return "paused_disk_full"

    if action == "skip_permanently_this_machine":
        print(f"[ablator] {job['id']}: {category} — quarantining", flush=True)
        return "quarantined"

    if action == "requeue_backoff_5min":
        q.update(job["id"], status="pending", health=None,
                claimed_by=None, claimed_at=None,
                not_before=time.time() + 5 * 60)
        return "pending"

    if action == "requeue_backoff_2min":
        q.update(job["id"], status="pending", health=None,
                claimed_by=None, claimed_at=None,
                not_before=time.time() + 2 * 60)
        return "pending"

    if action == "requeue_once_needs_review":
        if job.get("oom_killed_once"):
            print(f"[ablator] {job['id']}: second oom_killed — quarantining", flush=True)
            return "quarantined"
        q.update(job["id"], status="pending", health=None,
                claimed_by=None, claimed_at=None,
                oom_killed_once=True, needs_review=True)
        return "pending"

    if action in ("quarantine_no_retry", "quarantine_code_fix_needed"):
        return "quarantined"

    # unknown -> retry_once_then_quarantine: preserve existing uniform
    # retry-then-quarantine behavior in run_loop().
    return "retry"


def run_job(cfg: dict, job: dict, machine: str,
           q: Queue | None = None) -> tuple[str, int | None]:
    """Execute one job; returns (status, exit_code). Logs to <log_dir>/<id>.log.

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
        return "failed", None
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
        rc = proc.returncode
        if override is not None:
            return override, rc
        return ("done" if rc == 0 else "failed"), rc
    except Exception as e:
        print(f"[ablator] {job['id']} crashed: {e}", flush=True)
        return "failed", None


def _job_base_dir(cfg: dict, job: dict, machine: str) -> str:
    try:
        tcfg = cfgmod.type_cfg(cfg, job.get("type", ""), machine)
    except KeyError:
        tcfg = {}
    return tcfg.get("cwd") or os.getcwd()


def run_loop(cfg: dict, once: bool = False) -> None:
    machine = cfgmod.machine_name(cfg)
    q = Queue(cfgmod.queue_path(cfg))
    print(f"[ablator] runner on {machine} watching {q.path}", flush=True)
    last_tick = time.monotonic()
    while True:
        # Watchdog: if the previous iteration (probes + sleeps, NOT a job
        # run) took absurdly long, say so loudly — this is the 'stuck loop'
        # tell.
        now = time.monotonic()
        if now - last_tick > STALL_WARN_S:
            print(f"[ablator] WARNING: loop iteration took {now - last_tick:.0f}s "
                  f"(> {STALL_WARN_S}s) — a probe or lock likely hung",
                  flush=True)
        last_tick = now
        try:
            if resources.machine_busy(cfg, machine):
                write_heartbeat(cfg, machine, "busy-wait")
                if once:
                    return
                time.sleep(BUSY_POLL_S)
                continue
            write_heartbeat(cfg, machine, "idle")
            job = q.claim_next(machine, can_run=make_can_run(cfg, machine))
            if job is None:
                if once:
                    return
                time.sleep(IDLE_POLL_S)
                continue
            write_heartbeat(cfg, machine, f"running:{job['id']}")
            base_dir = _job_base_dir(cfg, job, machine)
            status, exit_code = run_job(cfg, job, machine, q)
            if status == "failed":
                disposition = handle_failure(cfg, job, exit_code, machine, base_dir, q)
                if disposition == "retry":
                    # unknown category: preserve existing uniform
                    # retry-once-then-quarantine behavior.
                    if not job.get("retried"):
                        job["retried"] = True
                        q.update(job["id"], retried=True)
                        print(f"[ablator] retrying {job['id']} once", flush=True)
                        status, exit_code = run_job(cfg, job, machine, q)
                        if status == "failed":
                            classify_and_record(cfg, job, exit_code, base_dir, q)
                            status = "quarantined"
                    else:
                        status = "quarantined"
                elif disposition == "pending":
                    status = "pending"  # already persisted by handle_failure
                else:
                    status = disposition  # paused_disk_full | quarantined
            if status == "preempted":
                # lane-1 job yielded to a lane-3 job: back to pending with the
                # anti-thrash bookkeeping; the loop claims the lane-3 job next
                q.update(job["id"], status="pending", health=None,
                         claimed_by=None, claimed_at=None,
                         preempt_count=int(job.get("preempt_count", 0)) + 1,
                         last_preempt_at=time.time())
            elif status == "requeue":
                # manual requeue: back to pending, clear claim/health so any
                # runner (including this one) can retake it fresh
                q.update(job["id"], status="pending", health=None,
                         claimed_by=None, claimed_at=None)
            elif status in ("pending", "paused_disk_full"):
                pass  # handle_failure() already persisted this status
            else:
                if status == "failed_no_retry":  # manual stop: never retried
                    status = "failed"
                q.finish(job["id"], status)
            write_heartbeat(cfg, machine, f"finished:{job['id']}:{status}")
            last_tick = time.monotonic()  # job runs are legitimately long
            print(f"[ablator] {job['id']} -> {status}", flush=True)
            if once:
                return
        except Exception as e:
            import traceback
            print(f"[ablator] loop iteration crashed: {e!r}\n"
                  f"{traceback.format_exc()}", flush=True)
            write_heartbeat(cfg, machine, "error")
            time.sleep(60)


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
