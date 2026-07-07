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

import json
import os
import re
import shlex
import signal
import subprocess
import threading
import time

from . import config as cfgmod
from . import error as errormod
from . import health as healthmod
from . import provenance as provmod
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


def _infer_tum_sequence(scene: str, extra_args: str) -> str:
    """Append --tum_sequence freiburgN when the job's HOST scene path names a
    TUM Freiburg sequence and extra_args doesn't already set it.

    Command templates mount `scene` at a generic in-container path (e.g.
    /data/scene), which defeats scene/readers/tum.py's path-basename
    inference of Freiburg1/2/3 intrinsics inside the container — it only
    ever sees the generic basename, never matches, and SILENTLY falls back
    to wrong Freiburg1 intrinsics for the whole run (no crash, just a
    warning line in train.log). Confirmed live: fr3par_hybrid_ctrl/gate
    both trained and evaluated an entire freiburg3 run against Freiburg1
    intrinsics this way, a ~6-8dB PSNR cost that compounded over training.
    `scene` is still the HOST path here (pre-mount), so recover the real
    sequence from it before rendering the command.
    """
    if "--tum_sequence" in (extra_args or ""):
        return extra_args
    name = os.path.basename(os.path.normpath(scene or "")).lower()
    for seq in ("freiburg1", "freiburg2", "freiburg3"):
        # Only the unambiguous "freiburgN" substring — a short "frN" form
        # false-triggers on unrelated scene names (e.g. a generic test
        # fixture path like "/data/fr3" is not a TUM scene at all).
        if seq in name:
            return f"{extra_args} --tum_sequence {seq}".strip()
    return extra_args


_CHKPT_RE = re.compile(r"chkpnt(\d+)\.pth$")


def find_latest_checkpoint(model_path: str, base_dir: str) -> tuple[str, int] | None:
    """Return (path, iteration) of the highest-iteration chkpntN.pth under a
    job's model_path, or None if none exists.

    Used both to decide whether a failed job is resumable (handle_failure)
    and to thread ``--start_checkpoint`` into a requeued job's command
    (_job_vars). Splatograph writes checkpoints via a temp-file + os.replace
    atomic rename (train.py / train_streaming.py), so any chkpntN.pth that
    exists here is guaranteed to be a complete, loadable file — never a
    truncated one from a SIGKILL mid-write.
    """
    mp = healthmod.resolve_model_path(model_path, base_dir)
    if not mp or not os.path.isdir(mp):
        return None
    best: tuple[str, int] | None = None
    try:
        for fn in os.listdir(mp):
            m = _CHKPT_RE.match(fn)
            if m:
                it = int(m.group(1))
                if best is None or it > best[1]:
                    best = (os.path.join(mp, fn), it)
    except OSError:
        return None
    return best


def _git_state_path(cfg: dict, machine: str) -> str:
    """Shared-storage status file one runner writes and another reads, so
    the r9700-vs-main drift check works across two SEPARATE ablator
    processes (each machine runs its own `ablator run`) without any
    SSH-specific plumbing: queue.jsonl (and therefore log_dir) already
    lives on shared NFS (/mnt/cps_scratch1_tmp), so this file is visible
    to both machines the moment either one writes it."""
    return os.path.join(cfgmod.log_dir(cfg), f"git_state_{machine}.json")


def write_git_state_file(cfg: dict, machine: str, state: dict) -> None:
    try:
        with open(_git_state_path(cfg, machine), "w") as f:
            json.dump({**state, "written_at": time.time()}, f)
    except OSError as e:
        print(f"[ablator] write_git_state_file({machine}) failed: {e!r}", flush=True)


def read_git_state_file(cfg: dict, machine: str) -> dict | None:
    try:
        with open(_git_state_path(cfg, machine)) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def capture_and_record_provenance(cfg: dict, job: dict, machine: str,
                                  cwd: str, q: Queue | None) -> dict:
    """Bare-metal code-provenance capture at job-dispatch time.

    Runs identically wherever `ablator run` actually executes (main or
    r9700's own separate process) — each runner captures its OWN local git
    state at the job type's configured `cwd`, since bare-metal jobs execute
    via a live bind-mount of that SAME checkout. Recorded into (a) the
    job's ledger entry, so `queue.jsonl` durably answers "what code ran
    this job", and (b) a cross-machine status file other runners can read
    for drift comparison (see check_r9700_drift below).
    """
    state = provmod.capture_local_git_state(cwd)
    if q is not None:
        q.update(job["id"], provenance=state)
    write_git_state_file(cfg, machine, state)
    return state


def check_r9700_drift(cfg: dict, job: dict, machine: str, state: dict,
                      q: Queue | None) -> None:
    """Proactive, loud (WARN-not-refuse) check: is r9700's actual git
    commit the same as main's most-recently-observed commit?

    Only meaningful when THIS dispatch is happening on r9700 (each machine
    runs its own ablator process — main never directly dispatches a
    bare-metal job onto r9700 over SSH, so there is no single call site
    where "before dispatching to r9700" can compare live processes
    directly). Instead: r9700's own runner, right before running one of
    its own jobs, reads the shared git_state_main.json file main's runner
    last wrote and compares. This is necessarily best-effort (stale if
    main hasn't run a bare-metal job recently) but requires zero
    SSH-specific plumbing and needs no reachability from r9700 to main.

    WARN, never refuse: a user may deliberately want different code on
    r9700 (e.g. testing a branch there only) — a loud warning in both the
    runner log and the job's ledger entry is enough to make the drift
    impossible to miss without blocking a job the user may have wanted to
    run exactly as-is.
    """
    if machine != "r9700":
        return
    main_state = read_git_state_file(cfg, "main")
    if not main_state or not main_state.get("commit") or not state.get("commit"):
        return
    if main_state["commit"] == state["commit"]:
        return
    warning = (f"CODE PROVENANCE DRIFT: r9700 is executing job {job['id']!r} at "
              f"commit {state['commit'][:12]} (branch {state.get('branch')}) but "
              f"main's checkout was last observed at commit "
              f"{main_state['commit'][:12]} — these two machines' checkouts have "
              f"diverged. If intentional (e.g. testing a branch on r9700 only), "
              f"ignore; otherwise sync the checkouts before trusting cross-machine "
              f"comparisons.")
    print(f"[ablator] {warning}", flush=True)
    if q is not None:
        q.update(job["id"], drift_warning=warning,
                main_commit_at_check=main_state["commit"])


def _dispatch_host_commit(cfg: dict, job: dict) -> str | None:
    """The current git commit of the checkout on THIS host (whichever host
    is running `ablator run` right now, main or r9700 acting as a k8s
    dispatcher) — used as the reference point for k8s image-drift checks.
    Uses the base (non-machine-overridden) type cwd, i.e. the real
    bare-metal repo path, never a container path."""
    cwd = cfg.get("types", {}).get(job.get("type", ""), {}).get("cwd") or os.getcwd()
    state = provmod.capture_local_git_state(cwd)
    return state.get("commit")


def _job_vars(job: dict, machine: str) -> dict:
    scene = job.get("scene", "")
    extra_args = _infer_tum_sequence(scene, job.get("extra_args", ""))
    resume_checkpoint = job.get("resume_checkpoint")
    if resume_checkpoint and "--start_checkpoint" not in extra_args:
        # Threaded in by handle_failure() when a preempted/crashed job has a
        # resumable checkpoint on durable (NFS-backed) storage — see there.
        extra_args = f"{extra_args} --start_checkpoint {resume_checkpoint}".strip()
    return {
        "scene": scene,
        "model_path": job.get("model_path", ""),
        "extra_args": extra_args,
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


_CONTAINER_RUNTIMES = ("podman", "docker")


def container_name_from_argv(argv: list[str]) -> str | None:
    """Extract the `--name X` / `--name=X` value from a rendered command,
    if the command is a podman/docker `run` invocation. Returns None for
    non-container commands or container commands with no explicit name
    (in which case guaranteed-teardown-by-name is not possible and only
    the process-group signal path applies)."""
    if not argv or argv[0] not in _CONTAINER_RUNTIMES:
        return None
    if "run" not in argv[:2]:
        return None
    for i, tok in enumerate(argv):
        if tok == "--name" and i + 1 < len(argv):
            return argv[i + 1]
        if tok.startswith("--name="):
            return tok.split("=", 1)[1]
    return None


def force_remove_container(runtime: str, name: str) -> None:
    """Best-effort, idempotent `stop` then `rm -f` by name. Safe to call
    whether or not the container exists — never raises. This is the
    guaranteed teardown path: unlike signalling the local `podman run`
    client process (which only forwards SIGTERM, and does nothing on
    SIGKILL since an uncatchable signal can't be proxied), this acts on
    the container itself regardless of whether the local client process
    responded to signals or was already reaped."""
    for args, timeout in (([runtime, "stop", "-t", "10", name], 20),
                          ([runtime, "rm", "-f", name], 15)):
        try:
            subprocess.run(args, capture_output=True, timeout=timeout, check=False)
        except (OSError, subprocess.TimeoutExpired) as e:
            print(f"[ablator] force_remove_container {args}: {e!r}", flush=True)


def kill_job(proc: subprocess.Popen, job: dict, argv: list[str] | None = None) -> None:
    """Kill the job's process group (SIGTERM, then SIGKILL), AND — if this
    is a podman/docker `run` command with a `--name` — forcibly stop/rm the
    container by name regardless of whether the signals reaped the local
    client process.

    Why both: `podman run <fg>` proxies SIGTERM to the container, but a
    SIGKILL to the *local client* process is uncatchable and therefore
    cannot be proxied to anything — it just kills our view into the run
    while the container (and the training process inside it) keeps going,
    orphaned. This is exactly what happened with job spp39f3_ctrl on
    2026-07-06: OOM crashed inside the container, our health check called
    kill_job(), SIGTERM didn't reap the hung post-OOM process within 30s,
    SIGKILL reaped the *client* fine (so ablator's ledger correctly moved
    on to 'quarantined') but the actual container ran for another ~13h
    leaking GPU GTT memory (114GB) with nothing left tracking it.
    """
    print(f"[ablator] killing {job['id']} (pid {proc.pid})", flush=True)
    name = container_name_from_argv(argv) if argv else None
    if name:
        print(f"[ablator] {job['id']}: also force-tearing-down container "
              f"'{name}' by name (guaranteed, independent of client signals)",
              flush=True)
        force_remove_container(argv[0], name)
    for sig, wait_s in ((signal.SIGTERM, 30), (signal.SIGKILL, 10)):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(timeout=wait_s)
            break
        except subprocess.TimeoutExpired:
            continue
    if name:
        # Belt-and-suspenders: repeat after signal escalation in case the
        # container was (re)created or missed by the first pass (e.g. name
        # not yet visible to `podman ps` at the time of the first call).
        force_remove_container(argv[0], name)


def supervise(cfg: dict, job: dict, proc: subprocess.Popen, base_dir: str,
              q: Queue | None = None,
              poll_s: float = HEALTH_POLL_S,
              sleep=None,
              health_fn=None,
              kill=None,
              record=None,
              control=None,
              preempt=None,
              argv: list[str] | None = None) -> str | None:
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
    kill = kill or (lambda: kill_job(proc, job, argv))
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

    Preemption-aware resume: rather than trying to definitively distinguish
    "was this SIGTERM'd by KAI Scheduler preemption" from "did this crash",
    which the k8s poll loop cannot reliably tell apart (see
    build_k8s_job_manifest/_poll_k8s_job — no pod-eviction-reason inspection
    today), we use a strictly simpler and more robust rule: if a resumable
    checkpoint exists AND it represents real progress beyond the last resume
    attempt, resume from it — regardless of why the job died. Either way
    (preemption or a transient crash after checkpointing) resuming is the
    right action; re-running from scratch is not. The progress-guard (only
    resume if checkpoint iteration advanced past job["last_resumed_iter"])
    prevents infinite resume->immediate-crash->resume loops for jobs with a
    genuine, reproducible bug that happens to occur just after a checkpoint.
    """
    ckpt = find_latest_checkpoint(job.get("model_path", ""), base_dir)
    if ckpt is not None:
        ckpt_path, ckpt_iter = ckpt
        last_resumed = job.get("last_resumed_iter")
        if last_resumed is None or ckpt_iter > last_resumed:
            print(f"[ablator] {job['id']}: resumable checkpoint at iter={ckpt_iter} "
                  f"(prior resume point: {last_resumed}) — requeuing with "
                  f"--start_checkpoint {ckpt_path} instead of restarting from scratch",
                  flush=True)
            q.update(job["id"], status="pending", health=None,
                    claimed_by=None, claimed_at=None,
                    last_resumed_iter=ckpt_iter, resume_checkpoint=ckpt_path,
                    error_category="resumable_from_checkpoint",
                    suggested_action="resume_from_checkpoint")
            job["last_resumed_iter"] = ckpt_iter
            job["resume_checkpoint"] = ckpt_path
            return "pending"
        print(f"[ablator] {job['id']}: checkpoint exists (iter={ckpt_iter}) but did not "
              f"advance past the last resume point ({last_resumed}) — treating as a real, "
              "reproducible failure rather than looping resume attempts.", flush=True)

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


def _k8s_job_name(job_id: str) -> str:
    """Kubernetes object names must be lowercase RFC-1123 (alnum + '-')."""
    name = re.sub(r"[^a-z0-9-]", "-", job_id.lower()).strip("-")
    return f"ablator-{name}"[:63].rstrip("-")


def _kubectl(args: list[str], input_text: str | None = None,
             timeout: float | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["kubectl", *args], input=input_text, text=True,
                          capture_output=True, timeout=timeout)


def build_k8s_job_manifest(mcfg: dict, job: dict, argv: list[str],
                          cwd: str | None) -> dict:
    """Build the Job manifest dict for one job on a k8s-backend machine.

    Mounts: `pvc_persistent` (read-only, subPath'd to the job's real dataset
    directory so the in-container "/data/scene" path used by the rendered
    command template resolves correctly) and `pvc_scratch` (read-write, at
    the SAME absolute path bare-metal machines use, /mnt/cps_scratch1_tmp --
    so the command's own `ln -sfn .../output output/scratch` step, and
    ablator's own status/collect reading the shared queue.jsonl under that
    same tree, both work identically to a bare-metal job).
    """
    name = _k8s_job_name(job["id"])
    scene = job.get("scene", "")
    persistent_root = "/mnt/cps_persistent1_shared"
    sub_path = scene[len(persistent_root):].lstrip("/") if scene.startswith(persistent_root) else ""
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": name,
            "namespace": mcfg["namespace"],
            "labels": {"kai.scheduler/queue": mcfg["kai_queue"], "app": "ablator-job"},
        },
        "spec": {
            "backoffLimit": 0,
            "ttlSecondsAfterFinished": 86400,
            # Hard safety net independent of ablator's own control-file polling
            # loop -- covers the coordinator process itself dying/being killed
            # mid-job, which would otherwise leave an orphaned Job running
            # forever on the cluster. 24h is generous for a single replay run
            # (image pulls included) but still bounds worst-case orphan cost.
            "activeDeadlineSeconds": mcfg.get("active_deadline_s", 86400),
            "template": {
                "metadata": {"labels": {"kai.scheduler/queue": mcfg["kai_queue"],
                                        "app": "ablator-job"}},
                "spec": {
                    "schedulerName": "kai-scheduler",
                    "priorityClassName": mcfg["priority_class"],
                    # KAI Scheduler preempts kai-batch-low (lowest priority) pods
                    # for any higher-priority queue at any time via SIGTERM, then
                    # SIGKILL after this many seconds. splatograph's SIGTERM
                    # handler (train.py / train_streaming.py) writes an emergency
                    # checkpoint SYNCHRONOUSLY before exiting -- anchor this to
                    # AsyncSaveWorker's own 120s shutdown-join timeout plus margin
                    # for the synchronous capture()+torch.save() itself, rather
                    # than k8s's 30s default (too short: SIGKILL would truncate
                    # the save via os.replace's rename never completing, losing
                    # the resume point).
                    "terminationGracePeriodSeconds":
                        mcfg.get("termination_grace_period_s", 150),
                    "restartPolicy": "Never",
                    "imagePullSecrets": [{"name": mcfg["image_pull_secret"]}],
                    "containers": [{
                        "name": "trainer",
                        "image": mcfg["image"],
                        # Non-":latest" tags default to imagePullPolicy=IfNotPresent,
                        # which silently reuses a stale cached layer on whatever node
                        # a Job lands on after a fresh push -- found live: a rebuilt
                        # cuda-dev image (fixing a real ModuleNotFoundError) was
                        # ignored by a node that had cached the broken prior build.
                        "imagePullPolicy": "Always",
                        "workingDir": cwd or "/workspace/splatograph",
                        "command": argv,
                        "resources": {
                            "requests": {"cpu": "4", "memory": "16Gi"},
                            "limits": {"cpu": "8", "memory": "32Gi",
                                      "nvidia.com/gpu": str(mcfg.get("gpu_count", 1))},
                        },
                        "volumeMounts": [
                            {"name": "dataset", "mountPath": "/data/scene",
                             "subPath": sub_path, "readOnly": True},
                            {"name": "scratch", "mountPath": "/mnt/cps_scratch1_tmp"},
                        ],
                    }],
                    "volumes": [
                        {"name": "dataset",
                         "persistentVolumeClaim": {"claimName": mcfg["pvc_persistent"],
                                                   "readOnly": True}},
                        {"name": "scratch",
                         "persistentVolumeClaim": {"claimName": mcfg["pvc_scratch"]}},
                    ],
                },
            },
        },
    }


def _poll_k8s_job(cfg: dict, job: dict, machine: str, mcfg: dict, tcfg: dict,
                  name: str, ns: str, log_path: str,
                  append: bool = False) -> tuple[str, int | None]:
    """Poll an existing (already-submitted) k8s Job to completion.

    Shared by the initial-submission path (`run_job_k8s`) and the
    restart-recovery re-attach path (`reconcile_stale_running`): both cases
    reduce to "there is a k8s Job `ns/name` already out there, watch it
    until it finishes". `append=True` is used for re-attach, since the
    Job (and its log history) predates this process and truncating the
    existing log file would destroy that history.
    """
    log_proc = None
    try:
        with open(log_path, "a" if append else "w") as lf:
            if append:
                lf.write(f"# {time.strftime('%Y-%m-%dT%H:%M:%S')} {job['id']} "
                         f"re-attached to existing k8s Job {ns}/{name} "
                         "(runner restart)\n")
            lf.flush()
            missing_polls = 0
            while True:
                if read_control(cfg, job["id"]) is not None:
                    _kubectl(["delete", "job", name, "-n", ns, "--ignore-not-found",
                             "--wait=false"])
                    return "requeue", None
                status = _kubectl(["get", "job", name, "-n", ns, "-o", "json"],
                                  timeout=30)
                if status.returncode != 0:
                    # A transient API hiccup looks identical to "the Job was
                    # deleted out from under us" (e.g. a manual `kubectl
                    # delete`, or the namespace being cleaned up) -- both
                    # give a nonzero `kubectl get` exit. Without a giveup
                    # count this loop spun forever once the Job was gone,
                    # leaving the job stuck "running" in the ledger with no
                    # pod/Job to show for it (found live, 2026-07-06).
                    missing_polls += 1
                    if missing_polls >= 3:
                        print(f"[ablator] {job['id']} k8s Job {ns}/{name} "
                              f"no longer found after {missing_polls} polls "
                              "-- treating as failed", flush=True)
                        rc = 1
                        break
                    time.sleep(HEALTH_POLL_S)
                    continue
                missing_polls = 0
                st = json.loads(status.stdout).get("status", {})
                if log_proc is None:
                    pods = _kubectl(["get", "pods", "-n", ns, "-l",
                                     f"job-name={name}", "-o",
                                     "jsonpath={.items[0].metadata.name}"])
                    if pods.returncode == 0 and pods.stdout.strip():
                        log_proc = subprocess.Popen(
                            ["kubectl", "logs", "-f", pods.stdout.strip(), "-n", ns],
                            stdout=lf, stderr=subprocess.STDOUT)
                if st.get("succeeded", 0) >= 1:
                    rc = 0
                    break
                if st.get("failed", 0) >= 1:
                    rc = 1
                    break
                time.sleep(HEALTH_POLL_S)
        if log_proc is not None:
            try:
                log_proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                log_proc.kill()
    finally:
        _kubectl(["delete", "job", name, "-n", ns, "--ignore-not-found", "--wait=false"])

    if rc == 0 and _require_result_artifact(cfg, tcfg):
        # `cwd` above is the CONTAINER workingDir (/workspace/splatograph),
        # meaningless on the host running this coordinator process. Health
        # checks always run on whichever host executes `ablator run`/
        # `ablator status` -- resolve model_path against THAT host's repo
        # checkout instead, same as bare-metal jobs do (the base, non-
        # machine-overridden [types.<t>].cwd, since output/scratch there is
        # the symlink into the same NFS tree the k8s pod's scratch PVC
        # mounts, so both paths land on the identical files).
        host_base_dir = _job_base_dir(cfg, job, machine)
        h = healthmod.job_health(job, host_base_dir, cfg.get("queue", {}),
                                 process_alive=False)
        if h["state"] != "done":
            print(f"[ablator] {job['id']} k8s Job succeeded but no completion "
                  f"artifact found (state={h['state']!r}) — treating as failed",
                  flush=True)
            return "failed", rc
    return ("done" if rc == 0 else "failed"), rc


def run_job_k8s(cfg: dict, job: dict, machine: str, mcfg: dict,
               q: Queue | None = None) -> tuple[str, int | None]:
    """Execute one job as a Kubernetes Job on a k8s-backend machine.

    Submits via `kubectl apply` (stdlib-only, no k8s Python client
    dependency, matching this project's zero-dependency philosophy), then
    delegates to `_poll_k8s_job` for the completion wait, log tailing, and
    Job teardown. Honors the same control-file protocol (stop/skip/requeue)
    as the subprocess path by deleting the Job when one is found.
    """
    log_path = os.path.join(cfgmod.log_dir(cfg), f"{job['id']}.log")
    try:
        tcfg = cfgmod.type_cfg(cfg, job.get("type", ""), machine)
        argv, _env, cwd = render_command(tcfg, job, machine)
    except (KeyError, TemplateError) as e:
        print(f"[ablator] {job['id']} unrunnable: {e}", flush=True)
        return "failed", None

    manifest = build_k8s_job_manifest(mcfg, job, argv, cwd)
    name = manifest["metadata"]["name"]
    ns = mcfg["namespace"]
    print(f"[ablator] running {job['id']} -> {job.get('model_path')} "
          f"(k8s Job {ns}/{name}, log {log_path})", flush=True)

    local_commit = _dispatch_host_commit(cfg, job)
    image_prov = provmod.check_image_drift(mcfg["image"], local_commit)
    if image_prov.get("warning"):
        print(f"[ablator] {image_prov['warning']}", flush=True)
    if q is not None:
        q.update(job["id"], image_provenance=image_prov,
                dispatch_host_commit=local_commit)

    apply = _kubectl(["apply", "-f", "-"], input_text=json.dumps(manifest))
    if apply.returncode != 0:
        print(f"[ablator] {job['id']} k8s apply failed: {apply.stderr}", flush=True)
        return "failed", None

    with open(log_path, "w") as lf:
        lf.write(f"# {time.strftime('%Y-%m-%dT%H:%M:%S')} {job['id']} "
                 f"(k8s Job {ns}/{name})\n# {shlex.join(argv)}\n")
        lf.write(provmod.format_banner("k8s", image_prov) + "\n")
    return _poll_k8s_job(cfg, job, machine, mcfg, tcfg, name, ns, log_path,
                        append=True)


def run_job(cfg: dict, job: dict, machine: str,
           q: Queue | None = None) -> tuple[str, int | None]:
    """Execute one job; returns (status, exit_code). Logs to <log_dir>/<id>.log.

    While the subprocess runs, a supervision loop mirrors artifact-derived
    health into the queue record, honors control files, and kills
    hung/crashed jobs.
    """
    mcfg = cfgmod.machine_cfg(cfg, machine)
    if mcfg.get("backend") == "k8s":
        return run_job_k8s(cfg, job, machine, mcfg, q)
    log_path = os.path.join(cfgmod.log_dir(cfg), f"{job['id']}.log")
    try:
        tcfg = cfgmod.type_cfg(cfg, job.get("type", ""), machine)
        argv, env, cwd = render_command(tcfg, job, machine)
    except (KeyError, TemplateError) as e:
        print(f"[ablator] {job['id']} unrunnable: {e}", flush=True)
        return "failed", None
    print(f"[ablator] running {job['id']} -> {job.get('model_path')} (log {log_path})",
          flush=True)
    prov_state = capture_and_record_provenance(cfg, job, machine, cwd or os.getcwd(), q)
    check_r9700_drift(cfg, job, machine, prov_state, q)
    container_name = container_name_from_argv(argv)
    if container_name:
        # Pre-launch safety net: if a prior attempt for this same job id
        # leaked a container under this name (e.g. crashed before this fix,
        # or a future bug reintroduces the gap), `podman run --name X` would
        # otherwise fail loudly with "name already in use" and the job would
        # spuriously fail every retry. Force-clear it first so retries never
        # collide, and log loudly so a real leak doesn't pass silently.
        force_remove_container(argv[0], container_name)
    try:
        with open(log_path, "w") as lf:
            lf.write(f"# {time.strftime('%Y-%m-%dT%H:%M:%S')} {job['id']}\n"
                     f"# cwd={cwd or os.getcwd()}\n# {shlex.join(argv)}\n")
            lf.write(provmod.format_banner("bare-metal", prov_state) + "\n")
            lf.flush()
            proc = subprocess.Popen(argv, env=env, cwd=cwd,
                                    stdout=lf, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            override = supervise(cfg, job, proc, cwd or os.getcwd(), q, argv=argv)
        rc = proc.returncode
        if override is not None:
            # Control-triggered stop/skip/requeue (or a lane preemption)
            # already returned an explicit terminal/backoff status above in
            # supervise() — the exit code of the killed subprocess is never
            # consulted here, so a manual kill can never read as "done".
            return override, rc
        if rc == 0 and _require_result_artifact(cfg, tcfg):
            h = healthmod.job_health(job, cwd or os.getcwd(), cfg.get("queue", {}),
                                     process_alive=False)
            if h["state"] != "done":
                print(f"[ablator] {job['id']} exited 0 but no completion "
                      f"artifact found (result_glob unmatched, state="
                      f"{h['state']!r}) — treating as failed, not done",
                      flush=True)
                return "failed", rc
        return ("done" if rc == 0 else "failed"), rc
    except Exception as e:
        print(f"[ablator] {job['id']} crashed: {e}", flush=True)
        return "failed", None


def _job_base_dir(cfg: dict, job: dict, machine: str) -> str:
    # k8s-backend machines have a machine-override `cwd` that is the
    # CONTAINER workingDir (e.g. /workspace/splatograph), meaningless for
    # host-side path resolution -- the coordinator process checking health
    # always runs on a real bare-metal host (main/r9700), never inside the
    # pod. Fall back to the base (non-machine-overridden) type cwd instead,
    # same fix as run_job_k8s's own health check.
    if cfgmod.machine_cfg(cfg, machine).get("backend") == "k8s":
        return cfg.get("types", {}).get(job.get("type", ""), {}).get("cwd") or os.getcwd()
    try:
        tcfg = cfgmod.type_cfg(cfg, job.get("type", ""), machine)
    except KeyError:
        tcfg = {}
    return tcfg.get("cwd") or os.getcwd()


def _require_result_artifact(cfg: dict, tcfg: dict) -> bool:
    """Per-type (falls back to [queue]) toggle: an exit code of 0 is not
    sufficient for 'done' — a result_glob artifact must also exist.

    Defaults to False for backward compat; set `require_result_artifact =
    true` under a [types.<t>] (or its per-machine override) or under
    [queue] to opt a job type in. This exists because a wrapper script can
    swallow a killed subprocess's real exit code (e.g. `docker wait X ||
    true` followed by unconditional `echo "Done."`), which would otherwise
    let a manually-killed or crashed run masquerade as a successful one.
    """
    if "require_result_artifact" in tcfg:
        return bool(tcfg["require_result_artifact"])
    return bool(cfg.get("queue", {}).get("require_result_artifact", False))


def _k8s_job_still_active(mcfg: dict, name: str) -> tuple[bool, bool]:
    """Query real k8s liveness for a Job name: (exists, still_active).

    `exists` is False if `kubectl get job` fails (Job gone / API error).
    `still_active` is True only when the Job exists and has neither
    succeeded nor failed yet — i.e. it is genuinely still running (or
    still pending scheduling), as opposed to a real completion/failure
    that happened while this runner process was down.
    """
    ns = mcfg["namespace"]
    status = _kubectl(["get", "job", name, "-n", ns, "-o", "json"], timeout=30)
    if status.returncode != 0:
        return False, False
    st = json.loads(status.stdout).get("status", {})
    if st.get("succeeded", 0) >= 1 or st.get("failed", 0) >= 1:
        return True, False
    return True, True


_ABLATOR_CONTAINER_RE = re.compile(r"^ablator-(?P<job_id>.+)$")

# Defense-in-depth, independent of any specific job's own supervise() path
# (found necessary live 2026-07-06: spp39f3_ctrl's container leaked for
# ~13h after ablator's own ledger had already marked it 'quarantined',
# because the targeted kill_job() teardown had a gap — see kill_job()'s
# docstring). This reaper does not assume that gap is fully closed by the
# name-based teardown above; it independently audits *every* container
# named 'ablator-<job_id>' against the ledger's current terminal/non-
# terminal state on every loop tick, regardless of which code path was
# supposed to have cleaned it up. Loud logging on every find, per this
# project's established taste for visible telemetry over silent failure
# (see CLAUDE.md run-contracts banner precedent).
def reap_orphaned_containers(cfg: dict, q: Queue, runtime: str = "podman") -> int:
    """Scan `podman ps -a` for ablator-managed containers whose job is no
    longer genuinely running per the ledger, and force-remove them.

    A container is orphaned if its name matches 'ablator-<job_id>' and the
    ledger entry for that job_id is missing or not status=='running'. (A
    'running' job might legitimately still be claimed-and-supervised by
    THIS OR ANOTHER machine's live runner process — bare-metal job ids are
    unique across the queue so name collisions across machines can't
    happen; leave those alone.) Returns the number of containers reaped.
    """
    try:
        out = subprocess.run(
            [runtime, "ps", "-a", "--format", "{{.Names}}\t{{.Status}}"],
            capture_output=True, text=True, timeout=15, check=False,
        ).stdout
    except (OSError, subprocess.TimeoutExpired) as e:
        print(f"[ablator] reap_orphaned_containers: {runtime} ps failed: {e!r}",
              flush=True)
        return 0
    by_id = {j["id"]: j for j in q.read()}
    reaped = 0
    for line in out.splitlines():
        parts = line.split("\t", 1)
        name = parts[0].strip()
        status = parts[1].strip() if len(parts) > 1 else "?"
        m = _ABLATOR_CONTAINER_RE.match(name)
        if not m:
            continue
        job_id = m.group("job_id")
        job = by_id.get(job_id)
        if job is not None and job.get("status") == "running":
            continue  # legitimately in flight (this or another machine)
        print(f"[ablator] REAPER: orphaned container '{name}' (podman status: "
              f"{status!r}) found, ledger status: "
              f"{job.get('status') if job else '<no such job>'!r} — force-removing",
              flush=True)
        force_remove_container(runtime, name)
        reaped += 1
    return reaped


def reconcile_stale_running(cfg: dict, machine: str, q: Queue,
                            busy: bool | None = None,
                            inflight: "_K8sInflight | None" = None) -> None:
    """Self-heal 'running' jobs this machine claimed but is no longer
    supervising (this runner process just (re)started, so any job still
    marked running-and-claimed-by-us predates this process and has no
    live supervise() loop watching it).

    Only ever touches jobs with claimed_by == machine — a different
    machine's claim is untouched, it self-heals on its own restart.

    Never runs while the machine's busy-guards say something is still
    actually executing (e.g. the training container is still up): we
    cannot tell *which* job that process belongs to, and reconciling
    while it might still be writing results would risk a second runner
    launching a duplicate job against the same model_path. In that case
    the entry is left stuck for now and a warning is printed; the next
    idle poll (once the process/container is gone) reconciles it.

    For a k8s-backend `machine`, in-memory in-flight tracking (thread +
    _K8sInflight bookkeeping) is necessarily lost across a runner process
    restart, but the k8s Job itself is NOT — it is a real cluster object
    outside this process, still executing under kai-scheduler regardless
    of whether anything here is polling it. Blindly requeuing every
    'running' k8s-targeted job on restart (the bare-metal-only check
    above) would falsely mark genuinely-still-running cluster jobs as
    crashed, and re-dispatching them would depend on lucky `kubectl apply`
    idempotency against a byte-identical manifest rather than correct
    behavior (found live 2026-07-07: exactly this happened after a
    restart). Instead, query the real k8s Job status via `kubectl get job`
    before assuming anything is dead, and re-attach a polling thread to a
    genuinely-still-running Job rather than requeuing it.
    """
    if busy is None:
        busy = resources.machine_busy(cfg, machine)
    if busy:
        return
    is_k8s = cfgmod.machine_cfg(cfg, machine).get("backend") == "k8s"
    mcfg = cfgmod.machine_cfg(cfg, machine) if is_k8s else None
    for job in q.read():
        if job.get("status") != "running" or job.get("claimed_by") != machine:
            continue

        if is_k8s:
            name = _k8s_job_name(job["id"])
            exists, still_active = _k8s_job_still_active(mcfg, name)
            if still_active:
                print(f"[ablator] reconcile: {job['id']} k8s Job "
                      f"{mcfg['namespace']}/{name} is still genuinely running "
                      "after a runner restart — re-attaching, not requeuing",
                      flush=True)
                try:
                    tcfg = cfgmod.type_cfg(cfg, job.get("type", ""), machine)
                except KeyError:
                    tcfg = {}
                log_path = os.path.join(cfgmod.log_dir(cfg), f"{job['id']}.log")
                ns = mcfg["namespace"]

                def run_fn(job=job, tcfg=tcfg, name=name, ns=ns,
                          log_path=log_path):
                    return _poll_k8s_job(cfg, job, machine, mcfg, tcfg, name,
                                        ns, log_path, append=True)

                t = threading.Thread(
                    target=_dispatch_and_finalize,
                    args=(cfg, machine, job, machine, q),
                    kwargs={"run_fn": run_fn},
                    daemon=True,
                    name=f"k8s-reattach-{job['id']}",
                )
                t.start()
                if inflight is not None:
                    inflight.add(machine, t, job["id"])
                continue
            if exists:
                # The Job finished (succeeded or failed) while this runner
                # process was down. A real completion is not a crash — fall
                # through to the same artifact-gated done/requeue check
                # bare-metal jobs get below; only the absence of a live
                # in-memory process differs, and that's expected here.
                print(f"[ablator] reconcile: {job['id']} k8s Job "
                      f"{mcfg['namespace']}/{name} already reached a terminal "
                      "state while this runner was down — checking for a "
                      "completion artifact", flush=True)
            else:
                print(f"[ablator] reconcile: {job['id']} k8s Job "
                      f"{mcfg['namespace']}/{name} no longer exists on the "
                      "cluster — treating as crashed", flush=True)

        base_dir = _job_base_dir(cfg, job, machine)
        h = healthmod.job_health(job, base_dir, cfg.get("queue", {}),
                                 process_alive=False)
        if h["state"] == "done":
            print(f"[ablator] reconcile: {job['id']} has a completion artifact "
                  f"but was stuck at 'running' (orphaned by a runner "
                  f"restart/crash) — marking done", flush=True)
            q.finish(job["id"], "done", health=h, reconciled=True,
                    reconciled_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
        else:
            print(f"[ablator] reconcile: {job['id']} stuck at 'running' with no "
                  f"live process and no completion artifact (state={h['state']}) "
                  f"— requeuing to pending", flush=True)
            q.update(job["id"], status="pending", health=h,
                    claimed_by=None, claimed_at=None, reconciled=True,
                    reconciled_at=time.strftime("%Y-%m-%dT%H:%M:%S"))


DEFAULT_K8S_MAX_CONCURRENT = 4


def _k8s_max_concurrent(cfg: dict, k8s_name: str) -> int:
    """Concurrency cap for a k8s-backend dispatch machine.

    KAI Scheduler already queues excess Jobs beyond real cluster capacity on
    the cluster side, so a slightly-too-high cap here is not dangerous — it
    just means some submitted Jobs sit Pending in kubectl until a GPU frees
    up. Defaults conservatively (not "all 8 GPUs are free") since ablator
    has no visibility into what else might be using the cluster.
    """
    mcfg = cfgmod.machine_cfg(cfg, k8s_name)
    try:
        return max(1, int(mcfg.get("max_concurrent", DEFAULT_K8S_MAX_CONCURRENT)))
    except (TypeError, ValueError):
        return DEFAULT_K8S_MAX_CONCURRENT


def _dispatch_and_finalize(cfg: dict, machine: str, job: dict, job_machine: str,
                           q: Queue, run_fn=None) -> str:
    """Run one job to completion and apply the full success/failure/retry/
    quarantine/preempt/requeue bookkeeping.

    Shared by the serial bare-metal path and each concurrent k8s dispatch
    thread in run_loop() — every job, regardless of which machine it targets
    or whether it runs synchronously or on a background thread, goes through
    exactly this same disposition logic.

    `run_fn` defaults to the normal submit-and-run path (`run_job`), but a
    caller re-attaching to a k8s Job that was already running before this
    process started (restart recovery, see `reconcile_stale_running`) can
    pass a callable that resumes polling that existing Job instead of
    re-submitting it.
    """
    base_dir = _job_base_dir(cfg, job, job_machine)
    run_fn = run_fn or (lambda: run_job(cfg, job, job_machine, q))
    status, exit_code = run_fn()
    if status == "failed":
        disposition = handle_failure(cfg, job, exit_code, job_machine, base_dir, q)
        if disposition == "retry":
            # unknown category: preserve existing uniform
            # retry-once-then-quarantine behavior.
            if not job.get("retried"):
                job["retried"] = True
                q.update(job["id"], retried=True)
                print(f"[ablator] retrying {job['id']} once", flush=True)
                status, exit_code = run_job(cfg, job, job_machine, q)
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
    print(f"[ablator] {job['id']} -> {status}", flush=True)
    return status


class _K8sInflight:
    """Tracks concurrently-dispatched k8s job threads, keyed by k8s machine
    name, so run_loop can cap concurrency per machine and report a live
    count in the heartbeat.

    Only ever touched from the main run_loop thread (append when spawning,
    reap when a thread finishes) — no locking needed for the bookkeeping
    dict itself, only the Queue file operations inside each thread need
    (and already have, via flock) their own synchronization.
    """

    def __init__(self):
        self._threads: dict[str, list[tuple[threading.Thread, str]]] = {}

    def reap(self) -> None:
        for name, entries in list(self._threads.items()):
            alive = [(t, jid) for t, jid in entries if t.is_alive()]
            if alive:
                self._threads[name] = alive
            else:
                del self._threads[name]

    def count(self, name: str) -> int:
        return len(self._threads.get(name, []))

    def total(self) -> int:
        return sum(len(v) for v in self._threads.values())

    def add(self, name: str, thread: threading.Thread, job_id: str) -> None:
        self._threads.setdefault(name, []).append((thread, job_id))

    def join_all(self) -> None:
        for entries in self._threads.values():
            for t, _jid in entries:
                t.join()
        self._threads.clear()


def run_loop(cfg: dict, once: bool = False) -> None:
    machine = cfgmod.machine_name(cfg)
    q = Queue(cfgmod.queue_path(cfg))
    # k8s-backend machines (e.g. a100cluster) have no hostname_patterns --
    # nothing's identity ever resolves to them, since this repo's code never
    # runs ON a cluster node itself. Whichever bare-metal host runs `ablator
    # run` (main/r9700) instead acts as the DISPATCHER: it also claims jobs
    # explicitly targeting a k8s machine name and submits/polls them via
    # kubectl (run_job_k8s), on top of claiming its own bare-metal identity's
    # jobs. This is safe to do from every runner (kubectl apply/get are
    # idempotent-ish and Queue.claim_next's file lock already serializes
    # claims across machines — flock is per-open-file-description, so it
    # serializes correctly across threads within this same process too, not
    # just across separate processes/machines), so no extra "dispatcher"
    # config is needed.
    #
    # Bare-metal dispatch (this runner's own `machine` identity) stays
    # exactly as serial as before: one job claimed and run to completion per
    # loop iteration, blocking. k8s-targeted jobs are different: KAI
    # Scheduler and the cluster's real GPUs provide the actual parallelism,
    # so this loop must not be the bottleneck — each claimed k8s job is
    # dispatched via run_job_k8s (through _dispatch_and_finalize) on its own
    # background thread, up to a configurable per-machine concurrency cap,
    # so multiple cluster jobs can be in flight at once without blocking
    # bare-metal claiming/dispatch.
    k8s_machines = [
        name for name, m in cfg.get("machines", {}).items()
        if m.get("backend") == "k8s"
    ]
    dispatch_machines = [machine] + k8s_machines
    print(f"[ablator] runner on {machine} watching {q.path} "
          f"(dispatching for: {', '.join(dispatch_machines)})", flush=True)
    inflight = _K8sInflight()
    reconcile_stale_running(cfg, machine, q)
    for k8s_name in k8s_machines:
        reconcile_stale_running(cfg, k8s_name, q, busy=False, inflight=inflight)
    last_tick = time.monotonic()
    while True:
        # Watchdog: if the previous iteration (probes + sleeps, NOT a job
        # run) took absurdly long, say so loudly — this is the 'stuck loop'
        # tell. Background k8s threads run their own blocking polls, so a
        # long-running k8s job never shows up here (it doesn't hold up the
        # main loop's own tick).
        now = time.monotonic()
        if now - last_tick > STALL_WARN_S:
            print(f"[ablator] WARNING: loop iteration took {now - last_tick:.0f}s "
                  f"(> {STALL_WARN_S}s) — a probe or lock likely hung",
                  flush=True)
        last_tick = now
        try:
            inflight.reap()
            if resources.machine_busy(cfg, machine):
                write_heartbeat(cfg, machine,
                                f"busy-wait k8s_inflight={inflight.total()}")
                if once:
                    inflight.join_all()
                    return
                time.sleep(BUSY_POLL_S)
                continue
            write_heartbeat(cfg, machine, f"idle k8s_inflight={inflight.total()}")

            # 1. Fill k8s concurrency slots — non-blocking: each claimed job
            # is handed to a background thread and this loop moves straight
            # on to bare-metal claiming below without waiting for it.
            for k8s_name in k8s_machines:
                cap = _k8s_max_concurrent(cfg, k8s_name)
                can_run = make_can_run(cfg, k8s_name)
                while inflight.count(k8s_name) < cap:
                    kjob = q.claim_next(k8s_name, can_run=can_run)
                    if kjob is None:
                        break
                    print(f"[ablator] dispatching {kjob['id']} to {k8s_name} "
                          f"({inflight.count(k8s_name) + 1}/{cap} in flight)",
                          flush=True)
                    t = threading.Thread(
                        target=_dispatch_and_finalize,
                        args=(cfg, machine, kjob, k8s_name, q),
                        daemon=True,
                        name=f"k8s-{kjob['id']}",
                    )
                    t.start()
                    inflight.add(k8s_name, t, kjob["id"])

            # 2. Claim and run (serially, blocking) at most one bare-metal
            # job for this runner's own identity — unchanged from before.
            job = q.claim_next(machine, can_run=make_can_run(cfg, machine))
            if job is None:
                if once:
                    inflight.join_all()
                    return
                time.sleep(IDLE_POLL_S)
                continue
            write_heartbeat(cfg, machine, f"running:{job['id']}")
            status = _dispatch_and_finalize(cfg, machine, job, machine, q)
            write_heartbeat(cfg, machine,
                            f"finished:{job['id']}:{status} "
                            f"k8s_inflight={inflight.total()}")
            last_tick = time.monotonic()  # job runs are legitimately long
            if once:
                inflight.join_all()
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
        # --config is a top-level flag (`ablator --config X run`), not a
        # subcommand argument (`ablator run --config X` fails argparse
        # with "unrecognized arguments") -- found live 2026-07-06 when
        # this exact ordering crashed instantly on relaunch.
        cmd = (f"setsid nohup ablator --config {shlex.quote(cfg['_path'])} run "
               f"</dev/null > {shlex.quote(log)} 2>&1 &")
        subprocess.run(["bash", "-c", cmd], check=True)
        print(f"[start] launched runner on {me} (log {log})")

    for name, m in cfg.get("machines", {}).items():
        if name == me or not m.get("ssh"):
            continue
        runner_cmd = m.get("runner_command", "ablator run")
        log = os.path.join(ldir, f"runner_{name}.log")
        # Two separate ssh calls, not one conditional one-liner: if the
        # check and the launch command are sent as a single script, the
        # *checking* shell process's own argv contains the launch
        # command's literal text (e.g. "ablator run" in the else branch)
        # even when that branch never executes — `pgrep -f` then matches
        # the checking process itself and always reports "already
        # running" on the first real attempt. Keeping the check's argv
        # free of the launch text avoids this self-match.
        check = subprocess.run(["ssh", m["ssh"], "pgrep -f '[a]blator run'"],
                               capture_output=True, text=True)
        if check.stdout.strip():
            print(f"[start] runner already running on {name}")
            continue
        remote = (f"mkdir -p {shlex.quote(ldir)} && "
                  f"setsid nohup {runner_cmd} </dev/null > {shlex.quote(log)} 2>&1 &")
        r = subprocess.run(["ssh", m["ssh"], remote])
        if r.returncode == 0:
            print(f"[start] launched runner on {name}")
        else:
            print(f"[start] WARNING: could not reach {m['ssh']} — "
                  f"runner on {name} not started")
    print(f"[start] done. Watch: tail -f {ldir}/runner_*.log")
