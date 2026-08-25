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
import shutil
import signal
import subprocess
import threading
import time

from . import config as cfgmod
from . import error as errormod
from . import execution_receipt as receiptmod
from . import experiment_declaration as declarations
from . import health as healthmod
from . import provenance as provmod
from . import resources
from . import source_checkout as sourcecheckout
from . import self_check as selfcheckmod
from .pause_revalidation import revalidate_pause
from .queue import Queue, is_paused, pause_flag_path, write_pause_flag
from .urgent_fixes import enforce_urgent_fixes, load_urgent_fixes

IDLE_POLL_S = 30
BUSY_POLL_S = 30
HEALTH_POLL_S = 60
STALL_WARN_S = 600  # loudly log if one loop iteration took longer than this
SELF_CHECK_INTERVAL_S = 3600  # re-check ablator's own git currency hourly

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
    the satellite-vs-main drift check works across two SEPARATE ablator
    processes (each machine runs its own `ablator run`) without any
    SSH-specific plumbing: queue.jsonl (and therefore log_dir) already
    lives on shared NFS (/mnt/cps_scratch1_tmp), so this file is visible
    to both machines the moment either one writes it."""
    return os.path.join(cfgmod.log_dir(cfg), f"git_state_{machine}.json")


def write_git_state_file(cfg: dict, machine: str, state: dict) -> None:
    path = _git_state_path(cfg, machine)
    temporary_path = (
        f"{path}.tmp.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}"
    )
    try:
        fd = os.open(temporary_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        with os.fdopen(fd, "w") as f:
            json.dump({**state, "written_at": time.time()}, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_path, path)
    except OSError as e:
        print(f"[ablator] write_git_state_file({machine}) failed: {e!r}", flush=True)
    finally:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass


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
    for drift comparison (see check_checkout_drift below).
    """
    state = provmod.capture_local_git_state(cwd)
    if q is not None:
        q.update(job["id"], provenance=state)
    write_git_state_file(cfg, machine, state)
    return state


def refresh_idle_provenance(cfg: dict, machine: str) -> dict | None:
    """Refresh cross-machine workload state without requiring a job claim.

    The urgent-fix checkout is the runner's canonical live workload checkout,
    including its per-machine path override.  Refreshing it at runner startup
    and on the existing self-check cadence prevents an idle machine's shared
    state from remaining pinned to the last job it happened to dispatch.
    """
    repo_cwd, _fixes, _auto_sync_ref = load_urgent_fixes(cfg, machine)
    if repo_cwd is None:
        return None
    state = provmod.capture_local_git_state(repo_cwd)
    write_git_state_file(cfg, machine, state)
    return state


def expected_branch(cfg: dict) -> str:
    """The branch a runner's shared checkout is expected to sit on.

    Derived from `[urgent_fixes] auto_sync_ref` (the only place the config
    already states "this checkout must track <ref>"), with the remote
    prefix stripped: `origin/main` -> `main`. Falls back to `main`, which
    is what every runner has in practice been assumed to be on.
    """
    ref = ((cfg.get("urgent_fixes") or {}).get("auto_sync_ref") or "main").strip()
    return ref.split("/")[-1] or "main"


def check_checkout_drift(cfg: dict, job: dict, machine: str, state: dict,
                         q: Queue | None) -> None:
    """Proactive, loud (WARN-not-refuse) check on the shared, mutable git
    checkout this job is about to execute from.

    Runners execute training from a fixed, mutable, shared checkout, so
    whatever branch/working-tree state that checkout happens to be in at
    claim time is what the job runs (splatograph#259). Three independent
    ways that silently changes what ran, all checked here from the git
    state `capture_and_record_provenance()` already captured — no extra
    subprocesses, no new config:

    1. **Cross-machine divergence** (any satellite vs. main). Each machine
       runs its own ablator process, so a satellite's runner reads the
       shared `git_state_main.json` that main's runner last wrote and
       compares. Best-effort (stale if main hasn't dispatched a bare-metal
       job recently), but needs no SSH plumbing or reachability. Note this
       cannot report "N commits behind": main's commit may not even exist
       in a stale satellite's object store without a fetch — a real
       descendant *gate* is deliberately left as follow-up.
       Previously hardcoded to `machine != "r9700"`, which is why rtx3090
       executed a job from a 25-commit-stale checkout carrying no warning
       at all; now every non-main machine checks itself.
    2. **Dirty working tree**, on every machine including main. Never
       flagged before, and it is exactly splatograph#259's Instance 2 (a
       rewound `ablator` submodule pointer showed up only as ` M ablator`).
       Only `dirty is True` warns — `dirty is None` means git was
       unreadable, which is reported by `capture_local_git_state`'s own
       `error` field rather than mislabelled as drift here.
    3. **Off the expected branch** (see `expected_branch`). A *clean*
       checkout parked on an agent branch is the headline #259 case and is
       invisible to both checks above.

    WARN, never refuse: a user may deliberately want different code on a
    given machine (e.g. testing a branch there only), and a refusal path
    here would stall a lane exactly as the `urgent_fix_unsynced` pause
    once did. A loud warning in both the runner log and the job's ledger
    entry (`drift_warning`, read by splatograph's
    `scripts/audit_run_drift.py`) makes the drift impossible to miss
    without blocking a job the user may have wanted to run as-is.
    """
    if job.get("requested_git_sha"):
        return

    warnings: list[str] = []
    fields: dict = {}
    commit = state.get("commit")

    if machine != "main" and commit:
        main_state = read_git_state_file(cfg, "main")
        main_commit = (main_state or {}).get("commit")
        if main_commit and main_commit != commit:
            warnings.append(
                f"CODE PROVENANCE DRIFT: {machine} is executing job {job['id']!r} at "
                f"commit {commit[:12]} (branch {state.get('branch')}) but "
                f"main's checkout was last observed at commit "
                f"{main_commit[:12]} — these two machines' checkouts have "
                f"diverged. If intentional (e.g. testing a branch on {machine} "
                f"only), ignore; otherwise sync the checkouts before trusting "
                f"cross-machine comparisons.")
            fields["main_commit_at_check"] = main_commit

    if state.get("dirty") is True:
        warnings.append(
            f"CODE PROVENANCE DRIFT: {machine}'s checkout at "
            f"{state.get('cwd')!r} has UNCOMMITTED CHANGES while claiming job "
            f"{job['id']!r} at commit {(commit or '?')[:12]} — the code this job "
            f"runs is not any commit that exists anywhere, so its results are "
            f"not reproducible from the recorded SHA alone (a rewound submodule "
            f"pointer looks exactly like this). Commit or clean the tree before "
            f"trusting these results.")

    branch = state.get("branch")
    want = expected_branch(cfg)
    if branch and branch not in (want, "HEAD") and not state.get("error"):
        warnings.append(
            f"CODE PROVENANCE DRIFT: {machine}'s checkout is on branch "
            f"{branch!r}, not the expected {want!r}, while claiming job "
            f"{job['id']!r} at commit {(commit or '?')[:12]} — this job runs that "
            f"branch's code, but nothing downstream attributes it to anything "
            f"other than {want!r}. If intentional, ignore; otherwise switch the "
            f"checkout back before trusting these results.")

    if not warnings:
        return
    warning = " | ".join(warnings)
    print(f"[ablator] {warning}", flush=True)
    if q is not None:
        q.update(job["id"], drift_warning=warning, **fields)


# Back-compat alias: this check has not been r9700-specific since
# splatograph#259 (see check_checkout_drift's docstring).
check_r9700_drift = check_checkout_drift


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
    params = job.get("params") or {}
    if not isinstance(params, dict):
        raise TemplateError(f"job {job.get('id')} params must be a mapping")
    # External schedulers may provide typed, workload-agnostic template
    # variables. Legacy queue fields remain authoritative and deliberately
    # overwrite same-named params for backward compatibility and safety.
    values = {str(key): str(value) for key, value in params.items()}
    values.update({
        "scene": scene,
        "model_path": job.get("model_path", ""),
        "extra_args": extra_args,
        "iterations": str(job.get("iterations", "")),
        "id": job.get("id", ""),
        "machine": machine,
    })
    return values


def _fmt(s: str, vars: dict) -> str:
    try:
        return s.format(**vars)
    except (KeyError, IndexError) as e:
        raise TemplateError(f"unknown template variable in {s!r}: {e}")


def _sanitize_container_name(raw: str) -> str:
    """docker/podman container names must match [a-zA-Z0-9][a-zA-Z0-9_.-]*."""
    out = "".join(c if c.isalnum() or c in "_.-" else "_" for c in raw)
    return out if out and (out[0].isalnum()) else f"j_{out}"


def _ensure_container_name(argv: list[str], job: dict) -> list[str]:
    """Inject `--name splat_train_<job_id>` into a rendered docker/podman
    `run` command if the template didn't already set one.

    Every busy-guard in this project's configs (see [[machines.*.busy_guards]]
    in configs/ablator.toml / configs/ablator.json) checks `docker ps
    --format {{.Names}}` for the substring "splat_train" to tell "GPU is
    genuinely occupied by a training job" apart from leftover viewer/router
    containers. That guard can only ever work if the launched container's
    name actually contains that substring -- but no command template here
    ever passed `--name`, so every job ran under Docker's random
    adjective-scientist name generator and the guard silently never matched
    anything. The direct consequence (found live 2026-08-12): `busy` always
    read False for a machine with a real training container running, so
    `reconcile_stale_running()`'s `if busy: return` early-out never fired,
    and a job that was simply slow to write its first health-check artifact
    got requeued to pending and picked up by a second launch -- two
    processes training the identical job, each holding a full share of GPU
    memory, on a real run. This centralizes the fix at the one place every
    docker/podman command is assembled, rather than requiring every current
    and future command template to remember `--name` itself.
    """
    if not argv or argv[0] not in _CONTAINER_RUNTIMES or "run" not in argv[:2]:
        return argv
    if container_name_from_argv(argv) is not None:
        return argv  # template already set an explicit name -- respect it
    name = f"splat_train_{_sanitize_container_name(str(job.get('id', 'job')))}"
    run_idx = argv.index("run")
    return argv[: run_idx + 1] + ["--name", name] + argv[run_idx + 1 :]


def _inject_container_environment(argv: list[str], child_env: dict[str, str]) -> list[str]:
    """Inject protected declaration env into a direct Docker/Podman run."""
    if not argv or argv[0] not in _CONTAINER_RUNTIMES:
        return argv
    if "run" not in argv[:2]:
        return argv

    protected = declarations.PROTECTED_ENV
    for index, token in enumerate(argv):
        if child_env and (token == "--env-file" or token.startswith("--env-file=")):
            raise TemplateError(
                "declared container job cannot use --env-file because protected "
                "declaration values could be overridden"
            )
        if token in ("-e", "--env") and index + 1 < len(argv):
            key = argv[index + 1].split("=", 1)[0]
            if key in protected:
                raise TemplateError(f"command template overrides protected env {key}")
        if token.startswith("-e") and not token.startswith("--") and len(token) > 2:
            key = token[2:].removeprefix("=").split("=", 1)[0]
            if key in protected:
                raise TemplateError(f"command template overrides protected env {key}")
        if token.startswith("--env="):
            key = token[len("--env="):].split("=", 1)[0]
            if key in protected:
                raise TemplateError(f"command template overrides protected env {key}")

    if not child_env:
        return argv
    run_index = argv.index("run")
    flags: list[str] = []
    for key, value in child_env.items():
        flags.extend(["--env", f"{key}={value}"])
    return argv[: run_index + 1] + flags + argv[run_index + 1 :]


def render_command(
    tcfg: dict, job: dict, machine: str, *, include_protected_env: bool = True,
) -> tuple[list[str], dict, str | None]:
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
    try:
        declaration_env = declarations.experiment_environment(job)
    except declarations.ExperimentDeclarationError as exc:
        raise TemplateError(str(exc)) from exc
    argv = _ensure_container_name(argv, job)
    if include_protected_env:
        argv = _inject_container_environment(argv, declaration_env)
    env = os.environ.copy()
    for k, v in (tcfg.get("env") or {}).items():
        env[k] = _fmt(str(v), vars)
    for key in declarations.PROTECTED_ENV:
        env.pop(key, None)
    if include_protected_env:
        env.update(declaration_env)
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
              argv: list[str] | None = None,
              machine: str | None = None,
              mem_sampler=None) -> str | None:
    """Watch a running job until its process exits or intervention is needed.

    Returns None when the process exited on its own (caller reads
    returncode), or an override status: 'failed_no_retry' (manual stop),
    'cancelled' (manual skip), 'requeue' (manual requeue), 'preempted'
    (lane-1 job yielding to a pending lane-3 job), 'failed'
    (hung/crashed OR sustained GPU-memory danger — normal retry->quarantine
    path applies either way; the memory case additionally stamps
    job['_gpu_memory_exhausted'] so handle_failure()/classify_and_record()
    record the definitive category instead of guessing from the log tail).

    Health comes ONLY from the run's own artifacts (health module); the run
    itself stays fully standalone and never talks to the runner.

    The GPU-memory guard below is memory-based, not progress-based: it
    samples actual GTT/VRAM usage every poll_s regardless of whether the
    job's own health state looks fine. This means it also naturally covers
    Incident 1's pattern (a process hung/leaking memory post-crash but not
    yet reaped, health state possibly still "ok"/"hung" ambiguous) — a
    stuck process that is still holding a lot of GPU memory keeps tripping
    this guard even if `hung`/`crashed` health detection is inconclusive.
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
    mem_sampler = mem_sampler or (
        (lambda: resources.sample_gpu_mem_pct(cfg, machine)) if machine else (lambda: None))
    rcfg = cfg.get("resources", {})
    mem_danger_pct = rcfg.get("mem_kill_danger_pct", resources.DEFAULT_MEM_KILL_DANGER_PCT)
    mem_grace_cycles = rcfg.get("mem_kill_grace_cycles", resources.DEFAULT_MEM_KILL_GRACE_CYCLES)
    mem_breach_streak = 0
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
        mem_pct = mem_sampler()
        if mem_pct is not None and mem_pct >= mem_danger_pct:
            mem_breach_streak += 1
            print(f"[GPU MEMORY DANGER] {time.strftime('%Y-%m-%dT%H:%M:%S')} "
                  f"host={machine} pid={proc.pid} model_path={job.get('model_path')} "
                  f"usage={mem_pct:.1f}% threshold={mem_danger_pct}% "
                  f"consecutive_breaches={mem_breach_streak}/{mem_grace_cycles}",
                  flush=True)
            if mem_breach_streak >= mem_grace_cycles:
                print(f"[ablator] {job['id']} GPU memory danger threshold sustained "
                      f"for {mem_breach_streak} consecutive polls — killing", flush=True)
                job["_gpu_memory_exhausted"] = True
                job["_gpu_memory_pct"] = mem_pct
                kill()
                return "failed"
        else:
            mem_breach_streak = 0
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


def _measure_write_speed_mb_s(path: str, size_mb: int = 16) -> float | None:
    """Write ``size_mb`` of random data to a throwaway file under ``path``,
    fsync, time it, delete it. Best-effort: any OSError (permissions,
    read-only mount, path doesn't exist yet) returns None rather than
    raising -- this must never be able to fail a job on its own.
    """
    try:
        os.makedirs(path, exist_ok=True)
        probe_path = os.path.join(path, f".ablator_speedtest_{os.getpid()}.tmp")
        chunk = os.urandom(1024 * 1024)
        t0 = time.monotonic()
        with open(probe_path, "wb") as f:
            for _ in range(size_mb):
                f.write(chunk)
            f.flush()
            os.fsync(f.fileno())
        elapsed = time.monotonic() - t0
        os.remove(probe_path)
        if elapsed <= 0:
            return None
        return size_mb / elapsed
    except OSError:
        return None


def output_folder_preflight(model_path: str, cwd: str | None) -> str:
    """Mandatory pre-dispatch check: resolve a job's output folder to a real
    host filesystem path and report free space + measured write speed.

    Always runs, for every job, informational only -- never blocks
    dispatch. This project has been bitten repeatedly by silent NFS
    disk-full/write-stall incidents that were only noticed well after a job
    had been silently degraded or a queue file corrupted (a scratch1
    disk-full incident truncated queue.jsonl; a separate NFS write-queue
    stall on the same mount degraded a live training run's iteration rate
    for tens of minutes before it was caught). Surfacing both numbers in
    every job's own log up front, before training starts, catches this
    class of problem at dispatch time instead of only after a stall is
    already suspected.
    """
    resolved = model_path if os.path.isabs(model_path) else os.path.join(cwd or os.getcwd(), model_path)
    free = _disk_free_bytes(resolved)
    speed = _measure_write_speed_mb_s(resolved)
    free_str = f"{free / (1024 ** 3):.1f}GB" if free is not None else "unknown"
    speed_str = f"{speed:.1f}MB/s" if speed is not None else "unknown"
    return f"[ablator] output folder preflight: path={resolved} free={free_str} write_speed={speed_str}"


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
    """Classify a failed job's log/exit-code and persist error_* fields.

    If runner.supervise()'s GPU-memory guard already killed this job
    in-flight (job['_gpu_memory_exhausted']), that verdict is definitive —
    skip log-tail heuristics entirely and record 'gpu_memory_exhaustion'
    directly, still through this same function so the ledger bookkeeping
    (q.update of error_category/evidence/confidence/suggested_action) is
    identical to every other failure path.
    """
    if job.get("_gpu_memory_exhausted"):
        result = errormod.gpu_memory_exhaustion_result(job.get("_gpu_memory_pct"))
    else:
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


def k8s_dispatch_enabled(cfg: dict, machine: str) -> bool:
    """Whether this runner, acting as `machine`, should ever attempt to act
    as a k8s dispatcher (build dispatch_machines beyond its own identity,
    invoke kubectl, etc).

    Two independent gates, both must pass:
      1. [machines.<machine>].k8s_dispatch config flag, default True
         (preserves pre-2026-07-07 behavior for machines like `main` that
         genuinely have working cluster access). Set to `false` for a
         machine that should NEVER dispatch to k8s under any
         circumstances -- e.g. r9700, which has no route to the cluster's
         API server and raced/quarantined jobs main would have handled
         correctly. This is a permanent policy switch, not a fallback.
      2. `kubectl` binary actually present on PATH (shutil.which). Defense
         in depth independent of (1): a future misconfigured machine (or
         one where `k8s_dispatch` was left at the default True by mistake)
         must never CRASH its entire runner process -- bare-metal jobs and
         all -- just because kubectl isn't installed. r9700 did exactly
         this on 2026-07-07 (FileNotFoundError at startup) before this
         flag existed.
    """
    mcfg = cfg.get("machines", {}).get(machine, {})
    if not mcfg.get("k8s_dispatch", True):
        return False
    if shutil.which("kubectl") is None:
        return False
    return True


def build_k8s_job_manifest(mcfg: dict, job: dict, argv: list[str],
                          cwd: str | None, local_commit: str | None = None,
                          image_override: str | None = None) -> dict:
    """Build the Job manifest dict for one job on a k8s-backend machine.

    Workload-agnostic: nothing here assumes Gaussian-splatting specifically.
    All cluster/scheduling/mount specifics are config-driven via `mcfg`, with
    defaults chosen to keep every existing splatograph-style config
    byte-identical to pre-generalization behavior:

    - `scheduler_name` (default "kai-scheduler"), `kai_queue`, `priority_class`
      are plain mcfg fields -- different chair users/teams point these at
      different KAI queues/priority classes without touching this code.
    - `pvc_persistent` / `pvc_scratch` are both OPTIONAL now. If neither is
      set, no dataset/scratch volumes are mounted at all (a plain PyTorch
      job with no shared-dataset PVC need not configure them). If either is
      set, the "dataset"/"scratch" volumes below are built as before.
    - `dataset_mount_path` (default "/data/scene", the splatograph
      convention) and `persistent_mount_root` / `scratch_mount_root`
      (default "/mnt/cps_persistent1_shared" / "/mnt/cps_scratch1_tmp") are
      now config fields, not hardcoded constants, so a generic job can route
      its own dataset layout/mount path instead of splatograph's.
    - `image_pull_secret` is optional; omitted -> no `imagePullSecrets` (a
      public image needs none).
    - `image_override`: optional, lets a caller run a DIFFERENT image than
      `mcfg["image"]` for this one job (e.g. a job-type-specific image such
      as a ROS2+CUDA bag-training image, set via `[types.<type>.machines.
      <machine>].image` in the TOML, vs. the machine's plain CUDA-only
      default image used by other job types on the same machine). `mcfg`
      only ever holds ONE image per machine otherwise -- this is the sole
      per-job image escape hatch.
    - `cpu_request`/`memory_request`/`cpu_limit`/`memory_limit` are
      overridable (defaults match the prior hardcoded 4/16Gi/8/32Gi).
    - `extra_volumes`: optional list of
      `{"name", "claim_name", "mount_path", "read_only"}` dicts for mounting
      any additional PVC (e.g. a shared checkpoint volume) generically.
    - `mps`: optional bool (default false). When true, wires the trainer
      container as an MPS (Multi-Process Service) client of the cluster's own
      per-node MPS control daemon -- needed on GPU nodes left in NVIDIA
      `Exclusive_Process` compute mode with no per-pod permission to change it,
      where a plain `nvidia.com/gpu: 1` request can otherwise fail its first
      CUDA call with "CUDA-capable device(s) is/are busy or unavailable" even
      on a fully idle GPU. See the `mps_enabled` block below for exactly what
      this adds (hostPath volume, env vars, soft anti-affinity) and why.

    Mounts (when configured): `pvc_persistent` (read-only, subPath'd to the
    job's real dataset directory so the in-container dataset mount path used
    by the rendered command template resolves correctly) and `pvc_scratch`
    (read-write, at the SAME absolute path bare-metal machines use -- so the
    command's own `ln -sfn .../output output/scratch` step, and ablator's own
    status/collect reading the shared queue.jsonl under that same tree, both
    work identically to a bare-metal job).

    Git-sync init container (OPT-IN, gated on `git_sync_repo_url` being set
    in mcfg -- absent by default, so this is a no-op / byte-identical
    manifest for every machine that hasn't configured it):

    The cuda-dev image's /workspace/splatograph source tree is COPY'd at
    image-build time, so it silently goes stale relative to the dispatching
    host's actual checkout (this is the gap commit 2e173b2's drift-WARNING
    flags but doesn't fix). Instead of trusting the baked source, an
    `alpine/git`-based init container clones the repo at the EXACT commit
    SHA the dispatching host was at when the job was submitted (not just a
    branch head, which could itself move between queueing and scheduling)
    into a shared `emptyDir` volume, and the trainer container mounts that
    SAME emptyDir OVER /workspace/splatograph -- overlaying the fresh
    checkout on top of (replacing, for that one path) the baked copy,
    rather than introducing a second parallel path the trainer would need
    to know about. This is safe because the heavy compiled deps
    (gsplat/fused-ssim/simple-knn wheels, torch, CUDA toolchain) install
    into /opt/venv's site-packages, never under /workspace/splatograph
    (verified against Dockerfile: COPY targets there are only source dirs
    like arguments/, scene/, utils/, train.py, etc.) -- so overlaying only
    ever replaces the .py/.json/config source tree the image install step
    the wheels are already built.

    KNOWN LIMITATION (out of scope to solve here): overlaying source
    without rebuilding the image means a freshly-pulled commit that
    requires an incompatible/newer gsplat, fused-ssim, or simple-knn wheel
    version than what's actually baked into `mcfg["image"]` will fail or
    misbehave at runtime with no compatibility check performed -- this
    decouples "source freshness" from "environment freshness" by design,
    and that tradeoff is deliberate (rebuilding the heavy wheels per job
    would be far too slow), but it does reintroduce a (much narrower,
    dependency-shaped rather than arbitrary-code-shaped) drift risk that
    provenance.py's drift-WARNING doesn't cover and this feature doesn't
    add new detection for.
    """
    try:
        declaration_env = declarations.experiment_environment(job)
    except declarations.ExperimentDeclarationError as exc:
        raise TemplateError(str(exc)) from exc
    name = _k8s_job_name(job["id"])
    image = image_override or mcfg["image"]
    scene = job.get("scene", "")
    persistent_root = mcfg.get("persistent_mount_root", "/mnt/cps_persistent1_shared")
    scratch_root = mcfg.get("scratch_mount_root", "/mnt/cps_scratch1_tmp")
    dataset_mount_path = mcfg.get("dataset_mount_path", "/data/scene")
    has_persistent_pvc = "pvc_persistent" in mcfg
    has_scratch_pvc = "pvc_scratch" in mcfg
    trainer_volume_mounts: list[dict] = []
    volumes: list[dict] = []
    if has_persistent_pvc or has_scratch_pvc:
        # Datasets can live under EITHER shared mount (e.g. TUM/floor3 under
        # persistent, ScanNet++'s cache under scratch) -- the "dataset"
        # volume must subPath into whichever PVC actually contains the
        # scene, not always pvc_persistent. Found live: a ScanNet++ job
        # silently mounted pvc_persistent's ROOT at /data/scene (empty
        # subPath, since the scene path didn't match persistent_root at
        # all) and crashed with "No supported RGB-D dataset layout found"
        # -- not a training-code bug.
        if scene.startswith(persistent_root) and has_persistent_pvc:
            dataset_pvc = mcfg["pvc_persistent"]
            sub_path = scene[len(persistent_root):].lstrip("/")
        elif scene.startswith(scratch_root) and has_scratch_pvc:
            dataset_pvc = mcfg["pvc_scratch"]
            sub_path = scene[len(scratch_root):].lstrip("/")
        else:
            dataset_pvc = mcfg.get("pvc_persistent") or mcfg.get("pvc_scratch")
            sub_path = ""
        trainer_volume_mounts.append(
            {"name": "dataset", "mountPath": dataset_mount_path,
             "subPath": sub_path, "readOnly": True})
        volumes.append(
            {"name": "dataset",
             "persistentVolumeClaim": {"claimName": dataset_pvc, "readOnly": True}})
    if has_scratch_pvc:
        trainer_volume_mounts.append(
            {"name": "scratch", "mountPath": scratch_root})
        volumes.append(
            {"name": "scratch",
             "persistentVolumeClaim": {"claimName": mcfg["pvc_scratch"]}})
    # Generic extra PVC mounts (checkpoints, additional shared datasets,
    # etc.) -- fully config-driven, no assumption about what they're for.
    for extra in mcfg.get("extra_volumes", []):
        vol_name = extra["name"]
        trainer_volume_mounts.append({
            "name": vol_name,
            "mountPath": extra["mount_path"],
            "readOnly": extra.get("read_only", False),
        })
        volumes.append({
            "name": vol_name,
            "persistentVolumeClaim": {
                "claimName": extra["claim_name"],
                "readOnly": extra.get("read_only", False),
            },
        })
    # MPS (Multi-Process Service) client wiring -- OPT-IN via mcfg["mps"] = true,
    # absent/false by default (byte-identical manifest for every machine that
    # hasn't configured it). Needed on cluster GPU nodes left in NVIDIA
    # `Exclusive_Process` compute mode with no per-pod permission to change it and
    # no MPS-sharing arbitration for a plain `nvidia.com/gpu: 1` request -- a pod's
    # first CUDA call can otherwise fail with `CUDA error: CUDA-capable device(s)
    # is/are busy or unavailable` even on a fully idle GPU (found live running
    # semantic-gaussian-particles on this cluster's A100 nodes; same root cause
    # independently documented in gs-slam-bench's gs_icp_slam/gaus_slam READMEs).
    # The fix is to wire the pod as a CLIENT of the cluster's own already-running
    # per-node MPS control daemon -- NOT to start a second, self-hosted daemon
    # inside the pod, which was tried elsewhere and found to non-deterministically
    # race the real one. Three parts, all needed together:
    #   1. A hostPath volume at /run/nvidia/mps (host) mounted at /mps (container).
    #   2. CUDA_MPS_PIPE_DIRECTORY=/mps/nvidia.com/gpu/pipe and
    #      CUDA_MPS_LOG_DIRECTORY=/mps/nvidia.com/gpu/log env vars on the trainer
    #      container, so any CUDA-using process in it becomes an MPS client
    #      automatically (no application code changes needed).
    #   3. Soft (preferred, not required) anti-affinity against other
    #      `app: ablator-job` pods on the same node: the MPS server's startup
    #      attempt grabs every physical GPU on a node at once and can crash-loop
    #      if one is legitimately busy from an unrelated job, which would also
    #      block a brand-new client asking only for the OTHER, idle GPU.
    # A job's first CUDA call can still transiently fail even with this wiring
    # correct, because the control daemon spawns its real server process LAZILY on
    # a client's first connection -- callers should retry the first CUDA call a
    # few times (this is workload-side, not something the manifest can fix).
    mps_enabled = bool(mcfg.get("mps"))
    trainer_env = [
        {"name": key, "value": value}
        for key, value in declaration_env.items()
    ]
    if job.get("requested_git_sha"):
        trainer_env.append({"name": "PYTHONDONTWRITEBYTECODE", "value": "1"})
    if mps_enabled:
        volumes.append({
            "name": "mps-root",
            "hostPath": {"path": "/run/nvidia/mps", "type": "DirectoryOrCreate"},
        })
        trainer_volume_mounts.append({"name": "mps-root", "mountPath": "/mps"})
        trainer_env.extend([{
            "name": "CUDA_MPS_PIPE_DIRECTORY",
            "value": "/mps/nvidia.com/gpu/pipe",
        }, {
            "name": "CUDA_MPS_LOG_DIRECTORY",
            "value": "/mps/nvidia.com/gpu/log",
        }])

    shm_size_gb = mcfg.get("shm_size_gb")
    if shm_size_gb:
        volumes.append({
            "name": "dshm",
            "emptyDir": {"medium": "Memory", "sizeLimit": f"{shm_size_gb}Gi"},
        })
        trainer_volume_mounts.append({"name": "dshm", "mountPath": "/dev/shm"})

    git_sync_repo_url = mcfg.get("git_sync_repo_url")
    git_sync_enabled = bool(git_sync_repo_url)
    init_containers: list[dict] = []
    if git_sync_enabled:
        workspace_path = cwd or mcfg.get("default_workdir", "/workspace")
        volumes.append({"name": "repo-src", "emptyDir": {}})
        trainer_volume_mounts.append(
            {"name": "repo-src", "mountPath": workspace_path, "readOnly": True})
        sha = job.get("requested_git_sha") or local_commit or "HEAD"
        # git_sync_http_secret_name takes precedence if both are somehow set --
        # rewrite the remote URL to embed the token as an x-access-token
        # credential, rather than relying on GIT_SSH_COMMAND (which needs an
        # ssh:// or git@ URL, not the https:// one this and git_sync_repo_url's
        # existing SSH path both otherwise assume unchanged).
        git_sync_http_secret_name = mcfg.get("git_sync_http_secret_name")
        remote_url_expr = git_sync_repo_url
        if git_sync_http_secret_name:
            stripped = git_sync_repo_url.removeprefix("https://")
            remote_url_expr = f"https://x-access-token:$(cat /etc/git-creds/token)@{stripped}"
        # git init + fetch-by-sha + checkout (rather than `git clone
        # --branch`) is the only way to pin an EXACT commit rather than a
        # moving branch head -- the whole point of this feature is running
        # the precise code the dispatching host had at submit time, which
        # may have since moved on the branch by the time the cluster
        # actually schedules the pod.
        clone_script = (
            "set -eu; "
            f"git init -q {workspace_path}; "
            f"cd {workspace_path}; "
            f"git remote add origin \"{remote_url_expr}\"; "
            f"git fetch --depth 1 origin {sha}; "
            "git checkout -q --detach FETCH_HEAD; "
            "git submodule sync --recursive; "
            "git -c protocol.file.allow=always submodule update --init --recursive --checkout; "
            f"test \"$(git rev-parse HEAD)\" = \"{sha}\"; "
            "test -z \"$(git status --porcelain --untracked-files=all)\"; "
            "git submodule foreach --quiet --recursive "
            "'test -z \"$(git status --porcelain --untracked-files=all)\"'; "
            "submodules_sha256=$(git submodule status --recursive | sha256sum | cut -d' ' -f1); "
            f"printf 'ABLATOR_SOURCE_V1 requested={sha} executed=%s "
            "ref=DETACHED dirty=false submodules_sha256=%s\\n' "
            "\"$(git rev-parse HEAD)\" \"$submodules_sha256\" > /dev/termination-log; "
            f"echo \"git-sync: checked out {sha} from {git_sync_repo_url}\""
        )
        init_container: dict = {
            "name": "git-sync",
            "image": mcfg.get("git_sync_image", "alpine/git:2.45.2"),
            "command": ["sh", "-c", clone_script],
            "volumeMounts": [{"name": "repo-src", "mountPath": workspace_path}],
        }
        git_sync_secret_name = mcfg.get("git_sync_secret_name")
        if git_sync_secret_name:
            # SSH deploy-key secret, mounted READ-ONLY into the init
            # container ONLY -- the trainer container never needs git
            # access and must never see it. Mirrors the existing
            # image_pull_secret pattern but as its own, narrower-scoped
            # secret (registry pull != repo read access).
            volumes.append({
                "name": "git-creds",
                "secret": {"secretName": git_sync_secret_name,
                          "defaultMode": 0o400},
            })
            init_container["volumeMounts"].append(
                {"name": "git-creds", "mountPath": "/etc/git-creds", "readOnly": True})
            init_container["env"] = [
                {"name": "GIT_SSH_COMMAND",
                 "value": "ssh -i /etc/git-creds/ssh-privatekey "
                          "-o StrictHostKeyChecking=no -o IdentitiesOnly=yes"},
            ]
        elif git_sync_http_secret_name:
            # HTTPS token secret (a plain Opaque secret with a single "token"
            # key) -- found live: some clusters require authenticated git
            # fetches even for a genuinely public GitHub repo (anonymous
            # `git fetch` from the cluster's egress got "could not read
            # Username for 'https://github.com'" while the SAME fetch worked
            # fine from a dispatching host with a cached `gh` credential
            # helper -- an ambient-auth difference between hosts, not a
            # private-repo requirement). A PAT already provisioned for
            # `image_pull_secret`/GHCR push (scoped to also cover repo read)
            # can often be reused here instead of provisioning a separate
            # SSH deploy key.
            volumes.append({
                "name": "git-creds",
                "secret": {"secretName": git_sync_http_secret_name,
                          "defaultMode": 0o400},
            })
            init_container["volumeMounts"].append(
                {"name": "git-creds", "mountPath": "/etc/git-creds", "readOnly": True})
        init_containers.append(init_container)
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
                    # Configurable so a job can target any scheduler; defaults
                    # to this cluster's KAI Scheduler.
                    "schedulerName": mcfg.get("scheduler_name", "kai-scheduler"),
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
                    # the resume point). A generic job with no such graceful-save
                    # behavior can simply lower this in its own mcfg.
                    "terminationGracePeriodSeconds":
                        mcfg.get("termination_grace_period_s", 150),
                    "restartPolicy": "Never",
                    # Optional: a public image (or one on a registry the
                    # cluster already trusts) needs no pull secret at all.
                    **({"imagePullSecrets": [{"name": mcfg["image_pull_secret"]}]}
                       if mcfg.get("image_pull_secret") else {}),
                    **({"initContainers": init_containers} if init_containers else {}),
                    # Soft anti-affinity against other ablator-job pods on the same
                    # node -- see the mps_enabled comment above for why this matters
                    # specifically for MPS, but it's requested unconditionally
                    # whenever MPS is on (not itself separately configurable) since
                    # it has no meaning/benefit without MPS.
                    **({"affinity": {"podAntiAffinity": {
                        "preferredDuringSchedulingIgnoredDuringExecution": [{
                            "weight": 100,
                            "podAffinityTerm": {
                                "labelSelector": {"matchExpressions": [
                                    {"key": "app", "operator": "In", "values": ["ablator-job"]},
                                ]},
                                "topologyKey": "kubernetes.io/hostname",
                            },
                        }],
                    }}} if mps_enabled else {}),
                    "containers": [{
                        "name": "trainer",
                        "image": image,
                        # Non-":latest" tags default to imagePullPolicy=IfNotPresent,
                        # which silently reuses a stale cached layer on whatever node
                        # a Job lands on after a fresh push -- found live: a rebuilt
                        # cuda-dev image (fixing a real ModuleNotFoundError) was
                        # ignored by a node that had cached the broken prior build.
                        "imagePullPolicy": "Always",
                        "workingDir": cwd or mcfg.get("default_workdir", "/workspace"),
                        "command": argv,
                        **({"env": trainer_env} if trainer_env else {}),
                        "resources": {
                            "requests": {
                                "cpu": mcfg.get("cpu_request", "4"),
                                "memory": mcfg.get("memory_request", "16Gi"),
                            },
                            "limits": {
                                "cpu": mcfg.get("cpu_limit", "8"),
                                "memory": mcfg.get("memory_limit", "32Gi"),
                                "nvidia.com/gpu": str(mcfg.get("gpu_count", 1)),
                            },
                        },
                        "volumeMounts": trainer_volume_mounts,
                    }],
                    "volumes": volumes,
                },
            },
        },
    }


def _log_stall_tracker(stall_after_s: float):
    """Return a `check(size, now) -> float | None` closure for detecting a
    stalled log file (pure decision logic, no I/O -- kept separate from
    _poll_k8s_job's real polling loop so it's directly unit-testable without
    mocking subprocess calls or the time module).

    Call `check(current_log_size_or_None, current_time)` on every poll:
    - Returns None (not stalled) whenever size has grown since the last
      call, or hasn't been observed long enough yet.
    - Returns the number of seconds since the log last grew, once that
      duration reaches `stall_after_s`.
    A None size (log unreadable, e.g. not yet created) never counts as
    growth -- treated the same as "no change" for stall-timing purposes.
    """
    state = {"last_size": -1, "last_growth_ts": None}

    def check(size, now):
        if state["last_growth_ts"] is None:
            state["last_growth_ts"] = now
        if size is not None and size > state["last_size"]:
            state["last_size"] = size
            state["last_growth_ts"] = now
            return None
        elapsed = now - state["last_growth_ts"]
        return elapsed if elapsed >= stall_after_s else None

    return check


def _k8s_execution_attestation(pods_payload: dict, expected_sha: str) -> dict:
    """Validate init-container source proof and capture actual pod identity."""
    items = pods_payload.get("items") or []
    if not items:
        return {
            "schema": "ablator.execution-attestation/v1",
            "verdict": "REJECTED",
            "source": None,
            "runtime": None,
            "error": "k8s source attestation unavailable: no job pod found",
        }
    pod = items[0]
    status = pod.get("status") or {}
    init_statuses = status.get("initContainerStatuses") or []
    git_status = next(
        (item for item in init_statuses if item.get("name") == "git-sync"), None
    )
    message = (((git_status or {}).get("state") or {}).get("terminated") or {}).get(
        "message"
    )
    fields: dict[str, str] = {}
    if isinstance(message, str) and message.startswith("ABLATOR_SOURCE_V1 "):
        for token in message.strip().split()[1:]:
            key, separator, value = token.partition("=")
            if separator:
                fields[key] = value
    source = {
        "requested_git_sha": fields.get("requested"),
        "executed_git_sha": fields.get("executed"),
        "ref": fields.get("ref"),
        "dirty": ({"true": True, "false": False}.get(fields.get("dirty", ""))),
        "submodules_sha256": fields.get("submodules_sha256"),
    }
    reasons: list[str] = []
    if not fields:
        reasons.append("git-sync termination message is missing or malformed")
    if source["requested_git_sha"] != expected_sha:
        reasons.append("init-container requested SHA differs from queue intent")
    if source["executed_git_sha"] != expected_sha:
        reasons.append("init-container executed SHA differs from queue intent")
    if source["ref"] != "DETACHED":
        reasons.append("init-container checkout is not detached")
    if source["dirty"] is not False:
        reasons.append("init-container checkout is dirty or unreadable")
    if not source["submodules_sha256"]:
        reasons.append("recursive submodule fingerprint is missing")

    trainer_statuses = status.get("containerStatuses") or []
    trainer = next(
        (item for item in trainer_statuses if item.get("name") == "trainer"), {}
    )
    runtime = {
        "pod": (pod.get("metadata") or {}).get("name"),
        "node": (pod.get("spec") or {}).get("nodeName"),
        "image": trainer.get("image"),
        "image_id": trainer.get("imageID"),
    }
    return {
        "schema": "ablator.execution-attestation/v1",
        "verdict": "REJECTED" if reasons else "ACCEPTED",
        "source": source,
        "runtime": runtime,
        "error": "; ".join(reasons) if reasons else None,
    }


def _capture_k8s_execution_attestation(
    name: str, ns: str, expected_sha: str,
) -> dict:
    pods = _kubectl([
        "get", "pods", "-n", ns, "-l", f"job-name={name}", "-o", "json"
    ], timeout=30)
    if pods.returncode != 0:
        return {
            "schema": "ablator.execution-attestation/v1",
            "verdict": "REJECTED",
            "source": None,
            "runtime": None,
            "error": "k8s source attestation query failed: "
                     f"{(pods.stderr or pods.stdout).strip()[:400]}",
        }
    try:
        payload = json.loads(pods.stdout)
    except json.JSONDecodeError as exc:
        return {
            "schema": "ablator.execution-attestation/v1",
            "verdict": "REJECTED",
            "source": None,
            "runtime": None,
            "error": f"k8s source attestation JSON is invalid: {exc}",
        }
    return _k8s_execution_attestation(payload, expected_sha)


def _poll_k8s_job(cfg: dict, job: dict, machine: str, mcfg: dict, tcfg: dict,
                  name: str, ns: str, log_path: str,
                  append: bool = False, q: Queue | None = None) -> tuple[str, int | None]:
    """Poll an existing (already-submitted) k8s Job to completion.

    Shared by the initial-submission path (`run_job_k8s`) and the
    restart-recovery re-attach path (`reconcile_stale_running`): both cases
    reduce to "there is a k8s Job `ns/name` already out there, watch it
    until it finishes". `append=True` is used for re-attach, since the
    Job (and its log history) predates this process and truncating the
    existing log file would destroy that history.
    """
    log_proc = None
    # Zombie-pod stall detection (2026-07-27): st.get("succeeded"/"failed") is
    # a JOB-level signal that only fires once a pod actually starts and then
    # finishes/fails -- a pod stuck forever in Pending/Init (e.g. the node's
    # kubelet/container-runtime wedged, an image-pull hang with no
    # ImagePullBackOff yet) never reaches either state, so this loop
    # previously spun forever (confirmed live: a splatograph scannetpp job
    # sat at Init:0/1 for 4h54m with zero runner-side detection, discovered
    # only by a human `kubectl get pods` check). Reuses the exact same
    # hung_after_min config knob and staleness semantics as the bare-metal
    # supervise() path (health.hung_after_s): if the job's own log file
    # (banner line already written above) hasn't grown in that long AND the
    # Job hasn't reached a terminal state, treat it as stuck regardless of
    # WHY (Init hang, image-pull hang, a training process that silently
    # wedged post-start) and kill it -- one general staleness check instead
    # of enumerating every specific k8s failure mode. Decision logic lives in
    # _log_stall_tracker (pure, no I/O) so it's directly unit-testable
    # without driving this function's real polling loop.
    _stall_after_s = healthmod.hung_after_s(cfg.get("queue", {}), job)
    _stall_tracker = _log_stall_tracker(_stall_after_s)
    rc: int | None = None
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
                try:
                    _cur_size = os.path.getsize(log_path)
                except OSError:
                    _cur_size = None
                _now = time.time()
                _stalled_s = _stall_tracker(_cur_size, _now)
                if _stalled_s is not None:
                    print(f"[ablator] {job['id']} k8s Job {ns}/{name} stalled: "
                          f"log has not grown in {_stalled_s / 60:.1f}min "
                          f"(threshold {_stall_after_s / 60:.1f}min) with no terminal "
                          f"Job status -- likely a stuck pod (Init/image-pull hang or "
                          f"a wedged process). Killing and reporting failed so the "
                          f"queue retries/quarantines it instead of hanging forever.",
                          flush=True)
                    job["_k8s_pod_stalled"] = True
                    _kubectl(["delete", "job", name, "-n", ns, "--ignore-not-found",
                             "--wait=false"])
                    return "failed", None
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
        expected_sha = job.get("requested_git_sha")
        if expected_sha:
            attestation = _capture_k8s_execution_attestation(name, ns, expected_sha)
            job["execution_attestation"] = attestation
            if attestation["verdict"] == "ACCEPTED":
                job["executed_git_sha"] = expected_sha
            elif rc == 0:
                rc = 1
                print(
                    f"[ablator] {job['id']} k8s source attestation rejected: "
                    f"{attestation['error']}", flush=True,
                )
            if q is not None:
                q.update(
                    job["id"],
                    executed_git_sha=job.get("executed_git_sha"),
                    execution_attestation=attestation,
                )
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
        sourcecheckout.job_git_target(
            job,
            required=(
                job.get("gradeability") == "GRADEABLE_DECLARED"
                or bool(tcfg.get("require_pinned_git"))
            ),
        )
        job = dict(job)
        source_identity = None
        if job.get("requested_git_sha"):
            if not mcfg.get("git_sync_repo_url"):
                raise sourcecheckout.SourcePreparationError(
                    "pinned k8s job requires machine git_sync_repo_url so the "
                    "pod can materialize the requested SHA")
            source_cwd = (cfg.get("types", {}).get(job.get("type", ""), {})
                          .get("cwd"))
            source_identity = sourcecheckout.validate_requested_revision_policy(
                cfg, job, machine, source_cwd)
            job["source_repo"] = source_identity
        argv, _env, cwd = render_command(
            tcfg, job, machine, include_protected_env=False
        )
    except (KeyError, TemplateError, sourcecheckout.SourcePreparationError) as e:
        print(f"[ablator] {job['id']} unrunnable: {e}", flush=True)
        if q is not None and isinstance(e, sourcecheckout.SourcePreparationError):
            q.update(job["id"], source_prepare_error=str(e))
        return "failed", None

    local_commit = _dispatch_host_commit(cfg, job)
    image_override = tcfg.get("image")
    preliminary_manifest = build_k8s_job_manifest(
        mcfg, job, argv, cwd, local_commit, image_override=image_override
    )
    from .external import capture_runner_provenance
    runner_provenance = capture_runner_provenance(cfg, machine)
    execution_receipt = receiptmod.build_prelaunch_receipt(
        cfg=cfg,
        job=job,
        machine=machine,
        type_config=tcfg,
        argv=argv,
        cwd=cwd,
        source_state=None,
        source_repo=source_identity,
        source_checkout=cwd if job.get("requested_git_sha") else None,
        source_lease_id=None,
        runner_provenance=runner_provenance,
    )
    trainer = preliminary_manifest["spec"]["template"]["spec"]["containers"][0]
    execution_receipt["launch"].update({
        "runtime": "kubernetes",
        "image": trainer["image"],
        "mounts": [
            {
                "name": mount["name"],
                "target": mount["mountPath"],
                "read_only": bool(mount.get("readOnly", False)),
            }
            for mount in trainer.get("volumeMounts", [])
        ],
        "namespace": mcfg["namespace"],
    })
    job["runner_provenance"] = runner_provenance
    job["execution_receipt"] = execution_receipt
    if q is not None:
        q.update(
            job["id"],
            source_repo=source_identity,
            requested_git_sha=job.get("requested_git_sha"),
            runner_provenance=runner_provenance,
            execution_receipt=execution_receipt,
            source_prepare_error=None,
        )
    manifest = build_k8s_job_manifest(
        mcfg, job, argv, cwd, local_commit, image_override=image_override
    )
    name = manifest["metadata"]["name"]
    ns = mcfg["namespace"]
    print(f"[ablator] running {job['id']} -> {job.get('model_path')} "
          f"(k8s Job {ns}/{name}, log {log_path})", flush=True)

    image_prov = provmod.check_image_drift(image_override or mcfg["image"], local_commit)
    if image_prov.get("warning"):
        print(f"[ablator] {image_prov['warning']}", flush=True)
    if q is not None:
        q.update(job["id"], image_provenance=image_prov,
                dispatch_host_commit=local_commit)

    with open(log_path, "w") as lf:
        lf.write(f"# {time.strftime('%Y-%m-%dT%H:%M:%S')} {job['id']} "
                 f"(k8s Job {ns}/{name})\n# {shlex.join(argv)}\n")
        lf.write(declarations.runner_log_banner(job) + "\n")
        lf.write(provmod.format_banner("k8s", image_prov) + "\n")

    apply = _kubectl(["apply", "-f", "-"], input_text=json.dumps(manifest))
    if apply.returncode != 0:
        print(f"[ablator] {job['id']} k8s apply failed: {apply.stderr}", flush=True)
        return "failed", None

    return _poll_k8s_job(
        cfg, job, machine, mcfg, tcfg, name, ns, log_path, append=True, q=q
    )


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
    prepared_source: sourcecheckout.PreparedSource | None = None
    execution_receipt: dict | None = None
    status = "failed"
    exit_code: int | None = None
    try:
        tcfg = cfgmod.type_cfg(cfg, job.get("type", ""), machine)
        sourcecheckout.job_git_target(
            job,
            required=(
                job.get("gradeability") == "GRADEABLE_DECLARED"
                or bool(tcfg.get("require_pinned_git"))
            ),
        )
        prepared_source = sourcecheckout.prepare_job_source(cfg, job, machine, tcfg)
        tcfg = prepared_source.type_config
        if prepared_source.checkout_path:
            job["_prepared_repo_cwd"] = prepared_source.checkout_path
            job.update({
                "source_checkout": prepared_source.checkout_path,
                "source_repo": prepared_source.source_repo,
                "requested_git_sha": prepared_source.requested_git_sha,
                "source_lease": {
                    "lease_id": prepared_source.lease.lease_id,
                    "checkout": prepared_source.lease.checkout,
                    "sidecar": prepared_source.lease.sidecar,
                },
            })

        # Render the semantic command before protected env injection. The
        # receipt hashes this form so ABLATOR_JOB_JSON can contain the receipt
        # without recursively hashing an argv that embeds ABLATOR_JOB_JSON.
        argv, _discard_env, cwd = render_command(
            tcfg, job, machine, include_protected_env=False
        )
        prov_state = capture_and_record_provenance(
            cfg, job, machine, cwd or os.getcwd(), q
        )
        executed_git_sha = sourcecheckout.verify_executed_provenance(job, prov_state)
        job["provenance"] = prov_state
        if executed_git_sha is not None:
            job["executed_git_sha"] = executed_git_sha
        from .external import capture_runner_provenance
        runner_provenance = capture_runner_provenance(cfg, machine)
        job["runner_provenance"] = runner_provenance
        receipt_source_state = prepared_source.state or {
            "commit": prov_state.get("commit"),
            "ref": prov_state.get("branch"),
            "dirty": prov_state.get("dirty"),
            "submodules": [],
        }
        execution_receipt = receiptmod.build_prelaunch_receipt(
            cfg=cfg,
            job=job,
            machine=machine,
            type_config=tcfg,
            argv=argv,
            cwd=cwd or os.getcwd(),
            source_state=receipt_source_state,
            source_repo=prepared_source.source_repo,
            source_checkout=prepared_source.checkout_path,
            source_lease_id=(
                prepared_source.lease.lease_id if prepared_source.lease else None
            ),
            runner_provenance=runner_provenance,
        )
        job["execution_receipt"] = execution_receipt
        if q is not None:
            q.update(
                job["id"],
                source_checkout=job.get("source_checkout"),
                source_repo=job.get("source_repo"),
                requested_git_sha=job.get("requested_git_sha"),
                executed_git_sha=job.get("executed_git_sha"),
                source_lease=job.get("source_lease"),
                provenance=prov_state,
                runner_provenance=runner_provenance,
                execution_receipt=execution_receipt,
                source_prepare_error=None,
            )
        # Re-render only after the local queue envelope contains actual
        # source/runner/receipt identity. This is the form the child executes.
        argv, env, cwd = render_command(tcfg, job, machine)
        check_checkout_drift(cfg, job, machine, prov_state, q)
        container_name = container_name_from_argv(argv)
        if container_name:
            # Clear only a leaked prior attempt with this deterministic name.
            force_remove_container(argv[0], container_name)

        print(f"[ablator] running {job['id']} -> {job.get('model_path')} "
              f"(log {log_path})", flush=True)
        with open(log_path, "w") as lf:
            lf.write(f"# {time.strftime('%Y-%m-%dT%H:%M:%S')} {job['id']}\n"
                     f"# cwd={cwd or os.getcwd()}\n# {shlex.join(argv)}\n")
            lf.write(declarations.runner_log_banner(job) + "\n")
            lf.write(provmod.format_banner("bare-metal", prov_state) + "\n")
            preflight_line = output_folder_preflight(str(job.get("model_path", "")), cwd)
            lf.write(preflight_line + "\n")
            print(preflight_line, flush=True)
            lf.flush()
            proc = subprocess.Popen(argv, env=env, cwd=cwd,
                                    stdout=lf, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            override = supervise(cfg, job, proc, cwd or os.getcwd(), q, argv=argv,
                                 machine=machine)
        exit_code = proc.returncode
        if override is not None:
            # Control-triggered stop/skip/requeue (or a lane preemption)
            # already returned an explicit terminal/backoff status above in
            # supervise() — the exit code of the killed subprocess is never
            # consulted here, so a manual kill can never read as "done".
            status = override
        elif exit_code == 0 and _require_result_artifact(cfg, tcfg):
            h = healthmod.job_health(job, cwd or os.getcwd(), cfg.get("queue", {}),
                                     process_alive=False)
            if h["state"] != "done":
                print(f"[ablator] {job['id']} exited 0 but no completion "
                      f"artifact found (result_glob unmatched, state="
                      f"{h['state']!r}) — treating as failed, not done",
                      flush=True)
                status = "failed"
            else:
                status = "done"
        else:
            status = "done" if exit_code == 0 else "failed"
    except (KeyError, TemplateError, sourcecheckout.SourcePreparationError) as e:
        print(f"[ablator] {job['id']} unrunnable: {e}", flush=True)
        if q is not None and isinstance(e, sourcecheckout.SourcePreparationError):
            q.update(job["id"], source_prepare_error=str(e))
    except Exception as e:
        print(f"[ablator] {job['id']} crashed: {e}", flush=True)
    finally:
        if prepared_source is not None and prepared_source.checkout_path:
            try:
                final_state = sourcecheckout.inspect_checkout_state(
                    prepared_source.checkout_path
                )
                attestation = receiptmod.build_final_attestation(
                    execution_receipt or {
                        "source": {
                            "requested_git_sha": job.get("requested_git_sha"),
                            "submodules": (prepared_source.state or {}).get(
                                "submodules", []
                            ),
                        }
                    },
                    source_state=final_state,
                )
            except Exception as exc:
                attestation = receiptmod.build_final_attestation(
                    execution_receipt or {"source": {}}, error=str(exc)
                )
            if attestation["verdict"] != "ACCEPTED":
                status = "failed"
                print(
                    f"[ablator] {job['id']} final source attestation rejected: "
                    f"{attestation['error']}", flush=True,
                )
            if q is not None:
                q.update(job["id"], execution_attestation=attestation)
            try:
                sourcecheckout.release_source(prepared_source)
            except sourcecheckout.SourcePreparationError as exc:
                status = "failed"
                if q is not None:
                    q.update(job["id"], source_release_error=str(exc))
                print(
                    f"[ablator] {job['id']} source lease release failed: {exc}",
                    flush=True,
                )
    return status, exit_code


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

    Requeue/crash handling below is only ever applied to jobs with
    claimed_by == machine — a different machine's claim is left to that
    behavior, self-healing on its own restart. But a job with a real
    completion artifact is marked done regardless of which machine
    claimed it (see the cross-machine dead-man's-switch pass below) —
    that transition can never cause a duplicate launch, so it does not
    need to wait for the owning machine's own runner to restart.

    Never requeues while the machine's busy-guards say something is still
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
    is_k8s = cfgmod.machine_cfg(cfg, machine).get("backend") == "k8s"
    mcfg = cfgmod.machine_cfg(cfg, machine) if is_k8s else None
    grace_s = cfg.get("queue", {}).get("reconcile_grace_s", DEFAULT_RECONCILE_GRACE_S)
    for job in q.read():
        if job.get("status") != "running":
            continue
        claimed_by = job.get("claimed_by")

        if claimed_by != machine:
            # Cross-machine dead-man's-switch (found 2026-08-22,
            # pixel10a_champ_champion): the OWN-claim self-heal below only
            # ever runs from within a freshly (re)started/idle-ticking
            # run_loop for the machine that claimed the job. If that
            # machine's runner process dies outright (crashes silently,
            # gets killed, or the host itself never restarts it) and
            # nothing ever launches `ablator run`/`ablator start` there
            # again, that machine's own self-heal never gets a chance to
            # fire — confirmed live: main's runner process died mid-job
            # with no traceback, the container it had launched kept
            # training independently and finished cleanly with a real
            # completion artifact, and the ledger entry stayed stuck at
            # status="running" (serving an increasingly stale cached
            # health snapshot) for over 24h until a human ran `ablator
            # stop` by hand. A completion artifact is authoritative
            # regardless of which machine claimed the job or whether that
            # machine's own runner is still alive to notice it, and
            # marking it done here can never race a duplicate dispatch
            # (unlike the requeue-to-pending transition below, this one
            # only ever moves a job further away from being claimable).
            # Still respect the grace window so a job claimed moments ago
            # (result_glob/log not written yet) isn't misread — though
            # that can only ever produce a false "not done" here, never a
            # false "done".
            age_s = _claimed_age_s(job)
            if age_s is not None and age_s < grace_s:
                continue
            base_dir = _job_base_dir(cfg, job, claimed_by or machine)
            h = healthmod.job_health(job, base_dir, cfg.get("queue", {}),
                                     process_alive=False)
            if h["state"] == "done":
                print(f"[ablator] reconcile: {job['id']} (claimed by "
                      f"{claimed_by!r}) has a completion artifact but is "
                      "stuck at 'running' -- that machine's own runner may "
                      f"be dead with nothing left to notice; marking done "
                      f"from {machine}'s idle tick", flush=True)
                q.finish(job["id"], "done", health=h, reconciled=True,
                        reconciled_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
            continue

        if busy:
            continue

        # Grace window (issue splatograph#295): a job that was claimed only
        # moments ago has not necessarily written a log/artifact yet -- a
        # container can take real wall-clock time to pull/start before its
        # own health-check artifacts exist. job_health(process_alive=False)
        # below has no live process handle to consult (this runner process
        # just (re)started) and, with no log file yet, unconditionally
        # reports state="crashed" for ANY job with no log -- indistinguishable
        # from a job seconds into its container start. Confirmed live
        # 2026-08-12: a job claimed at 18:18:16 was reconciled to pending
        # and re-claimed/relaunched at 18:21:15 (2m59s later) while its first
        # attempt's container was still starting, producing two concurrent
        # training processes that OOM'd each other. `--name`-based busy-guard
        # detection (see _ensure_container_name) closes most of this window,
        # but not the sliver between process launch and the container
        # actually appearing in `docker ps` -- skip reconciling (leave the
        # job at 'running', untouched) until claimed_at is older than
        # `[queue] reconcile_grace_s` (default 180s), regardless of health
        # state or busy-guard result for this tick.
        age_s = _claimed_age_s(job)
        if age_s is not None and age_s < grace_s:
            print(f"[ablator] reconcile: {job['id']} claimed {age_s:.0f}s ago, "
                  f"still within the {grace_s:.0f}s startup grace window -- "
                  "not reconciling this tick", flush=True)
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
                                        ns, log_path, append=True, q=q)

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


DEFAULT_RECONCILE_GRACE_S = 180.0


def _claimed_age_s(job: dict, now: float | None = None) -> float | None:
    """Seconds since `job['claimed_at']` was stamped, or None if unset/
    unparseable. `claimed_at` is always written by Queue.claim_next() via
    `_now()` (`time.strftime('%Y-%m-%dT%H:%M:%S')`, local time, matching
    `time.strptime` below)."""
    claimed_at = job.get("claimed_at")
    if not claimed_at:
        return None
    try:
        claimed_epoch = time.mktime(time.strptime(claimed_at, "%Y-%m-%dT%H:%M:%S"))
    except (TypeError, ValueError):
        return None
    return (time.time() if now is None else now) - claimed_epoch


DEFAULT_K8S_MAX_CONCURRENT = 4

# How fresh a heartbeat must be to trust its "idle" state as "about to try
# claiming for itself this tick / very soon" — bigger than any single poll
# interval so a slightly-slow tick doesn't look stale, small enough that a
# genuinely dead/hung runner stops being treated as idle within a couple of
# minutes (fail OPEN toward k8s dispatch, i.e. never toward starving k8s).
IDLE_HEARTBEAT_FRESH_S = 120


def _bare_metal_machines(cfg: dict) -> list[str]:
    return [name for name, m in cfg.get("machines", {}).items()
            if m.get("backend") != "k8s"]


def _other_idle_baremetal(cfg: dict, machine: str) -> bool:
    """True if some OTHER bare-metal machine looks idle-with-capacity right
    now (fresh heartbeat, state starts with "idle", not paused) — i.e. it's
    about to attempt claim_next() for itself on its own very next tick.

    Used to make run_loop defer claiming machine="any" jobs for k8s this
    tick (see only_pinned= in Queue.claim_next), giving that idle bare-metal
    machine first shot instead of losing every "any" job to whichever
    process's k8s-fill loop happens to run first. Deliberately narrow: only
    fires when another machine is BOTH idle and unpaused, so it can't defer
    "any" work into a black hole — the deferred-to machine reassesses and
    claims on its own next poll, or this check stops returning True and k8s
    resumes claiming "any" jobs the very next tick.
    """
    qdir = os.path.dirname(cfgmod.queue_path(cfg))
    now = time.time()
    for name in _bare_metal_machines(cfg):
        if name == machine:
            continue
        path = os.path.join(qdir, f"heartbeat_{name}.txt")
        try:
            with open(path) as f:
                line = f.read().strip()
        except OSError:
            continue
        m = re.search(r"epoch=(\d+(?:\.\d+)?)\s+state=(\S+)", line)
        if not m:
            continue
        epoch, state = float(m.group(1)), m.group(2)
        if now - epoch > IDLE_HEARTBEAT_FRESH_S:
            continue  # stale — don't trust it, fail open toward k8s
        if not state.startswith("idle"):
            continue
        if is_paused(cfgmod.queue_path(cfg), name):
            continue  # reports idle but can never claim — don't defer to it
        return True
    return False


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


def _record_runner_provenance(
    cfg: dict, job: dict, job_machine: str, q: Queue,
) -> None:
    """Persist the Ablator process/config identity that will execute a job."""
    try:
        from .external import capture_runner_provenance
        runner_provenance = capture_runner_provenance(cfg, job_machine)
    except Exception as exc:
        runner_provenance = {
            "schema": "ablator.runner-provenance/v1",
            "machine": job_machine, "identity_complete": False,
            "error": repr(exc)}
    q.update(job["id"], runner_provenance=runner_provenance)


def _dispatch_and_finalize(cfg: dict, machine: str, job: dict, job_machine: str,
                           q: Queue, run_fn=None,
                           runner_provenance_recorded: bool = False) -> str:
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
    # Persist the identity of the actual Ablator process/config that
    # executes this job. This is distinct from workload checkout provenance
    # and is required for trustworthy cross-machine experiment comparison.
    if not runner_provenance_recorded:
        _record_runner_provenance(cfg, job, job_machine, q)
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
    if k8s_dispatch_enabled(cfg, machine):
        k8s_machines = [
            name for name, m in cfg.get("machines", {}).items()
            if m.get("backend") == "k8s"
        ]
    else:
        # This runner must NEVER act as a k8s dispatcher -- either
        # explicitly configured off (k8s_dispatch = false, e.g. r9700,
        # which has no route to the cluster API server) or kubectl isn't
        # even installed here. Zero k8s-related code paths are touched
        # below: no _kubectl() calls, no reconcile_stale_running() for a
        # k8s machine, no claiming of k8s-pinned/"any" jobs for k8s.
        k8s_machines = []
        reason = ("k8s_dispatch=false in config"
                  if not cfg.get("machines", {}).get(machine, {}).get("k8s_dispatch", True)
                  else "kubectl not found on PATH")
        print(f"[ablator] k8s dispatch DISABLED for {machine} ({reason}) -- "
              f"bare-metal-only mode", flush=True)
    dispatch_machines = [machine] + k8s_machines
    print(f"[ablator] runner on {machine} watching {q.path} "
          f"(dispatching for: {', '.join(dispatch_machines)})", flush=True)
    inflight = _K8sInflight()
    reconcile_stale_running(cfg, machine, q)
    for k8s_name in k8s_machines:
        reconcile_stale_running(cfg, k8s_name, q, busy=False, inflight=inflight)

    # Self-drift check: is THIS host's own ablator installation behind
    # origin/main? (Different from urgent_fixes.py, which checks the
    # TARGET repo, splatograph.) Caught live 2026-07-07: r9700 was 9
    # commits behind, silently, for an unknown period. Runs once at
    # startup unconditionally (this is exactly when a stale checkout
    # matters most — right before this process starts making dispatch
    # decisions with it), then periodically since runner processes stay up
    # for hours. Dispatched on a daemon thread — `git fetch` against a
    # remote can be slow/blocked (exactly the offline-r9700 case this is
    # meant to catch), and this check is purely informational, so it must
    # never delay bare-metal/k8s dispatch decisions the way a blocking
    # call in the hot startup/loop path would. Never raises regardless —
    # see self_check.py for the never-raises contract.
    def _bg_self_check():
        selfcheckmod.run_self_check(cfg, machine)
        refresh_idle_provenance(cfg, machine)
    threading.Thread(target=_bg_self_check, daemon=True,
                     name="ablator-self-check").start()
    last_self_check = time.monotonic()
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
        if now - last_self_check > SELF_CHECK_INTERVAL_S:
            threading.Thread(target=_bg_self_check, daemon=True,
                             name="ablator-self-check").start()
            last_self_check = now
        try:
            inflight.reap()
            baremetal_busy = resources.machine_busy(cfg, machine)
            if baremetal_busy:
                # Local GPU busy with this machine's own bare-metal job only
                # blocks steps 1/3 (bare-metal claim/run) below -- it must
                # NOT block step 2 (k8s-fill). This runner is the sole
                # dispatcher for every k8s-backend machine (e.g.
                # a100cluster); k8s dispatch never touches this machine's
                # own GPU, so gating it behind local GPU busy starved the
                # cluster of new work for the entire duration of every
                # bare-metal job this process ran. Found live 2026-07-25:
                # an a100cluster-pinned job sat "pending" for 10+ minutes
                # while main ran back-to-back bare-metal jobs, with zero
                # k8s dispatch log lines the entire time.
                write_heartbeat(cfg, machine,
                                f"busy-wait k8s_inflight={inflight.total()}")
            else:
                write_heartbeat(cfg, machine, f"idle k8s_inflight={inflight.total()}")

            # Steps 0/1 (bare-metal self-heal, urgent-fix gate, bare-metal
            # claim) only make sense/are only safe while this machine's own
            # GPU is idle -- in particular, enforce_urgent_fixes can do a
            # local git pull, which must never yank code out from under a
            # running bind-mounted bare-metal job. None of this touches k8s
            # dispatch (step 2 below), which always runs regardless of
            # baremetal_busy.
            job = None
            currency_ok = True
            if not baremetal_busy:
                # Re-run bare-metal self-heal on every idle tick, not just at
                # startup. The startup-only call can legitimately skip a job
                # that predates this process (busy=True because that prior
                # process's own job was still genuinely training) and then
                # never get a second chance -- confirmed live 2026-07-07:
                # frdeskw01main_fr1desk_w01_plus_admission_fix stayed stuck at
                # status="running" for 2h45m because the runner restarted
                # 2.5 minutes before that in-flight job actually finished, the
                # startup call correctly deferred (busy=True at that instant),
                # and nothing ever retried once the machine went idle seconds
                # later. reconcile_stale_running() is a no-op once an entry has
                # already been reconciled (it only touches status=="running"
                # entries), so calling it every idle tick is cheap in steady
                # state -- one extra q.read() scan, not a busy-poll.
                reconcile_stale_running(cfg, machine, q, busy=False)

                # -1. Pause re-validation: if this machine is currently
                # paused, re-run the SPECIFIC check that caused it (never a
                # blind timer/TTL -- see pause_revalidation.py and
                # splatograph issue #629) and clear the flag only if that
                # check now passes. Human-set pauses (`ablator pause`) and
                # any category without a registered re-checker are left
                # untouched. Only reached once machine_busy() above already
                # confirmed this machine's own GPU is idle -- the same
                # scoping the urgent-fix gate below relies on -- so
                # auto-clearing here can never race a foreign or
                # bind-mounted job for the GPU; it only makes claim_next()
                # eligible again on a subsequent, still fully-guarded tick.
                revalidate_pause(cfg, machine, q)

                # 0. Urgent-fix currency gate: verify this dispatcher's own
                # checkout has every registered urgent fix before dispatching
                # ANYTHING this tick -- both the k8s path (git-sync pins to
                # this host's HEAD SHA) and the bare-metal path (live bind
                # mount) run whatever is on disk here right now. Only reached
                # once machine_busy() above already confirmed idle, so an
                # auto-pull here can never yank code out from under a running
                # bind-mounted job. See urgent_fixes.py for the full incident
                # writeup and design rationale.
                currency_ok = enforce_urgent_fixes(cfg, machine, q)

                # 1. Claim (non-blocking) this runner's own bare-metal job
                # FIRST, before k8s claiming — this is what actually gives an
                # idle bare-metal machine first shot at machine="any" jobs
                # instead of losing every race to whichever process's
                # k8s-fill loop happens to run first. Running it is deferred
                # to step 3 (it's blocking) so it doesn't starve k8s
                # concurrency in the meantime — see the design comment above
                # for why that split matters.
                job = q.claim_next(
                    machine, can_run=make_can_run(cfg, machine),
                    allow_pinned_git_while_paused=not currency_ok,
                )

            # Preserve bare-metal priority through the new runner-provenance
            # write as well as through queue claiming.  If a k8s thread wins
            # the queue lock first, its provenance/finalization can otherwise
            # delay this already-claimed local job until the k8s work ends.
            baremetal_provenance_recorded = False
            if job is not None:
                _record_runner_provenance(cfg, job, machine, q)
                baremetal_provenance_recorded = True

            # 2. Fill k8s concurrency slots — non-blocking: each claimed job
            # is handed to a background thread and this loop moves straight
            # on to running the bare-metal job (if any) below without
            # waiting for it. When another bare-metal machine looks
            # idle-with-capacity right now (fresh heartbeat), defer
            # claiming machine="any" jobs for k8s this tick so that machine
            # gets first shot on its own next poll instead of losing every
            # "any" job to this process's k8s dispatch purely because it
            # gets to call claim_next() more often. Jobs explicitly pinned
            # to a k8s machine are never affected by this.
            defer_any = _other_idle_baremetal(cfg, machine)
            for k8s_name in k8s_machines:
                cap = _k8s_max_concurrent(cfg, k8s_name)
                base_can_run = make_can_run(cfg, k8s_name)
                if currency_ok:
                    can_run = base_can_run
                else:
                    can_run = lambda candidate, inner=base_can_run: (
                        bool(candidate.get("requested_git_sha")) and inner(candidate)
                    )
                while inflight.count(k8s_name) < cap:
                    kjob = q.claim_next(k8s_name, can_run=can_run,
                                        only_pinned=defer_any)
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

            # 3. Run (serially, blocking) the bare-metal job claimed in step
            # 1, if any.
            if job is None:
                if once:
                    inflight.join_all()
                    return
                time.sleep(IDLE_POLL_S)
                continue
            write_heartbeat(cfg, machine, f"running:{job['id']}")
            status = _dispatch_and_finalize(
                cfg, machine, job, machine, q,
                runner_provenance_recorded=baremetal_provenance_recorded,
            )
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
