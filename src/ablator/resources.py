"""Pluggable machine-busy checks.

Two independent guard families, both of which must say "idle" before the
runner claims a job:

1. GPU-utilization guard — auto-detects rocm-smi / amd-smi / nvidia-smi
   and treats the machine as busy only if BOTH of two samples ~gap
   seconds apart exceed the threshold (single spikes from a desktop
   compositor must not block claiming). Threshold from config
   [resources] gpu_busy_pct (default 20), env ABLATOR_GPU_BUSY_PCT
   overrides. A failed probe means "not busy by this guard" — the
   configured guards below remain the backstop.

2. Configured guards — per-machine list of generic checks in
   [machines.<name>] busy_guards: each entry runs a command and reports
   busy if its output contains a substring (empty contains = any
   non-empty output). This expresses "podman ps shows splat_train",
   "pgrep -f chain.sh", etc., without ablator knowing any names.

3. GPU memory guard — reads the machine's actual GTT/VRAM usage from
   sysfs (paths configured per-machine under [resources.mem_budgets.<m>])
   and treats the machine as busy if usage is at/above
   [resources] mem_dispatch_busy_pct. This is deliberately a LOWER,
   more conservative threshold than the kill-threshold used by
   supervise()'s memory guard (mem_kill_danger_pct) -- it exists to stop
   a NEW job from being dispatched onto a machine that is already
   sitting dangerously close to its memory ceiling (e.g. a leaked
   container from a crashed prior job silently holding GTT memory for
   hours, invisible to `podman ps`/`free -h`). See runner.supervise()
   for the in-flight kill-threshold check.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import UTC, datetime

DEFAULT_GPU_BUSY_PCT = 20.0
DEFAULT_SAMPLE_GAP_S = 3.0


# --------------------------------------------------------------- GPU util

def _run(cmd: list[str], timeout: float = 15) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout if r.returncode == 0 else None
    except Exception:
        return None


def sample_gpu_util() -> float | None:
    """One GPU-utilization sample in percent, or None if unprobeable.

    Tries rocm-smi (JSON then plain), amd-smi, nvidia-smi. Never raises.
    """
    out = _run(["rocm-smi", "--showuse", "--json"])
    if out and out.strip():
        try:
            data = json.loads(out)
            for card in data.values():
                if isinstance(card, dict):
                    for k, v in card.items():
                        if "use" in k.lower():
                            return float(str(v).rstrip("% "))
        except Exception:
            pass
    out = _run(["rocm-smi", "--showuse"])
    if out:
        for line in out.splitlines():
            if "GPU use" in line:
                try:
                    return float(line.split(":")[-1].strip().rstrip("%"))
                except ValueError:
                    pass
    out = _run(["amd-smi", "metric", "--usage"])
    if out:
        for line in out.splitlines():
            if "GFX_ACTIVITY" in line.upper():
                try:
                    return float(line.split(":")[-1].strip().rstrip("% "))
                except ValueError:
                    pass
    out = _run(["nvidia-smi", "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits"])
    if out:
        try:
            return max(float(x) for x in out.split() if x.strip())
        except ValueError:
            pass
    return None


def _cpu_percent() -> float | None:
    try:
        return float(os.getloadavg()[0])
    except OSError:
        return None


def _memory_gb() -> tuple[float | None, float | None]:
    try:
        values = dict(line.split(":", 1) for line in open("/proc/meminfo") if ":" in line)
        total = float(values["MemTotal"].split()[0]) / 1048576
        available = float(values["MemAvailable"].split()[0]) / 1048576
        return total - available, total
    except (OSError, KeyError, ValueError):
        return None, None


def machine_telemetry_snapshot(cfg: dict, machine: str, run=_run,
                               cpu_sampler=_cpu_percent, memory_sampler=_memory_gb,
                               gpu_sampler=sample_gpu_util) -> dict:
    """Best-effort machine state: token-only Beszel CLI, then local probes.

    Absence of Beszel, credentials, or a CLI binary never blocks a runner;
    consumers must retain the returned source instead of treating fallback
    values as fleet-authoritative telemetry.
    """
    mcfg = cfg.get("machines", {}).get(machine, {})
    resource_cfg = cfg.get("resources", {})
    provider = resource_cfg.get("telemetry_provider", "auto")
    now = datetime.now(UTC).isoformat()
    if provider != "local":
        cli = resource_cfg.get("beszel_cli", os.environ.get("BESZEL_CLI", "clusterstat"))
        system = mcfg.get("beszel_system", machine)
        output = run([cli, "--json", system])
        if output:
            try:
                payload = json.loads(output)
                systems = payload.get("systems", [])
                item = systems[0] if len(systems) == 1 and isinstance(systems[0], dict) else None
                if item:
                    return {"schema": "ablator.machine-telemetry/v1", "captured_at": now,
                            "source": "beszel", "machine": machine, "system": item.get("name"),
                            "status": item.get("status"), "cpu_percent": item.get("cpu_percent"),
                            "memory_percent": item.get("memory_percent"), "gpu_percent": item.get("gpu_percent"),
                            "vram_used_gb": item.get("vram_used_gb"), "vram_total_gb": item.get("vram_total_gb"),
                            "gpu_power_watts": item.get("gpu_power_watts")}
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
    memory_used, memory_total = memory_sampler()
    return {"schema": "ablator.machine-telemetry/v1", "captured_at": now,
            "source": "local-fallback", "machine": machine, "system": None, "status": "unknown",
            "cpu_percent": cpu_sampler(), "memory_percent": None, "memory_used_gb": memory_used,
            "memory_total_gb": memory_total, "gpu_percent": gpu_sampler(),
            "vram_used_gb": None, "vram_total_gb": None, "gpu_power_watts": None}


def gpu_util_busy(cfg: dict | None = None, sampler=sample_gpu_util,
                  sleep=time.sleep) -> bool:
    """True only if BOTH of two samples ~gap seconds apart exceed threshold."""
    res = (cfg or {}).get("resources", {})
    threshold = res.get("gpu_busy_pct", DEFAULT_GPU_BUSY_PCT)
    gap = res.get("sample_gap_s", DEFAULT_SAMPLE_GAP_S)
    env = os.environ.get("ABLATOR_GPU_BUSY_PCT")
    if env is not None:
        try:
            threshold = float(env)
        except ValueError:
            pass
    first = sampler()
    if first is None or first <= threshold:
        return False
    sleep(gap)
    second = sampler()
    return second is not None and second > threshold


# --------------------------------------------------------- config guards

def guard_busy(guard: dict) -> bool:
    """One configured guard: {command = [...], contains = "substr"}.

    Busy if the command's stdout contains `contains`; with contains
    empty/absent, busy if stdout is non-empty. Failed commands are idle.
    """
    out = _run(list(guard.get("command", [])))
    if out is None:
        return False
    needle = guard.get("contains", "")
    return (needle in out) if needle else bool(out.strip())


def machine_busy(cfg: dict, machine: str, sampler=sample_gpu_util,
                 sleep=time.sleep, mem_sampler=None) -> bool:
    """Combined busy check for this machine (GPU util OR any config guard OR
    GPU memory usage already above the pre-dispatch danger threshold)."""
    if gpu_util_busy(cfg, sampler=sampler, sleep=sleep):
        return True
    guards = cfg.get("machines", {}).get(machine, {}).get("busy_guards", [])
    if any(guard_busy(g) for g in guards):
        return True
    return gpu_mem_busy(cfg, machine, mem_sampler=mem_sampler)


# ------------------------------------------------------------ GPU memory

DEFAULT_MEM_DISPATCH_BUSY_PCT = 70.0
DEFAULT_MEM_KILL_DANGER_PCT = 90.0
DEFAULT_MEM_KILL_GRACE_CYCLES = 3


def _read_sysfs_int(path: str) -> int | None:
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def sample_gpu_mem_pct(cfg: dict, machine: str, reader=_read_sysfs_int) -> float | None:
    """Current GPU memory usage for `machine` as a percentage of its
    configured budget, or None if unconfigured/unreadable.

    Budgets are per-machine under [resources.mem_budgets.<machine>]:
      used_path  = sysfs path to the current-usage counter (bytes)
      total_path = sysfs path to the total-budget counter (bytes), or
      total_bytes = a literal fallback if no live total_path exists.
    A machine with no [resources.mem_budgets.<machine>] entry (e.g. a
    k8s-backend cluster where GPU isolation is per-pod, not host-level)
    always returns None -- callers must treat None as "unknown", never
    as "not busy"/"safe" by assumption.
    """
    budgets = cfg.get("resources", {}).get("mem_budgets", {}).get(machine)
    if not budgets:
        return None
    used = reader(budgets.get("used_path", ""))
    total = reader(budgets.get("total_path", "")) if budgets.get("total_path") else None
    if total is None:
        total = budgets.get("total_bytes")
    if used is None or not total:
        return None
    return 100.0 * used / total


def gpu_mem_busy(cfg: dict, machine: str, mem_sampler=None) -> bool:
    """True if current GPU memory usage is at/above mem_dispatch_busy_pct."""
    mem_sampler = mem_sampler or (lambda: sample_gpu_mem_pct(cfg, machine))
    pct = mem_sampler()
    if pct is None:
        return False
    threshold = cfg.get("resources", {}).get(
        "mem_dispatch_busy_pct", DEFAULT_MEM_DISPATCH_BUSY_PCT)
    return pct >= threshold


# ---------------------------------------------------------- capabilities

def images_present(runtime: str, images: list[str]) -> bool:
    """True if every image is already present locally (never pulls/builds)."""
    for img in images:
        out = _run([runtime, "images", "-q", img])
        if not (out or "").strip():
            return False
    return True
