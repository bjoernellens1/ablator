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
"""
from __future__ import annotations

import json
import os
import subprocess
import time

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
                 sleep=time.sleep) -> bool:
    """Combined busy check for this machine (GPU util OR any config guard)."""
    if gpu_util_busy(cfg, sampler=sampler, sleep=sleep):
        return True
    guards = cfg.get("machines", {}).get(machine, {}).get("busy_guards", [])
    return any(guard_busy(g) for g in guards)


# ---------------------------------------------------------- capabilities

def images_present(runtime: str, images: list[str]) -> bool:
    """True if every image is already present locally (never pulls/builds)."""
    for img in images:
        out = _run([runtime, "images", "-q", img])
        if not (out or "").strip():
            return False
    return True
