"""Passive failure-classification for queue jobs.

Pure diagnosis, no side effects: given a job record, its log tail, exit
code and some machine-context flags, classify_failure() returns a
category + evidence + confidence + suggested_action. Callers (runner)
decide what to actually do about it (requeue/quarantine/pause).

Categories, roughly in priority order when multiple could match:
  disk_full, image_missing, gpu_busy_conflict, oom_killed, scene_missing,
  network_transient, code_error, unknown

Marker lists are config-driven: built-in defaults below, overridable per
category via the host TOML config's [error_patterns] table, e.g.

  [error_patterns]
  image_missing = ["pull access denied", "manifest unknown", "custom marker"]

A category present in the config REPLACES the built-in list for that
category; categories not mentioned keep their defaults. Load with
patterns_from_config(cfg) and pass the result as classify_failure(...,
patterns=...).
"""
from __future__ import annotations

import re

SUGGESTED_ACTION = {
    "disk_full": "pause_queue_alert",
    "image_missing": "skip_permanently_this_machine",
    "gpu_busy_conflict": "requeue_backoff_5min",
    "oom_killed": "requeue_once_needs_review",
    "scene_missing": "quarantine_no_retry",
    "network_transient": "requeue_backoff_2min",
    "code_error": "quarantine_code_fix_needed",
    "unknown": "retry_once_then_quarantine",
}

DEFAULT_PATTERNS: dict[str, tuple[str, ...]] = {
    "image_missing": (
        "pull access denied",
        "manifest unknown",
        "no such image",
        "unable to find image",
    ),
    "network_transient": (
        "temporary failure in name resolution",
        "connection refused",
        "could not resolve host",
        "network is unreachable",
        "timed out",
        "timeout",
        "stale file handle",
    ),
    "gpu_oom": (
        "cuda out of memory",
        "hip out of memory",
        "out of memory",
        "device busy",
        "device or resource busy",
    ),
    "disk_full": ("no space left on device",),
}

# Backward-compatible module-level aliases (used by callers/tests that
# import the tuples directly rather than going through patterns_from_config).
_IMAGE_MISSING_MARKERS = DEFAULT_PATTERNS["image_missing"]
_NETWORK_MARKERS = DEFAULT_PATTERNS["network_transient"]
_GPU_OOM_MARKERS = DEFAULT_PATTERNS["gpu_oom"]
_DISK_FULL_MARKER = DEFAULT_PATTERNS["disk_full"][0]

_TRACEBACK_RE = re.compile(r"Traceback \(most recent call last\)")


def patterns_from_config(cfg: dict | None) -> dict[str, tuple[str, ...]]:
    """Merge [error_patterns] overrides from a loaded TOML/JSON config on
    top of DEFAULT_PATTERNS. A category present in cfg replaces its default
    list wholesale; absent categories keep the built-in markers."""
    merged = dict(DEFAULT_PATTERNS)
    overrides = (cfg or {}).get("error_patterns") or {}
    for category, markers in overrides.items():
        if category in merged:
            merged[category] = tuple(markers)
    return merged


def _snippet(text: str, marker: str, width: int = 160) -> str:
    """Return a short evidence snippet centered on the (case-insensitive) marker."""
    low = text.lower()
    idx = low.find(marker.lower())
    if idx == -1:
        return text[:width].strip()
    start = max(0, idx - width // 2)
    end = min(len(text), idx + len(marker) + width // 2)
    return text[start:end].strip()


def _result(category: str, evidence: str, confidence: float) -> dict:
    return {
        "category": category,
        "evidence_snippet": evidence,
        "confidence": confidence,
        "suggested_action": SUGGESTED_ACTION[category],
    }


def classify_failure(job: dict, log_tail: str, exit_code: int | None,
                      machine_context: dict | None = None,
                      patterns: dict[str, tuple[str, ...]] | None = None) -> dict:
    """Classify a job's failure.

    job: the queue job record (used for e.g. job['scene'] and
         job.get('gpu_busy_at_claim') set by claim time).
    log_tail: last N bytes/lines of the job's log as text.
    exit_code: process exit code, or None if unknown.
    machine_context: optional dict with extra signals, e.g.
        {"disk_free_bytes": int, "docker_storage_free_bytes": int,
         "dmesg_tail": str}
    patterns: marker lists per category (from patterns_from_config());
        defaults to DEFAULT_PATTERNS.
    """
    log_tail = log_tail or ""
    low = log_tail.lower()
    machine_context = machine_context or {}
    patterns = patterns or DEFAULT_PATTERNS
    image_missing_markers = patterns.get("image_missing", _IMAGE_MISSING_MARKERS)
    network_markers = patterns.get("network_transient", _NETWORK_MARKERS)
    gpu_oom_markers = patterns.get("gpu_oom", _GPU_OOM_MARKERS)
    disk_full_marker = patterns.get("disk_full", (_DISK_FULL_MARKER,))[0]

    # --- disk_full ---------------------------------------------------------
    disk_free = machine_context.get("disk_free_bytes")
    docker_free = machine_context.get("docker_storage_free_bytes")
    low_disk = (disk_free is not None and disk_free < 2 * 1024 ** 3) or \
               (docker_free is not None and docker_free < 2 * 1024 ** 3)
    if disk_full_marker in low:
        return _result("disk_full", _snippet(log_tail, disk_full_marker), 0.95)
    if low_disk:
        free = disk_free if disk_free is not None else docker_free
        return _result("disk_full", f"free space below 2GB threshold: {free} bytes", 0.85)

    # --- image_missing -------------------------------------------------------
    for marker in image_missing_markers:
        if marker in low:
            return _result("image_missing", _snippet(log_tail, marker), 0.95)

    # --- gpu_busy_conflict ---------------------------------------------------
    gpu_oom_hit = any(marker in low for marker in gpu_oom_markers)
    if gpu_oom_hit and (job.get("gpu_busy_at_claim") or machine_context.get("gpu_busy_at_claim")):
        marker = next(m for m in gpu_oom_markers if m in low)
        return _result("gpu_busy_conflict", _snippet(log_tail, marker), 0.9)

    # --- oom_killed ------------------------------------------------------
    if exit_code == 137 and not gpu_oom_hit:
        dmesg = machine_context.get("dmesg_tail", "")
        evidence = "exit code 137, no CUDA/HIP OOM signature in log"
        if dmesg and ("out of memory" in dmesg.lower() or "oom-killer" in dmesg.lower()):
            evidence += f"; dmesg: {_snippet(dmesg, 'oom')}"
            return _result("oom_killed", evidence, 0.9)
        return _result("oom_killed", evidence, 0.6)

    # --- scene_missing -------------------------------------------------------
    scene = job.get("scene", "")
    if "no such file or directory" in low and scene and scene.lower() in low:
        return _result("scene_missing", _snippet(log_tail, "no such file or directory"), 0.9)
    if "no such file or directory" in low and scene:
        # Still likely scene-related if the job type reads directly from scene path
        return _result("scene_missing", _snippet(log_tail, "no such file or directory"), 0.55)

    # --- network_transient ---------------------------------------------------
    for marker in network_markers:
        if marker in low:
            return _result("network_transient", _snippet(log_tail, marker), 0.8)

    # --- gpu_busy_conflict without prior claim-time flag but still OOM/busy --
    if gpu_oom_hit:
        marker = next(m for m in gpu_oom_markers if m in low)
        return _result("gpu_busy_conflict", _snippet(log_tail, marker), 0.5)

    # --- code_error ------------------------------------------------------
    if _TRACEBACK_RE.search(log_tail):
        return _result("code_error", _snippet(log_tail, "Traceback (most recent call last)"), 0.85)

    return _result("unknown", _snippet(log_tail, "") if log_tail else "no matching signature", 0.3)
