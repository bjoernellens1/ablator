"""Stable stdlib-only interface for external workflow schedulers.

This module intentionally knows nothing about Snakemake or ResearchFlow.  It
accepts one already-resolved typed job, persists an idempotency fingerprint,
and exposes exact-job inspection/cancellation.  The normal Ablator runner
continues to own machine selection, capability checks, containers, retries,
and terminal state bookkeeping.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

from . import config as cfgmod
from .identity import package_source_sha256
from .queue import Queue

SCHEMA = "ablator.external-job/v1"
RESERVED_PARAMS = frozenset(
    {"id", "machine", "type", "status", "scene", "model_path", "extra_args", "iterations"}
)
TERMINAL_STATES = frozenset(
    {"done", "failed", "quarantined", "cancelled", "failed_no_retry"}
)


class ExternalJobError(SystemExit):
    """Invalid or conflicting external scheduler request."""


def _canonical_json(value: Any) -> str:
    """Return deterministic compact JSON for hashing and machine output."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_json(value: Any) -> str:
    """Hash a JSON-compatible value using canonical serialization."""
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def parse_key_values(values: list[str] | None, *, field: str = "parameter") -> dict[str, str]:
    """Parse repeated ``key=value`` CLI arguments without shell evaluation."""
    result: dict[str, str] = {}
    for raw in values or []:
        key, sep, value = raw.partition("=")
        key = key.strip()
        if not sep or not key:
            raise ExternalJobError(f"invalid {field} {raw!r}; expected key=value")
        if key in result:
            raise ExternalJobError(f"duplicate {field} key {key!r}")
        result[key] = value
    return result


def _validate_id(value: str) -> str:
    """Validate a scheduler-supplied stable job identifier."""
    value = str(value).strip()
    if not value or len(value) > 180:
        raise ExternalJobError("external job id must contain 1..180 characters")
    if any(ch.isspace() for ch in value) or "/" in value or "\\" in value:
        raise ExternalJobError("external job id may not contain whitespace or path separators")
    return value


def build_job(
    cfg: dict[str, Any],
    *,
    job_id: str,
    job_type: str,
    machine: str = "any",
    params: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    lane: int = 2,
    depends_on: str | None = None,
) -> dict[str, Any]:
    """Validate and normalize one external job into the existing queue schema."""
    job_id = _validate_id(job_id)
    job_type = str(job_type).strip()
    machine = str(machine).strip() or "any"
    if job_type not in cfg.get("types", {}):
        raise ExternalJobError(
            f"external job type {job_type!r} is not defined in config {cfg.get('_path', '?')}"
        )
    if machine != "any" and machine not in cfg.get("machines", {}):
        raise ExternalJobError(f"unknown machine {machine!r}")
    try:
        lane = int(lane)
    except (TypeError, ValueError) as exc:
        raise ExternalJobError("lane must be 1, 2 or 3") from exc
    if lane not in (1, 2, 3):
        raise ExternalJobError("lane must be 1, 2 or 3")

    params = dict(params or {})
    conflicts = sorted(RESERVED_PARAMS.intersection(params))
    if conflicts:
        raise ExternalJobError(
            "external params may not override reserved fields: " + ", ".join(conflicts)
        )
    for key in params:
        if not isinstance(key, str) or not key or "{" in key or "}" in key:
            raise ExternalJobError(f"invalid external parameter name {key!r}")

    metadata = dict(metadata or {})
    immutable = {
        "id": job_id,
        "type": job_type,
        "machine": machine,
        "params": params,
        "metadata": metadata,
        "lane": lane,
        "depends_on": depends_on,
    }
    return {
        "id": job_id,
        "external_id": job_id,
        "external_schema": SCHEMA,
        "external_spec_sha256": _sha256_json(immutable),
        "external_metadata": metadata,
        "params": params,
        "machine": machine,
        "type": job_type,
        "lane": lane,
        "depends_on": depends_on,
        "status": "pending",
        "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": "external",
    }


def submit_job(cfg: dict[str, Any], job: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Idempotently enqueue a normalized job and return ``(record, created)``.

    Repeating the same external id with the same immutable specification is
    a no-op and returns the existing queue record.  Reusing the id for a
    different specification fails closed instead of silently changing the
    meaning of a scheduler job already known to Snakemake or another caller.
    """
    queue = Queue(cfgmod.queue_path(cfg))
    with queue._open_locked() as handle:
        jobs = queue._load(handle)
        existing = next((item for item in jobs if item.get("id") == job["id"]), None)
        if existing is not None:
            if existing.get("external_spec_sha256") == job.get("external_spec_sha256"):
                return existing, False
            raise ExternalJobError(
                f"job id {job['id']!r} already exists with a different specification"
            )
        jobs.append(job)
        queue._save(handle, jobs)
    return job, True


def inspect_job(cfg: dict[str, Any], job_id: str) -> dict[str, Any]:
    """Return a stable exact-job JSON projection for external schedulers."""
    job_id = _validate_id(job_id)
    record = next(
        (item for item in Queue(cfgmod.queue_path(cfg)).read() if item.get("id") == job_id),
        None,
    )
    if record is None:
        raise ExternalJobError(f"no job {job_id!r} in queue")
    return {
        "schema": SCHEMA,
        "job_id": job_id,
        "status": str(record.get("status") or "unknown"),
        "type": record.get("type"),
        "machine": record.get("machine", "any"),
        "claimed_by": record.get("claimed_by"),
        "claimed_at": record.get("claimed_at"),
        "finished_at": record.get("finished_at"),
        "lane": record.get("lane", 2),
        "depends_on": record.get("depends_on"),
        "external_spec_sha256": record.get("external_spec_sha256"),
        "metadata": record.get("external_metadata") or {},
        "params": record.get("params") or {},
        "runner_provenance": record.get("runner_provenance"),
        "workload_provenance": record.get("provenance"),
        "image_provenance": record.get("image_provenance"),
        "dispatch_host_commit": record.get("dispatch_host_commit"),
        "error_category": record.get("error_category"),
        "error_evidence": record.get("error_evidence"),
        "suggested_action": record.get("suggested_action"),
        "health": record.get("health"),
        "terminal": str(record.get("status")) in TERMINAL_STATES,
    }


def cancel_jobs(cfg: dict[str, Any], job_ids: list[str]) -> list[dict[str, Any]]:
    """Cancel exact jobs idempotently, including running jobs via control files."""
    wanted = [_validate_id(value) for value in job_ids]
    queue = Queue(cfgmod.queue_path(cfg))
    results: list[dict[str, Any]] = []
    running: list[str] = []
    with queue._open_locked() as handle:
        jobs = queue._load(handle)
        by_id = {str(item.get("id")): item for item in jobs}
        missing = [job_id for job_id in wanted if job_id not in by_id]
        if missing:
            raise ExternalJobError("unknown job ids: " + ", ".join(missing))
        for job_id in wanted:
            record = by_id[job_id]
            status = str(record.get("status") or "unknown")
            if status == "pending":
                record["status"] = "cancelled"
                record["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                results.append({"job_id": job_id, "status": "cancelled", "action": "cancelled"})
            elif status == "running":
                running.append(job_id)
                results.append({"job_id": job_id, "status": "running", "action": "cancel_requested"})
            else:
                results.append({"job_id": job_id, "status": status, "action": "no_op"})
        queue._save(handle, jobs)

    # Running jobs are supervised by another process.  Use the same control
    # file protocol as ``ablator skip`` so cancellation stays backend-agnostic.
    queue_dir = os.path.dirname(cfgmod.queue_path(cfg))
    for job_id in running:
        path = os.path.join(queue_dir, f"control_{job_id}")
        with open(path, "w") as handle:
            handle.write("skip\n")
    return results


def _git(command: list[str], cwd: Path) -> str | None:
    """Run a best-effort read-only git query without making git mandatory."""
    try:
        proc = subprocess.run(
            ["git", *command], cwd=cwd, capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def capture_runner_provenance(cfg: dict[str, Any], machine: str) -> dict[str, Any]:
    """Capture the exact Ablator runner/config identity executing a job."""
    module_path = Path(__file__).resolve()
    root_raw = _git(["rev-parse", "--show-toplevel"], module_path.parent)
    root = Path(root_raw) if root_raw else module_path.parent
    commit = _git(["rev-parse", "HEAD"], root) if root_raw else None
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], root) if root_raw else None
    status = _git(["status", "--porcelain=v1", "--untracked-files=all"], root) if root_raw else None
    dirty = bool(status) if status is not None else None
    dirty_fingerprint = hashlib.sha256(status.encode()).hexdigest() if status else None

    config_path = Path(str(cfg.get("_path") or "")).expanduser()
    config_sha256 = None
    try:
        if config_path.is_file():
            config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    except OSError:
        pass
    try:
        version = importlib.metadata.version("ablator")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    source_sha256 = package_source_sha256(module_path.parent)

    return {
        "schema": "ablator.runner-provenance/v1",
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "machine": machine,
        "hostname": socket.gethostname(),
        "package_version": version,
        "source_sha256": source_sha256,
        "module_path": str(module_path),
        "git_root": str(root) if root_raw else None,
        "git_commit": commit,
        "git_branch": branch,
        "git_dirty": dirty,
        "git_dirty_fingerprint": dirty_fingerprint,
        "config_path": str(config_path) if str(config_path) else None,
        "config_sha256": config_sha256,
        # Source+config identity is complete even for an installed wheel that
        # has no Git checkout. Git commit/dirty state remains additional
        # provenance whenever the runner is an editable checkout.
        "identity_complete": bool(source_sha256 and config_sha256),
    }


def parse_metadata_json(raw: str | None) -> dict[str, Any]:
    """Parse scheduler metadata while requiring a JSON object."""
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise ExternalJobError(f"invalid --metadata-json: {exc}") from exc
    if not isinstance(value, dict):
        raise ExternalJobError("--metadata-json must contain a JSON object")
    return value


def print_json(value: Any) -> None:
    """Write one stable compact JSON object/array to stdout."""
    print(_canonical_json(value))
