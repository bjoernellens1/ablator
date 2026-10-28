"""Opt-in passive observation of run_job's existing host ownership interval."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from uuid import UUID, uuid4


def _clock_domain() -> dict:
    domain = dict(schema="monotonic-clock-domain/v1", clock=None, boot_id=None,
                  time_namespace=None, status="unavailable", checked_through=None,
                  unavailable_reason=None)
    try:
        info = time.get_clock_info("monotonic")
        if (sys.platform != "linux" or not info.monotonic or info.adjustable
                or info.implementation != "clock_gettime(CLOCK_MONOTONIC)"):
            raise ValueError("unverified Linux CLOCK_MONOTONIC")
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        namespace = os.readlink("/proc/self/ns/time")
        if str(UUID(boot)) != boot or not re.fullmatch(r"time:\[[0-9]+\]", namespace):
            raise ValueError("invalid monotonic identity")
        domain.update(clock="CLOCK_MONOTONIC", boot_id=boot,
                      time_namespace=namespace, status="known")
    except (OSError, ValueError) as exc:
        domain["unavailable_reason"] = str(exc)
    return domain


def start(job: dict, machine: str) -> LifecycleCost | None:
    """Ordinary jobs perform no new clock sampling, filesystem access or copying."""
    if (job.get("external_metadata") or {}).get("record_outer_lifecycle_cost") is not True:
        return None
    try:
        return LifecycleCost(job, machine)
    except Exception as exc:
        print(f"[ablator] lifecycle observation unavailable: {exc}", flush=True)
        return None


class LifecycleCost:
    def __init__(self, job: dict, machine: str):
        self.started = time.monotonic_ns()
        self.domain = _clock_domain()
        self.envelope_id = str(uuid4())
        metadata = job.get("external_metadata") or {}
        self.identity = {
            "scheduler_job_id": job.get("id"), "external_id": job.get("external_id"),
            "machine": machine,
            **{key: metadata.get(key) for key in (
                "researchflow_run_id", "researchflow_plan_sha256", "researchflow_job_id")},
        }
        self.declaration = {
            "requested_git_sha": job.get("requested_git_sha"),
            "experiment_declaration_sha256": job.get("experiment_declaration_sha256"),
            "jobscript_sha256": metadata.get("jobscript_sha256"),
        }
        self.origin = metadata.get("researchflow_origin_path")
        self.child_started = None
        self.child_wait_completed = None

    def _artifacts(self) -> tuple[list, list, list]:
        attempts, artifacts, errors = [], [], []
        job_id = self.identity["researchflow_job_id"]
        if not self.origin or not isinstance(job_id, str) or Path(job_id).name != job_id:
            return attempts, artifacts, errors
        state = Path(self.origin).resolve().parent
        references: dict[Path, tuple[dict, bytes | None]] = {}

        def reference(path, *, parse=False):
            resolved = path.resolve()
            cached = references.get(resolved)
            if cached is not None and (not parse or cached[1] is not None):
                return cached
            raw = resolved.read_bytes() if parse else None
            digest = hashlib.sha256()
            if raw is not None:
                digest.update(raw)
            else:
                with resolved.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(64 * 1024), b""):
                        digest.update(chunk)
            ref = {"path": str(resolved), "sha256": digest.hexdigest()}
            if cached is not None and cached[0] != ref:
                raise ValueError(f"artifact changed during collection: {resolved}")
            references[resolved] = (ref, raw)
            return ref, raw

        for suffix in ("execution.json", "evidence.json", "done"):
            path = state / f"{job_id}.{suffix}"
            if path.exists():
                try:
                    artifacts.append(reference(path)[0])
                except OSError as exc:
                    errors.append(str(exc))
        for path in sorted((state / "attempts").glob("*/attempt.json")):
            try:
                ref, raw = reference(path, parse=True)
                record = json.loads(raw)
                if (record.get("schema") != "researchflow.execution-attempt/v1"
                        or record.get("job_id") != job_id
                        or record.get("run_id") != self.identity["researchflow_run_id"]
                        or record.get("plan_sha256") != self.identity["researchflow_plan_sha256"]):
                    continue
                attempt_id = record.get("attempt_id")
                if not isinstance(attempt_id, str) or path.parent.name != attempt_id:
                    errors.append(f"attempt identity/path mismatch: {path}")
                    continue
                execution_path = path.parent / "execution.json"
                execution = reference(execution_path)[0] if execution_path.exists() else None
                stamp = record.get("attempt_started_monotonic_ns")
                attempts.append((stamp if type(stamp) is int else -1, attempt_id,
                                 {"attempt_id": attempt_id, "record": ref, "execution": execution}))
            except (OSError, ValueError, AttributeError) as exc:
                errors.append(f"{path}: {exc}")
        return [item[2] for item in sorted(attempts)], artifacts, errors

    def publish(self, log_dir: str, *, bindings: dict, status: str, exit_code: int | None,
                error: str | None, proc, source_release: str,
                owner_unwinding: bool) -> None:
        """Observation failure never changes run_job's result or cleanup decisions."""
        try:
            attempts, artifacts, errors = self._artifacts()
            domain = _clock_domain()
            if domain != self.domain:
                domain.update(status="unavailable", checked_through=None,
                              unavailable_reason="clock identity unavailable or changed")
            elif domain["status"] == "known":
                domain["checked_through"] = "host_owner_return"
            observed = time.monotonic_ns()
            direct_child = "not_launched" if proc is None else (
                "reaped" if self.child_wait_completed is not None else "unknown")
            closed = (not owner_unwinding and direct_child != "unknown"
                      and source_release != "unknown")
            record = {
                "schema": "outer-lifecycle-cost/v1", "envelope_id": self.envelope_id,
                "identity": self.identity, "declaration": self.declaration,
                "clock_domain": domain, "started_monotonic_ns": self.started,
                "child_started_monotonic_ns": self.child_started,
                "child_wait_completed_monotonic_ns": self.child_wait_completed,
                "observed_through_monotonic_ns": observed,
                "closed_monotonic_ns": observed if closed else None,
                "elapsed_seconds": (observed - self.started) / 1e9 if closed else None,
                "observed_seconds": (observed - self.started) / 1e9,
                "outcome": {"status": status, "exit_code": exit_code, "error": error},
                "closure": {
                    "host_owner": "closed" if closed else "unknown",
                    "direct_child": direct_child, "source_release": source_release,
                    "workload": "unknown",
                    "reason": "opaque descendants, durable output completeness and teardown are unverified",
                },
                "bindings": bindings,
                "observed_attempts": attempts, "attempt_roster": "observed_only",
                "durable_artifacts": artifacts, "binding_errors": errors,
                "resources": dict(cpu_seconds=None, gpu_active_seconds=None, energy_joules=None,
                                  reason="resource accounting not observed at this host-owner boundary"),
                "receipt_publication": {"scope": "outside_closed_host_owner_interval"},
            }
            path = Path(log_dir) / f"outer-lifecycle-cost-{self.envelope_id}.json"
            raw = json.dumps(record, sort_keys=True, allow_nan=False).encode() + b"\n"
            with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as handle:
                handle.write(raw)
        except Exception as exc:
            print(f"[ablator] lifecycle receipt unavailable: {exc}", flush=True)
