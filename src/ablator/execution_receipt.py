"""Canonical, credential-free receipts for one resolved job launch."""

from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from typing import Any, Mapping


_CONTAINER_RUNTIMES = {"docker", "podman"}
_OPTIONS_WITH_VALUE = {
    "--add-host", "--device", "--entrypoint", "--env", "--env-file",
    "--group-add", "--hostname", "--ipc", "--label", "--name",
    "--network", "--publish", "--security-opt", "--shm-size", "--user",
    "--volume", "--workdir", "-e", "-h", "-l", "-p", "-u", "-v", "-w",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _volume_mount(spec: str) -> dict[str, Any] | None:
    parts = spec.split(":")
    if len(parts) < 2:
        return None
    options = ",".join(parts[2:]).split(",") if len(parts) > 2 else []
    return {
        "source": parts[0],
        "target": parts[1],
        "read_only": "ro" in options or "readonly" in options,
    }


def _long_mount(spec: str) -> dict[str, Any] | None:
    fields: dict[str, str] = {}
    flags: set[str] = set()
    for item in spec.split(","):
        key, separator, value = item.partition("=")
        if separator:
            fields[key] = value
        else:
            flags.add(key)
    if fields.get("type", "bind") != "bind":
        return None
    source = fields.get("src") or fields.get("source")
    target = fields.get("dst") or fields.get("destination") or fields.get("target")
    if not source or not target:
        return None
    return {
        "source": source,
        "target": target,
        "read_only": (
            "readonly" in flags
            or "ro" in flags
            or fields.get("readonly", "").lower() in {"1", "true", "yes"}
            or fields.get("ro", "").lower() in {"1", "true", "yes"}
        ),
    }


def container_launch(argv: list[str]) -> tuple[str | None, str | None, list[dict]]:
    """Return ``(runtime, image, normalized bind mounts)`` without env values."""
    if not argv or os.path.basename(argv[0]) not in _CONTAINER_RUNTIMES:
        return None, None, []
    try:
        run_index = argv.index("run")
    except ValueError:
        return None, None, []

    mounts: list[dict] = []
    image: str | None = None
    index = run_index + 1
    while index < len(argv):
        token = argv[index]
        if token in {"-v", "--volume", "--mount"} and index + 1 < len(argv):
            value = argv[index + 1]
            mount = _long_mount(value) if token == "--mount" else _volume_mount(value)
            if mount is not None:
                mounts.append(mount)
            index += 2
            continue
        if token.startswith("--volume="):
            mount = _volume_mount(token.split("=", 1)[1])
            if mount is not None:
                mounts.append(mount)
            index += 1
            continue
        if token.startswith("--mount="):
            mount = _long_mount(token.split("=", 1)[1])
            if mount is not None:
                mounts.append(mount)
            index += 1
            continue
        if token.startswith("-v") and not token.startswith("--") and len(token) > 2:
            mount = _volume_mount(token[2:].removeprefix("="))
            if mount is not None:
                mounts.append(mount)
            index += 1
            continue
        if token in _OPTIONS_WITH_VALUE:
            index += 2
            continue
        if token.startswith("-"):
            index += 1
            continue
        image = token
        break
    return os.path.basename(argv[0]), image, mounts


def build_prelaunch_receipt(
    *,
    cfg: Mapping[str, Any],
    job: Mapping[str, Any],
    machine: str,
    type_config: Mapping[str, Any],
    argv: list[str],
    cwd: str | None,
    source_state: Mapping[str, Any] | None,
    source_repo: str | None,
    source_checkout: str | None,
    source_lease_id: str | None,
    runner_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the exact pre-launch identity without copying environment values."""
    runtime, image, mounts = container_launch(argv)
    state = dict(source_state or {})
    return {
        "schema": "ablator.execution/v1",
        "phase": "prelaunch",
        "job_id": str(job.get("id") or ""),
        "machine": machine,
        "source": {
            "requested_git_sha": job.get("requested_git_sha"),
            "executed_git_sha": state.get("commit"),
            "ref": state.get("ref"),
            "dirty": state.get("dirty"),
            "submodules": deepcopy(state.get("submodules") or []),
            "repo": source_repo or job.get("git_repo"),
            "checkout": source_checkout,
            "lease_id": source_lease_id,
        },
        "runner": {
            "machine": runner_provenance.get("machine", machine),
            "hostname": runner_provenance.get("hostname"),
            "package_version": runner_provenance.get("package_version"),
            "source_sha256": runner_provenance.get("source_sha256"),
            "git_commit": runner_provenance.get("git_commit"),
            "git_dirty": runner_provenance.get("git_dirty"),
            "config_sha256": runner_provenance.get("config_sha256"),
        },
        "launch": {
            "cwd": cwd,
            "runtime": runtime or "process",
            "image": image,
            "mounts": mounts,
            "argv_sha256": _sha256_json(argv),
            # Hashing the merged type config captures all runtime knobs while
            # avoiding credential-bearing env values in the receipt itself.
            "type_config_sha256": _sha256_json(type_config),
        },
    }


def build_final_attestation(
    receipt: Mapping[str, Any],
    *,
    source_state: Mapping[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Compare final source state with the pre-launch immutable contract."""
    expected = dict(receipt.get("source") or {})
    actual = dict(source_state or {})
    reasons: list[str] = []
    if error:
        reasons.append(error)
    if actual:
        if actual.get("commit") != expected.get("requested_git_sha"):
            reasons.append("final commit differs from requested Git SHA")
        if actual.get("ref") != "DETACHED":
            reasons.append("final checkout is not detached")
        if actual.get("dirty") is not False:
            reasons.append("final checkout is dirty")
        if actual.get("submodules") != expected.get("submodules"):
            reasons.append("final recursive submodule state changed")
    elif not reasons:
        reasons.append("final source state is unavailable")
    return {
        "schema": "ablator.execution-attestation/v1",
        "verdict": "REJECTED" if reasons else "ACCEPTED",
        "source": actual or None,
        "error": "; ".join(reasons) if reasons else None,
    }
