"""Canonical, credential-free receipts for one resolved job launch."""

from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from typing import Any, Mapping

from . import experiment_declaration as declarations


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


def receipt_sha256(receipt: Mapping[str, Any]) -> str:
    """Return the canonical digest that identifies one immutable receipt."""
    return _sha256_json(receipt)


def argv_sha256(argv: list[str]) -> str:
    """Return the canonical digest for one argument vector."""
    return _sha256_json(argv)


def protected_environment_projection(entries: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Hash only Ablator-owned trainer variables, never arbitrary pod env."""
    variables: dict[str, str] = {}
    for entry in entries:
        name = entry.get("name")
        if name not in declarations.PROTECTED_ENV:
            continue
        if name in variables:
            raise ValueError(f"duplicate protected trainer environment {name}")
        value = entry.get("value")
        if not isinstance(value, str):
            raise ValueError(f"protected trainer environment {name} has no literal value")
        variables[str(name)] = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return {
        "schema": "ablator.protected-env/v1",
        "variables": dict(sorted(variables.items())),
    }


def protected_environment_sha256(projection: Mapping[str, Any]) -> str:
    return _sha256_json(projection)


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


def _nonrecursive_argv(argv: list[str]) -> list[str]:
    """Remove trusted protected-env values that recursively contain the receipt."""
    if not argv or os.path.basename(argv[0]) not in _CONTAINER_RUNTIMES:
        return list(argv)
    projected: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token in {"-e", "--env"} and index + 1 < len(argv):
            name = argv[index + 1].split("=", 1)[0]
            if name in declarations.PROTECTED_ENV:
                index += 2
                continue
        if token.startswith("--env="):
            name = token[len("--env="):].split("=", 1)[0]
            if name in declarations.PROTECTED_ENV:
                index += 1
                continue
        if token.startswith("-e") and not token.startswith("--") and len(token) > 2:
            name = token[2:].removeprefix("=").split("=", 1)[0]
            if name in declarations.PROTECTED_ENV:
                index += 1
                continue
        projected.append(token)
        index += 1
    return projected


_LAUNCH_FINGERPRINT_FIELDS = (
    "cwd", "runtime", "image", "image_digest", "mounts", "argv_sha256",
)


def launch_sha256(launch: Mapping[str, Any]) -> str:
    """Hash the finite, nonrecursive expected-vs-actual launch projection."""
    return _sha256_json({key: deepcopy(launch.get(key)) for key in _LAUNCH_FINGERPRINT_FIELDS})


def build_actual_launch(
    argv: list[str],
    cwd: str | None,
    *,
    container_id: str | None = None,
    image_digest: str | None = None,
) -> dict[str, Any]:
    """Capture the launch form handed to the runtime without receipt recursion."""
    runtime, image, mounts = container_launch(argv)
    actual = {
        "cwd": cwd,
        "runtime": runtime or "process",
        "image": image,
        "image_digest": image_digest,
        "mounts": mounts,
        "argv_sha256": argv_sha256(_nonrecursive_argv(argv)),
        "container_id": container_id,
    }
    actual["launch_sha256"] = launch_sha256(actual)
    return actual


def _runner_identity(
    runner_provenance: Mapping[str, Any], machine: str | None = None,
) -> dict[str, Any]:
    """Select the credential-free runner fields covered by a receipt."""
    return {
        "machine": runner_provenance.get("machine", machine),
        "hostname": runner_provenance.get("hostname"),
        "package_version": runner_provenance.get("package_version"),
        "source_sha256": runner_provenance.get("source_sha256"),
        "git_commit": runner_provenance.get("git_commit"),
        "git_dirty": runner_provenance.get("git_dirty"),
        "config_sha256": runner_provenance.get("config_sha256"),
    }


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
    resolved_image_digest: str | None = None,
) -> dict[str, Any]:
    """Build the exact pre-launch identity without copying environment values."""
    runtime, image, mounts = container_launch(argv)
    state = dict(source_state or {})
    receipt = {
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
        "runner": _runner_identity(runner_provenance, machine),
        "launch": {
            "cwd": cwd,
            "runtime": runtime or "process",
            "image": image,
            "image_digest": resolved_image_digest,
            "mounts": mounts,
            "argv_sha256": argv_sha256(argv),
            # Hashing the merged type config captures all runtime knobs while
            # avoiding credential-bearing env values in the receipt itself.
            "type_config_sha256": _sha256_json(type_config),
        },
    }
    receipt["launch"]["launch_sha256"] = launch_sha256(receipt["launch"])
    return receipt


def build_final_attestation(
    receipt: Mapping[str, Any],
    *,
    expected_receipt_sha256: str,
    source_state: Mapping[str, Any] | None = None,
    actual_launch: Mapping[str, Any] | None = None,
    type_config: Mapping[str, Any] | None = None,
    semantic_argv: list[str] | None = None,
    runner_provenance: Mapping[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Bind final source and launch evidence to the immutable prelaunch receipt."""
    expected = dict(receipt.get("source") or {})
    actual = dict(source_state or {})
    reasons: list[str] = []
    if error:
        reasons.append(error)
    actual_receipt_sha256 = receipt_sha256(receipt)
    if not expected_receipt_sha256 or actual_receipt_sha256 != expected_receipt_sha256:
        reasons.append("receipt SHA-256 mismatch")
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

    launch = dict(receipt.get("launch") or {})
    binding: dict[str, Any] = {
        "actual_launch": deepcopy(dict(actual_launch or {})) or None,
        "runner": deepcopy(dict(receipt.get("runner") or {})),
        "type_config_sha256": None,
        "semantic_argv_sha256": None,
    }
    if type_config is not None:
        binding["type_config_sha256"] = _sha256_json(type_config)
        if binding["type_config_sha256"] != launch.get("type_config_sha256"):
            reasons.append("runtime config differs from prelaunch receipt")
    elif launch.get("type_config_sha256") is not None:
        reasons.append("runtime config is unavailable")
    if semantic_argv is not None:
        binding["semantic_argv_sha256"] = _sha256_json(semantic_argv)
        if binding["semantic_argv_sha256"] != launch.get("argv_sha256"):
            reasons.append("semantic argv differs from prelaunch receipt")
    elif launch.get("argv_sha256") is not None:
        reasons.append("semantic argv is unavailable")
    if runner_provenance is not None:
        receipt_runner = dict(receipt.get("runner") or {})
        if _runner_identity(
            runner_provenance, receipt_runner.get("machine")
        ) != receipt_runner:
            reasons.append("runner identity differs from prelaunch receipt")
    elif receipt.get("runner"):
        reasons.append("runner identity is unavailable")
    if actual_launch is not None:
        actual_launch_dict = dict(actual_launch)
        expected_launch_sha256 = launch_sha256(launch)
        if (not launch.get("launch_sha256")
                or launch.get("launch_sha256") != expected_launch_sha256):
            reasons.append("prelaunch fingerprint is missing or invalid")
        actual_launch_sha256 = launch_sha256(actual_launch_dict)
        if (not actual_launch_dict.get("launch_sha256")
                or actual_launch_dict.get("launch_sha256") != actual_launch_sha256):
            reasons.append("actual launch fingerprint is missing or invalid")
        if actual_launch_dict.get("argv_sha256") != launch.get("argv_sha256"):
            reasons.append("actual argv fingerprint differs from prelaunch receipt")
        if actual_launch_sha256 != expected_launch_sha256:
            reasons.append("actual launch differs from prelaunch receipt")
        if launch.get("runtime") in _CONTAINER_RUNTIMES:
            if not actual_launch_dict.get("container_id"):
                reasons.append("actual container ID is missing")
            if not launch.get("image_digest"):
                reasons.append("prelaunch image digest is missing")
            if not actual_launch_dict.get("image_digest"):
                reasons.append("actual image digest is missing")
            elif actual_launch_dict.get("image_digest") != launch.get("image_digest"):
                reasons.append("actual image digest differs from prelaunch receipt")
    elif launch:
        reasons.append("actual launch is unavailable")
    return {
        "schema": "ablator.execution-attestation/v1",
        "verdict": "REJECTED" if reasons else "ACCEPTED",
        "receipt_sha256": expected_receipt_sha256 or None,
        "source": actual or None,
        "binding": binding,
        "error": "; ".join(reasons) if reasons else None,
    }
