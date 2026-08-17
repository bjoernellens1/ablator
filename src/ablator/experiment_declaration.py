"""Immutable, content-addressed experiment declarations."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Mapping


DECLARATION_ENV = "ABLATOR_EXPERIMENT_DECLARATION_JSON"
DECLARATION_SHA_ENV = "ABLATOR_EXPERIMENT_DECLARATION_SHA256"
JOB_ID_ENV = "ABLATOR_JOB_ID"
PROTECTED_ENV = frozenset({DECLARATION_ENV, DECLARATION_SHA_ENV, JOB_ID_ENV})

SUPPORTED_SCHEMA_VERSION = 1
GRADEABLE_RUN_CLASSES = frozenset({"experiment", "benchmark", "verification"})
NON_GRADEABLE_RUN_CLASSES = {
    "developer_smoke": "NON_GRADEABLE_DEVELOPER_SMOKE",
    "debug": "NON_GRADEABLE_DEBUG",
}
IMMUTABLE_JOB_FIELDS = frozenset(
    {
        "experiment_declaration",
        "experiment_declaration_json",
        "experiment_declaration_sha256",
        "gradeability",
    }
)


class ExperimentDeclarationError(ValueError):
    """A declaration is incomplete, ambiguous, or inconsistent."""


def canonical_declaration_json(declaration: Mapping[str, Any]) -> str:
    """Serialize exactly as the versioned Splatograph consumer expects."""
    return json.dumps(
        dict(declaration),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def declaration_sha256(declaration: Mapping[str, Any]) -> str:
    canonical = canonical_declaration_json(declaration).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def validate_declaration(declaration: Mapping[str, Any]) -> None:
    """Validate the generic envelope accepted by Splatograph schema v1."""
    if not isinstance(declaration, Mapping):
        raise ExperimentDeclarationError("experiment declaration must be a JSON object")

    errors: list[str] = []
    schema_version = declaration.get("schema_version")
    if schema_version != SUPPORTED_SCHEMA_VERSION:
        errors.append(
            f"unsupported schema_version {schema_version!r}; "
            f"expected {SUPPORTED_SCHEMA_VERSION}"
        )

    run_class = declaration.get("run_class")
    known_classes = GRADEABLE_RUN_CLASSES | set(NON_GRADEABLE_RUN_CLASSES)
    if run_class not in known_classes:
        errors.append(f"unsupported run_class {run_class!r}")

    for key in ("title", "purpose"):
        if not _present(declaration.get(key)):
            errors.append(f"missing non-empty {key}")

    if run_class in GRADEABLE_RUN_CLASSES:
        for key in ("experiment_id", "expected_evidence"):
            if not _present(declaration.get(key)):
                errors.append(f"missing non-empty {key}")
        if not (
            _present(declaration.get("owner_refs"))
            or _present(declaration.get("standalone_reason"))
        ):
            errors.append(
                "gradeable declaration requires owner_refs or standalone_reason"
            )

        arm = declaration.get("arm")
        if not isinstance(arm, Mapping):
            errors.append("gradeable declaration requires arm object")
        else:
            for key in ("id", "comparison_role", "manipulation"):
                if not _present(arm.get(key)):
                    errors.append(f"gradeable declaration arm missing non-empty {key}")

    if errors:
        raise ExperimentDeclarationError("; ".join(errors))


def _merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge JSON objects without retaining caller-owned values."""
    resolved = deepcopy(dict(base))
    for key, value in overlay.items():
        prior = resolved.get(key)
        if isinstance(prior, Mapping) and isinstance(value, Mapping):
            resolved[key] = _merge(prior, value)
        else:
            resolved[key] = deepcopy(value)
    return resolved


def resolve_declaration(
    shared: Mapping[str, Any] | None,
    arm_overlay: Mapping[str, Any] | None,
    arm_id: str,
) -> dict[str, Any] | None:
    """Resolve shared experiment metadata and one arm's overlay."""
    if shared is None:
        if arm_overlay is not None:
            raise ExperimentDeclarationError(
                "arm declaration requires a top-level experiment object"
            )
        return None
    if not isinstance(shared, Mapping):
        raise ExperimentDeclarationError("top-level experiment must be a JSON object")
    if arm_overlay is not None and not isinstance(arm_overlay, Mapping):
        raise ExperimentDeclarationError(
            "arm declaration overlay must be a JSON object"
        )

    resolved = _merge(shared, arm_overlay or {})
    declared_arm = resolved.get("arm")
    if declared_arm is None:
        declared_arm = {}
    if not isinstance(declared_arm, Mapping):
        raise ExperimentDeclarationError("declaration arm must be a JSON object")
    declared_id = declared_arm.get("id")
    if _present(declared_id) and declared_id != arm_id:
        raise ExperimentDeclarationError(
            f"declared arm id {declared_id!r} conflicts with spec arm id {arm_id!r}"
        )
    resolved["arm"] = {**deepcopy(dict(declared_arm)), "id": arm_id}
    validate_declaration(resolved)
    return resolved


def gradeability(declaration: Mapping[str, Any]) -> str:
    run_class = declaration["run_class"]
    if run_class in GRADEABLE_RUN_CLASSES:
        return "GRADEABLE_DECLARED"
    return NON_GRADEABLE_RUN_CLASSES[str(run_class)]


def freeze_declaration(declaration: Mapping[str, Any]) -> dict[str, Any]:
    """Return the immutable queue fields for a validated declaration."""
    validate_declaration(declaration)
    frozen = deepcopy(dict(declaration))
    canonical = canonical_declaration_json(frozen)
    return {
        "experiment_declaration": frozen,
        "experiment_declaration_json": canonical,
        "experiment_declaration_sha256": hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest(),
        "gradeability": gradeability(frozen),
    }


def validate_frozen_job(job: Mapping[str, Any]) -> dict[str, Any] | None:
    """Validate the declaration fields stored in a queue job."""
    declaration = job.get("experiment_declaration")
    present_fields = IMMUTABLE_JOB_FIELDS.intersection(job)
    if declaration is None:
        if present_fields:
            raise ExperimentDeclarationError(
                "partial experiment declaration fields without experiment_declaration"
            )
        return None
    if not isinstance(declaration, Mapping):
        raise ExperimentDeclarationError("experiment_declaration must be a JSON object")

    expected = freeze_declaration(declaration)
    if (
        job.get("experiment_declaration_json")
        != expected["experiment_declaration_json"]
    ):
        raise ExperimentDeclarationError(
            "canonical experiment declaration JSON mismatch"
        )
    if (
        job.get("experiment_declaration_sha256")
        != expected["experiment_declaration_sha256"]
    ):
        raise ExperimentDeclarationError("experiment declaration SHA-256 mismatch")
    if job.get("gradeability") != expected["gradeability"]:
        raise ExperimentDeclarationError("experiment declaration gradeability mismatch")
    return expected


def validate_immutable_update(
    job: Mapping[str, Any], fields: Mapping[str, Any]
) -> None:
    for key in IMMUTABLE_JOB_FIELDS.intersection(fields):
        if fields[key] != job.get(key):
            raise ExperimentDeclarationError(
                f"immutable {key} cannot be changed after enqueue"
            )


def experiment_environment(job: Mapping[str, Any]) -> dict[str, str]:
    """Return only the protected child-process declaration environment."""
    frozen = validate_frozen_job(job)
    if frozen is None:
        return {}
    job_id = job.get("id")
    if not _present(job_id):
        raise ExperimentDeclarationError("declared job requires a non-empty job id")
    return {
        DECLARATION_ENV: frozen["experiment_declaration_json"],
        DECLARATION_SHA_ENV: frozen["experiment_declaration_sha256"],
        JOB_ID_ENV: str(job_id),
    }


def runner_log_banner(job: Mapping[str, Any]) -> str:
    """Render the exact frozen declaration transported to the child."""
    env = experiment_environment(job)
    if not env:
        return "# experiment declaration: MISSING (NON-GRADEABLE LEGACY JOB)"
    return "\n".join(
        [
            f"# experiment declaration: {job['gradeability']}",
            f"# {JOB_ID_ENV}={env[JOB_ID_ENV]}",
            f"# {DECLARATION_SHA_ENV}={env[DECLARATION_SHA_ENV]}",
            f"# {DECLARATION_ENV}={env[DECLARATION_ENV]}",
        ]
    )
