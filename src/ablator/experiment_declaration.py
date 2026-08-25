"""Immutable, content-addressed experiment declarations and launch provenance."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Mapping


DECLARATION_ENV = "ABLATOR_EXPERIMENT_DECLARATION_JSON"
DECLARATION_SHA_ENV = "ABLATOR_EXPERIMENT_DECLARATION_SHA256"
JOB_ID_ENV = "ABLATOR_JOB_ID"
JOB_JSON_ENV = "ABLATOR_JOB_JSON"
SUBMISSION_ENV = "ABLATOR_SUBMISSION_JSON"
SOURCE_PROOF_ENV = "ABLATOR_SOURCE_PROOF_JSON"
PROTECTED_ENV = frozenset(
    {
        DECLARATION_ENV,
        DECLARATION_SHA_ENV,
        JOB_ID_ENV,
        JOB_JSON_ENV,
        SUBMISSION_ENV,
        SOURCE_PROOF_ENV,
    }
)

SUPPORTED_SCHEMA_VERSION = 1
GRADEABLE_RUN_CLASSES = frozenset({"experiment", "benchmark", "verification"})
NON_GRADEABLE_RUN_CLASSES = {
    "developer_smoke": "NON_GRADEABLE_DEVELOPER_SMOKE",
    "debug": "NON_GRADEABLE_DEBUG",
}
DECLARATION_JOB_FIELDS = frozenset(
    {
        "experiment_declaration",
        "experiment_declaration_json",
        "experiment_declaration_sha256",
        "gradeability",
    }
)
IMMUTABLE_JOB_FIELDS = DECLARATION_JOB_FIELDS | frozenset(
    {"id", "submission_provenance", "requested_git_sha", "git_repo"}
)
EXTERNAL_HASHED_JOB_FIELDS = frozenset(
    {
        "external_id",
        "external_schema",
        "external_spec_sha256",
        "external_metadata",
        "params",
        "machine",
        "type",
        "lane",
        "depends_on",
        "requested_git_sha",
        "git_repo",
    }
)


class ExperimentDeclarationError(ValueError):
    """A declaration or protected launch-provenance envelope is inconsistent."""


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
    present_fields = DECLARATION_JOB_FIELDS.intersection(job)
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
    immutable = IMMUTABLE_JOB_FIELDS
    if job.get("external_schema"):
        immutable = immutable | EXTERNAL_HASHED_JOB_FIELDS
    for key in immutable.intersection(fields):
        if fields[key] != job.get(key):
            raise ExperimentDeclarationError(
                f"immutable {key} cannot be changed after enqueue"
            )


def _canonical_json(value: Any) -> str:
    """Canonical JSON used by the generic launch-provenance transport."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _external_submission_content(job: Mapping[str, Any]) -> dict[str, Any]:
    """Return every external input whose identity is frozen at submission."""
    return {
        "schema": "ablator.submission/v1",
        "surface": "submit",
        "job_id": str(job.get("id") or ""),
        "external_id": job.get("external_id"),
        "type": job.get("type"),
        "machine": job.get("machine", "any"),
        "params": deepcopy(job.get("params") or {}),
        "metadata": deepcopy(job.get("external_metadata") or {}),
        "lane": job.get("lane", 2),
        "depends_on": job.get("depends_on"),
        "requested_git_sha": job.get("requested_git_sha"),
        "git_repo": job.get("git_repo"),
        "external_schema": job.get("external_schema"),
    }


def freeze_external_submission(job: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    """Build the canonical external envelope and its immutable digest."""
    content = _external_submission_content(job)
    digest = hashlib.sha256(_canonical_json(content).encode("utf-8")).hexdigest()
    return {**content, "external_spec_sha256": digest}, digest


def validate_external_submission(job: Mapping[str, Any]) -> dict[str, Any] | None:
    """Recompute external identity from queue fields and verify its envelope."""
    if not job.get("external_schema"):
        return None
    expected, digest = freeze_external_submission(job)
    if job.get("external_spec_sha256") != digest:
        raise ExperimentDeclarationError("external specification SHA-256 mismatch")
    recorded = job.get("submission_provenance")
    if not isinstance(recorded, Mapping):
        raise ExperimentDeclarationError(
            "external job requires a frozen submission_provenance envelope"
        )
    if dict(recorded) != expected:
        raise ExperimentDeclarationError("external submission provenance mismatch")
    return expected


def submission_provenance(job: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the exact queue-submission envelope for a job when known.

    ``ablator plan`` freezes its loaded spec into ``submission_provenance``.
    Current external jobs must already contain the frozen envelope written by
    ``ablator submit``. Older external records without it fail closed; the
    runner never reconstructs mutable intent at launch time.
    """
    if job.get("external_schema"):
        return validate_external_submission(job)

    recorded = job.get("submission_provenance")
    if isinstance(recorded, Mapping):
        resolved = deepcopy(dict(recorded))
        if resolved.get("surface") == "plan":
            spec = resolved.get("spec")
            expected_hash = resolved.get("spec_sha256")
            if not isinstance(spec, Mapping) or not _present(expected_hash):
                raise ExperimentDeclarationError(
                    "plan submission provenance requires spec and spec_sha256"
                )
            actual_hash = hashlib.sha256(_canonical_json(dict(spec)).encode("utf-8")).hexdigest()
            if actual_hash != expected_hash:
                raise ExperimentDeclarationError("plan submission spec SHA-256 mismatch")
        return resolved

    return None


def experiment_environment(job: Mapping[str, Any]) -> dict[str, str]:
    """Return protected child-process provenance environment.

    Queue-backed launch records get job identity and canonical job JSON even
    without an experiment declaration. Pure in-memory pending jobs keep the
    historical declaration-only behavior until they carry plan/submit
    provenance or have actually been claimed (``status == running``). This
    preserves the library-level rendering contract while ensuring every real
    runner-launched legacy job receives ``ABLATOR_JOB_ID``.
    """
    job_id = job.get("id")
    if not _present(job_id):
        raise ExperimentDeclarationError("job requires a non-empty job id")

    frozen = validate_frozen_job(job)
    submission = submission_provenance(job)
    queue_backed_launch = bool(
        submission is not None
        or job.get("external_schema")
        or job.get("status") == "running"
    )

    env: dict[str, str] = {}
    if frozen is not None or queue_backed_launch:
        env[JOB_ID_ENV] = str(job_id)
    if queue_backed_launch:
        env[JOB_JSON_ENV] = _canonical_json(dict(job))
    if submission is not None:
        env[SUBMISSION_ENV] = _canonical_json(submission)
    if frozen is not None:
        env.update(
            {
                DECLARATION_ENV: frozen["experiment_declaration_json"],
                DECLARATION_SHA_ENV: frozen["experiment_declaration_sha256"],
            }
        )
    return env


def runner_log_banner(job: Mapping[str, Any]) -> str:
    """Render the exact protected provenance transported to the child."""
    env = experiment_environment(job)
    lines: list[str] = []
    if JOB_ID_ENV in env:
        lines.append(f"# {JOB_ID_ENV}={env[JOB_ID_ENV]}")
    if JOB_JSON_ENV in env:
        lines.append(f"# {JOB_JSON_ENV}={env[JOB_JSON_ENV]}")
    if SUBMISSION_ENV in env:
        lines.append(f"# {SUBMISSION_ENV}={env[SUBMISSION_ENV]}")

    if DECLARATION_ENV not in env:
        lines.insert(0, "# experiment declaration: MISSING (NON-GRADEABLE LEGACY JOB)")
        return "\n".join(lines)

    lines.insert(0, f"# experiment declaration: {job['gradeability']}")
    lines.extend(
        [
            f"# {DECLARATION_SHA_ENV}={env[DECLARATION_SHA_ENV]}",
            f"# {DECLARATION_ENV}={env[DECLARATION_ENV]}",
        ]
    )
    return "\n".join(lines)
