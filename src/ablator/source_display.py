"""Operator-facing formatting for job source provenance.

Queue records keep full immutable values.  This module is the one place that
turns them into compact CLI/TUI labels so different UIs cannot silently assign
different meanings to the same fields.
"""
from __future__ import annotations


def short_sha(value: object, width: int = 12) -> str:
    if not isinstance(value, str) or not value:
        return "-"
    return value[:width]


def source_state(job: dict) -> str:
    """Compact source state for table columns.

    ``mutable`` means legacy mutable-checkout semantics.  For pinned jobs the
    requested SHA is shown immediately; ``=`` marks a verified execution,
    ``!=`` a recorded mismatch, and ``!`` a preparation/provenance failure.
    """
    requested = job.get("requested_git_sha")
    if not requested:
        return "mutable"
    req = short_sha(requested)
    if job.get("source_prepare_error"):
        return f"{req}!"
    executed = job.get("executed_git_sha")
    if not executed:
        return req
    exe = short_sha(executed)
    if executed == requested:
        return f"{req}="
    return f"{req}!={exe}"


def source_detail(job: dict) -> str:
    """Human-readable multi-field source provenance for selected-job views."""
    requested = job.get("requested_git_sha")
    if not requested:
        return "source: mutable checkout (not Git-pinned)"
    parts = [
        f"requested_git_sha={requested}",
        f"executed_git_sha={job.get('executed_git_sha') or '-'}",
        f"source_repo={job.get('source_repo') or job.get('git_repo') or '-'}",
        f"source_checkout={job.get('source_checkout') or '-'}",
    ]
    if job.get("source_prepare_error"):
        parts.append(f"source_prepare_error={job['source_prepare_error']}")
    return "source: " + " ".join(parts)


def inspect_fields(job: dict) -> dict:
    """Stable structured projection used by ``ablator inspect``."""
    return {
        "source_mode": "pinned" if job.get("requested_git_sha") else "mutable",
        "requested_git_sha": job.get("requested_git_sha"),
        "executed_git_sha": job.get("executed_git_sha"),
        "git_repo": job.get("git_repo"),
        "source_repo": job.get("source_repo"),
        "source_checkout": job.get("source_checkout"),
        "source_prepare_error": job.get("source_prepare_error"),
    }
