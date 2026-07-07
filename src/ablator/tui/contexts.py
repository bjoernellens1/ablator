"""Kubeconfig context listing/switching -- logic only (no textual import),
so it's unit-testable by monkeypatching subprocess.run.

Respects the same resolution `kubectl` itself uses: the `KUBECONFIG` env
var if set, else `~/.kube/config` -- we don't parse the file ourselves,
we just shell out to `kubectl config ...`, which already implements that
resolution (and the KUBECONFIG colon-separated-merge semantics) exactly.
"""
from __future__ import annotations

import subprocess


class KubectlError(RuntimeError):
    pass


def _run(args: list[str]) -> str:
    try:
        proc = subprocess.run(["kubectl", *args], capture_output=True,
                              text=True, timeout=15)
    except FileNotFoundError:
        raise KubectlError("kubectl not found on PATH")
    except subprocess.TimeoutExpired:
        raise KubectlError(f"kubectl {' '.join(args)} timed out")
    if proc.returncode != 0:
        raise KubectlError(proc.stderr.strip() or f"kubectl {' '.join(args)} failed")
    return proc.stdout


def list_contexts() -> list[str]:
    """All context names in the resolved kubeconfig, in file order."""
    out = _run(["config", "get-contexts", "-o", "name"])
    return [line.strip() for line in out.splitlines() if line.strip()]


def current_context() -> str | None:
    try:
        out = _run(["config", "current-context"])
    except KubectlError:
        return None
    ctx = out.strip()
    return ctx or None


def use_context(name: str) -> None:
    """Switch the resolved kubeconfig's current-context to `name`.

    Raises KubectlError if `name` isn't a valid context -- callers (the
    TUI) should catch this and show it as a user-facing error, not crash.
    """
    _run(["config", "use-context", name])
