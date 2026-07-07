"""Pod status/log lookup for a single ablator k8s-dispatched job -- scoped
to "what pod is this job's, and is it healthy", not a general Kubernetes
resource browser. Shells out to kubectl against whatever context is
currently active (see contexts.py to switch it).
"""
from __future__ import annotations

import subprocess


def pod_status_line(namespace: str, job_id: str, k8s_job_name: str) -> str:
    """One-line pod phase summary for the given ablator job, e.g.
    'Running (node k3s-wk-gpu2)' or 'no pod found' / an error string.
    Never raises -- this feeds a display widget, a failed lookup is just
    shown as text, not a crash.
    """
    try:
        proc = subprocess.run(
            ["kubectl", "get", "pods", "-n", namespace,
             "-l", f"job-name={k8s_job_name}",
             "-o", "custom-columns=PHASE:.status.phase,NODE:.spec.nodeName",
             "--no-headers"],
            capture_output=True, text=True, timeout=15)
    except FileNotFoundError:
        return "kubectl not found on PATH"
    except subprocess.TimeoutExpired:
        return "kubectl timed out"
    if proc.returncode != 0:
        return (proc.stderr or proc.stdout).strip() or "kubectl error"
    line = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""
    if not line:
        return "no pod found (not yet scheduled, or already cleaned up)"
    parts = line.split()
    phase = parts[0] if parts else "?"
    node = parts[1] if len(parts) > 1 else "?"
    return f"{phase} (node {node})"


def recent_log_tail(namespace: str, job_id: str, k8s_job_name: str,
                    lines: int = 20) -> str:
    """Best-effort recent log tail for the job's pod. Empty string (not an
    exception) if there's no pod/logs yet."""
    try:
        pods = subprocess.run(
            ["kubectl", "get", "pods", "-n", namespace,
             "-l", f"job-name={k8s_job_name}",
             "-o", "jsonpath={.items[0].metadata.name}"],
            capture_output=True, text=True, timeout=15)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    pod = pods.stdout.strip()
    if pods.returncode != 0 or not pod:
        return ""
    try:
        logs = subprocess.run(
            ["kubectl", "logs", pod, "-n", namespace, f"--tail={lines}"],
            capture_output=True, text=True, timeout=15)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    return logs.stdout
