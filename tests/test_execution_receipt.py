"""End-to-end contracts for immutable source execution receipts."""

import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from ablator import execution_receipt as receipts
from ablator import runner
from ablator.queue import Queue


def _run(*args, cwd=None):
    return subprocess.run(
        args, cwd=cwd, capture_output=True, text=True, check=True
    ).stdout.strip()


def _repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run("git", "init", "-b", "main", cwd=repo)
    _run("git", "config", "user.email", "test@example.com", cwd=repo)
    _run("git", "config", "user.name", "Ablator Test", cwd=repo)
    (repo / "payload.txt").write_text("clean\n")
    _run("git", "add", "payload.txt", cwd=repo)
    _run("git", "commit", "-m", "source", cwd=repo)
    return repo, _run("git", "rev-parse", "HEAD", cwd=repo)


def _cfg(tmp_path: Path, repo: Path, command: list[str], *, require_pin=True) -> dict:
    config_path = tmp_path / "ablator.json"
    config_path.write_text("{}")
    (tmp_path / "logs").mkdir()
    return {
        "_path": str(config_path),
        "queue": {
            "path": str(tmp_path / "queue.jsonl"),
            "log_dir": str(tmp_path / "logs"),
        },
        "git": {"worktree_root": str(tmp_path / "cache")},
        "machines": {"main": {}},
        "types": {
            "replay": {
                "cwd": str(repo),
                "command": command,
                "require_pinned_git": require_pin,
            }
        },
    }


def _wait_supervisor(_cfg, _job, proc, _base_dir, _q, **_kwargs):
    proc.wait(timeout=10)
    return None


def test_receipt_normalizes_runtime_image_mounts_and_hashes():
    argv = [
        "podman", "run", "--rm", "-v", "/src:/workspace:ro",
        "--mount", "type=bind,src=/data,dst=/data,readonly",
        "image@sha256:abc", "python", "train.py",
    ]
    receipt = receipts.build_prelaunch_receipt(
        cfg={"_path": "/config"},
        job={"id": "j", "requested_git_sha": "a" * 40},
        machine="main",
        type_config={"command": argv},
        argv=argv,
        cwd="/src",
        source_state={
            "commit": "a" * 40, "ref": "DETACHED", "dirty": False,
            "submodules": [],
        },
        source_repo="https://example.invalid/repo.git",
        source_checkout="/src",
        source_lease_id="lease",
        runner_provenance={"config_sha256": "c" * 64},
    )
    assert receipt["schema"] == "ablator.execution/v1"
    assert receipt["source"]["executed_git_sha"] == "a" * 40
    assert receipt["launch"]["runtime"] == "podman"
    assert receipt["launch"]["image"] == "image@sha256:abc"
    assert receipt["launch"]["mounts"] == [
        {"source": "/src", "target": "/workspace", "read_only": True},
        {"source": "/data", "target": "/data", "read_only": True},
    ]
    assert receipt["launch"]["argv_sha256"] == hashlib.sha256(
        json.dumps(argv, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def test_required_pin_type_rejects_unpinned_job_before_launch(tmp_path, monkeypatch):
    repo, _sha = _repo(tmp_path)
    cfg = _cfg(tmp_path, repo, ["python3", "-c", "raise SystemExit(0)"])
    called = False

    def popen(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("must not launch")

    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    status, exit_code = runner.run_job(
        cfg,
        {"id": "unpinned", "type": "replay", "status": "running"},
        "main",
    )
    assert (status, exit_code) == ("failed", None)
    assert called is False


def test_child_job_json_contains_prelaunch_execution_receipt(tmp_path, monkeypatch):
    repo, sha = _repo(tmp_path)
    captured = tmp_path / "captured-job.json"
    code = (
        "import os,pathlib; "
        f"pathlib.Path({str(captured)!r}).write_text(os.environ['ABLATOR_JOB_JSON'])"
    )
    cfg = _cfg(tmp_path, repo, ["python3", "-c", code])
    job = {
        "id": "receipt", "type": "replay", "status": "running",
        "requested_git_sha": sha,
    }
    queue = Queue(cfg["queue"]["path"])
    queue.append([job])
    monkeypatch.setattr(runner, "supervise", _wait_supervisor)

    status, exit_code = runner.run_job(cfg, job, "main", queue)

    assert (status, exit_code) == ("done", 0)
    envelope = json.loads(captured.read_text())
    receipt = envelope["execution_receipt"]
    assert receipt["source"]["requested_git_sha"] == sha
    assert receipt["source"]["executed_git_sha"] == sha
    assert receipt["source"]["ref"] == "DETACHED"
    assert receipt["source"]["dirty"] is False
    queued = queue.read()[0]
    assert queued["execution_receipt"] == receipt
    assert queued["execution_attestation"]["verdict"] == "ACCEPTED"


def test_post_run_mutation_rejects_zero_exit_and_releases_lease(tmp_path, monkeypatch):
    repo, sha = _repo(tmp_path)
    code = "from pathlib import Path; Path('payload.txt').write_text('mutated\\n')"
    cfg = _cfg(tmp_path, repo, ["python3", "-c", code])
    job = {
        "id": "mutator", "type": "replay", "status": "running",
        "requested_git_sha": sha,
    }
    queue = Queue(cfg["queue"]["path"])
    queue.append([job])
    monkeypatch.setattr(runner, "supervise", _wait_supervisor)

    status, exit_code = runner.run_job(cfg, job, "main", queue)

    assert (status, exit_code) == ("failed", 0)
    queued = queue.read()[0]
    assert queued["execution_attestation"]["verdict"] == "REJECTED"
    assert "dirty" in queued["execution_attestation"]["error"]
    with open(queued["source_lease"]["sidecar"]) as handle:
        assert json.load(handle)["active"] is False


def test_prelaunch_receipt_never_contains_environment_values():
    secret = "top-secret-token"
    receipt = receipts.build_prelaunch_receipt(
        cfg={},
        job={"id": "j"},
        machine="main",
        type_config={"command": ["python3", "train.py"], "env": {"TOKEN": secret}},
        argv=["python3", "train.py"],
        cwd="/src",
        source_state=None,
        source_repo=None,
        source_checkout=None,
        source_lease_id=None,
        runner_provenance={"config_sha256": None},
    )
    assert secret not in json.dumps(receipt)


def test_k8s_attestation_records_actual_source_and_image_digest():
    sha = "a" * 40
    submodules_hash = "b" * 64
    payload = {
        "items": [{
            "metadata": {"name": "job-pod"},
            "spec": {"nodeName": "gpu-1"},
            "status": {
                "initContainerStatuses": [{
                    "name": "git-sync",
                    "state": {"terminated": {"message": (
                        f"ABLATOR_SOURCE_V1 requested={sha} executed={sha} "
                        f"ref=DETACHED dirty=false submodules_sha256={submodules_hash}\n"
                    )}},
                }],
                "containerStatuses": [{
                    "name": "trainer",
                    "image": "registry/image:tag",
                    "imageID": "registry/image@sha256:deadbeef",
                }],
            },
        }],
    }
    attestation = runner._k8s_execution_attestation(payload, sha)
    assert attestation["verdict"] == "ACCEPTED"
    assert attestation["source"]["executed_git_sha"] == sha
    assert attestation["source"]["dirty"] is False
    assert attestation["source"]["submodules_sha256"] == submodules_hash
    assert attestation["runtime"] == {
        "pod": "job-pod",
        "node": "gpu-1",
        "image": "registry/image:tag",
        "image_id": "registry/image@sha256:deadbeef",
    }


def test_k8s_attestation_rejects_wrong_commit_or_missing_proof():
    expected = "a" * 40
    wrong = "c" * 40
    payload = {
        "items": [{
            "status": {"initContainerStatuses": [{
                "name": "git-sync",
                "state": {"terminated": {"message": (
                    f"ABLATOR_SOURCE_V1 requested={expected} executed={wrong} "
                    "ref=DETACHED dirty=false submodules_sha256=x\n"
                )}},
            }]},
        }],
    }
    assert runner._k8s_execution_attestation(payload, expected)["verdict"] == "REJECTED"
    assert runner._k8s_execution_attestation({"items": []}, expected)["verdict"] == "REJECTED"
