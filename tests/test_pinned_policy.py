"""Policy tests for SHA-pinned jobs and machine-level checkout pauses."""

import json
import subprocess

import pytest

from ablator import source_checkout as source
from ablator.queue import Queue, write_pause_flag


def _run(*args, cwd=None):
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


def _repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _run("git", "init", "-b", "main", cwd=repo)
    _run("git", "config", "user.email", "test@example.com", cwd=repo)
    _run("git", "config", "user.name", "Ablator Test", cwd=repo)
    (repo / "payload.txt").write_text("base\n")
    _run("git", "add", "payload.txt", cwd=repo)
    _run("git", "commit", "-m", "base", cwd=repo)
    base = _run("git", "rev-parse", "HEAD", cwd=repo)
    return repo, base


def _tcfg(repo):
    return {"cwd": str(repo), "command": ["python", "train.py"]}


def _cfg(tmp_path, **extra):
    cfg = {
        "git": {"worktree_root": str(tmp_path / "cache")},
        "machines": {"main": {}, "rtx3090": {
            "git_worktree_root": str(tmp_path / "rtx-cache")}},
    }
    cfg.update(extra)
    return cfg


def test_auto_sync_ref_is_mutable_checkout_policy_not_pinned_revision_policy(tmp_path):
    repo, old_sha = _repo(tmp_path)
    (repo / "payload.txt").write_text("newer\n")
    _run("git", "add", "payload.txt", cwd=repo)
    _run("git", "commit", "-m", "newer", cwd=repo)

    cfg = _cfg(tmp_path, urgent_fixes={"auto_sync_ref": "origin/main"})
    prepared = source.prepare_job_source(
        cfg, {"id": "old", "requested_git_sha": old_sha}, "main", _tcfg(repo)
    )
    assert _run("git", "rev-parse", "HEAD", cwd=prepared.checkout_path) == old_sha


def test_explicit_urgent_fix_is_required_of_pinned_revision(tmp_path):
    repo, base = _repo(tmp_path)
    _run("git", "switch", "-c", "fix", cwd=repo)
    (repo / "fix.txt").write_text("required\n")
    _run("git", "add", "fix.txt", cwd=repo)
    _run("git", "commit", "-m", "required fix", cwd=repo)
    fix_sha = _run("git", "rev-parse", "HEAD", cwd=repo)

    _run("git", "switch", "main", cwd=repo)
    (repo / "feature.txt").write_text("feature without fix\n")
    _run("git", "add", "feature.txt", cwd=repo)
    _run("git", "commit", "-m", "feature", cwd=repo)
    feature_sha = _run("git", "rev-parse", "HEAD", cwd=repo)

    cfg = _cfg(tmp_path, urgent_fixes={"repo_cwd": str(repo), "fixes": [
        {"sha": fix_sha, "subject": "required safety fix"}
    ]})
    with pytest.raises(source.SourcePreparationError, match="omits mandatory urgent fix"):
        source.prepare_job_source(
            cfg, {"id": "feature", "requested_git_sha": feature_sha},
            "main", _tcfg(repo)
        )

    prepared = source.prepare_job_source(
        cfg, {"id": "fixed", "requested_git_sha": fix_sha}, "main", _tcfg(repo)
    )
    assert prepared.requested_git_sha == fix_sha


def test_target_machine_uses_its_own_worktree_root(tmp_path):
    repo, sha = _repo(tmp_path)
    cfg = _cfg(tmp_path)
    prepared = source.prepare_job_source(
        cfg, {"id": "remote", "requested_git_sha": sha},
        "rtx3090", _tcfg(repo)
    )
    assert prepared.checkout_path.startswith(str(tmp_path / "rtx-cache"))
    assert _run("git", "rev-parse", "HEAD", cwd=prepared.checkout_path) == sha


def _write_jobs(path, jobs):
    with open(path, "w") as handle:
        for job in jobs:
            handle.write(json.dumps(job) + "\n")


def test_urgent_fix_pause_allows_only_sha_pinned_jobs_when_opted_in(tmp_path):
    q = Queue(str(tmp_path / "queue.jsonl"))
    _write_jobs(q.path, [
        {"id": "legacy", "status": "pending", "machine": "main", "lane": 3},
        {"id": "pinned", "status": "pending", "machine": "main", "lane": 3,
         "requested_git_sha": "1" * 40},
    ])
    write_pause_flag(q.path, "main", "urgent_fix_unsynced", "mutable checkout detached")

    job = q.claim_next("main", allow_pinned_git_while_paused=True)
    assert job["id"] == "pinned"
    remaining = {item["id"]: item for item in q.read()}
    assert remaining["legacy"]["status"] == "pending"


def test_manual_pause_still_blocks_pinned_jobs(tmp_path):
    q = Queue(str(tmp_path / "queue.jsonl"))
    _write_jobs(q.path, [
        {"id": "pinned", "status": "pending", "machine": "main", "lane": 3,
         "requested_git_sha": "1" * 40},
    ])
    write_pause_flag(q.path, "main", "manual_pause", "operator request")
    assert q.claim_next("main", allow_pinned_git_while_paused=True) is None
