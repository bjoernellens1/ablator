"""Tests for immutable per-job Git checkout materialization."""

import os
import subprocess

import pytest

from ablator import source_checkout as source


def _run(*args, cwd=None):
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


def _repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _run("git", "init", "-b", "main", cwd=repo)
    _run("git", "config", "user.email", "test@example.com", cwd=repo)
    _run("git", "config", "user.name", "Ablator Test", cwd=repo)
    (repo / "payload.txt").write_text("one\n")
    _run("git", "add", "payload.txt", cwd=repo)
    _run("git", "commit", "-m", "one", cwd=repo)
    sha = _run("git", "rev-parse", "HEAD", cwd=repo)
    return repo, sha


def _cfg(tmp_path):
    return {
        "git": {"worktree_root": str(tmp_path / "cache")},
        "machines": {"main": {}},
    }


def _tcfg(repo):
    return {
        "cwd": str(repo),
        "command": [
            "podman", "run", "--rm",
            "-v", f"{repo}:/workspace/project",
            "-w", "/workspace/project",
            "image:test", "python", "train.py",
        ],
        "env": {"HOST_SOURCE": str(repo)},
    }


def test_unpinned_job_is_noop_copy(tmp_path):
    repo, _sha = _repo(tmp_path)
    tcfg = _tcfg(repo)
    prepared = source.prepare_job_source(_cfg(tmp_path), {"id": "legacy"}, "main", tcfg)
    assert prepared.checkout_path is None
    assert prepared.type_config == tcfg
    assert prepared.type_config is not tcfg


def test_pinned_job_materializes_detached_clean_worktree_and_rewrites_paths(tmp_path):
    repo, sha = _repo(tmp_path)
    job = {"id": "pin", "requested_git_sha": sha}
    prepared = source.prepare_job_source(_cfg(tmp_path), job, "main", _tcfg(repo))

    assert prepared.checkout_path
    assert prepared.checkout_path != str(repo)
    assert _run("git", "rev-parse", "HEAD", cwd=prepared.checkout_path) == sha
    assert _run("git", "rev-parse", "--abbrev-ref", "HEAD", cwd=prepared.checkout_path) == "HEAD"
    assert _run("git", "status", "--porcelain", cwd=prepared.checkout_path) == ""
    assert prepared.type_config["cwd"] == prepared.checkout_path
    assert any(prepared.checkout_path in token for token in prepared.type_config["command"])
    assert prepared.type_config["env"]["HOST_SOURCE"] == prepared.checkout_path
    assert all(str(repo) not in token for token in prepared.type_config["command"])

    again = source.prepare_job_source(_cfg(tmp_path), job, "main", _tcfg(repo))
    assert again.checkout_path == prepared.checkout_path


def test_repo_cwd_template_is_supported_for_container_mounts(tmp_path):
    repo, sha = _repo(tmp_path)
    tcfg = {
        "cwd": str(repo),
        "command": ["docker", "run", "-v", "{repo_cwd}:/src", "image:test"],
    }
    prepared = source.prepare_job_source(
        _cfg(tmp_path), {"id": "pin", "requested_git_sha": sha}, "main", tcfg
    )
    assert f"{prepared.checkout_path}:/src" in prepared.type_config["command"]


def test_pinned_container_without_source_mount_is_rejected(tmp_path):
    repo, sha = _repo(tmp_path)
    tcfg = {"cwd": str(repo), "command": ["docker", "run", "image:test"]}
    with pytest.raises(source.SourcePreparationError, match="does not expose"):
        source.prepare_job_source(
            _cfg(tmp_path), {"id": "pin", "requested_git_sha": sha}, "main", tcfg
        )


def test_dirty_cached_worktree_is_never_reused(tmp_path):
    repo, sha = _repo(tmp_path)
    job = {"id": "pin", "requested_git_sha": sha}
    prepared = source.prepare_job_source(_cfg(tmp_path), job, "main", _tcfg(repo))
    with open(os.path.join(prepared.checkout_path, "payload.txt"), "a") as handle:
        handle.write("dirty\n")
    with pytest.raises(source.SourcePreparationError, match="dirty"):
        source.prepare_job_source(_cfg(tmp_path), job, "main", _tcfg(repo))


def test_git_repo_can_bootstrap_when_type_cwd_is_absent(tmp_path):
    repo, sha = _repo(tmp_path)
    bare = tmp_path / "origin.git"
    _run("git", "clone", "--bare", str(repo), str(bare))
    tcfg = {"command": ["python", "train.py"]}
    prepared = source.prepare_job_source(
        _cfg(tmp_path),
        {"id": "pin", "requested_git_sha": sha, "git_repo": str(bare)},
        "main",
        tcfg,
    )
    assert _run("git", "rev-parse", "HEAD", cwd=prepared.checkout_path) == sha
    assert prepared.type_config["cwd"] == prepared.checkout_path


def test_verify_executed_provenance_is_hard_contract():
    sha = "1" * 40
    job = {"id": "pin", "requested_git_sha": sha}
    assert source.verify_executed_provenance(
        job, {"commit": sha, "dirty": False}
    ) == sha

    with pytest.raises(source.SourcePreparationError, match="mismatch"):
        source.verify_executed_provenance(job, {"commit": "2" * 40, "dirty": False})

    with pytest.raises(source.SourcePreparationError, match="dirty or unreadable"):
        source.verify_executed_provenance(job, {"commit": sha, "dirty": True})

    assert source.verify_executed_provenance({"id": "legacy"}, {}) is None
