"""Tests for immutable per-job Git checkout materialization."""

import json
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
    assert again.checkout_path != prepared.checkout_path
    assert prepared.lease is not None
    assert again.lease is not None
    assert prepared.lease.lease_id != again.lease.lease_id


def test_distinct_jobs_at_same_sha_never_share_execution_worktree(tmp_path):
    repo, sha = _repo(tmp_path)
    first = source.prepare_job_source(
        _cfg(tmp_path), {"id": "first", "requested_git_sha": sha}, "main", _tcfg(repo)
    )
    second = source.prepare_job_source(
        _cfg(tmp_path), {"id": "second", "requested_git_sha": sha}, "main", _tcfg(repo)
    )
    assert first.checkout_path != second.checkout_path


def test_release_marks_only_own_lease_inactive(tmp_path):
    repo, sha = _repo(tmp_path)
    first = source.prepare_job_source(
        _cfg(tmp_path), {"id": "first", "requested_git_sha": sha}, "main", _tcfg(repo)
    )
    second = source.prepare_job_source(
        _cfg(tmp_path), {"id": "second", "requested_git_sha": sha}, "main", _tcfg(repo)
    )

    source.release_source(first)

    assert source.read_source_lease(first.lease)["active"] is False
    assert source.read_source_lease(second.lease)["active"] is True


def test_recursive_submodules_are_initialized_at_recorded_commits(tmp_path):
    child = tmp_path / "child"
    child.mkdir()
    _run("git", "init", "-b", "main", cwd=child)
    _run("git", "config", "user.email", "test@example.com", cwd=child)
    _run("git", "config", "user.name", "Ablator Test", cwd=child)
    (child / "child.txt").write_text("child\n")
    _run("git", "add", "child.txt", cwd=child)
    _run("git", "commit", "-m", "child", cwd=child)
    child_sha = _run("git", "rev-parse", "HEAD", cwd=child)

    repo, _ = _repo(tmp_path)
    _run(
        "git", "-c", "protocol.file.allow=always", "submodule", "add",
        str(child), "deps/child", cwd=repo,
    )
    _run("git", "commit", "-am", "add child", cwd=repo)
    sha = _run("git", "rev-parse", "HEAD", cwd=repo)

    prepared = source.prepare_job_source(
        _cfg(tmp_path), {"id": "submodules", "requested_git_sha": sha},
        "main", _tcfg(repo),
    )

    assert (prepared.checkout_path and
            os.path.exists(os.path.join(prepared.checkout_path, "deps/child/child.txt")))
    assert prepared.state is not None
    assert prepared.state["submodules"] == [{
        "path": "deps/child",
        "sha": child_sha,
        "dirty": False,
    }]


def test_submodule_drift_is_rejected_by_checkout_state(tmp_path):
    child = tmp_path / "child"
    child.mkdir()
    _run("git", "init", "-b", "main", cwd=child)
    _run("git", "config", "user.email", "test@example.com", cwd=child)
    _run("git", "config", "user.name", "Ablator Test", cwd=child)
    (child / "child.txt").write_text("child\n")
    _run("git", "add", "child.txt", cwd=child)
    _run("git", "commit", "-m", "child", cwd=child)
    repo, _ = _repo(tmp_path)
    _run(
        "git", "-c", "protocol.file.allow=always", "submodule", "add",
        str(child), "deps/child", cwd=repo,
    )
    _run("git", "commit", "-am", "add child", cwd=repo)
    sha = _run("git", "rev-parse", "HEAD", cwd=repo)
    prepared = source.prepare_job_source(
        _cfg(tmp_path), {"id": "submodules", "requested_git_sha": sha},
        "main", _tcfg(repo),
    )
    with open(os.path.join(prepared.checkout_path, "deps/child/child.txt"), "a") as handle:
        handle.write("drift\n")

    with pytest.raises(source.SourcePreparationError, match="submodule.*dirty"):
        source.capture_checkout_state(prepared.checkout_path)


def test_failed_submodule_initialization_retains_recoverable_evidence(tmp_path, monkeypatch):
    repo, sha = _repo(tmp_path)

    def fail(_checkout):
        raise source.SourcePreparationError("submodule setup failed")

    monkeypatch.setattr(source, "_initialize_submodules", fail)
    with pytest.raises(
        source.SourcePreparationError,
        match="submodule setup failed.*retained.*ablator.json",
    ):
        source.prepare_job_source(
            _cfg(tmp_path), {"id": "broken", "requested_git_sha": sha},
            "main", _tcfg(repo),
        )

    cache = tmp_path / "cache"
    sidecars = list(cache.rglob("*.ablator.json"))
    assert len(sidecars) == 1
    evidence = json.loads(sidecars[0].read_text())
    assert evidence["active"] is False
    assert evidence["materialization_state"] == "failed"
    assert evidence["materialization_error"] == "submodule setup failed"
    assert os.path.isdir(evidence["checkout"])


def test_post_materialization_config_error_releases_lease(tmp_path):
    repo, sha = _repo(tmp_path)
    # The source path makes the container-reachability preflight pass, but it
    # is only a working-directory flag, not a bind. Read-only bind validation
    # therefore fails after the unique worktree has been materialized.
    tcfg = {
        "cwd": str(repo),
        "command": ["docker", "run", "-w", str(repo), "image:test"],
    }

    with pytest.raises(source.SourcePreparationError, match="does not bind"):
        source.prepare_job_source(
            _cfg(tmp_path), {"id": "bad-mount", "requested_git_sha": sha},
            "main", tcfg,
        )

    sidecars = list((tmp_path / "cache").rglob("*.ablator.json"))
    assert len(sidecars) == 1
    lease = json.loads(sidecars[0].read_text())
    assert lease["active"] is False
    assert lease["released_at"] >= lease["created_at"]


def test_unavailable_commit_retains_failed_lease_evidence(tmp_path):
    repo, _sha = _repo(tmp_path)
    with pytest.raises(source.SourcePreparationError, match="fetch requested Git SHA"):
        source.prepare_job_source(
            _cfg(tmp_path),
            {"id": "missing", "requested_git_sha": "1" * 40},
            "main", _tcfg(repo),
        )

    cache = tmp_path / "cache"
    sidecars = list(cache.rglob("*.ablator.json"))
    assert len(sidecars) == 1
    evidence = json.loads(sidecars[0].read_text())
    assert evidence["active"] is False
    assert evidence["materialization_state"] == "failed"
    assert evidence["sha"] == "1" * 40


def test_repo_cwd_template_is_supported_for_container_mounts(tmp_path):
    repo, sha = _repo(tmp_path)
    tcfg = {
        "cwd": str(repo),
        "command": ["docker", "run", "-v", "{repo_cwd}:/src", "image:test"],
    }
    prepared = source.prepare_job_source(
        _cfg(tmp_path), {"id": "pin", "requested_git_sha": sha}, "main", tcfg
    )
    assert f"{prepared.checkout_path}:/src:ro" in prepared.type_config["command"]


def test_long_container_mount_form_is_read_only(tmp_path):
    repo, sha = _repo(tmp_path)
    tcfg = {
        "cwd": str(repo),
        "command": [
            "podman", "run", "--mount",
            "type=bind,src={repo_cwd},dst=/workspace/project",
            "image:test",
        ],
    }
    prepared = source.prepare_job_source(
        _cfg(tmp_path), {"id": "pin", "requested_git_sha": sha}, "main", tcfg
    )
    assert (
        f"type=bind,src={prepared.checkout_path},dst=/workspace/project,readonly"
        in prepared.type_config["command"]
    )


def test_every_checkout_descendant_bind_is_read_only(tmp_path):
    repo, sha = _repo(tmp_path)
    (repo / "configs").mkdir()
    tcfg = {
        "cwd": str(repo),
        "command": [
            "podman", "run",
            "-v", "{repo_cwd}:/workspace/project",
            "--mount", "type=bind,src={repo_cwd}/configs,dst=/configs",
            "image:test",
        ],
    }
    prepared = source.prepare_job_source(
        _cfg(tmp_path), {"id": "descendants", "requested_git_sha": sha},
        "main", tcfg,
    )
    command = prepared.type_config["command"]
    assert f"{prepared.checkout_path}:/workspace/project:ro" in command
    assert (
        f"type=bind,src={prepared.checkout_path}/configs,dst=/configs,readonly"
        in command
    )


def test_checkout_descendant_symlink_escape_is_rejected(tmp_path):
    repo, _sha = _repo(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    os.symlink(outside, repo / "escape")
    _run("git", "add", "escape", cwd=repo)
    _run("git", "commit", "-m", "add escape", cwd=repo)
    sha = _run("git", "rev-parse", "HEAD", cwd=repo)
    tcfg = {
        "cwd": str(repo),
        "command": [
            "podman", "run",
            "-v", "{repo_cwd}:/workspace/project",
            "-v", "{repo_cwd}/escape:/escape",
            "image:test",
        ],
    }
    with pytest.raises(source.SourcePreparationError, match="bind.*escapes"):
        source.prepare_job_source(
            _cfg(tmp_path), {"id": "escape", "requested_git_sha": sha},
            "main", tcfg,
        )


def test_explicit_git_repo_must_match_configured_checkout_origin(tmp_path):
    repo, sha = _repo(tmp_path)
    _run(
        "git", "remote", "add", "origin",
        "https://github.com/example/actual.git", cwd=repo,
    )

    with pytest.raises(source.SourcePreparationError, match="git.repo.*origin"):
        source.prepare_job_source(
            _cfg(tmp_path),
            {
                "id": "wrong-repo",
                "requested_git_sha": sha,
                "git_repo": "https://github.com/example/unrelated.git",
            },
            "main", _tcfg(repo),
        )


def test_explicit_local_git_repo_matching_checkout_origin_is_accepted(tmp_path):
    repo, sha = _repo(tmp_path)
    bare = tmp_path / "origin.git"
    _run("git", "clone", "--bare", str(repo), str(bare), cwd=tmp_path)
    _run("git", "remote", "add", "origin", str(bare), cwd=repo)

    prepared = source.prepare_job_source(
        _cfg(tmp_path),
        {"id": "local-origin", "requested_git_sha": sha, "git_repo": str(bare)},
        "main", _tcfg(repo),
    )

    assert prepared.checkout_path
    assert source.capture_checkout_state(prepared.checkout_path)["commit"] == sha


def test_pinned_direct_process_disables_bytecode_writes(tmp_path):
    repo, sha = _repo(tmp_path)
    prepared = source.prepare_job_source(
        _cfg(tmp_path),
        {"id": "pin", "requested_git_sha": sha},
        "main",
        {"cwd": str(repo), "command": ["python3", "train.py"]},
    )
    assert prepared.type_config["env"]["PYTHONDONTWRITEBYTECODE"] == "1"


def test_pinned_container_without_source_mount_is_rejected(tmp_path):
    repo, sha = _repo(tmp_path)
    tcfg = {"cwd": str(repo), "command": ["docker", "run", "image:test"]}
    with pytest.raises(source.SourcePreparationError, match="does not expose"):
        source.prepare_job_source(
            _cfg(tmp_path), {"id": "pin", "requested_git_sha": sha}, "main", tcfg
        )


def test_dirty_prior_worktree_is_never_reused(tmp_path):
    repo, sha = _repo(tmp_path)
    job = {"id": "pin", "requested_git_sha": sha}
    prepared = source.prepare_job_source(_cfg(tmp_path), job, "main", _tcfg(repo))
    with open(os.path.join(prepared.checkout_path, "payload.txt"), "a") as handle:
        handle.write("dirty\n")
    with pytest.raises(source.SourcePreparationError, match="dirty"):
        source.capture_checkout_state(prepared.checkout_path)
    replacement = source.prepare_job_source(_cfg(tmp_path), job, "main", _tcfg(repo))
    assert replacement.checkout_path != prepared.checkout_path
    assert replacement.state["dirty"] is False


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
