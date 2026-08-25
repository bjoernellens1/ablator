import json
import os
import subprocess
from contextlib import contextmanager

import pytest

from ablator import source_checkout
from ablator import source_gc


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
        "git": {
            "worktree_root": str(tmp_path / "cache"),
            "gc_max_age_days": 30,
        },
        "machines": {"main": {}},
    }


def _prepared(tmp_path):
    repo, sha = _repo(tmp_path)
    prepared = source_checkout.prepare_job_source(
        _cfg(tmp_path),
        {"id": "pin", "requested_git_sha": sha},
        "main",
        {"cwd": str(repo), "command": ["python", "train.py"]},
    )
    return repo, sha, prepared


def _unrelated_repo(tmp_path):
    repo = tmp_path / "unrelated"
    repo.mkdir()
    _run("git", "init", "-b", "main", cwd=repo)
    _run("git", "config", "user.email", "test@example.com", cwd=repo)
    _run("git", "config", "user.name", "Ablator Test", cwd=repo)
    (repo / "unrelated.txt").write_text("unrelated\n")
    _run("git", "add", "unrelated.txt", cwd=repo)
    _run("git", "commit", "-m", "unrelated", cwd=repo)
    return repo


def _age_sidecar(checkout, *, last_used_at):
    sidecar = checkout + ".ablator.json"
    data = json.loads(open(sidecar).read())
    data["last_used_at"] = last_used_at
    with open(sidecar, "w") as handle:
        json.dump(data, handle)
    return sidecar


def test_dry_run_lists_stale_entry_without_removing_it(tmp_path):
    _repo_path, _sha, prepared = _prepared(tmp_path)
    source_checkout.release_source(prepared)
    _age_sidecar(prepared.checkout_path, last_used_at=0)
    result = source_gc.gc_worktrees(
        _cfg(tmp_path), "main", [], dry_run=True, max_age_days=1, now=200000,
    )
    assert prepared.checkout_path in result.candidates
    assert result.removed == ()
    assert os.path.isdir(prepared.checkout_path)


def test_running_job_protects_stale_worktree(tmp_path):
    _repo_path, _sha, prepared = _prepared(tmp_path)
    source_checkout.release_source(prepared)
    _age_sidecar(prepared.checkout_path, last_used_at=0)
    jobs = [{
        "id": "pin",
        "status": "running",
        "source_checkout": prepared.checkout_path,
    }]
    result = source_gc.gc_worktrees(
        _cfg(tmp_path), "main", jobs, max_age_days=0, now=200000,
    )
    assert prepared.checkout_path in result.protected
    assert os.path.isdir(prepared.checkout_path)


def test_stale_worktree_is_removed_with_git_metadata(tmp_path):
    repo, _sha, prepared = _prepared(tmp_path)
    source_checkout.release_source(prepared)
    sidecar = _age_sidecar(prepared.checkout_path, last_used_at=0)
    result = source_gc.gc_worktrees(
        _cfg(tmp_path), "main", [], max_age_days=0, now=200000,
    )
    assert prepared.checkout_path in result.removed
    assert not os.path.exists(prepared.checkout_path)
    assert not os.path.exists(sidecar)
    worktrees = _run("git", "worktree", "list", "--porcelain", cwd=repo)
    assert prepared.checkout_path not in worktrees


def test_recent_worktree_is_retained(tmp_path):
    _repo_path, _sha, prepared = _prepared(tmp_path)
    source_checkout.release_source(prepared)
    _age_sidecar(prepared.checkout_path, last_used_at=199999)
    result = source_gc.gc_worktrees(
        _cfg(tmp_path), "main", [], max_age_days=1, now=200000,
    )
    assert prepared.checkout_path in result.retained
    assert os.path.isdir(prepared.checkout_path)


def test_unprovable_orphan_sidecar_and_checkout_are_retained(tmp_path):
    root = tmp_path / "cache"
    checkout = root / "repo-deadbeef" / ("1" * 40)
    checkout.mkdir(parents=True)
    (checkout / "payload").write_text("x")
    sidecar = str(checkout) + ".ablator.json"
    with open(sidecar, "w") as handle:
        json.dump({
            "checkout": str(checkout),
            "sha": "1" * 40,
            "repo": "gone",
            "source_repo_path": str(tmp_path / "missing-repo"),
            "last_used_at": 0,
        }, handle)
    cfg = {"git": {"worktree_root": str(root)}, "machines": {"main": {}}}
    result = source_gc.gc_worktrees(cfg, "main", [], max_age_days=0, now=200000)
    assert result.removed == ()
    assert checkout.exists()
    assert os.path.exists(sidecar)
    assert any("cannot prove" in error for error in result.errors)


def test_active_sidecar_protects_checkout_before_queue_update(tmp_path):
    _repo_path, _sha, prepared = _prepared(tmp_path)
    _age_sidecar(prepared.checkout_path, last_used_at=0)
    result = source_gc.gc_worktrees(
        _cfg(tmp_path), "main", [], max_age_days=0, now=200000,
    )
    assert prepared.checkout_path in result.protected
    assert os.path.isdir(prepared.checkout_path)


def test_sidecar_cannot_escape_managed_cache_root(tmp_path):
    root = tmp_path / "cache"
    root.mkdir()
    outside = tmp_path / "evidence"
    outside.mkdir()
    (outside / "report.json").write_text("important")
    sidecar = root / "malicious.ablator.json"
    sidecar.write_text(json.dumps({
        "checkout": str(outside),
        "sha": "1" * 40,
        "repo": "gone",
        "source_repo_path": str(tmp_path / "missing"),
        "active": False,
        "last_used_at": 0,
    }))

    result = source_gc.gc_worktrees(
        _cfg(tmp_path), "main", [], max_age_days=0, now=200000,
    )

    assert outside.exists()
    assert (outside / "report.json").read_text() == "important"
    assert result.removed == ()
    assert any("outside managed root" in error for error in result.errors)


def test_sidecar_must_be_adjacent_to_claimed_checkout(tmp_path):
    root = tmp_path / "cache"
    checkout = root / "repo" / "sha" / "job"
    checkout.mkdir(parents=True)
    sidecar = root / "wrong.ablator.json"
    sidecar.write_text(json.dumps({
        "checkout": str(checkout),
        "source_repo_path": str(tmp_path / "missing"),
        "active": False,
        "last_used_at": 0,
    }))
    result = source_gc.gc_worktrees(
        _cfg(tmp_path), "main", [], max_age_days=0, now=200000,
    )
    assert checkout.exists()
    assert any("not adjacent" in error for error in result.errors)


def test_forged_source_repo_cannot_mutate_unrelated_repository(tmp_path, monkeypatch):
    _repo_path, _sha, prepared = _prepared(tmp_path)
    unrelated = _unrelated_repo(tmp_path)
    source_checkout.release_source(prepared)
    sidecar = _age_sidecar(prepared.checkout_path, last_used_at=0)
    data = json.loads(open(sidecar).read())
    data["source_repo_path"] = str(unrelated)
    with open(sidecar, "w") as handle:
        json.dump(data, handle)
    mutations = []
    original = source_gc._run_git

    def observe(repo, *args):
        if args[:2] in (("worktree", "remove"), ("worktree", "prune")):
            mutations.append((repo, args))
        return original(repo, *args)

    monkeypatch.setattr(source_gc, "_run_git", observe)
    result = source_gc.gc_worktrees(
        _cfg(tmp_path), "main", [], max_age_days=0, now=200000,
    )

    assert result.removed == ()
    assert mutations == []
    assert os.path.exists(prepared.checkout_path)
    assert os.path.exists(sidecar)
    assert any("does not own checkout" in error for error in result.errors)


def test_forged_lock_path_is_rejected_before_lock_or_removal(tmp_path, monkeypatch):
    _repo_path, _sha, prepared = _prepared(tmp_path)
    source_checkout.release_source(prepared)
    sidecar = _age_sidecar(prepared.checkout_path, last_used_at=0)
    data = json.loads(open(sidecar).read())
    forged = tmp_path / "cache" / "_locks" / "forged.lock"
    data["lock_path"] = str(forged)
    with open(sidecar, "w") as handle:
        json.dump(data, handle)
    acquired = []
    original = source_checkout._locked

    @contextmanager
    def observe(path):
        acquired.append(path)
        with original(path):
            yield

    monkeypatch.setattr(source_checkout, "_locked", observe)
    result = source_gc.gc_worktrees(
        _cfg(tmp_path), "main", [], max_age_days=0, now=200000,
    )

    assert result.removed == ()
    assert acquired == []
    assert os.path.exists(prepared.checkout_path)
    assert any("repository lock does not match" in error for error in result.errors)


def test_missing_checkout_sidecar_cannot_prune_unrelated_repo(tmp_path, monkeypatch):
    unrelated = _unrelated_repo(tmp_path)
    root = tmp_path / "cache"
    checkout = root / "repo" / ("1" * 40) / "missing"
    checkout.parent.mkdir(parents=True)
    sidecar = str(checkout) + ".ablator.json"
    with open(sidecar, "w") as handle:
        json.dump({
            "checkout": str(checkout),
            "source_repo_path": str(unrelated),
            "lock_path": str(root / "_locks" / "forged.lock"),
            "active": False,
            "last_used_at": 0,
        }, handle)
    mutations = []
    original = source_gc._run_git

    def observe(repo, *args):
        if args[:2] in (("worktree", "remove"), ("worktree", "prune")):
            mutations.append((repo, args))
        return original(repo, *args)

    monkeypatch.setattr(source_gc, "_run_git", observe)
    result = source_gc.gc_worktrees(
        _cfg(tmp_path), "main", [], max_age_days=0, now=200000,
    )

    assert result.removed == ()
    assert mutations == []
    assert os.path.exists(sidecar)
    assert any("cannot prove" in error for error in result.errors)


def test_prune_failure_is_reported_and_sidecar_retained(tmp_path, monkeypatch):
    _repo_path, _sha, prepared = _prepared(tmp_path)
    source_checkout.release_source(prepared)
    sidecar = _age_sidecar(prepared.checkout_path, last_used_at=0)
    original = source_gc._run_git

    def fail_prune(repo, *args):
        if args == ("worktree", "prune"):
            return subprocess.CompletedProcess([], 1, "", "prune failed")
        return original(repo, *args)

    monkeypatch.setattr(source_gc, "_run_git", fail_prune)
    result = source_gc.gc_worktrees(
        _cfg(tmp_path), "main", [], max_age_days=0, now=200000,
    )
    assert result.removed == ()
    assert os.path.exists(sidecar)
    assert any("worktree prune failed" in error for error in result.errors)


def test_invalid_negative_age_remains_rejected(tmp_path):
    with pytest.raises(ValueError, match=">= 0"):
        source_gc.gc_worktrees(
            _cfg(tmp_path), "main", [], max_age_days=-1,
        )


def test_interrupted_removal_is_reported_and_preserves_sidecar(tmp_path, monkeypatch):
    _repo_path, _sha, prepared = _prepared(tmp_path)
    source_checkout.release_source(prepared)
    sidecar = _age_sidecar(prepared.checkout_path, last_used_at=0)

    def _fail_remove(_root, _entry, _trusted):
        return "error", "simulated interrupted cleanup"

    monkeypatch.setattr(source_gc, "_remove_entry", _fail_remove)
    result = source_gc.gc_worktrees(
        _cfg(tmp_path), "main", [], max_age_days=0, now=200000,
    )

    assert result.removed == ()
    assert result.errors == ("simulated interrupted cleanup",)
    assert os.path.isdir(prepared.checkout_path)
    assert os.path.exists(sidecar)
