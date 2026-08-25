import json
import os
import subprocess

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


def _age_sidecar(checkout, *, last_used_at):
    sidecar = checkout + ".ablator.json"
    data = json.loads(open(sidecar).read())
    data["last_used_at"] = last_used_at
    with open(sidecar, "w") as handle:
        json.dump(data, handle)
    return sidecar


def test_dry_run_lists_stale_entry_without_removing_it(tmp_path):
    _repo_path, _sha, prepared = _prepared(tmp_path)
    _age_sidecar(prepared.checkout_path, last_used_at=0)
    result = source_gc.gc_worktrees(
        _cfg(tmp_path), "main", [], dry_run=True, max_age_days=1, now=200000,
    )
    assert prepared.checkout_path in result.candidates
    assert result.removed == ()
    assert os.path.isdir(prepared.checkout_path)


def test_running_job_protects_stale_worktree(tmp_path):
    _repo_path, _sha, prepared = _prepared(tmp_path)
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
    _age_sidecar(prepared.checkout_path, last_used_at=199999)
    result = source_gc.gc_worktrees(
        _cfg(tmp_path), "main", [], max_age_days=1, now=200000,
    )
    assert prepared.checkout_path in result.retained
    assert os.path.isdir(prepared.checkout_path)


def test_orphan_sidecar_and_checkout_can_be_removed(tmp_path):
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
    assert str(checkout) in result.removed
    assert not checkout.exists()
    assert not os.path.exists(sidecar)


def test_sidecar_cannot_escape_the_configured_cache_root(tmp_path):
    root = tmp_path / "cache"
    root.mkdir()
    outside = tmp_path / "must-survive"
    outside.mkdir()
    (outside / "payload").write_text("keep")
    sidecar = root / "forged.ablator.json"
    sidecar.write_text(json.dumps({
        "checkout": str(outside),
        "sha": "1" * 40,
        "repo": "gone",
        "source_repo_path": str(tmp_path / "missing-repo"),
        "last_used_at": 0,
    }))
    cfg = {"git": {"worktree_root": str(root)}, "machines": {"main": {}}}

    result = source_gc.gc_worktrees(cfg, "main", [], max_age_days=0, now=200000)

    assert result.candidates == ()
    assert outside.is_dir()
    assert (outside / "payload").read_text() == "keep"


def test_interrupted_removal_is_reported_and_preserves_sidecar(tmp_path, monkeypatch):
    _repo_path, _sha, prepared = _prepared(tmp_path)
    sidecar = _age_sidecar(prepared.checkout_path, last_used_at=0)

    def _fail_remove(_entry):
        return "simulated interrupted cleanup"

    monkeypatch.setattr(source_gc, "_remove_entry", _fail_remove)
    result = source_gc.gc_worktrees(
        _cfg(tmp_path), "main", [], max_age_days=0, now=200000,
    )

    assert result.removed == ()
    assert result.errors == ("simulated interrupted cleanup",)
    assert os.path.isdir(prepared.checkout_path)
    assert os.path.exists(sidecar)
