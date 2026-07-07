"""Tests for the pre-dispatch urgent-fix currency gate (urgent_fixes.py)
and its wiring into runner.run_loop().

Uses real throwaway git repos (cheap, and exercises actual `git
merge-base --is-ancestor` / `git pull --ff-only` semantics rather than
mocking subprocess, since this module's whole job is to get those git
semantics right)."""
import subprocess

import pytest

from ablator import urgent_fixes as ufmod
from ablator.queue import Queue, is_paused, pause_flag_path


def _git(cwd, *args):
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True,
                       text=True, check=True)
    return r.stdout.strip()


def _init_repo(path):
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "t@example.com")
    _git(path, "config", "user.name", "t")
    (path / "f.txt").write_text("0\n")
    _git(path, "add", "f.txt")
    _git(path, "commit", "-q", "-m", "init")
    return _git(path, "rev-parse", "HEAD")


def _commit(path, msg):
    (path / "f.txt").write_text(msg + "\n")
    _git(path, "add", "f.txt")
    _git(path, "commit", "-q", "-m", msg)
    return _git(path, "rev-parse", "HEAD")


# ------------------------------------------------------------- unit-level

def test_missing_fixes_empty_when_ancestor(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    fix_sha = _commit(repo, "the fix")
    assert ufmod.missing_fixes(str(repo), [{"sha": fix_sha}]) == []


def test_missing_fixes_reports_unreached_commit(tmp_path):
    origin = tmp_path / "origin"
    _init_repo(origin)
    fix_sha = _commit(origin, "the fix")

    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(origin), str(clone)], check=True)
    # Roll the clone back before the fix commit landed upstream.
    parent = _git(clone, "rev-parse", "HEAD~1")
    _git(clone, "reset", "-q", "--hard", parent)

    missing = ufmod.missing_fixes(str(clone), [{"sha": fix_sha}])
    assert len(missing) == 1


def test_missing_fixes_unknown_sha_counts_as_missing(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    assert ufmod.missing_fixes(str(repo), [{"sha": "deadbeef" * 5}]) != []


# --------------------------------------------------------- enforce_urgent_fixes

def _cfg(tmp_path, repo_cwd=None, fixes=None):
    return {
        "queue": {"path": str(tmp_path / "queue.jsonl")},
        "urgent_fixes": ({"repo_cwd": repo_cwd, "fixes": fixes or []}
                        if repo_cwd else {}),
    }


def test_enforce_noop_when_not_configured(tmp_path):
    cfg = _cfg(tmp_path)
    q = Queue(str(tmp_path / "queue.jsonl"))
    assert ufmod.enforce_urgent_fixes(cfg, "main", q) is True


def test_enforce_noop_when_already_current(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    fix_sha = _commit(repo, "the fix")
    cfg = _cfg(tmp_path, repo_cwd=str(repo), fixes=[{"sha": fix_sha}])
    q = Queue(str(tmp_path / "queue.jsonl"))
    assert ufmod.enforce_urgent_fixes(cfg, "main", q) is True
    assert not is_paused(cfg["queue"]["path"], "main")


def test_enforce_auto_pulls_clean_fast_forward(tmp_path):
    origin = tmp_path / "origin"
    _init_repo(origin)
    fix_sha = _commit(origin, "the fix")

    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(origin), str(clone)], check=True)
    parent = _git(clone, "rev-parse", "HEAD~1")
    _git(clone, "reset", "-q", "--hard", parent)
    assert ufmod.missing_fixes(str(clone), [{"sha": fix_sha}])

    cfg = _cfg(tmp_path, repo_cwd=str(clone), fixes=[{"sha": fix_sha}])
    q = Queue(str(tmp_path / "queue.jsonl"))
    result = ufmod.enforce_urgent_fixes(cfg, "main", q)
    assert result is True
    assert not is_paused(cfg["queue"]["path"], "main")
    # the clone's HEAD should now include the fix commit
    assert ufmod.missing_fixes(str(clone), [{"sha": fix_sha}]) == []


def test_enforce_pauses_on_dirty_tree(tmp_path):
    origin = tmp_path / "origin"
    _init_repo(origin)
    fix_sha = _commit(origin, "the fix")

    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(origin), str(clone)], check=True)
    parent = _git(clone, "rev-parse", "HEAD~1")
    _git(clone, "reset", "-q", "--hard", parent)
    (clone / "f.txt").write_text("dirty edit\n")  # uncommitted change

    cfg = _cfg(tmp_path, repo_cwd=str(clone), fixes=[{"sha": fix_sha}])
    q = Queue(str(tmp_path / "queue.jsonl"))
    result = ufmod.enforce_urgent_fixes(cfg, "main", q)
    assert result is False
    assert is_paused(cfg["queue"]["path"], "main")
    # never silently mutated the dirty file
    assert (clone / "f.txt").read_text() == "dirty edit\n"


def test_enforce_pauses_on_diverged_history(tmp_path):
    """A clone with a local commit the origin doesn't have -- ff-only
    pull cannot resolve this; the gate must pause, never force/rebase."""
    origin = tmp_path / "origin"
    _init_repo(origin)
    fix_sha = _commit(origin, "the fix")

    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(origin), str(clone)], check=True)
    parent = _git(clone, "rev-parse", "HEAD~1")
    _git(clone, "reset", "-q", "--hard", parent)
    _commit(clone, "divergent local commit")  # diverges from origin

    cfg = _cfg(tmp_path, repo_cwd=str(clone), fixes=[{"sha": fix_sha}])
    q = Queue(str(tmp_path / "queue.jsonl"))
    result = ufmod.enforce_urgent_fixes(cfg, "main", q)
    assert result is False
    assert is_paused(cfg["queue"]["path"], "main")


def test_enforce_respects_existing_pause_flag(tmp_path):
    from ablator.queue import write_pause_flag
    repo = tmp_path / "repo"
    _init_repo(repo)
    fix_sha = _commit(repo, "the fix")
    cfg = _cfg(tmp_path, repo_cwd=str(repo), fixes=[{"sha": "unreachable" * 4}])
    q = Queue(str(tmp_path / "queue.jsonl"))
    write_pause_flag(cfg["queue"]["path"], "main", "manual_pause", "operator")
    # Already paused for an unrelated reason -- gate must not attempt any
    # git operations or double-pause; just report unsafe-to-dispatch.
    assert ufmod.enforce_urgent_fixes(cfg, "main", q) is False


def test_load_urgent_fixes_missing_config_returns_none(tmp_path):
    assert ufmod.load_urgent_fixes({}) == (None, [])
    assert ufmod.load_urgent_fixes({"urgent_fixes": {}}) == (None, [])
