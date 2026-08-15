"""Tests for pause_revalidation.py (splatograph issue #629): a pause flag
must be re-validated against the SPECIFIC check that caused it, not left
to latch forever nor cleared on a blind timer.

Covers the four contractual cases named in #629:
  1. transient failure -> auto-clears once the guarded condition re-passes
  2. persistent failure -> stays paused
  3. human-set pause (manual_pause) -> never auto-cleared
  4. unknown/unregistered category -> stays paused (bottom of the lattice,
     never a permissive default)

Also covers the urgent_fixes.py-specific re-checker
(_revalidate_urgent_fix_unsynced) against real throwaway git repos, since
that is the actual category #629 was filed against, and the auditability
requirement (a cleared pause is recorded, not silently gone).
"""
from __future__ import annotations

import subprocess

import pytest

from ablator import pause_revalidation as pr
from ablator import urgent_fixes as ufmod
from ablator.queue import (Queue, is_paused, pause_flag_path,
                           read_pause_flag, write_pause_flag)


def _cfg(tmp_path, **urgent_fixes_kwargs):
    uf = {}
    if urgent_fixes_kwargs:
        uf = dict(urgent_fixes_kwargs)
    return {
        "queue": {"path": str(tmp_path / "queue.jsonl")},
        "urgent_fixes": uf,
    }


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


# --------------------------------------------------------- generic registry


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch):
    """Each test gets its own copy of the revalidator registry so tests
    registering a throwaway category don't leak into other tests (the
    real registry is process-global, populated at import time by modules
    like urgent_fixes.py)."""
    saved = dict(pr._AUTO_REVALIDATORS)
    yield
    pr._AUTO_REVALIDATORS.clear()
    pr._AUTO_REVALIDATORS.update(saved)


def test_revalidate_noop_when_not_paused(tmp_path):
    cfg = _cfg(tmp_path)
    q = Queue(str(tmp_path / "queue.jsonl"))
    assert pr.revalidate_pause(cfg, "main", q) is True
    assert not is_paused(cfg["queue"]["path"], "main")


def test_register_auto_revalidator_rejects_manual_pause():
    with pytest.raises(ValueError):
        pr.register_auto_revalidator("manual_pause", lambda cfg, m: (True, "x"))


def test_transient_failure_auto_clears_on_recheck(tmp_path):
    """category with a registered revalidator that now reports resolved ->
    the flag is cleared and the machine is dispatch-eligible again."""
    calls = []

    def _recheck(cfg, machine):
        calls.append(machine)
        return True, "transient condition cleared"

    pr.register_auto_revalidator("flaky_transient", _recheck)
    cfg = _cfg(tmp_path)
    q = Queue(str(tmp_path / "queue.jsonl"))
    write_pause_flag(cfg["queue"]["path"], "main", "flaky_transient",
                     "evidence of a transient failure")

    assert pr.revalidate_pause(cfg, "main", q) is True
    assert not is_paused(cfg["queue"]["path"], "main")
    assert calls == ["main"]

    # And the clear is auditable, not silent.
    audit_path = tmp_path / "pause_audit.log"
    assert audit_path.exists()
    text = audit_path.read_text()
    assert "SET machine=main category=flaky_transient" in text
    assert "CLEAR machine=main was_category=flaky_transient" in text
    assert "auto_revalidate:flaky_transient" in text
    assert "transient condition cleared" in text


def test_persistent_failure_stays_paused(tmp_path):
    def _recheck(cfg, machine):
        return False, "still broken"

    pr.register_auto_revalidator("persistent_thing", _recheck)
    cfg = _cfg(tmp_path)
    q = Queue(str(tmp_path / "queue.jsonl"))
    write_pause_flag(cfg["queue"]["path"], "main", "persistent_thing",
                     "evidence of a real, ongoing failure")

    assert pr.revalidate_pause(cfg, "main", q) is False
    assert is_paused(cfg["queue"]["path"], "main")
    # Not cleared -- no spurious CLEAR audit line for this pause.
    info = read_pause_flag(cfg["queue"]["path"], "main")
    assert info["category"] == "persistent_thing"


def test_human_set_pause_never_auto_cleared(tmp_path):
    """manual_pause has no registered revalidator by construction
    (register_auto_revalidator refuses to register one) -- confirm the
    end-to-end behavior through revalidate_pause is "leave it alone"."""
    cfg = _cfg(tmp_path)
    q = Queue(str(tmp_path / "queue.jsonl"))
    write_pause_flag(cfg["queue"]["path"], "main", "manual_pause",
                     "operator-issued via `ablator pause`")

    assert pr.revalidate_pause(cfg, "main", q) is False
    assert is_paused(cfg["queue"]["path"], "main")


def test_unknown_category_stays_paused(tmp_path):
    """A category nobody has registered a re-checker for (a garbled flag,
    or a future auto-pause category whose revalidator hasn't shipped yet)
    must never be treated as auto-clearable by default."""
    cfg = _cfg(tmp_path)
    q = Queue(str(tmp_path / "queue.jsonl"))
    write_pause_flag(cfg["queue"]["path"], "main", "some_future_category",
                     "nobody wrote a re-checker for this yet")

    assert pr.revalidate_pause(cfg, "main", q) is False
    assert is_paused(cfg["queue"]["path"], "main")


def test_revalidator_exception_leaves_paused(tmp_path):
    """A misbehaving revalidator must not crash the loop tick or
    accidentally clear the flag -- fail safe, stay paused."""
    def _boom(cfg, machine):
        raise RuntimeError("revalidator bug")

    pr.register_auto_revalidator("buggy_category", _boom)
    cfg = _cfg(tmp_path)
    q = Queue(str(tmp_path / "queue.jsonl"))
    write_pause_flag(cfg["queue"]["path"], "main", "buggy_category", "evidence")

    assert pr.revalidate_pause(cfg, "main", q) is False
    assert is_paused(cfg["queue"]["path"], "main")


# --------------------------------------- urgent_fix_unsynced-specific re-checker


def test_urgent_fix_unsynced_transient_fetch_failure_autoclears(tmp_path):
    """The actual #629 shape: local checkout already has the fix (the
    ancestor test passes with zero network calls), so revalidation clears
    the pause without needing git ls-remote/fetch to succeed at all --
    this is the fix for 'fetch failed' meaning 'could not verify', not
    'fix absent'."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    fix_sha = _commit(repo, "the fix")

    cfg = _cfg(tmp_path, repo_cwd=str(repo), fixes=[{"sha": fix_sha}])
    q = Queue(str(tmp_path / "queue.jsonl"))
    # Simulate the exact incident: paused with "git fetch failed" evidence,
    # even though (as in the real incident) the fix was already present
    # locally the whole time.
    write_pause_flag(cfg["queue"]["path"], "main", "urgent_fix_unsynced",
                     f"git fetch failed at {repo}")

    assert pr.revalidate_pause(cfg, "main", q) is True
    assert not is_paused(cfg["queue"]["path"], "main")
    audit = (tmp_path / "pause_audit.log").read_text()
    assert "auto_revalidate:urgent_fix_unsynced" in audit
    assert "no network needed" in audit


def test_urgent_fix_unsynced_persistent_genuinely_missing_stays_paused(tmp_path):
    """A clone that genuinely lacks the fix, with no working remote at all
    (repo_cwd is not even a clone of anything) -- re-check must correctly
    report unresolved and leave the pause in place."""
    repo = tmp_path / "repo"
    _init_repo(repo)

    cfg = _cfg(tmp_path, repo_cwd=str(repo), fixes=[{"sha": "deadbeef" * 5}])
    q = Queue(str(tmp_path / "queue.jsonl"))
    write_pause_flag(cfg["queue"]["path"], "main", "urgent_fix_unsynced",
                     "git fetch failed at " + str(repo))

    assert pr.revalidate_pause(cfg, "main", q) is False
    assert is_paused(cfg["queue"]["path"], "main")


def test_urgent_fix_unsynced_dirty_tree_stays_paused_on_recheck(tmp_path):
    origin = tmp_path / "origin"
    _init_repo(origin)
    fix_sha = _commit(origin, "the fix")

    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(origin), str(clone)], check=True)
    parent = _git(clone, "rev-parse", "HEAD~1")
    _git(clone, "reset", "-q", "--hard", parent)
    (clone / "f.txt").write_text("dirty edit\n")

    cfg = _cfg(tmp_path, repo_cwd=str(clone), fixes=[{"sha": fix_sha}])
    q = Queue(str(tmp_path / "queue.jsonl"))
    write_pause_flag(cfg["queue"]["path"], "main", "urgent_fix_unsynced",
                     f"dirty or unreadable working tree at {clone}")

    assert pr.revalidate_pause(cfg, "main", q) is False
    assert is_paused(cfg["queue"]["path"], "main")
    assert (clone / "f.txt").read_text() == "dirty edit\n"


def test_urgent_fix_unsynced_recheck_falls_through_to_fetch_and_pull(tmp_path):
    """The fix landed upstream (origin) after the pause was set, and the
    local checkout's ancestor test alone can't see it yet -- ls-remote and
    fetch both work now, so the existing safe fetch/ff-pull path should
    close the gap on re-check, mirroring what enforce_urgent_fixes() does
    on a fresh tick."""
    origin = tmp_path / "origin"
    _init_repo(origin)

    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(origin), str(clone)], check=True)

    # The fix lands on origin only AFTER the clone already exists/paused.
    fix_sha = _commit(origin, "the fix, landed after the pause was set")

    cfg = _cfg(tmp_path, repo_cwd=str(clone), fixes=[{"sha": fix_sha}])
    q = Queue(str(tmp_path / "queue.jsonl"))
    write_pause_flag(cfg["queue"]["path"], "main", "urgent_fix_unsynced",
                     f"git fetch failed at {clone}")

    assert pr.revalidate_pause(cfg, "main", q) is True
    assert not is_paused(cfg["queue"]["path"], "main")
    assert ufmod.missing_fixes(str(clone), [{"sha": fix_sha}]) == []


def test_urgent_fix_unsynced_registered_with_pause_revalidation():
    """Confirms urgent_fixes.py actually registers itself at import time
    (the wiring #629 asked for), independent of any test that pauses/
    clears through the full revalidate_pause() path above."""
    assert "urgent_fix_unsynced" in pr._AUTO_REVALIDATORS
    assert pr._AUTO_REVALIDATORS["urgent_fix_unsynced"] is (
        ufmod._revalidate_urgent_fix_unsynced)
