"""Tests for ablator's own-codebase drift detection (self_check.py) --
distinct from urgent_fixes.py, which checks the TARGET repo (splatograph)."""
import subprocess
from unittest import mock

import pytest

from ablator import self_check


def _fake_run(results):
    """results: dict mapping a tuple of args (after 'git') -> stdout string
    (or None for a nonzero-exit / failure)."""
    def _run(cmd, cwd=None, capture_output=None, text=None, timeout=None, check=None):
        key = tuple(cmd[1:])
        out = results.get(key)
        if out is None:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom")
        return subprocess.CompletedProcess(cmd, 0, stdout=out + "\n", stderr="")
    return _run


def test_check_self_drift_behind_count(tmp_path, monkeypatch):
    results = {
        ("rev-parse", "HEAD"): "local123abc",
        ("fetch", "--quiet", "origin", "main"): "",
        ("rev-parse", "origin/main"): "remote456def",
        ("rev-list", "--count", "local123abc..remote456def"): "9",
        ("rev-list", "--count", "remote456def..local123abc"): "0",
    }
    monkeypatch.setattr(subprocess, "run", _fake_run(results))
    drift = self_check.check_self_drift(repo_root=str(tmp_path))
    assert drift["behind"] == 9
    assert drift["ahead"] == 0
    assert drift["local_commit"] == "local123abc"
    assert drift["remote_commit"] == "remote456def"
    assert drift["error"] is None


def test_check_self_drift_up_to_date(tmp_path, monkeypatch):
    results = {
        ("rev-parse", "HEAD"): "samecommit",
        ("fetch", "--quiet", "origin", "main"): "",
        ("rev-parse", "origin/main"): "samecommit",
        ("rev-list", "--count", "samecommit..samecommit"): "0",
        ("rev-list", "--count", "samecommit..samecommit"): "0",
    }
    monkeypatch.setattr(subprocess, "run", _fake_run(results))
    drift = self_check.check_self_drift(repo_root=str(tmp_path))
    assert drift["behind"] == 0
    assert self_check.format_drift_banner("main", drift) is None


def test_check_self_drift_offline_never_crashes(tmp_path, monkeypatch):
    def _raise(*a, **k):
        raise subprocess.TimeoutExpired(cmd="git", timeout=15)
    monkeypatch.setattr(subprocess, "run", _raise)
    drift = self_check.check_self_drift(repo_root=str(tmp_path))
    assert drift["behind"] is None
    assert drift["error"] is not None
    # must not raise
    assert self_check.format_drift_banner("main", drift) is None


def test_check_self_drift_missing_repo_root():
    drift = self_check.check_self_drift(repo_root=None)
    # repo_root=None with no importable ablator.__file__ fallback still
    # must not crash -- ablator_source_root() is exercised for real here,
    # so just assert the contract shape.
    assert "behind" in drift
    assert "error" in drift or drift["behind"] is not None


def test_format_drift_banner_loud_and_unmissable():
    drift = {"behind": 9, "ahead": 0, "local_commit": "69caa7bxxxx",
             "remote_commit": "fb33d3axxxx", "repo_root": "/home/bjoern/git/ablator"}
    banner = self_check.format_drift_banner("r9700", drift)
    assert banner is not None
    assert "SELF-DRIFT" in banner
    assert "r9700" in banner
    assert "9 commit" in banner
    assert "69caa7b" in banner
    assert "fb33d3a" in banner
    assert "ACTION REQUIRED" in banner
    # loud framing convention
    assert banner.splitlines()[0].startswith("!" * 10)


def test_run_self_check_does_not_auto_pull(tmp_path, monkeypatch):
    """run_self_check must never invoke `git pull` / `git merge` on
    ablator's own checkout -- only fetch + read-only comparisons."""
    calls = []
    results = {
        ("rev-parse", "HEAD"): "local123abc",
        ("fetch", "--quiet", "origin", "main"): "",
        ("rev-parse", "origin/main"): "remote456def",
        ("rev-list", "--count", "local123abc..remote456def"): "9",
        ("rev-list", "--count", "remote456def..local123abc"): "0",
    }
    fake = _fake_run(results)

    def _spy(cmd, **kwargs):
        calls.append(tuple(cmd))
        return fake(cmd, **kwargs)

    monkeypatch.setattr(subprocess, "run", _spy)
    monkeypatch.setattr(self_check, "ablator_source_root", lambda: str(tmp_path))

    cfg = {"queue": {"path": str(tmp_path / "queue.jsonl")}}
    drift = self_check.run_self_check(cfg, "r9700")

    assert drift["behind"] == 9
    for c in calls:
        assert "pull" not in c
        assert "merge" not in c
        assert "reset" not in c


def test_write_self_version_file_cross_machine_visible(tmp_path):
    qpath = tmp_path / "queue.jsonl"
    cfg = {"queue": {"path": str(qpath)}}
    drift = {"local_commit": "abc123", "remote_commit": "def456",
             "behind": 9, "ahead": 0, "error": None}
    self_check.write_self_version_file(cfg, "r9700", drift)
    out = (tmp_path / "ablator_version_r9700.txt").read_text()
    assert "behind=9" in out
    assert "local=abc123" in out
    assert "remote=def456" in out


def test_write_self_version_file_never_raises(tmp_path):
    cfg = {"queue": {"path": "/nonexistent/dir/queue.jsonl"}}
    drift = {"local_commit": "x", "remote_commit": "y", "behind": 1,
             "ahead": 0, "error": None}
    self_check.write_self_version_file(cfg, "r9700", drift)  # must not raise
