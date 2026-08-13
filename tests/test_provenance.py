"""Tests for ablator.provenance and its wiring into runner.py's dispatch
paths: git-state capture, r9700-vs-main drift warning, k8s image-commit-
label drift check, and that provenance data lands in the ledger entry."""
import json
import subprocess
from unittest import mock

import pytest

from ablator import provenance as provmod
from ablator import runner
from ablator.queue import Queue


# ------------------------------------------------------------- git capture

def _fake_run(commit="abc123", branch="main", dirty=False):
    def run(args, cwd=None, capture_output=None, text=None, timeout=None,
            check=None):
        cmd = args[1:]
        if cmd[:2] == ["rev-parse", "HEAD"]:
            out = commit
        elif cmd[:3] == ["rev-parse", "--abbrev-ref", "HEAD"]:
            out = branch
        elif cmd[:2] == ["status", "--porcelain"]:
            out = " M some_file.py\n" if dirty else ""
        else:
            out = ""
        return subprocess.CompletedProcess(args, 0, stdout=out, stderr="")
    return run


def test_capture_local_git_state_clean(tmp_path):
    with mock.patch("subprocess.run", side_effect=_fake_run(dirty=False)):
        state = provmod.capture_local_git_state(str(tmp_path), host="mainhost")
    assert state["commit"] == "abc123"
    assert state["branch"] == "main"
    assert state["dirty"] is False
    assert state["host"] == "mainhost"
    assert "error" not in state


def test_capture_local_git_state_dirty(tmp_path):
    with mock.patch("subprocess.run", side_effect=_fake_run(dirty=True)):
        state = provmod.capture_local_git_state(str(tmp_path))
    assert state["dirty"] is True


def test_capture_local_git_state_no_repo(tmp_path):
    def run(args, cwd=None, capture_output=None, text=None, timeout=None,
            check=None):
        return subprocess.CompletedProcess(args, 128, stdout="", stderr="not a repo")
    with mock.patch("subprocess.run", side_effect=run):
        state = provmod.capture_local_git_state(str(tmp_path))
    assert state["commit"] is None
    assert "error" in state


# --------------------------------------------------------------- r9700 drift

def test_check_remote_drift_matching():
    out = "commit1\nmain\n"
    def run(args, capture_output=None, text=None, timeout=None, check=None):
        return subprocess.CompletedProcess(args, 0, stdout=out, stderr="")
    with mock.patch("subprocess.run", side_effect=run):
        res = provmod.check_remote_drift("bjoern@r9700", "/repo", "commit1")
    assert res["drift"] is False
    assert res["warning"] is None


def test_check_remote_drift_mismatch():
    out = "commit2\nmain\n"
    def run(args, capture_output=None, text=None, timeout=None, check=None):
        return subprocess.CompletedProcess(args, 0, stdout=out, stderr="")
    with mock.patch("subprocess.run", side_effect=run):
        res = provmod.check_remote_drift("bjoern@r9700", "/repo", "commit1")
    assert res["drift"] is True
    assert res["warning"] is not None
    assert "DRIFT" in res["warning"]


def test_check_remote_drift_unreachable():
    def run(args, capture_output=None, text=None, timeout=None, check=None):
        raise OSError("no route to host")
    with mock.patch("subprocess.run", side_effect=run):
        res = provmod.check_remote_drift("bjoern@r9700", "/repo", "commit1")
    assert res["drift"] is None
    assert res["warning"] is not None


def _write_state_file(cfg, machine, commit):
    runner.write_git_state_file(cfg, machine, {"commit": commit, "branch": "main",
                                               "dirty": False, "host": machine})


def test_check_r9700_drift_warns_on_mismatch(tmp_path):
    qpath = tmp_path / "queue.jsonl"
    cfg = {"queue": {"path": str(qpath)}, "machines": {}, "types": {}}
    q = Queue(str(qpath))
    q.append([{"id": "j1", "type": "replay", "model_path": "output/j1"}])
    _write_state_file(cfg, "main", "commitA")
    job = {"id": "j1"}
    runner.check_r9700_drift(cfg, job, "r9700",
                             {"commit": "commitB", "branch": "feat/x"}, q)
    entry = next(j for j in q.read() if j["id"] == "j1")
    assert entry.get("drift_warning")
    assert entry.get("main_commit_at_check") == "commitA"


def test_check_r9700_drift_silent_on_match(tmp_path):
    qpath = tmp_path / "queue.jsonl"
    cfg = {"queue": {"path": str(qpath)}, "machines": {}, "types": {}}
    q = Queue(str(qpath))
    q.append([{"id": "j1", "type": "replay", "model_path": "output/j1"}])
    _write_state_file(cfg, "main", "commitA")
    job = {"id": "j1"}
    runner.check_r9700_drift(cfg, job, "r9700", {"commit": "commitA"}, q)
    entry = next(j for j in q.read() if j["id"] == "j1")
    assert "drift_warning" not in entry


def test_check_r9700_drift_noop_for_main(tmp_path):
    qpath = tmp_path / "queue.jsonl"
    cfg = {"queue": {"path": str(qpath)}, "machines": {}, "types": {}}
    q = Queue(str(qpath))
    q.append([{"id": "j1", "type": "replay", "model_path": "output/j1"}])
    job = {"id": "j1"}
    # No git_state_main.json written for "main" itself -> nothing to compare.
    runner.check_r9700_drift(cfg, job, "main", {"commit": "commitA"}, q)
    entry = next(j for j in q.read() if j["id"] == "j1")
    assert "drift_warning" not in entry


# ------------------------------------- splatograph#259: generalized drift

def _drift_cfg(tmp_path):
    qpath = tmp_path / "queue.jsonl"
    cfg = {"queue": {"path": str(qpath)}, "machines": {}, "types": {}}
    q = Queue(str(qpath))
    q.append([{"id": "j1", "type": "replay", "model_path": "output/j1"}])
    return cfg, q


def _warning(q):
    entry = next(j for j in q.read() if j["id"] == "j1")
    return entry.get("drift_warning"), entry


def test_drift_check_is_not_r9700_specific(tmp_path):
    """rtx3090 ran a job from a 25-commit-stale checkout with NO warning,
    because the check returned early for every machine but r9700."""
    cfg, q = _drift_cfg(tmp_path)
    _write_state_file(cfg, "main", "commitA")
    runner.check_checkout_drift(cfg, {"id": "j1"}, "rtx3090",
                                {"commit": "commitB", "branch": "main",
                                 "dirty": False}, q)
    warning, entry = _warning(q)
    assert warning and "rtx3090" in warning
    assert entry.get("main_commit_at_check") == "commitA"


def test_dirty_tree_warns_even_on_main(tmp_path):
    """splatograph#259 Instance 2: a rewound submodule pointer on main's
    own checkout showed up only as ` M ablator` and was never flagged."""
    cfg, q = _drift_cfg(tmp_path)
    runner.check_checkout_drift(cfg, {"id": "j1"}, "main",
                                {"commit": "commitA", "branch": "main",
                                 "dirty": True, "cwd": "/home/x/splatograph"}, q)
    warning, entry = _warning(q)
    assert warning and "UNCOMMITTED CHANGES" in warning
    # No cross-machine comparison happened, so no misleading reference commit.
    assert "main_commit_at_check" not in entry


def test_off_branch_warns_even_when_clean(tmp_path):
    """The headline #259 case: a CLEAN main checkout parked on an agent
    branch, invisible to both the dirty and cross-machine checks."""
    cfg, q = _drift_cfg(tmp_path)
    runner.check_checkout_drift(
        cfg, {"id": "j1"}, "main",
        {"commit": "commitA", "branch": "agent/report-holdout-hash-field",
         "dirty": False}, q)
    warning, _ = _warning(q)
    assert warning and "agent/report-holdout-hash-field" in warning


def test_clean_on_expected_branch_is_silent(tmp_path):
    cfg, q = _drift_cfg(tmp_path)
    runner.check_checkout_drift(cfg, {"id": "j1"}, "main",
                                {"commit": "commitA", "branch": "main",
                                 "dirty": False}, q)
    warning, _ = _warning(q)
    assert warning is None


def test_undeterminable_dirty_state_does_not_warn(tmp_path):
    """dirty=None means git was unreadable, reported via the state's own
    `error` field — not drift. Warning on it would be pure noise."""
    cfg, q = _drift_cfg(tmp_path)
    runner.check_checkout_drift(cfg, {"id": "j1"}, "main",
                                {"commit": None, "branch": None, "dirty": None,
                                 "error": "no git repo"}, q)
    warning, _ = _warning(q)
    assert warning is None


def test_expected_branch_derives_from_auto_sync_ref():
    assert runner.expected_branch({}) == "main"
    assert runner.expected_branch(
        {"urgent_fixes": {"auto_sync_ref": "origin/main"}}) == "main"
    assert runner.expected_branch(
        {"urgent_fixes": {"auto_sync_ref": "origin/release"}}) == "release"


def test_detached_head_is_not_reported_as_branch_drift(tmp_path):
    """A pinned/detached checkout (`git rev-parse --abbrev-ref HEAD` ->
    "HEAD") is the *goal* state of #259's worktree-isolation follow-up —
    it must not be flagged as drift."""
    cfg, q = _drift_cfg(tmp_path)
    runner.check_checkout_drift(cfg, {"id": "j1"}, "main",
                                {"commit": "commitA", "branch": "HEAD",
                                 "dirty": False}, q)
    warning, _ = _warning(q)
    assert warning is None


def test_multiple_conditions_are_combined_into_one_warning(tmp_path):
    cfg, q = _drift_cfg(tmp_path)
    _write_state_file(cfg, "main", "commitA")
    runner.check_checkout_drift(cfg, {"id": "j1"}, "r9700",
                                {"commit": "commitB", "branch": "feat/x",
                                 "dirty": True, "cwd": "/repo"}, q)
    warning, entry = _warning(q)
    assert warning.count("CODE PROVENANCE DRIFT") == 3
    assert entry.get("main_commit_at_check") == "commitA"


# ---------------------------------------------------------------- k8s image

def _skopeo_ok(revision):
    def run(args, capture_output=None, text=None, timeout=None, check=None):
        if args[0] == "skopeo":
            data = json.dumps({"Labels": {"org.opencontainers.image.revision": revision}})
            return subprocess.CompletedProcess(args, 0, stdout=data, stderr="")
        raise AssertionError("should not fall back past skopeo")
    return run


def test_capture_image_commit_label_via_skopeo():
    with mock.patch("subprocess.run", side_effect=_skopeo_ok("deadbeef")):
        info = provmod.capture_image_commit_label("ghcr.io/x/y:cuda-dev")
    assert info["baked_commit"] == "deadbeef"
    assert info["source"] == "skopeo"


def test_capture_image_commit_label_falls_back_to_podman():
    def run(args, capture_output=None, text=None, timeout=None, check=None):
        if args[0] == "skopeo":
            raise FileNotFoundError("no skopeo")
        if args[0] == "podman" and args[1] == "inspect":
            data = json.dumps([{"Config": {"Labels": {
                "org.opencontainers.image.revision": "cafef00d"}}}])
            return subprocess.CompletedProcess(args, 0, stdout=data, stderr="")
        return subprocess.CompletedProcess(args, 1, stdout="", stderr="no such image")
    with mock.patch("subprocess.run", side_effect=run):
        info = provmod.capture_image_commit_label("ghcr.io/x/y:cuda-dev", runtime="podman")
    assert info["baked_commit"] == "cafef00d"
    assert info["source"] == "podman"


def test_check_image_drift_mismatch():
    with mock.patch("subprocess.run", side_effect=_skopeo_ok("oldcommit")):
        res = provmod.check_image_drift("ghcr.io/x/y:cuda-dev", "newcommit")
    assert res["drift"] is True
    assert "DRIFT" in res["warning"]


def test_check_image_drift_match():
    with mock.patch("subprocess.run", side_effect=_skopeo_ok("samecommit")):
        res = provmod.check_image_drift("ghcr.io/x/y:cuda-dev", "samecommit")
    assert res["drift"] is False
    assert res["warning"] is None


def test_check_image_drift_no_label():
    with mock.patch("subprocess.run", side_effect=_skopeo_ok(None)):
        res = provmod.check_image_drift("ghcr.io/x/y:cuda-dev", "newcommit")
    assert res["drift"] is None
    assert "UNKNOWN" in res["warning"]


# ------------------------------------------------------- ledger integration

def test_capture_and_record_provenance_lands_in_ledger(tmp_path):
    qpath = tmp_path / "queue.jsonl"
    cfg = {"queue": {"path": str(qpath)}, "machines": {}, "types": {}}
    q = Queue(str(qpath))
    q.append([{"id": "j1", "type": "replay", "model_path": "output/j1"}])
    with mock.patch("subprocess.run", side_effect=_fake_run(commit="feedface")):
        state = runner.capture_and_record_provenance(cfg, {"id": "j1"}, "main",
                                                      str(tmp_path), q)
    assert state["commit"] == "feedface"
    entry = next(j for j in q.read() if j["id"] == "j1")
    assert entry["provenance"]["commit"] == "feedface"
    # Also persisted to the shared cross-machine status file.
    disk_state = runner.read_git_state_file(cfg, "main")
    assert disk_state["commit"] == "feedface"
