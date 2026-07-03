"""Tests for features ported from splatograph scripts/{queue_runner,run_error}.py:
pause flags, not_before backoff, flock deadline, and error classification."""
from __future__ import annotations

import time

import pytest

from ablator import error as errormod
from ablator.queue import (Queue, clear_pause_flag, is_paused, job_lane,
                            not_before_ok, pause_flag_path, write_pause_flag)


@pytest.fixture
def cfg(tmp_path):
    return {"_path": "x", "queue": {"path": str(tmp_path / "queue.jsonl")},
            "machines": {"m": {}}, "types": {}, "resources": {}}


# ------------------------------------------------------------- pause flags

def test_pause_flag_roundtrip(cfg):
    qpath = cfg["queue"]["path"]
    assert not is_paused(qpath, "m")
    path = write_pause_flag(qpath, "m", "disk_full", "free space below 2GB")
    assert is_paused(qpath, "m")
    assert path == pause_flag_path(qpath, "m")
    with open(path) as f:
        text = f.read()
    assert "category=disk_full" in text
    assert clear_pause_flag(qpath, "m")
    assert not is_paused(qpath, "m")


def test_claim_next_skips_paused_machine(cfg):
    q = Queue(cfg["queue"]["path"])
    q.append([{"id": "a", "status": "pending", "machine": "m"}])
    write_pause_flag(cfg["queue"]["path"], "m", "disk_full", "evidence")
    assert q.claim_next("m") is None
    assert q.read()[0]["status"] == "pending"  # untouched


# ------------------------------------------------------------ not_before

def test_not_before_ok():
    assert not_before_ok({}) is True
    assert not_before_ok({"not_before": time.time() - 10}) is True
    assert not_before_ok({"not_before": time.time() + 300}) is False


def test_claim_next_respects_not_before(cfg):
    q = Queue(cfg["queue"]["path"])
    q.append([{"id": "a", "status": "pending", "machine": "any",
               "not_before": time.time() + 300}])
    assert q.claim_next("m") is None
    q.update("a", not_before=time.time() - 1)
    claimed = q.claim_next("m")
    assert claimed is not None and claimed["id"] == "a"


# --------------------------------------------------------- flock deadline

def test_open_locked_times_out_when_contended(cfg, tmp_path):
    q = Queue(cfg["queue"]["path"])
    q.append([])  # create the file
    holder = open(q.path, "r+")
    import fcntl
    fcntl.flock(holder, fcntl.LOCK_EX)
    try:
        with pytest.raises(TimeoutError):
            q._open_locked(timeout_s=0.3)
    finally:
        fcntl.flock(holder, fcntl.LOCK_UN)
        holder.close()


def test_claim_next_returns_none_on_lock_timeout(cfg, monkeypatch):
    q = Queue(cfg["queue"]["path"])
    q.append([{"id": "a", "status": "pending", "machine": "any"}])
    monkeypatch.setattr(q, "_open_locked",
                        lambda *a, **k: (_ for _ in ()).throw(TimeoutError("busy")))
    assert q.claim_next("m") is None


# ---------------------------------------------------------- error classify

def test_classify_disk_full():
    r = errormod.classify_failure({}, "no space left on device", 1)
    assert r["category"] == "disk_full"
    assert r["suggested_action"] == "pause_queue_alert"


def test_classify_image_missing():
    r = errormod.classify_failure({}, "Error: pull access denied for ghcr.io/x", 1)
    assert r["category"] == "image_missing"
    assert r["suggested_action"] == "skip_permanently_this_machine"


def test_classify_gpu_busy_conflict_requires_claim_flag():
    job = {"gpu_busy_at_claim": True}
    r = errormod.classify_failure(job, "HIP out of memory", 1)
    assert r["category"] == "gpu_busy_conflict"
    assert r["confidence"] == 0.9


def test_classify_oom_killed_exit_code_137():
    r = errormod.classify_failure({}, "", 137)
    assert r["category"] == "oom_killed"
    assert r["suggested_action"] == "requeue_once_needs_review"


def test_classify_scene_missing():
    job = {"scene": "/data/kitchen1"}
    r = errormod.classify_failure(
        job, "FileNotFoundError: [Errno 2] No such file or directory: '/data/kitchen1'", 1)
    assert r["category"] == "scene_missing"


def test_classify_network_transient():
    r = errormod.classify_failure({}, "Temporary failure in name resolution", 1)
    assert r["category"] == "network_transient"
    assert r["suggested_action"] == "requeue_backoff_2min"


def test_classify_code_error():
    r = errormod.classify_failure(
        {}, "Traceback (most recent call last):\nValueError: bad", 1)
    assert r["category"] == "code_error"


def test_classify_unknown_default():
    r = errormod.classify_failure({}, "some unrelated log line", 1)
    assert r["category"] == "unknown"
    assert r["suggested_action"] == "retry_once_then_quarantine"


def test_classify_disk_full_low_free_space_without_marker():
    r = errormod.classify_failure({}, "", 1,
                                  machine_context={"disk_free_bytes": 100})
    assert r["category"] == "disk_full"


def test_patterns_from_config_overrides_category():
    cfg = {"error_patterns": {"image_missing": ["custom marker text"]}}
    patterns = errormod.patterns_from_config(cfg)
    assert patterns["image_missing"] == ("custom marker text",)
    # untouched category keeps its default
    assert patterns["network_transient"] == errormod.DEFAULT_PATTERNS["network_transient"]
    r = errormod.classify_failure({}, "hit custom marker text here", 1,
                                  patterns=patterns)
    assert r["category"] == "image_missing"


# --------------------------------------------------------------- job_lane

def test_job_lane_default_and_bounds():
    assert job_lane({}) == 2
    assert job_lane({"lane": 3}) == 3
    assert job_lane({"lane": "bogus"}) == 2
    assert job_lane({"lane": 99}) == 2


# ------------------------------------------------------------- heartbeat

def test_write_heartbeat_creates_file(cfg):
    from ablator import runner
    runner.write_heartbeat(cfg, "m", "idle")
    import os
    path = os.path.join(os.path.dirname(cfg["queue"]["path"]), "heartbeat_m.txt")
    assert os.path.exists(path)
    with open(path) as f:
        text = f.read()
    assert "state=idle" in text


# --------------------------------------------------------- handle_failure

def test_handle_failure_disk_full_pauses_machine(cfg, tmp_path):
    from ablator import runner
    q = Queue(cfg["queue"]["path"])
    log_dir = tmp_path
    cfg["queue"]["log_dir"] = str(log_dir)
    job = {"id": "j1", "model_path": str(tmp_path / "out"), "scene": ""}
    q.append([{**job, "status": "running", "machine": "m"}])
    with open(log_dir / "j1.log", "w") as f:
        f.write("no space left on device\n")
    disposition = runner.handle_failure(cfg, job, 1, "m", str(tmp_path), q)
    assert disposition == "paused_disk_full"
    assert is_paused(cfg["queue"]["path"], "m")
    assert q.read()[0]["status"] == "paused_disk_full"


def test_handle_failure_gpu_busy_requeues_with_backoff(cfg, tmp_path):
    from ablator import runner
    q = Queue(cfg["queue"]["path"])
    cfg["queue"]["log_dir"] = str(tmp_path)
    job = {"id": "j2", "model_path": str(tmp_path / "out"), "scene": "",
          "gpu_busy_at_claim": True}
    q.append([{**job, "status": "running", "machine": "m"}])
    with open(tmp_path / "j2.log", "w") as f:
        f.write("HIP out of memory\n")
    disposition = runner.handle_failure(cfg, job, 1, "m", str(tmp_path), q)
    assert disposition == "pending"
    rec = q.read()[0]
    assert rec["status"] == "pending"
    assert rec["not_before"] > time.time()
