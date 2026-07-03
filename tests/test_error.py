"""Tests for ablator.error classification + pause-flag/not_before wiring in
ablator.queue and status/errors CLI output."""
import json
import time

import pytest

from ablator import cli, error as errormod
from ablator.queue import Queue, is_paused, write_pause_flag, clear_pause_flag, not_before_ok


def _job(**kw):
    j = {"id": "j1", "scene": "/mnt/data/kitchen1", "model_path": "output/scratch/j1"}
    j.update(kw)
    return j


def make_cfg(tmp_path, **extra):
    cfg = {
        "_path": str(tmp_path / "config.json"),
        "queue": {"path": str(tmp_path / "queue.jsonl")},
        "machines": {"main": {"hostname_patterns": ["*"]}},
        "resources": {},
        "types": {"replay": {"cwd": "/repo", "command": ["true"]}},
    }
    cfg.update(extra)
    return cfg


def write_queue(path, jobs):
    with open(path, "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")


# --- one test per category ----------------------------------------------

def test_disk_full():
    r = errormod.classify_failure(_job(), "No space left on device", 1, {})
    assert r["category"] == "disk_full"
    assert r["suggested_action"] == "pause_queue_alert"


def test_image_missing():
    r = errormod.classify_failure(_job(), "pull access denied for ghcr.io/x", 125, {})
    assert r["category"] == "image_missing"
    assert r["suggested_action"] == "skip_permanently_this_machine"


def test_gpu_busy_conflict():
    r = errormod.classify_failure(_job(gpu_busy_at_claim=True),
                                  "RuntimeError: HIP out of memory", 1, {})
    assert r["category"] == "gpu_busy_conflict"
    assert r["suggested_action"] == "requeue_backoff_5min"


def test_oom_killed():
    r = errormod.classify_failure(_job(), "process killed", 137, {})
    assert r["category"] == "oom_killed"
    assert r["suggested_action"] == "requeue_once_needs_review"


def test_scene_missing():
    job = _job(scene="/mnt/data/kitchen1")
    log = "FileNotFoundError: No such file or directory: '/mnt/data/kitchen1/rgb.txt'"
    r = errormod.classify_failure(job, log, 1, {})
    assert r["category"] == "scene_missing"
    assert r["suggested_action"] == "quarantine_no_retry"


def test_network_transient():
    r = errormod.classify_failure(_job(), "Temporary failure in name resolution", 1, {})
    assert r["category"] == "network_transient"
    assert r["suggested_action"] == "requeue_backoff_2min"


def test_code_error():
    log = "Traceback (most recent call last):\nValueError: bad shape"
    r = errormod.classify_failure(_job(), log, 1, {})
    assert r["category"] == "code_error"
    assert r["suggested_action"] == "quarantine_code_fix_needed"


def test_unknown():
    r = errormod.classify_failure(_job(), "training ended for unclear reasons", 1, {})
    assert r["category"] == "unknown"
    assert r["suggested_action"] == "retry_once_then_quarantine"


# --- config-driven patterns ---------------------------------------------

def test_patterns_from_config_overrides_category():
    cfg = {"error_patterns": {"image_missing": ["totally custom marker"]}}
    patterns = errormod.patterns_from_config(cfg)
    r = errormod.classify_failure(_job(), "totally custom marker in log", 1, {},
                                  patterns=patterns)
    assert r["category"] == "image_missing"
    # default marker should NOT match anymore since category was replaced
    r2 = errormod.classify_failure(_job(), "pull access denied", 1, {}, patterns=patterns)
    assert r2["category"] != "image_missing"


def test_patterns_from_config_keeps_defaults_for_unmentioned_categories():
    cfg = {"error_patterns": {"image_missing": ["custom"]}}
    patterns = errormod.patterns_from_config(cfg)
    r = errormod.classify_failure(_job(), "No space left on device", 1, {}, patterns=patterns)
    assert r["category"] == "disk_full"


# --- not_before backoff --------------------------------------------------

def test_not_before_future_not_claimable():
    assert not_before_ok({"not_before": time.time() + 300}) is False


def test_not_before_past_claimable():
    assert not_before_ok({"not_before": time.time() - 300}) is True


def test_not_before_absent_claimable():
    assert not_before_ok({}) is True


def test_claim_next_skips_future_not_before(tmp_path):
    q = Queue(str(tmp_path / "queue.jsonl"))
    write_queue(q.path, [
        {"id": "future", "status": "pending", "machine": "any", "scene": "x",
         "model_path": "output/scratch/future", "not_before": time.time() + 300},
        {"id": "ready", "status": "pending", "machine": "any", "scene": "x",
         "model_path": "output/scratch/ready"},
    ])
    claimed = q.claim_next("main")
    assert claimed["id"] == "ready"


# --- pause-flag lifecycle -------------------------------------------------

def test_pause_flag_create_and_clear(tmp_path):
    qpath = str(tmp_path / "queue.jsonl")
    assert not is_paused(qpath, "main")
    path = write_pause_flag(qpath, "main", "disk_full", "evidence text")
    assert is_paused(qpath, "main")
    content = open(path).read()
    assert "category=disk_full" in content
    assert "evidence text" in content
    assert clear_pause_flag(qpath, "main") is True
    assert not is_paused(qpath, "main")


def test_clear_pause_flag_missing_returns_false(tmp_path):
    qpath = str(tmp_path / "queue.jsonl")
    assert clear_pause_flag(qpath, "main") is False


def test_claim_next_refuses_when_paused(tmp_path):
    q = Queue(str(tmp_path / "queue.jsonl"))
    write_queue(q.path, [
        {"id": "a", "status": "pending", "machine": "any", "scene": "x",
         "model_path": "output/scratch/a"},
    ])
    write_pause_flag(q.path, "main", "disk_full", "evidence")
    assert q.claim_next("main") is None
    clear_pause_flag(q.path, "main")


# --- status/errors CLI output --------------------------------------------

def test_cmd_status_shows_pause_flags(tmp_path, capsys):
    cfg = make_cfg(tmp_path)
    write_queue(cfg["queue"]["path"], [
        {"id": "a", "status": "pending", "machine": "any"},
    ])
    write_pause_flag(cfg["queue"]["path"], "main", "disk_full", "free space below 2GB")
    cli.cmd_status(cfg, None)
    out = capsys.readouterr().out
    assert "main is PAUSED (disk_full)" in out
    clear_pause_flag(cfg["queue"]["path"], "main")


def test_status_lines_tag_error_category(tmp_path):
    cfg = make_cfg(tmp_path)
    jobs = [{"id": "j1", "status": "quarantined", "machine": "any",
             "error_category": "scene_missing"}]
    lines = cli._status_lines(cfg, jobs)
    assert any("[scene_missing!]" in l for l in lines)


def test_cmd_errors_lists_classified_jobs(tmp_path, capsys):
    cfg = make_cfg(tmp_path)
    write_queue(cfg["queue"]["path"], [
        {"id": "bad1", "status": "quarantined", "machine": "any",
         "error_category": "code_error", "error_evidence": "Traceback...",
         "suggested_action": "quarantine_code_fix_needed",
         "finished_at": "2026-07-01T10:00:00"},
        {"id": "ok1", "status": "done", "machine": "any"},
    ])
    cli.cmd_errors(cfg, None)
    out = capsys.readouterr().out
    assert "bad1" in out
    assert "code_error" in out
    assert "ok1" not in out


def test_cmd_unpause_clears_flag(tmp_path, capsys):
    cfg = make_cfg(tmp_path)
    write_pause_flag(cfg["queue"]["path"], "main", "disk_full", "evidence")
    cli.cmd_unpause(cfg, "main")
    assert not is_paused(cfg["queue"]["path"], "main")
    assert "cleared" in capsys.readouterr().out


def test_cmd_unpause_missing_flag_errors(tmp_path):
    cfg = make_cfg(tmp_path)
    with pytest.raises(SystemExit):
        cli.cmd_unpause(cfg, "nonexistent-machine")
