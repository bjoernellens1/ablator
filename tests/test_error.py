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


def test_oom_killed_confirmed_by_dmesg():
    """Exit 137 IS classified oom_killed when dmesg actually confirms the
    kernel OOM-killer fired."""
    r = errormod.classify_failure(
        _job(), "process killed", 137,
        {"dmesg_tail": "Out of memory: Killed process 12345 (train.py)"})
    assert r["category"] == "oom_killed"
    assert r["suggested_action"] == "requeue_once_needs_review"


def test_oom_killed_confirmed_by_container_inspect():
    """Exit 137 IS classified oom_killed when the container's own
    `docker/podman inspect --format='{{.State.OOMKilled}}'` says true --
    the authoritative signal (see runner._container_oom_killed)."""
    r = errormod.classify_failure(
        _job(), "process killed", 137, {"container_oom_killed": True})
    assert r["category"] == "oom_killed"


def test_killed_externally_when_no_oom_evidence():
    """Bug fix (incident 2026-09-12): exit code 137 ALONE (no dmesg
    OOM-killer signature, no container inspect confirmation -- e.g. the
    container was already removed by the time this ran) is genuinely
    ambiguous -- a SIGKILL covers both the OOM-killer and a plain operator
    `docker kill`/`kill -9`. Must NOT be guessed as oom_killed."""
    r = errormod.classify_failure(_job(), "process killed", 137, {})
    assert r["category"] == "killed_externally"
    assert r["category"] != "oom_killed"


def test_killed_externally_when_container_inspect_says_not_oom():
    """An operator's `docker kill` on a container is exactly this case:
    docker/podman inspect's State.OOMKilled positively reports false, so
    exit 137 must be classified killed_externally, not oom_killed."""
    r = errormod.classify_failure(
        _job(), "process killed", 137, {"container_oom_killed": False})
    assert r["category"] == "killed_externally"


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


def test_unknown_evidence_shows_tail_not_head_of_giant_header():
    """`log_tail` here is already the LAST CRASH_TAIL_BYTES of a job's log.
    ablator's own log preamble is one giant, no-newline JSON line (the
    ABLATOR_JOB_JSON/ABLATOR_SUBMISSION_JSON env var dump); for a job whose
    real output after that preamble is short, that byte-window can still
    fall mostly inside the header. `_snippet(text, "")` used to return
    `text[:width]` -- the FIRST chars of that window, i.e. still the header
    -- instead of the run's own final, most-recent, most-diagnostic output
    at the very end. Confirmed live 2026-08-29 (r9700 fr3batchfull_b2 /
    scannetppfull_full): each printed a specific, actionable one-line
    failure as its last output, but the recorded evidence was an unrelated
    substring of that job's own spec `notes` field."""
    header_json = '{"notes": "' + ("x" * 4000) + '"}\n'
    real_error = "--streaming_replay is retired for production runs."
    log_tail = (header_json + real_error)[-4096:]
    r = errormod.classify_failure(_job(), log_tail, 0, {})
    assert r["category"] == "unknown"
    assert "streaming_replay is retired" in r["evidence_snippet"]
    assert "notes" not in r["evidence_snippet"]


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


def test_last_traceback_block_extraction():
    """Test that last_traceback_block extracts traceback to end of text."""
    log = "some output\nTraceback (most recent call last):\n  File \"train.py\", line 100\n    foo()\nValueError: bad value"
    tb = errormod.last_traceback_block(log)
    assert "Traceback (most recent call last)" in tb
    assert "ValueError: bad value" in tb
    assert "some output" not in tb  # Should not include text before traceback


def test_last_traceback_block_multiple():
    """Test that last_traceback_block returns the LAST traceback, not the first."""
    log = (
        "Traceback (most recent call last):\n  File \"a.py\"\nError: first\n\n"
        "... cleanup ...\n\n"
        "Traceback (most recent call last):\n  File \"b.py\"\nError: second"
    )
    tb = errormod.last_traceback_block(log)
    assert "File \"b.py\"" in tb
    assert "Error: second" in tb
    # First traceback should not be in the result (it's before the last one)
    assert tb.count("Traceback") == 1


def test_synthetic_large_stderr_with_traceback():
    """Test ~35 KB synthetic stderr with traceback near the end.

    Simulates a job with a large preamble (like ABLATOR_JOB_JSON) followed by
    the actual error. Evidence should contain the exception line and innermost
    frames, not get cut off at the marker's fixed 160-char window.
    """
    # Build a ~35 KB log: preamble + some noise + traceback at the end
    preamble = "x" * 20000  # 20 KB of noise
    middle = "y" * 15000   # 15 KB more noise
    traceback_text = (
        "\nTraceback (most recent call last):\n"
        "  File \"/splatograph/train.py\", line 2818, in <module>\n"
        "    training(config)\n"
        "  File \"/splatograph/train.py\", line 2500, in training\n"
        "    result = model.train_step(batch)\n"
        "  File \"/splatograph/model.py\", line 1200, in train_step\n"
        "    loss = compute_loss(output)\n"
        "  File \"/splatograph/loss.py\", line 450, in compute_loss\n"
        "    raise RuntimeError(\"tensor shape mismatch\")\n"
        "RuntimeError: tensor shape mismatch\n"
    )
    full_log = preamble + middle + traceback_text
    assert len(full_log) > 30000  # Verify size exceeds evidence tail size

    # Classify as code_error (has traceback)
    r = errormod.classify_failure(_job(), full_log, 1, {})
    assert r["category"] == "code_error"

    # Evidence must contain the exception line and inner frames, not just marker
    evidence = r["evidence_snippet"]
    assert "RuntimeError: tensor shape mismatch" in evidence
    assert "compute_loss" in evidence  # Inner frame
    assert len(evidence) > 160  # Must be more than just the 160-char snippet window


def test_gpu_oom_with_traceback_in_evidence():
    """Test that gpu_busy_conflict classifies OOM with traceback in evidence."""
    log = (
        "RuntimeError: HIP out of memory: tried to allocate 16.00 GiB\n"
        "... device full ...\n\n"
        "Traceback (most recent call last):\n"
        "  File \"/train.py\", line 100\n"
        "    result = model(batch)\n"
        "  File \"/model.py\", line 50\n"
        "    raise RuntimeError('OOM')\n"
        "RuntimeError: Out of memory\n"
    )
    r = errormod.classify_failure(_job(gpu_busy_at_claim=True), log, 1, {})
    assert r["category"] == "gpu_busy_conflict"

    # Evidence should include both the OOM marker and the traceback
    evidence = r["evidence_snippet"]
    assert "HIP out of memory" in evidence
    assert "RuntimeError: Out of memory" in evidence
