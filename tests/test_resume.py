"""Tests for preemption-aware resume: find_latest_checkpoint(), the
resume-from-checkpoint disposition in handle_failure(), --start_checkpoint
threading in _job_vars()/render_command(), and terminationGracePeriodSeconds
in build_k8s_job_manifest().

Context: a100cluster jobs run at KAI Scheduler priority kai-batch-low (lowest
queue, subject to preemption at any time). Before this, a preempted/crashed
job with a real, recent checkpoint was requeued and re-run from scratch
(model weights + all streaming/ingestion progress lost) rather than resumed.
"""
from __future__ import annotations

import os
import time

import pytest

from ablator import runner
from ablator.queue import Queue


@pytest.fixture
def cfg(tmp_path):
    return {"_path": "x", "queue": {"path": str(tmp_path / "queue.jsonl")},
            "machines": {"m": {}}, "types": {}, "resources": {}}


# --------------------------------------------------- find_latest_checkpoint

def test_find_latest_checkpoint_none_when_missing(tmp_path):
    assert runner.find_latest_checkpoint(str(tmp_path / "out"), str(tmp_path)) is None


def test_find_latest_checkpoint_picks_highest_iteration(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    for it in (1000, 5000, 500):
        (out / f"chkpnt{it}.pth").write_text("x")
    path, it = runner.find_latest_checkpoint(str(out), str(tmp_path))
    assert it == 5000
    assert path == str(out / "chkpnt5000.pth")


def test_find_latest_checkpoint_ignores_non_checkpoint_files(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "chkpnt100.pth").write_text("x")
    (out / "report.json").write_text("{}")
    (out / "chkpnt100.pth.tmp").write_text("partial")  # in-flight atomic write
    path, it = runner.find_latest_checkpoint(str(out), str(tmp_path))
    assert it == 100
    assert path.endswith("chkpnt100.pth")


# --------------------------------------------- handle_failure: resume path

def test_handle_failure_resumes_when_checkpoint_exists_and_advanced(cfg, tmp_path):
    q = Queue(cfg["queue"]["path"])
    cfg["queue"]["log_dir"] = str(tmp_path)
    model_path = tmp_path / "out"
    model_path.mkdir()
    (model_path / "chkpnt2000.pth").write_text("x")
    job = {"id": "j1", "model_path": str(model_path), "scene": ""}
    q.append([{**job, "status": "running", "machine": "m"}])
    with open(tmp_path / "j1.log", "w") as f:
        f.write("Traceback (most recent call last):\nRuntimeError: pod evicted\n")

    disposition = runner.handle_failure(cfg, job, None, "m", str(tmp_path), q)

    assert disposition == "pending"
    rec = q.read()[0]
    assert rec["status"] == "pending"
    assert rec["last_resumed_iter"] == 2000
    assert rec["resume_checkpoint"] == str(model_path / "chkpnt2000.pth")
    assert rec["error_category"] == "resumable_from_checkpoint"
    # in-memory job dict mutated too, so the same dispatch cycle picks it up
    assert job["resume_checkpoint"] == str(model_path / "chkpnt2000.pth")


def test_handle_failure_does_not_resume_without_checkpoint(cfg, tmp_path):
    """No checkpoint at all -> falls through to ordinary classification
    (a traceback here classifies as code_error -> quarantine)."""
    q = Queue(cfg["queue"]["path"])
    cfg["queue"]["log_dir"] = str(tmp_path)
    job = {"id": "j2", "model_path": str(tmp_path / "out_missing"), "scene": ""}
    q.append([{**job, "status": "running", "machine": "m"}])
    with open(tmp_path / "j2.log", "w") as f:
        f.write("Traceback (most recent call last):\nValueError: bad config\n")

    disposition = runner.handle_failure(cfg, job, 1, "m", str(tmp_path), q)

    assert disposition == "quarantined"
    assert "resume_checkpoint" not in job


def test_handle_failure_stops_resuming_once_progress_stalls(cfg, tmp_path):
    """Guards against an infinite resume->immediate-crash->resume loop: if the
    checkpoint iteration hasn't advanced past the last resume point, this is
    treated as a real, reproducible failure (falls through to normal
    classification) rather than resumed again."""
    q = Queue(cfg["queue"]["path"])
    cfg["queue"]["log_dir"] = str(tmp_path)
    model_path = tmp_path / "out"
    model_path.mkdir()
    (model_path / "chkpnt2000.pth").write_text("x")
    job = {"id": "j3", "model_path": str(model_path), "scene": "",
          "last_resumed_iter": 2000}  # already resumed to this same point once
    q.append([{**job, "status": "running", "machine": "m"}])
    with open(tmp_path / "j3.log", "w") as f:
        f.write("Traceback (most recent call last):\nRuntimeError: same bug again\n")

    disposition = runner.handle_failure(cfg, job, 1, "m", str(tmp_path), q)

    # Falls through to ordinary code_error classification, not another resume.
    assert disposition == "quarantined"
    assert job.get("error_category") == "code_error"


def test_handle_failure_resume_beats_normal_classification(cfg, tmp_path):
    """Even a log tail that would otherwise classify as a hard, non-retryable
    failure (e.g. scene_missing) is resumed first if a real checkpoint with
    new progress exists -- the checkpoint-progress signal is checked before
    classify_and_record runs at all."""
    q = Queue(cfg["queue"]["path"])
    cfg["queue"]["log_dir"] = str(tmp_path)
    model_path = tmp_path / "out"
    model_path.mkdir()
    (model_path / "chkpnt3000.pth").write_text("x")
    job = {"id": "j4", "model_path": str(model_path), "scene": "/data/some_scene"}
    q.append([{**job, "status": "running", "machine": "m"}])
    with open(tmp_path / "j4.log", "w") as f:
        f.write("FileNotFoundError: No such file or directory: '/data/some_scene'\n")

    disposition = runner.handle_failure(cfg, job, 1, "m", str(tmp_path), q)

    assert disposition == "pending"
    assert q.read()[0]["last_resumed_iter"] == 3000


# ------------------------------------------- _job_vars: --start_checkpoint

def test_job_vars_threads_start_checkpoint_when_resume_pending():
    job = {"scene": "", "model_path": "/out/j1", "extra_args": "--cap_max 100000",
          "id": "j1", "resume_checkpoint": "/out/j1/chkpnt2000.pth"}
    vars = runner._job_vars(job, "m")
    assert "--start_checkpoint /out/j1/chkpnt2000.pth" in vars["extra_args"]
    assert "--cap_max 100000" in vars["extra_args"]


def test_job_vars_no_start_checkpoint_on_fresh_job():
    job = {"scene": "", "model_path": "/out/j1", "extra_args": "--cap_max 100000",
          "id": "j1"}
    vars = runner._job_vars(job, "m")
    assert "--start_checkpoint" not in vars["extra_args"]


def test_job_vars_does_not_double_inject_if_already_present():
    job = {"scene": "", "model_path": "/out/j1",
          "extra_args": "--start_checkpoint /out/j1/chkpnt1000.pth",
          "id": "j1", "resume_checkpoint": "/out/j1/chkpnt2000.pth"}
    vars = runner._job_vars(job, "m")
    assert vars["extra_args"].count("--start_checkpoint") == 1


# --------------------------------------- build_k8s_job_manifest: grace period

def test_k8s_manifest_default_termination_grace_period():
    mcfg = {"namespace": "jupyterhub", "kai_queue": "kai-batch-low",
           "priority_class": "kai-batch-low", "image_pull_secret": "regcred",
           "pvc_persistent": "p", "pvc_scratch": "s", "image": "img:1"}
    job = {"id": "j1", "scene": "/mnt/cps_persistent1_shared/scene"}
    manifest = runner.build_k8s_job_manifest(mcfg, job, ["python", "train.py"], None)
    pod_spec = manifest["spec"]["template"]["spec"]
    assert pod_spec["terminationGracePeriodSeconds"] == 150


def test_k8s_manifest_termination_grace_period_overridable():
    mcfg = {"namespace": "jupyterhub", "kai_queue": "kai-batch-low",
           "priority_class": "kai-batch-low", "image_pull_secret": "regcred",
           "pvc_persistent": "p", "pvc_scratch": "s", "image": "img:1",
           "termination_grace_period_s": 300}
    job = {"id": "j1", "scene": "/mnt/cps_persistent1_shared/scene"}
    manifest = runner.build_k8s_job_manifest(mcfg, job, ["python", "train.py"], None)
    pod_spec = manifest["spec"]["template"]["spec"]
    assert pod_spec["terminationGracePeriodSeconds"] == 300
