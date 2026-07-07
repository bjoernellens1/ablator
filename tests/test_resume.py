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


# -------------------------------- build_k8s_job_manifest: dataset PVC routing

def test_k8s_manifest_dataset_pvc_persistent_root():
    mcfg = {"namespace": "jupyterhub", "kai_queue": "kai-batch-low",
           "priority_class": "kai-batch-low", "image_pull_secret": "regcred",
           "pvc_persistent": "persist-pvc", "pvc_scratch": "scratch-pvc", "image": "img:1"}
    job = {"id": "j1", "scene": "/mnt/cps_persistent1_shared/peyman/TUM/fr3"}
    manifest = runner.build_k8s_job_manifest(mcfg, job, ["python", "train.py"], None)
    volumes = {v["name"]: v for v in manifest["spec"]["template"]["spec"]["volumes"]}
    assert volumes["dataset"]["persistentVolumeClaim"]["claimName"] == "persist-pvc"
    mounts = manifest["spec"]["template"]["spec"]["containers"][0]["volumeMounts"]
    dataset_mount = next(m for m in mounts if m["name"] == "dataset")
    assert dataset_mount["subPath"] == "peyman/TUM/fr3"


def test_k8s_manifest_dataset_pvc_scratch_root():
    # Found live: a ScanNet++ scene under /mnt/cps_scratch1_tmp silently
    # mounted pvc_persistent's ROOT (empty subPath) at /data/scene, since
    # the scene path never matched persistent_root at all -- crashed with
    # "No supported RGB-D dataset layout found", not a training-code bug.
    mcfg = {"namespace": "jupyterhub", "kai_queue": "kai-batch-low",
           "priority_class": "kai-batch-low", "image_pull_secret": "regcred",
           "pvc_persistent": "persist-pvc", "pvc_scratch": "scratch-pvc", "image": "img:1"}
    job = {"id": "j1", "scene": "/mnt/cps_scratch1_tmp/bjoern/scannetpp_cache/8b5caf3398"}
    manifest = runner.build_k8s_job_manifest(mcfg, job, ["python", "train.py"], None)
    volumes = {v["name"]: v for v in manifest["spec"]["template"]["spec"]["volumes"]}
    assert volumes["dataset"]["persistentVolumeClaim"]["claimName"] == "scratch-pvc"
    mounts = manifest["spec"]["template"]["spec"]["containers"][0]["volumeMounts"]
    dataset_mount = next(m for m in mounts if m["name"] == "dataset")
    assert dataset_mount["subPath"] == "bjoern/scannetpp_cache/8b5caf3398"


def test_k8s_manifest_dataset_pvc_unknown_root_falls_back():
    mcfg = {"namespace": "jupyterhub", "kai_queue": "kai-batch-low",
           "priority_class": "kai-batch-low", "image_pull_secret": "regcred",
           "pvc_persistent": "persist-pvc", "pvc_scratch": "scratch-pvc", "image": "img:1"}
    job = {"id": "j1", "scene": "/some/other/path/scene"}
    manifest = runner.build_k8s_job_manifest(mcfg, job, ["python", "train.py"], None)
    volumes = {v["name"]: v for v in manifest["spec"]["template"]["spec"]["volumes"]}
    assert volumes["dataset"]["persistentVolumeClaim"]["claimName"] == "persist-pvc"
    mounts = manifest["spec"]["template"]["spec"]["containers"][0]["volumeMounts"]
    dataset_mount = next(m for m in mounts if m["name"] == "dataset")
    assert dataset_mount["subPath"] == ""


# --------------------------------------------- build_k8s_job_manifest: git-sync

BASE_MCFG = {"namespace": "jupyterhub", "kai_queue": "kai-batch-low",
            "priority_class": "kai-batch-low", "image_pull_secret": "regcred",
            "pvc_persistent": "p", "pvc_scratch": "s", "image": "img:1"}
GIT_JOB = {"id": "j1", "scene": "/mnt/cps_persistent1_shared/scene"}


def test_k8s_manifest_no_git_sync_by_default():
    """Absent git_sync_repo_url -> byte-identical manifest to today (no
    initContainers key, no repo-src volume/mount) -- opt-in, not a
    behavior change for any machine that hasn't configured it."""
    manifest = runner.build_k8s_job_manifest(
        BASE_MCFG, GIT_JOB, ["python", "train.py"], None, local_commit="abc123")
    pod_spec = manifest["spec"]["template"]["spec"]
    assert "initContainers" not in pod_spec
    volume_names = {v["name"] for v in pod_spec["volumes"]}
    assert "repo-src" not in volume_names
    mount_names = {m["name"] for m in pod_spec["containers"][0]["volumeMounts"]}
    assert "repo-src" not in mount_names


def test_k8s_manifest_git_sync_adds_init_container_and_shared_volume():
    mcfg = {**BASE_MCFG, "git_sync_repo_url": "git@github.com:bjoernellens1/splatograph.git"}
    manifest = runner.build_k8s_job_manifest(
        mcfg, GIT_JOB, ["python", "train.py"], "/workspace/splatograph",
        local_commit="deadbeef1234")
    pod_spec = manifest["spec"]["template"]["spec"]
    assert len(pod_spec["initContainers"]) == 1
    init = pod_spec["initContainers"][0]
    assert init["name"] == "git-sync"
    assert init["image"] == "alpine/git:2.45.2"
    script = init["command"][-1]
    # Pins the EXACT dispatching-host commit SHA, not a moving branch head.
    assert "deadbeef1234" in script
    assert "git@github.com:bjoernellens1/splatograph.git" in script

    volumes_by_name = {v["name"]: v for v in pod_spec["volumes"]}
    assert "emptyDir" in volumes_by_name["repo-src"]

    init_mounts = {m["name"]: m for m in init["volumeMounts"]}
    assert init_mounts["repo-src"]["mountPath"] == "/workspace/splatograph"

    trainer = pod_spec["containers"][0]
    trainer_mounts = {m["name"]: m for m in trainer["volumeMounts"]}
    # Overlay decision: fresh checkout replaces the baked source at the
    # SAME path the trainer already uses, no new path/env for it to know
    # about.
    assert trainer_mounts["repo-src"]["mountPath"] == "/workspace/splatograph"


def test_k8s_manifest_git_sync_custom_image_overridable():
    mcfg = {**BASE_MCFG, "git_sync_repo_url": "git@github.com:x/y.git",
           "git_sync_image": "my-registry/git-sync:v1"}
    manifest = runner.build_k8s_job_manifest(
        mcfg, GIT_JOB, ["python", "train.py"], None, local_commit="sha1")
    init = manifest["spec"]["template"]["spec"]["initContainers"][0]
    assert init["image"] == "my-registry/git-sync:v1"


def test_k8s_manifest_git_sync_no_secret_means_no_creds_mount():
    """No git_sync_secret_name configured -> no secret volume/mount at all
    (public-repo case); easy to add later without restructuring."""
    mcfg = {**BASE_MCFG, "git_sync_repo_url": "https://github.com/bjoernellens1/splatograph.git"}
    manifest = runner.build_k8s_job_manifest(
        mcfg, GIT_JOB, ["python", "train.py"], None, local_commit="sha1")
    pod_spec = manifest["spec"]["template"]["spec"]
    volume_names = {v["name"] for v in pod_spec["volumes"]}
    assert "git-creds" not in volume_names
    init = pod_spec["initContainers"][0]
    mount_names = {m["name"] for m in init["volumeMounts"]}
    assert "git-creds" not in mount_names


def test_k8s_manifest_git_sync_with_secret_mounts_readonly_init_container_only():
    mcfg = {**BASE_MCFG, "git_sync_repo_url": "git@github.com:bjoernellens1/splatograph.git",
           "git_sync_secret_name": "splatograph-git-readonly"}
    manifest = runner.build_k8s_job_manifest(
        mcfg, GIT_JOB, ["python", "train.py"], None, local_commit="sha1")
    pod_spec = manifest["spec"]["template"]["spec"]
    volumes_by_name = {v["name"]: v for v in pod_spec["volumes"]}
    creds_vol = volumes_by_name["git-creds"]
    assert creds_vol["secret"]["secretName"] == "splatograph-git-readonly"
    assert creds_vol["secret"]["defaultMode"] == 0o400

    init = pod_spec["initContainers"][0]
    init_mount_names = {m["name"] for m in init["volumeMounts"]}
    assert "git-creds" in init_mount_names
    # The secret must NEVER be mounted into the trainer container.
    trainer = pod_spec["containers"][0]
    trainer_mount_names = {m["name"] for m in trainer["volumeMounts"]}
    assert "git-creds" not in trainer_mount_names

    ssh_cmd = next(e["value"] for e in init["env"] if e["name"] == "GIT_SSH_COMMAND")
    assert "/etc/git-creds/ssh-privatekey" in ssh_cmd


# --------------------------------- build_k8s_job_manifest: generic PyTorch jobs

def test_k8s_manifest_no_dataset_volume_when_no_pvc_configured():
    """A plain PyTorch job with no shared-dataset PVC configured mounts no
    dataset/scratch volumes at all -- not every k8s job is a splatograph
    scene replay."""
    mcfg = {"namespace": "cps-users", "kai_queue": "kai-batch-low",
           "priority_class": "kai-batch-low", "image": "docker.io/pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime"}
    job = {"id": "j1", "scene": ""}
    manifest = runner.build_k8s_job_manifest(mcfg, job, ["python", "train.py"], None)
    pod_spec = manifest["spec"]["template"]["spec"]
    assert pod_spec["volumes"] == []
    assert pod_spec["containers"][0]["volumeMounts"] == []


def test_k8s_manifest_no_image_pull_secret_when_not_configured():
    """Public image needs no imagePullSecrets key at all."""
    mcfg = {"namespace": "cps-users", "kai_queue": "kai-batch-low",
           "priority_class": "kai-batch-low", "image": "pytorch/pytorch:2.4.0"}
    job = {"id": "j1", "scene": ""}
    manifest = runner.build_k8s_job_manifest(mcfg, job, ["python", "train.py"], None)
    pod_spec = manifest["spec"]["template"]["spec"]
    assert "imagePullSecrets" not in pod_spec


def test_k8s_manifest_scheduler_name_configurable():
    mcfg = {"namespace": "cps-users", "kai_queue": "kai-batch-low",
           "priority_class": "kai-batch-low", "image": "img:1",
           "scheduler_name": "default-scheduler"}
    job = {"id": "j1", "scene": ""}
    manifest = runner.build_k8s_job_manifest(mcfg, job, ["python", "train.py"], None)
    assert manifest["spec"]["template"]["spec"]["schedulerName"] == "default-scheduler"


def test_k8s_manifest_scheduler_name_defaults_to_kai_scheduler():
    mcfg = {"namespace": "cps-users", "kai_queue": "kai-batch-low",
           "priority_class": "kai-batch-low", "image": "img:1"}
    job = {"id": "j1", "scene": ""}
    manifest = runner.build_k8s_job_manifest(mcfg, job, ["python", "train.py"], None)
    assert manifest["spec"]["template"]["spec"]["schedulerName"] == "kai-scheduler"


def test_k8s_manifest_resource_requests_overridable():
    mcfg = {"namespace": "cps-users", "kai_queue": "kai-batch-low",
           "priority_class": "kai-batch-low", "image": "img:1",
           "cpu_request": "2", "memory_request": "8Gi",
           "cpu_limit": "4", "memory_limit": "16Gi", "gpu_count": 2}
    job = {"id": "j1", "scene": ""}
    manifest = runner.build_k8s_job_manifest(mcfg, job, ["python", "train.py"], None)
    resources = manifest["spec"]["template"]["spec"]["containers"][0]["resources"]
    assert resources["requests"] == {"cpu": "2", "memory": "8Gi"}
    assert resources["limits"] == {"cpu": "4", "memory": "16Gi", "nvidia.com/gpu": "2"}


def test_k8s_manifest_extra_volumes_generic_pvc_mount():
    """A generic extra PVC (e.g. a shared checkpoint volume) mounts via
    plain config fields, no scene/dataset-specific naming required."""
    mcfg = {"namespace": "cps-users", "kai_queue": "kai-batch-low",
           "priority_class": "kai-batch-low", "image": "img:1",
           "extra_volumes": [
               {"name": "checkpoints", "claim_name": "shared-checkpoints-pvc",
                "mount_path": "/mnt/checkpoints"},
           ]}
    job = {"id": "j1", "scene": ""}
    manifest = runner.build_k8s_job_manifest(mcfg, job, ["python", "train.py"], None)
    pod_spec = manifest["spec"]["template"]["spec"]
    volumes = {v["name"]: v for v in pod_spec["volumes"]}
    assert volumes["checkpoints"]["persistentVolumeClaim"]["claimName"] == "shared-checkpoints-pvc"
    mounts = {m["name"]: m for m in pod_spec["containers"][0]["volumeMounts"]}
    assert mounts["checkpoints"]["mountPath"] == "/mnt/checkpoints"


def test_k8s_manifest_dataset_mount_path_configurable():
    mcfg = {"namespace": "cps-users", "kai_queue": "kai-batch-low",
           "priority_class": "kai-batch-low", "image": "img:1",
           "pvc_persistent": "persist-pvc", "pvc_scratch": "scratch-pvc",
           "dataset_mount_path": "/mnt/data"}
    job = {"id": "j1", "scene": "/mnt/cps_persistent1_shared/some/scene"}
    manifest = runner.build_k8s_job_manifest(mcfg, job, ["python", "train.py"], None)
    mounts = manifest["spec"]["template"]["spec"]["containers"][0]["volumeMounts"]
    dataset_mount = next(m for m in mounts if m["name"] == "dataset")
    assert dataset_mount["mountPath"] == "/mnt/data"


def test_k8s_manifest_default_workdir_configurable():
    mcfg = {"namespace": "cps-users", "kai_queue": "kai-batch-low",
           "priority_class": "kai-batch-low", "image": "img:1",
           "default_workdir": "/workspace/myproject"}
    job = {"id": "j1", "scene": ""}
    manifest = runner.build_k8s_job_manifest(mcfg, job, ["python", "train.py"], None)
    assert (manifest["spec"]["template"]["spec"]["containers"][0]["workingDir"]
            == "/workspace/myproject")


def test_k8s_manifest_generic_job_no_scene_no_pvc_still_valid():
    """The full end-to-end minimal case for a non-splatting PyTorch job:
    no scene, no dataset PVCs, no image_pull_secret, default scheduler/
    resources -- must still produce a valid, submittable manifest."""
    mcfg = {"namespace": "cps-users", "kai_queue": "kai-batch-low",
           "priority_class": "kai-batch-low",
           "image": "docker.io/pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime",
           "gpu_count": 1}
    job = {"id": "pytorch_generic_ctrl", "scene": ""}
    manifest = runner.build_k8s_job_manifest(
        mcfg, job, ["python", "train.py", "--epochs", "10"], "/workspace")
    pod_spec = manifest["spec"]["template"]["spec"]
    assert pod_spec["schedulerName"] == "kai-scheduler"
    assert pod_spec["containers"][0]["image"] == mcfg["image"]
    assert pod_spec["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] == "1"
    assert pod_spec["volumes"] == []


def test_k8s_manifest_git_sync_pins_head_when_no_local_commit():
    """No local_commit resolved (e.g. git not available on dispatcher) ->
    falls back to fetching HEAD of the default branch rather than crashing
    manifest construction; a degraded-but-safe path, not a silent SHA
    substitution."""
    mcfg = {**BASE_MCFG, "git_sync_repo_url": "git@github.com:x/y.git"}
    manifest = runner.build_k8s_job_manifest(
        mcfg, GIT_JOB, ["python", "train.py"], None, local_commit=None)
    script = manifest["spec"]["template"]["spec"]["initContainers"][0]["command"][-1]
    assert "git fetch --depth 1 origin HEAD" in script
