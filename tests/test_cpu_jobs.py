"""`[types.<t>] requires_gpu = false`: CPU-only jobs are claimed while the
GPU is busy and run on a background thread; the done/failed verdict honours
the type's own result_glob. See runner.type_requires_gpu / run_loop step 1b
and runner._health_qcfg."""

from __future__ import annotations

import json
import os

from ablator import config as cfgmod
from ablator import health as healthmod
from ablator import resources, runner
from ablator.queue import Queue


def _cfg(tmp_path):
    return {
        "_path": str(tmp_path / "config.json"),
        "queue": {"path": str(tmp_path / "queue.jsonl"), "log_dir": str(tmp_path)},
        "machines": {"main": {"hostname_patterns": ["*"]}},
        "types": {
            "replay": {"command": ["true"]},
            "cpu": {"command": ["true"], "requires_gpu": False},
        },
    }


def _write(path, jobs):
    with open(path, "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")


def _read(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def test_type_requires_gpu_defaults_true():
    assert runner.type_requires_gpu({"command": ["x"]}) is True
    assert runner.type_requires_gpu({"command": ["x"], "requires_gpu": False}) is False


def test_make_can_run_filters_by_requires_gpu(tmp_path):
    cfg = _cfg(tmp_path)
    gpu_only = runner.make_can_run(cfg, "main", requires_gpu=True)
    cpu_only = runner.make_can_run(cfg, "main", requires_gpu=False)
    either = runner.make_can_run(cfg, "main")
    assert gpu_only({"id": "a", "type": "replay"}) and not gpu_only({"id": "b", "type": "cpu"})
    assert cpu_only({"id": "b", "type": "cpu"}) and not cpu_only({"id": "a", "type": "replay"})
    assert either({"id": "a", "type": "replay"}) and either({"id": "b", "type": "cpu"})


def test_cpu_job_runs_while_gpu_busy_and_gpu_job_waits(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    q = Queue(cfg["queue"]["path"])
    _write(q.path, [
        {"id": "gpu1", "machine": "main", "type": "replay", "scene": "/s",
         "model_path": str(tmp_path / "gpu1"), "status": "pending"},
        {"id": "cpu1", "machine": "main", "type": "cpu", "scene": "",
         "model_path": str(tmp_path / "cpu1"), "status": "pending"},
    ])
    monkeypatch.setattr(resources, "machine_busy", lambda *a, **k: True)
    monkeypatch.setattr(cfgmod, "machine_name", lambda c: "main")
    runner.run_loop(cfg, once=True)
    by_id = {j["id"]: j for j in _read(q.path)}
    assert by_id["cpu1"]["status"] == "done"
    assert by_id["gpu1"]["status"] == "pending"


def test_cpu_max_concurrent_bounds_claims(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg["resources"] = {"cpu_max_concurrent": 0}
    q = Queue(cfg["queue"]["path"])
    _write(q.path, [{"id": "cpu1", "machine": "main", "type": "cpu", "scene": "",
                     "model_path": str(tmp_path / "cpu1"), "status": "pending"}])
    monkeypatch.setattr(resources, "machine_busy", lambda *a, **k: True)
    monkeypatch.setattr(cfgmod, "machine_name", lambda c: "main")
    runner.run_loop(cfg, once=True)
    assert _read(q.path)[0]["status"] == "pending"


def test_idle_machine_still_runs_gpu_job_serially(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    q = Queue(cfg["queue"]["path"])
    _write(q.path, [{"id": "gpu1", "machine": "main", "type": "replay", "scene": "/s",
                     "model_path": str(tmp_path / "gpu1"), "status": "pending"}])
    monkeypatch.setattr(resources, "machine_busy", lambda *a, **k: False)
    monkeypatch.setattr(cfgmod, "machine_name", lambda c: "main")
    runner.run_loop(cfg, once=True)
    assert _read(q.path)[0]["status"] == "done"


def test_health_verdict_honours_type_result_glob(tmp_path):
    cfg = _cfg(tmp_path)
    cfg["queue"]["result_glob"] = "{model_path}/comparison/*/report.json"
    cfg["types"]["cpu"]["result_glob"] = "junit.xml"
    mp = tmp_path / "cpu1"
    mp.mkdir()
    (mp / "junit.xml").write_text("<testsuite/>")
    job = {"id": "cpu1", "type": "cpu", "model_path": str(mp)}
    queue_only = healthmod.job_health(job, str(tmp_path), cfg["queue"], process_alive=False)
    typed = healthmod.job_health(job, str(tmp_path), runner._health_qcfg(cfg, cfg["types"]["cpu"]),
                                 process_alive=False)
    assert queue_only["state"] != "done"
    assert typed["state"] == "done"


def test_cpu_job_exit0_without_artifact_is_failed_when_required(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg["types"]["cpu"].update(require_result_artifact=True, result_glob="junit.xml")
    q = Queue(cfg["queue"]["path"])
    _write(q.path, [{"id": "cpu1", "machine": "main", "type": "cpu", "scene": "",
                     "model_path": str(tmp_path / "cpu1"), "status": "pending"}])
    monkeypatch.setattr(resources, "machine_busy", lambda *a, **k: True)
    monkeypatch.setattr(cfgmod, "machine_name", lambda c: "main")
    runner.run_loop(cfg, once=True)
    assert _read(q.path)[0]["status"] in ("failed", "quarantined")
