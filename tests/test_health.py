"""Tests for ablator.health and the runner supervision/control machinery."""
import json
import os
import time

import pytest

from ablator import cli, health, runner
from ablator.queue import Queue


# ---------------------------------------------------------------- health

def make_run(tmp_path, log_text=None, log_age_s=0.0, report=False):
    mp = tmp_path / "run"
    mp.mkdir(exist_ok=True)
    if log_text is not None:
        log = mp / "train.log"
        log.write_text(log_text)
        t = time.time() - log_age_s
        os.utime(log, (t, t))
    if report:
        rd = mp / "comparison" / "iter_30000"
        rd.mkdir(parents=True, exist_ok=True)
        (rd / "report.json").write_text("{}")
    return {"id": "j1", "model_path": str(mp), "extra_args": ""}


def test_starting_no_log(tmp_path):
    h = health.job_health(make_run(tmp_path), str(tmp_path))
    assert h["state"] == "starting"
    assert h["iter"] is None and h["log_age_s"] is None


def test_training(tmp_path):
    job = make_run(tmp_path, "Training: 12345/30000 [12:00<18:00]")
    h = health.job_health(job, str(tmp_path))
    assert h["state"] == "training"
    assert (h["iter"], h["total"]) == (12345, 30000)


def test_hung_by_mtime(tmp_path):
    job = make_run(tmp_path, "Training: 100/30000", log_age_s=30 * 60)
    h = health.job_health(job, str(tmp_path))  # default 20 min
    assert h["state"] == "hung"
    h = health.job_health(job, str(tmp_path), {"hung_after_min": 60})
    assert h["state"] == "training"
    # per-job override beats config
    job["hung_after_min"] = 10
    h = health.job_health(job, str(tmp_path), {"hung_after_min": 60})
    assert h["state"] == "hung"


def test_crashed_markers(tmp_path):
    for marker in ("Traceback (most recent call last):",
                   "RuntimeError: HIP error: out of memory",
                   "what(): std::exception"):
        job = make_run(tmp_path, f"Training: 5/30000\n{marker}\n")
        assert health.job_health(job, str(tmp_path))["state"] == "crashed"


def test_crashed_process_dead_no_report(tmp_path):
    job = make_run(tmp_path, "Training: 5/30000")
    h = health.job_health(job, str(tmp_path), process_alive=False)
    assert h["state"] == "crashed"
    job = make_run(tmp_path, "Training: 30000/30000", report=True)
    h = health.job_health(job, str(tmp_path), process_alive=False)
    assert h["state"] == "done"


def test_reporting_and_done(tmp_path):
    job = make_run(tmp_path, "Training: 30000/30000")
    assert health.job_health(job, str(tmp_path))["state"] == "reporting"
    job = make_run(tmp_path, "Training: 30000/30000", report=True)
    assert health.job_health(job, str(tmp_path))["state"] == "done"


def test_online_sentinel_uses_cap(tmp_path):
    job = make_run(tmp_path, f"Training: 500/{2**31 - 1}")
    job["extra_args"] = "--streaming_max_iterations 6000"
    h = health.job_health(job, str(tmp_path))
    assert (h["iter"], h["total"]) == (500, 6000)
    job["extra_args"] = ""
    h = health.job_health(job, str(tmp_path))
    assert (h["iter"], h["total"]) == (500, None)
    assert h["state"] == "training"


# ------------------------------------------------------------ supervision

@pytest.fixture
def cfg(tmp_path):
    return {"_path": "x", "queue": {"path": str(tmp_path / "queue.jsonl")},
            "machines": {}, "types": {}, "resources": {}}


class FakePopen:
    pid = 4242

    def __init__(self, exits_after=None, rc=0):
        self.exits_after, self.rc, self.polls = exits_after, rc, 0

    def poll(self):
        self.polls += 1
        if self.exits_after is not None and self.polls > self.exits_after:
            return self.rc
        return None


def H(state, **kw):
    return {"state": state, "iter": kw.get("iter"), "total": kw.get("total"),
            "log_age_s": kw.get("log_age_s")}


def drive(cfg, proc, healths, controls=None):
    hi, ci = iter(healths), iter(controls or [])
    recorded, kills = [], []
    result = runner.supervise(
        cfg, {"id": "j"}, proc, ".", sleep=lambda s: None,
        health_fn=lambda alive: next(hi),
        control=lambda: next(ci, None),
        kill=lambda: kills.append(True),
        record=recorded.append)
    return result, recorded, kills


def test_supervise_normal_exit(cfg):
    result, recorded, kills = drive(cfg, FakePopen(exits_after=3),
                                    [H("training")] * 10)
    assert result is None and kills == [] and recorded


def test_supervise_kills_hung(cfg):
    result, _, kills = drive(cfg, FakePopen(),
                             [H("training"), H("hung", log_age_s=1500)] + [H("hung")] * 3)
    assert result == "failed" and kills == [True]


def test_supervise_kills_crashed(cfg):
    result, _, kills = drive(cfg, FakePopen(), [H("crashed")] * 3)
    assert result == "failed" and kills == [True]


@pytest.mark.parametrize("action,status", [
    ("stop", "failed_no_retry"), ("skip", "cancelled"), ("requeue", "requeue"),
])
def test_supervise_control_actions(cfg, action, status):
    result, _, kills = drive(cfg, FakePopen(), [H("training")] * 5,
                             controls=[None, action])
    assert result == status and kills == [True]


def test_read_control_consumes_file(cfg):
    os.makedirs(os.path.dirname(cfg["queue"]["path"]), exist_ok=True)
    path = runner.control_path(cfg, "jx")
    with open(path, "w") as f:
        f.write("requeue\n")
    assert runner.read_control(cfg, "jx") == "requeue"
    assert not os.path.exists(path)
    assert runner.read_control(cfg, "jx") is None
    with open(path, "w") as f:
        f.write("garbage")
    assert runner.read_control(cfg, "jx") is None  # ignored but consumed
    assert not os.path.exists(path)


def test_cmd_control_writes_file(cfg, capsys):
    q = Queue(cfg["queue"]["path"])
    q.append([{"id": "a", "status": "running"},
              {"id": "p", "status": "pending"}])
    cli.cmd_control(cfg, "stop", "a")
    with open(runner.control_path(cfg, "a")) as f:
        assert f.read().strip() == "stop"
    with pytest.raises(SystemExit, match="not running"):
        cli.cmd_control(cfg, "skip", "p")
    with pytest.raises(SystemExit, match="no job"):
        cli.cmd_control(cfg, "stop", "ghost")


def test_cmd_health_named_job(cfg, tmp_path, capsys):
    job = make_run(tmp_path, "Training: 10/100")
    job.update(status="running", type="replay")
    Queue(cfg["queue"]["path"]).append([job])
    cli.cmd_health(cfg, "j1")
    out = capsys.readouterr().out
    h = json.loads(out.split(":", 1)[1])
    assert h["state"] == "training" and h["iter"] == 10


def test_status_shows_health_note(cfg):
    Queue(cfg["queue"]["path"]).append([
        {"id": "a", "status": "running", "model_path": "",
         "health": {"state": "hung", "log_age_s": 1500.2}}])
    lines = cli._status_lines(cfg, Queue(cfg["queue"]["path"]).read())
    assert any("[hung 1500s]" in l for l in lines)


def test_requeue_status_resets_job(cfg, monkeypatch):
    """run_loop maps 'requeue' back to pending with claim/health cleared."""
    q = Queue(cfg["queue"]["path"])
    q.append([{"id": "j", "status": "pending", "machine": "any",
               "type": "replay"}])
    cfg["types"] = {"replay": {"command": ["true"]}}
    cfg["machines"] = {"m": {}}
    monkeypatch.setattr(runner.cfgmod, "machine_name", lambda c: "m")
    monkeypatch.setattr(runner.resources, "machine_busy", lambda c, m: False)
    monkeypatch.setattr(runner, "run_job", lambda c, j, m, q=None: "requeue")
    runner.run_loop(cfg, once=True)
    j = q.read()[0]
    assert j["status"] == "pending"
    assert j["health"] is None and j["claimed_by"] is None
