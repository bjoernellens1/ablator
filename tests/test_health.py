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


def test_done_with_model_path_prefixed_result_glob(tmp_path):
    """cli.py's `collect` documents/uses "{model_path}/comparison/*/report.json"
    (str.format()-templated against job vars). job_health() never templates —
    it joins result_glob directly onto the already-resolved model_path — so
    without normalizing this specific prefix, a real completed job with the
    documented config convention was ALWAYS misread as incomplete (glob'd for
    a literal "{model_path}" subdirectory that can never exist), causing
    require_result_artifact to falsely fail every job and trigger a wasted
    full retry. Confirmed live in production before this fix."""
    job = make_run(tmp_path, "Training: 30000/30000", report=True)
    qcfg = {"result_glob": "{model_path}/comparison/*/report.json"}
    assert health.job_health(job, str(tmp_path), qcfg)["state"] == "done"
    # bare form (this module's own documented/default convention) still works
    qcfg = {"result_glob": "comparison/*/report.json"}
    assert health.job_health(job, str(tmp_path), qcfg)["state"] == "done"


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


def drive(cfg, proc, healths, controls=None, mem_values=None, job=None):
    hi, ci = iter(healths), iter(controls or [])
    mi = iter(mem_values) if mem_values is not None else None
    recorded, kills = [], []
    job = job if job is not None else {"id": "j"}
    result = runner.supervise(
        cfg, job, proc, ".", sleep=lambda s: None,
        health_fn=lambda alive: next(hi),
        control=lambda: next(ci, None),
        kill=lambda: kills.append(True),
        record=recorded.append,
        mem_sampler=(lambda: next(mi, None)) if mi is not None else None)
    return result, recorded, kills


# --------------------------------------------------------- GPU memory guard

def test_supervise_kills_after_sustained_memory_danger(cfg):
    """3 consecutive polls >= mem_kill_danger_pct (default 90) triggers a
    kill, and stamps the job so classify_and_record() records the
    definitive category rather than guessing from the log tail."""
    job = {"id": "j", "model_path": "output/scratch/j"}
    result, _, kills = drive(cfg, FakePopen(), [H("training")] * 5,
                             mem_values=[95.0, 96.0, 97.0], job=job)
    assert result == "failed"
    assert kills == [True]
    assert job["_gpu_memory_exhausted"] is True
    assert job["_gpu_memory_pct"] == 97.0


def test_supervise_does_not_kill_on_brief_memory_spike(cfg):
    """A single poll above the danger threshold, followed by usage
    dropping back down, must not trigger a kill (grace period resets)."""
    job = {"id": "j", "model_path": "output/scratch/j"}
    result, _, kills = drive(cfg, FakePopen(exits_after=5), [H("training")] * 6,
                             mem_values=[95.0, 10.0, 12.0, 11.0, 13.0], job=job)
    assert result is None
    assert kills == []
    assert "_gpu_memory_exhausted" not in job


def test_memory_kill_lands_terminal_with_error_category(tmp_path, monkeypatch):
    """End-to-end: a job killed by the memory guard must land in a
    terminal ledger state (not stuck 'running') with
    error_category == 'gpu_memory_exhaustion', via the SAME
    handle_failure()/classify_and_record() path normal crash detection
    uses -- not a separate ad-hoc kill call bypassing ledger bookkeeping."""
    cfg = {"_path": "x", "queue": {"path": str(tmp_path / "queue.jsonl")},
          "machines": {"m": {}}, "types": {"replay": {"command": ["true"]}},
          "resources": {}}
    q = Queue(cfg["queue"]["path"])
    q.append([{"id": "j1", "status": "pending", "machine": "any",
               "type": "replay", "model_path": str(tmp_path / "run")}])

    def fake_supervise(cfg, job, proc, base_dir, q=None, **kw):
        job["_gpu_memory_exhausted"] = True
        job["_gpu_memory_pct"] = 96.0
        return "failed"

    monkeypatch.setattr(runner, "supervise", fake_supervise)
    monkeypatch.setattr(runner.cfgmod, "machine_name", lambda c: "m")
    monkeypatch.setattr(runner.resources, "machine_busy", lambda c, m: False)
    runner.run_loop(cfg, once=True)

    j = q.read()[0]
    assert j["status"] == "quarantined"
    assert j["error_category"] == "gpu_memory_exhaustion"


def test_machine_busy_treats_high_memory_as_busy(monkeypatch):
    from ablator import resources
    cfg = {"resources": {"mem_dispatch_busy_pct": 70,
                        "mem_budgets": {"main": {"used_path": "u", "total_path": "t"}}},
          "machines": {"main": {}}}
    idle = lambda: 0.0
    assert resources.machine_busy(cfg, "main", sampler=idle,
                                  mem_sampler=lambda: 75.0) is True
    assert resources.machine_busy(cfg, "main", sampler=idle,
                                  mem_sampler=lambda: 40.0) is False


def test_beszel_snapshot_uses_cli_then_local_fallback():
    from ablator import resources

    def beszel(command, timeout=15):
        return '{"systems":[{"name":"node","status":"up","cpu_percent":5,"memory_percent":10,"gpu_percent":3,"vram_used_gb":1,"vram_total_gb":24,"gpu_power_watts":40}]}'

    snapshot = resources.machine_telemetry_snapshot(
        {"machines": {"node": {"beszel_system": "node"}}}, "node", run=beszel
    )
    assert snapshot["source"] == "beszel" and snapshot["gpu_power_watts"] == 40
    fallback = resources.machine_telemetry_snapshot(
        {}, "node", run=lambda *args, **kwargs: None, cpu_sampler=lambda: 2, memory_sampler=lambda: (3, 8), gpu_sampler=lambda: 4
    )
    assert fallback["source"] == "local-fallback" and fallback["gpu_percent"] == 4


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
    monkeypatch.setattr(runner, "run_job", lambda c, j, m, q=None: ("requeue", None))
    runner.run_loop(cfg, once=True)
    j = q.read()[0]
    assert j["status"] == "pending"
    assert j["health"] is None and j["claimed_by"] is None
