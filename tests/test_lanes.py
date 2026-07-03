"""Priority-lane tests: claim ordering, default backfill, preemption state
machine, promote CLI command, once-per-30-minutes preemption guard."""
import json
import time

from ablator import cli, spec as specmod
from ablator.queue import Queue, job_lane, PREEMPT_COOLDOWN_S


def write_queue(path, jobs):
    with open(path, "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")


def read_queue(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def make_cfg(tmp_path):
    return {
        "_path": str(tmp_path / "config.json"),
        "queue": {"path": str(tmp_path / "queue.jsonl")},
        "machines": {"main": {"hostname_patterns": ["*"]}},
        "resources": {},
        "types": {"replay": {"cwd": "/repo", "command": ["true"]}},
    }


# ------------------------------------------------------------- job_lane

def test_job_lane_default_backfill():
    assert job_lane({"id": "a"}) == 2  # missing field -> lane 2


def test_job_lane_explicit_values():
    assert job_lane({"lane": 1}) == 1
    assert job_lane({"lane": 3}) == 3


def test_job_lane_invalid_falls_back_to_2():
    assert job_lane({"lane": 99}) == 2
    assert job_lane({"lane": "nonsense"}) == 2
    assert job_lane({"lane": None}) == 2


# ------------------------------------------------------------- claim ordering

def test_claim_next_lane_order(tmp_path):
    path = str(tmp_path / "queue.jsonl")
    write_queue(path, [
        {"id": "bg1", "lane": 1, "status": "pending", "machine": "any"},
        {"id": "std1", "lane": 2, "status": "pending", "machine": "any"},
        {"id": "urg1", "lane": 3, "status": "pending", "machine": "any"},
    ])
    q = Queue(path)
    first = q.claim_next("main")
    assert first["id"] == "urg1"
    second = q.claim_next("main")
    assert second["id"] == "std1"
    third = q.claim_next("main")
    assert third["id"] == "bg1"


def test_claim_next_default_lane_backfill(tmp_path):
    """Jobs missing 'lane' are treated as lane 2 and ordered accordingly."""
    path = str(tmp_path / "queue.jsonl")
    write_queue(path, [
        {"id": "legacy1", "status": "pending", "machine": "any"},  # no lane field
        {"id": "urg1", "lane": 3, "status": "pending", "machine": "any"},
        {"id": "bg1", "lane": 1, "status": "pending", "machine": "any"},
    ])
    q = Queue(path)
    order = [q.claim_next("main")["id"] for _ in range(3)]
    assert order == ["urg1", "legacy1", "bg1"]


def test_claim_next_preserves_file_order_within_lane(tmp_path):
    path = str(tmp_path / "queue.jsonl")
    write_queue(path, [
        {"id": "std_b", "lane": 2, "status": "pending", "machine": "any"},
        {"id": "std_a", "lane": 2, "status": "pending", "machine": "any"},
    ])
    q = Queue(path)
    assert q.claim_next("main")["id"] == "std_b"
    assert q.claim_next("main")["id"] == "std_a"


# ------------------------------------------------------------- preemption

def test_urgent_pending_true_when_lane3_pending(tmp_path):
    path = str(tmp_path / "queue.jsonl")
    write_queue(path, [{"id": "u", "lane": 3, "status": "pending", "machine": "any"}])
    q = Queue(path)
    assert q.urgent_pending("main") is True


def test_urgent_pending_false_when_no_lane3(tmp_path):
    path = str(tmp_path / "queue.jsonl")
    write_queue(path, [{"id": "s", "lane": 2, "status": "pending", "machine": "any"}])
    q = Queue(path)
    assert q.urgent_pending("main") is False


def test_preemption_due_only_for_lane1(tmp_path):
    path = str(tmp_path / "queue.jsonl")
    write_queue(path, [{"id": "u", "lane": 3, "status": "pending", "machine": "any"}])
    q = Queue(path)
    lane2_job = {"id": "s", "lane": 2}
    lane1_job = {"id": "b", "lane": 1}
    assert q.preemption_due(lane2_job, "main") is False  # never preempted
    assert q.preemption_due(lane1_job, "main") is True


def test_preemption_due_false_without_urgent(tmp_path):
    path = str(tmp_path / "queue.jsonl")
    write_queue(path, [])
    q = Queue(path)
    assert q.preemption_due({"id": "b", "lane": 1}, "main") is False


def test_preemption_guard_once_per_30_minutes(tmp_path):
    path = str(tmp_path / "queue.jsonl")
    write_queue(path, [{"id": "u", "lane": 3, "status": "pending", "machine": "any"}])
    q = Queue(path)
    now = time.time()
    job = {"id": "b", "lane": 1, "last_preempt_at": now - 10}  # 10s ago
    assert q.preemption_due(job, "main", now=now) is False  # within cooldown
    job2 = {"id": "b", "lane": 1, "last_preempt_at": now - PREEMPT_COOLDOWN_S - 1}
    assert q.preemption_due(job2, "main", now=now) is True  # cooldown elapsed


def test_supervise_returns_preempted_for_lane1_job(monkeypatch):
    """Monkeypatch kill/preempt in supervise() — no real process spawned."""
    from ablator import runner as runnermod

    class FakeProc:
        def poll(self):
            return None  # still running

        def wait(self, timeout=None):
            raise TimeoutError()

    proc = FakeProc()
    job = {"id": "bg1", "lane": 1}
    killed = []
    status = runnermod.supervise(
        cfg={}, job=job, proc=proc, base_dir="/tmp",
        sleep=lambda s: None,
        health_fn=lambda alive: {"state": "ok"},
        kill=lambda: killed.append(True),
        record=lambda h: None,
        control=lambda: None,
        preempt=lambda: True,
    )
    assert status == "preempted"
    assert killed == [True]


def test_supervise_never_preempts_lane2(monkeypatch):
    """The real preempt callback (Queue.preemption_due) is what enforces the
    lane-2-never-preempted rule; supervise() itself just calls it. Wire the
    real callback bound to a lane-2 job and confirm no kill happens even
    with an urgent job pending."""
    from ablator import runner as runnermod
    from ablator.queue import Queue

    path_jobs = [{"id": "u", "lane": 3, "status": "pending", "machine": "any"}]

    class FakeQueue(Queue):
        def __init__(self):
            pass

        def read(self):
            return path_jobs

    q = FakeQueue()
    job = {"id": "std1", "lane": 2}

    calls = {"n": 0}

    class FakeProc:
        def poll(self):
            calls["n"] += 1
            return None if calls["n"] < 2 else 0

        def wait(self, timeout=None):
            raise TimeoutError()

    proc = FakeProc()
    status = runnermod.supervise(
        cfg={}, job=job, proc=proc, base_dir="/tmp", q=q,
        sleep=lambda s: None,
        health_fn=lambda alive: {"state": "ok"},
        kill=lambda: (_ for _ in ()).throw(AssertionError("must not kill lane2")),
        record=lambda h: None,
        control=lambda: None,
        preempt=lambda: q.preemption_due(job, "main"),
    )
    assert status is None  # process exits on its own; lane-2 never preempted


# ------------------------------------------------------------- spec lane

def test_spec_expand_default_lane_is_2():
    spec = {"name": "abl", "base": {"scene": "/data/x"},
            "arms": [{"id": "a"}]}
    jobs = specmod.expand_spec(spec)
    assert jobs[0]["lane"] == 2


def test_spec_expand_top_level_lane():
    spec = {"name": "abl", "lane": 1, "base": {"scene": "/data/x"},
            "arms": [{"id": "a"}]}
    jobs = specmod.expand_spec(spec)
    assert jobs[0]["lane"] == 1


def test_spec_expand_per_arm_lane_override():
    spec = {"name": "abl", "lane": 1, "base": {"scene": "/data/x"},
            "arms": [{"id": "a"}, {"id": "b", "lane": 3}]}
    jobs = specmod.expand_spec(spec)
    assert jobs[0]["lane"] == 1
    assert jobs[1]["lane"] == 3


def test_spec_expand_invalid_lane_rejected():
    spec = {"name": "abl", "lane": 7, "base": {"scene": "/data/x"},
            "arms": [{"id": "a"}]}
    try:
        specmod.expand_spec(spec)
        assert False, "expected SystemExit"
    except SystemExit:
        pass


# ------------------------------------------------------------- promote

def test_cmd_promote_moves_pending_job(tmp_path, capsys):
    cfg = make_cfg(tmp_path)
    write_queue(cfg["queue"]["path"],
               [{"id": "abl_a", "lane": 1, "status": "pending", "machine": "any"}])
    cli.cmd_promote(cfg, "abl_a", "3")
    jobs = read_queue(cfg["queue"]["path"])
    assert jobs[0]["lane"] == 3


def test_cmd_promote_refuses_non_pending(tmp_path):
    cfg = make_cfg(tmp_path)
    write_queue(cfg["queue"]["path"],
               [{"id": "abl_a", "lane": 1, "status": "running", "machine": "any"}])
    try:
        cli.cmd_promote(cfg, "abl_a", "3")
        assert False, "expected SystemExit"
    except SystemExit:
        pass
    jobs = read_queue(cfg["queue"]["path"])
    assert jobs[0]["lane"] == 1  # unchanged


def test_cmd_promote_rejects_invalid_lane(tmp_path):
    cfg = make_cfg(tmp_path)
    write_queue(cfg["queue"]["path"],
               [{"id": "abl_a", "lane": 1, "status": "pending", "machine": "any"}])
    try:
        cli.cmd_promote(cfg, "abl_a", "9")
        assert False, "expected SystemExit"
    except SystemExit:
        pass


def test_cmd_promote_unknown_job(tmp_path):
    cfg = make_cfg(tmp_path)
    write_queue(cfg["queue"]["path"], [])
    try:
        cli.cmd_promote(cfg, "nope", "3")
        assert False, "expected SystemExit"
    except SystemExit:
        pass


# ------------------------------------------------------------- status restock warning

def test_restock_warning_when_lane1_dry():
    jobs = [{"id": "a", "lane": 2, "status": "pending"}]
    assert cli._lane1_restock_warning(jobs) is not None


def test_restock_warning_absent_when_lane1_well_stocked():
    jobs = [{"id": "a", "lane": 1, "status": "pending"},
            {"id": "b", "lane": 1, "status": "pending"}]
    assert cli._lane1_restock_warning(jobs) is None
