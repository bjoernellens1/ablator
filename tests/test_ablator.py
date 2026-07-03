"""CPU-only tests for ablator (spec expansion, queue, resources, templates)."""
import json
import os

import pytest

from ablator import cli, config as cfgmod, progress as progmod, resources, runner, spec as specmod
from ablator.queue import Queue


# ------------------------------------------------------------------ helpers

def make_spec(parallel=True, machine="any", arms=None):
    return {
        "name": "abl",
        "parallel": parallel,
        "base": {
            "type": "replay",
            "scene": "/data/fr3",
            "iterations": 30000,
            "machine": machine,
            "base_args": "--opacity_reg 0.001",
        },
        "arms": arms or [
            {"id": "ctrl", "extra_args": ""},
            {"id": "consol", "extra_args": "--foo bar"},
        ],
    }


def make_cfg(tmp_path):
    return {
        "_path": str(tmp_path / "config.json"),
        "queue": {"path": str(tmp_path / "queue.jsonl")},
        "machines": {
            "r9700": {"hostname_patterns": ["*r9700*"]},
            "main": {"hostname_patterns": ["*"]},
        },
        "resources": {},
        "types": {
            "replay": {
                "cwd": "/repo",
                "command": ["podman", "run", "-v", "{scene}:/data/scene:ro",
                            "img:gfx1151", "python", "train.py",
                            "-m", "{model_path}",
                            "--iterations", "{iterations}", "{extra_args}"],
                "env": {"SCENE_SOURCE": "{scene}"},
                "machines": {
                    "r9700": {"command": ["docker", "run", "img:gfx1201",
                                          "{id}", "{extra_args}"]},
                },
            },
            "bag": {
                "command": ["bash", "launch.sh", "{scene}"],
                "env": {"TRAIN_EXTRA_ARGS": "{extra_args} --report"},
                "machines": {
                    "r9700": {
                        "require_images": ["img:a", "img:b"],
                        "image_probe_runtime": "docker",
                        "env": {"CONTAINER_RUNTIME": "docker"},
                    },
                },
            },
        },
    }


def read_queue(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def write_queue(path, jobs):
    with open(path, "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")


# ------------------------------------------------------------- spec expansion

def test_expand_ids_paths_args():
    jobs = specmod.expand_spec(make_spec())
    assert [j["id"] for j in jobs] == ["abl_ctrl", "abl_consol"]
    assert jobs[0]["model_path"] == "output/scratch/abl_ctrl"
    assert jobs[0]["extra_args"] == "--opacity_reg 0.001"
    assert jobs[1]["extra_args"] == "--opacity_reg 0.001 --foo bar"
    assert all(j["status"] == "pending" for j in jobs)
    assert all(j["machine"] == "any" for j in jobs)
    assert all(j["iterations"] == 30000 for j in jobs)
    assert all("depends_on" not in j for j in jobs)  # parallel


def test_expand_sequential_chains_depends_on():
    spec = make_spec(parallel=False, arms=[{"id": "a"}, {"id": "b"}, {"id": "c"}])
    jobs = specmod.expand_spec(spec)
    assert "depends_on" not in jobs[0]
    assert jobs[1]["depends_on"] == "abl_a"
    assert jobs[2]["depends_on"] == "abl_b"


def test_arm_overrides_machine_type_iterations():
    spec = make_spec(arms=[
        {"id": "x", "machine": "main", "type": "bag", "iterations": 60000},
        {"id": "y"},
    ])
    jobs = specmod.expand_spec(spec)
    assert jobs[0]["machine"] == "main"
    assert jobs[0]["type"] == "bag"
    assert jobs[0]["iterations"] == 60000
    assert jobs[1]["machine"] == "any"
    assert jobs[1]["type"] == "replay"


def test_arm_scene_and_base_args_override():
    spec = make_spec(arms=[
        {"id": "x", "scene": "/data/kitchen1", "base_args": "--other 2",
         "extra_args": "--foo bar"},
        {"id": "y"},
    ])
    jobs = specmod.expand_spec(spec)
    assert jobs[0]["scene"] == "/data/kitchen1"
    assert jobs[0]["extra_args"] == "--other 2 --foo bar"  # base_args replaced
    assert jobs[1]["scene"] == spec["base"]["scene"]


def test_expand_refuses_duplicate_arm_ids():
    spec = make_spec(arms=[{"id": "x"}, {"id": "x"}])
    with pytest.raises(SystemExit, match="duplicate arm id"):
        specmod.expand_spec(spec)


def test_model_path_template_configurable():
    jobs = specmod.expand_spec(make_spec(), model_path_template="/out/{name}/{arm}")
    assert jobs[0]["model_path"] == "/out/abl/ctrl"


# ---------------------------------------------------------------------- plan

def test_plan_enqueues_and_refuses_duplicates(tmp_path):
    cfg = make_cfg(tmp_path)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(make_spec()))
    cli.cmd_plan(cfg, str(spec_path))
    jobs = read_queue(cfg["queue"]["path"])
    assert {j["id"] for j in jobs} == {"abl_ctrl", "abl_consol"}
    with pytest.raises(SystemExit, match="duplicate"):
        cli.cmd_plan(cfg, str(spec_path))
    assert len(read_queue(cfg["queue"]["path"])) == 2  # unchanged


def test_plan_rejects_unknown_type(tmp_path):
    cfg = make_cfg(tmp_path)
    spec = make_spec()
    spec["base"]["type"] = "nosuch"
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    with pytest.raises(SystemExit, match="nosuch"):
        cli.cmd_plan(cfg, str(spec_path))
    assert not os.path.exists(cfg["queue"]["path"])


# --------------------------------------------------------------------- queue

def test_claim_respects_depends_on(tmp_path):
    q = Queue(str(tmp_path / "q.jsonl"))
    write_queue(q.path, [
        {"id": "a", "machine": "any", "type": "replay", "status": "pending"},
        {"id": "b", "machine": "any", "type": "replay", "status": "pending",
         "depends_on": "a"},
    ])
    assert q.claim_next("main")["id"] == "a"
    assert q.claim_next("main") is None  # b blocked while a running
    q.finish("a", "done")
    assert q.claim_next("main")["id"] == "b"


def test_claim_blocked_by_failed_dependency(tmp_path):
    q = Queue(str(tmp_path / "q.jsonl"))
    write_queue(q.path, [
        {"id": "a", "status": "quarantined"},
        {"id": "b", "status": "pending", "depends_on": "a"},
    ])
    assert q.claim_next("main") is None


def test_claim_skips_cancelled_and_wrong_machine(tmp_path):
    q = Queue(str(tmp_path / "q.jsonl"))
    write_queue(q.path, [
        {"id": "a", "status": "cancelled"},
        {"id": "b", "machine": "r9700", "status": "pending"},
        {"id": "c", "machine": "any", "status": "pending"},
    ])
    job = q.claim_next("main")
    assert job["id"] == "c"
    assert job["claimed_by"] == "main"
    assert read_queue(q.path)[2]["status"] == "running"


def test_claim_respects_can_run_predicate(tmp_path):
    q = Queue(str(tmp_path / "q.jsonl"))
    write_queue(q.path, [
        {"id": "bag1", "machine": "any", "type": "bag", "status": "pending"},
        {"id": "rep1", "machine": "any", "type": "replay", "status": "pending"},
    ])
    only_replay = lambda j: j["type"] == "replay"
    assert q.claim_next("r9700", can_run=only_replay)["id"] == "rep1"
    assert q.claim_next("r9700", can_run=lambda j: True)["id"] == "bag1"


def test_cancel_marks_pending_only(tmp_path):
    cfg = make_cfg(tmp_path)
    write_queue(cfg["queue"]["path"], [
        {"id": "abl_a", "ablation": "abl", "status": "pending"},
        {"id": "abl_b", "ablation": "abl", "status": "running"},
        {"id": "other_x", "ablation": "other", "status": "pending"},
    ])
    cli.cmd_cancel(cfg, "abl")
    jobs = {j["id"]: j for j in read_queue(cfg["queue"]["path"])}
    assert jobs["abl_a"]["status"] == "cancelled"
    assert jobs["abl_b"]["status"] == "running"
    assert jobs["other_x"]["status"] == "pending"


# --------------------------------------------------------- machine identity

def test_machine_name_patterns(tmp_path):
    cfg = make_cfg(tmp_path)
    assert cfgmod.machine_name(cfg, hostname="cps-wkstn-amd1R9700") == "r9700"
    assert cfgmod.machine_name(cfg, hostname="strix-halo") == "main"
    assert cfgmod.machine_name({"machines": {}}, hostname="x") == "unknown"


# ------------------------------------------------------------ GPU util guard

def make_sampler(values):
    it = iter(values)
    return lambda: next(it)


def test_gpu_util_busy_two_samples_required(monkeypatch):
    monkeypatch.delenv("ABLATOR_GPU_BUSY_PCT", raising=False)
    nosleep = lambda s: None
    busy = lambda vals: resources.gpu_util_busy({}, make_sampler(vals), nosleep)
    assert busy([50.0, 45.0]) is True     # both above -> busy
    assert busy([90.0, 5.0]) is False     # spike then idle -> not busy
    assert busy([3.0]) is False           # idle first -> no second sample
    assert busy([20.0]) is False          # exactly at threshold not busy
    assert busy([None]) is False          # probe failure -> not busy
    assert busy([80.0, None]) is False


def test_gpu_util_busy_config_and_env_threshold(monkeypatch):
    nosleep = lambda s: None
    cfg = {"resources": {"gpu_busy_pct": 60}}
    monkeypatch.delenv("ABLATOR_GPU_BUSY_PCT", raising=False)
    assert resources.gpu_util_busy(cfg, make_sampler([50.0]), nosleep) is False
    assert resources.gpu_util_busy(cfg, make_sampler([70.0, 65.0]), nosleep) is True
    monkeypatch.setenv("ABLATOR_GPU_BUSY_PCT", "90")
    assert resources.gpu_util_busy(cfg, make_sampler([70.0]), nosleep) is False


def test_machine_busy_config_guards(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    cfg["machines"]["main"]["busy_guards"] = [
        {"command": ["fake-ps"], "contains": "splat_train"},
        {"command": ["fake-pgrep"]},
    ]
    outputs = {"fake-ps": "web\nsplat_train\n", "fake-pgrep": ""}
    monkeypatch.setattr(resources, "_run", lambda cmd, timeout=15: outputs.get(cmd[0]))
    idle = lambda: 0.0
    assert resources.machine_busy(cfg, "main", sampler=idle) is True
    outputs["fake-ps"] = "web\n"
    assert resources.machine_busy(cfg, "main", sampler=idle) is False
    outputs["fake-pgrep"] = "1234\n"  # empty contains -> any output is busy
    assert resources.machine_busy(cfg, "main", sampler=idle) is True


# ------------------------------------------------------- template rendering

JOB = {"id": "abl_ctrl", "type": "replay", "scene": "/data/fr3",
       "model_path": "output/scratch/abl_ctrl",
       "extra_args": "--opacity_reg 0.001 --foo bar", "iterations": 30000}


def test_render_command_substitution_and_extra_args_split(tmp_path):
    cfg = make_cfg(tmp_path)
    tcfg = cfgmod.type_cfg(cfg, "replay", "main")
    argv, env, cwd = runner.render_command(tcfg, JOB, "main")
    assert argv[0] == "podman"
    assert "/data/fr3:/data/scene:ro" in argv
    assert argv[-4:] == ["--opacity_reg", "0.001", "--foo", "bar"]  # split
    assert "30000" in argv
    assert env["SCENE_SOURCE"] == "/data/fr3"
    assert cwd == "/repo"


def test_render_command_per_machine_override(tmp_path):
    cfg = make_cfg(tmp_path)
    tcfg = cfgmod.type_cfg(cfg, "replay", "r9700")
    argv, env, _ = runner.render_command(tcfg, JOB, "r9700")
    assert argv[:2] == ["docker", "run"]
    assert "abl_ctrl" in argv
    assert env["SCENE_SOURCE"] == "/data/fr3"  # base env survives override


def test_render_env_merge_and_templating(tmp_path):
    cfg = make_cfg(tmp_path)
    tcfg = cfgmod.type_cfg(cfg, "bag", "r9700")
    _, env, _ = runner.render_command(tcfg, {**JOB, "type": "bag"}, "r9700")
    assert env["TRAIN_EXTRA_ARGS"] == "--opacity_reg 0.001 --foo bar --report"
    assert env["CONTAINER_RUNTIME"] == "docker"


def test_render_unknown_variable_fails(tmp_path):
    tcfg = {"command": ["run", "{nosuch}"]}
    with pytest.raises(SystemExit, match="nosuch"):
        runner.render_command(tcfg, JOB, "main")


# ---------------------------------------------------------------- capability

def test_type_capable_probes_images(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    tcfg = cfgmod.type_cfg(cfg, "bag", "r9700")
    present = {"img:a"}
    probed = []

    def fake_run(cmd, timeout=15):
        assert cmd[:3] == ["docker", "images", "-q"]
        probed.append(cmd[3])
        return "abc123\n" if cmd[3] in present else ""

    monkeypatch.setattr(resources, "_run", fake_run)
    assert runner.type_capable(tcfg) is False
    assert "img:b" in probed
    present.add("img:b")
    assert runner.type_capable(tcfg) is True
    # no require_images (main) -> capable without probing
    monkeypatch.setattr(resources, "_run",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("probed")))
    assert runner.type_capable(cfgmod.type_cfg(cfg, "bag", "main")) is True


def test_make_can_run_gates_by_capability(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    q = Queue(cfg["queue"]["path"])
    write_queue(q.path, [
        {"id": "bag1", "machine": "any", "type": "bag", "status": "pending"},
        {"id": "rep1", "machine": "any", "type": "replay", "status": "pending"},
    ])
    monkeypatch.setattr(resources, "images_present", lambda rt, imgs: False)
    assert q.claim_next("r9700", runner.make_can_run(cfg, "r9700"))["id"] == "rep1"
    monkeypatch.setattr(resources, "images_present", lambda rt, imgs: True)
    assert q.claim_next("r9700", runner.make_can_run(cfg, "r9700"))["id"] == "bag1"


# ---------------------------------------------------------- runner end-to-end

def test_run_loop_once_executes_and_retries(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    cfg["queue"]["log_dir"] = str(tmp_path)
    cfg["types"]["replay"] = {"command": ["false"]}  # always fails
    q = Queue(cfg["queue"]["path"])
    write_queue(q.path, [{"id": "j1", "machine": "any", "type": "replay",
                          "scene": "/s", "model_path": "m", "status": "pending"}])
    monkeypatch.setattr(resources, "machine_busy", lambda *a, **k: False)
    monkeypatch.setattr(cfgmod, "machine_name", lambda c: "main")
    runner.run_loop(cfg, once=True)
    jobs = read_queue(q.path)
    assert jobs[0]["status"] == "quarantined"  # failed -> retry -> quarantine
    assert jobs[0]["retried"] is True

    cfg["types"]["replay"] = {"command": ["true"]}
    write_queue(q.path, [{"id": "j2", "machine": "any", "type": "replay",
                          "scene": "/s", "model_path": "m", "status": "pending"}])
    runner.run_loop(cfg, once=True)
    assert read_queue(q.path)[0]["status"] == "done"


# -------------------------------------------------------------- config load

def test_load_config_json_and_toml(tmp_path, monkeypatch):
    j = tmp_path / "c.json"
    j.write_text(json.dumps({"queue": {"path": "/q/queue.jsonl"}}))
    cfg = cfgmod.load_config(str(j))
    assert cfgmod.queue_path(cfg) == "/q/queue.jsonl"
    assert cfgmod.log_dir(cfg) == "/q"
    t = tmp_path / "c.toml"
    t.write_text('[queue]\npath = "/q2/queue.jsonl"\n')
    try:
        import tomllib  # noqa: F401
    except ImportError:
        pytest.skip("tomllib unavailable (py3.10)")
    assert cfgmod.queue_path(cfgmod.load_config(str(t))) == "/q2/queue.jsonl"


def test_example_config_parses_and_renders(tmp_path):
    pytest.importorskip("tomllib")
    ex = os.path.join(os.path.dirname(__file__), "..", "examples", "splatograph.toml")
    cfg = cfgmod.load_config(ex)
    assert cfgmod.machine_name(cfg, hostname="cps-wkstn-amd1R9700.local") == "r9700"
    for machine in ("main", "r9700"):
        tcfg = cfgmod.type_cfg(cfg, "replay", machine)
        argv, env, cwd = runner.render_command(tcfg, JOB, machine)
        assert "train.py" in argv
        assert "SCENE_SOURCE=/data/fr3" in argv  # passed as -e in the template
    argv, _, _ = runner.render_command(cfgmod.type_cfg(cfg, "replay", "r9700"),
                                       JOB, "r9700")
    assert argv[0] == "docker" and any("gfx1201" in a for a in argv)
    tcfg = cfgmod.type_cfg(cfg, "bag", "r9700")
    assert tcfg["require_images"]
    _, env, _ = runner.render_command(tcfg, {**JOB, "type": "bag"}, "r9700")
    assert env["CONTAINER_RUNTIME"] == "docker"
    assert env["TRAIN_EXTRA_ARGS"].endswith("--streaming_report_final")


# --------------------------------------------------------------- progress

def test_parse_progress_last_counter_wins():
    tail = ("Streaming training:  10%| | 100/60000 [00:01<...]\r"
            "Streaming training:  89%| | 53570/60000 [27:05<02:53, 37.09it/s]")
    assert progmod.parse_progress(tail) == "iter 53570/60000 (89%)"


def test_parse_progress_no_counter():
    assert progmod.parse_progress("no numbers here") == ""


def test_parse_progress_sentinel_cap_fallback():
    tail = "train:  0%| | 5000/2147483647 [..]"
    extra = "--foo bar --streaming_max_iterations 60000"
    assert progmod.parse_progress(tail, extra) == "iter 5000/60000 (8%)"


def test_parse_progress_sentinel_without_cap():
    tail = "train: 5000/2147483647"
    assert progmod.parse_progress(tail, "--foo bar") == "iter 5000/?"


def test_job_progress_reads_configured_log(tmp_path):
    mp = tmp_path / "run1"
    mp.mkdir()
    (mp / "custom.log").write_text("x" * 5000 + " 42/100 [..]")
    job = {"model_path": "run1", "extra_args": "", "status": "running"}
    out = progmod.job_progress(job, str(tmp_path), {"progress_log": "custom.log"})
    assert out == "iter 42/100 (42%)"
    # default train.log missing -> empty
    assert progmod.job_progress(job, str(tmp_path), {}) == ""
