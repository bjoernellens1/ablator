"""CPU-only tests for ablator (spec expansion, queue, resources, templates)."""
import json
import os
import shutil
import subprocess
import threading
import time
from unittest import mock

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


def test_sample_gpu_mem_pct_reads_configured_budget():
    cfg = {"resources": {"mem_budgets": {"main": {
        "used_path": "/fake/used", "total_path": "/fake/total"}}}}
    reader = lambda p: {"/fake/used": 90, "/fake/total": 100}.get(p)
    assert resources.sample_gpu_mem_pct(cfg, "main", reader=reader) == 90.0
    assert resources.sample_gpu_mem_pct(cfg, "r9700", reader=reader) is None


def test_gpu_mem_busy_threshold(monkeypatch):
    cfg = {"resources": {"mem_dispatch_busy_pct": 70}}
    assert resources.gpu_mem_busy(cfg, "main", mem_sampler=lambda: 75.0) is True
    assert resources.gpu_mem_busy(cfg, "main", mem_sampler=lambda: 69.9) is False
    assert resources.gpu_mem_busy(cfg, "main", mem_sampler=lambda: None) is False


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


def test_tum_sequence_inferred_from_host_scene_path(tmp_path):
    """Command templates mount `scene` at a generic in-container path, which
    defeats scene/readers/tum.py's basename inference of Freiburg intrinsics
    inside the container. render_command must inject --tum_sequence from the
    HOST scene path (still available pre-mount) so this can't silently train
    against wrong (Freiburg1-fallback) intrinsics. Confirmed live: this cost
    fr3par_hybrid_ctrl/gate ~6-8dB, compounding over training."""
    cfg = make_cfg(tmp_path)
    tcfg = cfgmod.type_cfg(cfg, "replay", "main")
    job = {**JOB, "scene": "/data/rgbd_dataset_freiburg3_long_office_household",
           "extra_args": "--opacity_reg 0.001"}
    argv, _, _ = runner.render_command(tcfg, job, "main")
    assert argv[-2:] == ["--tum_sequence", "freiburg3"]


def test_tum_sequence_not_overridden_if_already_set(tmp_path):
    cfg = make_cfg(tmp_path)
    tcfg = cfgmod.type_cfg(cfg, "replay", "main")
    job = {**JOB, "scene": "/data/rgbd_dataset_freiburg3_long_office_household",
           "extra_args": "--tum_sequence freiburg1"}
    argv, _, _ = runner.render_command(tcfg, job, "main")
    assert argv[-2:] == ["--tum_sequence", "freiburg1"]


def test_tum_sequence_short_form_does_not_false_trigger():
    """A generic scene path basename like "fr3" (used as a test fixture
    elsewhere in this suite, not an actual TUM path) must not be mistaken
    for a Freiburg3 sequence — only the unambiguous "freiburgN" substring
    counts."""
    from ablator.runner import _infer_tum_sequence
    assert _infer_tum_sequence("/data/fr3", "--foo bar") == "--foo bar"


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


def test_pytorch_generic_example_config_parses_and_builds_k8s_manifest(tmp_path):
    """examples/pytorch-generic.toml proves the k8s dispatch path is
    workload-agnostic: an arbitrary PyTorch job (no scene/opacity_reg/
    Gaussian-splatting fields anywhere) renders a command and a valid k8s
    Job manifest through the exact same code path splatograph uses."""
    pytest.importorskip("tomllib")
    ex = os.path.join(os.path.dirname(__file__), "..", "examples",
                      "pytorch-generic.toml")
    cfg = cfgmod.load_config(ex)
    mcfg = cfgmod.machine_cfg(cfg, "a100cluster")
    assert mcfg["backend"] == "k8s"

    job = {"id": "generic_ctrl", "scene": "", "model_path": "output/generic_ctrl",
          "extra_args": "--lr 0.001", "iterations": 10}
    tcfg = cfgmod.type_cfg(cfg, "train", "a100cluster")
    argv, env, cwd = runner.render_command(tcfg, job, "a100cluster")
    assert argv[:2] == ["python", "train.py"]
    assert "--lr" in argv and "0.001" in argv
    assert cwd == "/workspace"

    manifest = runner.build_k8s_job_manifest(mcfg, job, argv, cwd)
    pod_spec = manifest["spec"]["template"]["spec"]
    assert manifest["metadata"]["namespace"] == "cps-users"
    assert manifest["metadata"]["labels"]["kai.scheduler/queue"] == "batch"
    assert pod_spec["priorityClassName"] == "kai-batch-low"
    assert pod_spec["schedulerName"] == "kai-scheduler"
    assert "imagePullSecrets" not in pod_spec  # public image, none configured
    assert pod_spec["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] == "1"
    volume_names = {v["name"] for v in pod_spec["volumes"]}
    assert volume_names == {"checkpoints"}  # extra_volumes only, no dataset/scratch PVC

    # Dispatcher-identity pitfall: this runner's own resolved identity must
    # be the bare-metal "local" fallback, NOT "a100cluster" itself -- if it
    # resolved to the k8s machine, jobs pinned to a100cluster would be
    # claimed through the serial bare-metal path instead of run_loop's
    # concurrent, thread-pooled k8s dispatch (see [machines.local]'s
    # comment in the example config).
    assert cfgmod.machine_name(cfg, hostname="some-random-laptop") == "local"


def test_k8s_manifest_git_sync_ssh_secret_unaffected_by_http_option():
    """SSH path (git_sync_secret_name) must render exactly as before -- the
    new HTTPS-token option is purely additive."""
    mcfg = {"namespace": "ns", "kai_queue": "q", "priority_class": "p",
            "image": "img:tag", "git_sync_repo_url": "https://github.com/o/r",
            "git_sync_secret_name": "deploy-key"}
    job = {"id": "j1", "model_path": "output/j1"}
    manifest = runner.build_k8s_job_manifest(mcfg, job, ["python", "x.py"], "/workspace", "abc123")
    init = manifest["spec"]["template"]["spec"]["initContainers"][0]
    assert init["env"] == [{"name": "GIT_SSH_COMMAND",
                            "value": "ssh -i /etc/git-creds/ssh-privatekey "
                                     "-o StrictHostKeyChecking=no -o IdentitiesOnly=yes"}]
    assert "git remote add origin \"https://github.com/o/r\"" in init["command"][2]


def test_k8s_manifest_git_sync_http_token_secret():
    """git_sync_http_secret_name rewrites the remote URL to embed the token
    from a mounted secret file, and mounts that secret read-only into the
    init container only -- no GIT_SSH_COMMAND env var (that's the SSH path's
    mechanism, not this one)."""
    mcfg = {"namespace": "ns", "kai_queue": "q", "priority_class": "p",
            "image": "img:tag", "git_sync_repo_url": "https://github.com/o/r",
            "git_sync_http_secret_name": "gh-token"}
    job = {"id": "j1", "model_path": "output/j1"}
    manifest = runner.build_k8s_job_manifest(mcfg, job, ["python", "x.py"], "/workspace", "abc123")
    pod_spec = manifest["spec"]["template"]["spec"]
    init = pod_spec["initContainers"][0]
    assert "env" not in init
    creds_volumes = [v for v in pod_spec["volumes"] if v["name"] == "git-creds"]
    assert creds_volumes == [{"name": "git-creds",
                              "secret": {"secretName": "gh-token", "defaultMode": 0o400}}]
    assert {"name": "git-creds", "mountPath": "/etc/git-creds", "readOnly": True} in init["volumeMounts"]
    # trainer container never gets the git-creds mount
    trainer_mounts = pod_spec["containers"][0]["volumeMounts"]
    assert not any(m["name"] == "git-creds" for m in trainer_mounts)
    script = init["command"][2]
    assert 'https://x-access-token:$(cat /etc/git-creds/token)@github.com/o/r' in script
    # the ORIGINAL (credential-free) URL must still appear in the echo, for
    # readable logs -- the token must never be echoed
    assert "checked out abc123 from https://github.com/o/r" in script
    assert "x-access-token" not in script.split("echo")[1]


def test_k8s_manifest_omits_mps_wiring_by_default():
    """Byte-identical manifest (no mps-root volume/env/affinity) for every
    machine that hasn't set mcfg["mps"] = true -- the opt-in must be a pure
    no-op otherwise."""
    mcfg = {"namespace": "ns", "kai_queue": "q", "priority_class": "p",
            "image": "img:tag"}
    job = {"id": "j1", "model_path": "output/j1"}
    manifest = runner.build_k8s_job_manifest(mcfg, job, ["python", "x.py"], "/workspace")
    pod_spec = manifest["spec"]["template"]["spec"]
    volume_names = {v["name"] for v in pod_spec["volumes"]}
    assert "mps-root" not in volume_names
    assert "affinity" not in pod_spec
    assert "env" not in pod_spec["containers"][0]


def test_k8s_manifest_adds_mps_wiring_when_enabled():
    """mcfg["mps"] = true adds the hostPath volume + mount, the
    CUDA_MPS_PIPE_DIRECTORY/CUDA_MPS_LOG_DIRECTORY env vars, and soft
    anti-affinity against other ablator-job pods -- all three parts of the
    documented fix for the Exclusive_Process/no-MPS-arbitration cluster GPU
    issue (see docs/EVALUATION_RESULTS.md in semantic-gaussian-particles,
    "CUDA path verified for real: A100 cluster dispatch, MPS wiring
    required")."""
    mcfg = {"namespace": "ns", "kai_queue": "q", "priority_class": "p",
            "image": "img:tag", "mps": True}
    job = {"id": "j1", "model_path": "output/j1"}
    manifest = runner.build_k8s_job_manifest(mcfg, job, ["python", "x.py"], "/workspace")
    pod_spec = manifest["spec"]["template"]["spec"]

    mps_volumes = [v for v in pod_spec["volumes"] if v["name"] == "mps-root"]
    assert len(mps_volumes) == 1
    assert mps_volumes[0]["hostPath"] == {"path": "/run/nvidia/mps", "type": "DirectoryOrCreate"}

    trainer = pod_spec["containers"][0]
    mps_mounts = [m for m in trainer["volumeMounts"] if m["name"] == "mps-root"]
    assert mps_mounts == [{"name": "mps-root", "mountPath": "/mps"}]

    env_by_name = {e["name"]: e["value"] for e in trainer["env"]}
    assert env_by_name["CUDA_MPS_PIPE_DIRECTORY"] == "/mps/nvidia.com/gpu/pipe"
    assert env_by_name["CUDA_MPS_LOG_DIRECTORY"] == "/mps/nvidia.com/gpu/log"

    anti_affinity_terms = pod_spec["affinity"]["podAntiAffinity"][
        "preferredDuringSchedulingIgnoredDuringExecution"]
    assert len(anti_affinity_terms) == 1
    label_selector = anti_affinity_terms[0]["podAffinityTerm"]["labelSelector"]
    assert label_selector["matchExpressions"] == [
        {"key": "app", "operator": "In", "values": ["ablator-job"]}]
    assert anti_affinity_terms[0]["podAffinityTerm"]["topologyKey"] == "kubernetes.io/hostname"


def test_k8s_manifest_mps_wiring_coexists_with_extra_volumes():
    """MPS's hostPath volume must not clobber or be clobbered by
    extra_volumes' PVC mounts -- both should be present."""
    mcfg = {"namespace": "ns", "kai_queue": "q", "priority_class": "p",
            "image": "img:tag", "mps": True,
            "extra_volumes": [{"name": "checkpoints", "claim_name": "ckpt-pvc",
                               "mount_path": "/mnt/checkpoints"}]}
    job = {"id": "j1", "model_path": "output/j1"}
    manifest = runner.build_k8s_job_manifest(mcfg, job, ["python", "x.py"], "/workspace")
    pod_spec = manifest["spec"]["template"]["spec"]
    volume_names = {v["name"] for v in pod_spec["volumes"]}
    assert volume_names == {"mps-root", "checkpoints"}


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
    tail = "train: 5000/2147483647 [..]"
    assert progmod.parse_progress(tail, "--foo bar") == "iter 5000/?"


def test_parse_progress_ignores_log_timestamp_after_last_tqdm_line():
    # Regression: "[07/07 18:02:47]" (this project's log-timestamp format,
    # MM/DD) also matches bare "(\d+)/(\d+)". A save/report-phase line
    # carrying a timestamp landing in the tail AFTER the last real tqdm
    # progress line must not be picked up as the iteration counter.
    tail = (
        "Streaming training:   3%|2 | 2620/100000 [11:10<8:33:28, 3.16it/s]"
        "\n[FRAME 5250] Saving Gaussians (iter=2623) [07/07 18:02:55]\n"
    )
    assert progmod.parse_progress(tail) == "iter 2620/100000 (2%)"


def test_parse_progress_ignores_report_line_view_count_after_last_tqdm_line():
    # Regression: this project's periodic report lines, e.g.
    # "[report] iter=7000 train PSNR mean=... (128/323 views)", also match
    # bare "(\d+)/(\d+)" and can land in the tail after the last real tqdm
    # line during a mid-training report pause. Same failure mode as the
    # timestamp case above, found live immediately after that fix landed.
    tail = (
        "Streaming training:   5%|5 | 5000/98000 [10:20<1:14:49, 20.72it/s]"
        "\n[report] iter=7000 train PSNR mean=20.64dB min=15.27 max=26.47 "
        "(128/323 views) [07/07 18:20:05]\n"
    )
    assert progmod.parse_progress(tail) == "iter 5000/98000 (5%)"


def test_job_progress_reads_configured_log(tmp_path):
    mp = tmp_path / "run1"
    mp.mkdir()
    (mp / "custom.log").write_text("x" * 5000 + " 42/100 [..]")
    job = {"model_path": "run1", "extra_args": "", "status": "running"}
    out = progmod.job_progress(job, str(tmp_path), {"progress_log": "custom.log"})
    assert out == "iter 42/100 (42%)"
    # default train.log missing -> empty
    assert progmod.job_progress(job, str(tmp_path), {}) == ""


# ------------------------------------------------------- stale-running reconciliation

def test_reconcile_marks_done_when_completion_artifact_present(tmp_path):
    cfg = make_cfg(tmp_path)
    mp = tmp_path / "run_done"
    (mp / "comparison" / "iter_1000").mkdir(parents=True)
    (mp / "comparison" / "iter_1000" / "report.json").write_text("{}")
    q = Queue(cfg["queue"]["path"])
    write_queue(q.path, [{"id": "j1", "type": "replay", "model_path": str(mp),
                          "status": "running", "claimed_by": "main"}])
    runner.reconcile_stale_running(cfg, "main", q, busy=False)
    job = read_queue(q.path)[0]
    assert job["status"] == "done"
    assert job["reconciled"] is True


def test_reconcile_requeues_when_no_artifact_and_no_process(tmp_path):
    cfg = make_cfg(tmp_path)
    mp = tmp_path / "run_orphaned"
    mp.mkdir()
    q = Queue(cfg["queue"]["path"])
    write_queue(q.path, [{"id": "j1", "type": "replay", "model_path": str(mp),
                          "status": "running", "claimed_by": "main"}])
    runner.reconcile_stale_running(cfg, "main", q, busy=False)
    job = read_queue(q.path)[0]
    assert job["status"] == "pending"
    assert job["claimed_by"] is None
    assert job["reconciled"] is True


def test_reconcile_skips_while_machine_busy(tmp_path):
    """A still-running process could belong to the stuck job; never touch
    it (and risk a duplicate launch) while busy-guards say something is
    still actually executing."""
    cfg = make_cfg(tmp_path)
    q = Queue(cfg["queue"]["path"])
    write_queue(q.path, [{"id": "j1", "type": "replay", "model_path": "nope",
                          "status": "running", "claimed_by": "main"}])
    runner.reconcile_stale_running(cfg, "main", q, busy=True)
    assert read_queue(q.path)[0]["status"] == "running"


def test_reconcile_ignores_other_machines_claims(tmp_path):
    cfg = make_cfg(tmp_path)
    q = Queue(cfg["queue"]["path"])
    write_queue(q.path, [{"id": "j1", "type": "replay", "model_path": "nope",
                          "status": "running", "claimed_by": "r9700"}])
    runner.reconcile_stale_running(cfg, "main", q, busy=False)
    assert read_queue(q.path)[0]["status"] == "running"
    assert read_queue(q.path)[0].get("claimed_by") == "r9700"


def test_run_loop_reconciles_orphan_that_was_busy_at_startup(tmp_path, monkeypatch):
    """Regression for the 2026-07-07 live incident: a runner restart's
    startup-only reconcile_stale_running() call correctly deferred (the
    prior process's job was still genuinely training, busy=True), but
    nothing ever retried once the machine went idle moments later --
    frdeskw01main_fr1desk_w01_plus_admission_fix stayed stuck at
    status="running" for 2h45m with no live supervisor. run_loop() must
    re-run reconciliation on every idle tick, not just once at startup."""
    cfg = make_cfg(tmp_path)
    cfg["queue"]["log_dir"] = str(tmp_path)
    mp = tmp_path / "run_done"
    (mp / "comparison" / "iter_1000").mkdir(parents=True)
    (mp / "comparison" / "iter_1000" / "report.json").write_text("{}")
    q = Queue(cfg["queue"]["path"])
    write_queue(q.path, [{"id": "j1", "type": "replay", "model_path": str(mp),
                          "status": "running", "claimed_by": "main"}])
    monkeypatch.setattr(cfgmod, "machine_name", lambda c: "main")
    # busy=True at the moment run_loop() calls its own STARTUP reconcile
    # (mirrors the prior process's job still genuinely training at restart
    # time); by the time the loop's first idle-tick check runs, it's gone
    # idle -- exactly the race that orphaned the real job.
    busy_calls = {"n": 0}
    def fake_busy(*a, **k):
        busy_calls["n"] += 1
        return busy_calls["n"] == 1
    monkeypatch.setattr(resources, "machine_busy", fake_busy)
    runner.run_loop(cfg, once=True)
    job = read_queue(q.path)[0]
    assert job["status"] == "done"
    assert job["reconciled"] is True


# ------------------------------------------------------ hard completion validity

def test_exit_zero_without_artifact_is_failed_when_required(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    cfg["queue"]["log_dir"] = str(tmp_path)
    mp = tmp_path / "run_no_report"
    cfg["types"]["replay"] = {"cwd": str(tmp_path), "command": ["true"],
                              "require_result_artifact": True}
    q = Queue(cfg["queue"]["path"])
    write_queue(q.path, [{"id": "j1", "machine": "any", "type": "replay",
                          "scene": "/s", "model_path": str(mp),
                          "status": "pending"}])
    monkeypatch.setattr(resources, "machine_busy", lambda *a, **k: False)
    monkeypatch.setattr(cfgmod, "machine_name", lambda c: "main")
    runner.run_loop(cfg, once=True)
    assert read_queue(q.path)[0]["status"] == "quarantined"  # failed -> retry -> quarantine


def test_exit_zero_with_artifact_is_done_when_required(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    cfg["queue"]["log_dir"] = str(tmp_path)
    mp = tmp_path / "run_with_report"
    (mp / "comparison" / "iter_1000").mkdir(parents=True)
    (mp / "comparison" / "iter_1000" / "report.json").write_text("{}")
    cfg["types"]["replay"] = {"cwd": str(tmp_path), "command": ["true"],
                              "require_result_artifact": True}
    q = Queue(cfg["queue"]["path"])
    write_queue(q.path, [{"id": "j1", "machine": "any", "type": "replay",
                          "scene": "/s", "model_path": str(mp),
                          "status": "pending"}])
    monkeypatch.setattr(resources, "machine_busy", lambda *a, **k: False)
    monkeypatch.setattr(cfgmod, "machine_name", lambda c: "main")
    runner.run_loop(cfg, once=True)
    assert read_queue(q.path)[0]["status"] == "done"


def test_require_result_artifact_off_by_default(tmp_path, monkeypatch):
    """Backward compat: unset -> exit code 0 alone is still 'done'."""
    cfg = make_cfg(tmp_path)
    cfg["queue"]["log_dir"] = str(tmp_path)
    mp = tmp_path / "run_no_report2"
    cfg["types"]["replay"] = {"cwd": str(tmp_path), "command": ["true"]}
    q = Queue(cfg["queue"]["path"])
    write_queue(q.path, [{"id": "j1", "machine": "any", "type": "replay",
                          "scene": "/s", "model_path": str(mp),
                          "status": "pending"}])
    monkeypatch.setattr(resources, "machine_busy", lambda *a, **k: False)
    monkeypatch.setattr(cfgmod, "machine_name", lambda c: "main")
    runner.run_loop(cfg, once=True)
    assert read_queue(q.path)[0]["status"] == "done"


def test_control_requeue_never_yields_done_regardless_of_exit_code(tmp_path):
    """A manual control action (stop/skip/requeue), even against a process
    that goes on to exit 0, must never be reported as 'done' — supervise()
    returns the CONTROL_STATUS override, and run_job() returns that
    override directly without ever consulting the exit code."""
    cfg = make_cfg(tmp_path)
    cfg["queue"]["log_dir"] = str(tmp_path)
    cfg["types"]["replay"] = {"cwd": str(tmp_path), "command": ["sleep", "0.2"]}
    job = {"id": "j1", "type": "replay", "model_path": "m", "status": "running"}

    # Exercise run_job with a stubbed supervise() that simulates a control
    # action firing while the underlying process goes on to exit 0.
    import subprocess as sp
    real_popen = sp.Popen

    def fake_popen(argv, **kw):
        return real_popen(["sleep", "0.05"], **kw)

    for action, expected in runner.CONTROL_STATUS.items():
        sp.Popen = fake_popen
        orig_supervise = runner.supervise
        runner.supervise = lambda *a, **k: expected
        try:
            status, rc = runner.run_job(cfg, job, "main", q=None)
        finally:
            runner.supervise = orig_supervise
            sp.Popen = real_popen
        assert status == expected
        assert status != "done"


# --------------------------------------------------------------- pause/unpause CLI

def test_pause_cli_roundtrips_with_unpause(tmp_path, capsys):
    cfg = make_cfg(tmp_path)
    from ablator.queue import is_paused
    assert not is_paused(cfg["queue"]["path"], "main")
    cli.cmd_pause(cfg, "main")
    assert is_paused(cfg["queue"]["path"], "main")
    q = Queue(cfg["queue"]["path"])
    write_queue(q.path, [{"id": "j1", "machine": "any", "type": "replay",
                          "status": "pending"}])
    assert q.claim_next("main") is None  # paused -> no new claims
    cli.cmd_unpause(cfg, "main")
    assert not is_paused(cfg["queue"]["path"], "main")
    assert q.claim_next("main")["id"] == "j1"  # claims resume


def test_pause_cli_requires_machine_arg(tmp_path):
    cfg = make_cfg(tmp_path)
    with pytest.raises(SystemExit):
        cli.cmd_pause(cfg, None)


# --------------------------------------------------- concurrent k8s dispatch

def make_k8s_cfg(tmp_path, max_concurrent=2):
    cfg = make_cfg(tmp_path)
    cfg["queue"]["log_dir"] = str(tmp_path)
    cfg["machines"]["a100cluster"] = {"backend": "k8s",
                                      "max_concurrent": max_concurrent}
    return cfg


def _k8s_jobs(n, prefix="kjob"):
    return [{"id": f"{prefix}{i}", "machine": "a100cluster", "type": "replay",
            "scene": "/s", "model_path": f"m{i}", "status": "pending"}
           for i in range(n)]


def test_k8s_jobs_dispatched_concurrently_up_to_cap(tmp_path, monkeypatch):
    """Multiple k8s-targeted jobs run overlapping in time, not serially."""
    cfg = make_k8s_cfg(tmp_path, max_concurrent=3)
    q = Queue(cfg["queue"]["path"])
    write_queue(q.path, _k8s_jobs(3))
    monkeypatch.setattr(resources, "machine_busy", lambda *a, **k: False)
    monkeypatch.setattr(cfgmod, "machine_name", lambda c: "main")

    lock = threading.Lock()
    state = {"concurrent": 0, "max_concurrent": 0}

    def fake_run_job_k8s(cfg, job, machine, mcfg, q=None):
        with lock:
            state["concurrent"] += 1
            state["max_concurrent"] = max(state["max_concurrent"], state["concurrent"])
        time.sleep(0.2)
        with lock:
            state["concurrent"] -= 1
        return "done", 0

    monkeypatch.setattr(runner, "run_job_k8s", fake_run_job_k8s)
    runner.run_loop(cfg, once=True)

    # The real assertion is about overlap in run_job_k8s calls, not overall
    # wall-clock (three threads finishing ~simultaneously then all racing
    # for the queue file's flock to record "done" adds its own — unrelated
    # — contention overhead on top of the concurrent dispatch itself).
    assert state["max_concurrent"] >= 2, "jobs never overlapped in time"
    jobs = read_queue(q.path)
    assert all(j["status"] == "done" for j in jobs)


def test_k8s_dispatch_enabled_default_true(tmp_path):
    """Default (no k8s_dispatch key) preserves pre-existing behavior for a
    machine like main that genuinely has cluster access."""
    cfg = make_k8s_cfg(tmp_path)
    cfg["machines"]["main"] = {}
    with mock.patch("shutil.which", return_value="/usr/bin/kubectl"):
        assert runner.k8s_dispatch_enabled(cfg, "main") is True


def test_k8s_dispatch_enabled_false_when_configured_off(tmp_path):
    """k8s_dispatch = false in config disables k8s dispatch even though
    kubectl is present and k8s machines exist in cfg["machines"]."""
    cfg = make_k8s_cfg(tmp_path)
    cfg["machines"]["r9700"] = {"k8s_dispatch": False}
    with mock.patch("shutil.which", return_value="/usr/bin/kubectl"):
        assert runner.k8s_dispatch_enabled(cfg, "r9700") is False


def test_k8s_dispatch_enabled_false_when_kubectl_missing(tmp_path):
    """Missing kubectl binary disables k8s dispatch regardless of the
    config flag's value -- defense in depth against crashing the whole
    runner process on a machine without kubectl installed."""
    cfg = make_k8s_cfg(tmp_path)
    cfg["machines"]["main"] = {}  # k8s_dispatch defaults True
    with mock.patch("shutil.which", return_value=None):
        assert runner.k8s_dispatch_enabled(cfg, "main") is False


def test_dispatch_machines_excludes_k8s_when_disabled(tmp_path, monkeypatch):
    """run_loop must build a bare-metal-only dispatch_machines list (and
    never touch any k8s codepath) when k8s_dispatch is disabled for this
    machine, even though a k8s machine exists in cfg["machines"]."""
    cfg = make_k8s_cfg(tmp_path)
    cfg["machines"]["r9700"] = {"k8s_dispatch": False}
    cfg["types"]["replay"] = {"cwd": str(tmp_path), "command": ["true"]}
    q = Queue(cfg["queue"]["path"])
    write_queue(q.path, [
        {"id": "baremetal1", "machine": "r9700", "type": "replay",
         "scene": "/s", "model_path": "m", "status": "pending"},
    ])
    monkeypatch.setattr(resources, "machine_busy", lambda *a, **k: False)
    monkeypatch.setattr(cfgmod, "machine_name", lambda c: "r9700")

    def boom(*a, **k):
        raise AssertionError("kubectl must never be invoked when k8s_dispatch=false")

    monkeypatch.setattr(runner, "_kubectl", boom)
    monkeypatch.setattr(runner, "run_job_k8s", boom)

    runner.run_loop(cfg, once=True)

    jobs = read_queue(q.path)
    assert jobs[0]["status"] == "done"


def test_dispatch_machines_excludes_k8s_when_kubectl_missing(tmp_path, monkeypatch):
    """Even with k8s_dispatch left at its True default, a missing kubectl
    binary must degrade gracefully to bare-metal-only dispatch instead of
    crashing the runner."""
    cfg = make_k8s_cfg(tmp_path)
    cfg["types"]["replay"] = {"cwd": str(tmp_path), "command": ["true"]}
    q = Queue(cfg["queue"]["path"])
    write_queue(q.path, [
        {"id": "baremetal1", "machine": "main", "type": "replay",
         "scene": "/s", "model_path": "m", "status": "pending"},
    ])
    monkeypatch.setattr(resources, "machine_busy", lambda *a, **k: False)
    monkeypatch.setattr(cfgmod, "machine_name", lambda c: "main")
    monkeypatch.setattr(shutil, "which", lambda name: None)

    def boom(*a, **k):
        raise AssertionError("kubectl must never be invoked when the binary is missing")

    monkeypatch.setattr(runner, "_kubectl", boom)
    monkeypatch.setattr(runner, "run_job_k8s", boom)

    runner.run_loop(cfg, once=True)

    jobs = read_queue(q.path)
    assert jobs[0]["status"] == "done"


def test_bare_metal_job_not_blocked_by_inflight_k8s(tmp_path, monkeypatch):
    """A bare-metal job must be claimed/run promptly even while a slow k8s
    job is in flight — the bare-metal path must never wait on it."""
    cfg = make_k8s_cfg(tmp_path, max_concurrent=2)
    cfg["types"]["replay"] = {"cwd": str(tmp_path), "command": ["true"]}
    q = Queue(cfg["queue"]["path"])
    write_queue(q.path, _k8s_jobs(1) + [
        {"id": "baremetal1", "machine": "any", "type": "replay",
         "scene": "/s", "model_path": "m", "status": "pending"},
    ])
    monkeypatch.setattr(resources, "machine_busy", lambda *a, **k: False)
    monkeypatch.setattr(cfgmod, "machine_name", lambda c: "main")

    bare_metal_started_at = {}

    def fake_run_job_k8s(cfg, job, machine, mcfg, q=None):
        time.sleep(0.3)
        return "done", 0

    real_run_job = runner.run_job

    def spying_run_job(cfg, job, machine, q=None):
        if job["id"] == "baremetal1":
            bare_metal_started_at["t"] = time.monotonic()
        return real_run_job(cfg, job, machine, q)

    monkeypatch.setattr(runner, "run_job_k8s", fake_run_job_k8s)
    monkeypatch.setattr(runner, "run_job", spying_run_job)
    t0 = time.monotonic()
    runner.run_loop(cfg, once=True)

    assert "t" in bare_metal_started_at
    # The bare-metal job must have started almost immediately, well before
    # the 0.3s the k8s job takes to "finish" in the background.
    assert bare_metal_started_at["t"] - t0 < 0.15
    jobs = {j["id"]: j for j in read_queue(q.path)}
    assert jobs["baremetal1"]["status"] == "done"
    assert jobs["kjob0"]["status"] == "done"  # join_all() waits before once=True returns


def test_k8s_concurrency_cap_respected_and_resumes(tmp_path, monkeypatch):
    """Only `max_concurrent` k8s jobs are claimed per pass; the remainder
    stays pending and is picked up once slots free on the next pass."""
    cfg = make_k8s_cfg(tmp_path, max_concurrent=2)
    q = Queue(cfg["queue"]["path"])
    write_queue(q.path, _k8s_jobs(3))
    monkeypatch.setattr(resources, "machine_busy", lambda *a, **k: False)
    monkeypatch.setattr(cfgmod, "machine_name", lambda c: "main")

    seen_concurrent = []
    lock = threading.Lock()
    state = {"concurrent": 0}

    def fake_run_job_k8s(cfg, job, machine, mcfg, q=None):
        with lock:
            state["concurrent"] += 1
            seen_concurrent.append(state["concurrent"])
        time.sleep(0.1)
        with lock:
            state["concurrent"] -= 1
        return "done", 0

    monkeypatch.setattr(runner, "run_job_k8s", fake_run_job_k8s)
    # once=True joins in-flight threads before returning, so a single pass
    # only ever claims up to the cap (2 of the 3 pending jobs); the 3rd is
    # picked up by dispatching once more.
    runner.run_loop(cfg, once=True)
    assert max(seen_concurrent) <= 2
    jobs = read_queue(q.path)
    done = [j for j in jobs if j["status"] == "done"]
    pending = [j for j in jobs if j["status"] == "pending"]
    assert len(done) == 2
    assert len(pending) == 1

    runner.run_loop(cfg, once=True)  # resumes: claims the remaining job
    jobs = read_queue(q.path)
    assert all(j["status"] == "done" for j in jobs)


def test_concurrent_k8s_job_failure_bookkeeping(tmp_path, monkeypatch):
    """A concurrently-dispatched k8s job that fails still goes through the
    normal retry-once-then-quarantine disposition logic."""
    cfg = make_k8s_cfg(tmp_path, max_concurrent=2)
    q = Queue(cfg["queue"]["path"])
    write_queue(q.path, _k8s_jobs(1))
    monkeypatch.setattr(resources, "machine_busy", lambda *a, **k: False)
    monkeypatch.setattr(cfgmod, "machine_name", lambda c: "main")

    def fake_run_job_k8s(cfg, job, machine, mcfg, q=None):
        return "failed", 1

    monkeypatch.setattr(runner, "run_job_k8s", fake_run_job_k8s)
    runner.run_loop(cfg, once=True)
    jobs = read_queue(q.path)
    assert jobs[0]["status"] == "quarantined"
    assert jobs[0]["retried"] is True


# --------------------------------------------- k8s restart-recovery reconcile

def make_k8s_reconcile_cfg(tmp_path, max_concurrent=2):
    cfg = make_k8s_cfg(tmp_path, max_concurrent=max_concurrent)
    cfg["machines"]["a100cluster"]["namespace"] = "jupyterhub"
    return cfg


def _running_k8s_job(job_id="kjob0", model_path="m0"):
    return {"id": job_id, "machine": "a100cluster", "type": "replay",
           "scene": "/s", "model_path": model_path, "status": "running",
           "claimed_by": "a100cluster"}


def test_reconcile_k8s_reattaches_genuinely_running_job(tmp_path, monkeypatch):
    """A k8s Job that is still genuinely Running on the cluster after a
    runner restart must be re-attached (polling resumed), never requeued
    or marked crashed."""
    cfg = make_k8s_reconcile_cfg(tmp_path)
    q = Queue(cfg["queue"]["path"])
    write_queue(q.path, [_running_k8s_job()])

    def fake_kubectl(args, input_text=None, timeout=None):
        assert args[:2] == ["get", "job"]
        return subprocess.CompletedProcess(args, 0,
            stdout=json.dumps({"status": {}}), stderr="")

    release = threading.Event()

    def fake_poll(cfg, job, machine, mcfg, tcfg, name, ns, log_path, append=False):
        release.wait(timeout=5)
        return "done", 0

    monkeypatch.setattr(runner, "_kubectl", fake_kubectl)
    monkeypatch.setattr(runner, "_poll_k8s_job", fake_poll)

    inflight = runner._K8sInflight()
    runner.reconcile_stale_running(cfg, "a100cluster", q, busy=False,
                                   inflight=inflight)

    # Not touched synchronously: still "running", not pending/crashed.
    job = read_queue(q.path)[0]
    assert job["status"] == "running"
    assert inflight.count("a100cluster") == 1

    release.set()
    inflight.join_all()
    job = read_queue(q.path)[0]
    assert job["status"] == "done"


def test_reconcile_k8s_succeeded_while_down_with_artifact_marks_done(tmp_path, monkeypatch):
    """The k8s Job reached 'succeeded' while the runner process was down --
    a real success, not a crash -- and a completion artifact exists."""
    cfg = make_k8s_reconcile_cfg(tmp_path)
    mp = tmp_path / "run_done_k8s"
    (mp / "comparison" / "iter_1000").mkdir(parents=True)
    (mp / "comparison" / "iter_1000" / "report.json").write_text("{}")
    q = Queue(cfg["queue"]["path"])
    write_queue(q.path, [_running_k8s_job(model_path=str(mp))])

    def fake_kubectl(args, input_text=None, timeout=None):
        return subprocess.CompletedProcess(args, 0,
            stdout=json.dumps({"status": {"succeeded": 1}}), stderr="")

    monkeypatch.setattr(runner, "_kubectl", fake_kubectl)
    inflight = runner._K8sInflight()
    runner.reconcile_stale_running(cfg, "a100cluster", q, busy=False,
                                   inflight=inflight)

    job = read_queue(q.path)[0]
    assert job["status"] == "done"
    assert job["reconciled"] is True
    assert inflight.count("a100cluster") == 0  # no thread spawned, terminal


def test_reconcile_k8s_succeeded_while_down_without_artifact_requeues(tmp_path, monkeypatch):
    """Succeeded while down but no completion artifact -- a genuine crash
    (or a wrapper script swallowing a real failure), matches
    require_result_artifact semantics used elsewhere: not 'done'."""
    cfg = make_k8s_reconcile_cfg(tmp_path)
    mp = tmp_path / "run_no_artifact_k8s"
    mp.mkdir()
    q = Queue(cfg["queue"]["path"])
    write_queue(q.path, [_running_k8s_job(model_path=str(mp))])

    def fake_kubectl(args, input_text=None, timeout=None):
        return subprocess.CompletedProcess(args, 0,
            stdout=json.dumps({"status": {"succeeded": 1}}), stderr="")

    monkeypatch.setattr(runner, "_kubectl", fake_kubectl)
    inflight = runner._K8sInflight()
    runner.reconcile_stale_running(cfg, "a100cluster", q, busy=False,
                                   inflight=inflight)

    job = read_queue(q.path)[0]
    assert job["status"] == "pending"
    assert job["claimed_by"] is None
    assert job["reconciled"] is True


def test_reconcile_k8s_job_gone_requeues(tmp_path, monkeypatch):
    """The k8s Job no longer exists on the cluster at all (real crash while
    the runner was down, or manually deleted) -- existing crashed/requeue
    behavior is preserved."""
    cfg = make_k8s_reconcile_cfg(tmp_path)
    mp = tmp_path / "run_gone_k8s"
    mp.mkdir()
    q = Queue(cfg["queue"]["path"])
    write_queue(q.path, [_running_k8s_job(model_path=str(mp))])

    def fake_kubectl(args, input_text=None, timeout=None):
        return subprocess.CompletedProcess(args, 1, stdout="", stderr="not found")

    monkeypatch.setattr(runner, "_kubectl", fake_kubectl)
    inflight = runner._K8sInflight()
    runner.reconcile_stale_running(cfg, "a100cluster", q, busy=False,
                                   inflight=inflight)

    job = read_queue(q.path)[0]
    assert job["status"] == "pending"
    assert job["claimed_by"] is None
    assert job["reconciled"] is True


def test_reconcile_k8s_reattach_counts_toward_max_concurrent(tmp_path, monkeypatch):
    """A re-attached job must occupy an in-flight slot so a subsequent
    dispatch pass correctly sees reduced (not full) spare capacity."""
    cfg = make_k8s_reconcile_cfg(tmp_path, max_concurrent=1)
    q = Queue(cfg["queue"]["path"])
    write_queue(q.path, [_running_k8s_job()] + _k8s_jobs(1, prefix="waiting"))

    def fake_kubectl(args, input_text=None, timeout=None):
        return subprocess.CompletedProcess(args, 0,
            stdout=json.dumps({"status": {}}), stderr="")

    release = threading.Event()

    def fake_poll(cfg, job, machine, mcfg, tcfg, name, ns, log_path, append=False):
        release.wait(timeout=5)
        return "done", 0

    monkeypatch.setattr(runner, "_kubectl", fake_kubectl)
    monkeypatch.setattr(runner, "_poll_k8s_job", fake_poll)

    inflight = runner._K8sInflight()
    runner.reconcile_stale_running(cfg, "a100cluster", q, busy=False,
                                   inflight=inflight)

    cap = runner._k8s_max_concurrent(cfg, "a100cluster")
    assert cap == 1
    assert inflight.count("a100cluster") == 1
    # At cap already -- a dispatch pass must not claim the waiting job.
    assert inflight.count("a100cluster") >= cap

    release.set()
    inflight.join_all()


# ------------------------------------------------- "any" job dispatch fairness

def test_any_job_deferred_from_k8s_when_other_baremetal_idle(tmp_path, monkeypatch):
    """When another bare-metal machine (r9700) reports a fresh idle
    heartbeat, this process's k8s-fill loop must not vacuum up a
    machine="any" job that r9700 would otherwise claim for itself on its
    own next tick — only jobs explicitly pinned to the k8s machine may
    still be claimed."""
    cfg = make_k8s_cfg(tmp_path, max_concurrent=2)
    cfg["types"]["replay"] = {"cwd": str(tmp_path), "command": ["true"]}
    q = Queue(cfg["queue"]["path"])
    write_queue(q.path, [
        {"id": "any1", "machine": "any", "type": "replay",
         "scene": "/s", "model_path": "m1", "status": "pending"},
        {"id": "any2", "machine": "any", "type": "replay",
         "scene": "/s", "model_path": "m2", "status": "pending"},
    ])
    monkeypatch.setattr(resources, "machine_busy", lambda *a, **k: False)
    monkeypatch.setattr(cfgmod, "machine_name", lambda c: "main")
    monkeypatch.setattr(runner, "run_job", lambda cfg, job, machine, q=None: ("done", 0))
    monkeypatch.setattr(runner, "run_job_k8s",
                        lambda cfg, job, machine, mcfg, q=None: ("done", 0))

    # r9700 looks idle-with-capacity right now.
    runner.write_heartbeat(cfg, "r9700", "idle k8s_inflight=0")

    runner.run_loop(cfg, once=True)

    jobs = {j["id"]: j for j in read_queue(q.path)}
    # main's own bare-metal claim (step 1) takes exactly one of the two
    # "any" jobs and runs it to completion.
    done_ids = {jid for jid, j in jobs.items() if j["status"] == "done"}
    pending_ids = {jid for jid, j in jobs.items() if j["status"] == "pending"}
    assert len(done_ids) == 1
    assert len(pending_ids) == 1
    # Crucially: the k8s side must NOT have grabbed the second "any" job
    # out from under the (simulated) idle r9700 — it's left pending for
    # r9700 to claim on its own next tick.
    assert done_ids | pending_ids == {"any1", "any2"}


def test_any_job_claimed_by_k8s_when_other_baremetal_busy(tmp_path, monkeypatch):
    """No idle bare-metal elsewhere (no heartbeat at all, i.e. the classic
    'r9700 is off/never started' case) — k8s must still be able to claim
    machine="any" jobs normally; the fairness gate must not starve k8s."""
    cfg = make_k8s_cfg(tmp_path, max_concurrent=2)
    cfg["types"]["replay"] = {"cwd": str(tmp_path), "command": ["true"]}
    q = Queue(cfg["queue"]["path"])
    write_queue(q.path, [
        {"id": "any1", "machine": "any", "type": "replay",
         "scene": "/s", "model_path": "m1", "status": "pending"},
        {"id": "any2", "machine": "any", "type": "replay",
         "scene": "/s", "model_path": "m2", "status": "pending"},
    ])
    monkeypatch.setattr(resources, "machine_busy", lambda *a, **k: False)
    monkeypatch.setattr(cfgmod, "machine_name", lambda c: "main")
    monkeypatch.setattr(runner, "run_job", lambda cfg, job, machine, q=None: ("done", 0))
    monkeypatch.setattr(runner, "run_job_k8s",
                        lambda cfg, job, machine, mcfg, q=None: ("done", 0))
    # no heartbeat_r9700.txt written at all — no idle machine to defer to.

    runner.run_loop(cfg, once=True)

    jobs = {j["id"]: j for j in read_queue(q.path)}
    # Both jobs are claimed and finished this tick: one by main's own
    # bare-metal slot, one by k8s.
    assert all(j["status"] == "done" for j in jobs.values())


def test_any_job_claimed_by_k8s_when_other_baremetal_paused(tmp_path, monkeypatch):
    """r9700 reports a fresh idle heartbeat but is actually paused (can
    never claim_next itself) — the fairness gate must recognize it can't
    really take the job and let k8s claim it rather than stranding it."""
    cfg = make_k8s_cfg(tmp_path, max_concurrent=2)
    cfg["types"]["replay"] = {"cwd": str(tmp_path), "command": ["true"]}
    q = Queue(cfg["queue"]["path"])
    write_queue(q.path, [
        {"id": "any1", "machine": "any", "type": "replay",
         "scene": "/s", "model_path": "m1", "status": "pending"},
        {"id": "any2", "machine": "any", "type": "replay",
         "scene": "/s", "model_path": "m2", "status": "pending"},
    ])
    monkeypatch.setattr(resources, "machine_busy", lambda *a, **k: False)
    monkeypatch.setattr(cfgmod, "machine_name", lambda c: "main")
    monkeypatch.setattr(runner, "run_job", lambda cfg, job, machine, q=None: ("done", 0))
    monkeypatch.setattr(runner, "run_job_k8s",
                        lambda cfg, job, machine, mcfg, q=None: ("done", 0))

    runner.write_heartbeat(cfg, "r9700", "idle k8s_inflight=0")
    from ablator.queue import write_pause_flag
    write_pause_flag(cfg["queue"]["path"], "r9700", "manual", "test")

    runner.run_loop(cfg, once=True)

    jobs = {j["id"]: j for j in read_queue(q.path)}
    assert all(j["status"] == "done" for j in jobs.values())


def test_pinned_k8s_job_unaffected_by_fairness_gate(tmp_path, monkeypatch):
    """A job explicitly pinned to the k8s machine must be claimable by k8s
    even while an idle bare-metal machine would otherwise cause "any" jobs
    to be deferred — only machine="any" jobs are ever gated."""
    cfg = make_k8s_cfg(tmp_path, max_concurrent=2)
    q = Queue(cfg["queue"]["path"])
    write_queue(q.path, _k8s_jobs(1))  # pinned to a100cluster
    monkeypatch.setattr(resources, "machine_busy", lambda *a, **k: False)
    monkeypatch.setattr(cfgmod, "machine_name", lambda c: "main")
    monkeypatch.setattr(runner, "run_job_k8s",
                        lambda cfg, job, machine, mcfg, q=None: ("done", 0))

    runner.write_heartbeat(cfg, "r9700", "idle k8s_inflight=0")

    runner.run_loop(cfg, once=True)

    jobs = {j["id"]: j for j in read_queue(q.path)}
    assert jobs["kjob0"]["status"] == "done"


def test_claim_next_only_pinned_skips_any_jobs(tmp_path):
    """Direct unit test of the only_pinned= gate added to claim_next."""
    import json as jsonmod
    path = tmp_path / "queue.jsonl"
    jobs = [
        {"id": "a", "machine": "any", "status": "pending"},
        {"id": "b", "machine": "a100cluster", "status": "pending"},
    ]
    with open(path, "w") as f:
        for j in jobs:
            f.write(jsonmod.dumps(j) + "\n")
    q = Queue(str(path))
    job = q.claim_next("a100cluster", only_pinned=True)
    assert job["id"] == "b"  # "any" job skipped, pinned job claimed
    assert q.claim_next("a100cluster", only_pinned=True) is None
