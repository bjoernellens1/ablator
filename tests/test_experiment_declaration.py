"""Cross-repository ExperimentDeclaration producer contract (#705)."""

import copy
import json
import subprocess

import pytest

from ablator import cli
from ablator import experiment_declaration as declarations
from ablator import runner, spec as specmod
from ablator.queue import Queue


def test_canonical_declaration_matches_splatograph_contract():
    """Changing key order must not change the exact bytes/hash consumed downstream."""
    declaration = {
        "purpose": "Prove café declaration transport",
        "schema_version": 1,
        "title": "Producer integration",
        "run_class": "experiment",
    }

    canonical = declarations.canonical_declaration_json(declaration)

    assert canonical == (
        '{"purpose":"Prove café declaration transport","run_class":"experiment",'
        '"schema_version":1,"title":"Producer integration"}'
    )
    assert declarations.declaration_sha256(declaration) == (
        "c191903c81dd2331c1468954726acc98020e2865ad5055deb7c9ade7e9c8e282"
    )


def _declared_spec() -> dict:
    return {
        "name": "issue705",
        "git_sha": "0123456789abcdef0123456789abcdef01234567",
        "experiment": {
            "schema_version": 1,
            "run_class": "experiment",
            "experiment_id": "issue705_transport",
            "title": "Immutable declaration transport",
            "purpose": "Prove the producer/consumer boundary",
            "owner_refs": ["splatograph#705"],
            "expected_evidence": ["matching hash at every boundary"],
            "comparison_group_id": "issue705_pair",
            "domain": {
                "system": "splatograph",
                "payload": {"replay_capsule_sha256": "abc123"},
            },
        },
        "base": {"type": "replay", "scene": "/data/fr3"},
        "arms": [
            {
                "id": "ctrl",
                "declaration": {
                    "arm": {
                        "title": "Control",
                        "comparison_role": "control",
                        "manipulation": "No treatment",
                    }
                },
            },
            {
                "id": "candidate",
                "declaration": {
                    "arm": {
                        "title": "Candidate",
                        "comparison_role": "treatment",
                        "manipulation": "Enable producer under test",
                    },
                    "domain": {"payload": {"arm_kind": "candidate"}},
                },
            },
        ],
    }


def test_expand_spec_freezes_fully_resolved_declaration_per_arm():
    """Losing an overlay or retaining shared mutable input would misidentify an arm."""
    source = _declared_spec()
    original = copy.deepcopy(source)

    jobs = specmod.expand_spec(source)

    control = jobs[0]
    candidate = jobs[1]
    assert control["experiment_declaration"]["arm"] == {
        "id": "ctrl",
        "title": "Control",
        "comparison_role": "control",
        "manipulation": "No treatment",
    }
    assert candidate["experiment_declaration"]["arm"]["id"] == "candidate"
    assert candidate["experiment_declaration"]["domain"] == {
        "system": "splatograph",
        "payload": {
            "replay_capsule_sha256": "abc123",
            "arm_kind": "candidate",
        },
    }
    assert (
        json.loads(candidate["experiment_declaration_json"])
        == (candidate["experiment_declaration"])
    )
    assert len(candidate["experiment_declaration_sha256"]) == 64
    assert candidate["gradeability"] == "GRADEABLE_DECLARED"
    assert source == original


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda d: d.pop("expected_evidence"), "expected_evidence"),
        (lambda d: d.pop("owner_refs"), "owner_refs or standalone_reason"),
        (lambda d: d.__setitem__("schema_version", 2), "schema_version"),
        (lambda d: d.__setitem__("run_class", "paper_magic"), "run_class"),
    ],
)
def test_expand_spec_rejects_invalid_gradeable_declaration(mutation, message):
    """A malformed declared run must fail before it can become a queue job."""
    spec = _declared_spec()
    mutation(spec["experiment"])

    with pytest.raises(SystemExit, match=message):
        specmod.expand_spec(spec)


def test_expand_spec_rejects_conflicting_declared_arm_id():
    """Two different arm identities must never be silently resolved by precedence."""
    spec = _declared_spec()
    spec["arms"][0]["declaration"]["arm"]["id"] = "not-ctrl"

    with pytest.raises(SystemExit, match="conflicts with spec arm id"):
        specmod.expand_spec(spec)


def test_legacy_spec_stays_runnable_without_gradeability_fields():
    """Migration must not fabricate declarations for existing debug/developer specs."""
    spec = {
        "name": "legacy",
        "base": {"type": "replay"},
        "arms": [{"id": "smoke"}],
    }

    job = specmod.expand_spec(spec)[0]

    assert not any(key.startswith("experiment_declaration") for key in job)
    assert "gradeability" not in job


def test_plan_invalid_declaration_leaves_queue_untouched(tmp_path):
    """Preflight must validate every arm before append makes any job claimable."""
    spec = _declared_spec()
    spec["arms"][1]["declaration"]["arm"].pop("manipulation")
    spec_path = tmp_path / "invalid.json"
    spec_path.write_text(json.dumps(spec))
    cfg = {
        "_path": str(tmp_path / "config.toml"),
        "queue": {"path": str(tmp_path / "queue.jsonl")},
        "types": {"replay": {"command": ["true"]}},
    }

    with pytest.raises(SystemExit, match="manipulation"):
        cli.cmd_plan(cfg, str(spec_path))

    assert not (tmp_path / "queue.jsonl").exists()


def test_queue_refuses_tampered_frozen_hash_before_append(tmp_path):
    """A queue record whose object and hash disagree must never become claimable."""
    job = specmod.expand_spec(_declared_spec())[0]
    job["experiment_declaration_sha256"] = "0" * 64
    queue = Queue(str(tmp_path / "queue.jsonl"))

    with pytest.raises(SystemExit, match="SHA-256 mismatch"):
        queue.append([job])

    assert not (tmp_path / "queue.jsonl").exists()


def test_queue_refuses_declaration_mutation_after_enqueue(tmp_path):
    """Bookkeeping updates must not be able to rewrite pre-run intent."""
    job = specmod.expand_spec(_declared_spec())[0]
    queue = Queue(str(tmp_path / "queue.jsonl"))
    queue.append([job])

    changed = {**job["experiment_declaration"], "purpose": "Changed after enqueue"}
    with pytest.raises(SystemExit, match="immutable experiment_declaration"):
        queue.update(job["id"], experiment_declaration=changed)

    stored = queue.read()[0]
    assert stored["experiment_declaration"]["purpose"] == (
        "Prove the producer/consumer boundary"
    )


def test_render_command_propagates_exact_declaration_to_direct_child():
    """A non-container child must receive the frozen bytes, hash, and job id."""
    job = specmod.expand_spec(_declared_spec())[0]
    tcfg = {
        "command": ["python", "train.py"],
        "env": {declarations.JOB_ID_ENV: "config-must-not-override-job"},
    }

    _, env, _ = runner.render_command(tcfg, job, "main")

    assert env[declarations.DECLARATION_ENV] == job["experiment_declaration_json"]
    assert (
        env[declarations.DECLARATION_SHA_ENV] == (job["experiment_declaration_sha256"])
    )
    assert env[declarations.JOB_ID_ENV] == job["id"]


def test_legacy_child_cannot_inherit_or_configure_stale_declaration(monkeypatch):
    """An undeclared job must not be upgraded by ambient runner or type env."""
    monkeypatch.setenv(declarations.DECLARATION_ENV, '{"stale":true}')
    monkeypatch.setenv(declarations.DECLARATION_SHA_ENV, "f" * 64)
    monkeypatch.setenv(declarations.JOB_ID_ENV, "old-job")
    job = specmod.expand_spec({"name": "legacy", "arms": [{"id": "smoke"}]})[0]
    tcfg = {
        "command": ["true"],
        "env": {declarations.JOB_ID_ENV: "configured-fake"},
    }

    _, env, _ = runner.render_command(tcfg, job, "main")

    assert declarations.DECLARATION_ENV not in env
    assert declarations.DECLARATION_SHA_ENV not in env
    assert declarations.JOB_ID_ENV not in env


def test_render_command_injects_declaration_into_container_environment():
    """Setting host env alone is insufficient for Docker/Podman child processes."""
    job = specmod.expand_spec(_declared_spec())[0]
    tcfg = {"command": ["podman", "run", "--rm", "image:tag", "python", "train.py"]}

    argv, _, _ = runner.render_command(tcfg, job, "main")

    injected = {
        argv[index + 1] for index, token in enumerate(argv[:-1]) if token == "--env"
    }
    assert injected == {
        f"{declarations.DECLARATION_ENV}={job['experiment_declaration_json']}",
        f"{declarations.DECLARATION_SHA_ENV}={job['experiment_declaration_sha256']}",
        f"{declarations.JOB_ID_ENV}={job['id']}",
    }
    assert argv.index("--env") < argv.index("image:tag")


@pytest.mark.parametrize("runtime", ["docker", "podman"])
@pytest.mark.parametrize("name", sorted(declarations.PROTECTED_ENV))
@pytest.mark.parametrize("env_flag", ["-e{name}=evil", "-e={name}=evil"])
def test_render_command_rejects_compact_protected_container_env(
    runtime, name, env_flag
):
    """Compact ``-eNAME=value`` must not override trusted injected values."""
    job = specmod.expand_spec(_declared_spec())[0]
    tcfg = {
        "command": [
            runtime,
            "run",
            env_flag.format(name=name),
            "-eUNRELATED=allowed",
            "image:tag",
        ]
    }

    with pytest.raises(runner.TemplateError, match=name):
        runner.render_command(tcfg, job, "main")


def test_render_command_fails_closed_on_tampered_queue_declaration():
    """Manual queue edits after enqueue must not reach a child process."""
    job = specmod.expand_spec(_declared_spec())[0]
    job["experiment_declaration_json"] = "{}"

    with pytest.raises(runner.TemplateError, match="canonical.*mismatch"):
        runner.render_command({"command": ["true"]}, job, "main")


def test_k8s_manifest_propagates_exact_declaration_to_trainer_container():
    """Kubernetes must not discard the declaration env rendered for bare metal."""
    job = specmod.expand_spec(_declared_spec())[0]
    mcfg = {
        "namespace": "jobs",
        "kai_queue": "batch",
        "priority_class": "batch-low",
        "image": "image:tag",
    }

    manifest = runner.build_k8s_job_manifest(
        mcfg, job, ["python", "train.py"], "/workspace"
    )

    container = manifest["spec"]["template"]["spec"]["containers"][0]
    env = {item["name"]: item["value"] for item in container["env"]}
    assert env == {
        declarations.DECLARATION_ENV: job["experiment_declaration_json"],
        declarations.DECLARATION_SHA_ENV: job["experiment_declaration_sha256"],
        declarations.JOB_ID_ENV: job["id"],
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def test_bare_metal_runner_log_records_same_frozen_declaration_before_child(
    tmp_path, monkeypatch
):
    """The outer audit log must expose the exact object/hash used at launch."""
    job = specmod.expand_spec(_declared_spec())[0]
    job["model_path"] = str(tmp_path / "model")
    cfg = {
        "queue": {"path": str(tmp_path / "queue.jsonl"), "log_dir": str(tmp_path)},
        "machines": {"main": {}},
        "types": {"replay": {"cwd": str(tmp_path), "command": ["true"]}},
    }
    monkeypatch.setattr(
        runner,
        "capture_and_record_provenance",
        lambda *args, **kwargs: {
            "commit": job["requested_git_sha"], "branch": "HEAD", "dirty": False
        },
    )
    monkeypatch.setattr(
        runner.sourcecheckout,
        "prepare_job_source",
        lambda cfg, job, machine, tcfg: runner.sourcecheckout.PreparedSource(
            type_config=tcfg
        ),
    )
    monkeypatch.setattr(runner, "check_checkout_drift", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "output_folder_preflight", lambda *args: "preflight ok")
    monkeypatch.setattr(
        runner,
        "supervise",
        lambda cfg, job, proc, base_dir, q, **kwargs: (proc.wait(), None)[1],
    )

    status, exit_code = runner.run_job(cfg, job, "main")

    log = (tmp_path / f"{job['id']}.log").read_text()
    assert status == "done" and exit_code == 0
    assert f"ABLATOR_JOB_ID={job['id']}" in log
    assert (
        f"ABLATOR_EXPERIMENT_DECLARATION_SHA256={job['experiment_declaration_sha256']}"
    ) in log
    assert (
        f"ABLATOR_EXPERIMENT_DECLARATION_JSON={job['experiment_declaration_json']}"
    ) in log


def test_k8s_runner_log_records_same_frozen_declaration_before_submit(
    tmp_path, monkeypatch
):
    """Cluster dispatch logs must preserve the same audit chain as bare metal."""
    job = specmod.expand_spec(_declared_spec())[0]
    cfg = {
        "queue": {"path": str(tmp_path / "queue.jsonl"), "log_dir": str(tmp_path)},
        "types": {"replay": {"cwd": "/workspace", "command": ["python", "train.py"]}},
    }
    mcfg = {
        "namespace": "jobs",
        "kai_queue": "batch",
        "priority_class": "batch-low",
        "image": "image:tag",
        "git_sync_repo_url": "https://github.com/example/project.git",
    }
    monkeypatch.setattr(
        runner.sourcecheckout,
        "validate_requested_revision_policy",
        lambda *args: "https://github.com/example/project.git",
    )
    monkeypatch.setattr(runner, "_dispatch_host_commit", lambda *args: "abc123")
    monkeypatch.setattr(
        runner.provmod,
        "check_image_drift",
        lambda *args: {"image": "image:tag", "warning": None},
    )
    monkeypatch.setattr(
        runner.provmod,
        "format_banner",
        lambda *args: "provenance banner",
    )

    def apply_after_log(*args, **kwargs):
        log = (tmp_path / f"{job['id']}.log").read_text()
        assert f"ABLATOR_JOB_ID={job['id']}" in log
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(runner, "_kubectl", apply_after_log)
    monkeypatch.setattr(runner, "_poll_k8s_job", lambda *args, **kwargs: ("done", 0))

    status, exit_code = runner.run_job_k8s(cfg, job, "a100", mcfg)

    log = (tmp_path / f"{job['id']}.log").read_text()
    assert status == "done" and exit_code == 0
    assert f"ABLATOR_JOB_ID={job['id']}" in log
    assert (
        f"ABLATOR_EXPERIMENT_DECLARATION_SHA256={job['experiment_declaration_sha256']}"
    ) in log
    assert (
        f"ABLATOR_EXPERIMENT_DECLARATION_JSON={job['experiment_declaration_json']}"
    ) in log


def test_gradeable_job_cannot_be_rerun_in_place(tmp_path):
    """A scientific rerun needs a new identity/output and explicit lineage."""
    job = specmod.expand_spec(_declared_spec())[0]
    job["status"] = "done"
    queue_path = tmp_path / "queue.jsonl"
    Queue(str(queue_path)).append([job])
    cfg = {"queue": {"path": str(queue_path)}}

    with pytest.raises(SystemExit, match="new job identity.*lineage"):
        cli.cmd_rerun(cfg, job["id"])

    assert Queue(str(queue_path)).read()[0]["status"] == "done"
