"""Queue/submission provenance transport contract (issue #34)."""

import hashlib
import json

import pytest

from ablator import experiment_declaration as declarations
from ablator import external, runner, spec as specmod


def _cfg():
    return {
        "_path": "/tmp/ablator.toml",
        "machines": {"main": {}},
        "types": {"replay": {"command": ["true"]}},
    }


def test_loaded_plan_freezes_exact_spec_and_transports_it(tmp_path):
    source = {
        "name": "issue34",
        "base": {
            "type": "replay",
            "scene": "/data/fr3",
            "iterations": 40000,
            "machine": "main",
        },
        "arms": [{"id": "ctrl", "extra_args": "--seed 5"}],
    }
    path = tmp_path / "issue34.json"
    path.write_text(json.dumps(source, indent=2))

    loaded = specmod.load_spec(str(path))
    job = specmod.expand_spec(loaded)[0]
    submission = job["submission_provenance"]

    canonical = json.dumps(source, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert submission == {
        "schema": "ablator.submission/v1",
        "surface": "plan",
        "spec_path": str(path.resolve()),
        "spec_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "spec": source,
        "ablation": "issue34",
    }

    env = declarations.experiment_environment(job)
    assert env[declarations.JOB_ID_ENV] == "issue34_ctrl"
    assert json.loads(env[declarations.JOB_JSON_ENV])["submission_provenance"] == submission
    assert json.loads(env[declarations.SUBMISSION_ENV]) == submission
    assert declarations.DECLARATION_ENV not in env
    assert declarations.DECLARATION_SHA_ENV not in env


def test_pure_in_memory_spec_does_not_fabricate_plan_submission():
    job = specmod.expand_spec({"name": "memory", "arms": [{"id": "x"}]})[0]
    assert "submission_provenance" not in job


def test_external_submit_gets_equivalent_structured_submission_envelope():
    job = external.build_job(
        _cfg(),
        job_id="rf-train-fr3",
        job_type="replay",
        machine="main",
        params={"jobscript": "/shared/train.sh"},
        metadata={"scheduler": "snakemake"},
        lane=3,
        depends_on="prep",
    )

    env = declarations.experiment_environment(job)
    submission = json.loads(env[declarations.SUBMISSION_ENV])

    assert env[declarations.JOB_ID_ENV] == "rf-train-fr3"
    assert submission["surface"] == "submit"
    assert submission["job_id"] == "rf-train-fr3"
    assert submission["type"] == "replay"
    assert submission["machine"] == "main"
    assert submission["params"] == {"jobscript": "/shared/train.sh"}
    assert submission["metadata"] == {"scheduler": "snakemake"}
    assert submission["lane"] == 3
    assert submission["depends_on"] == "prep"
    assert submission["external_spec_sha256"] == job["external_spec_sha256"]
    assert json.loads(env[declarations.JOB_JSON_ENV])["external_id"] == "rf-train-fr3"


def test_claimed_legacy_job_gets_identity_and_job_snapshot_without_declaration():
    job = {
        "id": "legacy_job",
        "type": "replay",
        "machine": "any",
        "status": "running",
        "claimed_by": "main",
    }

    env = declarations.experiment_environment(job)

    assert env[declarations.JOB_ID_ENV] == "legacy_job"
    assert json.loads(env[declarations.JOB_JSON_ENV]) == job
    assert declarations.SUBMISSION_ENV not in env
    assert declarations.DECLARATION_ENV not in env
    assert declarations.DECLARATION_SHA_ENV not in env


def test_plan_submission_hash_is_revalidated_before_launch(tmp_path):
    path = tmp_path / "spec.json"
    path.write_text(json.dumps({"name": "x", "arms": [{"id": "a"}]}))
    job = specmod.expand_spec(specmod.load_spec(str(path)))[0]
    job["submission_provenance"]["spec"]["name"] = "tampered"

    with pytest.raises(declarations.ExperimentDeclarationError, match="spec SHA-256 mismatch"):
        declarations.experiment_environment(job)


def test_new_transport_variables_are_protected():
    assert declarations.JOB_ID_ENV in declarations.PROTECTED_ENV
    assert declarations.JOB_JSON_ENV in declarations.PROTECTED_ENV
    assert declarations.SUBMISSION_ENV in declarations.PROTECTED_ENV


def test_claimed_legacy_direct_child_replaces_stale_protected_environment(monkeypatch):
    job = {
        "id": "legacy_direct",
        "type": "replay",
        "machine": "main",
        "status": "running",
    }
    stale = {name: "stale" for name in declarations.PROTECTED_ENV}
    for name, value in stale.items():
        monkeypatch.setenv(name, value)

    _, env, _ = runner.render_command(
        {"command": ["true"], "env": stale}, job, "main"
    )

    assert env[declarations.JOB_ID_ENV] == job["id"]
    assert json.loads(env[declarations.JOB_JSON_ENV]) == job
    assert declarations.SUBMISSION_ENV not in env
    assert declarations.DECLARATION_ENV not in env
    assert declarations.DECLARATION_SHA_ENV not in env


@pytest.mark.parametrize("runtime", ["docker", "podman"])
def test_claimed_legacy_container_receives_trusted_job_environment(runtime):
    job = {
        "id": f"legacy_{runtime}",
        "type": "replay",
        "machine": "main",
        "status": "running",
    }

    argv, _, _ = runner.render_command(
        {"command": [runtime, "run", "--rm", "image:tag", "true"]},
        job,
        "main",
    )

    injected = {
        argv[index + 1] for index, token in enumerate(argv[:-1]) if token == "--env"
    }
    assert injected == {
        f"{declarations.JOB_ID_ENV}={job['id']}",
        f"{declarations.JOB_JSON_ENV}="
        f"{json.dumps(job, sort_keys=True, separators=(',', ':'))}",
    }


def test_absolute_container_runtime_receives_protected_environment():
    job = {
        "id": "absolute-runtime",
        "type": "replay",
        "machine": "main",
        "status": "running",
    }
    argv, _, _ = runner.render_command(
        {
            "command": [
                "/usr/bin/podman", "run", "--rm", "image:tag", "true",
            ],
        },
        job,
        "main",
    )

    assert argv[0] == "/usr/bin/podman"
    assert any(token.startswith("ABLATOR_JOB_JSON=") for token in argv)
    assert runner.container_name_from_argv(argv) == "splat_train_absolute-runtime"


def test_claimed_legacy_kubernetes_trainer_receives_job_environment():
    job = {
        "id": "legacy_k8s",
        "type": "replay",
        "machine": "cluster",
        "status": "running",
    }
    machine = {
        "namespace": "jobs",
        "kai_queue": "batch",
        "priority_class": "batch-low",
        "image": "image:tag",
    }

    manifest = runner.build_k8s_job_manifest(
        machine, job, ["python", "train.py"], "/workspace"
    )

    trainer = manifest["spec"]["template"]["spec"]["containers"][0]
    env = {item["name"]: item["value"] for item in trainer["env"]}
    assert env == {
        declarations.JOB_ID_ENV: job["id"],
        declarations.JOB_JSON_ENV: json.dumps(
            job, sort_keys=True, separators=(",", ":")
        ),
    }
