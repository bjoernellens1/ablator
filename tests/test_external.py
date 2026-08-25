"""Contract tests for the generic external scheduler interface."""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import pytest

from ablator import cli, experiment_declaration, runner
from ablator.external import (
    ExternalJobError,
    build_job,
    cancel_jobs,
    capture_runner_provenance,
    inspect_job,
    submit_job,
)
from ablator.queue import Queue


SHA_A = "0123456789abcdef0123456789abcdef01234567"


def _cfg(tmp_path: Path, *, require_pin: bool = False) -> dict:
    config = tmp_path / "ablator.json"
    queue = tmp_path / "queue.jsonl"
    raw = {
        "queue": {"path": str(queue), "log_dir": str(tmp_path / "logs")},
        "machines": {"main": {"hostname_patterns": ["*"]}},
        "types": {
            "researchflow": {
                "command": ["bash", "{jobscript}"],
                "cwd": str(tmp_path),
                "require_pinned_git": require_pin,
            }
        },
        "resources": {},
    }
    config.write_text(json.dumps(raw))
    return {**raw, "_path": str(config)}


def _last_json(text: str) -> dict:
    """Parse the command's final JSON line despite unrelated daemon-test output."""
    lines = [line for line in text.splitlines() if line.strip()]
    return json.loads(lines[-1])


def test_submit_is_idempotent_and_conflicts_fail_closed(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    job = build_job(
        cfg,
        job_id="rf-plan-job1",
        job_type="researchflow",
        params={"jobscript": "/shared/job.sh"},
        metadata={"scheduler": "snakemake"},
    )
    first, created = submit_job(cfg, job)
    assert created is True
    second, created = submit_job(cfg, job)
    assert created is False
    assert second["external_spec_sha256"] == first["external_spec_sha256"]

    changed = build_job(
        cfg,
        job_id="rf-plan-job1",
        job_type="researchflow",
        params={"jobscript": "/shared/other.sh"},
    )
    with pytest.raises(ExternalJobError):
        submit_job(cfg, changed)


def test_concurrent_identical_external_submissions_create_exactly_once(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    job = build_job(
        cfg, job_id="concurrent-same", job_type="researchflow",
        params={"jobscript": "/shared/job.sh"},
    )
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(
            lambda _index: submit_job(cfg, deepcopy(job)), range(16)
        ))

    assert sum(created for _item, created in results) == 1
    queued = Queue(cfg["queue"]["path"]).read()
    assert len(queued) == 1
    assert queued[0]["external_spec_sha256"] == job["external_spec_sha256"]


def test_concurrent_conflicting_external_submissions_never_mix_envelopes(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    variants = [
        build_job(
            cfg, job_id="concurrent-conflict", job_type="researchflow",
            params={"jobscript": f"/shared/{name}.sh"},
        )
        for name in ("a", "b")
    ]

    def attempt(job):
        try:
            stored, created = submit_job(cfg, deepcopy(job))
            return ("stored", stored["external_spec_sha256"], created)
        except ExternalJobError:
            return ("rejected", job["external_spec_sha256"], False)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, variants))

    queued = Queue(cfg["queue"]["path"]).read()
    assert len(queued) == 1
    assert sorted(item[0] for item in results) == ["rejected", "stored"]
    winner = next(item for item in results if item[0] == "stored")
    assert winner[2] is True
    assert queued[0]["external_spec_sha256"] == winner[1]


def test_strict_external_submit_rejects_unpinned_atomically(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, require_pin=True)
    job = build_job(cfg, job_id="strict", job_type="researchflow")

    with pytest.raises(ExternalJobError, match="requires an immutable Git target"):
        submit_job(cfg, job)

    assert Queue(cfg["queue"]["path"]).read() == []


def test_external_dependency_mixed_sha_rejects_atomically(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, require_pin=True)
    parent = build_job(
        cfg, job_id="parent", job_type="researchflow", git_sha=SHA_A,
    )
    child = build_job(
        cfg, job_id="child", job_type="researchflow", depends_on="parent",
        git_sha="f" * 40,
    )
    submit_job(cfg, parent)

    with pytest.raises(ExternalJobError, match="dependency chain changes Git target"):
        submit_job(cfg, child)

    assert [item["id"] for item in Queue(cfg["queue"]["path"]).read()] == ["parent"]


def test_mutated_external_hash_input_rejects_before_enqueue(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    job = build_job(
        cfg, job_id="mutated", job_type="researchflow",
        params={"jobscript": "/shared/original.sh"},
    )
    job["params"]["jobscript"] = "/shared/tampered.sh"

    with pytest.raises(ExternalJobError, match="external specification SHA-256 mismatch"):
        submit_job(cfg, job)

    assert Queue(cfg["queue"]["path"]).read() == []


def test_complete_external_submission_is_frozen_and_protected(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    job = build_job(
        cfg,
        job_id="frozen",
        job_type="researchflow",
        machine="main",
        params={"jobscript": "/shared/job.sh"},
        metadata={"scheduler": "snakemake"},
        lane=3,
        depends_on=None,
        git_sha=SHA_A,
        git_repo="https://github.com/example/project.git",
    )
    submission = job["submission_provenance"]
    assert submission == experiment_declaration.submission_provenance(job)
    assert submission["external_spec_sha256"] == job["external_spec_sha256"]
    submit_job(cfg, job)
    queue = Queue(cfg["queue"]["path"])

    for field, changed in (
        ("external_id", "other"),
        ("external_schema", "other/v1"),
        ("external_spec_sha256", "0" * 64),
        ("external_metadata", {"scheduler": "other"}),
        ("params", {"jobscript": "/other.sh"}),
        ("machine", "any"),
        ("type", "other"),
        ("lane", 1),
        ("depends_on", "other"),
    ):
        with pytest.raises(SystemExit, match=f"immutable {field}"):
            queue.update("frozen", **{field: changed})


def test_external_hash_is_reverified_before_protected_environment() -> None:
    cfg = {
        "types": {"researchflow": {"command": ["true"]}},
        "machines": {},
    }
    job = build_job(
        cfg, job_id="env-tamper", job_type="researchflow",
        params={"jobscript": "/shared/original.sh"},
    )
    job["status"] = "running"
    job["params"] = {"jobscript": "/shared/tampered.sh"}

    with pytest.raises(
        experiment_declaration.ExperimentDeclarationError,
        match="external specification SHA-256 mismatch",
    ):
        experiment_declaration.experiment_environment(job)


def test_external_git_target_is_immutable_submit_identity(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    job = build_job(
        cfg,
        job_id="rf-pinned",
        job_type="researchflow",
        params={"jobscript": "/shared/job.sh"},
        git_sha=SHA_A.upper(),
        git_repo="https://github.com/example/project.git",
    )
    assert job["requested_git_sha"] == SHA_A
    assert job["git_repo"] == "https://github.com/example/project.git"
    submission = experiment_declaration.submission_provenance(job)
    assert submission["requested_git_sha"] == SHA_A
    assert submission["git_repo"] == "https://github.com/example/project.git"

    changed = build_job(
        cfg,
        job_id="rf-pinned",
        job_type="researchflow",
        params={"jobscript": "/shared/job.sh"},
        git_sha="f" * 40,
        git_repo="https://github.com/example/project.git",
    )
    submit_job(cfg, job)
    with pytest.raises(ExternalJobError, match="different specification"):
        submit_job(cfg, changed)


@pytest.mark.parametrize("sha", ["main", "a" * 39, "g" * 40, ""])
def test_external_git_target_requires_full_commit_sha(tmp_path: Path, sha: str) -> None:
    with pytest.raises(ExternalJobError, match="full 40-character hexadecimal"):
        build_job(
            _cfg(tmp_path),
            job_id="rf-bad-pin",
            job_type="researchflow",
            git_sha=sha,
        )


def test_exact_inspection_and_pending_cancel(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    job = build_job(cfg, job_id="job-a", job_type="researchflow", params={"jobscript": "a.sh"})
    submit_job(cfg, job)
    before = inspect_job(cfg, "job-a")
    assert before["status"] == "pending"
    assert before["params"]["jobscript"] == "a.sh"

    result = cancel_jobs(cfg, ["job-a"])
    assert result == [{"job_id": "job-a", "status": "cancelled", "action": "cancelled"}]
    assert inspect_job(cfg, "job-a")["status"] == "cancelled"
    # Repeated cancellation is a stable no-op rather than an error.
    assert cancel_jobs(cfg, ["job-a"])[0]["action"] == "no_op"


def test_running_cancel_uses_existing_control_protocol(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    job = build_job(cfg, job_id="job-running", job_type="researchflow", params={"jobscript": "a.sh"})
    submit_job(cfg, job)
    Queue(cfg["queue"]["path"]).update("job-running", status="running", claimed_by="main")
    result = cancel_jobs(cfg, ["job-running"])
    assert result[0]["action"] == "cancel_requested"
    assert (tmp_path / "control_job-running").read_text() == "skip\n"


def test_external_params_become_generic_template_variables(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    job = build_job(
        cfg,
        job_id="job-template",
        job_type="researchflow",
        params={"jobscript": "/shared/run.sh", "seed": 3},
    )
    values = runner._job_vars(job, "main")
    assert values["jobscript"] == "/shared/run.sh"
    assert values["seed"] == "3"
    argv, _env, cwd = runner.render_command(cfg["types"]["researchflow"], job, "main")
    assert argv == ["bash", "/shared/run.sh"]
    assert cwd == str(tmp_path)


def test_reserved_params_cannot_change_legacy_queue_meaning(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    with pytest.raises(ExternalJobError):
        build_job(
            cfg,
            job_id="bad",
            job_type="researchflow",
            params={"machine": "other"},
        )


def test_runner_provenance_contains_config_identity(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    prov = capture_runner_provenance(cfg, "main")
    assert prov["schema"] == "ablator.runner-provenance/v1"
    assert prov["machine"] == "main"
    assert prov["config_sha256"] == hashlib.sha256(Path(cfg["_path"]).read_bytes()).hexdigest()
    assert "git_commit" in prov
    assert "git_dirty" in prov


def test_cli_submit_inspect_cancel_json_contract(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = _cfg(tmp_path)
    cli.main(
        [
            "--config",
            cfg["_path"],
            "submit",
            "--format",
            "json",
            "--id",
            "cli-job",
            "--type",
            "researchflow",
            "--param",
            "jobscript=/shared/cli.sh",
            "--metadata-json",
            '{"scheduler":"snakemake"}',
        ]
    )
    submitted = _last_json(capsys.readouterr().out)
    assert submitted["job_id"] == "cli-job"
    assert submitted["created"] is True

    cli.main(["--config", cfg["_path"], "inspect", "--format", "json", "cli-job"])
    inspected = _last_json(capsys.readouterr().out)
    assert inspected["status"] == "pending"
    assert inspected["metadata"]["scheduler"] == "snakemake"

    cli.main(["--config", cfg["_path"], "cancel-jobs", "--format", "json", "cli-job"])
    cancelled = _last_json(capsys.readouterr().out)
    assert cancelled["jobs"][0]["status"] == "cancelled"


def test_cli_submit_transports_git_target(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = _cfg(tmp_path)
    cli.main(
        [
            "--config", cfg["_path"], "submit", "--format", "json",
            "--id", "cli-pinned", "--type", "researchflow",
            "--git-sha", SHA_A,
            "--git-repo", "https://github.com/example/project.git",
        ]
    )
    _last_json(capsys.readouterr().out)
    cli.main(["--config", cfg["_path"], "inspect", "--format", "json", "cli-pinned"])
    inspected = _last_json(capsys.readouterr().out)
    assert inspected["requested_git_sha"] == SHA_A
    assert inspected["git_repo"] == "https://github.com/example/project.git"
