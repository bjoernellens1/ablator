"""Contract tests for the generic external scheduler interface."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ablator import cli, runner
from ablator.external import (
    ExternalJobError,
    build_job,
    cancel_jobs,
    capture_runner_provenance,
    inspect_job,
    submit_job,
)
from ablator.queue import Queue


def _cfg(tmp_path: Path) -> dict:
    config = tmp_path / "ablator.json"
    queue = tmp_path / "queue.jsonl"
    raw = {
        "queue": {"path": str(queue), "log_dir": str(tmp_path / "logs")},
        "machines": {"main": {"hostname_patterns": ["*"]}},
        "types": {
            "researchflow": {
                "command": ["bash", "{jobscript}"],
                "cwd": str(tmp_path),
            }
        },
        "resources": {},
    }
    config.write_text(json.dumps(raw))
    return {**raw, "_path": str(config)}


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
    submitted = json.loads(capsys.readouterr().out)
    assert submitted["job_id"] == "cli-job"
    assert submitted["created"] is True

    cli.main(["--config", cfg["_path"], "inspect", "--format", "json", "cli-job"])
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["status"] == "pending"
    assert inspected["metadata"]["scheduler"] == "snakemake"

    cli.main(["--config", cfg["_path"], "cancel-jobs", "--format", "json", "cli-job"])
    cancelled = json.loads(capsys.readouterr().out)
    assert cancelled["jobs"][0]["status"] == "cancelled"
