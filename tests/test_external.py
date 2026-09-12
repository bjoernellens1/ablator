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


def _declaration(arm_id: str = "control") -> dict:
    return {
        "schema_version": 1,
        "run_class": "experiment",
        "experiment_id": "external_transport",
        "title": "External declaration transport",
        "purpose": "Verify immutable declaration forwarding",
        "owner_refs": ["splatograph#705"],
        "expected_evidence": ["declaration hash"],
        "arm": {
            "id": arm_id,
            "comparison_role": "control",
            "manipulation": "none",
        },
    }


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
        ("id", "other"),
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


def test_external_submit_freezes_declaration_for_runner_and_idempotency(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    job = build_job(
        cfg,
        job_id="declared-external",
        job_type="researchflow",
        experiment_declaration=_declaration(),
        git_sha=SHA_A,
    )

    assert job["experiment_declaration"] == _declaration()
    assert job["experiment_declaration_sha256"] == experiment_declaration.declaration_sha256(
        _declaration()
    )
    env = experiment_declaration.experiment_environment(job)
    assert env[experiment_declaration.DECLARATION_ENV] == job[
        "experiment_declaration_json"
    ]
    assert env[experiment_declaration.DECLARATION_SHA_ENV] == job[
        "experiment_declaration_sha256"
    ]

    submit_job(cfg, job)
    changed = build_job(
        cfg,
        job_id="declared-external",
        job_type="researchflow",
        experiment_declaration=_declaration("candidate"),
        git_sha=SHA_A,
    )
    with pytest.raises(ExternalJobError, match="different specification"):
        submit_job(cfg, changed)


def test_legacy_external_record_without_frozen_envelope_fails_closed():
    job = build_job(
        {"types": {"researchflow": {"command": ["true"]}}, "machines": {}},
        job_id="old-external", job_type="researchflow",
    )
    job.pop("submission_provenance")
    job["status"] = "running"

    with pytest.raises(
        experiment_declaration.ExperimentDeclarationError,
        match="frozen submission_provenance",
    ):
        experiment_declaration.experiment_environment(job)


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


def test_running_cancel_uses_the_shared_control_path_helper(tmp_path: Path, monkeypatch) -> None:
    """Bug fix (incident 2026-09-12): cancel_jobs() must derive the control
    file path via runner.control_path() -- the exact same helper
    ablator skip/stop/requeue use -- rather than an independently
    hand-built f"control_{job_id}" string. Proven here by monkeypatching
    control_path() to a distinctive value and asserting cancel_jobs()
    actually followed it: if cancel_jobs ever regresses to a duplicated,
    independent path construction, this fails even though the two happen
    to compute the same string today."""
    cfg = _cfg(tmp_path)
    job = build_job(cfg, job_id="job-running2", job_type="researchflow", params={"jobscript": "a.sh"})
    submit_job(cfg, job)
    Queue(cfg["queue"]["path"]).update("job-running2", status="running", claimed_by="main")

    calls = []

    def fake_control_path(cfg_arg, job_id):
        calls.append(job_id)
        return str(tmp_path / f"CUSTOM_control_{job_id}")

    monkeypatch.setattr(runner, "control_path", fake_control_path)
    result = cancel_jobs(cfg, ["job-running2"])
    assert result[0]["action"] == "cancel_requested"
    assert calls == ["job-running2"]
    assert (tmp_path / "CUSTOM_control_job-running2").read_text() == "skip\n"


def test_cancel_jobs_control_file_honored_by_supervise_within_poll_cadence(
    tmp_path: Path,
) -> None:
    """End-to-end: a control file written by cancel_jobs() must be picked
    up by a real supervise() loop at its normal poll cadence, exactly like
    `ablator skip` -- same mechanism, same promptness, no separate/slower
    path. Uses a real subprocess (a plain `sleep`) and a real, short
    poll_s rather than mocking supervise() itself."""
    import subprocess as sp
    cfg = _cfg(tmp_path)
    job_record = build_job(cfg, job_id="job-cancel-e2e", job_type="researchflow",
                          params={"jobscript": "a.sh"})
    submit_job(cfg, job_record)
    Queue(cfg["queue"]["path"]).update("job-cancel-e2e", status="running", claimed_by="main")

    result = cancel_jobs(cfg, ["job-cancel-e2e"])
    assert result[0]["action"] == "cancel_requested"

    job = {"id": "job-cancel-e2e", "type": "researchflow", "model_path": "m",
          "status": "running"}
    proc = sp.Popen(["sleep", "5"], start_new_session=True)
    try:
        outcome = runner.supervise(cfg, job, proc, str(tmp_path), q=None,
                                   poll_s=0.05, argv=["sleep", "5"])
        assert outcome == "cancelled"
        assert proc.poll() is not None  # actually killed, not just flagged
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


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


def test_cli_submit_transports_external_experiment_declaration(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = _cfg(tmp_path)
    cli.main(
        [
            "--config", cfg["_path"], "submit", "--format", "json",
            "--id", "cli-declared", "--type", "researchflow",
            "--git-sha", SHA_A,
            "--experiment-declaration-json", json.dumps(_declaration()),
        ]
    )
    _last_json(capsys.readouterr().out)
    cli.main(["--config", cfg["_path"], "inspect", "--format", "json", "cli-declared"])
    inspected = _last_json(capsys.readouterr().out)
    assert inspected["gradeability"] == "GRADEABLE_DECLARED"
    assert inspected["experiment_declaration"] == _declaration()
