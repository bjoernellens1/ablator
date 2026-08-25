"""Tests for immutable Git targets in ablation specs."""

import pytest

from ablator import spec as specmod
from ablator.queue import Queue


SHA_A = "0123456789abcdef0123456789abcdef01234567"
SHA_B = "89abcdef0123456789abcdef0123456789abcdef"
SHA_C = "ABCDEF0123456789ABCDEF0123456789ABCDEF01"


def _spec(**extra):
    spec = {
        "name": "gitpin",
        "base": {"machine": "main", "type": "replay"},
        "arms": [{"id": "a"}, {"id": "b"}],
    }
    spec.update(extra)
    return spec


def _gradeable_spec(**extra):
    value = _spec(**extra)
    value["experiment"] = {
        "schema_version": 1,
        "run_class": "experiment",
        "title": "immutable source contract",
        "purpose": "prove every scientific arm executes registered code",
        "experiment_id": "rf-issue-7",
        "expected_evidence": ["report.json"],
        "owner_refs": ["ablator#7"],
        "arm": {
            "comparison_role": "candidate",
            "manipulation": "source pin contract",
        },
    }
    return value


def test_spec_level_git_sha_is_frozen_into_every_job():
    jobs = specmod.expand_spec(_spec(git_sha=SHA_A))
    assert [job["requested_git_sha"] for job in jobs] == [SHA_A, SHA_A]
    assert all("git_repo" not in job for job in jobs)


def test_structured_git_target_persists_repo_and_normalizes_sha():
    jobs = specmod.expand_spec(_spec(git={
        "repo": "https://github.com/example/project.git",
        "sha": SHA_C,
    }))
    assert jobs[0]["requested_git_sha"] == SHA_C.lower()
    assert jobs[0]["git_repo"] == "https://github.com/example/project.git"


def test_git_target_inheritance_is_arm_then_base_then_spec():
    spec = _spec(git_sha=SHA_A)
    spec["base"]["git_sha"] = SHA_B
    spec["arms"][1]["git_sha"] = SHA_C
    jobs = specmod.expand_spec(spec)
    assert jobs[0]["requested_git_sha"] == SHA_B
    assert jobs[1]["requested_git_sha"] == SHA_C.lower()


def test_legacy_spec_has_no_git_fields():
    jobs = specmod.expand_spec(_spec())
    assert all("requested_git_sha" not in job for job in jobs)
    assert all("git_repo" not in job for job in jobs)


def test_gradeable_spec_requires_immutable_git_target():
    with pytest.raises(SystemExit, match="requires an immutable Git target"):
        specmod.expand_spec(_gradeable_spec())


def test_non_gradeable_declaration_can_remain_unpinned():
    value = _spec()
    value["experiment"] = {
        "schema_version": 1,
        "run_class": "developer_smoke",
        "title": "local smoke",
        "purpose": "debug runner wiring",
    }
    jobs = specmod.expand_spec(value)
    assert all("requested_git_sha" not in job for job in jobs)


def test_sequential_dependency_chain_rejects_mixed_git_targets():
    value = _spec(parallel=False, git_sha=SHA_A)
    value["arms"][1]["git_sha"] = SHA_B
    with pytest.raises(SystemExit, match="dependency chain changes Git target"):
        specmod.expand_spec(value)


def test_parallel_arms_may_explicitly_target_distinct_commits():
    value = _spec(parallel=True, git_sha=SHA_A)
    value["arms"][1]["git_sha"] = SHA_B
    jobs = specmod.expand_spec(value)
    assert [job["requested_git_sha"] for job in jobs] == [SHA_A, SHA_B]


def test_queue_append_rejects_dependency_git_target_drift(tmp_path):
    queue = Queue(str(tmp_path / "queue.jsonl"))
    with pytest.raises(SystemExit, match="dependency chain changes Git target"):
        queue.append([
            {"id": "parent", "status": "pending", "requested_git_sha": SHA_A},
            {
                "id": "child",
                "status": "pending",
                "depends_on": "parent",
                "requested_git_sha": SHA_B,
            },
        ])


def test_queue_append_checks_dependency_already_in_queue(tmp_path):
    queue = Queue(str(tmp_path / "queue.jsonl"))
    queue.append([
        {"id": "parent", "status": "done", "requested_git_sha": SHA_A},
    ])
    with pytest.raises(SystemExit, match="dependency chain changes Git target"):
        queue.append([
            {
                "id": "child",
                "status": "pending",
                "depends_on": "parent",
                "requested_git_sha": SHA_B,
            },
        ])


def test_requested_git_target_cannot_change_after_enqueue(tmp_path):
    queue = Queue(str(tmp_path / "queue.jsonl"))
    queue.append([
        {
            "id": "pinned",
            "status": "pending",
            "requested_git_sha": SHA_A,
            "git_repo": "https://github.com/example/project.git",
        },
    ])
    with pytest.raises(SystemExit, match="immutable requested_git_sha"):
        queue.update("pinned", requested_git_sha=SHA_B)
    with pytest.raises(SystemExit, match="immutable git_repo"):
        queue.update("pinned", git_repo="https://github.com/example/other.git")


@pytest.mark.parametrize("value", [
    "deadbeef",
    "g" * 40,
    "0" * 39,
    "0" * 41,
    None,
    123,
])
def test_invalid_git_sha_is_rejected(value):
    with pytest.raises(SystemExit, match="full 40-character hexadecimal"):
        specmod.expand_spec(_spec(git_sha=value))


def test_same_scope_cannot_mix_compact_and_structured_forms():
    with pytest.raises(SystemExit, match="only one"):
        specmod.expand_spec(_spec(git_sha=SHA_A, git={"sha": SHA_A}))


def test_structured_git_rejects_unknown_fields():
    with pytest.raises(SystemExit, match="unsupported git field"):
        specmod.expand_spec(_spec(git={"sha": SHA_A, "branch": "main"}))


def test_structured_git_rejects_empty_repo():
    with pytest.raises(SystemExit, match="git.repo"):
        specmod.expand_spec(_spec(git={"sha": SHA_A, "repo": "  "}))
