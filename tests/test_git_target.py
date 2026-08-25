"""Tests for immutable Git targets in ablation specs."""

import pytest

from ablator import spec as specmod


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
