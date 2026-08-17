from ablator import source_display as display
from ablator import cli
from ablator.tui import queue_view


SHA_A = "0123456789abcdef0123456789abcdef01234567"
SHA_B = "89abcdef0123456789abcdef0123456789abcdef"


def test_source_state_distinguishes_mutable_pending_verified_and_errors():
    assert display.source_state({}) == "mutable"
    assert display.source_state({"requested_git_sha": SHA_A}) == SHA_A[:12]
    assert display.source_state({
        "requested_git_sha": SHA_A,
        "executed_git_sha": SHA_A,
    }) == SHA_A[:12] + "="
    assert display.source_state({
        "requested_git_sha": SHA_A,
        "executed_git_sha": SHA_B,
    }) == f"{SHA_A[:12]}!={SHA_B[:12]}"
    assert display.source_state({
        "requested_git_sha": SHA_A,
        "source_prepare_error": "fetch failed",
    }) == SHA_A[:12] + "!"


def test_inspect_fields_keep_full_values():
    job = {
        "requested_git_sha": SHA_A,
        "executed_git_sha": SHA_A,
        "git_repo": "https://example.invalid/repo.git",
        "source_repo": "https://example.invalid/repo.git",
        "source_checkout": "/cache/repo/sha",
    }
    fields = display.inspect_fields(job)
    assert fields["source_mode"] == "pinned"
    assert fields["requested_git_sha"] == SHA_A
    assert fields["executed_git_sha"] == SHA_A
    assert fields["source_checkout"] == "/cache/repo/sha"


def test_tui_row_has_git_column_and_compact_state():
    job = {
        "id": "run_a",
        "lane": 2,
        "status": "pending",
        "machine": "main",
        "requested_git_sha": SHA_A,
    }
    assert queue_view.COLUMNS[-1] == "git"
    assert queue_view.job_row(job)[-1] == SHA_A[:12]


def test_cli_status_has_git_column_without_affecting_progress_lookup():
    cfg = {"queue": {}, "types": {}}
    jobs = [{
        "id": "run_a",
        "lane": 2,
        "status": "pending",
        "machine": "main",
        "requested_git_sha": SHA_A,
    }]
    text = "\n".join(cli._status_lines(cfg, jobs))
    assert "git" in text.splitlines()[0]
    assert SHA_A[:12] in text


def test_source_detail_marks_legacy_and_pinned_modes():
    assert "mutable checkout" in display.source_detail({})
    detail = display.source_detail({
        "requested_git_sha": SHA_A,
        "executed_git_sha": SHA_A,
        "source_checkout": "/cache/sha",
    })
    assert SHA_A in detail
    assert "/cache/sha" in detail
