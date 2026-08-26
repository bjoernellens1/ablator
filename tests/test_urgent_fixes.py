"""Tests for the pre-dispatch urgent-fix currency gate (urgent_fixes.py)
and its wiring into runner.run_loop().

Uses real throwaway git repos (cheap, and exercises actual `git
merge-base --is-ancestor` / `git pull --ff-only` semantics rather than
mocking subprocess, since this module's whole job is to get those git
semantics right)."""
import subprocess

import pytest

from ablator import urgent_fixes as ufmod
from ablator.queue import Queue, is_paused, pause_flag_path


def _git(cwd, *args):
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True,
                       text=True, check=True)
    return r.stdout.strip()


def _init_repo(path):
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "t@example.com")
    _git(path, "config", "user.name", "t")
    (path / "f.txt").write_text("0\n")
    _git(path, "add", "f.txt")
    _git(path, "commit", "-q", "-m", "init")
    return _git(path, "rev-parse", "HEAD")


def _commit(path, msg):
    (path / "f.txt").write_text(msg + "\n")
    _git(path, "add", "f.txt")
    _git(path, "commit", "-q", "-m", msg)
    return _git(path, "rev-parse", "HEAD")


# ------------------------------------------------------------- unit-level

def test_missing_fixes_empty_when_ancestor(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    fix_sha = _commit(repo, "the fix")
    assert ufmod.missing_fixes(str(repo), [{"sha": fix_sha}]) == []


def test_missing_fixes_reports_unreached_commit(tmp_path):
    origin = tmp_path / "origin"
    _init_repo(origin)
    fix_sha = _commit(origin, "the fix")

    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(origin), str(clone)], check=True)
    # Roll the clone back before the fix commit landed upstream.
    parent = _git(clone, "rev-parse", "HEAD~1")
    _git(clone, "reset", "-q", "--hard", parent)

    missing = ufmod.missing_fixes(str(clone), [{"sha": fix_sha}])
    assert len(missing) == 1


def test_missing_fixes_unknown_sha_counts_as_missing(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    assert ufmod.missing_fixes(str(repo), [{"sha": "deadbeef" * 5}]) != []


def test_missing_fixes_auto_sync_ref_behind(tmp_path):
    origin = tmp_path / "origin"
    _init_repo(origin)
    _commit(origin, "a commit the clone never gets")

    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(origin), str(clone)], check=True)
    branch = _git(clone, "rev-parse", "--abbrev-ref", "HEAD")
    parent = _git(clone, "rev-parse", "HEAD~1")
    _git(clone, "reset", "-q", "--hard", parent)
    _git(clone, "fetch", "-q")  # so origin/<branch> reflects the new commit

    assert ufmod.missing_fixes(str(clone), [], auto_sync_ref=f"origin/{branch}")


def test_missing_fixes_auto_sync_ref_current(tmp_path):
    origin = tmp_path / "origin"
    _init_repo(origin)

    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(origin), str(clone)], check=True)
    branch = _git(clone, "rev-parse", "--abbrev-ref", "HEAD")

    assert ufmod.missing_fixes(str(clone), [], auto_sync_ref=f"origin/{branch}") == []


# --------------------------------------------------------- enforce_urgent_fixes

def _cfg(tmp_path, repo_cwd=None, fixes=None, auto_sync_ref=None):
    uf = {}
    if repo_cwd:
        uf = {"repo_cwd": repo_cwd, "fixes": fixes or []}
        if auto_sync_ref:
            uf["auto_sync_ref"] = auto_sync_ref
    return {
        "queue": {"path": str(tmp_path / "queue.jsonl")},
        "urgent_fixes": uf,
    }


def test_enforce_noop_when_not_configured(tmp_path):
    cfg = _cfg(tmp_path)
    q = Queue(str(tmp_path / "queue.jsonl"))
    assert ufmod.enforce_urgent_fixes(cfg, "main", q) is True


def _external_researchflow_job():
    from ablator import experiment_declaration as declarations
    job = {
        "id": "rf-job",
        "external_id": "rf-job",
        "external_schema": "ablator.external-job/v1",
        "external_metadata": {},
        "params": {},
        "source": "external",
        "type": "researchflow",
        "machine": "main",
        "lane": 2,
        "depends_on": None,
        "status": "pending",
    }
    submission, digest = declarations.freeze_external_submission(job)
    job["submission_provenance"] = submission
    job["external_spec_sha256"] = digest
    return job


def test_queue_can_claim_external_researchflow_during_urgent_fix_pause(tmp_path):
    from ablator.queue import write_pause_flag
    path = str(tmp_path / "queue.jsonl")
    q = Queue(path)
    q.append([_external_researchflow_job()])
    write_pause_flag(path, "main", "urgent_fix_unsynced", "stale checkout")

    assert q.claim_next("main") is None
    job = q.claim_next("main", allow_pinned_git_while_paused=True)
    assert job["id"] == "rf-job"

    ordinary_path = str(tmp_path / "ordinary-queue.jsonl")
    ordinary_q = Queue(ordinary_path)
    ordinary_q.append([{
        "id": "replay-job",
        "source": "internal",
        "type": "replay",
        "machine": "main",
        "status": "pending",
    }])
    write_pause_flag(ordinary_path, "main", "urgent_fix_unsynced", "stale checkout")
    assert ordinary_q.claim_next(
        "main", allow_pinned_git_while_paused=True
    ) is None


def test_manual_pause_still_blocks_external_researchflow(tmp_path):
    from ablator.queue import write_pause_flag
    path = str(tmp_path / "queue.jsonl")
    q = Queue(path)
    q.append([_external_researchflow_job()])
    write_pause_flag(path, "main", "manual_pause", "operator request")

    assert q.claim_next(
        "main", allow_pinned_git_while_paused=True
    ) is None


def test_mutated_ordinary_job_cannot_forge_researchflow_pause_bypass(tmp_path):
    from ablator.queue import write_pause_flag
    path = str(tmp_path / "queue.jsonl")
    q = Queue(path)
    q.append([{
        "id": "ordinary",
        "source": "internal",
        "type": "replay",
        "machine": "main",
        "status": "pending",
    }])
    q.update("ordinary", source="external", type="researchflow")
    write_pause_flag(path, "main", "urgent_fix_unsynced", "stale checkout")

    assert q.claim_next(
        "main", allow_pinned_git_while_paused=True
    ) is None


def test_enforce_noop_when_already_current(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    fix_sha = _commit(repo, "the fix")
    cfg = _cfg(tmp_path, repo_cwd=str(repo), fixes=[{"sha": fix_sha}])
    q = Queue(str(tmp_path / "queue.jsonl"))
    assert ufmod.enforce_urgent_fixes(cfg, "main", q) is True
    assert not is_paused(cfg["queue"]["path"], "main")


def test_enforce_auto_pulls_clean_fast_forward(tmp_path):
    origin = tmp_path / "origin"
    _init_repo(origin)
    fix_sha = _commit(origin, "the fix")

    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(origin), str(clone)], check=True)
    parent = _git(clone, "rev-parse", "HEAD~1")
    _git(clone, "reset", "-q", "--hard", parent)
    assert ufmod.missing_fixes(str(clone), [{"sha": fix_sha}])

    cfg = _cfg(tmp_path, repo_cwd=str(clone), fixes=[{"sha": fix_sha}])
    q = Queue(str(tmp_path / "queue.jsonl"))
    result = ufmod.enforce_urgent_fixes(cfg, "main", q)
    assert result is True
    assert not is_paused(cfg["queue"]["path"], "main")
    # the clone's HEAD should now include the fix commit
    assert ufmod.missing_fixes(str(clone), [{"sha": fix_sha}]) == []


def test_enforce_pauses_on_dirty_tree(tmp_path):
    origin = tmp_path / "origin"
    _init_repo(origin)
    fix_sha = _commit(origin, "the fix")

    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(origin), str(clone)], check=True)
    parent = _git(clone, "rev-parse", "HEAD~1")
    _git(clone, "reset", "-q", "--hard", parent)
    (clone / "f.txt").write_text("dirty edit\n")  # uncommitted change

    cfg = _cfg(tmp_path, repo_cwd=str(clone), fixes=[{"sha": fix_sha}])
    q = Queue(str(tmp_path / "queue.jsonl"))
    result = ufmod.enforce_urgent_fixes(cfg, "main", q)
    assert result is False
    assert is_paused(cfg["queue"]["path"], "main")
    # never silently mutated the dirty file
    assert (clone / "f.txt").read_text() == "dirty edit\n"


def test_enforce_pauses_on_diverged_history(tmp_path):
    """A clone with a local commit the origin doesn't have -- ff-only
    pull cannot resolve this; the gate must pause, never force/rebase."""
    origin = tmp_path / "origin"
    _init_repo(origin)
    fix_sha = _commit(origin, "the fix")

    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(origin), str(clone)], check=True)
    parent = _git(clone, "rev-parse", "HEAD~1")
    _git(clone, "reset", "-q", "--hard", parent)
    _commit(clone, "divergent local commit")  # diverges from origin

    cfg = _cfg(tmp_path, repo_cwd=str(clone), fixes=[{"sha": fix_sha}])
    q = Queue(str(tmp_path / "queue.jsonl"))
    result = ufmod.enforce_urgent_fixes(cfg, "main", q)
    assert result is False
    assert is_paused(cfg["queue"]["path"], "main")


def test_enforce_respects_existing_pause_flag(tmp_path):
    from ablator.queue import write_pause_flag
    repo = tmp_path / "repo"
    _init_repo(repo)
    fix_sha = _commit(repo, "the fix")
    cfg = _cfg(tmp_path, repo_cwd=str(repo), fixes=[{"sha": "unreachable" * 4}])
    q = Queue(str(tmp_path / "queue.jsonl"))
    write_pause_flag(cfg["queue"]["path"], "main", "manual_pause", "operator")
    # Already paused for an unrelated reason -- gate must not attempt any
    # git operations or double-pause; just report unsafe-to-dispatch.
    assert ufmod.enforce_urgent_fixes(cfg, "main", q) is False


def test_enforce_auto_pulls_via_auto_sync_ref_no_pinned_fixes(tmp_path):
    """auto_sync_ref alone (no [[fixes]] entries) drives the same safe
    fetch/ff-pull/pause machinery -- this is the actual fix for the
    2026-08-12 incident (a merged PR nobody remembered to pin)."""
    origin = tmp_path / "origin"
    _init_repo(origin)
    _commit(origin, "a commit no human registered as an urgent fix")

    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(origin), str(clone)], check=True)
    branch = _git(clone, "rev-parse", "--abbrev-ref", "HEAD")
    parent = _git(clone, "rev-parse", "HEAD~1")
    _git(clone, "reset", "-q", "--hard", parent)

    cfg = _cfg(tmp_path, repo_cwd=str(clone), auto_sync_ref=f"origin/{branch}")
    q = Queue(str(tmp_path / "queue.jsonl"))
    result = ufmod.enforce_urgent_fixes(cfg, "main", q)
    assert result is True
    assert not is_paused(cfg["queue"]["path"], "main")
    assert ufmod.missing_fixes(str(clone), [], auto_sync_ref=f"origin/{branch}") == []


def test_enforce_detects_gap_when_local_origin_ref_is_stale(tmp_path):
    """Regression for splatograph#752: the origin advances AFTER the clone
    was made, with no fetch in between, so the clone's locally-cached
    refs/remotes/origin/<branch> is stale and still equals local HEAD.

    Before the fix, missing_fixes()'s first call resolved auto_sync_ref
    from that stale cached ref (identical to HEAD -> "nothing missing")
    and enforce_urgent_fixes returned True immediately without ever
    fetching -- permanently blind to the real, growing gap on origin,
    since fetching only happened as a *consequence* of already believing
    something was missing. This is a self-reinforcing deadlock on any
    host that goes idle long enough for its cached remote-tracking ref to
    go stale relative to the true remote."""
    origin = tmp_path / "origin"
    _init_repo(origin)

    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(origin), str(clone)], check=True)
    branch = _git(clone, "rev-parse", "--abbrev-ref", "HEAD")

    # Clone's cached origin/<branch> == its own HEAD right after cloning --
    # exactly the "looks current" trap. Now origin moves on without the
    # clone ever fetching.
    fix_sha = _commit(origin, "a real fix landed upstream after the clone")

    cfg = _cfg(tmp_path, repo_cwd=str(clone), auto_sync_ref=f"origin/{branch}")
    q = Queue(str(tmp_path / "queue.jsonl"))
    result = ufmod.enforce_urgent_fixes(cfg, "main", q)
    assert result is True
    assert not is_paused(cfg["queue"]["path"], "main")
    # The clone must have actually fetched and fast-forwarded -- not just
    # returned True on stale information.
    assert _git(clone, "rev-parse", "HEAD") == fix_sha


def test_enforce_pauses_on_dirty_tree_via_auto_sync_ref(tmp_path):
    origin = tmp_path / "origin"
    _init_repo(origin)
    _commit(origin, "a commit no human registered as an urgent fix")

    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(origin), str(clone)], check=True)
    branch = _git(clone, "rev-parse", "--abbrev-ref", "HEAD")
    parent = _git(clone, "rev-parse", "HEAD~1")
    _git(clone, "reset", "-q", "--hard", parent)
    (clone / "f.txt").write_text("dirty edit\n")

    cfg = _cfg(tmp_path, repo_cwd=str(clone), auto_sync_ref=f"origin/{branch}")
    q = Queue(str(tmp_path / "queue.jsonl"))
    result = ufmod.enforce_urgent_fixes(cfg, "main", q)
    assert result is False
    assert is_paused(cfg["queue"]["path"], "main")
    assert (clone / "f.txt").read_text() == "dirty edit\n"


def test_load_urgent_fixes_missing_config_returns_none(tmp_path):
    assert ufmod.load_urgent_fixes({}) == (None, [], None)
    assert ufmod.load_urgent_fixes({"urgent_fixes": {}}) == (None, [], None)


def test_load_urgent_fixes_auto_sync_ref_only_is_configured(tmp_path):
    """auto_sync_ref with no [[fixes]] entries must still be treated as
    configured (not the empty/no-op case) -- this is the whole point of
    the generalized mode."""
    repo_cwd, fixes, ref = ufmod.load_urgent_fixes(
        {"urgent_fixes": {"repo_cwd": "/some/repo", "auto_sync_ref": "origin/main"}}
    )
    assert repo_cwd == "/some/repo"
    assert fixes == []
    assert ref == "origin/main"


def test_load_urgent_fixes_per_machine_repo_cwd_override(tmp_path):
    """A machine with a different user/home (e.g. rtx3090 running as
    bjoern1) or a dedicated worktree must be able to override repo_cwd
    (and optionally auto_sync_ref) without affecting other machines."""
    cfg = {
        "urgent_fixes": {
            "repo_cwd": "/home/bjoern/git/splatograph",
            "auto_sync_ref": "origin/main",
            "machines": {
                "rtx3090": {"repo_cwd": "/home/bjoern1/git/splatograph"},
            },
        }
    }
    # Unlisted machine falls back to the global repo_cwd.
    repo_cwd, _, ref = ufmod.load_urgent_fixes(cfg, "main")
    assert repo_cwd == "/home/bjoern/git/splatograph"
    assert ref == "origin/main"

    # Overridden machine gets its own repo_cwd, inherits auto_sync_ref.
    repo_cwd, _, ref = ufmod.load_urgent_fixes(cfg, "rtx3090")
    assert repo_cwd == "/home/bjoern1/git/splatograph"
    assert ref == "origin/main"

    # No machine given (back-compat callers) -> global default.
    repo_cwd, _, _ = ufmod.load_urgent_fixes(cfg)
    assert repo_cwd == "/home/bjoern/git/splatograph"
