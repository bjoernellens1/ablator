"""Concurrency contracts for immutable source preparation."""

from concurrent.futures import ThreadPoolExecutor
import os
import subprocess

from ablator import source_checkout as source
from ablator import source_gc


def _run(*args, cwd=None):
    return subprocess.run(
        args, cwd=cwd, capture_output=True, text=True, check=True
    ).stdout.strip()


def _repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _run("git", "init", "-b", "main", cwd=repo)
    _run("git", "config", "user.email", "test@example.com", cwd=repo)
    _run("git", "config", "user.name", "Ablator Test", cwd=repo)
    shas = []
    for index in range(2):
        (repo / "payload.txt").write_text(f"{index}\n")
        _run("git", "add", "payload.txt", cwd=repo)
        _run("git", "commit", "-m", f"commit {index}", cwd=repo)
        shas.append(_run("git", "rev-parse", "HEAD", cwd=repo))
    return repo, shas


def _prepare(cfg, repo, item):
    job_id, sha = item
    return source.prepare_job_source(
        cfg,
        {"id": job_id, "requested_git_sha": sha},
        "main",
        {
            "cwd": str(repo),
            "command": ["python", "train.py"],
        },
    )


def test_parallel_same_and_different_sha_materialization_is_per_job(tmp_path):
    repo, shas = _repo(tmp_path)
    cfg = {
        "git": {"worktree_root": str(tmp_path / "cache")},
        "machines": {"main": {}},
    }
    jobs = [(f"job-{index}", shas[index % 2]) for index in range(8)]

    with ThreadPoolExecutor(max_workers=8) as pool:
        prepared = list(pool.map(lambda item: _prepare(cfg, repo, item), jobs))

    paths = [item.checkout_path for item in prepared]
    assert len(set(paths)) == len(jobs)
    assert all(item.lease is not None for item in prepared)
    assert all(source.read_source_lease(item.lease)["active"] for item in prepared)
    assert [source.capture_checkout_state(path)["commit"] for path in paths] == [
        sha for _job_id, sha in jobs
    ]


def test_gc_racing_materialization_cannot_remove_active_leases(tmp_path):
    repo, shas = _repo(tmp_path)
    cfg = {
        "git": {"worktree_root": str(tmp_path / "cache")},
        "machines": {"main": {}},
    }
    jobs = [(f"job-{index}", shas[index % 2]) for index in range(12)]

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(_prepare, cfg, repo, item) for item in jobs]
        gc_futures = [
            pool.submit(
                source_gc.gc_worktrees,
                cfg,
                "main",
                [],
                max_age_days=0,
                now=10 ** 12,
            )
            for _ in range(4)
        ]
        prepared = [future.result() for future in futures]
        results = [future.result() for future in gc_futures]

    assert all(item.checkout_path and os.path.isdir(item.checkout_path)
               for item in prepared)
    assert all(result.removed == () for result in results)
