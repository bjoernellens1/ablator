"""Unit tests for the TUI's data/logic layer: contexts.py, wizard.py,
queue_view.py. Deliberately does NOT import ablator.tui.app (that needs
`textual`, which is an optional extra and may not be installed) -- the
core logic modules here never import textual and are always testable.
"""
from __future__ import annotations

import os
import subprocess

import pytest

from ablator.tui import contexts as ctxmod
from ablator.tui import queue_view as qvmod
from ablator.tui import wizard as wizardmod
from ablator.queue import Queue


# ------------------------------------------------------------------ contexts

class _FakeCompleted:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout, self.stderr, self.returncode = stdout, stderr, returncode


def test_list_contexts_parses_names(monkeypatch):
    def fake_run(args, **kw):
        assert args[:3] == ["kubectl", "config", "get-contexts"]
        return _FakeCompleted(stdout="ctx-a\nctx-b\n")
    monkeypatch.setattr(subprocess, "run", fake_run)
    assert ctxmod.list_contexts() == ["ctx-a", "ctx-b"]


def test_current_context_returns_none_on_error(monkeypatch):
    def fake_run(args, **kw):
        return _FakeCompleted(returncode=1, stderr="no context")
    monkeypatch.setattr(subprocess, "run", fake_run)
    assert ctxmod.current_context() is None


def test_current_context_strips_output(monkeypatch):
    def fake_run(args, **kw):
        return _FakeCompleted(stdout="my-ctx\n")
    monkeypatch.setattr(subprocess, "run", fake_run)
    assert ctxmod.current_context() == "my-ctx"


def test_use_context_raises_kubectl_error_on_failure(monkeypatch):
    def fake_run(args, **kw):
        return _FakeCompleted(returncode=1, stderr="no such context \"bogus\"")
    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(ctxmod.KubectlError):
        ctxmod.use_context("bogus")


def test_use_context_succeeds(monkeypatch):
    calls = []
    def fake_run(args, **kw):
        calls.append(args)
        return _FakeCompleted()
    monkeypatch.setattr(subprocess, "run", fake_run)
    ctxmod.use_context("good-ctx")
    assert calls[0] == ["kubectl", "config", "use-context", "good-ctx"]


def test_kubectl_missing_raises_kubectl_error(monkeypatch):
    def fake_run(args, **kw):
        raise FileNotFoundError()
    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(ctxmod.KubectlError):
        ctxmod.list_contexts()


# -------------------------------------------------------------------- wizard

def test_collect_answers_uses_defaults_on_blank_input():
    def fake_input(prompt):
        return "my-image:1" if "image" in prompt.lower() and "pull" not in prompt.lower() else ""
    answers = wizardmod.collect_answers(input_fn=fake_input)
    assert answers["namespace"] == "cps-users"
    assert answers["kai_queue"] == "batch"
    assert answers["priority_class"] == "kai-batch-low"


def test_collect_answers_required_field_blank_raises():
    # 'image' has no default and is required -> blank answer must raise,
    # not silently produce an unusable config.
    with pytest.raises(ValueError):
        wizardmod.collect_answers(input_fn=lambda p: "")


def test_build_config_dict_shape():
    answers = {
        "queue_path": "/tmp/x/queue.jsonl", "machine_name": "a100cluster",
        "namespace": "cps-users", "kai_queue": "batch",
        "priority_class": "kai-batch-low", "image": "img:1", "gpu_count": "1",
        "image_pull_secret": "", "dataset_pvc_name": "", "dataset_mount_path": "",
    }
    cfg = wizardmod.build_config_dict(answers)
    assert cfg["queue"]["path"] == "/tmp/x/queue.jsonl"
    assert cfg["machines"]["local"]["hostname_patterns"] == ["*"]
    mcfg = cfg["machines"]["a100cluster"]
    assert mcfg["backend"] == "k8s"
    assert mcfg["gpu_count"] == 1
    assert "image_pull_secret" not in mcfg
    assert "extra_volumes" not in mcfg


def test_build_config_dict_optional_fields_included_when_set():
    answers = {
        "queue_path": "/tmp/x/queue.jsonl", "machine_name": "a100cluster",
        "namespace": "cps-users", "kai_queue": "batch",
        "priority_class": "kai-batch-low", "image": "img:1", "gpu_count": "2",
        "image_pull_secret": "regcred", "dataset_pvc_name": "shared-pvc",
        "dataset_mount_path": "/mnt/data",
    }
    cfg = wizardmod.build_config_dict(answers)
    mcfg = cfg["machines"]["a100cluster"]
    assert mcfg["image_pull_secret"] == "regcred"
    assert mcfg["extra_volumes"] == [
        {"name": "dataset", "claim_name": "shared-pvc", "mount_path": "/mnt/data"}]


def test_render_toml_and_write_config_roundtrip(tmp_path):
    pytest.importorskip("tomllib")
    from ablator import config as cfgmod
    answers = {
        "queue_path": str(tmp_path / "queue.jsonl"), "machine_name": "a100cluster",
        "namespace": "cps-users", "kai_queue": "batch",
        "priority_class": "kai-batch-low", "image": "img:1", "gpu_count": "1",
        "image_pull_secret": "", "dataset_pvc_name": "", "dataset_mount_path": "",
    }
    cfg = wizardmod.build_config_dict(answers)
    path = str(tmp_path / "config.toml")
    wizardmod.write_config(cfg, path)
    loaded = cfgmod.load_config(path)
    assert loaded["queue"]["path"] == str(tmp_path / "queue.jsonl")
    assert loaded["machines"]["a100cluster"]["namespace"] == "cps-users"
    assert loaded["machines"]["local"]["hostname_patterns"] == ["*"]


def test_run_wizard_end_to_end(tmp_path):
    pytest.importorskip("tomllib")
    from ablator import config as cfgmod
    path = str(tmp_path / "config.toml")
    answers_queue = iter(["", "", "", "", "", "my-image:1", "1", "", "", ""])
    printed = []
    wizardmod.run_wizard(path, input_fn=lambda p: next(answers_queue),
                         print_fn=printed.append)
    assert os.path.exists(path)
    loaded = cfgmod.load_config(path)
    assert loaded["machines"]["a100cluster"]["image"] == "my-image:1"
    assert any("Wrote" in line for line in printed)


# ---------------------------------------------------------------- queue_view

def test_job_row_shape():
    job = {"id": "j1", "status": "running", "machine": "a100cluster",
          "claimed_by": "a100cluster", "claimed_at": "2026-01-01T00:00:00",
          "lane": 2}
    row = qvmod.job_row(job)
    assert row[0] == "j1"
    assert row[2] == "running"
    assert row[3] == "a100cluster"


def test_queue_rows_filters_by_ablation_name():
    jobs = [{"id": "foo_ctrl", "ablation": "foo", "status": "done"},
           {"id": "bar_ctrl", "ablation": "bar", "status": "done"}]
    rows = qvmod.queue_rows(jobs, name="foo")
    assert len(rows) == 1 and rows[0][0] == "foo_ctrl"


def test_running_rows_only_includes_running():
    jobs = [{"id": "a", "status": "running"}, {"id": "b", "status": "done"}]
    rows = qvmod.running_rows(jobs)
    assert [r[0] for r in rows] == ["a"]


def test_load_jobs_reads_the_configured_queue(tmp_path):
    q = Queue(str(tmp_path / "queue.jsonl"))
    q.append([{"id": "j1", "status": "pending", "machine": "any", "type": "train"}])
    cfg = {"queue": {"path": str(tmp_path / "queue.jsonl")}}
    jobs = qvmod.load_jobs(cfg)
    assert len(jobs) == 1 and jobs[0]["id"] == "j1"


def test_k8s_job_name_matches_runner_naming():
    from ablator import runner as runnermod
    assert qvmod.k8s_job_name("myjob_ctrl") == runnermod._k8s_job_name("myjob_ctrl")
