"""Tests for installation-independent Ablator runner identity."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ablator.external import capture_runner_provenance
from ablator.identity import package_source_sha256


def test_package_source_fingerprint_is_stable_and_content_sensitive(tmp_path: Path) -> None:
    package = tmp_path / "ablator"
    package.mkdir()
    (package / "a.py").write_text("A=1\n")
    (package / "b.py").write_text("B=2\n")
    first = package_source_sha256(package)
    assert first == package_source_sha256(package)
    (package / "b.py").write_text("B=3\n")
    assert package_source_sha256(package) != first


def test_runner_identity_is_complete_without_requiring_git(tmp_path: Path) -> None:
    config = tmp_path / "ablator.json"
    config.write_text(json.dumps({"queue": {"path": str(tmp_path / "queue.jsonl")}}))
    cfg = {"_path": str(config)}
    provenance = capture_runner_provenance(cfg, "main")
    assert provenance["source_sha256"]
    assert provenance["config_sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert provenance["identity_complete"] is True
