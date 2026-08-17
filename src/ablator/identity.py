"""Stable source identities for Ablator runner provenance."""
from __future__ import annotations

import hashlib
from pathlib import Path


def package_source_sha256(package_dir: str | Path | None = None) -> str:
    """Hash the installed Ablator Python sources deterministically.

    Git metadata is the strongest identity for an editable checkout, but a
    wheel/site-packages installation may have no ``.git`` directory.  This
    content fingerprint provides the same immutable source identity in both
    cases without introducing a packaging or Git dependency.
    """
    root = Path(package_dir).resolve() if package_dir else Path(__file__).resolve().parent
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*.py") if path.is_file())
    for path in files:
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()
