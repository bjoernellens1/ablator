"""Optional k9s-style TUI for ablator, scoped to ablator's own domain
(queue jobs, runs, config, kubeconfig context) -- not a general Kubernetes
resource browser.

This subpackage is the ONLY part of ablator allowed to depend on a
non-stdlib package (`textual`, an optional extra: `pip install
ablator[tui]`). The host runner (`ablator.runner`/`ablator.cli run`)
never imports anything here and must keep working with zero installs.
"""
from __future__ import annotations
