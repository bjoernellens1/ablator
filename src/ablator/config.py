"""Host configuration loading and machine identity.

The config file (TOML or JSON) defines everything machine/workload
specific: queue path, machine identities (hostname patterns, ssh
addresses), busy guards, and job-type command templates. The ablator
package itself contains zero knowledge of any particular workload.

Default location: ~/.config/ablator/config.toml (override with
--config or env ABLATOR_CONFIG).

TOML parsing uses stdlib tomllib (Python >= 3.11). On 3.10 use a
.json config file instead.
"""
from __future__ import annotations

import fnmatch
import json
import os
import socket

DEFAULT_CONFIG_PATH = os.path.expanduser("~/.config/ablator/config.toml")


def load_config(path: str | None = None) -> dict:
    path = path or os.environ.get("ABLATOR_CONFIG") or DEFAULT_CONFIG_PATH
    if not os.path.exists(path):
        raise SystemExit(f"ablator: config file not found: {path}\n"
                         "Create one (see examples/splatograph.toml) or pass --config.")
    if path.endswith(".json"):
        with open(path) as f:
            cfg = json.load(f)
    else:
        try:
            import tomllib
        except ImportError:  # Python 3.10
            raise SystemExit("ablator: TOML config requires Python >= 3.11 (tomllib); "
                             "use a .json config on 3.10")
        with open(path, "rb") as f:
            cfg = tomllib.load(f)
    cfg.setdefault("queue", {})
    cfg.setdefault("machines", {})
    cfg.setdefault("types", {})
    cfg.setdefault("resources", {})
    if "path" not in cfg["queue"]:
        raise SystemExit(f"ablator: config {path} missing required [queue] path")
    cfg["_path"] = path
    return cfg


def queue_path(cfg: dict) -> str:
    return os.environ.get("ABLATOR_QUEUE_FILE") or cfg["queue"]["path"]


def log_dir(cfg: dict) -> str:
    return cfg["queue"].get("log_dir") or os.path.dirname(queue_path(cfg))


def machine_name(cfg: dict, hostname: str | None = None) -> str:
    """Resolve this host's machine name via hostname glob patterns.

    Each [machines.<name>] entry may define hostname_patterns = ["*r9700*"].
    First match wins (dict order); a machine with no patterns (or pattern
    "*") acts as the fallback. If nothing matches, returns "unknown".
    """
    host = (hostname or socket.gethostname()).lower()
    fallback = None
    for name, m in cfg.get("machines", {}).items():
        pats = m.get("hostname_patterns", [])
        if not pats or pats == ["*"]:
            fallback = fallback or name
            continue
        if any(fnmatch.fnmatch(host, p.lower()) for p in pats):
            return name
    return fallback or "unknown"


def machine_cfg(cfg: dict, name: str) -> dict:
    return cfg.get("machines", {}).get(name, {})


def type_cfg(cfg: dict, job_type: str, machine: str) -> dict:
    """Job-type config with per-machine overrides merged in (shallow)."""
    base = dict(cfg.get("types", {}).get(job_type) or {})
    if not base:
        raise KeyError(f"job type '{job_type}' not defined in config")
    override = (base.pop("machines", {}) or {}).get(machine, {})
    merged = dict(base)
    for k, v in override.items():
        if k == "env":
            merged["env"] = {**base.get("env", {}), **v}
        else:
            merged[k] = v
    return merged
