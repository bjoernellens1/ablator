"""Guided first-run config creation: prompts for the fields needed for
k8s dispatch and writes a ready-to-use TOML config.

Split into pure/testable pieces:
  - `WIZARD_FIELDS` -- the prompt list (label, config path, default,
    required-ness).
  - `collect_answers()` -- runs the prompts via an injectable `input_fn`
    (defaults to stdlib `input`), returns a flat dict of raw answers.
  - `build_config_dict()` -- pure function: raw answers -> the nested
    config dict `ablator.config.load_config()` would have produced from
    a hand-written TOML file.
  - `render_toml()` -- pure function: nested config dict -> TOML text.
    Hand-rolled and deliberately narrow (covers exactly the shapes this
    wizard produces: strings, ints, bools, lists of strings, one list of
    flat string-valued dicts for extra_volumes) -- NOT a general TOML
    serializer.
  - `write_config()` -- render_toml() + write to disk.

`run_wizard()` ties these together for interactive use (from the TUI's
first-run flow or a bare `ablator` invocation with no config yet).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class WizardField:
    key: str            # dotted path into the answers dict, e.g. "machine.namespace"
    prompt: str
    default: str = ""
    required: bool = True


# Kept in prompt order; mirrors examples/pytorch-generic.toml's fields.
WIZARD_FIELDS: list[WizardField] = [
    WizardField("queue_path", "Path for ablator's local queue file",
               os.path.expanduser("~/ablator/queue/queue.jsonl")),
    WizardField("machine_name", "Name for this k8s dispatch target", "a100cluster"),
    WizardField("namespace", "Kubernetes namespace", "cps-users"),
    WizardField("kai_queue", "KAI Scheduler queue", "batch"),
    WizardField("priority_class", "KAI priorityClassName", "kai-batch-low"),
    WizardField("image", "Container image for your training job", ""),
    WizardField("gpu_count", "GPUs per job", "1"),
    WizardField("image_pull_secret",
               "Image pull secret (blank if image is public)", "", required=False),
    WizardField("dataset_pvc_name",
               "Shared dataset/checkpoint PVC claim name (blank to skip)",
               "", required=False),
    WizardField("dataset_mount_path", "Mount path for that PVC in the container",
               "/mnt/data", required=False),
]


def collect_answers(input_fn: Callable[[str], str] = input) -> dict[str, str]:
    """Run the prompts, returning {field.key: answer}. Blank input on a
    field keeps its default."""
    answers: dict[str, str] = {}
    for f in WIZARD_FIELDS:
        suffix = f" [{f.default}]" if f.default else ""
        raw = input_fn(f"{f.prompt}{suffix}: ").strip()
        value = raw or f.default
        if f.required and not value:
            raise ValueError(f"'{f.prompt}' is required")
        answers[f.key] = value
    return answers


def build_config_dict(answers: dict[str, str]) -> dict:
    """Raw wizard answers -> the same nested shape `load_config()` returns
    for a hand-written TOML file (queue/machines/types keys)."""
    machine_name = answers.get("machine_name") or "a100cluster"
    mcfg: dict = {
        "backend": "k8s",
        "namespace": answers["namespace"],
        "kai_queue": answers["kai_queue"],
        "priority_class": answers["priority_class"],
        "image": answers["image"],
        "gpu_count": int(answers.get("gpu_count") or 1),
    }
    if answers.get("image_pull_secret"):
        mcfg["image_pull_secret"] = answers["image_pull_secret"]
    if answers.get("dataset_pvc_name"):
        mcfg["extra_volumes"] = [{
            "name": "dataset",
            "claim_name": answers["dataset_pvc_name"],
            "mount_path": answers.get("dataset_mount_path") or "/mnt/data",
        }]
    return {
        "queue": {"path": answers["queue_path"]},
        "machines": {
            "local": {"hostname_patterns": ["*"]},
            machine_name: mcfg,
        },
        "types": {
            "train": {
                "cwd": "/workspace",
                "command": ["python", "train.py",
                           "--epochs", "{iterations}",
                           "--output-dir", "{model_path}", "{extra_args}"],
            },
        },
    }


def _toml_scalar(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    raise TypeError(f"unsupported scalar type for TOML: {type(v)}")


def _toml_array(items: list) -> str:
    parts = []
    for item in items:
        if isinstance(item, dict):
            inline = ", ".join(f"{k} = {_toml_scalar(v)}" for k, v in item.items())
            parts.append("{ " + inline + " }")
        else:
            parts.append(_toml_scalar(item))
    return "[" + ", ".join(parts) + "]"


def render_toml(cfg: dict) -> str:
    """Narrow, hand-rolled serializer covering exactly the shapes
    `build_config_dict()` produces (queue/[machines.*]/[types.*] tables of
    strings/ints/bools/string-lists/one dict-list level). Not general."""
    lines: list[str] = []

    def emit_table(header: str, table: dict) -> None:
        scalars = {k: v for k, v in table.items()
                  if not isinstance(v, dict)}
        nested = {k: v for k, v in table.items() if isinstance(v, dict)}
        lines.append(f"[{header}]")
        for k, v in scalars.items():
            if isinstance(v, list):
                lines.append(f"{k} = {_toml_array(v)}")
            else:
                lines.append(f"{k} = {_toml_scalar(v)}")
        lines.append("")
        for k, v in nested.items():
            emit_table(f"{header}.{k}", v)

    for top_key in ("queue", "machines", "types"):
        table = cfg.get(top_key, {})
        if top_key in ("machines", "types"):
            for name, sub in table.items():
                emit_table(f"{top_key}.{name}", sub)
        else:
            emit_table(top_key, table)
    return "\n".join(lines).rstrip() + "\n"


def write_config(cfg: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(render_toml(cfg))


def run_wizard(config_path: str, input_fn: Callable[[str], str] = input,
              print_fn: Callable[[str], None] = print) -> str:
    """Full interactive flow: prompt, build, write. Returns the path
    written to."""
    print_fn("No ablator config found -- let's create one for k8s dispatch "
             "against the CPS GPU cluster.")
    answers = collect_answers(input_fn)
    cfg = build_config_dict(answers)
    write_config(cfg, config_path)
    print_fn(f"Wrote {config_path}")
    return config_path
