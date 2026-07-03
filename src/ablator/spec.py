"""Ablation spec expansion.

Spec format (JSON):
  {
    "name": "consol_sweep",
    "parallel": true,               // false = strictly sequential chain
    "base": {"type": "replay", "scene": "/mnt/.../fr3",
             "iterations": 30000, "machine": "any",
             "base_args": "--opacity_reg 0.001"},
    "arms": [
      {"id": "ctrl",   "extra_args": ""},
      {"id": "consol", "extra_args": "--foo bar",
       "machine": "main", "type": "bag", "iterations": 60000}  // overrides
    ]
  }

model_path per job comes from a template (config [queue]
model_path_template, default "output/scratch/{name}_{arm}").
"""
from __future__ import annotations

import json

DEFAULT_MODEL_PATH_TEMPLATE = "output/scratch/{name}_{arm}"


def load_spec(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def expand_spec(spec: dict,
                model_path_template: str = DEFAULT_MODEL_PATH_TEMPLATE) -> list[dict]:
    """Pure expansion of an ablation spec into queue-job dicts."""
    name = spec["name"]
    base = spec.get("base", {})
    parallel = spec.get("parallel", True)
    jobs: list[dict] = []
    prev_id: str | None = None
    seen_ids: set[str] = set()
    for arm in spec["arms"]:
        arm_id = arm["id"]
        if arm_id in seen_ids:
            raise SystemExit(f"spec '{name}': duplicate arm id '{arm_id}'")
        seen_ids.add(arm_id)
        job_id = f"{name}_{arm_id}"
        base_args = arm.get("base_args", base.get("base_args", "")).strip()
        extra = " ".join(x for x in (base_args, arm.get("extra_args", "").strip()) if x)
        job = {
            "id": job_id,
            "ablation": name,
            "machine": arm.get("machine", base.get("machine", "any")),
            "type": arm.get("type", base.get("type", "replay")),
            "scene": arm.get("scene", base.get("scene", "")),
            "model_path": model_path_template.format(name=name, arm=arm_id, id=job_id),
            "extra_args": extra,
            "iterations": arm.get("iterations", base.get("iterations", 30000)),
            "status": "pending",
        }
        if not parallel and prev_id is not None:
            job["depends_on"] = prev_id
        jobs.append(job)
        prev_id = job_id
    return jobs
