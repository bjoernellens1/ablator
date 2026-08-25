"""Ablation spec expansion.

Spec format (JSON):
  {
    "name": "consol_sweep",
    "parallel": true,               // false = strictly sequential chain
    "git_sha": "0123456789abcdef0123456789abcdef01234567", // optional
    "base": {"type": "replay", "scene": "/mnt/.../fr3",
             "iterations": 30000, "machine": "any",
             "base_args": "--opacity_reg 0.001"},
    "arms": [
      {"id": "ctrl",   "extra_args": ""},
      {"id": "consol", "extra_args": "--foo bar",
       "machine": "main", "type": "bag", "iterations": 60000}  // overrides
    ]
  }

A Git target may also use the structured form::

  "git": {
    "repo": "https://github.com/example/project.git",
    "sha": "0123456789abcdef0123456789abcdef01234567"
  }

Git targets inherit spec -> base -> arm, with the nearest declaration winning.
Only immutable full 40-character commit SHAs are accepted.  Floating refs are
intentionally not stored in queue jobs.

model_path per job comes from a template (config [queue]
model_path_template, default "output/scratch/{name}_{arm}").
"""
from __future__ import annotations

import json
import re

from . import experiment_declaration as declarations

DEFAULT_MODEL_PATH_TEMPLATE = "output/scratch/{name}_{arm}"
_FULL_GIT_SHA = re.compile(r"^[0-9a-fA-F]{40}$")


def load_spec(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _git_target_from_scope(scope: dict, *, where: str) -> tuple[str, str | None] | None:
    """Return ``(sha, repo)`` for one declaration scope.

    ``git_sha`` is the compact form.  ``git`` is the extensible structured
    form.  A single scope may not provide both: making one spelling canonical
    at each level avoids a subtle class of configuration where two fields look
    equivalent but disagree.
    """
    has_simple = "git_sha" in scope
    has_structured = "git" in scope
    if not has_simple and not has_structured:
        return None
    if has_simple and has_structured:
        raise SystemExit(f"{where}: specify only one of 'git_sha' or 'git'")

    repo: str | None = None
    if has_simple:
        sha = scope.get("git_sha")
    else:
        target = scope.get("git")
        if not isinstance(target, dict):
            raise SystemExit(f"{where}: 'git' must be an object with a 'sha' field")
        unknown = sorted(set(target) - {"sha", "repo"})
        if unknown:
            raise SystemExit(f"{where}: unsupported git field(s): {', '.join(unknown)}")
        sha = target.get("sha")
        repo = target.get("repo")
        if repo is not None and (not isinstance(repo, str) or not repo.strip()):
            raise SystemExit(f"{where}: git.repo must be a non-empty string")
        if isinstance(repo, str):
            repo = repo.strip()

    if not isinstance(sha, str) or _FULL_GIT_SHA.fullmatch(sha) is None:
        raise SystemExit(
            f"{where}: Git target must be a full 40-character hexadecimal commit SHA"
        )
    return sha.lower(), repo


def _resolve_git_target(spec: dict, base: dict, arm: dict, *, name: str,
                        arm_id: str) -> tuple[str, str | None] | None:
    """Resolve the nearest Git target using arm > base > spec precedence."""
    for scope, where in (
        (arm, f"spec '{name}' arm '{arm_id}'"),
        (base, f"spec '{name}' base"),
        (spec, f"spec '{name}'"),
    ):
        target = _git_target_from_scope(scope, where=where)
        if target is not None:
            return target
    return None


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
        lane = arm.get("lane", spec.get("lane", base.get("lane", 2)))
        if lane not in (1, 2, 3):
            raise SystemExit(f"spec '{name}' arm '{arm_id}': lane must be "
                             f"1, 2 or 3 (got {lane!r})")
        job = {
            "id": job_id,
            "ablation": name,
            "machine": arm.get("machine", base.get("machine", "any")),
            "type": arm.get("type", base.get("type", "replay")),
            "scene": arm.get("scene", base.get("scene", "")),
            "model_path": model_path_template.format(name=name, arm=arm_id, id=job_id),
            "extra_args": extra,
            "iterations": arm.get("iterations", base.get("iterations", 30000)),
            "lane": lane,
            "status": "pending",
        }
        git_target = _resolve_git_target(spec, base, arm, name=name, arm_id=arm_id)
        if git_target is not None:
            git_sha, git_repo = git_target
            job["requested_git_sha"] = git_sha
            if git_repo is not None:
                job["git_repo"] = git_repo
        try:
            declaration = declarations.resolve_declaration(
                spec.get("experiment"), arm.get("declaration"), arm_id
            )
        except declarations.ExperimentDeclarationError as exc:
            raise SystemExit(f"spec '{name}' arm '{arm_id}': {exc}") from exc
        if declaration is not None:
            job.update(declarations.freeze_declaration(declaration))
        if not parallel and prev_id is not None:
            job["depends_on"] = prev_id
        jobs.append(job)
        prev_id = job_id
    return jobs
