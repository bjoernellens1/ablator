# Immutable Per-Job Checkouts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every scientific Ablator job execute from an auditable, immutable, per-attempt checkout pinned to registered Git intent.

**Architecture:** Extend the existing SHA/worktree stack with authoring and queue policy, unique leased worktrees with recursive submodule verification, a protected execution receipt plus post-run attestation, and containment-safe GC. Keep mutable execution only for explicitly non-gradeable legacy types, then enable the strict type policy in Splatograph.

**Tech Stack:** Python 3 standard library, Git CLI/worktrees/submodules, POSIX `flock`, pytest, Docker/Podman argv contracts, Kubernetes Job manifests.

**Spec:** `docs/designs/2026-08-25-immutable-job-checkouts.md`

## Global Constraints

- Preserve published Ablator PR #35/#37/#38/#39/#40 history; make changes only on a new stacked follow-up branch.
- Never execute a gradeable or `require_pinned_git` job without a full 40-character SHA.
- Never share an execution worktree between jobs or attempts.
- Never delete outside the configured worktree cache or delete result/evidence paths.
- Do not add third-party runtime dependencies.
- Synthetic tests are contract evidence, not production or scientific validation.

---

### Task 1: Pin policy and dependency-chain identity

**Files:**
- Modify: `src/ablator/spec.py`
- Modify: `src/ablator/queue.py`
- Modify: `src/ablator/external.py`
- Modify: `src/ablator/cli.py`
- Test: `tests/test_git_target.py`
- Test: `tests/test_external.py`

**Interfaces:**
- Consumes: existing `_resolve_git_target()` and frozen declaration fields.
- Produces: `validate_job_git_policy(job, type_config=None)` and external `git_sha`/`git_repo` inputs.

- [ ] **Step 1: Write failing authoring and queue tests**

```python
def test_gradeable_spec_requires_git_sha():
    with pytest.raises(SystemExit, match="requires an immutable Git target"):
        expand_spec(gradeable_spec_without_git())

def test_sequential_chain_rejects_mixed_git_targets():
    with pytest.raises(SystemExit, match="dependency chain changes Git target"):
        expand_spec(sequential_spec_with_two_shas())

def test_queue_rejects_dependency_sha_drift():
    with pytest.raises(SystemExit, match="dependency.*Git target"):
        Queue(path).append([parent_at_sha_a, child_at_sha_b])
```

- [ ] **Step 2: Run the focused tests and confirm they fail**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python3 -m pytest \
  tests/test_git_target.py tests/test_external.py -q
```

- [ ] **Step 3: Implement minimal policy and external transport**

```python
def validate_job_git_policy(job: Mapping[str, Any], type_config: Mapping[str, Any] | None = None) -> None:
    required = job.get("gradeability") == "GRADEABLE_DECLARED" or bool((type_config or {}).get("require_pinned_git"))
    if required and not _full_sha(job.get("requested_git_sha")):
        raise GitTargetError("job requires an immutable Git target")
```

Add `--git-sha` and `--git-repo` to `ablator submit`, include both in its
idempotency fingerprint/submission envelope, and validate every queue
dependency edge against the parent's repository/SHA.

- [ ] **Step 4: Run focused tests green**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python3 -m pytest \
  tests/test_git_target.py tests/test_external.py tests/test_experiment_declaration.py -q
```

- [ ] **Step 5: Commit the policy contract**

```bash
git add src/ablator/spec.py src/ablator/queue.py src/ablator/external.py \
  src/ablator/cli.py tests/test_git_target.py tests/test_external.py
git commit -m "fix(#7): require immutable scientific git intent"
```

### Task 2: Unique leased worktrees and submodules

**Files:**
- Modify: `src/ablator/source_checkout.py`
- Test: `tests/test_source_checkout.py`
- Create: `tests/test_source_checkout_concurrency.py`

**Interfaces:**
- Consumes: `prepare_job_source(cfg, job, machine, tcfg)`.
- Produces: `PreparedSource.lease`, `release_source(prepared)`, and `capture_checkout_state(path)` including recursive submodules.

- [ ] **Step 1: Write failing per-attempt, submodule, and concurrency tests**

```python
def test_same_sha_jobs_get_distinct_worktrees():
    assert prepare(job_a).checkout_path != prepare(job_b).checkout_path

def test_recursive_submodules_are_pinned_and_clean():
    prepared = prepare(superproject_job)
    assert prepared.state["submodules"] == [{"path": "deps/lib", "sha": child_sha, "dirty": False}]

def test_parallel_materialization_and_gc_cannot_remove_active_checkout():
    with ThreadPoolExecutor(max_workers=3) as pool:
        prepared = list(pool.map(prepare, jobs))
    assert len({item.checkout_path for item in prepared}) == len(jobs)
```

- [ ] **Step 2: Run the focused tests and confirm they fail**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python3 -m pytest \
  tests/test_source_checkout.py tests/test_source_checkout_concurrency.py -q
```

- [ ] **Step 3: Implement unique paths, active sidecars, and recursive verification**

```python
@dataclass(frozen=True)
class SourceLease:
    checkout: str
    sidecar: str
    lock_path: str
    lease_id: str

def capture_checkout_state(path: str) -> dict:
    return {"commit": head, "ref": "detached", "dirty": False, "submodules": submodules}

def release_source(prepared: PreparedSource) -> None:
    # Atomically mark only this lease inactive after final attestation.
    ...
```

Use a sanitized job ID plus a random lease ID in each checkout path. Under the
repository lock, initialize every submodule recursively and write the active
sidecar before returning.

- [ ] **Step 4: Run focused tests green**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python3 -m pytest \
  tests/test_source_checkout.py tests/test_source_checkout_concurrency.py -q
```

- [ ] **Step 5: Commit worktree isolation**

```bash
git add src/ablator/source_checkout.py tests/test_source_checkout.py \
  tests/test_source_checkout_concurrency.py
git commit -m "fix(#7): isolate and lease each execution checkout"
```

### Task 3: Read-only container binds and execution receipts

**Files:**
- Create: `src/ablator/execution_receipt.py`
- Modify: `src/ablator/experiment_declaration.py`
- Modify: `src/ablator/runner.py`
- Test: `tests/test_execution_receipt.py`
- Modify: `tests/test_source_checkout.py`
- Modify: `tests/test_resume.py`

**Interfaces:**
- Consumes: prepared source state, runner provenance, rendered argv/cwd.
- Produces: `build_prelaunch_receipt(...)`, `final_attestation(...)`, and a protected queue/job envelope.

- [ ] **Step 1: Write failing mount, receipt, and mutation tests**

```python
def test_checkout_bind_is_read_only_and_mount_identity_is_recorded():
    prepared, receipt = prepare_and_receipt(podman_type())
    assert checkout_mount(prepared.type_config)["read_only"] is True
    assert receipt["launch"]["mounts"] == [expected_mount]

def test_post_run_source_mutation_turns_zero_exit_into_failure():
    status, _ = run_job_with_source_mutating_child()
    assert status == "failed"
    assert queued()["execution_attestation"]["verdict"] == "REJECTED"

def test_job_json_is_rendered_after_executed_sha_and_receipt_exist():
    envelope = json.loads(child_env["ABLATOR_JOB_JSON"])
    assert envelope["execution_receipt"]["source"]["executed_git_sha"] == sha
```

- [ ] **Step 2: Run the focused tests and confirm they fail**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python3 -m pytest \
  tests/test_execution_receipt.py tests/test_source_checkout.py tests/test_resume.py -q
```

- [ ] **Step 3: Implement normalized mounts and two-phase attestation**

```python
def build_prelaunch_receipt(*, cfg, job, machine, type_config, argv, cwd, source_state, runner_provenance) -> dict:
    return {"schema": "ablator.execution/v1", "phase": "prelaunch", "source": source_state, "launch": normalized_launch}

def final_attestation(requested_sha: str, checkout: str) -> dict:
    state = capture_checkout_state(checkout)
    return {"schema": "ablator.execution-attestation/v1", "verdict": "ACCEPTED" if state_is_exact else "REJECTED", "source": state}
```

Render argv first, persist the receipt into the local/queue job, then render
protected environment so `ABLATOR_JOB_JSON` contains the actual pre-launch
identity. Inject `PYTHONDONTWRITEBYTECODE=1`. Finalize and release the lease in
a `finally` path, and reject nominal success when final attestation is not exact.

- [ ] **Step 4: Make Kubernetes source initialization equivalent**

```sh
git fetch --depth 1 origin "$SHA"
git checkout --detach FETCH_HEAD
git submodule sync --recursive
git submodule update --init --recursive --checkout
test "$(git rev-parse HEAD)" = "$SHA"
test -z "$(git status --porcelain --untracked-files=no)"
```

Mount the repo emptyDir read-only in the trainer and include receipt JSON in its
protected environment.

- [ ] **Step 5: Run focused tests green**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python3 -m pytest \
  tests/test_execution_receipt.py tests/test_source_checkout.py \
  tests/test_experiment_declaration.py tests/test_resume.py -q
```

- [ ] **Step 6: Commit receipt and attestation**

```bash
git add src/ablator/execution_receipt.py src/ablator/experiment_declaration.py \
  src/ablator/runner.py tests/test_execution_receipt.py \
  tests/test_source_checkout.py tests/test_resume.py
git commit -m "fix(#7): attest immutable source through job completion"
```

### Task 4: Containment-safe, lease-aware cleanup

**Files:**
- Modify: `src/ablator/source_gc.py`
- Modify: `tests/test_source_gc.py`
- Modify: `tests/test_source_checkout_concurrency.py`

**Interfaces:**
- Consumes: active/inactive lease sidecars and shared repository locks.
- Produces: GC classification that retains active/orphan leases and removes only validated inactive entries.

- [ ] **Step 1: Write failing cleanup safety tests**

```python
def test_gc_rejects_sidecar_checkout_outside_cache_root():
    assert outside_path.exists()
    assert "outside managed root" in gc_result.errors[0]

def test_gc_never_removes_active_lease_even_when_queue_update_lags():
    assert prepared.checkout_path in gc_result.protected

def test_git_prune_failure_is_reported_and_sidecar_retained():
    assert gc_result.removed == ()
    assert sidecar.exists()
```

- [ ] **Step 2: Run focused tests and confirm they fail**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python3 -m pytest \
  tests/test_source_gc.py tests/test_source_checkout_concurrency.py -q
```

- [ ] **Step 3: Implement containment, locks, and error-preserving cleanup**

```python
def _contained(root: Path, candidate: Path) -> bool:
    return candidate != root and candidate.is_relative_to(root)
```

Validate both checkout and sidecar real paths, lock the repository namespace,
re-read the lease under lock, refuse active leases, require successful worktree
remove/prune before deleting the sidecar, and report orphan leases separately.

- [ ] **Step 4: Run focused tests green**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python3 -m pytest \
  tests/test_source_gc.py tests/test_source_checkout_concurrency.py -q
```

- [ ] **Step 5: Commit cleanup hardening**

```bash
git add src/ablator/source_gc.py tests/test_source_gc.py \
  tests/test_source_checkout_concurrency.py
git commit -m "fix(#7): make checkout cleanup lease-safe"
```

### Task 5: Documentation, full Ablator verification, and PR

**Files:**
- Modify: `docs/git-pinned-jobs.md`
- Modify: `docs/worktree-cache.md`
- Modify: `docs/external-scheduler.md`
- Modify: `docs/kubernetes.md`
- Modify: `docs/config-reference.md`
- Modify: `mkdocs.yml`

**Interfaces:**
- Consumes: all preceding contracts.
- Produces: operator documentation and review evidence.

- [ ] **Step 1: Document strict pinning, receipts, submodules, leases, and rollout gates**

```markdown
Pinned scientific jobs use one exact SHA across every dependency edge. The
runner records `execution_receipt` before launch and `execution_attestation`
after the workload exits; `done` is refused if the final source state drifts.
```

- [ ] **Step 2: Run all Ablator tests**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python3 -m pytest tests/ -q
```

- [ ] **Step 3: Run package and docs checks**

```bash
python3 -m compileall -q src tests
python3 -m build
python3 -m mkdocs build --strict
```

- [ ] **Step 4: Commit docs and push the stacked branch**

```bash
git add docs mkdocs.yml
git commit -m "docs(#7): document immutable execution receipts"
git push -u origin agent/issue-7-immutable-checkout-hardening
```

### Task 6: Splatograph integration after Ablator dependency is reviewable

**Files:**
- Modify: `ablator`
- Modify: `configs/ablator.toml`
- Modify: `utils/ablator_provenance.py`
- Modify: `tests/test_ablator_provenance.py`
- Create: `tests/test_issue259_immutable_checkout_contract.py`
- Modify: `docs/architecture/2026-08-13-issue259-ablator-checkout-isolation.md`

**Interfaces:**
- Consumes: released/merged Ablator immutable-checkout contract.
- Produces: Splatograph strict configuration, persisted receipt, and submodule pin.

- [ ] **Step 1: Create a separate Splatograph worktree and failing integration tests**

```python
def test_every_scientific_ablator_type_requires_pinned_git():
    assert all(tcfg["require_pinned_git"] for tcfg in scientific_types())

def test_config_uses_repo_cwd_template_for_source_mounts():
    assert "{repo_cwd}:/workspace/splatograph:ro" in replay_command()
```

- [ ] **Step 2: Update config paths and receipt persistence**

Use `{repo_cwd}` for cwd, build context, script, and source bind paths; preserve
writable dataset/scratch mounts. Persist the execution receipt already embedded
in `ablator_job.json` and surface its source verdict in run-summary validation.

- [ ] **Step 3: Run Splatograph checks only in the project container**

```bash
docker compose run --rm train python -m pytest \
  tests/test_ablator_provenance.py \
  tests/test_issue259_immutable_checkout_contract.py -q
docker compose run --rm train python -m pytest tests/test_issue1013_mandatory_researchflow_enforcement.py -q
```

- [ ] **Step 4: Commit, push, and open the Splatograph PR**

```bash
git add ablator configs/ablator.toml utils/ablator_provenance.py \
  tests/test_ablator_provenance.py tests/test_issue259_immutable_checkout_contract.py \
  docs/architecture/2026-08-13-issue259-ablator-checkout-isolation.md
git commit -m "fix(#259): require immutable Ablator source checkouts"
git push -u origin agent/issue-259-immutable-checkout-integration
```

### Task 7: Review and deployment gate

**Files:**
- No source changes unless review identifies a verified defect.

**Interfaces:**
- Consumes: green Ablator and Splatograph PRs.
- Produces: review results and an explicit deployed/not-deployed evidence verdict.

- [ ] **Step 1: Request code review for both PRs**

```bash
gh pr edit <ablator-pr> --add-reviewer <reviewer>
gh pr edit <splatograph-pr> --add-reviewer <reviewer>
```

- [ ] **Step 2: Check runner and queue state read-only**

```bash
ablator status
podman ps --format '{{.Names}} {{.Status}}'
```

- [ ] **Step 3: Stop at the deployment gate unless every runner is idle**

No runner restart, config mutation, or queue write is permitted while work is
active. If idle and separately authorized, install the merged release and run a
fresh non-scientific pinned diagnostic; otherwise report the exact remaining
gate and leave Splatograph #259 open.
