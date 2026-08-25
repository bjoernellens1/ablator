# Workload launch provenance

Ablator transports queue identity and submission provenance into launched workloads through protected environment variables. This lets applications persist their own self-contained reproduction artifacts without reading the shared queue or invoking Ablator from inside a container.

## Protected variables

Every real queue-backed launch receives:

```text
ABLATOR_JOB_ID
ABLATOR_JOB_JSON
```

`ABLATOR_JOB_JSON` is the canonical JSON snapshot of the queue job as seen by
the runner when the command is rendered. For a pinned bare-metal launch, that
render occurs only after the runner has persisted `executed_git_sha`, source
lease, runner provenance, and the canonical pre-launch `execution_receipt`.

When the original queue surface is known, the child also receives:

```text
ABLATOR_SUBMISSION_JSON
```

For jobs produced by `ablator plan`, the submission envelope contains the exact parsed spec, its canonical SHA-256, the informational source path, the ablation name, and `surface: "plan"`.

For jobs produced by `ablator submit`, the envelope contains the immutable typed submit inputs already stored in the external queue record and `surface: "submit"`.

Gradeable/declared jobs additionally retain the existing declaration transport:

```text
ABLATOR_EXPERIMENT_DECLARATION_JSON
ABLATOR_EXPERIMENT_DECLARATION_SHA256
```

All of these names are protected. Ambient environment variables and type-configured environment entries with the same names are removed/rejected before the trusted values are injected into direct children, Docker/Podman containers, and Kubernetes trainer containers.

A pinned Kubernetes trainer additionally receives
`ABLATOR_SOURCE_PROOF_JSON`. It is populated only by the command wrapper from
the init container's verified, read-only proof file; ambient or configured
values are scrubbed like the other protected variables.

The execution receipt records requested/executed source identity, detached
ref/dirty/submodule state, checkout and lease identity, runner/config identity,
resolved cwd, runtime/image, normalized mounts, and hashes of the rendered argv
and merged type configuration. It excludes arbitrary environment values and
credentials. Its canonical `execution_receipt_sha256` and exact
`actual_launch` are stored before execution. The queue's later
`execution_attestation` verifies the receipt hash and binds source, runner,
config, semantic argv, and actual launch evidence; Kubernetes additionally
binds observed command/mount/pod/node/image/image-ID identity.

## `ablator plan`

`ablator plan spec.json` loads the JSON through `load_spec()`. The exact parsed object is copied into each generated job under `submission_provenance`, together with:

```json
{
  "schema": "ablator.submission/v1",
  "surface": "plan",
  "spec_path": "/absolute/informational/path/spec.json",
  "spec_sha256": "...",
  "spec": {"...": "exact parsed spec"},
  "ablation": "..."
}
```

The spec contents and hash are authoritative. `spec_path` is only provenance about where the submitting process read the file; a consumer must not assume that path exists on another machine.

## `ablator submit`

External jobs already store their immutable submit inputs in the queue record. At launch, Ablator derives the equivalent `ablator.submission/v1` envelope from those fields, including job id, type, requested machine, typed params, metadata, lane, dependency, and the existing `external_spec_sha256`.

## Compatibility

Old queue records remain runnable. Once an old pending job is claimed and becomes `running`, the runner can still transport `ABLATOR_JOB_ID` and `ABLATOR_JOB_JSON` even though there is no recoverable original spec/submission envelope. Consumers must therefore treat `ABLATOR_SUBMISSION_JSON` as optional and must never invent a spec from job naming conventions.

## Intended consumer behavior

A workload may persist the protected values as artifacts such as:

```text
ablator_job.json
ablator_submission.json
ablator_spec.json
```

For a plan-created job, `ablator_spec.json` should be written from the exact `submission.spec` object. Historical inspection uses `ablator inspect --format json <job-id>`; a new gradeable scientific replication must use a new job identity/output with explicit lineage rather than `ablator rerun` in place.
