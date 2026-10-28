# Passive host lifecycle receipt

Opt in with `external_metadata.record_outer_lifecycle_cost: true` (literal JSON
boolean). Local/managed-container `run_job` only; Kubernetes is not instrumented.
Normal jobs perform no additional lifecycle I/O. One fresh, private log-directory
`outer-lifecycle-cost-<uuid>.json` is written exclusively after the owner boundary.
Absence (including SIGKILL or publication failure) is unknown, never zero cost.
The receipt is not a scheduler state or a new cleanup owner.

`outer-lifecycle-cost/v1` has these exact top-level fields:

- `schema`, `envelope_id` (fresh UUID, invocation identity only).
- `identity`: `scheduler_job_id` (actual Ablator job id), `external_id`, `machine`,
  `researchflow_run_id`, `researchflow_plan_sha256`, `researchflow_job_id`. RF
  values come from the identically named canonical external metadata fields.
  Missing values are null; no identities are inferred from filenames.
  These bind the actual host owner and RF job, not every upstream scheduler alias.
- `declaration`: `requested_git_sha`, `experiment_declaration_sha256`,
  `jobscript_sha256`. These are submitted claims, separate from observed bindings.
- `clock_domain`: exactly `schema`, `clock`, `boot_id`, `time_namespace`, `status`,
  `checked_through`, `unavailable_reason`, following `monotonic-clock-domain/v1`.
  Known Linux CLOCK_MONOTONIC identity is checked at start and owner return.
- `started_monotonic_ns`, `child_started_monotonic_ns`,
  `child_wait_completed_monotonic_ns`, `observed_through_monotonic_ns`,
  `closed_monotonic_ns`: integer nanoseconds or null. Start precedes source
  preparation; launch is sampled immediately after Popen; wait completion after
  the existing supervisor returns with a reaped child. Observation follows
  final attestation, queue publication, source release and raw-byte binding.
  Closed is null if release/child completion is not observed or owner unwinds.
- `elapsed_seconds`: `(closed-start)/1e9` or null. `observed_seconds` is the
  elapsed lower bound through the last observation. Neither is endpoint time.
- `outcome`: `status`, `exit_code`, `error` (first caught execution error or null).
- `closure`: `host_owner` (`closed`/`unknown`), `direct_child`
  (`not_launched`/`reaped`/`unknown`), `source_release`
  (`not_acquired`/`released`/`unknown`), `workload` (always `unknown`), `reason`.
  Closed host ownership does not prove remote descendants, container teardown,
  retained-attempt completeness or opaque workload durability. Return zero does
  not upgrade workload closure. Failed/cancelled jobs can close host ownership.
- `bindings`: `execution_receipt`, `execution_receipt_sha256`, `actual_launch`,
  `execution_attestation`: observed existing Ablator records or null. The receipt
  digest uses existing canonical Ablator receipt hashing, not a raw file hash.
- `observed_attempts`: records ordered by observed attempt-start integer then
  attempt id; each has `attempt_id`, `record` and `execution`. Each byte reference
  is `{path, sha256}` (absolute path, lowercase SHA256); execution may be null.
  Only matching job/run/plan `researchflow.execution-attempt/v1` records in the
  canonical RF `jobs/attempts/<id>` layout are included. `attempt_roster` is
  `observed_only`; omission/completeness and scientific admission need cold
  validation. Raw bytes are hashed before the owner-boundary sample.
- `durable_artifacts`: observed canonical RF job `.execution.json`, `.evidence.json`,
  `.done` byte references, when present. `binding_errors`: read/parse errors; these
  cannot establish complete output closure. Paths resolve from canonical
  `researchflow_origin_path` parent and the RF job id, as in existing health code.
- `resources`: `cpu_seconds`, `gpu_active_seconds`, `energy_joules` (all null),
  `reason`. No wall-to-resource inference or unavailable-as-zero conversion.
- `receipt_publication`: `scope` = `outside_closed_host_owner_interval`.
  Its own write and any later export are excluded; no self-inclusive rewrite.

A cold reader must verify byte hashes, canonical RF identities, complete retained
roster, clock domain/enclosure, and the relevant teardown/durability scope before
admitting any whole-workload bound. Nested timers never add to this envelope.
