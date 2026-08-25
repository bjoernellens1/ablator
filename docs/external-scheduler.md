# External scheduler API

Ablator can act as a deliberately small execution backend for an external workflow scheduler. The external scheduler decides **what is ready to run**; Ablator decides **where and how that ready job executes**.

The interface is workload-agnostic and stdlib-only. It does not depend on Snakemake, ResearchFlow, or any experiment-management package.

## Configure a trusted job type

External callers select an existing Ablator type. They cannot submit arbitrary host commands.

For a Snakemake/ResearchFlow jobscript stored on shared storage:

```toml
[types.researchflow]
command = ["bash", "{jobscript}"]
cwd = "/mnt/cps_scratch1_tmp/researchflow"
```

`jobscript` is a typed external parameter. Any parameter submitted with `--param key=value` becomes a command/environment template variable. Reserved legacy queue fields such as `machine`, `scene`, `iterations`, and `extra_args` cannot be overridden through `params`.

The jobscript path must be readable by the machine that ultimately claims the job. For multi-machine deployments, place ResearchFlow/Snakemake runtime state on shared storage.

## Submit

```bash
ablator submit \
  --format json \
  --id rf-0123456789ab-train-fr3-seed1 \
  --type researchflow \
  --machine any \
  --git-sha 0123456789abcdef0123456789abcdef01234567 \
  --git-repo https://github.com/example/project.git \
  --param jobscript=/mnt/cps_scratch1_tmp/researchflow/jobs/train.sh \
  --metadata-json '{"scheduler":"snakemake","researchflow_plan_sha256":"..."}'
```

Response:

```json
{
  "schema": "ablator.external-job/v1",
  "job_id": "rf-0123456789ab-train-fr3-seed1",
  "status": "pending",
  "created": true,
  "external_spec_sha256": "..."
}
```

Submission is idempotent. Repeating the same ID with the same immutable specification returns the existing job with `created=false`. Reusing an ID for a different specification fails closed.

`--git-sha` accepts only a full 40-character commit SHA. `--git-repo` is
optional when the configured type `cwd` is already a usable repository. Both
fields participate in the idempotency fingerprint, and a dependent job must
use the same Git target as its parent. A type with `require_pinned_git = true`
rejects an external job that omits the SHA before launch.

## Inspect an exact job

```bash
ablator inspect --format json rf-0123456789ab-train-fr3-seed1
```

The JSON projection exposes the stable queue state plus execution provenance, including:

- exact job status;
- requested and claimed machine;
- typed parameters and opaque scheduler metadata;
- workload checkout provenance;
- container/image provenance when available;
- the canonical `execution_receipt` and post-run `execution_attestation`,
  including requested/executed commit, ref/dirty/submodules, resolved cwd,
  runtime image and mounts, and hashed argv/type configuration;
- **Ablator runner provenance**: package version, runner Git commit/branch/dirty state, config SHA-256, machine and hostname;
- failure classification and health state.

External schedulers should consume this JSON rather than scraping the human `ablator status` table.

## Cancel exact jobs

```bash
ablator cancel-jobs --format json job-a job-b job-c
```

Pending jobs are atomically marked `cancelled`. Running jobs use Ablator's existing `skip` control-file protocol, so the supervising runner performs the same backend-aware teardown used by the human CLI. Terminal jobs are stable no-ops.

## Stable states

The external API preserves Ablator's queue vocabulary:

- `pending`
- `running`
- `done`
- `failed`
- `quarantined`
- `cancelled`
- transient/backoff states such as `paused_disk_full`

A workflow adapter maps these to its own scheduler states. For example, the ResearchFlow Snakemake adapter maps `pending/running` to `running`, `done` to `success`, and terminal failures to `failed`.

## Provenance boundary

Ablator records two separate identities:

1. **workload provenance** — the checkout/image used by the application being run;
2. **runner provenance** — the actual Ablator code/config/machine that claimed and executed the job.

Both are necessary when comparing scientific results across machines. A run can execute successfully while still being scientifically incomparable if mandatory provenance is absent or ambiguous; that interpretation belongs to the research layer, not Ablator.
