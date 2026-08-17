# Spec reference

An ablation spec is a JSON file describing one **base** configuration
and a set of **arms** (variants) that get expanded into individual
queue jobs. `ablator plan spec.json` reads it, expands it, and appends
the resulting jobs to the shared queue (refusing duplicate job ids).

```json
{
  "name": "consol_sweep",
  "parallel": true,
  "base": {
    "type": "replay",
    "scene": "/mnt/data/fr3",
    "iterations": 30000,
    "machine": "any",
    "base_args": "--opacity_reg 0.001"
  },
  "arms": [
    {"id": "ctrl", "extra_args": ""},
    {"id": "consol", "extra_args": "--foo bar",
     "machine": "main", "type": "bag", "iterations": 60000}
  ]
}
```

## Top-level fields

| Field | Type | Default | Meaning |
|---|---|---|---|
| `name` | string | required | Ablation name. Job ids are `<name>_<arm_id>`. |
| `base` | object | `{}` | Defaults every arm inherits unless it overrides them. |
| `arms` | array of objects | required | The variants to expand. |
| `parallel` | bool | `true` | `false` chains arms sequentially via `depends_on` (see below). |
| `lane` | int (1, 2 or 3) | `2` | Spec-level default lane, overridable per arm. |

## `base` fields

`base` may set any of: `type`, `scene`, `iterations`, `machine`,
`base_args`, `lane`. Every one of these is overridable per arm.

## Arm fields

| Field | Type | Default | Meaning |
|---|---|---|---|
| `id` | string | required | Arm id, must be unique within the spec (`plan` raises `SystemExit` on a duplicate). |
| `type` | string | `base.type` or `"replay"` | Job type — must match a `[types.<type>]` entry in the host config. |
| `scene` | string | `base.scene` or `""` | Passed to the command template as `{scene}`. |
| `iterations` | int | `base.iterations` or `30000` | Passed as `{iterations}`. |
| `machine` | string | `base.machine` or `"any"` | Target machine name, or `"any"` for the first idle runner to claim it. |
| `extra_args` | string | `""` | Appended after `base_args`+arm's own `base_args` (see below). |
| `base_args` | string | `base.base_args` or `""` | Arm-level override of the base args string. |
| `lane` | int (1, 2 or 3) | spec/base `lane` or `2` | Queue lane (see [queue semantics](queue-semantics.md)). `plan` raises `SystemExit` if not 1/2/3. |

`extra_args` fed to the command template is
`" ".join([effective base_args, arm's own extra_args])`, both
`.strip()`'d and empty parts dropped. That is: `base_args` sets the
"always-on" flags for an arm, `extra_args` is appended after it.

## What gets built per job

```json
{
  "id": "<name>_<arm_id>",
  "ablation": "<name>",
  "machine": "...",
  "type": "...",
  "scene": "...",
  "model_path": "<model_path_template rendered with {name} {arm} {id}>",
  "extra_args": "...",
  "iterations": ...,
  "lane": 1|2|3,
  "status": "pending",
  "depends_on": "<previous arm's job id>"   // only when parallel: false
}
```

`model_path` is rendered from `[queue] model_path_template` in the host
config (default `"output/scratch/{name}_{arm}"`); available variables
are `{name}`, `{arm}`, `{id}`.

## `parallel: false` (sequential chains)

When `parallel` is `false`, each arm after the first gets
`depends_on` set to the **previous arm's job id** (declaration order in
the `arms` array, not lane or id order). A job with `depends_on` set is
only claimable once that dependency's status is `done` — see
[queue semantics](queue-semantics.md) for exactly how failed/quarantined
dependencies block the chain.

## Immutable experiment declarations

Add a top-level `experiment` object when a run may count as scientific
evidence. Each arm supplies its overlay under `declaration`; Ablator resolves
the two objects recursively, forces `arm.id` to the spec arm id, validates the
result, and freezes the resolved object plus canonical JSON and SHA-256 into the
queue job before enqueue.

```json
{
  "name": "issue705_transport",
  "experiment": {
    "schema_version": 1,
    "run_class": "experiment",
    "experiment_id": "issue705_transport",
    "title": "Immutable declaration transport",
    "purpose": "Prove the producer/consumer boundary",
    "owner_refs": ["splatograph#705"],
    "expected_evidence": ["matching hash at every boundary"],
    "comparison_group_id": "issue705_pair",
    "domain": {
      "system": "splatograph",
      "payload": {"replay_capsule_sha256": "abc123"}
    }
  },
  "base": {"type": "replay", "scene": "/mnt/data/fr3"},
  "arms": [
    {
      "id": "ctrl",
      "declaration": {
        "arm": {
          "title": "Control",
          "comparison_role": "control",
          "manipulation": "No treatment"
        }
      }
    },
    {
      "id": "candidate",
      "declaration": {
        "arm": {
          "title": "Candidate",
          "comparison_role": "treatment",
          "manipulation": "Enable the change under test"
        }
      }
    }
  ]
}
```

Schema version 1 accepts these `run_class` values:

- `experiment`, `benchmark`, `verification`: gradeable intent. These require
  non-empty `experiment_id`, `title`, `purpose`, `expected_evidence`, either
  `owner_refs` or `standalone_reason`, and per-arm `id`, `comparison_role`, and
  `manipulation`.
- `developer_smoke`, `debug`: explicitly non-gradeable declared work. These
  require only non-empty `title` and `purpose`.

Additional keys are preserved opaquely and covered by the hash. This is where
domain contracts, decision-profile/tooling references, result-sink references,
and `replicate_of`/`rerun_of`/`supersedes` lineage belong. Ablator does not
interpret domain payloads.

Canonical JSON is UTF-8 JSON with object keys sorted, no insignificant
whitespace, and Unicode preserved. The queue record stores:

```text
experiment_declaration
experiment_declaration_json
experiment_declaration_sha256
gradeability
```

At launch the runner revalidates all four fields, writes the exact JSON/hash to
the outer job log, and propagates these protected variables to direct children,
Docker/Podman containers, and Kubernetes trainer containers:

```text
ABLATOR_EXPERIMENT_DECLARATION_JSON
ABLATOR_EXPERIMENT_DECLARATION_SHA256
ABLATOR_JOB_ID
```

Specs without `experiment` remain runnable during migration but have no
declaration environment and are non-gradeable. Ablator also scrubs ambient or
type-configured values for the three protected variables so an undeclared job
cannot inherit a stale declaration accidentally.

A gradeable terminal job cannot use `ablator rerun` in place. Create a fresh
spec/job id and output directory and declare `rerun_of`, `replicate_of`, or
`supersedes` lineage instead.

## Dry run

```bash
ablator plan spec.json --dry-run
```

Prints the expanded jobs without writing them to the queue — use this
to sanity-check arm expansion, especially `extra_args` composition and
`depends_on` chaining, before committing jobs other machines might
start claiming immediately.
