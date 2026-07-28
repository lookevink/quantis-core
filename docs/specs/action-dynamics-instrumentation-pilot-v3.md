# Action-dynamics instrumentation pilot v3

## Status

Preregistered before collection on 2026-07-28.

The machine-readable authorities are:

- `lab/action_dynamics/smoke-protocol-v3.json`; and
- `lab/action_dynamics/pilot-protocol-v3.json`.

The v2 smoke and pilot artifacts and their failed decision remain immutable.
V3 uses new generator seeds, case IDs, pair IDs, execution plans, captures,
and output directories. V2 data may explain the prospective repair but may
not be rescored into v3.

## Evidence boundary

V3 qualifies the action-and-observation apparatus. It is not model training,
validation, confirmation, or evidence for a world-model claim. The effect
thresholds remain `0.15` for API rejection and the recovery-ratio threshold
remains `0.30`; neither was relaxed after v2.

The smoke must pass every gate before the pilot is opened. The pilot must pass
every gate before a separate 120-pair development corpus may be generated.
Smoke and pilot captures must never enter model fitting.

## Non-circular identity binding

Preparation writes a canonical schema-v2 `plan.json` before wrapper
manifests. The execution plan contains:

- protocol, image, build, observation-schema, and graph identities;
- all opaque pair and case assignments;
- all action schedules, workload schedules, seeds, and topologies; and
- for the pilot, the qualifying-smoke digest.

The plan deliberately excludes wrapper-file hashes. Its final canonical
SHA-256 is embedded in every schema-v2 wrapper manifest. A separate
`manifest-index.json` maps case IDs to final wrapper-file hashes. This avoids
the circular v2 pseudo-plan digest.

The smoke assessment artifact manifest must verify with zero mismatches before
pilot preparation. `smoke-qualification.json` records and hashes:

- the smoke protocol;
- the smoke execution plan;
- the smoke data-quality assessment;
- the smoke artifact manifest;
- the smoke collection attestation;
- every smoke gate; and
- the zero-mismatch artifact verification count.

Its file SHA-256 is embedded in the pilot execution plan, every pilot wrapper,
the collection attestation, assessment, report, and final artifact manifest.
The assessor exposes explicit `final_plan_binding` and
`qualifying_smoke_binding` gates.

## Fixed trajectory and API estimand

Every v3 capture has 108 state windows of 250 milliseconds.

API rejection retains severities `0.25`, `0.50`, and `0.75` and minimum effect
`0.15`. Every API pair uses:

- exactly 20 active transitions;
- exactly 12 requests per transition in both twins;
- exactly 240 active requests per twin; and
- exact `request_count` and `error_count` evidence metrics at every state.

The primary active effect is:

`sum(treatment errors) / sum(treatment requests)`

minus:

`sum(control errors) / sum(control requests)`.

The equal-duration immediately preceding placebo uses the same pooled-count
estimand. Missing, fractional, inconsistent, or non-240 active denominators
fail the pair. The smoke uses the worst-case configured cell:
three workers, magnitude `0.25`, and duration 20.

## Enqueue drain and recovery

Every enqueue-delay pair uses this fixed tail:

- normal matched workload through transition 83;
- transitions 84 through 91: zero-request drain;
- transitions 92 through 106: eight-request probe workload; and
- recovery scored on final states 100 through 107.

Before recovery is eligible, both twins must show four consecutive final drain
states with:

- `queue_depth == 0`;
- `api_inflight_current == 0`;
- `worker_busy_count == 0`; and
- a clean action-controller close boundary with no active intervention.

Failure to drain fails the pair and is not an exclusion or retry. Eligible
recovery retains the v2 final-eight paired median-absolute-delta ratio and the
fixed maximum `0.30`.

The smoke contains two independent three-worker, 20 ms enqueue-delay pairs
with duration 20.

## Designs and execution

Smoke contains six matched pairs:

- one full worker pause;
- one PostgreSQL lock;
- two low-severity enqueue delays;
- one dequeue delay; and
- one low-severity API rejection.

The pilot remains the complete factorial:

`5 action kinds × 3 worker topologies × 2 replicates = 30 pairs`

There are 60 pilot captures, six isolated Compose lanes, five pair-atomic
batches, no automatic retries, and no attrition replacement. Treatment and
control twins run sequentially in their lane from fresh Compose projects.

Frozen identities:

- smoke generator seed `26072821`;
- pilot generator seed `26072822`;
- smoke output `artifacts/action-dynamics/lab-smoke-v3`;
- pilot output
  `artifacts/action-dynamics/instrumentation-pilot-v3`;
- smoke Compose prefix `quantis-action-smoke-v3`; and
- pilot Compose prefix `quantis-action-pilot-v3`.

## Go/no-go gates

All prior structural gates remain: exact capture/pair counts, schedule
alignment, exact metric windows, start/stop and cleanup commands, truth
exclusion, at least 95% eligible-event trace linkage, at least 95% exact
completed paths, zero cross-case trace references, effect, recovery, placebo,
lane isolation, and artifact hashing.

V3 additionally requires:

1. exact final-plan binding in every wrapper;
2. valid qualifying-smoke binding in the pilot;
3. exact API count evidence and 240 active requests per twin;
4. both smoke enqueue pairs and all pilot enqueue pairs to drain cleanly; and
5. every treatment to retain the original signed effect and recovery
   thresholds.

Any failure stops the workflow. It must remain visible and requires another
versioned repair; it cannot be retried or replaced under v3.

## Training authorization

A fully qualified pilot authorizes design and collection of a new 120-pair
development corpus. It does not itself authorize fitting on pilot captures.

Before development collection, a separate protocol must freeze:

- 90 training pairs and 30 development-validation pairs;
- whole-pair, action/topology-balanced, seed-disjoint splits;
- the real-capture graph/event compiler and truth-exclusion checks;
- persistence, seasonal persistence, action-conditioned and action-agnostic
  graph VARX baselines;
- the neural action-conditioned graph state-space model and ablations; and
- immutable multi-seed training and assessment artifacts.

Only that separately qualified development corpus may start model fitting.
