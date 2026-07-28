# Action-dynamics instrumentation pilot v1

## Status

Preregistered for execution on 2026-07-28.

The machine-readable protocols are:

- `lab/action_dynamics/smoke-protocol.json`; and
- `lab/action_dynamics/pilot-protocol.json`.

Those files are the authority for collection constants and acceptance
thresholds. Before collection, the runner must record their canonical SHA-256
digests and bind the relevant digest into every generated plan, manifest,
attestation, assessment, and report.

Changing a collection constant, action range, effect threshold, recovery
threshold, trace denominator, or gate requires a new protocol version. The
runner must refuse to overwrite an existing output directory.

## Evidence boundary

This stage qualifies an intervention-and-observation instrument. It asks
whether isolated copies of the fixed Quantis checkout lab can execute
predeclared reversible actions while producing aligned metrics, structured
events, and traces suitable for later dynamics learning.

The smoke and pilot captures are not model training, validation, or
confirmation data. No result from this stage supports a forecasting,
attribution, topology-benefit, production-transfer, or world-model claim.

The embedded v3 `ActionConditionedCaseManifest` currently accepts only
`training`, `validation`, or `confirmation`. The lab wrapper therefore uses
`training` solely as a v3 compatibility sentinel. The wrapper protocol stage
is the authoritative corpus role. Corpus compilation and model fitting must
reject wrappers whose stage is `smoke` or `instrumentation_pilot`, regardless
of the embedded compatibility value.

The strongest permitted positive statement is:

> On the fixed Quantis checkout lab, the content-addressed concurrent harness
> executed five predeclared reversible intervention/location families across
> randomized matched treatment/control pairs and declared worker topologies
> with aligned commands, preregistered observable effects, recovery,
> trace-linked request paths, and isolated capture.

The report must replace “executed” with the observed counts and must list every
failed gate. Action kind and target are one-to-one in this library, so any
future inverse result must be called intervention-family/location-pair
attribution rather than independent target localization.

## Stage ordering

Collection has two sequential stages:

1. Run six smoke treatment/control pairs.
2. Recompute the complete smoke assessment from raw artifacts.
3. Continue only if every smoke gate passes.
4. Run the separately generated 30-pair instrumentation pilot.
5. Recompute the complete pilot assessment from raw artifacts.

Smoke and pilot may be invoked by one command, but they are not concurrent
stages. Smoke case IDs, pair IDs, manifests, captures, and estimates are
excluded from the pilot.

If smoke reveals that an action range or gate is unsuitable, the existing
pilot must not run. Revising the pilot requires a new protocol version and
digest; the smoke result remains visible as a bounded negative result.

## Fixed trajectory and workload

Every capture contains 84 logical windows of 250 milliseconds, or 21 seconds
of scheduled workload before final drain and cleanup.

- At least the first 20 windows are clean context.
- Action onset is between logical indices 28 and 39 inclusive.
- Active duration is between 8 and 20 windows inclusive.
- Every selected onset/duration combination leaves at least 24 post-stop
  recovery windows.
- A command at logical index `t` may first affect observation `state[t+1]`.
- Stop at logical index `t` means recovery begins at `state[t+1]`.

Each pair receives an explicit 84-value request-count schedule. Counts are
seeded integers from 6 through 10 inclusive. The schedule is materialized
before either twin starts and is copied byte-for-byte into the treatment and
control manifests. It is not regenerated independently inside a capture.

The API request queue size is 128. Any additional capacity, load, or topology
control must be explicit in the wrapper manifest and identical within a pair.

## Intervention library

Each treatment contains exactly one v3 `InterventionAction`; each control
contains none. All magnitudes are finite and positive, intervals do not
overlap, and `recovery_tolerance` is the dimensionless value `0.30`.

| Action | Target | Physical severities | Primary paired effect | Minimum signed effect |
|---|---|---:|---|---:|
| `worker_pause` | `worker_pool` | `1.0` worker fraction | `worker_rate` decreases | `1.0` request/s |
| `postgres_lock` | `worker_writes_postgresql` | `1.0` binary lock | `db_write_rate` decreases | `1.0` write/s |
| `redis_enqueue_delay` | `api_enqueues_queue` | `20`, `40`, `60` ms | `request_latency_ms` increases | `10.0` ms |
| `redis_dequeue_delay` | `queue_dequeues_to_worker` | `20`, `40`, `60` ms | `redis_dequeue_latency_ms` increases | `10.0` ms |
| `api_rejection` | `api` | `0.25`, `0.50`, `0.75` probability | `error_rate` increases | `0.15` |

Worker fractions are commanded fractions, not guaranteed realized fractions
on every topology. The capture must record the selected worker IDs and
realized paused-worker count. No worker-severity dose-response claim is
permitted from this instrumentation corpus.

`postgres_lock` remains a binary action; its two pilot replicates estimate
capture repeatability rather than two severity levels. A richer busy-age edge
metric may be reported as a diagnostic, but the frozen primary effect remains
the paired `db_write_rate` decrease in the protocol.

## Smoke design

The smoke protocol contains six fixed cells:

- all five action kinds appear at least once;
- one-, two-, and three-worker topologies are exercised;
- `worker_pause` appears once with one worker and once with two workers, both
  at full pause. The v1 smoke showed that partial pause was not identifiable
  from aggregate throughput under the frozen light-load schedule, so partial
  targeting is deferred until per-worker observations are available; and
- onset, duration, workload seed, and intervention seed are explicit in each
  cell.

There are exactly six pairs and twelve captures. Smoke is supported only if
all six pairs are structurally valid and all preregistered gates pass.

## Pilot design

The pilot is a complete factorial:

`5 action kinds × 3 worker topologies × 2 replicates = 30 pairs`

This yields exactly:

- six pairs per action kind;
- ten pairs per worker topology; and
- two pairs per action/topology cell.

For three-level actions, severity index is
`(topology_index + replicate_index) mod 3`. Across an action kind, every
commanded severity therefore appears exactly twice. PostgreSQL lock remains
`1.0`.

Onsets and durations are selected from the frozen inclusive ranges using the
protocol generator seed before capture. A draw may be rejected only when it
would leave fewer than 24 recovery windows. Request schedules are likewise
materialized before capture. The generated plan must pass action, topology,
replicate, severity, timing, and schedule-bounds validation before any
Compose project starts.

There are only two pairs per action/topology cell. Cell and topology results
are descriptive instrumentation diagnostics, not estimates of topology
generalization.

## Pairing and concurrent scheduling

Every matched pair contains one treatment and one control with:

- the same explicit request schedule;
- the same topology and worker count;
- the same workload and intervention seeds;
- the same logical timing and observation schema;
- the same resolved application and dependency image identities; and
- different opaque case IDs.

Case and pair IDs are opaque UUIDs and must not encode action kind, target,
severity, or treatment/control role.

Six isolated Compose lanes are used. Smoke has one batch of six pairs. The
pilot has five batches of six pairs. Within a lane, treatment and control run
sequentially from fresh Compose projects. Treatment runs first for even pair
ordinals and second for odd pair ordinals. The application image is built once
before collection.

No host ports, named volumes, capture directories, Compose project names,
networks, or action-control keys may be shared across active projects. Each
capture is torn down with volumes and orphans removed before its twin starts.

## Static identity and truth separation

The capture wrapper must resolve and store:

- the canonical v3 action-case manifest;
- the explicit request schedule;
- sampling and API-capacity settings;
- the observation-schema SHA-256;
- resolved image digests and application build-context SHA-256;
- protocol and generated-plan SHA-256 values; and
- the opaque case and pair identities.

`unverified`, empty, or non-digest build identities are invalid.

Action commands are emitted only to the dedicated conditioning stream.
Metrics, application logs, and traces must not contain action ID, action kind,
legacy fault kind, matched-pair ID, target entity, action phase, action
magnitude, or intervention seed as observation fields. Opaque case identity
and manifest digest may exist as provenance used to validate and partition a
capture, but they must not enter compiled model features.

Trace and span IDs are linkage keys, not numeric or categorical model
features.

## Trace eligibility

An eligible application event is a schema-declared trace-capable event emitted
while handling a case request in the API or worker. Runner action events,
readiness messages, collector diagnostics, and application events unrelated to
a request are excluded.

An eligible completed checkout is an accepted request that reaches successful
completion before the case drain boundary. Rejected API requests are eligible
for API trace-creation checks but are not expected to have a downstream
checkout path.

At least 95% of eligible application events must have structurally valid trace
and span IDs. At least 95% of eligible completed checkouts must form a linked
path covering:

`API admission → Redis enqueue → queue residence → Redis dequeue → worker processing → PostgreSQL write`

No trace may reference a span outside its opaque case capture.

## Effect, recovery, and placebo estimands

All metric comparisons align treatment and control by logical index. Let
`delta[t] = treatment[t] - control[t]` for the action’s declared primary
feature.

Active effect uses observations from `start_index + 1` through `stop_index`
inclusive:

`active_effect = median(delta[active_windows])`

For a declared decrease, multiply the result by `-1`; for an increase, retain
its sign. The signed value must meet the action-specific minimum in the table.

Recovery uses the final eight scheduled windows:

`recovery_delta = median(abs(delta[final_8_windows]))`

`recovery_ratio = recovery_delta / max(abs(active_effect), feature_floor)`

The fixed feature floors are:

- `0.25` for `worker_rate`;
- `0.25` for `db_write_rate`;
- `1.0` ms for request and Redis latency; and
- `0.05` for error rate.

Every treatment must have `recovery_ratio <= 0.30`.

The placebo interval for a pair is the equally long interval immediately
before action onset. A placebo false positive occurs when its signed paired
median meets the action’s primary minimum effect. Smoke permits zero placebo
false positives; the 30-pair pilot permits at most three.

These are apparatus gates, not causal effect-size estimates. Any uncertainty
summary resamples whole matched pairs, stratified by action kind. Windows,
events, spans, and requests inside one capture are not independent resampling
units.

## Acceptance gates

The assessor must recompute, rather than trust, every gate from raw files and
the frozen protocol:

1. Expected pair and capture counts are exact.
2. Every pair has exactly one treatment and one control.
3. All captures have exactly 84 complete, unique, monotonic metric windows.
4. Pair schedules, topology, seeds, and schemas match exactly.
5. Every capture binds to one manifest, protocol, plan, build, and schema
   identity.
6. Every treatment has exactly one successful start and one successful stop
   command at the declared logical indices.
7. Controls have zero action commands, action IDs are globally unique, and
   cleanup shows no active intervention.
8. Forbidden truth fields are absent from observation channels.
9. Event trace linkage and completed-path coverage are each at least 95%.
10. There are no cross-case trace references.
11. Every treatment passes its signed effect and recovery gates.
12. Placebo false-positive rate is at most 10%.
13. Every declared lane is observed and active projects share no project
    namespace, capture directory, network, or volume.
14. The artifact manifest hashes every final evidence file.

Pilot collection additionally requires a supported smoke assessment and its
bound protocol digest.

## Retry and attrition policy

There are no automatic retries. Each pair has one attempt.

Effect, recovery, trace-link, and other scientific gate failures are never
retryable. Infrastructure failures, partial captures, health-check timeouts,
and operator interruption also remain visible in the assessment and block the
current protocol rather than being silently replaced.

A manual rerun requires a new output directory and protocol version. All
failures remain in the denominator of the reported stage result.

## Evidence package

The stage artifact must contain:

- the frozen protocol and its canonical digest;
- the generated, fully materialized pair plan;
- every wrapper manifest;
- raw metrics, application logs, action commands, and traces by opaque case;
- runner logs and Compose-lane attestation;
- build, image, graph, feature-schema, and Collector identities;
- the recomputed data-quality assessment;
- a human-readable report with all denominators and failures; and
- a SHA-256 manifest covering every final file except itself.

The report must give per-action counts, per-topology counts, exact failed pair
IDs, effect and recovery summaries, placebo behavior, trace denominators, and
attrition. It must not hide failed pairs behind a per-protocol subset.

## Decision

If every gate passes, proceed to the separately planned 120-pair open
development corpus. Pilot ranges may inform that generator, but any change is
recorded in a new development protocol.

If any gate fails, publish the bounded negative instrumentation result and
stop. The report must identify whether the failure came from action delivery,
effect observability, recovery, trace continuity, truth separation, capture
identity, or concurrent isolation. No model training is authorized by a
failed instrumentation pilot.
