# Action-conditioned graph dynamics v1

## Status

Phase 0 synthetic tracer bullet completed on 2026-07-28. The next executable
stage is Phase 1 lab actions, recovery, and trace instrumentation.

This milestone is designed to produce a publishable result even if the
strongest generalization panels fail. The primary claim is deliberately
restricted to the fixed Quantis checkout stack and intervention families seen
during development. Location, composition, and topology transfer are reported
as separate generalization panels.

## Problem Statement

The current hybrid telemetry JEPA encodes nominal history well enough to
improve a frozen future-state probe over raw history and matched PCA. It does
not preserve recoverable local state, benefit measurably from declared
topology, or benefit materially from structured application events. The
nominal corpus contains no interventions and therefore cannot identify how an
action changes the stack, where a disturbance originated, or how it propagates
and recovers.

The next milestone needs intervention-conditioned transition evidence rather
than another nominal representation sweep.

## Claim under test

Given recent node/edge telemetry, the declared checkout graph, future workload
controls, and a declared intervention trajectory, a learned temporal graph
state-space model can:

1. predict multi-step metric and event trajectories during intervention and
   recovery on held-out schedules and severities;
2. outperform matched action-agnostic, topology-shuffled, linear, persistence,
   and raw-history controls;
3. preserve recoverable per-entity state with a non-collapsed latent;
4. rank the injected intervention and target from observations when the true
   action label is hidden; and
5. estimate a propagation path whose affected entities and timing agree with
   matched no-action runs and trace-linked observations.

This is a constrained claim about one fixed local stack and a finite,
predeclared intervention library. It is not a claim of arbitrary root-cause
analysis, unseen-mechanism discovery, production transfer, or a universal
software world model.

## Solution

Collect randomized, reversible intervention/recovery trajectories from
isolated copies of the existing API, queue, worker, Redis, and PostgreSQL lab.
Pair every intervention capture with a no-action capture using the same
workload and topology seed.

Compile metrics, structured application events, traces, workload controls,
topology, and the external action schedule into graph-owned trajectories.
Train a supervised action-conditioned temporal graph state-space model to
roll forward node/edge state distributions and event intensities. Use the same
rollout interface for forecasting and counterfactual attribution.

At attribution time, hide the injected action label. Roll out every candidate
action/target trajectory plus a no-action candidate and rank them by the
likelihood of the observed future. The predicted difference between the
winning action rollout and its no-action rollout is the estimated propagation
path.

## Primary interface

The highest external seam is one rollout operation:

`rollout(history, future_controls, candidate_actions, graph) -> trajectory_distribution`

The interface includes the following invariants:

- `history` contains only observable graph-owned metrics and events;
- lab truth such as fault kind, target, phase, manifest identity, and split is
  excluded from observation features;
- `future_controls` contains exogenous demand and declared capacity;
- `candidate_actions` contains external commands, not observations inferred
  from their effects;
- the graph and all feature, control, action, entity, and relation names are
  part of artifact identity;
- rollout steps are aligned to logical event time;
- output distributions preserve entity and feature ownership; and
- restored models reject incompatible schemas or topology unless an explicit
  ablation mode is requested.

Forecasting supplies the actual planned action trajectory. Attribution calls
the same interface repeatedly with candidate trajectories while withholding
the true label.

## User Stories

1. As a researcher, I want an explicit action trajectory separated from
   observed state, so that the model learns transitions rather than fault-label
   leakage.
2. As a researcher, I want each action assigned to a declared node or edge, so
   that location transfer and propagation can be measured.
3. As a researcher, I want randomized onset, duration, severity, workload, and
   topology, so that intervention identity is not confounded with schedule.
4. As a researcher, I want a matched no-action twin for each intervention run,
   so that intervention effects can be evaluated against the trajectory that
   the same workload would otherwise produce.
5. As a researcher, I want reversible actions and a long post-action interval,
   so that the model must predict both degradation and recovery.
6. As a researcher, I want request context propagated through the queue and
   worker, so that graph propagation can be checked against trace-linked
   observations.
7. As a researcher, I want preprocessing fitted on training runs only, so that
   held-out results are not contaminated.
8. As a researcher, I want multi-step probabilistic rollouts, so that error,
   calibration, and uncertainty degradation are visible by horizon.
9. As a researcher, I want strong linear and nonlinear controls, so that a
   neural graph model receives credit only for behavior simpler models cannot
   explain.
10. As a researcher, I want the true action hidden during attribution, so that
    intervention ranking measures inference rather than label decoding.
11. As a researcher, I want a no-action attribution candidate, so that nominal
    trajectories are not forced to have a root cause.
12. As a researcher, I want held-out schedule and severity confirmation, so
    that the primary result is publishable within the fixed stack.
13. As a researcher, I want location, combination, and topology panels
    separated from the primary gate, so that ambitious transfer tests do not
    blur the narrow supported claim.
14. As a researcher, I want per-entity state recovery and effective-rank
    diagnostics, so that a predictive shortcut is not mistaken for a usable
    system state.
15. As an operator, I want candidate intervention ranking with calibrated
    uncertainty, so that unsupported mechanisms can be returned as
    out-of-distribution instead of receiving a confident attribution.
16. As an operator, I want predicted affected nodes, edges, onset, and recovery
    time, so that an attribution can be checked against telemetry.
17. As an engineer, I want one content-addressed experiment protocol, so that
    every capture, split, model, and report is reproducible.
18. As an engineer, I want isolated lab stacks to run concurrently, so that
    corpus collection is bounded by machine capacity rather than serial Docker
    startup.
19. As an engineer, I want one accelerator training job at a time and CPU
    baselines in parallel, so that MPS contention does not invalidate runtime
    comparisons.
20. As a reviewer, I want the evidence report regenerated from raw captures
    and frozen artifacts, so that reported gates are independently checkable.

## Intervention protocol

### Action representation

Each manifest declares an ordered action timeline. One action contains:

- immutable action and matched-pair identifiers;
- action kind and parameter schema version;
- target graph entity;
- command phase: start, update, or stop;
- logical start and stop indices;
- magnitude values in physical units;
- workload seed, intervention seed, and topology identifier; and
- the expected raw-effect and recovery checks used to validate the capture.

The compiled action tensor contains:

- a no-action indicator;
- action-kind identity;
- target-entity identity;
- phase identity;
- normalized magnitude;
- elapsed and remaining duration; and
- an action-applicability mask.

An active-fault label is not included among observation features. Manifest
truth is available only as the supplied conditioning action and as evaluation
truth.

### Initial reversible intervention library

1. `worker_pause` targets `worker_pool` and pauses consumption without killing
   the process. Stop resumes consumption.
2. `postgres_lock` targets `worker_writes_postgresql` and holds the advisory
   lock already used by the lab. Stop releases it.
3. `redis_enqueue_delay` targets `api_enqueues_queue` and injects bounded
   client-side delay. Stop clears it.
4. `redis_dequeue_delay` targets `queue_dequeues_to_worker` and injects bounded
   client-side delay. Stop clears it.
5. `api_rejection` targets `api` and applies a declared rejection probability.
   Stop restores normal admission.

Worker crash remains an evaluation-only irreversible stress case. It is not a
training action in v1 because it does not provide a clean recovery transition.

Severity is parameterized rather than encoded as a different action kind.
Delay and rejection actions use at least three development severity bands;
pause and lock actions vary duration and affected worker fraction where the
topology permits it.

### Trace and event instrumentation

- Generate one W3C trace context at API admission.
- Carry it in the Redis work item and restore it in the worker.
- Emit spans for API admission, Redis enqueue, queue residence, Redis dequeue,
  worker processing, and PostgreSQL write.
- Attach trace/span identifiers to eligible application events.
- Add a trace pipeline to the pinned Collector and content-address its output.
- Emit action-command events from the runner, not the application, with action
  identity, target, phase, magnitude, and logical index.
- Keep runner action events in the conditioning channel; never compile them as
  application observations.

### Trajectory shape

Each development trajectory has at least:

- 20 clean context windows;
- 8 randomized pre-action windows;
- 8 to 20 active-action windows; and
- 24 post-stop recovery windows.

Onset, duration, severity, load pattern, and worker replica count are
randomized independently within predeclared bounds. The matched no-action twin
uses the identical workload and topology schedule.

## Corpus plan

### Stage A: instrumentation pilot

Collect 30 intervention/no-action pairs across all five action kinds. This
stage validates action alignment, effect magnitude, recovery, trace linkage,
parallel stack isolation, and capture determinism. It is not used for a
scientific claim.

### Stage B: open development corpus

Collect 120 additional pairs:

- every action kind appears across every supported topology;
- severity and duration are continuously jittered within declared ranges;
- workload schedules are generated before capture;
- the same raw run never appears in more than one split; and
- split assignment is by matched-pair identifier, not by window.

Use this corpus for architecture and threshold selection. Results remain
development-only.

### Stage C: sealed confirmation corpus

After freezing the compiler, action vocabulary, model, hyperparameters,
candidate library, and acceptance gates, generate 60 new matched pairs from
previously unused seeds. Do not open confirmation results until every capture
passes independent data-quality and raw-effect checks.

The planned total is 210 matched pairs, or 420 isolated captures. At roughly
24 seconds of workload per capture, six concurrent stacks have a lower bound
near 28 minutes; Docker startup, drain, hashing, and retries will make 45 to 75
minutes a more realistic collection budget.

## Generalization panels

### Primary: held-out schedule and severity

All action kinds and target entities are represented during development.
Confirmation uses unseen workload seeds, onset times, durations, and severity
values. This is the only panel required for the first constrained
action-conditioned dynamics claim.

### Secondary: topology transfer

Fit on one- and two-worker topologies and evaluate three workers without
refitting. Report separately; do not merge it into the primary mean.

### Secondary: location transfer

For the shared latency mechanism, withhold one dependency edge from fitting
and evaluate it as a target at confirmation. This measures whether typed graph
factorization transfers a known mechanism to a new location.

### Exploratory: composition

Fit only single interventions and evaluate pairs of simultaneous or partially
overlapping known interventions. Composition is not a v1 promotion gate.

### Exploratory: unseen mechanism

Use worker crash as an out-of-library disturbance. The desired behavior is
high predictive uncertainty or rejection of every known candidate, not correct
closed-set classification.

## Model design

### State encoder

Encode the last 20 steps of graph-owned operational metrics and structured
events into one latent per node and edge. Preserve a direct state-recovery
decoder throughout training. Do not make JEPA loss the primary objective.

### Transition model

Use a residual temporal graph transition:

`z[t+1] = z[t] + transition(z[t], control[t], action[t], graph)`

Typed message passing handles graph propagation. A temporal module summarizes
recent state and permits autoregressive multi-step rollout. Action target and
kind embeddings enter at the targeted entity; propagation must occur through
declared typed edges.

### Observation heads

- Predict continuous operational state with a heteroscedastic robust
  distribution.
- Predict bounded rates with an appropriate transformed continuous head.
- Predict structured-event occurrence or intensity as sparse auxiliary
  targets.
- Predict time to the next selected state-transition event.
- Decode the current observed state from the exported latent.

Exact log text generation is out of scope. Template identity, outcome, event
intensity, and timing are the useful supervision.

### Training objective

The primary loss is supervised multi-step state and event prediction with
horizon weighting. Add:

- current-state reconstruction;
- intervention-effect prediction relative to the matched no-action twin;
- event occurrence/intensity and time-to-event losses;
- variance/covariance regularization only if rank diagnostics require it; and
- optional masked JEPA prediction as a low-weight auxiliary ablation.

Scheduled sampling may be introduced after one-step behavior is verified.
Teacher-forced and free-running metrics must be reported separately.

### Baselines and ablations

Compare against:

1. persistence and seasonal persistence;
2. training-only linear VARX/state-space regression with actions;
3. the same neural model without actions;
4. the same model with shuffled action targets;
5. the same model with shuffled topology;
6. a parameter-budget-matched all-entity temporal model without graph message
   passing;
7. metrics-only and no-application-event variants; and
8. optional JEPA-auxiliary versus no-JEPA training.

The action-conditioned model does not earn a graph-dynamics claim merely by
beating the current nominal JEPA.

## Attribution

At evaluation time:

1. provide pre-event history and future workload controls;
2. hide the manifest action from the attributor;
3. create a finite candidate set over allowed action kinds, targets,
   severities, onset times, and the no-action candidate;
4. roll out each candidate;
5. score observed future likelihood with model uncertainty included; and
6. rank candidates and return calibrated rejection when all candidates are
   implausible.

The estimated propagation path is the per-entity difference between the
winning action rollout and its no-action rollout. Validate affected-entity
ranking, onset ordering, peak effect, and recovery time against the paired
capture and trace-linked observations.

This is causal attribution only relative to randomized actions and the finite
candidate library. It is not proof of an arbitrary production root cause.

## Acceptance gates

### Data quality

1. Every capture is content-addressed and bound to one static manifest.
2. Action command start/stop coverage is 100%, with no duplicate action IDs.
3. No lab-truth field enters metric, application-event, or trace observation
   features.
4. Eligible application-event trace-link coverage is at least 95%.
5. Every intervention passes its independent effect and recovery checks.
6. Every matched pair has identical workload and topology schedules.
7. Preprocessing and normalization are fitted on development training pairs
   only.

### Representation and rollout

1. Current-state recovery NRMSE is at most 0.15.
2. Minimum per-entity effective-rank fraction is at least 0.25.
3. At horizons 5 and 10, action-conditioned intervention/recovery NRMSE beats
   the best action-agnostic nonlinear control by at least 10% relative.
4. At horizon 10, it beats linear action-conditioned VARX by at least 5%
   relative.
5. The declared graph beats shuffled topology by at least 5% relative on
   downstream-entity intervention-effect error.
6. All three training seeds favor the declared action-conditioned model and
   the paired 95% confidence interval excludes zero for the primary effect.
7. The nominal 90% predictive interval covers between 85% and 95% of primary
   confirmation observations.
8. Recovery-time mean absolute error is at most three logical windows.

### Attribution

1. Closed-library action-and-target hit@1 is at least 70%.
2. Hit@3 is at least 90%.
3. Median correct attribution occurs no later than three observed windows
   after action onset.
4. No-action specificity is at least 90% on matched control captures.
5. Affected-entity hit@3 is at least 90%.
6. Worker-crash out-of-library rejection is reported separately and is not a
   primary promotion gate.

Failure of a gate produces a bounded negative result and identifies whether
the failure lies in data, state preservation, transition prediction, topology,
or inverse attribution.

## Modules and seams

### Intervention manifest module

Own validation, canonical serialization, hashing, randomization provenance,
action timelines, matched-pair identity, and split assignment. Extend the
existing fault-matrix manifest semantics without exposing runner-specific
Redis keys to callers.

### Lab intervention module

Expose one command operation over an action manifest. Internally adapt action
kinds to Redis controls, PostgreSQL locks, and application delay/rejection
controls. Emit command evidence and guarantee cleanup.

### Action-conditioned corpus module

Fit training-only state/event/action transforms and compile aligned graph
trajectories. Own truth exclusion, trace alignment, pair alignment, masks,
normalization, caching, and schema identity behind one compiler interface.

### Graph dynamics module

Expose fit, rollout, serialize, and restore. Hide PyTorch network construction,
batching, device selection, scheduled sampling, and distribution heads.

### Counterfactual attribution module

Expose candidate ranking over the rollout interface. Own candidate expansion,
likelihood aggregation, rejection calibration, and propagation-difference
calculation.

### Assessment module

Consume frozen model outputs, matched-pair truth, and the protocol. Return
versioned gates, confidence intervals, generalization panels, limitations, and
the promotion decision.

## Testing Decisions

- Test modules through their public interfaces; do not assert private network
  layer structure.
- Extend existing manifest round-trip and invalid-input tests for action
  timelines, matched pairs, and truth exclusion.
- Use deterministic synthetic graph trajectories to prove that the corpus
  compiler aligns action onset, recovery, controls, traces, and targets without
  crossing run boundaries.
- Use a small synthetic causal graph to prove that rollout changes only when
  supplied actions change and that the attributor can recover a known action.
- Verify restored artifacts reject renamed controls, actions, entities,
  relations, features, or changed topology.
- Run an end-to-end six-pair smoke corpus before the 30-pair instrumentation
  pilot.
- Recompile checked-in evidence from raw captures and frozen artifacts rather
  than trusting report booleans.
- Preserve existing hybrid JEPA and fault-matrix tests as regression coverage.

## Execution sequence

### Phase 0: protocol and synthetic tracer bullet

1. Freeze manifest v3, action vocabulary, rollout interface, metrics, and
   evidence boundary.
2. Implement compiler/model/attributor interfaces against synthetic graph
   trajectories.
3. Establish persistence, VARX, and action-agnostic baselines.

Exit when a synthetic intervention can be rolled out, recovered by candidate
ranking, serialized, restored, and assessed end to end.

### Phase 1: lab actions, recovery, and traces

1. Add reversible intervention adapters.
2. Propagate W3C trace context through the queue.
3. Add trace and runner-action Collector outputs.
4. Build a parameterized, isolated, concurrent capture runner.

Exit when six smoke pairs pass raw effects, recovery, trace linkage, hashing,
and truth-exclusion checks.

### Phase 2: instrumentation pilot

Collect 30 pairs using up to six concurrent Compose projects. Inspect only
data-quality and raw-effect reports, then adjust intervention ranges before
freezing the development generator.

### Phase 3: open development

Collect 120 pairs, train the baseline matrix and graph model across three
seeds, and select the smallest model meeting state, rollout, calibration, and
attribution development gates.

CPU baselines, corpus compilation, and report generation may run in parallel.
Run only one MPS training job at a time. Benchmark MLX only if the frozen
PyTorch/MPS training matrix exceeds 45 minutes; do not maintain two training
implementations speculatively.

### Phase 4: freeze and confirm

Freeze all model and assessment choices, generate 60 new matched pairs, run
the confirmation once, and publish the primary result plus separate topology,
location, composition, and unseen-mechanism panels.

## Publishable outcomes

### Positive primary outcome

On the fixed Quantis checkout stack and a finite known intervention library,
the action-conditioned graph state-space model predicts held-out
intervention/recovery trajectories and ranks injected action targets better
than matched controls.

This supports the phrase **constrained action-conditioned world model for the
lab stack** only if every primary representation, rollout, topology, and
attribution gate passes.

### Negative primary outcome

Randomized actions improve forecasting but topology or attribution gates fail.
The publishable conclusion is that action supervision identifies transition
regimes, while the telemetry/graph factorization remains insufficient for
dependable propagation or localization.

### Strong negative outcome

Action conditioning does not beat linear VARX or action-agnostic controls.
The publishable conclusion is that this stack's observed dynamics are either
mostly linear, insufficiently excited, or not identifiable from the available
telemetry. Further neural representation work should stop until the corpus or
instrumentation changes.

## Out of Scope

- arbitrary natural-language generation from logs;
- production deployment or online remediation;
- actions outside the finite intervention vocabulary;
- a graph inferred from telemetry alone;
- causal guarantees under unrandomized production incidents;
- cross-organization or cross-stack zero-shot transfer;
- autonomous execution of counterfactual recovery actions; and
- replacing the existing nominal JEPA artifact.

## Further Notes

The model should first earn the right to use graph topology and actions. JEPA
is retained only as an optional auxiliary objective after state preservation
and supervised transition prediction are healthy.

The project has no configured issue-tracker target in the current workspace,
so this specification is stored with the repository specifications until the
user chooses a tracker.
