# Quantis held-out fault-matrix milestone

## Status

Accepted for implementation on 2026-07-26 as the milestone following the
instrumented worker-stall fault lab.

## Claim under test

The compiler and detector fitted by the development fault lab can be frozen and
used, without refitting or threshold adjustment, to detect three independently
captured fault classes under schedules not used to fit the model:

1. a worker process crash;
2. PostgreSQL advisory-lock contention; and
3. a logical Redis-backed cache outage at the API boundary.

This is controlled local evidence about transfer across fault mechanisms and
load schedules. It is not evidence of production-scale performance, arbitrary
zero-day detection, or causal root-cause analysis.

## Frozen public seams

### Operator entrypoint

`./lab/fault_matrix/run-fault-matrix.sh`

The command builds one content-addressed application image, runs every static
case in a fresh topology, captures OTLP metrics, evaluates all captures with the
checked-in development compiler and detector, writes aggregate evidence, and
exits non-zero when an acceptance gate fails.

### Evaluation boundary

`evaluate_fault_matrix(runs, feature_spec, window_compiler_artifact_bytes,
detector_artifact_bytes) -> FaultMatrixReport`

Each run contains a raw `TelemetryCapture` and a static
`FaultMatrixCaseManifest`. The evaluator restores fitted artifacts with
The evaluator hashes and decodes the exact supplied bytes, restores state with
`from_dict`, and never calls `fit`. Acceptance limits are module constants, not
evaluator arguments. The frozen artifact hashes, per-case
capture hashes, application image identity, build-context hash, schedules, raw
effects, detection results, and attribution results are recorded.

## Experimental controls

- The development artifacts in `artifacts/fault-lab` are frozen before any
  matrix capture is scored. Their file hashes are recorded before the first
  case, compared again after the last case, passed to the evaluator, and
  verified against the files used for scoring.
- Every case uses a new Redis volume, PostgreSQL volume, API process, worker
  process, runner, and Collector output.
- Case manifests differ in load-pattern phase and structural-fault timing from
  the development schedule.
- Every OTLP resource records its case ID, fault kind, and canonical manifest
  hash; the evaluator rejects a capture paired with different manifest truth.
- Routine noise remains one isolated delayed request. Its alert accounting
  includes the injection plus the complete six-window predictor-context
  response horizon.
- Pre-fault false-positive accounting includes every scoreable held-out point
  before routine noise, not the development run's eight validation points.
- The matrix runner may repair implementation bugs, but may not change the
  frozen compiler, detector, threshold, context-repair rule, or acceptance
  limits after observing case outcomes.

## Fault mechanisms and independent raw-effect gates

### Worker crash

The worker exits with a non-zero status after receiving an external control
key. The process is not restarted during the case. Redis backlog must grow by
at least 20 jobs and median worker and database write rates must fall to at most
20% of their pre-fault medians.

### Database lock

The runner holds a PostgreSQL advisory lock also required by the worker's insert
transaction. The API and Redis remain available. Redis backlog must grow by at
least 20 jobs and median worker and database write rates must fall to at most
20% of their pre-fault medians.

### Cache outage

An external control key makes the API return HTTP 503 instead of enqueuing
checkout work while Redis remains available for observability. Median error
rate must reach at least 80%, and median worker and database write rates must
fall to at most 20% of their pre-fault medians.

Every case must also demonstrate that the routine-noise injection raised median
request latency to at least three times its pre-fault median.

## Acceptance gates fixed before capture

1. All three declared fault kinds have exactly one complete capture.
2. Frozen compiler and detector artifacts are content-addressed and are
   unchanged by evaluation.
3. Every case passes its independent raw-effect gates.
4. Structural event recall is 100% (three of three).
5. Every detected event begins within six logical windows of injection.
6. Aggregate routine-noise response-horizon alert rate is at most 20%.
7. Aggregate pre-noise held-out alert rate is at most 20%.
8. Attribution hit@3 is 100% (three of three), using affected features declared
   before capture.
9. Captures, feature schema, manifests, service images, application image, and
   application build context are content-addressed.

## Required limitations in the evidence report

- Three local cases are not representative of production fault diversity.
- The topology and telemetry vocabulary remain the same as development.
- The cache fault is a logical application-path outage, not a killed Redis
  process, because Redis also carries this lab's public counters.
- Load schedules and fault mechanisms are held out from fitting, but were
  authored by the same development team.
- Logical event-time windows are sampled faster than wall clock.
- Feature evidence is associative attribution, not causal proof.
- The target encoder remains linear PCA rather than a learned JEPA encoder.
