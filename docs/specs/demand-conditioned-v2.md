# Quantis demand-conditioned model v2 milestone

## Status

Accepted for implementation on 2026-07-26 after the frozen v1 detector failed
the held-out fault matrix on false-positive rate.

## Evidence motivating the change

The v1 detector achieved 3/3 structural recall and 3/3 attribution hit@3 but
alerted on 87/108 normal pre-noise windows and 18/21 routine-noise response
windows. Median feature evidence moved together for request, worker, and
database rates under unseen load patterns. The model learned the development
schedule rather than the invariant relationship between admitted demand and
completed work.

The v1 matrix is development evidence from this point onward. It may be used as
a regression set but never again described as untouched confirmation.

## Confirmed public seams

### Demand conditioning

`DemandConditioner.transform(values, feature_names) -> ConditionedTelemetry`

The transform treats observed request rate as exogenous demand. It removes
request rate as an anomaly target and replaces worker and database write rates
with their ratios to request rate. Latency, error rate, queue depth, and worker
heartbeat age remain directly observed features.

The six model features are:

1. `request_latency_ms`
2. `error_rate`
3. `queue_depth`
4. `worker_completion_ratio`
5. `worker_heartbeat_age_s`
6. `db_write_completion_ratio`

The transform is deterministic, versioned, serializable, and rejects zero or
negative request demand rather than inventing a denominator.

### Multi-schedule training

`train_demand_conditioned_model(runs, feature_spec) ->
DemandConditionedModel`

Every training run declares one fault-free interval. Normalization is fitted on
the union of those points. Temporal windows are compiled separately per run and
combined only afterward, so no context crosses a capture boundary. Training
must contain at least three distinct load schedules and zero structural points.

The v2 coherent detector uses consensus rank two and a residual-scale floor of
0.001 normalized units. This makes a deviation in a normally constant
completion-ratio feature observable without changing the frozen v1 detector.
The alert threshold remains twice the 98th percentile of training scores.

### Frozen evaluation

`evaluate_demand_conditioned_fault_matrix(runs, feature_spec,
model_artifact_bytes, confirmation_protocol_bytes=None,
preregistered_git_commit=None) -> FaultMatrixReport`

The evaluator hashes and restores the exact model artifact bytes, performs no
fitting, applies the serialized conditioner, and maps raw affected-feature
truth to conditioned feature names for attribution. It compares canonical
realized request schedules and fault-kind/timing pairs with serialized training
provenance. Unseen data without a protocol is only out-of-sample validation.

## Development regression gate

Before confirmation, v2 must score the existing three-case matrix with:

- 3/3 structural event recall;
- 3/3 attribution hit@3;
- detection within six logical windows;
- at most 20% aggregate pre-noise alerts; and
- at most 20% aggregate routine-noise response alerts.

Passing this gate is development evidence only.

## Untouched confirmation protocol

After v2 artifacts and all evaluator limits are frozen, declare three new
manifests with realized load schedules and fault-kind/timing pairs absent from
both v1 development and regression data. Commit a protocol containing hashes of
the model, manifests, feature specification, preprocessing, evaluator, and
acceptance limits before capture. Record that full Git commit in the report and
verify every frozen file is recoverable from it. Run worker-crash,
database-lock, and cache-outage cases in fresh topologies. Do not alter v2, the
manifests, preprocessing, or limits after observing results.

Confirmation uses the same nine operational gates as the first held-out matrix:
complete fault coverage, raw effects, 100% recall, six-window maximum delay,
20% aggregate pre-noise and noise-response alert limits, 100% attribution
hit@3, one application image/build, and content-addressed inputs.

## Required limitations

- Demand ratios assume positive observed request demand in every window.
- Three confirmation cases do not establish production fault diversity.
- Training and confirmation share one local topology and telemetry vocabulary.
- Completion ratios encode a domain assumption that admitted requests should
  lead to worker and database completions.
- Feature evidence remains associative rather than causal.
- The target encoder remains linear PCA rather than a learned JEPA encoder.
