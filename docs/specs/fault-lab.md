# Quantis instrumented fault-lab milestone

## Status

Accepted for implementation on 2026-07-26 as the milestone following the
deterministic OTLP replay slice.

## Claim under test

Telemetry produced by a running API, worker, Redis queue, and PostgreSQL
database can pass through the pinned OpenTelemetry Collector and provide enough
multivariate evidence for the coherence-aware latent predictor to:

1. ignore an isolated request-latency disturbance;
2. detect a worker-stall fault promptly; and
3. attribute the event to at least one independently declared affected signal.

This is a controlled local fault experiment. It is not evidence of general
zero-day detection, production scale, or causal root-cause analysis.

## Confirmed public seams

### Operator entrypoint

`./lab/fault/run-fault-lab.sh`

The command builds the pinned topology, applies the experiment schedule,
captures OTLP metrics, evaluates the capture, writes evidence, and exits
non-zero if an acceptance gate fails.

### Telemetry boundary

The Collector file exporter writes newline-delimited OTLP JSON. The existing
`read_otlp_capture` and `OtlpWindowCompiler` public seams compile that capture.
No evaluator may read service-private state to manufacture model inputs.

### Evaluation boundary

`evaluate_fault_lab(capture, feature_spec, manifest) -> FaultLabReport`

The static experiment manifest is the independent source of phase labels and
affected-feature truth. The report is versioned and includes capture identity,
data quality, raw fault effects, detection latency, false-positive rate,
attribution, acceptance gates, and limitations.

## Topology and experiment

- A threaded HTTP API enqueues checkout work in Redis.
- A worker consumes the queue and writes completed work to PostgreSQL.
- A load runner sends a stable baseline workload.
- Routine noise delays one request per window without changing the other
  services.
- The structural fault sets an external control key that stalls worker
  consumption and heartbeat updates while API load continues.
- A metrics runner samples public counters and datastore probes, then exports
  gauges through OTLP/HTTP JSON.

The detector's false-positive accounting covers the injection window and the
following six-window predictor-context horizon. Before scoring, a fixed robust
context rule replaces a normalized context value with the cross-feature median
of the non-outlying signals in that timestep only when it exceeds eight robust
standard units and fewer than the detector's three consensus features are
outliers. Correlated outlier context is retained. This rule is fixed before the
fault run and is included in the hashed evaluator configuration.

The final alert threshold is twice the 98th-percentile training score. This
safety margin is derived only from baseline training scores; validation, noise,
and structural intervals do not calibrate model state or threshold.

The schedule uses logical one-second event-time windows sampled faster than
wall clock to keep the local experiment short. Both durations are recorded.

## Data separation

- Detector normalization, prediction weights, residual scales, and threshold
  are fitted only on the manifest's baseline training interval.
- Routine-noise and structural intervals are excluded from training.
- The checked-in evidence test re-hashes and recompiles the raw capture instead
  of trusting report booleans.

## Acceptance gates

1. Every expected telemetry cell is present and finite.
2. Redis backlog grows by at least 20 jobs during the worker stall.
3. The injected latency disturbance raises observed request latency to at
   least three times the baseline median.
4. Median worker and database write rates during the stall fall below 20% of
   their baseline medians.
5. The structural event produces at least one alert.
6. Detection delay is no more than six logical windows.
7. Routine-noise alert rate is no more than 20%.
8. Pre-fault validation alert rate is no more than 20%.
9. The baseline produces a positive, non-degenerate calibrated threshold.
10. At least one of the independently declared affected features appears in the
   top three attributed features at first detection.
11. The checked-in capture, feature schema, service images, experiment manifest,
   and evaluator configuration are content-addressed.

The application build-context hash covers, in sorted order, the Dockerfile,
experiment manifest, pinned requirements, API/worker source, and runner source.
The run also records the Docker image ID actually used by API, worker, and
runner; both values travel inside the OTLP resource attributes.
