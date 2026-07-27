# OTLP replay vertical slice

## Status

Accepted for implementation on 2026-07-26 from the build plan agreed in the
preceding conversation.

## Claim under test

An OTLP JSON capture can be replayed into deterministic, semantically correct
model features without silently treating absent telemetry as zero.

This slice covers metrics. Logs, traces, exponential histograms, overlapping
writers, spatial aggregation, and a production OTLP receiver remain out of scope.

## Confirmed public seams

### Capture reader

`read_otlp_capture(path) -> TelemetryCapture`

The reader accepts newline-delimited OTLP JSON
`ExportMetricsServiceRequest` messages produced by the OpenTelemetry Collector
file exporter. It returns canonical metric points, capture SHA-256, resource and
point attributes, scope identity, metric kind, temporality, and flags.

Malformed or unsupported inputs fail with an error containing the JSON line and
metric name where available.

### OTLP window compiler

`OtlpWindowCompiler(feature_spec).compile(capture) -> CompiledTelemetry`

The compiler aligns event-time observations to fixed windows and implements:

- Gauge: last observed value in the window.
- Delta sum: interval value divided by its duration.
- Cumulative sum: difference between consecutive points divided by elapsed time.
- Delta histogram mean: interval sum divided by interval count.
- Cumulative histogram mean: difference in sum divided by difference in count.
- Counter and histogram reset detection using start time and decreasing values.
- `NO_RECORDED_VALUE` and absent features as explicit missing values.

The result includes values, observed masks, reset masks, event-time window ends,
feature-schema ID, capture hash, and data-quality counts.

### Evaluation adapter

`materialize_compiled_telemetry(compiled, policy) -> MaterializedTelemetry`

Materialization is explicit. The initial supported policy is bounded forward
fill, which returns both model values and an imputation mask. Leading missing
values or gaps beyond the configured limit fail rather than becoming zero.

## Feature specification

Each output feature declares:

- Stable output name.
- OTLP metric name.
- Statistic: `gauge_last`, `sum_rate`, `histogram_mean`, or
  `histogram_count_rate`.
- Optional resource-attribute and point-attribute equality filters.

Feature order is specification order and becomes part of the schema ID.

## Evidence protocol

- Golden OTLP JSON fixtures use the official integer enum representation.
- Replaying identical capture bytes and feature specification must produce
  identical values, masks, schema ID, and capture hash.
- Tests use worked counter reset, delta, histogram, late-order, flag, and missing
  examples with independently calculated expected values.
- A Collector configuration and repeatable emitter demonstrate a real
  Collector file-exporter round trip.
- Generated evidence records Collector image version, config hash, capture hash,
  feature schema, data-quality counts, and replay results.

## Acceptance gates

1. All worked semantic examples pass.
2. Golden replay is deterministic.
3. Missing telemetry remains explicit through compilation and materialization.
4. A capture emitted through the pinned Collector image replays successfully.
5. Existing synthetic evaluation and type checking remain green.

Passing supports only the deterministic metrics-replay claim. It does not
establish production receiver throughput, clock-skew handling, full OTLP
coverage, or real-world anomaly detection.
