# Quantis multimodal JEPA v0 milestone

## Status

Accepted for implementation on 2026-07-27 after the metrics-only JEPA corpus
showed partial transfer to held-out request schedules.

## Claim under development

Structured application events add useful state to the metrics-only world model
without allowing unbounded message text, identifiers, secrets, or stack traces
to enter the learned feature vocabulary.

This is a development tracer bullet. It does not establish that application
logs improve anomaly detection or production generalization.

## Public seams

### OTLP application-log capture

`read_otlp_log_capture(path) -> OtlpLogCapture`

The reader accepts newline-delimited OTLP JSON Logs export requests, preserves
resource, scope, severity, event body, trace identity, and record attributes,
and fails loudly when a record cannot be interpreted without data loss.

### Deterministic log windows

`OtlpLogWindowCompiler(spec).compile(capture, window_count) -> CompiledLogTelemetry`

The compiler:

- assigns new lab records by emission time against observed metric-window
  boundaries, while retaining the request's origin index as provenance;
- aggregates only preregistered event-name and severity filters;
- sums events from multiple application instances in the same window;
- materializes explicit zeros for windows with no matching event;
- for legacy logical-window captures, rejects missing, negative, fractional,
  or out-of-range window identities;
- rejects event timestamps outside recorded run boundaries;
- never derives features from unrestricted message text; and
- records the source capture and feature-spec identities.

### Multimodal corpus

`compile_multimodal_telemetry_corpus(runs, log_captures, metric_spec, log_spec, split_spec) -> MultimodalTelemetryCorpus`

The corpus compiler verifies metric, log, and manifest identity for every run,
selects only declared normal intervals, fits preprocessing on training runs
only, and compiles each run independently. Metric and log contexts cannot cross
run boundaries. Entire canonical request schedules remain isolated between
training and validation.

### Multimodal JEPA

`MultimodalJepaWorldModelDetector.fit(windows) -> MultimodalJepaWorldModelDetector`

The metrics and application-log channels have separate online and EMA target
encoders. Their latent states are concatenated behind the detector interface,
and one predictor learns the next joint latent state. Serialization includes
both encoders, preprocessing state, calibration, seed, and training losses.

The command-line seam is:

`python -m quantis_core train-multimodal-jepa-world-model ...`

## Initial application vocabulary

- `checkout.accepted`
- `checkout.rejected`
- `checkout.completed`
- all records at error severity or higher

Every event carries a stable `event.name` and logical-window index. Request IDs,
payloads, exception messages, stack traces, and arbitrary message bodies are
excluded from model features.

## Development gates

- OTLP log parsing preserves stable semantics and identity.
- Log aggregation has independently worked expected values.
- Metric and log captures must match the same manifest case identity.
- Preprocessing is fitted on training runs only.
- Context never crosses run boundaries.
- Training and validation schedules are disjoint.
- Repeated training with the same inputs and seed is byte-identical.
- Restored artifacts reproduce scores exactly.
- Metrics-only and multimodal results are reported separately; neither is
  confirmation evidence.
