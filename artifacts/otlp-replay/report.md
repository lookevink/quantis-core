# Quantis OTLP Collector round-trip verification

Overall acceptance: **PASS**

- Collector: `ghcr.io/open-telemetry/opentelemetry-collector-releases/opentelemetry-collector-contrib:0.153.0@sha256:93aad750175cbf1a973ae1c5886c3371f4d800f61be25cdd26870b8441ffe9fa`
- Capture SHA-256: `d62f959f1f6cef4e5d8baa7e69f2d5980891de0bd954d6b2c7b1ff0514714eb2`
- Feature schema: `6fd422e11f03b343ab9084179fda8234d11945cf16bbd2a5ebd3bda7b8b34eac`
- Windows × features: 240 × 12
- Missing cells: 0
- Runtime-format action counts: {'quantis.experiment.error_count': 3.0, 'quantis.experiment.request_count': 12.0}
- Maximum value difference: 0
- Maximum score difference: 0
- Alerts identical: True
- Structural event detected: True
- Routine-noise alert rate: 0.000

## Gates

- PASS: `capture_matches_golden`
- PASS: `no_missing_cells`
- PASS: `values_match_direct_path`
- PASS: `scores_match_direct_path`
- PASS: `alerts_match_direct_path`
- PASS: `structural_event_detected`
- PASS: `runtime_format_action_counts`
- PASS: `compiled_matches_golden`

## Limitations

- The Collector round trip carries deterministic synthetic gauges, not telemetry from a production workload.
- The parity result validates transport and replay semantics, not real-world anomaly detection.
- The checked-in capture covers gauges; worked fixtures separately cover sums, histograms, resets, flags, and missingness.
