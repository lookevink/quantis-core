# Quantis instrumented fault-lab verification

Overall acceptance: **PASS**

## Observed system effects

- Redis queue growth: 96.0 jobs
- Worker fault/baseline rate ratio: 0.000
- Database fault/baseline write ratio: 0.000
- Routine-noise/baseline latency ratio: 46.3×

## Detection

- Structural event detected: True
- Detection delay: 0 logical windows
- Detection wall-time upper bound: 0.250s
- Routine-noise response-horizon alert rate: 0.000
- Pre-fault validation alert rate: 0.000
- Attribution top three: worker_heartbeat_age_s, db_write_rate, worker_rate
- Attribution hit@3: True

## Provenance

- Capture SHA-256: `a9106f93e33f7127eda2bda6de401f6733a8e656458f99c25bec5aee5d28b714`
- Application image ID: `sha256:abde408ce71d54de3de4a211e358c0a3a6efb7c33267fa400ba32b0fe51b24e9`
- Application build context SHA-256: `5934a0342f4f1d75a2b8096a659919ca926f9c34ee5b87c062e9e36664ef22da`

## Acceptance gates

- PASS: `complete_telemetry`
- PASS: `backlog_growth_at_least_minimum`
- PASS: `routine_noise_has_observed_effect`
- PASS: `worker_rate_collapses`
- PASS: `db_write_rate_collapses`
- PASS: `structural_event_detected`
- PASS: `non_degenerate_calibration_threshold`
- PASS: `detection_delay_within_limit`
- PASS: `routine_noise_alert_rate_within_limit`
- PASS: `validation_alert_rate_within_limit`
- PASS: `attribution_hit_at_3`
- PASS: `content_addressed_inputs`

## Limitations

- This is one controlled local topology and one injected worker stall.
- Evaluator preprocessing and threshold margin were developed against this topology and schedule; this is development evidence, not an untouched confirmatory experiment.
- False-positive evidence covers one noise point and eight validation points, plus the six-window noise-response horizon, in a single run.
- Logical event-time windows are sampled faster than wall clock.
- The detector is fitted and tested on different intervals of one run.
- Feature evidence is associative attribution, not causal proof.
- The target encoder is linear PCA, not a learned JEPA encoder.
