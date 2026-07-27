# Quantis held-out fault-matrix verification

Overall acceptance: **FAIL**

## Aggregate evidence

- Structural event recall: 3/3
- Attribution hit@3: 3/3
- Maximum detection delay: 0 logical windows
- Routine-noise response alerts: 18/21
- Pre-noise held-out alerts: 87/108

## Cases

### cache-outage-held-out-01

- Fault kind: `cache_outage`
- Detected: True
- Detection delay: 0 logical windows
- Attribution top three: request_latency_ms, db_write_rate, worker_rate
- Pre-noise score median / threshold: 17.599 / 5.461
- Raw-effect gates passed: True

### database-lock-held-out-01

- Fault kind: `database_lock`
- Detected: True
- Detection delay: 0 logical windows
- Attribution top three: worker_heartbeat_age_s, db_write_rate, worker_rate
- Pre-noise score median / threshold: 9.753 / 5.461
- Raw-effect gates passed: True

### worker-crash-held-out-01

- Fault kind: `worker_crash`
- Detected: True
- Detection delay: 0 logical windows
- Attribution top three: worker_heartbeat_age_s, db_write_rate, worker_rate
- Pre-noise score median / threshold: 10.533 / 5.461
- Raw-effect gates passed: True

## Acceptance gates

- PASS: `complete_fault_kind_coverage`
- PASS: `frozen_artifacts_unchanged`
- PASS: `one_application_image_and_build`
- PASS: `all_raw_fault_effects_observed`
- PASS: `all_captures_match_manifests`
- PASS: `structural_event_recall_is_one`
- PASS: `all_detection_delays_within_limit`
- FAIL: `aggregate_routine_noise_alert_rate_within_limit`
- FAIL: `aggregate_pre_noise_alert_rate_within_limit`
- PASS: `attribution_hit_rate_at_3_is_one`
- PASS: `content_addressed_inputs`

## Diagnostic interpretation

The frozen predictor rejects most normal held-out windows. Median pre-noise evidence is jointly elevated for request, worker, and database rates, which move together with the new load schedules. This is evidence of schedule-pattern overfitting in the development model, not a threshold-only problem. Structural recall is therefore not operationally useful at the observed false-positive rate.

## Limitations

- Three local cases are not representative of production fault diversity.
- The topology and telemetry vocabulary remain the same as development.
- The cache fault is a logical application-path outage, not a killed Redis process, because Redis also carries this lab's public counters.
- Load schedules and fault mechanisms are held out from fitting, but were authored by the same development team.
- Logical event-time windows are sampled faster than wall clock.
- Feature evidence is associative attribution, not causal proof.
- The target encoder is linear PCA, not a learned JEPA encoder.
