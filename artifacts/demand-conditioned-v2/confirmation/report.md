# Quantis demand-conditioned v2 confirmation

Overall acceptance: **PASS**

## Aggregate evidence

- Structural event recall: 3/3
- Attribution hit@3: 3/3
- Maximum detection delay: 0 logical windows
- Routine-noise response alerts: 3/21
- Pre-noise confirmation alerts: 22/148

## Cases

### cache-outage-confirmation-04

- Fault kind: `cache_outage`
- Detected: True
- Detection delay: 0 logical windows
- Attribution top three: db_write_completion_ratio, worker_completion_ratio, error_rate
- Pre-noise score median / threshold: 0.597 / 2.951
- Raw-effect gates passed: True

### database-lock-confirmation-04

- Fault kind: `database_lock`
- Detected: True
- Detection delay: 0 logical windows
- Attribution top three: queue_depth, db_write_completion_ratio, worker_completion_ratio
- Pre-noise score median / threshold: 0.597 / 2.951
- Raw-effect gates passed: True

### worker-crash-confirmation-04

- Fault kind: `worker_crash`
- Detected: True
- Detection delay: 0 logical windows
- Attribution top three: queue_depth, db_write_completion_ratio, worker_completion_ratio
- Pre-noise score median / threshold: 0.510 / 2.951
- Raw-effect gates passed: True

## Acceptance gates

- PASS: `complete_fault_kind_coverage`
- PASS: `frozen_artifacts_unchanged`
- PASS: `one_application_image_and_build`
- PASS: `all_raw_fault_effects_observed`
- PASS: `all_captures_match_manifests`
- PASS: `structural_event_recall_is_one`
- PASS: `all_detection_delays_within_limit`
- PASS: `aggregate_routine_noise_alert_rate_within_limit`
- PASS: `aggregate_pre_noise_alert_rate_within_limit`
- PASS: `attribution_hit_rate_at_3_is_one`
- PASS: `content_addressed_inputs`

## Limitations

- Preregistration attests frozen inputs and disjoint cases, canonical realized request schedules, and fault timings; three local cases still provide limited external validity.
- Demand ratios assume positive observed request demand in every window.
- Completion ratios encode a domain assumption that admitted requests should lead to worker and database completions.
- Three local cases are not representative of production fault diversity.
- The topology and telemetry vocabulary remain the same as development.
- The cache fault is a logical application-path outage, not a killed Redis process, because Redis also carries this lab's public counters.
- Logical event-time windows are sampled faster than wall clock.
- Feature evidence is associative attribution, not causal proof.
- The target encoder is linear PCA, not a learned JEPA encoder.
