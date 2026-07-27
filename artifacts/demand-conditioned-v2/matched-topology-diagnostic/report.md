# Quantis matched-topology v2 diagnostic

Overall acceptance: **FAIL**

## Aggregate evidence

- Structural event recall: 9/9
- Attribution hit@3: 9/9
- Maximum detection delay: 0 logical windows
- Routine-noise response alerts: 27/63
- Pre-noise diagnostic alerts: 133/393

## Topology strata

- `workers-1`: recall 3/3, attribution 3/3, pre-noise 46/131, noise 7/21
- `workers-2`: recall 3/3, attribution 3/3, pre-noise 44/131, noise 9/21
- `workers-3`: recall 3/3, attribution 3/3, pre-noise 43/131, noise 11/21

## Matched topology diagnostic

- Classification: `no_material_topology_effect`
- Reference topology: `workers-1`
- Material paired risk difference: 20.0%
- `workers-2` paired pre-noise risk differences: -9.5%, -13.6%, +17.8% (mean -1.8%)
- `workers-3` paired pre-noise risk differences: -4.8%, -13.6%, +11.1% (mean -2.4%)

## Cases

### matched-workers-1-cache-outage-11

- Fault kind: `cache_outage`
- Topology: `workers-1` (1 workers)
- Detected: True
- Detection delay: 0 logical windows
- Attribution top three: db_write_completion_ratio, worker_completion_ratio, error_rate
- Pre-noise score median / threshold: 1.009 / 2.951
- Raw-effect gates passed: True

### matched-workers-1-database-lock-11

- Fault kind: `database_lock`
- Topology: `workers-1` (1 workers)
- Detected: True
- Detection delay: 0 logical windows
- Attribution top three: queue_depth, db_write_completion_ratio, worker_completion_ratio
- Pre-noise score median / threshold: 80.128 / 2.951
- Raw-effect gates passed: True

### matched-workers-1-worker-crash-11

- Fault kind: `worker_crash`
- Topology: `workers-1` (1 workers)
- Detected: True
- Detection delay: 0 logical windows
- Attribution top three: queue_depth, db_write_completion_ratio, worker_completion_ratio
- Pre-noise score median / threshold: 0.978 / 2.951
- Raw-effect gates passed: True

### matched-workers-2-cache-outage-11

- Fault kind: `cache_outage`
- Topology: `workers-2` (2 workers)
- Detected: True
- Detection delay: 0 logical windows
- Attribution top three: db_write_completion_ratio, worker_completion_ratio, error_rate
- Pre-noise score median / threshold: 0.687 / 2.951
- Raw-effect gates passed: True

### matched-workers-2-database-lock-11

- Fault kind: `database_lock`
- Topology: `workers-2` (2 workers)
- Detected: True
- Detection delay: 0 logical windows
- Attribution top three: queue_depth, db_write_completion_ratio, worker_completion_ratio
- Pre-noise score median / threshold: 1.496 / 2.951
- Raw-effect gates passed: True

### matched-workers-2-worker-crash-11

- Fault kind: `worker_crash`
- Topology: `workers-2` (2 workers)
- Detected: True
- Detection delay: 0 logical windows
- Attribution top three: queue_depth, db_write_completion_ratio, worker_completion_ratio
- Pre-noise score median / threshold: 1.178 / 2.951
- Raw-effect gates passed: True

### matched-workers-3-cache-outage-11

- Fault kind: `cache_outage`
- Topology: `workers-3` (3 workers)
- Detected: True
- Detection delay: 0 logical windows
- Attribution top three: db_write_completion_ratio, worker_completion_ratio, error_rate
- Pre-noise score median / threshold: 1.027 / 2.951
- Raw-effect gates passed: True

### matched-workers-3-database-lock-11

- Fault kind: `database_lock`
- Topology: `workers-3` (3 workers)
- Detected: True
- Detection delay: 0 logical windows
- Attribution top three: queue_depth, db_write_completion_ratio, worker_completion_ratio
- Pre-noise score median / threshold: 1.112 / 2.951
- Raw-effect gates passed: True

### matched-workers-3-worker-crash-11

- Fault kind: `worker_crash`
- Topology: `workers-3` (3 workers)
- Detected: True
- Detection delay: 0 logical windows
- Attribution top three: queue_depth, db_write_completion_ratio, worker_completion_ratio
- Pre-noise score median / threshold: 0.812 / 2.951
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
- PASS: `complete_fault_topology_coverage`
- FAIL: `all_topology_strata_within_limits`
- PASS: `matched_topology_design_complete`

## Diagnostic interpretation

Holding the request schedule fixed removes the material normal-alert difference between worker-count treatments. The earlier expanded result is therefore better explained by its schedule confound than by worker count alone.

## Limitations

- Preregistration attests frozen inputs and disjoint cases, canonical realized request schedules, and fault timings; nine local cases across three worker-count strata still provide limited external validity.
- Demand ratios assume positive observed request demand in every window.
- Completion ratios encode a domain assumption that admitted requests should lead to worker and database completions.
- Worker replica count is only one dimension of topology diversity.
- Redis, PostgreSQL, API, Collector, host, and telemetry vocabulary remain unchanged.
- Nine controlled local cases do not estimate production incident prevalence.
- The cache fault is a logical application-path outage, not a killed Redis process, because Redis also carries this lab's public counters.
- Logical event-time windows are sampled faster than wall clock.
- Feature evidence is associative attribution, not causal proof.
- The target encoder is linear PCA, not a learned JEPA encoder.
