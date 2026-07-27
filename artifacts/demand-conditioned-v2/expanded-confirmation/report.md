# Quantis demand-conditioned v2 confirmation

Overall acceptance: **FAIL**

## Aggregate evidence

- Structural event recall: 9/9
- Attribution hit@3: 9/9
- Maximum detection delay: 0 logical windows
- Routine-noise response alerts: 34/63
- Pre-noise confirmation alerts: 216/377

## Topology strata

- `workers-1`: recall 3/3, attribution 3/3, pre-noise 12/120, noise 0/21
- `workers-2`: recall 3/3, attribution 3/3, pre-noise 86/122, noise 13/21
- `workers-3`: recall 3/3, attribution 3/3, pre-noise 118/135, noise 21/21

## Cases

### expanded-workers-1-cache-outage-10

- Fault kind: `cache_outage`
- Topology: `workers-1` (1 workers)
- Detected: True
- Detection delay: 0 logical windows
- Attribution top three: db_write_completion_ratio, worker_completion_ratio, error_rate
- Pre-noise score median / threshold: 0.656 / 2.951
- Raw-effect gates passed: True

### expanded-workers-1-database-lock-10

- Fault kind: `database_lock`
- Topology: `workers-1` (1 workers)
- Detected: True
- Detection delay: 0 logical windows
- Attribution top three: queue_depth, db_write_completion_ratio, worker_completion_ratio
- Pre-noise score median / threshold: 0.707 / 2.951
- Raw-effect gates passed: True

### expanded-workers-1-worker-crash-10

- Fault kind: `worker_crash`
- Topology: `workers-1` (1 workers)
- Detected: True
- Detection delay: 0 logical windows
- Attribution top three: queue_depth, db_write_completion_ratio, worker_completion_ratio
- Pre-noise score median / threshold: 0.932 / 2.951
- Raw-effect gates passed: True

### expanded-workers-2-cache-outage-10

- Fault kind: `cache_outage`
- Topology: `workers-2` (2 workers)
- Detected: True
- Detection delay: 0 logical windows
- Attribution top three: db_write_completion_ratio, worker_completion_ratio, error_rate
- Pre-noise score median / threshold: 1.485 / 2.951
- Raw-effect gates passed: True

### expanded-workers-2-database-lock-10

- Fault kind: `database_lock`
- Topology: `workers-2` (2 workers)
- Detected: True
- Detection delay: 0 logical windows
- Attribution top three: queue_depth, db_write_completion_ratio, worker_completion_ratio
- Pre-noise score median / threshold: 125.000 / 2.951
- Raw-effect gates passed: True

### expanded-workers-2-worker-crash-10

- Fault kind: `worker_crash`
- Topology: `workers-2` (2 workers)
- Detected: True
- Detection delay: 0 logical windows
- Attribution top three: queue_depth, db_write_completion_ratio, worker_completion_ratio
- Pre-noise score median / threshold: 129.167 / 2.951
- Raw-effect gates passed: True

### expanded-workers-3-cache-outage-10

- Fault kind: `cache_outage`
- Topology: `workers-3` (3 workers)
- Detected: True
- Detection delay: 0 logical windows
- Attribution top three: error_rate, db_write_completion_ratio, worker_completion_ratio
- Pre-noise score median / threshold: 142.857 / 2.951
- Raw-effect gates passed: True

### expanded-workers-3-database-lock-10

- Fault kind: `database_lock`
- Topology: `workers-3` (3 workers)
- Detected: True
- Detection delay: 0 logical windows
- Attribution top three: queue_depth, db_write_completion_ratio, worker_completion_ratio
- Pre-noise score median / threshold: 142.857 / 2.951
- Raw-effect gates passed: True

### expanded-workers-3-worker-crash-10

- Fault kind: `worker_crash`
- Topology: `workers-3` (3 workers)
- Detected: True
- Detection delay: 0 logical windows
- Attribution top three: queue_depth, db_write_completion_ratio, worker_completion_ratio
- Pre-noise score median / threshold: 100.000 / 2.951
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

## Diagnostic interpretation

Normal alert rates are low in the one-worker stratum and high in the observed two- and three-worker strata. Worker count co-varies with workload schedule, so this establishes an association with multi-worker operation rather than isolated causality. The frozen model does not transfer operationally at the observed false-positive rates.

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
