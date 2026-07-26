# Quantis synthetic evaluation

Overall acceptance: **PASS**

This report evaluates a linear latent predictive detector and a coherence-aware variant. It does not establish the behavior of a learned JEPA.

## Results

| Detector | Noise alert rate | Structural event recall | Attribution hit@3 | Attribution recall@3 | Mean delay | ms / point |
|---|---:|---:|---:|---:|---:|---:|
| persistence | 0.196 | 0.000 | 0.000 | 0.000 | n/a | 0.013864 |
| robust_feature | 0.375 | 1.000 | 1.000 | 1.000 | 49.12 | 0.009375 |
| linear_latent_predictive | 0.250 | 0.500 | 1.000 | 0.167 | 28.75 | 0.019790 |
| coherent_latent_predictive | 0.080 | 1.000 | 1.000 | 0.708 | 12.25 | 0.014838 |

## Acceptance gates

- PASS: `structural_event_recall_at_least_0_8`
- PASS: `routine_noise_alert_rate_at_most_0_1`
- PASS: `attribution_hit_at_3_at_least_0_8`
- PASS: `attribution_hit_at_3_above_random_chance`
- PASS: `fewer_noise_alerts_than_persistence`
- PASS: `mean_scoring_below_1_ms_per_point`

## Limitations

- Synthetic scenarios share one generator family and are not evidence of real-world zero-day detection.
- The linear latent target encoder is fitted with PCA; this is not a learned JEPA encoder or evidence for JEPA-specific advantages.
- Injected affected-feature labels support associative attribution evaluation, not causal root-cause identification.
- The runtime measurement scores one window at a time in Python; it is not a production OpenTelemetry throughput benchmark.
