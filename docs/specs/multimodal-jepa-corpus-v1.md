# Quantis multimodal JEPA corpus v1

## Status

Preregistered for collection on 2026-07-27. No run from this design may be
replaced, moved between splits, or excluded after its result is inspected.

## Question

Does an aligned, bounded-vocabulary application-log channel improve a distinct
metrics-plus-logs JEPA over metrics-only capacity and unaligned-log controls on
fresh normal lab runs?

The resulting model remains separate from the published metrics-only JEPA. A
passing result supports publication as development evidence; it is not
confirmation evidence or a production-generalization claim.

## Corpus

The target is 30 new, fault-free runs and 10,020 run-isolated next-point
windows:

- ten new request-schedule families;
- each family repeated at one, two, and three worker replicas;
- the first eight complete families, 24 runs and 8,016 windows, are training;
- the last two complete families, six runs and 2,004 windows, are validation;
- 340 points per run, a six-point lookback, and a 0.1-second sample period; and
- case IDs under `multimodal-normal-*`, with seed label 47.

The ten canonical request schedules are:

1. `6, 4, 5, 7, 3`
2. `9, 5, 8, 6, 7, 10`
3. `5, 10, 8, 12, 9, 7`
4. `13, 11, 8, 12, 10, 14, 9`
5. `5, 9, 6, 4, 8, 7`
6. `11, 7, 5, 10, 8, 6, 9`
7. `8, 14, 9, 10, 12, 7`
8. `7, 13, 16, 10, 12, 14, 11`
9. `10, 6, 12, 8, 11, 7, 9`
10. `7, 15, 12, 18, 10, 14, 13, 16`

These schedules and case IDs are disjoint from the earlier 30-run metrics-only
corpus. Families 9 and 10 remain validation-only.

Every included run must satisfy the existing capture and manifest identity,
image provenance, completeness, channel-alignment, safe-log-vocabulary, and
run-boundary checks. There is no post-result run-quality exclusion rule:
failure of an invariant fails the corpus rather than silently removing a run.

## Models and controls

All models use 300 epochs, learning rate 0.02, EMA decay 0.98, weight decay
0.0001, calibration quantile 0.98, and seed 47.

Four distinct artifacts are trained:

- `model.json`: aligned multimodal JEPA, with a three-dimensional metric latent
  and two-dimensional application-log latent;
- `metrics-only-model.json`: the existing three-dimensional metrics-only
  baseline retained for continuity;
- `capacity-matched-metrics-only-model.json`: a five-dimensional metrics-only
  JEPA, matching the fused model's total latent width; and
- `shuffled-log-model.json`: the same fused architecture trained after a
  deterministic permutation breaks metric/log alignment while preserving each
  log context-target pair.

The training and validation log permutations use seeds 1048 and 2048,
respectively. Validation is never used for fitting, preprocessing, threshold
calibration, or hyperparameter choice.

## Promotion gates

The aligned multimodal model is eligible for publication only if all four
preregistered validation gates pass:

- alert rate is at most 10%;
- alert rate is no worse than the five-dimensional capacity-matched
  metrics-only baseline;
- alert rate is no worse than the shuffled-log ablation; and
- mean squared latent loss is no worse than the shuffled-log ablation.

Latent loss is not compared with either metrics-only model because those
models learn a different latent target. A failed gate is reported as a result;
parameters, schedules, split membership, or thresholds are not revised for
this corpus.

## Publication boundary

If every gate passes, the application may publish a new multimodal model,
corpus provenance, development metrics, and model-specific inference path. It
must not replace, rename, or silently route through the metrics-only model.
Raw captures and unrestricted application-log content are not published.
