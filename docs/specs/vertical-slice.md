# Quantis evidence-producing vertical slice

## Status

Accepted for implementation on 2026-07-26 based on the architecture agreed in
the preceding conversation.

## Claim under test

A detector that predicts a compressed representation of multivariate telemetry
can be less sensitive to isolated feature noise than featurewise and persistence
baselines while still detecting correlated structural drift not present in
training.

This slice evaluates both a linear latent-loss detector and a coherence-aware
latent predictive detector. The latter scores a consensus-ranked feature
residual decoded from the predicted latent state. The evaluation config uses
consensus rank three, explicitly requiring three signals to disagree. Both
detectors are precursors to, not evidence for, a learned JEPA encoder.
Feature evidence and the consensus score are divided by per-feature median
prediction residuals fitted on training data, preventing naturally volatile
signals from dominating attribution. Attribution averages signed residuals over
the preceding context window before taking magnitude, so transient errors cancel
instead of accumulating as false evidence.

## Confirmed public seams

### Scenario Engine

`generate_scenario(spec) -> Scenario`

Given an explicit seed and scenario specification, it returns telemetry,
point-level phase labels, affected-feature ground truth, and a serializable
manifest. Generation must be deterministic and must not use global random state.

### Window Compiler

`WindowCompiler.fit(telemetry) -> WindowCompiler`

`WindowCompiler.transform(telemetry) -> ModelWindows`

The compiler owns robust normalization and temporal alignment. Its fitted state is
serializable so preprocessing can be reproduced at inference time.

### Detector

`detector.fit(normal_windows) -> detector`

`detector.score(windows) -> DetectionScores`

Every detector returns a scalar anomaly score and per-feature evidence. Threshold
calibration uses training data only.

### Evaluation

`run_evaluation(config) -> EvaluationReport`

The report compares detectors over identical held-out scenarios and includes
noise false-positive rate, structural event recall, detection delay, attribution
hit rate, runtime, and acceptance-gate results.

## Experimental protocol

- Training data contains normal operation and labelled routine isolated noise.
- Training data contains no structural anomalies.
- Test scenarios use seeds not present in training.
- Structural faults change at least three related features over a sustained
  interval.
- All detectors see the same compiled inputs.
- Calibration uses only training scores.
- Scenario manifests and fitted detector artifacts are retained with the report.

## Acceptance gates

The latent detector must:

1. Detect at least 80% of held-out structural events.
2. Alert on no more than 10% of routine-noise points.
3. Reach at least 80% attribution hit@3 for detected structural events.
4. Produce fewer routine-noise alerts than the persistence baseline.
5. Complete one-window-at-a-time held-out scoring at an average below 1 ms per
   point on the evaluation machine.

Passing these gates supports only the synthetic claim above. It does not establish
real-world zero-day detection, causal attribution, or production performance.

Attribution hit@3 is binary per detected event: at least one injected affected
feature appears in the top three buffered evidence features. The report also
includes the stricter affected-feature recall@3 over all events, assigning zero
to missed events. Structural faults affect three of twelve available features;
the report records the analytical random-ranking hit@3 rate and requires the
detector to exceed it.
