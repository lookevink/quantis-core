# Contextual metrics + logs JEPA v1

## Status

Implemented as a development experiment on 2026-07-27. It reuses the existing
30-run multimodal corpus for model development only. Families 9 and 10 have
already been inspected, so their results are diagnostic rather than fresh
publication evidence.

The completed preflight selected L1 loss with one log latent dimension for the
next untouched corpus. See the
[development result](../research/contextual-multimodal-jepa-v1-results.md).

This protocol implements the recommendations in the
[JEPA telemetry research note](../research/jepa-telemetry-lessons.md). It does
not alter or replace the separately serialized metrics-only or multimodal-v0
models.

## Claim under development

A contextual JEPA can use bounded structured application logs without learning
absolute request demand as a schedule shortcut when:

- raw application-event counts are expressed relative to observed demand;
- demand and worker topology are explicit exogenous predictor controls;
- target representations cover contiguous future blocks at multiple horizons;
- modality-specific target residuals are calibrated before fusion; and
- model selection is performed inside the original training families.

The experiment does not establish production generalization, fault detection,
or publication-quality improvement.

## Public seams

### Demand-relative application state

`DemandResidualLogTransformer.transform(values, feature_names, request_demand)`

The transform emits:

- checkout completion ratio: `completed / max(accepted, 1)`;
- checkout backlog delta ratio: `(accepted - completed) / demand`;
- checkout rejection rate: `rejected / demand`; and
- application error-event rate: `error_events / demand`.

Only preregistered structured event counts enter this transform. Raw message
text, identifiers, payloads, stack traces, and arbitrary attributes remain
excluded.

### Contextual corpus

`compile_contextual_multimodal_telemetry_corpus(base, runs, horizons, target_block_size)`

The compiler reconstructs each run separately from the validated multimodal
corpus, derives observable request demand and worker replicas, fits log and
control normalization on training runs only, and emits:

- six-point metric and log contexts;
- two-point contiguous target blocks beginning at horizons 1, 3, and 6; and
- demand and topology controls aligned to every target block.

Neither context nor targets cross run boundaries.

### Contextual detector

`ContextualMultimodalJepaWorldModelDetector.fit(windows)`

The detector has separate metric and log temporal block encoders. During
representation pretraining, online context encoders receive gradients while
stop-gradient target encoders move only by EMA. A conditioned two-layer
predictor forecasts future joint embeddings with a configured L1, Huber, or
MSE loss. Explicit metric-to-metric, log-to-log, metric-to-log, and
log-to-metric heads train alongside the joint predictor.

The six-point context is encoded as three non-overlapping two-point patches.
This makes one rollout step advance exactly one target block: the oldest
context patch is dropped and the predicted horizon-1 block is appended before
the horizon-3 prediction.

The last training stage freezes both encoders and refines only the causal
predictors. The appended horizon-1 prediction is stop-gradient, so the
horizon-3 rollout trains shared predictor dynamics without updating the first
prediction through the second step.

Scoring uses the horizon-1 block. Metric and log latent energies are calibrated
separately on training data before equal fusion, adding one window of detection
latency.

## Controls and development selection

Every run reports:

- metrics-only;
- capacity-matched metrics-only;
- shuffled demand-relative logs;
- log-only;
- metric-context-only dropout; and
- log-context-only dropout.

Metrics-only, capacity-matched metrics-only, and log-only controls use the same
contextual target blocks, exogenous controls, nonlinear predictor class,
multi-horizon loss, rollout, and frozen-encoder refinement. Capacity matching
sets the metric latent width equal to the fused model's total joint width; it
does not claim exact scalar-parameter equality across modality-specific heads.

When enabled, leave-one-schedule-family-out development runs only over the
original training cases. Metric, log, and control normalization is refitted
inside each fold using only its training families. Each fold includes both EMA
representation pretraining and predictor-only frozen-encoder refinement.
Selection requires conditional mean improvement over metrics-only, no-worse
performance on at least half the folds, and active metric and log latent rank.
Passing development selection is never publication eligibility: a new
untouched corpus remains required.

## JEPA rationale and primary sources

The contextual target blocks follow I-JEPA's finding that useful prediction
targets should be sufficiently large and contextual rather than local
pointwise details ([Assran et al., 2023](https://arxiv.org/abs/2301.08243)).

Continuous temporal blocks, an EMA target encoder, explicit target-position
conditioning, and robust latent prediction follow V-JEPA
([Bardes et al., 2024](https://arxiv.org/abs/2404.08471)).

The frozen-encoder causal stage, exogenous control conditioning, and short
latent rollout are small-data adaptations of V-JEPA 2's action-conditioned
world model
([Assran et al., 2025](https://arxiv.org/abs/2506.09985)).

Separate modality stems and explicit intra- and cross-modal objectives are
motivated by MJEPA's finding that naïve multimodal parameter sharing can
degrade unimodal representations
([Teotia et al., 2026](https://arxiv.org/abs/2606.25225)). That evidence comes
from audio and video at far larger scale, so it is treated as an ablation
motivation rather than proof for telemetry.

The shortcut diagnostics also follow evidence that joint-embedding predictive
systems can prioritize slow or easily predictable distractors
([Sobal et al., 2022](https://arxiv.org/abs/2211.10831)). That study used toy
environments and different JEPA variants, so the connection is diagnostic
analogy only.

## Reproducibility

Run the v1 development experiment against the preserved v2 corpus:

```bash
./lab/fault_matrix/run-contextual-multimodal-jepa-development.sh
```

Artifacts are written to
`artifacts/jepa-world-model-v1/contextual-multimodal-development`.

Reproduce the fixed loss/width preflight and its aggregate selection report:

```bash
./lab/fault_matrix/run-contextual-multimodal-jepa-ablations.sh
```
