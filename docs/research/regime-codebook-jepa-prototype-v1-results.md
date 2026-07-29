# Soft regime-codebook JEPA prototype v1 result

## Decision

**Reject this recipe. Do not absorb the prototype into durable model code or
run multi-seed robustness.**

The soft codebook solved the narrow collapse symptom but did not preserve the
observable effect magnitude needed by the predictive core. It passed the
preregistered code-usage, state-probe, restoration, attribution, and action
sanity checks. It failed both raw low-rank predictive safety gates and every
alert-policy false-alarm gate.

The corrected evidence is:

`artifacts/action-dynamics/prototype-regime-codebook-jepa-v1/prototype-result.json`

This is non-production open-development evidence, not sealed confirmation. The
exact runner is retained so the negative result remains reproducible.

## Evidence correction

The first completed training run was invalidated before interpretation because
the prototype inferred entity-feature ownership from normalized magnitude.
Zero-padded slots inherit nonzero normalized constants and tiny floating-point
spread, so that implementation incorrectly marked all 189 entity-feature slots
as observed.

The invalid artifact is retained at:

`artifacts/action-dynamics/prototype-regime-codebook-jepa-v1-invalid-ownership-mask`

The corrected run derives the 27 metric owners from the declared graph and
admits structured-event slots only when their training variation exceeds the
frozen numerical-noise threshold. It identifies 37 observed slots and leaves
the terminal PostgreSQL node unobserved. No model configuration, role,
baseline, threshold, or gate changed.

## Data and runtime

- Fitting: 40 matched pairs, 6,320 windows, worker topologies one and two.
- Selection: 10 matched pairs, 1,580 windows.
- Calibration: 10 matched pairs, 1,580 windows.
- In-distribution evaluation: 20 matched pairs, 3,160 windows.
- Primary topology transfer: 10 topology-three pairs, 1,580 windows.
- Neural seed: 127, deterministic CPU, 40 epochs.
- Continuous-null training: 9.75 seconds.
- Codebook training: 28.03 seconds.
- Switching-regime fitting: 0.21 seconds.
- Codebook inference parameters: 558,751.
- Local codebook batch-one CPU latency: 0.233 ms.

The timing is a local Python/PyTorch microbenchmark, not a target-device claim.

## Topology-transfer prediction

| Model | Action MSE | Overall MSE | Downstream-effect MSE | Action + target hit@1 | No-action specificity |
|---|---:|---:|---:|---:|---:|
| Raw rank-32 low-rank | **0.5512** | **0.1057** | **0.06627** | 100% | 100% |
| Switching-regime ridge | 0.7592 | 0.1712 | 0.06981 | 100% | 100% |
| Continuous JEPA null | 0.6948 | 0.3196 | 0.13930 | 90% | 100% |
| Soft regime-codebook JEPA | 1.4135 | 0.2486 | 0.16390 | 100% | 100% |

Relative to raw low-rank, the codebook produced:

- `2.56x` action-overlap MSE;
- `2.35x` overall MSE; and
- `2.47x` downstream-effect MSE.

Correct action beat both no-action and whole-pair-deranged action on all ten
held-out treatment pairs for every model. The codebook therefore learned that
actions matter, but its decoded magnitude and downstream effects were not
dependable.

## Representation result

The codebook did not collapse:

- 27 of 32 codes exceeded 0.5% marginal usage;
- marginal perplexity was `23.58`;
- every entity used multiple codes;
- mean assignment entropy was `2.895`; and
- mean maximum assignment probability was `0.165`.

Topology-transfer current-state frozen-probe NRMSE was:

| Representation | Aggregate NRMSE |
|---|---:|
| Soft regime codebook | **0.5655** |
| Continuous JEPA null | 0.7038 |
| Matched width-16 PCA | 0.9185 |

The codebook improved state probing and restored hit@1 from 90% for the
continuous null to 100%. It did not improve on the already-perfect raw
low-rank or classical switching controls, and it did so while making observable
forecasting materially worse.

The assignments also remained very soft. A mean maximum probability of 0.165
is evidence of distributed prototype use, not a crisp operational regime
partition. Increasing sharpness after seeing this result would be a new
experiment and cannot rescue the frozen recipe.

## Alert-policy result

Sequential hidden-action detection was:

| Model | Control false alarms | Treatment detection | Median delay |
|---|---:|---:|---:|
| Raw low-rank | 20% | 90% | 20.0 |
| Switching-regime ridge | 10% | 90% | 15.0 |
| Continuous JEPA null | 100% | 100% | 6.5 |
| Soft regime-codebook JEPA | 100% | 100% | 8.5 |

The codebook's apparent sensitivity is operationally unusable because every
held-out control trajectory alarmed. It failed the 5% false-alarm gate and did
not improve delay over the continuous null.

## Gate outcome

Passed:

- active-code count and marginal perplexity;
- multiple codes for every entity;
- frozen-probe non-inferiority to the continuous null;
- deterministic finite restoration;
- 100% action-and-target hit@1;
- 100% no-action specificity; and
- 100% correct-action sanity.

Failed:

- action and overall MSE within 5% of raw low-rank;
- 10% downstream-effect improvement;
- control-trajectory false alarms at most 5%; and
- any complete predictive or alert value lane.

The investigation lane beat the continuous JEPA null, but the same action
metrics were already perfect for both raw low-rank and switching-regime
controls. It therefore does not establish incremental investigation value.

## Interpretation

The experiment separates collapse from usefulness:

1. A balanced soft codebook can sustain broad prototype usage.
2. It can preserve current observable state better than the tested continuous
   latent and PCA controls.
3. Those properties do not imply accurate future effect magnitude, calibrated
   anomaly scores, or value over a strong raw-state model.
4. The 0.21-second classical switching-regime fit retained perfect attribution
   and stayed close to raw low-rank downstream effects without JEPA training.

Exact LeJEPA/SIGReg regularization is therefore no longer the highest-priority
next test: this run achieved a noncollapsed representation and still failed.
The next materially different JEPA tracer should change the information and
prediction task by using trace-linked events natively—template/outcome and
time-to-next-event targets—rather than adding another anti-collapse loss to
the same binned state.
