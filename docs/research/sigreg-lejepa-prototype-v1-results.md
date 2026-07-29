# Exact SIGReg regularizer-substitution tracer v1 result

## Decision

**Reject this exact residual-JEPA/SIGReg substitution. Do not advance it to
multi-seed robustness or sealed confirmation.**

The candidate passed every safety gate, but selection assigned its learned
residual correction a gain of zero. Its deployed forecast therefore reduced
exactly to the frozen raw rank-32 low-rank baseline and no predictive,
investigation, or alert value lane passed.

This is a negative result for the pinned regularizer substitution, not for
SIGReg or LeJEPA generally. The experiment did not implement the complete
LeJEPA multi-view objective.

The exact runner remains at
`lab/action_dynamics/prototype_sigreg_lejepa.py`, and the local immutable
evidence is under
`artifacts/action-dynamics/prototype-sigreg-lejepa-v1`.

## Evidence boundary

- One deterministic seed over already-open development data.
- Fit and selection used worker topologies one and two.
- The primary diagnostic held out worker topology three.
- The three neural variants used identical inference architecture and 55,030
  parameters.
- SIGReg used the official-minimal preset pinned to LeJEPA commit
  `c293d291ca87cd4fddee9d3fffe4e914c7272052`: 256 sketches, 17 knots over
  `[0, 3]`, weight `0.02`.
- This is neither sealed confirmation nor production-paging evidence.

## Selection and topology-transfer prediction

Selection chose correction gains `1.0`, `1.0`, and `0.0` for the
no-regularizer, variance/covariance, and SIGReg variants respectively.

| Model | Selected gain | Transfer action MSE | Transfer overall MSE | Transfer effect MSE |
|---|---:|---:|---:|---:|
| Raw rank-32 low-rank | — | 0.551154 | 0.105744 | 0.066273 |
| No-regularizer JEPA | 1.0 | 0.551048 | 0.105745 | 0.066327 |
| Variance/covariance JEPA | 1.0 | **0.550614** | **0.105723** | 0.066302 |
| SIGReg JEPA | **0.0** | 0.551154 | 0.105744 | **0.066273** |

The SIGReg row equals the raw baseline because the frozen selector disabled
the learned correction. Its effect MSE is numerically below both learned
controls, but it does not improve raw by the required 10%, and a zero selected
gain fails the predictive value lane.

All models retained 100% action-and-target hit@1 and 100% no-action
specificity. Correct action beat both no-action and whole-pair shuffled-action
ablations on every held-out treatment pair.

## Representation result

SIGReg produced the broadest latent spectrum but the least useful neural
observable-state representation:

| Representation | Effective rank | Minimum observed-entity rank | State-probe NRMSE |
|---|---:|---:|---:|
| No-regularizer JEPA | 4.417 | 1.541 | 0.4786 |
| Variance/covariance JEPA | 6.012 | 1.470 | **0.3899** |
| SIGReg JEPA | **9.967** | **6.606** | 0.5869 |
| Matched width-16 PCA | — | — | 0.6025 |

The SIGReg probe was slightly better than matched PCA but 50.5% worse than
the current variance/covariance regularizer, far from the required 5%
improvement. This again separates non-collapse from task-relevant state
retention: a higher effective rank did not yield a better frozen observable
probe or a useful residual forecast.

## Alert result

Topology-transfer trajectory alerting used calibration-control trajectory
maxima at the shared 5% false-alarm budget.

| Model | Control false alarms | Treatment detection | Median delay |
|---|---:|---:|---:|
| No-regularizer JEPA | 0% | 30% | 8 transitions |
| Variance/covariance JEPA | 0% | 0% | — |
| SIGReg JEPA | 0% | 0% | — |

SIGReg met the false-alarm bound but detected no treatment trajectory. It did
not reach 80% sensitivity, had no finite post-onset delay, and did not improve
over both matched JEPA controls.

## Safety, restoration, and edge measurements

Every preregistered safety gate passed:

- every reported numeric measurement was finite;
- residual-model inference parameter counts matched;
- raw-baseline forecast safety, attribution, and action-ablation gates passed;
  and
- every serialized model restored with public-output parity.

The SIGReg artifact contains 55,030 inference parameters and is 1,043,293
bytes. Its selected-gain-zero batch-one CPU path measured 0.149 ms. Training
took 64.88 seconds, versus 46.22 seconds for variance/covariance and 43.07
seconds without a regularizer. These are local Python/PyTorch measurements,
not target-device benchmarks.

## Interpretation

The exact official-minimal regularizer substantially broadened the latent
spectrum. That change did not preserve more observable state than the existing
regularizer, improve calibrated alerts, or earn a nonzero residual correction.
For this corpus and residual architecture, representation isotropy is not the
missing ingredient.

The bounded conclusion is:

> Do not deploy or further tune the official-minimal SIGReg substitution in
> this residual JEPA. Retain the implementation and artifact as a reproducible
> negative result.

A future complete LeJEPA experiment would be a different hypothesis: it would
replace the surrounding objective and view construction, not merely change
the anti-collapse term after seeing this result.

## Artifact identity

- Manifest SHA-256:
  `912548d38056ce910394a6b675f65277becd2590b5b65b88029ace0af830385d`
- Result SHA-256:
  `95b8016d07cb0e6cdc50d3d1579ed11307ef9254d91b7906c1710ca6d4395918`
- Runner SHA-256:
  `98ba521bcfea4d13b89dbb70906c10440c952b29320c23f8e251047ee1ec039e`
- Production SIGReg source SHA-256:
  `edacd8c49aa1a0d3ef17111e7b6bc44f8b3011a0c923cd03836977be37594b66`

The artifact manifest was independently rehashed, the pure assessor was
recomputed from the stored measurements, and all implementation-identity
hashes matched.
