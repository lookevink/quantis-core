# SALT-JEPA telemetry tracer v2 results

## Decision

**Reject this SALT-JEPA telemetry recipe. Do not run seed robustness or
sealed confirmation.**

The evidence-review-corrected run confirms the v1 numerical disposition.
Aligned targets became materially easier on selection, but the transfer
advantage remained below the frozen 10% mechanism threshold. More
importantly, the deployed student still substantially regressed the raw
action-conditioned dynamics core.

This rejects the exact telemetry translation on the fixed Quantis lab stack.
It does not reject SALT on video or static-teacher representation learning in
general.

## Reproducible evidence

- implementation commit:
  `14048f53be77db35c314044926818cd7721932f5`;
- conclusion-bearing immutable artifact:
  `artifacts/action-dynamics/prototype-salt-jepa-v2`;
- preserved superseded artifact:
  `artifacts/action-dynamics/prototype-salt-jepa-v1`;
- artifact-manifest SHA-256:
  `76fed1e39208e50f5b4ecfa4ca2dc5b7f5d052e8e9e9975c2b8951d2d9a8562b`;
- source recipe: Li et al.,
  [*Rethinking JEPA: Compute-Efficient Video SSL with Frozen Teachers*](https://arxiv.org/abs/2509.24317);
- 320 reconstructive-teacher steps and 1,280 static-target student steps per
  SALT cell; and
- 40 fitting, 10 selection, 10 calibration, 20 IID evaluation, and 10
  held-topology evaluation pairs.

The 278 MiB artifact retains both complete teacher/student cells, the
reconstructive teacher, PCA and raw controls, deployable student/probe JSON,
raw latency samples, connected-block mask provenance, all original/restored
public outputs and diagnostics, copied reproduction sources, and the
independently recomputed assessment.

## Held-topology result

| representation | overall MSE | action-overlap MSE | downstream-effect MSE |
|---|---:|---:|---:|
| raw rank-32 | 0.105744 | 0.859940 | 0.143833 |
| SALT-JEPA student | 0.142895 | 1.398272 | 0.274727 |
| deranged-target SALT | 0.145124 | 1.397310 | 0.277695 |
| reconstructive teacher | 0.144763 | 1.398071 | 0.278517 |
| matched PCA | 0.147181 | 1.444091 | 0.274703 |

Relative to raw, the SALT student increased downstream-effect error by
`1.91×`. It improved effect MSE by 1.36% over its teacher and 1.07% over the
deranged cell, but was fractionally worse than PCA. It beat the teacher on
seven of ten transfer pairs, passing only that one value gate.

Every representation lacked a raw-safe ridge on selection; the assessor
recomputed that status and the ridge fallback independently. SALT also failed
selection-best and per-entity state-retention gates.

## Mechanism result

| role | SALT-JEPA | deranged-target SALT | aligned improvement |
|---|---:|---:|---:|
| selection | 0.068188 | 0.077518 | 12.04% |
| transfer | 0.103066 | 0.113124 | 8.89% |

The corrected connected mask makes the identity-specific signal clearer than
v1, but the transfer result still misses 10%. The signal therefore remains
role-fragile and does not explain useful held-topology forecasting.

## Evidence and edge behavior

All eleven protocol checks recomputed true:

- every mask was reconstructed from connected rectangles and
  adjacency-preserving fill provenance;
- role identifiers remained disjoint;
- serialized trainable capacity matched exactly at 83,760 inference and
  206,783 total training parameters per SALT cell;
- both teacher hashes remained unchanged;
- student, teacher, probe, target, mask, loss, and PCA restoration error was
  exactly zero;
- the copied public inference signature was causal;
- the actual deployable bundle was 11,930,568 bytes, below 16 MiB; and
- 100 retained latency samples yielded 36.34 ms median and 40.31 ms p95 local
  batch-one CPU latency.

The artifact manifest has no identity mismatches and pure reassessment matches
the stored assessment exactly. Predictive raw-safety gates remain false; those
are scientific failures, not evidence-contract failures.

## Consequence

SALT is closed on this stack. Proceed to exact LeNEPA with a disposable
prediction projection and temporal SIGReg. Do not carry SALT's teacher,
decoder, or mask schedule into that experiment.
