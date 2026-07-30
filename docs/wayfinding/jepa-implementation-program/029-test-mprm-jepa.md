# 029 — Test mean-preserving residual mixture JEPA

## Status

Complete; exact recipe rejected by the primary and independent assessors.

See the
[result](../../research/mean-preserving-residual-mixture-jepa-v1-results.md).

## Objective

Test whether four JEPA residual forecast hypotheses improve proper trajectory
score while their weighted predictive mean remains exactly equal to the
frozen raw rank-32 predictive core.

## Required order

- Fit and freeze the candidate and all controls on qualified v1 fit evidence.
- Restore models and verify prediction parity.
- Bind the model-freeze identity into a new 90-pair selection campaign.
- Collect once with zero retry or post-hoc attrition.
- Qualify all provenance, telemetry, action-effect, and recovery gates.
- Score only the qualified complete corpus.
- Reproduce the binary decision with the independent assessor.

## Stop conditions

Any operational or qualification failure stops the attempt without a model
claim. Any selection-gate failure rejects the exact recipe. A pass authorizes
only a separate calibration/evaluation design.
