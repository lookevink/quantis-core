---
status: closed
label: wayfinder:prototype
parent: Evaluate the unimplemented JEPA families
assignee: codex
blocked_by:
  - Test a multi-hypothesis trajectory JEPA tracer
---

# Test an exact SIGReg LeJEPA tracer

## Question

Does the exact sketched-isotropic-Gaussian regularizer from LeJEPA improve
topology-transfer observable-state retention, prediction, alerting, or
investigation when substituted into the strongest entity-preserving
action-conditioned JEPA, compared with its current variance/covariance
regularizer, a no-regularizer JEPA null, matched PCA, and raw low-rank
prediction without changing inference-time capacity?

## Resolution comment

Resolved on 2026-07-28 by the frozen tracer in
[`Exact SIGReg regularizer-substitution tracer v1`](../../specs/sigreg-lejepa-prototype-v1.md).
The implementation pins the LeJEPA authors' `official-minimal-c293d29`
quadrature and changes only the anti-collapse regularizer, with matched
no-regularizer and variance/covariance controls.

SIGReg increased aggregate latent effective rank from 6.012 for the current
regularizer to 9.967, but its topology-transfer observable-state probe NRMSE
was 0.5869 versus 0.3899 for the current regularizer. Selection assigned the
candidate a correction gain of zero, so prediction reduced exactly to the raw
low-rank baseline. It detected 0% of held-out treatment trajectories at 0%
control false alarms. All safety and restoration gates passed, but no
predictive, investigation, or alert value lane passed.

The durable measurements and bounded claim are recorded in
[`Exact SIGReg regularizer-substitution tracer v1 result`](../../research/sigreg-lejepa-prototype-v1-results.md).
Reject this exact residual-JEPA/SIGReg recipe without multi-seed or sealed
work. The runner and immutable local artifact are retained for reproduction.
