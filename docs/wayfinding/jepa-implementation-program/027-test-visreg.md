---
status: completed
label: wayfinder:ticket
title: Test VISReg scale-shape regularization
---

# Test VISReg scale-shape regularization

## Objective

Implement, independently assess, and retain the
[VISReg telemetry tracer v1](../../specs/visreg-telemetry-tracer-v1.md).

## Dependency

PEIRA is complete and rejected under ticket 026. VISReg returns to the
complete multi-view LeJEPA stack and changes only the explicit
representation regularizer. It does not reuse PEIRA's moments, trace
objective, projector state, or learned representation.

## Result

Completed and rejected. The frozen radial-collapse diagnostic confirmed both
claimed gradient inequalities, but the detached candidate still collapsed to
projector rank `1.12`, reached scale loss approximately `1.0`, retained
`1.97×` raw held-topology effect error, lost to no-detach and reconstruction,
and won five of ten transfer pairs. All protocol checks pass. Code, the
1.2 GiB immutable artifact, and the
[result record](../../research/visreg-telemetry-v1-results.md) are retained.

Proceed to the bounded JEPA-SCORE edge-feasibility screen. Do not carry the
VISReg projector or regularizer forward.
