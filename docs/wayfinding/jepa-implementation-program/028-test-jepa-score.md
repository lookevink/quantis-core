---
status: in-progress
label: wayfinder:ticket
title: Test exact JEPA-SCORE edge alerting
---

# Test exact JEPA-SCORE edge alerting

## Objective

Implement, independently assess, and retain the
[JEPA-SCORE edge-feasibility screen v1](../../specs/jepa-score-edge-screen-v1.md).

## Dependency

VISReg is complete and rejected under ticket 027. JEPA-SCORE does not train
another representation. It applies the source paper's exact Jacobian
singular-value score to the frozen complete-LeJEPA projector and compares its
alert policy with a fit-only raw telemetry delta score.

## Exit

Conclude with an immutable artifact and result record. Promote only if exact
edge runtime, role-clean safety, and held-topology alert value all pass. Keep
all code and evidence after either outcome.

