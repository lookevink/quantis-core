---
status: complete
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

## Result

Rejected. All 13 protocol gates and all eight edge-safety gates passed, with
`51.359 ms` median and `60.195 ms` p95 exact scorer latency. All five value
gates failed: selection pair wins were `0.40`, IID treatment detection was
`0.10`, and transfer treatment detection was `0.00`.

See the
[retained result](../../research/jepa-score-edge-screen-v1-results.md). The
immutable artifact manifest SHA-256 is
`e678101945c3b99cd325e003f23fdbef334c09ef29ef68f89220cc244012ed86`.
