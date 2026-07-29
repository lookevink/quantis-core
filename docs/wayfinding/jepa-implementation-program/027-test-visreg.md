---
status: active
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

Active. The paper/current-code boundary, clean-room equations, paired batch,
views, candidate/no-detach falsifier, retained controls, explicit direction
RNG, optimizer, collapse curve, operational gates, and artifact contract are
frozen before implementation or fitting.
