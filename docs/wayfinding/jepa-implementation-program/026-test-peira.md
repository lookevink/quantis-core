---
status: active
label: wayfinder:ticket
title: Test PEIRA inter-view regressor alignment
---

# Test PEIRA inter-view regressor alignment

## Objective

Implement, independently assess, and retain the
[PEIRA telemetry tracer v1](../../specs/peira-telemetry-tracer-v1.md).

## Dependency

Discrete-JEPA is complete and rejected under ticket 025. PEIRA starts from
the retained complete-LeJEPA paired semantic views but replaces invariance
plus SIGReg with the trace-of-optimal-regularized-regressor objective. It
does not reuse Discrete-JEPA's hard codebook.

## Result

Active. The paper/code boundary, paired views, aligned and deranged cells,
prior controls, exact moment updates, auxiliary gradient, optimization,
roles, diagnostics, gates, and artifact contract are frozen before
implementation or fitting.
