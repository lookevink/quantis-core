---
status: completed
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

Completed and rejected. Aligned PEIRA learned a strong noncollapsed trace
signal and sharply separated from pair derangement, but retained `1.91×` raw
held-topology effect error, lost to the deranged and reconstruction controls,
won only three of ten transfer pairs, and failed raw safety. The corrected
float32 replay check changes no scientific metric or decision. Code, the
673 MiB immutable artifact, and the
[result record](../../research/peira-telemetry-v1-results.md) are retained.

Proceed to VISReg. Do not carry PEIRA moment or projector state forward.
