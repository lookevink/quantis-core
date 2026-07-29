---
status: active
label: wayfinder:ticket
title: Test Discrete-JEPA semantic tokenization
---

# Test Discrete-JEPA semantic tokenization

## Objective

Implement, independently assess, and retain the
[Discrete-JEPA telemetry tracer v1](../../specs/discrete-jepa-telemetry-tracer-v1.md).

## Dependency

Exact LeNEPA is complete and rejected under ticket 024. Discrete-JEPA starts
from a fresh masked same-history tokenizer and does not reuse LeNEPA's causal
backbone, projector, or SIGReg objective.

## Result

Active. Primary-source ambiguities, telemetry adaptations, cells, controls,
gates, seeds, roles, and artifact contract are frozen before implementation
or fitting.
