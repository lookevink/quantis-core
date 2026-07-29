---
status: completed
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

Completed and rejected. The complete cell collapsed to one code per entity,
tied the P2P-only control on next-code accuracy and all forecast scores,
retained `1.92×` raw transfer effect error, and failed every value gate.
All implementation and conclusion-bearing evidence remains retained under
`artifacts/action-dynamics/prototype-discrete-jepa-v1`. See the
[result record](../../research/discrete-jepa-telemetry-v1-results.md).
