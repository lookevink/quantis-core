---
status: completed
label: wayfinder:ticket
title: Test exact LeNEPA disposable projection
---

# Test exact LeNEPA disposable projection

## Objective

Implement, independently assess, and retain the
[LeNEPA telemetry tracer v1](../../specs/lenepa-telemetry-tracer-v1.md).

## Dependency

SALT-JEPA is complete and rejected under ticket 023. LeNEPA starts from a
fresh causal no-augmentation encoder and does not reuse SALT's teacher,
decoder, or mask schedule.

## Result

Rejected. All twelve protocol checks passed, but the projected candidate
failed raw selection/transfer safety, failed the projected mechanism gate,
retained `1.92×` raw transfer effect error, and beat the unprojected control
on only two of ten transfer pairs.

See the
[retained result](../../research/lenepa-telemetry-v1-results.md) and immutable
`artifacts/action-dynamics/prototype-lenepa-jepa-v1` artifact. Proceed to
Discrete-JEPA.
