---
status: closed
label: wayfinder:task
parent: Evaluate the unimplemented JEPA families
assignee: codex
blocked_by: []
---

# Freeze the shared JEPA evaluation contract

## Question

Which public model, representation, alert-policy, artifact, and assessment
seams—and which common baselines and promotion gates—must every JEPA tracer use
so results remain comparable without forcing unlike candidates into an
implementation-specific interface?

## Resolution comment

Resolved on 2026-07-28 after the user approved the six public seams.

The shared contract is recorded in
[`JEPA implementation ladder v1`](../../specs/jepa-experiment-ladder-v1.md).
Every tracer now uses fitting, entity-preserving encoding, observable-state
predictive distribution, serialization/restoration, pure assessment, and
immutable artifact seams. It shares the topology-transfer data boundary,
baseline categories, common metrics, lane-specific value gates, TDD boundary,
and promotion boundary defined there.

The contract deliberately permits predictive-core, alert-policy, and
investigation candidates to earn value through different lanes while requiring
every JEPA-specific claim to beat a matched null.
