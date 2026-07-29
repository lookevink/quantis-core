---
status: closed
label: wayfinder:grilling
parent: Evaluate the unimplemented JEPA families
assignee: codex
blocked_by:
  - Test an exact SIGReg LeJEPA tracer
---

# Freeze the complete multi-view LeJEPA telemetry contract

## Question

Which telemetry view construction, entity-preserving encoder and projector,
exact convex LeJEPA loss, matched controls, frozen downstream adapters, and
promotion gates can test the complete predictor-free multi-view LeJEPA recipe
rather than SIGReg alone, without allowing view augmentations to erase
operational state or leak selection and evaluation roles?

## Resolution comment

Resolved on 2026-07-28 through a live contract review with the user. The
complete decision is recorded in
[`Complete multi-view LeJEPA telemetry contract v1`](../../specs/complete-lejepa-telemetry-contract-v1.md).

The candidate is a representation candidate only. It uses one independent
anchor from each of 40 matched pairs, two semantic-preserving global views,
six rooted local views, an entity-preserving two-block width-64 transformer,
and the pinned complete LeJEPA objective with 1,024 official-package SIGReg
directions and `lambda=0.05`.

The matched controls separate cross-view invariance, SIGReg, masked
reconstruction, PCA, and raw-state prediction. Promotion requires state-probe
safety plus a shared frozen rank-32 action-conditioned representation probe
that remains within raw low-rank safety and improves held-out downstream
effects by at least 5% over every representation control. Loss, rank,
Gaussianity, and view agreement cannot pass by themselves.

The contract freezes 1,600 deterministic pair-blocked optimizer steps, exact
selection boundaries, a pure assessor, immutable evidence, and public test
prerequisites. Passing authorizes multi-seed representation robustness and
design of a separate action-conditioned stage. Failure rejects the exact
recipe. The runner, assessor, tests, specification, and artifacts remain
retained under either outcome.
