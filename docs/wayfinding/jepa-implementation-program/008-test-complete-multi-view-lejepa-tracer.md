---
status: closed
label: wayfinder:prototype
parent: Evaluate the unimplemented JEPA families
assignee: codex
blocked_by:
  - Freeze the complete multi-view LeJEPA telemetry contract
---

# Test the complete multi-view LeJEPA telemetry tracer

## Question

Does the frozen complete multi-view LeJEPA recipe learn a restorable,
entity-preserving telemetry representation whose fit-only action-conditioned
probe preserves raw low-rank safety and improves held-out-topology downstream
effects over invariance-only, SIGReg-only, masked-autoencoder, and matched-PCA
controls?

## Resolution comment

Resolved on 2026-07-28 under the frozen
[`Complete multi-view LeJEPA telemetry contract v1`](../../specs/complete-lejepa-telemetry-contract-v1.md).
The implementation used one anchor from each of 40 matched pairs per step,
eight aligned semantic views, the exact `0.05 SIGReg + 0.95 invariance`
objective, 1,024 fresh sketches, four matched representation controls, and a
fit-only rank-32 action probe.

Complete LeJEPA transfer downstream-effect MSE was `0.274014`, versus
`0.269572` for the best masked-autoencoder control and `0.143833` for the raw
rank-32 reference. It was not best on selection, won only 50% of held-out
pairs, and exceeded all three raw predictive safety bounds. Aggregate
state-probe NRMSE was strong at `0.015221`, but the candidate failed the
per-entity PCA bound. No representation had a ridge candidate satisfying all
raw-selection safety constraints.

The stored-array assessor independently reproduced
`reject_exact_complete_multi_view_lejepa_recipe`. The durable measurements and
bounded claim are recorded in
[`Complete multi-view LeJEPA telemetry tracer v1 result`](../../research/complete-lejepa-telemetry-prototype-v1-results.md).
Do not advance this recipe to multi-seed robustness, sealed confirmation, or
a two-stage action model. The runner, assessor, tests, and immutable local
artifact remain retained.
