---
status: closed
label: wayfinder:prototype
parent: Evaluate the unimplemented JEPA families
assignee: codex
blocked_by:
  - Freeze the multi-hypothesis JEPA scoring contract
---

# Test a multi-hypothesis trajectory JEPA tracer

## Question

Does a fixed four-component, responsibility-weighted latent trajectory JEPA
improve topology-transfer proper scores, hidden-action alerting, or
closed-library investigation over its one-component JEPA, capacity-matched
single-Gaussian, supervised-mixture, and raw low-rank controls under the
frozen multi-hypothesis scoring contract?

## Resolution comment

Resolved on 2026-07-28 by the corrected tracer preregistered in
[`Multi-hypothesis trajectory JEPA prototype v2 correction`](../../specs/multi-hypothesis-jepa-prototype-v2.md).
The first decision artifact was invalidated during review because evaluation
finiteness could enter selection and its promised assessment was incomplete;
its numeric sidecars were preserved and content-bound into a corrected,
selection-only pure assessment.

The four-component JEPA produced supported alternatives on 24.51% of
action-overlap selection windows, but improved over one-component JEPA by only
0.000868 nats per coordinate, was 0.010676 nats worse than the supervised
mixture, and produced 2.379x raw overall MSE plus 4.387x raw action-overlap
MSE. It failed the safe-null gate, so calibration, alerting, and investigation
were not reached. The frozen safe null is the rank-32 raw low-rank model.

The durable measurements and bounded claim are recorded in
[`Multi-hypothesis trajectory JEPA prototype v2 result`](../../research/multi-hypothesis-jepa-prototype-v2-results.md).
The direct responsibility-weighted four-head recipe is rejected without
multi-seed or sealed work. The public mixture moment and exact proper-score
seam remains as tested infrastructure.
