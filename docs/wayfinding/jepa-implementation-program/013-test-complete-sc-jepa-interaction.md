---
status: completed
label: wayfinder:prototype
parent: Evaluate the unimplemented JEPA families
assignee: codex
blocked_by:
  - Test a horizon-conditioned event-predictive JEPA alert tracer
---

# Test the complete SC-JEPA codebook × multi-resolution interaction

## Question

Does the complete SC-JEPA mechanism—soft regime quantization trained jointly
with distinct fine- and coarse-resolution future-prediction objectives—improve
held-topology alert or predictive value over each isolated component and the
raw low-rank reference?

Use a four-cell capacity-matched factorial:

1. continuous, single-resolution;
2. continuous, multi-resolution;
3. codebook, single-resolution; and
4. codebook, multi-resolution (the complete interaction).

The treatment advances only if the interaction is positive rather than merely
recovering a main effect already tested by the earlier codebook or
multi-resolution tracers. Freeze the exact target construction, capacity
matching, selection rule, and minimum interaction gate before fitting.

## Result

Rejected. The complete cell produced a negligible positive Brier interaction
(`0.0000174` versus the required `0.001468`), zero detection interaction, and
no treatment detections. It also lost to the raw low-rank reference on Brier,
failed per-entity code-use diversity, and had 2.73 times the matched-PCA state
NRMSE. Capacity, restoration, protocol, and edge gates passed.

The implementation remains at commit
`5b6db454619aa1d6555a4cc457535c0bad86a446`; the immutable artifact is
`artifacts/action-dynamics/prototype-sc-jepa-interaction-v1`. See
[`sc-jepa-interaction-v1-results.md`](../../research/sc-jepa-interaction-v1-results.md)
for the complete interpretation.
