---
status: completed
label: wayfinder:prototype
parent: Evaluate the unimplemented JEPA families
assignee: codex
blocked_by:
  - Test the complete SC-JEPA codebook × multi-resolution interaction
---

# Test CF-JEPA mask-free multi-horizon alerting

## Question

Does the complete CF-JEPA three-zone forward-crop objective produce a
smoother EMA representation with held-topology alert value beyond its online
representation, the authors’ one-zone and masked-latent objectives, and
matched raw PCA?

Freeze and run the source-faithful comparison in
[`cf-jepa-alert-tracer-v1.md`](../../specs/cf-jepa-alert-tracer-v1.md).
Preserve the implementation and every artifact regardless of outcome.

## Result

Rejected. The source mechanism appeared: the EMA target was much smoother
and lower rank than the online encoder, improved Brier over the online route
and PCA, and achieved `0.0437` state NRMSE. But the simpler one-zone target
had slightly better Brier, every neural EMA route had 10% transfer control
false alarms, and all routes tied at 90% treatment detection with
one-transition median delay.

The implementation is frozen at
`3b7cb81a277b5d7c48a6946735c9e5e0012bcc54`; the immutable artifact is
`artifacts/action-dynamics/prototype-cf-jepa-alert-v1`. See
[`cf-jepa-alert-v1-results.md`](../../research/cf-jepa-alert-v1-results.md)
for the complete interpretation.
