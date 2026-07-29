# Test SD-JEPA progression/content telemetry

- Status: complete
- Depends on:
  - [Audit remaining JEPA frontier](011-audit-remaining-jepa-frontier.md)
  - [Test CF-JEPA alert tracer](014-test-cf-jepa-alert-tracer.md)

## Question

Does SD-JEPA's fixed progression/content decomposition produce a distinct,
edge-runnable semantic-event localization signal in telemetry?

## Frozen contract

Implement and execute
[SD-JEPA alert tracer v1](../../specs/sd-jepa-alert-tracer-v1.md) against the
pinned primary-source mechanism documented in
[SD-JEPA primary-source notes](../../research/sd-jepa-primary-source-notes.md).

Keep canonical A2, LeWorldModel A0, and A2-full capacity matched. Preserve all
code, selected checkpoints, smoke bundles, failures, and final evidence
regardless of outcome.

## Completion

- [x] Unit tests cover the exact split, triplet, restoration, public causal
  seam, and stored-array assessor.
- [x] A non-interpretable smoke run completes.
- [x] Implementation is reviewed and committed before the official run.
- [x] The frozen official run is independently assessed.
- [x] Result and next-target decision are recorded without deleting code or
  evidence.

## Result

Canonical A2 improved pooled angular current-event AUROC to `0.751656` from
the A0 angle's `0.565924`, but missed the required margin over its own z-MSE
by `0.003885`. More decisively, its progression `R2=0.077549` was below both
A0's first two coordinates (`0.081910`) and its own content (`0.205141`).
The calibrated transfer policy emitted no alerts. All safety and edge gates
passed. Reject the recipe and preserve the implementation and artifacts.

See [the complete result](../../research/sd-jepa-alert-v1-results.md).
