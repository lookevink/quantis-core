# Test SD-JEPA progression/content telemetry

- Status: active
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
- [ ] Implementation is reviewed and committed before the official run.
- [ ] The frozen official run is independently assessed.
- [ ] Result and next-target decision are recorded without deleting code or
  evidence.
