---
status: closed
label: wayfinder:prototype
parent: Evaluate the unimplemented JEPA families
assignee: codex
blocked_by:
  - Audit the remaining JEPA frontier and select the next tracer
---

# Test a horizon-conditioned event-predictive JEPA alert tracer

## Question

Does a clean-room HEPA telemetry implementation—future-interval JEPA
pretraining followed by frozen-encoder, horizon-conditioned survival
finetuning—produce a finite, calibrated, monotone event-time distribution
that improves held-out-topology treatment detection over a
whole-trajectory horizon-deranged JEPA null by at least 10 percentage points
at no more than 5% control-trajectory false alarms, while retaining
entity-local state, restoring exactly, and satisfying the frozen edge budget?

The exact role, treatment, data boundary, null, controls, and ten gates are
frozen in
[`JEPA frontier technique audit, July 2026`](../../research/jepa-frontier-technique-audit-2026.md#first-tracer-contract-hepa-telemetry).

## Outcome

Rejected. The valid HEPA branch was finite, restorable, state-retentive, and
edge-feasible. It improved held-topology Brier over the horizon-deranged null
but detected only 50% of treatments, exactly matching the null and trailing
the 60% supervised/raw controls. It failed the 80% sensitivity and ten-point
JEPA-specific improvement gates.

The complete result and evidence-correction record are in
[`HEPA telemetry tracer v1 result`](../../research/hepa-jepa-telemetry-tracer-v1-results.md).
The valid artifact and the invalid calibration-topology run are both retained.
