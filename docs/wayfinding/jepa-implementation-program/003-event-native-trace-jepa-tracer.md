---
status: closed
label: wayfinder:prototype
parent: Evaluate the unimplemented JEPA families
assignee: codex
blocked_by:
  - Test a soft regime-codebook JEPA tracer
---

# Test an event-native trace JEPA tracer

## Question

Do trace-linked template, outcome, entity, and inter-event-time tokens provide
incremental topology-transfer alert or investigation value when trained with
masked event and time-to-next-event objectives, compared with metrics-only,
binned-event, alignment-shuffled, and simple event n-gram controls?

## Resolution comment

Resolved on 2026-07-28 by the causally compiled tracer preregistered in
[`Event-native trace JEPA prototype v1`](../../specs/event-native-trace-jepa-prototype-v1.md).

The trace encoder was non-collapsed (effective rank 27.53), reached perfect
masked categorical accuracy, and fit within 86,848 parameters. It nevertheless
detected 0% of held-out-topology treatments at 0% control false alarms. Its
80% action-and-target investigation hit@1 was exactly matched by the
alignment-shuffled null. The recipe is rejected without durable
implementation or multi-seed work.

The durable measurements and clock-domain correction are recorded in
[`Event-native trace JEPA prototype v1 result`](../../research/event-native-trace-jepa-prototype-v1-results.md).
The next frontier must test whether explicitly multimodal futures add value;
it first needs a proper-scoring and calibration contract that separates
uncertainty quality from added capacity.
