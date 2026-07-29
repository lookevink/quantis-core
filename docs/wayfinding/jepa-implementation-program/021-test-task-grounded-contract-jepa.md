---
status: complete
label: wayfinder:ticket
title: Test task-grounded Contract-JEPA
---

# Test task-grounded Contract-JEPA

## Destination

Implement and conclude the
[task-grounded Contract-JEPA tracer](../../specs/task-grounded-contract-jepa-tracer-v1.md).

## Evidence required

- An immutable raw action-conditioned path and bounded correction.
- Equal-capacity JEPA, supervised, and ungrounded cells.
- Joint current-state, paired-effect, and effect-score witnesses.
- Selection-only safe gain and calibration-only witness threshold.
- Independent stored-evidence assessment and retained reproduction code.

## Result

Rejected. All safety gates passed and the bounded residual improved raw
transfer downstream-effect MSE by 1.05%, but the supervised and ungrounded
controls were both better and the required improvement was 10%. The witness
alert detected every treatment with zero delay but also alarmed on every
transfer control. See the
[result](../../research/task-grounded-contract-jepa-v1-results.md).
