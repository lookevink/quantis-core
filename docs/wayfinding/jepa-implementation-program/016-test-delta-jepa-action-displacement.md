# Test Delta-JEPA action-sensitive displacement

- Status: complete
- Depends on:
  - [Audit remaining JEPA frontier](011-audit-remaining-jepa-frontier.md)
  - [Test SD-JEPA progression/content telemetry](015-test-sd-jepa-alert-tracer.md)

## Question

Does five-step latent-difference action decoding cure the recurring
action-insensitive bottleneck and improve held-topology downstream effects?

## Frozen contract

Implement and execute the
[Delta-JEPA action-displacement tracer](../../specs/delta-jepa-action-displacement-tracer-v1.md)
from the pinned
[primary-source notes](../../research/delta-jepa-primary-source-notes.md).

Keep LDAD, endpoint-concat, and prediction-only cells parameter identical.
Preserve code and all evidence regardless of outcome.

## Completion

- [x] Unit tests cover displacement-only input, capacity, restoration,
  pair-blocking, public causality, and pure assessment.
- [x] A non-interpretable smoke run completes.
- [x] Implementation is reviewed and committed before the official run.
- [x] The official run and fresh-process assessment complete.
- [x] The result and next geometry-screen decision are recorded.

## Result

The displacement predicted observable five-step state change very accurately
(`NRMSE=0.052359`, `Pearson=0.998203`) and retained current state, but it did
not isolate action better than endpoints. Treatment action reconstruction was
`0.028554`, versus `0.015846` for endpoint concatenation, and sequence
retrieval was `15.43%`, versus `43.09%`. Transfer downstream-effect MSE was
`0.274364`, essentially tied with both neural controls and `90.75%` worse than
raw. Reject this recipe and retain its code, smoke bundle, and official
artifact.

See
[the complete result](../../research/delta-jepa-action-displacement-v1-results.md).
