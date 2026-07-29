---
status: closed
label: wayfinder:prototype
parent: Evaluate the unimplemented JEPA families
assignee: codex
blocked_by:
  - Freeze the shared JEPA evaluation contract
---

# Test a soft regime-codebook JEPA tracer

## Question

Does a fine/coarse, soft-prototype JEPA with balanced code usage and an
observable-state anchor improve topology-transfer prediction or investigation
utility over continuous JEPA, frozen PCA, switching-regime, and raw low-rank
baselines without subsystem-level collapse?

## Resolution comment

Resolved on 2026-07-28 by the retained non-production tracer preregistered in
[`Soft regime-codebook JEPA prototype v1`](../../specs/regime-codebook-jepa-prototype-v1.md).

The corrected codebook used 27 of 32 codes, reached marginal perplexity 23.58,
and improved frozen state-probe NRMSE to 0.5655 from 0.7038 for the continuous
null. It nevertheless produced 2.56x raw low-rank action MSE, 2.47x downstream
effect MSE, and 100% control-trajectory sequential false alarms. The recipe is
rejected without multi-seed or durable implementation.

The durable interpretation is
[`Soft regime-codebook JEPA prototype v1 result`](../../research/regime-codebook-jepa-prototype-v1-results.md).
Because noncollapse did not produce predictive or alert value, the next
frontier changes the information and objective through native trace events
rather than applying another anti-collapse loss.
