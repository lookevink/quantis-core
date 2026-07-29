---
status: closed
label: wayfinder:grilling
parent: Evaluate the unimplemented JEPA families
assignee: codex
blocked_by:
  - Test the complete multi-view LeJEPA telemetry tracer
---

# Freeze the retrieval-JEPA evidence and abstention contract

## Question

Can an episode-level retrieval JEPA improve topology-transfer incident-evidence
retrieval, calibrated abstention, and investigation utility over raw telemetry,
matched supervised retrieval, deterministic PCA, and non-JEPA metric-learning
controls without reusing direct trajectory error as its value claim or leaking
action labels into the query representation?

## Resolution comment

Resolved on 2026-07-28 under the frozen
[`Retrieval-JEPA evidence and abstention contract v1`](../../specs/retrieval-jepa-evidence-contract-v1.md).

The width-64 V-JEPA-style episode-predictive retriever was causal, restorable,
non-collapsed, state-safe, and locally edge-feasible. It did not add
investigation value. On held-out worker topology, it reached `0.40` hit@1,
`0.50` hit@3, and zero empirical abstention coverage. Raw telemetry and
deterministic PCA both reached `1.00` hit@1, `1.00` hit@3, and `1.00` mean
reciprocal rank. The candidate also lost fixed retrieval to the CPC control.

The stored-array assessor independently reproduced
`reject_episode_predictive_retrieval_jepa_recipe`, including causal
counterfactuals, complete risk-coverage curves, and restored bank, ranking,
and accept-decision parity. The bounded conclusion is recorded in
[`Retrieval-JEPA tracer v1 result`](../../research/retrieval-jepa-prototype-v1-results.md).
Do not advance this recipe. The runner, assessor, tests, and immutable local
artifact remain retained.
