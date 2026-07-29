# Retrieval-JEPA tracer v1 result

## Decision

Reject the frozen episode-predictive retrieval-JEPA recipe.

The candidate passed every causal-input, role, finiteness, restoration,
observable-state, and local edge-feasibility gate. It did not establish
retrieval or empirical-abstention value. On the ten held-out-topology
treatment episodes, raw telemetry and deterministic PCA both achieved
`1.00` hit@1, `1.00` hit@3, and `1.00` mean reciprocal rank. The
episode-predictive JEPA achieved `0.40` hit@1, `0.50` hit@3, and `0.519180`
mean reciprocal rank. Its calibration-role threshold abstained from every
held-out-topology treatment query, yielding zero coverage and zero
accepted-and-correct rate.

Do not advance this recipe to multi-seed robustness, sealed confirmation,
ANN/quantization work, or an alerting integration.

## Frozen test

The run followed
[`Retrieval-JEPA evidence and abstention contract v1`](../specs/retrieval-jepa-evidence-contract-v1.md)
without result-driven changes:

- the content-addressed `eb542...b3ff` preprocessing cache;
- 40 fit pairs on worker topologies one and two;
- one midpoint treatment episode and matched control per pair;
- a 40-item, treatment-only evidence bank sorted by episode ID;
- a width-64, two-block, four-head V-JEPA-style context encoder and
  positional predictor;
- 400 CPU AdamW steps, seed `9019`, and target EMA `0.996`;
- a whole-pair-deranged JEPA null, CPC/InfoNCE, a capacity-matched supervised
  retriever, raw telemetry, and deterministic width-64 PCA;
- exact cosine search with `K=3`; and
- one model-specific empirical class-margin threshold fit on the 30
  calibration episode queries.

The query path received only the 20-point context and declared graph.
Counterfactual stored-array checks altered future evidence, topology,
action labels, and pair IDs while holding the context fixed; every query
vector remained bit-identical.

## Primary held-out-topology result

| representation | hit@1 | hit@3 | pair-balanced MRR | accepted correct | treatment coverage | selective accuracy | control specificity |
|---|---:|---:|---:|---:|---:|---:|---:|
| episode-predictive JEPA | 0.40 | 0.50 | 0.519 | 0.00 | 0.00 | 0.00 | 1.00 |
| deranged-target JEPA | 0.20 | 0.60 | 0.418 | 0.00 | 0.00 | 0.00 | 1.00 |
| CPC / InfoNCE | 0.50 | 0.50 | 0.580 | 0.20 | 0.20 | 1.00 | 0.90 |
| raw telemetry | 1.00 | 1.00 | 1.000 | 0.40 | 0.40 | 1.00 | 1.00 |
| deterministic PCA-64 | 1.00 | 1.00 | 1.000 | 0.40 | 0.40 | 1.00 | 1.00 |
| supervised retriever | 1.00 | 1.00 | 1.000 | 0.10 | 0.10 | 1.00 | 1.00 |

The selection topology told the same story. The candidate reached `0.20`
hit@1, `0.60` hit@3, and zero accepted-and-correct rate, while raw and PCA
were perfect on fixed retrieval and reached `0.60` accepted-and-correct
rate.

The candidate improved hit@1 over its deranged-target null (`0.40` versus
`0.20`) and preserved useful state, but it did not beat CPC and remained far
behind the non-learned controls. It retrieved both Redis delay mechanisms
better than PostgreSQL lock and worker pause, but the two queries per action
are diagnostics, not stable action-level estimates.

## Safety and feasibility

All safety gates passed:

- held-out owned-state NRMSE was `0.631223`, within the PCA safety bound
  (`1.278562`);
- the 40-item candidate bank had effective rank `39`;
- restored query and bank vectors were exact, and all rankings and abstention
  decisions were identical;
- every stored similarity recomputed from its stored vectors;
- the query counterfactual, role-disjointness, bank-identity, and episode-count
  checks passed; and
- all stored numeric evidence was finite.

The local CPU microbenchmark reported:

- 149,568 online parameters and 221,056 retained training parameters;
- 1,152,107 serialized candidate bytes;
- 20,480 bytes for the 40-by-64 float64 bank;
- `1.1253 ms` median batch-one query latency; and
- `0.0051 ms` median exact-search latency.

These measurements establish local feasibility only. They are not a
target-device latency claim.

The episode-predictive loss fell from `1.152703` to `0.066435`, while the
invalid whole-pair-deranged objective fell even lower to `0.040111`. Loss
reduction therefore did not establish useful episode semantics.

## Abstention boundary

The candidate found no calibration threshold that both rejected every
control query and accepted a correct treatment query. Its abstain-all policy
remained specific but had no investigation utility.

Raw and PCA fixed retrieval were perfect, yet their calibration thresholds
covered only `0.40` of transfer treatments. Even the supervised retriever,
despite perfect fixed retrieval, covered only `0.10`. The confidence margin
therefore shifts more than the neighbor ordering on this small corpus.

There are only 30 calibration trajectories from 15 matched pairs. This is
insufficient for a 10%-risk, 95%-confidence SGR guarantee. No threshold in
this tracer is production-calibrated; collecting more overlapping windows
would not fix the independent-unit shortfall.

## What the evidence establishes

Allowed:

- this exact episode-predictive JEPA recipe is not a promotion candidate on
  the current topology-transfer corpus;
- direct raw/PCA precedent retrieval is a stronger next investigation
  baseline than a learned JEPA query map on this corpus;
- the self-supervised candidate is restorable, non-collapsed, state-safe, and
  locally edge-runnable, but those properties are insufficient for retrieval
  value; and
- empirical reject-option evidence needs substantially more independent
  calibration episodes.

Not allowed:

- that retrieval-oriented JEPA is generally ineffective;
- that raw or PCA retrieval is ready for production;
- that the intervention labels are real-world root causes;
- that the empirical threshold has a finite-sample risk guarantee; or
- that this open-development result is sealed confirmation.

## Artifact and reproduction

The immutable local bundle is
[`artifacts/action-dynamics/prototype-retrieval-jepa-v1`](../../artifacts/action-dynamics/prototype-retrieval-jepa-v1).
It contains the fitted models and probes, every episode and raw evidence
reference, representations, similarity matrices, complete risk-coverage
curves, causal counterfactuals, restoration evidence, assessment, report,
reproduction-source copies, and a manifest for every evidence-bearing file.

- Artifact bytes: `9,426,284`
- Manifest files: `16`
- `artifact-manifest.json` SHA-256:
  `676b0cba10ebb66a77fe66a33376b06f418ee10736a12da3a3de3e8d991cc0ba`
- Standalone decision:
  `reject_episode_predictive_retrieval_jepa_recipe`

Recompute the conclusion without loading a fitted model:

```bash
.venv/bin/python \
  lab/action_dynamics/prototype_retrieval_jepa_assessor.py \
  artifacts/action-dynamics/prototype-retrieval-jepa-v1
```

The independent assessor reproduced `assessment.json` byte-for-byte. The
runner, assessor, production primitives, tests, frozen contract, source note,
and negative artifact remain retained.
