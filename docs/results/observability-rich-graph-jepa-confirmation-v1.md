# Observability-rich graph-JEPA confirmation v1

## Decision

**Not supported. Do not publish the preregistered positive claim.**

The experiment supports a narrower negative result:

> In the fixed Quantis checkout lab, declared one-hop topology contains useful
> predictive signal, but the transferred-width learned EMA graph-JEPA does not
> preserve enough operational state to beat raw or frozen-PCA predictors.

This is not a world-model result. The corpus is fault-free and contains no
action-conditioned transitions.

## Execution

- Source instrumentation and corpus protocol commit: `511c8a5`
- Learned confirmation pipeline commit: `fccfa31`
- Final preregistered training execution commit: `1f8ef1a`
- Cases: 72 fresh runs, 24 schedule families, 1–3 workers
- Split: 12 training families / 12 untouched validation families
- Collection: 3 isolated Docker lanes, 24 batches
- Collection wall time: 19.78 minutes
- Raw evidence size: 1.1 GB
- Structured log records: 1,539,667
- Tensor cache: 12 MB, content key
  `66a7a840a0110807a9ac8221ed4c69152f7bf73e05a645f6f3aecab1084d8c77`
- Model windows: 11,160 training / 11,160 validation
- Training wall time: 6.09 minutes with four parallel seed workers
- Deterministic repeat: byte-identical

## Main results

| Predictor | Mean validation normalized MSE |
| --- | ---: |
| Raw one-hop ridge | **0.4641** |
| Raw flat ridge | 0.4740 |
| Frozen equal-width PCA one-hop | 0.4898 |
| Training mean | 0.7198 |
| Persistence | 0.9424 |
| Learned all-entity EMA JEPA | 1.2024 |
| Learned one-hop EMA JEPA | 1.2376 |
| Learned shuffled-topology EMA JEPA | 1.2987 |
| Learned entity-local EMA JEPA | 1.3434 |

The learned one-hop model beat entity-local and shuffled topology on all five
seeds, and was no worse than entity-local on all 12 validation families. That
is evidence that the declared topology is informative. It is not sufficient
evidence that the learned representation is useful: frozen PCA and raw ridge
were much better.

## Gate outcome

Passed:

- raw flat and raw one-hop prediction beat the training mean;
- raw one-hop mean normalized MSE was below 1.0;
- one-hop learned JEPA beat entity-local and shuffled-topology JEPA;
- one-hop retained all-entity performance within the frozen 5% margin;
- all 12 validation families were no worse than entity-local;
- all five seeds beat both entity-local and shuffled topology;
- the active context compressed from 740 values to 260 latent values,
  or 2.846:1; and
- the primary-seed repeat was byte-identical.

Failed:

- the critical raw Redis group had normalized MSE 1.315, driven by sparse
  thresholded latency log events;
- learned target reconstruction normalized MSE was 0.583, above the 0.10
  gate; and
- learned one-hop prediction did not beat equal-width frozen PCA.

## Interpretation in the JEPA literature

[I-JEPA](https://arxiv.org/abs/2301.08243) predicts representations of target
blocks from an informative context block. [V-JEPA](https://arxiv.org/abs/2404.08471)
extends feature prediction over time and deliberately does not optimize pixel
reconstruction. Our raw-state reconstruction gate is therefore an
observability and attribution requirement, not a generic requirement for
calling an objective “JEPA.”

That distinction does not rescue this result. For the declared downstream
task, the learned representation must retain subsystem state well enough to
decode and attribute prediction error. It did not. Its latent training loss
decreased for every seed, but validation raw-state prediction remained much
worse than both raw ridge and frozen PCA.

A post-confirmation, training-only width sweep found that the richer state
requires 40 dimensions per temporal patch to reach mean per-entity
reconstruction error at or below 0.10:

- API: 4
- checkout queue: 6
- worker pool: 11
- Redis: 3
- PostgreSQL: 3
- enqueue edge: 4
- dequeue edge: 4
- PostgreSQL-write edge: 5

That still gives 1.85:1 context compression. The transferred 26-dimensional
budget was too small, and the learned EMA basis was also substantially less
state-preserving than a PCA basis at the same 26-dimensional budget.

## Next development sequence

1. Treat this complete corpus as development data; its validation split has
   now been opened and cannot serve as untouched evidence again.
2. Replace sparse Redis threshold-event targets with continuous Redis node
   state, such as generic dependency-call latency, availability, and
   last-success age. Keep enqueue/dequeue edge timing separate.
3. Use the training-only 40-dimension widths and add an attribution-oriented
   state-preservation objective or decoder alongside latent JEPA prediction.
   This is intentionally stricter than V-JEPA because dependable subsystem
   attribution is part of our claim.
4. Require the learned model to match or beat frozen PCA in development,
   while retaining the local and shuffled-topology advantages, before
   collecting another untouched schedule corpus.
5. Only after nominal graph prediction passes, collect paired
   disturbance/action/recovery episodes. Condition transitions on explicit
   actions such as worker scaling, retry policy, lock release, and queue
   backpressure; test multi-step rollout and held-out interventions.

The fifth step is the first experiment that could support a constrained world
model claim. This follows the important boundary demonstrated by
[V-JEPA 2](https://arxiv.org/abs/2506.09985): latent prediction becomes useful
for planning only after an action-conditioned model is trained and evaluated
for interaction.
