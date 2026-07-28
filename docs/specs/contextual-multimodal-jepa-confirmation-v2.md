# Contextual metrics + dependency logs JEPA confirmation v2

## Status

The confirmation design is frozen and ready to run after its implementation,
protocol, feature specifications, and runner are committed together. No case
from this corpus may be collected from a dirty worktree or assessed against a
different commit.

This is a confirmation of a deliberately narrow representation-learning
claim. It is not a promotion to production and it is not yet a validated
world-model claim.

## Preregistered claim

Within the fixed Quantis checkout stack, aligned bounded application and
dependency logs improve compact frozen-state prediction and normal schedule
transfer beyond capacity-matched metrics-only and alignment-shuffled controls.

The stack boundary is the checkout API, Redis queue, PostgreSQL database, and
one to three checkout workers. The evidence is limited to fault-free schedule
transfer. Passing does not establish fault detection, counterfactual
prediction, closed-loop control, cross-application generalization, or a
general-purpose world model.

This wording follows the distinction in the JEPA literature between predicting
in representation space and showing that the learned representation preserves
useful state. I-JEPA and V-JEPA evaluate frozen representations on downstream
tasks, while V-JEPA 2 adds action-conditioned prediction and closed-loop
planning before making a world-model utility claim
([Assran et al., 2023](https://arxiv.org/abs/2301.08243),
[Bardes et al., 2024](https://arxiv.org/abs/2404.08471),
[Assran et al., 2025](https://arxiv.org/abs/2506.09985)).

## Untouched corpus and collection schedule

The corpus has 24 new request-schedule families crossed with three worker
topologies, for 72 runs:

- families 1–12 and their 36 runs form the training split;
- families 13–24 and their 36 runs form the untouched validation split;
- every run contains 340 points sampled at 0.1 seconds;
- the context lookback is six points;
- targets are contiguous two-point blocks at horizons 1, 3, and 6; and
- the application request queue is fixed at 128.

The 24 canonical request schedules are unique and disjoint from the inspected
development corpus. The validation half exactly matches the training half's
marginal request-rate and schedule-length design in new orderings. This makes the
primary transfer question schedule dynamics rather than an easy shift in
aggregate demand.

Collection uses three isolated Docker Compose projects. Each family is one
batch containing its one-, two-, and three-worker runs. A Latin rotation maps
topology to lane, so each topology appears in each lane exactly eight times
over the full corpus. Batches execute sequentially and the three cases inside a
batch execute concurrently. The collector records case, lane, batch, project,
and timing in a collection attestation.

Training and validation families alternate in the frozen collection order:
1, 13, 2, 14, through 12, 24. Validation labels still remain untouched by
training and assessment. Interleaving collection prevents a slow machine or
lab-state drift from being perfectly confounded with the train/validation
boundary.

The old process spent roughly 43 seconds of wall time per case, about 21
minutes collecting 30 cases, and about seven minutes compiling and training.
The three-lane design targets approximately 17 minutes for 72 new cases rather
than approximately 51 minutes sequentially. The exact elapsed time remains an
observed result, not an acceptance gate.

Profiling also showed that most host-side time was log loading and event
matching, while the selected 300 training epochs added roughly 24 seconds on
the 30-run development corpus. The confirmation therefore parallelizes the
stack and keeps the existing NumPy/Apple Accelerate implementation. It does not
introduce an MLX backend into a preregistered scientific comparison where GPU
kernel time is not the primary bottleneck. MLX remains a later option if a
larger nonlinear model makes training dominant; its relevant mechanisms are
compiled graph execution and unified CPU/GPU memory
([MLX compilation](https://ml-explore.github.io/mlx/build/html/usage/compile.html),
[MLX unified memory](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html)).

## Fixed model and controls

The selected recipe is frozen before validation:

- metric latent width 3 and log latent width 1;
- 200 EMA representation-pretraining epochs;
- 100 frozen-encoder predictor-refinement epochs;
- no cross-validation or post-collection candidate selection;
- learning rate 0.02, EMA decay 0.98, and weight decay 0.0001;
- Huber loss with delta 1.0;
- auxiliary and rollout weights 0.2; and
- calibration quantile 0.98.

The aligned model is compared with continuity metrics-only,
capacity-matched metrics-only, shuffled-log, log-only, metric-context-only,
and log-context-only controls. Raw-context ridge and 12-dimensional PCA-context
ridge baselines are included for frozen-state transfer.

The shuffled-log control preserves complete log context/target blocks but
breaks their alignment with metrics using fixed split-specific seed offsets.
That control is central because predictive joint-embedding objectives can
prefer an easy, stable signal over task-relevant state
([Sobal et al., 2022](https://arxiv.org/abs/2211.10831)).

## Frozen representation transfer

Every fitted encoder is frozen before the probes are trained. Context is
reduced to a 12-dimensional vector by concatenating the four-dimensional
metric-plus-log state from each of the three context patches. Ridge probes are
fit only on training families and evaluated only on validation families.

The eight fixed future-state targets cover latency, queue depth, worker
completion, checkout completion and backlog, PostgreSQL pressure, Redis
pressure, and checkout queue-wait pressure. Each target block is reduced by
its mean. The same probe protocol is applied to aligned JEPA, metrics-only,
capacity-matched metrics-only, shuffled logs, raw 108-dimensional context, and
training-fit 12-dimensional PCA context.

This answers the compression question directly: can a 12-value frozen state
retain the future-state information available in 108 raw context values? A
small training objective alone is not treated as evidence of meaningful
compression.

## Seeds, statistics, and determinism

The five fixed training seeds are 89, 97, 101, 103, and 107. Seed 89 is
trained a second time in a distinct, non-overlapping execution. Its serialized
artifacts must be byte-identical to the primary seed-89 artifacts.

The independent statistical unit is the schedule family, not the window.
Within each validation family, scores are averaged across the five seeds.
Comparisons then use an exact one-sided paired sign-randomization test over the
12 validation-family differences. The maximum p-value is 0.05. This avoids
treating heavily overlapping windows or five models trained on the same corpus
as independent observations.

## Confirmation gates

The narrow claim is supported only if every gate passes:

- overall aligned validation alert rate is at most 3%;
- every validation family's alert rate is at most 5%;
- aligned logs are no worse than metrics-only in at least 75% of families;
- aligned logs beat both capacity-matched metrics-only and shuffled logs under
  the exact paired test;
- at least 80% of training seeds beat both controls in aggregate;
- metric and log effective ranks are at least 1.5 and 0.5 respectively;
- at least five of the eight frozen probe targets are evaluable;
- aligned 12-dimensional state has at most 1.25 times the raw-context probe
  error; and
- aligned state has normalized probe MSE at most 1.0, so it improves on the
  training-mean reference; and
- aligned frozen state beats capacity-matched, shuffled-log, and
  12-dimensional PCA state under the exact paired test.

Thresholds, schedules, seeds, and target variables may not be revised after
collection. A failed gate is not a failed experiment.

## Publication package

Either outcome is publication-ready if the preregistration, clean commit,
collection attestation, five seed artifacts, deterministic repeat, raw corpus,
family-level statistics, frozen probes, and limitations are intact.

A positive result supports only the preregistered constrained-stack claim. A
negative result reports which of three interpretations survived:

1. aligned compact state and normal transfer both worked;
2. compact state transferred but aligned logs did not clear the controls; or
3. the compact state did not preserve enough held-out operational state.

The assessor chooses the subsequent milestone without retuning this result:

- supported claim → collect an action-conditioned intervention corpus;
- compression only → repair log alignment before dynamics work; or
- unsupported compression → improve state observability before dynamics work.

The first branch is the next step toward a constrained world model: introduce
observable actions such as worker scaling and dependency recovery, train a
causal latent predictor, test multi-step rollouts, and evaluate whether its
predictions select better interventions. That is the first experiment
comparable in kind—though not scale—to V-JEPA 2's action-conditioned evidence.

## Run

After committing the frozen design:

```bash
./lab/fault_matrix/run-contextual-multimodal-jepa-confirmation-v2.sh
```

The runner refuses dirty state or an existing output directory. It verifies
the committed file hashes, builds the stack once, collects 24 three-case
batches, trains five seeds plus the deterministic repeat, and writes the final
assessment under
`artifacts/jepa-world-model-v2/contextual-confirmation-v2/assessment`.
