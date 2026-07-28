# Hybrid telemetry JEPA development v1 results

## Decision

Do not promote this representation and do not describe it as a world model.

On the fixed nominal Quantis checkout stack, the corrected temporal graph
JEPA exposed a small, seed-stable improvement over representation-budget-
matched PCA and raw-history controls on a frozen linear future-state probe.
It did not expose a stable benefit from declared topology or structured
application events. It simultaneously collapsed to approximately one
effective latent direction in its weakest entity and failed to preserve
recoverable local state. The result is useful as a bounded negative finding,
not as evidence of a dependable system representation.

The final review-corrected artifact is:

`artifacts/jepa-world-model-v3/hybrid-telemetry-jepa-development-v1-reviewed-final/assessment.json`

Earlier diagnostic and intermediate corrected artifacts omitted parts of the
final control suite or used confounded diagnostics. They are retained only as
implementation history and must not be used as scientific evidence.

## Data and runtime

- 72 nominal runs: 36 training and 36 held-out schedule validation runs.
- 1,539,667 application-log records: 777,603 training and 762,064 validation.
- 13 training-fitted event templates; no validation-only templates.
- 0 populated trace/span links, so no trace-supervised propagation claim.
- 11,160 graph windows per split, each with 20 context steps, 9 declared
  node/edge entities, 9 padded operational slots, and 20 structured-event
  slots.
- 88 observed/applicable entity-feature slots: 37 operational and 51 event.
- 1,760 observed context scalars per sample versus 576 nominal latent values,
  a nominal 3.06x compression (3.44x if the entity with no observations is
  excluded).
- First structured-event ingestion took about 75 seconds with 12 reader
  threads. Verified cache reuse took 0.36 seconds.
- The final 3-seed x 3-ablation, 12-epoch matrix took 1,519 seconds
  (25.3 minutes). Training was serialized on one MPS accelerator.

## Corrected experiment

The corrected run:

- fits the event vocabulary, numeric scaling, PCA, and ridge probes only on
  training data;
- binds its event cache to the graph corpus, structured-event protocol, and
  all 72 raw-capture hashes;
- preserves typed `event:` versus `body:` template identity;
- masks both fine and pooled coarse views;
- injects identical entity/time masks into declared and shuffled models;
- predicts masked current EMA targets as well as future EMA targets;
- conditions future prediction on the supplied controls;
- measures current local-state recovery separately from future forecasting;
- compares raw context, JEPA, and 64-wide per-entity PCA with identical
  one-hop frozen, training-standardized ridge probes carrying explicit horizon
  and target-block identity;
- evaluates the shuffled representation through the same declared one-hop
  probe neighborhood, so topology is the only changed variable;
- reconstructs the full local history only from the compressed per-entity
  state used by the probes; and
- applies anti-collapse regularization and effective-rank measurement to each
  entity encoder across samples, before controls enter the predictor.

## Results

| Representation/control | Operational one-hop probe NRMSE | Local probe NRMSE | Context recovery NRMSE | Minimum rank fraction |
|---|---:|---:|---:|---:|
| Declared-topology hybrid JEPA | 0.434573 | 0.436188 | 0.808835 | 0.016883 |
| Shuffled-topology hybrid JEPA | 0.434607 | 0.436815 | 0.718770 | 0.020721 |
| No-event JEPA | 0.434792 | 0.437255 | 0.682771 | 0.019521 |
| Matched 64-wide PCA | 0.4425 | 0.4455 | n/a | n/a |
| Raw hybrid context | 0.4452 | n/a | n/a | n/a |

Across seeds, declared-topology frozen-probe NRMSE was 0.434106, 0.434781,
and 0.434833 (standard deviation 0.000331).

The declared model improved the matched PCA probe by 0.007895 absolute (1.78%
relative) and raw context by 0.010674 (2.40% relative). Its improvement over
shuffled topology was only 0.000034 (0.0078% relative), below the
preregistered 1% minimum. Paired relative topology effects were +0.0670%,
-0.0406%, and -0.0030%; the 95% confidence interval was -0.1279% to +0.1435%.
The direction therefore did not replicate. The event-feature improvement over
the no-event model was 0.000219 (0.0504% relative), also negligible.

They do not establish a topology-shaped world representation:

- The one-hop probe barely outperformed the entity-local probe (0.434573
  versus 0.436188), while the topology-specific and event-specific effects
  were too small to support a meaningful propagation or application-log
  claim.
- Minimum effective-rank fraction was approximately 1/64. The nominal
  64-dimensional state had effectively collapsed to one varying direction in
  its weakest entity/horizon stratum.
- Current-state recovery missed its gate by more than 8x (0.809 versus <=0.1).
  A representation that cannot recover its own node/edge observations is not
  dependable for attribution.
- The jointly trained decoder was unstable and much worse than the frozen
  probe. Structured event targets were especially poorly decoded.
- The corpus contains nominal workload variation but no interventions and no
  trace linkage. It cannot identify causal transitions or propagation.

## Narrow publishable claim

The defensible claim is:

> On 72 runs from one fixed software stack, a masked temporal JEPA
> representation produced a small, seed-stable improvement in frozen one-hop
> future-state probing over raw-context and representation-budget-matched PCA
> controls. Structured application events contributed only a negligible
> incremental effect, and the declared-topology effect was indistinguishable
> from zero under a matched probe scope. The encoder latent collapsed and
> failed local-state recovery, so the experiment does not support an
> attributable state representation or world-model claim.

This is consistent with the earlier paper analysis: JEPA can discard
unpredictable detail, but nominal repetitive telemetry provides too little
state diversity to determine which discarded information is essential.
[I-JEPA](https://arxiv.org/abs/2301.08243) and
[V-JEPA](https://arxiv.org/abs/2404.08471) motivate prediction in
representation space, while [V-JEPA 2](https://arxiv.org/abs/2506.09985)
separates action-free pretraining from action-conditioned world-model
learning. This run only addresses the former.

## Subsequent step toward a metrics-and-logs world model

Stop further nominal-only JEPA architecture sweeps. The next experiment should
change the information content, supervision, and evaluation:

1. Collect randomized intervention/recovery trajectories over the same
   declared graph: worker scaling, dependency latency, queue pressure,
   process kill/restart, lock contention, and retry/backpressure changes.
2. Emit explicit action tokens and trace/span correlations. Keep continuous
   metrics as state targets; treat structured logs as sparse auxiliary events
   with template/outcome classification and time-to-next-event losses rather
   than exact count reconstruction.
3. Train a supervised action-conditioned temporal graph state-space model
   first. Predict multi-step node/edge state distributions and event
   intensities from state plus action.
4. Evaluate held-out intervention types, rollout horizon degradation,
   recovery timing, counterfactual action ranking, and node/edge attribution
   against topology-shuffled and no-action controls.
5. Reintroduce JEPA only as an auxiliary masked-pretraining loss if the
   action-conditioned model has healthy per-entity rank and state recovery.

That sequence offers the highest probability of reaching a valid constrained
world-model claim because it supplies the intervention-conditioned transitions
that nominal logs and metrics cannot identify.
