# Contextual metrics + logs JEPA v1 development result

## Outcome

Demand correction and contextual prediction removed the severe schedule
transfer failure seen in multimodal v0, but the current application-log
vocabulary still provides, at most, a small and unstable incremental signal.

The selected hypothesis for a new untouched corpus is **L1 loss with one log
latent dimension**. This is a development choice, not a confirmation result.
Its family-held-out advantage is only 0.038 percentage points, and shuffled
logs perform better on the previously exposed validation families. The next
corpus must therefore retain the shuffled-log and metrics-only controls.

## Development comparison

All comparisons use the original training families 1–8 in eight
leave-one-schedule-family-out folds. The original validation families 9–10
were already inspected and appear only in the diagnostic columns.

| Candidate | Training-family alert rate | Metrics-only | No-worse folds | Exposed validation | Exposed shuffled logs | Selection |
|---|---:|---:|---:|---:|---:|---|
| Huber, 1 log latent | 2.134% | 2.096% | 6/8 | 1.931% | 1.931% | Failed |
| Huber, 2 log latents | 2.121% | 2.096% | 3/8 | 1.474% | 1.524% | Failed |
| L1, 1 log latent | 2.007% | 2.045% | 7/8 | 2.795% | 2.744% | Passed |
| MSE, 1 log latent | 2.160% | 2.071% | 7/8 | 1.778% | 1.677% | Failed |

Selection ignores the exposed-validation columns. Because the same eight folds
were used to compare candidates, even the training-family rate is a
model-selection statistic rather than an unbiased generalization estimate.

Each fold refits metric, log, and control normalization without its held-out
family, pretrains its encoders for 40 epochs, and performs 20 additional
predictor-only epochs with the encoders frozen. Its metrics-only comparator
uses the same contextual block encoder, conditioned nonlinear predictor,
multi-horizon objective, rollout, and loss; only the log stem is removed.

The L1 result is uneven: seven folds are no worse than metrics-only, but one
schedule family reaches a 9.45% alert rate versus 4.27% metrics-only. That
single-family failure is the clearest warning against treating the small mean
advantage as established transfer.

## Representation diagnostics

The selected L1/1 model's metric effective rank is 2.21 of 3 and its log
effective rank is 1.00 of 1. Huber/2 reaches only 1.008 effective log
dimensions out of
2. This confirms that the representation is active but that the current
normal-run logs remain essentially one-dimensional.

Frozen joint-latent probes on L1/1 recover:

- worker completion ratio with training \(R^2 = 0.580\);
- queue transition direction with \(R^2 = 0.586\);
- checkout completion and backlog ratios with \(R^2 = 0.554\);
- request latency with \(R^2 = 0.418\); and
- absolute queue depth with only \(R^2 = 0.006\).

The latency-tertile probe reaches 65.4% training accuracy. Queue depth does not
have enough normal-run variation to form three buckets. The EMA decay of 0.98
corresponds to an effective update half-life of 34.31 pretraining epochs.

These are same-training-data probes. They establish decodability, not causal
usefulness or held-out probe generalization.

## Interpretation

This is no longer evidence of representation collapse. Multi-horizon targets,
observable demand/topology controls, and demand-relative logs bring held-out
normal alert rates back to the metrics-only range. The remaining problem is
information content: accepted and completed events still describe almost the
same fact, and the second log dimension has almost no independent variance.

The results also do not establish positive multimodal transfer. Improvements
are small, differ across schedule families, and often survive shuffled log
alignment. The next collection should add bounded endogenous application-state
events—such as queue/backlog transitions, database latency buckets, worker
state changes, and completion outcomes—without reintroducing identifiers or
free text.

L1 is retained because it has the best training-family development statistic
and because V-JEPA reported more stable training with L1 feature prediction
([Bardes et al., 2024](https://arxiv.org/abs/2404.08471)). Contextual block
targets and the frozen conditioned dynamics stage follow
[I-JEPA](https://arxiv.org/abs/2301.08243) and
[V-JEPA 2](https://arxiv.org/abs/2506.09985), respectively. These sources
motivate the architecture; they do not validate its transfer to this small
telemetry corpus.

## Evidence boundary

The artifacts live under
`artifacts/jepa-world-model-v1/contextual-multimodal-development` and
`artifacts/jepa-world-model-v1/preflight`. They are intentionally ignored by
Git like the earlier JEPA development artifacts and are reproducible from the
preserved v2 captures.

A new untouched corpus, collected only after the event vocabulary and L1/1
configuration are frozen, is required before any publication or application
promotion decision.
