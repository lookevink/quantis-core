# Edge dynamics development v1 results

## Result

All seven adjacent edge-runnable techniques were exercised on one
corpus-bound preprocessing cache of the qualified development-v1 corpus. The
bounded graph residual won the candidate selection split by `0.0054%`, then
failed to improve on the dense or low-rank references on evaluation. The most
useful engineering result is therefore the contractive low-rank model: it
preserved the dense model's predictive and closed-library attribution scores
with about half the parameters and serialized size.

This is open development evidence. The evaluation split had already informed
the redesign, so the result is not sealed confirmation and does not establish
a world model.

## Data boundary

The 120 matched pairs were assigned without pair leakage:

- 60 fit pairs, producing 9,480 windows;
- 15 selection pairs, producing 2,370 windows;
- 15 calibration pairs, producing 2,370 windows; and
- 30 existing development-evaluation pairs, producing 4,740 windows.

Normalization was fit on the fit role only. The 568 MB cache preserves
20-state histories, 10-step futures, controls, action tensors, whole-pair
identity, and the frozen 108-candidate attribution queries. Its directory is
addressed by the SHA-256 of the source artifact manifest, and cache reuse is
rejected if that source identity does not match.

## Predictive models

| Model | Action-overlap MSE | Overall MSE | Downstream effect MSE | Action+target hit@1 | No-action specificity | Parameters | CPU latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| Contractive low-rank | 0.3211 | 0.08194 | 0.08004 | 100% | 100% | 34,503 | 0.177 ms |
| Dense VARX reference | 0.3211 | 0.08194 | 0.08004 | 100% | 100% | 67,704 | 0.224 ms |
| Bounded graph residual | 0.3211 | 0.08195 | 0.08006 | 100% | 100% | 62,217 | 0.947 ms |
| Echo-state network | 0.3551 | 0.1152 | 0.1356 | 100% | 100% | 76,408 | 0.446 ms |
| Direct temporal convolution | 1.4861 | 0.3097 | 0.2878 | 20% | 100% | 112,350 | 1.223 ms |
| Persistence | 2.7305 | 0.6278 | 0.3176 | — | — | — | — |

All learned rollouts remained finite over the frozen ten-step horizon. Their
maximum state-norm growth over that horizon ranged from `5.11×` for the TCN to
`15.10×` for the other candidates. Finite rollout and a bounded transition
operator do not establish empirical contraction or long-horizon stability.
Batch-one latency is a local CPU microbenchmark, not a portable hardware
guarantee.

The selected rank-32 low-rank configuration had spectral radius `0.8714`.
Relative to persistence, it improved action-overlap error by `88.2%` and
paired downstream-effect error by `74.8%`. It matched the dense reference to
displayed precision while using `49.0%` fewer parameters, a `51.6%` smaller
JSON artifact, and `21.0%` lower local batch-one latency.

The perfect attribution scores apply only to the closed, randomized,
known-action library on this lab stack. They do not imply attribution of
arbitrary production incidents.

## Technique interpretations

### Echo-state network

The 16-unit reservoir beat persistence substantially and preserved
closed-library attribution. It was `10.6%` worse than low-rank on
action-overlap MSE and worse on downstream effects. Nonlinear reservoir
capacity did not earn its extra parameters here.

### Direct causal temporal convolution

The TCN remained finite but underfit the intervention mechanisms. Its action
error was `4.63×` the low-rank model and action attribution fell to `20%`.
This edge-sized direct-horizon configuration is rejected; a different
architecture or training budget would be a new experiment.

### Contractive low-rank transition

Rank 8 was too small, rank 16 was close, and ranks 24 and 32 recovered dense
performance. The stable global channel retained the cross-entity information
lost by the earlier graph-only factorization. This model is a strong
engineering reference and a candidate for the next preregistration, but it
was not the protocol-selected winner of this open tournament.

### Bounded graph residual

The graph residual was the protocol-selected adjacent candidate, beating
low-rank on selection action MSE by about `0.0054%`. On evaluation it was
`0.0023%` worse, used 27,714 more parameters, and had about `5.3×` the local
latency. This provides no evidence that the graph correction helped.

### Structured log and trace features

On the 27 metric targets, removing all four structured event inputs changed
action-overlap MSE from `0.2981271995` to `0.2981270809`. The negligible
difference slightly favors removing them. The current aggregate events added
no measurable predictive information.

The streaming audit processed 442,917 messages into exactly three templates:

- `checkout accepted`: 219,989;
- `checkout completed`: 219,989; and
- `checkout rejected`: 2,939.

The three-key template payload was only 76 bytes. This validates inexpensive
streaming parser plumbing, not natural-language generalization.

### Conformal and sequential detection

The detector received no action tensor. A 1% point threshold detected `96.7%`
of treatment trajectories with median delay zero, but alarmed on `40%` of
control trajectories because each capture contains 79 overlapping chances to
alarm.

Sequential accumulation eliminated control false alarms and pre-onset alarms
in this evaluation, but detected only `60%` of treatments with median delay
`17.5` transitions. The warning mechanism remains useful, but this
calibration does not yet offer a dependable sensitivity/false-alarm tradeoff.

### Streaming sketch

A 4×128 Count-Min Sketch used 4,096 bytes and reconstructed all 28
denormalized entity-event counts exactly. The current collision-free address
round-tripped the selected predictor's historical event inputs within
`1.3e-8` normalized units, with zero change in its reported scores. However,
the equivalent exact key/value payload was only 1,109 bytes. The sketch is
therefore larger for this tiny vocabulary and provides no evidence of a
high-cardinality advantage.

## Bounded conclusion

The experiment supports a narrow open-development claim:

> On this fixed Quantis lab and closed randomized action library, compact
> action-conditioned global dynamics substantially outperform persistence
> for ten-step prediction and support exact library attribution.

It rejects or fails to support four stronger readings:

1. the graph residual did not improve the global model on evaluation;
2. current structured logs did not add predictive information;
3. the TCN and reservoir did not improve the linear state-space reference; and
4. anomaly calibration is not yet operationally dependable.

The next scientific step is to preregister one low-rank predictor and
run/block-aware detector, freeze preprocessing and thresholds, then collect a
fresh sealed confirmation corpus. Richer application and SDK log templates
and a substantially larger event vocabulary should be collected separately
before another NLP or sketch-compression claim.
