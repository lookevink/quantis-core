# Edge dynamics development v1 results

## Result

All seven adjacent edge-runnable techniques were exercised on a single
content-addressed preprocessing cache of the qualified development-v1 corpus.
The practical recommendation is the **contractive low-rank
action-conditioned model**.

This is open development evidence. The evaluation split had already informed
the redesign, so the result is not sealed confirmation and does not establish
a world model.

## Data boundary

The 120 matched pairs were assigned without pair leakage:

- 60 fit pairs, producing 9,480 windows;
- 15 selection pairs, producing 2,370 windows;
- 15 calibration pairs, producing 2,370 windows; and
- 30 existing development-evaluation pairs, producing 4,740 windows.

Normalization was fit on the fit role only. The compiled 568 MB cache preserves
20-state histories, 10-step futures, controls, action tensors, whole-pair
identity, and the frozen 108-candidate attribution queries.

## Predictive models

| Model | Action-overlap MSE | Overall MSE | Downstream effect MSE | Action+target hit@1 | No-action specificity | Parameters | CPU latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| Contractive low-rank | 0.3211 | 0.08194 | 0.08004 | 100% | 100% | 34,503 | 0.153 ms |
| Dense VARX | 0.3211 | 0.08194 | 0.08004 | 100% | 100% | 67,704 | 0.191 ms |
| Bounded graph residual | 0.3211 | 0.08195 | 0.08006 | 100% | 100% | 62,217 | 0.812 ms |
| Echo-state network | 0.3551 | 0.1152 | 0.1356 | 100% | 100% | 76,408 | 0.374 ms |
| Direct temporal convolution | 1.4861 | 0.3097 | 0.2878 | 20% | 100% | 112,350 | 1.089 ms |
| Persistence | 2.7305 | 0.6278 | 0.3176 | — | — | — | — |

All learned rollouts remained finite over the frozen ten-step horizon.
Batch-one latency is a local CPU microbenchmark and is not a portable hardware
guarantee.

The rank-32 low-rank model had spectral radius `0.8714`. Relative to
persistence, it improved action-overlap error by `88.2%` and paired downstream
effect error by `74.8%`. It matched the dense model to displayed precision
while using `49.0%` fewer parameters and a `51.6%` smaller JSON artifact.

The perfect attribution scores are meaningful only for the closed,
randomized, known-action library on this lab stack. They do not imply
attribution of arbitrary production incidents.

## Technique interpretations

### Echo-state network

The 16-unit reservoir was viable and stable, beating persistence substantially
and preserving closed-library attribution. It was `10.6%` worse than the
low-rank model on action-overlap MSE and worse on downstream effects. Nonlinear
reservoir capacity did not earn its extra parameters here.

### Direct causal temporal convolution

The TCN was stable but underfit the intervention mechanisms. Its action error
was `4.63×` the low-rank model and action attribution fell to `20%`. A larger
network or longer training might improve it, but the present edge-sized direct
horizon model is rejected.

### Contractive low-rank transition

This was the strongest edge tradeoff. The rank sweep showed rank 8 was too
small, rank 16 was close, and ranks 24/32 recovered dense performance. The
stable global channel retained the cross-entity information lost by the
earlier graph-only factorization.

### Bounded graph residual

The graph residual produced the lowest selection MSE, but only by about
`0.005%`. It then became `0.002%` worse than low-rank on evaluation while
using 27,714 more parameters and roughly five times the batch-one latency.
This is no evidence that the graph correction helped.

The raw selection winner remains recorded as graph residual. A post-hoc edge
recommendation chose the fewest-parameter model among candidates within 1% of
the selection minimum, yielding low-rank. That parsimony rule is explicitly
not a frozen scientific gate.

### Structured log and trace features

On the 27 metric targets, removing all four structured event inputs changed
action-overlap MSE from `0.2981190241` to `0.2981190249`. The difference is
negligible. The current aggregate events neither helped nor hurt this model in
a measurable way.

The streaming template audit processed 442,917 messages into exactly three
templates:

- `checkout accepted`: 219,989;
- `checkout completed`: 219,989; and
- `checkout rejected`: 2,939.

This validates inexpensive parser plumbing, not natural-language
generalization.

### Conformal and sequential detection

The detector received no action tensor. A 1% point threshold detected `96.7%`
of treatment trajectories with median delay zero, but alarmed on `40%` of
control trajectories because each capture contains 79 overlapping chances to
alarm.

Sequential accumulation eliminated control false alarms and pre-onset alarms
in this evaluation, but detected only `60%` of treatments with median delay
`17.5` transitions. The warning mechanism remains useful, but this calibration
does not yet offer an acceptable sensitivity/false-alarm tradeoff.

### Streaming sketch

A 4×128 Count-Min Sketch used 4,096 bytes and reconstructed all 28 observed
entity-event keys exactly. With only 28 keys, this demonstrates correctness
but provides no evidence of a high-cardinality advantage.

## Bounded conclusion

The experiment strengthens a narrow claim:

> The fixed Quantis lab contains compact, stable, action-conditioned global
> dynamics that support accurate ten-step prediction and closed-library
> attribution on open development data.

It rejects three stronger readings:

1. the graph residual did not improve the global model;
2. current structured logs did not add predictive information; and
3. the anomaly calibration is not yet operationally dependable.

The next scientific step is to freeze the low-rank model and preprocessing,
replace overlapping point calibration with run- or block-aware calibration,
then collect a fresh sealed confirmation corpus. Richer application and SDK
log templates should be collected separately before another NLP claim.
