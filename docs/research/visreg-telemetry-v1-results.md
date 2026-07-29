# VISReg telemetry tracer v1 results

## Decision

**Reject this exact detached VISReg telemetry recipe. Do not run seed
robustness or sealed confirmation.**

The preregistered source mechanism is real on the synthetic radial-collapse
screen: detached VISReg produced much stronger small-radius gradients than
exact SIGReg, and detaching the scale denominator made the shape gradient
strictly stronger than the no-detach falsifier. That mechanism did not
prevent collapse during telemetry training. The detached candidate reached
projector effective rank `1.12`, population-scale loss approximately `1.0`,
and almost zero invariance loss on both held roles. Its held-topology
downstream-effect error was `1.97×` raw, it lost to the no-detach falsifier
and masked reconstruction, and it won only half of held pairs.

This rejects the fixed `0.4` invariance plus `0.6` detached VISReg
translation on the Quantis telemetry stack. It does not reject the paper's
image results.

## Reproducible evidence

- implementation commit:
  `985d448308dd32b65782a462abd27f7fcbca3859`;
- conclusion-bearing immutable artifact:
  `artifacts/action-dynamics/prototype-visreg-v1`;
- artifact-manifest SHA-256:
  `029dd91e4b158b82ca9658ac4e97bf825ba7097a2988d5dbb1a594d573b44b18`;
- artifact size: 1,310,350,134 bytes across 60 files;
- 1,600 steps for each of two equal-capacity VISReg cells; and
- 40 fitting, 10 selection, 10 calibration, 20 IID evaluation, and 10
  held-topology evaluation pairs.

The artifact retains both cells, all four identity-verified complete-LeJEPA
controls, PCA and raw models, every probe, per-step embeddings and
directions, quantile and sorted-projection receipts, all explicit RNG states,
fixed diagnostics, collapse curves, float64 original/restored outputs, 100
latency samples, copied transitive reproduction source, an isolated assessor
receipt, and the content manifest. All fourteen protocol checks pass.

## Held-topology result

| representation | overall MSE | action-overlap MSE | downstream-effect MSE |
|---|---:|---:|---:|
| raw rank-32 | 0.105744 | 0.859940 | 0.143833 |
| detached VISReg | 0.146024 | 1.362790 | 0.283530 |
| no-detach VISReg | 0.141472 | 1.425808 | 0.269214 |
| masked autoencoder | 0.140543 | 1.391670 | 0.269572 |
| complete LeJEPA | 0.142355 | 1.428005 | 0.274014 |
| SIGReg-only | 0.142154 | 1.419560 | 0.273962 |
| invariance-only | 0.149393 | 1.364735 | 0.286114 |
| matched PCA | 0.147181 | 1.444091 | 0.274703 |

The candidate increased raw held-topology effect error by 97.13%. It was
5.32% worse than the no-detach falsifier, the best representation control,
and beat that control on five of ten transfer pairs. On selection its effect
MSE was `0.188455`, versus `0.187476` for masked reconstruction and
`0.128783` for raw. Every representation exhausted the frozen ridge set
without a selection-safe choice.

## Mechanism result

| cell and role | fixed shape | fixed scale | projector rank | backbone rank |
|---|---:|---:|---:|---:|
| detached, selection | 0.1114 | 1.0000 | 1.12 | 1.32 |
| detached, transfer | 0.1218 | 1.0000 | 1.12 | 1.27 |
| no-detach, selection | 0.2143 | 0.0282 | 8.41 | 11.82 |
| no-detach, transfer | 0.2621 | 0.0809 | 7.82 | 10.80 |

The detached cell's final training loss decomposed into invariance
`4.55e-10`, scale `0.999998`, shape `0.093045`, and center `8.15e-6`.
The population standard deviations fell into the clamped regime; scale then
had no useful derivative, while normalized shape could remain numerically
good. The no-detach falsifier did not follow that collapse path. Step-zero
regularizer gradients differed by maximum absolute `0.4727`, final network
and projector hashes differed, and public fit-anchor tokens differed by
`6.56`, so the falsifier was behaviorally enforced.

The synthetic source diagnostic passed exactly:

| radius | detached VISReg regularizer gradient | SIGReg gradient | detached shape gradient | no-detach shape gradient |
|---:|---:|---:|---:|---:|
| `1e-2` | 0.089798 | 0.0000662 | 0.090325 | 0.064862 |
| `1e-3` | 0.902723 | 0.00000662 | 0.903253 | 0.648620 |
| `1e-4` | 9.031996 | 0.000000662 | 9.032526 | 6.486201 |

Thus the failure is not an implementation-null result. The preregistered
gradient inequalities hold, but they do not establish that optimizer
trajectories stay outside the clamp's dead-scale regime.

## Safety and edge envelope

All exact-math, source identity, role isolation, schedule, RNG, copied
control, independent objective, mode enforcement, state-probe, causal
inference, bundle, latency, and restoration protocol checks pass. Original
and restored representation/probe outputs agree exactly for every model.

Operational safety still fails:

- candidate selection and transfer overall/action errors exceed every raw
  `1.05×` margin;
- the aggregate transfer state probe is excellent (`0.00412` NRMSE versus
  PCA `0.24019`), but two edge entities that PCA decodes almost exactly miss
  the per-entity `1.15×` bound; and
- candidate projector rank misses the frozen minimum by a wide margin.

Attribution hit@1, no-action specificity, and action sanity are all `1.0`.
The two cells use 116,848 training and 83,760 inference parameters. The
strict inference bundle is 3,217,398 bytes; 100 batch-one CPU samples yield
71.39 ms median and 80.00 ms p95.

## Consequence

VISReg closes the scale/shape regularizer omission and supplies a useful
negative mechanism result: stronger detached radial gradients do not
guarantee a noncollapsed fitted representation when the clamped scale path
can become gradient-dead. The no-detach cell is diagnostically healthier,
but it remains almost `1.87×` raw on held-topology effect error and is not a
promotion candidate.

Retain the implementation and immutable artifact. Proceed to the bounded
JEPA-SCORE edge-feasibility screen; do not carry either VISReg projector or
regularizer into that screen.
