# Event-native trace JEPA prototype v1 result

## Verdict

Reject this recipe without a durable implementation or multi-seed run.

The causally compiled trace JEPA was compact, non-collapsed, and perfectly
learned the bounded span schema. It nevertheless detected none of the ten
held-out-topology treatments. Its 80% action-and-target investigation score
was matched by the event-alignment-shuffled null, so it did not demonstrate
case-aligned trace value.

The valid result is:

- `artifacts/action-dynamics/prototype-event-native-trace-jepa-v1/prototype-result.json`
- SHA-256:
  `38aa22cf268003d29dbf1cb59da926ced417d977a33747466e3957399a093b43`
- size: 264,490 bytes

This is single-seed, already-open development evidence. It is not sealed
confirmation or production evidence.

## Frozen recipe

The prototype followed
[`Event-native trace JEPA prototype v1`](../specs/event-native-trace-jepa-prototype-v1.md):

- fit on 6,320 windows and 73,181 completed traces from worker topologies one
  and two;
- six training-fitted span templates and six observed graph entities;
- a 48-dimensional, two-layer masked trace Transformer;
- masked template, entity, and outcome prediction plus time-to-next-span;
- a training-only rank-32 future trace embedding;
- a 64-dimensional action-conditioned temporal latent predictor;
- a metrics-only rank-32 low-rank control;
- a capacity-matched binned-event predictor;
- a complete 6,320-sample event-alignment derangement within topology,
  transition, treatment status, and action family; and
- an event bigram and timing-surprise control.

Hidden-action thresholds used calibration controls only. The primary
evaluation held out worker topology three.

## Input and causal-timing audit

The 180 eligible cases contained:

- 164,764 retained completed traces;
- 992,533 raw spans and 977,569 retained spans;
- 100% trace linkage;
- no incomplete parent references;
- no truncated retained traces;
- no unknown transfer templates or entities; and
- 2,494 traces excluded after the final logical window as drain events.

Metric timestamps in this corpus use a synthetic logical clock, while spans
use wall-clock nanoseconds. The valid compiler therefore reconstructed
wall-clock window ends from the independently recorded
`action.run.boundary` start event and the manifest's declared 250 ms period.
Each trace became visible only at the first window end after its final span.
The request-origin index was not used for placement.

## Representation and pretext result

The masked trace encoder learned the bounded trace grammar:

| Diagnostic | Selection result |
| --- | ---: |
| Masked template accuracy | 100% |
| Masked entity accuracy | 100% |
| Masked outcome accuracy | 100% |
| Standardized next-time MAE | 0.0531 |
| Total masked objective | 0.0110 |

Its mean fitting loss fell from 0.6371 in epoch one to 0.0131 in epoch 12.
The transfer target representation had effective rank 27.53, with every
dimension varying. Its fitting PCA retained 97.75% of standardized variance.
This rules out simple representation collapse.

The frozen event-context probe reconstructed the 33 varying normalized
observable positions with transfer NRMSE 0.7311. That is meaningful state
content, but not enough to establish a predictive-core claim.

## Latent prediction

On selection:

| Model | Overall latent MSE | Action-overlap latent MSE |
| --- | ---: | ---: |
| Event-native trace JEPA | 0.8201 | 2.0019 |
| Binned event | 0.3370 | 2.1589 |
| Alignment-shuffled null | 0.9713 | 2.8530 |

The candidate beat the null on latent loss and slightly beat binned events
during action overlap, but its overall event future was substantially less
predictable than the binned representation. Lower latent loss was not treated
as a value result.

## Held-out-topology alerting

| Model | Control false alarms | Treatment detection | Median delay |
| --- | ---: | ---: | ---: |
| Event-native trace JEPA | 0% | 0% | — |
| Alignment-shuffled null | 0% | 40% | 31 |
| Binned event | 40% | 60% | 18 |
| Event n-gram | 20% | 20% | 16.5 |
| Metrics-only low-rank | 0% | 70% | 16 |

The trace candidate was well calibrated but inert: no control or treatment
trajectory crossed its sequential threshold. It failed detection, delay, and
alignment-null gates.

The same problem was visible in distribution. The candidate detected 15% of
treatments at zero control false alarms with median delay 48, while the
metrics-only control detected 80% at 5% false alarms with median delay 15.5.

## Held-out-topology investigation

| Model | Action + target hit@1 | Exact variant hit@1 | No-action specificity |
| --- | ---: | ---: | ---: |
| Event-native trace JEPA | 80% | 40% | 100% |
| Alignment-shuffled null | 80% | 20% | 90% |
| Binned event | 70% | 30% | 100% |

The candidate's 10-point gain over binned events did not survive the
alignment null. In-distribution action-and-target hit@1 was only 70%, below
the binned control's 90%. The investigation lane therefore failed.

The null result matters: because the derangement preserved topology,
transition, treatment status, and action family, candidate-conditioned
predictions could still exploit action-family marginals. An 80% score shared
with that null is not evidence that the observed trace history and future
were correctly aligned.

## Edge cost

- 86,848 total trace-encoder plus predictor parameters;
- 347,392 serialized tensor bytes; and
- 0.815 ms local CPU batch-one predictor latency.

These are local Apple-arm64/PyTorch 2.5.1 microbenchmarks, not target-runtime
claims. They show that compute was not the limiting issue.

## Interpretation

This corpus has a deliberately bounded six-span request path. Template,
entity, and outcome are almost completely determined by path position, which
made masked reconstruction easy without forcing the representation to retain
incident-discriminating information. Completion-time and gap variation added
some action-family signal, but not enough case-aligned surprise for alerting
or attribution.

The bounded negative conclusion is:

> This masked-path plus deterministic future-latent recipe did not add
> topology-transfer alert or investigation value beyond its controls.

This does not rule out event-native JEPA on heterogeneous production traces,
where path structure, service vocabulary, errors, and partial traces vary
substantially. It does rule out spending durable implementation effort on
this recipe for the current corpus.

## Execution corrections

Three implementation issues were corrected before the valid result:

1. The initial command pointed to the legacy cache root instead of its
   content-addressed topology-transfer child and stopped before loading cases.
2. The first causal parser attempt compared synthetic metric timestamps with
   span wall-clock timestamps, classified all traces as drain, and stopped
   before fitting. The clock-domain correction above was frozen before any
   model result.
3. A PyTorch compatibility call stopped before the first optimization step,
   and a NumPy metric-axis indexing error stopped assessment before any result
   was written. The final command reran the complete deterministic recipe and
   produced the artifact identified above.

No invalid run created a result directory.

