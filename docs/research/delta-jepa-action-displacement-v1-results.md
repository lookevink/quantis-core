# Delta-JEPA action-displacement tracer v1 result

## Decision

**Reject this edge telemetry Delta-JEPA recipe.**

The learned displacement is compact, state-rich, and strongly predictive of
observable five-step state change. It is not more action-specific than a
capacity-matched endpoint-concatenation decoder, and it does not improve
held-topology downstream-effect prediction. The exact candidate failed both
mechanism gates, every downstream value gate except the selection-only
ordering, and the raw forecast safety bounds.

This rejects the frozen shared-encoder, five-step LDAD telemetry adaptation.
It does not reject Delta-JEPA on the authors' visual-control benchmarks or a
corpus with less state/action confounding.

## Evidence identity

- Implementation commit:
  `80ce4b5`
- Published local artifact:
  `artifacts/action-dynamics/prototype-delta-jepa-v1`
- Retained non-interpretable smoke:
  `artifacts/action-dynamics/prototype-delta-jepa-v1-smoke-1step`
- Artifact-manifest SHA-256:
  `935f5b1c7d21447835f96c3f63ec653dcad23093dcd3eeb85cfddcf11987eeb4`
- Assessment SHA-256:
  `3910d4d6deee353a8d32e24128c8d7517ee909c58207f79fccfd87ec35cbc9ae`
- Independent fresh-process decision:
  `reject_delta_jepa_edge_recipe`

The official run used seed `16016`, 1,600 steps per cell, checkpoints every
200 steps, the frozen pair-atomic roles, and 100 latency repetitions. All
three cells selected step 1,600.

## Mechanism result

The LDAD loss learned a real signal. Its held-selection action loss fell from
`0.005686` at step 200 to `0.002214` at step 1,600. Its held-topology latent
displacement predicted observable state change with `NRMSE=0.052359` and
`Pearson=0.998203`; displacement effective rank was `8.143263`.

But the preregistered comparison shows that this signal was not specifically
created by subtracting endpoints:

| cell | treatment action MSE | treatment sequence retrieval |
|---|---:|---:|
| Delta-JEPA displacement | 0.028554 | 0.154255 |
| endpoint concatenation | 0.015846 | 0.430851 |
| prediction only | 0.243167 | 0.037234 |

The endpoint null's treatment reconstruction error was `44.50%` lower than
Delta-JEPA's, and its retrieval rate was `27.66` percentage points higher.
Delta-JEPA therefore failed the required 10% reconstruction improvement and
10-point retrieval improvement. Its displacement was useful, but endpoint
identity carried substantially more action-disambiguating information.

## Downstream result

All three frozen encoders produced nearly indistinguishable reduced-rank
downstream results:

| representation | selection effect MSE | transfer effect MSE |
|---|---:|---:|
| Delta-JEPA | 0.189565 | 0.274364 |
| endpoint concatenation | 0.189765 | 0.274360 |
| prediction only | 0.189399 | 0.274329 |
| raw rank-32 dynamics | 0.128783 | 0.143833 |

Delta-JEPA was only `0.11%` better than endpoint concatenation on selection,
then was fractionally worse on transfer. It beat the endpoint null on 40% of
held-topology pairs, below the required 60%. More importantly, its transfer
effect MSE was `90.75%` worse than raw.

The same regression appeared in the broader safety metrics:

- transfer overall MSE was `0.149452`, versus raw `0.105744`;
- transfer action-overlap MSE was `1.390836`, versus raw `0.859940`; and
- the selected ridge was `1.0` for every neural representation.

Attribution hit@1, no-action specificity, and action sanity were all perfect
for every neural cell. Those metrics therefore reflect the shared
action-conditioned downstream probe, not a distinct Delta-JEPA advantage.

## State retention and edge feasibility

The candidate was not collapsed:

- current-state probe NRMSE was `0.104116`, versus `0.602463` for matched
  rank-16 entity PCA;
- all cells had exactly `133,844` training parameters and `7,696` deployed
  encoder parameters;
- original/restored representations, decoder outputs, probe outputs, and
  attribution predictions agreed within `1e-6`;
- the model-plus-probe bundle was `6,933,564` bytes;
- batch-one CPU encoding was `0.084321 ms` mean and `0.086604 ms` p95;
- peak process RSS was `7,802,994,688` bytes; and
- public encoding rejected future-state, future-control, and future-action
  keywords.

The rejection is thus not caused by collapse, capacity, serialization, or
edge latency. The likely failure is semantic: five-step state displacement is
dominated by predictable system evolution and action consequences that are
not uniquely identifiable from the difference alone. Endpoint context helps
inverse decoding, while neither objective improves the downstream probe over
the raw state/action model.

## Next target

Proceed to the exact LeWorldModel treatment and bounded latent-geometry
screen. Delta-JEPA shows that adding a specialized inverse objective does not
repair the observable forecast bottleneck. The next falsifier should isolate
whether the exact ambient SIGReg geometry or a subspace-only variant preserves
state and effect information under the otherwise identical world-model
objective.
