# LeWorldModel bounded geometry screen v1 result

## Decision

**Reject this edge telemetry LeWorldModel geometry recipe.**

No regularized cell was selection-safe relative to raw rank-32 dynamics.
The diagnostic selection winner, SPHERE-JEPA, improved selection
downstream-effect MSE by only `0.15%` relative to prediction-only, then
regressed held-topology effect prediction, state retention, and ordinary
forecast accuracy. Every geometry failed at least one anti-collapse or value
gate.

This rejects the frozen width-32, end-to-end telemetry adaptations. It does
not reject the source methods on their published visual-control or
representation-learning benchmarks.

## Evidence identity

- Implementation commit: `25dbef0`
- Published local artifact:
  `artifacts/action-dynamics/prototype-leworld-geometry-v1`
- Retained non-interpretable smoke:
  `artifacts/action-dynamics/prototype-leworld-geometry-v1-smoke-1step`
- Artifact-manifest SHA-256:
  `8a337131f86e0e9be052883c3af51b295cb0d4dccdea4d837f6e35e5a63bae7d`
- Assessment SHA-256:
  `b3b4756e933e75acddc607f33ecea907416387037375350d567487d572691a4a`
- Independent fresh-process decision:
  `reject_leworld_geometry_edge_recipe`

The official run used seed `17017`, 800 steps per cell, checkpoints every
100 steps, the frozen pair-atomic roles, and 100 latency repetitions. All
seven cells selected step 800.

## Selection and transfer result

No regularized cell had a selected downstream ridge within the frozen 5%
raw safety envelope. The assessor therefore followed the preregistered
diagnostic fallback and selected the regularized cell with lowest selection
downstream-effect error.

| representation | selection effect MSE | transfer effect MSE | effective rank | state NRMSE |
|---|---:|---:|---:|---:|
| SPHERE-JEPA | **0.189288** | 0.282957 | 5.474 | 0.287719 |
| Sub-JEPA | 0.189348 | **0.275037** | 2.066 | 0.060953 |
| KerJEPA | 0.189545 | 0.281361 | **7.802** | 0.062080 |
| ambient LeWorldModel | 0.189711 | 0.275665 | 2.855 | **0.054055** |
| spherical MMD | 0.190334 | 0.279040 | 4.852 | 0.247375 |
| Rectified LpJEPA | 0.193815 | 0.280406 | 4.825 | 0.198747 |
| prediction only | 0.189563 | 0.281442 | 7.814 | 0.072606 |
| raw rank-32 dynamics | 0.128783 | **0.143833** | n/a | n/a |

SPHERE-JEPA beat prediction-only by `0.000276` on selection, then was
`0.54%` worse on transfer effect and won only 40% of held-topology pairs. Its
transfer effect error was `96.73%` worse than raw. Transfer overall MSE was
`0.223687` versus raw `0.105744`; action-overlap MSE was `1.439584` versus
raw `0.859940`.

## Geometry result

The regularizers changed the representation, but none produced the required
combination of rank, state retention, and downstream value.

- Ambient SIGReg and Sub-JEPA collapsed to effective ranks `2.855` and
  `2.066`.
- KerJEPA preserved the broadest regularized geometry at rank `7.802`, almost
  identical to prediction-only's `7.814`, but did not improve its selection
  or transfer effect.
- Both spherical cells enforced unit norm but reduced rank and sharply
  worsened state probes.
- Rectification made `48.9%` of coordinates zero in the one-step smoke and
  remained low-rank and state-poor in the official run.
- Matched entity PCA state NRMSE was `0.428863`; every neural cell beat this
  loose safety baseline, but the chosen spherical cell was nearly four times
  worse than prediction-only.

The geometry lane failed because the selected cell had rank below eight and
worse state NRMSE than prediction-only. The value lane failed all three
gates.

## Shared safety and edge feasibility

Attribution hit@1, no-action specificity, and action sanity were all 100% for
every cell. As in prior tracers, these metrics show that the shared
action-conditioned probe can use declared actions; they do not establish
representation-specific value.

The non-value apparatus behaved correctly:

- every cell had equal training and inference parameter counts;
- original/restored representations, temporal scene histories, probe
  predictions, and attribution predictions agreed within `1e-6`;
- pair-blocking and all selection-only choices independently recomputed;
- public encoders rejected future state, control, and action inputs;
- winner model-plus-probe size was `5,341,542` bytes;
- winner batch-one CPU encoding was `0.156663 ms` mean and `0.158302 ms` p95;
- peak process RSS was `8,024,326,144` bytes; and
- the official artifact and 372 MiB smoke artifact were both retained.

## UR-JEPA prerequisite

**Do not run UR-JEPA.** Ambient and Sub-JEPA did share a low-effective-rank
failure, but neither remained otherwise competitive with raw dynamics.
Their transfer overall, action-overlap, and downstream-effect errors all
missed the frozen 5% raw envelope. The preregistered prerequisite is therefore
false, preventing a post-hoc uniformity sweep.

## Next target

Proceed to Causal-JEPA's materially different whole-entity trajectory
intervention. The geometry screen shows that changing only the marginal
latent prior does not repair held-topology effect prediction. The next
experiment should test whether reconstructing one service trajectory from
the other entities and declared exogenous controls learns interaction
structure that ordinary temporal masking misses.
