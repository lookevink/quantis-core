# Complete multi-view LeJEPA telemetry tracer v1 result

## Decision

**Reject the exact frozen complete multi-view LeJEPA representation recipe.
Do not advance it to multi-seed robustness, sealed confirmation, or an
action-conditioned second stage.**

The representation was finite, restorable, entity ordered, and highly
linearly accessible to current-state probes. It did not preserve the raw
rank-32 predictive safety boundary and did not outperform every matched
representation control. The independently recomputed decision is
`reject_exact_complete_multi_view_lejepa_recipe`.

The exact retained runner is
`lab/action_dynamics/prototype_complete_lejepa.py`; the standalone stored-array
assessor is
`lab/action_dynamics/prototype_complete_lejepa_assessor.py`; and the immutable
local evidence is under
`artifacts/action-dynamics/prototype-complete-lejepa-v1`.

## Evidence boundary

- One deterministic open-development seed; no sealed evidence was consumed.
- Fit and selection used worker topologies one and two. Worker topology three
  was held out.
- Every neural variant trained for 1,600 pair-blocked steps with 40 independent
  matched-pair anchors per step.
- Complete LeJEPA used eight aligned `20 × 7` semantic views, 1,024 fresh
  directions per step, 17 knots over `[0, 3]`, and the pinned official
  implementation semantics from commit
  `c293d291ca87cd4fddee9d3fffe4e914c7272052`.
- Complete LeJEPA, invariance-only, SIGReg-only, and masked-autoencoder
  representations each deployed the same 83,760-parameter backbone.
- The pure assessor reloaded stored representations, predictions, role
  identities, queries, action ablations, and all five ridge candidates. It did
  not invoke a fitted encoder or trust stored metric summaries or gates.

This is neither production-paging evidence nor a general rejection of
predictor-free representation learning.

## Downstream result

No representation had a ridge candidate that met all three raw-selection
safety bounds. The listed ridge is therefore the deterministic diagnostic
fallback and the missing-safe-ridge gate fails.

| Representation | Selected ridge | Selection effect MSE | Transfer overall MSE | Transfer action MSE | Transfer effect MSE |
|---|---:|---:|---:|---:|---:|
| Complete LeJEPA | 1.0 | 0.190176 | 0.142355 | 1.428005 | 0.274014 |
| Invariance-only | 0.01 | 0.188721 | 0.149393 | 1.364735 | 0.286114 |
| SIGReg-only | 1.0 | 0.188041 | 0.142154 | 1.419560 | 0.273962 |
| Masked autoencoder | 1.0 | **0.187476** | **0.140543** | **1.391670** | **0.269572** |
| Matched PCA | 1.0 | 0.189167 | 0.147181 | 1.444091 | 0.274703 |
| Raw rank-32 low-rank | — | 0.128783 | **0.105744** | **0.859940** | **0.143833** |

Complete LeJEPA was 34.6% worse than raw on overall MSE, 66.1% worse on
action-overlap MSE, and 90.5% worse on downstream-effect MSE. It was not best
on selection, was 1.65% worse than the masked-autoencoder transfer control,
and beat that best control on only 50% of held-out pairs rather than the
required 60%.

All five representations achieved 100% action-and-target hit@1, 100%
no-action specificity, and 100% of treatment pairs where the correct action
beat both no-action and whole-pair shuffled-action ablations. Those checks
show that the action inputs were used, but they do not rescue the large
trajectory and effect errors.

## Representation result

| Representation | Transfer state-probe NRMSE | Fixed-view SIGReg | Global-view agreement MSE | Local/global agreement MSE |
|---|---:|---:|---:|---:|
| Complete LeJEPA | 0.015221 | 4.6583 | 0.009676 | 0.010656 |
| Invariance-only | **0.001100** | 16.7386 | **0.000002** | **0.000002** |
| SIGReg-only | 0.012519 | 5.9345 | 0.034833 | 0.826561 |
| Masked autoencoder | 0.010109 | 21.5824 | 0.000344 | 0.024484 |
| Matched PCA | 0.240192 | — | — | — |

Complete LeJEPA easily passed the aggregate `1.05 × PCA` state-probe gate.
It failed the per-entity `1.15 × PCA` gate because width-64 PCA is effectively
lossless for the two owned coordinates on `api_enqueues_queue` and
`queue_dequeues_to_worker`: their PCA NRMSE values were approximately
`1.6e-7`, versus `0.00615` and `0.00548` for complete LeJEPA. This is a real
consequence of the frozen comparator, not a numerical non-finiteness.

Invariance-only produced the strongest current-state accessibility but did
not produce useful future-effect prediction. SIGReg-only improved the fixed
Gaussianity diagnostic while severely weakening local/global view agreement.
Together these results again separate non-collapse and present-state
accessibility from intervention-relevant future information.

## Safety and value gates

The candidate passed:

- pair-blocked anchor and aligned-view schedule validation;
- finite stored evidence and exact public-output restoration;
- aggregate state-probe safety;
- action-and-target hit@1, no-action specificity, and action-ablation sanity;
  and
- recomputation of the selection-only ridge rule.

It failed:

- the per-entity state-probe bound;
- availability of a raw-safe ridge for every representation;
- all three raw overall/action/effect prediction bounds;
- strict selection superiority over all four controls;
- 5% transfer improvement over the best control; and
- the 60% per-pair win threshold.

The full recipe therefore fails both the safety block and the
LeJEPA-specific value block.

## Runtime and edge interpretation

Complete LeJEPA training took 570.16 seconds. The four neural fits took
2,252.65 seconds in total, with peak process memory of 4.56 GB. Batch-one
Python/CPU encoding measured 82.89 ms and its rank-32 probe measured
0.096 ms. The complete-LeJEPA representation and probe JSON artifacts were
2,780,077 and 6,566,863 bytes respectively.

These are local CPU microbenchmarks, not target-device measurements. Since the
scientific value gates failed, no deployment optimization or target-device
benchmark is justified.

## Interpretation

The complete objective corrected the earlier experiment's central limitation:
this was genuine predictor-free multi-view LeJEPA, not SIGReg added to an EMA
residual predictor. That broader change still did not make the representation
competitive with the raw predictive reference or even consistently better
than reconstruction.

The bounded conclusion is:

> The exact pinned complete multi-view LeJEPA telemetry representation did
> not earn a second-stage action model or broader robustness work. Retain the
> implementation and artifact as a reproducible negative result.

A future retrieval-JEPA would be a materially different hypothesis because
its value contract would concern episode/evidence retrieval and abstention,
not direct trajectory prediction. It must be separately preregistered rather
than tuning this rejected recipe.

## Artifact identity

- Artifact manifest SHA-256:
  `00639afaee81cd3844e8b60014ed87f886a8e7a7e0b20bc50834416da1deb265`
- Result SHA-256:
  `adbee2ac569bf13cf1e222e184d6a2d991b744d5788c296d8305541e9f456435`
- Evidence-array SHA-256:
  `5aa894a823b92b7e0273a457d173b5559f379c5a404506f1911a65f13da58bfb`
- Runner SHA-256:
  `8a4f60dbf95c5e07a1ae68e5b0e3f91e84b658b327202fa9680e91aec700725e`
- Standalone assessor SHA-256:
  `edb947fe3a80342a8d811a29307d8f4f78f567912a313d238cbd3ba0e9c565a2`
- Representation module SHA-256:
  `9c59c62d163f67d42df1dbbffd8445c0f258a6635a4095668028e77e5da123fd`
- Frozen contract SHA-256:
  `3db903ffd9d21050153c0bf8b456605489245a40e18012687daa59323c447356`

The bundle contains 42 content-bound evidence files (597 MB total). Its
manifest was independently rehashed and the standalone assessor reproduced
the recorded decision from stored arrays.
