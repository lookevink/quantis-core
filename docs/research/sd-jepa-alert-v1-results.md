# SD-JEPA alert tracer v1 result

## Decision

**Reject this edge telemetry SD-JEPA recipe.**

Canonical A2 did create a distinct angular event-localization signal, but it
did not create the claimed progression coordinate and did not produce a
usable calibrated alert policy. Every safety gate passed; both value lanes
failed.

This result rejects the frozen two-coordinate, pair-blocked telemetry
adaptation. It does not reject SD-JEPA on the authors' visual-control tasks or
a future task with coherent episodic phase labels.

## Evidence identity

- Implementation commit:
  `8ece9c9e91061db17a0399af0e7be0f15ab1e0b3`
- Official source revision:
  `1cc121065e83220a495808f4c65ef4b0b1915f9f`
- Published local artifact:
  `artifacts/action-dynamics/prototype-sd-jepa-alert-v1`
- Artifact-manifest SHA-256:
  `45bf49091c33553c2f06fcac1c9260762741f833fdd34a5e85376e44e7f6903b`
- Assessment SHA-256:
  `c8a3c71d6f368a1375303b4ac49d5670bd56e9c5c5721983f391fe18f0bb69ab`
- Independent fresh-process decision:
  `reject_sd_jepa_edge_recipe`

The official run used the frozen content-addressed role split, seed `15015`,
300 steps per cell, checkpoints every 50 steps, and 100 latency repetitions.
All cells selected step 300.

## What the mechanism did learn

The triplet was active and separated the canonical progression coordinates
from the controls. Its selection loss fell to `0.052532`, while the A2-full
triplet on the whole latent remained `0.411958`.

On held-topology current-event localization:

| score | pooled AUROC | mean per-trajectory AUROC |
|---|---:|---:|
| SD-JEPA angle | 0.751656 | 0.716894 |
| SD-JEPA z-MSE | 0.705541 | 0.723526 |
| A0 angle | 0.565924 | 0.491508 |
| A0 z-MSE | 0.714140 | 0.714896 |
| A2-full angle | 0.556051 | 0.473082 |
| A2-full z-MSE | 0.764841 | 0.751610 |

The A2 angle therefore improved pooled AUROC by `0.185732` over the A0
angle. The split was load-bearing for the angular readout; simply applying
the triplet to the full latent did not work.

However, A2 angle beat its own z-MSE by only `0.046115`. The frozen mechanism
gate required at least `0.05`, so it missed by `0.003885`. It also lost to
A2-full z-MSE. The IID result told the same qualitative story: A2 angle AUROC
was `0.700414`, versus `0.622732` for its own z-MSE and `0.578227` for A0
angle.

## Why this is not a progression coordinate

The within-trajectory progress diagnostic contradicted the primary claim:

| feature | pooled progress R2 |
|---|---:|
| SD-JEPA progression coordinates | 0.077549 |
| A0 first two coordinates | 0.081910 |
| SD-JEPA content coordinates | 0.205141 |

The designated progression coordinates were `0.004361` worse than the A0
first-two-coordinate null and far below the candidate's content coordinates.
Their mean unwrapped angular span was only `0.598600` radians, with maximum
span `0.945679`; radius coefficient of variation was `0.566783`. This is not
the near-circular, phase-carrying geometry reported on suitable control
episodes.

The corpus explains the mismatch. The triplet sees local temporal adjacency,
but Quantis trajectories contain long normal operation, sparse
interventions, recovery, and matched controls rather than a consistent
start-to-goal task phase. SD-JEPA's own stated failure conditions include
episodes without coherent task progression. The learned angle is a useful
two-coordinate change detector here, not a compass.

## Alerting result

Calibration erased the modest ranking benefit:

| score | transfer Brier | control FPR | treatment detection |
|---|---:|---:|---:|
| SD-JEPA angle | 0.006265 | 0.00 | 0.00 |
| SD-JEPA z-MSE | 0.006292 | 0.00 | 0.00 |
| A0 angle | 0.007140 | 0.30 | 0.30 |
| A0 z-MSE | 0.006295 | 0.00 | 0.00 |
| A2-full angle | 0.007289 | 0.60 | 0.60 |
| A2-full z-MSE | 0.006277 | 0.00 | 0.10 |

Candidate Brier was only `0.18%` better than the best reference, far below
the required 5% relative improvement. The strict calibration-control
threshold produced zero candidate alerts on transfer and only 10% treatment
detection on IID. The representation is therefore not deployable as the
real-world alert adapter.

## Safety and edge feasibility

All safety gates passed:

- original/restored scores, calibrated probabilities, decisions, scene
  tokens, and entity tokens were exact;
- all three cells had `27,136` training parameters and `8,736` deployed
  encoder parameters;
- candidate content state NRMSE was `0.092206`, versus `0.438723` for matched
  rank-30 entity PCA;
- the stored model/calibrator/probe/event payload was `4,165,306` bytes;
- candidate angle latency was `0.167489 ms` mean and `0.178363 ms` p95;
- peak process RSS was `3,626,680,320` bytes; and
- public outputs changed with history, ignored stored forbidden-array
  counterfactuals, and rejected future-state/control/action keywords.

The recipe is exceptionally small and state-rich. Its rejection is about
value and semantics, not runtime feasibility or collapse.

## Next target

Proceed to Delta-JEPA. SD-JEPA shows that unconstrained temporal adjacency can
shape a small coordinate without making it task-progressive. Delta-JEPA asks
a narrower question that matches Quantis's recurring failure directly:
whether the intervening action can be decoded from latent displacement alone.

