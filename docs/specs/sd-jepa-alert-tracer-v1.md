# SD-JEPA alert tracer v1

## Question

Does the canonical SD-JEPA progression/content split create an angular
progression-change signal that localizes held-topology telemetry events better
than scalar latent prediction error and the same-width unsplit controls,
without losing entity-local observable state or violating edge constraints?

This is an open-development, one-seed tracer. Passing authorizes a fixed
multi-seed robustness run only. It does not authorize sealed evaluation or
production paging.

## Frozen role and data

The candidate is an action-blind current-event alert adapter. It does not
replace the raw rank-32 action-conditioned predictive core.

Use the content-addressed roles frozen by the JEPA experiment ladder:

- fitting: 40 pairs from worker topologies one and two;
- selection: 10 disjoint in-distribution pairs;
- calibration: 10 disjoint in-distribution pairs;
- IID evaluation: 20 disjoint in-distribution pairs; and
- transfer evaluation: 10 topology-three pairs.

No evaluation tensor may affect normalization, representation fitting,
checkpoint selection, score calibration, or the alert threshold. Public
inference accepts current histories and the declared graph only.

## Frozen architecture and optimization

Encode each owned entity observation with a shared two-layer MLP plus a
learned entity embedding. Use ownership-masked entity means as the global
scene latent. Preserve all entity tokens for the state probe. Use total latent
width 32, hidden width 64, and a two-layer MLP predictor conditioned on the
last three global latents plus the declared future control/action vector.

Fit three cells with identical architecture and initialization:

| cell | SIGReg domain | triplet domain |
|---|---|---|
| `sd_jepa` | content coordinates 2:32 | progression coordinates 0:2 |
| `lewm_unsplit` | full coordinates 0:32 | none |
| `a2_full` | full coordinates 0:32 | full coordinates 0:32 |

Use deterministic CPU AdamW, seed `15015`, 300 steps, one pair-blocked anchor
per fitting pair per step, initial learning rate `5e-4` with cosine decay,
weight decay `1e-3`, and checkpoints every 50 steps. Keep the source weights
`lambda_S=0.09`, `lambda_T=0.10`, triplet margin `0.2`, temporal radius one,
17 SIGReg knots, and 256 projections. The smaller projection count is the
only objective approximation and is fixed for all cells.

Fit normalization on fitting histories plus fitting future states. At each
step concatenate the 20-step history and ten-step fitting future. Predict all
ten next future embeddings from a three-latent context and the aligned
control/action vector. Apply SIGReg across independent pair-blocked samples at
each of the 30 timesteps. Apply the exact released middle-anchor,
next-timestep-positive, different-trajectory-negative triplet.

Select every cell's checkpoint by its own deterministic selection-role
self-supervised objective. Do not compare loss magnitudes across cells.

## Scores and calibration

For `sd_jepa`, the primary score is the ownership-masked scene angle change
between the final two history steps:

```text
abs(wrap(theta[-1] - theta[-2]))
theta = atan2(z_progression[1], z_progression[0])
```

Report these matched references:

1. `sd_jepa_angle`, the candidate;
2. `sd_jepa_z_mse`, zero-action one-step latent prediction error from the
   candidate model;
3. `lewm_unsplit_angle`, the first-two-coordinate angle change from A0;
4. `lewm_unsplit_z_mse`, A0 prediction error;
5. `a2_full_angle`, the first-two-coordinate angle change from A2-full; and
6. `a2_full_z_mse`, A2-full prediction error.

Fit the existing robust normalized one-step state-change event definition on
fitting controls. Current-event labels are whether the most recent observed
transition exceeds its threshold. Fit an increasing one-dimensional logit
calibrator on the calibration role. Set the alert threshold strictly above
the maximum calibrated score on calibration-control trajectories.

## Diagnostics

On transfer trajectories, report:

- pooled and per-trajectory current-event AUROC;
- within-trajectory ridge `R^2` for normalized episode progress using
  progression coordinates, content coordinates, and A0's first two
  coordinates;
- progression angular span and radius dispersion;
- entity-local current-state ridge-probe NRMSE for candidate content tokens
  and matched rank-30 entity PCA.

The PCA width is 30 so both probes expose the same per-entity content width.

## Gates

All safety gates must pass:

1. all stored values are finite and original/restored outputs agree within
   `1e-6`;
2. all neural cells have identical inference and training capacity;
3. candidate content state NRMSE is no worse than `1.05` times matched PCA;
4. the candidate bundle including calibrator and state probe is at most
   16 MiB, with batch-one CPU latency and peak RSS recorded;
5. future-state/control/action counterfactuals cannot change public outputs
   and forbidden public keywords are rejected; and
6. a fresh stored-array assessor reproduces all metrics and gates.

The progression mechanism lane passes only if:

- transfer pooled AUROC for `sd_jepa_angle` is at least `0.05` above both
  `sd_jepa_z_mse` and `lewm_unsplit_angle`;
- candidate progression progress `R^2` is at least `0.10` above A0's
  first-two-coordinate `R^2`; and
- candidate progression progress `R^2` exceeds candidate content progress
  `R^2`.

The calibrated alert lane passes only if:

- candidate transfer Brier is no greater than `0.95` times the best
  reference Brier;
- transfer control false alarms are at most 5%;
- transfer treatment detection is at least 80%;
- median post-onset detection delay is at most ten transitions; and
- candidate detection exceeds every reference by at least ten percentage
  points.

Advance the mechanism only if every safety gate and at least one value lane
passes. Failure rejects this edge telemetry recipe, not SD-JEPA on the
authors' visual-control benchmarks.

## Artifact contract

The non-overwriting runner writes through a staging directory and publishes
atomically. Preserve protocol, source revision, data identity, all selected
checkpoints, calibrators, event definition, probe payloads, original/restored
outputs, raw evidence arrays, independent assessment, report, reproduction
sources, and SHA-256 manifest. Preserve smoke and failed bundles.

