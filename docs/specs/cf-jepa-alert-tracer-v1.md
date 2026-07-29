# CF-JEPA alert tracer v1

## Question

Does mask-free three-zone forward joint-embedding prediction produce a
source-faithful EMA representation that improves held-topology alerting over
the same model’s online representation, the authors’ one-zone and
masked-latent ablations, and a matched raw PCA representation?

This is an open-development, one-seed tracer. Passing authorizes a fixed
multi-seed robustness run only. It does not authorize sealed evaluation or
production paging.

## Frozen data roles

Use the content-addressed edge-dynamics cache and pair-atomic roles frozen by
the [JEPA experiment ladder](jepa-experiment-ladder-v1.md):

- fitting: 40 pairs and 6,320 windows from worker topologies one and two;
- selection: 10 disjoint in-distribution pairs;
- calibration: 10 disjoint in-distribution pairs;
- IID evaluation: 20 disjoint in-distribution pairs; and
- transfer evaluation: 10 topology-three pairs.

No evaluation tensor may affect normalization, encoder fitting, checkpoint
choice, Gaussian fitting, probability calibration, or the alert threshold.
Public inference accepts current histories and the declared graph only.

## Views and encoder

For fitting only, concatenate each 20-step history and ten-step future state
into a length-30 series. Zero unowned entity-feature positions. Fit the
normalizer on fitting tensors only. Each declared entity is encoded as a
separate multivariate series by one shared encoder with a learned entity
embedding.

Use hidden width 64, representation width 32, three multi-scale dilated
depthwise-convolution blocks, kernels `{3, 9, 15}`, dilations `{1, 2, 4}`,
batch normalization, GELU, pointwise convolutions, residual connections, and
a final linear projection.

Use deterministic CPU AdamW, seed `14014`, 300 pretraining steps, batch size
64, four crops per sample, learning rate `2.25e-4`, weight decay `1e-5`,
gradient clipping at one, and checkpoints every 50 steps. Series shorter
than 50 use crop ratios sampled uniformly from `[0.6, 0.8]`. Crop sampling
reserves at least three future positions.

The base EMA momentum is `0.983` and follows the source cosine schedule to
one. Apply source weights `0.081`, `0.076`, and `1.101` to VICReg variance,
VICReg covariance, and multi-scale invariance. Pool sizes are `{2, 4, 8}`.

Select each objective’s checkpoint by its own deterministic selection-role
self-supervised loss. Checkpoint selection may compare steps within one
objective; losses are not compared across objectives.

## Frozen cells

Train three source-faithful objectives:

| Objective | View and prediction |
|---|---|
| `three_zone` | mask-free random crops; three near-identity linear predictors target contiguous short, middle, and long future zones; horizon weight anneals toward zero |
| `one_zone` | mask-free random crops; one near-identity linear predictor targets the complete following portion; fixed prediction weight |
| `masked_latent` | 30% random timestep masks on the full series; a two-layer MLP predicts EMA latents at masked positions; no crop invariance; fixed prediction weight |

The deployed alert representations are:

1. `cf_jepa_target`: target encoder from `three_zone`, the candidate;
2. `cf_jepa_online`: online encoder from the same `three_zone` fit;
3. `one_zone_target`: target encoder from `one_zone`;
4. `masked_latent_target`: target encoder from `masked_latent`; and
5. `matched_pca`: fitting-only entity-local rank-32 raw-history PCA.

The online and target candidate are not separately trained. All neural cells
have the same deployed encoder capacity. Active training capacity is
reported, not required to match, because the official predictor ablations
have different heads.

## Alert adapter

For every representation, mean-pool the encoder’s per-timestep history
tokens per entity. Fit one full-covariance Gaussian with ridge `1e-3` to
fitting-control tokens for each entity. Average the summed entity Mahalanobis
distance over entity and latent dimensions, then apply the fixed monotone map
`d / (1 + d)` to obtain a bounded action-blind anomaly score.

Reuse the HEPA robust normalized one-step state-change event definition,
fitted on fitting controls only. Fit an increasing one-dimensional logit
calibrator on calibration scores and labels. Set the alert threshold strictly
above the maximum calibrated score observed on any calibration-control
trajectory. No learned supervised risk head is used.

## Mechanism diagnostics

On held-topology histories, compute for the paired online and target temporal
representations:

- mean cosine similarity of adjacent timesteps; and
- 90%-variance effective rank after flattening sample, entity, and time.

The claimed asymmetric geometry is present only if target adjacent cosine
similarity is strictly greater than online similarity and target effective
rank is no greater than online rank.

Fit identical entity-local ridge probes on fitting candidate-target tokens
and matched-PCA tokens to predict the current observable entity state. The
candidate’s transfer aggregate NRMSE must be no worse than `1.05` times the
matched-PCA NRMSE.

## Primary estimands and gates

Let `B_*` be transfer calibrated Brier score and `D_*` treatment-trajectory
detection. Lower Brier and higher detection are better.

All safety gates must pass:

1. finite outputs and restoration within `1e-6`;
2. exact deployed neural capacity across neural routes/objectives;
3. candidate state NRMSE no worse than `1.05 ×` matched PCA;
4. the target/online geometry asymmetry above;
5. candidate payload plus Gaussian adapter and state probe no larger than
   16 MiB, with batch-one CPU latency and peak RSS recorded; and
6. an independent stored-array assessor reproduces every metric and gate.

The predictive alert-score lane passes only if:

- `B_cf_jepa_target <= 0.95 × min(B_cf_jepa_online,
  B_one_zone_target, B_masked_latent_target, B_matched_pca)`.

The trajectory alert lane passes only if:

- transfer control false alarms are at most 5%;
- transfer treatment detection is at least 80%;
- median post-onset detection delay is at most ten transitions; and
- candidate detection exceeds each neural ablation by at least ten
  percentage points.

Advance only if every safety gate and at least one value lane passes. Failure
rejects this edge CF-JEPA recipe, not CF-JEPA on the authors’ benchmarks or
forward cropping at other scales.

## Artifact contract

The non-overwriting runner writes through a staging directory and publishes
atomically. Preserve:

- protocol, source revision, data identity, models, Gaussian adapters, event
  definition, and probe payload;
- original and restored representations, scores, calibrated scores, alert
  decisions, labels, geometry tensors, state-probe evidence, and causal-input
  counterfactual evidence;
- independent assessment and report;
- exact reproduction sources; and
- a SHA-256 artifact manifest.

Failed and smoke bundles remain available and are never deleted by the
runner.
