# CF-JEPA primary-source notes

## Source identity

- Paper: Jaehoon Lee and Sunghyun Sim, “Crop-based Forward Joint-Embedding
  Predictive Architecture for Time Series Self-Supervised Learning,” arXiv
  `2606.07031v1`, submitted 2026-06-05.
- Paper URL: <https://arxiv.org/abs/2606.07031>
- Official implementation: <https://github.com/WDSLab/CF-JEPA>
- Official implementation revision inspected for this tracer:
  `4968faf731c8c56e89d78d944716e212392eb5a0`.
- The repository declares the implementation MIT licensed.

These notes freeze the source interpretation used by the Quantis tracer. The
implementation is an edge-sized clean-room adaptation, not a claim that the
authors evaluated operational telemetry.

## Mechanism

CF-JEPA replaces masked context construction with random contiguous forward
crops. For a length-`T` series it samples a crop ratio uniformly, chooses a
valid start, and reserves at least three later timesteps. The online encoder
processes the crop. A stop-gradient target encoder, updated by exponential
moving average, processes the complete series.

The portion after the crop is partitioned into three contiguous zones. Three
near-identity linear predictors map the crop representation to short-,
middle-, and long-horizon target-encoder representations. When a zone is
shorter than the crop, the official implementation adaptive-average-pools the
prediction to the zone length. The loss is L1 after unit normalization along
the representation dimension.

The official encoder is a stack of multi-scale dilated depthwise-convolution
blocks. Each block sums kernels `{3, 9, 15}`, then applies batch
normalization, GELU, a pointwise convolution, a second batch normalization,
GELU, and a residual connection. Dilations grow as `2**i`.

Collapse prevention combines:

- VICReg variance and covariance losses on mean-pooled crop
  representations;
- multi-view invariance after adaptive pooling at temporal scales
  `{2, 4, 8}`; and
- a linearly annealed horizon-loss weight from one toward zero.

The EMA momentum follows a cosine schedule from its base value toward one.
The paper configuration uses four crops, representation width 320, hidden
width 256, depth five, AdamW at `2.25e-4`, base EMA `0.983`, and weights
`0.081`, `0.076`, and `1.101` for variance, covariance, and invariance. For
series shorter than 50, crop ratios are raised to `[0.6, 0.8]`.

## Downstream asymmetry

The paper routes the online encoder to sample classification and the EMA
target encoder to forecasting and anomaly detection. Its explanation is
geometric: online representations remain more discriminative, while EMA
representations become temporally smoother and lower rank. The anomaly path
uses per-timestep target representations and either nearest-neighbor or
Gaussian distance from the training distribution.

This is not merely an implementation detail. Quantis therefore treats
online-versus-target alerting as a paired comparison from the same trained
model and requires the claimed smoothness/rank asymmetry to be observable.

## Official ablations

The official repository registers both:

- `forward_linear`: one-zone forward latent prediction; and
- `masked_latent`: 30% full-sequence timestep masking with an MLP predicting
  target-encoder latents at the masked positions.

The masked-latent ablation uses a single full-sequence view and therefore
does not apply the crop multi-scale invariance loss. The one-zone ablation
uses one linear predictor and a fixed prediction-loss weight. Those are
source-faithful ablations, but they are not training-parameter-matched to the
three-zone recipe. All deployed encoders are architecture matched; Quantis
records active training capacity separately and does not disguise that
difference.

## Quantis adaptation boundary

Operational telemetry is represented as one owned multivariate series per
declared graph entity. The encoder weights are shared across entities and an
entity embedding preserves identity. Pretraining concatenates each
20-timestep history with its ten self-supervised future states. The complete
30-timestep sequence is fitting-only; public inference accepts the
20-timestep history and graph.

The architecture is scaled to hidden width 64, representation width 32, and
depth three. Objective weights, crops, momentum schedule, predictor
initialization, temporal alignment, and the three official objectives remain
unchanged. This tests the mechanism at the intended edge budget; it does not
reproduce the paper’s benchmark scale.

The alert adapter follows the paper’s Gaussian anomaly route. A regularized
Gaussian is fit only on fitting-control representations, then an increasing
calibrator and a strict trajectory threshold are fit on the calibration
role. Quantis maps the non-negative Mahalanobis distance through the fixed
monotone transform `d / (1 + d)` because the shared logit calibrator accepts
bounded inputs. A matched entity-local rank-32 PCA passes through the
identical Gaussian/calibration protocol.
