# VISReg primary-source notes

## Source and code identity

- Haiyu Wu, Randall Balestriero, and Morgan Levine,
  [*VISReg: Variance-Invariance-Sketching Regularization for JEPA
  training*](https://arxiv.org/abs/2606.02572), arXiv `2606.02572v1`,
  submitted 2026-06-01.
- The inspected
  [arXiv source archive](https://export.arxiv.org/e-print/2606.02572)
  has SHA-256
  `1dbf9d9331ce519f60ae2bcce374f0b7d8aa315e9724f9ab803ee5f5a9e5e0c3`.
- Official code:
  [HaiyuWu/visreg](https://github.com/HaiyuWu/visreg), pinned at
  [`47b1cf4d725b6cbc76dae1394eb46acc2d282fc1`](https://github.com/HaiyuWu/visreg/tree/47b1cf4d725b6cbc76dae1394eb46acc2d282fc1)
  on 2026-07-29.

The repository's README declares the code and weights CC BY-NC 4.0, while
`pyproject.toml` declares MIT and refers to a license file that is absent.
Quantis must honor the explicit non-commercial declaration and use a
clean-room implementation of the published equations rather than copy code.

## Exact regularizer

For projector embeddings `Z` with shape `(V, N, D)`, the released
implementation applies every statistic independently per view over the
independent-sample axis `N`, then averages over views and coordinates.

For each view:

```text
mu       = mean_N(Z)
Z_center = Z - mu
std_j    = clamp_min(||Z_center[:, j]||_2 / sqrt(N), 1e-6)

L_center = mean(mu^2)
L_scale  = mean_j((std_j - 1)^2)

Z_shape  = Z_center / stop_gradient(std)
W        = column_normalize(randn(D, K))
S        = sort_N(Z_shape @ W)
q_i      = Phi^-1(i / (N + 1)), i = 1..N
L_shape  = mean((S - q)^2)

L_reg = lambda_scale * L_scale
      + lambda_shape * L_shape
      + lambda_center * L_center.
```

`std` is the population standard deviation (`unbiased=False`). The same
`D x K` matrix of unit Gaussian directions is shared across all views in one
call and resampled on the next call. The detached denominator is the defining
scale/shape decoupling; gradients from `L_shape` still flow through centered
embeddings. The Gaussian targets use `i/(N+1)`, not endpoints, and the latest
official implementation computes and caches them in float32. See
[paper §3.1](https://arxiv.org/html/2606.02572#S3.SS1) and the pinned
[`VISReg` implementation](https://github.com/HaiyuWu/visreg/blob/47b1cf4d725b6cbc76dae1394eb46acc2d282fc1/visreg/losses/visreg.py).

The official full multi-view objective uses the configurable leading
`n_global` views to define

```text
global_mean_n = mean_global_view(Z[:n_global, n])
L_pred        = mean_(view,n,d) (z_view,n,d - global_mean_n,d)^2
L_VISReg    = (1 - lambda) * L_pred + lambda * L_reg,
```

where `i` includes global and local views and nothing is detached in
`L_pred`. Default component weights are all one. This is a predictor-free,
EMA-free representation objective; the projector and regularizer are
training-only. The source main/default image recipes use four global views;
the Quantis translation separately freezes `n_global=2` to preserve the
complete-LeJEPA telemetry view contract.

## Released architecture and optimization boundary

The paper's main ImageNet-1K runs use four global and six local DINO-style
crops, a timm ViT backbone, and a three-layer projector with two width-2,048
hidden layers, BatchNorm, and GELU. AdamW uses weight decay `5e-2`, bfloat16
mixed precision, five warmup epochs, cosine decay to `lr_max/1000`, and global
gradient clipping at one.

| Recipe | Backbone | Epochs | LR | Overall `lambda` | Projector `D` | Slices/GPU | Effective batch |
|---|---|---:|---:|---:|---:|---:|---:|
| VISReg-B | ViT-B/16 | 400 | `9e-4` | 0.9 | 256 | 2,048 | 512 on 32 GPUs |
| VISReg-L | ViT-L/14 | 400 | `8e-4` | 0.7 | 384 | 4,096 | 512 on 32 GPUs |

Global crops are `224x224` at scale `[0.3, 1.0]`; local crops are `96x96`
for ViT-B or `98x98` for ViT-L at scale `[0.05, 0.3]`. Augmentations include
horizontal flip, color jitter at probability `0.8`, grayscale at `0.2`, blur
at `0.5`, and solarization at `0.2`. Full settings are in
[Appendix A.1](https://arxiv.org/html/2606.02572#A1.SS1).

The current official code uses Python 3.12, PyTorch `>=2.8,<2.9`,
HuggingFace Accelerate, Hydra, timm, torchvision, and optional Kornia. Its
default ViT-B config differs from the final paper recipe: it specifies 100
epochs, 4,096 slices, and only 16 samples per GPU without pinning world size.
The paper's exact run therefore requires overrides not captured by one
checked-in configuration.

The released trainer:

- seeds each rank with `process_index`;
- draws projection directions from the rank-local global Torch RNG;
- all-gathers projector embeddings before regularization;
- lets each rank evaluate the gathered batch with independently drawn slices;
- averages those rank-local gradients through distributed training;
- excludes biases and one-dimensional normalization parameters from weight
  decay;
- uses AdamW defaults for otherwise unspecified optimizer parameters;
- trains an online probe on detached backbone features, so probe gradients do
  not reach the encoder; and
- does not save or restore Python, Torch, CUDA, sampler, worker, or VISReg RNG
  state in checkpoints.

Consequently, the current resume path is not an exact stochastic replay.
Quantis must use an explicit serialized generator rather than inherit that
limitation.

## Paper-versus-code differences

1. The current implementation's float32 Gaussian-quantile target and
   `clamp_min(1e-6)` were added at commit
   [`06f65b1`](https://github.com/HaiyuWu/visreg/commit/06f65b11617e5190800be813994c4645b4fd2a20)
   after arXiv v1 to fix AMP precision and numerical stability. Earlier code
   cast the target to the embedding dtype and added `1e-6` to every standard
   deviation.
2. Released checkpoints predate that July fix, and the repository does not
   say they were retrained. Paper metrics and current-code metrics therefore
   cannot be assumed bitwise or scientifically identical.
3. Appendix A says the projector is applied to concatenated CLS tokens from
   the final two backbone layers. Current code applies the projector only to
   the final CLS token; the two-layer concatenation feeds the detached online
   probe.
4. Paper ablations use two global plus six local views, main runs use four
   plus six, and the checked-in default uses four plus six.
5. Algorithm 1 shows `K=64` as an illustrative default, the Python class
   constructor defaults to 256, the checked-in loss config uses 4,096, and
   paper main runs use 2,048 or 4,096.
6. The paper writes an epsilon in the shape denominator but does not specify
   it; current code defines the effective floor as `1e-6`.
7. The paper does not freeze random seeds, direction generators, direction
   sharing across devices, sorting determinism, or checkpoint selection.

These differences prevent an exact reproduction of the reported image
training. They do **not** prevent an exact, version-pinned test of the current
VISReg regularizer within Quantis.

## Published diagnostics and limits

The source mechanism claim is that the explicit scale term preserves a strong
corrective gradient near collapse, while detached standardization lets SWD
control shape separately. The authors diagnose:

- gradient norm as an embedding is radially collapsed;
- scale, shape, and center knockout performance;
- detached versus non-detached shape normalization;
- effective performance versus `K`, `D`, batch size, and overall `lambda`;
- loss/online-accuracy correlation; and
- downstream frozen-feature linear probes and transfer.

The paper reports that scale and shape knockouts fail on Imagenette, omitting
center costs only 0.41 accuracy points, and detachment improves each of three
tested datasets. It recommends `lambda=0.6` as a starting point for small
datasets and `0.9` for ImageNet-1K. It also finds OT-based variants usable at
far fewer slices than the largest training recipe.

These are image-representation results. They do not establish action
sensitivity, forecast value, alert calibration, or edge inference value.
VISReg adds no deployed operation if its projector is discarded.

## Bounded Quantis telemetry translation

This tracer should be a direct regularizer substitution in the already-frozen
complete multi-view LeJEPA representation experiment.

### Frozen candidate

- Reuse its 40-pair independent optimizer batch, anchor schedule, two global
  plus six local semantic-preserving telemetry views, width-64 two-block
  entity Transformer, and `64 -> 256 -> 64` training projector.
- Preserve `Z` as `(V=8, N=40, D=64)`. Views never increase `N`.
- Use the latest official numerical definition: population standard
  deviation, `clamp_min(1e-6)`, detached scale in the shape path, float32
  `i/(N+1)` Gaussian quantiles, and equal component weights.
- Freeze overall `lambda=0.6`, the paper's small-dataset starting value.
- Freeze `K=256` fresh unit-Gaussian directions per step. This is four times
  projector width, matches the official class default, and is a declared
  edge-training adaptation from the 2,048/4,096-slice image runs.
- Draw directions from a dedicated CPU Torch generator, shared across all
  eight views, record its seed and step counter, and serialize exact direction
  matrices or enough generator state for replay. Do not consume global RNG.
- Use deterministic CPU float32, 1,600 final-state-only AdamW steps, learning
  rate `5e-4`, weight decay `5e-2`, 80-step warmup, cosine decay to `5e-7`,
  and no gradient clipping. This reuses the complete-LeJEPA control schedule
  so the primary comparison changes only the regularizer; it is a declared
  adaptation from the source image trainer's clip-at-one schedule.
- Deploy only the complete-history seven width-64 entity tokens and the
  selected frozen action-conditioned probe. VISReg, projector, masks, and
  slice RNG are training-only.

### Controls and falsifiers

The primary control is the retained complete LeJEPA cell with exact SIGReg.
Also retain its invariance-only, SIGReg-only, masked-autoencoder, matched-PCA,
and raw low-rank references.
Reuse the same fit-only state probes and selection-only rank-32
action-conditioned probe family.

Add one equal-capacity source falsifier:

- **VISReg without detach:** remove only stop-gradient and use the same
  differentiable standard deviation after `clamp_min(1e-6)`, while keeping
  every other tensor, direction, seed, step, and parameter identical.

This isolates the published scale/shape decoupling without multiplying the
experiment into an unfrozen component sweep.

### Required evidence

Record and independently reassess:

- literal-reference equality for center, scale, shape, invariance, and total
  losses;
- projection directions, norms, RNG advancement, Gaussian targets, streaming
  reconstruction of sorted projections, and per-step component losses;
- fixed-direction post-fit SWD, per-dimension means/standard deviations,
  covariance, effective rank, and per-entity variance;
- a frozen radial-collapse curve comparing VISReg, no-detach VISReg, and
  SIGReg gradient norms at positive radii down to `1e-4`;
- public token and probe restoration on every selection and transfer sample;
- state-probe safety, raw/PCA forecast safety, downstream-effect error,
  correct-action attribution, pair wins, capacity, bundle size, and latency;
  and
- source, role, code, configuration, generator, and artifact identities.

VISReg earns a mechanism only if it remains non-collapsed, preserves a larger
finite corrective gradient than SIGReg near collapse, and beats the no-detach
cell on the frozen mechanism diagnostics. Promotion still requires the shared
operational gates: selection safety and at least five-percent held-topology
downstream-effect improvement over every representation control. Better SWD,
variance, rank, or training loss without raw/PCA operational value rejects
this exact translation while retaining its code and immutable artifact.
