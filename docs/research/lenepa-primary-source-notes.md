# LeNEPA primary-source implementation notes

## Sources

- Chemeris, Jin, and Balestriero,
  [*LeNEPA: No-Augmentation Next-Latent Prediction for Time-Series
  Representation Learning*](https://arxiv.org/abs/2607.00958), v1, 1 July
  2026.
- Authors'
  [official reproduction repository](https://github.com/langotime/lenepa-milets-2026).
- Official
  [`models/NEPA.py`](https://github.com/langotime/lenepa-milets-2026/blob/main/models/NEPA.py),
  [`losses/sigreg.py`](https://github.com/langotime/lenepa-milets-2026/blob/main/losses/sigreg.py),
  and
  [`models/projector_build.py`](https://github.com/langotime/lenepa-milets-2026/blob/main/models/projector_build.py).

## Literal objective

LeNEPA is a single-view, no-augmentation objective over a causal encoder.
The final-layer token at time `t` predicts the input-layer token at `t+1`.
Under SIGReg stabilization, the target is not detached and there is no EMA
teacher.

For the paper's principal PTB-XL and Aionoscope recipe:

- target layer: `0`;
- predictor: none (`pred_depth=0`);
- projector mode: `both`;
- prediction loss: MSE;
- projector output dimension: 64;
- projector hidden dimension: 1,536;
- projector hidden-layer count: one;
- projector hidden normalization/activation: BatchNorm1d then ReLU;
- temporal SIGReg weight: 20;
- temporal SIGReg layers: input layer and final layer (`[0, 8]`);
- other SIGReg sites: disabled;
- SIGReg normalization: disabled;
- EMA encoder: disabled; and
- learning rate `1e-4`, initial weight decay `1e-2`, final weight decay
  `1e-1`.

The official SIGReg implementation uses 17 quadrature knots over `[0, 3]`,
256 random unit projections, and the positive-half Epps-Pulley statistic.
For temporal SIGReg, each selected layer/sample is a view and the time-token
axis supplies the independent samples.

## Telemetry translation

The frozen telemetry port keeps the objective literal while scaling the
backbone for the edge lane:

- each of the 20 telemetry time slices is one causal token;
- owned entity coordinates are projected and summed into that token;
- the causal backbone remains eight layers but uses width 64, four heads,
  and MLP width 256;
- the exact 64/1,536/64 projector, MSE shift, `[0,8]` temporal SIGReg, and
  weight 20 are retained;
- the deployable representation combines the final causal token with the
  final-time entity contributions; and
- the projector is training-only and absent from the deployed bundle.

This is an objective-faithful edge translation, not a claim to reproduce the
paper's ViT-XS architecture, datasets, or frozen-probe benchmarks.

## Decisive controls

1. `unprojected_lenepa`: identical backbone and retained projector, but both
   next-token MSE and temporal SIGReg operate in backbone space.
2. `projected_sigreg_only`: identical projected SIGReg path with next-token
   MSE weight zero.
3. matched entity PCA and raw rank-32 action-conditioned dynamics.

The projector is retained in every neural cell so training and inference
capacity comparisons do not confound the objective.
