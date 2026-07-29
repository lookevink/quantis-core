# LeWorldModel geometry-screen primary-source notes

## Sources and pinned revisions

- [LeWorldModel](https://arxiv.org/abs/2603.19312), arXiv v1,
  2026-03-13; official code
  [lucas-maes/le-wm](https://github.com/lucas-maes/le-wm), revision
  `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`.
- [Sub-JEPA](https://arxiv.org/abs/2605.09241), arXiv v1,
  2026-05-10; official code
  [intcomp/Sub-JEPA](https://github.com/intcomp/Sub-JEPA), revision
  `ef945ed434ce529bc7c5f1995f2e1cf173954843`.
- [Rectified LpJEPA](https://arxiv.org/abs/2602.01456), arXiv v2,
  2026-05-28; official code
  [YilunKuang/rectified-lp-jepa](https://github.com/YilunKuang/rectified-lp-jepa),
  revision `5ae61ab49a4ae489c68f2328ba614472163518d6`.
- [KerJEPA](https://arxiv.org/abs/2512.19605), arXiv v1 source archive
  SHA-256
  `0585c24ea061c0d621b7aedaeae454aff431dd715c3e8fc319751452f8355888`.
- [SPHERE-JEPA](https://arxiv.org/abs/2605.26900), arXiv v1 source archive
  SHA-256
  `561a374c8f6657f0d6150a5e204ca87d83e3679139745a3ea341ac6c54e45770`.
- [Expanding SPHERE-JEPA](https://arxiv.org/abs/2606.17603), arXiv v1
  source archive SHA-256
  `f983b29e7e1202d2f0f3df39da39facd1c20a9ffc497b065a5c9f12f2727e67b`.

The revisions and source archives were read as scientific references. Quantis
uses a clean telemetry implementation and does not vendor external code.

## Exact LeWorldModel core

LeWorldModel jointly trains one encoder and an action-conditioned
autoregressive predictor:

```text
z[1:T] = encoder(o[1:T])
zhat[t+1] = predictor(z[t-history+1:t], a[t])
L = MSE(zhat[t+1], z[t+1]) + 0.09 * SIGReg(z[1:T])
```

Both sides use the same trainable encoder. There is no EMA teacher,
stop-gradient, pretrained encoder, reconstruction decoder, reward, or
auxiliary action loss.

The official implementation applies SIGReg independently at every time index
over the sample batch. SIGReg draws normalized Gaussian directions, projects
the embeddings, and integrates the Epps-Pulley statistic with 17 knots on
`[0,3]`. The source defaults to 1,024 projections. Optimization uses AdamW,
learning rate `5e-5`, weight decay `1e-3`, cosine decay, and gradient clipping
at one.

## Geometry variants

The screen changes only the latent support and training-only distribution
regularizer.

### Sub-JEPA

Sub-JEPA replaces ambient SIGReg with the mean SIGReg loss over frozen
row-orthonormal random subspaces. The official default uses 32 subspaces and
sets each dimension to ambient width divided by 32. Quantis uses eight
four-dimensional subspaces for its width-32 edge latent; this is the smallest
non-degenerate integer scaling of the published construction. Projections are
initialized by QR decomposition, frozen, and excluded from parameter counts.

### Rectified LpJEPA

Rectified LpJEPA applies ReLU to learned features and matches them to a
coordinate-wise rectified generalized Gaussian target. For the paper's
`p=2`, `mu=0` case, this is `ReLU(N(0,I))`. RDMReg projects learned and target
samples along shared random directions, sorts each one-dimensional sample,
and averages squared two-sample Wasserstein distances. Quantis transports
this geometry into the LeWorldModel objective with the shared `0.09`
regularizer coefficient and the same projection budget.

### KerJEPA

KerJEPA identifies SIGReg as a sliced Gaussian-kernel MMD and develops direct
Euclidean kernel discrepancies. The bounded screen uses exact Gaussian-prior
MMD with an RBF kernel and analytic Gaussian expectations. With
`k(x,y)=exp(-||x-y||²/(2d))`, both cross-prior and prior-prior terms are
closed form, so no target samples or random projections are needed.

### SPHERE-JEPA and deterministic spherical MMD

SPHERE-JEPA normalizes the representation and predictor to the unit sphere,
then uses sliced distribution matching toward the hyperspherical uniform
distribution. Quantis uses shared random projections and two-sample sliced
Wasserstein distance against seeded uniform-sphere samples.

The follow-up derives deterministic full-dimensional MMD, KSD, and KL
families on the sphere. Its strongest generic classification setting uses
MMD/KSD weight `0.05` and a heat-kernel temperature `5/d`. The bounded screen
tests the MMD member: the uniform-prior cross term is constant by rotational
invariance, leaving the deterministic heat-kernel pair energy over normalized
embeddings.

The screen does not separately test every kernel, temperature, KSD, or KL
variant. Those are estimator/kernel ablations of the same spherical-uniform
hypothesis, and an unbounded sweep would use evaluation data to select a
regularizer.

## Telemetry adaptation

All cells use the same width-32 entity-preserving encoder and history-three
action-conditioned predictor. The global scene latent is the mean of observed
entity tokens. Training predicts all ten fitting-role future steps and applies
the cell's regularizer at each of the 30 history-plus-future positions.

Public encoding returns current entity tokens from histories and the declared
graph only. A shared rank-32 action-conditioned probe evaluates observable
forecast, held-topology downstream effect, attribution, and action sanity.

This adaptation tests whether a published anti-collapse geometry makes the
same telemetry world-model representation more useful. It is not a claim made
by the source authors about operational telemetry.
