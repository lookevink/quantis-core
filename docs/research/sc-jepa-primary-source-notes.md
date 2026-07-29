# SC-JEPA primary-source notes

## Sources and version

These notes freeze the clean-room interpretation used by the Quantis
SC-JEPA interaction tracer.

- Paper: [SC-JEPA: Stabilizing Latent Predictive Learning for Time-Series
  Anomaly Prediction, arXiv:2602.04643v2](https://arxiv.org/abs/2602.04643),
  dated 17 July 2026.
- Author repository, pinned at commit
  [`2ea322a`](https://github.com/Echoo113/SC-JEPA/tree/2ea322af6c60c7125d064b3fe157b7bbd611736c).

The repository does not declare an open-source software license. Quantis
therefore uses a clean-room implementation of the published mechanism rather
than copying repository code.

The paper is the normative source for this tracer. The pinned repository is
illustrative but does not provide a checkpoint-selection or artifact protocol,
uses a one-epoch example configuration, and differs from the paper in some
defaults and architectural details. In particular, its example temperature is
`1.0` rather than the paper's `0.1`. Those discrepancies are recorded here so
that an executable example cannot silently override the frozen hypothesis.

## Mechanism identity

SC-JEPA forms consecutive context/future window pairs. Each window is
instance-normalized and divided into five fine patches. A coarse future view
averages every five normalized time points and treats the resulting sequence
as one patch.

The online branch sees only the fine context. An EMA target encoder sees the
fine and coarse future. A shared soft codebook maps L2-normalized features and
prototypes to temperature-scaled cosine-similarity distributions. The
expected prototype embedding anchors reconstruction.

The fine predictor maps the context code sequence to a future code sequence.
The coarse predictor uses a learned query and cross-attention to predict one
global future code. Training combines:

- fine target-to-prediction KL;
- fine expected-code latent MSE;
- coarse target-to-prediction KL;
- stopped-gradient embedding and commitment losses;
- low per-sample assignment entropy and high batch-marginal entropy; and
- context reconstruction after reversing instance normalization.

The published weights are `1.0` for fine KL, `0.1` for fine latent MSE,
`0.5` for coarse KL, `1.0` for embedding alignment, `0.25` for commitment,
`0.005` for sample entropy, and `0.01` for batch entropy. Reconstruction is
annealed from `0.5` to `0.1`. The EMA decay is `0.996`.

For downstream anomaly prediction, the paper freezes the encoder and
codebook, max-pools code probabilities across variables, flattens the patch
sequence, and trains an MLP to predict whether the next window contains an
anomaly. Because max-pooling nonnegative code probabilities and signed
continuous latents would confound the factorial, Quantis freezes mean pooling
across entities for every cell.

## Edge adaptation

The frozen Quantis cache provides a 20-step context and a 10-step future, not
two 100-step windows. The tracer therefore uses the most recent ten context
steps and all ten future steps. Both are split into five patches of length
two. The coarse target averages the two five-step temporal blocks, producing
one length-two patch.

Each declared telemetry entity is treated as one channel with its owned state
features. The patch encoder is shared across entities; a learned entity
embedding preserves the public entity-token contract. Padded non-owned
features are excluded from reconstruction and state probing.

The edge model uses width and code count `32`, two predictor blocks, four
attention heads, deterministic CPU execution, and no dropout. These are
capacity reductions, not claims of exact benchmark reproduction.

The published downstream role is retained: self-supervised pretraining and
checkpoint selection occur before a frozen-encoder MLP alert head is trained.
The action-blind event label is shared with the HEPA tracer so the alert
comparison changes representation learning rather than event semantics.

## Factorial interpretation

The paper reports separate ablations without the codebook module and without
temporal downsampling, but does not publish the complete two-by-two
interaction. Quantis freezes four cells:

1. continuous bottleneck, fine target only;
2. continuous bottleneck, fine targets plus a separately encoded coarse
   target;
3. soft codebook, fine target only; and
4. soft codebook, fine targets plus a separately encoded coarse target.

All cells instantiate and train the same encoder, fine predictor, global
predictor, decoder, and downstream head. In a single-resolution cell, the
global target is the mean of the EMA fine targets. In a multi-resolution
cell, the same predictor instead targets a separately encoded, temporally
downsampled future view. This changes target provenance without changing
branch capacity. The soft-codebook cells use a `32 × 32` prototype matrix;
continuous cells replace it with a bias-free `32 × 32` linear bottleneck.
This keeps parameter count fixed while changing the intended
predictive-state geometry.

The continuous cells use MSE in the same locations where codebook cells use
distributional KL. This means the codebook factor represents the paper's
complete discretized-prediction mechanism, not prototype lookup in isolation.
The multi-resolution factor changes only the provenance of the global target;
its loss has weight `0.5` in every cell.
