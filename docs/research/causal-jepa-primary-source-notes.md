# Causal-JEPA primary-source notes

## Sources and identity

- [Causal-JEPA](https://arxiv.org/abs/2602.11389), arXiv source archive
  SHA-256
  `74e3ae2337e82aa8f2f55d0f0f853f0eef9667c10297565cd5190f6242fe6e8a`.
- Official code:
  [galilai-group/cjepa](https://github.com/galilai-group/cjepa), revision
  `412337d0210bf98cee2ca90c3586ab2ea7ca519e`.

The paper source and repository were read as scientific references. Quantis
uses a clean telemetry implementation and does not vendor external code.

## Material mechanism

Causal-JEPA freezes an object-centric encoder and trains a bidirectional
masked predictor. Selected object slots retain their earliest history state
as an identity anchor, while their remaining history trajectory is replaced
with a learned query constructed from the anchor and time embedding. All
future object slots are also queries.

The predictor jointly:

1. reconstructs the wholly masked object histories; and
2. predicts every future object latent from the partially observed history
   and optional auxiliary variables.

The loss is the sum of mean squared latent error over masked history slots
and mean squared latent error over all future slots. At inference, history is
fully visible and only future tokens remain masked.

The authors explicitly describe this as an intervention on predictor
observability, not a do-intervention on the data-generating system and not
proof of causal identification. Their "influence neighborhoods" are
predictively sufficient sets under masking.

## Architecture and training

The released predictor uses full bidirectional attention, a learned mask
token, temporal positional embeddings, a linear identity-anchor projector,
and an output projection. The paper uses six Transformer layers, 16 heads,
width-128 object slots, and MLP width 2,048.

For CLEVRER, the paper uses seven object slots, six history frames, and ten
future frames. The paper reports masking between zero and four objects; the
released default configuration and predictor use two masked slots. Adam uses
learning rate `5e-4`. Object representations are pre-extracted from a frozen
encoder.

Actions and proprioceptive inputs are auxiliary variables. The paper reports
better performance when auxiliary variables are modeled as separate entities
than when concatenated into object slots.

## Frozen telemetry adaptation

The Quantis corpus already has seven declared graph entities, a six-step
history suffix, and ten future steps, so it uses the CLEVRER temporal layout.
A fit-only frozen orthonormal state projection replaces the pretrained visual
slot encoder. The predictor is scaled to width 32, two layers, four heads,
and MLP width 128 for the edge screen.

One separate condition token per future step contains declared controls and
all entity actions. The candidate masks two complete entity trajectories
after their earliest identity anchors. A coordinate-time masking null hides
the same number of history tokens without preserving an entity-level
intervention, and an unmasked future-prediction null omits history recovery.
All cells have identical trainable capacity.

The adaptation tests the paper's object-level observability intervention. It
does not claim that declared telemetry entities are causally identified
objects or that masked predictive influence is causal attribution.
