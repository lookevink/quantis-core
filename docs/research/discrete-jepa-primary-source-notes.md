# Discrete-JEPA primary-source notes

## Primary source

Baek et al., [*Discrete JEPA: Learning Discrete Token Representations without
Reconstruction*](https://arxiv.org/abs/2506.14373), ICML Tokenization
Workshop 2025.

No author implementation was publicly discoverable when this tracer was
frozen on 2026-07-29. The paper is therefore the only implementation
authority.

## Mechanism that must survive translation

The paper's tokenizer has an online context encoder, an EMA target encoder,
learnable semantic tokens, continuous patch tokens, and a shared hard
nearest-code semantic VQ bottleneck. It applies three complementary latent
objectives:

1. semantic-to-patch (S2P), predicting masked target patch latents from
   quantized context semantic tokens;
2. patch-to-semantic (P2S), predicting continuous target semantic latents
   from context patch latents; and
3. patch-to-patch (P2P), predicting masked target patch latents from context
   patch latents.

The target encoder is updated by EMA. The codebook uses standard VQ
commitment loss and EMA updates. Continuous patch tokens are training-only;
the discrete semantic codes are the tokenizer output.

For its smaller visual task the paper uses 40%-60% random masking, eight
semantic tokens, 96-dimensional tokens, a 1,024-entry codebook, AdamW-style
cosine training with 5% warmup, and batch size 128. It evaluates the learned
indices with a separate index world model.

## Paper ambiguities

The paper does not publish:

- source code or an implementation commit;
- the three objective weights;
- the commitment coefficient or codebook EMA decay;
- an exact definition of the table's `SVQ` label;
- encoder EMA scheduling; or
- tokenizer training-step count for the Dancing-Sprites experiment.

Calling a telemetry implementation “exact Discrete-JEPA” would therefore be
false. This tracer pins a conventional nearest-neighbor straight-through VQ
with EMA codebook updates, equal S2P/P2S/P2P weights, and all remaining
controls before fitting.

## Telemetry translation

One telemetry “image” is the current 20-step history. Each of seven declared
entities contributes five contiguous four-step patch tokens and one
entity-aligned semantic token. The context encoder receives 40%-60% masked
patches; the EMA target encoder receives the complete same history. Future
states, controls, actions, role labels, and outcomes never enter tokenizer
training.

Seven semantic tokens replace the paper's eight because the public alert
contract requires one stable token per declared entity. Width 64 and 64 codes
bound edge cost and avoid a 1,024-way codebook on only 40 independent fitting
pairs. These are declared deployment adaptations, not paper claims.

The deployed candidate retains only the online encoder, hard codebook, and
selected action-conditioned forecast probe. Target encoder and all three
predictors are training-only.
