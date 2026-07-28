# Hybrid telemetry JEPA development v1

## Objective

Build the highest-probability nominal representation experiment identified in
`docs/research/jepa-software-telemetry-gap-analysis.md`.

This is a development experiment over the already-open observability corpus.
It must not make a new confirmation or world-model claim.

## Public seams

### Structured events

Fit a training-only vocabulary over OTLP application logs and compile each
capture into deterministic typed events. Prefer `event.name`; otherwise use a
normalized body template. Preserve service identity, severity, trace/span
linkage, event time/delta, and an explicit allowlist of numeric parameters.
Validation-only templates map to an unknown token.

### Hybrid graph tokens

Compile `GraphStateWindows` into model-ready fine and coarse graph tokens.
Tokens retain entity identity, entity kind, relation type, observation masks,
and declared adjacency. A seeded multi-mask sampler produces multiple
contiguous entity/time target masks without mutating source tensors.

### Training and assessment

Train an optional-PyTorch temporal graph JEPA with:

- a shared metric/log projection;
- learned entity, kind, relation/time, and horizon embeddings;
- relational graph message passing and temporal self-attention;
- an EMA target encoder;
- multi-mask L1 latent prediction;
- a local raw-state recovery head;
- variance/covariance anti-collapse regularization; and
- deterministic effective-rank, per-dimension variance, and covariance
  diagnostics.

The trainer accepts training and development `GraphStateWindows`, uses
Apple MPS when available, and falls back to CPU. It serializes configuration,
losses, diagnostics, frozen-probe performance, and model parameters.

## Development comparisons

Report the learned frozen representation against:

- raw one-hop ridge;
- equal-width frozen-PCA one-hop;
- local learned context;
- shuffled topology; and
- an optional soft-regime/codebook ablation.

No development candidate advances unless it beats the existing frozen-PCA
one-hop baseline, preserves recoverable node/edge state, has healthy effective
rank across seeds, and benefits from declared rather than shuffled topology.

## Boundaries

- Fit vocabularies, normalizers, PCA, probes, and codebooks on training only.
- Do not alter or reuse the prior confirmation claim.
- Do not add action-conditioned/world-model language until intervention data
  and multi-step rollout evaluation exist.
- Do not make generic language-model embeddings a required dependency.
- Keep PyTorch in an optional `training` dependency group so ingestion remains
  NumPy-only.
