# Discrete-JEPA telemetry tracer v1

## Status and claim boundary

Frozen before implementation and fitting. This is a single-seed,
open-development, paper-faithful telemetry translation, not an exact
reproduction, production alert policy, robustness study, or sealed
confirmation.

The hypothesis is:

> Hard entity-semantic tokens trained by the complete S2P + P2S + P2P
> objective preserve repeatable telemetry regimes and improve held-topology
> action-effect forecasting over continuous semantics and hard P2P-only
> tokenization without regressing raw dynamics.

## Frozen data and roles

Use the content-addressed edge preprocessing artifact
`eb54271132f88c9a431b01e786ea66279a563776434cca2290e47e6b7ae9b3ff`.
Retain the existing worker-topology partition and role boundaries:

- fitting: 40 in-topology pairs;
- selection: 10 in-topology pairs;
- calibration: 10 in-topology pairs;
- IID evaluation: 20 in-topology pairs; and
- transfer evaluation: 10 held-topology pairs.

Tokenizer fitting uses current histories and declared graph ownership only.
Forecast probes use fitting only. Ridge choice uses selection only.
Calibration is retained but does not tune the representation. Evaluation
roles never alter weights, codebooks, masks, ridges, or gates.

## Frozen cells

Train three capacity-identical cells from identical initialization:

1. `discrete_complete`: hard VQ semantics with S2P + P2S + P2P;
2. `continuous_complete`: continuous semantic tokens with S2P + P2S + P2P;
3. `discrete_p2p_only`: hard VQ semantics with P2P only.

The inactive predictors and codebook remain present so architecture and
training capacity match. Also fit width-64 entity PCA and raw rank-32
action-conditioned dynamics on fitting only.

## Frozen architecture and optimization

- 20 history steps become five contiguous four-step patches for each of
  seven declared entities;
- seven learnable entity-aligned semantic tokens;
- width 64, two pre-norm Transformer blocks, four heads, feed-forward width
  128, and zero dropout;
- one shared 64-entry Euclidean codebook;
- straight-through nearest-code context semantics;
- EMA codebook decay 0.99 and commitment weight 0.25;
- EMA target encoder decay 0.996;
- deterministic 40%-60% random patch masking, with at least one visible
  patch per entity;
- one pair-blocked anchor per fitting pair and step;
- 800 steps, AdamW, learning rate `1e-4`, weight decay `1e-2`, 40-step
  linear warmup, cosine decay, gradient clip 1.0; and
- seed 25025 for initialization, anchors, and masks.

S2P, P2S, and P2P use equal unit weights and mean squared latent error.
Targets are stop-gradient outputs of the full-history EMA encoder. The
context and target semantic streams both update the EMA codebook; only the
context stream receives straight-through gradients.

Public `encode(histories, graph)` accepts no future, action, role, outcome,
or target input and returns seven entity-ordered semantic tokens. Discrete
cells return code vectors; the continuous cell returns continuous online
semantic vectors.

## Evaluation

Fit rank-32 action-conditioned probes over ridges
`{1e-4, 1e-3, 1e-2, 1e-1, 1}`. A ridge is selection-safe only when overall
and action-overlap MSE are each within 5% of raw. Among safe ridges choose
the lowest selection downstream-effect MSE, tie-breaking toward smaller
ridge. If none is safe, retain the lowest-effect ridge and explicitly record
selection safety failure.

Report selection, IID, and transfer overall/action/effect MSE, per-pair
transfer effect error, current-state retention, attribution hit@1,
no-action specificity, and absent/shuffled-action sanity.

## Mechanism diagnostics

On every selection and transfer history, independently report:

- hard-code counts, perplexity, and per-entity code usage;
- S2P, P2S, and P2P diagnostic losses under a fixed 50% mask;
- next-window per-entity code accuracy from a fitting-only categorical
  transition table; and
- original/restored semantic tokens and indices.

The discrete mechanism passes only if:

1. candidate perplexity is at least 8 and every varying entity uses at least
   two codes on both selection and transfer;
2. candidate S2P and P2S error are each at least 10% below the corresponding
   untrained heads in `discrete_p2p_only`;
3. candidate P2P error is within 5% of `discrete_p2p_only`; and
4. candidate next-window code accuracy exceeds `discrete_p2p_only` by at
   least five percentage points on both selection and transfer.

## Gates

All safety gates must pass:

1. finite evidence and disjoint pair/trajectory roles;
2. identical independently recomputed neural capacity;
3. model, code index, probe, and deployment replay within `1e-6`;
4. copied public inference signatures are causal;
5. anchor and mask schedules recompute exactly;
6. ridge choice and no-safe-ridge status recompute;
7. candidate selection and transfer overall/action MSE are within 5% of raw;
8. attribution hit@1 is at least 95%, no-action specificity is 100%, and
   correct actions beat absent/shuffled actions for at least 80% of treatment
   pairs;
9. the exact loadable candidate bundle is at most 16 MiB and retains 100 raw
   batch-one CPU latency samples; and
10. an artifact-only assessor recomputes every gate.

The value lane passes only if candidate selection effect MSE is strictly best
among learned controls, transfer effect MSE improves the best learned control
and raw by at least 10%, and candidate wins at least 60% of transfer pairs.

Advance only if all safety, mechanism, and value gates pass. Otherwise reject
this exact telemetry translation, not Discrete-JEPA on synthetic visual
symbolic tasks.

## Artifact contract

Publish atomically without overwrite after verifying the staging bundle.
Retain all cells, target encoders, predictors, codebooks, PCA/raw controls,
deployment bundle, anchor/mask schedules, ridge curves, predictions,
diagnostic tensors, restoration evidence, latency samples, independent
assessment, copied sources, report, manifest, smoke bundles, and failures.
