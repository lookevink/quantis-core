# VISReg telemetry tracer v1

## Status and question

This is the preregistered contract for a single-seed open-development
representation tracer. It asks:

> Does the exact current VISReg scale/shape/center regularizer, substituted
> into the complete multi-view LeJEPA telemetry stack, preserve a
> noncollapsed edge representation and improve held-topology
> action-conditioned effect prediction over exact SIGReg, reconstruction,
> invariance-only, PCA, and raw controls?

The candidate is a representation candidate, not an alert policy, predictive
core, or world model. Passing authorizes fixed-seed robustness only. Every
runner, assessor, test, source snapshot, and artifact remains retained after
either outcome; published result directories are immutable.

## Source identity and evidence boundary

The paper is Wu, Balestriero, and Levine,
[*VISReg: Variance-Invariance-Sketching Regularization for JEPA
training*](https://arxiv.org/abs/2606.02572), arXiv `2606.02572v1`.
The inspected arXiv source archive has SHA-256
`1dbf9d9331ce519f60ae2bcce374f0b7d8aa315e9724f9ab803ee5f5a9e5e0c3`.

The numerical definition is pinned to the official repository at commit
[`47b1cf4d725b6cbc76dae1394eb46acc2d282fc1`](https://github.com/HaiyuWu/visreg/tree/47b1cf4d725b6cbc76dae1394eb46acc2d282fc1),
including the post-paper float32 Gaussian-quantile and `clamp_min(1e-6)`
fix. The repository README declares CC BY-NC 4.0 while package metadata
declares MIT without the referenced license file. Quantis therefore uses a
clean-room implementation of the published equations and does not vendor or
copy author code.

The current regularizer is exact and testable. The paper's reported image
recipe is not claimed reproducible because checked-in configurations,
projector-layer use, slice counts, checkpoints, precision fix, and stochastic
resume state differ or are incomplete. The complete boundary is recorded in
[VISReg primary-source notes](../research/visreg-primary-source-notes.md).

The frozen source data is the same content-addressed action-dynamics cache
used by complete LeJEPA and PEIRA:

- cache:
  `artifacts/action-dynamics/edge-preprocessing-v1/eb54271132f88c9a431b01e786ea66279a563776434cca2290e47e6b7ae9b3ff`;
- source-corpus SHA-256:
  `df03af282f48591216251f22f934b97e1555147df9ed6f6c6d1f7684f9644a26`;
- source artifact-manifest SHA-256:
  `d02afa33a5977cba69b255cd1a5f31470751b6681da1ecae31b11e86307e65b1`;
- preprocessing protocol:
  `action_conditioned_jepa_topology_transfer_v1`;
- fitting and selection use worker topologies one and two;
- evaluation holds out worker topology three;
- whole matched pairs remain atomic and disjoint across roles;
- the fixed roles contain 40 fit, 10 selection, 10 calibration, 20 IID
  evaluation, and 10 held-topology evaluation pairs; and
- no evaluation result may select architecture, objective weight,
  directions, ridge set, checkpoint, diagnostic, or threshold.

The runner and independent assessor require exact equality to all four source
identities above; any mismatch makes a run non-interpretable and cannot
publish to the frozen output.

## Independent batch, views, encoder, and projector

Reuse the complete LeJEPA pair-blocked schedule. Every optimizer step selects
exactly one context anchor from each of the 40 fit pairs. Arm and transition
choices are action-blind, balanced, deterministic, and recorded. The
independent VISReg sample axis is `N=40`; views never count as independent
samples.

Reuse the eight identity-preserving telemetry views:

1. two global views over the complete and most recent 16 context points,
   each with a seeded 10% owned-token mask; and
2. six local views over the most recent ten points and a connected
   three-entity block rooted at each fit-varying observed entity.

Views retain absolute time, entity, kind, relation, graph distance, and
owned-feature identity. They contain no current/future action truth, future
state, future controls, numeric jitter, synthetic values, or cross-trajectory
pairing.

The backbone is the frozen width-64, two-block, four-head, graph-biased
pre-norm entity Transformer with feed-forward width 128, GELU, zero dropout,
and the complete LeJEPA tokenization. A masked mean of visible backbone
tokens feeds the training-only projector:

```text
Linear(64, 256) -> GELU -> Linear(256, 64)
```

Projector outputs are not normalized. Public inference encodes the complete,
unaugmented 20-point history as seven entity-ordered width-64 tokens. The
projector, VISReg, masks, and direction generator are discarded.

## Exact clean-room VISReg objective

Let projector embeddings be `Z` with shape `(V=8, N=40, D=64)`. For each
view independently:

```text
mu       = mean_N(Z)
Z_center = Z - mu
std      = clamp_min(||Z_center||_2 / sqrt(N), 1e-6)

L_center = mean(mu^2)
L_scale  = mean((std - 1)^2)

Z_shape  = Z_center / stop_gradient(std)
W        = column_normalize(randn(D, K))
S        = sort_N(Z_shape @ W)
q_i      = sqrt(2) * erfinv(2 * i/(N+1) - 1), i=1..N
L_shape  = mean((S - q)^2)

L_reg = L_scale + L_shape + L_center
```

`std` is the population standard deviation. `q` is constructed and cached in
float32. `W` is float32, has `K=256` unit-norm columns, is shared across all
eight views in one step, and is freshly generated on the next step from an
explicit CPU generator. Initialize one `torch.Generator(device="cpu")` with
seed 3509 and consume exactly one contiguous float32
`randn(64, 256, generator=...)` draw followed by column normalization per
step. Reset an identical stream for the second cell, then retain both final
generator states and the 1,600-draw counters. VISReg does not consume ambient
NumPy or Torch RNG state.

Let `g_n` be the mean of the first two global embeddings. The undetached
multi-view invariance loss is:

```text
L_pred = mean_(v,n,d) (Z_vnd - g_nd)^2
```

The candidate optimizes the paper's small-dataset starting weight:

```text
L_candidate = 0.4 * L_pred + 0.6 * L_reg
```

All eight views receive gradients. There is no EMA teacher, learned
predictor, negative pair, reconstruction term, whitening, covariance loss,
or downstream-probe gradient in representation training.

## Cells and retained controls

Train two equal-capacity cells from identical backbone/projector
initialization, anchors, views, direction matrices, and optimizer schedule:

1. **Detached VISReg candidate:** exact objective above.
2. **No-detach VISReg falsifier:** replace only
   `stop_gradient(clamp_min(std, 1e-6))` with the differentiable clamped
   standard deviation.

Forward regularizer values match at identical embeddings; only gradient
semantics differ. The no-detach cell removes only `stop_gradient` and uses
the same clamped `std` tensor. The cells must diverge behaviorally after
fitting while retaining identical training and inference capacity.

Before the first optimizer update, retain and independently recompute the
two regularizer gradients with respect to the shared step-zero projector
embedding tensor and direction matrix. Their float32 tensor SHA-256 values
must differ and their maximum absolute difference must exceed `1e-7`.
After fitting, candidate and no-detach network/projector tensor hashes must
differ, and complete public tokens for the fit role's pair-blocked
step-1599 anchors must differ by more than `1e-6` in maximum absolute value.
These are mode-enforcement checks, not value claims.

Copy and identity-verify these models from
`artifacts/action-dynamics/prototype-complete-lejepa-v1`, whose frozen
manifest SHA-256 is
`00639afaee81cd3844e8b60014ed87f886a8e7a7e0b20bc50834416da1deb265`:

- complete LeJEPA with exact SIGReg;
- invariance-only;
- SIGReg-only; and
- the matched masked autoencoder.

Also refit matched entity PCA on fit only and the frozen raw rank-32
contractive low-rank dynamics. No retained control may be modified.

## Frozen fitting schedule

Both VISReg cells receive exactly 1,600 optimizer steps.

- Optimizer: AdamW with default beta/epsilon values.
- Initial learning rate: `5e-4`.
- Weight decay: `5e-2` in one parameter group over every backbone and
  projector parameter. This deliberately preserves the complete-LeJEPA
  comparison stack rather than the source image trainer's bias and
  one-dimensional-normalization exclusions.
- Warmup: 80 linear steps.
- Remaining schedule: cosine decay to `5e-7`.
- No gradient clipping. Record the finite gradient norm at every step. This
  deliberately preserves the retained complete-LeJEPA optimizer schedule so
  the primary comparison changes only the regularizer; it deviates from the
  source image trainer's clip-at-one schedule.
- Runtime: deterministic CPU float32.
- No early stopping or checkpoint selection.
- Only the final state is eligible for assessment.

Freeze these seeds, matching complete LeJEPA where applicable:

| Purpose | Seed |
|---|---:|
| Backbone/projector initialization | 509 |
| Pair/anchor schedule | 1509 |
| Telemetry views | 2509 |
| Training directions | 3509 |
| Fixed post-fit diagnostic directions | 6509 |
| Collapse-curve directions | 7509 |
| Collapse-curve base embeddings | 8509 |

Serialize the exact direction schedule, initial/final generator states, and
draw counters needed to reconstruct every matrix. Seed 6509 initializes one
fresh CPU generator and one contiguous `(64, 1024)` fixed-diagnostic draw.
Seed 7509 likewise initializes one generator and one contiguous `(64, 256)`
collapse-curve draw. Both fitted cells and diagnostics must leave global
Torch RNG unchanged.

## Downstream instruments

Use the fit-only observable-state probe and action-conditioned rank-32
reduced-rank probe from complete LeJEPA. Choose each representation's ridge
from `{1e-4, 1e-3, 1e-2, 1e-1, 1}` using selection downstream-effect MSE,
subject to selection overall and action-overlap MSE each remaining within
`1.05` times raw. If no ridge is safe, record that fact and choose the
minimum-effect ridge only for complete reporting.

Report selection, IID, and held-topology overall/action/effect MSE, current
state retention, transfer per-pair effect error, action-and-target hit@1,
no-action specificity, and absent/shuffled-action sanity. Calibration is
retained only to prove role isolation; this tracer has no alert-threshold
lane.

## Mechanism diagnostics

Independently recompute:

- literal center, scale, shape, invariance, and total loss on small and
  retained tensors;
- every direction norm, seed/step identity, Gaussian target, sorted
  projection, component loss, learning rate, and clip decision;
- fixed `K=1,024` post-fit shape loss using seed 6509;
- per-dimension means and population standard deviations, covariance,
  projector/backbone effective rank, and per-entity backbone variance on
  selection and transfer; and
- a synthetic radial-collapse curve at radii
  `{1, 1e-1, 1e-2, 1e-3, 1e-4}` using shared `(8, 40, 64)` base embeddings
  and `K=256` directions.

For selection and transfer projector diagnostics, instantiate the same
pair-blocked anchor schedule with seed 1509 on that role and take
`batch(step=1599)`, yielding one independent anchor per role pair. Generate
all eight semantic views with the frozen step-1599 masks and feed their
`(8, N_role_pairs, 64)` projector outputs to the fixed-direction shape loss,
per-dimension center/scale statistics, covariance, and projector effective
rank. Backbone effective rank and per-entity variance instead use complete,
unaugmented public tokens for every window in the role. No diagnostic result
selects a training or probe choice.

The fixed shape scalar is the official mean over views, samples, and
directions. Retain projector means and population standard deviations as
`(8, 64)` arrays and population covariance as eight separate `(64, 64)`
matrices; do not flatten views for these statistics. Projector effective rank
is the exponential entropy of the normalized nonzero singular values of the
uncentered matrix `Z.reshape(8 * N_role_pairs, 64)`. Backbone effective rank
uses the same singular-value entropy on uncentered complete tokens reshaped
to `(role_windows, 7 * 64)`. Per-entity variance is the mean feature variance
of each complete public token over role windows.

Construct the collapse base exactly as the pinned official analysis:
draw CPU float32 `randn(8, 40, 64)` with seed 8509, then divide each
`(view, sample)` vector by its last-axis L2 norm plus `1e-12`. At radius
`r`, create a fresh leaf `Z_r = r * Z_base`. Generate one float32
column-normalized `(64, 256)` direction matrix with seed 7509 and reuse it at
every radius for detached VISReg, no-detach VISReg, and SIGReg. The SIGReg
comparison uses the complete-LeJEPA 17-knot `[0, 3]` symmetric-trapezoid
quadrature and the same directions. Reduce every embedding gradient as the
mean L2 norm over the flattened 320 `(view, sample)` rows:

```text
mean(norm(gradient.reshape(-1, 64), dim=1))
```

For the collapse curve, retain regularizer-only embedding-gradient norms for
detached VISReg and exact SIGReg, plus shape-only gradient norms for detached
and no-detach VISReg. Inputs, directions, projection count, and reduction are
identical; overall training weights and invariance do not confound this
source-mechanism diagnostic.

The VISReg mechanism passes only if:

1. all exact-math, RNG, schedule, and numerical checks pass;
2. candidate projector effective rank is at least eight and every
   fit-varying entity has positive token variance on selection and transfer;
3. detached VISReg has a finite, strictly larger regularizer gradient than
   SIGReg at each radius at or below `1e-2`; and
4. detached shape-only gradient is finite and strictly larger than the
   no-detach shape-only gradient at each radius at or below `1e-2`.

Better shape loss, variance, or rank without these causal gradient
differences does not establish the source mechanism.

## Safety and value gates

All safety gates must pass:

1. finite evidence and exact disjoint 40/10/10/20/10 role cardinalities;
2. pair-blocked anchors, views, directions, optimizer, and RNG schedules
   independently reconstruct;
3. the step-zero gradient identity and final behavioral-divergence checks
   above pass, while candidate and no-detach cells have identical
   independently recomputed training and inference capacity;
4. full float64-retained original/restored public tokens, probes, diagnostics,
   copied controls, PCA, and deployed bundle agree within `1e-6`;
5. public inference signatures are causal and the exact deployment envelope
   excludes projector, VISReg, masks, directions, optimizer, and training
   evidence;
6. selection-only ridge choice and no-safe-ridge status reconstruct;
7. candidate transfer aggregate observable-state probe NRMSE is at most
   `1.05` times matched PCA, and no fit-varying observed entity exceeds
   `1.15` times its matched-PCA NRMSE;
8. candidate selection and transfer overall/action MSE are each within
   `1.05` times raw;
9. action-and-target hit@1 is at least 95%, no-action specificity is 100%,
   and correct action beats absent and shuffled actions on at least 80% of
   transfer treatment pairs;
10. the exact candidate inference bundle is at most 16 MiB and retains 100
   raw batch-one CPU latency samples; and
11. an isolated `-I -S` copied-source assessor reconstructs every gate and
    canonically agrees before atomic publication.

All value gates must pass:

1. candidate selection effect MSE is strictly best among every learned
   representation and PCA;
2. candidate transfer effect MSE improves the best representation control and
   raw by at least 5%; and
3. candidate beats the best transfer representation control on at least 60%
   of matched pairs.

Advance only if every safety, mechanism, and value gate passes. Otherwise
reject this exact telemetry translation, not VISReg's published image
results.

## Artifact and test contract

Publish atomically to
`artifacts/action-dynamics/prototype-visreg-v1` without overwrite. Retain both
VISReg cells, copied prior controls and their manifest, PCA and raw models,
all probes, the exact strict inference bundle, role identities, schedules,
per-step projector embeddings/directions/component losses, and scalar plus
hash receipts from streaming independent reconstruction of sorted projections
(do not retain redundant full `S` tensors), fixed diagnostics, collapse curves,
representations, predictions, restoration evidence, float64 parity arrays,
100 latency samples, copied transitive reproduction source, assessments,
report, and SHA-256 manifest.

Before fitting, public tests must establish:

1. literal VISReg equality, target quantiles, direction sharing/norms, and
   finite gradients;
2. detached and no-detach forward equality but gradient inequality;
3. explicit RNG advancement and ambient global-RNG isolation;
4. a small public `.fit` integration runs both modes, proves their final
   representations diverge, preserves ambient global RNG, and matches
   capacity;
5. pair/view/sample-axis invariants and equal cell capacity;
6. ordered causal encoding plus full/training-only and strict inference
   serialization;
7. pure gate rejection for failed mechanism, safety, or value;
8. exact source/control binding, atomic non-overwrite, restoration, manifest,
   and isolated copied-source smoke reassessment.

If the tracer fails, preserve all code and artifacts, record the failed gates,
and proceed to the bounded JEPA-SCORE edge-feasibility screen.
