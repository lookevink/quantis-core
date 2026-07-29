# PEIRA telemetry tracer v1

## Status and claim boundary

Frozen before implementation and fitting. This is a single-seed,
open-development, paper-faithful telemetry translation, not an exact
reproduction, production alert policy, robustness study, or sealed
confirmation. The authors have not released PEIRA code.

The hypothesis is:

> PEIRA's trace-of-optimal-regressor objective learns noncollapsed shared
> telemetry structure that improves held-topology action-effect forecasting
> over a pair-deranged PEIRA null, the retained complete LeJEPA and
> masked-reconstruction controls, matched PCA, and raw dynamics.

## Frozen data, roles, and prior control

Use the content-addressed edge preprocessing artifact
`eb54271132f88c9a431b01e786ea66279a563776434cca2290e47e6b7ae9b3ff`
and the existing worker-topology partition:

- fitting: 40 in-topology pairs;
- selection: 10 in-topology pairs;
- calibration: 10 in-topology pairs;
- IID evaluation: 20 in-topology pairs; and
- transfer evaluation: 10 held-topology pairs.

The retained complete LeJEPA artifact is
`artifacts/action-dynamics/prototype-complete-lejepa-v1`, with manifest
SHA-256
`00639afaee81cd3844e8b60014ed87f886a8e7a7e0b20bc50834416da1deb265`.
Copy its `complete_lejepa` and `masked_autoencoder` model states into the new
artifact and independently recompute their representations and downstream
probes. Refit matched entity PCA and raw rank-32 dynamics on fitting only.

Representation fitting uses current fitting histories and the declared graph
only. Selection chooses downstream ridge values only. Calibration does not
tune the representation. Evaluation never updates a weight, moment,
normalizer, ridge, schedule, threshold, or gate.

## Frozen views

At every optimizer step, retain one pair-blocked anchor from each fitting
pair. Reuse the complete-LeJEPA view schedule but use only its two global
views:

1. the full 20-step by seven-entity current history; and
2. the aligned most-recent 16-step history.

Each view independently hides 10% of owned time/entity tokens under the
existing seeded schedule. Numeric jitter, value rescaling, entity
permutation, synthetic telemetry, future states, controls, actions, pair
identity, outcomes, and cross-role records are forbidden.

`aligned_peira` uses corresponding views. `deranged_peira` applies a seeded
no-fixed-point cyclic rotation to view B across matched pairs before moment
updates and loss evaluation. The two cells otherwise receive identical
marginals, initialization, anchors, masks, architecture, and optimization.

## Frozen architecture

Use one shared complete-LeJEPA telemetry encoder for both views:

- 20 by seven absolute time/entity tokens;
- width 64;
- two pre-norm graph-biased Transformer blocks;
- four heads and feed-forward width 128;
- GELU and zero dropout; and
- the existing declared time, entity, kind, presence, relation, and
  graph-distance identities.

Pool visible tokens and apply the existing training-only
`Linear(64,256) -> GELU -> Linear(256,64)` projector. The deployed
representation is the seven width-64 anchor-time entity tokens from the
complete unaugmented current history. The projector and all PEIRA moment
state are training-only.

The public `encode(histories, graph)` method accepts no role, future,
control, action, target, pair identity, or outcome.

## Exact PEIRA translation

For paired projector features `U,V` shaped `B x 64`, use the paper's
uncentered moment convention:

```text
Sigma_batch = (U^T V + V^T U) / B
N_batch     = (U^T U + V^T V) / B
Sigma       = (1 - eta) Sigma + eta Sigma_batch
N           = (1 - eta) N + eta N_batch
Q           = (N + lambda I)^-1
P           = Sigma Q
```

Update the moments before computing `P`, `Q`, and the step gradient. Hold
`P` and `Q` fixed for the auxiliary loss:

```text
L_aux = 0.5 E[
    u^T Q(Pu - v) + v^T Q(Pv - u)
] + 0.5 lambda E[||u||^2 + ||v||^2].
```

Both views receive gradients. There is no learned predictor, target encoder,
stop-gradient view, negative pair, covariance centering, whitening, feature
normalization, or bias correction.

Freeze `lambda=0.1`. Initialize `Sigma` and `N` to exact float64 zeros.
Compute float64 `Q` with `torch.linalg.solve`, compute `P=Sigma Q`, and cast
fixed `P/Q` to float32 for the encoder gradient. Anneal the new-batch weight
`eta` linearly from `0.9` to `0.5`, including both endpoints.

Also report the post-hoc trace objective

```text
E_PEIRA = -0.5 trace(P)
          + 0.5 lambda E[||u||^2 + ||v||^2].
```

Do not use `L_aux` as the mechanism score: the paper shows that it is zero at
a stable optimum while `E_PEIRA` is negative.

## Frozen optimization and randomness

Fit both PEIRA cells for 1,600 final-state-only steps using CPU float32
AdamW, learning rate `5e-4`, weight decay `5e-2`, 80-step linear warmup, and
cosine decay to `5e-7`. Apply global gradient-norm clipping at 1.0 only after
warmup.

| purpose | seed |
|---|---:|
| shared initialization | 26026 |
| pair-blocked anchors | 26126 |
| two-view masks | 26226 |
| derangement | 26326 |

The optimizer is a declared telemetry adaptation. Algorithm 1 accepts an
optimizer; importing the paper's image-scale LARS recipe would confound the
objective comparison.

## Shared downstream evaluation

Freeze every representation. Fit the existing rank-32 action-conditioned
probe over ridges `{1e-4, 1e-3, 1e-2, 1e-1, 1}` using fitting only. A ridge
is selection-safe when overall and action-overlap MSE are both within 5% of
raw. Choose the safe ridge with lowest selection downstream-effect MSE,
tie-breaking toward the smaller ridge. If none is safe, retain the
lowest-effect ridge and record selection-safety failure.

For selection, IID, and transfer, report overall/action/effect MSE and
current-state retention. Also retain transfer per-pair effect error,
attribution hit@1, no-action specificity, and absent/shuffled-action sanity.

## Mechanism diagnostics

Independently recompute on selection and transfer:

- minibatch and final running `Sigma/N`, derived `P/Q`, symmetry error,
  solve residual, and condition number;
- `L_aux`, exact `E_PEIRA`, and `trace(P)`;
- spectra of `Sigma`, `N`, and the symmetric part of `P`;
- mean alignment of the leading eight `Sigma` eigenvectors with `N`;
- projector and backbone effective rank; and
- per-entity backbone variance.

The PEIRA mechanism passes only if, on both roles:

1. every matrix and scalar is finite, `Sigma/N` symmetry error is at most
   `1e-8`, solve residual is at most `1e-8`, and condition number is at most
   `1e6`;
2. candidate projector effective rank is at least eight and every
   fit-varying entity has nonzero token variance;
3. candidate `-E_PEIRA` is at least 10% greater than deranged PEIRA; and
4. candidate top-eight signal/noise eigenvector alignment exceeds deranged
   PEIRA by at least five percentage points.

## Gates

All safety gates must pass:

1. finite evidence and exact 40/10/10/20/10 disjoint role cardinalities;
2. identical independently recomputed candidate/null training and inference
   capacity;
3. original/restored public representations, PEIRA state, probes, prior
   controls, PCA, and deployment bundle agree within `1e-6`;
4. copied public inference signatures are causal;
5. anchor, view, derangement, learning-rate, clip, and eta schedules
   recompute;
6. selection-only ridge choice and no-safe-ridge status recompute;
7. candidate selection and transfer overall/action MSE are within 5% of raw;
8. attribution hit@1 is at least 95%, no-action specificity is 100%, and
   correct actions beat absent/shuffled actions on at least 80% of transfer
   treatment pairs;
9. the exact candidate inference bundle is at most 16 MiB and retains 100
   raw batch-one CPU latency samples; and
10. an isolated copied-source assessor recomputes every gate.

The value lane passes only if:

- candidate selection effect MSE is strictly best among all learned
  representations;
- candidate transfer effect MSE improves the best learned control and raw by
  at least 5%; and
- candidate beats the best learned control on at least 60% of transfer
  pairs.

Advance only if every safety, mechanism, and value gate passes. Otherwise
reject this exact telemetry translation, not PEIRA on the paper's image
benchmarks.

## Artifact contract

Publish atomically without overwrite. Retain both PEIRA cells, copied prior
controls, PCA and raw models, probes, exact inference bundle, source artifact
identities, every schedule, per-step minibatch/running moments and scalar
diagnostics, final `P/Q`, role representations, predictions, per-pair errors,
restoration evidence, 100 latency samples, independent assessment, copied
transitive reproduction source, report, and SHA-256 manifest.
