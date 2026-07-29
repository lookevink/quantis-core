# LeJEPA SIGReg: primary-source implementation notes

Research date: 2026-07-28.

This note uses only the authors' [LeJEPA paper, arXiv v3](https://arxiv.org/html/2511.08544v3), the [official repository](https://github.com/galilai-group/lejepa), and author-written documentation in that repository. Repository links are pinned to commit
[`c293d291ca87cd4fddee9d3fffe4e914c7272052`](https://github.com/galilai-group/lejepa/tree/c293d291ca87cd4fddee9d3fffe4e914c7272052).

## Bottom line

SIGReg in LeJEPA is not covariance regularization. It is a sliced
Epps--Pulley characteristic-function statistic that tries to make every
one-dimensional projection of a batch of embeddings look like
`Normal(0, 1)`. The exact loss contains:

- fresh random unit directions;
- the empirical cosine and sine characteristic functions;
- the standard-normal characteristic function `exp(-t²/2)`;
- a second `exp(-t²/2)` factor as the Gaussian integration window;
- trapezoidal quadrature;
- multiplication by the number of independent samples; and
- a mean across slice directions and views.

The paper and the later official code expose two different finite-quadrature
presets. A reproduction must name which one it uses rather than combining
their constants.

## Mathematical objective

For one view, let `Z = [z₁, ..., z_N]ᵀ ∈ R^(N×D)`. Draw `M` directions
`a_m ∈ S^(D-1)`. For slice `m`,

```text
u_nm = a_mᵀ z_n
φ̂_m(t) = (1/N) Σ_n exp(i t u_nm)
φ_G(t) = exp(-t²/2)
EP_m = N ∫ |φ̂_m(t) - φ_G(t)|² w(t) dt
SIGReg(Z) = (1/M) Σ_m EP_m
```

LeJEPA chooses the Epps--Pulley statistic and its implementation sets
`w(t) = exp(-t²/2) = φ_G(t)`. The paper first defines SIGReg as the average
of a univariate test over directions and explains that the average replaces
a maximum to avoid sparse gradients. It then defines the Epps--Pulley
statistic as the weighted squared distance between empirical and target
characteristic functions. See [paper §4.2 and §4.2.3](https://arxiv.org/html/2511.08544v3#S4.SS2).
There is no `1/sqrt(2π)` factor, division by `D`, empirical centering,
whitening, or per-sample L2 normalization.

Writing real and imaginary parts explicitly, the integrand actually computed
is

```text
exp(-t²/2) * [
  ((1/N) Σ_n cos(t u_nm) - exp(-t²/2))²
  + ((1/N) Σ_n sin(t u_nm))²
].
```

This is also the exact real-valued form in the official
[`EppsPulley.forward`](https://github.com/galilai-group/lejepa/blob/c293d291ca87cd4fddee9d3fffe4e914c7272052/lejepa/univariate/epps_pulley.py#L80-L99)
and avoids complex-number operations.

For `V` views and batch size `B`, the full objective is

```text
L_LeJEPA =
  (λ/V) Σ_v SIGReg({z_nv}_n=1..B)
  + ((1-λ)/B) Σ_n L_pred({z_nv}_v=1..V).
```

`L_pred` is the mean squared distance from every view embedding to the mean
of the global-view embeddings for the same sample. Thus the coefficients are
`λ` and `1-λ`, not `1` and `λ`. See
[paper §5.1, equations 6--9](https://arxiv.org/html/2511.08544v3#S5.SS1).

## Sketch distribution and tensor dimensions

The paper's Algorithm 1 and the official slicing module agree on the sketch:

```text
A_raw[d, m] ~ iid Normal(0, 1),  A[:, m] = A_raw[:, m] / ||A_raw[:, m]||₂
```

Therefore `A ∈ R^(D×M)` and each column is uniformly distributed on the unit
sphere by rotational invariance. There is no `1/sqrt(D)` multiplier after
normalization. Embeddings of shape `(..., N, D)` become projected samples of
shape `(..., N, M)`. The sample mean is over `N`, and the default reduction
is a mean over all `M` slices and any leading view dimensions. See the
official
[`SlicingUnivariateTest`](https://github.com/galilai-group/lejepa/blob/c293d291ca87cd4fddee9d3fffe4e914c7272052/lejepa/multivariate/slicing.py#L91-L153).

Directions must be resampled every minibatch/optimizer step. The paper says
this cumulative coverage is important and reports that even 16 resampled
directions can outperform thousands of fixed directions
([paper §4.3](https://arxiv.org/html/2511.08544v3#S4.SS3)). The official
module seeds a device-local generator from a `global_step` buffer and
increments the buffer after every forward call. Passing all views as a
leading dimension uses the same `A` for every view in that call.

## Constants: paper preset versus official optimized preset

| Source | Directions `M` | Quadrature grid | Knots | Window | Reduction |
|---|---:|---|---:|---|---|
| Paper Algorithm 1 | 256 default | `linspace(-5, 5, 17)` | 17 total | `exp(-t²/2)` | returns per-slice values; SIGReg definition/caller averages |
| Paper recommended starting point | 1024 | `[-5, 5]` | 17 total | `exp(-t²/2)` | mean |
| Official `MINIMAL.md` | 256 | `linspace(0, 3, 17)` | 17 nonnegative | `exp(-t²/2)` | mean |
| Official package class | caller-supplied; README uses 1024 | `linspace(0, 3, 17)` by default | 17 nonnegative | `exp(-t²/2)` | mean by default |

Sources:

- [Paper Algorithm 1](https://arxiv.org/html/2511.08544v3#S4.SS2) contains
  the full-range `[-5, 5]`, 17-knot, 256-slice implementation.
- [Paper §6.1](https://arxiv.org/html/2511.08544v3#S6.SS1) recommends 17
  points over `[-5, 5]` and 1024 slices after ablation.
- The author-written
  [`MINIMAL.md` SIGReg](https://github.com/galilai-group/lejepa/blob/c293d291ca87cd4fddee9d3fffe4e914c7272052/MINIMAL.md#L54-L75)
  uses symmetry, 17 points over `[0, 3]`, and 256 slices.
- The packaged
  [`EppsPulley`](https://github.com/galilai-group/lejepa/blob/c293d291ca87cd4fddee9d3fffe4e914c7272052/lejepa/univariate/epps_pulley.py#L62-L99)
  defaults to `t_max=3`, `n_points=17`; the
  [official README example](https://github.com/galilai-group/lejepa/blob/c293d291ca87cd4fddee9d3fffe4e914c7272052/README.md#L101-L119)
  requests 1024 slices.

For the nonnegative grid, let `Δ = 3/(K-1)`. The official code uses

```text
q_0 = q_(K-1) = Δ
q_k = 2Δ, 0 < k < K-1
effective_weight_k = q_k * exp(-t_k²/2).
```

Those weights use even symmetry to approximate the corresponding integral
over `[-3, 3]`. It is mathematically the same symmetric integral in the
continuous limit, but 17 nonnegative knots are not the same finite
quadrature as 17 knots spanning the full interval. For strict reproduction:

- use `paper-v3`: full `[-5, 5]`, 17 total knots, `M=1024` for the paper's
  recommended starting point; or
- use `official-minimal-c293d29`: half `[0, 3]`, 17 knots, explicit symmetric
  weights, `M=256`.

Do not describe one preset as an exact reproduction of the other.

One further source-level distinction matters: the prose describes a generic
Gaussian weight `exp(-t²/σ²)` with `σ=1`, but Algorithm 1 and both official
implementations use `exp(-t²/2)` as the window. Reproduce the code, not the
generic prose, when exact constants matter.

## Where SIGReg is applied

SIGReg is applied to the projector output, separately to the `B` embeddings
of each view, not to a covariance matrix and not to already standardized
features. The official minimal encoder returns both backbone features and
projector features; both the invariance loss and SIGReg consume `proj`, while
the detached backbone feature is used only by the auxiliary online probe.
See
[`MINIMAL.md` encoder and training step](https://github.com/galilai-group/lejepa/blob/c293d291ca87cd4fddee9d3fffe4e914c7272052/MINIMAL.md#L78-L93)
and
[`MINIMAL.md` loss construction](https://github.com/galilai-group/lejepa/blob/c293d291ca87cd4fddee9d3fffe4e914c7272052/MINIMAL.md#L149-L183).

There is no stop-gradient in the LeJEPA encoder/projector path, and neither a
predictor nor teacher--student network is needed to prevent collapse. The
paper recommends training without a predictor or register tokens; SWA for a
ViT is optional rather than part of SIGReg
([paper §6.1](https://arxiv.org/html/2511.08544v3#S6.SS1)).

## Optimizer and training assumptions

- The paper assumes the `N` samples along the batch axis are independent and
  identically distributed; views of one sample are not additional
  independent samples.
- The paper reports an `O(1/N)` minibatch bias in the unscaled empirical
  characteristic-function discrepancy and says it was not concerning at
  batch sizes as small as 16. It does not use U-statistic debiasing or sample
  splitting ([paper §5.1](https://arxiv.org/html/2511.08544v3#S5.SS1)).
- The paper recommends `λ=0.05` and batch size at least 128 as starting
  points. Its canonical optimizer is AdamW with learning rate in
  `{5e-3, 5e-4}`, weight decay in `{1e-1, 1e-2, 1e-5}`, no weight-decay
  schedule, and standard linear warmup plus cosine learning-rate decay
  ([paper §6.1](https://arxiv.org/html/2511.08544v3#S6.SS1)).
- The author README gives the narrower defaults `lr=5e-4`,
  `weight_decay=5e-2` for ViTs or `5e-4` for ResNets, bfloat16 mixed
  precision, and linear warmup/cosine decay to `lr/1000`
  ([official training configuration](https://github.com/galilai-group/lejepa/blob/c293d291ca87cd4fddee9d3fffe4e914c7272052/README.md#L68-L76)).
- The official minimal example uses AdamW, one epoch of linear warmup, cosine
  decay, and a fixed seed of zero
  ([minimal optimizer loop](https://github.com/galilai-group/lejepa/blob/c293d291ca87cd4fddee9d3fffe4e914c7272052/MINIMAL.md#L137-L183)).

The sources are inconsistent about the view count: the paper's experiment
detail and official README use eight total views (two global plus six local),
while a paper recommendation paragraph says two global plus eight local.
Record `V_g` and `V_l` explicitly in any reproduction artifact.

The claims of "no schedulers" in the abstract/README concern removal of
JEPA-specific teacher/EMA machinery; the same sources explicitly use a
standard optimizer learning-rate schedule.

## Faithful CPU PyTorch checklist

1. Compute in `float32` on CPU. This preserves the mathematical objective and
   avoids CPU bfloat16/complex limitations, but it is not bit-for-bit
   identical to the authors' GPU bfloat16 training.
2. Register `t`, `φ_G`, and quadrature weights as buffers. Create `A` on the
   input device and in a dtype compatible with `Z @ A`.
3. Prefer separate `cos` and `sin` means, as in the official package, instead
   of constructing a complex tensor.
4. Normalize each column of `A` along dimension zero. Do not normalize rows,
   whiten `Z`, or multiply directions by `sqrt(D)`.
5. Average the empirical characteristic function over the independent sample
   axis only. For input `(V, N, D)`, that is the `N` axis, not the view axis.
6. Preserve the leading factor `N`. Omitting it changes both gradient scale
   and the meaning of the published `λ`.
7. Average over slices and views. The paper's Algorithm 1 returns a
   per-slice vector, but the SIGReg definition and LeJEPA caller take its
   mean; the official minimal implementation calls `.mean()`.
8. Reuse one sampled `A` across leading views within a step, then resample on
   the next step. Persist the step counter or RNG state in the artifact for a
   deterministic rerun.
9. Do not detach projector embeddings on either the invariance or SIGReg
   branch. Do not add a stop-gradient, EMA teacher, predictor, covariance
   penalty, or post-hoc standardization and still call the result exact
   LeJEPA.
10. In distributed reproduction, average cosine/sine sufficient statistics
    across ranks and multiply local `N` by `world_size`, as the official code
    does. This assumes equal local batch sizes; `drop_last=True` in the
    minimal example enforces that.
11. Leave statistic clipping disabled. The official slicing wrapper exposes
    `clip_value`, but its default is `None`.
12. Use `n_points=17` with the pinned package class. The pinned README spells
    this keyword `num_points`, but the actual constructor is `n_points`;
    copying the README call literally raises `TypeError`.
13. The intermediate tensor has shape `(..., N, M, K)` with `K=17`. On a
    memory-bound CPU, it is valid to project and accumulate fixed chunks of
    the already-sampled `A` along `M`, provided the final reduction is the
    same mean. Sample the complete `A` first if bitwise seed compatibility
    with the unchunked implementation matters.

## Recommended reproduction declaration

Every result should store at least:

```text
source_preset: paper-v3 | official-minimal-c293d29
source_commit: c293d291ca87cd4fddee9d3fffe4e914c7272052
t_max, n_points, half_range_symmetry
num_slices, embedding_dim
batch_size, num_views, num_global_views
lambda, optimizer, learning_rate, weight_decay
projection_seed, initial_global_step
dtype, torch_version
```

For an edge-oriented CPU tracer, run the 256-slice official-minimal preset
first and the 1024-slice paper preset as the preregistered fidelity
sensitivity. A run containing SIGReg alone tests the isotropic-Gaussian
regularizer; it is not a full LeJEPA experiment unless it also implements the
multi-view prediction loss and the convex `λ/(1-λ)` combination above.
