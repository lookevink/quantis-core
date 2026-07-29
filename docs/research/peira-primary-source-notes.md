# PEIRA primary-source notes

## Source identity

- Michael Arbel, Basile Terver, and Jean Ponce,
  [*PEIRA: Learning Predictive Encoders through Inter-View Regressor
  Alignment*](https://arxiv.org/abs/2605.17671), arXiv
  `2605.17671v1`, submitted 2026-05-17.
- The inspected
  [arXiv source archive](https://export.arxiv.org/e-print/2605.17671)
  has SHA-256
  `52eddedc5e932959701a963605151065eb742e5ed43e0aa415514d68b9335167`.
- No author implementation is linked from the paper, arXiv record, or the
  [first author's software page](https://michaelarbel.github.io/software/).
  The paper explicitly says that code will be released upon publication.

The paper and its source archive are therefore the only implementation
authority for this tracer. Quantis must describe its implementation as a
clean telemetry translation, not an exact reproduction of unpublished code.

## Exact objective

PEIRA takes paired views `(x, y)` and maps them to `k`-dimensional
representations

```text
u = U(x)
v = V(y).
```

The encoders may be nonlinear. The inter-view predictor is linear and is
always treated as the optimal regularized regressor. Define the **uncentered**
population moments

```text
Sigma = E[v u^T + u v^T]
N     = E[u u^T + v v^T]
Q*    = (N + lambda I)^-1
P*    = Sigma Q*.
```

PEIRA minimizes

```text
E_PEIRA(U, V)
  = -0.5 * trace(P*)
    + 0.5 * lambda * E[||u||^2 + ||v||^2].
```

The same `lambda` regularizes the matrix inverse and controls feature scale.
The moments are not mean-centered, whitened, normalized, or standardized.
PEIRA has no learned predictor network, negative pairs, stop-gradient teacher,
or EMA target encoder. Its EMA state contains only running feature moments.
Both encoded views receive gradients. See
[§3 and Equations 5–6](https://arxiv.org/html/2605.17671#S3).

The population analysis associates stable global minima with leading
nonlinear canonical-correlation subspaces, with `lambda` selecting active
modes. Those guarantees assume an optimal predictor and function-space
encoders; the paper explicitly says they do not directly transfer to finite
parametric neural networks. A telemetry tracer may test spectral alignment
and non-collapse, but must not claim that its finite learned tokens are
identified nonlinear CCA coordinates.

## Exact stochastic-compositional algorithm

Directly differentiating a minibatch matrix inverse can produce unstable,
high-variance gradients. PEIRA instead maintains running moment estimates and
uses an auxiliary loss whose partial derivative in the encoders equals the
gradient of the trace-form objective when `P*` and `Q*` are exact.

For minibatch feature matrices `Phi_X, Phi_Y` with shape `(B, k)`, Algorithm 1
updates

```text
Sigma <- (1 - eta) * Sigma
         + eta/B * (Phi_X^T Phi_Y + Phi_Y^T Phi_X)

N     <- (1 - eta) * N
         + eta/B * (Phi_X^T Phi_X + Phi_Y^T Phi_Y)

Q_hat <- (N + lambda I)^-1
P_hat <- Sigma Q_hat.
```

Here `eta` is the **new-minibatch weight**, not the old-state momentum. The
encoder update differentiates the minibatch estimate of

```text
L_aux(U, V; P, Q)
  = 0.5 * E[
        u^T Q(Pu - v)
      + v^T Q(Pv - u)
    ]
    + 0.5 * lambda * E[||u||^2 + ||v||^2],
```

while `P_hat` and `Q_hat` are held fixed. The algorithm updates the running
moments before forming `P_hat`, `Q_hat`, and the step gradient. It is
reasonable to use a linear solve rather than materialize the inverse, but the
resulting matrices must be numerically identical within tolerance. See
[§5 and Algorithm 1](https://arxiv.org/html/2605.17671#S5).

For symmetric views, the authors allow a single shared encoder by setting
`V = U`. They anneal `eta` from a high initial value to a lower final value so
fresh statistics dominate early and variance decreases later. The paper does
not specify the interpolation function.

`L_aux` and `E_PEIRA` are not interchangeable scalar diagnostics:
`L_aux` is zero at a stable optimum while `E_PEIRA` is negative. The paper's
loss-versus-accuracy plots compute the trace-form `E_PEIRA` post hoc from the
running moment buffers. A tracer must retain and report both values. See
[Appendix G.2](https://arxiv.org/html/2605.17671#A7.SS2).

## Published architecture and view construction

The method is architecture-agnostic; its experiments use a shared encoder for
the two views and apply the objective to projector embeddings.

| Setting | Backbone | Pretraining | Projector |
|---|---|---:|---|
| ImageNet-1K | ResNet-50 | 100 epochs | 3-layer MLP, hidden 8,192, output 256, final bias |
| CIFAR-10 | CIFAR-style ResNet-18, `3x3` stem, no max-pool | 1,000 epochs | 3-layer MLP, hidden 2,048, output 1,024 |

ImageNet uses two `224x224` crops and the asymmetric VICReg/BYOL augmentation
family: Gaussian blur probabilities `1.0` and `0.1`, and solarization
probabilities `0.0` and `0.2`, for views one and two respectively.

CIFAR-10 uses symmetric independent views with random-resized `32x32` crops
at scale `[0.2, 1.0]`, color jitter probability `0.8`
(`brightness=0.4`, `contrast=0.4`, `saturation=0.2`, `hue=0.1`), grayscale
probability `0.2`, solarization probability `0.1`, horizontal-flip
probability `0.5`, and no blur. These details are in
[Appendix G.1](https://arxiv.org/html/2605.17671#A7.SS1).

## Published optimization

Both experiments use LARS, linear warmup, cosine learning-rate decay, and
weight decay.

| Parameter | ImageNet-1K | CIFAR-10 |
|---|---:|---:|
| Batch size | 2,048 | 256 |
| Base learning rate | 2.4 | 0.04 |
| Weight decay | `1e-6` | `1e-4` |
| Warmup | 10 epochs | 10 epochs |
| Minimum learning rate | `2e-3` | 0 |
| `lambda` | 0.1 | 0.7 |
| `eta_init` | 0.9 | 0.8 |
| `eta_min` | 0.5 | 0.5 |
| Gradient clipping | norm 1.0 | norm 1.0 from epoch 4 |

The ImageNet recipe additionally publishes LARS momentum `0.9` and trust
coefficient `1e-3`. The paper reports that clipping CIFAR from the first epoch
plateaued performance, motivating its four-epoch delay. It reports only
preliminary results and approximately 20 times and 10 times the final-run
compute for preliminary ImageNet and CIFAR tuning, respectively.

## Published evaluation and result boundary

The authors freeze the backbone and train an offline linear classifier.
ImageNet's probe runs for 100 epochs with SGD, batch size 1,024, learning rate
`0.25`, cosine decay, and weight decay `1e-6`; training uses random-resized
crop and horizontal flip, and validation uses resize-to-256 plus center crop.
CIFAR follows the same frozen-feature linear-probe family but does not publish
the complete probe optimizer table.

Across three seeds, PEIRA reports:

| Dataset | PEIRA | VICReg | SIGReg |
|---|---:|---:|---:|
| ImageNet-1K top-1 | `66.50 +/- 0.02` | `68.81 +/- 0.09` | `66.26 +/- 0.28` |
| CIFAR-10 top-1 | `90.97 +/- 0.10` | `90.92 +/- 0.14` | `92.34 +/- 0.21` |

Thus the primary evidence establishes competitive image linear-probe behavior,
not telemetry forecasting, alert value, action sensitivity, or edge
deployment. The paper also reports that ImageNet accuracy varies by 1.36
percentage points across `lambda` in `[0.025, 0.5]`, and uses effective rank
plus signal/noise eigenspace alignment as mechanism diagnostics. See
[§6 and Table 1](https://arxiv.org/html/2605.17671#S6).

## What remains ambiguous

The paper does not publish:

- executable code, an implementation revision, or dependency versions;
- initial values for the running `Sigma` and `N` matrices;
- the functional form or update cadence of the `eta` annealing schedule;
- bias correction, if any, for the running moments;
- the exact normalization, activation, and intermediate-bias layout of the
  three-layer projectors;
- complete LARS details for CIFAR-10 or parameter exclusions from LARS and
  weight decay;
- whether the `1.0` clip is a single global norm or another clipping rule;
- the full ImageNet crop-scale, color-jitter, grayscale, and flip settings;
- the full CIFAR offline linear-probe optimizer and duration;
- the seeds, sample order, floating-point precision, or matrix-solve method;
  or
- a finite-sample rule translating the population active-mode theory into a
  mechanism threshold.

These values must be preregistered as Quantis adaptations. They cannot be
silently attributed to PEIRA or selected from topology-transfer results.

## Recommended bounded telemetry contract

PEIRA should remain a **representation candidate**, not an action model or
alert policy. The minimal faithful tracer should change the representation
objective while reusing the completed LeJEPA evaluation stack.

### Data, views, and capacity

- Use exactly one independent anchor from each of the 40 fitting matched
  pairs per optimizer step. Overlapping windows and two arms remain dependent
  data, not extra samples.
- Reuse the complete-LeJEPA seeded pair/anchor schedule and its two global
  semantic views only: the full 20-step, seven-entity context and the aligned
  recent-16-step view, each with its existing 10% owned-token mask.
- Do not use the six local views: the published PEIRA moments are defined for
  one paired view `(x, y)`, and a new multi-view aggregation would be a new
  objective.
- Use one shared entity-preserving width-64, two-block Transformer and the
  existing training-only `64 -> 256 -> 64` projector. The deployed output is
  the seven width-64 anchor-time entity tokens from the complete unaugmented
  context. The projector and `Sigma/N/P/Q` state are training-only.
- Forbid numeric jitter, rescaling, entity permutation, future state, future
  controls, action truth, outcomes, and role identifiers in representation
  fitting.

This preserves PEIRA's paired-view, shared-encoder, projector-space objective
while isolating it from the already-rejected complete LeJEPA objective.

### Frozen numerical choices

- `k=64`, `lambda=0.1`, `eta_init=0.9`, `eta_min=0.5`.
- Initialize `Sigma` and `N` to exact zeros; apply the source recurrence
  without bias correction.
- Anneal `eta` linearly by optimizer step, including both endpoints. Record
  the full schedule.
- Use `torch.linalg.solve(N + lambda I, I)` in deterministic CPU float64 for
  `Q`, cast the fixed `P/Q` to the training dtype for `L_aux`, and preserve
  solve residual and condition-number diagnostics.
- Use 1,600 final-state-only steps with the existing AdamW schedule:
  learning rate `5e-4`, weight decay `5e-2`, 80 linear warmup steps, then
  cosine decay to `5e-7`. Apply global norm clipping at `1.0` after warmup.

AdamW and the small Transformer are declared telemetry adaptations. Algorithm
1 accepts an optimizer as an input; reproducing the image LARS recipe would
confound the objective comparison and is not required to preserve PEIRA's
mechanism.

### Cells and falsifiers

1. **Aligned PEIRA:** exact Algorithm 1 moments and auxiliary gradient.
2. **Deranged PEIRA:** identical capacity, initialization, schedules, and
   per-view marginals, but cyclically rotate view-B anchors across matched
   pairs before forming cross-view moments and `L_aux`.
3. **Complete LeJEPA:** the retained frozen complete-LeJEPA representation is
   the primary existing objective control.
4. **Masked reconstruction, matched PCA, and raw low-rank dynamics:** reuse
   the existing representation and operational baselines.

The candidate contributes a PEIRA-specific mechanism only if aligned pairing
beats the deranged cell in trace-form objective and downstream transfer while
remaining non-collapsed. Lower `L_aux` alone is not evidence because its
theoretical optimum is zero.

### Required diagnostics and value boundary

Retain, independently recompute, and report:

- every `eta`, minibatch moment, running `Sigma/N`, `P/Q`, `L_aux`, and exact
  trace-form objective checkpoint;
- symmetry errors, solve residuals, condition numbers, and all finite checks;
- `trace(P)`, spectra of `Sigma`, `N`, and the symmetric part of `P`;
- the paper's top-signal-eigenvector alignment with `N`;
- backbone/projector effective rank and per-entity variance;
- complete serialization/restoration parity for public tokens and for
  training-state replay;
- observable-state probes and the same frozen rank-32 action-conditioned
  forecast probes used by complete LeJEPA;
- raw/PCA/control forecast safety, downstream-effect error, pair wins,
  capacity, bundle size, and batch-one latency.

Promotion requires all shared safety gates, strict improvement over the
deranged PEIRA mechanism null, and at least the existing five-percent
held-topology downstream-effect improvement over every representation control.
If PEIRA improves spectral diagnostics but not raw/PCA operational value, the
correct conclusion is to reject this exact telemetry translation and retain
its implementation and immutable evidence.
