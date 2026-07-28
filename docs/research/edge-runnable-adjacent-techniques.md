# Edge-runnable techniques adjacent to the Quantis world-model work

Date: 2026-07-28

## Recommendation

The highest-probability next system is not a larger end-to-end neural model. It
is a small pipeline with separately testable jobs:

1. convert raw logs into stable event templates and typed parameters;
2. predict short-horizon telemetry with a compact, stable temporal model;
3. calibrate prediction residuals into an online warning signal; and
4. invoke a hybrid global-plus-graph investigator only after a warning.

This matches the development-v1 evidence: global action-conditioned dynamics
generalized, while a graph-only autoregressive rollout was unstable.

## Priority experiments

### 1. Echo-state network / reservoir computer

Use a fixed sparse recurrent reservoir with spectral radius set below one and
fit only a ridge-regression readout. This makes stability a design parameter,
training cheap, and recurrent capacity available without backpropagation
through time. It is the most direct nonlinear ablation for the observed
spectral-radius failure. Echo-state networks have also been used by turning
their forecast error into an anomaly score
([Heim and Avery, 2019](https://arxiv.org/abs/1909.01709)).

### 2. Small causal temporal convolution

A causal TCN or depthwise-separable Conv1D has fixed context and can predict the
whole ten-step horizon directly, avoiding repeated application of the same
state matrix. Generic TCNs have been shown to be a strong starting point across
sequence tasks
([Bai, Kolter, and Koltun, 2018](https://arxiv.org/abs/1803.01271)).
Compare it with the dense VARX and the reservoir; do not assume it will win.

### 3. Contractive low-rank state-space model

Factor the successful dense transition into a low-rank global channel, enforce
a contractive transition, and add the known action as an exogenous input. This
is likely the best investigator backbone because it preserves the cross-entity
information that the graph-only factorization removed while reducing parameter
count.

### 4. Bounded graph residual, not a graph-only transition

Add typed, finite-hop graph messages as a residual correction to the global
forecast. Polynomial or diffusion graph filters provide a bounded receptive
field; diffusion convolution explicitly models directed propagation
([Li et al., 2017](https://arxiv.org/abs/1707.01926)). Gate the residual norm
and test zero-, one-, and two-hop variants. The residual must earn improvement
over the global model and remain stable in free-running rollout.

### 5. Streaming log-template parsing

Use an online parser such as Drain to map free text to template IDs, keeping
typed variables such as duration, status, host, and key separately. Drain uses
a fixed-depth parse tree and was designed for online parsing
([He et al., 2017](https://jiemingzhu.github.io/pub/pjhe_icws2017.pdf)).
Feed template counts, transitions, novelty, and parameter summaries to the
temporal model. Reserve a tiny text encoder for previously unseen templates;
running a language model over every familiar log line is unnecessary.

### 6. Conformal calibration plus sequential change detection

Treat JEPA or temporal forecast error as a nonconformity score, calibrate it on
healthy held-out windows, and aggregate sustained evidence with CUSUM,
Page-Hinkley, or a conformal martingale. This preserves JEPA's intended role:
prediction divergence awakens investigation. Conformal change detectors can sit
on top of arbitrary predictors
([Vovk et al., 2021](https://arxiv.org/abs/2102.10439)); recent work also gives
finite-sample false-positive control for event-sequence anomaly tests
([Zhang et al., 2025](https://proceedings.mlr.press/v267/zhang25dn.html)).

### 7. Streaming sketches for high-cardinality events

If template or edge cardinality grows beyond the lab stack, use Count-Min
Sketch-style summaries for per-template, source-target, and parameter-frequency
features. Higher-order sketches can detect edge and subgraph anomalies with
constant update time and memory
([Bhatia et al., 2021](https://arxiv.org/abs/2106.04486)).

## Useful later ablations

- TinyTimeMixer is a compact multivariate forecasting baseline starting near
  one million parameters and supports CPU execution
  ([Ekambaram et al., 2024](https://arxiv.org/abs/2401.03955)). It is worth
  testing only after the smaller TCN/reservoir baselines.
- Switching linear dynamical models or HMMs can represent healthy, degrading,
  failed, and recovering regimes explicitly. They are especially useful if
  continuous transitions average over visibly distinct regimes.
- A tiny event n-gram or Markov model is a cheap, interpretable baseline for
  log-template order. Its surprise score should be evaluated independently
  from metric prediction error.
- KAN-AD is a recent lightweight anomaly detector reported with fewer than
  1,000 trainable parameters, but should be treated as a later benchmark rather
  than the backbone
  ([Zhou et al., 2025](https://proceedings.mlr.press/v267/zhou25u.html)).

## Deployment

Export the winning hot-path model to ONNX Runtime or ExecuTorch and measure it
on the actual target hardware. ONNX Runtime supports static and dynamic
8-bit quantization and recommends static quantization for CNNs and dynamic
quantization for recurrent or transformer models
([ONNX Runtime quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)).
ExecuTorch targets mobile, wearable, and embedded execution and provides
backend-specific quantization
([ExecuTorch concepts](https://docs.pytorch.org/executorch/stable/concepts.html)).
Quantization must have its own accuracy and attribution regression gates.

## Concrete next matrix

Hold the corpus and evaluation protocol fixed and compare:

1. dense VARX;
2. contractive low-rank VARX;
3. echo-state network;
4. small causal TCN; and
5. the best global model plus zero/one/two-hop bounded graph residuals.

Run each with metrics only and metrics plus Drain-derived events. Evaluate
one-step and ten-step prediction, treatment-minus-control effects, early-warning
lead time at a fixed false-alarm budget, attribution hit rates, spectral or
empirical rollout stability, model size, peak memory, and device latency.

Defer large transformers, graph transformers, unconstrained recursive GNNs,
and topology discovery from raw logs. They add capacity or ambiguity before the
current global-versus-local failure has been resolved.
