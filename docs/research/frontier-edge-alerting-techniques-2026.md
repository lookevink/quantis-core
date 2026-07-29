# Frontier edge-runnable techniques for Quantis alerting

Date: 2026-07-28

## Bottom line

The repository has already run most of the small neural and streaming
experiments proposed in the earlier
[edge-adjacent memo](edge-runnable-adjacent-techniques.md). The strongest
edge-runnable predictive result is now the **contractive rank-32 low-rank
action-conditioned transition**, not a JEPA, graph model, reservoir, or
temporal convolution. It matched the dense VARX reference on the open
development evaluation while using 34,503 parameters and taking 0.177 ms in a
local batch-one CPU microbenchmark
([result](edge-dynamics-development-v1-results.md#predictive-models)).

That model is promising as a **shadow predictor and closed-library
investigator**, but it is not ready to page operators. Its evidence is open
development evidence on one lab stack, its perfect attribution score is
limited to a known randomized action library, and its current residual alarm
policy did not find an operationally dependable sensitivity/false-alarm
tradeoff
([bounded conclusion](edge-dynamics-development-v1-results.md#bounded-conclusion)).

The highest-value omissions are therefore:

1. a sealed, run-aware alert-policy experiment around the frozen low-rank
   predictor;
2. export or native reimplementation with numerical parity, quantization, and
   measurement on the actual edge target;
3. OpenTelemetry-native Drain and Isolation Forest shadow baselines;
4. bounded-memory nonparametric streaming baselines such as online matrix
   profile or random-cut/isolation forests;
5. only then, tiny learned challengers such as KAN-AD and TinyTimeMixer.

Large transformers, full Mamba stacks, graph transformers, and another JEPA
loss sweep are technically edge-adjacent but poorly matched to the current
evidence. They add complexity before the alert policy and target runtime have
been validated.

## What edge-runnable techniques Quantis has actually tried

| Technique | What happened | Edge and alerting interpretation |
|---|---|---|
| Dense action-conditioned VARX | 67,704 parameters, 0.224 ms local CPU, strong forecast and closed-library attribution | Useful reference, but the low-rank factorization is strictly more attractive for the current edge target. |
| Contractive low-rank global transition | 34,503 parameters, 0.177 ms local CPU, spectral radius 0.8714, matched dense results to displayed precision | **Promising predictive core.** Freeze and confirm; do not yet use it as a paging policy. |
| Echo-state network | Preserved attribution and beat persistence, but was 10.6% worse than low-rank on action-overlap MSE, used 76,408 parameters, and took 0.446 ms | Viable compact nonlinear baseline, but not competitive on this corpus. Do not deploy this configuration. |
| Direct causal temporal convolution | 112,350 parameters, 1.223 ms, 4.63 times the low-rank action error, and only 20% action attribution | Reject the tested configuration. A depthwise-separable or differently trained TCN would be a new experiment, not a rescue of this result. |
| Strict graph VARX | Autoregressive instability, spectral radius 1.616, and severe failure for worker and PostgreSQL interventions | Do not deploy. It removed useful global cross-entity information. |
| Low-rank plus bounded graph residual | Won selection by 0.0054%, then was 0.0023% worse than low-rank on evaluation with 62,217 parameters and about 5.3 times the latency | No evidence that graph correction earns its cost. |
| Metrics plus aggregate structured events | Removing the four event inputs changed target MSE negligibly and slightly in favor of removal | Current event vocabulary does not earn hot-path use. |
| Streaming template audit | Processed 442,917 messages into three pre-existing structured templates | Plumbing success only. Quantis has **not** tested online free-text clustering, template drift, typed parameter extraction, or template novelty. |
| Fixed conformal point threshold plus sequential accumulation | Point alarms detected 96.7% of treatments but alarmed on 40% of control trajectories; sequential accumulation reduced control alarms to zero but detected 60% with median delay 17.5 transitions | The predictor may be useful, but this alert policy is not. Recalibration must be trajectory- and deployment-aware. |
| 4 x 128 Count-Min Sketch | Used 4,096 bytes and reconstructed current counts exactly, but the exact key/value representation was only 1,109 bytes | Correctly rejected for the current tiny vocabulary. Revisit only at production cardinality. |
| Learned latent JEPA bottleneck | 20,527 parameters but roughly 114 times the raw low-rank latency, materially worse errors, attribution loss, and per-node rank failures | Do not deploy this architecture. |
| JEPA residual correction | Preserved raw-path attribution, but slightly worsened every primary transfer error; 55,030 inference parameters and 0.874 ms | Do not deploy this residual branch. |

The numerical comparisons above come from the
[edge-dynamics result](edge-dynamics-development-v1-results.md), the
[action-conditioned JEPA result](action-conditioned-jepa-low-rank-development-v1-results.md),
the
[residual JEPA result](residual-jepa-correction-development-v1-results.md),
and the
[graph-dynamics stability diagnosis](action-dynamics-development-v1-results.md#post-hoc-stability-diagnosis).

## What is promising for a real-world alerting system

### 1. Freeze the low-rank predictor, but deploy it in shadow first

The rank-32 model is the only learned candidate that combines:

- a contractive fitted transition;
- preservation of observable state and action/control channels;
- dense-model prediction and closed-library attribution;
- a small matrix-based inference path; and
- sub-millisecond local Python/NumPy evidence.

Its 34,503 parameters correspond to approximately 134.8 KiB of raw float32
weights or 33.7 KiB of raw int8 weights. Those are arithmetic lower bounds,
not full deployed footprints: state buffers, normalizers, metadata, runtime
code, and alert-policy state are excluded. A Linux node agent, sidecar, or
OpenTelemetry gateway should have ample capacity; an MCU implementation is
plausible only after a native fixed-shape implementation and memory/power
measurement.

The production claim should initially be narrow: “predicts and scores in
shadow,” not “pages reliably” or “identifies arbitrary root cause.”

### 2. Make the alert policy the next scientific object

Quantis currently calibrates overlapping window scores and then reduces or
accumulates them within trajectories. The result exposes the deployment
problem cleanly: excellent point sensitivity can turn into poor
per-trajectory false-alarm behavior, while conservative trajectory thresholds
can erase sensitivity.

Three omitted, edge-cheap calibration families are relevant:

- **Run/block-aware conformal tests.** CADES constructs conformal tests for
  continuous-time event sequences with finite-sample and
  calibration-conditional false-positive-rate control
  ([Zhang et al., 2025](https://proceedings.mlr.press/v267/zhang25dn.html)).
  Its time-rescaling assumptions and event model would need validation for
  OTLP logs; the useful pattern is to calibrate the alert unit actually sent
  to an operator, not every overlapping window.
- **Online e-values and false-discovery budgeting.** e-LOND controls online
  false discovery rate under arbitrary, possibly unknown dependence when its
  inputs are valid e-values
  ([Xu and Ramdas, 2024](https://proceedings.mlr.press/v238/xu24a.html)).
  e-GAI adds dynamic testing-level allocation to improve power under more
  general dependence conditions
  ([Zhang et al., 2025](https://proceedings.mlr.press/v267/zhang25cd.html)).
  These methods do not automatically make Quantis residuals valid e-values;
  constructing and auditing that conversion is part of the experiment.
- **Adaptive conformal thresholds under drift.** Error-quantified conformal
  inference adapts threshold feedback using the magnitude of miscoverage and
  proves long-run coverage under arbitrary dependence and distribution shift
  ([Wu et al., ICLR 2025](https://arxiv.org/abs/2502.00818)). A 2026
  model-agnostic method similarly uses weighted adaptive quantiles for signal
  monitoring
  ([Martinez Gil et al., 2026](https://arxiv.org/abs/2604.20122)). These are
  development candidates, not permission to learn from unverified anomalous
  traffic; production adaptation needs quarantine, rollback, and poisoning
  controls.

All three add little compute compared with the predictor. They are more
important to operator trust than reducing 0.177 ms inference by another
fraction.

### 3. Put parsing and simple anomaly baselines in the telemetry plane

As of July 2026, the official OpenTelemetry Collector component list includes
an alpha Drain processor for logs and an alpha Isolation Forest processor for
traces, metrics, and logs
([Collector processor registry](https://opentelemetry.io/docs/collector/components/processor/)).

The Drain processor is especially well aligned with Quantis. It derives
templates online, optionally extracts wildcard parameters, bounds clusters
with LRU eviction, supports seeding for stable templates, and can persist its
parse tree across restarts
([official Drain processor documentation](https://pkg.go.dev/github.com/open-telemetry/opentelemetry-collector-contrib/processor/drainprocessor)).
Quantis only audited three already-structured messages, so a real Drain
processor on production-like SDK and application logs is genuinely omitted.
Template counts, new-template rate, parameter quantiles, and transitions
should be evaluated separately before being added to the low-rank state.

The official Isolation Forest processor supports multidimensional telemetry
attributes, sliding windows, configurable retraining, grouping, score
enrichment, and memory guardrails
([official Isolation Forest processor documentation](https://pkg.go.dev/github.com/open-telemetry/opentelemetry-collector-contrib/processor/isolationforestprocessor)).
Because it is alpha, it is a shadow comparator rather than a paging dependency.
Its advantage is integration realism: it can run in the same agent/gateway
plane in which a future Quantis scorer would live.

The Collector supports custom components, and its documented agent-to-gateway
pattern keeps agents small while assigning heavier processing to gateways
([custom components](https://opentelemetry.io/docs/collector/extend/custom-component/),
[agent-to-gateway deployment](https://opentelemetry.io/docs/collector/deploy/other/agent-to-gateway/)).
This gives Quantis two sensible targets:

1. a native Go processor for the linear low-rank hot path; or
2. a sidecar scorer fed by an agent, with aggregation at the gateway.

The first minimizes runtime footprint; the second preserves Python model
iteration speed. Both should be benchmarked with real OTLP throughput,
backpressure, restart, and missing-data tests.

## Important omitted edge baselines

### Bounded-memory online matrix profile

Online matrix profile is training-free and compares the current subsequence to
its closest historical match. It has been studied specifically for IT
operations series, including a cached implementation and spectral-residual
combination
([Lan et al., 2021](https://arxiv.org/abs/2108.12093)). A 2024 multidimensional
study benchmarks matrix-profile anomaly detection across 119 datasets and
reports consistent performance across unsupervised, supervised, and
semi-supervised setups
([Yeh et al., 2024](https://arxiv.org/abs/2409.09298)).

This is a strong omitted baseline for recurring schedules because it detects
subsequence novelty without fitting a global transition. On an edge agent it
must use a declared rolling history or prototype store; unbounded history is
not a viable memory policy. It will not provide action-conditioned
counterfactual attribution, so it should challenge the **alarm score**, not
replace the investigator.

### Random-cut or isolation forests

Streaming tree ensembles are a useful high-cardinality, nonlinear complement
to a linear predictor. OpenSearch already uses Random Cut Forest for near-real
time anomaly detection and alert integration
([official OpenSearch anomaly-detection repository](https://github.com/opensearch-project/anomaly-detection)).
The newer OpenTelemetry Isolation Forest processor makes a directly compatible
experiment possible without inventing an integration layer. Quantis has tried
neither family.

These models should be grouped by service/operation and fed residual,
rate-of-change, missingness, and saturation features. A single global forest
would likely repeat the schedule confounding already observed in Quantis.

### KAN-AD

KAN-AD replaces spline functions with truncated Fourier expansions and reports
fewer than 1,000 trainable parameters on cloud/web time-series anomaly
detection benchmarks
([Zhou et al., ICML 2025](https://proceedings.mlr.press/v267/zhou25u.html)).
That makes it a credible tiny diagnostic challenger. It is omitted from
Quantis.

Its limitations are equally important: the published task is anomaly scoring,
not action-conditioned multivariate rollout or counterfactual attribution.
Use it against the residual detector at the same trajectory false-alarm
budget, not as a replacement world model. Its tiny parameter count suggests
edge feasibility, but the paper does not substitute for an exported-runtime
benchmark on the target hardware.

### TinyTimeMixer

TinyTimeMixer starts around one million parameters, supports multivariate
forecasting and exogenous signals during fine-tuning, and is reported to run
on CPU-only machines
([Ekambaram et al., 2024](https://arxiv.org/abs/2401.03955)). It is omitted.
At one million parameters its raw weights are roughly 3.8 MiB in float32 or
0.95 MiB in int8, before runtime overhead. That is comfortable on a node-class
edge CPU but roughly 29 times the current low-rank parameter count.

TTM is worth testing only if transfer from broad pretraining improves unseen
schedule/topology behavior. It must beat a frozen 34,503-parameter model at a
fixed false-alarm budget and device resource envelope; forecast benchmark
accuracy alone is irrelevant.

### Cross-system residual calibration

M2AD separates a base expected-behavior model from global residual scoring and
calibrated thresholding across heterogeneous sensors and systems. Its 2025
paper reports evaluation on 130 Amazon fulfillment-center assets
([Alnegheimish et al., 2025](https://proceedings.mlr.press/v258/alnegheimish25a.html)).
The full deep framework is not necessarily edge-small, but its **residual
aggregation pattern** is directly applicable: keep the low-rank predictor,
then learn or fit a small global scorer that accounts for heterogeneous
services and channels. Quantis has not tested this separation.

This is higher priority than another latent encoder because Quantis already has
a competent predictor and an inadequate alarm aggregation rule.

## Frontier techniques to watch, not prioritize

### Wavelet plus lightweight multi-scale autoencoding

A July 2026 preprint combines a discrete wavelet transform with a lightweight
multi-scale autoencoder, reports a model below 500 KB, and evaluates latency
and power on an NVIDIA Jetson Nano
([Wani and Sarangi, 2026](https://arxiv.org/abs/2607.12599)). This is unusually
direct edge evidence, but it is univariate, extremely new, and based on
reconstruction rather than action-conditioned prediction. Treat it as a
research-only multiscale residual benchmark.

### Selective state-space models

Mamba offers linear sequence-length scaling and a hardware-aware recurrent
algorithm
([Gu and Dao, 2023](https://arxiv.org/abs/2312.00752)). It is an omitted
long-context technique, but Quantis windows are currently short and a
contractive linear state-space model already works. Edge quantization is also
not automatic: early Mamba post-training-quantization work identifies
activation-outlier channels as a difficulty
([Pierro and Abreu, 2024](https://arxiv.org/abs/2407.12397)).

A selective SSM becomes interesting only if production evidence shows that
20-step context misses long-duration regime information that bounded history,
switching linear models, and matrix profile cannot capture.

### Lightweight multi-scale autoencoders, Mamba, and foundation models are not
alert policies

These models can emit scores or forecasts, but they do not solve paging-unit
definition, repeated testing, alert grouping, cooldown, recovery, missing
telemetry, or calibration drift. Quantis' own experiments show that this
distinction is decisive.

## Runtime techniques Quantis has omitted

There is no ONNX, ExecuTorch, LiteRT, Core ML, int8, or quantization-aware path
in the current source tree; the edge results are Python/NumPy or PyTorch
microbenchmarks.

For future neural challengers, ONNX Runtime supports mobile CPU execution,
XNNPACK, NNAPI, Core ML, 8-bit weight reduction, and custom reduced-operator
builds
([ONNX Runtime mobile](https://onnxruntime.ai/docs/tutorials/mobile/),
[reduced operator configuration](https://onnxruntime.ai/docs/reference/operators/reduced-operator-config-file.html)).
ExecuTorch targets mobile, wearable, and embedded devices, and its XNNPACK
backend supports int8 linear and Conv1D paths
([ExecuTorch concepts](https://docs.pytorch.org/executorch/stable/concepts),
[XNNPACK quantization](https://docs.pytorch.org/executorch/stable/backends/xnnpack/xnnpack-quantization.html)).

Quantization is not guaranteed to improve latency on every device; ONNX
Runtime explicitly notes that gains depend on hardware instructions and that
quantization/dequantization overhead can make old devices slower
([ONNX Runtime quantization guide](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)).
Every export therefore needs:

- float implementation parity;
- quantized prediction, attribution, and alarm regressions;
- model plus runtime binary size;
- peak resident memory;
- batch-one and sustained OTLP throughput;
- p50/p95/p99 latency;
- cold start and restart behavior; and
- target-device energy or CPU utilization.

For the current low-rank linear model, a native fixed-shape implementation may
be smaller and easier to audit than importing a neural runtime.

## Recommended experiment order

1. **Freeze one low-rank bundle.** No predictor tournament. Use whole-run
   train/calibration/test separation and collect a fresh sealed corpus with
   schedule, topology, restart, telemetry-loss, and intervention diversity.
2. **Define the alert unit before fitting thresholds.** Compare the current
   sequential rule with a run/block conformal rule and one online
   e-value/adaptive-conformal rule. Report alerts per host-day or service-day,
   not only window false-positive rate.
3. **Build a target-runtime parity harness.** Compare current NumPy, native Go
   Collector component or sidecar, and int8 where applicable on the actual
   x86/ARM target.
4. **Run OpenTelemetry-native shadows.** Drain-derived templates and
   parameters, Isolation Forest, and a bounded online matrix-profile scorer.
   Do not let adaptive baselines learn from confirmed incident intervals.
5. **Add exactly one tiny learned challenger.** KAN-AD first; TinyTimeMixer
   only if transfer learning is the hypothesis. Require improvement at the
   same trajectory alert budget and device envelope.
6. **Defer Mamba, multiscale autoencoders, graph transformers, and new JEPA
   encoders** until the simpler system fails for a diagnosed capacity or
   context-length reason.

The likely real-world architecture is therefore deliberately hybrid:

> OpenTelemetry Drain/typed normalization -> bounded feature windows ->
> frozen low-rank forecast -> run-aware calibrated warning -> optional
> action-library investigator -> alert grouping and operator notification.

The low-rank predictor is the promising part already in hand. Reliable alert
calibration, production log semantics, and an audited edge runtime are the
important omissions.
