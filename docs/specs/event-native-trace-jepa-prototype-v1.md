# Event-native trace JEPA prototype v1

## Status and question

This is a preregistered, single-seed, open-development logic prototype. It
asks:

> Do raw trace paths add topology-transfer alert or investigation value beyond
> metrics-only dynamics, binned event features, an event-alignment null, and a
> simple event n-gram?

The prototype is deliberately non-production, but its exact runner remains in
the repository as reproduction code. A positive result authorizes a durable
implementation through the shared JEPA seams; it is not production or
sealed-confirmation evidence.

## Evidence boundary

- Source corpus:
  `artifacts/action-dynamics/development-v1`.
- Reuse the content-addressed pair roles in
  `artifacts/action-dynamics/edge-preprocessing-v1`.
- Fit on fitting-role pairs from worker topologies one and two.
- Use selection only for preregistered diagnostics; do not tune the recipe.
- Calibrate hidden-action alert thresholds on calibration-role controls from
  worker topologies one and two.
- Report both in-distribution evaluation and the primary held-out worker-three
  topology.
- Keep matched treatment/control pairs atomic.
- Seed: `211`.
- Runtime: local CPU PyTorch with deterministic algorithms and one thread.
- No fresh or sealed cases are collected or opened.

The frozen corpus already reports 442,917 of 442,917 eligible log events as
trace-linked and 219,914 of 219,989 eligible request traces as structurally
complete. The prototype independently recomputes the trace-path quantities it
uses.

## Causal event compilation

The candidate consumes spans, not the four graph-window event-count features
already present in the metric tensor.

For every case:

1. Parse the OTLP trace capture and reconstruct each request path from
   `traceId`, `spanId`, and `parentSpanId`.
2. Represent each span by its training-fitted template, graph entity, status
   outcome, duration, depth, gap from the previous linked span, and time to the
   next linked span.
3. Fit template and entity vocabularies on fitting cases only. Unseen values
   map to an explicit unknown token.
4. The metric capture uses a synthetic logical clock and cannot be compared
   directly with span wall-clock timestamps. Reconstruct wall-clock window
   ends from the independently recorded `action.run.boundary` start event and
   the manifest's declared logical-window period. Assign a completed trace to
   the first reconstructed window end at or after the trace's final span end.
   Events after the final window are excluded as drain events.
5. Never use `quantis.experiment.origin.window.index` for event placement. It
   is an identity check only, because assigning a late span to its request
   origin would leak future completion information into the past.
6. Keep at most the first eight causally ordered spans in one trace. Report
   truncation, unknown-template, unknown-entity, incomplete-parent, and
   excluded-drain rates.

The history for transition `t` contains only traces completed through window
`t`. Its target contains traces completed in windows `t+1` through `t+10`.
This clock-domain correction was frozen after the initial runner stopped
before fitting because synthetic metric timestamps were not comparable to
span timestamps; no model result had been produced.

## Candidate

### Masked trace encoder

- Token dimension: 48.
- Template, entity, outcome, and depth embeddings are summed with a projection
  of log-duration and log-gap numeric inputs.
- Two Transformer encoder layers, four heads, feed-forward width 96, zero
  dropout.
- A learned trace token pools each path.
- On every fitting batch, 30% of non-padding spans are masked, with at least
  one masked span per trace.
- The encoder predicts masked template, entity, and outcome and predicts
  `log1p(time-to-next-span)` with smooth-L1 loss.
- Pretraining: 12 epochs, AdamW, learning rate `0.002`, batch size 512.

### Window target encoder

For each graph entity in a logical window, pool the unmasked span states of
all traces completed in that window using their mean, standard deviation, and
log count. Concatenate the entity pools in declared graph order. Fit a
training-only rank-32 PCA and standardization transform. This fixed
32-dimensional vector is the future joint-embedding target.

Report effective rank and variance on every role. A rank below eight or
non-finite value is collapse.

### Action-conditioned latent predictor

- A one-layer GRU reads the preceding 20 window embeddings into a
  64-dimensional context.
- A shared action projection maps each horizon's ordered
  entity-by-action tensor to 32 dimensions.
- A recurrent 64-dimensional predictor consumes context, future controls,
  projected future actions, and an eight-dimensional horizon embedding.
- A linear head predicts ten future 32-dimensional event embeddings.
- Loss: mean squared error in the frozen target-embedding coordinates.
- Training: 40 epochs, AdamW, learning rate `0.002`, batch size 256.
- Seed and runtime are fixed as above.

This is an event-native JEPA because the online model predicts encoded future
trace state rather than reconstructing raw spans or graph metrics. The masked
event and next-time objectives shape the target encoder; the temporal
predictor never decodes events during fitting.

## Controls

### Metrics-only

Refit the rank-32 contractive raw-state low-rank model on the identical
topology-one/two fitting windows. Score its existing hidden-action conformal
detector on the same calibration and evaluation roles.

### Binned event

Aggregate training-fitted per-window span-template counts, status counts, and
duration/gap means by declared entity. Fit a rank-32 training PCA and the same
64-dimensional action-conditioned temporal predictor. This matches the
candidate's downstream capacity but removes trace paths, within-trace order,
and masked-event pretraining.

### Alignment-shuffled JEPA null

Use the candidate encoder and predictor capacity, but deterministically
derange future event targets among fitting trajectories within the same worker
topology, transition index, and treatment/action family. This preserves
marginals and action prevalence while breaking case-level context-to-future
event alignment.

### Event n-gram

Fit a Laplace-smoothed order-two Markov model over
template/entity/outcome trace tokens on fitting controls, plus robust
training-only location and scale for log time-to-next-span. Use its
per-window surprise as an alert score. It has no action-counterfactual
investigation claim.

## Evaluation

### Hidden-action alerting

For the candidate, binned control, and shuffled null:

1. Zero all future action tensors.
2. Compute one-step squared latent prediction error for each transition.
3. Convert errors to empirical upper-tail p-values using calibration control
   windows.
4. Accumulate nonnegative evidence
   `max(0, -log(p) - log(2))`.
5. Set the sequential threshold from complete calibration-control
   trajectories only.

The n-gram uses the same steps with per-window event surprise. Report:

- control-trajectory sequential false-alarm rate;
- treatment post-onset detection;
- pre-onset treatment alerts;
- median and worst post-onset delay; and
- alerts per logical run.

The held-out-topology alert lane passes only if:

- control false alarms are zero (the ten transfer controls make the shared
  five-percent bound discrete);
- treatment detection is at least 80%;
- median delay is at most 10 transitions;
- the candidate beats the alignment null by at least 10 percentage points of
  detection or two transitions of median delay at the same false-alarm bound;
  and
- it is non-inferior to the best binned-event or n-gram control on detection
  and median delay.

### Investigation

At the frozen onset query, predict ten future event embeddings for every
candidate action plan and choose the minimum latent error to the observed
future event embeddings. Report action-and-target hit@1, exact variant hit@1,
and no-action specificity.

The held-out-topology investigation lane passes only if:

- action-and-target hit@1 is at least 95%;
- no-action specificity is 100%;
- correct actions beat both no-action and whole-pair-deranged actions on at
  least 80% of treatment queries; and
- action-and-target hit@1 improves by at least 10 percentage points over both
  the binned-event control and alignment null.

### Secondary diagnostics

- masked template/entity/outcome accuracy and time-to-next error;
- target effective rank and per-dimension variance;
- training-only frozen ridge reconstruction of the current normalized
  observable state from the event context;
- batch-one CPU latency, parameter count, and serialized tensor bytes; and
- in-distribution versions of every primary metric.

Observable state reconstruction is diagnostic, not a predictive-core claim.

## Safety and decision

Reject the recipe before value assessment if any of these fail:

- trace-link coverage below 95%;
- unknown template or entity rate above 1% on transfer;
- any future-to-history, role, pair, or topology leakage;
- event target effective rank below eight;
- non-finite training or evaluation values; or
- the candidate does not outperform its alignment-shuffled null.

Advance to a durable, tested implementation and three fixed seeds only if one
held-out-topology value lane passes with every safety gate. Otherwise retain
the runner, measurements, and bounded interpretation as a reproducible
negative result, and move to the next materially different JEPA family.

Permitted positive claim:

> On the open development corpus, causally compiled trace paths added
> single-seed topology-transfer alert or closed-library investigation value
> under the frozen controls.

Permitted negative claim:

> This event-native trace recipe did not add sufficient open-development
> topology-transfer value to justify durable implementation.

Neither outcome confirms production paging, general incident semantics, or a
software world model.
