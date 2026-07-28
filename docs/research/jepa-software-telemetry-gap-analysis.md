# Is JEPA applicable to software telemetry?

Date: 2026-07-27

## Bottom line

The confirmation result does **not** show that JEPA is inapplicable to software
systems. It shows that our particular model—a per-entity linear/tanh encoder,
linear predictor, fixed past-to-future task, continuous EMA target, and
single-resolution loss—does not learn a representation competitive with raw
features or frozen PCA.

That is a materially smaller claim than “JEPA does not work for logs.” Several
parts of the published JEPA recipe were absent, and recent time-series JEPA
work reports both that the modality is viable and that a naive continuous
single-resolution formulation can suffer the same instability we observed.

The defensible hypothesis is:

> JEPA may learn a useful predictive state of a software system when it is
> trained over structured, multimodal telemetry with block masking,
> multi-resolution targets, explicit anti-collapse constraints, and an
> expressive relational encoder. Logs alone are not guaranteed to identify
> the system state.

## What our experiment actually tested

Our confirmation used 72 trajectories from 24 nominal schedule families,
producing 11,160 training and 11,160 validation windows. Although the raw
evidence contained 1,539,667 structured log records, those records are highly
repetitive emissions from only 72 controlled runs; they are not 1.5 million
independent semantic situations.

The learned model:

- linearly projects each entity's temporal patch and applies `tanh`;
- concatenates selected entity tokens and target-time controls;
- predicts each entity with a linear matrix;
- trains against a stop-gradient EMA projection with squared error;
- always predicts a fixed future block from a fixed past context; and
- has no random entity/time masking, attention, message passing,
  multi-resolution view, discrete regime bottleneck, or explicit collapse
  diagnostic.

It contains the online encoder / EMA target encoder / predictor skeleton of a
JEPA, but not the core masked-prediction and high-capacity token-processing
recipe used by I-JEPA and V-JEPA.

## Important ideas from the papers that we missed

### 1. The prediction task is designed, not merely “past predicts future”

[I-JEPA](https://arxiv.org/abs/2301.08243) reports that representation quality
depends critically on predicting sufficiently large semantic target blocks
from a spatially distributed, informative context. [V-JEPA](https://arxiv.org/html/2404.08471v1)
uses two complementary multi-block masks per clip, about 90% effective masking,
and found that low spatial or temporal coverage makes the task too easy and
hurts downstream representations.

Our fixed past-to-future regression never asks the encoder to infer missing
entities, missing intervals, or differently sized target regions. The
predictor can exploit stable correlations without being forced to construct a
general system state.

**Translation for telemetry:** randomly mask large, contiguous blocks across
time and topology. Each sample should include more than one task, such as:

- infer a masked subsystem interval from its neighbors and earlier state;
- infer several masked nodes/edges from a distributed graph context; and
- predict short- and long-horizon target blocks from the same context.

The predictor must receive entity, edge, time, horizon, and mask-position
embeddings so that it knows which missing state it is being asked to predict.

### 2. Our encoder and predictor are drastically underpowered

V-JEPA uses a transformer encoder and a narrow but still 12-block transformer
predictor. Even the small [TS-JEPA](https://arxiv.org/html/2509.25449v1)
time-series study uses a transformer with 128-dimensional embeddings and 70%
masking. Our model has independent linear projections followed by `tanh` and a
linear predictor. Selecting one-hop inputs is not graph representation
learning; there is no learned message passing or relational attention.

**Translation for telemetry:** use a shared temporal graph transformer with
typed node/edge tokens, relative time embeddings, and graph-distance or
relation embeddings. A practical development model can still be small—roughly
2–4 encoder blocks at width 64–128 and a 2–4 block predictor—but it must be
capable of conditional interaction among entities.

### 3. Continuous EMA self-distillation is not automatically stable

V-JEPA uses stop-gradient EMA targets and an L1 feature loss, which it reports
as more stable than the alternative it tested. Our implementation uses MSE,
a fixed EMA decay of 0.98, and orthonormalizes encoder columns, but does not
measure effective latent rank, per-dimension variance, covariance, target
drift, or predictor shortcuts.

The July 2026 revision of
[SC-JEPA](https://arxiv.org/html/2602.04643v2), a time-series anomaly-prediction
paper, explicitly reports that direct continuous JEPA self-distillation can be
unstable or collapse. It introduces a soft prototype codebook, entropy and
prototype-alignment terms, and a reconstruction anchor. Its ablation reports
near-random behavior without the codebook and collapse when the reconstruction
anchor is removed.

This is unusually close to our empirical pattern: latent training loss falls
while frozen downstream prediction is poor.

**Translation for telemetry:** measure collapse first, then test an explicit
regime bottleneck:

- 32–64 soft regime prototypes;
- batch code-usage entropy and per-sample sharpness;
- prototype separation/alignment;
- effective-rank and variance diagnostics; and
- a lightly weighted reconstruction or state-probe anchor.

The reconstruction term is not a universal JEPA requirement. Here it is a
domain-specific anchor because subsystem attribution requires local state to
remain recoverable.

### 4. Software dynamics are multi-scale

SC-JEPA jointly predicts fine and coarse future states, using a fine patched
view and a downsampled trend view. It argues that this separates transient
noise from slower regime changes. This is directly relevant to a stack where
request handling, queue growth, retries, connection pressure, and recovery
operate at different time constants.

Our target has one temporal resolution. Multiple horizons do not replace
multiple resolutions if every horizon is encoded from the same patch scale.

**Translation for telemetry:** construct at least:

- a fine view for request/queue/worker transitions; and
- a coarse view for saturation, backlog, dependency health, and recovery.

Predict both from the same context representation.

### 5. Dense/local grounding needs its own training signal

[V-JEPA 2.1](https://arxiv.org/html/2603.14482v3) adds loss on visible as well
as masked tokens and deep self-supervision at intermediate encoder layers. Its
ablation shows a large gain in dense localization tasks, while also showing
that this loss must be warmed up and weighted carefully to avoid sacrificing
global semantics.

Our use case requires both global regime understanding and precise
node/edge attribution. A single final latent-prediction loss is unlikely to
guarantee both.

**Translation for telemetry:** add intermediate node/edge probes or
deep predictive losses and warm in a local-state preservation term after the
global masked task begins learning. Do not force every latent dimension to
decode every raw event.

### 6. A world model needs action-conditioned transitions

[V-JEPA 2](https://arxiv.org/html/2506.09985v1) separates two stages:
action-free representation pretraining, then an action-conditioned predictor
trained on interaction data. The latter autoregressively predicts future
representations from previous state, actions, and proprioceptive state, and is
evaluated inside a planning loop.

Our request-demand and worker-count controls help forecasting, but nominal
schedule conditioning is not evidence that the model has learned causal
interventions. No nominal-only result can support a world-model claim in the
strong planning sense.

**Translation for telemetry:** after representation pretraining succeeds,
freeze or slowly tune the encoder and train a transition model on explicit
interventions: scale workers, inject latency, kill/restart a dependency,
change retry/backpressure policy, and release a lock. Evaluate held-out
multi-step intervention rollouts, not only one-step reconstruction.

## Logs are observations, not the state itself

JEPA's useful bias is to discard details that are not predictable. That fits
telemetry, but only if the remaining observations identify the operational
state. Logs are event-triggered, censored by logging policy, repetitive, and
often ambiguous when absent. Two different hidden states can emit the same
log stream. No representation learner can recover information that was never
observed.

The next model should therefore treat:

- continuous metrics and trace/span durations as primary state observations;
- structured log template IDs, entity, operation, outcome, duration, and
  trace/span correlation as sparse event tokens;
- free-form message text as an auxiliary field, preferably parsed into stable
  templates and attributes before using a language embedding; and
- declared topology and explicit controls/actions as first-class inputs.

The sparse Redis threshold events that dominated our error should not be
raw-value prediction targets. Continuous dependency latency, availability,
last-success age, and queue/connection pressure should define Redis state;
threshold logs can remain auxiliary evidence.

## One decisive JEPA experiment

Before abandoning JEPA, run one bounded, paper-informed development study on
the already-open corpus:

1. **Tokenize the graph.** Emit typed node and edge tokens at fine and coarse
   temporal resolutions, with structured logs as event tokens.
2. **Use a real relational encoder.** Train a small temporal graph transformer
   (width 64–128) and transformer predictor with entity, relation, time,
   horizon, and mask-query embeddings.
3. **Use multi-mask pretraining.** Sample two entity-time masks per window,
   covering roughly 70–90% of target tokens, including large contiguous
   subsystem/time blocks and distributed graph context.
4. **Stabilize the latent.** Start with L1 prediction, a scheduled EMA, collapse
   diagnostics, and the 40-dimension minimum state budget already established.
   Compare continuous latents against a 32–64 prototype soft codebook plus a
   small reconstruction/state-probe anchor.
5. **Train at two resolutions.** Predict fine transitions and coarse regimes.
6. **Evaluate frozen representations.** Fit identical lightweight probes on
   raw, PCA, and learned features for future state, per-subsystem state
   recovery, and schedule-family transfer. Retain the local, shuffled-topology,
   and all-entity ablations.

Advance only if the learned representation:

- beats the frozen PCA one-hop development baseline, not merely persistence;
- has healthy effective rank and code usage across all seeds;
- retains recoverable node/edge state for attribution;
- benefits consistently from true topology over shuffled topology; and
- transfers across held-out schedule families.

If this model still cannot beat PCA after a small preregistered sweep over
masking, continuous versus codebook latent, and one versus two resolutions,
stop investing in JEPA for nominal representation learning. Use a supervised
temporal graph/state-space model for attribution and revisit JEPA only when
substantially more varied or intervention-rich data exists.

## Verdict

JEPA is **plausibly applicable to structured software telemetry**, and recent
time-series results make it premature to reject. JEPA is **not established for
raw software logs alone**, and our corpus is too narrow and repetitive to
justify treating record count as state diversity.

The next experiment should be viewed as the final serious test of
action-free JEPA pretraining, not another incremental width tweak. If it passes,
the subsequent disturbance/action/recovery phase can test a genuinely
constrained software world model. If it fails, we will have tested the
important modality-specific ideas from the literature and can pivot with a
clean negative conclusion.

## Complementary NLP and ML techniques

JEPA should be the latent prediction objective, not the entire system. Several
other techniques can supply better observations, stronger inductive biases,
and auxiliary supervision.

### Log-template parsing and typed parameter extraction

[Drain](https://pinjiahe.github.io/files/pdf/research/ICWS17.pdf) converts raw
messages into stable templates using an online fixed-depth parse tree. For our
lab, the representation of a log event should contain:

- template ID and logger/source;
- emitting node and operation;
- severity and outcome;
- typed parameters such as duration, queue depth, attempt number, status code,
  and dependency name;
- trace/span/request correlation; and
- timestamp and time since the previous related event.

This is preferable to treating the whole rendered line as natural-language
text. The template identifies the event type, while typed parameters preserve
the operational quantities that generic sentence embeddings tend to blur.

### Masked event modeling

[LogBERT](https://arxiv.org/abs/2103.04475) demonstrates self-supervised masked
modeling over normal log sequences. A small log transformer can be pretrained
to predict masked template IDs, outcome fields, and parameter buckets.

This should be an auxiliary objective, not the world model itself. It teaches
the event encoder program-flow regularities while JEPA learns which event
information predicts the wider system state.

Useful masked tasks include:

- masked template or operation;
- masked emitting entity;
- masked outcome/status class;
- masked duration or count bucket; and
- missing span between two trace-correlated events.

Exact free-text reconstruction is low priority because it rewards incidental
wording and identifiers.

### Continuous-time event modeling

Logs are irregular event streams, not dense time series. The
[Neural Hawkes Process](https://arxiv.org/abs/1612.09328) jointly models which
event occurs next and when it occurs, allowing earlier events to excite or
inhibit later event types.

Add an event-time head to the log encoder:

- next-event type likelihood;
- time-to-next-event likelihood; and
- optionally, event intensity per subsystem or operation.

The point-process state becomes a log token supplied to the graph JEPA. This
preserves burstiness, silence, retries, and delayed completions without
arbitrarily turning every log into a fixed-bin count. A simpler implementation
can begin with template classification plus log-time or mixture-density
prediction before adopting a full point-process likelihood.

### Hierarchical contrastive or redundancy-reduction objectives

[TS2Vec](https://ojs.aaai.org/index.php/AAAI/article/view/20881) learns
timestamp and subsequence representations through hierarchical contrastive
learning across overlapping views.
[Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748) learns
predictive representations by distinguishing the actual future latent from
negative samples. These objectives can complement JEPA by making schedule,
regime, and time-local structure separable.

For our repetitive normal runs, naive negatives are risky: two windows from
different runs may represent the same state. Prefer:

- positives defined by the same trace, request, or aligned metric/log window;
- negatives from known different regimes or interventions;
- hierarchical positives at fine and coarse resolutions; or
- [VICReg-style](https://openreview.net/pdf?id=IXexLXymbZ9)
  variance/covariance regularization when trustworthy negatives are
  unavailable.

The contrastive term should be small enough that it structures the latent
without overriding the JEPA notion of predictable equivalence.

### Relational graph learning

[Graph Networks as Learnable Physics Engines](https://arxiv.org/abs/1806.01242)
shows how object- and relation-centric graph networks can serve as forward
dynamics models and generalize across system structure. For software, the
analogues of objects and relations are services/resources and calls,
queues, writes, locks, and ownership edges.

Use relation-specific message functions or graph attention rather than only
concatenating declared neighbors. Candidate learned messages include:

- demand and latency propagation along request edges;
- backlog propagation along enqueue/dequeue edges;
- contention propagation along dependency edges; and
- recovery signals propagating from restarted dependencies.

The declared topology should remain an input prior. The model can learn edge
strength or attention, but unconstrained causal-edge discovery from 72 nominal
runs is not identifiable enough to replace the declared graph.

### Process mining as a symbolic teacher

[Process discovery](https://vdaalst.com/publications/p711.pdf) extracts process
models such as Petri nets from timestamped event traces. Our trace-correlated
checkout events can yield coarse symbolic states such as accepted, enqueued,
dequeued, dependency-called, committed, completed, retried, or failed.

Those states can help in three ways:

- weak labels for event-encoder pretraining;
- conformance-violation features supplied to the learned model; and
- an interpretable baseline against which latent transitions are compared.

This hybrid is attractive because process mining handles discrete workflow
order explicitly, while JEPA handles continuous load, latency, and hidden
regime information.

### Discrete regime and switching-state models

The SC-JEPA soft codebook can be understood as a learned regime model. It is
closely aligned with the operational reality that the stack moves among a
finite set of recognizable states: idle, healthy-throughput, queue-building,
worker-saturated, dependency-degraded, retrying, draining, and recovering.

We should compare three alternatives:

- continuous JEPA latent;
- soft prototype/codebook JEPA latent; and
- a classical hidden or switching-state baseline over the same observations.

If the classical model wins, that is still scientifically valuable: it means
our lab is better described by explicit regime transitions than a smooth
high-dimensional manifold.

### Probabilistic transitions and calibrated uncertainty

A deterministic predictor tends to average multiple valid futures. Software
systems contain scheduler jitter, races, retry timing, batching, and other
stochastic outcomes. Predict either:

- a distribution over regime codes;
- a mixture over future latent states;
- quantiles for continuous node state; or
- an ensemble mean and epistemic uncertainty.

Uncertainty should be evaluated for calibration and used to abstain from
attribution when the observations do not identify one dependable explanation.
This is more honest than forcing every future into one latent vector.

### Trace-aligned multimodal learning

Trace/span IDs give much stronger supervision than temporal coincidence.
Metric, log, and span tokens belonging to the same request or operation should
be aligned through cross-attention or a small cross-modal matching loss. This
can teach, for example, that a Redis timeout event, a long dependency span, and
a queue increase are views of one evolving episode.

Keep modality-specific encoders:

- continuous metric encoder;
- structured log/event encoder;
- trace/span graph encoder; and
- topology-aware fusion encoder.

Fuse them after each has preserved its native timing and structure. Early
concatenation makes dense metrics dominate sparse events.

### Retrieval for attribution, not representation training

Store latent episodes with their raw metric/log/trace evidence. At inference,
retrieve nearby historical episodes and report both the learned subsystem
attribution and concrete precedents. Retrieval will not fix a bad latent, but
it makes a good latent auditable and exposes when a query lies outside the
training distribution.

## Recommended blend and ablation order

Do not introduce every component simultaneously. The highest-probability
sequence is:

1. **Structured event layer:** template IDs, typed parameters, and trace/span
   correlation.
2. **Native modality encoders:** a patched metric encoder and a
   transformer-plus-event-time log encoder.
3. **Relational fusion:** a temporal graph transformer using typed nodes and
   edges.
4. **Training objectives:** masked event modeling, multi-mask JEPA prediction,
   local state recovery, and variance/effective-rank regularization.
5. **Regime ablation:** continuous latent versus soft codebook versus a
   classical switching-state baseline.
6. **Probabilistic action model:** only after nominal representation passes,
   learn intervention-conditioned transition distributions.

A compact objective is:

`JEPA latent prediction + masked event loss + event-time likelihood + local
state probe + anti-collapse regularization`

Add cross-modal contrastive alignment only when positives and negatives can be
defined from trace identity or known regimes. Add free-text language embeddings
last, and retain them only if they improve held-out schedule and intervention
performance beyond the template-and-parameter representation.
