# JEPA frontier technique audit, July 2026

## Decision

The runnable queue identified by this audit is now complete. HEPA, complete
SC-JEPA, CF-JEPA, SD-JEPA, Delta-JEPA, exact LeWorldModel plus its bounded
geometry matrix, Causal-JEPA, and MoP-JEPA were all implemented and rejected
under their frozen gates. See the
[execution conclusion](jepa-frontier-execution-conclusion-2026.md).

The remaining work is not “try every paper with JEPA in the title.” Many
papers are modality or backbone ports of the same objective. The scientifically
useful queue is the set of materially different mechanisms: a different
training signal, target construction, collapse constraint, latent geometry,
predictive distribution, or deployment value path.

The completed execution order was:

1. HEPA horizon-conditioned event prediction;
2. the complete SC-JEPA codebook-plus-multi-resolution interaction;
3. CF-JEPA mask-free multi-horizon forward prediction;
4. SD-JEPA progression/content subspaces;
5. Delta-JEPA latent-difference action decoding;
6. exact LeWorldModel with a controlled ambient/subspace geometry screen;
7. Causal-JEPA entity-trajectory interventions;
8. uncertainty-aware joint embeddings, beginning with MoP-JEPA; and
9. lower-priority channel, cross-modal, physics, and belief variants remain
   conditional on their stated prerequisites.

This audit covers primary sources available through 2026-07-28. Claims below
about telemetry fit are inferences from those sources and the Quantis results,
not claims made by the paper authors.

## What Quantis has already tested

The repository already contains reproducible code and negative evidence for:

- canonical temporal/multimodal EMA-target masked latent prediction;
- fine/coarse and current/future target views;
- entity, time, and graph-neighborhood masks;
- variance/covariance anti-collapse and observable-state recovery;
- action-conditioned direct multi-horizon latent dynamics and low-rank
  transitions;
- a raw-state-preserving residual JEPA correction;
- an SC-JEPA-inspired soft regime codebook with fine/coarse context summaries;
- a separate masked hybrid JEPA with fine/coarse target views;
- event-native template/entity/outcome/time-to-next prediction;
- finite multi-hypothesis latent trajectories;
- official-formula SIGReg substituted into the existing EMA residual model;
- complete predictor-free multi-view LeJEPA;
- episode retrieval-JEPA; and
- a cross-stack representation contract, retained as an audit artifact but
  out of scope because the product deliberately targets one small stack.

Those experiments rejected exact recipes, not the whole JEPA family. The
recurring local failure was a learned latent bottleneck that lost
entity-local state or downstream action effect while the raw rank-32
contractive model remained strong. That observation determines the ordering
below.

## Omitted mechanisms that are runnable now

| Priority | Mechanism | What is materially new | Local overlap | Disposition |
|---|---|---|---|---|
| 1 | [HEPA](https://arxiv.org/abs/2605.11130) | Log-uniform horizon-conditioned future-interval latent prediction, followed by a frozen encoder and a finetuned predictor that emits a monotone discrete-time survival CDF | Quantis has future-latent prediction and alert scoring, but never this two-stage event-time objective or survival output | Run first |
| 2 | [SC-JEPA](https://arxiv.org/abs/2602.04643) | A soft regime codebook and explicitly separate fine/coarse future-prediction objectives are trained together | Quantis tested a codebook with fine/coarse context summaries and, separately, fine/coarse targets; it did not test their claimed joint interaction | Run the complete factorial after HEPA |
| 3 | [CF-JEPA](https://arxiv.org/abs/2606.07031) | Mask-free random crops with short-, middle-, and long-horizon forward targets; online and EMA encoders are routed to different downstream roles | Quantis used masking and EMA, but did not test mask-free forward crops or evaluate the online/EMA asymmetry | Run after SC-JEPA |
| 4 | [SD-JEPA](https://arxiv.org/abs/2605.31111) | Orthogonal progression and content subspaces; a cosine-margin triplet objective gives progression an explicit coordinate while SIGReg acts on content | No explicit progression coordinate has been tested | Run; its event-localization output is directly relevant |
| 5 | [Delta-JEPA](https://arxiv.org/abs/2606.31232) | An inverse decoder reconstructs the action sequence from the latent displacement \(z_{t+H}-z_t\), making transition geometry action-sensitive without reconstruction | Quantis conditioned predictors on action, but never forced action to be recoverable from displacement alone | Run in the action/investigation lane |
| 6 | [LeWorldModel](https://arxiv.org/abs/2603.19312) | End-to-end action-conditioned next-latent prediction plus SIGReg, with no EMA teacher, stop-gradient, or separately pretrained encoder | Neither the SIGReg substitution nor complete LeJEPA tested this exact world-model recipe | Run as a small controlled matrix |
| 6a | [Sub-JEPA](https://arxiv.org/abs/2605.09241) | Gaussian matching in random lower-dimensional subspaces rather than forcing the ambient latent to be isotropic | Exact subspace SIGReg is untested and directly addresses low intrinsic entity rank | Pair with LeWorldModel |
| 6b | [UR-JEPA](https://arxiv.org/abs/2606.01443) | A uniform-rectifiability regularizer targets a locally low-dimensional manifold | Untested; plausible response to the same local rank problem | Screen only if Sub-JEPA is promising |
| 6c | [Rectified LpJEPA](https://arxiv.org/abs/2602.01456) | Rectified distribution matching creates sparse, non-negative, maximum-entropy representations | Untested; sparsity may help edge inference and localization | Same geometry screen |
| 6d | [KerJEPA](https://arxiv.org/abs/2512.19605) | Kernel/prior choices generalize LeJEPA's sliced Gaussian discrepancy | Untested, but primarily a training-geometry choice | Do not create a separate pipeline |
| 7 | [Causal-JEPA](https://arxiv.org/abs/2602.11389) | Whole object trajectories are masked except for an identity anchor, forcing recovery from other objects and auxiliary variables | Entity/time masks overlap, but not this intervention-shaped whole-entity trajectory target | Run with services as objects |
| 8 | [MoP-JEPA](https://arxiv.org/abs/2607.05238) | Hard best-of-\(K\) predictor heads model several futures without iterative sampling | Quantis tried a likelihood-trained mixture, not winner-take-all predictor specialization | First stochastic tracer |
| 9 | [T-JEPA](https://arxiv.org/abs/2410.05016) | Arbitrary feature-subset targets and learned regularization tokens for tabular observations | Entity/time masking does not test missing-channel robustness | Run only in a missing-telemetry lane |
| 10 | [V-JEPA 2.1](https://arxiv.org/abs/2603.14482) | Visible-token prediction and deep intermediate-layer self-supervision improve dense localization | Current-state recovery overlaps, but not deep supervision | Low-cost localization ablation |
| 11 | [BiJEPA](https://arxiv.org/abs/2603.00049) | Bidirectional prediction and cycle consistency, stabilized by norm regularization | Quantis predicts forward only | Conditional; backward predictability may encode recovery |

### Why HEPA is first

HEPA is unusually close to the actual Quantis deployment question. Its primary
output is \(P(\text{event within }\Delta t)\), not a latent loss that must later
be justified through an alert heuristic. It was evaluated on spacecraft
telemetry, server metrics, industrial-control attacks, process faults, and
lifecycle prediction. The published model is 2.16 million parameters and
finetunes only about 198 thousand parameters after pretraining.

Its mechanism is also distinct from the Quantis LeJEPA work:

- a causal context encoder predicts a future *interval* embedding;
- prediction horizons are sampled log-uniformly;
- the context and target paths share weights and are jointly trained;
- L1 latent prediction is mixed with SIGReg instead of using an EMA teacher;
- the encoder is frozen after pretraining;
- the horizon-conditioned predictor, not just a probe, is finetuned; and
- per-horizon hazards are composed into a monotone survival CDF.

There are two cautions. First, the paper reports that its probability surfaces
can be poorly calibrated, so calibration cannot be inferred from AUROC.
Second, the official
[HEPA repository](https://github.com/Forgis-Labs/HEPA) is published under a
non-commercial share-alike license. A deployable Quantis implementation should
be cleanly implemented from the paper or separately licensed; the repository
may be used as a scientific reference, not copied into production code.

## First tracer contract: HEPA telemetry

### Frozen role

The candidate is an **alert-policy adapter**, not a replacement predictive
core. The raw rank-32 action-conditioned low-rank model remains the predictive
and investigation reference.

The tracer uses only the existing open action-dynamics corpus and its frozen
topology-transfer roles. It opens no sealed data. Stage-one pretraining and
stage-two event-head fitting use fitting trajectories only. Selection chooses
the checkpoint and fixed hyperparameter option. Calibration is used only for
probability calibration and the alert threshold. Both evaluation roles remain
untouched until assessment.

The event is defined without action identity: using fitting control
trajectories only, fit the already-declared normalized downstream-effect norm
and freeze its event threshold. For each context \(t\) and horizon \(h\),
\(y(t,h)=1\) exactly when the trajectory's first threshold crossing occurs in
\((t,t+h]\). Action and target truth are never model inputs in the alert lane.
The assessor may use them only for the existing post-onset and per-action
breakdowns.

### Hypothesis

> On the existing held-out worker topology, horizon-conditioned future-interval
> JEPA pretraining followed by frozen-encoder predictor finetuning yields a
> useful monotone event-time distribution and improves trajectory-level alert
> detection over an alignment-broken JEPA at the same control-trajectory
> false-alarm budget.

### JEPA-specific null

Use a **whole-trajectory horizon-deranged null**:

- identical encoder, predictor, SIGReg, event head, parameter count, optimizer,
  training steps, seeds, and stage-two labels;
- during stage-one only, replace each future target interval with a
  length- and horizon-matched interval from another fitting trajectory;
- keep all derangement atomic at the logical-pair level; and
- never derange across roles.

This preserves future-target marginal statistics and training compute while
breaking the context-to-future predictive relationship that defines HEPA.
Also report the existing raw alert reference and a capacity-matched
supervised-from-scratch survival model, but neither replaces the
JEPA-specific null.

### Minimal exact treatment

- causal two-layer telemetry transformer;
- entity-preserving tokens at the public encoding seam, with pooling only
  inside the predictor/head;
- one shared, jointly optimized context/target encoder;
- future cumulative interval targets;
- log-uniform horizon sampling over the frozen alert horizon;
- L1 prediction plus SIGReg with the paper's `alpha = 0.1`;
- no EMA teacher and no stop-gradient;
- frozen encoder in stage two;
- horizon-conditioned predictor plus shared hazard head;
- cumulative survival CDF
  \(p(t,h)=1-\prod_{j=1}^{h}(1-\lambda_j(t))\); and
- training-fitted post-hoc calibration selected without evaluation data.

No mask family, graph message-passing layer, action decoder, retrieval index,
or second regularizer is added to this tracer.

### Safety and value gates

All shared ladder gates remain mandatory. In addition:

1. every restored CDF is finite, lies in \([0,1]\), and is non-decreasing in
   horizon;
2. the clean treatment and deranged null have matched inference parameter
   counts and differ only in the stored stage-one target alignment;
3. observed-entity frozen-probe retention remains within the preregistered
   matched-PCA margin, with every varying entity reported;
4. after calibration, Brier score is no worse than `1.05 ×` the
   horizon-deranged null, and ECE is reported rather than hidden;
5. control-trajectory false alarms are at most 5%;
6. treatment-trajectory post-onset detection is at least 80%;
7. median post-onset delay is at most 10 transitions;
8. at the same at-most-5% false-alarm budget, treatment-trajectory detection is
   at least 10 percentage points higher than the horizon-deranged null;
9. the serialized candidate plus sidecars is at most 16 MiB and all batch-one
   CPU and memory diagnostics are recorded; and
10. serialization/restoration reproduces representations, probability
    surfaces, calibrated outputs, and alert decisions within the frozen
    deterministic tolerance.

Passing all gates creates a one-seed alert representation candidate and
authorizes the existing fixed-seed robustness step. It does not authorize
sealed collection or production paging. Failure rejects this exact HEPA
telemetry recipe and preserves its implementation and artifacts.

## Next tracers after HEPA

### SC-JEPA: test the interaction we have not tested

[SC-JEPA](https://arxiv.org/abs/2602.04643) combines two ideas that Quantis
tested only in partially overlapping prototypes: a soft regime codebook and
separate fine/coarse future-prediction objectives. The retained codebook
tracer summarizes its *context* at fine and coarse scales but encodes every
future state at one resolution. The hybrid graph tracer predicts fine and
pooled coarse targets but has no codebook. A four-cell factorial—continuous
single-resolution, continuous multi-resolution, codebook single-resolution,
and full codebook multi-resolution—can therefore isolate whether the
published interaction adds value rather than rerunning either rejected
component by itself.

### CF-JEPA: remove masking

[CF-JEPA](https://arxiv.org/abs/2606.07031) is the cleanest test of whether
masking itself harmed temporal continuity in the earlier Quantis work. It
predicts short-, medium-, and long-horizon future crops and reports a useful
online/EMA encoder asymmetry: online features were more discriminative while
EMA features were smoother and better for forecasting/anomaly detection.
The null should restore the prior masked target construction while matching
all other capacity and horizons.

### SD-JEPA: give surprise a direction

[SD-JEPA](https://arxiv.org/abs/2605.31111) splits the latent into a small
progression subspace and a larger SIGReg content subspace. Its angular
progression change localized semantic events better than scalar prediction
error in the reported control tasks. For Quantis, freeze progression as
distance from normal operation to downstream impact and use content for
service state. The null is the same total width with no orthogonal split and
no progression triplet loss.

### Delta-JEPA: make action effect recoverable

[Delta-JEPA](https://arxiv.org/abs/2606.31232) directly targets Quantis's
learned-bottleneck failure. Its latent-difference action decoder sees only
\(z_{t+H}-z_t\) and reconstructs the intervening action sequence. The primary
null is the paper's capacity-matched endpoint-concatenation decoder
\([z_t,z_{t+H}]\). Advance only if held-topology downstream-effect MSE improves
by at least 10% over both that null and raw low-rank while the shared 5%
overall/action-overlap bounds, 95% hit@1, and 100% no-action specificity pass.

### Exact LeWorldModel and geometry screen

[LeWorldModel](https://arxiv.org/abs/2603.19312) should be implemented before
concluding that SIGReg world models do not work here. Run one treatment matrix
with the same end-to-end action-conditioned predictor:

- ambient SIGReg, the exact LeWorldModel treatment;
- [Sub-JEPA](https://arxiv.org/abs/2605.09241) random-subspace SIGReg;
- [UR-JEPA](https://arxiv.org/abs/2606.01443) only if both reveal the same
  low-intrinsic-rank tension; and
- the raw low-rank and prediction-only matched controls.

[Rectified LpJEPA](https://arxiv.org/abs/2602.01456) and
[KerJEPA](https://arxiv.org/abs/2512.19605) belong in this screen, not in
separate end-to-end programs. These mechanisms change training geometry and
add essentially no inference cost. The screen may also include
[SPHERE-JEPA](https://arxiv.org/abs/2605.26900), which targets a uniform
hypersphere, and
[deterministic spherical discrepancies](https://arxiv.org/abs/2606.17603),
which replace stochastic projections with MMD, KSD, or KL objectives. They
must share one architecture and selection budget; otherwise a regularizer
sweep becomes an unbounded model search.

### Causal-JEPA: services as interacting objects

[Causal-JEPA](https://arxiv.org/abs/2602.11389) is not the similarly named
contrastive C-JEPA. It masks selected object trajectories across history and
future while retaining an identity anchor, so the model must infer their
states through other objects and auxiliary variables. This is a materially
different intervention from Quantis's contiguous entity/time masks. It becomes
high priority if HEPA succeeds globally but still fails per-service
localization.

## Conditional and blocked families

### Probabilistic futures

Quantis's failed finite-mixture experiment does not cover all probabilistic
JEPAs. The remaining ladder is:

1. [MoP-JEPA](https://arxiv.org/abs/2607.05238), hard best-of-\(K\) predictor
   specialization;
2. [Gaussian/mixture joint embeddings](https://arxiv.org/abs/2603.26799),
   closed-form conditional embedding distributions;
3. [VJEPA/BJEPA](https://arxiv.org/abs/2601.14354), a predictive belief
   distribution and modular product-of-experts prior;
4. [Var-JEPA](https://arxiv.org/abs/2603.20111), an explicit coupled ELBO;
5. [JEDI](https://arxiv.org/abs/2605.13013), iterative latent diffusion; and
6. [UWM-JEPA](https://arxiv.org/abs/2605.25313), density-matrix beliefs with a
   unitary predictor.

MoP-JEPA is runnable and edge-plausible. Variational families are conditional
on demonstrating heteroscedastic residual structure that improves a proper
score over calibrated deterministic residuals. JEDI and UWM-JEPA are deferred:
the current corpus has not shown enough irreducible multimodality to justify
iterative sampling or quadratic belief-state machinery at the edge.

### Cross-modal and cross-stack

- [MJEPA](https://arxiv.org/abs/2606.25225) explicitly predicts within and
  across modalities. Quantis fused metrics and event tokens but did not test
  metrics-to-events and events-to-metrics targets. It is runnable, but prior
  events added negligible value and trace links are incomplete.
- [CHARM](https://arxiv.org/abs/2605.31580) uses channel descriptions and
  channel-order equivariance for cross-dataset generalization. It requires
  meaningful descriptions and multiple independent stacks; the current
  seven-record, one-stack corpus blocks its claimed value.
- [VL-JEPA](https://arxiv.org/abs/2512.10942) predicts continuous language
  embeddings. It requires aligned operator language, incidents, or tickets
  that Quantis does not yet have.
- A frozen domain-teacher recipe such as
  [US-JEPA](https://arxiv.org/abs/2602.19322) is blocked because no trustworthy
  pretrained telemetry teacher exists.

### Dynamical and structural priors

- [Koopman-invariant JEPA](https://arxiv.org/abs/2511.09783) shows that a
  near-identity linear predictor can select regime-indicator eigenfunctions.
  It is cheap and interpretable, but Quantis already rejected a regime
  codebook; run it only if HEPA reveals separable precursor regimes.
- [Phys-JEPA](https://arxiv.org/abs/2606.16076) separates physical and residual
  latent state and applies constraints to latent transitions. It is applicable
  only where queue conservation, flow balance, or another invariant is
  trustworthy and fully observed.
- [Fast-LeWorldModel](https://arxiv.org/abs/2606.26217) predicts all
  action-prefix futures in parallel. Quantis already predicts direct
  multi-horizon trajectories, so this is mostly an efficiency/rollout
  ablation rather than a new primary hypothesis.
- [Temporal-Distance JEPA](https://arxiv.org/abs/2607.25337) learns a directed
  progress cost from trajectory order, negatives, and rollout consistency.
  It is a planning objective; for alerting, SD-JEPA's explicit progression
  coordinate is the narrower first test.
- [BiJEPA](https://arxiv.org/abs/2603.00049) adds backward prediction and cycle
  consistency. It is relevant only if recovery trajectories are sufficiently
  represented.
- [seq-JEPA](https://arxiv.org/abs/2505.03176) explicitly separates
  action-equivariant from action-invariant state. It overlaps Delta-JEPA's
  action-sensitivity question and should follow, not precede, that simpler
  falsifier.
- [temporal straightening](https://arxiv.org/abs/2603.12231) regularizes
  trajectory curvature. It is low priority because Quantis is not deploying a
  latent planner.

### Lower-priority target construction

- [DMT-JEPA](https://arxiv.org/abs/2405.17995) forms targets by aggregating
  semantically similar local neighbors. A telemetry analogue would pool
  causal-graph neighbors, but prior graph-context evidence was negative.
- [DSeq-JEPA](https://arxiv.org/abs/2511.17354) predicts targets in a learned
  salience order rather than in parallel. Alert spikes can dominate such an
  ordering, so it needs a salience-deranged null before consideration.
- [P-JEPA](https://arxiv.org/abs/2606.23256) pools long procedural sequences
  before masked latent prediction. Current Quantis windows do not contain the
  long procedural structure that motivates it.
- [ER-JEPA](https://arxiv.org/abs/2607.01145) composes interval-level and
  sequence-level prediction hierarchically. Quantis tested parallel
  fine/coarse views, not this compositional hierarchy, but the evidence is
  currently limited to an application-specific ECG setting.
- [NextLat](https://arxiv.org/abs/2511.05963) adds next-latent prediction to an
  autoregressive sequence model to encourage a compact belief state. The
  event-native tracer already combines event targets with future-latent
  prediction, so only a much longer event stream would make this distinct.

## Techniques that should not become separate experiments

The following are implementations or domain ports, not independent Quantis
hypotheses:

- I-JEPA/V-JEPA mask-ratio changes and larger Transformer backbones;
- Audio-JEPA, Point-JEPA, LiDAR-JEPA, skeleton, ECG, brain, particle, and
  other modality-specific tokenizers without a new objective;
- Graph-JEPA's hyperbolic hierarchy coordinate when the deployment target is
  temporal alerting rather than graph classification;
- alternative positional encodings, patch sizes, or attention kernels;
- predictor depth and width sweeps; and
- a different implementation of an objective whose exact mechanism has
  already been rejected.

[JEPA-SCORE](https://arxiv.org/abs/2510.05949) is a distinct value path, but
not a new trainer: it derives an implicit density score from encoder Jacobian
singular values. It may be useful for offline drift analysis. Exact
Jacobian/SVD work is unlikely to be an edge-event scorer unless it first
passes a separate latency feasibility test.

## Conclusion

The omission was real and bounded; the runnable set is now preserved as
reproducible tracers and negative evidence. None should advance. Cross-stack
semantic methods remain blocked by data, while diffusion, density-matrix, and
Jacobian-score approaches should not consume edge implementation effort until
their recorded prerequisites establish a predictable residual, distinct
future, or alert benefit.
