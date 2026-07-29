# JEPA frontier exhaustion refresh, 2026-07-29

## Answer

Quantis has **not literally exhausted every known JEPA model or technique**.
It has exhausted the materially distinct experiments in its previously frozen
runnable queue, and it has unusually broad negative coverage of masking,
target-encoder choices, temporal horizons, action conditioning, latent
geometry, retrieval, event-time prediction, and finite multi-future heads.
That supports the existing decision not to deploy a tested JEPA.

It does **not** support the stronger statement that every objective-level
alternative has been tried. A primary-source refresh identifies three
one-stack mechanisms that were understated or absent from the prior audit:

1. jointly training the representation with a task/value auxiliary that
   declares which state distinctions must survive;
2. selecting targets around rare interactions or intervention effects rather
   than sampling ordinary time uniformly; and
3. using a separately trained, frozen reconstructive target teacher instead
   of an EMA or jointly optimized target.

All three are runnable on the deliberately small stack. None requires adding
independent stacks or pretending the system is conditionally multimodal.

## Exact coverage boundary

The retained matrix already covers the major generic JEPA axes:

- EMA, non-EMA, and predictor-free target paths;
- masked and mask-free forward prediction;
- one-step and direct multi-horizon targets;
- entity, time, graph-neighborhood, fine/coarse, and whole-trajectory masks;
- reconstruction/state-recovery auxiliaries;
- action-conditioned prediction, action decoding from latent displacement,
  and raw-state-preserving residual correction;
- SIGReg, subspace, rectified, kernel, hyperspherical, and deterministic
  spherical regularization;
- event-time survival output, retrieval, soft codebooks, likelihood mixtures,
  and hard winner-take-all mixtures.

See the [execution conclusion](jepa-frontier-execution-conclusion-2026.md) and
[prior audit](jepa-frontier-technique-audit-2026.md). This is enough to reject
routine modality ports, mask-ratio changes, wider predictors, and another
anti-collapse regularizer as separate scientific programs.

## Material known mechanisms still untested

### 1. Decision-anchored or value-grounded JEPA — high priority

[Why and How Auxiliary Tasks Improve JEPA Representations](https://arxiv.org/abs/2509.12249)
proves, for deterministic MDPs, that a jointly trained auxiliary regression
head can force a JEPA to preserve distinctions induced by transition dynamics
or the auxiliary value. A later controlled study found that reward-free
predictive, action-conditioned, controllability, and inverse-dynamics
objectives all discarded an exogenous but control-relevant feature, while a
small amount of reward grounding recovered it
([primary paper](https://arxiv.org/abs/2606.30068)).
A bisimulation-augmented JEPA similarly enforces control-relevant state
equivalence rather than mere temporal predictability
([primary paper](https://arxiv.org/abs/2602.18639)).

Quantis trained reconstruction and cross-modal auxiliary heads, and HEPA
finetuned an alert head after freezing its encoder. It did **not** jointly
train an encoder with the actual bounded alert/effect decision as the
auxiliary equivalence relation. This is the most direct known response to the
observed failure: predictive latents repeatedly discarded information that
raw telemetry retained.

This is runnable with the existing fitting-role intervention labels. It is no
longer a purely self-supervised representation experiment, so it must be
compared with a capacity-matched supervised alert/effect model. JEPA earns a
role only if the predictive auxiliary improves transfer beyond that control.

### 2. Interaction-aware target selection — high priority

[Interaction-Aware JEPA](https://arxiv.org/abs/2605.15466) changes the
training signal by preferentially masking entities and moments involved in
interactions instead of allowing common static content to dominate. That is a
materially different target sampler, not a backbone port.

Quantis tried whole-entity masking, action-conditioned prediction, event
targets, and progression coordinates, but did not use the matched
control/treatment effect itself to select and pair predictive targets. The
existing intervention onsets and action windows are enough for a
small-stack tracer. A label-free motion/change-score sampler and an
action-onset sampler should be separate cells so target-label leakage is
visible.

### 3. Static target teacher (SALT) — medium priority

[SALT](https://arxiv.org/abs/2509.24317) first trains a target encoder with
masked observation reconstruction, freezes it, and only then trains a student
to predict its target latents. This decouples target semantics from
student/teacher co-adaptation.

Quantis has tested EMA targets, jointly optimized SIGReg encoders, frozen
encoders in downstream stages, and a masked-autoencoder control. It has not
tested the exact two-stage combination “reconstructive teacher, then frozen
latent-prediction student.” It is edge-runnable because the extra decoder is
training-only. Its likely ceiling is modest here: masked reconstruction already
beat complete LeJEPA but still lost badly to raw dynamics.

**Executed result:** rejected. The evidence-review-corrected v2 run passed all
eleven protocol checks. Aligned masked-target prediction improved over
derangement by 12.04% on selection but only 8.89% on transfer. The student
retained `1.91×` raw held-topology downstream-effect MSE. See the
[retained result](salt-jepa-telemetry-v2-results.md).

### 4. LeNEPA's disposable prediction projection — lower priority

[LeNEPA](https://arxiv.org/abs/2607.00958) uses a causal, no-augmentation
next-latent objective, SIGReg, and a lightweight projected prediction space
that is discarded for evaluation. Quantis covered no-mask prediction,
next/multi-horizon prediction, and end-to-end SIGReg separately, but not this
exact routing of the predictive loss away from the deployed backbone.

This is an exact recipe omission, although its scientific overlap with
CF-JEPA and LeWorldModel is large. It should follow decision anchoring and
paired-effect training, not precede them.

### 5. Discrete-JEPA — material tokenizer/objective omission

[Discrete-JEPA](https://arxiv.org/abs/2506.14373) learns discrete semantic
tokens with complementary latent-prediction objectives rather than treating a
soft prototype codebook only as a continuous regularizer. Quantis's SC-JEPA
regime codebook and event-native path grammar overlap with its motivation, but
neither is an exact hard semantic-token JEPA.

This is runnable with existing event templates, but the current event-native
experiment found no alert or investigation gain and the corpus may be too
small to identify stable discrete semantics. It is a real omission with lower
expected value than paired-effect or decision-grounded training.

### 6. PEIRA — material predictor-free objective omission

[PEIRA](https://arxiv.org/abs/2605.17671) defines its objective through the
trace of an optimal regularized inter-view linear regressor and targets
nonlinear canonical-correlation subspaces. This is not EMA prediction,
invariance plus SIGReg, or a different kernel choice.

It is training-only and therefore edge-runnable. Quantis already found that
the complete predictor-free LeJEPA representation was state-accessible but
operationally inferior to raw dynamics, so PEIRA should be treated as a
bounded representation ablation rather than a likely alert system.

**Executed result:** rejected. The aligned cell learned a real non-collapsed
trace mechanism but lost to derangement and reconstruction controls, retained
1.91× raw transfer effect error, and won three of ten pairs. See the
[retained result](peira-telemetry-v1-results.md).

### 7. VISReg — exact regularizer omission, low value priority

[VISReg](https://arxiv.org/abs/2606.02572) combines an explicit variance floor
with a sliced-Wasserstein distribution sketch, separating scale control from
full-distribution shape matching. The geometry screen did not contain this
exact objective.

It is runnable and adds no inference cost, but Quantis already showed that
several active changes in latent rank and geometry did not restore state or
effect value. VISReg is therefore a bounded ablation, not a new program.

**Executed result:** rejected. The exact small-radius gradient mechanism
passed, but the detached candidate collapsed to projector rank 1.12, retained
1.97× raw transfer effect error, and lost to no-detach and reconstruction
controls. See the [retained result](visreg-telemetry-v1-results.md).

### 8. JEPA-SCORE — executed value path, not a trainer

[JEPA-SCORE](https://arxiv.org/abs/2510.05949) derives a local density score
from encoder Jacobian singular values. It is a genuinely different way to use
a fitted encoder, but it does not repair the predictive representation.
The frozen screen resolved the latency concern without approximation.

**Executed result:** rejected for alerting, retained for feasibility. Exact
full-Jacobian/SVD scoring passed every protocol and edge gate at 51.4 ms
median and 60.2 ms p95, but won 40% of selection pairs, detected 10% of IID
treatments, and detected no transfer treatments. See the
[retained result](jepa-score-edge-screen-v1-results.md).

## Known mechanisms that do not become small-stack alert experiments

- [INTACT](https://arxiv.org/abs/2607.26056), published after the prior
  refresh was assembled, learns a search-free intent-to-action law. It is a
  material new control interface, but Quantis is evaluating alerts rather
  than choosing goal-directed actions.
- [HamJEPA](https://arxiv.org/abs/2605.20107) and homomorphic latent dynamics
  ([primary paper](https://arxiv.org/abs/2603.20048)) impose symplectic or Lie
  structure. They need a defensible phase-space or group action, which the
  current telemetry contract does not provide.
- [EPM-JEPA](https://arxiv.org/abs/2606.12979) modulates predictor weights
  from an online experience buffer. It needs a declared online-shift and
  update-safety lane; its own preregistered comparison was a null result.
- [D-JEPA](https://arxiv.org/abs/2410.03755), JEDI, variational beliefs, and
  density-matrix JEPAs model generative or stochastic futures. A deliberately
  small stack does not itself establish multiple valid futures, and their
  extra machinery is not justified by the current deterministic residuals.
- Siamese students, deeper intermediate supervision, frozen visual
  backbones, and modality-specific tokenizers change architecture or
  optimization but do not address the local loss of alert/effect information.

## Three locally formulated candidates

These names describe Quantis hypotheses, not claims of wholly new JEPA
theory.

### PairEffect-JEPA

Use every matched control/treatment pair as the primitive training unit.
Decompose the representation into:

- a shared baseline coordinate aligned across the pair before intervention;
- an effect coordinate that predicts the treatment-minus-control trajectory;
  and
- an action-equivariant coordinate required to identify action and target.

Train the future-prediction loss on both arms, a paired difference loss on the
effect coordinate, and a deranged-pair null with identical capacity. Deploy
only the predicted effect correction on top of the frozen raw rank-32 path.

Novelty caution: invariance/equivariance is already present in
[seq-JEPA](https://arxiv.org/abs/2505.03176), auxiliary-value anchoring is
known, bisimulation grounding is known, and Delta-JEPA already decodes actions
from latent differences. The locally new part is their **matched-twin
effect-supervision contract**, not the ingredients individually. Causal-JEPA's
whole-entity masking did not test this intervention contrast.

This is the strongest next tracer because it exploits information the small
corpus uniquely has rather than asking a generic predictive objective to
rediscover it.

**Executed result:** rejected. The paired objective was slightly worse than
its deranged null on selection and transfer observable effect, and composing
its predicted effect onto raw dynamics increased held-topology downstream
effect MSE by `3.38×`. See the
[retained result](pair-effect-jepa-v1-results.md).

### Contract-JEPA

Define the deployed state as
\(z_t=[P_{\mathrm{raw}}x_t,\ r_\theta(x_{\leq t})]\), where the first block is
a frozen, directly auditable raw/PCA state and only the bounded residual block
is learned. Add joint heads for current observable recovery, downstream
effect, and the alert decision. Constrain the learned correction by a
selection-fitted trust region and fall back exactly to raw when it earns no
gain.

Novelty caution: Quantis already rejected a raw-preserving residual JEPA and
already trained reconstruction anchors. A fixed raw bypass by itself is
therefore **not new**. Contract-JEPA is worth another run only when coupled to
the decision-anchored objective above or PairEffect supervision. Its real
hypothesis is hard sufficiency plus task grounding, not “residual JEPA v2.”

**Executed result:** rejected. The bounded residual improved raw transfer
effect MSE by 1.05%, but the supervised and ungrounded controls were both
better. The explicit effect witness learned its training quantity yet emitted
100% false alarms on transfer controls. See the
[retained result](task-grounded-contract-jepa-v1-results.md).

### Error-Certificate-JEPA

Keep a single deterministic future. Train a small head to predict a scalar or
per-horizon upper certificate for the frozen raw model's error—such as an
absolute-residual quantile or exceedance probability—from context and JEPA
features. Calibrate the certificate on the existing calibration role, then
emit prediction, bound, and abstain/alert decision. Do not emit \(K\) futures.

Novelty caution: this overlaps HEPA's monotone probability interface,
JEPA-SCORE's deterministic density signal, and ordinary residual
quantile/conformal calibration. It should be described as a **JEPA-assisted
error certificate**, not a new probabilistic world model and not epistemic
uncertainty. Its decisive controls are the same certificate from raw features
alone and a conformalized raw residual. The JEPA feature path must improve
sharpness at fixed coverage and the alert operating point without increasing
false alarms.

**Executed result:** rejected. The corrected run preserved raw exactly and
passed every safety gate, but held-topology simultaneous control coverage was
80%, treatment detection was zero, and the learned JEPA bound was no sharper
than derangement or constant conformal. See the
[retained result](error-certificate-jepa-v1-results.md).

## Execution outcome

The first three experiments and the corrected SALT tracer were
executed with frozen controls, independent stored-evidence assessment, and
retained artifacts. All four were rejected before robustness or sealed
confirmation.

Interaction-aware target selection was exercised through matched-pair effect
supervision and task-grounded witnesses; neither established incremental
value. SALT, exact LeNEPA, Discrete-JEPA, PEIRA, VISReg, and exact
JEPA-SCORE were then run as bounded omissions and rejected before robustness
or sealed confirmation. JEPA-SCORE did establish that exact, unapproximated
Jacobian/SVD scoring fits the edge budget.

This outcome accepts that the stack is intentionally small. The next action
is shadow deployment of the retained non-JEPA baselines, not data expansion
or another generic representation sweep.
