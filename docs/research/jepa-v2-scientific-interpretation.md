# Scientific interpretation of the dependency-log JEPA v2 result

## Verdict

**Continue the research, but do not promote or describe the current artifact
as a validated world model.**

The result is real progress from the earlier shortcut-dominated experiments:
the model is non-constant, transfers its calibration across held-out schedule
families, and uses aligned logs slightly better than shuffled logs. The
evidence for *useful log semantics*, however, is still weak. Across the eight
family-held-out folds, the selected model produced 160 alerts in 7,872 normal
windows, versus 163 for shuffled logs and 166 for a capacity-matched
metrics-only model. Its decisive margin is therefore three alerts over
shuffled logs and six over the capacity control.

Most importantly, all 30 manifests in this development corpus have
`fault_kind: none`. The experiment measures normal-schedule calibration
transfer only. It cannot show that the representation detects, predicts, or
helps respond to Redis, PostgreSQL, queue, or worker faults.

The appropriate decision is:

- **go** for an untouched, preregistered confirmation experiment;
- **no-go** for promotion, publication claims, or calling this operationally
  useful until it shows fault sensitivity and downstream utility.

The local evidence is in the
[candidate selection](../../artifacts/jepa-world-model-v2/contextual-development-v2/training/candidate-selection.json),
[selected candidate development artifact](../../artifacts/jepa-world-model-v2/contextual-development-v2/training/candidates/v2_log_latent_1/development.json),
and [development protocol](../specs/contextual-multimodal-jepa-v2-development.md).

## What the JEPA papers count as evidence

The original JEPA proposal defines the central move as predicting the
representation of a target from the representation of a context, optionally
conditioned on a latent variable that accounts for transformations or
multiple plausible futures. Its claimed advantage is that the target encoder
can discard irrelevant detail; low prediction energy alone is not evidence
that it retained the *right* information
([LeCun, 2022, §4.4](https://openreview.net/forum?id=BZ5a1r-kVsf)).

I-JEPA tests representation usefulness with frozen-encoder linear probes,
low-shot adaptation, transfer classification, counting, and depth prediction.
It also ablates representation-space versus pixel-space targets and the
masking design. Large target blocks and spatially distributed context are
important because they make the task semantic and prevent easy local
interpolation
([Assran et al., 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.html)).

V-JEPA likewise evaluates a frozen backbone on both appearance and motion
tasks. It uses stop-gradient EMA targets and a predictor as its practical
anti-collapse mechanism, and finds L1 feature prediction more stable. Its
continuous, high-ratio masks deliberately limit leakage from video
redundancy. The paper's claim rests on downstream frozen evaluations and
feature-versus-pixel ablations, not merely a falling pretraining loss
([Bardes et al., 2024, §§3–4](https://arxiv.org/html/2404.08471v1)).

V-JEPA 2 raises the bar for the term *world model*. It first evaluates frozen
representations on diverse understanding tasks, then freezes the encoder and
trains an action-conditioned causal predictor with teacher-forcing and a
two-step rollout loss. World-model utility is finally demonstrated through
closed-loop planning and zero-shot robot success in two new labs
([Assran et al., 2025, §§3–4](https://arxiv.org/html/2506.09985v1)).

The slow-feature study supplies a directly relevant warning: a predictive
joint-embedding objective can favor a stable, easy-to-predict distractor over
the task-relevant state. In its controlled environments, JEPAs performed well
with temporally changing distractors but failed when a fixed background
became the easiest predictable signal
([Sobal et al., 2022](https://arxiv.org/abs/2211.10831)). This is an analogy,
not direct evidence about telemetry, but it is exactly why the shuffled-log
and family-held-out controls matter.

LeJEPA makes the anti-collapse standard stricter still: it identifies both
complete and dimensional collapse as shortcut solutions and explicitly
regularizes embeddings toward an isotropic Gaussian, then validates them
with frozen probes across datasets and architectures
([Balestriero and LeCun, 2025](https://arxiv.org/html/2511.08544v3)). Our EMA,
nonzero variance, and effective-rank diagnostics are useful, but they are not
equivalent to that distributional guarantee.

## Comparison with the current model

### We are doing genuine representation-space prediction

The selected model predicts learned metric and log targets over contextual
blocks at horizons 1, 3, and 6, conditions on observable request demand,
worker topology, and horizon, uses EMA targets, and includes a short rollout.
That is recognizably JEPA-shaped and materially closer to I-JEPA/V-JEPA 2
than the earlier pointwise predictor.

But the 12 log inputs are already engineered ratios and thresholded event
rates. The encoder is not discovering structure from raw application text or
SDK output; much of the abstraction was supplied by the vocabulary design.
This experiment can validate predictive dynamics over curated telemetry, but
it cannot yet support the broader claim that JEPA learned application
semantics from logs.

### This is not complete collapse

The selected log target has nonzero variance (0.169), low tanh saturation
(1.58%), and effective rank 1.0. The metric target has effective rank 1.624
of three. These facts reject a constant-output interpretation.

They do not prove a rich representation. A one-dimensional log latent has no
room to reveal dimensional collapse. The two- and three-dimensional
candidates reached log ranks of roughly 1.75–2.42, so the new dependency
vocabulary contains more than one learnable direction, but those candidates
reduced metric rank and generalized less reliably. The current scientific
reading is **capacity/interference tension**, not “the logs are intrinsically
one-dimensional.”

The balanced masked two-dimensional candidate is especially informative: it
had stronger margins over its controls and was no worse than metrics-only on
six of eight folds, but missed the preregistered metric-rank gate (1.460
versus 1.5). Because masking, loss balance, and log width changed together,
this is not a clean masking ablation and must not be used to claim that
I-JEPA-style masking helped or failed.

### The control advantage is too small to establish semantic use

The selected model's family-held-out alert rate was 2.033%, compared with:

- 2.426% for metrics-only;
- 2.109% for capacity-matched metrics-only; and
- 2.071% for shuffled logs.

Only four of eight folds were no worse than metrics-only, and only four beat
shuffled logs. The three-alert aggregate edge over shuffled logs is compatible
with a small useful signal, seed/calibration noise, or schedule-specific
effects. Since candidate selection used these same folds, the reported margin
is also a model-selection statistic, not an unbiased effect estimate.

The selected representation's frozen probes are in-sample diagnostics:
request latency reaches training \(R^2=0.334\), most operational ratios are
around \(0.12\)–\(0.18\), and queue depth is \(0.033\). Unlike the frozen
transfer evaluations in I-JEPA and V-JEPA, these do not yet demonstrate
held-out decodability.

### We have not yet tested world-model utility

All observations are normal. A lower normal alert rate could mean better
state modeling, but it could also mean lower anomaly sensitivity. V-JEPA 2's
world-model claim is supported by conditioned rollouts that select actions
and succeed in new environments. Our request-demand and topology variables
are conditioning inputs, not an evaluated intervention model, and no action
is optimized or executed from the latent predictions.

The scientifically defensible name today is **a contextual JEPA anomaly-model
candidate**, not a demonstrated infrastructure world model.

## What would constitute the next real result

Freeze the selected recipe and evaluate it once on an untouched corpus with:

1. new normal schedule families for false-positive calibration transfer;
2. preregistered Redis, PostgreSQL, queue-pressure, and worker faults for
   detection recall, detection delay, and fault-family transfer;
3. paired metrics-only, capacity-matched, shuffled-log, and modality-dropout
   controls;
4. frozen probes trained on development representations and evaluated on
   untouched runs;
5. multiple preregistered training seeds or paired uncertainty intervals, so
   a three-alert difference is not overinterpreted; and
6. if the *world model* claim is retained, a downstream intervention test:
   predict the result of an observable action such as worker scaling or
   dependency recovery and use the prediction to choose an action.

A clean win on held-out faults, especially over shuffled logs and the
capacity-matched model, would show that aligned dependency events contribute
state information. Accurate action-conditioned rollouts that improve an
operational decision would be the first result comparable in kind—not
scale—to V-JEPA 2's world-model evidence.
