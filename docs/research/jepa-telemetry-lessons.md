# JEPA lessons for the next metrics + logs experiment

## Conclusion

The v2 result does not show that JEPA cannot generalize. It shows that the
current learning problem makes a schedule shortcut easier than learning
application state.

The highest-priority change is to treat request demand and topology as
conditioning variables—JEPA's \(z\), or an action—not as future state that the
latent must rediscover. The second is to replace the pointwise next-window
target with contextual, masked temporal targets. Representation width,
predictor capacity, and additional data should be tested only after those two
changes.

This conclusion is an application of JEPA research to a very small structured
telemetry model, not a claim that video-scale recipes transfer unchanged.

## What v2 actually trained

The aligned model used six normalized metrics, four normalized log counts, a
six-window lookback, three metric latent dimensions, and two log latent
dimensions. Separate linear/tanh encoders transform each point; their outputs
are concatenated and flattened; one affine predictor forecasts one joint
future latent. The target is another independently encoded point, not a
contextual representation of a future block. Training minimizes MSE, updates
target encoders by fixed-decay EMA, and scoring takes one RMS across all five
latent dimensions
([implementation](../../src/quantis_core/multimodal_world_model.py),
[protocol](../specs/multimodal-jepa-corpus-v2.md)).

On held-out schedules, its alert rate was 6.74%, versus 1.85% for the
capacity-matched metrics-only control and 4.44% for shuffled logs
([frozen result](../../artifacts/jepa-world-model-v0/multimodal-normal-corpus-v2/training/development.json)).
The log vocabulary was almost degenerate in normal runs: rejected and error
counts had zero variance; accepted and completed counts correlated at 0.9965
in training and were exactly equal in 99.73% of windows. One principal
direction explained 99.83% of training log variance. The two learned log
target dimensions consequently had an effective covariance rank of about
1.02. Log residuals contributed 32.1% of training squared error but 49.6% of
validation squared error.

This is not classic constant-output collapse. It is a useful-looking,
near-one-dimensional representation of the easiest predictable variable:
absolute demand.

## JEPA-specific interpretation

### 1. Demand is missing predictor conditioning

The generic JEPA predictor is conditioned on \(z\), which describes the
transformation between context and target. This lets the same context support
different predictions rather than forcing the encoder to infer the
transformation. The original JEPA proposal also allows \(z\) to represent
information needed for multiple plausible futures
([LeCun 2022, §4.4](https://openreview.net/forum?id=BZ5a1r-kVsf)).
V-JEPA makes target position explicit predictor conditioning
([Bardes et al. 2024, §3](https://arxiv.org/html/2404.08471#S3)).
V-JEPA 2-AC goes further: its causal predictor forecasts future latent states
conditioned on actions and proprioceptive state
([Assran et al. 2025, §3.1](https://arxiv.org/html/2506.09985#S3.SS1)).

For Quantis, intended or observed request demand, worker topology, deployment
state, and configuration changes are analogous exogenous controls. Metrics
are already conditioned against request rate, while raw accepted/completed
log counts reintroduce it. The fused predictor therefore has to extrapolate
unseen request sequences from past counts.

Before another run, either:

- condition prediction on an operationally observable demand/topology token;
  or
- make log targets demand-invariant, using acceptance, completion, rejection,
  error, and backlog residuals or ratios.

If future demand truly is unavailable at scoring time, a deterministic
single-output predictor cannot distinguish an anomalous transition from one
of several legitimate futures. Then the target must be demand-invariant or
the model must represent multiple futures; secretly conditioning on a lab
schedule would not be deployable.

### 2. The target is latent, but not yet contextual

JEPA's advantage is not merely applying a projection before forecasting.
I-JEPA predicts several sufficiently large target blocks from a distributed,
informative context. It encodes the complete input with the target encoder and
selects target blocks from the encoder output, producing contextual targets
([Assran et al. 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.html)).
V-JEPA similarly masks large continuous spatiotemporal blocks to reduce
leakage from local redundancy
([Bardes et al. 2024, §§3.2, 4.4](https://arxiv.org/html/2404.08471#S3.SS2)).

The current target encoder sees only one four- or six-feature point. It cannot
represent a completion pattern, queue episode, or worker transition as a
contextual state. A more faithful small-data design is two-stage:

1. Pretrain modality-specific temporal encoders by predicting masked,
   contiguous blocks inside normal metric/log sequences.
2. Freeze or strongly stabilize those encoders, then train a causal predictor
   over their states, conditioned on demand and topology.

V-JEPA 2 uses this separation between masked representation pretraining and a
new frozen-encoder action-conditioned predictor. Its data scale is vastly
larger, so separation must be an ablation here, not an assumption of success.

### 3. One horizon encourages sequence memorization

I-JEPA and V-JEPA use multiple target blocks and both short- and long-range
masks. V-JEPA 2-AC trains next-state predictions throughout a sequence and
adds a two-step rollout loss to expose accumulated forecast error
([Assran et al. 2025, equations 2–4](https://arxiv.org/html/2506.09985#S3.SS1)).

For telemetry, use contiguous targets at several preregistered horizons—for
example 1, 3, and 6 windows—with a horizon/position embedding. Keep levels and
deltas as distinct target groups. This asks for state dynamics rather than
the next member of a periodic request pattern.

### 4. EMA prevents collapse, not shortcuts

V-JEPA uses stop-gradient targets, an EMA encoder, and a predictor to prevent
constant representations; it chose L1 over L2 because L1 was more stable
([Bardes et al. 2024, §3.1](https://arxiv.org/html/2404.08471#S3.SS1)).
The current manual optimization has the same broad online/EMA asymmetry and
also orthonormalizes encoder weights. But orthogonal weights do not make
representations diverse when the data itself is rank one.

JEPA has a documented tendency to focus on slow or easily predictable
distractors: in simple environments, fixed background structure displaced
task-relevant motion from the representation
([Sobal et al. 2022](https://arxiv.org/abs/2211.10831)). That study used toy
environments and VICReg/SimCLR-style JEPAs, not EMA V-JEPA, so it is a
diagnostic analogy rather than proof. It nevertheless closely matches the
demand-count shortcut.

Record per-modality target variance, covariance/effective rank, tanh
saturation, online-target distance, and predictor performance with each
channel dropped. Specify EMA in effective update half-life; copying V-JEPA's
numeric momentum is inappropriate because Quantis performs full-batch epoch
updates.

### 5. Fusion needs explicit objectives and native semantics

MJEPA found that naïvely sharing an encoder across audio and video degraded
both unimodal baselines. It recovered positive transfer with modality-specific
tokenizers/embeddings, intra-modal prediction, and explicit cross-modal
predictors
([Teotia et al. 2026, §§4.1–4.4](https://arxiv.org/html/2606.25225#S4)).
V-JEPA 2.1 likewise uses modality-specific tokenizers and modality tokens so
images and videos are processed in their native form
([Mur-Labadia et al. 2026, §2.3.4](https://arxiv.org/html/2603.14482#S2.SS3.SSS4)).

Keep separate metric and log stems. Train metric-to-metric and log-to-log
objectives, then test metric-to-log and log-to-metric prediction explicitly.
Do not add cross-modal alignment until demand has been residualized: otherwise
alignment will strengthen the trivial accepted-count/request-rate
correspondence. MJEPA's evidence is from millions of audio/video samples, not
10,020 overlapping telemetry windows.

## Recommended preflight, in order

Use only v2 training families with nested family-held-out development; its
published validation families are now exposed and cannot be fresh evidence.

1. **Demand/action correction:** residual log features plus observable
   demand/topology conditioning. Compare one log latent against two.
2. **Loss and scoring:** compare MSE with L1/Huber; calibrate metric and log
   energies separately before fusion rather than averaging uncalibrated latent
   dimensions.
3. **Contextual JEPA:** masked contiguous temporal targets, EMA target encoder,
   then a frozen-encoder causal predictor.
4. **Dynamics:** multi-horizon targets and a short rollout loss.
5. **Fusion:** add explicit intra- and cross-modal heads only after shortcut
   removal.

Retain metrics-only, capacity-matched, shuffled-log, log-only, and modality
dropout controls. Add promotion gates for conditional improvement over
metrics-only, active latent rank relative to allocated width, and frozen-latent
probes for backlog delta, completion ratio, latency/queue buckets, worker
state, and transition direction.

Only after that ablation sequence is frozen should a new untouched corpus be
collected. More runs under the present objective would mostly provide more
examples of the same demand shortcut.

## Primary sources

- [LeCun, *A Path Towards Autonomous Machine Intelligence* (2022)](https://openreview.net/forum?id=BZ5a1r-kVsf)
- [Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture* (2023)](https://arxiv.org/abs/2301.08243)
- [Bardes et al., *Revisiting Feature Prediction for Learning Visual Representations from Video* (2024)](https://arxiv.org/abs/2404.08471)
- [Assran et al., *V-JEPA 2* (2025)](https://arxiv.org/abs/2506.09985)
- [Sobal et al., *Joint Embedding Predictive Architectures Focus on Slow Features* (2022)](https://arxiv.org/abs/2211.10831)
- [Teotia et al., *MJEPA* (2026)](https://arxiv.org/abs/2606.25225)
