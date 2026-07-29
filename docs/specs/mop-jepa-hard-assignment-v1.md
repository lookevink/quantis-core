# MoP-JEPA hard-assignment tracer v1

## Question

Does hard winner-take-all latent prediction recover several
context-conditioned, transition-valid telemetry futures while preserving the
proper-score and point-prediction safety required by the edge alerting
system?

This is a one-seed open-development tracer. Passing authorizes fixed-seed
robustness only.

## Frozen roles

Use the content-addressed pair-atomic ladder roles:

- fitting: 40 in-distribution pairs;
- calibration: 10 disjoint in-distribution pairs;
- selection: 10 disjoint in-distribution pairs; and
- transfer evaluation: 10 held-topology pairs.

Fit encoders, predictors, routers, the codebook, and raw dynamics on fitting
only. Calibrate observable component variances and the transition-validity
radius on calibration only. No selection or transfer target may affect a
model, variance, threshold, or configuration.

## Frozen cells

1. `mop_jepa`: eight cosine-trained latent heads, trajectory-level hard
   winner, EMA target, and context-only router.
2. `dense_jepa`: the same JEPA recipe with one head.
3. `supervised_hard_wta`: the same eight-head architecture, but hard winners
   minimize observable future MSE directly.
4. `context_free_codebook`: eight deterministic fitting-role future centers
   with global mixture weights and no context.
5. `raw_low_rank`: the existing rank-32 edge dynamics baseline.

The MoP candidate and supervised hard-WTA control have identical trainable
capacity. The codebook and deterministic shuffled-context candidate are
mechanism controls, not promotion competitors.

## Architecture and optimization

Use 20 history steps, ten future steps, seven declared entity slots, latent
width 12, entity-context width 16, predictor width 128, batch size 256,
AdamW learning rate `1e-3`, weight decay `1e-4`, 40 epochs, target EMA
`0.996`, and seed `19019`.

Encode every state slot with one shared linear layer and GELU, then L2
normalize it. Summarize each entity by its final, mean, and history-delta
latent. Concatenate all entity summaries with flattened future controls and
actions. A shared hidden layer feeds independent output blocks and a router.

For MoP-JEPA, choose the head minimizing mean cosine distance across the
complete entity/time successor. Optimize winning cosine distance, router
cross-entropy with coefficient one, EMA-target reconstruction with
coefficient `0.10`, and current-state reconstruction with coefficient
`0.05`. Freeze the paper-supported zero load-balance setting. The supervised
control replaces cosine winner loss with winning observable MSE. Dense JEPA
uses the same latent objective with one head. Use the final epoch; do not
select a restart or checkpoint.

Fit each codebook center by deterministic farthest-first initialization and
20 Lloyd iterations over flattened fitting futures.

## Stored-array assessment

For selection and transfer, store every observable component mean, router
weight, shared calibrated variance, target, history, control, action, pair
identity, and transition identity. Also store the candidate under one
deterministic context permutation. A fresh assessor recomputes:

- exact complete-trajectory Gaussian-mixture NLL;
- mixture-mean overall, action-overlap, and downstream-effect MSE;
- raw and router-gated oracle component MSE;
- router effective-head count and observable winner-usage count;
- correct-versus-shuffled context coverage;
- codebook-relative gated coverage; and
- calibrated realized-transition precision.

Router gating keeps predictor heads with `pi_k > 0.5 / K`. The input-agnostic
codebook has no router and is never gated. The transition radius is the 95th
percentile calibration-role trajectory RMSE of raw rank-32 dynamics.

## Gates

Every safety gate must pass:

1. all stored evidence is finite and fresh restoration agrees within `1e-6`;
2. candidate and supervised hard-WTA trainable capacities are identical;
3. public inference rejects realized future inputs;
4. the candidate artifact is at most 16 MiB and CPU latency is reported;
5. selection and transfer mixture-mean overall and action-overlap MSE are no
   worse than `1.05` times raw; and
6. artifact publication is accepted only after the assessor CLI, in a fresh
   process, reproduces the stored decision and verifies the completed
   manifest. This external integrity check cannot be a numeric eligibility
   conjunct without making the manifest self-referential.

The hard-assignment mechanism passes only if, on selection:

- observable winner usage has at least two effective heads;
- router probabilities have at least `1.5` effective heads;
- router-gated realized-transition precision is at least 80%;
- raw oracle MSE improves dense JEPA point MSE by at least 10%;
- gated oracle MSE improves the context-free codebook by at least 10%; and
- correct-context oracle MSE improves shuffled-context oracle MSE by at
  least 10%.

The value lane passes only if:

- selection NLL improves dense JEPA by at least `0.01` nats per coordinate;
- selection NLL improves supervised hard-WTA by at least `0.01` nats per
  coordinate;
- transfer downstream-effect MSE improves raw by at least 5%; and
- candidate transfer downstream-effect error beats raw on at least 60% of
  matched pairs.

Advance only if safety, mechanism, and value all pass. Otherwise reject this
edge telemetry recipe while retaining every implementation, model,
prediction, smoke, failure, assessment, and manifest.
