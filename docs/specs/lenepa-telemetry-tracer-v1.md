# LeNEPA telemetry tracer v1

## Question

Does LeNEPA's disposable projected next-latent objective produce a causal
edge representation whose held-topology action-effect forecasts beat an
unprojected next-latent control, projected SIGReg without prediction, matched
PCA, and raw dynamics?

This is a one-seed open-development tracer. Passing authorizes fixed-seed
robustness only; it does not authorize production alerts.

## Frozen roles

Use the content-addressed pair-atomic edge cache:

- fitting: 40 in-distribution pairs;
- selection: 10 disjoint in-distribution pairs;
- calibration: 10 disjoint in-distribution pairs;
- IID evaluation: 20 disjoint pairs;
- transfer evaluation: 10 held-topology pairs; and
- the frozen held-topology attribution query library.

Representation fitting uses fitting histories only. Evaluation tensors may
not affect optimization, ridge selection, normalization, or configuration.

## Frozen cells

All neural cells have identical initialization, backbone, retained
projector, optimizer, anchor schedule, trainable parameter count, and
deployable parameter count.

| cell | next-token loss space | temporal SIGReg space | MSE weight |
|---|---|---|---:|
| `projected_lenepa` | projected | projected | 1 |
| `unprojected_lenepa` | backbone | backbone | 1 |
| `projected_sigreg_only` | projected | projected | 0 |

The unprojected control retains but does not call its projector. The
SIGReg-only control retains and calls the exact projected path. No cell uses
augmentation, masking, future states, controls, actions, stop-gradient, EMA,
or a separate predictor.

## Architecture and objective

At each of 20 times, zero undeclared coordinates, project each entity's owned
state with an entity-specific linear map, sum the seven contributions, and
add learned time, entity-kind, and graph-summary embeddings. Process the
sequence with an eight-layer pre-norm causal Transformer:

- width 64;
- four attention heads;
- MLP width 256;
- no dropout; and
- CPU inference.

The final-time public entity tokens are each entity contribution plus the
final causal token and learned entity identity.

For the candidate, apply the same training-only
`64 -> 1536 -> 64` BatchNorm/ReLU projector to input-layer and final-layer
tokens. Minimize MSE between projected final-layer tokens at `t` and
projected input-layer tokens at `t+1`, with gradients through both sides.
Add temporal SIGReg at weight 20 over projected input/final tokens. Use 17
knots and 256 seeded random projections.

Fit for 1,600 pair-blocked steps with seed `24024`, AdamW, learning rate
`1e-4`, 80 warmup steps, cosine learning-rate decay, and cosine weight-decay
increase from `1e-2` to `1e-1`. Clip gradient norm at one.

## Shared downstream evaluation

Freeze each encoder. Fit the shared rank-32 reduced-rank action probe over
ridges `{1e-4, 1e-3, 1e-2, 1e-1, 1}`. Select the lowest selection
downstream-effect MSE among rows whose overall and action-overlap MSE are
within 5% of raw; if none are safe, select the lowest effect MSE and record
the safety failure explicitly.

Report selection, IID, and transfer:

- overall, action-overlap, and downstream-effect MSE;
- transfer per-pair effect error;
- current-state probe NRMSE;
- attribution hit@1 and no-action specificity; and
- correct-action versus absent/shuffled-action sanity.

Fit matched width-64 entity PCA and raw rank-32 dynamics on fitting only.

## Mechanism diagnostics

On selection and transfer histories, independently report:

- cosine error between every predicted/target next-token pair;
- aligned next-token retrieval hit@1 against all same-time batch targets;
- prediction and target effective rank;
- temporal SIGReg for input and final layers under a fresh fixed diagnostic
  seed; and
- prefix invariance after perturbing all telemetry later than a chosen time.

The projection mechanism passes only if, on both selection and transfer:

1. projected LeNEPA cosine error is at most 90% of unprojected LeNEPA; and
2. projected LeNEPA aligned retrieval hit@1 exceeds unprojected LeNEPA by at
   least ten percentage points.

## Gates

All safety gates must pass:

1. evidence is finite and pair/trajectory roles are disjoint;
2. all neural cells have identical independently recomputed training and
   inference capacity;
3. original/restored representation, sequence, diagnostic, and probe outputs
   agree within `1e-6`;
4. copied public signatures accept histories and graph only, and retained
   prefix-invariance evidence is exact within `1e-6`;
5. the pair-blocked anchor schedule recomputes;
6. selection ridge choice and explicit no-safe-ridge status recompute;
7. candidate selection and transfer overall/action MSE are within 5% of raw;
8. attribution hit@1 is at least 95%, no-action specificity is 100%, and
   correct actions beat absent/shuffled actions on at least 80% of treatment
   pairs;
9. the actual deployable candidate bundle is at most 16 MiB and 100 raw
   batch-one CPU latency samples are retained; and
10. an independent artifact-only assessor reproduces every gate.

The downstream value lane passes only if:

- candidate selection effect MSE is strictly best among learned controls;
- candidate transfer effect MSE improves the best learned control and raw by
  at least 10%; and
- candidate beats the best learned control on at least 60% of transfer
  pairs.

Advance only if all safety, mechanism, and value gates pass. Otherwise reject
this exact telemetry recipe, not LeNEPA on the paper's time-series tasks.

## Artifact contract

Publish atomically without overwrite after verifying the staging bundle.
Retain all neural cells, projector state, PCA and raw controls, deployable
bundle, anchors, ridge curves, predictions, diagnostic tensors, restoration
evidence, raw latency samples, independent assessment, copied reproduction
sources, report, manifest, smoke bundles, and failures.
