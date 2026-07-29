# Exact SIGReg regularizer-substitution tracer v1

## Status and question

This is a preregistered, single-seed, open-development tracer. It asks:

> Does the exact official-minimal SIGReg regularizer improve the strongest
> entity-preserving action-conditioned residual JEPA's topology-transfer
> observable-state retention, prediction, alerting, or investigation over its
> current variance/covariance regularizer and a no-regularizer JEPA null,
> without changing inference-time capacity?

This tests a SIGReg substitution, not the complete LeJEPA training recipe. The
existing EMA target, masked latent predictor, and residual prediction losses
remain fixed. The exact retained runner is
`lab/action_dynamics/prototype_sigreg_lejepa.py`.

## Primary-source identity

The implementation is pinned to the LeJEPA authors' official
`official-minimal-c293d29` preset at commit
[`c293d291ca87cd4fddee9d3fffe4e914c7272052`](https://github.com/galilai-group/lejepa/tree/c293d291ca87cd4fddee9d3fffe4e914c7272052),
not to the numerically different paper-v3 finite quadrature.

For entity tokens `Z` with shape `(entity, batch, dimension)`, every optimizer
step:

1. draws 256 shared `Normal(0, 1)` sketch columns and L2-normalizes each;
2. projects every 16-dimensional token onto those directions;
3. evaluates empirical cosine and sine characteristic functions at 17 knots
   over `[0, 3]`;
4. compares them with `exp(-t²/2)`;
5. integrates with the official symmetric trapezoidal weights and Gaussian
   window;
6. multiplies by batch size; and
7. averages across directions and entities.

Entities are leading groups, analogous to views in the reference code, so
their identities are not mixed into the independent sample axis. Directions
are shared across entities within a step and freshly sampled on the next step.
The complete source comparison and formula are recorded in
[`LeJEPA SIGReg primary-source implementation notes`](../research/lejepa-sigreg-primary-source-notes.md).

## Evidence boundary

- Source corpus: `artifacts/action-dynamics/development-v1`.
- Reuse its content-addressed topology-transfer preprocessing cache.
- Fit and selection use worker topologies one and two.
- Calibration uses separate topology-one/two control trajectories.
- The primary open transfer diagnostic is worker topology three.
- Whole treatment/control pairs remain atomic.
- No new or sealed evidence is collected.

## Frozen architecture and training

Reuse the prior residual-JEPA configuration because it was the strongest
entity-preserving action-conditioned JEPA: its raw rank-32 low-rank path
preserved 100% attribution while its learned branch remained a bounded
correction.

Every neural variant has:

- seven entity tokens of width 16;
- a rank-32 latent transition;
- a zero-initialized residual decoder over a frozen raw low-rank baseline;
- 30% time masking over 25% of entities;
- latent L1 weight `0.2`;
- decoded residual MSE weight `1.0`;
- no context reconstruction;
- EMA decay `0.996`;
- correction-gain selection over `{0, 0.25, 0.5, 0.75, 1}`;
- deterministic CPU PyTorch;
- seed `401`;
- SIGReg projection seed `1401`, advanced deterministically by optimizer step;
- 60 epochs, batch size 256, AdamW at `1e-3`, and weight decay `1e-4`.

The SIGReg candidate uses the official minimal loss at weight `0.02`, matching
the authors' minimal worked example. This is not the paper's recommended full
LeJEPA `lambda=0.05`, because the surrounding loss here is a fixed residual
JEPA rather than LeJEPA's multi-view convex objective.

## Controls

1. **Variance/covariance JEPA:** same model, seed, batches, masking, and
   residual objective with the prior variance weight `0.01` and covariance
   weight `0.005`.
2. **No-regularizer JEPA:** same model with every anti-collapse term zero.
3. **Matched PCA:** training-only per-entity width-16 PCA, evaluated by the
   same frozen local observable-state ridge probe.
4. **Raw low-rank:** the frozen rank-32 observable-state predictive core.

SIGReg, variance/covariance, and no-regularizer variants must have identical
inference parameter counts. SIGReg is training-only.

## Measurements

Report selection, in-distribution, and topology-transfer forecast metrics;
selected correction gains; per-entity effective rank; frozen observable-state
probe NRMSE; action attribution and action-ablation sanity; trajectory-level
latent-divergence alerting; batch-one latency; serialized model identity; and
restoration parity.

The observable-state probe is fitted on the fit role only. It maps each
entity's token to that entity's latest varying observed features and is applied
unchanged to topology transfer. Training-fitted feature scales define NRMSE.

## Safety gates

All must pass:

1. every reported score and probe is finite;
2. all three neural variants have the same inference parameter count;
3. SIGReg action-overlap and overall MSE are each within 5% of raw low-rank;
4. action-and-target hit@1 is at least 95%;
5. no-action specificity is 100%; and
6. correct action beats both ablations on at least 80% of treatment pairs.

## Value lanes

At least one lane must pass with every safety gate.

Predictive:

- topology-transfer downstream-effect MSE improves by at least 10% over raw;
- it is strictly better than both JEPA regularizer controls; and
- selection chooses a nonzero correction gain.

Investigation:

- topology-transfer state-probe NRMSE improves by at least 5% over the current
  variance/covariance regularizer; and
- it is no worse than matched PCA.

Alert:

- control-trajectory false alarms are at most 5%;
- treatment detection is at least 80%;
- median post-onset delay is at most 10 transitions; and
- sensitivity or delay improves over both JEPA regularizer controls at the
  same false-alarm budget.

Passing authorizes fixed-seed robustness, not sealed confirmation. Failure
rejects this exact residual-JEPA/SIGReg substitution, not SIGReg or LeJEPA
generally.

## Reproducibility rule

The runner and assessor remain in the repository regardless of outcome.
Published artifacts are immutable; reruns use a fresh `--output` directory.
Production code must not import the tracer as a supported interface.
