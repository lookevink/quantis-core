# Residual JEPA correction development v1

## Purpose

Test whether JEPA is useful as an auxiliary residual representation after the
successful raw-state low-rank model, rather than as the sole dynamics state.

This is an open-development experiment over the already-inspected,
content-addressed action-dynamics corpus. It cannot confirm a world-model
claim.

## Hypothesis

A frozen rank-32 raw-state low-rank model preserves observable state, action,
and topology identity. A zero-start JEPA branch that predicts only the
baseline's future error can improve downstream-effect prediction without
damaging action attribution or ordinary forecasting.

The JEPA-specific null is that any improvement comes from supervised residual
fitting rather than joint-embedding prediction.

## Data boundary

Reuse only the topology-transfer preprocessing cache whose state and control
normalizers were fitted on worker topologies 1–2. Keep complete matched pairs
within one role.

- fit, selection, and calibration use topologies 1–2;
- the primary transfer evaluation uses unseen topology 3; and
- the remaining evaluation pairs provide an in-distribution diagnostic.

No held-out-topology observation may influence fitting, normalization,
correction-gain selection, or calibration.

## Public seams

Tests exercise:

1. the existing `EdgeDynamicsModel` fit/rollout/serialization boundary;
2. exact baseline identity when residual gain is zero;
3. zero initial output from each residual decoder;
4. selection-only correction-gain choice; and
5. the development assessment and immutable artifact boundary.

Tests do not inspect optimizer internals or private coefficients.

## Models

Fit the rank-32 `ContractiveLowRankDynamics` once. Hash its serialized state
before and after correction training to prove it remains frozen.

Train two capacity-matched residual branches:

1. **supervised residual correction** — decoded multi-step residual MSE only;
2. **JEPA residual correction** — the same decoded residual objective plus
   block-masked L1 future-state latent prediction, EMA target encoder, context
   variance/covariance anti-collapse, and a contractive rank-32 latent
   transition.

Both branches retain seven explicit entity tokens of 16 dimensions. Their
decoders are initialized to exactly zero, so the composed model equals the
frozen baseline before the first optimizer update. The latent target remains
the full observed future state; only the decoder target is

`observed future - frozen baseline future`.

This avoids asking an EMA encoder to equate raw state with a different
residual-valued input domain. Context reconstruction is disabled because the
residual decoder must not also be trained to reproduce raw state.

Use deterministic PyTorch CPU execution, seed 113, 60 epochs, batch size 256,
AdamW at `1e-3`, and EMA decay `0.996`. CPU is the confirmation backend because
two same-seed Apple MPS development runs were not bitwise reproducible despite
requesting deterministic algorithms. The supervised control has residual MSE
weight `1.0` and every auxiliary weight set to zero. The JEPA branch has
residual MSE weight `1.0`, latent loss weight `0.2`, and variance/covariance
weights `0.01` and `0.005`. Mask contiguous 30% time blocks for 25% of entity
tokens. These deliberately make observable residual accuracy primary and JEPA
a light regularizer.

## Selection

For each fitted branch, choose residual gain from
`{0, 0.25, 0.5, 0.75, 1}` using only action-overlap MSE on the selection role.
Break exact ties toward the smaller gain. Gain zero is an explicit safe
fallback and means the residual hypothesis did not earn deployment.

## Evaluation

Report in-distribution and topology-transfer:

- normalized overall and action-overlap MSE;
- paired treatment-minus-control downstream-effect MSE;
- action-and-target hit@1 and no-action specificity;
- correct-action versus no-action and deranged-action sanity;
- parameter count, serialized size, latency, finite rollout, and norm growth;
- selected correction gain and selection curve; and
- JEPA token effective-rank diagnostics.

Evaluate JEPA latent divergence separately. Reduce each in-distribution control
trajectory to the maximum divergence across every predicted horizon and
calibrate the threshold over those trajectory maxima. Then report
trajectory-level control false alarms,
post-onset treatment detection, and median delay for both in-distribution and
topology-transfer evaluation. This aligns the calibration and gate units and
avoids compounding a pointwise false-alarm probability over dozens of
overlapping windows. This detector is not part of the predictive correction
gate.

## Preregistered gates

The JEPA residual tracer advances to three-seed robustness only if, on unseen
topology 3, it:

1. improves downstream-effect MSE by at least 10% over frozen raw low-rank;
2. keeps action-overlap MSE within 5% of raw low-rank;
3. keeps overall MSE within 5% of raw low-rank;
4. retains at least 95% action-and-target hit@1;
5. retains 100% no-action specificity;
6. beats both action ablations on at least 80% of treatment pairs;
7. selects a nonzero correction gain; and
8. has downstream-effect MSE no worse than the capacity-matched supervised
   residual branch.

The last gate is required for a JEPA-specific claim. Passing the first seven
but not the eighth may support residual correction, but not JEPA.

The independent investigation-wakeup gates are:

- at most 5% control-trajectory false alarms;
- at least 80% treatment-trajectory detection; and
- median post-onset delay no greater than 10 transitions.

Failure rejects this configuration, not residual modeling or JEPA generally.
Passing this single-seed tracer permits three fixed seeds. Only all-seed
success permits freezing a recipe and collecting fresh sealed matched pairs.

## Claim boundary

Even a passing open run supports only:

> On this fixed lab stack and open corpus, a frozen low-rank transition plus
> JEPA-regularized residual correction improved action-conditioned
> topology-transfer prediction while preserving closed-library attribution.

It does not establish a general software world model, novel-fault
generalization, production reliability, or sealed confirmation.
