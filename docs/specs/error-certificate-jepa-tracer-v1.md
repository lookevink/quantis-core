# Error-Certificate-JEPA tracer v1

## Question

Can JEPA features make a deterministic, calibrated upper bound on the frozen
raw predictive core's future error materially sharper than raw-feature and
conformal controls while preserving held-topology coverage and useful alert
operation?

The raw mean and variance are immutable. This tracer predicts reliability; it
does not correct the forecast or emit multiple futures.

## Frozen data roles

Use the topology-transfer cache:

- fitting: 40 in-distribution matched pairs;
- selection: 10 disjoint in-distribution pairs;
- calibration: 10 disjoint in-distribution pairs used only for additive
  certificate calibration;
- IID evaluation: 20 disjoint in-distribution pairs; and
- transfer evaluation: 10 held-worker-topology pairs.

Fit one rank-32 raw action-conditioned predictive core. Hash it before and
after every certificate fit.

## Target and public output

For each window and horizon, the realized error target is the owned-coordinate
RMSE between the raw forecast and observed future.

Public inference accepts only current history, declared controls/actions, the
graph, and the internally computed raw forecast. It returns:

- the unchanged raw trajectory distribution;
- one non-negative upper certificate per horizon; and
- a serialized additive calibration adjustment.

No future observation, pair identity, incident label, or evaluation statistic
enters prediction.

## Equal-capacity cells

| Cell | certificate features | latent objective |
|---|---|---|
| `jepa_error_certificate` | raw state/forecast plus predicted JEPA tokens | matched future latent |
| `raw_error_certificate` | raw state/forecast plus a zero latent block | disabled |
| `deranged_jepa_certificate` | raw state/forecast plus predicted JEPA tokens | deterministically deranged future latent |

Every cell retains the same encoder, predictor, certificate head, and
parameter count. All heads receive current raw state, the raw forecast at the
requested horizon, controls, actions, a horizon embedding, and either the
predicted latent block or zeros.

Train the certificate with 0.95 pinball loss and the JEPA cells with an
additional `0.2` latent L1 loss. Use an EMA target encoder. Derangement cycles
future latent targets within each deterministic minibatch while leaving the
true realized-error target aligned.

Use deterministic CPU AdamW, seed `25021`, 800 steps, batch size 128,
learning rate `5e-4`, weight decay `1e-3`, EMA `0.996`, gradient clipping at
one, and checkpoints every 100 steps. Select each checkpoint by its own
selection pinball loss.

## Calibration and controls

For each learned cell, reduce `realized_error - predicted_bound` to one maximum
per calibration control trajectory. Add the 95th higher empirical quantile,
clipped below at zero, to every predicted bound.

The direct conformal control ignores learned certificates and uses the 95th
higher empirical quantile of calibration control-trajectory maximum raw error
as a constant bound.

Report point coverage, simultaneous trajectory coverage, mean and p95 bound,
control-trajectory false alarms, treatment detection, and post-onset delay.

## Gates

All safety gates must pass:

1. all stored and independently recomputed values are finite;
2. learned cells have identical training and inference capacity;
3. the raw artifact hash is unchanged by training;
4. every cell returns the exact same raw mean and variance;
5. public inference is causal and restored outputs and alert decisions match
   within `1e-6`;
6. all uncalibrated and calibrated bounds are finite and non-negative;
7. the candidate bundle is at most 16 MiB and batch-one CPU latency is
   recorded; and
8. selection and calibration are the only roles used for checkpoint and
   adjustment choice.

Coverage requires:

- at least 95% point coverage on transfer control trajectories;
- 100% simultaneous coverage across transfer control trajectories, equivalent
  to at most 5% control-trajectory false alarms with ten controls; and
- no evaluation data used to widen a bound.

Mechanism requires:

- candidate selection pinball loss at most `0.90` times the deranged cell; and
- candidate transfer-control mean bound at most `0.90` times the deranged
  bound while both satisfy coverage.

Value requires:

- candidate transfer-control mean bound at most `0.90` times both the raw-only
  learned certificate and constant conformal control;
- at least 80% transfer treatment-trajectory detection;
- median post-onset delay at most ten transitions; and
- candidate treatment detection at least ten percentage points above the
  raw-only learned certificate at the same control false-alarm ceiling.

Every safety, coverage, mechanism, and value gate must pass. Failure rejects
JEPA-assisted certification on this corpus while retaining raw/conformal
controls.

## Artifact and claim boundary

Write through a fresh staging directory. Retain raw and certificate models,
checkpoint evidence, calibration adjustments, raw predictions, realized
errors, original/restored bounds and decisions, independent assessment,
copied sources, and a SHA-256 manifest.

This is an empirical error certificate, not a formal uncertainty guarantee,
Bayesian posterior, or production service-level guarantee.

