# Task-grounded Contract-JEPA tracer v1

## Question

Can a bounded JEPA residual improve the frozen raw action-conditioned
predictive core when its representation is trained jointly against explicit
observable-state and paired-effect witnesses?

This tracer follows the PairEffect-JEPA rejection. It must never replace the
raw rollout. Passing authorizes fixed-seed robustness only.

## Frozen role and data

Use the existing topology-transfer cache:

- fitting: 40 in-distribution matched pairs;
- selection: 10 disjoint in-distribution pairs;
- calibration: 10 disjoint in-distribution pairs, used only for the witness
  alert policy;
- IID evaluation: 20 disjoint in-distribution pairs; and
- transfer evaluation: 10 held-worker-topology pairs.

Fit one rank-32 contractive raw action-conditioned predictive core and hash it
before and after every neural fit.

## Contract

The deployed mean is always:

`raw_action_conditioned_rollout + selected_gain * bounded_correction`.

The correction is bounded coordinate-wise by three fitting-role residual
standard deviations. Gain is selected from `{0, 0.25, 0.5, 0.75, 1}`.
Gain zero returns the raw distribution exactly.

The public representation is a tuple of:

- the unmodified current observable raw state; and
- seven learned width-16 entity tokens.

This makes observable-state sufficiency structural rather than a probe hope.

## Frozen equal-capacity cells

| Cell | latent JEPA | current-state witness | paired-effect and effect-score witnesses |
|---|---:|---:|---:|
| `task_grounded_contract_jepa` | 0.2 | 0.1 | 1.0 and 0.2 |
| `supervised_task_contract` | 0 | 0.1 | 1.0 and 0.2 |
| `ungrounded_contract_jepa` | 0.2 | 0 | 0 |

All heads and parameters remain present in every cell.

For every fitting step, select one aligned transition from every pair and use
both treatment and control arms. The online encoder consumes current states.
An EMA target encoder supplies future latent targets. The predictor consumes
current tokens, future controls, and declared actions.

The losses are:

1. observable residual MSE against
   `observed_future - frozen_raw_future`;
2. L1 predicted-latent versus EMA future-latent JEPA loss;
3. decoded current-state recovery from the context tokens;
4. MSE of the corrected treatment-minus-control future effect; and
5. MSE of a non-negative per-horizon effect-score head against the observed
   matched effect RMS.

Use deterministic CPU AdamW, seed `24021`, 800 steps, learning rate `5e-4`,
weight decay `1e-3`, EMA `0.996`, gradient clipping at one, and checkpoints
every 100 steps. Select checkpoints on each cell's own selection residual plus
paired-effect objective.

## Gain selection

On selection only, evaluate all five gains. Eligible gains keep overall and
action-overlap MSE within `1.05` times raw. Choose the eligible gain with the
lowest paired downstream-effect MSE, breaking exact ties toward the smaller
gain. Gain zero is always available and means the residual earned no role.

## Witness alert policy

The public effect-score head consumes the same causal inputs as correction.
Using calibration control trajectories, calibrate one alert-policy cutoff over each
trajectory's maximum score. Report control-trajectory false alarms, treatment
detection, and post-onset delay on IID and transfer evaluation.

This is a declared-action witness lane, not autonomous incident detection.

## Gates

All safety gates must pass:

1. all stored and independently assessed values are finite;
2. all neural cells have identical training and inference capacity;
3. the raw artifact hash is unchanged by every neural fit;
4. gain zero reproduces raw mean and variance exactly;
5. correction and witness public seams reject future observations, pair ids,
   and target truth;
6. every correction lies inside the serialized trust bound;
7. restoration reproduces tokens, corrections, witnesses, composed rollouts,
   and alert decisions within `1e-6`;
8. transfer overall and action-overlap MSE remain within `1.05` times raw;
9. action-and-target hit@1 is at least 95%, no-action specificity is 100%,
   and correct action beats both ablations on at least 80% of transfer pairs;
10. the candidate bundle is at most 16 MiB and batch-one CPU latency is
    recorded.

Mechanism requires the task-grounded cell's selection paired-effect objective
and transfer effect-score MSE each to be at most `0.90` times the ungrounded
cell.

Predictive value requires:

- a selected nonzero gain;
- transfer downstream-effect MSE at most `0.90` times raw, supervised
  task-contract, and ungrounded JEPA;
- at least 60% transfer-pair wins against both controls; and
- selection downstream-effect MSE strictly below both controls.

The witness lane separately requires at most 5% transfer control-trajectory
false alarms, at least 80% treatment detection, and median post-onset delay at
most ten transitions. It cannot rescue failed predictive or mechanism gates.

## Artifact and claim boundary

Write through a fresh staging directory and retain selected models, gain
curves, raw hashes, predictions, witnesses, alert decisions, original/restored
arrays, independent assessment, copied sources, and a SHA-256 manifest.

The result may establish only whether this exact hard-sufficiency,
task-grounded residual contract adds value on the open fixed-stack corpus.
