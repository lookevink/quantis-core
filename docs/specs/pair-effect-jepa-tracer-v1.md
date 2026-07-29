# PairEffect-JEPA tracer v1

## Question

Can a joint-embedding predictor trained on randomized matched
treatment/control differences improve held-topology intervention-effect
prediction without replacing the auditable raw-state path?

This is a one-seed open-development tracer. Passing authorizes fixed-seed
robustness only. It does not authorize paging, autonomous remediation, or
sealed confirmation.

## Frozen role and data

The candidate is an **effect correction** over a frozen no-action
rank-32 predictive core. The raw action-conditioned rank-32 model remains the
primary comparator.

Use the existing content-addressed topology-transfer cache and pair-atomic
roles:

- fitting: 40 in-distribution matched pairs;
- selection: 10 disjoint in-distribution pairs;
- calibration: unused by model fitting;
- IID evaluation: 20 disjoint in-distribution pairs; and
- transfer evaluation: 10 held-worker-topology pairs.

Normalizers, target encoders, checkpoints, and gains use fitting or selection
only. Evaluation roles never select a model, hyperparameter, or threshold.

## Frozen cells

Fit three equal-capacity cells:

| Cell | Paired latent loss | Observable effect target |
|---|---:|---|
| `pair_effect_jepa` | matched treatment minus control | matched treatment minus control |
| `supervised_pair_effect` | disabled | matched treatment minus control |
| `deranged_pair_jepa` | treatment minus a different cell-matched control | the same deranged difference |

Derangement is a deterministic cycle within action-kind and worker-topology
cells. It preserves treatment, control, action, topology, transition, marginal
targets, optimizer steps, and capacity while breaking randomized twin
identity.

## Model

Each fitting unit contains the treatment and control windows with the same
matched-pair id and transition index.

1. An online entity-preserving encoder maps the treatment history's current
   observable state to seven width-16 tokens.
2. An EMA target encoder maps the treatment and control future states.
3. Their per-horizon target difference is
   `stop_gradient(E_target(treatment) - E_target(control))`.
4. A horizon-conditioned predictor receives the online context tokens,
   future controls, and the declared treatment action and predicts that latent
   effect.
5. A shared entity decoder maps the predicted effect tokens to the observable
   treatment-minus-control trajectory.
6. The control arm, supplied with the no-action tensor, is trained to emit
   exactly zero effect.

The matched JEPA cell minimizes observable-effect MSE plus `0.2` times latent
effect L1. The supervised cell has the same graph and parameters but a zero
latent-loss weight. The deranged cell uses the same objective as the matched
JEPA cell against the frozen derangement.

Use deterministic CPU AdamW, seed `23021`, 800 steps, pair-blocked batches,
learning rate `5e-4`, weight decay `1e-3`, gradient clipping at one, EMA
decay `0.996`, and checkpoints every 100 steps. Select each cell's checkpoint
by its own selection-role observable effect MSE.

## Public inference

`predict_effect(histories, future_controls, future_actions, graph)` accepts
only current histories and declared future inputs. It returns one bounded
observable effect trajectory.

The composed predictive core is:

`raw_no_action_rollout(current, controls) + predicted_effect(current, controls, action)`.

No control trajectory, future observation, pair id, target action label, or
evaluation statistic enters public inference. A no-action request must return
a numerically zero correction. Serialization/restoration must preserve both
the effect prediction and composed rollout.

## Evaluation

Report for every cell and the raw action-conditioned comparator:

- pair-balanced overall and action-overlap MSE;
- paired treatment-minus-control downstream-effect MSE;
- per-pair downstream-effect errors and win fractions;
- action-and-target hit@1 and no-action specificity on the frozen library;
- correct-action versus no-action and whole-pair-shuffled-action sanity;
- matched versus deranged observable-effect error;
- training/inference parameters, serialized bytes, batch-one CPU latency,
  finite rollout, and restoration parity.

## Gates

All safety gates must pass:

1. all stored and independently recomputed values are finite;
2. all three cells have identical training and inference parameter counts;
3. restored effect and rollout outputs match originals within `1e-6`;
4. public inference rejects future observations, pair identities, and target
   truth;
5. no-action correction is zero within `1e-7`;
6. transfer overall and action-overlap MSE are each no worse than `1.05`
   times raw;
7. action-and-target hit@1 is at least 95%;
8. no-action specificity is exactly 100%;
9. correct actions beat both no-action and shuffled actions on at least 80%
   of treatment pairs; and
10. the candidate bundle is at most 16 MiB and batch-one CPU latency is
    recorded.

The mechanism gate passes only if matched observable-effect MSE is at most
`0.90` times the deranged-pair cell on selection and transfer.

The value gate passes only if:

- transfer downstream-effect MSE is at most `0.90` times both the raw
  action-conditioned model and the supervised paired-effect cell;
- the candidate beats the supervised cell on at least 60% of transfer pairs;
  and
- selection downstream-effect MSE is strictly lower than the supervised cell.

Every safety, mechanism, and value gate must pass. Failure rejects this exact
PairEffect-JEPA recipe while retaining its implementation and artifact.

## Artifact contract

Write through a fresh staging directory and publish atomically. Retain all
selected models, raw reference, original/restored predictions, role tensors
needed by the assessor, per-pair evidence, independent assessment, report,
source identities, reproduction-source copies, and a SHA-256 manifest.

The artifact and conclusion may establish only:

> On the fixed Quantis lab stack and open corpus, a paired latent-effect
> objective did or did not improve intervention-effect prediction over its
> supervised, deranged, and raw controls.

