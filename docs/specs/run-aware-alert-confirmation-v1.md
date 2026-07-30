# Run-aware alert confirmation v1

Status: **frozen before collection**

## Question

Does the already-confirmed rank-32 action-conditioned predictive core produce
useful warnings when its hidden-action residual is converted into a
run-aware alert under a frozen 5% false-alarm budget?

This experiment evaluates an alert policy. It does not refit or reconfirm the
predictive core.

## Primary references and adaptation

- Zhang et al., “Conformal anomaly detection in event sequences,” ICML 2025
  ([PMLR](https://proceedings.mlr.press/v267/zhang25dn.html)), motivates
  calibrating the operational alert unit rather than treating dependent
  event/window scores as independent tests.
- Page, “Continuous inspection schemes,” *Biometrika* 1954
  ([DOI](https://doi.org/10.1093/biomet/41.1-2.100)), supplies the resettable
  cumulative-sum pattern.

The adaptation is deliberately narrower than either source. It makes no
continuous-time rescaling or general e-value claim. Empirical tail
probabilities are used only as monotone evidence inputs; the finite-sample
false-alarm argument is attached to the maximum complete-run statistic on
disjoint control runs. The `log(4)` subtraction supplies negative drift under
routine probabilities and fixes the previous non-resetting implementation.

## Why this policy

The old open-development detector calibrated overlapping windows and then
accumulated only nonnegative evidence. Its statistic therefore grew with run
length and traded 60% treatment detection for a 17.5-transition median delay.

An open tracer on the already-exposed low-rank confirmation corpus corrected
that defect by allowing negative evidence to reset the cumulative statistic.
With the frozen 30/30/60 development-only role split and observable `t + 1`
timestamps, the candidate form had 0/60 control alarms, 60/60 treatment
detections, a six-transition median delay, and all five action families at
100%. The same policy form around persistence detected 50/60 treatments and
reached 58.3% in its weakest family. These numbers selected the policy form
only; they are not confirmation evidence.

## Frozen evidence roles

The fresh campaign has 120 matched treatment/control pairs across five action
families and three worker topologies, with eight pairs per action-topology
cell. Within each cell, matched-pair IDs are sorted lexicographically:

- positions 0-1 are `score_reference` pairs;
- positions 2-3 are `threshold_calibration` pairs; and
- positions 4-7 are `sealed_evaluation` pairs.

This produces 30 reference, 30 threshold-calibration, and 60 evaluation
pairs. Both twins stay in the same role. Collector labels named `training` or
`validation` are transport compatibility fields and have no analytic meaning.

## Frozen policy

At alert time the action tensor is replaced by the neutral/no-action value.
For each transition:

1. roll the frozen predictive core forward one transition;
2. compute mean normalized squared error over all observed entities and state
   features;
3. convert the score to an empirical upper-tail probability using every
   control window in the `score_reference` role and the plus-one correction;
4. update
   `C_t = max(0, C_(t-1) - log(p_t) - log(4))`; and
5. alert once per run at the first strict crossing of the run threshold.

The threshold is the
`ceil((n + 1) * (1 - 0.05))` order statistic of the maximum `C_t` from each
of the 30 disjoint threshold-calibration control runs. With 30 runs this is
the maximum calibration-control statistic and gives a conservative
split-conformal run-level exceedance bound of at most `1/31 = 3.23%` under
exchangeability. No point/window false-alarm claim is made.

Persistence receives its own reference probabilities and run threshold under
the identical policy. It is a value control, not a source of the candidate
threshold.

## Frozen usefulness conjunction

Only the 60 sealed-evaluation pairs enter the decision. The claim passes only
if every gate passes:

- candidate control-run false-alarm rate is at most 5%;
- a one-sided exact binomial lower-tail test rejects a 15% or worse
  control-run false-alarm rate at `p <= 0.05`;
- treatment pre-onset alert rate is at most 5%;
- treatment detection is at least 90%;
- detection while the intervention is active is at least 85%;
- median valid detection delay is at most eight transitions;
- every action family has at least 75% detection;
- candidate detection exceeds persistence by at least 10 percentage points;
  and
- candidate within-active detection exceeds persistence by at least 10
  percentage points.

An alert is latched. If its first crossing is observed at or before
intervention onset, it is a pre-onset alert and cannot later count as a valid
detection. A score for action transition `t` becomes observable only with
state `t + 1`, so its alert timestamp is `t + 1`. Detection delay is measured
from action onset to the first valid observed crossing, and “while active”
requires that observation timestamp to be no later than the final active
action-transition index.

## Execution and evidence boundary

The executable contract is
`lab/action_dynamics/run-aware-alert-confirmation-contract-v1.json`. It binds
the prior confirmed model and assessment, preprocessing artifact, collection
seed, policy, role rule, gates, implementation files, and independent
assessor by SHA-256.

The campaign has no automatic retries and no overwrite. Raw qualification
failure blocks scoring. After collection there is no model fitting, policy
selection, threshold-rule change, role change, or gate change. A failed
conjunction definitively rejects this exact warning policy; it does not erase
the prior predictive-core confirmation.

Passing supports useful warnings only on the fixed Quantis checkout lab and
declared intervention library. It does not establish production paging,
cross-stack transfer, unknown-intervention detection, root-cause attribution,
or drift-safe adaptive calibration.
