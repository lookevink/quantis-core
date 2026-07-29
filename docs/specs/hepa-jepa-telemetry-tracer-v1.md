# HEPA telemetry tracer v1

Status: frozen before implementation
Role: alert-policy adapter
Ticket: `012-test-hepa-event-alert-tracer`

## Claim boundary

This tracer asks whether a clean-room, edge-sized implementation of the HEPA
training recipe produces a useful event-time distribution on the existing
open action-dynamics corpus. It does not replace the raw rank-32 predictive
core, open sealed data, or authorize production paging.

The implementation is derived from the HEPA paper, not copied from the
non-commercial reference repository.

## Frozen data roles

Use the content-addressed
`action_conditioned_jepa_topology_transfer_v1` cache and its whole-pair roles.
Within each role, the largest worker topology is the transfer partition.

- stage-one pretraining: fitting in-distribution windows only;
- stage-two event finetuning: fitting in-distribution windows only;
- checkpoint selection: selection in-distribution windows only;
- probability calibration and alert threshold: calibration in-distribution
  windows (worker topologies one and two) only;
- assessment: in-distribution and held-topology evaluation windows, opened
  only after fitting, selection, and calibration are complete.

No pair crosses roles. No evaluation tensor influences fitting, selection, or
calibration.

## Action-blind event definition

The audit's phrase “normalized downstream-effect norm” is frozen here at the
observable single-trajectory seam so it is executable without action or target
identity.

1. Reconstruct each fitting control trajectory from its overlapping normalized
   windows.
2. On observed entity-feature slots, fit the median and robust scale
   (`1.4826 * MAD`, with standard-deviation and unit fallbacks) of one-step
   state changes.
3. At transition `s`, define the effect score as the root-mean-square of the
   standardized observed one-step change.
4. Reduce every fitting control trajectory to its maximum effect score.
5. Freeze the event threshold at the higher empirical 95th percentile of
   those trajectory maxima.
6. A trajectory event is its first transition whose effect score is strictly
   greater than the threshold.
7. For context transition `t` and horizon `h`,
   `y(t,h) = 1` exactly when that first event is in `(t,t+h]`.

Control membership is used only to fit the nominal label transform. Action
kind, target entity, magnitude, pair identity, controls, and future values are
not model inputs. Action truth is used later only to compute post-onset alert
breakdowns.

## Treatment

- two-layer causal telemetry Transformer;
- width 64, four heads, feed-forward width 128, zero dropout;
- one public width-64 token per entity at the context boundary;
- pooling only inside the predictor and event head;
- shared context/target encoder weights, jointly optimized;
- bidirectional target encoding of cumulative future intervals;
- one discrete horizon per sample, drawn with probability proportional to
  `1/h` over horizons 1 through 10;
- two-layer horizon-conditioned predictor;
- stage-one loss
  `0.9 * mean_absolute_error(predicted, target) + 0.1 * SIGReg(predicted)`;
- no EMA teacher, stop-gradient, masks, graph message passing, action decoder,
  retrieval index, or second regularizer;
- frozen encoder during stage two;
- the pretrained predictor and one shared linear hazard head are finetuned
  with positive-weighted BCE on the cumulative probability surface;
- `lambda_h(t) = sigmoid(w' g(h_t, h) + b)`;
- `p(t,h) = 1 - product_{j=1..h}(1-lambda_j(t))`.

The frozen interpretable run uses 400 stage-one and 300 stage-two optimizer
steps, deterministic CPU execution, batch size 64, AdamW, and seed 12012.
Stage-two checkpoints are scored every 50 steps by uncalibrated selection
Brier score; exact ties choose the earlier checkpoint.

## Controls

The JEPA-specific null is identical except that stage-one future intervals
come from a deterministic whole-pair derangement. Both arms of a logical pair
move together to another fitting pair; arm position and transition are
preserved. The mapping has no fixed points. Stage-two labels and all inference
parameters remain matched.

Also report:

- the current action-blind normalized effect score as the raw alert reference;
- a capacity-matched survival model trained from scratch on fitting labels;
- a width-64, entity-preserving PCA representation for state-retention
  comparison.

Neither replaces the horizon-deranged JEPA null.

## Calibration and alert decision

Fit a single increasing logit calibration map
`sigmoid(a * logit(p) + b)`, with `a > 0`, by a frozen finite grid minimizing
calibration Brier score. Apply the same map to every horizon, preserving the
CDF ordering.

For each calibration control trajectory, reduce the calibrated horizon-10
probability to its maximum over contexts. The alert threshold is the higher
empirical 95th percentile of these maxima. An alert occurs when calibrated
`p(t,10)` is strictly greater than that threshold.

## Gates

The tracer advances to fixed-seed robustness only if all of these pass on the
held-out worker topology:

1. restored CDFs are finite, in `[0,1]`, and non-decreasing in horizon;
2. treatment and deranged null have equal inference parameter counts and
   differ only in the stored stage-one target alignment;
3. the fitting-only frozen state probe has aggregate NRMSE no worse than
   `1.05 *` matched PCA, with every varying observed entity reported;
4. calibrated treatment Brier is no worse than `1.05 *` deranged-null Brier,
   with 10-bin ECE reported;
5. control-trajectory false alarms are at most 5%;
6. treatment-trajectory post-onset detection is at least 80%;
7. median post-onset delay is at most 10 transitions;
8. treatment detection is at least 10 percentage points above the deranged
   null at the same at-most-5% false-alarm budget;
9. serialized candidate plus sidecars is at most 16 MiB and batch-one CPU
   latency plus process peak RSS are recorded; and
10. restoration reproduces public tokens, probability surfaces, calibrated
    surfaces, thresholds, and alert decisions within `1e-6`.

Failure rejects this exact recipe and preserves its implementation and
immutable artifact bundle.
