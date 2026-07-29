# Task-grounded Contract-JEPA tracer v1 results

## Decision

**Reject this task-grounded Contract-JEPA recipe. Do not run seed robustness
or sealed confirmation.**

The hard contract worked as engineering: raw dynamics remained immutable,
gain zero was exact, corrections stayed bounded, every safety gate passed, and
all three residual cells selected nonzero gain. The task-grounded candidate
slightly improved raw transfer prediction, but it did not reach the frozen
10% value threshold and it lost to both the supervised and ungrounded
controls. Its witness alert had 100% treatment sensitivity and 100% control
false alarms under topology transfer.

## Reproducible evidence

- implementation commit:
  `97f27a3ebaa4561c440759122fa21206186d0fee`;
- immutable artifact:
  `artifacts/action-dynamics/prototype-task-grounded-contract-jepa-v1`;
- artifact size: about 142 MiB;
- 40 fitting, 10 selection, 10 calibration, 20 IID evaluation, and 10
  held-topology evaluation pairs; and
- independently recomputed assessment and verified SHA-256 manifest.

## Held-topology result

| model | selected gain | overall MSE | action-overlap MSE | downstream-effect MSE |
|---|---:|---:|---:|---:|
| raw rank-32 | 0 | 0.105744 | 0.859940 | 0.143833 |
| task-grounded Contract-JEPA | 1 | 0.104929 | 0.849399 | 0.142319 |
| supervised task contract | 1 | 0.105133 | 0.847610 | 0.141778 |
| ungrounded Contract-JEPA | 1 | 0.099700 | 0.850766 | 0.141466 |

The task-grounded cell improved raw:

- overall MSE by 0.77%;
- action-overlap MSE by 1.23%; and
- downstream-effect MSE by 1.05%.

Those are real open-development improvements, but far below the 10% gate. The
supervised cell had lower action and effect error, while the ungrounded JEPA
had the best overall and effect error. Task grounding did not contribute
incremental predictive value.

The candidate beat the supervised cell on four of ten transfer pairs and the
ungrounded cell on two. Selection also favored the ungrounded cell on
downstream effect.

## Mechanism result

Task grounding strongly trained the explicit effect-score head:

- transfer effect-score MSE was `1.8907`, versus `15.0091` for the ungrounded
  head; and
- selection effect-score MSE was `2.8089`, versus `23.2547`.

But the corrected paired-effect trajectory itself barely changed:

- selection paired-effect MSE was `1.5178`, versus `1.5212` ungrounded; and
- transfer paired-effect MSE was `2.6997`, versus `2.7267`.

The witness learned its supervised quantity, but that signal did not reshape
the residual representation enough to improve the predictive task.

## Witness alert result

The calibration-control trajectory maximum was `0.1834`. On held topology:

- control-trajectory false alarms: 100%;
- treatment-trajectory detection: 100%; and
- median post-onset delay: zero transitions.

This is clean topology-dependent score drift, not a usable operating point.
The ungrounded unused head had zero false alarms and zero detections. The
explicit witness separated declared treatments but did not transfer its
control scale.

## Safety and runtime

Every safety gate passed:

- raw artifact hash unchanged across all fits;
- exact raw mean and variance at gain zero;
- bounded correction and causal public seams;
- restoration within `1e-6`;
- raw-safe transfer overall and action-overlap errors;
- 100% attribution, 100% no-action specificity, and 100% action sanity;
- identical `49,551` training and `41,855` inference parameters per cell;
- 1.59 MiB candidate bundle; and
- 0.470 ms median local batch-one CPU latency.

This validates the contract pattern as a safe way to test auxiliary models.
It does not justify deploying the learned correction.

## Consequence for Error-Certificate-JEPA

The next tracer should preserve the raw prediction completely and stop trying
to improve its mean. It should test only whether JEPA features sharpen a
calibrated error bound over raw-feature and conformal controls. The topology
drift observed here makes coverage on the held topology a mandatory gate.

