# Error-Certificate-JEPA tracer v1 results

## Decision

**Reject this Error-Certificate-JEPA recipe. Do not run seed robustness or
sealed confirmation.**

The learned certificate preserved the raw forecast exactly and passed every
safety gate in the corrected source-bound run. It also reached 99.91% point
coverage on held-topology control windows. That high point coverage concealed
an unusable trajectory operating point: two of ten control trajectories
alarmed, no treatment trajectory was detected, and the candidate was no
sharper than its deranged, raw-only, or constant-conformal controls.

## Reproducible evidence

- implementation commit:
  `79458ea43dedc3acfe93b6fa6d57b72623242ba8`;
- conclusive immutable artifact:
  `artifacts/action-dynamics/prototype-error-certificate-jepa-v2`;
- retained invalid first attempt:
  `artifacts/action-dynamics/prototype-error-certificate-jepa-v1`;
- 40 fitting, 10 selection, 10 calibration, 20 IID evaluation, and 10
  held-topology evaluation pairs; and
- independently recomputed assessment and verified SHA-256 manifest.

The first artifact remains reproducible but is not the conclusion-bearing
run. Its certificate evidence was serialized as float32 while its exact
calibration gate used a `1e-7` tolerance, producing up to `9.54e-7` of
round-trip error. The runner now stores calibration evidence losslessly; the
v2 artifact passes the repaired regression and all safety gates.

## Held-topology result

| certificate | control point coverage | simultaneous coverage | control false alarms | mean bound | treatment detection | median delay |
|---|---:|---:|---:|---:|---:|---:|
| JEPA certificate | 99.91% | 80% | 20% | 5.08344 | 0% | — |
| raw-only learned certificate | 99.91% | 80% | 20% | 5.11504 | 0% | — |
| deranged-JEPA certificate | 99.91% | 80% | 20% | 5.08237 | 0% | — |
| constant conformal | 99.91% | 80% | 20% | 4.86001 | 20% | 8 |

The candidate missed the frozen requirement of 100% simultaneous coverage,
equivalently at most 5% false alarms with ten controls. More importantly,
widening it enough to cover trajectories would only reduce its already-zero
treatment sensitivity.

The constant conformal comparator was 4.40% sharper than the candidate and
was the only cell to detect any transfer treatment. Raw-only was 0.62% wider
than the candidate, far short of the required 10% JEPA advantage.

## Mechanism result

Selection unadjusted pinball losses were:

| cell | selection pinball |
|---|---:|
| JEPA certificate | 0.053573 |
| deranged-JEPA certificate | 0.053641 |
| raw-only certificate | 0.053307 |

The JEPA path improved on derangement by only 0.13%, not the required 10%,
and lost to raw-only. On transfer, its mean bound was slightly *wider* than
derangement (`5.08344` versus `5.08237`) while both had the same failed
trajectory coverage. The aligned latent target therefore contributed no
measurable reliability signal.

## Calibration and safety

Every safety gate passed in v2:

- all evidence was finite and every bound was non-negative;
- all three cells had identical `87,409` training and `79,713` inference
  parameters;
- the rank-32 raw artifact hash was unchanged after every fit;
- every wrapper returned the exact raw mean and variance;
- public inference was causal;
- restored forecasts, bounds, and alert decisions matched exactly;
- learned adjustments and the constant conformal bound independently
  recomputed from calibration-only evidence;
- role isolation was explicit and no evaluation data widened a bound;
- the candidate bundle was 2.28 MiB; and
- median local batch-one CPU latency was 0.495 ms.

The fitted additive adjustments were `4.29201` for the candidate, `4.32018`
for raw-only, and `4.29051` for derangement. Their near equality is another
direct indication that the learned latent path did not alter the usable
certificate.

## Program consequence

The three locally formulated one-stack hypotheses are now closed at tracer
stage:

1. PairEffect-JEPA did not learn matched effects beyond derangement and
   damaged the raw path.
2. Task-grounded Contract-JEPA learned its witness but did not beat its
   controls and its witness scale drifted across topology.
3. Error-Certificate-JEPA preserved raw safely but did not improve calibrated
   sharpness or alert operation.

The appropriate deployment direction remains the deliberately small
non-JEPA stack: immutable raw rank-32 dynamics, direct retrieval, explicit
calibration/abstention, and shadow evaluation. Future JEPA work should open
only when a measured prerequisite—such as missing-channel incidence,
recovery trajectories, or irreducible multimodal residuals—creates a distinct
testable lane.
