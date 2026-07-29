# HEPA telemetry tracer v1 result

## Decision

**Reject this exact HEPA telemetry recipe. Do not run fixed-seed robustness or
promote it into the alerting system.**

The clean-room treatment produced finite, monotone, calibrated event
probabilities; retained observable state; met the frozen edge budget; and
improved held-topology Brier score over the whole-pair horizon-deranged null.
It did not deliver the preregistered alert value:

- treatment detection was 50%, below the 80% gate;
- the deranged null also detected 50%, so improvement was zero rather than
  the required ten percentage points; and
- both the supervised-from-scratch control and the raw effect reference
  detected 60% at the same zero control false-alarm rate.

The valid immutable evidence is:

`artifacts/action-dynamics/prototype-hepa-jepa-v1`

This is open-development, one-seed evidence. It is not sealed confirmation or
production authorization.

## Evidence correction

The first completed run improperly included held-worker-topology calibration
pairs. The shared JEPA ladder freezes fitting, selection, and calibration to
worker topologies one and two. That run was invalidated before ticket closure
and is preserved rather than deleted:

`artifacts/action-dynamics/prototype-hepa-jepa-v1-invalid-calibration-topology-leak`

The corrected runner also separates checkpoint production (`fit`) from
checkpoint choice (`select`), so selection tensors no longer cross the public
fitting seam. No model architecture, seed, optimizer, step count, label,
checkpoint choice, gate, or evaluation value changed. The corrected
calibration uses ten in-distribution pairs.

Two further scientifically equivalent runs are retained for provenance:

- `artifacts/action-dynamics/prototype-hepa-jepa-v1-superseded-shared-helper-dependency`
  was produced before the HEPA module owned all of its implementation
  dependencies; and
- `artifacts/action-dynamics/prototype-hepa-jepa-v1-invalid-implementation-commit-identity`
  recorded the pre-implementation Git revision rather than the revision
  containing the runner it executed.

The final stored-array assessor derives protocol checks, parameter counts,
edge summaries, calibration, thresholds, and decisions from raw evidence. It
also reproduces serialized-model tokens, probability surfaces, calibrated
surfaces, and the public alert-policy decisions byte-for-byte.

## Frozen treatment

The implementation follows the mechanism described in the
[HEPA paper](https://arxiv.org/abs/2605.11130):

- a shared, jointly optimized two-layer causal context/target encoder;
- log-uniform cumulative future-interval targets;
- L1 latent prediction plus SIGReg at `alpha = 0.1`;
- no EMA teacher or stop-gradient;
- a frozen encoder in stage two;
- a horizon-conditioned predictor and shared hazard head; and
- a discrete survival CDF that is monotone by construction.

The width-64 edge tracer has 78,082 inference parameters. Stage one ran for
400 optimizer steps. Stage two ran for 300 steps, with selection choosing step
200 for HEPA and step 250 for the deranged null.

The action-blind event is the first crossing of a robustly standardized
observable one-step state-change norm. Its threshold, `45.3877443`, was fit
from the 95th percentile of fitting-control trajectory maxima. Action kind and
target identity are not model inputs.

## Held-topology result

| Model | Calibrated Brier | ECE | Control FPR | Treatment detection | Median delay | Worst delay |
|---|---:|---:|---:|---:|---:|---:|
| HEPA | **0.015137** | 0.014143 | 0% | 50% | 4.0 | 59 |
| Horizon-deranged JEPA | 0.017032 | **0.001919** | 0% | 50% | 6.0 | 59 |
| Supervised from scratch | **0.014652** | 0.006451 | 0% | **60%** | **2.5** | 59 |
| Raw effect reference | n/a | n/a | 0% | **60%** | 9.0 | 60 |

HEPA improved Brier by 11.1% relative to the alignment-broken null. This
establishes that aligned future-interval pretraining affected the probability
surface. It does not establish alert value: the candidate and null alerted on
the same number of treatments, and the candidate remained worse than
supervised-from-scratch Brier and detection.

The four-transition median delay passes its gate, but the 59-transition worst
delay exposes a long tail that the median alone would hide.

## Representation and edge result

The fitting-only state probe passed comfortably:

| Representation | Held-topology aggregate NRMSE |
|---|---:|
| HEPA entity tokens | **0.08330** |
| Matched width-64 entity PCA | 0.24019 |

All six varying observed entities were reported; the terminal unobserved
entity remained explicitly excluded.

Edge diagnostics for the candidate were:

- 78,082 inference parameters;
- 1,981,889 serialized candidate-and-sidecar bytes;
- 1.71 ms local batch-one CPU latency over 100 repetitions; and
- 2.75 GB process peak RSS while the complete training and assessment bundle
  was resident.

The RSS value is a process-level training/assessment diagnostic, not a claim
that candidate inference requires that memory.

## Gate outcome

Passed:

- finite, bounded, monotone restored CDF;
- equal treatment/null inference capacity with only target alignment changed;
- state retention within the matched-PCA margin;
- calibrated Brier within `1.05 ×` deranged;
- zero control-trajectory false alarms;
- median post-onset delay at most ten;
- serialized edge budget and runtime diagnostics; and
- exact restoration of tokens, probability surfaces, calibrated outputs,
  thresholds, and decisions.

Failed:

- treatment detection at least 80%; and
- at least ten percentage points of detection improvement over the deranged
  null.

## Interpretation

This experiment separates three claims:

1. The HEPA recipe can learn an edge-sized, restorable, state-rich monotone
   probability surface on this corpus.
2. Correct future alignment improves its proper score over a matched
   alignment-broken null.
3. That improvement is insufficient for deployment: sensitivity is low,
   identical to the JEPA null, and worse than simpler controls.

The bounded negative result rejects this label construction and training
recipe as the real-world alert-policy adapter. It does not reject
horizon-conditioned event prediction generally. The next independent target
is the complete SC-JEPA interaction: codebook and multi-resolution prediction
must be tested jointly because earlier Quantis tracers isolated only parts of
that mechanism.
