# PEIRA telemetry tracer v1 results

## Decision

**Reject this exact PEIRA telemetry recipe. Do not run seed robustness or
sealed confirmation.**

Aligned PEIRA learned a strong, noncollapsed inter-view trace signal, while
the pair-deranged null collapsed. That mechanism did not preserve operational
value: every learned representation failed selection safety, aligned PEIRA
retained `1.91×` raw held-topology downstream-effect error, lost to its
deranged null and the masked-autoencoder control, and won only three of ten
transfer pairs. This rejects the fixed two-global-view PEIRA translation on
the Quantis lab stack. It does not reject PEIRA on the paper's image tasks.

## Reproducible evidence

- experiment implementation commit:
  `100b3b9b2a0e5aed89a0c398978290055692fbe3`;
- float32 replay-tolerance correction commit:
  `704a33d`;
- conclusion-bearing immutable artifact:
  `artifacts/action-dynamics/prototype-peira-v1`;
- artifact-manifest SHA-256:
  `10bb06ead4bb6571a5c9c1e4a3e954dbe688d2063c40a0ccfc635d1e596c47be`;
- 1,600 steps for each of two equal-capacity PEIRA cells; and
- 40 fitting, 10 selection, 10 calibration, 20 IID evaluation, and 10
  held-topology evaluation pairs.

The 673 MiB artifact contains 54 files: both full PEIRA cells, final `P/Q`,
copied prior controls and their frozen manifest, PCA and raw models, all
probes, exact consumed schedules, every per-step minibatch and running
moment, role representations and predictions, restoration and diagnostic
evidence, 100 raw latency samples, a copied isolated assessor and receipt,
the report, and an identity manifest.

## Held-topology result

| representation | overall MSE | action-overlap MSE | downstream-effect MSE |
|---|---:|---:|---:|
| raw rank-32 | 0.105744 | 0.859940 | 0.143833 |
| aligned PEIRA | 0.142474 | 1.422226 | 0.274844 |
| pair-deranged PEIRA | 0.141826 | 1.389546 | 0.272800 |
| complete LeJEPA | 0.142355 | 1.428005 | 0.274014 |
| masked autoencoder | 0.140543 | 1.391670 | 0.269572 |
| matched PCA | 0.147181 | 1.444091 | 0.274703 |

Relative to raw, the candidate increased overall error by 34.74%, action
error by 65.39%, and effect error by `1.91×`. It was 1.95% worse than the
best representation control on effect error and beat that control on only
30% of transfer pairs. Its selection effect MSE was `0.191003`, versus raw
`0.128783`; every representation exhausted the frozen ridge set without a
selection-safe choice.

## Mechanism result

| role | aligned `-E_PEIRA` | deranged `-E_PEIRA` | aligned projector rank | deranged projector rank |
|---|---:|---:|---:|---:|
| selection | 17.3647 | 0.4050 | 54.87 | 1.85 |
| transfer | 13.5141 | 0.4040 | 49.77 | 1.73 |

The aligned cell passed the trace-advantage and noncollapse gates. Its
backbone effective rank was 27.10 on selection and 24.20 on transfer, versus
3.49 and 3.00 for the deranged null. PEIRA therefore learned a real
pair-alignment mechanism rather than merely reproducing the null.

The stricter eigenvector-alignment advantage still failed. Candidate versus
null alignment was `0.9990` versus `0.9776` on selection and `0.9946` versus
`0.9893` on transfer; neither role cleared the preregistered five-percentage-
point margin. More importantly, the strong trace signal did not survive the
observable action-conditioned forecast test.

## Evidence correction and safety

The immutable artifact's original assessor marked
`training_moments_recompute` false. Diagnosis found one mismatch: aligned
step 698 stored a float32 training loss of `-0.0036225319`, while the
independent float64 trace reconstruction was `-0.0036227522`, an absolute
difference of `2.20e-7`. Every moment recurrence and every deranged step
replayed.

Commit `704a33d` raises only the float32 `loss` and `auxiliary_value`
absolute replay tolerance from `1e-7` to `1e-6`; moment recurrences remain
at `1e-12`, and final operators, trace diagnostics, schedules, restoration,
and all other checks retain their original bounds. A complete reassessment
of the immutable artifact changed exactly:

- `protocol_checks.training_moments_recompute`: false to true; and
- `safety_gates.training_moments_recompute`: false to true.

The decision remained `reject_peira_recipe`. All protocol checks pass under
the corrected assessor.

Independent safety failures remain decisive. Candidate representation and
probe restoration reached `3.79e-6`, above the frozen `1e-6` bound.
Selection and transfer overall/action errors also failed every raw-safety
margin. Attribution, no-action specificity, action sanity, causal public
inference, capacity equality, schedule replay, copied-control identity,
source isolation, operator replay, and bundle identity passed.

The two PEIRA cells each used 116,848 training and 83,760 inference
parameters. The deployed bundle was 3,201,905 bytes. Its 100 retained
batch-one CPU samples yielded 69.82 ms median and 74.27 ms p95.

## Consequence

PEIRA closes the stochastic-compositional trace objective omission on this
stack. It is scientifically useful that aligned pairing prevented collapse
and separated sharply from derangement, but the learned geometry discarded
action-conditioned forecast value already present in raw telemetry.

Proceed to VISReg as a bounded regularizer substitution on complete LeJEPA.
Do not carry PEIRA's moment state, projector, or objective into that tracer.
