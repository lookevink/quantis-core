# PairEffect-JEPA tracer v1 results

## Decision

**Reject this PairEffect-JEPA recipe. Do not run seed robustness or sealed
confirmation.**

The matched-twin objective did not separate from its deranged-pair null and
the composed no-action-raw-plus-effect path substantially regressed the raw
action-conditioned predictive core. It preserved closed-library attribution
and remained edge-small, but neither mechanism nor operational value passed.

This is open-development evidence on the fixed Quantis lab stack. It rejects
the exact tracer, not paired experimental designs or JEPA generally.

## Reproducible evidence

- implementation commit:
  `90530cf7b07cfdf366c7ed72441e93b4c9871c7a`;
- immutable artifact:
  `artifacts/action-dynamics/prototype-pair-effect-jepa-v1`;
- fitting: 40 in-distribution matched pairs;
- selection: 10 disjoint in-distribution pairs;
- IID evaluation: 20 disjoint pairs; and
- transfer evaluation: 10 held-worker-topology pairs.

The artifact contains selected models, composed raw/effect models, stored
assessment tensors, a fresh independent assessment, copied reproduction
sources, and a verified SHA-256 manifest. It occupies about 93 MiB.

## Held-topology result

| model | overall MSE | action-overlap MSE | downstream-effect MSE |
|---|---:|---:|---:|
| raw rank-32 action-conditioned | 0.105744 | 0.859940 | 0.143833 |
| PairEffect-JEPA | 0.211724 | 2.518358 | 0.485602 |
| supervised paired effect | 0.217109 | 2.567372 | 0.493960 |
| deranged-pair JEPA | 0.211418 | 2.519301 | 0.485713 |

Relative to raw, PairEffect-JEPA:

- doubled overall error;
- increased action-overlap error by `2.93×`; and
- increased downstream-effect error by `3.38×`.

It was 1.69% better than the supervised effect bottleneck on downstream
effect and won six of ten transfer pairs, but the required improvement was
10%. More importantly, it was statistically and operationally
indistinguishable from the deranged-pair null.

## Mechanism result

Observable paired-effect MSE was:

| role | PairEffect-JEPA | deranged null | supervised |
|---|---:|---:|---:|
| selection | 8.7981 | 8.7455 | 8.9706 |
| transfer | 6.1255 | 6.1069 | 6.3224 |

The matched objective was slightly *worse* than the deranged null in both
roles. The model learned an action-conditioned average correction rather than
information specific to randomized twin identity. Pair matching therefore
added no demonstrated JEPA mechanism in this formulation.

## Safety and edge behavior

The candidate passed the engineering and attribution checks:

- finite stored evidence and exact independent reassessment;
- identical `41,727` training and `34,031` inference parameters per cell;
- causal public inference and restoration within `1e-6`;
- exactly zero no-action correction;
- 100% action-and-target hit@1 and 100% no-action specificity;
- correct action beat no-action and shuffled action on 90% of transfer pairs;
- 1.42 MiB composed artifact; and
- 0.426 ms median local batch-one CPU latency.

It failed raw-safety for both overall and action-overlap MSE. These failures
are decisive even though the inference path is small and attributable.

## Interpretation

The fixed raw model already conditions directly on the declared action. The
alternative composition asks a learned bottleneck to reconstruct the complete
effect and add it to a no-action rollout. Both matched and deranged training
converged to nearly the same result, indicating that action identity and
average intervention shape dominate the available paired variation.

Task-grounded Contract-JEPA should therefore not inherit PairEffect's
full-trajectory correction. It must keep the raw action-conditioned rollout
unchanged and test whether a bounded, selection-gated residual can add value
only when the training objective is jointly anchored to the actual alert and
effect witnesses.

