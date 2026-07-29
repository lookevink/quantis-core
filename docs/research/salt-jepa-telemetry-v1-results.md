# SALT-JEPA telemetry tracer v1 review record

## Review disposition

**Preserve v1, but do not treat it as conclusion-bearing. A corrected v2
evidence run is required before accepting the numerical rejection.**

The v1 numbers strongly indicate rejection, but post-run review found that
mask semantics, both teacher restorations, complete diagnostic restoration,
capacity, latency, causality, and deployed bytes were not all independently
recomputed from retained evidence. The producer recorded those claims and the
artifact remains reproducible, but a conclusion-bearing contract must not
trust them.

This rejects the exact telemetry translation on the fixed Quantis lab stack.
It does not reject SALT on video or frozen-teacher representation learning in
general.

## Reproducible evidence

- implementation commit:
  `a60b84123dc81890cbba594a1099c8f26368bd56`;
- preserved, superseded immutable artifact:
  `artifacts/action-dynamics/prototype-salt-jepa-v1`;
- source recipe: Li et al.,
  [*Rethinking JEPA: Compute-Efficient Video SSL with Frozen Teachers*](https://arxiv.org/abs/2509.24317);
- 320 reconstructive-teacher steps and 1,280 static-target student steps per
  SALT cell;
- 40 fitting, 10 selection, 10 calibration, 20 IID evaluation, and 10
  held-topology evaluation pairs; and
- stored assessment and a verified SHA-256 manifest under the v1 evidence
  contract.

The 262 MiB artifact retains both complete teacher/student cells, the
reconstructive teacher, PCA and raw controls, downstream probes, anchor and
mask schedules, original/restored public outputs, diagnostic tensors, role
identifiers, copied reproduction sources, and pure stored-array assessment.

## Held-topology result

| representation | overall MSE | action-overlap MSE | downstream-effect MSE |
|---|---:|---:|---:|
| raw rank-32 | 0.105744 | 0.859940 | 0.143833 |
| SALT-JEPA student | 0.141959 | 1.402418 | 0.274069 |
| deranged-target SALT | 0.144943 | 1.398683 | 0.277348 |
| reconstructive teacher | 0.145226 | 1.398229 | 0.278659 |
| matched PCA | 0.147181 | 1.444091 | 0.274703 |

Relative to raw, the SALT student:

- increased overall error by 34.25%;
- increased action-overlap error by 63.08%; and
- increased downstream-effect error by `1.91×`.

It improved downstream-effect MSE by 1.65% over its reconstructive teacher,
1.18% over deranged SALT, and 0.23% over PCA. Those small differences missed
the frozen 10% value requirement. The candidate beat its teacher on five of
ten transfer pairs rather than the required six.

## Mechanism result

Aligned masked-teacher latent L1 was:

| role | SALT-JEPA | deranged-target SALT | aligned improvement |
|---|---:|---:|---:|
| selection | 0.069934 | 0.075164 | 6.96% |
| transfer | 0.108744 | 0.114025 | 4.63% |

The frozen teacher supplied an identity-specific signal, but the advantage
was below 10% in both roles. It also did not translate into a useful action
probe. The aligned student had transfer aggregate state NRMSE `0.008313`
versus `0.008574` for the teacher, but it regressed some observed entities and
was much worse than the deranged student's `0.001623`. Target alignment
therefore did not explain useful observable-state retention.

## Safety and edge behavior

The engineering contract passed except for predictive raw safety:

- fitting, selection, calibration, IID, and transfer pair and trajectory
  identifiers were disjoint;
- every mask hid exactly 126 of 140 tokens while retaining declared
  anchor-time visibility;
- aligned and deranged cells had identical capacity;
- each teacher hash was unchanged throughout student fitting;
- model, probe, teacher, and diagnostic restoration error was exactly zero;
- attribution, no-action specificity, and action sanity were all 100%;
- the deployed student/probe bundle was 8.32 MB; and
- median local batch-one CPU latency was 42.33 ms.

Selection and transfer overall/action MSE each exceeded the frozen raw-safety
ceiling. A small, restorable inference path cannot compensate for that
regression.

## Consequence

SALT closes the separately reconstructed static-teacher omission on the
current stack. Its student learned the teacher slightly better when targets
were aligned, but the teacher semantics did not preserve the intervention
effect that raw telemetry already exposed.

Proceed to the exact LeNEPA omission: route next-latent prediction through a
disposable projected space with SIGReg on the deployed backbone. Compare it
with an equal-capacity unprojected cell, a no-prediction SIGReg cell, matched
PCA, and raw dynamics. Do not carry SALT's teacher or decoder into that
experiment.
