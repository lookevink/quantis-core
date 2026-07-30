# Low-rank action-dynamics sealed confirmation v3 results

## Decision

**Confirmed: the new data contains learnable action-conditioned dynamics.**

On the fixed Quantis checkout lab and declared randomized intervention
library, the frozen rank-32 contractive low-rank predictive core improved
ten-step observable-state and downstream-effect forecasts over both its own
neutral-action ablation and persistence on a fresh 120-pair campaign. Every
preregistered data-quality and model decision gate passed.

This confirms the supervised action-conditioned predictive core. It does not
show that graph message passing or topology helps: the earlier graph residual
added no development value and was not the confirmation candidate.

## Evidence boundary

The successful v3 campaign contains 120 matched pairs and 240 captures,
balanced as 24 pairs for each of five intervention/location families across
one-, two-, and three-worker topologies. Collection used six pair-atomic
lanes, 20 batches, the bound application image and build, and zero retries.

All 24 raw corpus gates passed, including complete capture coverage, action
command coverage, exact count resolution, treatment/control schedule
identity, raw effects, recovery, enqueue mechanistic recovery, trace
coverage, truth exclusion, and lane isolation. The corpus was qualified
before the frozen model was restored.

V1 and v2 are excluded. V1 was interrupted because a concurrent host test
process could contaminate telemetry. V2 stopped on a transient HTTP disconnect
and never produced a complete attestation. Neither attempt was qualified or
scored; v3 used a new generator and opaque identities.

## Frozen result

| Metric | Candidate | Neutral-action ablation | Persistence |
|---|---:|---:|---:|
| Pair-balanced action-overlap MSE | 0.366633 | 4.888081 | 3.027495 |
| Downstream-effect MSE | 0.040097 | 0.339078 | 0.227275 |
| Overall normalized MSE, descriptive | 0.066146 | 0.586190 | 0.724327 |

The candidate reduced action-overlap MSE by 92.5% relative to its
capacity-identical neutral-action ablation and by 87.9% relative to
persistence. It reduced downstream-effect MSE by 88.2% and 82.4%,
respectively. Both one-sided matched-pair sign-flip tests returned
`p = 0.00001` with the frozen 99,999 draws.

Every action family passed its at-most-0.90 ratio gate against both controls:

| Action family | Candidate MSE | Candidate / masked | Candidate / persistence |
|---|---:|---:|---:|
| API rejection | 0.411398 | 0.0677 | 0.1576 |
| Postgres lock | 0.910614 | 0.0834 | 0.1119 |
| Redis dequeue delay | 0.274430 | 0.1099 | 0.2401 |
| Redis enqueue delay | 0.021446 | 0.0111 | 0.0206 |
| Worker pause | 0.215275 | 0.0717 | 0.0978 |

The restored predictor remained finite, had spectral radius `0.871358`,
34,503 parameters, and a 543,464-byte canonical serialized artifact. These
passed the frozen 0.98, 40,000-parameter, and 1 MiB envelopes.

## Integrity identities

- contract SHA-256:
  `c7e41d8d47716a8e9372ade768e5526babf18cc7ed2aa2658bb04920d88f4e08`
- model SHA-256:
  `c3456d1314c0d186167c9b63fce608cf65ec923e004c626dfd0343c3fe8b582d`
- qualified source artifact-manifest SHA-256:
  `db8cbf5604a6ca0aac0606926e64d08960784397ef4aceb3736058009617f6ed`
- prediction-manifest SHA-256:
  `e5fa90d91285e21e96ad45f8747d358a18aa1ed86646760ff62a97fef548c085`
- bound execution source commit:
  `beb63a2ce5ffcf4209b81c5607d078a48121f031`

The standalone stored-array assessor reproduced `confirmed` from these exact
hashes after the primary scoring command completed.

## Bounded conclusion

The fresh evidence supports a stronger statement than the prior open
development result:

> Declared interventions carry large, reproducible predictive information
> about future observable telemetry on this lab, and a compact supervised
> contractive state-space model learned that information before the
> confirmation cases existed.

It does not establish unknown-incident causality, graph-topology benefit,
cross-stack transfer, operational alert quality, production readiness, or a
software world model. The next experiment can now focus on whether a
run-aware alert policy converts this confirmed predictive core into useful
warnings under a frozen false-alarm budget.
