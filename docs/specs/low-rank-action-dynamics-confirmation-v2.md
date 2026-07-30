# Low-rank action-dynamics sealed confirmation v2

This protocol retains every candidate, control, metric, margin, resampling,
edge-envelope, and claim choice from
[v1](low-rank-action-dynamics-confirmation-v1.md). It changes only the fresh
collection generator and execution boundary.

The v1 attempt was interrupted after 36 of 240 capture directories had been
materialized because the repository test suite was running concurrently on
the host. That known host workload could perturb latency-derived telemetry.
The partial attempt is therefore invalid, unqualified, unscored, and excluded
from every analysis.

V2 uses generator seed `26073051`, new opaque pair/case identities, and the
machine-readable
`lab/action_dynamics/low-rank-confirmation-contract-v2.json`. Collection runs
with the host dedicated to the campaign. As in v1, prepare, collect, qualify,
and score must all run from the exact clean Git commit inserted into the
materialized protocol and capture manifests.
