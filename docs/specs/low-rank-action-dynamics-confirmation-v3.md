# Low-rank action-dynamics sealed confirmation v3

This protocol retains every candidate, control, metric, margin, resampling,
edge-envelope, and claim choice from
[v1](low-rank-action-dynamics-confirmation-v1.md). It changes only the fresh
collection generator after two invalid, unscored attempts.

- V1 was interrupted after 36 capture directories because a concurrent host
  test process could contaminate telemetry.
- V2 stopped after 24 captures when a normal three-worker control runner
  received `RemoteDisconnected` at transition 23. It produced no complete
  attestation, was not qualified, and was not scored.

V3 uses generator seed `26073061`, new opaque pair/case identities, and
`lab/action_dynamics/low-rank-confirmation-contract-v3.json`. Prepare,
collect, qualify, and score must all use the exact clean source commit bound
into the materialized protocol and every capture manifest. V1 and V2 cases
are forbidden from V3 qualification or analysis.
