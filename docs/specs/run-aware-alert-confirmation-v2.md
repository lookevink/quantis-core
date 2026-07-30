# Run-aware alert confirmation v2

Status: **frozen before collection**

V2 inherits the question, predictive core, pair roles, alert policy, timing
semantics, false-alarm budget, usefulness conjunction, references, and claim
boundary from
[v1](run-aware-alert-confirmation-v1.md) without modification.

## Why v2 exists

The partial v1 campaign is invalid and unscored. Its 30th capture, a control
case in lane 6, completed transition 0 and then received
`http.client.RemoteDisconnected` from the checkout API. The no-retry runner
stopped immediately. No role assignment, prediction, calibration, threshold,
or alert outcome was computed.

The exact failed manifest subsequently completed in one isolated replay and
in 12 of 12 six-lane stress replays. This falsifies a deterministic workload
or case defect and bounds the event as transient collection infrastructure.
V1 remains retained at
`artifacts/action-dynamics/run-aware-alert-confirmation-v1-attempt-001` and
must never be resumed or scored.

## Only frozen changes

- generator seed: `26073080`;
- parallel collection lanes: four instead of six;
- output:
  `artifacts/action-dynamics/run-aware-alert-confirmation-v2-attempt-001`; and
- executable contract:
  `lab/action_dynamics/run-aware-alert-confirmation-contract-v2.json`.

The lower concurrency is a conservative host-contention mitigation. It does
not change any individual case, twin ordering, model input, score, calibration
rule, decision gate, or scientific claim. V2 still has no automatic retry and
no overwrite.
