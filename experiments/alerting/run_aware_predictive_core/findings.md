# Run-aware predictive-core alert confirmation findings

Status: **confirmed**

## Decision

**Confirm that the previously confirmed rank-32 predictive core yields useful
run-aware warnings under the frozen 5% control-run false-alarm budget on the
fixed Quantis checkout lab and declared intervention library.**

The independently recomputed decision is
`confirm_predictive_core_yields_useful_run_aware_warnings`. The candidate
passed all ten frozen decision gates. This is warning-policy evidence, not
production-paging readiness.

## Sealed result

| frozen sealed-evaluation metric | candidate | persistence |
|---|---:|---:|
| control false alarms | 1/60 (1.67%) | 0/60 (0%) |
| treatment detection | 60/60 (100%) | 12/60 (20%) |
| detection while intervention active | 60/60 (100%) | 5/60 (8.33%) |
| pre-onset alerts | 0/60 (0%) | 0/60 (0%) |
| median detection delay | 6 transitions | 19.5 transitions |

Candidate detection was 100% in every frozen action family: API rejection,
Postgres lock, Redis dequeue delay, Redis enqueue delay, and worker pause.
The candidate's false-alarm count rejects a 15% or worse control-run
false-alarm rate with exact lower-tail binomial p-value `0.0006747655`.

The candidate used a CUSUM threshold of `31.3995721091`, calibrated from 30
disjoint control runs after another 30 disjoint controls defined empirical
residual tails. The conclusion uses only the 60 preassigned sealed-evaluation
pairs.

## Evidence identity

- Execution source commit:
  `ab72c5a0ac799b6aee36b1c8a37980c82c877509`
- Conclusion-bearing artifact:
  `artifacts/action-dynamics/run-aware-alert-confirmation-v2-attempt-001`
- Contract SHA-256:
  `9dc8ea1706df37dcc7aa246e48c0edc4a129106c78e6e5cb946973569e45e4c0`
- Confirmed predictive-core model SHA-256:
  `c3456d1314c0d186167c9b63fce608cf65ec923e004c626dfd0343c3fe8b582d`
- Raw collection artifact-manifest SHA-256:
  `01d4f8ae4244553839d53b8f4291eba73be8a8a1341e404228220c22f791c3ba`
- Qualified data-quality SHA-256:
  `e6c267397d63bad05b1444a1828c5e904142e08f3fd193ec3adeb6794315207c`
- Prediction-manifest SHA-256:
  `2040d9a8b1aaf8688a30630a386d3db841c96821d6f898cb95632b40809c8720`
- Independent confirmation-assessment SHA-256:
  `70ba572cc7bd4e892bd04e809e6329c2e09ccb2422739d99ea1b85826d1859a6`
- Confirmation artifact-manifest SHA-256:
  `e66208ded3cf2b772e0a370ca2a565d193822a9a2aa280538c3ad10b6a90a90b`

Collection completed all 240 planned captures with zero missing captures and
zero automatic retries. All 24 raw qualification gates passed. A standalone
assessor then reloaded the raw corpus, verified every source manifest entry,
reconstructed compiler and model outputs, matched the stored prediction
arrays, and independently returned `confirmed` with the same metrics.

## Invalid predecessor

V1 stopped after 30 captures when a control case received a transient API
`RemoteDisconnected` immediately after transition 0. It was never qualified,
assigned analytic roles, calibrated, or scored. The exact failed case passed
an isolated replay and 12/12 six-lane stress replays. V2 retained the policy
and every scientific gate, used a fresh seed, and lowered host concurrency
from six to four lanes.

## Claim boundary

This result does not establish production paging, cross-stack or production
transfer, unknown-intervention detection, root-cause attribution, adaptive
calibration under drift, or more than one alert per run. It establishes the
frozen claim only on the declared lab stack and five intervention families.
