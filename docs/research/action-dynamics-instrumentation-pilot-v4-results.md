# Action-dynamics instrumentation pilot v4 results

## Decision

Freeze the v4 development generator and authorize a separately
preregistered 120-pair development corpus.

Both the six-pair smoke and the subsequent 30-pair instrumentation pilot
qualified under their frozen protocols. There were no retries, missing
captures, failed pairs, or post-run changes.

The pilot captures remain forbidden from model fitting or model selection.
This result qualifies the apparatus; it is not a trained-model result.

## Execution

The protocols and implementation were frozen in commit `05e64ad` before
either v4 artifact directory existed. The smoke used six parallel first
twins followed by six parallel second twins. The pilot used the same
two-wave barrier for each of five six-pair batches.

- Smoke: 12/12 captures, 6/6 pairs, 82.6 seconds.
- Pilot: 60/60 captures, 30/30 pairs, 407.3 seconds.
- Automatic retries: zero.
- Missing captures: zero.

The pilot was the complete factorial:

`5 action kinds × 3 worker topologies × 2 replicates = 30 matched pairs`

Pilot identities:

- protocol SHA-256:
  `eba673eccfd7ff69c3b93cdafd73075421e7d55ff4a7e5ee0e607a0d0dcc2cea`;
- execution-plan SHA-256:
  `a9be10d4fc6633ace04aafbd2d533292ddf34b1693606951a5cd6e96c69acac6`;
- qualifying-smoke SHA-256:
  `d8a3281432fe2c930812ffc3abf33fffbebc8eb31fc22f59a700fe45a62853b2`;
- application build-context SHA-256:
  `0512db6c63db20c8ff45db0b1fbd382b176b9903d47e324a6269f18dcefe2e7e`.

## Smoke result

All 21 smoke gates passed:

- every final plan, build, manifest, schedule, count/rate, action-command,
  controller-readback, truth-exclusion, lane, and twin-wave identity gate;
- 21,539/21,539 eligible application events were trace linked;
- 10,735/10,739 admitted, completed checkouts had the exact six-span path
  (99.963%);
- all six primary effects and recoveries passed;
- both enqueue pairs passed direct Redis mechanistic recovery, clean drain,
  and restart liveness; and
- the placebo false-positive rate was zero.

The two repaired enqueue pairs had request-latency recovery ratios of
`0.08484` and `0.05628`, both well below the unchanged `0.30` maximum.
Their direct Redis enqueue-latency recovery ratios were `0.03366` and
`0.03111`.

## Pilot result

All 22 pilot gates and all 30 pair-level decisions passed:

- each action appeared in six pairs and each topology in ten;
- 111,238/111,238 eligible application events were trace linked;
- 55,244/55,256 admitted, completed checkouts had the exact six-span path
  (99.978%);
- there were zero cross-case trace references;
- exact API count evidence contained 240 active requests in every twin;
- all controller cleanup, count/rate consistency, schedule, drain, restart,
  mechanistic recovery, wave-barrier, and isolation checks passed; and
- the placebo false-positive rate was zero.

The weakest signed primary effect in each family remained above its frozen
minimum:

- API rejection: `0.21667` versus `0.15`;
- PostgreSQL lock: `28.0` writes/s versus `1.0`;
- Redis dequeue delay: `20.702 ms` versus `10.0`;
- Redis enqueue delay: `18.361 ms` versus `10.0`; and
- worker pause: `28.0` workers/s versus `1.0`.

Maximum observed recovery ratios were `0.08167` for enqueue delay,
`0.01326` for dequeue delay, and zero for the other action families. The
maximum direct enqueue mechanistic recovery ratio was `0.05962`.

## Integrity and limitations

The smoke artifact manifest contains 91 file hashes; the pilot manifest
contains 433. The raw local evidence occupies approximately 465 MiB under
the ignored `artifacts/action-dynamics` tree. It has not yet been published
to an immutable external archive, so a fresh clone cannot independently
recompute the run.

The full repository suite passed with 236 tests, strict typing passed for 52
source files, and the pinned OpenTelemetry Collector round trip passed.

The retained bounded risk is that a fast same-wave lane may begin teardown
while a slower lane is near the legacy final-eight recovery window for a
non-enqueue action. It did not produce a pair failure. Enqueue recovery is
scored earlier under continuous workload and is not exposed to this overlap.

## Permitted claim

This result supports only:

> On the fixed Quantis checkout lab, a content-addressed six-lane harness
> completed 30 randomized matched treatment/control pairs spanning five
> reversible action/location families and three worker topologies with exact
> count and controller evidence, continuous-workload enqueue recovery,
> trace-linked graph paths, zero attrition, zero placebo false positives, and
> all preregistered pair and aggregate gates passing.

No model was trained. This is not evidence of forecasting, attribution,
topology generalization, production transfer, or a software world model.

## Authorized next step

Preregister 120 new matched pairs, with whole pairs split before any window
materialization into 90 training pairs and 30 development-validation pairs.
The new corpus must use fresh seeds and identities and must preserve all v4
collection and evidence gates.

Train the action-conditioned temporal graph state-space model only on the 90
training pairs. Evaluate it once on the 30 frozen development-validation
pairs against persistence, action-agnostic temporal, and non-graph
action-conditioned baselines. The claim must remain bounded to this stack and
the finite randomized action library.
