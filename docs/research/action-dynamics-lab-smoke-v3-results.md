# Action-dynamics lab smoke v3 results

## Decision

Do not open the 30-pair instrumentation pilot and do not begin model
training. The preregistered decision is
`stop_and_repair_instrumentation`.

The six-pair v3 smoke failed one recovery gate. The runner stopped before
creating pilot inputs or captures. This result is retained without rerunning
or rescoring any capture.

## Execution

The smoke ran from preregistered implementation commits `7dcba03` and
`705922f`. The final commit closed the independently reviewed evidence
bindings before the v3 artifact directory existed.

Six isolated Compose lanes produced 12/12 fresh captures and 6/6 matched
treatment/control pairs in 82.0 seconds. There were no retries, missing
captures, or attrition.

Frozen identities:

- protocol SHA-256:
  `8a8dcf1366d237be75ddde57cb04720b3acd84270285c956a65300870d220b89`;
- execution-plan SHA-256:
  `c38d9ebd716299023518152d722cc31e7927b25e656009b98d01ec4e38c588c0`;
- application build-context SHA-256:
  `d9a9fd78d3599dac1b627cd363ee7284e97cfebc5b23be71cde3b23013e34493`;
  and
- application image ID:
  `sha256:d0f1c5b82c3739f1c8ac0de7bf3f58ed17fa69f19499e2e603b37f18fc66b0d7`.

## Result

Sixteen of 17 aggregate gates passed:

- all 12 captures, six pairs, and six isolated lanes were present;
- final plan, build-context, manifest, and observation identities matched;
- exact request and error counts were complete, integral, schedule-bound,
  and reconciled with their derived rates at every state;
- API rejection had exactly 240 active requests in each twin;
- both enqueue-delay pairs met the zero-work drain eligibility condition;
- 21,527/21,527 eligible application events were trace linked;
- 10,728/10,731 admitted, completed checkouts had the exact parent-linked
  six-span path (99.972%);
- all six interventions passed their frozen primary-effect threshold;
- the placebo false-positive rate was zero; and
- action command, cleanup, truth-exclusion, schedule, cross-case isolation,
  and lane-isolation checks passed.

One `redis_enqueue_delay`/three-worker pair failed recovery. It produced a
clear `18.9442 ms` active request-latency effect, but the median absolute
treatment/control difference over the final eight probe states was
`9.011 ms`, yielding a recovery ratio of `0.475668` against the frozen
`0.30` maximum. The other worst-case enqueue-delay pair passed with a
recovery ratio of `0.137841`.

## Interpretation

The failure does not look like an action that remained enabled or queued work
left behind. The stop and cleanup evidence passed; the last four drain states
in both twins had zero queue depth, zero in-flight work, and zero busy
workers; and the final observed enqueue-latency difference was approximately
zero.

The failed statistic was dominated by high treatment/control latency
differences during the early part of the short post-drain probe. Each
250-millisecond state contained only eight requests, so the final-eight-state
median remained sensitive to cold-start and run-level latency variation even
though the directly observed queue state was clean. This is a diagnosis, not
a post-hoc pass: v3 remains failed under its frozen estimator.

## Integrity and boundary

The artifact manifest contains 91 file hashes. The raw 75 MiB evidence tree
remains in the local ignored artifact directory at
`artifacts/action-dynamics/lab-smoke-v3`; it has not yet been published to an
immutable external archive.

The full repository suite passed with 233 tests, strict typing passed for 52
source files, and the pinned OpenTelemetry Collector round trip accepted the
runtime-format exact count gauges.

No model was trained. This result supports no forecasting, attribution,
topology-generalization, production-transfer, or world-model claim.

## Prospective repair

Any retry requires a new protocol version. The next smoke should preserve the
effect and recovery thresholds, exact-count estimands, worst-case cells, and
all evidence bindings. It should lengthen the post-drain steady probe and
increase recovery observations so the frozen recovery statistic measures a
settled system rather than the first few windows after a zero-work drain.

Only a fully qualifying replacement smoke may open a fresh 30-pair pilot.
Only a fully qualifying pilot may authorize a separately preregistered
development corpus for training.
