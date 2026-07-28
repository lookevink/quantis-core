# Action-dynamics instrumentation pilot v4

## Status and purpose

This protocol prospectively repairs the failed v3 smoke. It does not rescore,
rewrite, or remove the v3 failure recorded in
`docs/research/action-dynamics-lab-smoke-v3-results.md`.

V4 remains an instrumentation-qualification experiment. Its smoke and pilot
captures are forbidden from model fitting, model selection, forecasting,
attribution, graph-benefit, transfer, and world-model claims.

## Motivating negative result

V3 completed all 12 smoke captures and passed every identity, count,
schedule, action, trace, isolation, primary-effect, drain, and placebo gate.
One low-severity, three-worker enqueue-delay pair failed the final-eight-state
recovery statistic.

The action had already returned to baseline during uninterrupted workload:
after the stop transition, request and direct Redis enqueue latency recovered,
the controller closed cleanly, and both twins later reached four fully idle
drain states. The failed residual appeared only after the forced idle/restart
probe. The two twins were sequential, not contemporaneous, and each recovery
state contained only eight requests. V3 therefore mixed intervention
recovery with cold-restart and host-lifecycle variation.

V3 remains failed under its frozen estimator. The exploratory diagnosis does
not authorize a v3 pass.

## Frozen v4 changes

All first twins in a batch run concurrently and finish teardown before any
second twin starts. All second twins then run concurrently. Treatment order
remains counterbalanced, projects remain fresh and isolated, and automatic
retry remains forbidden. The attestation must prove, for every batch, that
the latest first-wave completion precedes the earliest second-wave start.

For `redis_enqueue_delay`, intervention recovery is measured under continuous
workload:

- the first two states after the stop transition are washout;
- the next exactly 16 states are the recovery window;
- the window must end before the zero-request drain begins;
- the primary recovery statistic remains the median absolute paired
  treatment/control delta in `request_latency_ms`, divided by the larger of
  the absolute active effect or the frozen `1 ms` floor; and
- the maximum ratio remains `0.30`.

A second mechanistic gate applies the same active and recovery windows to
`redis_enqueue_latency_ms`. The active direct effect must be at least `10 ms`
and its recovery ratio must be at most `0.30`.

The later zero-request drain and eight-request restart probe remain frozen.
Both twins must show four clean drain states, exact scheduled counts,
count/rate consistency, finite nonnegative latency, and positive restart
traffic. Restart latency is diagnostic only; it is not evidence of lingering
intervention state.

The action truth stream records the Redis enqueue-delay controller key
immediately after both start and stop commands and at run close. Treatment
start must read back the commanded magnitude; stop and close must read back
zero. These fields remain excluded from model observations.

V4 intentionally does not redesign the recovery estimand for the other four
actions, because v3 showed no corresponding failure. Those actions retain
their frozen final-eight-state recovery window. A faster lane can therefore
begin teardown while a slower same-wave lane is still near its final states;
the inter-wave barrier prevents new twin startup but not that bounded
same-wave lifecycle risk. Any resulting pair failure remains in the
denominator and stops promotion; it cannot be retried away.

## Unchanged design

V4 preserves:

- the six smoke pairs, including two opposite-order low-severity
  three-worker enqueue-delay pairs;
- the complete 30-pair
  `5 actions × 3 topologies × 2 replicates` pilot;
- 108 states at 250 milliseconds;
- all action magnitudes, onset and duration ranges, effect thresholds,
  recovery thresholds, exact API-count estimand, trace gates, placebo gate,
  truth exclusion, content-addressed identity, lane isolation, and zero-retry
  policy; and
- the rule that a smoke failure prevents pilot preparation or collection.

V4 uses new generator seeds, opaque IDs, output directories, Compose
prefixes, plans, manifests, attestations, and artifact hashes. A qualifying
v4 pilot may authorize only a separately preregistered 120-pair development
corpus. Pilot captures themselves remain ineligible for training.

## Decision

Every frozen smoke gate must pass to open the pilot. Every frozen pilot gate
must pass to freeze the development generator.

Any failure yields `stop_and_repair_instrumentation`. No capture may be
silently discarded, retried, or rescored under a replacement estimator.

## Narrow permitted claim

If both stages qualify, the result supports only:

> On the fixed Quantis checkout lab, the content-addressed six-lane harness
> can execute the five predeclared reversible intervention families across
> randomized matched treatment/control pairs with exact action/count
> evidence, continuous-workload recovery, controller cleanup, trace-linked
> graph paths, and isolated capture.

It would not establish learned dynamics, attribution, topology
generalization, production transfer, or a software world model.
