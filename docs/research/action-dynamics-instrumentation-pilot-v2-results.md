# Action-dynamics instrumentation pilot v2 results

## Decision

Do not freeze the development generator or begin model training from this
corpus. The preregistered decision is `stop_and_repair_instrumentation`.

The repaired six-pair smoke qualified, but the subsequent 30-pair pilot
failed two pair-level gates. This result is retained without rerunning any
capture.

## Execution

The smoke and pilot ran from preregistered implementation commit `ffd2e02`.
The pilot was a complete factorial:

`5 action kinds × 3 worker topologies × 2 replicates = 30 matched pairs`

It produced 60 fresh captures in five batches across six isolated Compose
lanes. There were no retries, missing captures, or attrition. The smoke took
74.7 seconds and the pilot took 358.6 seconds of attested wall time.

Pilot identities:

- protocol SHA-256:
  `2041dc54df3acdd05a118d133c3934efdafbbf75d9c28cb7a19d1e1bb58555c2`;
- generated-plan SHA-256:
  `b5068368c8df4736ec54d5bcb402582b12dc98b2e9609b106819ff575f9ce347`;
- application build-context SHA-256:
  `9f64839538e51ba1aeece70a7348fd4578d3a4bc7f96bfb9bcd69d71edfdd058`;
  and
- application image ID:
  `sha256:ec89a189a90b64faff8b967307ff7402c560266e0f555013a0ac08e3cf1cb70a`.

## Smoke result

All 13 repaired smoke gates passed:

- 12/12 captures and 6/6 matched pairs were present;
- 15,710/15,710 eligible application events were trace linked;
- 7,840/7,840 admitted, completed checkouts had the exact parent-linked
  six-span path;
- every treatment passed its effect and recovery gates;
- the paired placebo false-positive rate was zero; and
- identity, truth exclusion, metric completeness, schedule alignment,
  cross-case isolation, and lane isolation passed.

This supported opening the pilot. It did not qualify a model or claim.

## Pilot result

Eleven of 13 pilot gates passed. The collection, identity, observation, and
isolation apparatus was strong:

- 60/60 captures and 30/30 matched pairs were present;
- each action appeared in six pairs and each topology in ten;
- 79,212/79,212 eligible application events were trace linked;
- 39,428/39,448 admitted, completed checkouts had the exact parent-linked
  six-span path (99.949%);
- there were zero cross-case trace references;
- the placebo false-positive rate was zero; and
- all implemented action-command, schedule, identity, hash, metric, truth
  exclusion, and lane-isolation checks passed.

Twenty-eight of 30 pairs passed both their primary effect and recovery gates.
Two preregistered pair gates failed:

1. One `api_rejection`/three-worker pair produced an active error-rate
   difference of `0.142857`, just below the fixed `0.15` minimum. Its recovery
   ratio was zero.
2. One `redis_enqueue_delay`/three-worker pair produced a clear `19.3689 ms`
   active latency effect, but its final-eight-window recovery ratio was
   `0.338770`, above the fixed `0.30` maximum.

The first failure caused the aggregate `raw_effects` gate to fail; the second
caused `recovery` to fail. Successful start/stop command evidence was present
for both interventions. The result therefore points to threshold sensitivity
and/or finite-window variability, not a silent command failure, but this pilot
does not identify the exact cause.

## Post-run binding audit

An independent review found two additional preregistration deviations that
the implemented identity gate did not detect:

1. the pilot attestation does not contain the qualifying smoke protocol and
   assessment digests, so the sealed pilot cannot independently prove which
   smoke result authorized it; and
2. each capture manifest stores a deterministic pre-plan identity digest
   (`04fdda…`), not the final canonical pilot plan digest (`b50683…`). The
   collection attestation contains the final digest, but the per-capture
   wrappers do not.

These defects do not change either measured pair failure or the correct stop
decision. They do mean that the implemented `identity_and_hash_binding` pass
overstates compliance with the written specification. Both bindings must be
added prospectively under a new protocol; the sealed artifacts must not be
rewritten.

## Integrity verification

The smoke artifact manifest contains 90 file hashes and the pilot manifest
contains 426. Independent verification found zero mismatches. Recomputing all
gates from the raw captures, frozen inputs, and collection attestations
reproduced both stored assessments exactly.

The full repository suite passed with 228 tests, strict typing passed for 48
source files, and the independent OTLP fixture passed the repository's
round-trip acceptance check.

The 392 MiB smoke-plus-pilot evidence remains in the local ignored artifact
tree. The compact result record is checked in, but the raw corpus has not yet
been published to an immutable external archive. A fresh clone therefore
cannot independently recompute this run until that archive is published.

## Permitted claim

This pilot supports only the following narrow statement:

> On the fixed Quantis checkout lab, a six-lane harness completed 30
> randomized matched treatment/control pairs spanning five reversible
> intervention/location families and three worker topologies with complete
> local capture, perfect eligible-event trace linkage, 99.949% complete
> checkout paths, and zero placebo false positives. Twenty-eight pairs met
> both preregistered effect and recovery criteria; two boundary failures and
> two post-run binding deviations prevented instrumentation qualification.

No model was trained. This is not evidence of forecasting, attribution,
topology generalization, production transfer, or a software world model.

## Next protocol

Any repair must be preregistered as a new protocol and must preserve this
negative result. Before another smoke, the capture envelope must bind the
final generated-plan digest and the pilot envelope must bind the qualifying
smoke protocol and assessment digests. The next smoke should then test two
apparatus changes:

1. increase the information in the API-rejection estimand, such as more
   admitted requests per logical window or a preregistered count-based paired
   effect, so a low-severity effect is not dominated by coarse error-rate
   increments; and
2. define a longer post-stop recovery horizon or an explicit queue-drained
   recovery condition for enqueue delay, so residual queued work is separated
   from intervention cleanup.

Only a fully qualifying replacement pilot should open the development corpus
and baseline action-conditioned graph-dynamics training.
