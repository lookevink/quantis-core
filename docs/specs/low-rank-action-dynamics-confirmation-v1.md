# Low-rank action-dynamics sealed confirmation v1

## Decision question

Does the already frozen rank-32 contractive low-rank predictive core use
declared intervention inputs to predict observable telemetry dynamics on new
cases, rather than merely extrapolating state and workload?

This is the missing scientific step after the open edge-development
tournament. The bounded graph residual did not improve the global model, so
this confirmation does not retest it. The candidate is a graph-owned
state-space predictor over the declared entity and edge telemetry schema; the
experiment does not claim that message passing or graph topology adds value.

## Frozen candidate

The candidate is restored byte-for-byte from
`artifacts/action-dynamics/edge-development-v1-reviewed2/models/contractive_low_rank.json`.
Its SHA-256 is
`c3456d1314c0d186167c9b63fce608cf65ec923e004c626dfd0343c3fe8b582d`.
It has rank 32, 34,503 parameters, and a fitted spectral radius of about
0.8714 under a maximum of 0.98. The normalization and action compiler are
restored from the content-addressed development preprocessing cache. No
parameter is fitted, selected, or calibrated from confirmation cases.

The machine-readable contract is
`lab/action_dynamics/low-rank-confirmation-contract-v1.json`. It binds the
candidate, source development artifact, compiler, collection generator, core
decision code, runner, and independent stored-array assessor by SHA-256. At
preparation, the clean Git source commit is inserted into the materialized
collection protocol and therefore into every capture manifest. Scoring
requires that exact clean commit, transitively freezing imported collection,
compiler, model, and qualification code.

## Fresh evidence

The campaign contains 120 new matched pairs and 240 captures:

- five declared intervention/location families;
- one-, two-, and three-worker topologies;
- eight replicates per action-by-topology cell;
- one treatment and one schedule-identical control per pair;
- 108 points per capture at 250 ms;
- six pair-atomic collection lanes;
- no automatic or pair retry.

Each action family contributes exactly 24 matched pairs.

The existing development collector is reused as a transport. Its `training`
and `validation` labels are compatibility fields only: all 120 pairs have one
sealed confirmation role, and none may be used for fitting or threshold
choice. The generator seed is new, so pair and case identities are disjoint
from the development campaign.

Collection must qualify every existing instrumentation, identity, count,
effect, recovery, trace, schedule, and isolation gate before model scoring.
A failed collection gate blocks scoring and confirmation.

## Frozen controls and metrics

Three predictions are evaluated on identical normalized windows:

1. **Candidate** — the frozen model with the true declared future action
   tensor.
2. **Neutral-action ablation** — the same frozen weights, state, controls, and
   capacity, with every future action replaced by the canonical no-action
   vector.
3. **Persistence** — the last observed normalized state repeated across the
   ten-step horizon.

The primary statistic is matched-pair-balanced normalized MSE on future
positions overlapping an active action. Secondary evidence is
treatment-minus-control downstream-effect MSE on graph-reachable entities.
The one-sided Monte Carlo sign-flip tests use the matched pair as the only
resampling unit, seed `26073042`, and 99,999 draws.

## Conjunctive decision

Confirmation requires every gate:

- candidate aggregate action-overlap MSE is at most 75% of both controls;
- candidate action-overlap MSE is at most 90% of both controls in every one
  of the five action families;
- candidate downstream-effect MSE is at most 80% of both controls;
- both pair-blocked sign-flip p-values are at most 0.05;
- all rollouts are finite;
- spectral radius is at most 0.98;
- parameter count is at most 40,000;
- canonical serialized model size is at most 1 MiB.

No gate or margin may change after collection begins. Passing yields
`confirm_learnable_action_dynamics`; any failed gate yields
`do_not_confirm_learnable_action_dynamics`.

## Claim boundary

Passing supports only this statement:

> On the fixed Quantis checkout lab and declared randomized intervention
> library, the frozen compact predictor extracts action-conditioned dynamics
> that improve ten-step observable-state and downstream-effect forecasts on a
> fresh sealed campaign.

It does not establish graph-topology benefit, unknown-incident causality,
cross-stack transfer, operational alert quality, production readiness, or a
software world model.
