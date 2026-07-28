# Action-conditioned graph dynamics development v1

## Status and evidence boundary

This protocol is preregistered after the v4 instrumentation smoke and pilot
qualified and before any development capture is generated.

The v4 smoke and 30-pair pilot are authorization evidence only. Their captures
are forbidden from preprocessing fit, model fit, model selection, threshold
selection, attribution scoring, or later confirmation.

This is open development. A positive result may select a frozen model and
claim for a new sealed confirmation experiment. It is not itself confirmation
evidence and does not establish a production or general-purpose world model.

## Development question

On the fixed Quantis checkout stack and the five randomized reversible actions,
does an action-conditioned temporal graph state-space model:

1. predict intervention and recovery trajectories better than models that do
   not know the action;
2. add value beyond a dense action-conditioned model and persistence; and
3. rank the injected action family and target from held-out trajectories while
   recognizing matched no-action controls?

The first question tests action-conditioned dynamics. The second tests whether
the declared topology adds information. The third tests bounded attribution
within the finite intervention library.

## Fresh corpus

Generate 120 matched pairs, or 240 isolated captures, from
`lab/action_dynamics/development-protocol-v1.json`.

The design is the complete product:

`5 action kinds × 3 worker topologies × 8 replicates`.

Within each action-by-topology cell, replicates zero through five are training
and replicates six and seven are development-validation. Both twins in a pair
always share a split. This yields exactly:

- 90 training pairs and 180 training captures;
- 30 development-validation pairs and 60 validation captures; and
- six training plus two validation pairs in every action-by-topology cell.

The generator seed, workload seeds, intervention seeds, opaque case IDs, and
opaque pair IDs are fresh and disjoint from v3 and v4. Collection uses six
concurrent isolated Compose lanes in 20 pair-atomic batches. Every first twin
in a batch finishes teardown before any second twin starts. Automatic retries,
case deletion, window deletion, and post-hoc reassignment are forbidden.

All instrumentation gates from v4 remain unchanged. Development preparation
must bind the exact qualifying v4 smoke protocol and plan, pilot protocol and
plan, pilot-bound smoke qualification, and application build-context hashes
listed in the development protocol's `authorization_identity`. A merely
similar or newly rerun pilot cannot authorize this corpus. If any of the 120
pairs fails identity, count, trace, effect, recovery, cleanup, schedule,
isolation, or placebo checks, the corpus does not qualify and model fitting
does not start.

## Graph state

The declared entity order is:

1. `api`;
2. `api_enqueues_queue`;
3. `checkout_queue`;
4. `queue_dequeues_to_worker`;
5. `worker_pool`;
6. `worker_writes_postgresql`; and
7. `postgresql`.

The directed path follows that order. The 27 operational metrics are assigned
to the entity that owns their mechanism. In particular, Redis enqueue and
dequeue observations live on their dependency edges, and all PostgreSQL write
rate, latency, error, event-age, and busy-age observations live on
`worker_writes_postgresql`. The terminal PostgreSQL node is retained as a
zero-padded structural endpoint in v1.

Structured logs and spans are not embedded as raw natural language. They are
aggregated by logical window and graph entity into:

- log event count;
- log error count;
- trace span count; and
- trace error count.

This uses stable event and trace structure while avoiding vocabulary leakage
from lab truth. Action IDs, kinds, targets, phases, magnitudes, seeds, matched
pair IDs, and fault fields remain forbidden from observations.

Exact request demand and worker replicas are exogenous controls. Action truth
is supplied only through the action tensor for action-conditioned models and
is never included in observed state or controls.

## Compilation

The compiler uses 20 observed states and a 10-transition rollout horizon.
With 108 states per capture, this produces 79 windows per capture:

- 14,220 training windows from 180 training captures; and
- 4,740 development-validation windows from 60 validation captures.

Every transform, normalization statistic, variance floor, and action scale is
fit from training captures only. Compilation cannot cross a run or pair
boundary. Report both all-window metrics and action-overlap metrics so nominal
windows cannot dominate the intervention result.

## Frozen first model matrix

The first pass is deliberately small and high probability:

1. graph-constrained linear VARX with actions;
2. the same graph VARX without actions;
3. a dense all-entity linear VARX with actions; and
4. persistence.

All linear models use ridge `1e-3` and variance floor `1e-4`. No hyperparameter
sweep is permitted on the development-validation split. This pass establishes
whether the action signal and graph factorization help before spending time on
a neural JEPA.

The existing neural residual graph state-space model and a low-weight masked
JEPA auxiliary are subsequent development candidates only if the linear
matrix leaves systematic nonlinear residuals. MLX is not used for the linear
pass because corpus collection, not linear algebra, is the bottleneck.

## Forecast evaluation

Fit only on the training split and score only on development-validation.
Report normalized mean squared error for:

- all forecast states;
- forecast states whose horizon overlaps an active intervention;
- the targeted entity;
- downstream entities; and
- recovery states.

Report results overall and by action kind and worker topology. The primary
dynamics comparison is the action-overlap score, not the all-window average.

Development gates are:

1. graph action-conditioned VARX improves at least 10% relative to graph
   action-agnostic VARX on action-overlap normalized MSE;
2. it improves at least 10% relative to persistence on the same score; and
3. declared-graph VARX improves at least 5% relative to dense
   action-conditioned VARX on downstream-entity intervention-effect error.

Failure of gate three blocks a graph/topology claim but does not erase a
possible action-conditioned forecasting result.

## Attribution evaluation

For each validation pair, construct one treatment query and one matched control
query with history ending immediately before the scheduled action onset.
Manifest action truth is hidden from the ranker.

The frozen candidate library contains:

- one no-action candidate; and
- every allowed action kind and fixed target, crossed with its declared
  severity and duration values, beginning at the next transition.

Candidate trajectories are ranked by normalized predictive likelihood over
the observed 10-transition future. Action-and-target hit@1 collapses severity
and duration variants to their action family and fixed graph target. Also
report exact-variant hit@1, action-family hit@3, likelihood margin, and results
by topology.

Development gates are:

1. treatment action-and-target hit@1 is at least 70%; and
2. no-action specificity on matched control queries is at least 90%.

These are closed-library randomized-action results. They do not attribute an
arbitrary production incident or infer an unseen root cause.

## Decisions and publishable bounds

If action forecasting and attribution gates pass but the graph gate fails, the
publishable development finding is narrowly:

> In this fixed lab and finite randomized action library, explicit action
> conditioning improves held-out transition prediction and bounded action
> ranking, while the declared topology does not add measurable value.

If all gates pass, freeze the compiler, feature ownership, candidate library,
model class, parameters, and thresholds for a fresh 60-pair confirmation. Only
a qualifying confirmation can support the phrase **constrained
action-conditioned world model for the Quantis lab stack**.

If action conditioning does not beat the action-agnostic and persistence
baselines, publish the bounded negative result and do not add JEPA capacity.
The likely limitation would be insufficient observability, excitation, or
identifiability rather than model width.
