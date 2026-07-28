# Graph JEPA observability pilot v1

## Status

This is a development tracer bullet following the negative contextual
metrics-plus-logs confirmation. It reuses the preserved 72-run confirmation
corpus only to validate graph compilation and to test whether graph structure
alone repairs the observed state-prediction failure.

The corpus and its validation results have already been inspected. No result
from this pilot is publication-confirmation evidence.

## Question

Before training a graph JEPA, do the captured observations contain enough
state to predict held-out future node and edge observations better than simple
references?

The pilot must stop before representation training when raw graph context does
not pass this gate. A nonlinear representation cannot establish useful
compression when its uncompressed inputs do not expose predictable state.

## Declared graph

The fixed lab topology is declared from the application and Compose
configuration. It is not inferred from log text.

Nodes:

- `api`
- `checkout_queue`
- `worker_pool`
- `redis`
- `postgresql`

Directed relationships:

- `api_enqueues_queue`: `api -> checkout_queue`
- `queue_dequeues_to_worker`: `checkout_queue -> worker_pool`
- `queue_hosted_on_redis`: `checkout_queue -> redis`
- `worker_writes_postgresql`: `worker_pool -> postgresql`

External request demand and worker replica count remain exogenous controls.
The declared operational graph is not called a causal graph. Controlled
interventions are required before assigning causal-propagation semantics.

## Observation ownership

Every semantic metric and structured-log feature is assigned to exactly one
node or edge. Unknown, duplicate, and silently dropped bindings are errors.

The compiler emits:

- run-isolated node and edge context tensors;
- matching future target blocks;
- an observation mask for padded entity slots;
- stable entity and local-feature schemas; and
- the original exogenous controls and horizons.

Unobserved relationships remain in the graph with an empty observation mask.
They may carry messages in a later predictor, but they are not scored as if
ground-truth edge telemetry existed.

## Development controls

Held-out future-state probes compare:

1. training-mean prediction;
2. last-observation persistence;
3. entity-local ridge using only the owning entity's history;
4. one-hop graph ridge using the entity and its incident neighbors; and
5. flat raw-context ridge using every captured feature.

All ridge models are fit only on training schedule families. Scores are
normalized by training-target variance and reported per target entity,
feature, horizon, and validation family.

The independent family identity is retained in the result; overlapping
windows are not treated as independent confirmation samples.

## Gate

State observability is supported only when:

- flat raw context beats the training-mean predictor in aggregate;
- one-hop graph context beats persistence in aggregate; and
- one-hop graph context is no worse than flat raw context by more than 5%.

These are development routing gates, not preregistered publication
thresholds. Failure selects `add_explicit_operational_state`.

The next collection must then add at least:

- in-flight request concurrency and busy-time accumulation;
- queue oldest-age and residence-time distributions;
- worker busy duration and active-worker count;
- exact Redis and PostgreSQL operation latency summaries; and
- event age and ordering features.

## Route to a constrained world model

Only after raw observability passes:

1. train node and edge target encoders plus a message-passing latent
   predictor;
2. require the graph representation to beat an equal-width PCA control;
3. collect paired disturbance/action/recovery episodes; and
4. require action-conditioned rollouts to rank interventions and predict
   recovery on held-out episodes.

Passing the representation stage supports a localized predictive-state claim.
Passing the action stage is required for a constrained world-model claim.

## Linear graph-JEPA tracer after a passing gate

The first representation model is intentionally linear and inspectable:

- each observed entity gets a training-fit PCA encoder over two-point blocks;
- six context points become three latent tokens per entity;
- the future predictor receives the owning entity and its declared one-hop
  neighborhood, plus demand, worker topology, and horizon controls;
- the predictor forecasts the target encoder's future entity token; and
- predicted tokens are decoded only for evaluation against raw node/edge
  targets.

The encoders are fitted only on training families and then frozen. This is a
joint-embedding predictive tracer bullet, not an attempt to reproduce the
scale or nonlinear optimization recipe of I-JEPA, V-JEPA, or Graph-JEPA.
Its purpose is to establish tensor ownership, localized latent prediction,
artifact round trips, and honest controls before adding learned EMA encoders.

The one-hop model is compared with entity-local and all-entity latent
predictors using the same encoders and ridge strength. It advances only when:

- target-token PCA reconstruction preserves at least 90% of the raw target
  variance in aggregate;
- decoded one-hop prediction beats the training-mean raw-target reference;
- one-hop prediction is no worse than the all-entity predictor by more than
  5%; and
- one-hop prediction beats the entity-local predictor.

These remain development gates on already-inspected data. Passing them starts
a new, separately collected observability-rich corpus; it does not promote the
reused corpus into confirmation evidence.

If no uniform entity width passes while compressing the raw context, a
development-only adaptive-width tracer may select the smallest independently
passing PCA width for each observed entity. Its total active latent width,
rather than zero padding used for tensor batching, is the compression
denominator. The adaptive profile advances only if it passes the same
representation gates and its active context ratio remains greater than
`1:1`. This profile is an architecture hypothesis for a fresh corpus, not a
post-hoc positive result.
