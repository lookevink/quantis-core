# Observability-rich graph-JEPA confirmation v1

## Status

This specification defines the fresh confirmation experiment selected by the
graph-JEPA observability tracer. Implementation and a three-topology smoke run
must pass before the protocol is committed and the 72 confirmation cases are
collected.

The previous 72-run contextual corpus is development evidence only. No capture
from it enters this confirmation.

## Narrow claim

Within the fixed Quantis checkout stack, a graph-structured representation
with frozen subsystem-specific widths preserves compact operational state and
improves localized future-state prediction across held-out workload schedules
and one-to-three-worker topologies.

Passing does not establish fault localization, causal propagation,
counterfactual prediction, action selection, or a general-purpose world
model.

## Declared operational graph

Nodes:

- `api`
- `checkout_queue`
- `worker_pool`
- `redis`
- `postgresql`

Directed relationships:

- `api_enqueues_queue`
- `queue_dequeues_to_worker`
- `queue_hosted_on_redis`
- `worker_writes_postgresql`

The graph is declared from the lab configuration. Runtime traces may attest
that an edge was exercised, but natural-language logs cannot create topology.
This is an operational graph, not yet a causal graph.

## Frozen active widths

The inspected development corpus selected these active dimensions using
training-family PCA reconstruction only:

- `api`: 2
- `checkout_queue`: 4
- `worker_pool`: 6
- `redis`: 3
- `postgresql`: 2
- `api_enqueues_queue`: 2
- `queue_dequeues_to_worker`: 3
- `worker_writes_postgresql`: 4

`queue_hosted_on_redis` currently has no directly observed target and therefore
has no active representation width. It remains available to message passing.

No confirmation width may be selected or revised after collection.

## Operational observations

The metric stream adds explicit state that was absent from the prior corpus:

### API

- request latency and error rate;
- current and peak in-flight request concurrency; and
- mean concurrent work derived from accumulated request busy time.

### Checkout queue

- depth and oldest-item age;
- enqueue and dequeue event age; and
- mean completed-item residence time.

### Worker pool

- completion ratio and heartbeat age;
- active-worker ratio and busy-worker ratio;
- maximum current busy age;
- accumulated worker-busy fraction; and
- processing latency.

### Redis and queue edges

- enqueue and dequeue operation latency;
- enqueue and dequeue operation error rate; and
- bounded structured dependency-pressure log events.

### PostgreSQL and write edge

- write-completion ratio;
- write latency and error rate;
- last successful write age; and
- bounded structured lock/latency/error events.

Request demand and declared worker replicas are exogenous controls, not latent
targets.

All values are observable from the application or dependency clients. Fault
labels, future outcomes, request identifiers, payloads, and evaluator labels
are excluded.

## Time design

- sampling period: 100 ms;
- points per run: 340;
- context: 20 points, covering two seconds;
- target blocks: two contiguous points;
- horizons: 1, 5, and 10 points; and
- queue capacity: 128.

The two-second context covers every declared schedule period and retains short
queue memory. Contexts and targets never cross run boundaries.

## Corpus

The corpus contains 24 new schedule families crossed with one, two, and three
workers:

- families 1–12: training;
- families 13–24: untouched validation;
- 72 total cases;
- three isolated Docker lanes; and
- training and validation families interleaved in collection order.

The schedules must be unique and disjoint from every committed JEPA
development or confirmation schedule.

## Cached graph tensors

Raw OTLP captures and manifests remain the source of truth. Compilation writes
one content-addressed cache containing:

- training and validation graph tensors;
- observation masks and entity schemas;
- case identity for every window;
- fitted training-only normalization;
- graph and feature specifications;
- capture, manifest, and protocol hashes; and
- a hash of every serialized cache file.

Loading refuses a changed source hash, graph schema, feature schema, or tensor
shape. Model seeds reuse this cache rather than recompiling 1.2 million log
records.

## Confirmation models and controls

The confirmation trains five fixed seeds plus one deterministic repeat for:

1. adaptive-width one-hop graph JEPA;
2. adaptive-width entity-local JEPA;
3. adaptive-width all-entity JEPA;
4. equal-active-width PCA context;
5. raw one-hop ridge;
6. flat raw ridge;
7. persistence and training-mean references; and
8. shuffled-topology graph JEPA.

The initial implementation may use the inspectable frozen-PCA entity encoder
as a smoke control. Publication confirmation requires a separately serialized
learned EMA target-encoder model or an explicit amendment narrowing the claim
to frozen-PCA joint-embedding prediction before validation is opened.

## Frozen confirmation gates

All gates must pass:

- every required operational target is present and variable in training;
- raw flat and one-hop state beat the actual training-mean reference;
- mean raw one-hop normalized MSE is below `1.0`;
- each critical node group has normalized MSE below `1.0`;
- target representation reconstruction MSE is at most `0.10`;
- active graph context compression is at least `1.25:1`;
- one-hop graph JEPA beats entity-local JEPA;
- one-hop graph JEPA is no worse than all-entity JEPA by more than 5%;
- one-hop graph JEPA beats equal-active-width PCA;
- one-hop graph JEPA beats shuffled topology;
- at least 9 of 12 validation families are no worse than entity-local;
- at least 4 of 5 seeds beat both entity-local and shuffled topology; and
- the repeated primary seed is byte-identical.

The schedule family is the independent statistical unit. Overlapping windows,
horizons, targets, and training seeds are not independent confirmation
replicates.

## Routing

- all gates pass: publish the constrained graph-representation claim and
  collect paired disturbance/action/recovery episodes;
- raw state passes but representation gates fail: improve the learned
  graph-JEPA objective without recollecting validation;
- raw state fails: revise observability using development-only cases and
  collect another untouched corpus.

Only the subsequent action-conditioned experiment can support a constrained
world-model claim.
