# Contextual metrics + dependency logs JEPA v2 development

Status: implemented development protocol. This specification does not
authorize promotion or publication.

## Question

Can bounded application and dependency events add schedule-transfer
information beyond a separately trained metrics-only JEPA?

The metrics-only and metrics-plus-logs models remain different model
artifacts. No encoder or calibration state is shared between them.

## Evidence boundary

The v1 promotion corpus and its ten schedule families have been inspected.
They are development evidence only. V2 therefore collects fresh executions
to exercise the new instrumentation, but the schedules remain exposed and
cannot become promotion evidence.

The first eight schedule families form the development pool. Candidate
selection uses leave-one-schedule-family-out folds inside that pool. The
last two families are diagnostic only and cannot affect selection.

After a candidate is selected, its complete recipe must be frozen before an
entirely new corpus with unexposed schedule families is collected.

## Fresh development corpus

- 10 schedule families
- worker topologies of 1, 2, and 3 replicas
- 30 fresh normal runs
- 340 points per run at 100 ms per point
- application request queue size fixed at 128
- deterministic case IDs with seed label 89
- immutable image digests and recorded application build-context hash

## Bounded dependency telemetry

The lab observes the Redis and PostgreSQL calls that the application
actually performs. Its finite dependency-event boundary supports:

- elevated or slow dependency latency;
- operation failure;
- an observed retry;
- elevated or slow client-pool wait; and
- elevated or slow checkout queue residence.

The current lab records dependency latency and failures plus checkout queue
residence. Retry and pool-wait events are reserved for clients that expose
those observations; synthetic retry or pool events are not generated.
Routine fast dependency success is not emitted. Event attributes contain
only the registered dependency name and logical origin window. Redis keys,
checkout payloads, SQL text, exception strings, and arbitrary debug bodies
are prohibited from the model vocabulary.

Queue residence begins at a Redis `TIME` value written into the payload by
the same Lua script that enqueues it. Dependency and queue-pressure records
retain their operation-completion timestamps even when emitted in a later
batch, so window assignment follows event time rather than batch time.

Redis latency is elevated at 500 µs and slow at 2 ms. PostgreSQL latency is
elevated at 2 ms and slow at 10 ms. Checkout queue residence is elevated at
10 ms and slow at 50 ms. These fixed lab thresholds are part of the
development protocol, not production SLO recommendations.

Redis documents client observability as metrics and traces; the lab
normalizes the corresponding client-operation semantics into its bounded
OTLP log vocabulary rather than treating raw SDK output as model-ready
events. See the
[Redis observability guide](https://redis.io/docs/latest/develop/clients/redis-py/observability/)
and
[OpenTelemetry redis-py instrumentation](https://opentelemetry-python-contrib.readthedocs.io/en/latest/instrumentation/redis/redis.html).

## V2 semantic log vector

The compiler removes routine complements such as low versus high, fast
versus slow, and worker busy versus idle. It produces:

1. checkout completion ratio;
2. checkout backlog delta per request;
3. checkout rejection rate;
4. queue pressure-transition rate;
5. queue-high transition rate;
6. PostgreSQL latency-pressure ratio;
7. PostgreSQL slow-or-error ratio;
8. worker activation rate;
9. Redis latency-pressure rate;
10. Redis slow-or-error rate;
11. checkout queue-wait pressure ratio; and
12. checkout queue-wait slow ratio.

This is intended to reduce the repetitive event skew and complementary
channels found in v1. The log representation is normalized using
development-training values only.

## Fixed JEPA candidate sequence

The candidate order is fixed before scoring:

1. log latent dimension 1;
2. log latent dimension 2;
3. log latent dimension 3;
4. log latent dimension 2 with modality masking and balanced objectives;
5. log latent dimension 3 with modality masking and balanced objectives.

Balanced candidates use deterministic 15% single-modality context masking,
multiply log-to-log self-prediction by 0.25, and multiply each cross-modal
objective by 1.5. Metrics and logs retain separate stems and target
encoders. These choices follow the predictive-embedding principles in
[I-JEPA](https://arxiv.org/abs/2301.08243),
[V-JEPA](https://arxiv.org/abs/2404.08471), and
[V-JEPA 2](https://arxiv.org/abs/2506.09985), adapted here as an explicitly
small-data experiment.

## Family-held-out controls

Every candidate fold trains and evaluates:

- contextual metrics plus logs;
- metrics-only;
- capacity-matched metrics-only;
- shuffled log alignment;
- log-only; and
- metric-context-only and log-context-only scoring ablations.

Every fold refits metric, log, and control normalization using only that
fold's training families.

## Selection

A candidate is eligible only when it:

- completes family-held-out evaluation;
- has a lower mean normal alert rate than metrics-only;
- is no worse than capacity-matched metrics-only;
- has a lower mean normal alert rate than shuffled logs;
- is no worse than metrics-only on at least half of folds; and
- retains at least half of the requested effective rank in each active
  latent.

Among eligible candidates, selection maximizes the minimum improvement
against metrics-only, capacity-matched, and shuffled-log controls. Exposed
diagnostic families are excluded from this calculation.

## Reproducible execution

Run:

```bash
./lab/fault_matrix/run-contextual-multimodal-jepa-v2-development.sh
```

The command refuses a dirty worktree or an output overwrite. It records the
commit, build context, input manifests, captures, candidate artifacts,
leaderboard, selected configuration, selected model when eligible, and an
evidence-boundary report.
