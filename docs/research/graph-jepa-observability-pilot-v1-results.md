# Graph JEPA observability pilot v1 results

## Outcome

The development tracer supports continuing to a fresh, observability-rich
graph corpus. It does not support a world-model, causal-attribution, or
publication-confirmation claim.

The pilot reused the inspected 72-run contextual confirmation corpus. Its
purpose was to validate graph ownership, test raw state observability, and
determine whether compact entity tokens could preserve future state before
collecting another corpus.

## Raw graph-state preflight

The declared graph assigned all 18 semantic metric and structured-log
features to five nodes and four directed relationships. An unobserved
`queue_hosted_on_redis` relationship remained present but unscored.

Held-out normalized MSE:

- actual training-mean prediction: `1.824916`;
- last-observation persistence: `3.351490`;
- entity-local raw ridge: `1.373599`;
- one-hop graph ridge: `1.338653`; and
- flat raw-context ridge: `1.327942`.

One-hop context retained flat raw performance within `1.0081x` and improved
on entity-local context. Families 15, 16, and 19 remained the dominant
failures. The result shows usable state signal and a small topology benefit;
it does not show that the existing observations are sufficient for dependable
rollout or attribution.

This comparison also clarifies the earlier confirmation interpretation. The
preregistered normalized-MSE threshold of `1.0` was a fixed transfer gate, not
the measured held-out error of the training-mean predictor. On this corpus the
actual training-mean error is materially above `1.0`, and raw predictors beat
it while still failing the stricter fixed threshold.

## Uniform entity width

A linear graph-JEPA tracer fitted training-only PCA target encoders over
two-point entity blocks. It predicted future tokens from three context
patches, exogenous controls, horizon, and either entity-local, one-hop, or
all-entity context.

Uniform width two compressed the 108-value raw context to 48 active values,
but target PCA reconstruction error was `0.581035`. The width sweep found:

| Width | Context ratio | PCA reconstruction error | One-hop error |
|---:|---:|---:|---:|
| 1 | `4.500:1` | `1.0354` | `1.4188` |
| 2 | `2.250:1` | `0.5810` | `1.3699` |
| 3 | `1.500:1` | `0.3203` | `1.3617` |
| 4 | `1.125:1` | `0.1369` | `1.3557` |
| 6 | `0.750:1` | `0.0027` | `1.3474` |
| 8 | `0.562:1` | approximately `0` | `1.3500` |

No uniform width simultaneously met the 90% state-retention gate and
compressed the context. Width six was the smallest fidelity-passing uniform
candidate, but it expanded rather than compressed the state.

## Adaptive entity widths

The next development tracer chose the smallest width passing the PCA
reconstruction gate independently for each entity, using training families
only:

- API: `2`;
- API-to-queue enqueue edge: `2`;
- checkout queue: `4`;
- PostgreSQL: `2`;
- queue-to-worker dequeue edge: `3`;
- Redis: `3`;
- worker pool: `6`; and
- worker-to-PostgreSQL edge: `4`.

This produced 26 active values per patch and 78 values across the three
context patches, a `1.3846:1` compression ratio from the raw 108-value
context.

Held-out results:

- target-token PCA reconstruction error: `0.042266`;
- entity-local decoded prediction error: `1.369770`;
- one-hop decoded prediction error: `1.346569`; and
- all-entity decoded prediction error: `1.339958`.

The adaptive model passed all frozen development gates. One-hop topology
improved over entity-local prediction by about 1.7% and remained within 0.5%
of the all-entity predictor.

## Interpretation

The result is evidence for a useful architectural hypothesis:

1. node and edge ownership can preserve almost all measured target state
   while compressing the context;
2. subsystem complexity differs enough that one global latent width is
   inefficient; and
3. declared one-hop context adds a small amount of predictive information.

It is not yet strong evidence that message passing is necessary. The
all-entity predictor remains slightly better, and the hard schedule families
still dominate aggregate error. The linear PCA encoders are an inspectable
joint-embedding tracer, not the final learned EMA graph-JEPA.

## Next experiment

Collect a fresh corpus with explicit operational state:

- request concurrency and accumulated API busy time;
- queue oldest age and residence-time summaries;
- worker busy duration and active-worker count;
- exact Redis and PostgreSQL operation-latency summaries; and
- event age and ordering.

The new corpus should freeze the adaptive width profile as a hypothesis,
cache compiled graph tensors, and compare raw, PCA, linear graph-JEPA, and a
learned graph-JEPA without selecting widths on validation data.

Only after that representation confirms should the lab collect paired
disturbance/action/recovery episodes and evaluate action-conditioned
rollouts, intervention ranking, and recovery prediction.
