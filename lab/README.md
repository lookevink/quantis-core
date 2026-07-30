# Lab infrastructure

`lab` contains shared collection, orchestration, and historical experiment
infrastructure. It is not the primary navigation interface for research
results; browse [`experiments`](../experiments) instead.

| Directory | Responsibility |
|---|---|
| [`otel`](otel) | Pinned OpenTelemetry Collector replay and round-trip verification |
| [`fault`](fault) | Real checkout-stack fault injection and capture |
| [`fault_matrix`](fault_matrix) | Schedule, topology, contextual JEPA, and graph-observability collections |
| [`graph_jepa`](graph_jepa) | Observability-rich graph training and confirmation infrastructure |
| [`action_dynamics`](action_dynamics) | Shared action-conditioned corpus tooling and retained historical tracer paths |

Historical action-dynamics runners remain under `lab/action_dynamics` because
their paths are recorded in immutable artifacts. Technique-centered capsule
links live under [`experiments/jepa`](../experiments/jepa).
