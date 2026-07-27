# Quantis expanded confirmation matrix

## Status

Accepted for implementation on 2026-07-26 after demand-conditioned v2 passed
one preregistered three-case confirmation with false-positive rates below, but
close to, the frozen 20% limits.

## Claim under test

Without fitting, threshold changes, or preprocessing changes, the exact v2
model can retain useful detection and attribution performance across a complete
cross-product of:

1. worker crash, database lock, and cache outage; and
2. one-, two-, and three-worker topologies.

Each of the nine cases also uses a distinct realized request schedule and
fault timing. This is controlled evidence about repeated transfer across
workload and worker-count variation. It is not production-scale validation.

## Public seams

`FaultMatrixCaseManifest.from_dict(...)` and `.to_dict()` carry a versioned
`topology_id` and `worker_replicas`. Schema-v1 manifests retain byte-identical
canonical serialization so prior evidence remains reproducible.

`evaluate_demand_conditioned_fault_matrix(...)` reports results for every
topology stratum and rejects an incomplete fault × topology cross-product.

`./lab/fault_matrix/run-expanded-v2-confirmation.sh` scales the worker service
to the declared replica count, verifies the runner observes the same count, and
runs every case in a fresh topology.

## Frozen acceptance gates

The expanded matrix must contain all nine fault × topology combinations. Both
the aggregate result and each topology stratum must achieve:

- 100% structural-event recall;
- 100% attribution hit@3;
- maximum detection delay of six logical windows;
- pre-noise alert rate at most 20%; and
- routine-noise response-horizon alert rate at most 20%.

All existing raw-effect, manifest/capture integrity, application-build,
content-addressing, and no-fit gates continue to apply.

## Preregistration

Before capture, commit:

- the exact model and evaluation dependency closure;
- all nine manifests;
- the feature schema and acceptance configuration;
- the runner, topology controls, and protocol verifier; and
- the protocol hashes and required topology IDs.

The final report must recover every frozen byte from that commit. New case IDs,
canonical schedules, and fault-kind/timing pairs must be disjoint from training
and prior result-bearing confirmation cases.

## Required limitations

- Worker replica count is only one dimension of topology diversity.
- Redis, PostgreSQL, API, Collector, host, and telemetry vocabulary remain
  unchanged.
- Nine controlled local cases do not estimate production incident prevalence.
- Completion-ratio features encode a domain assumption about admitted work.
- Attribution remains associative rather than causal.
- The target encoder remains linear PCA rather than a learned JEPA encoder.
