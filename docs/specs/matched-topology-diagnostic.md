# Quantis matched-topology diagnostic milestone

## Status

Accepted for implementation on 2026-07-27 after the expanded v2 confirmation
associated multi-worker operation with high normal-alert rates while confounding
worker count with request schedule.

## Question under test

When the absolute request schedule and every non-topology case field are held
fixed, does changing the observed worker replica count from one to two or three
reproduce a material increase in the frozen v2 model's pre-noise alert rate?

This is a diagnostic of the previous negative result. It is not a new model
confirmation and it does not authorize fitting, threshold changes, feature
changes, or preprocessing changes.

## Matched blocked design

The matrix contains three blocks, one for each existing fault kind. Every block
contains one-, two-, and three-worker treatments. Within a block, manifests are
identical except for:

- `case_id`;
- `topology_id`; and
- `worker_replicas`.

In particular, the absolute request schedule, phase intervals, fault timing,
routine-noise injection, affected-feature truth, sample period, point count, and
digest-pinned images are held fixed. Schedules differ between fault blocks so
the treatment is repeated against three workload shapes.

Treatment order is deterministically counterbalanced across blocks (1-2-3,
2-3-1, then 3-1-2) so topology is not identical to global run order. The
one-worker treatment is the reference. The primary outcome is the
pre-noise alert rate, measured before either the routine-noise injection or the
structural fault. For each block and non-reference topology, report the paired
risk difference:

> treatment pre-noise alert rate − reference pre-noise alert rate

A difference of at least 20 percentage points is preregistered as operationally
material. This is the same scale as the frozen maximum normal-alert rate, not a
post-capture statistical threshold.

## Frozen outcome classification

- `topology_effect_reproduced`: both multi-worker treatments have a paired risk
  difference of at least 20 percentage points in every block.
- `no_material_topology_effect`: every paired difference has absolute magnitude
  below 20 percentage points.
- `mixed_topology_effect`: any other result.

The original v2 acceptance gates remain visible and unchanged. The diagnostic
can complete validly even when those gates fail; its purpose is to isolate the
failure mechanism.

## Preregistration

Before any matched capture exists, commit:

- the exact frozen v2 model and evaluation dependency closure;
- all nine matched manifests;
- the feature schema and frozen acceptance configuration;
- the runner, topology controls, protocol verifier, and matched-design
  validator;
- the material-effect threshold and classification rule; and
- the protocol hashes and required topology mapping.

The three canonical request schedules and three fault-kind/timing pairs must be
disjoint from training and all prior result-bearing cases. Repetition within a
matched block is required by design and must not be mistaken for case leakage.

## Required limitations

- This controlled intervention isolates worker count only within one local lab.
- Three workload blocks do not estimate a production effect distribution.
- Run order is counterbalanced but not randomized, so unobserved time drift
  remains possible.
- Equal admitted demand does not imply equal per-worker utilization.
- Redis, PostgreSQL, API, Collector, host, and telemetry vocabulary remain
  unchanged.
- Completion ratios still encode a domain assumption about admitted work.
- Attribution remains associative rather than causal.
- The target encoder remains linear PCA rather than a learned JEPA encoder.
