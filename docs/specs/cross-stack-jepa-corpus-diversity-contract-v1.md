# Cross-stack JEPA corpus-diversity contract v1

## Decision

Do not fit another cross-stack JEPA until a corpus passes the strict
role-separated tracer floor in this contract. The existing Quantis evidence
contains one substantive checkout-stack family, regardless of its runs,
replica counts, build hashes, feature tables, or derived caches.

This contract audits a prerequisite. Passing it would authorize a cross-stack
tracer; it would not promote a representation or alert policy.

## Claim boundary

No finite number of environments proves generalization to arbitrary
out-of-distribution stacks. Quantis will test only the declared family:

> containerized request/queue/worker/persistence service stacks with a common
> mechanism-level intervention vocabulary and a preregistered canonical
> metrics/logs/traces mapping.

The research basis and limitations are recorded in
`docs/research/cross-stack-jepa-corpus-diversity-primary-sources.md`.
In particular, IRM requires genuinely different data-generating environments
and has model-dependent environment-diversity conditions; DomainBed makes
model selection part of the domain-generalization problem; WILDS keeps test
data out of fitting and selection; OpenTelemetry distinguishes logical
services from their horizontally scaled instances.

## Public seams

The retained implementation must expose and test:

1. corpus discovery and source-identity extraction;
2. independence, derivation, and evidence-role classification;
3. deterministic minimum-design gap calculation;
4. reassessment from stored inventory and protocol bytes without loading a
   fitted model.

The auditor is read-only. It must not open raw telemetry arrays or consume
sealed outcomes.

## Identity rules

A collection campaign is identified by its raw capture references plus the
application, deployment, instrumentation, workload, and reset manifests.

A distinct stack environment must differ in the logical application stack:
service implementations, dependency or protocol boundaries, and their
canonical telemetry mapping. None of the following creates a new stack:

- a new run, seed, date, container, compose project, or build hash;
- a worker replica-count or other topology change;
- a new feature schema, model, cache, replay, or preprocessing pass over the
  same raw captures;
- a different fault or workload inside the same logical stack.

Derived corpora inherit the source campaign and evidence role. Confirmation
captures remain confirmation evidence even when recompiled. Qualification and
synthetic artifacts cannot satisfy an operational stack role.

## Strict exploratory tracer floor

Six distinct stack environments are required:

| Role | Stack count | Permitted use |
| --- | ---: | --- |
| fit | 3 | Representation, preprocessing, vocabulary, normalization, and baseline fitting |
| selection | 1 | Recipe, checkpoint, and value-lane selection |
| calibration | 1 | Alert, abstention, or retrieval threshold fitting |
| evaluation | 1 | One untouched target opened after every other byte is frozen |

Every collection campaign belongs wholly to one role. Raw captures, matched
pairs, trajectories, and derived windows may not cross roles. The evaluation
stack supplies neither labeled nor unlabeled bytes before the frozen run.

This floor permits only “worked/failed on named unseen stack X.” A
claim-bearing program requires ten environments: 3 fit, 2 selection,
2 calibration, and 3 sealed evaluation stacks, reported by macro-average,
worst stack, and every stack separately.

## Within-stack factorial floor

Each stack must expose the same five portable mechanism/target cells:

1. service or worker pause/unavailability;
2. persistence contention or lock;
3. queue/message production delay;
4. queue/message consumption delay;
5. ingress/API request rejection.

Each mechanism must be crossed with:

- three canonical topology levels: small, medium, and large;
- three workload-shape families: steady, ramp-or-burst, and periodic-or-
  multi-phase;
- at least three fresh, independently randomized, separately reset matched
  treatment-control run pairs per complete cell.

Thus the strict diagnostic floor is:

```text
5 mechanisms × 3 topologies × 3 workload families × 3 pairs = 135 pairs/stack
6 stacks × 135 pairs = 810 pairs = 1,620 trajectories
```

Three pairs per cell is only a public-benchmark-informed diagnostic floor.
Prefer five for variance estimation. Before any promotion or claim-bearing
collection, use non-sealed pilot paired-effect variance and a declared minimum
worthwhile effect to freeze a power calculation; use the greater of the
powered count and three.

Stack-specific actuator commands, target names, rates, and replica counts may
differ. Their canonical mechanism, topology level, workload family, effect
direction, units, and success oracle must be frozen before capture. Missing
mechanisms cannot silently shrink the shared claim.

## Leakage invalidators

The audit must reject readiness when it finds:

- source campaign or raw-capture overlap between roles;
- any derived corpus counted as an independent environment;
- confirmation, qualification, or synthetic data assigned to a tracer role;
- one stack assigned to multiple roles;
- incomplete mechanism, topology, workload, or repetition coverage;
- intervention or workload families unique to a stack or role;
- encoder-visible stack, deployment, run, pair, host, container, pod, trace,
  schema, SDK, or role identifiers;
- preprocessing, schema alignment, or vocabulary fitted outside fit stacks;
- post-outcome environment construction or oracle evaluation-stack selection.

## Existing-corpus accounting

The existing action-dynamics development campaign is the only open operational
campaign with matched intervention/control coverage. It has:

- one logical checkout stack;
- five intervention mechanisms;
- three worker-replica topologies;
- one stationary random workload family;
- eight matched pairs per action/topology cell.

It may be assigned wholly to one fit-stack role after a canonical cross-stack
adapter is frozen. It is short two workload families, or 90 minimum pairs:

```text
5 mechanisms × 3 topologies × 2 missing workloads × 3 pairs = 90
```

Five additional stacks require 675 more pairs. The minimum incremental
collection for a strict tracer is therefore 765 matched pairs (1,530
trajectories), assuming the existing campaign passes canonicalization and is
used only as fit evidence.

The nominal multimodal corpus may supplement fit-only normal behavior, but it
does not count as a complete stack because it has no matched interventions.
Contextual, graph, hybrid-event, and edge-preprocessing caches derived from
existing captures add no environments. Result-bearing confirmation corpora
are ineligible.

## Audit outputs and decision

The immutable audit bundle must contain:

- the exact source files and their SHA-256 identities;
- one normalized inventory record per candidate corpus;
- exclusions and source-campaign equivalence classes;
- existing and qualifying distinct stack counts;
- role, mechanism, topology, workload, and repetition gaps;
- the minimum additional stack, pair, and trajectory counts;
- a pure stored-input assessment and byte-identical verification;
- the retained runner, assessor, module, tests, contract, research note, and
  a manifest binding every file.

Until every strict floor is met, the only valid decision is:

`collect_cross_stack_corpus_before_jepa`

No JEPA loss, representation, model selection, calibration, or held-out-stack
score is computed by this ticket.
