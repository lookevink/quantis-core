# Cross-stack JEPA corpus diversity: primary-source notes

## Decision

There is no literature-backed integer number of environments that makes an
arbitrary cross-stack JEPA identifiable. Generalization to an arbitrary
out-of-distribution target is impossible without assumptions about the allowed
shift, and even invariant-learning guarantees require particular linear models
and sufficiently diverse environments. A finite Quantis corpus can therefore
support only a claim over a preregistered family of stacks, workloads,
interventions, and telemetry schemas—not universal stack portability.

For Quantis, distinguish the scientific floor from role separation and
replication:

| Evidence level | Distinct stack environments | Permitted conclusion |
| --- | ---: | --- |
| Scientific development floor | `3 source + 1 untouched target = 4` | A cross-stack test is possible at all. Select by leave-one-stack-out across the three sources and calibrate on source-only, run-disjoint data. The conclusion is limited to the one named target. |
| Standard OOD role split | `3 source + 1 OOD selection/calibration + 1 target = 5` | Selection/calibration see a non-source shift, but still share one stack and must use disjoint runs. |
| Strict role-separated tracer | `3 fit + 1 selection + 1 calibration + 1 evaluation = 6` | Every role has a distinct stack. This is the minimum safe Quantis contract when abstention calibration is part of the claim. It still tests only one target stack. |
| Replicated claim-bearing program | `3 fit + 2 selection + 2 calibration + 3 sealed evaluation = 10` | Repeated success or failure on three preregistered unseen stacks from the declared stack family. This remains bounded benchmark evidence, not a production failure-rate estimate or universal portability proof. |

Four stacks is the minimum defensible development experiment, five mirrors the
source/OOD-validation/OOD-test pattern, and six is the minimum with
environment-distinct fit, selection, calibration, and evaluation. The
`3/2/2/3` ten-stack allocation is a stronger Quantis replication target. All
are design inferences, not theorems: the results below explain why fewer
environments are weak, but do not prove that any finite count is sufficient.

Within every stack environment, collect at least three separately reset,
independently randomized matched treatment-control run pairs per portable
intervention-target cell and prefer five before interpreting variance. This
`3–5` range follows public benchmark practice and is only a diagnostic floor,
not a power calculation. Estimate run-pair variance in a pilot, then freeze a
power calculation for the smallest effect worth promoting. Overlapping
telemetry windows never increase the number of independent runs or
environments.

## Unit definitions for Quantis

The following definitions keep changes in deployment identity from masquerading
as changes in domain.

### Run

A **run** is one freshly launched, bounded capture with:

- a unique raw-capture identifier;
- an independently drawn workload seed and, when applicable, intervention
  timing/magnitude seed;
- one fixed stack/environment manifest;
- no raw records, windows, matched pair, replay, or derived cache shared with
  another experimental role.

Multiple windows from a run are dependent observations nested inside that run.
Multiple runs of the same manifest are repetitions, not new environments.

### Stack

A **stack** is a logical application system, identified before outcomes are
observed by:

- application and service implementations and pinned versions;
- the logical service/dependency graph and protocol boundaries;
- stateful dependency kinds and ownership boundaries;
- the portable task and intervention semantics available on that application;
- the instrumentation and telemetry-semantic mapping used to expose the task.

A worker replica-count change, a fresh container, a pod restart, a new random
seed, or a new `service.instance.id` does **not** create a new stack.
OpenTelemetry defines `service.name` as the logical service name shared by
horizontally scaled instances and `service.instance.id` as the unique identity
of one such instance. This directly rules out counting replica identities as
stack diversity
([OpenTelemetry service semantic conventions](https://opentelemetry.io/docs/specs/semconv/resource/service/)).

### Environment

An **environment** is a preregistered generative regime:

```text
stack
× topology/configuration family
× workload policy family
× intervention policy
× platform/runtime class
× instrumentation/schema version
```

An independent *stack environment* must differ in the stack component, not only
in its run seed or replica count. A topology variant of the same checkout stack
is useful within-stack environment variation, but it is not cross-stack
evidence.

This factorization is consistent with AIOpsLab, which defines an operational
environment through the service, fault, and workload conditions and deploys
different applications as cloud environments
([AIOpsLab paper, Sections 2.2.3 and 2.3](https://www.microsoft.com/en-us/research/wp-content/uploads/2024/10/AIOpsLab-672458b635d2d.pdf)).
It is deliberately stricter than treating every benchmark problem or capture
as a domain.

### Corpus

A **corpus** is a collection of environments and their nested runs. Two files,
caches, or feature tables derived from the same raw run remain one run.
Likewise, hundreds of failure cases from three systems are three stack
environments, not hundreds of environments. RCAEval is a useful scale
reference: it reports 735 cases but only three microservice systems—Online
Boutique, Sock Shop, and Train Ticket—and explicitly describes three or five
repetitions per fault-service combination
([RCAEval source repository](https://github.com/phamquiluan/RCAEval#available-datasets)).

## Why environment count, not window count, matters

### Domain generalization needs real distributional variation

Invariant Risk Minimization (IRM) explicitly warns that random partitions of
one dataset follow the same distribution and therefore are not diverse
environments. It also warns that arbitrarily conditioning a dataset can create
spurious correlations or destroy the invariance of interest
([Arjovsky et al., 2019, pp. 10–12](https://leon.bottou.org/publications/pdf/tr-irm-2019.pdf)).
Consequently:

- slicing one Quantis trajectory into many windows adds no environment
  diversity;
- splitting one capture by action phase is not an environment construction;
- calling different seeds “domains” does not establish that shortcut
  correlations changed;
- environments must be defined from the data-generating intervention and stack
  manifest before observing results.

In the linear IRM analysis, a rank-`r` representation over `d` observed
dimensions needs training environments in a specified linear general position,
with a model-specific bound involving `d - r + d/r`. The same paper states that
it could not establish an analogous nonlinear general-position guarantee
([Arjovsky et al., 2019, Assumption 8 and Theorem 9](https://leon.bottou.org/publications/pdf/tr-irm-2019.pdf)).
This is a conditional theorem, not a recipe for declaring three or ten
telemetry stacks sufficient.

Rosenfeld, Ravikumar, and Risteski sharpen the limitation under their linear
classification model. If `E` is the number of environments and `d_e` the
dimension of environmental/spurious latent features, an IRM-feasible solution
is forced off those features only when `E > d_e`; when `E <= d_e`, the global
minimum can use non-invariant features and fail on unseen environments
([The Risks of Invariant Risk Minimization, Theorem 5 discussion](https://arxiv.org/abs/2010.05761)).
Telemetry has many potential environmental coordinates and no known `d_e`, so
a handful of topologies cannot provide an identifiability guarantee.

The practical formulation has further limits. Kamath et al. show failures of
IRMv1 even in simple population settings and report that sampling makes it
especially fragile
([Does Invariant Risk Minimization Capture Invariance?](https://proceedings.mlr.press/v130/kamath21a)).
These are warnings against treating an invariance-style loss—or a JEPA loss
trained across named environments—as evidence that the learned representation
is portable.

### Arbitrary OOD generalization is not learnable

The NeurIPS theoretical framework of Ye et al. states that generalization to
arbitrary OOD distributions is impossible and derives bounds only after
restricting how test variance expands relative to training domains
([Towards a Theoretical Framework of Out-of-Distribution Generalization](https://papers.nips.cc/paper/2021/hash/c5c1cb0bebd56ae38817b251ad72bedb-Abstract.html)).
Therefore the Quantis target population must be a declared family, for example:

> containerized request/queue/worker/persistence microservice stacks with a
> common mechanism-level intervention vocabulary and OTLP-derived
> metrics/logs/traces.

Passing stacks from that family does not justify claims about arbitrary
organizations, architectures, instrumentation, traffic processes, or incident
mechanisms.

## Frozen role design

### Four-stack scientific floor

Use three genuinely distinct source stacks and one untouched target stack.
Across the three sources, perform leave-one-stack-out model selection: fit on
two source stacks, validate on the third, rotate the held-out source, and
aggregate the selection score. Fit final source-only state after the recipe is
fixed. Calibration, if required, must use source-only runs disjoint from both
fit and selection. Open the target stack only after all recipe and threshold
bytes are frozen.

This is the smallest design in which every leave-one-source-out fit sees more
than one stack and final evaluation holds out a whole application. It supports
only “worked/failed on held-out stack X.”

### Five-stack standard split

Use three source stacks, one OOD validation stack, and one OOD test stack.
Selection and calibration may share the OOD validation stack only through
disjoint run blocks and only under a frozen order: select first, then calibrate
the selected recipe. This resembles WILDS' source/OOD-validation/OOD-test
separation, but one validation stack still represents only one shift.

### Exploratory six-stack floor

Use this only when the goal is to decide whether a recipe deserves a larger
corpus:

| Role | Stack environments | Allowed use |
| --- | ---: | --- |
| Fit | 3 | JEPA/self-supervised fitting, feature statistics, vocabulary fitting, normalization, and supervised-control fitting. |
| Selection | 1 | Choose the already enumerated recipe/checkpoint/value lane. |
| Calibration | 1 | Fit alert, abstention, or retrieval thresholds after selection. |
| Evaluation | 1 | One untouched target stack, opened once after all other bytes are frozen. |

This design can produce “worked/failed on held-out stack X.” It cannot produce
“generalizes across stacks,” because every non-fit role is represented by a
single shift.

### Minimum claim-bearing ten-stack floor

| Role | Stack environments | Required independence |
| --- | ---: | --- |
| Fit | 3 | Three distinct applications/stacks; all learned representation, preprocessing, and baseline parameters come only from these stacks. |
| Selection | 2 | Two additional stacks; select by macro-average and reject a recipe that wins on only one. |
| Calibration | 2 | Two further stacks; freeze one common calibration policy and report each stack separately. |
| Evaluation | 3 | Three sealed stacks; no labeled or unlabeled telemetry is consumed before the frozen run. |

Report the macro-average across stacks, the worst stack, and every stack result.
Do not weight a large or long-running stack more heavily merely because it
produces more windows.

Three fit stacks are the smallest practical Quantis source set that is not a
single contrast. It is not enough to satisfy the unknown-dimensional IRM bound.
Two selection and two calibration stacks are required so recipe and threshold
choices cannot be optimized to one shift. Three evaluation stacks are the
smallest replication set that supports the exact bounded wording “on three
preregistered unseen stacks”; it still gives a very weak population failure-rate
bound.

### Model selection is part of the contract

DomainBed concludes that a domain-generalization algorithm without a model
selection method is incomplete. Its source code distinguishes in-domain
selection, leave-one-domain-out selection, and an oracle method that consumes
the test domain
([DomainBed paper](https://arxiv.org/abs/2007.01434);
[official repository](https://github.com/facebookresearch/DomainBed)).
Quantis must never use oracle selection: evaluation stacks cannot select
architecture, checkpoint, mask, latent width, loss weight, score, or threshold.

WILDS similarly prohibits any test-set use for training or selection and
requires separate validation data. Its leaderboard also requires at least
three model seeds, with more for unstable datasets
([WILDS submission rules](https://wilds.stanford.edu/submit/)).
Quantis should freeze five model seeds and require the promotion gate to hold
for at least four; stack-level results remain the primary generalization units.

If any unlabeled target-stack telemetry is used in fitting or preprocessing,
the experiment is no longer zero-shot domain generalization. WILDS 2.0 treats
access to target-like unlabeled data as a separate unsupervised-adaptation
setting, including for self-supervised methods
([WILDS 2.0](https://arxiv.org/abs/2112.05090)).
Quantis must name such a run “unlabeled target adaptation,” not “cross-stack
zero-shot JEPA.”

## Schedule, intervention, and repetition coverage

### Portable intervention cells

Freeze a mechanism-level vocabulary that is meaningful on every included
stack. The current Quantis concepts can be generalized as:

1. service/worker unavailability or pause;
2. request rejection at an ingress or API boundary;
3. queue or message-production delay;
4. queue or message-consumption delay;
5. persistence contention or lock.

For a portability claim about action attribution, every claimed mechanism must
appear in every role and on every stack where the corresponding component
exists. Stack-specific actuator commands and service names may differ, but the
mechanism label and success oracle must be frozen before capture. If a
mechanism is absent on some stack, either:

- remove it from the shared cross-stack claim and report it in a
  stack-specific panel; or
- collect a different stack on which the full shared vocabulary exists.

Never let an intervention label identify the stack. For example, if database
lock occurs only on stack A and queue delay only on stack B, action
classification is environment classification.

### Workload schedules

Use at least three declared workload-shape families per stack:

- stationary/steady demand;
- ramp or burst demand;
- periodic or multi-phase demand.

Cross every portable intervention-target cell with all three families. Vary
absolute rates within the stack's feasible range; compare effects in
dimensionless or demand-relative units when raw service scales differ. Workload
family, intervention, and stack role must not be deterministically linked.

AIOpsLab's benchmark construction independently composes applications,
workload generators, fault generators, and task evaluators, demonstrating why
these are separate experimental factors rather than interchangeable “cases”
([AIOpsLab overview](https://microsoft.github.io/AIOpsLab/)).

### Independent repetitions

For each stack × workload family × intervention-target cell:

- collect at least three fresh matched treatment-control run pairs and prefer
  five before estimating variance;
- redraw workload realization and intervention timing/magnitude within the
  frozen range for every pair;
- launch treatment and control from equivalent clean state;
- balance pair order so time-of-day or global run order is not a label proxy;
- retain each raw capture and manifest as a separate immutable object.

Three to five pairs can expose gross instability and seed the pilot variance.
They are not automatically enough for a promotion gate. Before the
claim-bearing collection, estimate the paired-effect variance `sigma_d²` from
non-sealed pilots, state a minimum worthwhile effect `delta`, and calculate
the required pair count for the frozen paired analysis. Inflate that count for
within-stack clustering rather than treating windows as independent.

RCAEval's public corpus illustrates the difference between systems, cases, and
repetitions: its suites span three systems and use three or five repetitions
per fault-service pair
([RCAEval repository](https://github.com/phamquiluan/RCAEval#available-datasets)).
Those repetitions increase within-system precision; they do not turn three
systems into hundreds of independent domains.

## Dependence and finite-sample reporting

### Split and analyze by the highest shared group

Random cross-validation under temporal, spatial, or hierarchical dependence
can seriously underestimate prediction error. Blocked cross-validation is
recommended when the prediction target is new data or predictor space
([Roberts et al., 2017](https://www.biom.uni-freiburg.de/mitarbeiter/dormann/roberts-et-al-2017-ecography.pdf/at_download/file)).
Official `GroupKFold` guidance makes the corresponding implementation rule:
when samples within a group are dependent, no group may appear in both the
training and validation fold
([scikit-learn grouped cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data)).

Quantis grouping must therefore be nested:

```text
stack environment
  └── raw run / matched pair
        └── trajectory
              └── overlapping windows
```

Role splits occur at the stack level. Within a role, resampling and uncertainty
estimation use runs or matched pairs as blocks. Windows are inputs to the model,
not independent inferential samples.

If `m` dependent observations share a cluster and their intracluster
correlation is `rho`, the familiar design effect is

```text
DE = 1 + (m - 1) rho
effective_n ≈ nominal_n / DE
```

Cluster-trial sample-size methodology uses this factor to show why adding
correlated observations cannot replace adding clusters
([Rutterford, Copas, and Eldridge, 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4521133/)).
The formula is a planning diagnostic here, not permission to convert windows
into fractional independent environments.

### What zero failures means

For `n` independent Bernoulli evaluation units and zero observed failures, the
one-sided 95% upper bound on failure probability is:

```text
p_upper = 1 - 0.05 ** (1 / n) ≈ 3 / n
```

This is the exact zero-event calculation behind the “rule of three”
([Hanley and Lippman-Hand, 1983](https://doi.org/10.1001/jama.1983.03330370053031)).
If the unit is an unseen stack:

| Evaluation stacks with zero failures | 95% upper bound on stack failure probability |
| ---: | ---: |
| 1 | 95.0% |
| 3 | 63.2% |
| 10 | 25.9% |
| 29 | 9.8% |
| 59 | 5.0% |

Therefore three sealed evaluation stacks are enough for replicated bounded
evidence, not a reliability claim. Pair-level success across many runs can
bound within-sampled-stack run risk, but it cannot be substituted into this
stack-level table.

## Hard leakage and shortcut invalidators

Any item below invalidates a zero-shot cross-stack claim:

1. **Raw-source overlap:** a raw capture, replay, matched pair, trajectory, or
   derived window contributes to more than one role.
2. **Target access:** labeled or unlabeled evaluation-stack telemetry
   contributes to fitting, normalization, vocabulary construction, schema
   alignment, checkpoint selection, calibration, or early stopping.
3. **Oracle model selection:** evaluation results choose any recipe byte.
4. **Environment-label confounding:** an intervention, target, control status,
   workload family, or outcome exists only in one environment or role.
5. **Identifier leakage:** run IDs, pair IDs, deployment names, stack names,
   host/container/pod IDs, `service.instance.id`, trace IDs, filenames, wall
   clock epochs, or role labels are visible to the encoder or downstream
   policy.
6. **Schema-as-label leakage:** metric names, service vocabulary, missing-signal
   patterns, units, column order, SDK language/version, or instrumentation
   version uniquely identify both environment and answer.
7. **Role-specific preprocessing:** target-stack-specific imputation,
   normalization, vocabulary, topology mapping, PCA, feature selection, or
   dimensionality is fitted outside the fit role.
8. **Schedule shortcuts:** action timing, duration, magnitude, or global run
   order determines the label or stack.
9. **Pseudoreplication:** windows are used as degrees of freedom for a run- or
   stack-level claim.
10. **Post-outcome environment construction:** domains are split or merged
    after observing model errors to improve the result.
11. **Incommensurate tasks:** the label or success oracle changes meaning
    across stacks while the aggregate metric pretends it is one task.
12. **Missing matched controls:** a stack/workload/intervention cell contains
    treatments without equivalent nominal runs, making ordinary stack behavior
    indistinguishable from action effect.

OpenTelemetry resource attributes make schema shortcuts especially plausible:
`service.name`, `service.version`, `telemetry.sdk.language`,
`telemetry.sdk.version`, telemetry distribution, deployment, Kubernetes,
container, host, and cloud attributes are designed to identify resources and
environments
([OpenTelemetry resource semantic conventions](https://opentelemetry.io/docs/specs/semconv/resource/)).
Keep these fields in the evidence manifest, but exclude or explicitly
canonicalize them before encoder input. The canonicalizer itself must be
fit/preregistered without evaluation-stack outcomes.

Required falsification diagnostics, even after identifiers are removed:

- train a simple frozen-embedding probe for stack identity;
- evaluate a metadata-only baseline;
- shuffle intervention labels within each stack and repeat the downstream
  probe;
- mask service names and permute canonical entity order;
- compare declared topology against topology-shuffled and entity-local
  controls;
- report performance separately by stack, workload family, intervention, and
  target rather than only in aggregate.

High stack-probe accuracy is not automatically fatal—a useful representation
may retain legitimate structural state—but value that disappears after
macro-averaging or within-stack label shuffling is not portable incident
evidence.

## What fewer environments can and cannot establish

| Available diversity | Valid conclusion | Invalid conclusion |
| --- | --- | --- |
| One stack, many runs/windows | Within-stack repetition; workload, action, or topology transfer if those factors are genuinely held out. | Cross-stack representation portability. |
| One stack, several replica counts | Transfer across those named topology configurations of that stack. | Independent stack environments or general service-graph transfer. |
| Two training environments | A learned relation is consistent with two observed regimes. | Identification of invariant features; the IRM results explicitly permit spurious solutions when environment diversity is too small. |
| Three fit stacks, one unseen evaluation stack | A role-clean result on that named unseen stack. | Repeated cross-stack generalization or a stack-population failure rate. |
| Four-stack `3 source / 1 target` floor | A whole-application target test with leave-one-source-stack-out selection. | Environment-distinct selection/calibration or repeated target-stack portability. |
| Five-stack `3 source / 1 OOD validation / 1 target` split | A named-target result after tuning on a separate OOD stack. | Separate validation and calibration shifts, or a target-population failure rate. |
| Six-stack exploratory floor | A fully role-separated tracer that can reject an architecture without consuming a larger sealed corpus. | Promotion to production or a generic topology-portability claim. |
| Ten-stack `3/2/2/3` program | Repeated bounded evidence on three sealed stacks in the declared family, with selection and calibration replicated. | Universal zero-shot portability, a narrow production failure bound, or causal incident attribution. |
| Three public benchmark systems with hundreds of cases | Cross-system evaluation over those three systems, if roles and schemas are clean. | Hundreds of independent environments; cases remain nested within systems. |

## External stack candidates are evidence sources, not automatic domains

Primary telemetry benchmarks demonstrate feasible sources of distinct
applications:

- DeathStarBench publishes separate end-to-end Social Network, Media Service,
  and Hotel Reservation applications rather than treating service replicas as
  applications
  ([official repository](https://github.com/delimitrou/DeathStarBench)).
- AIOpsLab integrates HotelReservation and SocialNetwork as two cloud-service
  environments, composes workload and fault generators, and includes nominal
  no-op problems
  ([AIOpsLab paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2024/10/AIOpsLab-672458b635d2d.pdf)).
- RCAEval supplies three microservice systems and a common set of resource,
  delay, loss, socket, and code-level failure cases
  ([RCAEval paper](https://arxiv.org/abs/2412.17015);
  [source repository](https://github.com/phamquiluan/RCAEval)).

These corpora cannot simply be concatenated with Quantis. Their labels,
instrumentation, metric dimensions, workload generators, capture boundaries,
and fault oracles differ. They qualify only after a preregistered adapter maps a
shared mechanism vocabulary and common observable feature contract without
using evaluation outcomes. A public dataset previously used during recipe
development also cannot later become sealed confirmation evidence.

## Preregistration checklist

Before any claim-bearing capture, freeze:

1. the allowed stack family and exact `stack_id`/`environment_id` construction;
2. all ten role assignments and hashes of application, deployment,
   instrumentation, and schema manifests;
3. the shared mechanism-level intervention and target vocabulary;
4. three or more workload families and randomized schedule ranges;
5. matched treatment-control protocol and clean-state verifier;
6. pilot-variance-based run-pair power calculation, with three separately reset
   pairs per cell as the non-negotiable floor and five preferred;
7. the encoder-visible feature contract and explicit excluded identifier list;
8. schema canonicalization learned only from fit stacks;
9. exact JEPA candidate and raw/PCA/supervised/non-JEPA controls;
10. five model seeds and the across-seed promotion rule;
11. selection score, calibration procedure, per-stack and worst-stack value
    gates, and abstention/false-alert gates;
12. grouped resampling and confidence intervals at run and stack levels;
13. immutable raw-capture, compiled-corpus, runner, assessor, and manifest
    dependency hashes;
14. a rule that opening any evaluation stack before the frozen run retires it
    from evaluation and reassigns it to development.

## Bottom line for Quantis

The existing checkout-stack runs may count as many independent run pairs and
several within-stack topology environments, but they remain one stack family.
They can reject weak JEPA recipes and establish named topology transfer. They
cannot identify a cross-stack invariant representation.

Do not run another cross-stack JEPA on that corpus and call the result portable.
First assemble either:

- four distinct stacks for the scientific development floor (`3 source / 1
  untouched target`), using source-only leave-one-stack-out selection;
- five stacks for a source/OOD-validation/OOD-test split;
- six stacks when fit, selection, calibration, and evaluation must be
  environment-distinct; or
- ten stacks under the `3 fit / 2 selection / 2 calibration / 3 sealed
  evaluation` contract for repeated, bounded cross-stack evidence.

Even the ten-stack program should conclude only:

> The frozen representation and policy met (or failed) the preregistered gates
> on three unseen stacks drawn from the declared benchmark family.

Production alerting still requires broader stack coverage, a substantially
tighter stack-level failure bound, and sustained shadow evaluation.
