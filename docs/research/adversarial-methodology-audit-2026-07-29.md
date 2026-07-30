# Adversarial methodology audit, 2026-07-29

## Bottom line

The repository supports a strong, useful conclusion:

> None of the tested models is ready for a real paging path, and the raw
> rank-32 transition plus direct raw/PCA retrieval are the correct incumbents
> for the next **shadow** system.

That conclusion is robust because promotion was deliberately conservative,
false-alarm failures were common, the strongest JEPA losses to raw telemetry
were often large, and the documents usually keep their claims inside the
observed boundary.

The repository does **not** support either of these stronger conclusions:

1. that the rank-32 model is production-valid; or
2. that JEPA as a model family has been exhausted or generally falsified.

Almost every action-dynamics and JEPA result is open-development evidence from
one repeatedly inspected corpus, one stack, one schema, and one held worker
topology. Most JEPA tracers use one optimizer seed and only ten held-topology
matched pairs. The 23 tracers are not 23 independent replications: they share
the same data, split, baseline, and investigators.

The most scientifically rigorous model experiment is the five-seed contextual
metrics-and-logs confirmation. The most rigorous intervention corpus is the
120-pair action-dynamics collection. The strongest current engineering
candidate is rank-32 raw action-conditioned dynamics, but it still requires a
fresh sealed, run-blocked alert-policy confirmation before shadow promotion.

## Scope and audit method

This audit read the frozen protocols, executable runners/assessors, retained
manifests/results, implementation tickets, and source-adaptation notes. Summary
prose was used only to locate primary evidence. The inventory comes from:

- the [repository experiment directory](../../README.md);
- the [JEPA catalog](../../experiments/jepa/catalog.json);
- frozen protocols in [`docs/specs`](../specs);
- runners in [`lab`](../../lab);
- conclusion-bearing evidence under [`artifacts`](../../artifacts); and
- the [JEPA program tickets](../wayfinding/jepa-implementation-program/map.md).

For each family, the audit asked:

1. What is the independent unit: point, overlapping window, trajectory,
   matched pair, schedule family, or collection block?
2. Are fitting, selection, calibration, and evaluation genuinely separated?
3. Was an ostensibly held role adaptively reused after inspection?
4. Does the transfer split test the claim being made?
5. Are nulls, capacity controls, ablations, and raw comparators adequate?
6. Are configuration selection and alert calibration clean?
7. Is there enough independent evidence, uncertainty, and power?
8. Does the assessor recompute results without trusting the producer?
9. Are data, source, runtime, and artifact identities recoverable?
10. Were multiplicity and stopping rules defined?
11. Does the written conclusion stay inside the evidence boundary?

Severity in this report means:

- **Fatal validity threat:** prevents the stronger interpretation stated in
  the audit, even if the underlying run remains useful.
- **Material limitation:** does not erase the bounded finding but substantially
  lowers confidence or narrows transfer.
- **Defensible choice:** conservative or well controlled under the declared
  development purpose.

No experiment code or evidence artifact was modified during this audit.

## Cross-program findings

### Fatal for production and “frontier exhaustion”

#### 1. The action/JEPA evaluation role is adaptive, not held out

The initial edge tournament explicitly says its evaluation split had already
informed redesign and therefore is not sealed confirmation
([edge protocol, lines 83–89](../specs/edge-dynamics-development-v1.md);
[result, lines 13–15](edge-dynamics-development-v1-results.md)). The JEPA
ladder then reuses the same action corpus and a previously opened
worker-three transfer diagnostic, with no fresh case for any tracer
([ladder, lines 18–35](../specs/jepa-experiment-ladder-v1.md)).

Role identifiers are disjoint, so this is not row leakage. It is **adaptive
research-program reuse**: architecture ideas, nulls, gates, and later targets
were chosen after observing earlier results on the same transfer role.
Consequences:

- transfer remains a useful development falsifier;
- a large loss to raw is credible evidence against that exact recipe;
- close wins/losses are not unbiased generalization estimates; and
- the sequence cannot establish that the search space is exhausted.

This is the model-selection analogue of repeatedly consulting a test set.
Cawley and Talbot show that optimizing over a noisy finite-sample selection
criterion can create selection bias comparable to the reported differences
between algorithms
([JMLR, 2010](https://www.jmlr.org/papers/v11/cawley10a.html)).

#### 2. Sample sizes are too small for operational rates

The common tracer roles contain 40 fit, 10 selection, 10 calibration, 20 IID
evaluation, and 10 held-topology pairs
([ladder, lines 29–35](../specs/jepa-experiment-ladder-v1.md)). A held-topology
rate of 0/10 false alarms has an exact two-sided 95% binomial upper bound of
about 30.8%, not 5%. Detection of 8/10 has an interval of roughly 44%–97%;
10/10 only lowers the 95% bound to about 69%. There are generally only two
held-topology pairs per action family.

The gates are sensible engineering screens, but the observed proportions do
not estimate production false-alarm or detection rates precisely. Zero
empirical false alarms on ten controls cannot validate a 5% SLO.

The same problem is sharper for a nominal 95% split-conformal guarantee. With
`n=10` exchangeable calibration units, the finite-sample conformal rank is
`ceil((n+1) × 0.95) = 11`, which is not available among ten residuals. Using
the observed maximum instead gives rank coverage `10/11 ≈ 90.9%` under the
ideal continuous-exchangeable model; a valid 95% procedure would need an
infinite/randomized boundary or more independent calibration units. Therefore
the tracer thresholds are empirical screens, not 95% finite-sample
certificates.

The calibration controls also come from worker topologies one/two while the
headline certificate is evaluated on topology three. Ordinary split-conformal
coverage depends on exchangeability; work on non-exchangeable data requires
additional dependence/shift conditions and pays a coverage penalty
([Oliveira et al., JMLR 2024](https://www.jmlr.org/papers/v25/23-1553.html)).
No such transfer guarantee is established here.

#### 3. One training seed dominates the JEPA ladder

The ladder correctly says a one-seed pass only authorizes fixed-seed
robustness, followed by fresh confirmation
([ladder, lines 273–276 and 311–316](../specs/jepa-experiment-ladder-v1.md)).
None of the 23 tracers earned that step. A one-seed **failure** is enough for
cost-conscious recipe triage when the margin is overwhelming, but it does not
distinguish a structurally bad objective from an unlucky optimization outcome
when the margin is close.

This matters most for near-threshold results such as SALT's 8.89% transfer
alignment advantage against a 10% gate, HEPA's 50% versus 60% treatment
detection, and small geometry/rank differences. It matters less where a model
has roughly two-to-five times raw effect error.

#### 4. “Twenty-three experiments” is not multiplicity control

Later protocols freeze each tracer and often use conjunction gates: every
safety, mechanism, and value gate must pass. This is conservative for
promotion but creates a high false-negative probability for scientific claims.
There is no program-level alpha allocation, hierarchical model, or
preregistered stopping rule across the 23 adaptively ordered tracers.

Accordingly:

- a pass would have required fresh confirmation, which is defensible;
- a fail rejects a **recipe at tracer stage**, not the family;
- counting all failures as independent evidence against JEPA is invalid; and
- “try every known JEPA” is a search narrative, not a statistical exhaustion
  result.

#### 5. The transfer axis is narrow

The common transfer is worker replicas one/two to three. Redis, PostgreSQL,
API, Collector, host, feature vocabulary, action library, and graph schema
remain fixed. The earlier matched diagnostic correctly notes that worker count
is only one local-lab factor and that equal admitted demand does not imply equal
per-worker utilization
([matched-topology protocol, lines 75–85](../specs/matched-topology-diagnostic.md)).

This is topology-parameter interpolation within one deployment family, not
transfer across stacks, software versions, missing telemetry, collector
degradation, action mechanisms, or incident distributions.

### Material limitations

#### 6. Overlapping windows are predictions, not independent samples

Each action capture yields 79 heavily overlapping windows
([action protocol, lines 98–109](../specs/action-dynamics-development-v1.md)).
The best protocols correctly keep matched pairs atomic, use trajectory-level
alert gates, and in several newer tracers use pair-balanced scores or a
pair-blocked SIGReg sample axis. That is good.

However, aggregate MSEs and calibration residual pools can still be dominated
by correlated windows. Reports rarely attach cluster bootstrap intervals over
pairs, action-by-topology cells, or collection batches. Point coverage such as
99.91% can coexist with 20% control-trajectory false alarms, exactly as the
Error-Certificate result demonstrates
([result, lines 8–13 and 43–55](error-certificate-jepa-v1-results.md)).
Treating correlated subsamples as experimental replication is the classic
pseudoreplication error
([Hurlbert, 1984](https://esajournals.onlinelibrary.wiley.com/doi/10.2307/1942661));
the stronger Quantis protocols avoid the worst form by keeping pairs atomic,
but their uncertainty reporting should use that same unit.

#### 7. Matched twins control schedules, not time

Treatment and control share workload/topology seeds and are captured as
isolated twins. Collection forbids retries and deletions, and twin order is
counterbalanced in the instrumentation design. Those are strong choices.
But twins are sequential rather than simultaneous; six lanes also run in
parallel. Host load, thermal state, Docker startup, and neighboring lane
teardown can create batch/time effects. The v4 result itself retains a risk
that a faster lane tears down while a slower lane is still in recovery
([v4 result, lines 99–105](action-dynamics-instrumentation-pilot-v4-results.md)).

No downstream model reports a random collection-batch effect or a
block/bootstrap sensitivity analysis.

#### 8. The incumbent received more adaptive optimization

The raw rank-32 model is the product-relevant comparator and should be hard to
beat. But it emerged from an earlier open tournament and was repeatedly reused,
whereas most JEPA translations received one frozen configuration and one seed.
That is fair for the deployment question, “does this candidate beat the best
known small incumbent?” It is asymmetric for the scientific question, “does
this objective have value after comparable optimization effort?”

The reports should keep saying **exact recipe rejected**, not “technique
rejected.”

#### 9. “Independent assessor” usually means computational, not epistemic,
independence

Later artifacts are strong because assessors consume stored arrays, recompute
metrics/gates, verify manifests, and sometimes run from copied source in an
isolated interpreter. Nevertheless, they were developed in the same repository
by the same research process; several import producer-side helpers. This
protects against stale summaries and many implementation mistakes, not against
a shared conceptual error in labels, metrics, or adaptation.

The strongest tier is the isolated copied-source assessment used by LeNEPA,
Discrete-JEPA, PEIRA, VISReg, and JEPA-SCORE. The weakest tier is the
regime-codebook, event-native, and SIGReg prototypes, which have no cataloged
assessor.

#### 10. Artifact integrity is good, availability is incomplete

This audit recomputed every hash in the current cataloged artifact manifests.
All present entries matched. The later artifacts include source snapshots and
often transitive reproduction closure. This is unusually strong.

Two exceptions are material:

- the regime-codebook and event-native artifact directories contain a single
  `prototype-result.json`, no artifact manifest, and no bound implementation
  commit/source closure; and
- the multi-hypothesis catalog points to the invalid v1 producer bundle even
  though the conclusion-bearing correction is v2
  ([v2 result, lines 17–33](multi-hypothesis-jepa-prototype-v2-results.md)).

Also, most large bundles are Git-ignored and have not yet been published.
Hashes establish integrity only for bytes a reviewer can obtain. Until the
content-addressed bundles are uploaded, third-party reproducibility is a plan,
not a demonstrated property.

The central [JEPA reproduction guide](../../lab/action_dynamics/JEPA_REPRODUCTION.md)
also documents exact commands for only 11 of the 23 cataloged tracers. The
technique capsules link all runners/specs/findings, which is good navigation,
but a runner link is not a complete reproduction recipe: later experiments
often require exact source-artifact hashes, isolated assessor invocation,
non-default refusal flags, and version-specific output handling.

#### 11. Artifact hashes detect mutation, not authorship

A SHA-256 manifest is excellent for accidental-change detection and content
addressing. It is not an external timestamp, signature, or proof that the
protocol predated observation. Git commits and retained failed attempts improve
the chronology substantially. A release attestation or signed transparency
record would close the remaining adversarial provenance gap.

For several early tracers, “preregistered” means frozen inside the research
process rather than independently timestamped before results. Repository
history first adds the regime, event-native, multi-hypothesis, SIGReg,
complete-LeJEPA, and retrieval specs alongside their result-bearing change;
HEPA's spec and result likewise first appear together. That does not imply
post-hoc fabrication, but an adversarial reviewer cannot independently verify
the freeze date from Git history alone.

#### 12. Source papers motivate mechanisms, not telemetry efficacy

Several translations are substantial. The external sources make claims on
images, video, or their own time-series labels:

- [LeJEPA](https://arxiv.org/abs/2511.08544) introduces SIGReg and a
  predictor-free representation objective across image-style datasets;
- [HEPA](https://arxiv.org/abs/2605.11130) predicts labeled critical events
  across time-series benchmarks, whereas Quantis defines an event by a
  fitting-control state-change quantile;
- [SALT](https://arxiv.org/abs/2509.24317) uses a frozen reconstructive video
  teacher; and
- [JEPA-SCORE](https://arxiv.org/abs/2510.05949) derives density through the
  encoder Jacobian and an expectation over training transformations, while the
  Quantis screen freezes one Monte Carlo transform.

The source notes do a good job declaring these adaptation boundaries. Results
must preserve them: a source-faithful mechanism test at much smaller width,
different views/targets, and one seed is not a reproduction of the source
paper's empirical claim.

### Defensible methodological choices

- Whole matched pairs, not windows, are assigned to roles
  ([edge protocol, lines 16–29](../specs/edge-dynamics-development-v1.md)).
- Normalization and probes are generally fit on fitting data only.
- Selection and calibration have distinct roles.
- The true action is hidden for alerting and attribution; a no-action candidate
  is present.
- Raw/PCA, capacity-matched, supervised, and mechanism-breaking nulls are
  routinely included
  ([ladder, lines 145–162](../specs/jepa-experiment-ladder-v1.md)).
- Later assessors recompute gates from retained tensors instead of trusting
  producer booleans
  ([ladder, lines 113–143](../specs/jepa-experiment-ladder-v1.md)).
- Failed and invalid attempts are retained and explicitly labeled rather than
  deleted.
- Promotion gates are operationally meaningful and deliberately harder than
  representation loss, rank, or probe gains.
- Claims are usually narrow and explicitly deny production authorization.

## Experiment-family review

| Experiment family | Independence and separation | Adversarial verdict |
|---|---|---|
| Synthetic vertical slice | Test seeds differ from training; calibration is training-only, but all data come from one known generator and no independent scenario-family holdout is defined. Training explicitly includes labelled routine isolated noise, and test uses the same injection mechanism with fresh seeds. | **Material limitation.** Valid software/synthetic mechanism test and interpolation across seeds, not an adversarial test of an unseen noise mechanism. The pass does not estimate real detection. The report states this correctly ([spec, lines 62–92](../specs/vertical-slice.md); [report, lines 25–30](../../artifacts/evaluation/report.md)). |
| OTLP replay and Collector round trip | Worked examples and byte parity; no statistical sampling claim. | **Defensible.** Strong deterministic semantics test. It covers metrics/gauges in the live round trip, not production OTLP diversity ([spec, lines 67–88](../specs/otlp-replay.md); [report, lines 28–32](../../artifacts/otlp-replay/report.md)). |
| Single fault lab | Fitting and testing use intervals of one run; one noise point, one worker stall, one topology. | **Fatal for efficacy; defensible for plumbing.** The report openly calls it development evidence ([report, lines 43–51](../../artifacts/fault-lab/report.md)). |
| First held-out 3-fault matrix | Three fresh runs and frozen artifacts; only one case per fault, same team/topology, correlated point-rate gate. | **Valid negative transfer diagnostic.** It exposed 87/108 pre-noise and 18/21 noise alerts, making recall operationally irrelevant ([report, lines 5–12 and 56–68](../../artifacts/fault-matrix/report.md)). |
| Demand-conditioned v2 three-case confirmation | Protocol and model were frozen; three fresh cases. Rates pool windows, and one case can swing 3/3 recall/attribution. | **Materially underpowered positive.** The pass was legitimate under its frozen gates but never sufficient for deployment; the spec says three cases do not establish production diversity ([spec, lines 82–105](../specs/demand-conditioned-v2.md)). |
| Expanded 3×3 fault/topology matrix | Frozen model over nine fresh cells, one run per cell. Fault and topology are crossed, but schedules were initially confounded with topology. | **Strong falsification of dependable calibration.** It retained 9/9 detection while producing 57.3% pre-noise and 54.0% routine-noise alerts ([verification](../../artifacts/demand-conditioned-v2/expanded-confirmation/verification.json)). |
| Matched-topology diagnostic | Three workload/fault blocks; non-topology fields fixed, order counterbalanced but not randomized. | **Defensible diagnosis, low power.** It refuted the claimed large monotone topology effect, not schedule sensitivity. Three blocks cannot establish equivalence; the ±20-point classification is an engineering indifference zone, not a powered equivalence test ([spec, lines 19–53 and 75–80](../specs/matched-topology-diagnostic.md)). |
| Metrics-only JEPA v0 and pointwise metrics+logs | Run/family isolation and deterministic artifacts, but architecture and data were open development; the pointwise multimodal version learned demand. | **Useful failure discovery, not confirmation.** Schedule-family shortcuts were real and correctly motivated explicit controls ([corpus spec, lines 55–90](../specs/jepa-corpus-v1.md)). |
| Contextual multimodal development v1 | Eight leave-one-family-out development folds; already-exposed validation shown only diagnostically. | **Methodologically candid.** The selected 0.038-point advantage is a selection statistic, not generalization, and the result says so ([result, lines 15–41](contextual-multimodal-jepa-v1-results.md)). |
| Contextual multimodal confirmation v2 | Twelve untouched validation schedule families, five fixed seeds averaged within family, exact paired sign-randomization; no post-collection model selection. | **Strongest scientific experiment in the repo.** The unit and seed dependence are handled correctly ([spec, lines 150–180](../specs/contextual-multimodal-jepa-confirmation-v2.md)). Its negative conclusion is credible for fault-free schedule-family transfer in this stack, not for interventions or other logs. |
| Linear/adaptive graph observability pilot | Reuses the inspected 72-run corpus and adaptively selects per-entity widths on training families. | **Valid routing experiment only.** The reported 1.7% one-hop gain and compression justify a fresh hypothesis, not a topology claim ([result, lines 63–105](graph-jepa-observability-pilot-v1-results.md)). |
| Learned hybrid graph JEPA | 36 train/36 family-held runs, three seeds, matched PCA/raw/shuffled/no-event controls; corpus already open and no interventions. | **Credible bounded negative.** The tiny probe gain is real, but topology/event effects are negligible, one entity collapses, and recovery fails ([result, lines 66–115](hybrid-telemetry-jepa-development-v1-results.md)). |
| Action instrumentation and 120-pair corpus | Full 5 action × 3 topology × 8 replicate design, matched twins, fresh IDs, no retries/deletions, six isolated lanes. | **Strong corpus engineering.** Remaining threats are sequential twins, lane/batch interference, one host/stack, and no block-effect analysis. All 24 qualification gates passed ([result, lines 36–55](action-dynamics-development-v1-results.md)). |
| Graph/dense action-conditioned VARX | 90 fit and 30 validation pairs, two replicates/cell in validation, fixed ridge. | **Valid open-development falsification.** Dense action dynamics clearly beat persistence; graph-only rollout was unstable. No interval or sealed confirmation supports a world-model claim ([result, lines 57–77 and 129–205](action-dynamics-development-v1-results.md)). |
| Edge model tournament | 60/15/15/30 pair roles; clean role separation, but the 30-pair evaluation had informed redesign. | **Good model selection, not confirmation.** The graph residual's 0.0054% selection win reversed on evaluation. Rank-32's equality with dense is promising but requires fresh confirmation ([result, lines 17–30 and 76–89](edge-dynamics-development-v1-results.md)). |
| Echo-state and temporal convolution | Same clean roles and references; small frozen configuration grids. | **Recipe-specific rejection only.** The search budgets are too narrow to reject reservoirs or TCNs generally. |
| Structured-event ablation and Count-Min Sketch | Same windows; only three event templates. | **Defensible negative within vocabulary.** It says nothing about natural-language or high-cardinality logs; the report states this ([result, lines 91–105 and 119–127](edge-dynamics-development-v1-results.md)). |
| Conformal/sequential detector | Calibration is role-separated; point residuals are overlapping, trajectory is the operational outcome. | **Not operationally calibrated.** Point detection gives 40% control-trajectory alarms; sequential scoring trades that for 60% detection and long delay ([result, lines 107–117](edge-dynamics-development-v1-results.md)). |
| Action-conditioned latent JEPA | Pair-atomic topology split, raw/supervised controls, one seed, selection/calibration separation. | **Credible exact-recipe rejection.** It lost badly to raw low-rank; no seed or sealed step was warranted for deployment triage. It cannot reject action-conditioned JEPA architectures generally. |
| Residual JEPA correction | Raw bypass, zero-gain identity, selection-only gain, one deterministic CPU seed. | **Strong conservative design.** Failure supports “no added value for this residual recipe,” not absence of any useful auxiliary latent objective ([spec, lines 62–90 and 110–147](../specs/residual-jepa-correction-development-v1.md)). |

## The 23 comparable JEPA tracers

All 23 inherit the same central limitations: open adaptive transfer data,
single-seed fitting, ten held-topology pairs, no production incident sample,
and no fresh sealed confirmation. The table below focuses on what is specific
to each experiment.

| Tracer | Controls and assessor | Adversarial judgement |
|---|---|---|
| Soft regime-codebook | Continuous JEPA, switching/codebook control, raw/PCA; no independent assessor, manifest, or bound implementation commit. | **Exploratory rejection only.** The 2.35–2.56× raw prediction regression and 100% control alarms make nondeployment clear, but the artifact is below the later evidence standard. The stored result itself labels one seed and open data ([artifact, lines 129–137](../../artifacts/action-dynamics/prototype-regime-codebook-jepa-v1/prototype-result.json)). |
| Event-native trace JEPA | Binned event model, n-gram, alignment shuffle, raw dynamics; no manifest, bound implementation commit, or separate assessor. | **Exploratory rejection only.** Strong causal-clock/leakage safeguards, and 0/10 candidate detections versus 7/10 metrics-only is a decisive local failure. But the artifact is a single JSON, the six-span grammar is nearly deterministic, and action-family marginals remain a possible shortcut. |
| Multi-hypothesis trajectory JEPA | One-component, capacity-matched Gaussian, supervised mixture, raw; corrected artifact-only assessor. | **Valid selection-stage rejection after correction.** V1 improperly allowed transfer finiteness into selection and is invalid. V2 recomputes a pair-balanced selection-only decision and does **not** claim alert/investigation failure ([result, lines 13–33 and 35–75](multi-hypothesis-jepa-prototype-v2-results.md)). Catalog must point to v2. |
| Exact SIGReg substitution | Variance/covariance and no-regularizer nulls plus raw; no standalone assessor. | **Material evidence gap.** Single-seed negative metrics are useful, but the runner invokes its own assessment before artifact writing. Reject the substitution, not SIGReg or complete LeJEPA. |
| Complete multi-view LeJEPA | Invariance-only, reconstruction, supervised latent, PCA, raw; pair-blocked independent axis and stored-array assessor. | **Strong exact-recipe falsification.** Pair-blocking (`N=40`), fixed steps/no early stopping, controls, and role isolation are excellent. A ratio gate against near-zero PCA error is pathologically strict and raises false-negative risk, though several large raw/value failures independently reject the recipe. The assessor imports producer metric/gate utilities, so it is not a mathematical reimplementation. |
| Retrieval-JEPA | Raw, PCA, deranged JEPA, CPC, supervised contrastive; independent stored-array assessor. | **Strong local rejection with a split-claim inconsistency.** Episode/pair is the unit and the spec explicitly declines an infeasible selective-risk guarantee with only 30 calibration queries ([spec, lines 193–211](../specs/retrieval-jepa-evidence-contract-v1.md)). But topology-three pairs appear in selection/calibration before topology-three evaluation, so encoder fitting transfers while the complete abstention policy is topology-calibrated—not end-to-end unseen-topology transfer. Raw/PCA 100% versus JEPA 40% is still a large closed-bank failure. |
| HEPA | Deranged horizon, supervised-from-scratch, raw effect; corrected assessor and invalid attempts retained. | **Credible recipe rejection; near-rate uncertainty remains.** A real topology leak and commit-identity defect were corrected transparently ([result, lines 26–53](hepa-jepa-telemetry-tracer-v1-results.md)). Ten transfer treatments cannot precisely separate 50% from 60%, but HEPA tying its null defeats the mechanism lane. |
| SC-JEPA | Full codebook × resolution factorial with matched deployed capacity and assessor. | **Good recipe falsifier, imperfect factor isolation.** Negligible interaction and entity-local code collapse undermine this translation. The codebook factor also changes continuous MSE to distributional KL, so “quantization caused failure” is not separately identified. |
| CF-JEPA | EMA/online role cells, zone ablations, PCA/raw, stored-array assessor. | **Credible exact-recipe rejection.** The smooth target and state probe are mechanism evidence; 10% transfer-control FPR means one of ten controls, with very wide uncertainty. Official ablations match deployed capacity but not active training parameters, partially confounding objective and optimization capacity. |
| SD-JEPA | Angular/progression coordinates, controls and stored-array assessor. | **Credible deployment rejection; mechanism is inconclusive.** Angular AUROC misses its gate by only 0.003885 with no seed/paired interval. Progression semantics being worse than the A0/content controls and zero calibrated transfer alerts are more decisive. |
| Delta-JEPA | Endpoint, no-action/deranged, supervised/raw controls; assessor reuses complete-LeJEPA utilities. | **Strong representation comparison.** Displacement contains action signal but endpoint concatenation and raw forecasting win. This rejects the bottleneck, not temporal-difference supervision generally. |
| LeWorldModel geometry screen | Ambient/subspace/rectified/kernel/spherical cells, raw/PCA and assessor. | **Screening evidence, not a geometry-family test.** Seven cells × five ridges on ten selection pairs create multiplicity and winner's-curse risk; selection of the best diagnostic cell is exploratory. No raw-safe cell is still a valid reason not to promote. |
| Causal-JEPA | Whole-entity, coordinate-mask, persistence, raw; assessor. | **Scientific and engineering outcomes must be separated.** Whole-entity and coordinate masks match token count but not completion difficulty, and observability masking is not a causal `do`-intervention. Raw transfer is over three times better. The official bundle also misses restoration tolerance by `1.07e-6`; retaining the frozen tolerance is admirable, but restoration failure alone cannot falsify the learning hypothesis ([restore record](causal-jepa-attempt-1-restore-boundary.md)). |
| MoP-JEPA | Dense JEPA, supervised WTA, static codebook, shuffled context, raw; assessor. | **Strong corpus-specific mixture rejection.** Genuine specialization is established, but supervised/codebook/raw controls win. The mostly deterministic corpus has no oracle for several valid unobserved successors, so it is weak evidence about multimodal planning. |
| PairEffect-JEPA | Supervised matched-effect and deranged-pair controls, raw bypass/composition; source-bound assessor. | **Strong local-hypothesis falsification.** Matched twins are the right target; 3.38× raw regression makes nondeployment clear. The result's “statistically indistinguishable” phrasing is unsupported without a paired test or interval; use “numerically indistinguishable.” |
| Task-grounded Contract-JEPA | Raw bypass, supervised and deranged witnesses, frozen correction bound; source-bound assessor. | **Strong no-added-value conclusion.** Exact raw preservation prevents safety theater; both controls win and transfer witness scale drifts. A 1.05% raw-effect gain is too small and adaptively exposed to outweigh 100% witness false alarms. |
| Error-Certificate-JEPA | Raw-only, deranged-JEPA, constant conformal; five retained evidence versions and source-bound assessor. | **Strongest of the local conservative falsifiers, with no finite 95% certificate.** Exact raw preservation, separate calibration, and transparent evidence repair are good. With ten calibration controls the 95th higher quantile is the maximum; the required conformal rank is unavailable and max coverage is only 10/11 under ideal exchangeability. Point coverage is not trajectory coverage ([result, lines 15–41 and 43–99](error-certificate-jepa-v1-results.md)). |
| SALT-JEPA | Frozen reconstructive teacher, deranged target, PCA/raw; v2 independently repairs v1 evidence gaps. | **Credible exact-translation rejection.** The v1 assessor trusted several producer claims; v2 fixes that ([v1 review, lines 3–13](salt-jepa-telemetry-v1-results.md)). The 8.89% versus 10% mechanism miss is too close for a family claim, but 1.91× raw effect error is decisive for deployment ([v2 result, lines 41–69](salt-jepa-telemetry-v2-results.md)). |
| LeNEPA | Unprojected, SIGReg-only, PCA/raw; isolated copied-source assessor. | **Strong exact-recipe rejection.** Mechanism and raw safety both fail under a high-quality evidence contract. One projection design and seed cannot reject disposable projections generally. |
| Discrete-JEPA | P2P-only, complementary objectives, PCA/raw; isolated assessor. | **Strong compact-adaptation rejection, not an “exact” source reproduction.** One-code-per-entity collapse is a direct mechanism failure. No author code was available and the translation scales the paper's semantic-token setup substantially, so published Discrete-JEPA remains outside the claim. |
| PEIRA | Aligned/deranged PEIRA, complete LeJEPA, reconstruction, PCA/raw; isolated source closure. | **Credible operational-translation rejection.** The alignment mechanism is active but controls/raw win. Author code was unavailable, source details were ambiguous, and replay tolerance was amended after observation; paper-to-telemetry view construction is only one adaptation. |
| VISReg | Scale/shape cells, invariance-only, prior controls, PCA/raw; isolated assessor and gradient falsifier. | **Strong exact-recipe rejection.** Passing the small-radius gradient check proves the implementation mechanism; projector rank collapse and raw regression defeat operational value. |
| JEPA-SCORE | Three frozen representation cells and raw delta; exact Jacobian/SVD, action-blind calibration, isolated literal assessor. | **Strong feasibility result, weak efficacy sample.** Exact local-density scoring fits this machine's budget, but the screen uses one fixed Monte Carlo transform from a source expectation and an already weak encoder. The frozen combination yields 10% IID and 0% transfer detection; five anchors and ten transfer pairs do not reject density scoring generally. |

## Where conclusions should be tightened

### Rock-solid

- “Do not put any tested JEPA artifact in paging.”
- “The tested alert calibration is not dependable.”
- “Direct raw/PCA retrieval is the current closed-library incumbent.”
- “The rank-32 raw transition is the best compact predictive incumbent on the
  open action corpus.”
- “Representation rank, probe quality, alignment loss, or geometry alone did
  not predict alert/investigation value.”
- “The exact tested recipes failed their frozen promotion contracts.”

### Supported only as development prioritization

- “Stop spending on another generic JEPA port using the same corpus.”
- “Prefer raw dynamics and explicit calibration for the next shadow
  experiment.”
- “The current aggregate event vocabulary has no measurable added predictive
  value.”

### Not supported

- “The rank-32 model is ready for production.”
- “Topology-three is an untouched test set.”
- “0% observed false alarms means the true rate is below 5%.”
- “Twenty-three failures are independent replications.”
- “All known JEPA techniques have been exhausted.”
- “JEPA cannot work for logging, telemetry, or this stack.”

## Prioritized remediation

### P0 — before any deployment or new model-family conclusion

1. **Freeze one shadow candidate and collect a fresh sealed corpus.** Do not
   reuse the current evaluation/transfer pairs. Bind source, container images,
   protocol, feature schema, thresholds, and stopping rules before collection.
2. **Make the run/incident the inferential unit.** Predeclare cluster bootstrap
   or exact randomization intervals over matched pairs and collection blocks.
   Never use overlapping-window counts as the effective sample size.
3. **Power the alert SLO.** Ten controls cannot validate 5% false alarms.
   Choose the number of independent control runs from a one-sided confidence
   requirement. To observe zero alarms and put a 95% upper bound below 5%
   requires at least 59 independent controls (`1 - 0.05^(1/n) < 0.05`), before
   accounting for clustering or stratification.
4. **Separate shadow-alert confirmation from action-known forecasting.**
   Production alert inputs must exclude action truth; prediction and
   closed-library investigation should remain separately reported lanes.
5. **Run the frozen workload long enough to measure alerts per service-hour and
   incident-level recall.** Current short logical trajectories do not measure
   paging burden.

### P1 — repair scientific interpretation

6. **Adopt a hierarchical evidence model.** Report paired effects by action,
   topology, seed, and collection batch with uncertainty. Partial pooling is
   preferable to one aggregate percentage over two examples per action.
7. **Predeclare a program-level hypothesis tree.** A mechanism gate should open
   a value test, which should open seed robustness, which should open sealed
   confirmation. Record alpha/error spending if inferential p-values are used.
8. **Give near-miss candidates multiple seeds or label them optimization-
   inconclusive.** Reserve one-seed rejection for overwhelming raw/safety
   regressions or direct null failures.
9. **Distinguish deployment fairness from mechanism fairness.** Keep the tuned
   raw incumbent for deployment, but add a comparable search budget when
   drawing scientific conclusions about a JEPA objective.
10. **Add missingness, upgrade, and recovery panels only when measured product
    incidence justifies them.** These should be fresh prerequisite-driven
    protocols, not more reuse of topology three.

### P2 — harden reproducibility and review

11. **Publish content-addressed artifacts.** Upload all conclusion-bearing
    bundles, source closures, and raw-corpus manifests to immutable release
    assets or object storage.
12. **Fix catalog authority.** Point multi-hypothesis to the v2 corrected
    artifact; mark v1 invalid. Mark the regime-codebook and event-native
    artifacts as prototype-grade until they have manifests and artifact-only
    reassessment.
13. **Add signed provenance.** Sign release manifests or publish them to an
    external transparency log.
14. **Create an assessor-independence tier.** Label producer summary,
    stored-array reassessment, isolated copied-source assessment, and
    independent reimplementation separately.
15. **Invite a blind external reproduction of one negative and one promising
    result.** The best choices are Error-Certificate-JEPA and rank-32 raw
    dynamics because both have direct operational claims and mature artifact
    closure.

## Final assessment

The methodology evolved substantially and in the right direction. Early
experiments are conventional prototypes; the contextual confirmation,
action-corpus protocol, and later source-bound tracer artifacts show serious
attention to leakage, controls, role separation, and reproducibility.

The adversarial reading is not “the experiments are bad.” It is:

> They are very good **development falsifiers** and increasingly good
> reproducibility artifacts, but—with one narrow contextual confirmation
> exception—they are not independent confirmatory studies.

That distinction preserves the useful result: there is no rational basis to
deploy any tested JEPA today. It also prevents the evidence from being
overstated: the next justified step is a sealed, sufficiently powered shadow
evaluation of the small raw stack, not a declaration that the JEPA frontier
has been scientifically exhausted.
