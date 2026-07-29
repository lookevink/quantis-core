# JEPA implementation ladder v1

## Purpose

Evaluate materially distinct JEPA implementations under one comparable
development contract. Each tracer must establish value in observable
prediction, alerting, or investigation rather than treating a lower
representation loss as success.

The ladder begins with a soft regime-codebook JEPA and may later cover exact
LeJEPA/SIGReg regularization, event-native trace JEPA, multi-hypothesis latent
prediction, a complete two-stage action-conditioned pipeline, cross-run
representations, retrieval, and generative latent correction.

This is an open-development protocol. It cannot confirm production alerting or
a software world-model claim.

## Evidence boundary

Tracer experiments reuse the content-addressed action-dynamics development
corpus and its topology-transfer preprocessing cache:

- fitting, selection, and calibration use worker topologies one and two;
- the primary transfer diagnostic uses previously opened topology three;
- whole matched pairs remain atomic and cannot cross roles;
- state and control normalization is fit on fitting pairs only; and
- no fresh sealed case is collected or opened for a tracer.

The existing role counts remain:

- 40 fitting pairs and 6,320 windows;
- 10 selection pairs and 1,580 windows;
- 10 calibration pairs and 1,580 windows;
- 20 in-distribution evaluation pairs and 3,160 windows; and
- 10 held-out-topology evaluation pairs and 1,580 windows.

Every artifact records the source corpus, source artifact manifest,
preprocessing protocol, pair-role, graph-schema, observation-schema, feature,
configuration, implementation-commit, and runtime identities.

## Shared public model contract

Candidate implementations expose six public seams. Tests observe these seams
and do not inspect optimizer internals, private coefficients, prototype update
steps, or framework modules.

### 1. Fitting

`fit(training_windows) -> fitted_model`

- Input is one immutable `ActionConditionedWindows` fitting role.
- Fitting may use histories, future state, controls, actions, topology, and
  matched-pair identity declared by the candidate protocol.
- The fitted model records every training-fitted normalizer, vocabulary,
  prototype, and hyperparameter.
- Selection, calibration, and evaluation data are forbidden.

### 2. Encoding

`encode(histories, graph) -> representation_batch`

The returned representation batch contains:

- entity-preserving tokens with sample and time axes;
- an observation mask for meaningful entity-feature positions;
- any optional regime or mixture assignment probabilities;
- enough schema metadata to align tokens with graph entities; and
- finite values with deterministic ordering.

Assignment probabilities, when present, sum to one along their component axis.
The representation never exposes future observations or action truth through
the context encoding seam.

### 3. Predictive distribution

`predict_distribution(histories, future_controls, future_actions, graph) -> trajectory_distribution`

- The common output is the existing `TrajectoryDistribution` in normalized
  observable-state coordinates. A family-specific preregistration may instead
  return an additive probabilistic type, such as the
  `MixtureTrajectoryDistribution` frozen by the
  [multi-hypothesis scoring contract](multi-hypothesis-jepa-scoring-contract-v1.md),
  when it exposes an exact moment-matched `TrajectoryDistribution`
  compatibility view.
- Mean and diagonal variance align with sample, horizon, entity, and feature
  axes.
- A richer distribution must be assessed in its full form by preregistered
  proper scores. Its compatibility view is only for shared point metrics.
- Candidates that are not intended to replace the predictive core may expose
  a frozen-probe or adapter predictor through this seam; the assessment labels
  that role explicitly.
- Hidden-action alert evaluation supplies the no-action tensor rather than
  intervention truth.

An adapter may also expose this seam as the existing
`EdgeDynamicsModel.rollout` method. The two paths must return identical values.

### 4. Serialization and restoration

`to_dict()` returns a JSON-safe configuration and fitted-state description.
Large numeric arrays may be content-addressed sidecars. Restoration uses the
serialized `kind` and rejects:

- unknown versions;
- incompatible graph or observation schemas;
- missing or mismatched sidecars; and
- non-finite fitted state.

A restored model must reproduce encoding, predictive means and variances, and
optional assignments within a candidate's declared deterministic tolerance.
Any exact-repeat claim requires byte-identical evidence on the frozen runtime.

### 5. Pure assessment

The assessor consumes stored inputs, predictions, representations, attribution
queries, timings, and configuration identities. It must not call the fitted
model or trust stored metric summaries or gate booleans.

It independently recomputes:

- common predictive, representation, alert-policy, investigation, and runtime
  metrics;
- candidate-specific diagnostics;
- paired comparisons;
- gate outcomes; and
- the bounded decision.

### 6. Immutable artifact bundle

The runner refuses to overwrite a nonempty output directory. A completed
bundle contains at least:

- `protocol.json`;
- `data-identity.json`;
- one or more fitted model artifacts;
- stored representations and predictive distributions;
- stored attribution and alert-policy inputs;
- `assessment.json`;
- `report.md`; and
- `artifact-manifest.json` with SHA-256 for every evidence-bearing file.

Incomplete work remains in a staging directory with a machine-readable failure
record and cannot be interpreted as a result.

## Shared baselines

Every tracer declares its role before fitting and compares against the
smallest sufficient baseline set:

1. **Predictive reference:** the frozen rank-32 raw-state contractive low-rank
   transition.
2. **Representation reference:** training-only frozen PCA at the candidate's
   effective representation budget.
3. **Capacity control:** a capacity-matched supervised model without the
   JEPA-specific objective.
4. **Family-specific classical control:** for the regime-codebook tracer, a
   training-only switching-regime or hidden-state model.
5. **JEPA null:** remove or break the mechanism that distinguishes the
   candidate, such as the codebook, SIGReg term, event alignment, action
   conditioning, retrieval index, or stochastic component.

No baseline or candidate may be selected using either evaluation role.

## Shared metrics

### Observable prediction

Report in-distribution and held-out-topology:

- normalized overall and action-overlap MSE;
- paired treatment-minus-control downstream-effect MSE;
- preregistered proper scores and calibration diagnostics for candidates that
  expose a richer predictive distribution;
- finite rollout and maximum ten-step norm growth; and
- per-action and per-topology breakdowns.

### Observable-state retention

Fit every probe on fitting representations only and report:

- aggregate and per-observed-entity frozen-probe NRMSE;
- effective rank and variance by observed entity;
- comparison with representation-budget-matched PCA; and
- any candidate-specific component usage, entropy, or calibration.

An entity with no varying owned observation is reported separately and cannot
silently make the all-entity statistic pass or fail.

### Investigation utility

Using the frozen candidate library and stored query tensors, report:

- action-and-target hit@1;
- no-action specificity;
- per-action hit@1;
- correct-action versus no-action and whole-pair-deranged-action sanity; and
- abstention or uncertainty coverage when supported.

These scores apply only to the closed action library.

### Alert-policy utility

Action truth is hidden. Thresholds use calibration control trajectories only.
The matched trajectory, not an overlapping window, is the gate unit.

Report:

- control-trajectory false-alarm rate;
- treatment-trajectory post-onset detection;
- pre-onset treatment alerts;
- median and worst detection delay; and
- alerts per logical run in addition to point-level diagnostics.

### Edge feasibility

Report:

- inference parameter count;
- serialized model and sidecar size;
- batch-one CPU latency;
- peak resident memory when the runner supports it;
- finite output under missing or masked observations declared by the
  candidate; and
- exact target-runtime measurements only when executed on that runtime.

Local Python or PyTorch timings remain microbenchmarks, not portable latency
claims.

## Common gates

All candidates must pass:

1. training-only fitting and identity binding;
2. deterministic serialization/restoration;
3. finite representations, assignments, predictions, and assessments;
4. no future, role, topology, or action-truth leakage;
5. observable-state retention no worse than the candidate's preregistered
   safety margin; and
6. the family-specific JEPA null comparison.

A candidate then earns value through at least one declared lane. A
family-specific contract may add a lane based on a proper score when the
shared point-prediction lanes cannot measure the proposed value.

### Predictive-core lane

- downstream-effect MSE improves by at least 10% over raw low-rank;
- action-overlap and overall MSE remain within 5% of raw low-rank;
- action-and-target hit@1 is at least 95%;
- no-action specificity is 100%; and
- correct action beats both action ablations on at least 80% of treatment
  pairs.

### Alert-policy lane

- control-trajectory false alarms are at most 5%;
- treatment-trajectory detection is at least 80%;
- median post-onset delay is at most 10 transitions; and
- the candidate improves a preregistered sensitivity/delay measure over its
  JEPA null at the same trajectory false-alarm budget.

### Investigation lane

- action-and-target hit@1 is at least 95%;
- no-action specificity is 100%;
- correct action beats both action ablations on at least 80% of treatment
  pairs;
- observed-entity state retention is no worse than matched PCA by the
  preregistered margin; and
- the candidate improves a preregistered investigation measure over its JEPA
  null.

Passing one lane in a single-seed tracer authorizes fixed multi-seed
robustness. It does not authorize sealed collection. Every fixed seed must pass
the same value and safety gates before the recipe becomes a promotion
candidate.

## Candidate-specific preregistration

Before its first result directory exists, each tracer specification freezes:

- scientific hypothesis and JEPA-specific null;
- model role and value lane;
- data and split identities;
- architecture and parameter budget;
- fitting seed and runtime;
- family-specific diagnostics and safety margin;
- selection rule, including a safe null option;
- common and candidate-specific gates; and
- permitted positive and negative claims.

Changing any of these after observing a result creates a new version and
preserves the earlier outcome.

## TDD boundary

Tests are written one vertical slice at a time through the approved seams:

1. fitting consumes only the declared role;
2. encoding preserves entity/schema identity and optional probability
   invariants;
3. predictive distributions align with observable state, preserve any
   declared mixture invariants, and reproduce their compatibility moments;
4. serialization/restoration preserves public outputs;
5. the assessor recomputes outcomes from stored evidence; and
6. artifact creation is non-overwriting and content-addressed.

Tests do not assert private tensor layouts, optimizer steps, or implementation
class composition.

## Promotion boundary

Open development may reject a configuration or identify a promotion candidate.
Only a separately preregistered fresh corpus can confirm the latter. Production
paging additionally requires target-runtime parity and a sustained shadow
evaluation with an operator-level alert budget.
