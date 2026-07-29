# Quantis Experiment Language

Quantis evaluates compact predictive representations and alerting policies for
software telemetry. These terms keep model evidence distinct from operational
claims.

## Language

**Predictive core**:
A model that forecasts observable telemetry state or intervention effects.
_Avoid_: Alerting model, world model

**Alert policy**:
A rule that converts a sequence of model scores into operator-visible warning
events under a declared false-alarm budget.
_Avoid_: Threshold, detector

**Investigation model**:
A model invoked after a warning to rank known interventions, affected
subsystems, or historical precedents.
_Avoid_: Root-cause detector

**Tracer experiment**:
An open-development run that tests whether one fixed implementation earns
broader robustness work.
_Avoid_: Confirmation, pilot

**Promotion candidate**:
A fixed implementation that passes every tracer and multi-seed gate and is
eligible for a fresh sealed confirmation.
_Avoid_: Production model, validated model

**Sealed confirmation**:
A preregistered evaluation whose fitting, selection, and calibration cannot
observe the fresh confirmation cases.
_Avoid_: Test run, validation run

**Observable-state retention**:
The extent to which a representation preserves subsystem state required for
forecasting and attribution.
_Avoid_: Reconstruction quality

**Representation candidate**:
An encoder evaluated for transferable operational-state retention through
frozen downstream adapters. Passing authorizes a later predictive stage; it is
not itself a predictive core or alert policy.
_Avoid_: World model, predictive core, alerting model

**Pair-blocked anchor batch**:
A representation-training batch containing one context anchor from each
matched pair, with trajectory arm and transition balanced across steps. Its
views share an anchor; overlapping windows and matched arms are not counted as
independent samples.
_Avoid_: Window batch, augmented batch

**Telemetry view**:
A partial observation of one context anchor produced by identity-preserving
temporal, topology, or owned-coordinate masking. Multiple views describe one
sample and never enlarge the independent sample axis.
_Avoid_: Independent sample, synthetic trajectory

**Action-conditioned representation probe**:
A fit-only reduced-rank linear adapter that measures whether intervention
effects are accessible from a frozen representation. It is an evaluation
instrument, not a deployable transition model.
_Avoid_: Predictive core, latent dynamics model

**Forecast hypothesis**:
An exchangeable, weighted component of one calibrated predictive mixture,
representing a distinct plausible observable trajectory.
_Avoid_: Scenario, branch, named regime

**Predictive mixture**:
The probability distribution formed by all forecast hypotheses and their
normalized weights for one forecast.
_Avoid_: Candidate set, best-of-K forecast

**Forecast ambiguity**:
The condition in which several materially distinct forecast hypotheses retain
meaningful probability for one forecast. It is uncertainty, not evidence of
an anomaly by itself.
_Avoid_: Alert, incident evidence, model failure

**Mixture surprise**:
The degree to which an observed trajectory lacks support from the complete
predictive mixture. It is the forecast evidence used by uncertainty-aware
alerting.
_Avoid_: Hypothesis disagreement, branch error

**Unattributed alert**:
An alert supported by sufficient mixture surprise but not by enough evidence
to name one intervention from the closed investigation library.
_Avoid_: False alert, unknown anomaly

**Proper trajectory score**:
A score whose expected value is optimized by reporting the complete predictive
mixture believed to be true. For multi-hypothesis experiments, exact mixture
log score is the primary selection metric and multivariate energy score is a
required safety check.
_Avoid_: Best-branch error, oracle error, best-of-K error

**Retrieval episode**:
One independently generated trajectory observed at a declared query time,
together with its later raw evidence slice and immutable source reference.
Matched arms remain one calibration or assessment unit even when both are
queried.
_Avoid_: Window, sample, incident

**Evidence bank**:
An immutable, fit-role collection of independently generated retrieval
episodes whose raw telemetry and source references accompany their frozen
search vectors.
_Avoid_: Training set, memory, knowledge base

**Episode-predictive retriever**:
A representation candidate whose deployed query path predicts the latent
space of withheld episode evidence, then searches an evidence bank. Its value
claim is historical-evidence retrieval, not future-trajectory prediction.
_Avoid_: Root-cause detector, world model

**Empirical retrieval abstention**:
A calibration-role confidence rule that withholds a retrieved attribution
when evidence similarity is insufficient. Without enough independent
calibration episodes for a finite-sample bound, it is an empirical tracer
result rather than guaranteed selective risk.
_Avoid_: Calibrated probability, certified rejection, safe fallback

**Collection campaign**:
A preregistered set of fresh raw captures collected under one frozen
application, deployment, instrumentation, workload, and reset protocol.
Repeated runs and matched pairs within a campaign improve within-environment
precision but do not create new stack environments.
_Avoid_: Domain, stack

**Stack environment**:
A preregistered data-generating environment whose logical application stack
differs in service implementation, dependency or protocol boundaries, and
their canonical telemetry mapping. Replica counts, restarts, seeds, build
hashes, and derived feature schemas do not by themselves create a new stack.
_Avoid_: Run, topology, deployment instance

**Derived corpus**:
A feature table, event cache, replay, window collection, or other
transformation whose records ultimately refer to an existing raw collection
campaign. A derived corpus inherits the source campaign's evidence role and
never adds independent runs or environments.
_Avoid_: New corpus, independent environment

**Cross-stack identifiability**:
The ability of a role-clean experiment to distinguish a representation that
transfers across a declared family of stack environments from one that exploits
source-stack, schema, workload, or intervention shortcuts. It is always bounded
to the preregistered family and is not universal portability.
_Avoid_: Domain invariance, production generalization
