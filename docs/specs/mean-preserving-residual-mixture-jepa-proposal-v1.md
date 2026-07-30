# Mean-preserving residual mixture JEPA executable contract v1

**Status:** Complete; exact recipe rejected.

**Recommendation:** Do not advance this recipe. The valid fresh selection run
and independent assessor rejected it. See the
[result](../research/mean-preserving-residual-mixture-jepa-v1-results.md).

## Executive summary

The richer-regime campaign produced an unfavorable stored diagnostic for the
four-component JEPA, but its replacement selection corpus was not
execution-bound to the recollection amendment. That diagnostic cannot accept
or reject the recipe. It nevertheless identified a useful design constraint:
the candidate's mixture mean was substantially worse than the raw rank-32
predictive core.

The next target should not simply rerun that architecture. It should test a
**mean-preserving residual mixture JEPA**, abbreviated **MPRM-JEPA**:

1. fit the existing raw rank-32 predictive core as a strong anchor;
2. predict four weighted residual forecast hypotheses with a JEPA objective;
3. center the residual hypotheses so their weighted mean is exactly zero; and
4. evaluate the resulting complete predictive mixture on a fresh,
   preregistered selection campaign.

For anchor forecast `μ₀`, component residuals `δₖ`, and normalized weights
`wₖ`, define:

```text
δ̄ = Σₖ wₖ δₖ
μₖ = μ₀ + δₖ - δ̄
Σₖ wₖ μₖ = μ₀
```

The candidate's moment-matched mean is therefore exactly the raw anchor.
Overall and action-overlap point MSE cannot regress except for declared
floating-point tolerance. The experiment asks the narrower, more valuable
question: can a JEPA residual mixture improve proper trajectory score and
represent useful forecast ambiguity without sacrificing the strongest compact
point forecast?

## Problem Statement

We are bullish on multi-hypothesis prediction for telemetry because one
operating context can admit several plausible observable futures. The current
evidence does not tell us whether that mechanism is useful in richer local
operating regimes:

- the original narrow-corpus four-component JEPA was validly rejected;
- the richer-regime fit screen justified trying a multi-hypothesis model;
- the richer-regime stored selection diagnostic was unfavorable;
- all 135 retained richer-regime pairs pass the applicable action-specific
  effect and recovery gates; but
- the replacement selection collection did not bind its recollection
  authority into manifests and attestation, making its model decision
  inadmissible.

Repeating the old model would answer the provenance question but would not
address its largest technical failure. Tuning against the exposed replacement
selection data would create a new leakage problem. We need one structurally
motivated candidate, frozen before fitting, that uses valid fit-role evidence
and receives genuinely fresh selection evidence.

## Solution

Build and test one MPRM-JEPA recipe on the existing qualified v1 fit corpus.
The recipe inherits the original multi-hypothesis JEPA encoder, target encoder,
predictor objective, component count, seed, epoch count, and compact edge
envelope. Its substantive change is the mean-preserving residual
parameterization around the raw rank-32 anchor.

The executable source of truth is
[`mprm-jepa-protocol-v1.json`](../../lab/action_dynamics/mprm-jepa-protocol-v1.json).
It pins every recipe value, generator identity, score, randomization test,
numeric tolerance, runtime, and evidence boundary. This document explains that
protocol; where prose and JSON differ, execution must stop rather than infer.

The experiment has four hard boundaries:

1. **No reuse of exposed selection data.** The v1 failed attempt and v2
   replacement selection corpus remain diagnostic-only and are forbidden from
   fitting, normalization, model choice, threshold choice, or promotion.
2. **No sweep.** One recipe, one seed, one component count, and one set of
   inherited loss weights are frozen before model fitting.
3. **No scoring before qualification.** Fresh selection captures must
   completely pass identity, balance, telemetry, action-effect, recovery, and
   provenance checks before any model sees them.
4. **No amendment after failure.** An incomplete collection is retained and
   invalid. Continuing requires a newly versioned protocol and fresh opaque
   identities; no capture is silently replaced.

The fresh selection campaign crosses five reversible action kinds, three
worker topologies, and three workload families with two matched-pair
replicates per cell:

```text
5 actions × 3 topologies × 3 workload families × 2 replicates
= 90 matched pairs
= 180 captures
```

The fresh campaign reuses the exact v1 workload materialization semantics,
including action-specific API count and Redis drain/probe constraints. It does
not redefine workload families after fitting. Family-level interpretation must
therefore acknowledge regions where an action-specific constraint makes
schedules intentionally identical.

## Hypothesis

Conditional residual futures contain useful weighted alternatives that the
raw anchor mean does not express. A mean-preserving JEPA residual mixture will:

- preserve raw-anchor point prediction exactly;
- improve complete-trajectory mixture log score over the raw anchor and every
  fitted probabilistic control;
- retain at least two materially separated, non-negligible forecast
  hypotheses on a meaningful fraction of action-overlap samples; and
- remain compact and fast enough for the existing edge execution envelope.

The hypothesis is falsified if proper score does not improve. Distinct
components alone are not success.

## User Stories

1. As an alerting-system researcher, I want a multi-hypothesis candidate that
   cannot worsen the strongest point forecast, so that forecast ambiguity is
   tested without paying an avoidable accuracy penalty.
2. As an operator, I want forecast hypotheses to describe complete observable
   trajectories, so that ambiguity is not confused with an alert.
3. As an operator, I want the predictive mixture's mean to match the trusted
   raw predictive core, so that adding uncertainty does not destabilize
   ordinary forecasts.
4. As an experiment reviewer, I want the candidate recipe frozen before
   fitting, so that exposed diagnostics cannot tune the result.
5. As an experiment reviewer, I want old fit evidence and fresh selection
   evidence to have immutable, disjoint roles, so that model choice cannot
   leak across the boundary.
6. As an experiment reviewer, I want every fresh capture bound to the exact
   collection protocol, plan, action protocol, model-freeze manifest, image
   digests, schema, and build context, so that the assessed corpus has one
   verifiable identity.
7. As an experiment reviewer, I want the attestation to repeat those bindings,
   so that a directory label cannot substitute for execution provenance.
8. As an experiment reviewer, I want exact treatment/control twins and
   complete action×topology×workload balance, so that attrition cannot change
   the estimand.
9. As an experiment reviewer, I want action-specific effect and recovery rules
   recomputed from raw telemetry, so that generic validation semantics cannot
   misclassify pairs.
10. As an experiment reviewer, I want model scoring blocked until all three
    workload shards qualify, so that partial outcomes cannot influence retry
    decisions.
11. As a model developer, I want one deterministic numeric canonicalization
    for all mixture weights, so that float32 transport cannot invalidate one
    control selectively.
12. As a model developer, I want the supervised residual mixture and
    one-component residual model to receive the same anchor and output
    canonicalization, so that JEPA value is isolated fairly.
13. As a model developer, I want the original unanchored mixture retained as a
    diagnostic control, so that the structural benefit of mean preservation
    is measurable.
14. As a statistician, I want scores balanced first by trajectory and matched
    pair, so that overlapping windows are not treated as independent samples.
15. As a statistician, I want both a minimum effect size and a paired
    randomization test, so that a tiny but statistically convenient win cannot
    promote the candidate.
16. As a statistician, I want family-level non-inferiority gates, so that one
    demand family cannot hide a severe regression in another.
17. As an edge-system owner, I want serialized size and batch-one latency
    gates, so that a scientifically useful candidate remains locally
    deployable.
18. As a maintainer, I want every failed or superseded attempt retained in a
    fresh directory, so that reproduction code and operational evidence are
    never deleted.
19. As a maintainer, I want a pure stored-artifact assessor that requires
    externally pinned hashes, so that the reported decision is independently
    reproducible.
20. As a decision maker, I want a binary conclusion at selection—advance the
    exact recipe or reject it—so that calibration and evaluation are not used
    to rescue a failed predictive core.

## Implementation Decisions

### Candidate recipe

- The raw rank-32 predictive core is fitted on the retained, protocol-aware
  qualified v1 fit role and then frozen.
- The MPRM-JEPA uses four exchangeable forecast hypotheses.
- The JEPA encoder, target encoder, predictor widths, objective weights,
  optimizer, seed, and epoch count are the exact values in the executable
  protocol: seed `307`, 40 epochs, batch size 256, AdamW at `1e-3`, weight
  decay `1e-4`, 12-wide state latent, 16-wide context, and 128-wide predictor.
  There is no selection-time hyperparameter sweep.
- The predictor emits residual component means, positive diagonal component
  variances, and component logits.
- Component weights use one shared canonicalization for every mixture model:
  convert to float64, project onto the probability simplex with a component
  lower bound of `1e-9`, force the final component to the exact remaining
  mass, and validate before serialization.
- Residual means are weight-centered before being added to the raw anchor.
  The stored artifact verifies the weighted residual sum and moment-matched
  mean identity within `1e-10`.
- The representation objective remains JEPA. Exact mixture likelihood is the
  proper output score, not an oracle best-component loss.
- Component repulsion, uniform-usage rewards, named component semantics, and
  best-of-K selection remain forbidden.

### Required controls

Every fitted control uses the same fit windows, normalizer, raw anchor where
applicable, selection campaign, and scoring code:

1. raw rank-32 predictive core;
2. one-component anchored JEPA residual distribution;
3. supervised four-component mean-preserving residual mixture;
4. capacity-matched anchored single Gaussian;
5. original unanchored four-component JEPA, diagnostic only; and
6. MPRM-JEPA candidate.

The invalid supervised-null transport behavior from the previous run must be
covered by a regression test before fitting begins.

### Model-freeze order

1. Commit the frozen candidate and scoring protocol.
2. Fit the anchor, candidate, and controls on v1 fit evidence only.
3. Restore every model in a fresh process and verify prediction parity on a
   fit-only fixture.
4. Produce a content-addressed model-freeze manifest.
5. Freeze the fresh selection collection protocol with the model-freeze
   manifest hash embedded.
6. Collect all fresh selection captures without loading any model.
7. Qualify the complete collection.
8. Only then run the pure selection assessor.

### Fresh selection identity

- Use generator namespace `quantis:mprm-jepa:selection:v1` and seed
  `26072931`, unrelated to v1 pair identities.
- Use two fresh matched pairs per action×topology×workload cell.
- Treatment and control twins share the exact request schedule, workload seed,
  intervention seed, topology, and reset boundary.
- Every manifest binds:
  - collection protocol hash;
  - deterministic plan hash;
  - candidate/scoring protocol hash;
  - model-freeze manifest hash;
  - action protocol hash;
  - observation-schema hash;
  - application build-context hash;
  - image digests; and
  - attempt identity.
- The collection attestation repeats every campaign-level binding and
  content-addresses each prepared and captured manifest.
- The final source manifest hashes every protocol, plan, manifest, attestation,
  schema, log, trace, metric, action stream, and runner output consumed.

### Failure and attrition policy

- Automatic and pair-level retries are disabled.
- All 90 planned pairs are required.
- Every family requires exactly 30 unique pairs and 60 captures.
- Every pair requires exactly one treatment and one control.
- Any missing, duplicate, drifted, ineffective, nonrecovering, or operationally
  failed capture makes the campaign ineligible for model scoring.
- A failed campaign remains immutable. A new attempt requires a new protocol
  version and new opaque pair/case identities.
- The assessor refuses incomplete evidence rather than applying post-hoc
  attrition.

### Selection scores

All window values are averaged within logical trajectory, then within matched
pair. Matched pairs are the independent selection units.

The candidate passes only if every gate holds:

1. all protocol, model, source, and artifact identities verify;
2. all 90 pairs pass protocol-aware action realization and recovery;
3. the candidate's mixture log score improves over the raw anchor by at least
   `0.01` nats per observed coordinate;
4. it improves over both the one-component anchored JEPA and supervised
   anchored mixture by at least `0.01`;
5. it improves over the capacity-matched anchored single Gaussian by at least
   `0.01`;
6. its overall and action-overlap moment-mean MSE match the raw anchor within
   `0.1%`, with direct mean identity within `1e-10`;
7. its energy score is no worse than the raw anchor by more than `1%`;
8. at least `20%` of action-overlap samples contain two components with weight
   at least `0.10` and standardized separation at least `1.0`;
9. no workload family regresses raw-anchor log score by more than `0.01`;
10. a one-sided pair-level sign-randomization test for candidate-minus-raw
    pair-balanced log score has `p ≤ 0.05`, using 99,999 Monte Carlo draws,
    seed `26072932`, inclusive lower-tail comparison, and add-one correction;
    and
11. all restored outputs and scores are finite.

Failing any gate rejects this exact recipe. Passing authorizes a separate
calibration/evaluation proposal; it does not authorize production paging.

### Conditional multimodality diagnostic

The prior pooled two-cluster SSE statistic is retired. Diagnostic mechanism
analysis must:

- hold out complete trajectories;
- stratify by action, topology, and workload family;
- preserve matched-pair grouping;
- compare against a one-component residual permutation/bootstrap null; and
- report uncertainty over independent trajectories rather than windows.

This diagnostic explains a result but cannot override the proper-score
selection gates.

### Edge envelope

The candidate must remain:

- at or below 4 MiB serialized;
- at or below 5 ms batch-one p95 latency on Apple M1 Max / arm64, Python
  3.9.6, PyTorch 2.5.1, one Torch thread;
- free of network or accelerator dependencies at inference; and
- restorable with prediction parity in a fresh process.

These gates establish edge feasibility, not operational alert value.

## Testing Decisions

The primary test seam is one public **selection-campaign qualification**
interface. It receives the frozen protocols, model-freeze identity, prepared
plan/manifests, captured evidence, and attestation. It either returns a
content-addressed qualified corpus or rejects before any model scoring. No
lower-level flag may bypass this seam.

Tests exercise externally observable behavior:

- deterministic plan generation and fresh identity separation from v1;
- exact two-pair coverage for every action×topology×workload cell;
- exact treatment/control balance and schedule identity;
- rejection of missing, duplicate, extra, or cross-family pairs;
- rejection of any protocol, recipe, model, schema, build, image, or attempt
  hash drift;
- rejection when the attestation omits any required binding;
- protocol-aware action effect, recovery, API count, enqueue drain/probe, and
  mechanistic gates;
- rejection of partial evidence before model code can load it;
- rejection of any attempt to read v1/v2 selection artifacts as input;
- model-freeze restoration and prediction parity;
- mixture-weight floor behavior under float32 boundary values;
- component-permutation score invariance;
- algebraic mean preservation and MSE parity;
- pair/trajectory-balanced score recomputation;
- deterministic paired randomization inference;
- independent assessor rejection of source or result tampering; and
- refusal to overwrite an existing output directory.

The prior richer-regime plan, collector failure-output regression, public
action-protocol assessment, experiment-catalog, and artifact-manifest tests
are the closest existing patterns.

## Execution Plan

### Phase 0 — Freeze

- **Complete.** The executable JSON protocol freezes the one candidate recipe,
  controls, numeric canonicalization, gates, fresh generator seed, exact
  randomization test, runtime, and zero-retry policy.
- The protocol commit and content hash are recorded before fitting.

### Phase 1 — Fit and freeze models

- Load only the qualified v1 fit role.
- Fit the raw anchor, MPRM-JEPA, and controls.
- Restore and verify them in a fresh process.
- Write an immutable model-freeze artifact.

### Phase 2 — Prepare fresh selection

- Generate 90 new matched-pair identities and 180 manifests.
- Verify complete factorial balance and every required binding.
- Prepare only; do not collect until the plan assessment qualifies.

### Phase 3 — Collect and qualify

- Run the existing local Docker Compose stack.
- Retain every raw capture and runner output.
- Recompute all protocol-aware data-quality and provenance gates.
- Stop without model scoring if any gate fails.

### Phase 4 — Assess

- Load only the qualified, content-addressed fresh selection corpus.
- Restore the frozen model artifact.
- Compute every candidate/control score and paired inference in a fresh
  process.
- Have an independent stored-artifact assessor reproduce the decision.

### Phase 5 — Conclude

- **Pass:** freeze MPRM-JEPA as a promotion candidate for a separate
  calibration/evaluation design.
- **Fail:** reject this exact anchored residual-mixture recipe while retaining
  the raw predictive core and all mechanism diagnostics.
- **Operational invalidity:** retain the attempt, make no model claim, and
  require a newly versioned protocol for another collection.

## Out of Scope

- Calibration or alert-threshold fitting.
- Evaluation-role or sealed-confirmation collection.
- Production paging authorization.
- Cross-stack generalization.
- Online adaptation or continual learning.
- Component naming or causal interpretation.
- Architecture, component-count, seed, loss-weight, or threshold sweeps.
- Training on either exposed richer-regime selection attempt.
- Post-hoc pair exclusion or amendment-based recollection.
- Artifact publication or bucket configuration.

## Risks and Mitigations

- **The raw anchor may already dominate proper score.** This is the correct
  null. The candidate must add probability value, not merely ambiguity.
- **Mean preservation may encourage duplicate residual modes.** The supported
  pair gate and supervised-mixture control prevent diversity-only promotion.
- **Two fresh replicates per cell may still be noisy.** Pair-balanced inference,
  a minimum effect size, family gates, and 90 independent pairs limit
  window-level pseudoreplication.
- **Collection failure is expensive under zero retry.** Prepare-only
  validation, retained runner output, and a full local smoke pass occur before
  the scientific campaign.
- **The proposal is informed by exposed diagnostics.** The structural change
  and all thresholds are frozen; old selection tensors remain forbidden
  from fitting and tuning, and all model choice occurs on fresh selection.

## Further Notes

This proposal deliberately keeps the stack small. It changes the predictive
mixture and the evidence contract, not the application topology.

The design is novel to this repository: the raw predictive core owns the
mixture mean, while JEPA owns only weighted residual alternatives. That division
matches the evidence so far—the compact raw model is difficult to beat on
point prediction, while multi-hypothesis JEPA reliably produces distinct
alternatives. MPRM-JEPA makes those strengths complementary instead of forcing
one model to replace the other.

Primary internal references:

- [Richer-regime retry results](../research/richer-regime-alerting-retry-v1-results.md)
- [Richer-regime v1 contract](richer-regime-alerting-retry-v1.md)
- [Multi-hypothesis JEPA scoring contract](multi-hypothesis-jepa-scoring-contract-v1.md)
- [Multi-hypothesis JEPA corrected result](../research/multi-hypothesis-jepa-prototype-v2-results.md)

## Explicit scoring-contract supersessions

This selection-only tracer deliberately supersedes three clauses of the older
scoring contract:

1. raw-anchor variance is fitted on the fit role and frozen before collection;
   no calibration-role adjustment occurs in this tracer;
2. the original held-out-topology supported-hypothesis gate is deferred to a
   separate evaluation proposal if selection passes; and
3. the capacity-matched anchored Gaussian is promotion-bearing and must be
   beaten by the same `0.01` log-score margin.
