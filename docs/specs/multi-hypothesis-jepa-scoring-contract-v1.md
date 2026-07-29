# Multi-hypothesis JEPA scoring contract v1

## Status and purpose

This contract was frozen before fitting a multi-hypothesis JEPA tracer. It
extends the shared
[JEPA implementation ladder v1](jepa-experiment-ladder-v1.md) with the
smallest public distribution seam and evidence needed to distinguish a useful
multimodal forecast from:

- a larger predictive head;
- a supervised mixture without a JEPA objective;
- one Gaussian represented several times;
- an oracle best-of-K score; or
- uncertainty that does not improve alerting or investigation.

The tracer remains open-development evidence. Passing this contract authorizes
fixed multi-seed robustness, not sealed confirmation or production paging.

## Frozen interpretation

A forecast contains at most four exchangeable hypotheses. Each hypothesis is
one diagonal-Gaussian distribution over a complete observable trajectory and
has one probability for that complete trajectory. Hypothesis indices have no
persistent regime, incident, or causal meaning across samples.

Forecast ambiguity is not an alert. The alert policy uses the probability of
the observation under the complete no-action mixture. An alert may remain
unattributed when no intervention in the closed investigation library has
sufficient posterior support.

## Additive public distribution seam

The tracer adds an immutable `MixtureTrajectoryDistribution` with:

- `component_mean` shaped
  `[sample, component, horizon, entity, feature]`;
- `component_variance` with the same shape;
- `weight` shaped `[sample, component]`; and
- exactly one to four components, with v1 candidates fixed at four.

All arrays are finite. Variances are strictly positive. Weights are at least
`1e-12` after normalization and sum to one for each sample within the declared
floating-point tolerance. A component permutation must leave every public
score and moment unchanged.

`as_trajectory_distribution()` returns the existing compatibility type using
the exact marginal moments

```text
mean = sum_k weight_k * component_mean_k
variance = sum_k weight_k * (component_variance_k + component_mean_k^2)
           - mean^2
```

with only the existing numerical variance floor applied after the calculation.
The compatibility view is used for legacy point metrics. It must never replace
the full mixture for probabilistic selection, alert scoring, or investigation.

Serialization stores every component array, normalized weight, schema
identity, calibration scalar, and component count. Restoration must reproduce
the full mixture, its moment-matched compatibility view, and every score within
the declared deterministic tolerance.

## Exact primary score

For one observed trajectory `y` and its observed-coordinate mask, the primary
score is the exact mixture negative log likelihood:

```text
log p(y) = logsumexp_k(
    log(weight_k)
    + sum_d log Normal(y_d; component_mean_kd, component_variance_kd)
)
log_score(y) = -log p(y) / number_of_observed_coordinates
```

The component is chosen once for the complete trajectory, not independently
at each horizon or entity. The calculation uses stable log-sum-exp. Masked
coordinates contribute neither density nor normalization count.

Window scores are averaged within each logical trajectory, and logical
trajectories are weighted equally within a role. Paired comparisons first
average treatment and control scores inside each matched pair. Raw windows
cannot be treated as independent replicates.

Lower log score is better. MSE of the moment-matched mean remains a mandatory
safety diagnostic but cannot select component count, probabilities, or a
candidate.

Oracle best-component, minimum-component-error, and best-of-K scores are
forbidden for selection and claims. They may be reported only under an
explicitly diagnostic label.

## Required multivariate safety score

The multivariate energy score is recomputed from the stored full mixture:

```text
ES(P, y) = E ||X - y|| / sqrt(D)
           - 0.5 * E ||X - X'|| / sqrt(D)
```

where `D` is the number of observed coordinates. Assessment uses 256 fixed
common-random draws for `X` and 256 independent paired draws for `X'`.
Antithetic standard-normal draws and categorical uniforms are derived from
SHA-256 of `multi-hypothesis-energy-v1` plus the immutable sample identity.
Every model therefore receives identical base randomness. The assessment
stores the sampler identity and draw count.

Energy score cannot select the candidate, but a candidate may not materially
improve log score by becoming pathologically sharp or diffuse: its energy
score must be non-inferior under the gates below.

## Selection and calibration order

The order is immutable:

1. Fit every model and normalizer on fitting-role pairs only.
2. Assess uncalibrated selection-role outputs.
3. Apply the safe-null rule and freeze the candidate recipe.
4. Fit the permitted calibration scalars on calibration-role pairs.
5. Evaluate raw and calibrated frozen outputs on both evaluation roles.

No component count, architecture, seed, loss weight, or representation choice
may change after step 2. Evaluation roles cannot choose between raw and
calibrated output.

### Safe-null selection rule

The four-component candidate remains promotion-eligible only when, on the
selection role:

- its raw log score is at least `0.01` nats per observed coordinate better
  than both the one-component JEPA control and the supervised four-component
  control;
- its moment-matched overall and action-overlap MSE are within 5% of the raw
  low-rank reference; and
- its supported-hypothesis diagnostic below passes.

Otherwise the selected recipe is the best single-component control by raw log
score. The four-component evaluation may still be retained as bounded
diagnostic evidence, but it cannot become a promotion candidate.

### Permitted post-hoc calibration

The frozen candidate and every probabilistic control receive the same
calibration opportunity:

- one global softmax temperature applied to component log weights; and
- one global positive multiplier applied to component standard deviations.

Means, component count, relative component variances, and all representation
parameters remain fixed. Single-component controls use a temperature of one.

Calibration deterministically searches the Cartesian product of 33
log-spaced values from `0.25` through `4.0` for each applicable scalar. It
minimizes trajectory-balanced mixture log score on the calibration role.
Ties choose the pair closest to `(1, 1)` in absolute log distance, then the
lower temperature, then the lower scale. Raw and calibrated metrics are both
reported.

Alert thresholds are separate from probability calibration and use only
calibration-role control trajectories with no-action inputs.

## Calibration diagnostics

Assessment recomputes, raw and calibrated:

- probability-integral-transform histograms for every observable marginal;
- central 50%, 80%, and 90% interval coverage;
- pooled and entity-by-horizon absolute coverage error;
- mean and maximum calibration change in log score;
- component-weight entropy and effective hypothesis count; and
- realized posterior-responsibility concentration.

Entity-by-horizon groups with fewer than 100 observed coordinates are reported
but do not gate. Calibration must not be presented as evidence of causal or
component identity.

## Supported hypotheses and anti-duplication

For components `i` and `j`, define their permutation-invariant standardized
trajectory separation:

```text
distance(i, j) = sqrt(mean_d(
    (mean_id - mean_jd)^2
    / (0.5 * (variance_id + variance_jd))
))
```

A sample has a supported pair when two components each have weight at least
`0.10` and their distance is at least `1.0`. At least 20% of action-overlap
samples must contain a supported pair on both the selection and primary
held-out-topology roles.

Assessment also reports:

- `exp(-sum_k weight_k * log(weight_k))`;
- sorted component weights and sorted posterior responsibilities;
- the fraction of samples with one, two, three, and four supported
  hypotheses;
- pairwise standardized distances;
- a uniform-weight ablation; and
- a moment-matched one-Gaussian collapse.

The moment-matched collapse must worsen held-out-topology log score by at
least `0.01` nats per observed coordinate. Duplicate high-entropy components
therefore do not count as multimodal value.

Training may maximize exact mixture likelihood and may weight latent
prediction terms by soft posterior responsibility. Hard winner-take-all
best-of-K training, component-index labels, pairwise repulsion, and uniform
usage rewards are excluded from v1. Diversity must earn probability through
the observed data rather than through a diversity-for-diversity objective.

## Required controls

All controls use identical fitting, selection, calibration, and evaluation
roles.

1. **Raw predictive reference:** the frozen rank-32 contractive low-rank model,
   with its single Gaussian variance recalibrated under this contract.
2. **One-component JEPA control:** the same context encoder, latent target,
   predictor, optimizer, and fitting budget as the candidate with one
   trajectory Gaussian.
3. **Capacity-matched single Gaussian:** head parameters removed by the
   one-component output are reallocated to hidden width, within 5% of the
   candidate's inference parameter count.
4. **Supervised four-component mixture:** the same mixture seam and
   parameter budget, trained on observable trajectory likelihood without the
   JEPA target encoder or latent prediction objective. This is the
   JEPA-specific null.
5. **Assessment ablations:** the candidate's uniform-weight mixture and
   moment-matched one-Gaussian collapse, without refitting.

The four-component JEPA must beat both the one-component JEPA and supervised
four-component mixture. Beating only the raw reference cannot distinguish
multi-hypothesis value from neural capacity; beating only the one-component
JEPA cannot distinguish JEPA value from an ordinary mixture-density network.

The candidate is limited to one million inference parameters and a 16 MiB
serialized model plus sidecars. Batch-one CPU median and p95 latency and peak
resident memory are mandatory diagnostics. Absolute target-device latency
remains a promotion-stage measurement, not a portable claim from local
Python.

## Alert uncertainty policy

At each transition, hidden-action assessment:

1. supplies a no-action future plan;
2. marginalizes the complete mixture to the next observable step;
3. computes exact one-step mixture surprise per observed coordinate;
4. converts it to an empirical upper-tail p-value using calibration-control
   transitions,
   `p = (1 + count(calibration_score >= score)) / (1 + count(calibration))`;
5. updates
   `evidence_t = max(0, evidence_(t-1) - log(p) - log(2))`; and
6. alerts when evidence crosses the smallest threshold producing no complete
   calibration-control trajectory alerts.

Disagreement, entropy, or component separation alone cannot page. They are
recorded as forecast ambiguity. A low-probability hypothesis protects a
trajectory only in proportion to its calibrated mixture weight.

The gate unit is a complete logical trajectory. Action truth is used only
after stored alert decisions are complete.

## Investigation and abstention

For each frozen action-and-target candidate, assessment computes the complete
ten-step mixture likelihood of the observed future. With equal action-library
priors, those likelihoods define an action posterior.

The system returns the top action attribution only when:

- its posterior probability is at least `0.80`; and
- its posterior odds over the runner-up are at least `4:1`.

Otherwise an alert remains unattributed. Abstention is never converted into a
no-action attribution and counts as a miss in the shared unselective hit@1
metric. Assessment additionally reports selective accuracy, abstention rate,
and attribution coverage.

## Family-specific gates

Every common safety gate in the shared ladder still applies.

### Probabilistic safety

On both evaluation roles:

- calibrated energy score is no more than 1% worse than the best required
  control;
- calibrated 50%, 80%, and 90% pooled marginal coverage is within five
  percentage points of nominal;
- every entity-by-horizon group with at least 100 observations is within ten
  percentage points of nominal;
- overall and action-overlap MSE of the moment-matched mean remain within 5%
  of raw low-rank; and
- raw and calibrated distributions, scores, responsibilities, and
  serialization checks are finite.

Calibration that improves in-distribution score but violates a transfer
coverage or energy gate fails; evaluation cannot choose the raw fallback.

### Probabilistic forecast value lane

This family adds a probabilistic forecast lane. It passes only when:

- on held-out topology, calibrated log score improves by at least `0.02` nats
  per observed coordinate over the best required fitted control;
- on in-distribution evaluation, calibrated log score is no more than `0.01`
  nats per observed coordinate worse than that best control;
- at least 70% of held-out matched pairs improve over the best control;
- the supported-pair and moment-collapse gates pass; and
- every probabilistic safety gate passes.

### Alert and investigation value

The shared alert and investigation gates remain unchanged. In addition:

- an alert-lane claim must beat both the one-component JEPA and supervised
  mixture by at least ten percentage points of detection or two transitions
  of median delay at the same trajectory false-alarm bound; and
- an investigation-lane claim must retain at least 80% attribution coverage
  at 95% selective action-and-target accuracy, while preserving 100%
  no-action specificity, and beat both controls on unselective hit@1.

## Decision

One held-out-topology value lane plus every common and family-specific safety
gate authorizes an unchanged three-seed robustness run. Every seed must pass.
Failure rejects this four-component multi-hypothesis recipe, not mixture
forecasting or JEPA generally.

Permitted positive claim:

> On the open development corpus, a calibrated four-hypothesis JEPA improved
> topology-transfer probabilistic forecasting, alerting, or closed-library
> investigation over single-hypothesis, capacity-matched, and non-JEPA mixture
> controls under proper trajectory scoring.

Permitted negative claim:

> This four-hypothesis JEPA recipe did not add sufficient open-development
> topology-transfer value beyond its single-hypothesis or supervised-mixture
> controls to justify robustness or sealed confirmation.

Neither claim establishes incident semantics, production paging, or
target-device feasibility.
