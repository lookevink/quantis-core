# Complete multi-view LeJEPA telemetry contract v1

## Status and question

This is the preregistered contract for a single-seed open-development tracer.
It asks:

> Does a complete predictor-free multi-view LeJEPA objective learn a frozen,
> entity-preserving representation whose operational state supports a better
> action-conditioned linear probe than matched representation controls under
> held-out worker-topology transfer?

The candidate is a **representation candidate**. It is not a predictive core,
alert policy, latent dynamics model, or world model. Passing this contract
authorizes fixed multi-seed representation robustness and design of a separate
action-conditioned second stage.

Every runner, assessor, test, specification, and result artifact remains
retained whether the hypothesis passes or fails. Published result directories
are immutable, and reruns use fresh output directories.

## Source identity and scope

The exact objective is pinned to the LeJEPA authors' paper and official
implementation at commit
[`c293d291ca87cd4fddee9d3fffe4e914c7272052`](https://github.com/galilai-group/lejepa/tree/c293d291ca87cd4fddee9d3fffe4e914c7272052).
The formula and finite-quadrature distinctions are recorded in
[`LeJEPA SIGReg primary-source implementation notes`](../research/lejepa-sigreg-primary-source-notes.md).

This tracer implements the complete projector-space objective: cross-view
invariance plus SIGReg under the published convex weighting. It does not add
an EMA target encoder, predictor, reconstruction anchor, covariance penalty,
or action-conditioned latent transition to the candidate.

The source data remains the content-addressed action-dynamics development
corpus and topology-transfer cache:

- fitting and selection use worker topologies one and two;
- the primary open diagnostic holds out worker topology three;
- whole matched pairs remain atomic across roles;
- no fresh or sealed evidence is collected; and
- evaluation data cannot select masks, checkpoints, architecture, optimizer,
  regularization, or adapter settings.

## Independent fitting unit

The 6,320 in-distribution fit windows come from 40 matched pairs, with two
trajectories per pair and 79 overlapping windows per trajectory. Overlapping
windows and matched arms are not independent SIGReg samples.

Every optimizer step therefore uses a **pair-blocked anchor batch**:

1. Include exactly one context anchor from each of the 40 fit pairs.
2. Choose trajectory arm and transition without reading action values,
   outcomes, or future observations.
3. Balance arms within a step and across deterministic schedule cycles.
4. Cycle transition anchors through seeded permutations of all 79 valid
   positions.
5. Generate all eight views from that same selected anchor.

The SIGReg independent sample axis is therefore `N=40`. Views remain a leading
view axis and never increase `N`. Downstream metrics remain pair-balanced.

## Semantic-preserving telemetry views

A telemetry view is a partial observation of one anchor. Every view preserves
absolute time, entity, entity-kind, graph relation, and owned-feature identity.
Masked positions use explicit learned mask embeddings.

Permitted transformations are:

- contiguous temporal cropping;
- connected-topology cropping; and
- masking complete owned time/entity tokens.

Forbidden transformations are:

- numeric jitter or rescaling;
- channel or entity permutation;
- synthetic values;
- cross-trajectory or treatment/control pairing;
- future state or future controls; and
- current or future action truth.

The fixed layout uses eight views:

1. **Global A:** all 20 context points and all seven entities.
2. **Global B:** the most recent 16 points and all seven entities, aligned to
   the 20-point coordinate system with explicit leading masks.
3. **Six local views:** the most recent ten points and a connected
   three-entity block, with one local root for each entity that owns varying
   observed fit state.

Each global view independently masks 10% of its otherwise visible owned
time/entity tokens. Local masks may not empty a view or remove its root. The
current schema has six observed roots; a schema change must either produce a
deterministic round-robin root schedule or fail before fitting. View schedules
are seeded, recorded, and shared across matched neural variants.

Downstream assessment encodes the complete, unaugmented 20-point context.
Augmented training views are never used as evaluation inputs.

## Entity-preserving encoder and projector

The fixed backbone consumes `20 × 7` time/entity tokens:

- each token receives only the owning entity's declared observed coordinates;
- token width is 64;
- there are two pre-normalized transformer blocks;
- each block has four attention heads and feed-forward width 128;
- attention is global, with learned relation-type and graph-distance biases;
- absolute time, entity, entity-kind, and view-presence embeddings are added;
- hidden activation is GELU;
- LayerNorm epsilon is `1e-5`; and
- stochastic dropout is zero.

Graph relations bias attention but never forbid global interaction. The strict
local graph factorization is not repeated because it was unstable and removed
useful cross-entity information in the earlier action-dynamics experiment.

The public representation from an unaugmented context is the seven width-64
tokens at the anchor time, ordered by declared graph entity identity.

A masked mean over visible backbone tokens feeds a training-only projector:

```text
Linear(64, 256) -> GELU -> Linear(256, 64)
```

The projector output is not normalized. The projector is discarded after
fitting and is never presented as the operational representation.

## Exact LeJEPA objective

Let `Z` have shape `(V=8, N=40, D=64)`. Let the first two views be global and
let

```text
g_n = mean(z_0n, z_1n)
L_invariance = mean_(n,v,d) (z_vnd - g_nd)^2
```

No tensor in this expression is detached.

SIGReg uses the official package quadrature:

- 1,024 fresh unit-Gaussian projection directions per optimizer step;
- 17 knots over `[0, 3]`;
- official symmetric trapezoidal weights;
- empirical cosine and sine characteristic functions;
- standard-normal characteristic function and Gaussian window
  `exp(-t^2/2)`;
- multiplication by the independent sample count `N=40`; and
- mean reduction across directions and views.

One projection matrix is shared across all views in one call, then resampled
on the next step from an explicit serialized step-indexed generator.

The candidate loss is:

```text
L_LeJEPA = 0.05 * mean_v(SIGReg(Z_v))
         + 0.95 * L_invariance
```

There is no stop-gradient, EMA teacher, predictor, clipping, whitening,
post-hoc standardization, covariance penalty, or trainable downstream probe in
the candidate training path.

## Matched controls

All neural variants expose the same backbone architecture and seven width-64
tokens at inference.

1. **Invariance-only null:** identical views, initialization, and projector;
   optimize `0.95 * L_invariance` with the SIGReg term absent.
2. **SIGReg-only ablation:** identical views, initialization, and projector;
   optimize `0.05 * mean_v(SIGReg(Z_v))` with invariance absent.
3. **Capacity-matched masked autoencoder:** identical backbone and views; a
   training-only shared per-token decoder reconstructs the complete normalized
   anchor at declared owned coordinates. The decoder is discarded.
4. **Matched PCA:** fit-only deterministic per-entity PCA over the flattened
   20-point owned context, width 64 with zero padding when fitted rank is
   smaller and deterministic component signs.
5. **Raw predictive reference:** the frozen rank-32 contractive low-rank
   observable-state transition.

Training-only projector and decoder parameters, training cost, and artifact
size are reported even though deployed backbone capacity is matched.

## Frozen fitting schedule

Every neural variant receives exactly 1,600 optimizer steps.

- Each step contains the 40 pair-blocked anchors.
- LeJEPA and both ablations share backbone/projector initialization, anchor
  schedule, and view schedule.
- The masked autoencoder shares backbone initialization, anchors, views, and
  step count; its decoder has a separate seed.
- Optimizer: AdamW.
- Initial learning rate: `5e-4`.
- Weight decay: `5e-2`.
- Warmup: 80 linear steps.
- Remaining schedule: cosine decay to `5e-7`.
- Runtime: deterministic CPU float32.
- No early stopping.
- Only the final optimizer state is eligible for assessment.

The frozen seeds are:

| Purpose | Seed |
|---|---:|
| Backbone and projector initialization | 509 |
| Pair/anchor scheduling | 1509 |
| Telemetry views | 2509 |
| SIGReg directions | 3509 |
| Masked-autoencoder decoder | 4509 |

Every stochastic state or deterministic counter required to repeat the run is
serialized.

## Public representation and downstream probes

The encoder exposes the shared fitting, encoding, serialization/restoration,
pure-assessment, and immutable-artifact seams from
[`JEPA implementation ladder v1`](jepa-experiment-ladder-v1.md).

Encoding an unaugmented batch returns:

- finite tokens with shape `(sample, 7, 64)`;
- declared entity order and ownership metadata;
- the observation mask; and
- content identity for the graph, state schema, preprocessing, and encoder.

### Observable-state probe

For each entity with varying owned fit observations, fit an intercept-bearing
ridge map from its frozen token to its latest owned observable state.

- Ridge coefficient: fixed `1e-3`.
- Fit role only.
- Training-fitted target scales define NRMSE.
- Entities without varying owned observations are reported separately.

### Action-conditioned representation probe

This evaluation instrument is not a predictive core.

For each representation:

1. Flatten the seven width-64 frozen tokens.
2. Concatenate the complete declared ten-step future control tensor and one
   candidate-action tensor.
3. Fit an intercept-bearing rank-32 reduced-rank ridge map directly to the
   complete ten-step observable trajectory.
4. Fit all centers, scales, and coefficients on fit pairs only.
5. Choose the ridge coefficient independently per representation from
   `{1e-4, 1e-3, 1e-2, 1e-1, 1}` using pair-balanced selection
   downstream-effect MSE, subject to the raw-reference safety bounds.

The same probe form is used for complete LeJEPA, both ablations, the masked
autoencoder, and matched PCA. The raw low-rank model remains the external
predictive reference.

## Measurements

Report at least:

- training loss and its exact components;
- fixed-projection post-training SIGReg diagnostics separate from the
  stochastic training statistic;
- global/local view agreement;
- aggregate and per-observed-entity variance, covariance, and effective rank;
- aggregate and per-entity current-state probe NRMSE;
- selection, in-distribution, and held-out-topology probe trajectory metrics;
- pair-balanced overall, action-overlap, and downstream-effect MSE;
- action-and-target hit@1, no-action specificity, and whole-pair action
  ablations;
- inference parameter count and training-only parameter count;
- serialized backbone, projector, decoder, PCA, and probe sizes;
- batch-one encoding and probe latency;
- training runtime and peak resident memory when available; and
- exact public-output restoration parity.

Local CPU timings remain microbenchmarks, not target-device claims. This
representation tracer has no alert-policy lane.

## Safety gates

Every gate must pass:

1. Source, role, graph, schema, view, configuration, code, and runtime
   identities are complete.
2. Pair-blocked sampling and view generation satisfy the frozen invariants.
3. Training, representations, probes, predictions, and assessment are finite.
4. Every fitted public artifact restores its public outputs.
5. Aggregate topology-transfer state-probe NRMSE is at most `1.05` times
   matched PCA.
6. No varying observed entity's topology-transfer state-probe NRMSE is more
   than `1.15` times its matched-PCA value.
7. Complete-LeJEPA probe overall, action-overlap, and downstream-effect MSE
   are each at most `1.05` times the raw low-rank reference.
8. Complete-LeJEPA action-and-target hit@1 is at least 95%.
9. No-action specificity is 100%.
10. Correct action beats both no-action and whole-pair shuffled-action
    ablations on at least 80% of held-out treatment pairs.

Effective rank, Gaussianity, loss, and view agreement are diagnostics rather
than substitutes for these downstream safety gates.

## LeJEPA-specific value gates

All value gates must pass:

1. On selection, complete LeJEPA has strictly lower pair-balanced
   downstream-effect MSE than invariance-only, SIGReg-only, masked
   autoencoder, and matched-PCA probes.
2. On held-out topology, complete LeJEPA downstream-effect MSE is at most
   `0.95` times the best of those four controls.
3. Complete LeJEPA wins the per-pair downstream-effect comparison against
   that best held-out control on at least 60% of held-out matched pairs.
4. No evaluation result selected a checkpoint, ridge coefficient, mask,
   hyperparameter, threshold, or control.

Passing every safety and value gate creates a single-seed representation
candidate for fixed-seed robustness. Any failed gate rejects this exact recipe.

## Pure assessment and immutable evidence

The assessor consumes stored representations, targets, probe inputs,
predictions, attribution queries, timings, and identities. It does not call a
fitted model or trust stored metric summaries or gate booleans.

The runner refuses to overwrite an existing output directory. A complete
bundle contains:

- protocol and source identities;
- pair-blocked anchor and view schedules;
- fitted candidate and control artifacts;
- stored unaugmented representations;
- state-probe and action-conditioned-probe inputs and outputs;
- attribution and action-ablation inputs and outputs;
- pure assessment;
- Markdown report; and
- SHA-256 manifest over every evidence-bearing file.

Incomplete work remains explicitly non-interpretable. The runner and assessor
remain in the repository after either confirmation or rejection.

## Required implementation tests

Before the first result directory exists, public tests must establish:

1. Pair-blocked batches contain one anchor per pair, balance arms, cycle all
   valid transitions, and do not count views as samples.
2. View generation is seeded, non-mutating, correctly shaped, connected,
   identity-preserving, future-free, and action-truth-free.
3. The LeJEPA loss matches a literal small-tensor reference, preserves the
   factor `N`, shares directions across views, advances explicit RNG state,
   and has finite nonzero gradients.
4. Encoding returns ordered `(sample, 7, 64)` tokens and restoration preserves
   them.
5. Neural controls have identical inference backbone capacity.
6. PCA is fit-only, deterministic, entity-preserving, and width matched.
7. Reduced-rank probes respect rank, fit-role transforms, pair balance,
   selection-only ridge choice, and restoration.
8. The pure assessor rejects incomplete, non-finite, mismatched, or
   evaluation-selected evidence and recomputes every gate.
9. Artifact creation is non-overwriting and its manifest independently
   verifies.

Tests observe public seams rather than transformer internals or optimizer
implementation details.

## Permitted conclusions

If the tracer passes:

> Under one deterministic open-development seed, the complete pinned
> multi-view LeJEPA recipe produced a frozen telemetry representation that
> passed raw-state safety and improved held-out-topology intervention-effect
> accessibility over every matched representation control. Fixed-seed
> robustness is justified.

If it fails:

> The exact pinned multi-view LeJEPA telemetry recipe did not earn broader
> robustness work under the frozen representation contract.

Neither outcome establishes a production alert policy, deployable predictive
core, general root-cause model, cross-stack representation, or world model.
Changing views, masks, architecture, objective weights, checkpoints, or gates
after observing the result creates a new preregistered experiment and cannot
rescue this one.
