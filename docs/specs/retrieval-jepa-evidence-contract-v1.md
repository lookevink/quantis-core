# Retrieval-JEPA evidence and abstention contract v1

## Status

Frozen before the first result directory exists on 2026-07-28.

This contract tests one episode-predictive retriever in an investigation role.
It does not reopen the rejected complete-LeJEPA recipe and does not claim
direct trajectory prediction, root-cause discovery, production calibration,
or paging readiness.

The primary-source basis and the distinction between published JEPA objectives
and this retrieval role are recorded in
[`retrieval-jepa-primary-source-notes.md`](../research/retrieval-jepa-primary-source-notes.md).

## Question and hypothesis

Can a compact V-JEPA-style query path learn, without labels or negatives, to
predict the representation of withheld telemetry evidence and thereby retrieve
useful same-mechanism historical episodes on an unseen worker topology better
than raw telemetry, deterministic PCA, a whole-pair-deranged JEPA null, and a
capacity-matched CPC retriever, while remaining competitive with a
capacity-matched label-supervised retriever?

The value claim is exact historical-evidence retrieval with empirical
abstention. Future-trajectory MSE is not a selection metric or value gate.

## Evidence boundary and fixed identity

Use only the content-addressed preprocessing cache
`artifacts/action-dynamics/edge-preprocessing-v1/eb54271132f88c9a431b01e786ea66279a563776434cca2290e47e6b7ae9b3ff`.
Its compiler, graph, feature order, role identities, source corpus identity,
and source artifact identity are authoritative inputs.

- Fit uses worker topologies one and two: 40 matched pairs and 6,320 windows.
- The immutable evidence bank contains one treatment episode from each of
  those 40 fit pairs.
- Selection contains 15 pairs and 30 episode queries; five pairs are on the
  held-out worker topology.
- Calibration contains a disjoint 15 pairs and 30 episode queries; five pairs
  are on the held-out worker topology.
- Evaluation contains 30 pairs and 60 episode queries; ten pairs and 20
  queries form the primary held-out-topology result.
- A matched pair and both trajectories derived from it remain atomic. No
  overlapping window is counted as an independent retrieval query.
- Selection, calibration, and evaluation never fit normalization, weights,
  PCA, a bank vector, or a state-retention probe.

This is open-development evidence. Evaluation was previously available to the
program and is not a sealed confirmation role.

## Retrieval episode compiler

`compile_retrieval_episodes(windows) -> RetrievalEpisodes` is a public,
deterministic seam.

For every matched pair:

1. Identify the treatment trajectory only in the compiler by the presence of
   an applicable intervention in `future_actions`.
2. Identify its first active transition whose declared
   `elapsed_fraction >= 0.5`. The active action must have exactly one action
   kind and one target entity.
3. Select the matched control row at the identical transition.
4. For each arm, store the 20-point `histories` tensor as the alert-time query
   and the aligned 10-point `future_states` tensor as offline raw evidence.
5. Label the treatment episode by the exact `action kind @ target entity` and
   the control episode `no_action`.
6. Preserve pair ID, trajectory ID, transition index, topology, arm, and an
   immutable evidence reference for assessment and operator evidence only.

Action truth is allowed only for step 1 through step 5, supervised-control
fitting, and assessment truth. Query encoders accept only telemetry arrays and
the declared graph. They never accept action tensors, labels, IDs, topology
values outside ordinary observed controls, future controls, or raw evidence.

The compiler returns exactly two ordered episodes per pair, treatment followed
by control within lexicographically ordered pair IDs. It rejects absent or
ambiguous actions, misaligned matched arms, non-finite values, and duplicate
episode identities.

## Frozen representation treatment

The primary treatment is `episode_predictive_jepa`.

### Inputs and token identity

- A full fit anchor is the observed `20 x 7` context followed by its
  `10 x 7` future-state evidence, with 31 ordered state features per token.
- Values are standardized by per-entity/per-feature centers and scales fit on
  fit-role anchors only. Unowned coordinates are zeroed before and after
  standardization.
- Context tokens preserve absolute positions 0 through 19 and entity IDs.
- Target tokens preserve absolute positions 20 through 29 and entity IDs.
- No jitter, token permutation, trajectory splice, action conditioning,
  identifier embedding, label-informed mining, or future value enters the
  query path.

### Architecture

- context encoder: width 64, two pre-normalized Transformer blocks, four
  attention heads, feed-forward width 128, GELU, zero dropout;
- EMA target encoder: identical token projection and encoder;
- positional predictor: learned target mask tokens plus time and entity
  identity, two pre-normalized Transformer blocks, four heads, feed-forward
  width 128, GELU, zero dropout;
- public vector: mean of the 70 predicted or target evidence tokens followed
  by L2 normalization with epsilon `1e-12`;
- all arrays and training use CPU float32; public vectors are returned as
  finite float64 NumPy arrays.

The online query path is context preprocessing, context encoding, positional
prediction, mean pooling, and normalization. The target encoder is retained
for offline evidence-bank construction.

### Objective and schedule

At every step, the existing pair-blocked schedule supplies one anchor from
each of the 40 fit pairs. Arms alternate and transitions cycle without
changing the independent batch size.

The target encoder processes the complete 30-point anchor. The predictor
receives only its first 20 points plus fixed target-position masks. The loss is
the equally weighted mean tokenwise L1 error on:

- all 10 future time positions across all entities; and
- the nested final five future time positions across all entities.

Targets are stop-gradient. The target encoder is initialized from and updated
after each step as a fixed `0.996` EMA of the context encoder. The objective
has no reconstruction term, negatives, SIGReg, action label, or relevance
label.

Freeze AdamW, learning rate `5e-4`, weight decay `1e-4`, 400 optimizer steps,
seed `9019`, no checkpoint selection, and the final step. Any reduced-step run
is explicitly non-interpretable smoke evidence and cannot write the frozen
result path.

## Frozen controls

Every control searches the same 40-item bank with the same truth, exclusions,
similarity, `K`, ties, and assessment.

1. `raw_telemetry`: fit-only standardization, the final ten context points for
   queries and the ten evidence points for bank items, flattened over owned
   coordinates and L2 normalized.
2. `pca_64`: deterministic, sign-oriented width-64 PCA fit on the fit-role raw
   query and evidence vectors only, then L2 normalized.
3. `deranged_target_jepa`: identical architecture, initialization, anchors,
   optimizer, and steps as the treatment, but every target is rolled to the
   next lexicographically ordered matched pair. It tests directed
   same-episode prediction.
4. `cpc_infonce`: identical context, target, and predictor capacity and
   training budget. The positive is the same anchor's evidence vector; all
   other pairs in the pair-blocked batch are negatives. Matched arms and
   overlapping windows never appear together. Freeze temperature `0.07` and
   symmetric InfoNCE. It receives no action or relevance labels.
5. `supervised_retriever`: identical deployed query and evidence architecture,
   width, optimizer, and step budget. It uses only the 40 compiled fit
   treatment episodes and the fixed multi-positive action-and-target
   relevance loss at temperature `0.07`; an episode itself is excluded from
   its positive and denominator sets. This is a feasibility ceiling, not a
   self-supervised baseline the candidate must beat.

The candidate, deranged null, CPC, and supervised control all retain restorable
online and evidence encoders. Training-only heads or EMA updates are excluded
from online parameter and latency counts but retained in the artifact.

## Exact retrieval and immutable evidence

The bank contains the 40 compiled fit treatment episodes in deterministic
episode-ID order. Each entry stores its vector, action-and-target relevance
label, pair/trajectory/transition identity, raw evidence tensor, and evidence
reference. No query trajectory or matched arm can occur in the bank because
roles are disjoint.

For every model, exact cosine similarity is the dot product of L2-normalized
query and bank vectors. Sort descending by similarity and break exact ties by
bank episode ID. Freeze `K=3`. A relevant bank item has the same complete
action-and-target label as a treatment query. A control query has no relevant
bank item.

Report treatment hit@1, hit@3, reciprocal rank of the first relevant item,
complete first-relevant rank, per-action results, and pair-balanced means.
Every returned item remains joined to its raw evidence and source reference.

## Empirical abstention

Confidence is the top-item cosine similarity minus the highest similarity of
an item with a different action-and-target label. This class margin uses only
immutable bank metadata and no query truth.

For each representation independently, fit one threshold using only the 30
calibration episode queries:

1. enumerate the observed confidence values plus an abstain-all sentinel;
2. accept confidence greater than or equal to the threshold;
3. retain thresholds with zero accepted control queries and at most 10%
   empirical error among accepted treatment queries;
4. choose greatest treatment accepted-and-correct rate, then lowest selective
   risk, then lowest threshold;
5. if no treatment query can be accepted, freeze the abstain-all threshold.

The threshold is then unchanged on selection and both evaluation topologies.
Report treatment coverage, selective accuracy/risk, accepted-and-correct rate,
control specificity, and the complete empirical risk-coverage curve.

The calibration role has only 30 episode queries from 15 matched pairs.
It cannot support the SGR 10%-risk, 95%-confidence guarantee described in the
source note. Even treating the dependent arms as independent would be too
small. The artifact must record `sgr_guarantee_feasible=false`; no result from
this tracer may be called certified, guaranteed, or production-calibrated.

## Observable-state and leakage safety

Retrieval value cannot be bought by discarding the current observable state.
For every model, fit one intercept-bearing ridge probe (`ridge=1e-3`) from the
fit-role episode query vector to the owned coordinates at the final observed
context point. Store evaluation truth and predictions. Report normalized RMSE
over fit-varying owned coordinates. This is a safety diagnostic, not a
trajectory-prediction value claim.

The primary treatment must:

- be within `0.10` absolute NRMSE of `pca_64` on held-out topology;
- have effective rank at least 8 in the 40-item bank;
- restore query vectors, bank vectors, rankings, and accept decisions within
  `1e-6`;
- keep every representation, similarity, probe output, metric, and threshold
  finite;
- bind graph, feature, preprocessing, corpus, role, configuration, code, and
  bank identities; and
- pass explicit checks for role disjointness, query-future exclusion,
  action/identifier exclusion, immutable equal bank membership, and exact
  episode counts.

## Selection and promotion gates

The fixed recipe passes this tracer only if every safety gate passes and all of
the following are true.

### Selection diagnostic

On the five held-out-topology selection pairs:

- treatment hit@3 is at least `0.90`;
- accepted-and-correct rate is at least `0.80`;
- selective accuracy is at least `0.90`;
- control specificity is `1.00`; and
- accepted-and-correct rate exceeds both the deranged JEPA null and the best
  of raw telemetry, PCA, and CPC by at least `0.05`.

### Primary evaluation

On the ten held-out-topology evaluation pairs:

- treatment hit@1 is at least `0.80` and hit@3 at least `0.90`;
- accepted-and-correct rate is at least `0.80`;
- selective accuracy is at least `0.90`;
- control specificity is `1.00`;
- accepted-and-correct rate exceeds the deranged null and the best raw, PCA,
  and CPC control by at least `0.05`;
- its pair-balanced mean reciprocal rank exceeds the best non-supervised
  control by at least `0.05`;
- it is within `0.05` accepted-and-correct rate of the supervised retriever;
  and
- it wins hit@1 against the best non-supervised control on at least 60% of
  non-tied treatment pairs.

The pure assessor returns exactly one bounded decision:

- `advance_episode_predictive_retriever_to_fixed_multiseed_robustness`, or
- `reject_episode_predictive_retrieval_jepa_recipe`.

Passing authorizes only fixed multi-seed robustness and collection planning
for enough independent calibration episodes. It does not authorize sealed
confirmation, ANN/quantization work, production integration, or alert paging.

## Edge feasibility diagnostics

Report, but do not use to rescue failed scientific gates:

- online parameter count and total retained training parameter count;
- serialized model bytes;
- bank item count, dimension, dtype, and bytes;
- batch-one CPU preprocessing-plus-query latency and exact-search latency over
  100 measured repetitions after 10 warmups;
- process peak resident memory when available; and
- exact timing runtime and hardware identity.

A promotion candidate additionally requires at most 500,000 online parameters,
at most 10 MiB serialized model bytes, median batch-one query latency at most
100 ms, and median 40-item exact-search latency at most 5 ms on this local
runtime. These are local microbenchmarks, not portable edge-device claims.

## Pure assessment and artifact

The public pure assessor consumes stored episode truth, model vectors or
similarity matrices, state-probe truth and predictions, identities, and
timings. It recomputes rankings, thresholds, metrics, comparisons, every gate,
and the final decision. It never loads a fitted model and never trusts a stored
summary or gate boolean.

The runner writes to `<output>.building`, refuses an existing output or staging
directory, and only atomically publishes a completed immutable directory.
Required files are:

- `protocol.json`;
- `data-identity.json`;
- `episodes.npz` and `episode-metadata.json`;
- `models.json`;
- `representations.npz`;
- `retrieval-evidence.npz` and `retrieval-metadata.json`;
- `assessment.json`;
- `report.md`;
- copies of the runner, assessor, production module, tests, contract, and
  primary-source note; and
- `artifact-manifest.json` containing SHA-256 and byte size for every other
  file.

The manifest is the final file written in staging. A standalone assessor first
verifies it, then reproduces `assessment.json` byte-for-byte from stored arrays.
Failure leaves a machine-readable failure record in staging. No runner,
assessor, test, model artifact, negative result, or completed bundle is deleted
after interpretation.
