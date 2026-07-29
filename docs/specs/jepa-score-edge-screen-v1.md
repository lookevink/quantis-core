# JEPA-SCORE edge-feasibility screen v1

## Status and question

This is the frozen single-seed open-development contract for an exact
JEPA-SCORE value-path screen. It asks:

> Can the exact Jacobian-volume score of the already-fitted complete LeJEPA
> projector run inside the edge budget and improve action-blind
> held-topology alerts over a robust raw-telemetry delta score?

JEPA-SCORE is a scoring technique, not another trainer. The screen must not
modify, refit, fine-tune, or select a checkpoint from any retained encoder.
Passing authorizes a fresh fixed-seed robustness contract, not production or
sealed confirmation. Code and artifacts remain after either outcome.

The mathematical source and applicability boundary are recorded in
[the primary-source notes](../research/jepa-score-primary-source-notes.md).

## Frozen source identities

Use the content-addressed cache:

```text
artifacts/action-dynamics/edge-preprocessing-v1/
eb54271132f88c9a431b01e786ea66279a563776434cca2290e47e6b7ae9b3ff
```

Require:

- source-corpus SHA-256
  `df03af282f48591216251f22f934b97e1555147df9ed6f6c6d1f7684f9644a26`;
- source artifact-manifest SHA-256
  `d02afa33a5977cba69b255cd1a5f31470751b6681da1ecae31b11e86307e65b1`;
- cached preprocessing artifact-manifest raw-file SHA-256
  `525bd7e68b47336fad8eb0c39c0d93b0e99a7a80c0682119be3626d6066a3fa8`;
- preprocessing protocol
  `action_conditioned_jepa_topology_transfer_v1`; and
- pair-atomic fit, selection, calibration, IID evaluation, and
  worker-topology-three transfer roles.

Before loading, require the cache manifest to contain exactly:

| File | SHA-256 |
|---|---|
| `attribution-queries.npz` | `d649d238511da59e2f69aa9dc21c9f6a5513c13168f74cffd3e2129daf3c5e64` |
| `calibration.npz` | `9885f67751801b60479972e2d04f18dba7b31d3723e5991bbd94b332facaf9fb` |
| `evaluation.npz` | `cd861d41bbce2f660b921b654cac4061a5642df1e8781c71d5dbff5ac772b706` |
| `fit.npz` | `b481893f59cbd75c19a445c78b2c61e6d052ba8c70324993b552aec9a052a160` |
| `metadata.json` | `816cbff2642eb41ea0cf2565074f76d736ede7f365dc3ca0200587b52e0ee6f5` |
| `selection.npz` | `dd12288ec3cf650c250bab4e36be4530c4b60f513842fe26cf319513e3977622` |

The raw manifest hash, exact file set, each declared hash, and each actual
file hash must all agree. The derived directory address alone is not a data
identity.

Copy and hash-verify models from
`artifacts/action-dynamics/prototype-complete-lejepa-v1`, whose
artifact-manifest SHA-256 is
`00639afaee81cd3844e8b60014ed87f886a8e7a7e0b20bc50834416da1deb265`.
The primary model JSON SHA-256 is
`eda9795582f2965ba1091b1dca710bc74ce2098bbc747ddfc0de3a324e39e412`.
This is the SHA-256 of the raw file bytes and is named
`source_model_file_sha256`. A separately named
`source_model_payload_sha256` hashes its canonical parsed JSON; the two must
not be conflated.

The primary cell is `complete_lejepa`. Retain `sigreg_only` as a Gaussian
regularization control and `invariance_only` as a collapse falsifier. Their
scores are mechanism diagnostics only and cannot replace the primary cell.
No evaluation observation may select a model, sign, input coordinates,
epsilon, anchor, threshold, or alert rule.

## Exact scorer

Restore each frozen backbone and its training projector:

```text
Linear(64, 256) -> GELU -> Linear(256, 64)
```

Equation (4) of the paper averages the training transform distribution and
equation (5) is its one-transform Monte Carlo estimate. Freeze that estimate
to `TelemetryViewSchedule(seed=2509).batch(..., step=1600).global_a`.
Step 1600 is the first mask draw after the 0–1599 training draws. It is a
fresh member of the exact training-view support: all 20 points and seven
entities are present, with the schedule's fixed 10% owned-token visibility
mask. The visibility/presence arrays are shared across samples, copied into
the scorer bundle, and hash-verified. No identity, local, second-global, or
evaluation-selected view may replace it. A future multi-transform estimator
would be a separate contract and must aggregate equation (4) as
`-logmeanexp(-score_T)`.

Given one normalized complete history `x` with shape `(20, 7, 31)`, multiply
it by the frozen ownership mask **inside the differentiated function**, then
apply the frozen view:
`f_T(x) = projector(visible_mean(backbone_T(x * ownership)))`. Merely zeroing
a NumPy array before creating the Jacobian leaf is forbidden because that
leaves nonzero derivatives with respect to semantically unowned coordinates.
Apply the training mask embedding to invisible-but-present tokens and
mean-pool only visible tokens exactly as in complete-LeJEPA training. This
defines `f_T(x)` with 64 outputs. The input coordinate system is exactly the
cached normalized tensor; no second scaling is allowed. Every unowned
Jacobian column must be exactly zero and its maximum absolute value is
retained.

For a sample-separable batch, compute:

```text
J = jacobian(lambda x: f(x).sum(0), inputs=batch)
J = J.flatten(2).permute(1, 0, 2)
s = svdvals(J)
jepa_score = sum(log(clamp_min(s, 1e-6)), axis=1)
anomaly_score = -jepa_score
```

Use CPU float32, `torch.autograd.functional.jacobian` without a vectorized or
stochastic estimator, and `torch.linalg.svdvals`. Model parameters are
frozen. The scorer has no RNG, gradient update, finite difference,
Hutchinson trace, randomized SVD, rank truncation, or learned surrogate.
Retain all 64 singular values for every scored sample.

Batch-one and multi-sample results must agree within `1e-3` absolute score
and `2e-5` per singular value. Freeze the parity batch to the lexicographically
first three selection trajectory IDs at transition 39, in that order. The
literal assessor recomputes the lexicographically first trajectory at
transition 19 in each of the four scored roles for all three cells, plus the
three-row parity batch for all three cells. It must implement the model route,
mask-inside-Jacobian, Jacobian reshape, SVD, and reduction literally and must
not import or call the production `jepa_score` module.

## Action-blind sampling and roles

Score the fixed transition indices `(19, 39, 59, 79, 97)` for every
trajectory in:

- in-distribution selection;
- in-distribution calibration;
- in-distribution evaluation; and
- held-topology evaluation.

There is exactly one row per trajectory and fixed transition. Sampling never
uses action fields, action onsets, event labels, future state, or score
values. Retain sampled histories, role, trajectory ID, matched-pair ID,
transition, treatment status used only by the assessor, and source row
index. Onsets are read only after scoring for offline assessment.

Frozen role counts are 40 fit pairs/80 trajectories for raw fitting only,
10 selection pairs/20 trajectories, 10 calibration pairs/20 trajectories
including ten controls, 20 IID-evaluation pairs/40 trajectories, and ten
held-topology pairs/20 trajectories. Each scored trajectory contributes five
rows. `source_row_index` means the row in the cached role tensor before
worker-topology partitioning; retain enough partition receipts to replay it.

After scoring, use the cached `applicable` future-action coordinate only to
derive labels. A trajectory is treatment iff any applicable value is greater
than `0.5`. Its onset is the minimum
`transition_indices[row] + horizon_offset` over applicable future-action
coordinates, where offset zero is the action at transition `t` that predicts
state `t+1` under the compiler contract. Require exactly one treatment and
one control trajectory per matched pair. Retain the necessary action-label
source rows and recomputation receipt; the assessor may not trust stored
labels or onsets.

Selection reports the fraction of matched pairs for which the treatment
anomaly score at transition 39 exceeds its control twin. It does not choose
the score or policy.

## Raw comparator and calibration

Fit the raw comparator on fit controls only. Determine controls using the
same `applicable` rule above. Sort the 40 control trajectories by ID and each
trajectory's 79 cached rows by transition 19 through 97. Use exactly one
terminal normalized delta `history[-1]-history[-2]` per row: 3,160 fit
deltas total. Do not add interior-history deltas, reconstruct/deduplicate
trajectory states, or use `future_states`. Compute per entity/feature:

```text
center = median(delta)
mad = 1.4826 * median(abs(delta - center))
std = population_std(delta)
scale = where(mad > 1e-8, mad, where(std > 1e-8, std, 1))
center = where(ownership, center, 0)
scale = where(ownership, scale, 1)
```

For a sampled history, set `delta = history[-1] - history[-2]`,
`standardized = (delta-center)/scale`, and return the root mean square over
owned entity/feature slots only. Retain and independently recompute the
ownership mask, center, scale, sorted fit-control trajectory IDs, input
receipts, and every score.

For each method independently, negate JEPA-SCORE but leave raw delta
positive. On the ten in-distribution calibration control trajectories,
compute each trajectory's maximum across the five fixed anchors. The alert
threshold is the 0.95 higher quantile, which is the maximum with ten
controls. Alert only when a score is strictly greater than the threshold.
Calibration treatments are not used.

Apply both fixed thresholds unchanged to IID and held-topology evaluation.
Alerts are independent strict-exceedance decisions at the five anchors; they
do not latch. Control false-alarm rate is controls with any alert divided by
all controls. Pre-onset alert rate is treatments with any alert at an anchor
less than onset divided by all treatments. Detection rate is treatments with
any alert at an anchor at least onset divided by all treatments. First
detection is the minimum such alerted anchor and delay is first detection
minus onset. Median delay uses detected treatments only and is `None` when
none are detected. A trajectory with both early and later alerts counts in
both pre-onset and detection. Retain row-level decisions and replay every
reduction.

## Runtime, premise, and evidence

Run latency in a fresh scorer-only subprocess with
`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, Torch intra-op threads 1, and Torch
inter-op threads 1, all set before model work. Exclude process/import/model
load from latency. Warm up on the lexicographically first selection
trajectory at transition 19, then make exactly 20 batch-one calls at
transition 39 in lexicographic trajectory-ID order. Time the complete exact
score including transform, Jacobian, and SVD with `time.perf_counter_ns`.
Median uses NumPy's standard median; p95 is
`quantile(samples, 0.95, method="higher")`. Report every sample.

Record baseline RSS after imports but before bundle load, absolute peak RSS,
and their nonnegative difference in that fresh process, noting platform RSS
units. Record Torch version, OS/hardware, CPU affinity when exposed, and
power state when exposed.

The measured bundle is one uncompressed canonical UTF-8 JSON file named
`primary-scorer.json`. Its strict schema contains graph, ordered features,
ownership, backbone state, projector state, architecture/preprocessing
config needed by inference, fixed view visibility/presence, epsilon, source
identities, and parameter count. It must not contain training metrics,
optimizer/generator state, decoder state, runner evidence, or calibration.
Gate its exact file byte count. The frozen count is 116,848
backbone-plus-projector parameters.

On all sampled projector embeddings, report effective rank, marginal mean
and variance summaries, mean absolute off-diagonal covariance, singular
value clipping count, and score distributions by role/arm. These diagnose
the paper's Gaussian premise; they do not independently establish value.

The immutable artifact contains:

- copied source model JSON files and their hashes;
- sampled histories and all identifiers;
- every exact score and singular value;
- raw scores and calibration thresholds;
- alert decisions and per-trajectory rows;
- literal-assessor recomputation samples;
- latency samples and environment;
- a complete snapshot of the local `src/quantis_core` source tree plus
  runner, latency worker, assessor, spec, and primary-source notes;
- a manifest hashing every non-manifest file; and
- a pure `result.json` and human-readable report.

Finalize result and report, create the manifest excluding itself, then run
the copied assessor with an isolated import path rooted only at the copied
source closure. It verifies final contents and the manifest before atomic
rename. The runner refuses overwrite.

## Frozen gates

The run is interpretable only if all protocol gates pass:

1. every cache and source-model identity matches;
2. roles and matched pairs are disjoint and have frozen counts;
3. fixed anchors exist exactly once per trajectory;
4. sampling is action-blind and recomputes from source row indices;
5. all three model states restore exactly and remain unchanged;
6. exact score math, sign, epsilon, input coordinates, and projector route
   match this contract, including exactly zero unowned Jacobian columns;
7. batch parity and literal-assessor parity pass;
8. the latency input file, fixed sample rotation, receipt reductions, thread
   settings, timer, load exclusion, and RSS arithmetic recompute;
9. every retained array is finite and all singular values are nonnegative;
10. calibration uses only IID calibration controls and strict exceedance;
11. thresholds, decisions, onsets, and trajectory metrics recompute;
12. evaluation has no selection authority; and
13. source snapshots and artifact manifest verify.

The primary cell passes edge safety only if:

- median exact batch-one latency is at most `100 ms`;
- p95 exact batch-one latency is at most `125 ms`;
- the restored bundle is at most `8 MiB`;
- parameter count is at most `120,000`;
- IID and transfer control false-alarm rates are each at most `0.05`; and
- IID and transfer treatment pre-onset alert rates are each at most `0.05`.

It passes alert value only if:

- selection matched-pair directional win fraction is at least `0.60`;
- IID treatment detection is at least `0.80`;
- transfer treatment detection is at least `0.80`; and
- on transfer it Pareto-dominates raw delta: false-alarm rate is no higher,
  treatment detection is no lower, median delay is no higher when both are
  defined, and at least one improves materially (`0.05` false-alarm rate,
  `0.05` detection rate, or 20 transitions of median delay).

Pass only if every protocol, edge-safety, and alert-value gate passes.
Otherwise reject this exact JEPA-SCORE alert recipe while retaining whether
the score remains interesting for unlatched offline drift analysis.
Non-frozen smoke runs are never eligible to pass or authorize advancement;
they report `non_interpretable_jepa_score_smoke` while retaining provisional
gate outcomes for debugging.
