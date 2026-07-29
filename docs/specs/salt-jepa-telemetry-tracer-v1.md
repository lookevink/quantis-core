# SALT-JEPA telemetry tracer v1

## Question

Can a student trained against a separately reconstructed, frozen telemetry
teacher preserve intervention-relevant state better than the teacher itself
and a target-deranged SALT control, while remaining competitive with the raw
rank-32 predictive core under held-worker-topology transfer?

This is a one-seed open-development tracer. Passing every gate authorizes
fixed-seed robustness only. It does not authorize paging, autonomous
remediation, or sealed confirmation.

## Public seams

The confirmed test seams are:

1. `fit(windows)` for pair-blocked two-stage training;
2. `encode(histories, graph)` for causal student representation inference;
3. `encode_teacher(histories, graph)` for the frozen reconstructive control;
4. a public masked-prediction diagnostic using only context observations and
   a declared mask step;
5. `to_dict` / `from_dict` restoration; and
6. pure reassessment from stored artifact evidence.

Tests use these public seams and do not inspect private parameters or training
arrays.

## Frozen role and data contract

Use the existing content-addressed topology-transfer cache:

- fitting: 40 in-distribution matched pairs;
- selection: 10 disjoint in-distribution pairs;
- calibration: not used by representation or probe fitting;
- IID evaluation: 20 disjoint in-distribution pairs; and
- transfer evaluation: 10 held-worker-topology pairs.

Whole matched pairs remain atomic. Fitting selects normalizers, encoders,
teachers, masks, and predictors. Selection chooses only the ridge for each
frozen downstream probe. Evaluation cannot select a checkpoint, ridge,
architecture, mask, or gate.

## Mask contract

Every optimizer step contains one context anchor from each fitting pair.
Trajectory arm and transition follow the retained pair-blocked schedule.

For every anchor, a seeded multi-block mask hides exactly 90% of the 140
absolute time/entity tokens, rounded to 126 tokens. Masks are assembled from
contiguous time ranges crossed with connected entity blocks, with a
deterministic fill step that extends existing blocks. At least one owned token
per observed entity and one token at the anchor time remain visible. Numeric
jitter, entity permutation, synthetic values, future observations, actions,
and cross-role records are forbidden.

The schedule is serialized. Candidate and deranged cells receive identical
anchors and masks.

## Frozen architecture and training

Teacher and student use the complete LeJEPA telemetry backbone:

- 20 by 7 absolute time/entity tokens;
- width 64;
- two pre-normalized graph-biased transformer blocks;
- four attention heads;
- feed-forward width 128;
- GELU, no dropout; and
- declared time, entity, kind, presence, relation, and graph-distance
  identities.

### Stage 1: reconstructive teacher

Train the teacher for 320 steps. A training-only
`Linear(64,64) -> GELU -> Linear(64,F)` decoder predicts declared owned
coordinates at masked tokens. Minimize masked-coordinate MSE.

### Stage 2: static-target student

Freeze the teacher completely. Initialize a separate student and
`Linear(64,256) -> GELU -> Linear(256,64)` predictor. Train them for 1,280
steps. The teacher sees the complete context; the student sees the masked
context. Minimize L1 error between predicted student tokens and frozen teacher
tokens at masked positions.

Use deterministic CPU float32 AdamW, learning rate `5e-4`, weight decay
`5e-2`, 80-step warmup in each stage, cosine decay to `5e-7`, no early
stopping, and final states only.

Seeds are:

| Purpose | Seed |
|---|---:|
| Teacher initialization | 23023 |
| Teacher decoder | 24023 |
| Student initialization | 25023 |
| Student predictor | 26023 |
| Pair-blocked anchors | 27023 |
| Multi-block masks | 28023 |

## Frozen cells and controls

| Representation | Teacher targets during stage 2 |
|---|---|
| `salt_jepa` | same context anchor |
| `deranged_salt_jepa` | deterministic no-fixed-point cyclic pair reassignment |
| `reconstructive_teacher` | stage-1 teacher; no stage 2 |
| `matched_pca` | fit-only entity-preserving width-64 PCA |

The candidate and deranged cells have identical training and inference
capacity. Each trains its own deterministically identical teacher so the
artifacts remain independently restorable.

The immutable raw rank-32 action-conditioned predictive core is the primary
observable comparator.

## Frozen downstream evaluation

Fit the existing rank-32 reduced-rank action probe on each representation
using fitting-role representations, controls, and declared actions. Select
one ridge from `{1e-4, 1e-3, 1e-2, 1e-1, 1}` per representation using only
selection-role downstream-effect MSE subject to:

- selection overall MSE no worse than `1.05` times raw; and
- selection action-overlap MSE no worse than `1.05` times raw.

If no ridge is selection-safe, retain the lowest downstream-effect ridge and
record that selection safety failed.

Report pair-balanced overall, action-overlap, and downstream-effect MSE;
per-pair effect errors and win fractions; aggregate and per-entity
observable-state retention; action attribution and action sanity; aligned
masked-latent L1; rank; capacity; serialized bytes; and batch-one CPU latency.

## Gates

All safety gates must pass:

1. stored evidence and independent reassessment are finite;
2. fitting, selection, calibration, IID evaluation, and transfer identifiers
   are disjoint at pair and trajectory level;
3. candidate and deranged cells have identical training and inference
   parameter counts;
4. the teacher is unchanged throughout stage 2;
5. restored student, teacher, diagnostic, and probe outputs match originals
   within `1e-6`;
6. public encoding and diagnostics accept no future observation, pair
   identity, or outcome;
7. the candidate's selection and transfer overall and action-overlap MSE are
   each no worse than `1.05` times raw;
8. action-and-target hit@1 is at least 95%, no-action specificity is 100%,
   and correct action beats no-action and shuffled action on at least 80% of
   transfer treatment pairs;
9. deployed student bundle size is at most 16 MiB; and
10. batch-one CPU latency is recorded.

The mechanism gate passes only if candidate masked-latent L1 is at most
`0.90` times deranged SALT on selection and transfer diagnostics.

The representation-value gate passes only if:

- candidate transfer downstream-effect MSE is at most `0.90` times the
  reconstructive teacher, deranged SALT, and raw rank-32 reference;
- candidate selection downstream-effect MSE is strictly lower than all three
  learned representation controls;
- candidate beats the reconstructive teacher on at least 60% of transfer
  matched pairs; and
- candidate aggregate and every observed entity's transfer state-retention
  error are no worse than the reconstructive teacher.

Every safety, mechanism, and value gate must pass. Failure rejects this exact
SALT telemetry recipe while retaining its code and evidence.

## Artifact contract

Write through a fresh staging directory and publish atomically. Retain both
cell models, teachers, decoders, predictors, probes, raw reference, mask and
anchor schedules, role identifiers, original/restored public outputs,
masked-prediction diagnostics, downstream distributions, independent
assessment, report, source identities, reproduction-source copies, and a
SHA-256 manifest.

The artifact may establish only:

> On the fixed Quantis lab stack and open corpus, a reconstructive frozen
> teacher did or did not improve representation utility through the exact
> SALT-style two-stage objective relative to its teacher, a deranged target
> control, PCA, and raw dynamics.

