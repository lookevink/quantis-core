# SC-JEPA codebook × multi-resolution interaction v1

## Question

Does the complete SC-JEPA mechanism produce held-topology next-window alert
value that exceeds both isolated components and a raw low-rank classifier,
with a positive codebook-by-multi-resolution interaction?

This is an open-development, one-seed tracer. Passing authorizes fixed-seed
robustness only. It does not authorize sealed evaluation or production
paging.

## Frozen data roles

Use the content-addressed edge-dynamics cache frozen by the shared
[`JEPA experiment ladder`](jepa-experiment-ladder-v1.md).

- Fitting: 40 matched treatment/control pairs from worker topologies one and
  two.
- Selection: 10 disjoint in-distribution pairs.
- Calibration: 10 disjoint in-distribution pairs.
- IID evaluation: 20 disjoint in-distribution pairs.
- Transfer evaluation: 10 topology-three pairs.

Every role remains pair-atomic. No evaluation tensor may affect
normalization, representation fitting, checkpoint choice, the downstream
head, probability calibration, or alert threshold.

The model accepts only current histories and the declared graph at inference.
Future state is a self-supervised fitting target and an action-blind label
source. Future controls, future actions, action kind, target entity,
trajectory identity, and pair identity are forbidden model inputs.

## Event and downstream role

Reuse the HEPA robust normalized one-step state-change event definition. Fit
its center, scale, and 95th-percentile control-trajectory maximum on fitting
controls only. The binary target for each context is whether an event occurs
within the complete ten-transition future window.

Pretrain each cell self-supervised on fitting windows. Select its
self-supervised checkpoint on selection windows. Freeze the representation,
then fit the same width-64 MLP risk head on fitting labels and select its
checkpoint on selection Brier score. Fit an increasing logit calibrator and a
strict control-trajectory-maximum threshold on calibration only.

The raw control is a fitting-only rank-32 PCA of the owned last-ten-step raw
history followed by the same downstream-head, selection, calibration, and
threshold protocol.

## Frozen factorial

| Cell | Predictive bottleneck | Future targets |
|---|---|---|
| `continuous_single` | bias-free continuous `32 × 32` map | five fine patches plus their mean global target |
| `continuous_multi` | bias-free continuous `32 × 32` map | five fine patches plus one separately encoded coarse target |
| `codebook_single` | 32-prototype soft cosine codebook | five fine patches plus their mean global target |
| `codebook_multi` | 32-prototype soft cosine codebook | five fine patches plus one separately encoded coarse target |

All four cells use:

- the final ten context steps and all ten future steps;
- five non-overlapping length-two fine patches;
- a one-patch coarse future made from two five-step means;
- a shared entity-preserving width-32 patch encoder;
- identical two-block fine and coarse predictors;
- identical decoder and risk-head architectures;
- EMA target encoder and bottleneck at decay `0.996`;
- deterministic CPU AdamW, seed `13013`, 300 pretraining steps, batch size
  128, learning rate `5e-4`, weight decay `1e-5`, and gradient norm `0.5`;
- checkpoints every 50 steps; and
- 200 downstream steps with checkpoints every 25 steps.

Codebook cells use temperature `0.1` and the published objective weights:
fine KL `1.0`, fine latent MSE `0.1`, global KL `0.5`,
embedding `1.0`, commitment `0.25`, sample entropy `0.005`, batch entropy
`-0.01`, and linearly annealed reconstruction `0.5 → 0.1`.

Continuous cells replace fine/coarse KL with representation MSE and use the
same reconstruction schedule. The fine MSE weight is `1.1`, preserving the
combined fine-prediction weight; global MSE is `0.5`.

The global predictor is trained in every cell. For a single-resolution cell
its target is derived by averaging EMA fine targets. For a multi-resolution
cell the target comes from the separately downsampled future. Thus the
resolution factor changes target provenance, not effective predictor
capacity.

## Primary estimands

Let `B_cs`, `B_cm`, `B_ks`, and `B_km` denote held-topology calibrated Brier
scores for continuous-single, continuous-multi, codebook-single, and
codebook-multi. Lower is better. The preregistered Brier interaction is:

`I_B = B_cm + B_ks - B_cs - B_km`.

Let `D_*` denote treatment-trajectory detection at the frozen threshold. The
detection interaction is:

`I_D = D_km - D_cm - D_ks + D_cs`.

These difference-in-differences distinguish a joint interaction from either
main effect.

## Safety gates

The complete `codebook_multi` cell must satisfy all of:

1. all public representations, risks, calibrated risks, and restored outputs
   are finite and reproduce within `1e-6`;
2. every factorial cell has exactly the same trainable and deployed inference
   parameter count;
3. at least 8 of 32 codes have transfer marginal use above `0.5%`, marginal
   perplexity is at least 8, and every observed entity uses more than one
   dominant code;
4. a fitting-only state probe over mean entity representations has transfer
   aggregate NRMSE no worse than `1.05 ×` matched entity PCA;
5. serialized candidate plus event definition, calibration, and state probe
   is no more than 16 MiB, with batch-one CPU latency and process peak RSS
   recorded; and
6. the stored-array assessor derives every protocol check, metric, threshold,
   interaction, and decision from raw evidence.

## Value lanes

The tracer advances only if every safety gate and at least one lane passes.

Predictive lane:

- `I_B ≥ 0.05 × B_cs`; and
- `B_km ≤ 0.95 × min(B_cs, B_cm, B_ks, B_raw)`.

Alert lane:

- transfer control-trajectory false alarms are at most 5%;
- treatment detection is at least 80%;
- median post-onset delay is at most ten transitions;
- `I_D ≥ 0.10`; and
- `D_km` exceeds each isolated component and raw low-rank by at least ten
  percentage points.

Failure rejects this edge SC-JEPA interaction recipe, not soft codebooks,
multi-resolution prediction, or SC-JEPA on the authors' benchmark tasks.

## Artifact contract

The runner is non-overwriting and writes to a staging directory before atomic
publication. The retained bundle contains:

- protocol, data identity, model payloads, event definition, and raw
  assessment metadata;
- original and restored representations, risks, calibrated risks, alert
  decisions, labels, code assignments, state-probe evidence, latency samples,
  and causal-input counterfactual evidence;
- a stored-array assessment and report;
- exact reproduction sources; and
- a file-size and SHA-256 manifest verified before publication.
