# LeWorldModel bounded geometry screen v1

## Question

Does the exact end-to-end LeWorldModel objective, or one bounded published
alternative to its ambient Gaussian geometry, produce a state-rich,
edge-runnable telemetry representation that improves held-topology
downstream-effect prediction over prediction-only and raw rank-32 dynamics?

This is a one-seed open-development screen. Passing authorizes fixed-seed
robustness only.

## Frozen roles

Use the content-addressed pair-atomic ladder roles:

- fitting: 40 in-distribution pairs;
- selection: 10 disjoint in-distribution pairs;
- IID evaluation: 20 disjoint pairs;
- transfer evaluation: 10 held-topology pairs; and
- the frozen held-topology attribution query library.

No evaluation tensor may affect fitting, normalization, checkpoint or ridge
selection, or geometry-cell selection.

## Frozen cells

Fit seven parameter-identical cells:

| cell | latent support | training-only regularizer |
|---|---|---|
| `lewm_ambient` | Euclidean | ambient SIGReg |
| `sub_jepa` | Euclidean | SIGReg over 8 frozen orthogonal width-4 subspaces |
| `rectified_lp` | non-negative | rectified-Gaussian sliced Wasserstein |
| `ker_jepa` | Euclidean | analytic Gaussian-prior RBF MMD |
| `sphere_jepa` | unit sphere | sliced Wasserstein to uniform-sphere samples |
| `sphere_mmd` | unit sphere | deterministic heat-kernel MMD |
| `prediction_only` | Euclidean | none |

All regularized cells use coefficient `0.09`. Distribution-matching cells use
256 random projections except deterministic MMD cells. Random draws are
seeded by optimizer step. Frozen subspace matrices are buffers, not parameters.

## Architecture and optimization

Use a width-32 entity-preserving encoder, hidden width 64, learned entity
embeddings, and a two-layer residual MLP. Mean observed entities only inside
the world-model predictor. Use a two-hidden-layer MLP predictor over the last
three scene latents plus the aligned control/action vector.

Fit normalization on fitting histories and fitting futures. Predict each of
the ten future embeddings from its preceding three latent states and declared
condition. Fit with deterministic CPU AdamW, seed `17017`, 800 steps, one
pair-blocked anchor per fitting pair, learning rate `5e-5`, weight decay
`1e-3`, cosine decay, gradient clipping at one, and checkpoints every 100
steps. Select each cell by its own selection-role self-supervised objective.

## Shared downstream evaluation

Freeze every selected encoder. Fit the rank-32 reduced-rank
action-conditioned probe over ridges `{1e-4,1e-3,1e-2,1e-1,1}`. Select each
ridge on selection by lowest downstream-effect MSE among rows within 5% of
raw overall, action-overlap, and downstream effect; if none are safe, select
lowest downstream-effect MSE and record the safety failure.

Report selection, IID, and transfer forecast metrics; transfer per-pair
effect error; attribution hit@1 and no-action specificity; action sanity;
current-state NRMSE against width-32 entity PCA; variance, effective rank,
support sparsity, norm dispersion, and temporal straightness.

The selection-role screen winner is the safe regularized cell with lowest
selection downstream-effect MSE. If no regularized cell is selection-safe,
record that fact and still identify the lowest-error cell for diagnosis.
Evaluation never changes this choice.

## Gates

Every public safety gate must pass:

1. all evidence is finite and original/restored representations, probe
   predictions, and attribution predictions agree within `1e-6`;
2. every neural cell has identical training and inference parameter count;
3. the selected geometry's state NRMSE is no worse than `1.05` times PCA;
4. its transfer overall and action-overlap MSE are no worse than `1.05` times
   raw;
5. attribution hit@1 is at least 95%, no-action specificity is 100%, and
   correct action beats absent and shuffled actions for at least 80% of
   treatment pairs;
6. public inference is causal, its model-plus-probe bundle is at most 16 MiB,
   and CPU latency is reported; and
7. a fresh stored-array assessor reproduces all metrics, selection, and gates.

The geometry lane passes only if the selected regularized cell:

- is non-collapsed with effective rank at least eight;
- has selection downstream-effect MSE strictly below prediction-only; and
- has state NRMSE no worse than prediction-only.

The downstream value lane passes only if it:

- reduces transfer downstream-effect MSE by at least 5% versus both
  prediction-only and raw; and
- beats prediction-only on at least 60% of transfer pairs.

Advance only if all safety, geometry, and value gates pass. Report each
individual cell even if the selected winner fails. Run UR-JEPA only if both
ambient and subspace cells exhibit the same low-effective-rank failure while
remaining otherwise competitive; otherwise its stated prerequisite is absent.

## Artifact contract

Publish atomically without overwrite. Retain all selected models, probes,
raw/PCA references, ridge curves, protocol arrays, original/restored outputs,
assessment arrays, independent assessment, report, reproduction sources,
manifest, smoke bundles, and failures.
