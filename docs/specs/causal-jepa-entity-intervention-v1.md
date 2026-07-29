# Causal-JEPA entity-intervention tracer v1

## Question

Does whole-entity trajectory masking make interaction recovery functionally
necessary and improve held-topology action-effect prediction over
coordinate-time masking, unmasked latent prediction, and raw rank-32
dynamics?

This is a one-seed open-development tracer. Passing authorizes fixed-seed
robustness only.

## Frozen roles

Use the content-addressed pair-atomic ladder roles:

- fitting: 40 in-distribution pairs;
- selection: 10 disjoint in-distribution pairs;
- IID evaluation: 20 disjoint pairs;
- transfer evaluation: 10 held-topology pairs; and
- the frozen held-topology attribution query library.

No evaluation tensor may affect normalization, checkpoint selection, masks,
or any configuration.

## Frozen cells

Fit three parameter-identical predictors over one shared, fit-only frozen
state-slot projection:

| cell | history masking | loss |
|---|---|---|
| `causal_entity_mask` | two whole entity trajectories after their earliest anchor | masked-history MSE + future MSE |
| `coordinate_time_mask` | ten non-anchor entity/time tokens | masked-history MSE + future MSE |
| `prediction_only` | none | future MSE |

The two masked cells hide exactly ten of the 35 non-anchor history tokens per
batch. Mask schedules vary deterministically by optimizer step. The entity
cell preserves the same two target entities across all five non-anchor
times; the coordinate null distributes its ten targets over entity/time
coordinates.

## Architecture and optimization

Normalize each owned state coordinate on fitting history and future tensors.
Map each instantaneous entity state into width 32 with a frozen
row-orthonormal projection. Use the last six history steps, ten future
queries, seven declared entity slots, and one explicit condition token per
future step containing the declared control vector and flattened entity
actions.

Use a learned mask token, learned temporal embeddings, a linear
identity-anchor projector, a two-layer bidirectional Transformer with four
heads and MLP width 128, and a linear latent output. Decode forecasts through
the transpose of the frozen state projection and the fitting normalizer.

Fit only the predictor with deterministic CPU Adam, seed `18018`, 1,200
steps, one pair-blocked anchor per fitting pair, learning rate `5e-4`,
gradient clipping at one, and checkpoints every 200 steps. Select each cell
by its own selection-role self-supervised objective.

## Mechanism and downstream evidence

On every transfer window, mask one whole entity at a time after its anchor
and recompute its five hidden history states. Report owned-coordinate MSE
overall and on treatment windows. Compare with the coordinate null,
prediction-only, and an anchor-persistence baseline.

Using fully observed history, predict all ten future states. Report selection,
IID, and transfer overall, action-overlap, and downstream-effect MSE;
transfer per-pair effect error; attribution hit@1 and no-action specificity;
and correct-action versus absent/shuffled action sanity.

## Gates

Every public safety gate must pass:

1. all evidence is finite and original/restored forecasts, masked
   completions, and attribution predictions agree within `1e-6`;
2. every neural cell has identical trainable parameter count;
3. the pair-blocked anchor and matched mask schedules independently verify;
4. the candidate's transfer overall and action-overlap MSE are no worse than
   `1.05` times raw;
5. attribution hit@1 is at least 95%, no-action specificity is 100%, and
   correct actions beat absent and shuffled actions for at least 80% of
   treatment pairs;
6. public forecast inference rejects future-state inputs, the serialized
   model is at most 16 MiB, and CPU latency is reported; and
7. a fresh stored-array assessor reproduces every metric, checkpoint
   identity, mask check, and gate.

The interaction mechanism passes only if candidate transfer masked-history
MSE is at least 10% lower than both coordinate-time masking and anchor
persistence.

The downstream value lane passes only if:

- candidate selection downstream-effect MSE is strictly lower than both
  neural controls;
- candidate transfer downstream-effect MSE improves the best neural control
  and raw dynamics by at least 5%; and
- candidate beats the best neural control on at least 60% of transfer pairs.

Advance only if safety, mechanism, and downstream value all pass.

## Artifact contract

Publish atomically without overwrite. Retain all models, the frozen slot
projection, raw reference, anchor and mask schedules, protocol arrays,
forecasts, completions, attribution/action arrays, original/restored outputs,
independent assessment, report, reproduction sources, manifest, smoke
bundles, and failures.
