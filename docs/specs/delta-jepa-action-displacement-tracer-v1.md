# Delta-JEPA action-displacement tracer v1

## Question

Does decoding the executed five-step action sequence from latent displacement
alone create a state-rich, action-sensitive telemetry representation whose
held-topology downstream-effect predictions beat both a capacity-matched
endpoint-concatenation null and the raw rank-32 contractive model?

This is a one-seed open-development tracer. Passing authorizes a fixed
multi-seed robustness run only; it does not authorize production paging or
autonomous action.

## Frozen data roles

Use the content-addressed cache and pair-atomic roles from the JEPA experiment
ladder:

- fitting: 40 in-distribution pairs;
- selection: 10 disjoint in-distribution pairs;
- IID evaluation: 20 disjoint pairs;
- transfer evaluation: 10 held-topology pairs; and
- the frozen held-topology attribution query library.

No evaluation tensor may affect representation fitting, normalization,
checkpoint selection, ridge selection, or any probe. Public encoding accepts
only current histories and the declared graph.

## Frozen cells

Fit three cells with identical initialization, encoder, predictor, decoder,
parameter count, optimizer, schedule, and data:

| cell | action-decoder input | action-loss weight |
|---|---|---:|
| `delta_jepa` | `[z[t+5]-z[t], z[t+5]-z[t]]` | 10 |
| `endpoint_concat` | `[z[t], z[t+5]]` | 10 |
| `prediction_only` | identical decoder retained but unused | 0 |

The duplicated displacement preserves exact capacity without exposing an
endpoint shortcut.

## Architecture and objective

Use an entity-preserving shared encoder with width 16 and hidden width 64.
Ownership-mask normalized entity observations, add a learned entity
embedding, and apply a two-layer residual MLP. Flatten the seven entity
tokens only inside the world-model predictor and action decoder.

The predictor is a two-hidden-layer MLP that predicts the next complete
entity-token state from the current tokens, controls, and full declared
action. Apply next-latent MSE over all ten fitting future transitions.

Use `N=5` learned action queries. Modulate queries with decoder input through
learned shift and scale, then apply two non-causal Transformer layers with
four heads and hidden width 64. Decode the five compact entity-action vectors.
Use action weight 10, matching the paper.

Fit with deterministic CPU AdamW, seed `16016`, 1,600 steps, one pair-blocked
anchor per fitting pair, learning rate `5e-5`, weight decay `1e-3`, cosine
decay, gradient clipping at one, and checkpoints every 200 steps. Fit
normalization on fitting histories plus fitting futures. Select each cell's
checkpoint by its own selection-role two-term objective; never compare
objective values across cells.

## Shared downstream evaluation

Freeze every selected encoder. For each representation, fit the shared
rank-32 reduced-rank action-conditioned future probe over ridge values
`{1e-4, 1e-3, 1e-2, 1e-1, 1}`. On selection, choose the lowest
downstream-effect MSE among rows that keep overall, action-overlap, and
downstream-effect MSE within 5% of raw; if none are safe, choose the lowest
downstream-effect MSE and record the safety failure.

Report on IID and held-topology evaluation:

- pair-balanced overall MSE;
- pair-balanced action-overlap MSE;
- paired treatment-minus-control downstream-effect MSE;
- per-pair downstream-effect error;
- action-and-target hit@1 and no-action specificity on the frozen query
  library; and
- correct-action versus no-action and pair-shuffled-action sanity.

Fit the raw rank-32 contractive dynamics reference on fitting only.

## Mechanism diagnostics

On held-topology intervals, report:

- five-step compact action reconstruction MSE for all cells;
- treatment-only action reconstruction MSE;
- exact action-sequence nearest-candidate retrieval against correct,
  no-action, and pair-shuffled targets;
- delta-to-observable-state-change ridge NRMSE and Pearson correlation;
- latent displacement variance and effective rank; and
- frozen current-state probe NRMSE against matched rank-16 entity PCA.

## Gates

All safety gates must pass:

1. all evidence is finite and original/restored representations, decoder
   outputs, probe predictions, and attribution predictions agree within
   `1e-6`;
2. all three neural cells have identical training and inference capacity;
3. candidate current-state NRMSE is no worse than `1.05` times matched PCA;
4. candidate transfer overall and action-overlap MSE are each no worse than
   `1.05` times raw;
5. action-and-target hit@1 is at least 95%;
6. no-action specificity is exactly 100%;
7. correct action beats both no-action and shuffled action for at least 80%
   of treatment pairs;
8. public inference is causal, the candidate bundle is at most 16 MiB, and
   batch-one CPU latency is recorded; and
9. a fresh stored-array assessor reproduces every metric and gate.

The mechanism lane passes only if candidate treatment action-reconstruction
MSE is at most `0.90` times endpoint-concat and candidate action-sequence
retrieval exceeds endpoint-concat by at least ten percentage points.

The downstream value lane passes only if:

- candidate transfer downstream-effect MSE is at most `0.90` times both
  endpoint-concat and raw;
- candidate beats endpoint-concat on at least 60% of transfer pairs; and
- the selection-role candidate downstream-effect MSE is strictly lower than
  endpoint-concat.

Advance only if every safety gate and both value lanes pass. Failure rejects
this exact edge Delta-JEPA recipe, not latent-difference action decoding on
the paper's visual-control benchmarks.

## Artifact contract

Write through a staging directory and publish atomically without overwriting.
Preserve all selected checkpoints, raw and PCA references, ridge curves,
probes, original/restored outputs, query predictions, mechanism arrays,
independent assessment, report, reproduction sources, manifest, smoke
bundles, and failures.

