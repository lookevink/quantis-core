# Delta-JEPA primary-source implementation notes

## Sources

- Paper: [Delta-JEPA: Learning Action-Sensitive World Models via Latent
  Difference Decoding](https://arxiv.org/abs/2606.31232), arXiv v1,
  2026-06-30.
- ArXiv v1 source-archive SHA-256:
  `adb9c9d16c055c8c35eb1bc6ebad09e8954edb51932280612a963d0fa45a473a`.
- Parent comparison:
  [LeWorldModel](https://arxiv.org/abs/2603.19312).

The paper does not link released implementation code. The source contains a
commented AAAI example-code placeholder, not a project URL. Quantis therefore
pins the paper formulas and reported hyperparameters rather than inventing an
official repository revision.

## Core objective

Delta-JEPA trains one shared observation encoder, an action-conditioned
next-latent predictor, and a Latent Difference Action Decoder (LDAD)
end-to-end:

```text
z[t] = encoder(o[t])
z_hat[t+1] = predictor(z[t], action[t])
action_hat[t] = decoder(z[t+1] - z[t])

L = MSE(z_hat[t+1], z[t+1])
  + lambda * MSE(action_hat[t], action[t])
```

The paper uses the same trainable encoder on both endpoints. There is no EMA
teacher, stop-gradient branch, reconstruction decoder, SIGReg, or other
distribution-matching regularizer. The published action weight is
`lambda=10`; an ablation peaks at 50 but the main experiments use 10.

## Multi-step LDAD

The main implementation extends LDAD to horizon `N=5`. One long-horizon
displacement `z[t+N]-z[t]` conditions `N` learned action queries through
adaptive layer normalization. A lightweight three-layer non-causal
Transformer produces the five reconstructed continuous actions. The reported
decoder uses eight heads, head dimension 64, and FFN width 512; the encoder
and forward predictor match LeWorldModel's visual stack.

The Quantis edge tracer keeps five action queries, conditional query
modulation, non-causal self-attention, and the two-term objective while
scaling hidden widths and head count down with the telemetry encoder.

## Load-bearing null

The paper's direct ablation replaces displacement with concatenated endpoint
embeddings `[z[t], z[t+N]]`. It reports lower planning performance in every
environment, with the largest gap on Push-T. Quantis uses this as the primary
null.

To make decoder capacity exactly equal despite the different mathematical
input, both cells receive a vector of size `2D`:

- LDAD receives `[delta_z, delta_z]`; both halves contain only displacement
  information.
- The null receives `[z_start, z_end]`.

The decoder architecture and every learnable tensor are then identical. The
duplication does not add information to LDAD.

A prediction-only control retains the same decoder parameters but sets the
action-loss coefficient to zero. It tests the paper's anti-collapse claim
without creating another inference architecture.

## Telemetry action target

Quantis actions are sparse entity-indexed semantic vectors. The redundant
`no_action` complement is removed from the reconstruction target; all other
declared fields are flattened by entity for each of five steps. An all-zero
target therefore means no executed action. The forward predictor still
receives the complete declared future action tensor and future control vector.

This is the closest continuous target to the paper's raw action while avoiding
a loss dominated by repeated `no_action=1` entries on uninvolved entities.
The representation never sees outcome labels, attribution identities, or
evaluation state.

## Evaluation adaptation

The paper evaluates planning, physical-state probes, action-conditioned
latent responses, and state-delta probes. Quantis has no online planner or
pixel goal. It therefore uses the shared frozen representation evaluation:

- a rank-32 action-conditioned probe predicts the complete ten-step
  observable future from current entity tokens and declared candidate action;
- held-topology paired downstream-effect MSE is primary;
- closed-library action-and-target hit@1 and no-action specificity test
  planning-relevant discrimination;
- the trained LDAD's five-step action reconstruction and latent-displacement
  state-delta probe test the claimed mechanism directly; and
- raw rank-32 contractive dynamics remains the deployment reference.

These telemetry gates are Quantis adaptations, not claims made by the paper.

