# SD-JEPA primary-source implementation notes

## Sources and pinned revision

- Paper: [Subspace-Decomposed JEPAs: Disentangling Progression and
  Content in Latent World Models](https://arxiv.org/abs/2605.31111),
  arXiv v1, 2026-05-29.
- Released code: [LucasStill/SD-JEPA](https://github.com/LucasStill/SD-JEPA),
  revision `1cc121065e83220a495808f4c65ef4b0b1915f9f`.
- Parent world model: [LeWorldModel](https://arxiv.org/abs/2603.19312).

The source revision was read as a scientific reference. Quantis uses a clean
telemetry implementation and does not vendor source code.

## Canonical A2 mechanism

The paper's minimal A2 treatment leaves LeWorldModel's end-to-end
next-embedding prediction intact and makes two training-only changes:

1. the first `k_prog=2` coordinates of the latent are a fixed canonical
   progression subspace and the remaining coordinates are content; and
2. SIGReg acts only on content while a cosine-margin triplet acts only on
   progression.

The split is a coordinate view and adds no learnable parameter. The objective
is

```text
L = MSE(z_hat[t+1], z[t+1])
  + 0.09 * SIGReg(z_content)
  + 0.10 * relu(
        cosine(z_anchor, z_negative)
      - cosine(z_anchor, z_positive)
      + 0.2
    )
```

The released `cosine_triplet_loss` uses the middle timestep as anchor, the
adjacent timestep within radius one as positive, and the anchor timestep from
a different batch trajectory as negative. Quantis preserves this sampler
while constructing every optimizer batch with at most one trajectory from
each matched pair, so a rolled negative cannot be the paired counterfactual
or a second window from the anchor trajectory.

The source uses AdamW, cosine learning-rate decay, 17 SIGReg knots, 1,024
random projections, and a default 192-dimensional visual latent. The Quantis
tracer scales the latent and predictor down for edge deployment while keeping
the objective coefficients, margin, split dimension, sampler, and optimizer
family unchanged.

## Falsifiers

The paper's A0 baseline has no split, applies SIGReg to the full latent, and
has no triplet. Its A2-full falsifier also has no split and applies both
SIGReg and triplet to the full latent. The paper reports that A2-full returns
to A0 on Push-T, while applying the triplet to the full latent despite a split
is worse. Quantis therefore freezes:

- canonical A2 as the candidate;
- A0 as the primary same-width unsplit reference; and
- A2-full as the source ablation.

All three use the same encoder and predictor capacity and training budget.

## Progression readout

For the two progression coordinates,

```text
theta[t] = atan2(z_prog[t, 1], z_prog[t, 0])
score[t] = abs(wrap(theta[t] - theta[t-1]))
```

The paper describes this as a scene-aware compass. On held-out Cube episodes,
the paper reports that absolute angular change outperformed scalar
next-embedding prediction error for semantic-event localization. It also
reports an important limitation: the coordinate is local to an episode and
is identifiable only up to a global rotation. Quantis consequently evaluates
angular *change*, not absolute angle, and does not train a supervised phase
head.

## Deliberately excluded paths

The paper documents temporal straightening, polar predictor conditioning,
angular planning cost, and fixed first-frame anchoring as auxiliary
explorations, not the minimal A2 mechanism. They did not improve A2 in the
reported setup. This tracer excludes them. It also avoids a `k_prog` sweep:
the test is the canonical two-coordinate treatment, not an unbounded
architecture search.

## Telemetry adaptation

Each observation is encoded into entity-preserving tokens and their
ownership-masked mean is the scene latent used by the action-conditioned
predictor and progression loss. Training may use fitting-role future states,
controls, and actions. Public encoding and alert scoring accept only current
histories and the declared telemetry graph.

The primary mechanism test is event localization at the current observation
boundary. The event is the existing fitting-control robust normalized
state-change definition. This adaptation is a Quantis inference from the
paper's semantic-event result; it is not a claim made by the SD-JEPA authors.

