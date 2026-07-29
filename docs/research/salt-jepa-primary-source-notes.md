# SALT-JEPA primary-source notes

## Source

The recipe is pinned to:

- Xianhang Li, Chen Huang, Chun-Liang Li, Eran Malach, Josh Susskind,
  Vimal Thilak, and Etai Littwin, *Rethinking JEPA: Compute-Efficient Video
  SSL with Frozen Teachers*, arXiv:2509.24317, version dated 2026-03-22:
  <https://arxiv.org/abs/2509.24317>.

SALT means **Static-teacher Asymmetric Latent Training**.

## Objective identity

SALT separates representation fitting into two stages:

1. Train a target encoder with masked observation reconstruction.
2. Freeze that target encoder, then train a student encoder and predictor to
   predict the teacher's latent targets at masked positions.

The second-stage target is fixed. There is no EMA update, no jointly evolving
target encoder, no stop-gradient surrogate for an updated teacher, and no
explicit anti-collapse regularizer. The paper uses an L1 latent-prediction
loss and evaluates the frozen student representation.

The teacher and student masking follows the same multi-block family. The
reported best fixed-budget allocation spends 40,000 of 240,000 optimization
steps on the teacher and the remaining 200,000 on the student. The Quantis
tracer preserves that one-to-five ratio as 320 teacher steps and 1,280
student steps.

## Telemetry translation

Pixels become declared owned telemetry coordinates. Video patches become
absolute time/entity tokens. Multi-block masks become seeded contiguous
time-by-connected-entity blocks over the 20-point, seven-entity context.

The teacher decoder is training-only. The teacher, student predictor, and
decoder are retained for reproduction, but deployed representation inference
uses only the student encoder's seven anchor-time tokens.

The translation does not claim that telemetry is video. It tests the
objective-level hypothesis that a separately reconstructed and frozen teacher
provides more stable, useful targets than a co-adapting or predictor-free
representation on this corpus.

## Falsifier

A deterministically deranged SALT cell uses the identical reconstructive
teacher, student, predictor, masks, pair-blocked anchors, optimization steps,
and capacity. During student training only, its teacher target batch is
cyclically reassigned to a different matched-pair anchor. This preserves
target marginals while breaking context-target identity.

SALT contributes a mechanism only if aligned masked-latent prediction beats
that deranged control and the resulting student improves observable
held-topology utility over the reconstructive teacher and raw/PCA references.

