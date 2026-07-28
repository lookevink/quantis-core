# Action-conditioned JEPA + low-rank development v1

## Purpose

Test whether a node-preserving JEPA representation with a low-rank,
action-conditioned latent transition improves transfer beyond the raw-state
low-rank model on the fixed Quantis lab stack.

This is an open-development experiment over the already-inspected
`action-dynamics/development-v1` corpus. It cannot confirm a world-model claim.

## Source and split

Use only the content-addressed edge preprocessing cache associated with the
qualified source artifact manifest.

The primary open transfer diagnostic holds the largest observed
`worker_replicas` topology out of fitting:

- fit and selection: the two smaller worker topologies;
- calibration: the two smaller worker topologies; and
- transfer evaluation: the largest worker topology.

All filtering is by complete matched pair. No pair or capture crosses roles.
The remaining evaluation topologies form an in-distribution diagnostic.

## Public seams

The model implements the existing `EdgeDynamicsModel` fit, rollout,
serialization, parameter-count, and graph-schema boundary. The experiment
writes a non-overwriting result bundle with model artifacts, hashes, metrics,
gates, and a bounded Markdown interpretation.

Tests exercise these public seams rather than optimizer internals.

## Representation and dynamics

Each normalized state contains seven graph entities with 31 observed features.
A shared temporal node encoder emits one 16-dimensional token per entity.
Entity identity therefore remains explicit:

`7 entities × 16 dimensions = 112 latent state dimensions`.

The online encoder consumes the 20-state history. An exponential-moving-average
target encoder consumes the observed future block. The predictor rolls a
rank-32 global transition through the 112-dimensional latent state while
conditioning every step on request/replica controls and the complete
entity-owned action tensor.

Training combines:

- block-masked L1 latent prediction;
- a lightly weighted decoded future-state loss;
- current-state recovery;
- variance and covariance anti-collapse terms; and
- spectral projection of the latent transition.

The deterministic tracer uses seed 89, 60 epochs, batch size 256, AdamW at
`1e-3`, EMA decay `0.996`, and Apple MPS when available. Sixty epochs replace
the under-converged 12-epoch smoke run; seed robustness is required only if
the tracer passes the development gates.

The supervised-latent control uses the same encoder, latent budget, transition,
and decoder without the JEPA latent-prediction objective or masking.

## Comparisons

Report:

1. the existing raw-state contractive low-rank model;
2. the capacity-matched supervised latent low-rank model;
3. the JEPA latent low-rank model; and
4. no-action and deterministically shuffled-action evaluation of the fitted
   JEPA model.

The dense VARX result remains contextual evidence from the preceding
development ladder, not a selection candidate here.

## Metrics

For both in-distribution and held-out-topology evaluation report:

- normalized overall and action-overlap MSE;
- paired treatment-minus-control downstream-effect MSE;
- closed-library action-and-target hit@1 and no-action specificity;
- parameter count, serialized size, latency, finite rollout, and ten-step
  norm growth.

Also report:

- per-node-token effective rank, variance, and covariance;
- the fraction of treatment pairs for which correct-action prediction beats
  both no-action and shuffled-action prediction; and
- hidden-action point and sequential conformal detection.

The matched pair, not an overlapping window, is the unit for action-sensitivity
and future sealed statistical intervals.

## Development gates

The JEPA candidate advances to a sealed experiment only if, on the open
held-out-topology diagnostic, it:

1. improves downstream-effect MSE by at least 10% over raw low-rank;
2. keeps action-overlap MSE within 5% of raw low-rank;
3. retains at least 90% action-and-target hit@1;
4. has per-node-token effective rank at least 25% of latent width; and
5. beats both action ablations on at least 80% of treatment pairs.

The anomaly-investigation gate is separate:

- no more than 5% control-trajectory sequential false alarms;
- at least 80% treatment-trajectory sequential detection; and
- median sequential delay no greater than 10 transitions.

Failure of a development gate rejects advancement of this configuration, not
JEPA in general.

## Confirmation boundary

Passing development gates permits freezing the configuration and collecting a
fresh sealed corpus. A bounded predictive-world-model claim requires the same
primary gates to pass on sealed matched pairs with a paired 95% bootstrap
interval excluding no improvement. No result from this document is sealed
confirmation.
