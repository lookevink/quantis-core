# Edge dynamics development v1

## Purpose

Compare adjacent edge-runnable temporal, graph, log, and streaming techniques
on the already-qualified action-dynamics development-v1 corpus. This is open
development model selection. It is not sealed confirmation and cannot support
a production or world-model claim.

## Immutable source

The only source corpus is
`artifacts/action-dynamics/development-v1`. Its qualified 120 matched pairs,
240 captures, semantic schema, and content identity remain unchanged.

## Pair roles

The existing 30 development-validation pairs remain the `evaluation` role.
Within each of the 15 action-kind by worker-topology cells, the six existing
training pairs are deterministically ordered by SHA-256 of the source corpus
identity and pair ID, then assigned:

- four pairs to `fit`;
- one pair to `selection`; and
- one pair to `calibration`.

This produces 60 fit, 15 selection, 15 calibration, and 30 evaluation pairs.
No pair or capture may cross roles. Normalization is fit on `fit` captures
only.

## Common representation

Each capture contains 108 states over seven declared graph entities. Each
state has 27 operational metrics plus four structured log/trace aggregate
features. Exact request demand and worker replicas are exogenous controls.
The action tensor is kept separate. Windows use 20 observed states and a
10-transition horizon.

The preprocessing stage writes a non-overwriting, content-addressed cache.
Model experiments consume that cache rather than repeatedly parsing the
1.5 GB raw evidence bundle.

## Candidate ladder

Run candidates sequentially:

1. echo-state action dynamics with a fixed sparse reservoir and ridge readout;
2. a small causal temporal convolution predicting the entire horizon directly;
3. a contractive low-rank action-conditioned linear state-space model;
4. the low-rank global model plus a clipped one-hop graph residual;
5. metrics-only versus metrics-plus-structured-event feature ablation;
6. conformal one-step residual calibration plus sequential accumulation with
   action truth hidden; and
7. a Count-Min Sketch feature-system benchmark on the observed event stream.

The frozen dense action-conditioned VARX and persistence remain references.

## Selection and evaluation

Candidate configuration is selected using the `selection` role only.
Calibration thresholds use the `calibration` role only. Final development
scores use the `evaluation` role only.

For predictive candidates report:

- normalized MSE overall and over action-overlap states;
- paired treatment-minus-control downstream effect error;
- action-and-target hit@1 and no-action specificity using the frozen candidate
  library;
- empirical finite-horizon rollout stability;
- parameter count, serialized size, and batch-one CPU latency.

For the detector report evaluation-control false-alarm rate, treatment
detection rate, and detection delay. A detector may use predictions from a
selected model but receives no action truth.

For log/template and sketch experiments report feature vocabulary, collision
or reconstruction behavior, memory, and their effect on the selected
predictor. The current corpus has only three application event templates, so
these results cannot establish natural-language or high-cardinality
generalization.

## Bounded interpretation

The tournament can identify a promising compact predictor, investigator, and
detector for this fixed lab stack. The existing evaluation split has already
informed the redesign, so even a strong result remains development evidence.
A winning configuration must be frozen and evaluated on a fresh sealed corpus
before publication as a confirmed claim.
