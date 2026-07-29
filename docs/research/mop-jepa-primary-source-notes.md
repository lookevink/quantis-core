# MoP-JEPA primary-source notes

## Sources and identity

- [MoP-JEPA: Hard-Assigned Predictor Mixtures for Stochastic JEPA World
  Models](https://arxiv.org/abs/2607.05238), arXiv source archive SHA-256
  `38979f20eda082461180279baeb0792d793eff10b64980ddab00d66d712c6d6e`.

No official public implementation was located when this tracer was frozen.
The paper source is therefore the implementation authority. Quantis uses a
clean telemetry adaptation and does not vendor external code.

## Material mechanism

MoP-JEPA leaves the online encoder, EMA target encoder, and JEPA context
construction unchanged. It replaces one deterministic predictor with
parallel heads and a context-only router. Every predictor output and target
is L2-normalized. For each sample, the target selects the head with minimum
cosine distance; only that head receives the prediction loss. The winning
head index supervises router cross-entropy.

The deployed interface emits every `(candidate, router_probability)` pair in
one pass. The realized future is never an input to the router. This is
materially different from Quantis's earlier four-component JEPA, which
trained all heads through likelihood responsibilities rather than a hard
winner.

The paper defines a minibatch KL penalty on hard winner usage. As written,
the discrete winner histogram has no gradient. The reported fairness grid
includes zero load-balance weight. This tracer freezes that unambiguous
setting (`lambda_bal = 0`) instead of inventing an undocumented
straight-through estimator. The paper does not state one universal router
coefficient in the manuscript; the tracer freezes `lambda_route = 1`.

## Verification requirements

Raw best-of-K coverage is insufficient. The paper requires:

- a context-free codebook;
- shuffled contexts;
- router gating at `pi_k > 0.5 / K`;
- transition precision; and
- verified-route success for planning tasks.

The Quantis corpus has no executable environment or transition oracle, so
verified-route success cannot be claimed. The tracer instead measures
selection-only context dependence and an explicitly limited
realized-transition precision: an active prediction is valid when its
trajectory RMSE to the observed realization is below a calibration-only raw
model radius. This is conservative for genuinely stochastic systems because
an unobserved valid branch will count as invalid.

## Frozen telemetry adaptation

Use the complete ten-step normalized telemetry future as one successor.
Encode each entity/time state into a unit latent, predict eight complete
latent successor trajectories, and choose one trajectory-level hard winner
by mean cosine distance. A shared linear decoder, trained only to reconstruct
EMA targets, maps candidates back to observable telemetry.

The context contains entity-preserving summaries of the 20-step history plus
all declared future controls and actions. A context-only router produces
eight probabilities. Per-head observable variances are calibrated after
training on the disjoint calibration role; they do not change the predictor
or its hard assignments.

The paper's main experiments use `K = 8` and target EMA `0.996`, which this
tracer retains. The compact edge architecture and 40-epoch budget match the
earlier Quantis multi-hypothesis tracer so the hard-assignment mechanism,
rather than a larger backbone or longer search, is the new variable.
