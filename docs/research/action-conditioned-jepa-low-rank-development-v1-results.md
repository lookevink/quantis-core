# Action-conditioned JEPA + low-rank development v1 results

## Decision

**Reject this configuration. Do not advance it to sealed confirmation.**

The scientifically valid run used a topology-transfer preprocessing cache
whose state and control normalization was fit only on worker topologies 1–2.
Worker topology 3 was unseen until evaluation. The final run used exact L1
latent prediction, deterministic PyTorch execution on Apple MPS, a guaranteed
whole-pair action derangement, and all-node collapse gates.

This is open development evidence. It does not confirm or refute JEPA in
general and does not establish a world model.

## Data boundary

- 40 fit pairs and 6,320 windows from topologies 1–2;
- 10 selection pairs and 1,580 windows from topologies 1–2;
- 10 calibration pairs and 1,580 windows from topologies 1–2;
- 20 in-distribution evaluation pairs and 3,160 windows; and
- 10 held-out-topology evaluation pairs and 1,580 windows.

The 568 MB cache is addressed by both the source artifact manifest and
`action_conditioned_jepa_topology_transfer_v1`. Its compiler records 40
normalizer-fitting pairs.

## Held-out-topology result

| Model | Action MSE | Overall MSE | Downstream effect MSE | Action+target hit@1 | Parameters | Local latency |
|---|---:|---:|---:|---:|---:|---:|
| Raw low-rank | 0.5512 | 0.1057 | 0.0663 | 100% | 34,503 | 0.176 ms |
| Supervised latent low-rank | 0.9377 | 0.1499 | 0.1447 | 30% | 20,527 | 18.9 ms |
| JEPA latent low-rank | 1.6649 | 0.2835 | 0.2885 | 50% | 20,527 | 20.0 ms |

Relative to raw low-rank, JEPA had:

- `3.02×` action-overlap MSE;
- `4.35×` downstream-effect MSE;
- 50 percentage points lower attribution; and
- about `114×` local batch-one latency.

The supervised latent control was better than JEPA but remained materially
worse than raw low-rank. This localizes the primary failure to the learned
encoder/decoder bottleneck. The JEPA objective imposed an additional
predictive penalty rather than recovering transfer.

The same ordering held in distribution: action MSE was `0.3244` for raw
low-rank, `0.7989` for supervised latent, and `1.4896` for JEPA. The failure is
therefore not explained only by the held-out topology.

## What did work

The action-conditioning path was real. On all 10 held-out treatment pairs,
the correct action produced lower error than both:

- the no-action rollout; and
- a deterministic whole-pair derangement with no fixed points.

Mean treatment-pair MSE was `1.1583` with the correct action, `3.0070` without
an action, and `3.9703` with shuffled actions. JEPA learned that interventions
matter, but not a sufficiently accurate or attributable state transition.

Rollouts remained finite. The learned transition spectral constraint did not
fail numerically.

## Representation diagnostics

The aggregate per-node-token effective rank was `5.76` of 16, but aggregation
partly reflects entity identity. Entity-specific effective ranks ranged from
`0.0` to `4.39`; the PostgreSQL node had no varying owned observation and a
constant token. The minimum observed-node rank was only `2.75`, still below
the preregistered threshold of 4.

The representation therefore failed the all-node localization gate. A global
non-collapse statistic alone would have hidden this subsystem-level failure.

## Anomaly result

Sequential latent divergence detected all held-out treatments, but it also
alarmed on all held-out control trajectories. Median treatment delay was
`25.5` transitions.

This detector primarily recognized the unseen topology as anomalous. It did
not separate faults from legitimate topology transfer, so all anomaly gates
failed except raw sensitivity.

## Gate outcome

Only the action-sensitivity gate passed. The configuration failed:

1. 10% downstream-effect improvement;
2. action-MSE non-inferiority;
3. 90% action-and-target attribution;
4. every-node effective rank;
5. control false-alarm rate; and
6. detection delay.

Because the tracer failed, multi-seed robustness and sealed data collection
are not warranted for this configuration.

## Interpretation

The raw low-rank model succeeds because it compresses only the global
transition matrix while leaving the observable state and action/control
channels intact. The learned models instead force each node through a
16-dimensional nonlinear encoder and decoder before applying low-rank
dynamics. That bottleneck discarded local effect magnitude and target
identity—exactly the information required for counterfactual attribution.

The result rejects this architecture:

> A 16-dimension-per-node EMA JEPA bottleneck with rank-32 latent dynamics does
> not improve prediction or attribution over raw low-rank dynamics on the
> fixed Quantis stack, including transfer to an unseen worker topology.

It does not reject using JEPA as an auxiliary objective. A future attempt
would need to preserve the raw low-rank state path and apply JEPA to a residual
or auxiliary investigation representation, rather than making learned latent
tokens the sole dynamics state.
