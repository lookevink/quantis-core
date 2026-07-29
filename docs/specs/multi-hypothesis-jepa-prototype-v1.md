# Multi-hypothesis trajectory JEPA prototype v1

## Status and question

This is a preregistered, single-seed, open-development logic prototype under
the frozen
[multi-hypothesis scoring contract](multi-hypothesis-jepa-scoring-contract-v1.md).
It asks:

> Does a compact four-component latent trajectory JEPA improve
> topology-transfer probabilistic forecasts, hidden-action alerts, or
> closed-library investigation beyond a one-component JEPA, a
> parameter-matched single Gaussian, a supervised four-component mixture, and
> the raw rank-32 low-rank model?

The prototype is non-production, but its exact runner remains in the repository
as reproduction code. A positive result authorizes a durable three-seed
implementation; a negative result rejects only this recipe.

## Evidence boundary

- Source corpus: `artifacts/action-dynamics/development-v1`.
- Reuse the content-addressed topology-transfer preprocessing cache under
  `artifacts/action-dynamics/edge-preprocessing-v1`.
- Fit on 40 matched pairs and 6,320 windows from worker topologies one and two.
- Use 10 pairs and 1,580 windows each for selection and calibration.
- Report 20-pair, 3,160-window in-distribution evaluation and 10-pair,
  1,580-window worker-three topology transfer.
- Keep matched pairs atomic and trajectory-balanced in every proper score.
- Seed: `307`.
- Runtime: deterministic CPU PyTorch with one thread.
- No new, sealed, or previously unobserved evidence is collected.

## Candidate architecture

### Entity-preserving representation

One shared linear-GELU state encoder maps each of the 31 observable features
to a 12-dimensional entity token. For each of seven entities, concatenate the
last token, 20-step mean token, and end-minus-start trend token, then map the
36 values to a 16-dimensional context token. The public encoding is the seven
context tokens in graph order.

An EMA target encoder with decay `0.996` maps each future entity state to the
same 12-dimensional target space. A shared linear decoder reconstructs
observable entity state from either predicted or target tokens.

### Four complete trajectory hypotheses

Flatten the seven context tokens and the complete ten-step future control and
action plan. A two-layer GELU predictor of width 128 emits four complete
`[10, 7, 12]` latent trajectories. A separate linear head emits four
complete-trajectory weights.

The shared decoder maps each latent trajectory to its observable mean. Each
component learns one positive diagonal variance per entity and feature,
broadcast across the ten horizons. The four weights are floored and
renormalized through the public mixture seam.

### Objective

For 40 epochs with batch size 256, use AdamW at learning rate `1e-3` and
weight decay `1e-4`. Minimize:

- exact complete-trajectory mixture negative log likelihood per observed
  coordinate, weight `1.0`;
- posterior-responsibility-weighted L1 latent prediction, weight `0.20`;
- target-token observable reconstruction MSE, weight `0.10`; and
- last-context-token observable reconstruction MSE, weight `0.05`.

Posterior responsibilities are detached before weighting the latent loss.
There is no best-of-K loss, component repulsion, entropy reward, or component
label.

## Controls

1. **One-component JEPA:** identical encoders, latent width, predictor width,
   decoder, objective, optimizer, and fitting budget with one component.
2. **Capacity-matched single Gaussian:** one component with no JEPA losses.
   Its predictor width is the smallest integer giving an inference parameter
   count within 5% of the candidate.
3. **Supervised four-component mixture:** identical candidate inference
   architecture trained only by exact observable mixture likelihood.
4. **Raw low-rank:** the rank-32 contractive model fitted to the identical
   fitting windows and represented as a one-component mixture.
5. **Assessment ablations:** uniform candidate weights and the exact
   moment-matched one-Gaussian collapse, without refitting.

## Selection, calibration, and assessment

Apply the frozen contract without tuning:

- selection uses raw trajectory-balanced log score and the safe-null rule;
- calibration searches the frozen 33-by-33 global weight-temperature and
  standard-deviation-scale grid;
- probabilistic evaluation uses exact log score, 256-draw energy score,
  marginal interval coverage, supported-pair rate, and moment-collapse loss;
- legacy point metrics use only the exact compatibility moments;
- hidden-action alerting uses one-step no-action mixture surprise and
  calibration-control sequential thresholds; and
- investigation ranks the closed action library by complete mixture
  likelihood and applies the fixed posterior abstention policy.

The pure assessor consumes stored arrays and identities, not model metric
summaries.

## Prototype output

One command prints and stores:

- source, cache, split, schema, seed, runtime, and configuration identities;
- raw and calibrated distributions for every assessment role;
- trajectory-balanced log and energy scores;
- interval coverage and component-support diagnostics;
- point-prediction, alert, and investigation results;
- inference parameters, serialized bytes, batch-one latency, and restoration
  parity; and
- every gate plus the bounded decision.

The one-command entry point is:

```bash
.venv/bin/python lab/action_dynamics/prototype_multi_hypothesis_jepa.py
```

## Decision

Advance only if one held-out-topology value lane and every common and
multi-hypothesis safety gate pass. Otherwise preserve the runner, immutable
evidence, and result interpretation as a reproducible negative result, and
continue to the next materially different JEPA family.

Neither outcome establishes production paging, general incident semantics, or
target-device latency.
