# Soft regime-codebook JEPA prototype v1

## Question

Does a soft-prototype JEPA retain observable subsystem state and improve
topology-transfer prediction or investigation over a continuous-latent null,
frozen PCA, a classical switching-regime ridge model, and the frozen raw
low-rank predictive core?

This is a non-production logic prototype over open development evidence. It is
not a promotion experiment or sealed confirmation. Its exact runner is retained
as part of the experiment's reproducibility chain.

## Data boundary

Reuse the action-conditioned topology-transfer cache under the shared
[`JEPA implementation ladder v1`](jepa-experiment-ladder-v1.md):

- fit, selection, and calibration use worker topologies one and two;
- topology three remains the primary open transfer diagnostic;
- whole treatment/control pairs remain atomic; and
- no normalizer, representation, codebook, probe, threshold, or configuration
  is fit on either evaluation role.

The action corpus is preferred over the nominal graph corpus because it
contains explicit healthy, intervention, stop, and recovery transitions.

## Prototype model

Use seven entity-preserving tokens of width 16. The online context encoder
combines:

- state tokens from all 20 observed steps;
- a fine view over the last four steps;
- a coarse mean over all 20 steps;
- a full-window trend; and
- learned entity identity.

The EMA target encoder maps every future entity state into the same token
space. A direct ten-step action-conditioned predictor consumes the context
tokens, future demand/topology controls, and the complete action tensor.

The codebook has 32 shared prototypes. Soft squared-distance assignments use a
fixed temperature of 0.25. Training combines:

- decoded observable-future MSE with weight 1.0;
- L1 prediction of quantized future tokens with weight 0.2;
- target-versus-predicted assignment cross-entropy with weight 0.2;
- target-state reconstruction from the soft code with weight 0.2;
- current-state reconstruction from context tokens with weight 0.1;
- marginal code-balance regularization with weight 0.05; and
- per-token assignment sharpness with weight 0.01.

Use seed 127, deterministic PyTorch CPU execution, 40 epochs, batch size 256,
AdamW at `1e-3`, weight decay `1e-4`, and target EMA decay `0.996`.

## Controls

1. **Continuous JEPA null:** the same encoders, direct predictor, decoder,
   optimizer, seed, latent width, and observable-state losses without
   prototype quantization or code losses.
2. **Frozen PCA:** training-only per-entity PCA at width 16, evaluated through
   the same frozen observable-state probe boundary.
3. **Switching-regime ridge:** training-only 32-state k-means over fine/coarse
   observable summaries followed by a ridge future-state predictor
   conditioned on future controls and actions.
4. **Raw low-rank:** the rank-32 contractive raw-state model fitted on the same
   fitting windows.

## Prototype outputs

After loading, fitting each model, and evaluating, print the full prototype
state:

- data identities and role counts;
- training losses and runtime;
- selection metrics;
- in-distribution and topology-transfer predictive metrics;
- observable-state frozen-probe NRMSE;
- active code count, marginal perplexity, assignment entropy, and per-entity
  usage;
- action attribution and sanity;
- hidden-action trajectory false alarms, detection, and delay; and
- the bounded tracer decision.

The one-command entry point is:

```bash
.venv/bin/python lab/action_dynamics/prototype_regime_codebook_jepa.py
```

The script is explicitly non-production but retained. Existing result
artifacts are immutable; every reproduction run must use a fresh `--output`
directory. Production code must not import the runner as a supported library
interface.

## Gates

The codebook advances only if it passes every safety gate and at least one
value lane.

Safety:

1. at least 8 of 32 codes have marginal usage above 0.5%;
2. marginal code perplexity is at least 8;
3. no observed entity has constant assignments;
4. topology-transfer frozen-probe NRMSE is no worse than the continuous null;
5. action-overlap and overall MSE remain within 5% of raw low-rank; and
6. restored public outputs are finite and schema-aligned.

Predictive value:

- downstream-effect MSE improves by at least 10% over raw low-rank; and
- action-and-target hit@1 is at least 95% with 100% no-action specificity.

Investigation value:

- action-and-target hit@1 is at least 95%;
- no-action specificity is 100%;
- correct action beats no-action and deranged action on at least 80% of
  treatments; and
- the codebook improves hit@1 or action sanity over the continuous null.

Alert value:

- control-trajectory false alarms are at most 5%;
- treatment detection is at least 80%;
- median delay is at most 10 transitions; and
- the codebook improves sensitivity or delay over the continuous null at the
  same false-alarm budget.

Failure rejects this codebook recipe, not discrete regimes or JEPA generally.
