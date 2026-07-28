# Residual JEPA correction development v1 results

## Decision

**Reject this configuration. Do not run seed robustness or sealed
confirmation.**

Preserving the raw low-rank path solved the severe attribution regression from
the prior JEPA-bottleneck experiment: attribution and action sensitivity
remained perfect. It did not make JEPA useful. The JEPA residual branch
slightly worsened held-out-topology downstream-effect, action-overlap, and
overall prediction, and it was worse than the capacity-matched supervised
residual branch.

This is open development evidence. It rejects this auxiliary recipe on the
fixed corpus, not JEPA in general, and it does not establish or refute a
general software world model.

## Valid evidence boundary

The reviewed result is:

`artifacts/action-dynamics/residual-jepa-correction-development-v1-reviewed`

It reused the topology-transfer cache whose normalizers were fitted only on
worker topologies 1–2. The primary evaluation held out topology 3:

- 40 fit pairs and 6,320 windows;
- 10 selection pairs and 1,580 windows;
- 10 calibration pairs and 1,580 windows;
- 20 in-distribution evaluation pairs and 3,160 windows; and
- 10 held-out-topology evaluation pairs and 1,580 windows.

The rank-32 raw-state baseline was fitted once. Its SHA-256 was identical
before and after both neural correction fits. Both correction decoders emitted
exactly zero before their first optimizer update.

## Held-out-topology result

| Model | Action MSE | Overall MSE | Downstream effect MSE | Attribution | Gain | Latency |
|---|---:|---:|---:|---:|---:|---:|
| Frozen raw low-rank | 0.5512 | 0.1057 | 0.06627 | 100% | — | 0.151 ms |
| Supervised residual | 0.5504 | 0.1037 | 0.06586 | 100% | 0.25 | 19.3 ms |
| JEPA residual | 0.5522 | 0.1058 | 0.06635 | 100% | 1.0 | 20.1 ms |

Relative to raw low-rank, supervised residual correction improved:

- action-overlap MSE by about 0.13%;
- overall MSE by about 1.90%; and
- downstream-effect MSE by about 0.63%.

Those changes are directionally encouraging but far below the preregistered
10% downstream-effect gate. They cost 55,030 total parameters instead of
34,503 and roughly 128× local latency.

JEPA residual correction:

- worsened action-overlap MSE by 0.19%;
- worsened overall MSE by 0.02%;
- worsened downstream-effect MSE by 0.12%; and
- had 0.75% higher downstream-effect MSE than supervised residual correction.

Selection chose full JEPA gain, so this is not a safe-fallback artifact. The
selection action MSE improved monotonically from `0.29501` at gain zero to
`0.29425` at gain one, but that tiny development advantage did not transfer.

All models retained 100% action-and-target hit@1 and 100% no-action
specificity. Correct action beat both no-action and whole-pair deranged action
on all 10 held-out treatment pairs. Preserving the raw path therefore fixed
the prior architecture's attribution failure.

## In-distribution result

The same ordering held in distribution. Supervised residual correction
improved overall error and downstream-effect error modestly. JEPA residual
correction was slightly worse than raw low-rank. This argues against topology
shift as the sole explanation for JEPA's failure.

## Training interpretation

The JEPA latent objective did converge: mean latent L1 fell from `0.630` in the
first epoch to `0.045` in the last. Its decoded residual MSE ended at `0.0975`,
worse than the supervised branch's `0.0932`.

The most likely interpretation is objective interference over a small,
low-signal remainder:

1. the raw low-rank model already captures nearly all action-conditioned
   structure available in this corpus;
2. the remaining predictable residual is small and is better fit directly;
3. forcing the same limited tokens to satisfy masked future-state latent
   prediction consumes capacity without improving the residual quantities
   needed for counterfactual effects; and
4. the remaining error may contain observation noise or unobserved variables
   that neither objective can predict.

Aggregate token effective rank was `6.16/16`, while observed entity-specific
ranks ranged from `1.47` to `3.03`. PostgreSQL remained unobservable and
rank-zero. The latent representation was not globally collapsed, but its local
degrees of freedom remained thin.

## Investigation wake-up result

The initial implementation calibrated a 5% point threshold but judged whether
any point alarmed across a whole trajectory. That mismatched units and
compounded false alarms across overlapping windows:

- in distribution: 45% control-trajectory false alarms, 90% treatment
  detection, median delay 4;
- topology transfer: 60% control-trajectory false alarms, 90% treatment
  detection, median delay 1.

The reviewed protocol calibrates the maximum divergence of each control
trajectory, matching the threshold and gate unit. With only 10 calibration
control trajectories, the 5% higher quantile is the observed maximum. It
produced:

- 0% control false alarms; and
- 0% treatment detection

both in distribution and on topology transfer.

The useful conclusion is not that one threshold is preferable. It is that
fault and control divergence are not sufficiently separated: sensitivity is
available only at an unacceptable trajectory false-alarm rate.

## Reproducibility note

The original and reviewed MPS runs used the same seed, data, configuration, and
`torch.use_deterministic_algorithms(True)`. Their raw low-rank artifacts were
byte-identical, but their neural artifacts were not. Both runs selected JEPA
gain `1.0` and supervised gain `0.25`, preserved attribution, and rejected the
same gates. Numeric differences were small:

- JEPA downstream degradation was 0.22% then 0.12%;
- supervised downstream improvement was 0.18% then 0.63%.

The decision is qualitatively reproduced, but MPS was not bitwise
reproducible. Any future neural tracer that approaches a gate should use
multiple seeds and a CPU determinism check before promotion.

## Gate outcome

JEPA passed safety and attribution gates:

- action and overall MSE remained within 5%;
- attribution and specificity were 100%;
- action sanity was 100%; and
- selection chose a nonzero correction.

It failed the two gates that would establish value:

- no 10% downstream-effect improvement; and
- worse downstream-effect MSE than supervised residual correction.

The trajectory-calibrated divergence detector passed false-alarm control but
failed sensitivity and delay.

## What this changes

The experiment cleanly separates two conclusions:

1. **Keep the raw low-rank model as the predictive core.** It remains faster,
   smaller, and at least as accurate for the claim we care about.
2. **Do not add this JEPA residual branch.** It preserves attribution but adds
   no predictive or detector value.

If the project continues toward a bounded world model, the next high-value
step is not another JEPA loss sweep on the same targets. It is to enrich the
state with variables the residual may depend on—per-node resource saturation,
queue age distributions, trace-derived edge timing, recovery actions, and
fault intensity—then test whether a simple supervised structured residual
model finds a materially predictable remainder. JEPA should return only if
that richer corpus contains masked or multimodal structure that a direct model
cannot exploit.
