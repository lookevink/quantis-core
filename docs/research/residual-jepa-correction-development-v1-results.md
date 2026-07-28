# Residual JEPA correction development v1 results

## Decision

**Reject this configuration. Do not run seed robustness or sealed
confirmation.**

Preserving the raw low-rank path fixed the severe attribution regression from
the prior JEPA-bottleneck experiment. It did not make the residual branch
useful. The pure supervised correction was rejected by selection, while the
JEPA correction earned full gain on selection but slightly worsened every
primary transfer error.

This is open development evidence. It rejects this auxiliary recipe on the
fixed corpus, not JEPA in general, and it does not establish or refute a
general software world model.

## Final evidence boundary

The final result is:

`artifacts/action-dynamics/residual-jepa-correction-development-v1-final`

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

Earlier MPS bundles named without a suffix and with `-reviewed` are superseded.
Review found that their supervised null retained token regularizers, their
detector used only horizon one, and same-seed MPS training was not bitwise
reproducible. They remain local audit artifacts and are not cited as results.

## Held-out-topology result

| Model | Action MSE | Overall MSE | Downstream effect MSE | Attribution | Gain | Latency |
|---|---:|---:|---:|---:|---:|---:|
| Frozen raw low-rank | 0.55115 | 0.10574 | 0.066273 | 100% | — | 0.177 ms |
| Pure supervised residual | 0.55115 | 0.10574 | 0.066273 | 100% | 0.0 | 0.175 ms |
| JEPA residual | 0.55217 | 0.10576 | 0.066351 | 100% | 1.0 | 0.874 ms |

The supervised branch's action-overlap MSE was worst at every nonzero gain on
the selection role, increasing from `0.29501` at gain zero to `0.32247` at gain
one. Selection therefore chose the safe fallback. Its final predictions are
exactly the frozen baseline. This is stronger evidence than a small
held-out-test difference: direct residual learning did not earn use before
transfer evaluation.

JEPA's selection action MSE improved monotonically from `0.29501` at gain zero
to `0.29436` at gain one. That 0.22% development advantage did not transfer.
Relative to raw low-rank, JEPA:

- worsened action-overlap MSE by 0.184%;
- worsened overall MSE by 0.018%; and
- worsened downstream-effect MSE by 0.117%.

All models retained 100% action-and-target hit@1 and 100% no-action
specificity. Correct action beat both no-action and whole-pair deranged action
on all 10 held-out treatment pairs. Preserving the raw path therefore solved
the prior architecture's attribution failure.

## In-distribution result

The same ordering held in distribution. The supervised branch selected gain
zero and exactly matched raw low-rank. JEPA was slightly worse:

- action MSE `0.32532` versus `0.32441`;
- overall MSE `0.08617` versus `0.08607`; and
- downstream-effect MSE `0.10509` versus `0.10500`.

Topology shift is therefore not the sole explanation.

## Training interpretation

The JEPA objective did optimize successfully. Mean latent L1 fell from
`0.62955` in the first epoch to `0.03025` in the last. Its decoded residual MSE
ended at `0.09753`, while pure supervised residual MSE ended at `0.09296`.

The likely interpretation is a low-signal, poorly transferable remainder:

1. raw low-rank already captures nearly all action-conditioned structure
   available in this corpus;
2. direct residual fitting lowers training MSE but fails selection, consistent
   with overfitting noise or unobserved variables;
3. JEPA regularization makes a tiny selection improvement but does not preserve
   it in either evaluation distribution; and
4. future-state latent predictability is not the same as predictability of the
   small counterfactual effect error that matters to this claim.

Aggregate token effective rank was `6.15/16`, while observed entity-specific
ranks ranged from `1.46` to `3.10`. PostgreSQL remained unobservable and
rank-zero. The representation was not globally collapsed, but local degrees of
freedom remained thin.

## Investigation wake-up result

The final detector uses every predicted horizon, then reduces each control
trajectory to its maximum latent divergence for calibration. With only 10
calibration control trajectories, the 5% higher quantile is the observed
maximum.

It produced 0% control false alarms and 0% treatment detection both in
distribution and on topology transfer. No detection delay exists because
there were no post-onset detections.

This is not evidence that latent divergence is insensitive. Superseded
point-calibrated diagnostics were sensitive but accumulated unacceptable
trajectory false alarms. Together, they show that fault and control divergence
are not separated enough to satisfy both operational requirements.

## Runtime and reproducibility

The final CPU run took:

- `0.03 s` for raw low-rank;
- `36.50 s` for supervised residual training; and
- `42.34 s` for JEPA residual training.

CPU JEPA rollout latency was `0.874 ms`, about 4.9× raw low-rank. The composed
model stores 55,030 inference parameters versus 34,503 for raw low-rank.

A second full CPU run used the same seed, data, and configuration. All three
model JSON artifacts were byte-for-byte identical, and all non-timing result
fields matched. Their SHA-256 values were:

- raw low-rank:
  `f340290e7a8ac3322b6365029b13b4401e0f813e43caad3237ffd8d283fb1228`;
- supervised residual:
  `62c5c953b80f1f63a5b3c81f9b1ec270ea1ea4ce439f8abaf28c854a19319c6e`;
- JEPA residual:
  `75936f0d8491e991e4461b38a22c71c2e6b37dd71db09431e40da8ae496601a2`.

This closes the deterministic-execution requirement for the tracer. It does
not substitute for multiple seeds when a candidate approaches promotion
gates.

## Gate outcome

JEPA passed the safety and attribution gates:

- action and overall MSE remained within 5%;
- attribution and specificity were 100%;
- action sanity was 100%; and
- selection chose a nonzero correction.

It failed the gates that establish value:

- no 10% downstream-effect improvement; and
- downstream-effect MSE was worse than the supervised comparator, which itself
  selected no correction.

The trajectory-calibrated divergence detector passed false-alarm control but
failed sensitivity and delay.

## What this changes

The experiment cleanly separates two conclusions:

1. **Keep raw low-rank as the predictive core.** It remains faster, smaller,
   and more accurate for the claim we care about.
2. **Do not add this residual branch.** JEPA preserved attribution but added no
   predictive or detector value; direct residual learning did not even pass
   selection.

The next high-value step toward a bounded world model is not another JEPA loss
sweep on the same targets. Enrich the state with variables the residual may
depend on—per-node saturation, queue-age distributions, trace-derived edge
timing, recovery actions, and fault intensity—then first test whether a simple
supervised structured model finds a materially predictable residual. JEPA
should return only if that richer corpus contains masked or multimodal
structure a direct model cannot exploit.
