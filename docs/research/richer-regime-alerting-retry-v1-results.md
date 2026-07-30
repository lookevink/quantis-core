# Richer-regime alerting retry v1 results

## Conclusion

**Reject the richer-regime four-component multi-hypothesis JEPA retry. Do not
collect calibration or evaluation evidence for this recipe.**

Richer operating regimes exposed strong residual clustering, but the
four-component JEPA still lost to a one-component JEPA and badly regressed the
raw rank-32 predictor on independent selection evidence.

The result is local-stack, open-development evidence. It rejects this exact
recipe, not mixture forecasting or JEPA generally.

## Evidence collected

The retained v1 fit corpus contains 90 matched pairs / 180 captures:

- 30 steady pairs;
- 30 ramp/burst pairs; and
- 30 periodic/multiphase pairs.

All three fit shards passed capture count, pair count, protocol binding, plan
binding, file presence, and non-empty telemetry gates.

The v2 selection corpus contains 45 matched pairs / 90 captures, one pair per
action×topology×regime cell. All three selection shards passed the same gates.
No calibration or evaluation shard was collected.

Local artifacts:

- `artifacts/action-dynamics/richer-regime-retry-v1`
- `artifacts/action-dynamics/richer-regime-retry-v2`
- `artifacts/action-dynamics/richer-regime-multi-hypothesis-jepa-v1`

## Fit-only preflight

The preflight used replicate 0 for diagnostic fitting and replicate 1 as a
probe. Selection, calibration, and evaluation were not opened.

| Measurement | Result | Route |
|---|---:|---|
| Regime classification accuracy | 86.67% | below 90% gate |
| Contextual MSE ratio | 0.9968 | no contextual JEPA retry |
| Incremental event-context MSE ratio | 1.0000 | no HEPA retry |
| Residual variance ratio | 1.1182× | no Error-Certificate retry |
| Two-cluster residual SSE reduction | 59.81% | multi-hypothesis retry |

The imperfect regime classification is expected in part: API rejection uses a
fixed 12-request schedule, while Redis enqueue delay preserves a common
drain/probe tail. It is a failed contextual-mechanism gate, not a data-quality
failure.

## Selection result

Lower log score and MSE are better.

| Model | Log score | Overall MSE | Action-overlap MSE | Supported pair rate |
|---|---:|---:|---:|---:|
| Raw rank-32 low-rank | **-3.0353** | **0.1452** | **0.4091** | 0.0% |
| One-component JEPA | -0.4917 | 0.2492 | 1.3283 | 0.0% |
| Four-component JEPA | -0.4787 | 0.2692 | 1.2698 | **23.36%** |
| Capacity-matched single Gaussian | -0.4876 | 0.2786 | 1.4961 | 0.0% |

The multi-hypothesis mechanism was real: 23.36% of action-overlap samples
supported at least two non-negligible separated components. It was not useful:

- candidate log score was worse than one-component JEPA by 0.01294 rather than
  better by the required 0.01;
- candidate overall MSE was 1.85× raw; and
- candidate action-overlap MSE was 3.10× raw.

Thus the candidate fails three selection gates independently of the supervised
mixture null.

The standalone stored-array assessor verifies every artifact-manifest hash and
independently recomputes the valid-model metrics and frozen gates:

```bash
PYTHONPATH=src .venv/bin/python \
  lab/action_dynamics/assess_richer_regime_multi_hypothesis.py
```

## Invalid supervised null

The stored supervised four-component mixture hit the strict distribution
validator at selection. Its outputs were finite, variances were finite, and
weight sums differed from one by at most `1.19e-7`. The minimum float32 weight
was `9.99999996e-13`, which falls marginally below the required `1e-12` floor
after conversion.

This makes that null unusable under the frozen contract. It does not rescue the
candidate because the one-component and raw gates reject it independently. No
model was retrained during diagnosis.

## Operational incident

The first v1 steady-selection collection produced one partial case: an initial
metric window and clean shutdown boundary, before its scheduled intervention.
The collector then discarded the captured subprocess output on nonzero exit.

The partial attempt remains retained and excluded from scientific input. A
regression test now requires failed runner output to be written to
`runner.log`. The v2 operational amendment authorized one whole-shard
recollection in a new directory without changing scientific choices.

## Artifact identity

- Failure diagnosis SHA-256:
  `0a8d04ab34a11162d1d295e19f808b0eac58b1195025ae9516c7f51af80e4f2e`
- Artifact manifest SHA-256:
  `084a5d45d85f310358bacafbf5fba1736aa0649815ff1f21d133ec88f17f42b3`

## What changed our ranking

Before collection, contextual multimodal JEPA was the leading JEPA retry.
Fit-only evidence falsified its mechanism gate and did the same for HEPA and
Error-Certificate-JEPA. Multi-hypothesis JEPA alone earned selection, then
failed there.

The practical conclusion is stronger than the prior narrow-corpus result:
richer local demand regimes do not rescue this multi-hypothesis recipe, while
the raw low-rank baseline remains the model to beat.
