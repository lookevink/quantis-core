# Richer-regime alerting retry v1 results

## Conclusion

**The multi-hypothesis selection retry is methodologically inconclusive. Do
not use the replacement selection corpus to accept or reject that candidate,
and do not collect calibration or evaluation from it.**

The stored model outputs are reproducible and would reject the four-component
JEPA: it lost to a one-component JEPA and badly regressed the raw rank-32
predictor. A protocol-aware post-run audit found that all 135 pairs satisfy
the frozen action-specific effect and recovery rules. The campaign remains
scientifically inadmissible for a different reason:

- the v2 selection manifests do not bind the recollection amendment, so the
  one-recollection ceiling and whole-shard replacement rule were documentary
  rather than execution-enforced.

This is a replacement-selection provenance failure, not a model failure. The
v1 fit-only screening remains usable within its stated mechanism-routing
boundary. The earlier narrow-corpus multi-hypothesis result remains rejected
on its own evidence; this retry does not strengthen or reverse it.

## Evidence collected

The retained v1 fit corpus contains 90 matched pairs / 180 captures:

- 30 steady pairs;
- 30 ramp/burst pairs; and
- 30 periodic/multiphase pairs.

All three fit shards passed capture count, pair count, protocol binding, plan
binding, file presence, and non-empty telemetry gates. The post-run audit
subsequently recomputed the frozen action-specific effect/recovery gates from
the raw captures.

The v2 selection corpus contains 45 matched pairs / 90 captures, one pair per
action×topology×regime cell. All three selection shards passed the same gates.
No calibration or evaluation shard was collected.

Local artifacts:

- `artifacts/action-dynamics/richer-regime-retry-v1`
- `artifacts/action-dynamics/richer-regime-retry-v2`
- `artifacts/action-dynamics/richer-regime-multi-hypothesis-jepa-v1`
- `artifacts/action-dynamics/richer-regime-retry-v1-validity-audit-v4`

## Post-run validity audit

The final protocol-aware audit uses the development action protocol's median
active-effect and median recovery-ratio rules, including action-specific
recovery windows, API count resolution, and enqueue drain/probe and
mechanistic gates. It also
content-addresses the applicable action protocol and every consumed campaign,
manifest, capture, attestation, schema, amendment, and telemetry file.

| Role | Valid pairs | Failed pairs | Failure rate |
|---|---:|---:|---:|
| Fit | 90 | 0 | 0.00% |
| Selection | 45 | 0 | 0.00% |

An initial generic-validator audit is retained separately but superseded: it
treated the protocol's recovery ratio as an absolute raw-unit tolerance and
therefore overcounted failures. It is not cited as scientific evidence.

The amendment's three referenced file hashes are correct, but none of the 90
replacement selection manifests contains the amendment hash. The recollection
policy therefore remained documentary rather than execution-bound.

```bash
PYTHONPATH=src .venv/bin/python \
  lab/action_dynamics/audit_richer_regime_validity.py
```

## Fit-only preflight

The preflight used replicate 0 for diagnostic fitting and replicate 1 as a
probe. Selection, calibration, and evaluation were not opened.

| Measurement | Result | Route |
|---|---:|---|
| Regime classification accuracy | 86.67% | below 90% gate |
| Contextual MSE ratio | 0.9968 | no contextual JEPA retry |
| Incremental event-context MSE ratio | 1.0000 | no HEPA retry |
| Residual variance ratio | 1.1182× | no Error-Certificate retry |
| Pooled two-cluster residual SSE reduction | 59.81% | routed retry |

The imperfect regime classification is expected in part: API rejection uses a
fixed 12-request schedule, while Redis enqueue delay preserves a common
drain/probe tail. It is a failed contextual-mechanism gate, not a data-quality
failure.

The pooled clustering statistic mixes workload, topology, and time-phase
heterogeneity. It justified a screening run under the protocol, but it is not
evidence of conditional multimodality and needs a trajectory-held-out,
conditioning-stratified null before reuse.

## Stored selection diagnostic (scientifically inadmissible)

Lower log score and MSE are better.

| Model | Log score | Overall MSE | Action-overlap MSE | Supported pair rate |
|---|---:|---:|---:|---:|
| Raw rank-32 low-rank | **-3.0353** | **0.1452** | **0.4091** | 0.0% |
| One-component JEPA | -0.4917 | 0.2492 | 1.3283 | 0.0% |
| Four-component JEPA | -0.4787 | 0.2692 | 1.2698 | **23.36%** |
| Capacity-matched single Gaussian | -0.4876 | 0.2786 | 1.4961 | 0.0% |

Within the invalid corpus, 23.36% of action-overlap samples supported at least
two non-negligible separated components:

- candidate log score was worse than one-component JEPA by 0.01294 rather than
  better by the required 0.01;
- candidate overall MSE was 1.85× raw; and
- candidate action-overlap MSE was 3.10× raw.

The stored candidate fails three numeric gates independently of the supervised
mixture null. Because the replacement selection corpus was not execution-bound
to the recollection amendment, those failures cannot be promoted to a
scientific model decision.

The standalone stored-array assessor verifies every artifact-manifest hash and
independently recomputes the valid-model metrics and frozen gates:

```bash
PYTHONPATH=src .venv/bin/python \
  lab/action_dynamics/assess_richer_regime_multi_hypothesis.py
```

The numeric safe-null rule and model recipe predate this campaign in commit
`3da2d563c4bbfb86f5c082030ba374075a9a34b4`. At that commit, the scoring
contract SHA-256 was
`118f2dad09c62950e1762a4f71c48da43c046e24493be32bc04411aa2c357a8c`
and the prototype runner SHA-256 was
`0ae1b3a90f459cfda9fad9c4899af2b8bcf3f3e2521a4a34222b75f8be7ae103`.
This supports gate provenance, but it does not repair the missing amendment
binding.

## Invalid supervised null

The stored supervised four-component mixture hit the strict distribution
validator at selection. Its outputs were finite, variances were finite, and
weight sums differed from one by at most `1.19e-7`. The minimum float32 weight
was `9.99999996e-13`, which falls marginally below the required `1e-12` floor
after conversion.

This makes that null unusable under the frozen contract. It does not alter the
stored diagnostic because the one-component and raw gates fail independently.
No model was retrained during diagnosis.

## Operational incident

The first v1 steady-selection collection produced one partial case: an initial
metric window and clean shutdown boundary, before its scheduled intervention.
The collector then discarded the captured subprocess output on nonzero exit.

The partial attempt remains retained and excluded from model input. A
regression test now requires failed runner output to be written to
`runner.log`. The v2 operational amendment documented one whole-shard
recollection in a new directory, but the collection runner and replacement
manifests did not cryptographically bind that amendment.

## Artifact identity

- Failure diagnosis SHA-256:
  `0a8d04ab34a11162d1d295e19f808b0eac58b1195025ae9516c7f51af80e4f2e`
- Artifact manifest SHA-256:
  `084a5d45d85f310358bacafbf5fba1736aa0649815ff1f21d133ec88f17f42b3`
- Protocol-aware validity-audit artifact manifest SHA-256:
  `ce8cef807ba1feb557465b2868e8f9a5e955994211800f192fe424c021671c80`
- Consumed-source manifest SHA-256:
  `ae797d3f8075b21dc37fd931d6fbc6db796f359072a0ffe4443860622c1e40db`

## What changed our ranking

The fit-only screen did not justify contextual multimodal JEPA, HEPA, or
Error-Certificate-JEPA under their frozen mechanism gates. Multi-hypothesis
JEPA earned selection, but its stored selection failure is only a diagnostic.
Pair realization is not the blocker after protocol-aware recomputation; the
unbound recollection prevents treating the replacement selection corpus as
preregistered evidence.

The next campaign must make protocol-aware action realization/recovery a shard
qualification gate, bind any recollection amendment into every replacement
manifest and attestation, and use trajectory-held-out conditional
multimodality screening. Until then, the richer-regime question remains open.
