# Richer-regime alerting retry v1

## Purpose

This contract addresses the methodology audit's two highest-priority local
remediations: richer operating regimes and stronger independent replication.
It keeps the deployment stack deliberately small and runnable on the existing
local Docker Compose action-dynamics lab.

The executable protocol is
[`richer-regime-retry-protocol-v1.json`](../../lab/action_dynamics/richer-regime-retry-protocol-v1.json).
The deterministic generator and statistical checks are implemented in
[`richer_regime_retry.py`](../../src/quantis_core/richer_regime_retry.py).

## Frozen factorial

The campaign crosses:

- five reversible action kinds;
- worker topologies 1, 2, and 3;
- steady, ramp/burst, and periodic/multiphase demand; and
- eleven independent matched-pair replicates per cell.

This produces 45 action×topology×regime cells and 495 matched pairs. Every
treatment/control twin shares an explicit request schedule byte-for-byte.

Replicate ownership is frozen before capture:

| Replicates | Role | Pairs | Controls per workload family |
|---|---|---:|---:|
| 0–1 | fit | 90 | 30 |
| 2 | selection | 45 | 15 |
| 3–6 | calibration | 180 | 60 |
| 7–10 | evaluation | 180 | 60 |

At 60 independent controls per family, zero observed false alarms has an exact
one-sided 95% binomial upper bound below 5%. Pair roles cannot be reassigned
after capture.

## Fail-fast retry routing

Only fit evidence may enter the mechanism preflight. Replicate 0 fits
diagnostic predictors and replicate 1 probes them. The preflight routes four
expensive hypotheses:

- contextual multimodal JEPA requires at least a 5% probe-MSE improvement;
- HEPA requires at least a 5% incremental event-context improvement;
- Error-Certificate-JEPA requires at least a 1.5× residual-variance ratio;
- multi-hypothesis JEPA requires at least a 10% two-cluster residual-SSE
  reduction.

Selection is collected only for a routed candidate. Calibration and evaluation
are collected only if every candidate safe-null selection gate passes.

## Collection failure amendment

The first steady-selection attempt stopped before its first intervention after
one runner exited. The attempt remains immutable and is forbidden from model
input. The host collector was corrected to retain captured subprocess output
on failure.

The frozen
[`collection-amendment-v2`](../../lab/action_dynamics/richer-regime-collection-amendment-v2.json)
permits one whole-shard recollection in a new campaign directory because the
failure preceded treatment and any model outcome. It does not change the
scientific protocol, schedules, candidate, or gates.

Post-run protocol-aware audit cleared every pair's action-specific effect and
recovery gates, but found that the replacement manifests did not bind this
amendment. The retained campaign therefore does not satisfy this contract's
scientific intent; see the
[results and validity audit](../research/richer-regime-alerting-retry-v1-results.md).

## Claim boundary

This is local-stack, open-development evidence. A failed exact recipe may be
rejected. No result from this campaign alone authorizes production paging,
cross-site generalization, or rejection of a broad model family.
