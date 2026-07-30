# Multi-hypothesis trajectory JEPA prototype v2 result

> **2026 richer-regime retry:** the recipe was retried after a fit-only
> preflight found strong residual clustering across steady, ramp/burst, and
> periodic/multiphase demand. It again failed independent selection: overall
> MSE was 1.85× raw and action-overlap MSE was 3.10× raw. See the
> [richer-regime result](richer-regime-alerting-retry-v1-results.md)
> and [frozen protocol](../specs/richer-regime-alerting-retry-v1.md).

## Outcome

**Reject this four-component recipe at safe-null selection. Do not calibrate,
run value-lane assessment, or advance to multiple seeds.**

The candidate produced distinct, non-negligible hypotheses on some
action-overlap windows, but they did not improve proper score over the
supervised mixture or preserve the raw low-rank model's point prediction.
The frozen safe-null rule selected `raw_low_rank`.

Alert and investigation value were not assessed. This is a prerequisite
selection failure, not evidence that the candidate was tried in those lanes
and failed.

## Corrected evidence boundary

The first result bundle under
`artifacts/action-dynamics/prototype-multi-hypothesis-jepa-v1` is preserved but
invalid as a decision artifact. Review found that transfer finiteness could
enter its selection conjunction and that it stopped before outputs promised by
the v1 prototype document.

The
[v2 correction](../specs/multi-hypothesis-jepa-prototype-v2.md) binds that
complete numeric bundle by manifest and result SHA-256, verifies its frozen
seed, model, data, role, and preprocessing identities, and recomputes a
selection-only, pair-balanced decision. The corrected evidence is under
`artifacts/action-dynamics/prototype-multi-hypothesis-jepa-v2`.

This remains one deterministic seed over already-open development data. It is
not sealed confirmation or evidence for production paging.

## Selection result

Lower log score and MSE are better. Log score is exact complete-trajectory
mixture NLL, averaged within each trajectory and then within matched pairs.

| Model | Log score | Overall MSE | Action-overlap MSE | Supported pair | Effective hypotheses |
|---|---:|---:|---:|---:|---:|
| Raw rank-32 low-rank | **-3.083745** | **0.093968** | **0.295008** | 0.00% | 1.000 |
| Capacity-matched single Gaussian | 0.086398 | 0.212367 | 1.376714 | 0.00% | 1.000 |
| Supervised four-component mixture | **0.079876** | 0.218743 | 1.275322 | 15.18% | 1.076 |
| One-component JEPA | 0.091419 | 0.206351 | 1.307067 | 0.00% | 1.000 |
| Four-component JEPA | 0.090551 | 0.223560 | 1.294265 | **24.51%** | **1.119** |

The candidate:

- improved over one-component JEPA by only `0.000868` nats per observed
  coordinate, below the required `0.01`;
- was `0.010676` nats per coordinate worse than the supervised
  four-component mixture;
- had `2.379x` raw overall MSE; and
- had `4.387x` raw action-overlap MSE.

It passed only component support and finite-output selection gates. Proper
score versus both controls and both point-prediction safety gates failed.

## Held-out topology diagnostic

Transfer did not influence selection. It is retained only to show that the
selection failure was not a favorable in-distribution accident.

| Model | Log score | Overall MSE | Action-overlap MSE | Supported pair | Effective hypotheses |
|---|---:|---:|---:|---:|---:|
| Raw rank-32 low-rank | **-3.032492** | **0.105744** | **0.551154** | 0.00% | 1.000 |
| Capacity-matched single Gaussian | **0.050655** | 0.167946 | 1.007013 | 0.00% | 1.000 |
| Supervised four-component mixture | 0.113642 | 0.209019 | 1.240987 | 22.69% | 1.079 |
| One-component JEPA | 0.074327 | 0.182310 | 1.142911 | 0.00% | 1.000 |
| Four-component JEPA | 0.148357 | 0.222989 | 1.224004 | **27.73%** | **1.095** |

The candidate remained worse than every fitted probabilistic control on
transfer log score. Its transfer overall and action-overlap MSE were `2.109x`
and `2.221x` raw.

## Edge feasibility

All five models restored with prediction parity. The candidate used 569,707
inference parameters, a 2,131,368-byte compressed model artifact, and
0.228 ms median batch-one latency in the local deterministic CPU
microbenchmark. It fits the preregistered size envelope, but edge feasibility
without scientific value is not a promotion argument.

## Interpretation

The mixture mechanism did not merely duplicate four identical outputs:
supported pairs appeared on roughly one quarter of action-overlap windows.
However, an effective hypothesis count near `1.1` shows that most probability
still concentrated on one component, and the added alternatives did not earn
better likelihood than an ordinary supervised mixture.

The responsibility-weighted JEPA target therefore added no measurable
probabilistic advantage. More importantly, every neural variant remained far
behind the compact raw low-rank transition, whose strongly negative log score
reflects both substantially better means and tight fitted residual variance on
this deterministic corpus.

This result rejects:

> A direct four-head, responsibility-weighted latent trajectory JEPA with a
> shared observable decoder and global diagonal component variances.

It does not reject mixture forecasting generally. In particular, it does not
test discrete stochastic interventions absent from the current corpus,
autoregressive mode persistence, retrieval-conditioned hypotheses, or
generative latent residuals.

## Artifact identity

- Corrected manifest SHA-256:
  `aa47dd6b28dbe31ec99ccb908296a2a4f66a9ba3cf2a12299894d38e296f14a9`
- Corrected assessment SHA-256:
  `2b420e6c82a3349310a72154c905ecbb3264edeef4fa636b59df2262c4f4abd0`
- Bound v1 manifest SHA-256:
  `1a464d6182b4f0abd6987496453ef5f9ef403d9ab62779ffa87e7511184528f8`
- Bound v1 result SHA-256:
  `295ac75bbff1f85f3cb72833b11e6543fb082a5e027e9f20f814ff529a6c1760`
