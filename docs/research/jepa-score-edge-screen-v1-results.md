# Exact JEPA-SCORE edge screen v1 results

## Decision

**Reject this exact JEPA-SCORE edge-alert recipe. Do not run seed robustness
or sealed confirmation.**

The source paper's exact Appendix-B Jacobian singular-value score is practical
on the target CPU path, but it is not a useful alert score for the frozen
complete-LeJEPA representation. All 13 protocol gates and all eight
edge-safety gates passed. All five value gates failed.

This is a narrower result than “JEPA-SCORE does not work.” It rejects one
action-blind, single-transform Monte Carlo score over the frozen
complete-LeJEPA projector on the current telemetry contract. It also resolves
the open latency question: exact Jacobian plus full SVD is feasible at this
model size.

## Frozen evidence

- Implementation commit:
  `b12b3b6d040729b2b2479b94ad251174cd316c44`
- Artifact:
  `artifacts/action-dynamics/prototype-jepa-score-v1`
- Artifact manifest SHA-256:
  `e678101945c3b99cd325e003f23fdbef334c09ef29ef68f89220cc244012ed86`
- Complete-LeJEPA source manifest SHA-256:
  `00639afaee81cd3844e8b60014ed87f886a8e7a7e0b20bc50834416da1deb265`
- Preprocessing-cache manifest SHA-256:
  `525bd7e68b47336fad8eb0c39c0d93b0e99a7a80c0682119be3626d6066a3fa8`
- Primary scorer payload:
  116,848 parameters, 2,438,268 bytes, SHA-256
  `95e318f107f703c4f9600bbfda0dbbe53ecbaade3a0717b97a61c4a660a69a2d`
- Evidence boundary:
  single-seed open-development exact single-transform Monte Carlo density
  screen; not production or sealed evidence.

The retained artifact contains the canonical scorer, every scored row and
singular-value vector, source and role receipts, latency inputs and receipt,
full source closure, independent assessor, result, report, and manifest. Four
earlier failed smoke builds are also retained under
`artifacts/action-dynamics/prototype-jepa-score-v1-failed-smoke-*`; none has
scientific authority.

## Alert result

The frozen policy calibrated each method separately on the maximum of five
fixed anchors for each of ten IID control trajectories, using a strict
greater-than rule.

| method and role | control false alarms | pre-onset alerts | treatment detection | median detected delay |
|---|---:|---:|---:|---:|
| exact JEPA-SCORE, IID | 5% | 0% | 10% | 31.5 transitions |
| exact JEPA-SCORE, transfer | 0% | 0% | 0% | not applicable |
| raw terminal-delta score, IID | 0% | 0% | 55% | 6 transitions |
| raw terminal-delta score, transfer | 30% | 0% | 50% | 7 transitions |

JEPA-SCORE's selection pair-win fraction was `0.40`, below the frozen `0.60`
gate. Its anomaly threshold was `445.188416`; the raw threshold was
`4.859370`.

The candidate's clean transfer false-alarm behavior is not enough to make it
useful: it emitted no transfer treatment alerts. It therefore neither Pareto
dominated raw nor produced the required material transfer improvement.

## Exact-score and edge result

The scorer differentiated
`projector(visible_mean(backbone_T(x * ownership)))` with the ownership mask
inside the differentiated function, computed the full float32 Jacobian and
SVD, clipped singular values at `1e-6`, and summed their logarithms. The
independent assessor reconstructed the route literally and recomputed every
stored score.

| property | result | gate |
|---|---:|---:|
| median scorer latency | 51.359 ms | at most 100 ms |
| p95 scorer latency | 60.195 ms | at most 125 ms |
| incremental peak RSS | 41,041,920 bytes | diagnostic |
| scorer bundle | 2,438,268 bytes | at most 8 MiB |
| parameters | 116,848 | at most 120,000 |
| unowned Jacobian maximum | 0 | exactly 0 |
| complete-LeJEPA effective rank | 4.008 | diagnostic |
| clipped singular values, 500 rows | 52 | diagnostic |

Batch/single parity passed for complete LeJEPA, SIGReg-only, and
invariance-only controls. The invariance-only control clipped 1,000 singular
values and had only `4.90e-7` mean marginal variance, while the complete model
had substantial local variation. The negative alert result is therefore not
explained by a wholly constant primary score path.

## Interpretation

JEPA-SCORE answered a genuinely different question from the prediction and
representation tracers: whether local encoder density could expose drift even
when downstream prediction was weak. On this representation, treatment and
control score distributions overlap too strongly. The technique cannot repair
information that the fitted representation did not organize around the alert
boundary.

The useful retained result is operational: exact Jacobian/SVD density scoring
is small and fast enough for this edge budget. It may be reconsidered for an
offline drift or debugging lane if a future encoder first earns promotion and
local-density separation is demonstrated on selection data. It should not be
added to the current paging stack.

## Queue disposition

The materially distinct, currently runnable JEPA queue is now exhausted under
the present one-stack prerequisites. Keep rank-32 raw action-conditioned
dynamics and direct raw/PCA retrieval as the shadow-system baselines. Open a
new JEPA experiment only when a recorded prerequisite creates a different
hypothesis—for example, an explicit missing-channel lane, represented
recovery trajectories, a trusted physical invariant, or demonstrated
heteroscedastic/multimodal residuals.
