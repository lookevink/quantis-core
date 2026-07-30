# Mean-preserving residual mixture JEPA v1 result

## Outcome

**Reject the exact MPRM-JEPA recipe.**

The experiment was valid. All 90 fresh matched pairs / 180 captures completed
without retry, the complete campaign qualified, all six frozen models restored
with exact fresh-process prediction parity, and the independent stored-array
assessor reproduced the primary decision exactly.

Mean preservation worked as designed, and the candidate passed the compact
edge envelope and minimum supported-pair-rate gate. It nevertheless failed the
two strongest proper-score controls, the workload-family safety gate, and the
paired randomization test. Do not advance this recipe to calibration,
held-out evaluation, or production paging.

This result rejects one four-component, seed-307, 40-epoch anchored JEPA
residual-mixture recipe on the fixed local Docker Compose stack. It does not
reject mean-preserving mixtures in general. The supervised mean-preserving
mixture remains a useful diagnostic, not a promotion candidate under this
contract.

## Evidence identity

- Implementation commit:
  `c149a55a625c74dadd7716ca4bbbd884ccfe5edc`
- Protocol SHA-256:
  `da011fcb850a31395436766bb815ac57fdbd631b9d202723a58d838e8d02f410`
- Model-freeze manifest SHA-256:
  `1a7b356036324ba417439885f18d4e9c56ed2ed1d42e4a81795f9c129d74d736`
- Qualified-corpus identity:
  `3fafc172a1cdce091699b543e3f14068ba4d488fdc6c2ddba73163c3ccda2bb6`
- Prediction-manifest SHA-256:
  `9504e93e2a718406f10199d9878e85a178bd64e207b58ace71af4420f7a33bd7`
- Selection-assessment file SHA-256:
  `55c101dde52984a7ef67e76bd72cb7cc164c165f6061585e9e0d2012a5719a95`
- Official model artifact:
  `artifacts/action-dynamics/mprm-jepa-model-freeze-v1`
- Official selection artifact:
  `artifacts/action-dynamics/mprm-jepa-selection-v1-attempt-001`

The model artifact is 11 MiB and the complete selection artifact is 3.6 GiB.
The independent assessor was invoked with the externally pinned model-freeze,
qualified-corpus, and prediction-manifest identities above. It reproduced the
same decision, gate vector, metrics, prediction hashes, and randomization
`p`-value.

## Execution validity

The run followed the frozen order:

1. all preflight gates passed on the clean implementation commit;
2. fit-role-only evidence fitted and froze all six models;
3. every model passed exact fresh-process prediction parity;
4. the model-freeze identity was bound into a fresh 90-pair campaign;
5. all 180 captures completed across 15 batches and six isolated lanes;
6. the complete campaign qualified before any selection scoring;
7. the primary scorer ran once; and
8. the independent assessor reproduced the stored decision.

The qualified corpus content-addresses the collection attestation, pair
assessment, plan, protocol, every prepared manifest, and every capture's
actions, logs, metrics, traces, runner log, and capture manifest.

## Selection scores

Lower pair-balanced log score, energy score, and MSE are better.

| model | pair-balanced log score | energy score | overall MSE | action MSE | supported-pair rate |
|---|---:|---:|---:|---:|---:|
| raw rank-32 predictive core | **-2.915059** | 0.247841 | 0.283232 | 1.398441 | 0.00% |
| supervised four-component mean-preserving mixture | -0.555661 | **0.240812** | 0.283232 | 1.398441 | **38.73%** |
| MPRM-JEPA candidate | -0.550848 | 0.243420 | 0.283232 | 1.398441 | 20.29% |
| one-component anchored JEPA residual | -0.481696 | 0.249732 | 0.283232 | 1.398441 | 0.00% |
| capacity-matched anchored Gaussian | -0.481696 | 0.249732 | 0.283232 | 1.398441 | 0.00% |
| unanchored four-component JEPA diagnostic | -0.347876 | 0.313634 | 0.407701 | 2.807244 | 22.54% |

The candidate improved the capacity-matched Gaussian and one-component
controls by `0.069152` log-score units, clearing both `0.01` gates. It was
`2.364211` worse than the raw predictive core and `0.004813` worse than the
supervised four-component mixture, so it did not clear either strong-control
gate.

The candidate supported `20.29%` of eligible pairs, narrowly clearing the
frozen `20%` mechanism floor. The paired randomization test returned `p = 1.0`,
far above the required `0.05`, and at least one workload family regressed by
more than `0.01`.

## Structural and edge gates

The weighted predictive mean matched the raw anchor to a maximum absolute
error of `1.4210854715202004e-14`, below the `1e-10` tolerance. Candidate and
raw overall MSE were both `0.2832322497807999`; action-overlap MSE was
`1.3984407109327686` for both. The mean-preserving construction therefore
removed the point-forecast regression exactly as intended.

The unanchored diagnostic regressed overall and action MSE to `0.407701` and
`2.807244`, respectively. This confirms that anchoring solved a real safety
problem, but that structural success did not create enough proper-score value.

The candidate contained 569,707 parameters, serialized to 3,018,000 bytes,
and had 0.458 ms batch-one p95 CPU latency on the frozen Apple M1 Max runtime.
It passed size, latency, no-network/no-accelerator, runtime-identity, finite
score, restoration, mean-identity, MSE, energy-noninferiority, and
supported-pair gates.

## Gate decision

Passed:

- exact mean identity;
- overall and action MSE preservation;
- finite scores and fresh-process parity;
- energy-score noninferiority;
- capacity-matched Gaussian and one-component improvements;
- minimum supported-pair rate;
- frozen runtime, dependency, size, and latency gates.

Failed:

- beat raw rank-32 proper score by at least `0.01`;
- beat supervised four-component proper score by at least `0.01`;
- no workload-family regression over `0.01`; and
- paired randomization `p <= 0.05`.

The frozen conjunction therefore yields
`reject_exact_mprm_jepa_recipe`.

## Conclusion

The experiment answered its narrow question cleanly. Mean-preserving residual
centering is an effective architectural guardrail: it preserved the raw
anchor's point forecast while allowing nonzero mixture support. The JEPA
objective did not turn that safe ambiguity into competitive probabilistic
value. A supervised residual mixture was slightly better, and the compact raw
core remained overwhelmingly better on the primary proper score.

Retain the raw rank-32 predictive core and the complete mechanism diagnostics.
Do not tune or rerun this exposed selection attempt. Any successor must have a
new hypothesis, protocol version, seed, and fresh opaque evidence; it should
explain why it can close the raw-core proper-score gap rather than merely
preserve point MSE.
