# MoP-JEPA hard-assignment tracer v1 result

## Outcome

**Reject this edge telemetry MoP-JEPA recipe.**

Hard assignment was active and produced several observable winners. The
candidate also passed the frozen realized-transition precision gate and beat
dense JEPA on selection proper score. It nevertheless lost to supervised
hard-WTA, failed the context-free codebook and raw safety controls, and
regressed held-topology downstream-effect prediction. Do not advance to
restart selection, additional seeds, sealed confirmation, or alerting-system
integration.

This rejects one compact hard-assigned latent trajectory recipe on the
current largely deterministic corpus. It does not reject MoP-JEPA on
stochastic planning data, where several valid successors are observed or an
environment can verify proposed transitions.

## Evidence identity

- Implementation commit:
  `973aa283d88a856e82096aaf59dd3480170f9fef`
- Official artifact:
  `artifacts/action-dynamics/prototype-mop-jepa-v1`
- Artifact-manifest SHA-256:
  `fc88122515a688ab9c30a8361ef4767f6def43571f521c2c24af8961b3219a1e`
- Independent assessment SHA-256:
  `6e5766e1def22b07bf1d3402fe91281d394943eb9c5fa01d45dc7782b1c36d2c`
- Result SHA-256:
  `e22c7313a9762a2b7ea34e0109df3d42c8c8b18c515726a3264a710e7bde9acc`

The fresh assessor reproduced the stored decision and verified every manifest
entry. The 885 MiB bundle retains the candidate, dense JEPA, supervised
hard-WTA, context-free codebook, raw rank-32 model, calibration sidecars,
complete component predictions and weights, shuffled-context predictions,
restoration outputs, exact source snapshot, and assessment.

Three non-interpretable one-epoch smoke bundles are also retained. The first
two preserve pre-review apparatus states; the final qualifying smoke is
`prototype-mop-jepa-v1-smoke-final-1epoch-1pair`.

## Frozen recipe

The official matrix used seed `19019`, 40 epochs, eight candidate heads,
latent width 12, predictor width 128, target EMA `0.996`, cosine
trajectory-level hard assignment, and a context-only router. The supervised
control replaced only the winner loss with observable MSE. Observable
variances and the transition radius used calibration only.

The input-free eight-center codebook used farthest-first initialization and
20 Lloyd iterations. The paper's router threshold `pi_k > 0.5 / K` applied
only to predictor cells; every codebook component remained active.

## Proper score and observable prediction

Lower is better.

| selection cell | mixture NLL | point MSE | action MSE | effect MSE |
|---|---:|---:|---:|---:|
| raw rank-32 | **-3.092040** | **0.093968** | **0.313088** | **0.128783** |
| context-free codebook | -2.926213 | 0.833400 | 6.535213 | 0.672692 |
| supervised hard-WTA | -1.486712 | 0.679581 | 5.553284 | 0.594984 |
| MoP-JEPA | -0.797380 | 0.795826 | 6.245962 | 0.706847 |
| dense JEPA | -0.694080 | 0.735465 | 5.938351 | 0.623785 |

MoP-JEPA improved dense JEPA NLL by `0.103300` nats per coordinate, clearing
that one value gate. It was `0.689332` worse than supervised hard-WTA and
`2.294661` worse than raw. Its selection point MSE was `8.47` times raw and
its action-overlap MSE was `19.95` times raw.

| transfer cell | mixture NLL | point MSE | action MSE | effect MSE |
|---|---:|---:|---:|---:|
| raw rank-32 | **-3.072536** | **0.105744** | **0.859940** | **0.143833** |
| context-free codebook | -2.907992 | 0.568713 | 5.062522 | 0.673025 |
| supervised hard-WTA | -1.242116 | 0.471128 | 4.313583 | 0.603264 |
| dense JEPA | -0.684915 | 0.507737 | 4.642389 | 0.637280 |
| MoP-JEPA | -0.552849 | 0.582016 | 4.865530 | 0.668005 |

The candidate's transfer point and action MSE were `5.50` and `5.66` times
raw. Its transfer effect error was `4.65` times raw, and it beat raw on only
40% of matched held-topology pairs.

## Hard-assignment mechanism

| selection diagnostic | result | gate |
|---|---:|---:|
| observable winner effective heads | 2.224 | at least 2: pass |
| router effective heads | 1.406 | at least 1.5: fail |
| mean active router heads | 1.668 | diagnostic |
| gated realized-transition precision | 90.13% | at least 80%: pass |
| candidate oracle MSE | 0.794797 | improve dense point by 10%: fail |
| candidate gated-oracle MSE | 0.794888 | improve codebook 0.471948 by 10%: fail |
| shuffled-context oracle MSE | 0.882312 | improve by 10%: fail |

Correct context improved oracle error over shuffling by `9.92%`, narrowly
below the frozen 10% requirement. More importantly, its conditional candidate
set was worse than the static codebook and even its best component was worse
than dense JEPA's point prediction. The extra heads therefore specialized,
but not into useful telemetry successors.

On calibration, latent hard assignment used five heads with counts
`[0, 840, 41, 0, 0, 255, 412, 32]`. This is not the previous
likelihood-responsibility collapse, but active specialization alone did not
earn observable or operational value.

## Edge feasibility

The candidate had:

- 1,002,795 inference parameters;
- a 3,781,797-byte serialized model;
- 0.241 ms mean and 0.257 ms p95 batch-one CPU latency over 100 repetitions;
  and
- full restoration within `1e-6`.

The whole training, prediction, compression, and assessment process peaked at
6.40 GiB RSS; that is not an inference-memory measurement. The candidate is
edge-feasible in isolation, but feasibility cannot rescue failed safety and
value gates.

## Conclusion

MoP-JEPA resolves a narrow earlier omission: Quantis had tested a
likelihood-trained mixture but not hard winner specialization. The hard
mechanism genuinely activated, yet the current telemetry corpus rewarded a
static codebook and compact raw dynamics much more strongly. There is no
evidence of irreducible conditional multimodality sufficient to justify
variational, diffusion, or belief-state successors next.

The complete runnable frontier queue from the July 2026 audit is now
concluded. Proceed to the cross-experiment deployment and omission synthesis,
not another JEPA trainer on this one-stack corpus.
