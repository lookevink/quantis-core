# Discrete-JEPA telemetry tracer v1 results

## Decision

**Reject this exact Discrete-JEPA telemetry recipe. Do not run seed
robustness or sealed confirmation.**

The hard semantic codebook collapsed to one code per entity, the complete
S2P + P2S + P2P cell had no next-code advantage over the P2P-only control,
and every learned representation failed the frozen raw forecast-safety
margin. This rejects the fixed entity-token VQ translation on the Quantis
lab stack. It does not reject Discrete-JEPA on the paper's image tasks.

## Reproducible evidence

- implementation commit:
  `bfb69d4cc9132b467468365314c7dd049ce9e61a`;
- conclusion-bearing immutable artifact:
  `artifacts/action-dynamics/prototype-discrete-jepa-v1`;
- artifact-manifest SHA-256:
  `9fb5566463c1039617b0d16b437093f6ababd0ba59dd9adc4f62070bb211529c`;
- 800 steps for each of three equal-capacity neural cells; and
- 40 fitting, 10 selection, 10 calibration, 20 IID evaluation, and 10
  held-topology evaluation pairs.

The 468 MiB artifact contains 48 files: all model and control states, the
loadable inference bundle, exact schedules, predictions, full-role
diagnostics, semantic indices and restoration evidence, 100 raw latency
samples, an isolated copied-source assessor, independent assessment, report,
and identity manifest.

## Held-topology result

| representation | overall MSE | action-overlap MSE | downstream-effect MSE |
|---|---:|---:|---:|
| raw rank-32 | 0.105744 | 0.859940 | 0.143833 |
| Discrete-JEPA complete | 0.150505 | 1.541348 | 0.275837 |
| continuous complete | 0.166416 | 1.407938 | 0.291032 |
| discrete P2P-only | 0.150505 | 1.541348 | 0.275837 |
| matched PCA | 0.147181 | 1.444091 | 0.274703 |

The candidate retained `1.92×` raw downstream-effect error, was identical to
the hard P2P-only control on every reported forecast score, and beat the best
transfer control on only three of ten pairs. Its selection effect MSE
`0.183807` was also worse than raw `0.128783`.

## Mechanism result

| mechanism | selection | transfer | requirement |
|---|---:|---:|---:|
| code perplexity | 5.7423 | 5.7423 | at least 8 |
| active codes | 6 | 6 | noncollapsed |
| active codes per entity | 1 each | 1 each | at least 2 for varying entities |
| next-code accuracy | 1.0000 | 1.0000 | candidate at least control + 0.05 |
| P2P-only next-code accuracy | 1.0000 | 1.0000 | control |

The complementary heads were learned and P2P was preserved, but semantic
quantization encoded entity identity rather than varying regimes. Perfect
next-code accuracy was therefore trivial and exactly tied the P2P-only
control.

## Evidence and edge behavior

All ten independently recomputed protocol checks passed, including exact
role cardinalities and disjointness, frozen anchor and mask schedules,
selection-only ridge choice, matched neural capacity, causal public
inference, copied-source isolation, bundle identity, and raw latency
retention.

The three neural cells each used 125,504 training and 75,136 inference
parameters. The exact inference bundle was 1,982,333 bytes. Its 100 retained
batch-one CPU samples yielded 35.52 ms median and 39.54 ms p95.

Artifact reassessment was exact, but the original/restored representation,
index, probe, and bundle replay lane reached maximum absolute differences of
`3.55e-6` for the candidate and `7.61e-6` across all controls, above the
frozen `1e-6` deployment-fidelity threshold. This additional safety failure
does not affect the conclusion: code collapse, raw forecast regression, and
all value gates independently reject the recipe.

## Consequence

Discrete-JEPA is closed on this stack. Proceed to PEIRA as the next bounded
predictor-free objective omission. Do not carry the collapsed hard codebook
into that tracer.
