# LeNEPA telemetry tracer v1 results

## Decision

**Reject this exact LeNEPA telemetry recipe. Do not run seed robustness or
sealed confirmation.**

The disposable projected prediction space did not improve the matched
unprojected next-latent control on the frozen mechanism gate, and every
learned representation failed the raw forecast-safety gate. This rejects the
causal projected-next-latent plus temporal-SIGReg translation on the fixed
Quantis lab stack. It does not reject LeNEPA on the paper's time-series tasks.

## Reproducible evidence

- implementation commit:
  `eaa8ceba8b08db502d81809c3ceeb432b052ca68`;
- conclusion-bearing immutable artifact:
  `artifacts/action-dynamics/prototype-lenepa-jepa-v1`;
- preserved aborted pre-review staging artifact:
  `artifacts/action-dynamics/prototype-lenepa-jepa-v1-aborted-ec87ebb`;
- artifact-manifest SHA-256:
  `fc7658feef17fb0c309bf8a595fc185863d4e1f12f12e57783fda0853425553c`;
- 1,600 steps for each of three equal-capacity neural cells; and
- 40 fitting, 10 selection, 10 calibration, 20 IID evaluation, and 10
  held-topology evaluation pairs.

The 487 MiB artifact retains all models and projectors, PCA and raw controls,
the loadable deployment bundle, anchor schedule, predictions, full-role
diagnostic tensors, restoration evidence, 100 raw latency samples, copied
reproduction sources, independent assessment, report, and identity manifest.

## Held-topology result

| representation | overall MSE | action-overlap MSE | downstream-effect MSE |
|---|---:|---:|---:|
| raw rank-32 | 0.105744 | 0.859940 | 0.143833 |
| projected LeNEPA | 0.145600 | 1.435773 | 0.276266 |
| unprojected LeNEPA | 0.143395 | 1.441812 | 0.271797 |
| projected SIGReg-only | 0.145589 | 1.425890 | 0.275094 |
| matched PCA | 0.147181 | 1.444091 | 0.274703 |

The projected candidate retained `1.92×` raw downstream-effect error and was
worse than the unprojected control. It beat that control on only two of ten
transfer pairs. No representation had a raw-safe selection ridge.

## Mechanism result

| role | projected cosine | unprojected cosine | projected retrieval | unprojected retrieval |
|---|---:|---:|---:|---:|
| selection | 0.402790 | 0.412994 | 0.000600 | 0.001599 |
| transfer | 0.400692 | 0.416746 | 0.000666 | 0.001299 |

Projection reduced cosine error by only 2.47% on selection and 3.85% on
transfer, short of the frozen 10% requirement. Its aligned retrieval was
also lower, not ten percentage points higher. The projected prediction
advantage gate therefore failed.

## Evidence and edge behavior

All twelve protocol checks recomputed true:

- role identifiers and the pair-blocked anchor schedule were valid;
- full selection/transfer mechanism coverage and shifted diagnostic tensor
  identities recomputed;
- all three neural cells matched at 411,072 inference and 612,352 training
  parameters;
- public inference was causal and prefix invariance was exact;
- retained models, PCA, diagnostics, probes, and deployment replay restored
  within `2.4e-7`;
- selection ridge choice and no-safe-ridge status recomputed;
- the exact deployment payload matched and loaded without projector state;
  and
- the 6,204,083-byte bundle was below 16 MiB, with 100 retained local
  batch-one CPU samples yielding 173.69 ms median and 205.54 ms p95.

Forecast-safety, mechanism, and all value gates failed. These are scientific
failures rather than evidence-contract failures.

## Consequence

Exact LeNEPA is closed on this stack. Proceed to Discrete-JEPA as the next
bounded objective omission. Do not carry LeNEPA's projector, SIGReg loss, or
causal backbone into that tracer except as explicitly frozen controls.
