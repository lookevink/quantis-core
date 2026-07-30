# Contributing

Quantis is an evidence-producing system. Changes to detection behavior must include:

1. A behavioral test at a public module interface.
2. A reproducible evaluation scenario or worked example.
3. A statement of what the evidence does and does not establish.

Retain the exact non-production runner and any assessor that produced a
reported experiment. "Prototype" means the code is not a supported production
interface; it does not mean the reproduction code is disposable. Treat
published artifact directories as immutable and write every rerun to a fresh
output directory.

Production code lives under `src/quantis_core`. Tests use public interfaces
only and must not depend on private functions, model coefficients, or random
global state. All stochastic behavior accepts an explicit seed.

## Experiment organization

New experiment families live under `experiments/<program>/<technique>`. Each
technique capsule must expose:

- its frozen specification;
- primary references and adaptation notes;
- exact runner and independent assessor when applicable;
- conclusion-bearing findings;
- supported implementation and behavioral tests;
- immutable artifact path; and
- evidence boundary and disposition.

Register comparable JEPA experiments in `experiments/jepa/catalog.json`, then
run:

```bash
python tools/sync_experiment_catalog.py
python tools/sync_experiment_catalog.py --check
```

Do not relocate historical sources merely to improve appearance when their
paths are already bound into published artifact manifests. Use a generated
capsule link as the compatibility adapter. New experiments should place their
orchestration and documentation directly in the capsule from the start.

Cross-experiment comparisons belong in `docs/research`; execution order and
ticket history belong in `docs/wayfinding`; supported reusable behavior
belongs in `src/quantis_core`.

Run `python -m pytest` for the full test suite and
`python -m quantis_core evaluate --output artifacts/evaluation` for the thesis
evaluation.

Run `make check` for the fast catalog, documentation, maintenance-tool type,
and catalog-test checks before committing. Run `make full-check` before
merging changes that affect execution or model behavior.

Changes to OTLP semantics must include an independently worked JSON fixture.
Regenerate Collector evidence with `./lab/otel/run-roundtrip.sh`; never edit
capture, compiled-telemetry, or verification artifacts by hand.

Regenerate instrumented topology evidence with
`./lab/fault/run-fault-lab.sh`. Fault-lab reports must be recomputable from the
raw Collector capture and the checked-in experiment manifest.
