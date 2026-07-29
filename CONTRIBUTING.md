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

Production code lives under `src/quantis_core`. Tests use public interfaces only and
must not depend on private functions, model coefficients, or random global state.
All stochastic behavior accepts an explicit seed.

Run `python -m pytest` for the full test suite and
`python -m quantis_core evaluate --output artifacts/evaluation` for the thesis
evaluation.

Changes to OTLP semantics must include an independently worked JSON fixture.
Regenerate Collector evidence with `./lab/otel/run-roundtrip.sh`; never edit
capture, compiled-telemetry, or verification artifacts by hand.

Regenerate instrumented topology evidence with
`./lab/fault/run-fault-lab.sh`. Fault-lab reports must be recomputable from the
raw Collector capture and the checked-in experiment manifest.
