# Contributing

Quantis is an evidence-producing system. Changes to detection behavior must include:

1. A behavioral test at a public module interface.
2. A reproducible evaluation scenario or worked example.
3. A statement of what the evidence does and does not establish.

Production code lives under `src/quantis_core`. Tests use public interfaces only and
must not depend on private functions, model coefficients, or random global state.
All stochastic behavior accepts an explicit seed.

Run `python -m pytest` for the full test suite and
`python -m quantis_core evaluate --output artifacts/evaluation` for the thesis
evaluation.

Changes to OTLP semantics must include an independently worked JSON fixture.
Regenerate Collector evidence with `./lab/otel/run-roundtrip.sh`; never edit
capture, compiled-telemetry, or verification artifacts by hand.
