# Quantis Core

Quantis Core is an evidence-first experiment in latent predictive anomaly
detection for infrastructure telemetry.

The first vertical slice answers a deliberately narrow question:

> Can a latent predictive detector trained on normal operation tolerate isolated
> telemetry noise while detecting held-out, correlated structural drift?

The repository does not yet claim production OpenTelemetry ingestion, a full JEPA
training objective, or autonomous remediation. See
[`docs/specs/vertical-slice.md`](docs/specs/vertical-slice.md) for the executable
claim and acceptance criteria.

## Run it

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest
.venv/bin/python -m quantis_core evaluate --output artifacts/evaluation
```

The final command writes machine-readable results, a Markdown report, the exact
scenario manifest, and model artifacts.

## Replay OTLP metrics

Compile a newline-delimited OTLP JSON capture using a versioned feature
specification:

```bash
.venv/bin/python -m quantis_core replay-otlp \
  --capture tests/fixtures/otlp/semantic-metrics.jsonl \
  --feature-spec tests/fixtures/otlp/semantic-feature-spec.json \
  --output artifacts/replay
```

Run the pinned OpenTelemetry Collector round-trip verification:

```bash
./lab/otel/run-roundtrip.sh
```

The checked-in [OTLP verification report](artifacts/otlp-replay/report.md)
records the Collector image digest, capture and feature-schema hashes, data
quality, tensor parity, detector-score parity, and limitations.

## Run the instrumented fault lab

Run a real API, worker, Redis queue, PostgreSQL database, and OpenTelemetry
Collector while injecting an isolated latency disturbance and a worker stall:

```bash
./lab/fault/run-fault-lab.sh
```

The checked-in [fault-lab report](artifacts/fault-lab/report.md) records the
observed queue and database effects, held-out false-positive rates, detection
latency, attribution, content hashes, and explicit limitations.

## Train and regress the demand-conditioned v2 model

The first frozen model failed its multi-schedule fault matrix because it learned
the development load pattern. That negative result is preserved in the
[v1 matrix report](artifacts/fault-matrix/report.md).

Model v2 conditions worker and database throughput on observed request demand.
Its checked-in model and development regression are recomputable from the raw
v1 captures:

```bash
.venv/bin/python -m quantis_core train-demand-conditioned-v2 \
  --captures-directory artifacts/fault-matrix/cases \
  --manifests-directory lab/fault_matrix/experiments \
  --feature-spec lab/fault_matrix/feature-spec.json \
  --output artifacts/demand-conditioned-v2
```

The [v2 regression report](artifacts/demand-conditioned-v2/regression/report.md)
is development evidence. The separate confirmation protocol was committed as
`908e91d` before any `-04` capture existed. Run it with:

```bash
./lab/fault_matrix/run-v2-confirmation.sh
```

The [v2 confirmation report](artifacts/demand-conditioned-v2/confirmation/report.md)
records 3/3 recall, 3/3 attribution hit@3, zero-window maximum delay, 22/148
pre-noise alerts, and 3/21 noise-response alerts. Its machine-readable evidence
records the preregistration commit and hashes the full 27-file evaluation
dependency closure.

## Expanded topology confirmation

The expanded 3×3 matrix runs every fault across one-, two-, and three-worker
topologies:

```bash
./lab/fault_matrix/run-expanded-v2-confirmation.sh
```

The preregistered [expanded report](artifacts/demand-conditioned-v2/expanded-confirmation/report.md)
is an important negative result. Recall, attribution, and detection delay remain
perfect at 9/9, 9/9, and zero windows, but pre-noise alert rates rise from 10.0%
with one observed worker to 70.5% with two and 87.4% with three. Routine-noise
rates similarly rise from 0.0% to 61.9% and 100.0%. The aggregate false-positive
gates therefore fail at 216/377 pre-noise alerts and 34/63 routine-noise alerts.
Worker count co-varies with
the workload schedules in this matrix, so the result establishes an
association with multi-worker operation rather than isolated causality. V2
transfers operationally to the new one-worker cases but not to this expanded
multi-worker workload envelope.
