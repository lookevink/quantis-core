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

## Matched topology diagnostic

The next milestone removes that confound with three blocked comparisons. Within
each fault block, the one-, two-, and three-worker cases use the same absolute
request schedule and identical non-topology manifest fields:

```bash
./lab/fault_matrix/run-matched-v2-diagnostic.sh
```

The [matched diagnostic report](artifacts/demand-conditioned-v2/matched-topology-diagnostic/report.md)
finds no material topology effect. Pre-noise alert rates are 35.1%, 33.6%, and
32.8% for one, two, and three workers. Every preregistered paired difference is
below 20 percentage points, while aggregate pre-noise and routine-noise rates
still fail at 133/393 and 27/63. Holding schedule fixed removes the earlier
topology gradient: the v2 failure is better explained by schedule sensitivity
than worker count alone.

## Build the JEPA world-model v0

Compile run-isolated, demand-conditioned normal windows and train the first
learned joint-embedding predictor:

```bash
.venv/bin/python -m quantis_core train-jepa-world-model \
  --captures-directory path/to/fresh-normal-corpus/cases \
  --manifests-directory path/to/fresh-normal-manifests \
  --feature-spec lab/fault_matrix/feature-spec.json \
  --split-spec path/to/corpus-split.json \
  --output artifacts/jepa-world-model-v0
```

The split specification declares training, validation, and additional
reserved-evidence case IDs. The compiler automatically reserves every committed
result-bearing case and rejects case overlap, held-out schedule overlap,
capture/manifest mismatch, incomplete cells, and any attempt to place reserved
evidence in a model split. Normalization is fitted only on training runs and
windows are compiled separately per run.

The [JEPA corpus v1 specification](docs/specs/jepa-corpus-v1.md) describes the
single-step NumPy tracer bullet and the fresh-corpus gate required before model
selection. Existing result-bearing captures are reserved automatically; they are
not training data.

Run the three-schedule fresh-log pilot with:

```bash
./lab/fault_matrix/run-jepa-pilot.sh
```

The pilot collects two training schedules and one disjoint validation schedule,
then trains only on each capture's first 60 fault-free points. Each capture
injects a fault afterward to reuse the current lab runner, so these runs do not
count toward the normal-only corpus target. Development-only model and corpus
artifacts are written under `artifacts/jepa-world-model-v0/pilot`.

Collect the first target-sized normal-only corpus and train without fault
injection:

```bash
./lab/fault_matrix/run-jepa-normal-corpus.sh
```

This expands ten schedule families across one-, two-, and three-worker
topologies. Eight complete families train the model and two complete families
remain validation-only, yielding 30 runs and 10,020 run-isolated windows.

## Add structured application logs

The multimodal tracer bullet keeps metrics and application logs in separate
encoders, then predicts their next joint latent state. The lab API and worker
emit bounded-vocabulary OTLP events to a dedicated Collector Logs pipeline:

- `checkout.accepted`;
- `checkout.rejected`;
- `checkout.completed`; and
- error-severity events.

Raw message text, request identifiers, payloads, and stack traces do not become
model features. The log compiler aggregates declared events by logical window,
verifies capture/manifest identity, and joins them with metrics without crossing
run boundaries.

Run the preregistered three-schedule multimodal pilot with:

```bash
./lab/fault_matrix/run-multimodal-jepa-pilot.sh
```

The pilot writes a fused application-log JEPA and a metrics-only baseline from
the same training and validation runs under
`artifacts/jepa-world-model-v0/multimodal-pilot`. See the
[multimodal JEPA specification](docs/specs/multimodal-jepa-v0.md) for the
interfaces, safety constraints, and development gates.

Collect and train the separate target-sized metrics-plus-logs model with:

```bash
./lab/fault_matrix/run-multimodal-jepa-normal-corpus.sh
```

This creates 30 fresh fault-free runs across ten new schedule families and
trains the aligned model, two metrics-only baselines, and a shuffled-log
ablation as separate artifacts. Publication is allowed only if every frozen
promotion gate passes. The first collection attempt failed before training and
is not reused; the recovery protocol is in the
[multimodal corpus v2 specification](docs/specs/multimodal-jepa-corpus-v2.md).

The follow-up development model removes the observed request-volume shortcut,
uses demand and topology as predictor controls, and predicts contextual
two-point blocks at horizons 1, 3, and 6. It preserves the v0 model and reuses
the 30-run corpus only for development:

```bash
./lab/fault_matrix/run-contextual-multimodal-jepa-development.sh
```

The run produces the contextual model plus metrics-only, capacity-matched,
shuffled-log, log-only, and modality-dropout controls. Previously inspected
validation families remain diagnostic only; a new untouched corpus is required
before publication. See the
[contextual multimodal JEPA v1 specification](docs/specs/contextual-multimodal-jepa-v1.md)
for the cited design rationale and evaluation protocol.

The completed development comparison selected L1 loss with one log latent as a
hypothesis for the next untouched corpus; the advantage was small and did not
establish useful log transfer. See the
[v1 development result](docs/research/contextual-multimodal-jepa-v1-results.md).

The v2 development cycle replaces repetitive low/fast/idle complements with
bounded Redis, PostgreSQL, and checkout queue-pressure events emitted by real
lab operations. It evaluates a fixed latent-capacity and modality-balancing
sequence using schedule-family-held-out controls only:

```bash
./lab/fault_matrix/run-contextual-multimodal-jepa-v2-development.sh
```

This collects 30 fresh development runs, trains separate contextual and
metrics-only models plus controls, and writes the candidate leaderboard and
selected bundle under
`artifacts/jepa-world-model-v2/contextual-development-v2`. The v1 runs and the
final two v2 families are diagnostic only; promotion requires a new untouched
corpus after the v2 recipe is frozen. See the
[v2 development specification](docs/specs/contextual-multimodal-jepa-v2-development.md).

The next experiment freezes that recipe and tests a narrower publishable
claim on 72 untouched runs. It collects through three isolated Docker lanes,
trains five fixed seeds plus a deterministic repeat, and evaluates frozen
12-dimensional JEPA states against raw 108-dimensional and PCA baselines:

```bash
./lab/fault_matrix/run-contextual-multimodal-jepa-confirmation-v2.sh
```

Positive and negative outcomes are preserved under the same preregistration.
The assessor also selects the next world-model milestone from the evidence:
action-conditioned interventions if the claim passes, log-alignment repair if
only compression transfers, or improved observability if compression fails.
See the
[confirmation v2 specification](docs/specs/contextual-multimodal-jepa-confirmation-v2.md).

The negative confirmation routes to an observability-first graph tracer. It
declares the fixed API, queue, worker, Redis, and PostgreSQL topology; assigns
every semantic observation to one node or edge; and refuses representation
training until raw held-out graph state clears mean, persistence, and flat
controls:

```bash
./lab/fault_matrix/run-graph-observability-pilot.sh
./lab/fault_matrix/run-linear-graph-jepa-pilot.sh
./lab/fault_matrix/run-graph-jepa-width-sweep.sh
./lab/fault_matrix/run-adaptive-graph-jepa-pilot.sh
```

On the inspected corpus, a training-selected adaptive profile compresses 108
raw context values to 78 active graph-latent values while retaining 95.8% of
measured target state. One-hop prediction modestly beats entity-local
prediction but remains slightly behind the all-entity control. This is a
development architecture result, not a world-model claim. See the
[graph pilot specification](docs/specs/graph-jepa-observability-pilot-v1.md)
and [result interpretation](docs/research/graph-jepa-observability-pilot-v1-results.md).

## Reproduce the retained edge JEPA tracers

Rejected experiment recipes remain reproducible evidence. Twenty-two tracers
were run against the shared evaluation contract; none passed its frozen
observable-safety, mechanism, and operational-value gates. All runners,
assessors, and result documents are retained under `lab/action_dynamics`. The
full decision log is the
[wayfinding map](docs/wayfinding/jepa-implementation-program/map.md);
reproduction commands, seeds, and artifact identities are in the
[JEPA reproduction guide](lab/action_dynamics/JEPA_REPRODUCTION.md).

- **Soft regime-codebook JEPA** — Broad code usage and improved state probing
  did not preserve effect prediction or alert calibration.
- **Event-native trace JEPA** — A compact non-collapsed trace encoder learned
  the bounded path grammar but added no aligned alert or investigation value
  over its shuffled null.
- **Multi-hypothesis trajectory JEPA** — Distinct alternatives emerged but the
  mixture lost to its supervised control and badly regressed raw prediction. A
  v2 assessor applies the corrected scoring contract to the immutable v1
  sidecars.
- **Exact SIGReg LeJEPA substitution** — The regularizer broadened latent rank
  but worsened observable-state probing, earned zero correction gain, and
  detected no treatment trajectories.
- **Complete multi-view LeJEPA** — The predictor-free objective learned a
  restorable, state-accessible representation but lost to masked reconstruction
  and nearly doubled raw downstream-effect error.
- **Episode-predictive retrieval JEPA** — The causal query map was
  edge-feasible but reached only 40% transfer hit@1 while raw telemetry and
  PCA retrieved every transfer mechanism; direct retrieval is retained as the
  investigation baseline.
- **HEPA event-predictive alerting** — Learned a monotone, state-rich
  probability surface and improved Brier over its deranged null, but detected
  only 50% of treatments and trailed simpler controls.
- **Complete SC-JEPA codebook × multi-resolution interaction** — Negligible
  Brier interaction, zero detection interaction, no treatment detections,
  collapsed per-entity code use, and 2.73× matched-PCA state error.
- **CF-JEPA mask-free multi-horizon alerting** — The EMA target became
  smoother, lower rank, and state-rich, but three alert zones narrowly lost to
  one and the 10% control false-alarm rate exceeded the frozen ceiling.
- **SD-JEPA progression/content localization** — Angular change became more
  event-sensitive but the designated progression coordinates did not encode
  progression and the calibrated transfer policy emitted no alerts.
- **Delta-JEPA action-sensitive displacement** — The learned displacement
  contained a real action signal but endpoint concatenation decoded treatment
  actions much better and every neural cell nearly doubled raw transfer effect
  error.
- **LeWorldModel bounded geometry screen** — No regularized cell was
  selection-safe; SPHERE-JEPA won the diagnostic selection by a negligible
  margin and then lost transfer value and state information. Sub-JEPA and
  ambient cells collapsed without meeting the UR-JEPA prerequisite.
- **Causal-JEPA entity intervention** — Whole-entity masking learned
  completion beyond persistence but lost to matched coordinate-time masking and
  regressed raw held-topology effects by more than threefold.
- **MoP-JEPA hard-assigned predictors** — Hard assignment activated and
  produced transition-valid alternatives but lost to supervised hard-WTA and
  raw dynamics, regressing held-topology effects by 4.65×.
- **PairEffect-JEPA** — Matched-twin training was slightly worse than its
  deranged null and the composed path regressed raw transfer effect by 3.38×.
- **Task-grounded Contract-JEPA** — Passed every safety gate and improved raw
  effect error by 1.05%, but both controls predicted better and the effect
  witness produced 100% transfer control false alarms.
- **Error-Certificate-JEPA** — Passed every safety gate but achieved only 80%
  held-topology simultaneous control coverage, detected no treatments, and was
  no sharper than derangement or constant conformal.
- **SALT-JEPA static reconstructive teacher** — The corrected v2 run passed
  every evidence check but the transfer latent advantage was 8.89%, below the
  10% mechanism threshold, and the student retained 1.91× raw transfer effect
  error.
- **Exact LeNEPA disposable projection** — All twelve protocol checks passed
  but projection failed the mechanism gate, every learned cell failed raw
  forecast safety, and the candidate retained 1.92× raw transfer effect error.
- **Discrete-JEPA semantic tokenization** — The hard codebook collapsed to one
  code per entity, tied P2P-only on forecasts and transition accuracy, and
  retained 1.92× raw transfer effect error.
- **PEIRA inter-view regressor alignment** — Aligned PEIRA learned a real
  non-collapsed trace mechanism but lost to derangement and reconstruction
  controls, retained 1.91× raw transfer effect error, and won three of ten
  pairs.
- **VISReg scale-shape regularization** — Its small-radius gradient mechanism
  passed, but the detached candidate collapsed to projector rank 1.12,
  retained 1.97× raw transfer effect error, and lost to no-detach and
  reconstruction controls.

Raw action-conditioned dynamics and direct retrieval remain the shadow-system
baselines. Exact JEPA-SCORE edge alerting is the active final bounded screen
under [ticket 028](docs/wayfinding/jepa-implementation-program/map.md).
