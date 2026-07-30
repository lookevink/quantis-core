# Quantis Core

Quantis Core is an evidence-first research lab for predictive anomaly
detection and investigation over infrastructure telemetry.

The repository contains the complete progression from OTLP ingestion and a
real instrumented checkout stack through raw action-conditioned dynamics,
contextual metrics-and-logs models, graph observability experiments, and 23
retained JEPA tracers.

## Current conclusion
On the current deliberately small, one-stack, mostly deterministic problem,
learned representations repeatedly improved rank, geometry, pretraining loss,
or probe quality while losing entity-local state, action-effect magnitude, or
calibration that remained available in raw telemetry.

The strongest current shadow-system candidates are:

1. rank-32 contractive raw action-conditioned dynamics for prediction;
2. direct raw/PCA precedent retrieval for investigation; and
3. explicit trajectory-level calibration, abstention, and operator feedback
   around those baselines.

This is an open-development conclusion, not production authorization and not
a general rejection of JEPA. Read the
[cross-experiment conclusion](docs/research/jepa-frontier-execution-conclusion-2026.md)
for the evidence boundary and the
[JEPA experiment directory](experiments/jepa/) for every retained technique.

## Start here

| Goal | Entry point |
|---|---|
| Understand the current result | [JEPA frontier conclusion](docs/research/jepa-frontier-execution-conclusion-2026.md) |
| Browse every JEPA technique | [Technique-centered JEPA directory](experiments/jepa/) |
| Reproduce a retained tracer | [JEPA reproduction guide](lab/action_dynamics/JEPA_REPRODUCTION.md) |
| Follow the experiment sequence | [JEPA program map](docs/wayfinding/jepa-implementation-program/map.md) |
| Understand the evidence ladder | [JEPA experiment ladder](docs/specs/jepa-experiment-ladder-v1.md) |
| Inspect intentionally omitted techniques | [Frontier audit](docs/research/jepa-frontier-technique-audit-2026.md) |
| Run the core thesis evaluation | [Vertical-slice specification](docs/specs/vertical-slice.md) |
| Contribute safely | [Contributing guide](CONTRIBUTING.md) |

## Repository map

| Path | Responsibility |
|---|---|
| [`src/quantis_core`](src/quantis_core) | Supported library modules and command-line interfaces |
| [`experiments`](experiments) | Human-facing experiment directory organized by technique |
| [`experiments/jepa`](experiments/jepa) | Standardized capsules for the 23 comparable JEPA tracers |
| [`lab`](lab) | Shared data collection, Docker, OTLP, fault-injection, and historical runner infrastructure |
| [`docs/specs`](docs/specs) | Cross-cutting and historical frozen protocols |
| [`docs/research`](docs/research) | Cross-experiment syntheses and historical result documents |
| [`docs/wayfinding`](docs/wayfinding) | Execution sequence, ticket history, and dispositions |
| [`tests`](tests) | Behavioral tests at supported and experiment interfaces |
| [`artifacts`](artifacts) | Immutable local evidence; most large bundles are intentionally Git-ignored |
| [`tools`](tools) | Repository-maintenance and catalog verification tools |

The technique capsules are the navigation interface. Historical source paths
remain in place because published artifact manifests bind them by identity.
Capsules use generated relative links rather than copying code or evidence.
See the [repository-layout decision](docs/architecture/repository-layout.md).

## Quick start

Python 3.9 or newer is required.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,training]'
.venv/bin/python -m pytest
.venv/bin/python -m quantis_core evaluate --output artifacts/evaluation
```

Verify the experiment directory after changing catalog metadata:

```bash
.venv/bin/python tools/sync_experiment_catalog.py --check
```

## Experiment programs

### 1. Core anomaly-detection vertical slice

The original vertical slice asks whether a latent predictive detector trained
on normal operation can tolerate isolated telemetry noise while detecting
held-out correlated structural drift.

- [Specification](docs/specs/vertical-slice.md)
- Command: `.venv/bin/python -m quantis_core evaluate --output artifacts/evaluation`
- Output: `artifacts/evaluation`

### 2. OTLP replay and real fault lab

These experiments establish ingestion and evidence plumbing before model
claims.

| Experiment | Run | Evidence |
|---|---|---|
| OTLP metrics replay | `.venv/bin/python -m quantis_core replay-otlp ...` | [Protocol](docs/specs/otlp-replay.md), [checked-in report](artifacts/otlp-replay/report.md) |
| Collector round trip | `./lab/otel/run-roundtrip.sh` | [OTLP lab](lab/otel) |
| Instrumented checkout fault lab | `./lab/fault/run-fault-lab.sh` | [Protocol](docs/specs/fault-lab.md), [checked-in report](artifacts/fault-lab/report.md) |

### 3. Demand-conditioned detector and topology diagnostics

| Experiment | Result | Entry point |
|---|---|---|
| Demand-conditioned v2 confirmation | 3/3 recall and attribution, but exposed pre-noise and routine-noise alerts | [Specification](docs/specs/demand-conditioned-v2.md) |
| Expanded 3×3 topology matrix | Perfect recall but severe false-positive growth | [Specification](docs/specs/expanded-confirmation-matrix.md) |
| Matched-topology diagnostic | Removed the apparent topology gradient; schedule sensitivity remained | [Specification](docs/specs/matched-topology-diagnostic.md) |

Primary commands:

```bash
./lab/fault_matrix/run-v2-confirmation.sh
./lab/fault_matrix/run-expanded-v2-confirmation.sh
./lab/fault_matrix/run-matched-v2-diagnostic.sh
```

### 4. Foundational telemetry JEPA program

These experiments predate the common 23-tracer action-dynamics ladder.

| Experiment | Outcome | Evidence |
|---|---|---|
| Metrics-only JEPA world model v0 | Established the first run-isolated predictive-latent pipeline | [Corpus specification](docs/specs/jepa-corpus-v1.md) |
| Pointwise metrics + logs JEPA | Learned a request-demand shortcut | [Lessons](docs/research/jepa-telemetry-lessons.md) |
| Contextual multimodal JEPA v1/v2 | Removed the shortcut but produced only small, unstable log value | [Development result](docs/research/contextual-multimodal-jepa-v1-results.md) |
| Five-seed contextual confirmation | Narrow claim not supported; route to observability | [Confirmation specification](docs/specs/contextual-multimodal-jepa-confirmation-v2.md) |
| Adaptive graph-JEPA observability pilot | Preserved 95.8% of state with modest one-hop signal | [Result](docs/research/graph-jepa-observability-pilot-v1-results.md) |
| Learned hybrid graph JEPA | Small probe gain, but local collapse and failed state recovery | [Result](docs/research/hybrid-telemetry-jepa-development-v1-results.md) |
| Action-conditioned latent JEPA | Learned action sensitivity but lost badly to raw low-rank dynamics | [Result](docs/research/action-conditioned-jepa-low-rank-development-v1-results.md) |
| Raw-preserving residual JEPA | Preserved attribution but added no transferable value | [Result](docs/research/residual-jepa-correction-development-v1-results.md) |

### 5. Action-conditioned raw and adjacent edge techniques

The action-dynamics program evaluated dense and low-rank state-space models,
graph residuals, echo-state networks, direct temporal convolution, conformal
and sequential detection, and streaming sketches on a shared corpus.

The rank-32 contractive model matched dense VARX prediction and closed-library
attribution with about half the parameters and serialized size. See the
[edge-dynamics result](docs/research/edge-dynamics-development-v1-results.md).

### 6. Comparable JEPA frontier: 23 retained tracers

Every technique below has a capsule containing its runner, specification,
citations, findings, program ticket, and artifact reference. Assessors,
reference notes, supported implementations, and behavioral tests are linked
where those separate files exist.

| Technique | Main positive signal | Disposition |
|---|---|---|
| [Soft regime-codebook JEPA](experiments/jepa/regime_codebook/) | Broad code usage and improved state probing | Rejected: effect and calibration regressed |
| [Event-native trace JEPA](experiments/jepa/event_native_trace/) | Learned the bounded path grammar | Rejected: no aligned alert or investigation gain |
| [Multi-hypothesis trajectory JEPA](experiments/jepa/multi_hypothesis/) | Produced some distinct alternatives | Rejected: lost proper score and raw prediction |
| [Exact SIGReg substitution](experiments/jepa/sigreg_lejepa/) | Broadened latent rank | Rejected: worse state and no detection |
| [Complete multi-view LeJEPA](experiments/jepa/complete_lejepa/) | Restorable and state-accessible | Rejected: lost to reconstruction and raw |
| [Retrieval-JEPA](experiments/jepa/retrieval_jepa/) | Causal, state-safe, edge-feasible retrieval | Rejected: 40% hit@1 versus 100% raw/PCA |
| [HEPA](experiments/jepa/hepa/) | Monotone state-rich event probability | Rejected: 50% detection, tied its null |
| [SC-JEPA](experiments/jepa/sc_jepa/) | Exact codebook × resolution factorial | Rejected: negligible interaction and local collapse |
| [CF-JEPA](experiments/jepa/cf_jepa/) | Smooth state-rich EMA target | Rejected: no multi-zone gain and excess false alarms |
| [SD-JEPA](experiments/jepa/sd_jepa/) | Event-sensitive angular change | Rejected: progression coordinates and alerts failed |
| [Delta-JEPA](experiments/jepa/delta_jepa/) | Real action signal in displacement | Rejected: endpoints and raw dynamics were better |
| [LeWorldModel geometry screen](experiments/jepa/leworld_geometry/) | Geometry regularizers were active | Rejected: no raw-safe cell |
| [Causal-JEPA](experiments/jepa/causal_jepa/) | Whole-entity completion beat persistence | Rejected: coordinate masking and raw were better |
| [MoP-JEPA](experiments/jepa/mop_jepa/) | Hard predictors genuinely specialized | Rejected: supervised WTA, codebook, and raw won |
| [PairEffect-JEPA](experiments/jepa/pair_effect_jepa/) | Matched-twin effect formulation executed | Rejected: lost to derangement and raw |
| [Contract-JEPA](experiments/jepa/contract_jepa/) | Safe raw bypass and 1.05% raw effect gain | Rejected: both controls better; witness drifted |
| [Error-Certificate-JEPA](experiments/jepa/error_certificate_jepa/) | Exact raw preservation and calibrated bounds | Rejected: coverage and detection failed |
| [SALT-JEPA](experiments/jepa/salt_jepa/) | Aligned static-teacher signal | Rejected: missed transfer gate and raw safety |
| [LeNEPA](experiments/jepa/lenepa/) | Disposable projection reproduced exactly | Rejected: mechanism and raw safety failed |
| [Discrete-JEPA](experiments/jepa/discrete_jepa/) | Exact hard semantic-token objective | Rejected: one-code collapse and no value |
| [PEIRA](experiments/jepa/peira/) | Real non-collapsed alignment mechanism | Rejected: controls and raw won |
| [VISReg](experiments/jepa/visreg/) | Small-radius gradient mechanism passed | Rejected: projector collapse and raw regression |
| [JEPA-SCORE](experiments/jepa/jepa_score/) | Exact Jacobian/SVD fit the edge budget | Rejected: 10% IID and 0% transfer detection |

## Artifacts and reproducibility

Experiment code is retained. Published
artifact directories are immutable, and every rerun must use a fresh output
directory.

Read:

- [artifact policy](artifacts/README.md);
- [JEPA reproduction guide](lab/action_dynamics/JEPA_REPRODUCTION.md); and
- the relevant technique capsule under [`experiments/jepa`](experiments/jepa).

## Contribution workflow
 See [CONTRIBUTING.md](CONTRIBUTING.md).
