# Quantis JEPA corpus v1 milestone

## Status

Accepted for implementation on 2026-07-27 after the matched-topology diagnostic
showed that demand-conditioned v2 remains sensitive to unseen request schedules,
not to worker replica count alone.

## Claim under development

A jointly learned telemetry representation can predict the next normal latent
state across held-out request schedules with lower normal-alert rates than the
frozen linear v2 precursor, while preserving auditable run-level provenance.

This milestone starts a JEPA world-model implementation. Its first learned model
is a single-step temporal JEPA development baseline, not yet evidence for a
multi-horizon production world model.

## Public seams

### Telemetry corpus

`compile_telemetry_corpus(runs, feature_spec, split_spec) -> TelemetryCorpus`

The compiler:

- accepts complete OTLP captures paired with static manifests;
- verifies capture-to-manifest identity and complete feature cells;
- selects only each run's declared fault-free baseline interval;
- applies demand conditioning, then fits normalization on training runs only;
- compiles windows separately per run so context never crosses captures;
- requires disjoint training and validation case IDs and canonical schedules;
- automatically reserves committed result-bearing cases and rejects those plus
  any additional case declared as reserved evidence; and
- returns run-level provenance with the compiled train and validation windows.

### Learned JEPA detector

`JepaWorldModelDetector.fit(normal_windows) -> JepaWorldModelDetector`

`JepaWorldModelDetector.score(windows) -> DetectionScores`

The online point encoder maps each context observation into a learned latent
space. A predictor maps the encoded context sequence to the next latent state.
The target point encoder is an exponential-moving-average copy of the online
encoder and receives no direct prediction gradient. Training uses an explicit
seed and deterministic full-batch updates. The serialized artifact contains all
weights, hyperparameters, fitted shape, calibration threshold, and training
losses.

The initial encoder is deliberately small and implemented with the repository's
existing NumPy dependency. It is a tracer bullet for the training, provenance,
serialization, and scoring path.

## Development gates

- No reserved evidence case can enter either corpus split.
- No case or canonical request schedule can cross train and validation splits.
- Normalization is fitted only on training points.
- Context windows never cross run boundaries.
- Repeated training with the same seed and inputs produces byte-identical model
  artifacts.
- A restored artifact produces identical scores and feature evidence.
- Training and held-out latent losses are reported separately.

## Corpus collection target

Before using JEPA performance to choose a confirmation candidate, collect at
least:

- 30 fresh normal-only runs;
- 10 distinct request-schedule families;
- 10,000 normal model windows; and
- complete one-, two-, and three-worker coverage.

Entire schedule families, not individual windows, are held out for validation.
All existing v1, v2, expanded, and matched result-bearing cases remain reserved
evidence and cannot be used for fitting or model selection.

## Required limitations

- The v0 target is one future telemetry point rather than a future block.
- The six demand-conditioned metrics are a small state vocabulary.
- Request demand is handled by preprocessing rather than learned as an explicit
  control variable in v0.
- A local lab corpus does not establish production generalization.
- Feature evidence is a target-encoder sensitivity projection, not a decoder or
  causal attribution.
- A learned joint embedding is not by itself a complete world model.
