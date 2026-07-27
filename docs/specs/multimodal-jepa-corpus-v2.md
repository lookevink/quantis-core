# Quantis multimodal JEPA corpus v2

## Status

Preregistered for collection after corpus v1 failed before training. No model
was fitted and no validation metric was computed from v1.

The v1 attempt completed 28 runs, then the lab API reset a connection during
`multimodal-normal-f10-w2-47` at metric message 168 of 340. The API used the
standard library server's five-connection accept backlog while the declared
schedule issued bursts of up to 18 concurrent requests. The incomplete v1
attempt is preserved with `FAILED.json`; none of its captures may be used for
training, validation, or publication.

## Recovery change

The only lab-design change from
[`multimodal-jepa-corpus-v1.md`](multimodal-jepa-corpus-v1.md) is that the API
accept backlog is fixed at 128, above the maximum declared burst. The entire
30-run corpus is recollected under fresh `multimodal-normal-v2-*` case IDs with
seed label 48. Schedules, topology expansion, sample period, point count,
feature vocabularies, split membership by family, and promotion gates remain
unchanged.

The backlog defaults to the historical value of 5 for every other lab runner.
The v2 runner explicitly sets 128 and records it in both the input provenance
and every metric capture.

Reusing the schedule design does not follow an inspected model result: the v1
failure occurred during collection and the training command was never reached.

## Corpus

- 30 new fault-free runs and 10,020 run-isolated windows;
- families 1–8 across three topologies train: 24 runs and 8,016 windows;
- families 9–10 across three topologies validate: six runs and 2,004 windows;
- 340 fixed-duration points per run, six-point lookback, 0.1-second period;
- event-time log assignment with a required excluded drain interval; and
- one clean, unchanged Git commit and one content-addressed application build
  across all runs.

The exact ten schedules are those frozen in the v1 specification.

## Models

All models use 300 epochs, learning rate 0.02, EMA decay 0.98, weight decay
0.0001, calibration quantile 0.98, and seed 48:

- aligned multimodal JEPA: metric latent width 3 plus log latent width 2;
- continuity metrics-only JEPA: latent width 3;
- capacity-matched metrics-only JEPA: latent width 5; and
- shuffled-log multimodal JEPA using training permutation seed 1049 and
  validation permutation seed 2049.

All four remain separate artifacts.

## Promotion gates

The aligned multimodal artifact is eligible for publication only if its
validation alert rate:

- is at most 10%;
- is no worse than the capacity-matched metrics-only model;
- is no worse than the shuffled-log model; and
- is strictly better than at least one of those controls.

No cross-model latent loss is used. Failure of any invariant or gate is
reported without revising this corpus, its thresholds, or its split.
