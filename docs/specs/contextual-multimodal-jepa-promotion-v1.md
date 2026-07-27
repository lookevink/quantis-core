# Contextual metrics + logs JEPA promotion v1

## Status

Preregistered and ready for a new untouched collection. No run from this
corpus may begin until this specification, protocol, instrumentation, feature
vocabulary, trainer, assessor, and runner are committed in one clean Git
state.

The selected development hypothesis is fixed: L1 feature prediction with
three metric latent dimensions and one log latent dimension. The old
30-run corpus and its inspected validation families are development evidence
only; neither its checkpoint nor its metrics are eligible for promotion.

## Claim and evidence boundary

The confirmation claim is narrow: bounded endogenous application-state logs
improve normal schedule transfer beyond architecture-matched metrics-only and
shuffled-log controls.

Passing this protocol makes only the newly trained, separately serialized
contextual metrics-plus-logs model eligible for publication. It does not
replace the metrics-only model, establish fault-detection performance, or
authorize a production rollout. Fault evaluation remains a later milestone.

## Fresh corpus

The corpus contains 30 new fault-free runs under case IDs ending in seed label
73. Ten request-schedule families are each crossed with one, two, and three
workers. Families 1–8 train and families 9–10 remain untouched until the
fixed confirmation procedure. Two deterministic replicas score those families
without adaptation; only the first replica enters the final assessment. All
schedules and case IDs are disjoint from both earlier JEPA corpora.

Each run contains 340 points at a 0.1-second collection period, uses a
six-point lookback, and records the API accept backlog of 128 in its
provenance. Targets are two-point contiguous blocks beginning at horizons 1,
3, and 6. The runner refuses a dirty worktree, changed commit, existing output
directory, or protocol/hash mismatch.

The raw metric schema is fixed to the seven fault-lab gauges, and the model
schema is fixed to the six demand-conditioned metric features. Both schema
identities and both ordered vocabularies are checked before promotion.

## Bounded application-state vocabulary

The original acceptance, rejection, completion, and error events remain.
The promotion corpus adds only fixed event names:

- queue backlog transitions: low (0–2), elevated (3–8), high (9+);
- database write latency: fast (<2 ms), normal (2–10 ms), slow (10+ ms);
- worker state transitions: busy and idle.

Queue mutation, depth classification, and state advance occur in one Redis
script. The transition event time is captured inside that operation, so an
enqueue/dequeue race cannot erase or mis-time a threshold crossing. Related
events share one OTLP request, limiting instrumentation overhead. Each empty
run starts in the low state, so its first enqueue is not counted as a
synthetic transition.
Features count only the preregistered `event.name` values plus the existing
bounded severity-17 error count. Bodies, IDs, payloads, stack traces, and
arbitrary attributes remain excluded.

The contextual compiler converts transition counts to demand-relative rates
and database buckets to completion-relative ratios. The resulting richer
semantic log vector is still compressed to the selected one-dimensional log
latent, directly testing whether an active log representation captures state
rather than absolute request volume.

## Frozen model and controls

The new training run uses 200 EMA representation-pretraining epochs followed
by 100 predictor-only frozen-encoder epochs, learning rate 0.02, EMA decay
0.98, weight decay 0.0001, L1 loss, auxiliary and rollout weights 0.2,
calibration quantile 0.98, and seed 73. Cross-validation is disabled:
selection is over before the untouched validation families are scored.

The run must serialize aligned contextual multimodal, continuity metrics-only,
capacity-matched metrics-only, shuffled-log, and log-only models plus both
modality-dropout diagnostics. A second identical training invocation must
produce byte-identical JSON and report artifacts. Neither replica may change
configuration after validation scoring.

Each invocation also writes a separate execution attestation containing a
random execution ID, process ID, non-overlapping start/end times, output
directory, and hashes of the training result, model, corpus, and promotion
protocol. The assessor requires two distinct sequential attestations. These
are procedural evidence produced by the frozen clean-worktree runner, not
cryptographic proof against a malicious local operator who can forge files.

The preregistration freezes the complete local Python package, project
dependency declaration, collection code, runner, and feature specifications.
Training also refuses a Python, NumPy, or platform fingerprint other than the
one recorded in the protocol.

The shuffled-log control preserves each log context/target block, breaks its
alignment to metrics with training seed 1074 and validation seed 2074, and
keeps demand/topology controls aligned to metrics.

## Promotion gates

Every gate is fixed before collection:

- overall validation alert rate is at most 3%;
- every validation schedule family's alert rate is at most 5%;
- aligned logs are no worse than continuity metrics-only;
- aligned logs are no worse than capacity-matched metrics-only;
- aligned logs are strictly better than shuffled logs;
- aligned logs are no worse than metrics-only on at least 80% of validation
  schedule families;
- metric effective rank is at least 1.5 of 3; and
- log effective rank is at least 0.5 of 1.

The 3% aggregate ceiling allows one percentage point over the model's 2%
training calibration. The 5% family ceiling directly prevents the earlier
9.45% single-family failure from being hidden by the mean. Failure of any
gate is a completed negative result; the corpus, thresholds, split, and seed
must not be revised.

## Run

After committing this preparation:

```bash
./lab/fault_matrix/run-contextual-multimodal-jepa-promotion.sh
```

The runner will collect the corpus, train twice, verify deterministic
artifacts, reconcile schedule-family counts with aggregate validation counts,
verify every candidate/control artifact and embedded hash, and write the
preregistered promotion assessment. A failed gate is preserved as a negative
assessment and returns a nonzero status. The runner does not publish or deploy
anything automatically.
