# Causal-JEPA entity-intervention tracer v1 result

## Decision

**Reject this edge telemetry Causal-JEPA recipe.**

Whole-entity masking learned useful hidden-history completion relative to
anchor persistence, but coordinate-time masking learned it better. The
candidate also lost to that matched null on selection and transfer
downstream-effect prediction, won only half of held-topology pairs, and
regressed raw dynamics by more than threefold.

This rejects the frozen width-32 telemetry observability intervention. It
does not reject Causal-JEPA on object-centric visual reasoning or control,
and it does not establish or disprove causal identification.

## Evidence identity

- Implementation commit: `b2d3c42`
- Published local artifact:
  `artifacts/action-dynamics/prototype-causal-jepa-v1`
- Retained non-interpretable smoke:
  `artifacts/action-dynamics/prototype-causal-jepa-v1-smoke-1step`
- Retained superseded official attempt:
  `artifacts/action-dynamics/prototype-causal-jepa-v1-attempt-1-restore-failure`
- Artifact-manifest SHA-256:
  `3ea9af2a479d96c0eca5690ffa2ebe8865fff13cc3618bcfe9f97779fc413b52`
- Assessment SHA-256:
  `6e2bb61997d7751c7ff8e82760ac1ec96b0b507d57cdcb242c97b975b875f94a`
- Independent fresh-process decision:
  `reject_causal_jepa_edge_recipe`

The official run used seed `18018`, 1,200 steps per cell, checkpoints every
200 steps, and 100 latency repetitions. All three cells selected step 1,200.

## Interaction mechanism

| completion route | transfer history MSE | treatment history MSE |
|---|---:|---:|
| whole-entity Causal-JEPA | 1.443061 | 3.906892 |
| coordinate-time mask | **1.384777** | **3.629698** |
| prediction only | 2.743143 | 10.889930 |
| anchor persistence | 2.379124 | 3.940608 |

The candidate improved overall completion over anchor persistence by
`39.35%`, showing that the masked-history objective learned real relational
signal. It was nevertheless `4.21%` worse than coordinate-time masking,
failing the required 10% entity-level advantage. On treatment windows it was
only `0.86%` better than persistence and `7.64%` worse than coordinate-time
masking.

The result does not support the hypothesis that keeping a target entity
hidden across its complete non-anchor trajectory forces uniquely useful
interaction structure on this corpus. Distributed missing coordinates
provided a stronger completion curriculum.

## Downstream result

| forecast route | selection effect MSE | transfer effect MSE |
|---|---:|---:|
| whole-entity Causal-JEPA | 0.397178 | 0.519112 |
| coordinate-time mask | **0.370879** | **0.502943** |
| prediction only | 0.433503 | 0.513492 |
| raw rank-32 dynamics | **0.128783** | **0.143833** |

The candidate failed selection ordering, was `3.21%` worse than the selected
coordinate control on transfer, and beat it on only 50% of transfer pairs.
Its transfer effect error was `260.92%` worse than raw.

Broader forecast safety also failed:

- transfer overall MSE was `0.385549`, versus raw `0.105744`; and
- transfer action-overlap MSE was `2.954407`, versus raw `0.859940`.

Action-and-target attribution hit@1 and correct-action sanity were both 100%,
but no-action specificity was 90%, below the frozen 100% requirement.

## Edge and apparatus result

The model itself is edge-sized:

- all cells had exactly `31,328` trainable parameters;
- the candidate JSON model was `679,116` bytes;
- batch-one CPU forecast latency was `0.962997 ms` mean and `1.125386 ms`
  p95; and
- peak process RSS was `4,672,667,648` bytes.

Pair and mask schedules, checkpoint selection, finiteness, public causal
inputs, candidate restoration, forecasts, and attribution replay all passed.
The matrix-level restoration gate failed because the prediction-only
completion replay differed from its original in-memory output by a maximum
of `1.0728836059570312e-6`, just beyond the frozen `1e-6` tolerance. Two
freshly restored models agree exactly. The tolerance was not changed after
seeing the result. This safety failure is independent of the much larger
mechanism and downstream rejection margins.

The first complete official attempt and the identical-batch investigation are
retained in the
[restore-boundary record](causal-jepa-attempt-1-restore-boundary.md).

## Next target

Proceed to MoP-JEPA's hard best-of-K predictor specialization. Causal-JEPA
shows that forcing cross-entity completion does not repair deterministic
forecast error. The last planned material mechanism tests whether distinct
hard-assigned latent future hypotheses capture conditional alternatives that
the earlier likelihood-trained mixture did not.
