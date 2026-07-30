# Experiment directory

This is the human-facing directory for Quantis experiments. Library code stays
under `src/quantis_core`; shared collection and execution infrastructure stays
under `lab`; immutable evidence stays under `artifacts`.

## Programs

| Program | Question | Entry point |
|---|---|---|
| Core vertical slice | Can a predictive detector separate structural drift from isolated noise? | [Specification](../docs/specs/vertical-slice.md) |
| OTLP replay | Can semantic telemetry be compiled and replayed without changing detector inputs? | [Specification](../docs/specs/otlp-replay.md) |
| Instrumented fault lab | Can real service disturbances be observed, detected, and attributed? | [Specification](../docs/specs/fault-lab.md) |
| Demand-conditioned detector | Does conditioning remove the original workload shortcut? | [Specification](../docs/specs/demand-conditioned-v2.md) |
| Contextual metrics + logs JEPA | Do aligned bounded events improve compact predictive state? | [Scientific interpretation](../docs/research/jepa-v2-scientific-interpretation.md) |
| Graph observability | Can entity ownership and declared topology preserve useful state? | [Pilot result](../docs/research/graph-jepa-observability-pilot-v1-results.md) |
| Action-conditioned edge dynamics | Which compact dynamics and detector methods survive held-topology evaluation? | [Result](../docs/research/edge-dynamics-development-v1-results.md) |
| JEPA frontier | Does any materially distinct JEPA technique add alert, prediction, or investigation value? | [Technique directory](jepa/) |

## Organization rule

New experiment families should be added as technique-centered capsules. A
capsule owns the navigational interface for:

- the frozen hypothesis and specification;
- primary references and adaptation notes;
- exact runner and independent assessor;
- conclusion-bearing findings;
- supported implementation and behavioral tests;
- immutable artifact path and manifest identity; and
- the evidence boundary and disposition.

Historical files whose paths are already embedded in artifacts should not be
moved merely for aesthetics. Use a capsule alias and record the compatibility
reason. New experiments should place their orchestration and documentation
directly in the capsule from the beginning.
