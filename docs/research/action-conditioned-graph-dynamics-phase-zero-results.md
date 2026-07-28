# Action-conditioned graph dynamics Phase 0 results

## Decision

Advance to the instrumented pilot.

The synthetic tracer proves that the planned scientific interfaces work end
to end:

- a dedicated strict manifest-v3 action timeline;
- pair-atomic treatment/control validation;
- training-only state, control, and action compilation;
- `action[t] -> state[t+1]` alignment;
- graph-constrained action-aware and action-agnostic VARX rollouts;
- persistence control;
- manifest-independent candidate-action ranking;
- delayed counterfactual propagation through the declared graph;
- compiler and model artifact restoration; and
- independently recomputable, hashed assessment evidence.

This is synthetic proof of plumbing. It is not evidence about the running lab,
real fault attribution, production transfer, or a software world model.

The reviewed artifact is:

`artifacts/action-dynamics/synthetic-phase-zero-v4/assessment.json`

## Synthetic design

- 15 training treatment/control pairs and 5 validation pairs.
- Treatment and control twins share workload, initial state, and process
  noise; only the intervention differs.
- The graph is a five-entity causal chain:
  `source -> source_to_middle -> middle -> middle_to_sink -> sink`.
- Training interventions vary across five action-kind/target pairs.
- Attribution hides validation truth and searches a frozen 61-candidate grid:
  five kind/target pairs, four onset times, three magnitudes, and no action.
- The true candidate tensor is not copied from the held-out manifest.
- The action-aware model, action-agnostic model, and compiler are serialized
  and restored before validation is scored.

## Results

| Gate | Observed | Threshold | Result |
|---|---:|---:|---|
| Action-aware versus action-agnostic forecast | 99.966% relative improvement | >=10% | pass |
| Action-aware versus persistence | 99.977% relative improvement | >=10% | pass |
| Joint action kind/target/onset/magnitude hit@1 | 100% | >=70% | pass |
| No-action specificity | 100% | >=90% | pass |
| Graph propagation delay | `[1,2,3,4,5]` | exact chain order | pass |

Normalized rollout MSE was:

- action-conditioned graph VARX: `0.000262`;
- action-agnostic graph VARX: `0.765567`; and
- persistence: `1.116520`.

These large margins are expected from a deliberately identifiable linear
synthetic system. They validate alignment, conditioning, restoration,
candidate search, and assessment. They do not estimate performance on real
telemetry.

## Evidence integrity

The output directory contains:

- the fixed protocol;
- fitted compiler;
- action-aware and action-agnostic models;
- observed futures, predictions, per-candidate rollout distributions, and
  truth;
- recomputed assessment; and
- SHA-256 hashes for every evidence artifact.

The runner refuses to overwrite a nonempty output directory. The assessor
uses fixed preregistered Phase-0 gates and regenerates likelihood rankings,
winner-versus-no-action propagation, and the final assessment from the
stored rollout distributions without trusting derived scores or gate
booleans.

## Next step

Implement Phase 1 against the existing fault-matrix stack:

1. add reversible action adapters for worker pause, PostgreSQL lock, Redis
   enqueue/dequeue delay, and API rejection;
2. propagate W3C trace context through API admission, Redis work items,
   workers, and PostgreSQL writes;
3. emit runner action commands to a conditioning-only Collector stream;
4. build isolated concurrent Compose capture projects; and
5. collect six treatment/control smoke pairs before the 30-pair
   instrumentation pilot.

No neural graph model should be added until those six pairs pass action
alignment, raw-effect, recovery, trace-linkage, truth-exclusion, and artifact
integrity checks.
