# Test MoP-JEPA hard-assigned predictors

- Status: complete
- Depends on:
  - [Audit remaining JEPA frontier](011-audit-remaining-jepa-frontier.md)
  - [Test Causal-JEPA entity intervention](018-test-causal-jepa-entity-intervention.md)

## Question

Does hard best-of-eight latent predictor specialization produce
context-conditioned, transition-valid futures with enough proper-score and
action-effect value for edge alerting?

## Frozen contract

Execute the
[MoP-JEPA hard-assignment tracer](../../specs/mop-jepa-hard-assignment-v1.md)
using the pinned
[primary-source notes](../../research/mop-jepa-primary-source-notes.md).
Preserve the implementation, candidate, controls, smokes, failures, and
official bundle.

## Completion

- [x] Unit tests cover hard assignment, router causality, codebook context
  independence, capacity, restoration, and non-interpretable assessment.
- [x] A retained smoke completes and independently reassesses.
- [x] Implementation is reviewed, full-suite clean, and committed before the
  official run.
- [x] The 40-epoch official matrix and fresh-process assessment complete.
- [x] The downstream disposition and all-JEPA synthesis are recorded.

## Outcome

Resolved on 2026-07-28 by the retained
[result](../../research/mop-jepa-hard-assignment-v1-results.md).

Reject the recipe. Hard assignment produced 2.224 effective observable
winners and 90.13% gated realized-transition precision, but router
specialization and context controls failed. The candidate lost proper score
to supervised hard-WTA and raw dynamics, produced 4.65 times raw transfer
effect error, and won only 40% of held-topology pairs.
