# Test MoP-JEPA hard-assigned predictors

- Status: in progress
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
- [ ] The 40-epoch official matrix and fresh-process assessment complete.
- [ ] The downstream disposition and all-JEPA synthesis are recorded.
