---
status: closed
label: wayfinder:research
parent: Evaluate the unimplemented JEPA families
assignee: codex
blocked_by: []
---

# Audit the remaining JEPA frontier and select the next tracer

## Question

After collapsing paper names and backbone or masking variants into materially
distinct mechanisms, which JEPA families through 2026 have Quantis already
tested, which runnable families remain omitted, which are blocked by the
cross-stack corpus prerequisite or another missing precondition, and in what
evidence-first order should the remaining families be implemented and tested
while retaining every positive and negative tracer?

## Answer

Quantis has covered canonical EMA masked prediction, multimodal and graph
fusion, fine/coarse targets, action conditioning, residual correction,
codebooks, event-native objectives, soft finite mixtures, SIGReg substitution,
complete multi-view LeJEPA, and retrieval. It has not exhausted materially
different JEPA mechanisms.

Freeze the runnable order as:

1. HEPA horizon-conditioned event prediction;
2. the complete SC-JEPA codebook-plus-multi-resolution interaction;
3. CF-JEPA mask-free multi-horizon forward prediction;
4. SD-JEPA progression/content subspaces;
5. Delta-JEPA latent-difference action decoding;
6. exact LeWorldModel plus one controlled latent-geometry screen;
7. Causal-JEPA whole-entity trajectory masking; and
8. MoP-JEPA hard-assigned stochastic predictors.

T-JEPA missing-channel robustness, V-JEPA 2.1 deep supervision, and BiJEPA
cycle consistency are lower-priority conditional ablations. Variational,
diffusion, and density-matrix predictors require evidence of irreducible
conditional uncertainty. CHARM and other cross-stack semantic methods require
the cross-stack corpus. Planning-only, domain-tokenizer, backbone, mask-ratio,
and width/depth variants do not become independent telemetry experiments.

HEPA is first because it directly produces a monotone event-within-horizon
distribution and can be tested as an alert-policy adapter without replacing
the strong raw low-rank predictive core. Its role, treatment,
horizon-deranged JEPA null, and ten safety/value gates are frozen in the
primary-source audit.

## Evidence

- Primary-source and local-overlap audit:
  [`JEPA frontier technique audit, July 2026`](../../research/jepa-frontier-technique-audit-2026.md)
- Shared evidence boundary:
  [`JEPA implementation ladder v1`](../../specs/jepa-experiment-ladder-v1.md)

## Resolution comment

Resolved on 2026-07-28. The audit searched primary sources through that date,
collapsed modality and architecture ports into mechanism families, corrected
the earlier assumption that the full SC-JEPA interaction had been tested, and
selected HEPA as the next retained tracer. Every subsequent positive or
negative implementation remains in the repository with immutable result
artifacts.

## Next

Implement and test the frozen HEPA telemetry tracer end to end on open
development evidence.
