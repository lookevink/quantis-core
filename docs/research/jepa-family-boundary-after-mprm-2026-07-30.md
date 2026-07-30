# JEPA family boundary after MPRM-JEPA, 2026-07-30

## Decision

Quantis has **not ruled out the JEPA family in general**. It has ruled out
deployment of every tested JEPA recipe and has exhausted “another generic
JEPA objective on the same targets and one-stack evidence” as the best next
experiment.

No additional JEPA-first run is justified now. Reopen a specific conditional
family only after its missing prerequisite is measured. In the meantime, the
next model should be the already specified supervised,
intervention-conditioned temporal graph state-space model; a JEPA auxiliary
should earn inclusion only after that supervised model establishes useful
action-conditioned dynamics
([graph-dynamics specification](../specs/action-conditioned-graph-dynamics-v1.md),
[frontier conclusion](jepa-frontier-execution-conclusion-2026.md)).

## What the MPRM result falsified

The latest result rejects one exact treatment: a four-component, seed-307,
40-epoch EMA-target residual-mixture JEPA, with diagonal component variances
and a frozen raw rank-32 mean anchor, on the fixed local Docker Compose stack.
The protocol froze one width, optimizer, loss weighting, component count, and
seed with no sweep
([protocol](../../lab/action_dynamics/mprm-jepa-protocol-v1.json),
[implementation](../../lab/action_dynamics/mprm_jepa_model.py),
[contract](../specs/mean-preserving-residual-mixture-jepa-proposal-v1.md)).

Mean centering worked: candidate and raw point MSE were identical. The
candidate nevertheless lost the primary pair-balanced log score to both the
raw core (`-0.550848` versus `-2.915059`, lower is better) and the supervised
mean-preserving mixture (`-0.555661`), failed a workload-family safety gate,
and returned paired-randomization `p = 1.0`. The retained result explicitly
says this rejects the exact recipe, not mean-preserving mixtures generally
([MPRM result](mean-preserving-residual-mixture-jepa-v1-results.md)).

MPRM alone did not test every component count, seed, covariance structure,
autoregressive mode, retrieval-conditioned future, or generative latent
distribution. Those omissions do not automatically become good next
experiments: the wider Quantis program separately tested hard
winner-specialization, retrieval, codebooks, geometry changes, causal/effect
objectives, task grounding, and other materially different mechanisms
([multi-hypothesis result](multi-hypothesis-jepa-prototype-v2-results.md),
[MoP-JEPA result](mop-jepa-hard-assignment-v1-results.md),
[execution matrix](jepa-frontier-execution-conclusion-2026.md)).

## What the program has ruled out locally

The retained experiment matrix covers EMA and non-EMA targets, masked and
mask-free prediction, direct multi-horizon targets, several mask families,
multiple anti-collapse geometries, event-time prediction, retrieval, soft
codebooks, likelihood mixtures, hard winner-take-all mixtures, matched-pair
effect prediction, task-grounded residual contracts, error certificates,
static teachers, discrete tokens, and exact density scoring. Every tested
recipe failed its frozen conjunction of mechanism, observable-value, safety,
and operational gates
([execution conclusion](jepa-frontier-execution-conclusion-2026.md),
[exhaustion refresh](jepa-frontier-exhaustion-refresh-2026-07-29.md)).

The recurring local result is narrower than “JEPA does not work”: learned
bottlenecks repeatedly discarded entity-local state or action effect already
available to the compact raw dynamics, while representation diagnostics such
as rank, code use, state probing, or component specialization did not
translate into proper-score, attribution, calibration, or alert value. The
fixed corpus is also close to deterministic after declared controls, so extra
future heads have not demonstrated irreducible conditional ambiguity
([frontier conclusion](jepa-frontier-execution-conclusion-2026.md),
[MoP-JEPA result](mop-jepa-hard-assignment-v1-results.md),
[MPRM result](mean-preserving-residual-mixture-jepa-v1-results.md)).

Therefore the evidence supports these bounded conclusions:

| Claim | Status |
|---|---|
| The exact MPRM-JEPA v1 recipe should advance | Ruled out |
| A generic JEPA objective is the best next trainer on the current targets and corpus | Ruled out |
| Any tested JEPA is ready for the paging path | Ruled out |
| JEPA cannot work on other data regimes, products, or domains | Not supported |
| Every possible JEPA objective or probabilistic formulation has been tested | False |

## Genuinely untested conditional families

These are real omissions, but none is currently a high-value run without new
evidence:

| Family | Required evidence before reopening | Why not now |
|---|---|---|
| T-JEPA / arbitrary missing-channel targets | A declared missing-telemetry lane and measured corruption incidence | Missing-channel robustness is not the current alerting contract |
| BiJEPA / cycle-consistent forward-backward prediction | Adequately represented recovery and reverse trajectories | The present evidence does not establish the reverse-path coverage its claim needs |
| Gaussian-embedding, VJEPA/BJEPA, Var-JEPA, diffusion, or other belief JEPAs | Selection-role heteroscedasticity or multimodality that beats calibrated deterministic residuals | Both likelihood and hard-assignment multi-future tracers lost to raw or supervised controls |
| Koopman- or physics-constrained JEPA | Separable precursor regimes or a trusted, fully observed queue/flow invariant | Existing regime and geometry experiments did not establish those prerequisites |
| Cross-modal or language-conditioned JEPA | Aligned, information-bearing traces/events/operator language | Prior event evidence added no operational value and the required aligned modality is absent |

These prerequisites and deferrals are part of the retained execution decision,
not untracked ideas
([conditional-family table](jepa-frontier-execution-conclusion-2026.md),
[frontier audit](jepa-frontier-technique-audit-2026.md)).

## Recommendation

Do not tune MPRM against its exposed selection campaign, vary seeds or
component counts as a rescue, or port another JEPA backbone onto the same
target. Run the supervised intervention-conditioned graph state-space program
with raw, no-action, topology-shuffled, and capacity-matched controls. In
parallel, measure missing-channel frequency, recovery/reverse-path coverage,
and residual heteroscedasticity. Only a positive measurement on one of those
axes should reopen the corresponding JEPA family under a new protocol and
fresh opaque evidence
([graph-dynamics specification](../specs/action-conditioned-graph-dynamics-v1.md),
[MPRM evidence boundary](mean-preserving-residual-mixture-jepa-v1-results.md),
[frontier recommendation](jepa-frontier-execution-conclusion-2026.md)).
