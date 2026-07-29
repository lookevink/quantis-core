# Multi-hypothesis trajectory JEPA prototype v2 correction

## Status

This correction was frozen after review invalidated the first prototype
artifact, and before producing the corrected assessment artifact.

The v1 fitted models and stored predictions are reusable numeric sidecars:
their recipe, seed, training roles, restoration parity, and prediction values
were not implicated. The v1 decision is invalid because:

- transfer-output finiteness was accidentally included in the selection
  conjunction, allowing an evaluation role to influence safe-null selection;
  and
- the runner stopped after safe-null failure even though the v1 prototype
  document promised calibrated, energy, alert, and investigation outputs.

The v2 result binds the complete v1 artifact by SHA-256, recomputes selection
from its stored arrays, and makes the intended fail-fast boundary explicit.
It is a correction of already-open development evidence, not fresh evidence.

The only admissible v1 source has:

- artifact-manifest SHA-256
  `1a464d6182b4f0abd6987496453ef5f9ef403d9ab62779ffa87e7511184528f8`;
  and
- result SHA-256
  `295ac75bbff1f85f3cb72833b11e6543fb082a5e027e9f20f814ff529a6c1760`.

The assessor rejects any other bundle before reading predictions.

## Unchanged recipe and evidence

The candidate, controls, seed `307`, 40-epoch fitting budget, data roles,
architecture, objective, stored distributions, and thresholds remain exactly
those preregistered by
[multi-hypothesis trajectory JEPA prototype v1](multi-hypothesis-jepa-prototype-v1.md).
No model is refitted, tuned, or selected again.

The corrected assessor verifies the source artifact manifest and model
restoration records before interpreting any score.

## Corrected safe-null assessment

Selection uses only selection-role arrays:

1. Recompute exact complete-trajectory mixture log score per window.
2. Average windows within each logical trajectory.
3. Average the treatment and control trajectories within each matched pair.
4. Weight matched pairs equally.
5. Recompute moment-matched overall and action-overlap MSE.
6. Recompute supported-pair rate on action-overlap samples.
7. Require finite selection-role values.

Transfer metrics remain labelled diagnostics and cannot enter the selection
conjunction.

The four-component candidate remains eligible only if every safe-null rule in
the frozen scoring contract passes. Otherwise, select the fitted
single-component control with the lowest pair-balanced raw selection log score
and stop. In that case:

- do not fit calibration scalars;
- do not use in-distribution or transfer evaluation to rescue the candidate;
- do not spend the 256-draw energy, alert, or investigation assessment budget;
  and
- conclude only that this recipe cannot advance beyond safe-null selection.

This fail-fast result does not claim measured alert or investigation failure.
It rejects the candidate because the preregistered selection and
point-prediction safety requirements are necessary before those value lanes
can authorize promotion.

## Corrected artifact

The v2 bundle contains:

- the v2 protocol;
- hashes of the complete v1 source manifest and result;
- the independently recomputed selection and diagnostic transfer metrics;
- each corrected gate and the selected safe null;
- an explicit statement that calibration and value-lane assessment were not
  reached; and
- its own content-addressed manifest.

The corrected output directory is
`artifacts/action-dynamics/prototype-multi-hypothesis-jepa-v2`.
