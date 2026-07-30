# Soft regime-codebook JEPA

**Status:** Rejected

Broad code usage and improved state probing did not preserve effect prediction or alert calibration.

## Experiment interface

- [Frozen specification](spec.md)
- [Conclusion-bearing findings](findings.md)
- [Exact runner](run.py)
- [Program ticket](ticket.md)

## Primary references

- [Quantis soft regime-codebook formulation](../../../docs/specs/regime-codebook-jepa-prototype-v1.md)

## Artifact

- Local artifact: `artifacts/action-dynamics/prototype-regime-codebook-jepa-v1`
- Fetch after distribution metadata is recorded: `python tools/artifacts.py fetch regime_codebook`
- Published artifact directories are immutable.
- The artifact is intentionally not duplicated into this capsule;
  its manifest and result document bind the evidence identity.

## Evidence boundary

Open-development evidence on the fixed Quantis checkout stack; no entry authorizes production paging.

Use the [JEPA reproduction guide](../../../lab/action_dynamics/JEPA_REPRODUCTION.md)
for exact environment assumptions and fresh-output rules.
