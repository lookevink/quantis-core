# SALT-JEPA static reconstructive teacher

**Status:** Rejected

The corrected run passed its evidence checks, but transfer latent advantage missed the gate and effect error remained 1.91× raw.

## Experiment interface

- [Frozen specification](spec.md)
- [Conclusion-bearing findings](findings.md)
- [Exact runner](run.py)
- [Independent assessor](assess.py)
- [Library implementation](implementation.py)
- [Behavioral test](test.py)
- [Program ticket](ticket.md)
- [Additional retained findings](findings-2.md)

## Primary references

- [Pinned primary-source notes](references.md)
- [SALT: Compute-Efficient Video SSL with Frozen Teachers](https://arxiv.org/abs/2509.24317)

## Artifact

- Local artifact: `artifacts/action-dynamics/prototype-salt-jepa-v2`
- Published artifact directories are immutable.
- The artifact is intentionally not duplicated into this capsule;
  its manifest and result document bind the evidence identity.

## Evidence boundary

Open-development evidence on the fixed Quantis checkout stack; no entry authorizes production paging.

Use the [JEPA reproduction guide](../../../lab/action_dynamics/JEPA_REPRODUCTION.md)
for exact environment assumptions and fresh-output rules.
