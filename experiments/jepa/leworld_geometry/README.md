# LeWorldModel bounded geometry screen

**Status:** Rejected

No regularized cell was raw-safe; SPHERE-JEPA narrowly won a diagnostic before losing transfer value and state information.

## Experiment interface

- [Frozen specification](spec.md)
- [Conclusion-bearing findings](findings.md)
- [Exact runner](run.py)
- [Independent assessor](assess.py)
- [Library implementation](implementation.py)
- [Behavioral test](test.py)
- [Program ticket](ticket.md)

## Primary references

- [Pinned primary-source notes](references.md)
- [LeWorldModel](https://arxiv.org/abs/2603.19312)
- [Geometry-family primary-source matrix](../../../docs/research/leworld-geometry-primary-source-notes.md)

## Artifact

- Local artifact: `artifacts/action-dynamics/prototype-leworld-geometry-v1`
- Fetch after distribution metadata is recorded: `python tools/artifacts.py fetch leworld_geometry`
- Published artifact directories are immutable.
- The artifact is intentionally not duplicated into this capsule;
  its manifest and result document bind the evidence identity.

## Evidence boundary

Open-development evidence on the fixed Quantis checkout stack; no entry authorizes production paging.

Use the [JEPA reproduction guide](../../../lab/action_dynamics/JEPA_REPRODUCTION.md)
for exact environment assumptions and fresh-output rules.
