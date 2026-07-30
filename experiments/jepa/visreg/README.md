# VISReg scale-shape regularization

**Status:** Rejected

The small-radius mechanism passed, but the detached candidate collapsed and retained 1.97× raw transfer effect error.

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
- [VISReg](https://arxiv.org/abs/2606.02572)
- [Pinned VISReg implementation](https://github.com/HaiyuWu/visreg/tree/47b1cf4d725b6cbc76dae1394eb46acc2d282fc1)

## Artifact

- Local artifact: `artifacts/action-dynamics/prototype-visreg-v1`
- Published artifact directories are immutable.
- The artifact is intentionally not duplicated into this capsule;
  its manifest and result document bind the evidence identity.

## Evidence boundary

Open-development evidence on the fixed Quantis checkout stack; no entry authorizes production paging.

Use the [JEPA reproduction guide](../../../lab/action_dynamics/JEPA_REPRODUCTION.md)
for exact environment assumptions and fresh-output rules.
