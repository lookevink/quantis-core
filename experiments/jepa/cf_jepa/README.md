# CF-JEPA mask-free multi-horizon alerting

**Status:** Rejected

The EMA target was smooth and state-rich, but three alert zones lost to one and control false alarms exceeded the ceiling.

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
- [CF-JEPA](https://arxiv.org/abs/2606.07031)

## Artifact

- Local artifact: `artifacts/action-dynamics/prototype-cf-jepa-alert-v1`
- Published artifact directories are immutable.
- The artifact is intentionally not duplicated into this capsule;
  its manifest and result document bind the evidence identity.

## Evidence boundary

Open-development evidence on the fixed Quantis checkout stack; no entry authorizes production paging.

Use the [JEPA reproduction guide](../../../lab/action_dynamics/JEPA_REPRODUCTION.md)
for exact environment assumptions and fresh-output rules.
