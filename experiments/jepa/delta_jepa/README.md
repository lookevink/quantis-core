# Delta-JEPA action-sensitive displacement

**Status:** Rejected

The displacement contained a real action signal, but endpoint concatenation decoded actions better and raw effect prediction remained much stronger.

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
- [Delta-JEPA](https://arxiv.org/abs/2606.31232)

## Artifact

- Local artifact: `artifacts/action-dynamics/prototype-delta-jepa-v1`
- Fetch after distribution metadata is recorded: `python tools/artifacts.py fetch delta_jepa`
- Published artifact directories are immutable.
- The artifact is intentionally not duplicated into this capsule;
  its manifest and result document bind the evidence identity.

## Evidence boundary

Open-development evidence on the fixed Quantis checkout stack; no entry authorizes production paging.

Use the [JEPA reproduction guide](../../../lab/action_dynamics/JEPA_REPRODUCTION.md)
for exact environment assumptions and fresh-output rules.
