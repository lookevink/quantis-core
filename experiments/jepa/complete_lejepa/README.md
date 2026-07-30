# Complete multi-view LeJEPA

**Status:** Rejected

The predictor-free representation was restorable and state-accessible but lost to reconstruction and nearly doubled raw effect error.

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
- [LeJEPA](https://arxiv.org/abs/2511.08544)

## Artifact

- Local artifact: `artifacts/action-dynamics/prototype-complete-lejepa-v1`
- Fetch after distribution metadata is recorded: `python tools/artifacts.py fetch complete_lejepa`
- Published artifact directories are immutable.
- The artifact is intentionally not duplicated into this capsule;
  its manifest and result document bind the evidence identity.

## Evidence boundary

Open-development evidence on the fixed Quantis checkout stack; no entry authorizes production paging.

Use the [JEPA reproduction guide](../../../lab/action_dynamics/JEPA_REPRODUCTION.md)
for exact environment assumptions and fresh-output rules.
