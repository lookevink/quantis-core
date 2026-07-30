# Task-grounded Contract-JEPA

**Status:** Rejected

The raw-preserving contract passed every safety gate and improved raw effect error by 1.05%, but both controls were better and witness calibration failed.

## Experiment interface

- [Frozen specification](spec.md)
- [Conclusion-bearing findings](findings.md)
- [Exact runner](run.py)
- [Independent assessor](assess.py)
- [Library implementation](implementation.py)
- [Behavioral test](test.py)
- [Program ticket](ticket.md)

## Primary references

- [Quantis task-grounded residual contract and related work](../../../docs/specs/task-grounded-contract-jepa-tracer-v1.md)

## Artifact

- Local artifact: `artifacts/action-dynamics/prototype-task-grounded-contract-jepa-v2`
- Fetch after distribution metadata is recorded: `python tools/artifacts.py fetch contract_jepa`
- Published artifact directories are immutable.
- The artifact is intentionally not duplicated into this capsule;
  its manifest and result document bind the evidence identity.

## Evidence boundary

Open-development evidence on the fixed Quantis checkout stack; no entry authorizes production paging.

Use the [JEPA reproduction guide](../../../lab/action_dynamics/JEPA_REPRODUCTION.md)
for exact environment assumptions and fresh-output rules.
