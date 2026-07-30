# Multi-hypothesis trajectory JEPA

**Status:** Rejected

Distinct alternatives emerged, but the mixture badly regressed raw prediction both originally and in a stronger three-regime retry.

## Experiment interface

- [Frozen specification](spec.md)
- [Conclusion-bearing findings](findings.md)
- [Exact runner](run.py)
- [Independent assessor](assess.py)
- [Behavioral test](test.py)
- [Program ticket](ticket.md)
- [Additional retained findings](findings-2.md)
- [Supporting specification](supporting-spec-2.md)
- [Supporting specification](supporting-spec-3.md)
- [Additional behavioral test](test-2.py)
- [Supporting script](supporting-script-2.py)
- [Supporting script](supporting-script-3.py)
- [Supporting script](supporting-script-4.py)

## Primary references

- [Quantis likelihood-mixture scoring contract](../../../docs/specs/multi-hypothesis-jepa-scoring-contract-v1.md)

## Artifact

- Local artifact: `artifacts/action-dynamics/prototype-multi-hypothesis-jepa-v1`
- Fetch after distribution metadata is recorded: `python tools/artifacts.py fetch multi_hypothesis`
- Published artifact directories are immutable.
- The artifact is intentionally not duplicated into this capsule;
  its manifest and result document bind the evidence identity.

## Evidence boundary

Open-development evidence on the fixed Quantis checkout stack; no entry authorizes production paging.

Use the [JEPA reproduction guide](../../../lab/action_dynamics/JEPA_REPRODUCTION.md)
for exact environment assumptions and fresh-output rules.
