# Causal-JEPA entity intervention

**Status:** Rejected

Whole-entity masking learned completion beyond persistence but lost to matched coordinate-time masking and regressed raw effects.

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
- [Causal-JEPA](https://arxiv.org/abs/2602.11389)

## Artifact

- Local artifact: `artifacts/action-dynamics/prototype-causal-jepa-v1`
- Fetch after distribution metadata is recorded: `python tools/artifacts.py fetch causal_jepa`
- Published artifact directories are immutable.
- The artifact is intentionally not duplicated into this capsule;
  its manifest and result document bind the evidence identity.

## Evidence boundary

Open-development evidence on the fixed Quantis checkout stack; no entry authorizes production paging.

Use the [JEPA reproduction guide](../../../lab/action_dynamics/JEPA_REPRODUCTION.md)
for exact environment assumptions and fresh-output rules.
