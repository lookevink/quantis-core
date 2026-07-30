# Mean-preserving residual mixture JEPA

**Status:** Active

The executable selection tracer is frozen and ready for fit-only model freezing; no fresh selection evidence has been collected.

## Experiment interface

- [Frozen specification](spec.md)
- [Conclusion-bearing findings](findings.md)
- [Exact runner](run.py)
- [Independent assessor](assess.py)
- [Library implementation](implementation.py)
- [Behavioral test](test.py)
- [Program ticket](ticket.md)
- [Supporting script](supporting-script-2.py)

## Primary references

- [Quantis multi-hypothesis JEPA scoring contract](../../../docs/specs/multi-hypothesis-jepa-scoring-contract-v1.md)

## Artifact

- Local artifact: `artifacts/action-dynamics/mprm-jepa-selection-v1-attempt-001`
- Supporting artifact: `artifacts/action-dynamics/richer-regime-retry-v1`
- Supporting artifact: `artifacts/action-dynamics/richer-regime-retry-v1-validity-audit-v4`
- Fetch after distribution metadata is recorded: `python tools/artifacts.py fetch mprm_jepa`
- Published artifact directories are immutable.
- The artifact is intentionally not duplicated into this capsule;
  its manifest and result document bind the evidence identity.

## Evidence boundary

Open-development evidence on the fixed Quantis checkout stack; no entry authorizes production paging.

Use the [JEPA reproduction guide](../../../lab/action_dynamics/JEPA_REPRODUCTION.md)
for exact environment assumptions and fresh-output rules.
