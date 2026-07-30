# HEPA event-predictive alerting

**Status:** Rejected

HEPA learned a monotone, state-rich probability surface but detected only 50% of treatments and trailed simpler controls.

## Experiment interface

- [Frozen specification](spec.md)
- [Conclusion-bearing findings](findings.md)
- [Exact runner](run.py)
- [Independent assessor](assess.py)
- [Library implementation](implementation.py)
- [Behavioral test](test.py)
- [Program ticket](ticket.md)

## Primary references

- [HEPA: A Horizon-Conditioned Event Predictive Architecture](https://arxiv.org/abs/2605.11130)

## Artifact

- Local artifact: `artifacts/action-dynamics/prototype-hepa-jepa-v1`
- Fetch after distribution metadata is recorded: `python tools/artifacts.py fetch hepa`
- Published artifact directories are immutable.
- The artifact is intentionally not duplicated into this capsule;
  its manifest and result document bind the evidence identity.

## Evidence boundary

Open-development evidence on the fixed Quantis checkout stack; no entry authorizes production paging.

Use the [JEPA reproduction guide](../../../lab/action_dynamics/JEPA_REPRODUCTION.md)
for exact environment assumptions and fresh-output rules.
