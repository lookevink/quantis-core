# Exact JEPA-SCORE edge alerting

**Status:** Rejected

Exact Jacobian/SVD scoring passed every protocol and edge gate but detected only 10% of IID treatments and no transfer treatments.

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
- [Gaussian Embeddings: How JEPAs Secretly Learn Your Data Density](https://arxiv.org/abs/2510.05949)

## Artifact

- Local artifact: `artifacts/action-dynamics/prototype-jepa-score-v1`
- Fetch after distribution metadata is recorded: `python tools/artifacts.py fetch jepa_score`
- Published artifact directories are immutable.
- The artifact is intentionally not duplicated into this capsule;
  its manifest and result document bind the evidence identity.

## Evidence boundary

Open-development evidence on the fixed Quantis checkout stack; no entry authorizes production paging.

Use the [JEPA reproduction guide](../../../lab/action_dynamics/JEPA_REPRODUCTION.md)
for exact environment assumptions and fresh-output rules.
