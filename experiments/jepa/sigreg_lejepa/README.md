# Exact SIGReg LeJEPA substitution

**Status:** Rejected

SIGReg broadened latent rank but worsened state probing, earned zero correction gain, and detected no treatments.

## Experiment interface

- [Frozen specification](spec.md)
- [Conclusion-bearing findings](findings.md)
- [Exact runner](run.py)
- [Library implementation](implementation.py)
- [Behavioral test](test.py)
- [Program ticket](ticket.md)
- [Additional behavioral test](test-2.py)

## Primary references

- [Pinned primary-source notes](references.md)
- [LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics](https://arxiv.org/abs/2511.08544)
- [Pinned official LeJEPA implementation](https://github.com/galilai-group/lejepa/tree/c293d291ca87cd4fddee9d3fffe4e914c7272052)

## Artifact

- Local artifact: `artifacts/action-dynamics/prototype-sigreg-lejepa-v1`
- Fetch after distribution metadata is recorded: `python tools/artifacts.py fetch sigreg_lejepa`
- Published artifact directories are immutable.
- The artifact is intentionally not duplicated into this capsule;
  its manifest and result document bind the evidence identity.

## Evidence boundary

Open-development evidence on the fixed Quantis checkout stack; no entry authorizes production paging.

Use the [JEPA reproduction guide](../../../lab/action_dynamics/JEPA_REPRODUCTION.md)
for exact environment assumptions and fresh-output rules.
