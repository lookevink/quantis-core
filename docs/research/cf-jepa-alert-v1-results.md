# CF-JEPA alert tracer v1 results

## Decision

Reject this exact edge CF-JEPA recipe.

The experiment verified the paper’s online/EMA geometry claim and found an
unusually state-rich target representation, but the complete three-zone
candidate did not improve alert value over the simpler one-zone forward
objective. It also exceeded the frozen transfer control false-alarm ceiling.
The result rejects this telemetry adaptation, not CF-JEPA on the authors’
benchmarks or mask-free forward prediction at another scale.

## Frozen identity

- Implementation commit:
  `3b7cb81a277b5d7c48a6946735c9e5e0012bcc54`
- Official CF-JEPA revision:
  `4968faf731c8c56e89d78d944716e212392eb5a0`
- Artifact:
  `artifacts/action-dynamics/prototype-cf-jepa-alert-v1`
- Artifact-manifest SHA-256:
  `59dd147c359501d2ff10d32117c2dfbd5e65f836ec5ce31a3b19c149e7fd2c08`
- Independent assessment SHA-256:
  `9a35e0470aecb8ada13ffd0e9e232b0835c13ca2711884f260083b1427a7e260`
- Seed: `14014`
- Objective schedule: 300 steps, checkpoints every 50, four crops, cosine
  learning rate, and cosine EMA momentum
- Selected checkpoints: three-zone 250, one-zone 250, masked-latent 300

The independent assessor re-fitted every monotone calibrator from stored raw
scores, re-derived every control-maximum threshold and alert decision, and
verified the artifact manifest. Original/restored scores, probabilities,
decisions, and candidate temporal representations reproduced exactly.

## Held-topology alert result

| Route | Brier | control-trajectory FPR | treatment detection | median delay | alerts/run |
|---|---:|---:|---:|---:|---:|
| three-zone EMA target | 0.034094 | 0.10 | 0.90 | 1 | 7.30 |
| three-zone online | 0.036024 | 0.20 | 0.90 | 1 | 9.85 |
| one-zone EMA target | **0.034029** | 0.10 | 0.90 | 1 | **7.25** |
| masked-latent EMA target | 0.034127 | 0.10 | 0.90 | 1 | 7.60 |
| matched PCA | 0.037644 | 0.20 | 0.90 | 1 | 16.45 |

The target route improved Brier by 5.36% over the same model’s online route
and by 9.43% over matched PCA. Those are useful route-level results.
However, the complete candidate was 0.19% worse than the one-zone target and
only 0.10% better than masked latent. Its required Brier was at most
`0.95 × 0.034029 = 0.032328`; the observed `0.034094` missed that gate.

All neural EMA routes had 10% transfer control-trajectory false alarms,
twice the 5% ceiling, and the online/PCA routes had 20%. Every route detected
the same 90% of treatment trajectories with the same one-transition median
delay. Three zones therefore added neither sensitivity nor delay value over
one zone or masked latent.

## The source mechanism did appear

The online/EMA asymmetry was strong:

| Geometry metric | EMA target | Online |
|---|---:|---:|
| adjacent-timestep cosine similarity | 0.988752 | 0.768243 |
| 90%-variance effective rank | 6 | 16 |

The EMA target is much smoother and lower rank, exactly the direction
reported by CF-JEPA. This is not a null-result caused by a missing teacher
effect.

Observable-state retention was also excellent. The target representation’s
held-topology aggregate state-probe NRMSE was `0.043696`, versus `0.428863`
for matched entity-local PCA. It passed all six varying observed entities.
That is the strongest state-retention result in this tracer, but state access
alone did not distinguish the three-zone objective from its simpler alert
competitors.

## Safety and edge checks

The following gates passed:

- every stored numeric array was finite;
- restored neural and PCA models reproduced their outputs exactly;
- all neural alert routes deployed the same 23,590-parameter encoder;
- target state retention passed the `1.05 ×` PCA bound;
- target smoothness and effective-rank asymmetry both appeared;
- candidate model, Gaussian adapter, event definition, and probe used
  10,822,644 bytes, below 16 MiB;
- batch-one target latency was 4.86 ms mean and 5.25 ms p95 on the local CPU;
- causal-history changes changed output, while forbidden future/action
  counterfactuals could not enter the public seam; and
- independent stored-array assessment and the SHA-256 manifest verified.

Peak process RSS was 3,428,302,848 bytes. That is a training/evaluation
process measurement, not a target-device memory claim.

The predictive alert-score and trajectory alert lanes both failed. Since all
safety gates passed, the rejection is about incremental value rather than
collapse, leakage, restoration, capacity, or edge size.

## Retained failures

No code or evidence used to debug the tracer was deleted:

- `prototype-cf-jepa-alert-v1-smoke-1step.building` retains the unbounded
  Mahalanobis/calibrator contract failure;
- `prototype-cf-jepa-alert-v1-smoke-1step-bounded.building` retains the first
  missing PCA restoration failure; and
- `prototype-cf-jepa-alert-v1-attempt-b034783-pca-restore-failure` retains
  the full-run failure that exposed JSON’s lost `0×0` component shape.

The final runner writes and immediately restores each selected objective
checkpoint before continuing, so later failures preserve expensive fitting
work in staging.

## Conclusion and next target

CF-JEPA contributes two useful lessons for a real alerting system: route EMA
features to temporally smooth tasks, and retain the online representation
when discrimination matters. The three-zone forward objective itself did
not earn deployment or robustness work here; a single forward zone was
slightly better and no neural route controlled false alarms tightly enough.

Proceed to SD-JEPA. Its smallest falsifier should hold total width and alert
adapter fixed while testing whether an explicit orthogonal progression
subspace localizes operational change better than a same-width unsplit
representation. Unlike CF-JEPA, that next mechanism directly assigns a
coordinate to progress toward impact rather than relying on generic anomaly
distance.
