# JEPA experiment reproduction

The JEPA prototype runners and assessors are retained even when a recipe is
rejected. They are evidence-producing code, not supported production APIs.
The published artifact directories are immutable, and every rerun must use a
new output directory.

Run commands from the repository root with the project virtual environment.
Replace `rerun-001` before repeating a command.

## Soft regime-codebook JEPA

```bash
.venv/bin/python lab/action_dynamics/prototype_regime_codebook_jepa.py \
  --output artifacts/action-dynamics/reproductions/regime-codebook-rerun-001
```

- Seed: `127`
- Specification:
  `docs/specs/regime-codebook-jepa-prototype-v1.md`
- Result interpretation:
  `docs/research/regime-codebook-jepa-prototype-v1-results.md`
- Published local result:
  `artifacts/action-dynamics/prototype-regime-codebook-jepa-v1/prototype-result.json`
- Published result SHA-256:
  `b21bc61f2c333af7e4d6667bc15e83949955bfe8b7fdf988473f6a46333a29cf`

## Event-native trace JEPA

```bash
.venv/bin/python lab/action_dynamics/prototype_event_native_trace_jepa.py \
  --output artifacts/action-dynamics/reproductions/event-native-rerun-001
```

- Seed: `211`
- Specification:
  `docs/specs/event-native-trace-jepa-prototype-v1.md`
- Result interpretation:
  `docs/research/event-native-trace-jepa-prototype-v1-results.md`
- Published local result:
  `artifacts/action-dynamics/prototype-event-native-trace-jepa-v1/prototype-result.json`
- Published result SHA-256:
  `38aa22cf268003d29dbf1cb59da926ced417d977a33747466e3957399a093b43`

## Multi-hypothesis trajectory JEPA

Reproduce the original fit and stored prediction sidecars:

```bash
.venv/bin/python lab/action_dynamics/prototype_multi_hypothesis_jepa.py \
  --output artifacts/action-dynamics/reproductions/multi-hypothesis-v1-rerun-001
```

Reproduce the corrected decision from the immutable published v1 sidecars
without refitting any model:

```bash
.venv/bin/python \
  lab/action_dynamics/prototype_multi_hypothesis_jepa_v2_assessor.py \
  --source artifacts/action-dynamics/prototype-multi-hypothesis-jepa-v1 \
  --output artifacts/action-dynamics/reproductions/multi-hypothesis-v2-rerun-001
```

- Seed: `307`
- Original specification:
  `docs/specs/multi-hypothesis-jepa-prototype-v1.md`
- Corrected assessment specification:
  `docs/specs/multi-hypothesis-jepa-prototype-v2.md`
- Result interpretation:
  `docs/research/multi-hypothesis-jepa-prototype-v2-results.md`
- Published v1 manifest SHA-256:
  `1a464d6182b4f0abd6987496453ef5f9ef403d9ab62779ffa87e7511184528f8`
- Published v2 manifest SHA-256:
  `aa47dd6b28dbe31ec99ccb908296a2a4f66a9ba3cf2a12299894d38e296f14a9`

The v1 bundle is retained as the numeric source for the v2 correction, but its
original decision is invalid. The v2 assessor verifies the v1 artifact
manifest, source identities, and model restoration records before producing a
new corrected bundle.

## Exact SIGReg regularizer substitution

```bash
.venv/bin/python lab/action_dynamics/prototype_sigreg_lejepa.py \
  --output artifacts/action-dynamics/reproductions/sigreg-rerun-001
```

- Neural seed: `401`
- SIGReg projection seed: `1401`
- Source preset: LeJEPA `official-minimal-c293d29`
- Specification:
  `docs/specs/sigreg-lejepa-prototype-v1.md`
- Primary-source implementation notes:
  `docs/research/lejepa-sigreg-primary-source-notes.md`
- Result interpretation:
  `docs/research/sigreg-lejepa-prototype-v1-results.md`
- Published local result:
  `artifacts/action-dynamics/prototype-sigreg-lejepa-v1/prototype-result.json`
- Published manifest SHA-256:
  `912548d38056ce910394a6b675f65277becd2590b5b65b88029ace0af830385d`

This runner tests an exact SIGReg substitution in the existing residual JEPA.
It does not claim to reproduce the complete LeJEPA training objective.

## Complete multi-view LeJEPA representation

Run the exact frozen fit in a fresh output directory:

```bash
.venv/bin/python lab/action_dynamics/prototype_complete_lejepa.py \
  --output artifacts/action-dynamics/reproductions/complete-lejepa-rerun-001
```

Independently verify the manifest and recompute every assessment gate from
stored arrays without invoking a fitted encoder:

```bash
.venv/bin/python \
  lab/action_dynamics/prototype_complete_lejepa_assessor.py \
  artifacts/action-dynamics/reproductions/complete-lejepa-rerun-001
```

- Frozen seeds: initialization `509`, anchors `1509`, views `2509`, SIGReg
  `3509`, masked-autoencoder decoder `4509`
- Exact schedule: 1,600 steps, 40 pair-blocked anchors, eight views, 1,024
  SIGReg directions, 17 knots
- Specification:
  `docs/specs/complete-lejepa-telemetry-contract-v1.md`
- Result interpretation:
  `docs/research/complete-lejepa-telemetry-prototype-v1-results.md`
- Published local result:
  `artifacts/action-dynamics/prototype-complete-lejepa-v1/result.json`
- Published manifest SHA-256:
  `00639afaee81cd3844e8b60014ed87f886a8e7a7e0b20bc50834416da1deb265`

The runner retains incomplete work under a `.building` directory and refuses
to overwrite either incomplete or published results. Non-default `--steps`
or `--sketch-dimension` runs are labeled non-interpretable smoke runs.

## Episode-predictive retrieval JEPA

Run the frozen causal fit and exact evidence retrieval in a fresh directory:

```bash
.venv/bin/python lab/action_dynamics/prototype_retrieval_jepa.py \
  --output artifacts/action-dynamics/reproductions/retrieval-jepa-rerun-001
```

Independently verify the manifest and recompute rankings, empirical
abstention, risk-coverage curves, state safety, restoration parity, and the
decision from stored arrays:

```bash
.venv/bin/python \
  lab/action_dynamics/prototype_retrieval_jepa_assessor.py \
  artifacts/action-dynamics/reproductions/retrieval-jepa-rerun-001
```

- Seed: `9019`
- Exact schedule: 400 steps, 40 pair-blocked anchors, width 64
- Evidence bank: 40 treatment episodes, exact cosine `K=3`
- Specification:
  `docs/specs/retrieval-jepa-evidence-contract-v1.md`
- Primary-source notes:
  `docs/research/retrieval-jepa-primary-source-notes.md`
- Result interpretation:
  `docs/research/retrieval-jepa-prototype-v1-results.md`
- Published local result:
  `artifacts/action-dynamics/prototype-retrieval-jepa-v1/assessment.json`
- Published manifest SHA-256:
  `676b0cba10ebb66a77fe66a33376b06f418ee10736a12da3a3de3e8d991cc0ba`

The exact cache identity, 400 steps, and 100 latency repetitions are required
for an interpretable run. Overrides require
`--allow-noninterpretable-smoke`, cannot use the frozen result path, and do not
support a scientific decision.

## Complete SC-JEPA interaction

Run the exact capacity-matched four-cell factorial in a fresh output
directory:

```bash
.venv/bin/python lab/action_dynamics/prototype_sc_jepa_interaction.py \
  --output artifacts/action-dynamics/reproductions/sc-jepa-rerun-001
```

Independently verify the manifest and rederive the complete decision from
stored arrays:

```bash
.venv/bin/python \
  lab/action_dynamics/prototype_sc_jepa_interaction_assessor.py \
  artifacts/action-dynamics/reproductions/sc-jepa-rerun-001
```

- Seed: `13013`
- Exact schedule: 300 representation steps, 200 downstream-head steps, 100
  latency repetitions
- Implementation commit:
  `5b6db454619aa1d6555a4cc457535c0bad86a446`
- Specification:
  `docs/specs/sc-jepa-interaction-v1.md`
- Primary-source notes:
  `docs/research/sc-jepa-primary-source-notes.md`
- Result interpretation:
  `docs/research/sc-jepa-interaction-v1-results.md`
- Published local artifact:
  `artifacts/action-dynamics/prototype-sc-jepa-interaction-v1`
- Published manifest SHA-256:
  `ce88511e935edbc2704de12ac995224162d0165a2f6284f9a2c276aa6989fea8`

The exact content-addressed cache, frozen schedule, source-clean commit, and
100 latency repetitions are required for an interpretable run. Overrides
require `--allow-noninterpretable-smoke`, cannot use the frozen result path,
and are structurally ineligible for an advance decision.

## CF-JEPA mask-free multi-horizon alerting

Run the frozen three-objective fit and five-route Gaussian alert comparison:

```bash
.venv/bin/python -m lab.action_dynamics.prototype_cf_jepa \
  --output artifacts/action-dynamics/reproductions/cf-jepa-rerun-001
```

Independently verify the manifest and rederive calibrations, thresholds,
geometry, state retention, alerts, restoration, and the decision from stored
arrays:

```bash
.venv/bin/python -m \
  lab.action_dynamics.prototype_cf_jepa_assessor \
  artifacts/action-dynamics/reproductions/cf-jepa-rerun-001
```

- Seed: `14014`
- Exact schedule: 300 steps per objective, checkpoints every 50, four crops,
  cosine learning rate and EMA schedules, 100 latency repetitions
- Implementation commit:
  `3b7cb81a277b5d7c48a6946735c9e5e0012bcc54`
- Official CF-JEPA revision:
  `4968faf731c8c56e89d78d944716e212392eb5a0`
- Specification:
  `docs/specs/cf-jepa-alert-tracer-v1.md`
- Primary-source notes:
  `docs/research/cf-jepa-primary-source-notes.md`
- Result interpretation:
  `docs/research/cf-jepa-alert-v1-results.md`
- Published local artifact:
  `artifacts/action-dynamics/prototype-cf-jepa-alert-v1`
- Published manifest SHA-256:
  `59dd147c359501d2ff10d32117c2dfbd5e65f836ec5ce31a3b19c149e7fd2c08`

The frozen path is non-overwriting. Smoke overrides require
`--allow-noninterpretable-smoke` and cannot advance. Selected objective
payloads are written and restored under `objective-checkpoints/` before the
runner starts the next cell; failed staging bundles are preserved.

## SD-JEPA progression/content event localization

Run canonical A2 and its capacity-matched A0 and A2-full controls:

```bash
.venv/bin/python -m lab.action_dynamics.prototype_sd_jepa \
  --output artifacts/action-dynamics/reproductions/sd-jepa-rerun-001
```

Independently verify the manifest and rederive calibration, event
localization, progress probes, alert metrics, state retention, restoration,
and the decision from stored arrays:

```bash
.venv/bin/python -m \
  lab.action_dynamics.prototype_sd_jepa_assessor \
  artifacts/action-dynamics/reproductions/sd-jepa-rerun-001
```

- Seed: `15015`
- Exact schedule: 300 steps per cell, checkpoints every 50, pair-blocked
  batches, cosine learning-rate decay, 100 latency repetitions
- Implementation commit:
  `8ece9c9e91061db17a0399af0e7be0f15ab1e0b3`
- Official SD-JEPA revision:
  `1cc121065e83220a495808f4c65ef4b0b1915f9f`
- Specification:
  `docs/specs/sd-jepa-alert-tracer-v1.md`
- Primary-source notes:
  `docs/research/sd-jepa-primary-source-notes.md`
- Result interpretation:
  `docs/research/sd-jepa-alert-v1-results.md`
- Published local artifact:
  `artifacts/action-dynamics/prototype-sd-jepa-alert-v1`
- Published manifest SHA-256:
  `45bf49091c33553c2f06fcac1c9260762741f833fdd34a5e85376e44e7f6903b`

The frozen path is non-overwriting. Smoke overrides require
`--allow-noninterpretable-smoke`, reduce only the SIGReg sketch count, and
cannot advance. Selected checkpoints, smoke artifacts, failed staging
bundles, source copies, and negative evidence are retained.

## VISReg scale-shape regularization

Recompute the exact two-cell VISReg tracer in a fresh retained directory:

```bash
PYTHONPATH=src .venv/bin/python \
  lab/action_dynamics/prototype_visreg.py \
  --output artifacts/action-dynamics/reproductions/visreg-rerun-001 \
  --allow-noninterpretable-smoke
```

Independently verify the manifest and rederive the mechanism, state, effect,
restoration, latency, and final gate values from stored evidence:

```bash
PYTHONPATH=src .venv/bin/python \
  lab/action_dynamics/prototype_visreg_assessor.py \
  artifacts/action-dynamics/reproductions/visreg-rerun-001
```

- Exact schedule: 1,600 steps per VISReg cell and 100 latency repetitions
- Implementation commit:
  `985d448308dd32b65782a462abd27f7fcbca3859`
- Specification:
  `docs/specs/visreg-telemetry-tracer-v1.md`
- Primary-source notes:
  `docs/research/visreg-primary-source-notes.md`
- Result interpretation:
  `docs/research/visreg-telemetry-v1-results.md`
- Published local artifact:
  `artifacts/action-dynamics/prototype-visreg-v1`
- Published manifest SHA-256:
  `029dd91e4b158b82ca9658ac4e97bf825ba7097a2988d5dbb1a594d573b44b18`

The published path is immutable. A fresh output path is deliberately labeled
non-interpretable even when it uses the full schedule; it can reproduce the
mechanics and evidence but cannot mint a second scientific decision.

## Exact JEPA-SCORE edge alerting

Recompute all 500 rows for all three frozen representation cells, including
the exact full Jacobian and SVD, in a fresh retained directory:

```bash
PYTHONPATH=src .venv/bin/python \
  lab/action_dynamics/prototype_jepa_score.py \
  --output artifacts/action-dynamics/reproductions/jepa-score-rerun-001 \
  --allow-noninterpretable-smoke
```

Independently reconstruct the model route and rederive every score, role,
threshold, alert metric, latency receipt, and gate:

```bash
PYTHONPATH=src .venv/bin/python \
  lab/action_dynamics/prototype_jepa_score_assessor.py \
  --artifact artifacts/action-dynamics/reproductions/jepa-score-rerun-001 \
  --cache artifacts/action-dynamics/edge-preprocessing-v1/eb54271132f88c9a431b01e786ea66279a563776434cca2290e47e6b7ae9b3ff \
  --prior artifacts/action-dynamics/prototype-complete-lejepa-v1
```

- Exact score: Appendix-B full singular-value reduction with `1e-6` clipping
- Fixed anchors: 19, 39, 59, 79, and 97
- Implementation commit:
  `b12b3b6d040729b2b2479b94ad251174cd316c44`
- Specification:
  `docs/specs/jepa-score-edge-screen-v1.md`
- Primary-source notes:
  `docs/research/jepa-score-primary-source-notes.md`
- Result interpretation:
  `docs/research/jepa-score-edge-screen-v1-results.md`
- Published local artifact:
  `artifacts/action-dynamics/prototype-jepa-score-v1`
- Published manifest SHA-256:
  `e678101945c3b99cd325e003f23fdbef334c09ef29ef68f89220cc244012ed86`

The published path is immutable. Four failed smoke builds are intentionally
retained as provenance for contract hardening. Fresh-path reruns cannot
advance even when their numerical evidence matches the published result.

## Artifact availability

The experiment artifacts remain excluded by the repository's broad
`artifacts/*` ignore rule. `tools/artifacts.py` now provides deterministic
packing, GitHub Release publication, verified fetching, and safe extraction
for the 23 conclusion-bearing bundles. The first distribution has not yet been
uploaded; until its index is recorded under `experiments/jepa/releases`, the
artifacts remain local-only.

Shared development corpora, preprocessing caches, smoke runs, and failed
attempts are intentionally outside the first release. They retain their local
provenance status and can be published separately without mixing them into the
authoritative conclusion-bearing set.
