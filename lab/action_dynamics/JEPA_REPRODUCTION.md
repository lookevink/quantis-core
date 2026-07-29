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

## Artifact availability

The experiment artifacts are currently local and excluded by the repository's
broad `artifacts/*` ignore rule. The compact result documents and identities
describe the evidence, while exact reruns also require the generated
development corpus and preprocessing cache. A shared clone-to-reproduce
workflow therefore needs content-addressed external artifact storage or a
deliberate Git LFS policy; ordinary Git is unsuitable for the current bundles,
which include a 261 MiB multi-hypothesis artifact and multi-gigabyte source
caches.
