# Cross-stack JEPA corpus-diversity audit v1 results

## Conclusion

Do not fit a cross-stack JEPA on the current repository evidence. The audit
decision is:

`collect_cross_stack_corpus_before_jepa`

The existing artifacts contain one substantive stack family, not a
cross-stack corpus. Thirty nominal runs, 240 action-dynamics captures, three
worker replica counts, multiple build identities, and contextual, graph,
hybrid-event, and edge-preprocessing derivatives do not create independent
stacks.

The immutable audit is
`artifacts/action-dynamics/cross-stack-corpus-diversity-audit-v1`.
Its `artifact-manifest.json` SHA-256 is:

`e533c91b0794769a94dcc6265d06f26ebb0bf181609598988dcf8a5d064429dd`

## Inventory result

The catalog produced seven normalized records:

| Classification | Corpora | Cross-stack use |
| --- | --- | --- |
| Open primary | multimodal normal v2; action dynamics development v1 | One checkout-stack family |
| Derived | contextual multimodal development v1; edge preprocessing v1 | Inherit their source campaigns; add no runs or stacks |
| Derived confirmation | hybrid telemetry event cache v1 | Inherits graph-confirmation role; ineligible |
| Result-bearing confirmation | contextual confirmation v2; observability graph confirmation v1 | Ineligible for fitting, selection, calibration, or exploratory reassignment |

The contextual-development corpus has the same individual raw capture
fingerprints as multimodal normal v2. The graph event cache is explicitly
linked to observability-graph confirmation. The edge cache is explicitly
linked to action-dynamics development. These equivalence classes are stored in
the assessment rather than inferred from filenames.

The nominal corpus can supplement fit-only normal behavior, but it has no
matched intervention/control cells. The action-dynamics campaign is the only
open source with the five portable mechanisms and three topology levels. Its
workloads are independently randomized stationary schedules, not three
distinct workload-shape families.

## Frozen evidence levels

Primary-source review found no universal finite environment count that proves
out-of-distribution portability. Quantis therefore uses bounded engineering
levels:

| Level | Stack roles | Valid conclusion |
| --- | --- | --- |
| Scientific floor | 3 source + 1 untouched target | Cross-stack evaluation is possible; one named target only |
| OOD validation | 3 source + 1 OOD selection/calibration + 1 target | Non-source validation, but selection and calibration share a stack |
| Strict tracer | 3 fit + 1 selection + 1 calibration + 1 evaluation | Role-clean result on one named unseen stack |
| Repeated program | 3 fit + 2 selection + 2 calibration + 3 sealed evaluation | Repeated bounded result on three declared unseen stacks |

Ticket 010 freezes the strict six-stack tracer as the next runnable
prerequisite. It does not claim that six stacks are mathematically sufficient
or production-representative.

## Minimum acquisition gap

Every strict-tracer stack must cross:

```text
5 portable mechanisms
× 3 canonical topology levels
× 3 workload-shape families
× 3 separately reset matched pairs
= 135 matched pairs per stack
```

The current action campaign covers the five mechanisms and three topologies
with eight matched pairs per observed cell, but only the steady workload
family. Completing it requires:

```text
5 × 3 × 2 missing workload families × 3 pairs = 90 pairs
```

Five additional complete stacks require `5 × 135 = 675` pairs. The strict
tracer therefore needs at least:

- **5 additional distinct stacks**
- **765 additional matched pairs**
- **1,530 additional trajectories**

Three pairs per cell is a diagnostic floor informed by public benchmark
practice. Before promotion or a repeated claim, a non-sealed pilot must
estimate paired-effect variance and freeze a power calculation; use the
greater of the powered count and three, with five preferred for variance
estimation.

## Integrity checks

The retained implementation:

- extracts individual raw-capture fingerprints and rejects partial overlap
  across roles;
- rejects derived, confirmation, qualification, and synthetic records as
  independent stacks;
- rejects a stack or campaign assigned to multiple roles;
- rejects noncanonical intervention, topology, or workload families, including
  those introduced by incomplete supplemental corpora;
- reports source-campaign equivalence classes and per-factor completion gaps;
- copies the exact authoritative metadata inputs into `source-evidence/` and
  binds them, the implementation, tests, runner, assessor, contract, and
  research note in one manifest;
- reassesses from stored inventory and protocol bytes without loading a model.

The authoritative runner and standalone assessor produced byte-identical
canonical assessments. Five focused public-interface tests passed, including
partial raw-source overlap and stored-input reassessment. No raw telemetry
array or fitted model was loaded, and no confirmation outcome was used.

## Disposition

Ticket 010 is complete with a prerequisite failure, not a model failure. The
next useful target is an acquisition protocol and canonical semantic adapter
for five additional logical application stacks plus the two missing workload
families on the current stack. Another representation-objective tracer on the
existing cache cannot answer the cross-stack question.
