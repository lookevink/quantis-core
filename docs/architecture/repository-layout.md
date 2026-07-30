# Repository layout and experiment-capsule decision

## Decision

Quantis uses a technique-centered experiment directory as the human-facing
interface while retaining conventional locations for supported library code,
tests, shared lab infrastructure, and immutable evidence.

```text
README.md                    repository entry point and current conclusion
experiments/
  README.md                  directory of experiment programs
  jepa/
    catalog.json             metadata source of truth
    README.md                generated JEPA program index
    <technique>/             generated technique capsule
src/quantis_core/            supported library modules
tests/                       public-interface behavioral tests
lab/                         shared and historical execution infrastructure
docs/                        cross-cutting specifications and syntheses
artifacts/                   immutable evidence, usually Git-ignored
tools/                       repository-maintenance tools
```

## Capsule interface

Every comparable JEPA capsule exposes the same small interface:

- `README.md` — status, conclusion, citations, and evidence boundary;
- `run.py` — exact retained runner;
- `assess.py` — independent assessor, when the experiment has one;
- `spec.md` — frozen experiment contract;
- `findings.md` — conclusion-bearing interpretation;
- `references.md` — pinned primary-source notes, when separate notes exist;
- `implementation.py` — supported library implementation, when separated;
- `test.py` — behavioral test; and
- `ticket.md` — execution history and disposition.

These entries are relative links to the retained source files. They provide
locality for readers without duplicating implementation or evidence.

## Why historical files are not moved

Published artifact manifests bind exact runner, assessor, specification,
source-snapshot, and repository paths. Moving those files would create
avoidable ambiguity between historical evidence and the code now occupying
the old path.

The generated capsule therefore acts as a compatibility adapter at the
navigation seam. New experiments should live directly in their capsule;
historical experiments remain reachable through their original paths until a
future artifact format explicitly supports canonical relocation metadata.

## Source of truth

`experiments/jepa/catalog.json` owns technique metadata and declared paths.
`tools/sync_experiment_catalog.py` validates the catalog and generates capsule
READMEs and links.

Run:

```bash
.venv/bin/python tools/sync_experiment_catalog.py
.venv/bin/python tools/sync_experiment_catalog.py --check
```

The check fails on:

- duplicate or malformed technique identifiers;
- missing runners, specs, findings, citations, tests, or tickets;
- invalid artifact locations;
- stale generated READMEs;
- missing, stale, or broken capsule links; and
- uncataloged files inside a generated capsule.

## Evidence ownership

Artifacts are referenced, never copied into capsules. Published artifact
directories are immutable; reruns use fresh output paths. Large ignored
artifacts require a content-addressed external store before clone-to-reproduce
can be claimed.

Cross-experiment syntheses remain under `docs/research`, and the execution
sequence remains under `docs/wayfinding`. Those documents compare techniques;
they do not own technique-specific runner code.

## Status vocabulary

- `active` — preregistered work is currently executing.
- `accepted` — passed the current bounded ladder; not necessarily production.
- `rejected` — failed a frozen scientific gate; code and evidence are retained.
- `blocked` — a recorded prerequisite is absent.
- `superseded` — retained for provenance but not conclusion-bearing.
