# Artifact policy

Quantis artifacts are evidence, not disposable build output.

## Rules

1. Published artifact directories are immutable.
2. Every rerun writes to a fresh output directory.
3. Failed, invalidated, and superseded runs remain available with an explicit
   non-authoritative label.
4. Conclusion-bearing results record an implementation revision, source and
   data identities, role receipts, and a manifest hash where the experiment
   contract supports them.
5. Experiment capsules reference artifacts by path; they do not duplicate
   bundles.

## Repository boundary

Small thesis, OTLP, fault-lab, fault-matrix, and demand-conditioned evidence is
checked in through explicit `.gitignore` exceptions. Most action-dynamics and
JEPA artifacts are much larger and remain local under the broad
`artifacts/*` ignore rule.

The repository therefore supports:

- exact code and protocol review;
- local reassessment when the bound artifact is present; and
- deterministic reruns when the required generated corpus/cache is present.

It does not yet support clone-only reproduction of every experiment. That
requires a content-addressed bucket, release assets for selected compact
bundles, or a deliberate Git LFS policy. Ordinary Git should not absorb the
current multi-gigabyte caches.

## Future external storage

An external artifact index should key every bundle by its manifest SHA-256 and
record:

- experiment slug and artifact schema;
- implementation commit;
- byte size and media type;
- source-corpus and preprocessing-cache identities;
- object-store URI;
- retention class; and
- authoritative, invalid, failed-smoke, or superseded status.

The checked-in experiment catalog should contain identities and metadata, not
credentials or mutable download URLs.
