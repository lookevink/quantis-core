# JEPA artifact release indexes

This directory stores small, reviewable distribution metadata. Binary
artifacts remain outside Git and are attached to immutable-by-convention
GitHub Releases.

The planned first index is:

```text
evidence-jepa-frontier-v1.json
```

It is created only after all 23 conclusion-bearing artifact directories have
been deterministically packaged and independently verified:

```bash
make artifacts-plan
make artifacts-pack
make artifacts-record
```

Each asset entry binds:

- the experiment slug and original `artifacts/...` destination;
- unpacked byte and file counts;
- total archive-entry count, including empty directories;
- a canonical tree SHA-256;
- archive byte size, media type, and SHA-256; and
- the source commit, GitHub repository, and release tag.

The index pins its authoritative slug set and catalog identity when prepared.
Later catalog additions therefore do not invalidate an immutable older
release, while every pinned slug must still resolve to the same catalog
artifact path.

An index is committed before its matching draft release is uploaded. Published
assets are never replaced in place; changed evidence requires a new release
tag and new index.
