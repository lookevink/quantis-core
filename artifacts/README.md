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

The repository includes a content-addressed GitHub Release distribution tool.
The first release has not been uploaded yet, so clone-to-fetch remains pending
until its checked-in release index is populated.

## GitHub Release distribution

The authoritative JEPA artifacts are packaged one technique per deterministic
`.tar.gz` asset. The distribution records two independent identities:

- `tree_sha256` binds normalized paths, modes, sizes, and file contents; and
- `archive_sha256` binds the exact uploaded release object.

The release index also records the source commit used when preparing the
distribution. Publication is tied to the later commit that first records that
exact index.

The first release is planned as `evidence-jepa-frontier-v1` in
`lookevink/quantis-core`. It contains only the 23 conclusion-bearing catalog
artifacts. Smoke runs, failed attempts, shared caches, and superseded evidence
are not silently included.

### Prepare locally

```bash
make artifacts-plan
make artifacts-pack
make artifacts-record
```

`artifacts-plan` scans and hashes the real trees without writing archives.
`artifacts-pack` writes ignored distribution output under `dist/artifacts`.
`artifacts-record` copies only verified release metadata into the repository.

Now review and commit the recorded index. From that exact clean commit, run:

```bash
make artifacts-publish-plan
```

`artifacts-publish-plan` validates every archive, reads the current draft
release state, verifies existing remote digests, and prints the exact missing
upload or recoverable-starter operations without changing GitHub.

### Publish

Publishing is deliberately not exposed as a Make target. After reviewing and
committing the recorded index, use an explicit clean commit:

```bash
python tools/artifacts.py publish \
  --index dist/artifacts/evidence-jepa-frontier-v1/artifact-index-v1.json \
  --asset-directory dist/artifacts/evidence-jepa-frontier-v1 \
  --execute \
  --target <COMMIT_SHA>
```

The command refuses a dirty worktree, requires `<COMMIT_SHA>` to equal `HEAD`,
requires that commit to be the one that recorded the byte-identical index, and
uploads the index, checksums, and archives without clobbering existing assets.
If an upload is interrupted, rerunning safely resumes a matching draft after
verifying every asset GitHub already holds by size and SHA-256.

### Fetch and verify

After the release is published and its index is checked in:

```bash
make artifacts-fetch TECHNIQUE=visreg
make artifacts-verify TECHNIQUE=visreg
```

Fetch verifies the archive SHA-256 before extraction, rejects links and path
traversal, restores into a temporary directory, verifies the unpacked tree,
and refuses to overwrite an existing artifact.

See the checked-in [release-index contract](../experiments/jepa/releases/).

## Later storage boundary

Release assets are the distribution mechanism, not the sole archival copy.
If bundles outgrow GitHub's per-asset ceiling or need retention policies,
mirror the exact content-addressed archives to object storage. The checked-in
index should continue to contain identities and stable metadata, never
credentials or mutable signed URLs.
