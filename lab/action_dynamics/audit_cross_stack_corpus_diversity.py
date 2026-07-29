#!/usr/bin/env python3
"""Create an immutable, model-free cross-stack corpus diversity audit."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from quantis_core.cross_stack_corpus import (
    MinimumDiversityContract,
    assess_corpus_diversity,
    discover_corpus_inventory,
)


REPRODUCTION_PATHS = (
    "src/quantis_core/cross_stack_corpus.py",
    "tests/test_cross_stack_corpus.py",
    "lab/action_dynamics/audit_cross_stack_corpus_diversity.py",
    "lab/action_dynamics/audit_cross_stack_corpus_diversity_assessor.py",
    "lab/action_dynamics/cross-stack-corpus-catalog-v1.json",
    "docs/specs/cross-stack-jepa-corpus-diversity-contract-v1.md",
    "docs/research/cross-stack-jepa-corpus-diversity-primary-sources.md",
)


def run_audit(
    workspace_root: Path,
    catalog_path: Path,
    output_directory: Path,
) -> Mapping[str, Any]:
    """Discover corpus evidence and publish one immutable audit bundle."""

    root = Path(workspace_root).resolve()
    catalog = Path(catalog_path).resolve()
    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite audit bundle: {output}")
    records, source_identities = discover_corpus_inventory(root, catalog)
    contract = MinimumDiversityContract()
    assessment = assess_corpus_diversity(records, contract)
    inventory = {
        "schema_version": 1,
        "kind": "cross_stack_jepa_candidate_corpus_inventory",
        "corpora": [record.to_dict() for record in records],
    }
    identities = {
        "schema_version": 1,
        "kind": "cross_stack_jepa_source_identities",
        "files": source_identities,
    }

    output.mkdir(parents=True)
    _write_json(output / "protocol.json", contract.to_dict())
    _write_json(output / "inventory.json", inventory)
    _write_json(output / "source-identities.json", identities)
    _write_json(output / "assessment.json", assessment)
    (output / "report.md").write_text(_render_report(assessment))
    evidence_root = output / "source-evidence"
    for relative, expected_sha256 in source_identities.items():
        source = (root / relative).resolve()
        try:
            source.relative_to(root)
        except ValueError as error:
            raise ValueError(
                f"evidence source escapes workspace: {relative}"
            ) from error
        if _sha256_file(source) != expected_sha256:
            raise ValueError(f"evidence source changed during audit: {relative}")
        target = evidence_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    reproduction = output / "reproduction"
    for relative in REPRODUCTION_PATHS:
        source = root / relative
        target = reproduction / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    _write_manifest(output)
    return assessment


def _render_report(assessment: Mapping[str, Any]) -> str:
    inventory = _object(assessment["inventory"])
    gaps = _object(assessment["gaps"])
    role_counts = _object(assessment["role_counts"])
    requirements = _object(assessment["role_requirements"])
    exclusions = _object(assessment["exclusions"])
    rows = "\n".join(
        "| {} | {} | {} |".format(
            role, role_counts[role], requirements[role]
        )
        for role in ("fit", "selection", "calibration", "evaluation")
    )
    excluded_lines = "\n".join(
        "- `{}`: {}".format(
            status, ", ".join(f"`{value}`" for value in values)
        )
        for status, values in exclusions.items()
    )
    return """# Cross-stack JEPA corpus-diversity audit v1

## Decision

`{decision}`

The repository contains **{existing} substantive eligible stack family** and
**{qualifying} complete role-qualifying stack environments**. Repeated runs,
worker replica counts, build revisions, and derived caches did not increase
the stack count.

| Role | Complete stacks | Required |
| --- | ---: | ---: |
{rows}

## Minimum acquisition gap

- Additional distinct stacks: **{additional_stacks}**
- Complete the existing stack: **{completion_pairs} matched pairs**
- New stacks: **{new_pairs} matched pairs**
- Total minimum increment: **{additional_pairs} matched pairs /
  {additional_trajectories} trajectories**

The existing action-dynamics campaign covers all five mechanisms and all three
topology levels, but only the steady workload family. Its 90-pair gap is the
two missing workload families crossed with five mechanisms, three topologies,
and three independently reset pairs.

## Exclusions

{excluded_lines}

## Interpretation

The six-stack `3 fit / 1 selection / 1 calibration / 1 evaluation` floor only
authorizes a role-clean exploratory tracer and a conclusion about one named
unseen stack. It is not a portability or production claim. Repeated
claim-bearing evidence requires the separately documented ten-stack
`3 / 2 / 2 / 3` program, pilot-powered pair counts, and later shadow
evaluation.

No model was loaded or fit, no raw telemetry arrays were opened, and no sealed
outcomes were consumed by this audit.
""".format(
        decision=assessment["decision"],
        existing=inventory["distinct_existing_stack_count"],
        qualifying=inventory["qualifying_complete_stack_count"],
        rows=rows,
        additional_stacks=gaps["additional_distinct_stacks"],
        completion_pairs=gaps["existing_stack_completion_pairs"],
        new_pairs=gaps["new_stack_pairs"],
        additional_pairs=gaps["minimum_additional_pairs"],
        additional_trajectories=gaps["minimum_additional_trajectories"],
        excluded_lines=excluded_lines or "- None",
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(_pretty_json(payload))


def _pretty_json(payload: Any) -> str:
    return json.dumps(
        payload, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"


def _write_manifest(root: Path) -> None:
    files = {
        str(path.relative_to(root)): _sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "artifact-manifest.json"
    }
    manifest = {
        "schema_version": 1,
        "kind": "cross_stack_jepa_corpus_diversity_audit_manifest",
        "files": files,
    }
    _write_json(root / "artifact-manifest.json", manifest)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("expected JSON object")
    return value


def main(argv: Sequence[str] = ()) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path(
            "lab/action_dynamics/cross-stack-corpus-catalog-v1.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(None if not argv else argv)
    root = arguments.workspace_root.resolve()
    catalog = arguments.catalog
    if not catalog.is_absolute():
        catalog = root / catalog
    output = arguments.output
    if not output.is_absolute():
        output = root / output
    assessment = run_audit(root, catalog, output)
    print(_pretty_json(assessment), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
