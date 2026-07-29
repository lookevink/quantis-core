#!/usr/bin/env python3
"""Verify and recompute a stored cross-stack corpus audit."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from quantis_core.cross_stack_corpus import assess_serialized_inventory


def assess_stored_bundle(directory: Path) -> Mapping[str, Any]:
    """Verify bundle integrity and recompute the model-free assessment."""

    root = Path(directory)
    _verify_manifest(root)
    _verify_source_evidence(root)
    inventory = _read_object(root / "inventory.json")
    protocol = _read_object(root / "protocol.json")
    return assess_serialized_inventory(inventory, protocol)


def verify_stored_assessment(directory: Path) -> Mapping[str, Any]:
    """Require canonical stored and recomputed assessments to be identical."""

    root = Path(directory)
    recomputed = assess_stored_bundle(root)
    if (root / "assessment.json").read_text() != _pretty_json(recomputed):
        raise ValueError("stored corpus-diversity assessment does not recompute")
    return recomputed


def _verify_manifest(root: Path) -> None:
    manifest = _read_object(root / "artifact-manifest.json")
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported artifact manifest schema")
    expected = {
        str(name): str(digest)
        for name, digest in _object(manifest["files"]).items()
    }
    actual_names = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name != "artifact-manifest.json"
    }
    if actual_names != set(expected):
        raise ValueError("artifact manifest file set mismatch")
    for relative, digest in expected.items():
        if _sha256_file(root / relative) != digest:
            raise ValueError(f"artifact manifest digest mismatch: {relative}")


def _verify_source_evidence(root: Path) -> None:
    identities = _read_object(root / "source-identities.json")
    if identities.get("schema_version") != 1:
        raise ValueError("unsupported source identity schema")
    expected = {
        str(name): str(digest)
        for name, digest in _object(identities["files"]).items()
    }
    for relative, digest in expected.items():
        evidence_path = root / "source-evidence" / relative
        if not evidence_path.is_file():
            raise ValueError(f"missing stored evidence source: {relative}")
        if _sha256_file(evidence_path) != digest:
            raise ValueError(f"stored evidence source mismatch: {relative}")


def _read_object(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _object(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("expected JSON object")
    return value


def _pretty_json(payload: Any) -> str:
    return json.dumps(
        payload, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] = ()) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    arguments = parser.parse_args(None if not argv else argv)
    assessment = verify_stored_assessment(arguments.directory)
    print(_pretty_json(assessment), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
