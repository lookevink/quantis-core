import hashlib
import json
from pathlib import Path

from lab.fault_matrix.verify_v2_protocol import (
    _confirmation_manifest_hashes,
)


def test_confirmation_manifest_hashes_use_canonical_json(
    tmp_path: Path,
) -> None:
    relative_path = (
        "lab/fault_matrix/experiments_v2_expanded/example.json"
    )
    manifest_path = tmp_path / relative_path
    manifest_path.parent.mkdir(parents=True)
    payload = {
        "schema_version": 2,
        "case_id": "example-case",
        "load_pattern_offsets": [2, -1, 0],
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n")

    hashes = _confirmation_manifest_hashes(
        tmp_path, {relative_path: "raw-file-hash-is-not-used"}
    )

    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    assert hashes == {
        "example-case": hashlib.sha256(canonical).hexdigest()
    }
