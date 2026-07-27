"""Verify that confirmation inputs equal bytes in a preregistration commit."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Mapping


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return _sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )


def _confirmation_manifest_hashes(
    repository: Path, frozen_files: Mapping[str, Any]
) -> Mapping[str, str]:
    hashes: Dict[str, str] = {}
    for relative_path in frozen_files:
        if not (
            isinstance(relative_path, str)
            and relative_path.startswith("lab/fault_matrix/experiments")
            and relative_path.endswith(".json")
        ):
            continue
        payload = json.loads((repository / relative_path).read_text())
        if not isinstance(payload, dict) or not isinstance(
            payload.get("case_id"), str
        ):
            raise ValueError(
                f"invalid confirmation manifest: {relative_path}"
            )
        case_id = payload["case_id"]
        if case_id in hashes:
            raise ValueError(f"duplicate confirmation case_id: {case_id}")
        hashes[case_id] = _canonical_sha256(payload)
    return hashes


def _git_bytes(repository: Path, commit: str, path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=repository,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    arguments = parser.parse_args()

    repository = arguments.repository.resolve()
    protocol_path = arguments.protocol.resolve()
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    frozen_files = protocol.get("frozen_files")
    if not isinstance(frozen_files, Mapping) or not frozen_files:
        raise ValueError("protocol has no frozen_files mapping")

    protocol_relative = str(protocol_path.relative_to(repository))
    if _git_bytes(
        repository, arguments.commit, protocol_relative
    ) != protocol_bytes:
        raise ValueError(
            "working protocol differs from preregistration commit"
        )
    for relative_path, expected_sha256 in frozen_files.items():
        if not isinstance(relative_path, str) or not isinstance(
            expected_sha256, str
        ):
            raise ValueError("invalid frozen file entry")
        working_bytes = (repository / relative_path).read_bytes()
        if _sha256(working_bytes) != expected_sha256:
            raise ValueError(
                f"working frozen file hash mismatch: {relative_path}"
            )
        if _git_bytes(
            repository, arguments.commit, relative_path
        ) != working_bytes:
            raise ValueError(
                f"frozen file differs from commit: {relative_path}"
            )
    confirmation_manifest_sha256 = protocol.get(
        "confirmation_manifest_sha256"
    )
    if not isinstance(confirmation_manifest_sha256, Mapping):
        raise ValueError(
            "protocol has no confirmation_manifest_sha256 mapping"
        )
    if dict(confirmation_manifest_sha256) != dict(
        _confirmation_manifest_hashes(repository, frozen_files)
    ):
        raise ValueError(
            "protocol confirmation manifest hashes do not match"
        )
    print(
        f"Verified {len(frozen_files)} frozen files at "
        f"{arguments.commit}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
