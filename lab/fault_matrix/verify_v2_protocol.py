"""Verify that confirmation inputs equal bytes in a preregistration commit."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Mapping


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


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
    print(
        f"Verified {len(frozen_files)} frozen files at "
        f"{arguments.commit}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
