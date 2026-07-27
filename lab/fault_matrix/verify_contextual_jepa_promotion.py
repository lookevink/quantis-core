"""Verify frozen contextual-JEPA promotion inputs and prepared manifests."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


def verify_preregistration(
    repository: Path,
    protocol_path: Path,
    commit: str,
) -> Mapping[str, Any]:
    """Verify protocol bytes and every frozen implementation file."""

    repository = repository.resolve()
    resolved_commit = subprocess.run(
        ["git", "rev-parse", "--verify", f"{commit}^{{commit}}"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != resolved_commit:
        raise ValueError(
            "preregistered commit must be a full immutable Git ID"
        )
    protocol_path = protocol_path.resolve()
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    if (
        protocol.get("schema_version") != 1
        or protocol.get("kind")
        != "contextual_multimodal_jepa_promotion_v1"
    ):
        raise ValueError("unsupported contextual promotion protocol")
    protocol_relative = str(protocol_path.relative_to(repository))
    if _git_bytes(repository, commit, protocol_relative) != protocol_bytes:
        raise ValueError(
            "working protocol differs from preregistration commit"
        )
    frozen_files = protocol.get("frozen_files")
    if not isinstance(frozen_files, Mapping) or not frozen_files:
        raise ValueError("promotion protocol has no frozen files")
    for relative_path, expected_sha256 in frozen_files.items():
        if not isinstance(relative_path, str) or not isinstance(
            expected_sha256,
            str,
        ):
            raise ValueError("invalid frozen file entry")
        working = (repository / relative_path).read_bytes()
        if _sha256(working) != expected_sha256:
            raise ValueError(
                f"working frozen file hash mismatch: {relative_path}"
            )
        if _git_bytes(repository, commit, relative_path) != working:
            raise ValueError(
                f"frozen file differs from commit: {relative_path}"
            )
    return protocol


def verify_prepared_inputs(
    inputs_directory: Path,
    protocol: Mapping[str, Any],
) -> None:
    """Verify deterministic split membership and all manifest schedules."""

    split = json.loads(
        (inputs_directory / "split.json").read_text()
    )
    for split_name in ("training", "validation"):
        observed = list(split[f"{split_name}_case_ids"])
        expected = list(protocol[f"{split_name}_case_ids"])
        if observed != expected:
            raise ValueError(
                f"prepared {split_name} cases differ from protocol"
            )
    corpus = dict(protocol["corpus"])
    if int(split["lookback"]) != int(corpus["lookback"]):
        raise ValueError("prepared lookback differs from protocol")
    if split.get("expected_application_api_request_queue_size") != (
        corpus["expected_application_api_request_queue_size"]
    ):
        raise ValueError(
            "prepared API request queue size differs from protocol"
        )
    schedules = list(corpus["schedule_families"])
    manifests = sorted(
        (inputs_directory / "manifests").glob("*.json")
    )
    expected_case_ids = set(protocol["training_case_ids"]) | set(
        protocol["validation_case_ids"]
    )
    if len(manifests) != len(expected_case_ids):
        raise ValueError("prepared manifest count differs from protocol")
    for path in manifests:
        manifest = json.loads(path.read_text())
        case_id = str(manifest["case_id"])
        if case_id not in expected_case_ids:
            raise ValueError(f"unexpected prepared case: {case_id}")
        family_index = int(case_id.split("-f", 1)[1][:2]) - 1
        expected_schedule = dict(schedules[family_index])
        if (
            manifest["requests_per_window"]
            != expected_schedule["requests_per_window"]
            or manifest["load_pattern_offsets"]
            != expected_schedule["load_pattern_offsets"]
        ):
            raise ValueError(
                f"prepared schedule differs for {case_id}"
            )
        if (
            manifest["point_count"] != corpus["point_count"]
            or manifest["sample_period_seconds"]
            != corpus["sample_period_seconds"]
            or manifest["fault_kind"] != "none"
        ):
            raise ValueError(
                f"prepared manifest design differs for {case_id}"
            )


def main(arguments: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--inputs", type=Path)
    parsed = parser.parse_args(arguments)
    protocol = verify_preregistration(
        parsed.repository,
        parsed.protocol,
        parsed.commit,
    )
    if parsed.inputs is not None:
        verify_prepared_inputs(parsed.inputs, protocol)
    print(
        f"Verified contextual JEPA promotion protocol at "
        f"{parsed.commit}"
    )
    return 0


def _git_bytes(repository: Path, commit: str, path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=repository,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
