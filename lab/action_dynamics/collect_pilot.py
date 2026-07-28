"""Collect action-dynamics pairs in isolated, concurrent Compose lanes."""

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


def collect_action_cases(
    *,
    protocol_path: Path,
    plan_path: Path,
    manifests_directory: Path,
    captures_directory: Path,
    compose_file: Path,
    project_prefix: str,
    application_image_id: str,
    application_build_context_sha256: str,
    parallel_jobs: int,
    attestation_path: Path,
) -> Mapping[str, Any]:
    """Execute a frozen pair-atomic collection plan."""

    protocol = _read_object(protocol_path)
    plan = _read_object(plan_path)
    assignments = plan.get("assignments")
    if not isinstance(assignments, list) or not assignments:
        raise ValueError("collection plan assignments are invalid")
    expected_jobs = protocol["collection"]["parallel_jobs"]
    if (
        isinstance(expected_jobs, bool)
        or not isinstance(expected_jobs, int)
        or parallel_jobs != expected_jobs
    ):
        raise ValueError(
            "parallel job count differs from frozen protocol"
        )
    if captures_directory.exists():
        raise FileExistsError(
            f"refusing to overwrite captures: {captures_directory}"
        )
    if attestation_path.exists():
        raise FileExistsError(
            f"refusing to overwrite attestation: {attestation_path}"
        )
    manifests = {
        path.stem: path
        for path in manifests_directory.glob("*.json")
    }
    case_ids = {
        _required_text(assignment, "case_id")
        for assignment in assignments
    }
    if case_ids != set(manifests):
        raise ValueError(
            "collection plan and prepared manifests differ"
        )
    captures_directory.mkdir(parents=True)
    attestation_path.parent.mkdir(parents=True, exist_ok=True)

    pairs: Dict[str, list[Mapping[str, Any]]] = {}
    for raw_assignment in assignments:
        if not isinstance(raw_assignment, dict):
            raise ValueError("collection assignment is invalid")
        assignment = dict(raw_assignment)
        pairs.setdefault(
            _required_text(assignment, "pair_id"), []
        ).append(assignment)
    for pair_id, pair_assignments in pairs.items():
        if (
            len(pair_assignments) != 2
            or {item["role"] for item in pair_assignments}
            != {"treatment", "control"}
            or len({item["lane"] for item in pair_assignments}) != 1
            or len({item["batch"] for item in pair_assignments}) != 1
            or {item["order_in_pair"] for item in pair_assignments}
            != {0, 1}
        ):
            raise ValueError(
                f"pair-atomic assignment is invalid: {pair_id}"
            )

    started = time.time_ns()
    execution_id = str(uuid.uuid4())
    completed_cases: list[Mapping[str, Any]] = []
    batches = sorted(
        {
            _required_integer(assignment, "batch")
            for assignment in assignments
        }
    )
    try:
        for batch in batches:
            batch_pairs = [
                sorted(
                    pair_assignments,
                    key=lambda item: int(item["order_in_pair"]),
                )
                for pair_assignments in pairs.values()
                if _required_integer(
                    pair_assignments[0], "batch"
                )
                == batch
            ]
            if not batch_pairs or len(batch_pairs) > parallel_jobs:
                raise ValueError(
                    f"collection batch {batch} has invalid width"
                )
            for order_in_pair in (0, 1):
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=parallel_jobs
                ) as executor:
                    futures = [
                        executor.submit(
                            _collect_case,
                            assignment=pair_assignments[
                                order_in_pair
                            ],
                            manifest_path=manifests[
                                _required_text(
                                    pair_assignments[
                                        order_in_pair
                                    ],
                                    "case_id",
                                )
                            ],
                            manifests_directory=(
                                manifests_directory
                            ),
                            captures_directory=captures_directory,
                            compose_file=compose_file,
                            project_prefix=project_prefix,
                            application_image_id=(
                                application_image_id
                            ),
                            application_build_context_sha256=(
                                application_build_context_sha256
                            ),
                        )
                        for pair_assignments in batch_pairs
                    ]
                    for future in concurrent.futures.as_completed(
                        futures
                    ):
                        completed_cases.append(future.result())
    finally:
        _clean_all_lanes(
            compose_file, project_prefix, parallel_jobs
        )

    attestation = {
        "schema_version": 1,
        "kind": "action_dynamics_collection_attestation",
        "execution_id": execution_id,
        "started_unix_nano": started,
        "completed_unix_nano": time.time_ns(),
        "parallel_jobs": parallel_jobs,
        "batch_count": len(batches),
        "case_count": len(completed_cases),
        "pair_count": len(pairs),
        "application_image_id": application_image_id,
        "application_build_context_sha256": (
            application_build_context_sha256
        ),
        "protocol_sha256": _canonical_sha256(protocol),
        "plan_sha256": _canonical_sha256(plan),
        "qualifying_smoke_sha256": plan.get(
            "qualifying_smoke_sha256"
        ),
        "cases": sorted(
            completed_cases,
            key=lambda case: (
                int(case["batch"]),
                int(case["lane"]),
                int(case["order_in_pair"]),
            ),
        ),
    }
    attestation_path.write_text(
        json.dumps(
            attestation,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    return attestation


def _collect_case(
    *,
    assignment: Mapping[str, Any],
    manifest_path: Path,
    manifests_directory: Path,
    captures_directory: Path,
    compose_file: Path,
    project_prefix: str,
    application_image_id: str,
    application_build_context_sha256: str,
) -> Mapping[str, Any]:
    case_id = _required_text(assignment, "case_id")
    lane = _required_integer(assignment, "lane")
    worker_replicas = _required_integer(
        assignment, "worker_replicas"
    )
    project = f"{project_prefix}-lane-{lane}"
    capture_directory = captures_directory / case_id
    capture_directory.mkdir()
    shutil.copyfile(
        manifest_path,
        capture_directory / "capture-manifest.json",
    )
    environment = dict(os.environ)
    environment.update(
        {
            "EXPERIMENT_DIRECTORY": str(
                manifests_directory.resolve()
            ),
            "CAPTURE_DIRECTORY": str(
                capture_directory.resolve()
            ),
            "EXPERIMENT_PATH": (
                f"/experiments/{manifest_path.name}"
            ),
            "WORKER_REPLICAS": str(worker_replicas),
            "APPLICATION_IMAGE_ID": application_image_id,
            "APPLICATION_BUILD_CONTEXT_SHA256": (
                application_build_context_sha256
            ),
        }
    )
    compose = _compose_command(compose_file, project)
    started = time.time_ns()
    runner_output = ""
    try:
        _run(
            compose
            + [
                "up",
                "--detach",
                "--scale",
                f"worker={worker_replicas}",
                "redis",
                "postgres",
                "collector",
                "api",
                "worker",
            ],
            environment,
        )
        result = _run(
            compose + ["run", "--rm", "runner"],
            environment,
            capture=True,
        )
        runner_output = result.stdout
        _run(compose + ["stop", "collector"], environment)
    finally:
        (capture_directory / "runner.log").write_text(
            runner_output
        )
        _run(
            compose
            + ["down", "--volumes", "--remove-orphans"],
            environment,
            check=False,
        )
    return {
        **dict(assignment),
        "compose_project": project,
        "manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "started_unix_nano": started,
        "completed_unix_nano": time.time_ns(),
    }


def _clean_all_lanes(
    compose_file: Path,
    project_prefix: str,
    parallel_jobs: int,
) -> None:
    for lane in range(1, parallel_jobs + 1):
        _run(
            _compose_command(
                compose_file,
                f"{project_prefix}-lane-{lane}",
            )
            + ["down", "--volumes", "--remove-orphans"],
            dict(os.environ),
            check=False,
        )


def _compose_command(
    compose_file: Path, project: str
) -> list[str]:
    return [
        "docker",
        "compose",
        "--project-name",
        project,
        "--file",
        str(compose_file.resolve()),
    ]


def _run(
    command: Sequence[str],
    environment: Mapping[str, str],
    *,
    check: bool = True,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        env=dict(environment),
        check=check,
        text=True,
        stdout=(
            subprocess.PIPE if capture else subprocess.DEVNULL
        ),
        stderr=(
            subprocess.STDOUT if capture else subprocess.DEVNULL
        ),
    )


def _read_object(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return dict(value)


def _required_text(
    value: Mapping[str, Any], key: str
) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise ValueError(f"{key} must be nonempty text")
    return result


def _required_integer(
    value: Mapping[str, Any], key: str
) -> int:
    result = value.get(key)
    if (
        isinstance(result, bool)
        or not isinstance(result, int)
        or result < 0
    ):
        raise ValueError(f"{key} must be a nonnegative integer")
    return result


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def main(arguments: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--manifests-directory", type=Path, required=True
    )
    parser.add_argument(
        "--captures-directory", type=Path, required=True
    )
    parser.add_argument("--compose-file", type=Path, required=True)
    parser.add_argument("--project-prefix", required=True)
    parser.add_argument("--application-image-id", required=True)
    parser.add_argument(
        "--application-build-context-sha256", required=True
    )
    parser.add_argument("--parallel-jobs", type=int, required=True)
    parser.add_argument(
        "--attestation", type=Path, required=True
    )
    parsed = parser.parse_args(arguments)
    collect_action_cases(
        protocol_path=parsed.protocol,
        plan_path=parsed.plan,
        manifests_directory=parsed.manifests_directory,
        captures_directory=parsed.captures_directory,
        compose_file=parsed.compose_file,
        project_prefix=parsed.project_prefix,
        application_image_id=parsed.application_image_id,
        application_build_context_sha256=(
            parsed.application_build_context_sha256
        ),
        parallel_jobs=parsed.parallel_jobs,
        attestation_path=parsed.attestation,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
