"""Collect contextual confirmation cases in isolated Docker lanes."""

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from quantis_core.contextual_confirmation import (
    ConfirmationCollectionCase,
    plan_parallel_confirmation_collection,
)
from quantis_core.fault_matrix import FaultMatrixCaseManifest


def collect_contextual_confirmation(
    *,
    protocol_path: Path,
    manifests_directory: Path,
    captures_directory: Path,
    compose_file: Path,
    project_prefix: str,
    application_image_id: str,
    application_build_context_sha256: str,
    api_request_queue_size: int,
    parallel_jobs: int,
    attestation_path: Path,
) -> Mapping[str, Any]:
    """Run each family as one topology-balanced parallel batch."""

    protocol = json.loads(protocol_path.read_text())
    plans = plan_parallel_confirmation_collection(protocol)
    expected_jobs = int(protocol["collection"]["parallel_jobs"])
    if parallel_jobs != expected_jobs:
        raise ValueError(
            "parallel job count differs from confirmation protocol"
        )
    if api_request_queue_size != int(
        protocol["corpus"][
            "expected_application_api_request_queue_size"
        ]
    ):
        raise ValueError(
            "API queue size differs from confirmation protocol"
        )
    if application_build_context_sha256 != str(
        protocol["corpus"]["application_build_context_sha256"]
    ):
        raise ValueError(
            "application build context differs from protocol"
        )
    manifests = _load_manifests(manifests_directory)
    if set(manifests) != {plan.case_id for plan in plans}:
        raise ValueError(
            "prepared manifests differ from confirmation case plan"
        )
    _validate_plan_manifests(plans, manifests)
    if captures_directory.exists():
        raise FileExistsError(
            "refusing to overwrite confirmation captures: "
            f"{captures_directory}"
        )
    if attestation_path.exists():
        raise FileExistsError(
            "refusing to overwrite collection attestation: "
            f"{attestation_path}"
        )
    captures_directory.mkdir(parents=True)
    attestation_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.time_ns()
    execution_id = str(uuid.uuid4())
    completed_cases = []
    batches = sorted({plan.batch for plan in plans})
    try:
        for batch in batches:
            batch_plans = tuple(
                plan for plan in plans if plan.batch == batch
            )
            if len(batch_plans) != parallel_jobs:
                raise ValueError(
                    f"confirmation batch {batch} is not full-width"
                )
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=parallel_jobs
            ) as executor:
                futures = {
                    executor.submit(
                        _collect_case,
                        plan=plan,
                        manifest_path=manifests[plan.case_id],
                        manifests_directory=manifests_directory,
                        captures_directory=captures_directory,
                        compose_file=compose_file,
                        project_prefix=project_prefix,
                        application_image_id=application_image_id,
                        application_build_context_sha256=(
                            application_build_context_sha256
                        ),
                        api_request_queue_size=(
                            api_request_queue_size
                        ),
                    ): plan
                    for plan in batch_plans
                }
                for future in concurrent.futures.as_completed(
                    futures
                ):
                    completed_cases.append(future.result())
    finally:
        _clean_all_lanes(
            compose_file,
            project_prefix,
            parallel_jobs,
        )
    completed = time.time_ns()
    attestation = {
        "schema_version": 1,
        "kind": (
            "contextual_multimodal_jepa_confirmation_"
            "collection_attestation"
        ),
        "execution_id": execution_id,
        "started_unix_nano": started,
        "completed_unix_nano": completed,
        "parallel_jobs": parallel_jobs,
        "batch_count": len(batches),
        "case_count": len(completed_cases),
        "application_image_id": application_image_id,
        "application_build_context_sha256": (
            application_build_context_sha256
        ),
        "protocol_sha256": _canonical_sha256(protocol),
        "cases": sorted(
            completed_cases,
            key=lambda case: (
                int(case["batch"]),
                int(case["lane"]),
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
    plan: ConfirmationCollectionCase,
    manifest_path: Path,
    manifests_directory: Path,
    captures_directory: Path,
    compose_file: Path,
    project_prefix: str,
    application_image_id: str,
    application_build_context_sha256: str,
    api_request_queue_size: int,
) -> Mapping[str, Any]:
    project = f"{project_prefix}-lane-{plan.lane}"
    capture_directory = captures_directory / plan.case_id
    capture_directory.mkdir()
    environment = dict(os.environ)
    environment.update(
        {
            "EXPERIMENT_DIRECTORY": str(
                manifests_directory.resolve()
            ),
            "CAPTURE_DIRECTORY": str(capture_directory.resolve()),
            "EXPERIMENT_PATH": (
                f"/experiments/{manifest_path.name}"
            ),
            "WORKER_REPLICAS": str(plan.worker_replicas),
            "QUANTIS_API_REQUEST_QUEUE_SIZE": str(
                api_request_queue_size
            ),
            "APPLICATION_IMAGE_ID": application_image_id,
            "APPLICATION_BUILD_CONTEXT_SHA256": (
                application_build_context_sha256
            ),
        }
    )
    compose = _compose_command(compose_file, project)
    started = time.time_ns()
    try:
        _run(
            compose
            + [
                "up",
                "--detach",
                "--scale",
                f"worker={plan.worker_replicas}",
                "redis",
                "postgres",
                "collector",
                "api",
                "worker",
            ],
            environment,
        )
        _run(compose + ["run", "--rm", "runner"], environment)
        _run(compose + ["stop", "collector"], environment)
    finally:
        _run(
            compose
            + ["down", "--volumes", "--remove-orphans"],
            environment,
            check=False,
        )
    completed = time.time_ns()
    return {
        "case_id": plan.case_id,
        "family": plan.family,
        "worker_replicas": plan.worker_replicas,
        "split": plan.split,
        "batch": plan.batch,
        "lane": plan.lane,
        "compose_project": project,
        "started_unix_nano": started,
        "completed_unix_nano": completed,
    }


def _load_manifests(
    manifests_directory: Path,
) -> Mapping[str, Path]:
    manifests: Dict[str, Path] = {}
    for path in sorted(manifests_directory.glob("*.json")):
        manifest = FaultMatrixCaseManifest.from_dict(
            json.loads(path.read_text())
        )
        if manifest.case_id in manifests:
            raise ValueError(
                f"duplicate confirmation manifest: {manifest.case_id}"
            )
        manifests[manifest.case_id] = path
    return manifests


def _validate_plan_manifests(
    plans: Sequence[ConfirmationCollectionCase],
    manifests: Mapping[str, Path],
) -> None:
    for plan in plans:
        manifest = FaultMatrixCaseManifest.from_dict(
            json.loads(manifests[plan.case_id].read_text())
        )
        if (
            manifest.worker_replicas != plan.worker_replicas
            or manifest.topology_id
            != f"workers-{plan.worker_replicas}"
            or manifest.fault_kind != "none"
        ):
            raise ValueError(
                f"manifest differs from collection plan: {plan.case_id}"
            )


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
    compose_file: Path,
    project: str,
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
) -> None:
    subprocess.run(
        list(command),
        env=dict(environment),
        check=check,
    )


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
    parser = argparse.ArgumentParser(
        description=(
            "collect-contextual-confirmation in balanced Docker lanes"
        )
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument(
        "--manifests-directory",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--captures-directory",
        type=Path,
        required=True,
    )
    parser.add_argument("--compose-file", type=Path, required=True)
    parser.add_argument("--project-prefix", required=True)
    parser.add_argument("--application-image-id", required=True)
    parser.add_argument(
        "--application-build-context-sha256",
        required=True,
    )
    parser.add_argument(
        "--api-request-queue-size",
        type=int,
        required=True,
    )
    parser.add_argument("--parallel-jobs", type=int, required=True)
    parser.add_argument("--attestation", type=Path, required=True)
    parsed = parser.parse_args(arguments)
    attestation = collect_contextual_confirmation(
        protocol_path=parsed.protocol,
        manifests_directory=parsed.manifests_directory,
        captures_directory=parsed.captures_directory,
        compose_file=parsed.compose_file,
        project_prefix=parsed.project_prefix,
        application_image_id=parsed.application_image_id,
        application_build_context_sha256=(
            parsed.application_build_context_sha256
        ),
        api_request_queue_size=parsed.api_request_queue_size,
        parallel_jobs=parsed.parallel_jobs,
        attestation_path=parsed.attestation,
    )
    print(
        "Collected "
        f"{attestation['case_count']} confirmation cases in "
        f"{attestation['batch_count']} batches"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
