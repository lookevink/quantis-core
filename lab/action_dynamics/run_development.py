"""Collect and qualify the fresh action-dynamics development corpus."""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from quantis_core.action_dynamics_lab import (
    ActionCollectionProtocol,
    write_action_collection_assessment,
    write_prepared_action_collection,
)

from collect_pilot import collect_action_cases
from run_lab_pilot import (
    APPLICATION_IMAGE,
    RUNTIME_IMAGES,
    _canonical_sha256,
    _observation_schema,
    _output,
    _run,
)


def run_action_development(
    *,
    repository: Path,
    smoke_evidence: Path,
    pilot_evidence: Path,
    output: Path,
    parallel_jobs: int = 6,
) -> Mapping[str, Any]:
    """Build once, bind both v4 qualifications, and collect development."""

    root = Path(repository).resolve()
    lab = root / "lab" / "action_dynamics"
    compose_file = lab / "compose.yaml"
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite action development output: {output}"
        )
    for label, evidence in (
        ("smoke", smoke_evidence),
        ("pilot", pilot_evidence),
    ):
        if not (evidence / "data-quality.json").is_file():
            raise ValueError(
                f"qualifying {label} evidence is absent: {evidence}"
            )

    with tempfile.TemporaryDirectory(
        prefix="quantis-action-development-build-"
    ) as temporary:
        scratch = Path(temporary)
        environment = dict(os.environ)
        environment.update(
            {
                "CAPTURE_DIRECTORY": str(scratch / "captures"),
                "EXPERIMENT_DIRECTORY": str(scratch / "manifests"),
                "EXPERIMENT_PATH": "/experiments/placeholder.json",
                "WORKER_REPLICAS": "1",
            }
        )
        (scratch / "captures").mkdir()
        (scratch / "manifests").mkdir()
        _run(
            [
                "docker",
                "compose",
                "--file",
                str(compose_file),
                "build",
                "api",
            ],
            environment,
        )

    application_image_id = _output(
        [
            "docker",
            "image",
            "inspect",
            APPLICATION_IMAGE,
            "--format",
            "{{.Id}}",
        ],
        dict(os.environ),
    ).strip()
    if not application_image_id.startswith("sha256:"):
        raise RuntimeError(
            "action application image is not digest addressed"
        )
    build_context_sha256 = _output(
        [sys.executable, str(lab / "hash_build_context.py")],
        dict(os.environ),
    ).strip()
    _require_same_build(
        build_context_sha256, smoke_evidence, pilot_evidence
    )

    observation_schema = _observation_schema(lab)
    observation_schema_sha256 = _canonical_sha256(
        observation_schema
    )
    image_digests = {
        **RUNTIME_IMAGES,
        "application": application_image_id,
    }
    protocol_payload = json.loads(
        (lab / "development-protocol-v1.json").read_text()
    )
    if not isinstance(protocol_payload, dict):
        raise ValueError(
            "development protocol must be a JSON object"
        )
    protocol = ActionCollectionProtocol.from_dict(protocol_payload)
    if protocol.stage != "development":
        raise ValueError(
            "development runner requires a development protocol"
        )
    if protocol.parallel_jobs != parallel_jobs:
        raise ValueError(
            "requested lane count differs from frozen protocol"
        )

    inputs = output / "inputs"
    write_prepared_action_collection(
        protocol,
        inputs,
        image_digests=image_digests,
        observation_schema_sha256=observation_schema_sha256,
        application_build_context_sha256=build_context_sha256,
        qualifying_smoke_directory=smoke_evidence,
        qualifying_pilot_directory=pilot_evidence,
    )
    (output / "observation-schema.json").write_text(
        json.dumps(
            observation_schema,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    captures = output / "cases"
    attestation = output / "collection-attestation.json"
    collect_action_cases(
        protocol_path=inputs / "protocol.json",
        plan_path=inputs / "plan.json",
        manifests_directory=inputs / "manifests",
        captures_directory=captures,
        compose_file=compose_file,
        project_prefix="quantis-action-development-v1",
        application_image_id=application_image_id,
        application_build_context_sha256=build_context_sha256,
        parallel_jobs=parallel_jobs,
        attestation_path=attestation,
    )
    return write_action_collection_assessment(
        inputs,
        captures,
        attestation,
        output,
    )


def _require_same_build(
    current_build_sha256: str,
    smoke_evidence: Path,
    pilot_evidence: Path,
) -> None:
    for label, evidence in (
        ("smoke", smoke_evidence),
        ("pilot", pilot_evidence),
    ):
        raw = json.loads(
            (evidence / "inputs" / "plan.json").read_text()
        )
        if (
            not isinstance(raw, dict)
            or raw.get("application_build_context_sha256")
            != current_build_sha256
        ):
            raise ValueError(
                f"development runtime differs from qualifying {label}"
            )


def main(arguments: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--smoke-evidence",
        type=Path,
        default=Path(
            "artifacts/action-dynamics/lab-smoke-v4"
        ),
    )
    parser.add_argument(
        "--pilot-evidence",
        type=Path,
        default=Path(
            "artifacts/action-dynamics/instrumentation-pilot-v4"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/action-dynamics/development-v1"
        ),
    )
    parser.add_argument("--parallel-jobs", type=int, default=6)
    parsed = parser.parse_args(arguments)
    assessment = run_action_development(
        repository=parsed.repository,
        smoke_evidence=parsed.smoke_evidence,
        pilot_evidence=parsed.pilot_evidence,
        output=parsed.output,
        parallel_jobs=parsed.parallel_jobs,
    )
    print(json.dumps(assessment, indent=2, sort_keys=True))
    return 0 if assessment.get("status") == "qualified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
