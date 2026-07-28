"""Build once, qualify the action lab, then run its instrumentation pilot."""

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Optional, Sequence

from quantis_core.action_dynamics_lab import (
    ActionCollectionProtocol,
    write_action_collection_assessment,
    write_prepared_action_collection,
)

from collect_pilot import collect_action_cases


APPLICATION_IMAGE = "quantis-action-dynamics-app:local"
RUNTIME_IMAGES = {
    "redis": (
        "redis@sha256:"
        "e7723ff73d963f5cc6d9c4643ea3d989527a402a319239054e9472a7fb9219a2"
    ),
    "postgres": (
        "postgres@sha256:"
        "742f40ea20b9ff2ff31db5458d127452988a2164df9e17441e191f3b72252193"
    ),
    "collector": (
        "ghcr.io/open-telemetry/"
        "opentelemetry-collector-releases/"
        "opentelemetry-collector-contrib:0.153.0@sha256:"
        "93aad750175cbf1a973ae1c5886c3371f4d800f61be25cdd26870b8441ffe9fa"
    ),
}


def run_lab_pilot(
    *,
    repository: Path,
    smoke_output: Path,
    pilot_output: Path,
    parallel_jobs: int = 6,
) -> Mapping[str, Any]:
    """Run smoke and pilot with immutable inputs and a hard smoke gate."""

    root = Path(repository).resolve()
    lab = root / "lab" / "action_dynamics"
    compose_file = lab / "compose.yaml"
    for output in (smoke_output, pilot_output):
        if output.exists():
            raise FileExistsError(
                f"refusing to overwrite action lab output: {output}"
            )

    with tempfile.TemporaryDirectory(
        prefix="quantis-action-build-"
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
    observation_schema = _observation_schema(lab)
    observation_schema_sha256 = _canonical_sha256(
        observation_schema
    )
    image_digests = {
        **RUNTIME_IMAGES,
        "application": application_image_id,
    }

    smoke = _run_stage(
        protocol_path=lab / "smoke-protocol.json",
        output=smoke_output,
        compose_file=compose_file,
        project_prefix="quantis-action-smoke-v1",
        application_image_id=application_image_id,
        build_context_sha256=build_context_sha256,
        image_digests=image_digests,
        observation_schema=observation_schema,
        observation_schema_sha256=(
            observation_schema_sha256
        ),
        parallel_jobs=parallel_jobs,
    )
    if smoke["status"] != "qualified":
        return {"smoke": smoke, "pilot": None}
    pilot = _run_stage(
        protocol_path=lab / "pilot-protocol.json",
        output=pilot_output,
        compose_file=compose_file,
        project_prefix="quantis-action-pilot-v1",
        application_image_id=application_image_id,
        build_context_sha256=build_context_sha256,
        image_digests=image_digests,
        observation_schema=observation_schema,
        observation_schema_sha256=(
            observation_schema_sha256
        ),
        parallel_jobs=parallel_jobs,
    )
    return {"smoke": smoke, "pilot": pilot}


def _run_stage(
    *,
    protocol_path: Path,
    output: Path,
    compose_file: Path,
    project_prefix: str,
    application_image_id: str,
    build_context_sha256: str,
    image_digests: Mapping[str, str],
    observation_schema: Mapping[str, Any],
    observation_schema_sha256: str,
    parallel_jobs: int,
) -> Mapping[str, Any]:
    protocol_payload = json.loads(protocol_path.read_text())
    if not isinstance(protocol_payload, dict):
        raise ValueError("action protocol must be a JSON object")
    protocol = ActionCollectionProtocol.from_dict(protocol_payload)
    if protocol.parallel_jobs != parallel_jobs:
        raise ValueError(
            "requested lane count differs from frozen protocol"
        )
    inputs = output / "inputs"
    write_prepared_action_collection(
        protocol,
        inputs,
        image_digests=image_digests,
        observation_schema_sha256=(
            observation_schema_sha256
        ),
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
        project_prefix=project_prefix,
        application_image_id=application_image_id,
        application_build_context_sha256=(
            build_context_sha256
        ),
        parallel_jobs=parallel_jobs,
        attestation_path=attestation,
    )
    return write_action_collection_assessment(
        inputs,
        captures,
        attestation,
        output,
    )


def _observation_schema(lab: Path) -> Mapping[str, Any]:
    module = _load_module(
        "quantis_action_run_capture",
        lab / "run_capture.py",
    )
    feature_names = getattr(module, "FEATURE_NAMES")
    if (
        not isinstance(feature_names, tuple)
        or not feature_names
        or any(not isinstance(name, str) for name in feature_names)
    ):
        raise RuntimeError(
            "action runtime feature vocabulary is invalid"
        )
    return {
        "schema_version": 1,
        "kind": "action_dynamics_observation_schema",
        "feature_names": list(feature_names),
        "trace_span_names": [
            "api.admission",
            "redis.enqueue",
            "queue.residence",
            "redis.dequeue",
            "worker.processing",
            "postgresql.write",
        ],
        "truth_fields_permitted": [],
    }


def _load_module(name: str, path: Path) -> ModuleType:
    runtime = str(path.parent)
    if runtime not in sys.path:
        sys.path.insert(0, runtime)
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load runtime module: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _run(
    command: Sequence[str], environment: Mapping[str, str]
) -> None:
    subprocess.run(
        list(command), env=dict(environment), check=True
    )


def _output(
    command: Sequence[str], environment: Mapping[str, str]
) -> str:
    return subprocess.run(
        list(command),
        env=dict(environment),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout


def main(arguments: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--smoke-output",
        type=Path,
        default=Path(
            "artifacts/action-dynamics/lab-smoke-v1"
        ),
    )
    parser.add_argument(
        "--pilot-output",
        type=Path,
        default=Path(
            "artifacts/action-dynamics/instrumentation-pilot-v1"
        ),
    )
    parser.add_argument("--parallel-jobs", type=int, default=6)
    parsed = parser.parse_args(arguments)
    result = run_lab_pilot(
        repository=parsed.repository,
        smoke_output=parsed.smoke_output,
        pilot_output=parsed.pilot_output,
        parallel_jobs=parsed.parallel_jobs,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    pilot = result["pilot"]
    return (
        0
        if isinstance(pilot, Mapping)
        and pilot.get("status") == "qualified"
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
