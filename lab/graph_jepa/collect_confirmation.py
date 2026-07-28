"""Collect graph confirmation cases through the proven Docker lane runner."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Optional, Sequence

from prepare_confirmation import plan_collection


def _load_frozen_collector() -> ModuleType:
    path = (
        Path(__file__).resolve().parent.parent
        / "fault_matrix"
        / "collect_contextual_confirmation.py"
    )
    spec = importlib.util.spec_from_file_location(
        "quantis_graph_jepa_frozen_collector", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen parallel collector")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def collect(
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
    frozen = _load_frozen_collector()
    setattr(
        frozen,
        "plan_parallel_confirmation_collection",
        plan_collection,
    )
    raw = frozen.collect_contextual_confirmation(
        protocol_path=protocol_path,
        manifests_directory=manifests_directory,
        captures_directory=captures_directory,
        compose_file=compose_file,
        project_prefix=project_prefix,
        application_image_id=application_image_id,
        application_build_context_sha256=(
            application_build_context_sha256
        ),
        api_request_queue_size=api_request_queue_size,
        parallel_jobs=parallel_jobs,
        attestation_path=attestation_path,
    )
    attestation = dict(raw)
    attestation["kind"] = (
        "observability_rich_graph_jepa_confirmation_"
        "collection_attestation_v1"
    )
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


def main(arguments: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
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
    parser.add_argument(
        "--api-request-queue-size", type=int, required=True
    )
    parser.add_argument("--parallel-jobs", type=int, required=True)
    parser.add_argument("--attestation", type=Path, required=True)
    parsed = parser.parse_args(arguments)
    collect(
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
