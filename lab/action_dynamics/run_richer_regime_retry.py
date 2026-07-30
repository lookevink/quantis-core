"""Prepare and collect one frozen richer-regime retry shard locally."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from quantis_core.richer_regime_retry import (
    CORPUS_ROLES,
    WORKLOAD_FAMILIES,
    RicherRegimeRetryProtocol,
    assess_richer_regime_plan,
    build_richer_regime_plan,
    prepare_richer_regime_shard,
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


def run_richer_regime_shard(
    *,
    repository: Path,
    output: Path,
    corpus_role: str,
    workload_family: str,
    parallel_jobs: int = 6,
    prepare_only: bool = False,
    reuse_built_image: bool = False,
) -> Mapping[str, Any]:
    """Build once and execute one immutable role-by-family shard."""

    root = Path(repository).resolve()
    lab = root / "lab" / "action_dynamics"
    protocol_path = lab / "richer-regime-retry-protocol-v1.json"
    protocol_payload = _read_object(protocol_path)
    protocol = RicherRegimeRetryProtocol.from_dict(protocol_payload)
    if parallel_jobs != int(protocol.execution["parallel_jobs"]):
        raise ValueError(
            "requested lane count differs from frozen retry protocol"
        )
    full_plan = build_richer_regime_plan(protocol)
    plan_assessment = assess_richer_regime_plan(protocol, full_plan)
    if plan_assessment["status"] != "qualified":
        raise ValueError("richer-regime campaign plan did not qualify")
    campaign = output / "campaign"
    _write_campaign_once(
        campaign,
        protocol_payload,
        full_plan,
        plan_assessment,
    )
    shard = output / corpus_role / workload_family
    if shard.exists():
        raise FileExistsError(
            f"refusing to overwrite richer-regime shard: {shard}"
        )

    application_image_id, build_context_sha256 = _stack_identity(
        lab,
        prepare_only=prepare_only,
        reuse_built_image=reuse_built_image,
    )
    observation_schema = _observation_schema(lab)
    observation_schema_sha256 = _canonical_sha256(
        observation_schema
    )
    development_payload = _read_object(
        lab / "development-protocol-v1.json"
    )
    action_library = development_payload.get("action_library")
    if not isinstance(action_library, dict):
        raise ValueError("development action library is absent")
    image_digests = {
        **RUNTIME_IMAGES,
        "application": application_image_id,
    }
    prepared = prepare_richer_regime_shard(
        protocol,
        full_plan,
        corpus_role=corpus_role,
        workload_family=workload_family,
        action_library=action_library,
        image_digests=image_digests,
        observation_schema_sha256=observation_schema_sha256,
        application_build_context_sha256=build_context_sha256,
    )
    inputs = shard / "inputs"
    manifests = inputs / "manifests"
    manifests.mkdir(parents=True)
    _write_json(inputs / "protocol.json", prepared["protocol"])
    _write_json(inputs / "plan.json", prepared["plan"])
    _write_json(inputs / "prepared-summary.json", prepared["summary"])
    raw_manifests = prepared["manifests"]
    if not isinstance(raw_manifests, dict):
        raise AssertionError("prepared manifests changed type")
    for case_id, manifest in raw_manifests.items():
        _write_json(manifests / f"{case_id}.json", manifest)
    _write_json(shard / "observation-schema.json", observation_schema)
    if prepare_only:
        result = {
            **dict(prepared["summary"]),
            "status": "prepared",
            "decision": "execute_local_compose_shard",
        }
        _write_json(shard / "shard-assessment.json", result)
        return result

    captures = shard / "cases"
    attestation_path = shard / "collection-attestation.json"
    collect_action_cases(
        protocol_path=inputs / "protocol.json",
        plan_path=inputs / "plan.json",
        manifests_directory=manifests,
        captures_directory=captures,
        compose_file=lab / "compose.yaml",
        project_prefix=(
            "quantis-richer-regime-v1-"
            f"{corpus_role[:3]}-{workload_family[:3]}"
        ),
        application_image_id=application_image_id,
        application_build_context_sha256=build_context_sha256,
        parallel_jobs=parallel_jobs,
        attestation_path=attestation_path,
    )
    assessment = _assess_collected_shard(
        prepared, captures, attestation_path
    )
    _write_json(shard / "shard-assessment.json", assessment)
    return assessment


def _stack_identity(
    lab: Path, *, prepare_only: bool, reuse_built_image: bool
) -> tuple[str, str]:
    if not prepare_only and not reuse_built_image:
        with tempfile.TemporaryDirectory(
            prefix="quantis-richer-regime-build-"
        ) as temporary:
            scratch = Path(temporary)
            environment = dict(os.environ)
            environment.update(
                {
                    "CAPTURE_DIRECTORY": str(
                        scratch / "captures"
                    ),
                    "EXPERIMENT_DIRECTORY": str(
                        scratch / "manifests"
                    ),
                    "EXPERIMENT_PATH": (
                        "/experiments/placeholder.json"
                    ),
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
                    str(lab / "compose.yaml"),
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
    return application_image_id, build_context_sha256


def _write_campaign_once(
    directory: Path,
    protocol: Mapping[str, Any],
    plan: Mapping[str, Any],
    assessment: Mapping[str, Any],
) -> None:
    expected = {
        "protocol.json": protocol,
        "plan.json": plan,
        "plan-assessment.json": assessment,
    }
    if directory.exists():
        for name, payload in expected.items():
            path = directory / name
            if not path.is_file() or _read_object(path) != payload:
                raise ValueError(
                    "existing richer-regime campaign identity drifted"
                )
        return
    directory.mkdir(parents=True)
    for name, payload in expected.items():
        _write_json(directory / name, payload)


def _assess_collected_shard(
    prepared: Mapping[str, Any],
    captures: Path,
    attestation_path: Path,
) -> Mapping[str, Any]:
    attestation = _read_object(attestation_path)
    manifests = prepared["manifests"]
    if not isinstance(manifests, dict):
        raise AssertionError("prepared manifests changed type")
    expected_files = {
        "capture-manifest.json",
        "collector-actions.jsonl",
        "collector-logs.jsonl",
        "collector-metrics.jsonl",
        "collector-traces.jsonl",
        "runner.log",
    }
    missing = {}
    empty = {}
    for case_id in manifests:
        case_directory = captures / case_id
        missing_names = sorted(
            name
            for name in expected_files
            if not (case_directory / name).is_file()
        )
        empty_names = sorted(
            name
            for name in expected_files - {"runner.log"}
            if (case_directory / name).is_file()
            and (case_directory / name).stat().st_size == 0
        )
        if missing_names:
            missing[case_id] = missing_names
        if empty_names:
            empty[case_id] = empty_names
    expected_count = len(manifests)
    gates = {
        "attested_capture_count": attestation.get("case_count")
        == expected_count,
        "attested_pair_count": attestation.get("pair_count")
        == expected_count // 2,
        "all_capture_files_present": not missing,
        "telemetry_files_nonempty": not empty,
        "protocol_binding": attestation.get("protocol_sha256")
        == prepared["summary"]["protocol_sha256"],
        "plan_binding": attestation.get("plan_sha256")
        == prepared["summary"]["plan_sha256"],
    }
    return {
        "schema_version": 1,
        "kind": "richer_regime_retry_shard_assessment",
        "status": "qualified" if all(gates.values()) else "failed",
        "decision": (
            "retain_shard_for_declared_corpus_role"
            if all(gates.values())
            else "stop_without_retrying_failed_pairs"
        ),
        "corpus_role": prepared["summary"]["corpus_role"],
        "workload_family": prepared["summary"]["workload_family"],
        "pair_count": expected_count // 2,
        "capture_count": expected_count,
        "gates": gates,
        "missing_files": missing,
        "empty_files": empty,
    }


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def main(arguments: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/action-dynamics/richer-regime-retry-v1"
        ),
    )
    parser.add_argument(
        "--corpus-role", choices=CORPUS_ROLES, required=True
    )
    parser.add_argument(
        "--workload-family",
        choices=WORKLOAD_FAMILIES,
        required=True,
    )
    parser.add_argument("--parallel-jobs", type=int, default=6)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--reuse-built-image",
        action="store_true",
        help=(
            "Reuse the existing digest-addressed local application "
            "image; the build-context hash is still recomputed."
        ),
    )
    parsed = parser.parse_args(arguments)
    assessment = run_richer_regime_shard(
        repository=parsed.repository,
        output=parsed.output,
        corpus_role=parsed.corpus_role,
        workload_family=parsed.workload_family,
        parallel_jobs=parsed.parallel_jobs,
        prepare_only=parsed.prepare_only,
        reuse_built_image=parsed.reuse_built_image,
    )
    print(json.dumps(assessment, indent=2, sort_keys=True))
    return 0 if assessment.get("status") in {"prepared", "qualified"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
