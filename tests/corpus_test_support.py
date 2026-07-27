import hashlib
import json
from dataclasses import replace
from pathlib import Path

from quantis_core.fault_matrix import (
    FaultMatrixCaseManifest,
    FaultMatrixRun,
)
from quantis_core.otlp import MetricPoint, TelemetryCapture, read_otlp_capture
from quantis_core.otlp_windowing import OtlpFeatureSpec


FRESH_CASE_IDS = (
    "fresh-normal-cache-01",
    "fresh-normal-database-01",
    "fresh-normal-worker-01",
)


def fresh_development_runs() -> tuple[
    list[FaultMatrixRun],
    OtlpFeatureSpec,
]:
    source_runs, feature_spec = _source_runs()
    return [
        _fresh_run(run, fresh_case_id)
        for run, fresh_case_id in zip(source_runs, FRESH_CASE_IDS)
    ], feature_spec


def write_fresh_development_runs(
    root: Path,
) -> tuple[Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    repository = Path(__file__).resolve().parents[1]
    source_manifests = (
        repository / "lab" / "fault_matrix" / "experiments"
    )
    source_captures = (
        repository / "artifacts" / "fault-matrix" / "cases"
    )
    manifests_directory = root / "manifests"
    captures_directory = root / "cases"
    manifests_directory.mkdir()
    captures_directory.mkdir()
    for source_path, fresh_case_id in zip(
        sorted(source_manifests.glob("*.json")),
        FRESH_CASE_IDS,
    ):
        payload = json.loads(source_path.read_text())
        source_case_id = str(payload["case_id"])
        payload["case_id"] = fresh_case_id
        manifest_sha256 = _canonical_sha256(payload)
        (manifests_directory / f"{fresh_case_id}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        case_directory = captures_directory / fresh_case_id
        case_directory.mkdir()
        output_lines = []
        for line in (
            source_captures
            / source_case_id
            / "collector-output.jsonl"
        ).read_text().splitlines():
            message = json.loads(line)
            for resource_metrics in message["resourceMetrics"]:
                attributes = resource_metrics["resource"][
                    "attributes"
                ]
                for attribute in attributes:
                    key = attribute["key"]
                    if key == "quantis.experiment.case.id":
                        attribute["value"]["stringValue"] = (
                            fresh_case_id
                        )
                    elif (
                        key
                        == "quantis.experiment.manifest.sha256"
                    ):
                        attribute["value"]["stringValue"] = (
                            manifest_sha256
                        )
            output_lines.append(
                json.dumps(
                    message,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        (
            case_directory / "collector-output.jsonl"
        ).write_text("\n".join(output_lines) + "\n")
    return (
        captures_directory,
        manifests_directory,
        repository / "lab" / "fault_matrix" / "feature-spec.json",
    )


def _source_runs() -> tuple[
    list[FaultMatrixRun],
    OtlpFeatureSpec,
]:
    repository = Path(__file__).resolve().parents[1]
    lab = repository / "lab" / "fault_matrix"
    captures = repository / "artifacts" / "fault-matrix" / "cases"
    runs = []
    for manifest_path in sorted((lab / "experiments").glob("*.json")):
        manifest = FaultMatrixCaseManifest.from_dict(
            json.loads(manifest_path.read_text())
        )
        runs.append(
            FaultMatrixRun(
                manifest,
                read_otlp_capture(
                    captures
                    / manifest.case_id
                    / "collector-output.jsonl"
                ),
            )
        )
    feature_spec = OtlpFeatureSpec.from_dict(
        json.loads((lab / "feature-spec.json").read_text())
    )
    return runs, feature_spec


def _fresh_run(
    run: FaultMatrixRun,
    case_id: str,
) -> FaultMatrixRun:
    manifest = replace(run.manifest, case_id=case_id)
    manifest_sha256 = _canonical_sha256(manifest.to_dict())
    points = tuple(
        replace(
            point,
            resource_attributes=_fresh_attributes(
                point,
                case_id,
                manifest_sha256,
            ),
        )
        for point in run.capture.points
    )
    return FaultMatrixRun(
        manifest=manifest,
        capture=TelemetryCapture(
            points=points,
            sha256=hashlib.sha256(
                f"{run.capture.sha256}:{case_id}".encode("utf-8")
            ).hexdigest(),
            source_path=f"fresh://{case_id}",
            json_message_count=run.capture.json_message_count,
        ),
    )


def _fresh_attributes(
    point: MetricPoint,
    case_id: str,
    manifest_sha256: str,
):
    attributes = dict(point.resource_attributes)
    attributes["quantis.experiment.case.id"] = case_id
    attributes["quantis.experiment.manifest.sha256"] = (
        manifest_sha256
    )
    return attributes


def _canonical_sha256(payload) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
