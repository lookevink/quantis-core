"""Provenance-safe compilation of normal OTLP telemetry for learned models."""

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .demand_conditioning import (
    DemandConditioner,
    canonical_request_schedule,
)
from .fault_matrix import FaultMatrixRun
from .otlp_windowing import OtlpFeatureSpec, OtlpWindowCompiler
from .windowing import ModelWindows, WindowCompiler


RESERVED_EVIDENCE_CASE_IDS = frozenset(
    {
        "cache-outage-confirmation-04",
        "cache-outage-held-out-01",
        "database-lock-confirmation-04",
        "database-lock-held-out-01",
        "expanded-workers-1-cache-outage-10",
        "expanded-workers-1-database-lock-10",
        "expanded-workers-1-worker-crash-10",
        "expanded-workers-2-cache-outage-10",
        "expanded-workers-2-database-lock-10",
        "expanded-workers-2-worker-crash-10",
        "expanded-workers-3-cache-outage-10",
        "expanded-workers-3-database-lock-10",
        "expanded-workers-3-worker-crash-10",
        "matched-workers-1-cache-outage-11",
        "matched-workers-1-database-lock-11",
        "matched-workers-1-worker-crash-11",
        "matched-workers-2-cache-outage-11",
        "matched-workers-2-database-lock-11",
        "matched-workers-2-worker-crash-11",
        "matched-workers-3-cache-outage-11",
        "matched-workers-3-database-lock-11",
        "matched-workers-3-worker-crash-11",
        "worker-crash-confirmation-04",
        "worker-crash-held-out-01",
    }
)
FAILED_CORPUS_CASE_IDS = frozenset(
    f"multimodal-normal-f{family_index:02d}"
    f"-w{worker_replicas}-47"
    for family_index in range(1, 11)
    for worker_replicas in (1, 2, 3)
)


@dataclass(frozen=True)
class TelemetryCorpusSplitSpec:
    """Run-level corpus split and explicit evidence exclusions."""

    training_case_ids: Tuple[str, ...]
    validation_case_ids: Tuple[str, ...]
    reserved_case_ids: Tuple[str, ...]
    lookback: int = 6

    def __post_init__(self) -> None:
        if not self.training_case_ids:
            raise ValueError("telemetry corpus requires training cases")
        if not self.validation_case_ids:
            raise ValueError("telemetry corpus requires validation cases")
        if self.lookback < 1:
            raise ValueError("telemetry corpus lookback must be positive")
        for name, case_ids in (
            ("training", self.training_case_ids),
            ("validation", self.validation_case_ids),
            ("reserved", self.reserved_case_ids),
        ):
            if len(set(case_ids)) != len(case_ids):
                raise ValueError(f"{name} corpus case_ids must be unique")
            if any(not case_id for case_id in case_ids):
                raise ValueError(f"{name} corpus case_ids cannot be empty")
        training = set(self.training_case_ids)
        validation = set(self.validation_case_ids)
        reserved = (
            set(self.reserved_case_ids)
            | RESERVED_EVIDENCE_CASE_IDS
        )
        if training & validation:
            raise ValueError(
                "training and validation case_ids must be disjoint"
            )
        if (training | validation) & reserved:
            raise ValueError(
                "reserved evidence cannot enter a telemetry corpus split"
            )
        if (training | validation) & FAILED_CORPUS_CASE_IDS:
            raise ValueError(
                "failed corpus cases cannot enter a telemetry corpus split"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "training_case_ids": list(self.training_case_ids),
            "validation_case_ids": list(self.validation_case_ids),
            "reserved_case_ids": sorted(
                set(self.reserved_case_ids)
                | RESERVED_EVIDENCE_CASE_IDS
            ),
            "lookback": self.lookback,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "TelemetryCorpusSplitSpec":
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported telemetry corpus split schema")
        return cls(
            training_case_ids=tuple(
                str(value) for value in payload["training_case_ids"]
            ),
            validation_case_ids=tuple(
                str(value) for value in payload["validation_case_ids"]
            ),
            reserved_case_ids=tuple(
                str(value) for value in payload["reserved_case_ids"]
            ),
            lookback=int(payload.get("lookback", 6)),
        )


@dataclass(frozen=True)
class TelemetryCorpusSplit:
    """Compiled windows plus the source run for every window."""

    windows: ModelWindows
    case_ids: Tuple[str, ...]
    window_case_ids: Tuple[str, ...]


@dataclass(frozen=True)
class TelemetryCorpus:
    """Train and validation tensors with auditable fitted preprocessing."""

    training: TelemetryCorpusSplit
    validation: TelemetryCorpusSplit
    conditioner_artifact: Mapping[str, Any]
    window_compiler_artifact: Mapping[str, Any]
    protocol: Mapping[str, Any]

    def metadata_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "telemetry_corpus",
            "conditioner": dict(self.conditioner_artifact),
            "window_compiler": dict(self.window_compiler_artifact),
            "protocol": dict(self.protocol),
        }


def compile_telemetry_corpus(
    runs: Sequence[FaultMatrixRun],
    feature_spec: OtlpFeatureSpec,
    split_spec: TelemetryCorpusSplitSpec,
) -> TelemetryCorpus:
    """Compile normal intervals with run-level and schedule-level isolation."""

    run_by_case_id = _index_runs(runs)
    selected_ids = (
        split_spec.training_case_ids
        + split_spec.validation_case_ids
    )
    missing = set(selected_ids) - set(run_by_case_id)
    if missing:
        raise ValueError(
            f"telemetry corpus cases are missing: {sorted(missing)}"
        )

    training_schedules = _schedules(
        run_by_case_id,
        split_spec.training_case_ids,
    )
    validation_schedules = _schedules(
        run_by_case_id,
        split_spec.validation_case_ids,
    )
    schedule_overlap = sorted(
        training_schedules & validation_schedules
    )
    if schedule_overlap:
        raise ValueError(
            "training and validation canonical request schedules "
            "must be disjoint"
        )

    conditioner = DemandConditioner()
    compiler = OtlpWindowCompiler(feature_spec)
    conditioned_by_case_id = {}
    provenance_by_case_id = {}
    application_builds = set()
    application_queue_sizes = set()
    for case_id in selected_ids:
        run = run_by_case_id[case_id]
        manifest_sha256 = _canonical_sha256(
            run.manifest.to_dict()
        )
        _validate_capture_identity(run, manifest_sha256)
        application_image_id, application_build_hash = (
            _application_build_provenance(run)
        )
        application_builds.add(
            (application_image_id, application_build_hash)
        )
        application_queue_size = (
            _application_api_request_queue_size(run)
        )
        application_queue_sizes.add(application_queue_size)
        compiled = compiler.compile(run.capture)
        if len(compiled.values) != run.manifest.point_count:
            raise ValueError(
                f"{case_id} point count does not match manifest"
            )
        if compiled.data_quality["missing_cells"] != 0:
            raise ValueError(
                "telemetry corpus requires complete feature cells"
            )
        normal_values = compiled.values[
            run.manifest.baseline_slice
        ]
        conditioned = conditioner.transform(
            normal_values,
            compiled.feature_names,
        )
        conditioned_by_case_id[case_id] = conditioned
        provenance_by_case_id[case_id] = {
            "case_id": case_id,
            "capture_sha256": run.capture.sha256,
            "manifest_sha256": manifest_sha256,
            "application_image_id": application_image_id,
            "application_build_context_sha256": (
                application_build_hash
            ),
            "application_api_request_queue_size": (
                application_queue_size
            ),
            "normal_interval": list(
                run.manifest.baseline_interval
            ),
            "normal_point_count": len(normal_values),
            "canonical_request_schedule": list(
                canonical_request_schedule(
                    run.manifest.requests_per_window,
                    run.manifest.load_pattern_offsets,
                )
            ),
        }
    if len(application_builds) != 1:
        raise ValueError(
            "telemetry corpus runs must use the same application build"
        )
    application_image_id, application_build_hash = next(
        iter(application_builds)
    )
    if len(application_queue_sizes) != 1:
        raise ValueError(
            "telemetry corpus runs must use the same API request "
            "queue size"
        )
    application_queue_size = next(
        iter(application_queue_sizes)
    )

    training_values = [
        conditioned_by_case_id[case_id].values
        for case_id in split_spec.training_case_ids
    ]
    window_compiler = WindowCompiler(split_spec.lookback).fit(
        np.concatenate(training_values, axis=0)
    )
    training = _compile_split(
        split_spec.training_case_ids,
        conditioned_by_case_id,
        window_compiler,
    )
    validation = _compile_split(
        split_spec.validation_case_ids,
        conditioned_by_case_id,
        window_compiler,
    )
    protocol = {
        "split_spec": split_spec.to_dict(),
        "feature_schema_id": feature_spec.schema_id,
        "feature_spec_sha256": _canonical_sha256(
            feature_spec.to_dict()
        ),
        "training_point_count": sum(
            len(values) for values in training_values
        ),
        "validation_point_count": sum(
            len(conditioned_by_case_id[case_id].values)
            for case_id in split_spec.validation_case_ids
        ),
        "training_window_count": len(training.windows.targets),
        "validation_window_count": len(validation.windows.targets),
        "training_validation_schedule_overlap": [],
        "context_crosses_run_boundary": False,
        "application_image_id": application_image_id,
        "application_build_context_sha256": (
            application_build_hash
        ),
        "application_api_request_queue_size": (
            application_queue_size
        ),
        "runs": {
            case_id: provenance_by_case_id[case_id]
            for case_id in selected_ids
        },
    }
    return TelemetryCorpus(
        training=training,
        validation=validation,
        conditioner_artifact=conditioner.to_dict(),
        window_compiler_artifact=window_compiler.to_dict(),
        protocol=protocol,
    )


def _index_runs(
    runs: Sequence[FaultMatrixRun],
) -> Mapping[str, FaultMatrixRun]:
    indexed = {}
    for run in runs:
        case_id = run.manifest.case_id
        if case_id in indexed:
            raise ValueError(
                f"duplicate telemetry corpus case_id: {case_id}"
            )
        indexed[case_id] = run
    return indexed


def _schedules(
    runs: Mapping[str, FaultMatrixRun],
    case_ids: Sequence[str],
) -> set[Tuple[int, ...]]:
    return {
        canonical_request_schedule(
            runs[case_id].manifest.requests_per_window,
            runs[case_id].manifest.load_pattern_offsets,
        )
        for case_id in case_ids
    }


def _validate_capture_identity(
    run: FaultMatrixRun,
    manifest_sha256: str,
) -> None:
    case_ids = {
        point.resource_attributes.get(
            "quantis.experiment.case.id"
        )
        for point in run.capture.points
    }
    fault_kinds = {
        point.resource_attributes.get(
            "quantis.experiment.fault.kind"
        )
        for point in run.capture.points
    }
    manifest_hashes = {
        point.resource_attributes.get(
            "quantis.experiment.manifest.sha256"
        )
        for point in run.capture.points
    }
    if not (
        case_ids == {run.manifest.case_id}
        and fault_kinds == {run.manifest.fault_kind}
        and manifest_hashes == {manifest_sha256}
    ):
        raise ValueError(
            f"{run.manifest.case_id} capture does not match manifest"
        )


def _application_build_provenance(
    run: FaultMatrixRun,
) -> Tuple[str, str]:
    image_ids = {
        point.resource_attributes.get(
            "quantis.application.image.id"
        )
        for point in run.capture.points
    }
    build_hashes = {
        point.resource_attributes.get(
            "quantis.application.build_context.sha256"
        )
        for point in run.capture.points
    }
    image_id = (
        str(next(iter(image_ids))) if len(image_ids) == 1 else ""
    )
    build_hash = (
        str(next(iter(build_hashes)))
        if len(build_hashes) == 1
        else ""
    )
    if (
        re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) is None
        or re.fullmatch(r"[0-9a-f]{64}", build_hash) is None
    ):
        raise ValueError(
            f"{run.manifest.case_id} application image provenance "
            "is missing or invalid"
        )
    return image_id, build_hash


def _application_api_request_queue_size(
    run: FaultMatrixRun,
) -> Optional[int]:
    values = {
        point.resource_attributes.get(
            "quantis.application.api.request_queue_size"
        )
        for point in run.capture.points
    }
    if values == {None}:
        return None
    raw = next(iter(values)) if len(values) == 1 else None
    if (
        isinstance(raw, bool)
        or not isinstance(raw, int)
        or raw < 1
    ):
        raise ValueError(
            f"{run.manifest.case_id} API request queue size "
            "provenance is missing or invalid"
        )
    return raw


def _compile_split(
    case_ids: Tuple[str, ...],
    conditioned_by_case_id: Mapping[str, Any],
    compiler: WindowCompiler,
) -> TelemetryCorpusSplit:
    windows = []
    window_case_ids = []
    for case_id in case_ids:
        conditioned = conditioned_by_case_id[case_id]
        run_windows = compiler.transform(
            conditioned.values,
            conditioned.feature_names,
        )
        windows.append(run_windows)
        window_case_ids.extend([case_id] * len(run_windows.targets))
    return TelemetryCorpusSplit(
        windows=_combine_windows(windows),
        case_ids=case_ids,
        window_case_ids=tuple(window_case_ids),
    )


def _combine_windows(
    windows: Sequence[ModelWindows],
) -> ModelWindows:
    feature_names = windows[0].feature_names
    if any(item.feature_names != feature_names for item in windows):
        raise ValueError("telemetry corpus feature names do not match")
    return ModelWindows(
        contexts=np.concatenate(
            [item.contexts for item in windows],
            axis=0,
        ),
        targets=np.concatenate(
            [item.targets for item in windows],
            axis=0,
        ),
        point_indices=np.concatenate(
            [item.point_indices for item in windows],
            axis=0,
        ),
        feature_names=feature_names,
    )


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
