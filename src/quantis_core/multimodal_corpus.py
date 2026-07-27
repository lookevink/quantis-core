"""Run-isolated metrics and application-log windows for multimodal JEPA."""

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .fault_matrix import FaultMatrixRun
from .otlp_log_windowing import (
    OtlpLogFeatureSpec,
    OtlpLogWindowCompiler,
)
from .otlp_logs import OtlpLogCapture
from .otlp_windowing import OtlpFeatureSpec
from .telemetry_corpus import (
    TelemetryCorpusSplitSpec,
    compile_telemetry_corpus,
)
from .windowing import ModelWindows, WindowCompiler


@dataclass(frozen=True)
class MultimodalModelWindows:
    """Aligned model windows with independent metric and log channels."""

    metric: ModelWindows
    logs: ModelWindows

    def __post_init__(self) -> None:
        if (
            self.metric.contexts.shape[:2]
            != self.logs.contexts.shape[:2]
            or self.metric.targets.shape[0]
            != self.logs.targets.shape[0]
            or not np.array_equal(
                self.metric.point_indices,
                self.logs.point_indices,
            )
        ):
            raise ValueError(
                "metric and log model windows must be aligned"
            )


@dataclass(frozen=True)
class MultimodalTelemetryCorpusSplit:
    """One run-level split and the case identity of every window."""

    windows: MultimodalModelWindows
    case_ids: Tuple[str, ...]
    window_case_ids: Tuple[str, ...]


@dataclass(frozen=True)
class MultimodalTelemetryCorpus:
    """Train and validation channels plus fitted preprocessing provenance."""

    training: MultimodalTelemetryCorpusSplit
    validation: MultimodalTelemetryCorpusSplit
    metric_corpus_metadata: Mapping[str, Any]
    log_feature_spec_artifact: Mapping[str, Any]
    log_window_compiler_artifact: Mapping[str, Any]
    protocol: Mapping[str, Any]

    def metadata_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "multimodal_telemetry_corpus",
            "metric_corpus": dict(self.metric_corpus_metadata),
            "log_feature_spec": dict(
                self.log_feature_spec_artifact
            ),
            "log_window_compiler": dict(
                self.log_window_compiler_artifact
            ),
            "protocol": dict(self.protocol),
        }


def compile_multimodal_telemetry_corpus(
    runs: Sequence[FaultMatrixRun],
    log_captures: Mapping[str, OtlpLogCapture],
    metric_spec: OtlpFeatureSpec,
    log_spec: OtlpLogFeatureSpec,
    split_spec: TelemetryCorpusSplitSpec,
) -> MultimodalTelemetryCorpus:
    """Compile separately normalized metric and structured-log channels."""

    metric_corpus = compile_telemetry_corpus(
        runs,
        metric_spec,
        split_spec,
    )
    runs_by_case_id = {
        run.manifest.case_id: run for run in runs
    }
    selected_case_ids = (
        split_spec.training_case_ids
        + split_spec.validation_case_ids
    )
    missing_logs = set(selected_case_ids) - set(log_captures)
    if missing_logs:
        raise ValueError(
            "multimodal corpus log captures are missing: "
            f"{sorted(missing_logs)}"
        )

    log_values_by_case_id = {}
    log_provenance = {}
    log_compiler = OtlpLogWindowCompiler(log_spec)
    for case_id in selected_case_ids:
        run = runs_by_case_id[case_id]
        capture = log_captures[case_id]
        manifest_sha256 = _canonical_sha256(
            run.manifest.to_dict()
        )
        _validate_log_capture_identity(
            run,
            capture,
            manifest_sha256,
        )
        compiled = log_compiler.compile(
            capture,
            run.manifest.point_count,
        )
        normal_values = compiled.values[
            run.manifest.baseline_slice
        ]
        log_values_by_case_id[case_id] = normal_values
        log_provenance[case_id] = {
            "case_id": case_id,
            "log_capture_sha256": capture.sha256,
            "log_record_count": len(capture.records),
            "log_data_quality": dict(compiled.data_quality),
            "normal_interval": list(
                run.manifest.baseline_interval
            ),
        }

    fitted_log_window_compiler = WindowCompiler(
        split_spec.lookback
    ).fit(
        np.concatenate(
            [
                log_values_by_case_id[case_id]
                for case_id in split_spec.training_case_ids
            ],
            axis=0,
        )
    )
    training_logs = _compile_log_split(
        split_spec.training_case_ids,
        log_values_by_case_id,
        log_spec,
        fitted_log_window_compiler,
    )
    validation_logs = _compile_log_split(
        split_spec.validation_case_ids,
        log_values_by_case_id,
        log_spec,
        fitted_log_window_compiler,
    )
    training = _multimodal_split(
        metric_corpus.training.windows,
        training_logs,
        metric_corpus.training.case_ids,
        metric_corpus.training.window_case_ids,
    )
    validation = _multimodal_split(
        metric_corpus.validation.windows,
        validation_logs,
        metric_corpus.validation.case_ids,
        metric_corpus.validation.window_case_ids,
    )
    return MultimodalTelemetryCorpus(
        training=training,
        validation=validation,
        metric_corpus_metadata=metric_corpus.metadata_dict(),
        log_feature_spec_artifact=log_spec.to_dict(),
        log_window_compiler_artifact=(
            fitted_log_window_compiler.to_dict()
        ),
        protocol={
            "split_spec": split_spec.to_dict(),
            "training_window_count": len(
                training.windows.metric.targets
            ),
            "validation_window_count": len(
                validation.windows.metric.targets
            ),
            "context_crosses_run_boundary": False,
            "channels_aligned": True,
            "runs": {
                case_id: log_provenance[case_id]
                for case_id in selected_case_ids
            },
        },
    )


def _validate_log_capture_identity(
    run: FaultMatrixRun,
    capture: OtlpLogCapture,
    manifest_sha256: str,
) -> None:
    case_ids = {
        record.resource_attributes.get(
            "quantis.experiment.case.id"
        )
        for record in capture.records
    }
    fault_kinds = {
        record.resource_attributes.get(
            "quantis.experiment.fault.kind"
        )
        for record in capture.records
    }
    manifest_hashes = {
        record.resource_attributes.get(
            "quantis.experiment.manifest.sha256"
        )
        for record in capture.records
    }
    if not (
        case_ids == {run.manifest.case_id}
        and fault_kinds == {run.manifest.fault_kind}
        and manifest_hashes == {manifest_sha256}
    ):
        raise ValueError(
            f"{run.manifest.case_id} log capture does not match manifest"
        )


def _compile_log_split(
    case_ids: Tuple[str, ...],
    values_by_case_id: Mapping[str, NDArray[np.float64]],
    log_spec: OtlpLogFeatureSpec,
    compiler: WindowCompiler,
) -> ModelWindows:
    windows = [
        compiler.transform(
            values_by_case_id[case_id],
            tuple(feature.name for feature in log_spec.features),
        )
        for case_id in case_ids
    ]
    return _combine_windows(windows)


def _combine_windows(
    windows: Sequence[ModelWindows],
) -> ModelWindows:
    feature_names = windows[0].feature_names
    if any(item.feature_names != feature_names for item in windows):
        raise ValueError("log corpus feature names do not match")
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


def _multimodal_split(
    metric_windows: ModelWindows,
    log_windows: ModelWindows,
    case_ids: Tuple[str, ...],
    window_case_ids: Tuple[str, ...],
) -> MultimodalTelemetryCorpusSplit:
    windows = MultimodalModelWindows(
        metric=metric_windows,
        logs=log_windows,
    )
    if len(window_case_ids) != len(windows.metric.targets):
        raise ValueError(
            "multimodal case identities do not match model windows"
        )
    return MultimodalTelemetryCorpusSplit(
        windows=windows,
        case_ids=case_ids,
        window_case_ids=window_case_ids,
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
