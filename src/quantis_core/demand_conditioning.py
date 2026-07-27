"""Demand-conditioned telemetry and model training for schedule robustness."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Mapping, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .detectors import DemandConditionedCoherentDetector
from .otlp_windowing import OtlpFeatureSpec, OtlpWindowCompiler
from .windowing import ModelWindows, WindowCompiler

if TYPE_CHECKING:
    from .fault_matrix import FaultMatrixRun


CONDITIONED_FEATURE_NAMES = (
    "request_latency_ms",
    "error_rate",
    "queue_depth",
    "worker_completion_ratio",
    "worker_heartbeat_age_s",
    "db_write_completion_ratio",
)
REQUIRED_RAW_FEATURES = (
    "request_rate",
    "request_latency_ms",
    "error_rate",
    "queue_depth",
    "worker_rate",
    "worker_heartbeat_age_s",
    "db_write_rate",
)
LOOKBACK = 6
LATENT_DIMENSION = 1
RIDGE = 1e-2
CALIBRATION_QUANTILE = 0.98
THRESHOLD_SAFETY_MULTIPLIER = 2.0
CONSENSUS_RANK = 2
RESIDUAL_SCALE_FLOOR = 1e-3


def canonical_request_schedule(
    requests_per_window: int,
    load_pattern_offsets: Sequence[int],
) -> Tuple[int, ...]:
    """Return the shortest repeating sequence of realized request counts."""

    realized = tuple(
        requests_per_window + int(offset)
        for offset in load_pattern_offsets
    )
    if not realized or min(realized) < 1:
        raise ValueError("request schedule must keep demand positive")
    for period in range(1, len(realized) + 1):
        candidate = realized[:period]
        if all(
            value == candidate[index % period]
            for index, value in enumerate(realized)
        ):
            return candidate
    raise RuntimeError("finite request schedule has no canonical period")


@dataclass(frozen=True)
class ConditionedTelemetry:
    """Finite model values and their stable semantic names."""

    values: NDArray[np.float64]
    feature_names: Tuple[str, ...]


class DemandConditioner:
    """Express downstream throughput relative to observed request demand."""

    kind = "request_demand_ratios"

    def transform(
        self,
        values: NDArray[np.float64],
        feature_names: Sequence[str],
    ) -> ConditionedTelemetry:
        telemetry = np.asarray(values, dtype=np.float64)
        names = tuple(feature_names)
        if telemetry.ndim != 2 or telemetry.shape[1] != len(names):
            raise ValueError(
                "telemetry columns must match feature_names"
            )
        if not np.all(np.isfinite(telemetry)):
            raise ValueError("telemetry values must be finite")
        if len(set(names)) != len(names):
            raise ValueError("feature_names must be unique")
        missing = set(REQUIRED_RAW_FEATURES) - set(names)
        if missing:
            raise ValueError(
                f"demand-conditioning features are missing: "
                f"{sorted(missing)}"
            )
        index = {name: position for position, name in enumerate(names)}
        request_rate = telemetry[:, index["request_rate"]]
        if np.any(request_rate <= 0.0):
            raise ValueError(
                "demand conditioning requires positive request demand"
            )
        conditioned = np.column_stack(
            (
                telemetry[:, index["request_latency_ms"]],
                telemetry[:, index["error_rate"]],
                telemetry[:, index["queue_depth"]],
                telemetry[:, index["worker_rate"]] / request_rate,
                telemetry[:, index["worker_heartbeat_age_s"]],
                telemetry[:, index["db_write_rate"]] / request_rate,
            )
        )
        return ConditionedTelemetry(
            values=conditioned,
            feature_names=CONDITIONED_FEATURE_NAMES,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": self.kind,
        }

    def map_affected_features(
        self, raw_feature_names: Sequence[str]
    ) -> Tuple[str, ...]:
        mapping = {
            "worker_rate": "worker_completion_ratio",
            "db_write_rate": "db_write_completion_ratio",
            "request_rate": "",
        }
        mapped = tuple(
            mapping.get(name, name) for name in raw_feature_names
        )
        return tuple(name for name in mapped if name)

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "DemandConditioner":
        if (
            payload.get("schema_version") != 1
            or payload.get("kind") != cls.kind
        ):
            raise ValueError("unsupported DemandConditioner artifact")
        return cls()


@dataclass(frozen=True)
class DemandConditionedModel:
    """Serializable fitted v2 model plus auditable training provenance."""

    conditioner_artifact: Mapping[str, Any]
    window_compiler_artifact: Mapping[str, Any]
    detector_artifact: Mapping[str, Any]
    protocol: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "demand_conditioned_model",
            "conditioner": dict(self.conditioner_artifact),
            "window_compiler": dict(self.window_compiler_artifact),
            "detector": dict(self.detector_artifact),
            "protocol": dict(self.protocol),
        }

    def to_bytes(self) -> bytes:
        return (
            json.dumps(
                self.to_dict(),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")


def write_demand_conditioned_model(
    model: DemandConditionedModel, output_directory: Path
) -> Mapping[str, Path]:
    """Write the frozen v2 artifact and its training provenance."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "model": output / "model.json",
        "training": output / "training.json",
    }
    paths["model"].write_bytes(model.to_bytes())
    paths["training"].write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_file_sha256": hashlib.sha256(
                    model.to_bytes()
                ).hexdigest(),
                "protocol": dict(model.protocol),
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    return paths


def train_demand_conditioned_model(
    runs: Sequence["FaultMatrixRun"],
    feature_spec: OtlpFeatureSpec,
) -> DemandConditionedModel:
    """Fit v2 on separately windowed, fault-free schedule intervals."""

    if len(runs) < 3:
        raise ValueError(
            "demand-conditioned training requires at least three runs"
        )
    schedules = {
        (
            run.manifest.requests_per_window,
            tuple(run.manifest.load_pattern_offsets),
        )
        for run in runs
    }
    if len(schedules) < 3:
        raise ValueError(
            "demand-conditioned training requires three distinct schedules"
        )
    conditioner = DemandConditioner()
    conditioned_runs = []
    provenance = []
    structural_points = 0
    for run in runs:
        manifest_sha256 = _canonical_sha256(
            run.manifest.to_dict()
        )
        case_ids = {
            point.resource_attributes.get("quantis.experiment.case.id")
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
        compiled = OtlpWindowCompiler(feature_spec).compile(run.capture)
        if len(compiled.window_end_unix_nano) != run.manifest.point_count:
            raise ValueError(
                f"{run.manifest.case_id} point count does not match manifest"
            )
        if compiled.data_quality["missing_cells"] != 0:
            raise ValueError(
                "demand-conditioned training requires complete telemetry"
            )
        baseline = conditioner.transform(
            compiled.values[run.manifest.baseline_slice],
            compiled.feature_names,
        )
        conditioned_runs.append(baseline)
        structural_points += _overlap_size(
            run.manifest.baseline_interval,
            run.manifest.structural_interval,
        )
        provenance.append(
            {
                "case_id": run.manifest.case_id,
                "capture_sha256": run.capture.sha256,
                "manifest_sha256": manifest_sha256,
                "training_interval": list(
                    run.manifest.baseline_interval
                ),
                "load_pattern_offsets": list(
                    run.manifest.load_pattern_offsets
                ),
                "requests_per_window": (
                    run.manifest.requests_per_window
                ),
                "canonical_request_schedule": list(
                    canonical_request_schedule(
                        run.manifest.requests_per_window,
                        run.manifest.load_pattern_offsets,
                    )
                ),
                "fault_timing": {
                    "fault_kind": run.manifest.fault_kind,
                    "structural_interval": list(
                        run.manifest.structural_interval
                    ),
                },
            }
        )
    if structural_points != 0:
        raise ValueError("training intervals must contain no structural points")
    compiler = WindowCompiler(LOOKBACK).fit(
        np.concatenate(
            [telemetry.values for telemetry in conditioned_runs],
            axis=0,
        )
    )
    run_windows = [
        compiler.transform(
            telemetry.values, telemetry.feature_names
        )
        for telemetry in conditioned_runs
    ]
    training_windows = _combine_windows(run_windows)
    detector = DemandConditionedCoherentDetector(
        latent_dimension=LATENT_DIMENSION,
        ridge=RIDGE,
        calibration_quantile=CALIBRATION_QUANTILE,
        consensus_rank=CONSENSUS_RANK,
        residual_scale_floor=RESIDUAL_SCALE_FLOOR,
    ).fit(training_windows)
    detector.threshold *= THRESHOLD_SAFETY_MULTIPLIER
    config = {
        "lookback": LOOKBACK,
        "latent_dimension": LATENT_DIMENSION,
        "ridge": RIDGE,
        "calibration_quantile": CALIBRATION_QUANTILE,
        "threshold_safety_multiplier": (
            THRESHOLD_SAFETY_MULTIPLIER
        ),
        "consensus_rank": CONSENSUS_RANK,
        "residual_scale_floor": RESIDUAL_SCALE_FLOOR,
    }
    return DemandConditionedModel(
        conditioner_artifact=conditioner.to_dict(),
        window_compiler_artifact=compiler.to_dict(),
        detector_artifact=detector.to_dict(),
        protocol={
            "training_run_count": len(runs),
            "distinct_load_schedule_count": len(schedules),
            "training_point_count": sum(
                len(telemetry.values)
                for telemetry in conditioned_runs
            ),
            "training_window_count": len(training_windows.targets),
            "training_structural_points": structural_points,
            "window_boundary_policy": (
                "compile each run separately before concatenating windows"
            ),
            "training_runs": provenance,
            "config": config,
            "config_sha256": _canonical_sha256(config),
        },
    )


def _combine_windows(windows: Sequence[ModelWindows]) -> ModelWindows:
    if not windows:
        raise ValueError("cannot combine an empty window collection")
    feature_names = windows[0].feature_names
    if any(item.feature_names != feature_names for item in windows):
        raise ValueError("all training runs must share conditioned features")
    return ModelWindows(
        contexts=np.concatenate(
            [item.contexts for item in windows], axis=0
        ),
        targets=np.concatenate(
            [item.targets for item in windows], axis=0
        ),
        point_indices=np.concatenate(
            [item.point_indices for item in windows], axis=0
        ),
        feature_names=feature_names,
    )


def _overlap_size(
    left: Tuple[int, int], right: Tuple[int, int]
) -> int:
    return max(0, min(left[1], right[1]) - max(left[0], right[0]))


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
