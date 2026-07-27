"""Evaluation contracts for the instrumented API/worker fault lab."""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .detectors import CoherentLatentPredictiveDetector
from .otlp import TelemetryCapture
from .otlp_windowing import OtlpFeatureSpec, OtlpWindowCompiler
from .windowing import (
    WindowCompiler,
    repair_isolated_context_outliers,
)


Interval = Tuple[int, int]


@dataclass(frozen=True)
class FaultLabManifest:
    """Independent phase and affected-feature truth for one lab run."""

    point_count: int
    sample_period_seconds: float
    logical_window_period_nano: int
    training_interval: Interval
    routine_noise_interval: Interval
    structural_interval: Interval
    affected_features: Tuple[str, ...]
    requests_per_window: int = 1
    routine_noise_delay_ms: int = 0
    load_pattern_offsets: Tuple[int, ...] = (0,)
    images: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.point_count < 1:
            raise ValueError("point_count must be positive")
        if self.sample_period_seconds <= 0.0:
            raise ValueError("sample_period_seconds must be positive")
        if self.logical_window_period_nano <= 0:
            raise ValueError("logical_window_period_nano must be positive")
        for name, interval in (
            ("training", self.training_interval),
            ("routine noise", self.routine_noise_interval),
            ("structural", self.structural_interval),
        ):
            start, stop = interval
            if not 0 <= start < stop <= self.point_count:
                raise ValueError(f"{name} interval is outside the experiment")
        if self.training_interval[0] != 0:
            raise ValueError("training interval must begin at zero")
        if self.training_interval[1] > self.routine_noise_interval[0]:
            raise ValueError("training interval must contain baseline only")
        if self.training_interval[1] > self.structural_interval[0]:
            raise ValueError("training interval must contain baseline only")
        if self.routine_noise_interval[1] > self.structural_interval[0]:
            raise ValueError("routine noise must precede structural fault")
        if _overlap(self.routine_noise_interval, self.structural_interval):
            raise ValueError("fault intervals cannot overlap")
        if not self.affected_features or any(
            not name for name in self.affected_features
        ):
            raise ValueError("affected_features cannot be empty")
        if len(set(self.affected_features)) != len(self.affected_features):
            raise ValueError("affected_features must be unique")
        if self.requests_per_window < 1:
            raise ValueError("requests_per_window must be positive")
        if self.routine_noise_delay_ms < 0:
            raise ValueError("routine_noise_delay_ms cannot be negative")
        if (
            not self.load_pattern_offsets
            or self.requests_per_window + min(self.load_pattern_offsets) < 1
        ):
            raise ValueError("load pattern must keep request count positive")
        if self.images and any(
            "@sha256:" not in image for image in self.images.values()
        ):
            raise ValueError("fault-lab images must be digest pinned")

    @property
    def training_slice(self) -> slice:
        return slice(*self.training_interval)

    def phase_labels(self) -> NDArray[np.str_]:
        labels = np.full(self.point_count, "recovery", dtype="<U16")
        training_start, training_stop = self.training_interval
        labels[training_start:training_stop] = "baseline"
        noise_start, noise_stop = self.routine_noise_interval
        labels[training_stop:noise_start] = "validation"
        labels[noise_start:noise_stop] = "routine_noise"
        structural_start, structural_stop = self.structural_interval
        labels[structural_start:structural_stop] = "structural"
        return labels

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": 1,
            "point_count": self.point_count,
            "sample_period_seconds": self.sample_period_seconds,
            "logical_window_period_nano": self.logical_window_period_nano,
            "training_interval": list(self.training_interval),
            "routine_noise_interval": list(self.routine_noise_interval),
            "structural_interval": list(self.structural_interval),
            "affected_features": list(self.affected_features),
            "requests_per_window": self.requests_per_window,
            "routine_noise_delay_ms": self.routine_noise_delay_ms,
            "load_pattern_offsets": list(self.load_pattern_offsets),
            "images": dict(self.images),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FaultLabManifest":
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported FaultLabManifest schema_version")
        return cls(
            point_count=int(payload["point_count"]),
            sample_period_seconds=float(payload["sample_period_seconds"]),
            logical_window_period_nano=int(
                payload["logical_window_period_nano"]
            ),
            training_interval=_interval(payload["training_interval"]),
            routine_noise_interval=_interval(
                payload["routine_noise_interval"]
            ),
            structural_interval=_interval(payload["structural_interval"]),
            affected_features=tuple(
                str(name) for name in payload["affected_features"]
            ),
            requests_per_window=int(
                payload.get("requests_per_window", 1)
            ),
            routine_noise_delay_ms=int(
                payload.get("routine_noise_delay_ms", 0)
            ),
            load_pattern_offsets=tuple(
                int(offset)
                for offset in payload.get("load_pattern_offsets", [0])
            ),
            images={
                str(name): str(image)
                for name, image in payload.get("images", {}).items()
            },
        )


@dataclass(frozen=True)
class FaultLabEvaluationConfig:
    """Detector and operational gates fixed before observing fault results."""

    lookback: int = 6
    latent_dimension: int = 1
    ridge: float = 1e-2
    calibration_quantile: float = 0.98
    threshold_safety_multiplier: float = 2.0
    consensus_rank: int = 3
    maximum_detection_delay_windows: int = 6
    maximum_noise_alert_rate: float = 0.2
    maximum_validation_alert_rate: float = 0.2
    minimum_backlog_growth: float = 20.0
    maximum_fault_rate_ratio: float = 0.2
    minimum_noise_latency_ratio: float = 3.0
    isolated_context_z_threshold: float = 8.0


@dataclass(frozen=True)
class FaultLabReport:
    """Auditable measurements and acceptance decisions for one lab run."""

    capture: Mapping[str, Any]
    compiled: Mapping[str, Any]
    protocol: Mapping[str, Any]
    raw_effects: Mapping[str, float]
    detection: Mapping[str, Any]
    attribution: Mapping[str, Any]
    acceptance: Mapping[str, Any]
    window_compiler_artifact: Mapping[str, Any]
    detector_artifact: Mapping[str, Any]
    limitations: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "capture": dict(self.capture),
            "compiled": dict(self.compiled),
            "protocol": dict(self.protocol),
            "raw_effects": dict(self.raw_effects),
            "detection": dict(self.detection),
            "attribution": dict(self.attribution),
            "acceptance": dict(self.acceptance),
            "window_compiler_artifact": dict(
                self.window_compiler_artifact
            ),
            "detector_artifact": dict(self.detector_artifact),
            "limitations": list(self.limitations),
        }


def evaluate_fault_lab(
    capture: TelemetryCapture,
    feature_spec: OtlpFeatureSpec,
    manifest: FaultLabManifest,
    config: FaultLabEvaluationConfig = FaultLabEvaluationConfig(),
) -> FaultLabReport:
    """Compile, fit on baseline only, and score one instrumented lab capture."""

    if feature_spec.window_period_nano != manifest.logical_window_period_nano:
        raise ValueError("feature spec and manifest window periods differ")
    if (
        manifest.routine_noise_interval[1] + config.lookback
        > manifest.structural_interval[0]
    ):
        raise ValueError(
            "manifest must preserve the full routine-noise response horizon"
        )
    compiled = OtlpWindowCompiler(feature_spec).compile(capture)
    if len(compiled.window_end_unix_nano) != manifest.point_count:
        raise ValueError("compiled point count does not match fault-lab manifest")
    if compiled.data_quality["missing_cells"] != 0:
        raise ValueError("fault-lab evaluation requires complete telemetry")
    missing_affected = set(manifest.affected_features) - set(
        compiled.feature_names
    )
    if missing_affected:
        raise ValueError(
            f"affected features are absent from telemetry: "
            f"{sorted(missing_affected)}"
        )
    application_image_ids = {
        point.resource_attributes.get(
            "quantis.application.image.id"
        )
        for point in capture.points
    }
    if (
        len(application_image_ids) != 1
        or not _is_sha256_identifier(next(iter(application_image_ids)))
    ):
        raise ValueError(
            "capture must identify exactly one digest-addressed "
            "application image"
        )
    application_image_id = str(next(iter(application_image_ids)))
    application_build_hashes = {
        point.resource_attributes.get(
            "quantis.application.build_context.sha256"
        )
        for point in capture.points
    }
    if (
        len(application_build_hashes) != 1
        or not _is_sha256_hex(next(iter(application_build_hashes)))
    ):
        raise ValueError(
            "capture must identify exactly one application build context"
        )
    application_build_context_sha256 = str(
        next(iter(application_build_hashes))
    )

    training_values = compiled.values[manifest.training_slice]
    compiler = WindowCompiler(config.lookback).fit(training_values)
    training_windows = compiler.transform(
        training_values, compiled.feature_names
    )
    all_windows = compiler.transform(
        compiled.values, compiled.feature_names
    )
    detector = CoherentLatentPredictiveDetector(
        latent_dimension=config.latent_dimension,
        ridge=config.ridge,
        calibration_quantile=config.calibration_quantile,
        consensus_rank=config.consensus_rank,
    ).fit(training_windows)
    detector.threshold *= config.threshold_safety_multiplier
    robust_windows, repaired_context_cells = (
        repair_isolated_context_outliers(
            all_windows,
            z_threshold=config.isolated_context_z_threshold,
            consensus_rank=config.consensus_rank,
        )
    )
    scores = detector.score(robust_windows)
    phases = manifest.phase_labels()[all_windows.point_indices]
    structural_mask = phases == "structural"
    noise_response_start = manifest.routine_noise_interval[0]
    noise_response_stop = (
        manifest.routine_noise_interval[1] + config.lookback
    )
    routine_noise_mask = (
        (all_windows.point_indices >= noise_response_start)
        & (all_windows.point_indices < noise_response_stop)
    )
    validation_mask = phases == "validation"
    structural_alert_positions = np.flatnonzero(
        scores.alerts & structural_mask
    )
    structural_detected = len(structural_alert_positions) > 0
    first_detection_position: Optional[int] = None
    first_detection_point: Optional[int] = None
    detection_delay: Optional[int] = None
    top_features: Tuple[str, ...] = ()
    if structural_detected:
        first_detection_position = int(structural_alert_positions[0])
        first_detection_point = int(
            all_windows.point_indices[first_detection_position]
        )
        detection_delay = (
            first_detection_point - manifest.structural_interval[0]
        )
        evidence = scores.feature_evidence[first_detection_position]
        top_indices = np.argsort(evidence)[-3:][::-1]
        top_features = tuple(
            all_windows.feature_names[int(index)] for index in top_indices
        )
    attribution_hit = bool(
        set(top_features) & set(manifest.affected_features)
    )
    noise_points = int(np.count_nonzero(routine_noise_mask))
    noise_alerts = int(
        np.count_nonzero(scores.alerts & routine_noise_mask)
    )
    noise_alert_rate = noise_alerts / noise_points if noise_points else 0.0
    validation_points = int(np.count_nonzero(validation_mask))
    validation_alerts = int(
        np.count_nonzero(scores.alerts & validation_mask)
    )
    validation_alert_rate = (
        validation_alerts / validation_points
        if validation_points
        else 0.0
    )

    raw_effects = _raw_fault_effects(
        compiled.values,
        compiled.feature_names,
        manifest,
    )
    manifest_sha256 = _canonical_sha256(manifest.to_dict())
    config_payload = {
        key: value for key, value in vars(config).items()
    }
    evaluator_config_sha256 = _canonical_sha256(config_payload)
    gates = {
        "complete_telemetry": compiled.data_quality["missing_cells"] == 0,
        "backlog_growth_at_least_minimum": (
            raw_effects["queue_depth_growth"]
            >= config.minimum_backlog_growth
        ),
        "routine_noise_has_observed_effect": (
            raw_effects["routine_noise_latency_ratio"]
            >= config.minimum_noise_latency_ratio
        ),
        "worker_rate_collapses": (
            raw_effects["worker_rate_fault_ratio"]
            <= config.maximum_fault_rate_ratio
        ),
        "db_write_rate_collapses": (
            raw_effects["db_write_rate_fault_ratio"]
            <= config.maximum_fault_rate_ratio
        ),
        "structural_event_detected": structural_detected,
        "non_degenerate_calibration_threshold": scores.threshold > 0.0,
        "detection_delay_within_limit": (
            detection_delay is not None
            and detection_delay <= config.maximum_detection_delay_windows
        ),
        "routine_noise_alert_rate_within_limit": (
            noise_points > 0
            and noise_alert_rate <= config.maximum_noise_alert_rate
        ),
        "validation_alert_rate_within_limit": (
            validation_alert_rate
            <= config.maximum_validation_alert_rate
        ),
        "attribution_hit_at_3": attribution_hit,
        "content_addressed_inputs": (
            len(capture.sha256) == 64
            and len(compiled.feature_schema_id) == 64
            and len(manifest_sha256) == 64
            and len(evaluator_config_sha256) == 64
            and bool(manifest.images)
            and all(
                "@sha256:" in image
                for image in manifest.images.values()
            )
            and _is_sha256_identifier(application_image_id)
            and _is_sha256_hex(application_build_context_sha256)
        ),
    }
    detector_artifact = detector.to_dict()
    model_version = hashlib.sha256(
        json.dumps(
            detector_artifact, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return FaultLabReport(
        capture={
            "sha256": capture.sha256,
            "json_message_count": capture.json_message_count,
            "metric_point_count": len(capture.points),
        },
        compiled={
            "feature_schema_id": compiled.feature_schema_id,
            "window_count": len(compiled.window_end_unix_nano),
            "feature_count": len(compiled.feature_names),
            "feature_names": list(compiled.feature_names),
            "data_quality": dict(compiled.data_quality),
        },
        protocol={
            "training_interval": list(manifest.training_interval),
            "routine_noise_interval": list(
                manifest.routine_noise_interval
            ),
            "routine_noise_response_interval": [
                noise_response_start,
                noise_response_stop,
            ],
            "structural_interval": list(manifest.structural_interval),
            "training_structural_points": int(
                np.count_nonzero(
                    manifest.phase_labels()[manifest.training_slice]
                    == "structural"
                )
            ),
            "calibration_source": "baseline training interval only",
            "sample_period_seconds": manifest.sample_period_seconds,
            "logical_window_period_nano": (
                manifest.logical_window_period_nano
            ),
            "detector_kind": detector.kind,
            "model_version": model_version,
            "manifest_sha256": manifest_sha256,
            "evaluator_config_sha256": evaluator_config_sha256,
            "images": dict(manifest.images),
            "application_image_id": application_image_id,
            "application_build_context_sha256": (
                application_build_context_sha256
            ),
            "config": config_payload,
        },
        raw_effects=raw_effects,
        detection={
            "threshold": float(scores.threshold),
            "alert_count": int(np.count_nonzero(scores.alerts)),
            "structural_detected": structural_detected,
            "first_detection_point": first_detection_point,
            "detection_delay_windows": detection_delay,
            "detection_latency_wall_seconds_upper_bound": (
                (detection_delay + 1) * manifest.sample_period_seconds
                if detection_delay is not None
                else None
            ),
            "routine_noise_points": noise_points,
            "routine_noise_alerts": noise_alerts,
            "routine_noise_alert_rate": noise_alert_rate,
            "repaired_isolated_context_cells": (
                repaired_context_cells
            ),
            "validation_points": validation_points,
            "validation_alerts": validation_alerts,
            "validation_alert_rate": validation_alert_rate,
        },
        attribution={
            "expected_features": list(manifest.affected_features),
            "top_features": list(top_features),
            "hit_at_3": attribution_hit,
        },
        acceptance={
            "all_passed": all(gates.values()),
            "gates": gates,
        },
        window_compiler_artifact=compiler.to_dict(),
        detector_artifact=detector_artifact,
        limitations=(
            "This is one controlled local topology and one injected worker stall.",
            "Evaluator preprocessing and threshold margin were developed against "
            "this topology and schedule; this is development evidence, not an "
            "untouched confirmatory experiment.",
            "False-positive evidence covers one noise point and eight validation "
            "points, plus the six-window noise-response horizon, in a single run.",
            "Logical event-time windows are sampled faster than wall clock.",
            "The detector is fitted and tested on different intervals of one run.",
            "Feature evidence is associative attribution, not causal proof.",
            "The target encoder is linear PCA, not a learned JEPA encoder.",
        ),
    )


def write_fault_lab_artifacts(
    report: FaultLabReport, output_directory: Path
) -> Mapping[str, Path]:
    """Write the versioned machine report and human-readable summary."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "verification": output / "verification.json",
        "report": output / "report.md",
        "window_compiler": output / "window-compiler.json",
        "detector": output / "detector.json",
    }
    paths["verification"].write_text(
        json.dumps(
            report.to_dict(),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    paths["report"].write_text(_markdown_report(report))
    paths["window_compiler"].write_text(
        json.dumps(
            report.window_compiler_artifact,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    paths["detector"].write_text(
        json.dumps(
            report.detector_artifact,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    return paths


def _raw_fault_effects(
    values: NDArray[np.float64],
    feature_names: Tuple[str, ...],
    manifest: FaultLabManifest,
) -> Dict[str, float]:
    feature_index = {
        name: index for index, name in enumerate(feature_names)
    }
    required = {
        "request_latency_ms",
        "queue_depth",
        "worker_rate",
        "db_write_rate",
    }
    missing = required - set(feature_index)
    if missing:
        raise ValueError(
            f"raw fault-effect features are missing: {sorted(missing)}"
        )
    baseline = values[manifest.training_slice]
    structural = values[
        manifest.structural_interval[0] : manifest.structural_interval[1]
    ]
    baseline_queue = float(
        np.median(baseline[:, feature_index["queue_depth"]])
    )
    fault_queue_max = float(
        np.max(structural[:, feature_index["queue_depth"]])
    )
    baseline_worker = float(
        np.median(baseline[:, feature_index["worker_rate"]])
    )
    fault_worker = float(
        np.median(structural[:, feature_index["worker_rate"]])
    )
    baseline_db = float(
        np.median(baseline[:, feature_index["db_write_rate"]])
    )
    fault_db = float(
        np.median(structural[:, feature_index["db_write_rate"]])
    )
    baseline_latency = float(
        np.median(baseline[:, feature_index["request_latency_ms"]])
    )
    routine_noise = values[
        manifest.routine_noise_interval[0] :
        manifest.routine_noise_interval[1]
    ]
    noise_latency = float(
        np.median(
            routine_noise[:, feature_index["request_latency_ms"]]
        )
    )
    return {
        "baseline_queue_depth_median": baseline_queue,
        "fault_queue_depth_max": fault_queue_max,
        "queue_depth_growth": fault_queue_max - baseline_queue,
        "baseline_worker_rate_median": baseline_worker,
        "fault_worker_rate_median": fault_worker,
        "worker_rate_fault_ratio": _ratio(fault_worker, baseline_worker),
        "baseline_db_write_rate_median": baseline_db,
        "fault_db_write_rate_median": fault_db,
        "db_write_rate_fault_ratio": _ratio(fault_db, baseline_db),
        "baseline_request_latency_ms_median": baseline_latency,
        "routine_noise_request_latency_ms_median": noise_latency,
        "routine_noise_latency_ratio": _ratio(
            noise_latency, baseline_latency
        ),
    }


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        raise ValueError("baseline rate must be positive")
    return numerator / denominator


def _is_sha256_identifier(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    digest = value.removeprefix("sha256:")
    return len(digest) == 64 and all(
        character in "0123456789abcdef" for character in digest
    )


def _is_sha256_hex(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(
            character in "0123456789abcdef" for character in value
        )
    )


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _markdown_report(report: FaultLabReport) -> str:
    status = "PASS" if report.acceptance["all_passed"] else "FAIL"
    detection = report.detection
    attribution = report.attribution
    effects = report.raw_effects
    lines = [
        "# Quantis instrumented fault-lab verification",
        "",
        f"Overall acceptance: **{status}**",
        "",
        "## Observed system effects",
        "",
        f"- Redis queue growth: {effects['queue_depth_growth']:.1f} jobs",
        f"- Worker fault/baseline rate ratio: "
        f"{effects['worker_rate_fault_ratio']:.3f}",
        f"- Database fault/baseline write ratio: "
        f"{effects['db_write_rate_fault_ratio']:.3f}",
        f"- Routine-noise/baseline latency ratio: "
        f"{effects['routine_noise_latency_ratio']:.1f}×",
        "",
        "## Detection",
        "",
        f"- Structural event detected: {detection['structural_detected']}",
        f"- Detection delay: {detection['detection_delay_windows']} "
        "logical windows",
        f"- Detection wall-time upper bound: "
        f"{detection['detection_latency_wall_seconds_upper_bound']:.3f}s",
        f"- Routine-noise response-horizon alert rate: "
        f"{detection['routine_noise_alert_rate']:.3f}",
        f"- Pre-fault validation alert rate: "
        f"{detection['validation_alert_rate']:.3f}",
        f"- Attribution top three: "
        f"{', '.join(attribution['top_features'])}",
        f"- Attribution hit@3: {attribution['hit_at_3']}",
        "",
        "## Provenance",
        "",
        f"- Capture SHA-256: `{report.capture['sha256']}`",
        f"- Application image ID: "
        f"`{report.protocol['application_image_id']}`",
        f"- Application build context SHA-256: "
        f"`{report.protocol['application_build_context_sha256']}`",
        "",
        "## Acceptance gates",
        "",
    ]
    for gate, passed in report.acceptance["gates"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: `{gate}`")
    lines.extend(["", "## Limitations", ""])
    for limitation in report.limitations:
        lines.append(f"- {limitation}")
    lines.append("")
    return "\n".join(lines)


def _interval(value: Any) -> Interval:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("intervals must contain exactly two indices")
    return int(value[0]), int(value[1])


def _overlap(left: Interval, right: Interval) -> bool:
    return max(left[0], right[0]) < min(left[1], right[1])
