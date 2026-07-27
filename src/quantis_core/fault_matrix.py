"""Held-out, frozen-model evaluation across instrumented fault classes."""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .detectors import detector_from_dict
from .otlp import TelemetryCapture
from .otlp_windowing import OtlpFeatureSpec, OtlpWindowCompiler
from .windowing import (
    WindowCompiler,
    repair_isolated_context_outliers,
)


Interval = Tuple[int, int]
FAULT_KINDS = frozenset(
    {"worker_crash", "database_lock", "cache_outage"}
)


@dataclass(frozen=True)
class FaultMatrixCaseManifest:
    """Predeclared schedule, mechanism, and attribution truth for one case."""

    case_id: str
    fault_kind: str
    point_count: int
    sample_period_seconds: float
    logical_window_period_nano: int
    baseline_interval: Interval
    routine_noise_interval: Interval
    structural_interval: Interval
    affected_features: Tuple[str, ...]
    requests_per_window: int = 1
    routine_noise_delay_ms: int = 0
    load_pattern_offsets: Tuple[int, ...] = (0,)
    images: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("case_id cannot be empty")
        if self.fault_kind not in FAULT_KINDS:
            raise ValueError(f"unsupported fault kind: {self.fault_kind}")
        if self.point_count < 1:
            raise ValueError("point_count must be positive")
        if self.sample_period_seconds <= 0.0:
            raise ValueError("sample_period_seconds must be positive")
        if self.logical_window_period_nano <= 0:
            raise ValueError("logical_window_period_nano must be positive")
        for name, interval in (
            ("baseline", self.baseline_interval),
            ("routine noise", self.routine_noise_interval),
            ("structural", self.structural_interval),
        ):
            start, stop = interval
            if not 0 <= start < stop <= self.point_count:
                raise ValueError(f"{name} interval is outside the experiment")
        if self.baseline_interval[0] != 0:
            raise ValueError("baseline interval must begin at zero")
        if self.baseline_interval[1] > self.routine_noise_interval[0]:
            raise ValueError("baseline interval must precede routine noise")
        if self.routine_noise_interval[1] > self.structural_interval[0]:
            raise ValueError("routine noise must precede structural fault")
        if not self.affected_features:
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
            raise ValueError("fault-matrix images must be digest pinned")

    @property
    def baseline_slice(self) -> slice:
        return slice(*self.baseline_interval)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "case_id": self.case_id,
            "fault_kind": self.fault_kind,
            "point_count": self.point_count,
            "sample_period_seconds": self.sample_period_seconds,
            "logical_window_period_nano": self.logical_window_period_nano,
            "baseline_interval": list(self.baseline_interval),
            "routine_noise_interval": list(self.routine_noise_interval),
            "structural_interval": list(self.structural_interval),
            "affected_features": list(self.affected_features),
            "requests_per_window": self.requests_per_window,
            "routine_noise_delay_ms": self.routine_noise_delay_ms,
            "load_pattern_offsets": list(self.load_pattern_offsets),
            "images": dict(self.images),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "FaultMatrixCaseManifest":
        if payload.get("schema_version") != 1:
            raise ValueError(
                "unsupported FaultMatrixCaseManifest schema_version"
            )
        return cls(
            case_id=str(payload["case_id"]),
            fault_kind=str(payload["fault_kind"]),
            point_count=int(payload["point_count"]),
            sample_period_seconds=float(payload["sample_period_seconds"]),
            logical_window_period_nano=int(
                payload["logical_window_period_nano"]
            ),
            baseline_interval=_interval(payload["baseline_interval"]),
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
class FaultMatrixRun:
    """One static manifest paired with its raw Collector capture."""

    manifest: FaultMatrixCaseManifest
    capture: TelemetryCapture


@dataclass(frozen=True)
class FaultMatrixEvaluationConfig:
    """Acceptance limits fixed before held-out captures are observed."""

    maximum_detection_delay_windows: int = 6
    maximum_noise_alert_rate: float = 0.2
    maximum_pre_noise_alert_rate: float = 0.2
    minimum_backlog_growth: float = 20.0
    maximum_fault_rate_ratio: float = 0.2
    minimum_noise_latency_ratio: float = 3.0
    minimum_cache_error_rate: float = 0.8
    isolated_context_z_threshold: float = 8.0


FROZEN_EVALUATION_CONFIG = FaultMatrixEvaluationConfig()


@dataclass(frozen=True)
class FaultMatrixReport:
    """Versioned case evidence plus aggregate acceptance decisions."""

    protocol: Mapping[str, Any]
    cases: Mapping[str, Mapping[str, Any]]
    aggregate: Mapping[str, Any]
    acceptance: Mapping[str, Any]
    limitations: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "protocol": dict(self.protocol),
            "cases": {
                key: dict(value) for key, value in self.cases.items()
            },
            "aggregate": dict(self.aggregate),
            "acceptance": dict(self.acceptance),
            "limitations": list(self.limitations),
        }


def evaluate_fault_matrix(
    runs: Sequence[FaultMatrixRun],
    feature_spec: OtlpFeatureSpec,
    window_compiler_artifact_bytes: bytes,
    detector_artifact_bytes: bytes,
) -> FaultMatrixReport:
    """Score held-out captures using restored, immutable fitted artifacts."""

    config = FROZEN_EVALUATION_CONFIG
    if not runs:
        raise ValueError("fault matrix requires at least one run")
    window_compiler_artifact = _decode_artifact(
        window_compiler_artifact_bytes, "window compiler"
    )
    detector_artifact = _decode_artifact(
        detector_artifact_bytes, "detector"
    )
    artifact_file_sha256 = {
        "window_compiler": hashlib.sha256(
            window_compiler_artifact_bytes
        ).hexdigest(),
        "detector": hashlib.sha256(
            detector_artifact_bytes
        ).hexdigest(),
    }
    source_compiler_sha256 = _canonical_sha256(
        window_compiler_artifact
    )
    source_detector_sha256 = _canonical_sha256(detector_artifact)
    compiler_payload = copy_mapping(window_compiler_artifact)
    detector_payload = copy_mapping(detector_artifact)
    compiler_sha256 = _canonical_sha256(compiler_payload)
    detector_sha256 = _canonical_sha256(detector_payload)
    config_payload = dict(vars(config))
    case_reports: Dict[str, Mapping[str, Any]] = {}
    for run in runs:
        if run.manifest.case_id in case_reports:
            raise ValueError(
                f"duplicate fault-matrix case_id: {run.manifest.case_id}"
            )
        case_reports[run.manifest.case_id] = _evaluate_case(
            run,
            feature_spec,
            compiler_payload,
            detector_payload,
            config,
        )
    if (
        compiler_payload != window_compiler_artifact
        or source_compiler_sha256
        != _canonical_sha256(window_compiler_artifact)
    ):
        raise RuntimeError("window compiler artifact changed during evaluation")
    if (
        detector_payload != detector_artifact
        or source_detector_sha256
        != _canonical_sha256(detector_artifact)
    ):
        raise RuntimeError("detector artifact changed during evaluation")

    ordered_cases = [case_reports[key] for key in sorted(case_reports)]
    event_count = len(ordered_cases)
    detected_count = sum(
        bool(case["detection"]["structural_detected"])
        for case in ordered_cases
    )
    attribution_hits = sum(
        bool(case["attribution"]["hit_at_3"])
        for case in ordered_cases
    )
    noise_alerts = sum(
        int(case["detection"]["routine_noise_alerts"])
        for case in ordered_cases
    )
    noise_points = sum(
        int(case["detection"]["routine_noise_points"])
        for case in ordered_cases
    )
    pre_noise_alerts = sum(
        int(case["detection"]["pre_noise_alerts"])
        for case in ordered_cases
    )
    pre_noise_points = sum(
        int(case["detection"]["pre_noise_points"])
        for case in ordered_cases
    )
    noise_rate = noise_alerts / noise_points if noise_points else 0.0
    pre_noise_rate = (
        pre_noise_alerts / pre_noise_points if pre_noise_points else 0.0
    )
    delays = [
        int(case["detection"]["detection_delay_windows"])
        for case in ordered_cases
        if case["detection"]["detection_delay_windows"] is not None
    ]
    aggregate = {
        "structural_events": event_count,
        "structural_events_detected": detected_count,
        "structural_event_recall": detected_count / event_count,
        "attribution_hits_at_3": attribution_hits,
        "attribution_hit_rate_at_3": attribution_hits / event_count,
        "maximum_detection_delay_windows": max(delays) if delays else None,
        "routine_noise_alerts": noise_alerts,
        "routine_noise_points": noise_points,
        "routine_noise_alert_rate": noise_rate,
        "pre_noise_alerts": pre_noise_alerts,
        "pre_noise_points": pre_noise_points,
        "pre_noise_alert_rate": pre_noise_rate,
    }
    observed_kinds = {
        str(case["manifest"]["fault_kind"]) for case in ordered_cases
    }
    application_image_ids = {
        str(case["capture"]["application_image_id"])
        for case in ordered_cases
    }
    application_build_hashes = {
        str(case["capture"]["application_build_context_sha256"])
        for case in ordered_cases
    }
    gates = {
        "complete_fault_kind_coverage": (
            observed_kinds == FAULT_KINDS
            and event_count == len(FAULT_KINDS)
        ),
        "frozen_artifacts_unchanged": (
            compiler_sha256 == _canonical_sha256(compiler_payload)
            and detector_sha256 == _canonical_sha256(detector_payload)
            and source_compiler_sha256
            == _canonical_sha256(window_compiler_artifact)
            and source_detector_sha256
            == _canonical_sha256(detector_artifact)
        ),
        "one_application_image_and_build": (
            len(application_image_ids) == 1
            and len(application_build_hashes) == 1
        ),
        "all_raw_fault_effects_observed": all(
            bool(case["acceptance"]["raw_effects_observed"])
            for case in ordered_cases
        ),
        "all_captures_match_manifests": all(
            bool(case["acceptance"]["capture_matches_manifest"])
            for case in ordered_cases
        ),
        "structural_event_recall_is_one": detected_count == event_count,
        "all_detection_delays_within_limit": all(
            delay <= config.maximum_detection_delay_windows
            for delay in delays
        )
        and len(delays) == event_count,
        "aggregate_routine_noise_alert_rate_within_limit": (
            noise_points > 0
            and noise_rate <= config.maximum_noise_alert_rate
        ),
        "aggregate_pre_noise_alert_rate_within_limit": (
            pre_noise_points > 0
            and pre_noise_rate <= config.maximum_pre_noise_alert_rate
        ),
        "attribution_hit_rate_at_3_is_one": (
            attribution_hits == event_count
        ),
        "content_addressed_inputs": all(
            bool(case["acceptance"]["content_addressed_inputs"])
            for case in ordered_cases
        )
        and _is_sha256_hex(compiler_sha256)
        and _is_sha256_hex(detector_sha256),
    }
    return FaultMatrixReport(
        protocol={
            "evaluation_kind": "held_out_frozen_model_fault_matrix",
            "model_fit_calls": 0,
            "window_compiler_sha256": compiler_sha256,
            "detector_sha256": detector_sha256,
            "artifact_file_sha256": dict(artifact_file_sha256),
            "feature_schema_id": feature_spec.schema_id,
            "case_fault_kinds": sorted(observed_kinds),
            "config": config_payload,
            "evaluator_config_sha256": _canonical_sha256(config_payload),
        },
        cases={
            key: case_reports[key] for key in sorted(case_reports)
        },
        aggregate=aggregate,
        acceptance={
            "all_passed": all(gates.values()),
            "gates": gates,
        },
        limitations=(
            "Three local cases are not representative of production fault diversity.",
            "The topology and telemetry vocabulary remain the same as development.",
            "The cache fault is a logical application-path outage, not a killed "
            "Redis process, because Redis also carries this lab's public counters.",
            "Load schedules and fault mechanisms are held out from fitting, but "
            "were authored by the same development team.",
            "Logical event-time windows are sampled faster than wall clock.",
            "Feature evidence is associative attribution, not causal proof.",
            "The target encoder is linear PCA, not a learned JEPA encoder.",
        ),
    )


def write_fault_matrix_artifacts(
    report: FaultMatrixReport, output_directory: Path
) -> Mapping[str, Path]:
    """Write machine-readable and concise human-readable matrix evidence."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "verification": output / "verification.json",
        "report": output / "report.md",
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
    return paths


def _evaluate_case(
    run: FaultMatrixRun,
    feature_spec: OtlpFeatureSpec,
    compiler_payload: Mapping[str, Any],
    detector_payload: Mapping[str, Any],
    config: FaultMatrixEvaluationConfig,
) -> Mapping[str, Any]:
    manifest = run.manifest
    capture = run.capture
    lookback = int(compiler_payload["lookback"])
    if (
        manifest.routine_noise_interval[1] + lookback
        > manifest.structural_interval[0]
    ):
        raise ValueError(
            "manifest must preserve the full routine-noise response horizon"
        )
    if feature_spec.window_period_nano != manifest.logical_window_period_nano:
        raise ValueError("feature spec and manifest window periods differ")
    compiled = OtlpWindowCompiler(feature_spec).compile(capture)
    if len(compiled.window_end_unix_nano) != manifest.point_count:
        raise ValueError(
            f"{manifest.case_id} compiled point count does not match manifest"
        )
    if compiled.data_quality["missing_cells"] != 0:
        raise ValueError(
            f"{manifest.case_id} evaluation requires complete telemetry"
        )
    missing_affected = set(manifest.affected_features) - set(
        compiled.feature_names
    )
    if missing_affected:
        raise ValueError(
            f"affected features are absent from telemetry: "
            f"{sorted(missing_affected)}"
        )
    compiler = WindowCompiler.from_dict(dict(compiler_payload))
    detector = detector_from_dict(dict(detector_payload))
    windows = compiler.transform(
        compiled.values, compiled.feature_names
    )
    robust_windows, repaired_cells = repair_isolated_context_outliers(
        windows,
        z_threshold=config.isolated_context_z_threshold,
        consensus_rank=int(detector_payload["consensus_rank"]),
    )
    scores = detector.score(robust_windows)
    indices = windows.point_indices
    structural_mask = (
        (indices >= manifest.structural_interval[0])
        & (indices < manifest.structural_interval[1])
    )
    noise_stop = manifest.routine_noise_interval[1] + lookback
    noise_mask = (
        (indices >= manifest.routine_noise_interval[0])
        & (indices < noise_stop)
    )
    pre_noise_mask = indices < manifest.routine_noise_interval[0]
    alert_positions = np.flatnonzero(scores.alerts & structural_mask)
    detected = len(alert_positions) > 0
    first_point: Optional[int] = None
    delay: Optional[int] = None
    top_features: Tuple[str, ...] = ()
    if detected:
        position = int(alert_positions[0])
        first_point = int(indices[position])
        delay = first_point - manifest.structural_interval[0]
        top_indices = np.argsort(
            scores.feature_evidence[position]
        )[-3:][::-1]
        top_features = tuple(
            windows.feature_names[int(index)] for index in top_indices
        )
    attribution_hit = bool(
        set(top_features) & set(manifest.affected_features)
    )
    noise_points = int(np.count_nonzero(noise_mask))
    noise_alerts = int(np.count_nonzero(scores.alerts & noise_mask))
    pre_noise_points = int(np.count_nonzero(pre_noise_mask))
    pre_noise_alerts = int(
        np.count_nonzero(scores.alerts & pre_noise_mask)
    )
    pre_noise_feature_evidence_median = np.median(
        scores.feature_evidence[pre_noise_mask], axis=0
    )
    raw_effects = _raw_effects(
        compiled.values, compiled.feature_names, manifest
    )
    raw_gates = _raw_effect_gates(raw_effects, manifest.fault_kind, config)
    application_image_ids = {
        point.resource_attributes.get("quantis.application.image.id")
        for point in capture.points
    }
    application_build_hashes = {
        point.resource_attributes.get(
            "quantis.application.build_context.sha256"
        )
        for point in capture.points
    }
    application_image_id = (
        str(next(iter(application_image_ids)))
        if len(application_image_ids) == 1
        else ""
    )
    application_build_hash = (
        str(next(iter(application_build_hashes)))
        if len(application_build_hashes) == 1
        else ""
    )
    manifest_sha256 = _canonical_sha256(manifest.to_dict())
    case_ids = {
        point.resource_attributes.get("quantis.experiment.case.id")
        for point in capture.points
    }
    fault_kinds = {
        point.resource_attributes.get("quantis.experiment.fault.kind")
        for point in capture.points
    }
    manifest_hashes = {
        point.resource_attributes.get(
            "quantis.experiment.manifest.sha256"
        )
        for point in capture.points
    }
    capture_matches_manifest = (
        case_ids == {manifest.case_id}
        and fault_kinds == {manifest.fault_kind}
        and manifest_hashes == {manifest_sha256}
    )
    content_addressed = (
        _is_sha256_hex(capture.sha256)
        and _is_sha256_hex(compiled.feature_schema_id)
        and _is_sha256_hex(manifest_sha256)
        and bool(manifest.images)
        and all("@sha256:" in image for image in manifest.images.values())
        and _is_sha256_identifier(application_image_id)
        and _is_sha256_hex(application_build_hash)
        and capture_matches_manifest
    )
    return {
        "manifest": manifest.to_dict(),
        "manifest_sha256": manifest_sha256,
        "capture": {
            "sha256": capture.sha256,
            "json_message_count": capture.json_message_count,
            "metric_point_count": len(capture.points),
            "application_image_id": application_image_id,
            "application_build_context_sha256": application_build_hash,
        },
        "compiled": {
            "window_count": len(compiled.window_end_unix_nano),
            "feature_count": len(compiled.feature_names),
            "feature_names": list(compiled.feature_names),
            "data_quality": dict(compiled.data_quality),
        },
        "raw_effects": raw_effects,
        "raw_effect_gates": raw_gates,
        "detection": {
            "threshold": float(scores.threshold),
            "structural_detected": detected,
            "first_detection_point": first_point,
            "detection_delay_windows": delay,
            "detection_latency_wall_seconds_upper_bound": (
                (delay + 1) * manifest.sample_period_seconds
                if delay is not None
                else None
            ),
            "routine_noise_alerts": noise_alerts,
            "routine_noise_points": noise_points,
            "routine_noise_alert_rate": (
                noise_alerts / noise_points if noise_points else 0.0
            ),
            "pre_noise_alerts": pre_noise_alerts,
            "pre_noise_points": pre_noise_points,
            "pre_noise_alert_rate": (
                pre_noise_alerts / pre_noise_points
                if pre_noise_points
                else 0.0
            ),
            "pre_noise_score_median": float(
                np.median(scores.scores[pre_noise_mask])
            ),
            "pre_noise_score_maximum": float(
                np.max(scores.scores[pre_noise_mask])
            ),
            "pre_noise_feature_evidence_median": {
                name: float(pre_noise_feature_evidence_median[index])
                for index, name in enumerate(windows.feature_names)
            },
            "repaired_isolated_context_cells": repaired_cells,
        },
        "attribution": {
            "expected_features": list(manifest.affected_features),
            "top_features": list(top_features),
            "hit_at_3": attribution_hit,
        },
        "acceptance": {
            "raw_effects_observed": all(raw_gates.values()),
            "content_addressed_inputs": content_addressed,
            "capture_matches_manifest": capture_matches_manifest,
        },
    }


def _raw_effects(
    values: NDArray[np.float64],
    feature_names: Tuple[str, ...],
    manifest: FaultMatrixCaseManifest,
) -> Dict[str, float]:
    feature_index = {
        name: index for index, name in enumerate(feature_names)
    }
    required = {
        "request_latency_ms",
        "error_rate",
        "queue_depth",
        "worker_rate",
        "db_write_rate",
    }
    missing = required - set(feature_index)
    if missing:
        raise ValueError(
            f"raw fault-effect features are missing: {sorted(missing)}"
        )
    baseline = values[manifest.baseline_slice]
    structural = values[slice(*manifest.structural_interval)]
    noise = values[slice(*manifest.routine_noise_interval)]

    def median(data: NDArray[np.float64], name: str) -> float:
        return float(np.median(data[:, feature_index[name]]))

    baseline_queue = median(baseline, "queue_depth")
    fault_queue_max = float(
        np.max(structural[:, feature_index["queue_depth"]])
    )
    baseline_worker = median(baseline, "worker_rate")
    baseline_db = median(baseline, "db_write_rate")
    baseline_latency = median(baseline, "request_latency_ms")
    return {
        "baseline_queue_depth_median": baseline_queue,
        "fault_queue_depth_max": fault_queue_max,
        "queue_depth_growth": fault_queue_max - baseline_queue,
        "worker_rate_fault_ratio": _ratio(
            median(structural, "worker_rate"), baseline_worker
        ),
        "db_write_rate_fault_ratio": _ratio(
            median(structural, "db_write_rate"), baseline_db
        ),
        "fault_error_rate_median": median(structural, "error_rate"),
        "routine_noise_latency_ratio": _ratio(
            median(noise, "request_latency_ms"), baseline_latency
        ),
    }


def _raw_effect_gates(
    effects: Mapping[str, float],
    fault_kind: str,
    config: FaultMatrixEvaluationConfig,
) -> Dict[str, bool]:
    gates = {
        "routine_noise_has_observed_effect": (
            effects["routine_noise_latency_ratio"]
            >= config.minimum_noise_latency_ratio
        ),
        "worker_rate_collapses": (
            effects["worker_rate_fault_ratio"]
            <= config.maximum_fault_rate_ratio
        ),
        "db_write_rate_collapses": (
            effects["db_write_rate_fault_ratio"]
            <= config.maximum_fault_rate_ratio
        ),
    }
    if fault_kind in {"worker_crash", "database_lock"}:
        gates["backlog_growth_at_least_minimum"] = (
            effects["queue_depth_growth"]
            >= config.minimum_backlog_growth
        )
    if fault_kind == "cache_outage":
        gates["cache_error_rate_at_least_minimum"] = (
            effects["fault_error_rate_median"]
            >= config.minimum_cache_error_rate
        )
    return gates


def _markdown_report(report: FaultMatrixReport) -> str:
    aggregate = report.aggregate
    status = "PASS" if report.acceptance["all_passed"] else "FAIL"
    lines = [
        "# Quantis held-out fault-matrix verification",
        "",
        f"Overall acceptance: **{status}**",
        "",
        "## Aggregate evidence",
        "",
        f"- Structural event recall: "
        f"{aggregate['structural_events_detected']}/"
        f"{aggregate['structural_events']}",
        f"- Attribution hit@3: {aggregate['attribution_hits_at_3']}/"
        f"{aggregate['structural_events']}",
        f"- Maximum detection delay: "
        f"{aggregate['maximum_detection_delay_windows']} logical windows",
        f"- Routine-noise response alerts: "
        f"{aggregate['routine_noise_alerts']}/"
        f"{aggregate['routine_noise_points']}",
        f"- Pre-noise held-out alerts: {aggregate['pre_noise_alerts']}/"
        f"{aggregate['pre_noise_points']}",
        "",
        "## Cases",
        "",
    ]
    for case_id, case in report.cases.items():
        detection = case["detection"]
        attribution = case["attribution"]
        lines.extend(
            [
                f"### {case_id}",
                "",
                f"- Fault kind: `{case['manifest']['fault_kind']}`",
                f"- Detected: {detection['structural_detected']}",
                f"- Detection delay: "
                f"{detection['detection_delay_windows']} logical windows",
                f"- Attribution top three: "
                f"{', '.join(attribution['top_features'])}",
                f"- Pre-noise score median / threshold: "
                f"{detection['pre_noise_score_median']:.3f} / "
                f"{detection['threshold']:.3f}",
                f"- Raw-effect gates passed: "
                f"{case['acceptance']['raw_effects_observed']}",
                "",
            ]
        )
    lines.extend(["## Acceptance gates", ""])
    for gate, passed in report.acceptance["gates"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: `{gate}`")
    if not report.acceptance["all_passed"]:
        lines.extend(
            [
                "",
                "## Diagnostic interpretation",
                "",
                "The frozen predictor rejects most normal held-out windows. "
                "Median pre-noise evidence is jointly elevated for request, "
                "worker, and database rates, which move together with the "
                "new load schedules. This is evidence of schedule-pattern "
                "overfitting in the development model, not a threshold-only "
                "problem. Structural recall is therefore not operationally "
                "useful at the observed false-positive rate.",
            ]
        )
    lines.extend(["", "## Limitations", ""])
    for limitation in report.limitations:
        lines.append(f"- {limitation}")
    lines.append("")
    return "\n".join(lines)


def copy_mapping(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Round-trip an artifact to an isolated JSON-compatible mapping."""

    return dict(json.loads(json.dumps(payload, allow_nan=False)))


def _decode_artifact(raw: bytes, name: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} artifact is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{name} artifact must be a JSON object")
    return payload


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _is_sha256_identifier(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    return _is_sha256_hex(value.removeprefix("sha256:"))


def _is_sha256_hex(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        raise ValueError("baseline rate must be positive")
    return numerator / denominator


def _interval(value: Any) -> Interval:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("intervals must contain exactly two indices")
    return int(value[0]), int(value[1])
