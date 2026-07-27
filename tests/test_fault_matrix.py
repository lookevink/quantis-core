import copy
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

from quantis_core.detectors import CoherentLatentPredictiveDetector
from quantis_core.fault_matrix import (
    FaultMatrixCaseManifest,
    FaultMatrixReport,
    FaultMatrixRun,
    evaluate_fault_matrix,
    write_fault_matrix_artifacts,
)
from quantis_core.otlp import (
    MetricKind,
    MetricPoint,
    TelemetryCapture,
    Temporality,
)
from quantis_core.otlp_windowing import (
    FeatureDefinition,
    FeatureStatistic,
    OtlpFeatureSpec,
)
from quantis_core.windowing import WindowCompiler


FEATURE_NAMES = (
    "request_rate",
    "request_latency_ms",
    "error_rate",
    "queue_depth",
    "worker_rate",
    "worker_heartbeat_age_s",
    "db_write_rate",
)


def test_fault_matrix_scores_three_fault_kinds_without_refitting(monkeypatch):
    training = _baseline_values(44)
    compiler = WindowCompiler(6).fit(training)
    detector = CoherentLatentPredictiveDetector(
        latent_dimension=1,
        ridge=0.01,
        calibration_quantile=0.98,
        consensus_rank=3,
    ).fit(compiler.transform(training, FEATURE_NAMES))
    detector.threshold *= 2.0
    compiler_artifact = copy.deepcopy(compiler.to_dict())
    detector_artifact = copy.deepcopy(detector.to_dict())
    runs = tuple(_run(fault_kind) for fault_kind in (
        "worker_crash",
        "database_lock",
        "cache_outage",
    ))

    def forbid_fit(*args, **kwargs):
        raise AssertionError("held-out evaluation must not fit model state")

    monkeypatch.setattr(WindowCompiler, "fit", forbid_fit)
    monkeypatch.setattr(CoherentLatentPredictiveDetector, "fit", forbid_fit)

    report = evaluate_fault_matrix(
        runs,
        _feature_spec(),
        json.dumps(compiler_artifact).encode("utf-8"),
        json.dumps(detector_artifact).encode("utf-8"),
    )

    assert report.acceptance["all_passed"] is True, report.acceptance
    assert report.aggregate["structural_events"] == 3
    assert report.aggregate["structural_events_detected"] == 3
    assert report.aggregate["structural_event_recall"] == 1.0
    assert report.aggregate["attribution_hits_at_3"] == 3
    assert report.aggregate["routine_noise_alerts"] == 0
    assert report.aggregate["routine_noise_points"] == 21
    assert report.aggregate["pre_noise_alerts"] == 0
    assert report.aggregate["pre_noise_points"] == 114
    assert report.protocol["model_fit_calls"] == 0
    assert report.protocol["case_fault_kinds"] == [
        "cache_outage",
        "database_lock",
        "worker_crash",
    ]
    assert compiler_artifact == compiler.to_dict()
    assert detector_artifact == detector.to_dict()


def test_fault_matrix_rejects_capture_manifest_swaps_and_has_no_tuning_seam():
    training = _baseline_values(44)
    compiler = WindowCompiler(6).fit(training)
    detector = CoherentLatentPredictiveDetector(
        latent_dimension=1,
        ridge=0.01,
        calibration_quantile=0.98,
        consensus_rank=3,
    ).fit(compiler.transform(training, FEATURE_NAMES))
    detector.threshold *= 2.0
    worker = _run("worker_crash")
    database = _run("database_lock")
    cache = _run("cache_outage")
    swapped = (
        FaultMatrixRun(worker.manifest, database.capture),
        FaultMatrixRun(database.manifest, worker.capture),
        cache,
    )

    report = evaluate_fault_matrix(
        swapped,
        _feature_spec(),
        json.dumps(compiler.to_dict()).encode("utf-8"),
        json.dumps(detector.to_dict()).encode("utf-8"),
    )

    assert (
        report.acceptance["gates"]["all_captures_match_manifests"]
        is False
    )
    assert report.acceptance["all_passed"] is False
    assert "config" not in inspect.signature(
        evaluate_fault_matrix
    ).parameters


def test_report_does_not_mislabel_non_false_positive_failure(
    tmp_path: Path,
) -> None:
    report = FaultMatrixReport(
        protocol={
            "confirmation_status": "preregistered_held_out_confirmation",
            "config": {
                "maximum_noise_alert_rate": 0.2,
                "maximum_pre_noise_alert_rate": 0.2,
            },
            "confirmation_protocol": {
                "required_topologies": {
                    "workers-1": 1,
                    "workers-2": 2,
                }
            },
        },
        cases={},
        aggregate={
            "structural_events_detected": 1,
            "structural_events": 2,
            "attribution_hits_at_3": 2,
            "maximum_detection_delay_windows": 0,
            "routine_noise_alerts": 0,
            "routine_noise_points": 2,
            "pre_noise_alerts": 0,
            "pre_noise_points": 2,
            "topology_strata": {
                "workers-1": {
                    "structural_events_detected": 0,
                    "structural_events": 1,
                    "attribution_hits_at_3": 1,
                    "pre_noise_alerts": 0,
                    "pre_noise_points": 1,
                    "routine_noise_alerts": 0,
                    "routine_noise_points": 1,
                    "pre_noise_alert_rate": 0.0,
                    "routine_noise_alert_rate": 0.0,
                },
                "workers-2": {
                    "structural_events_detected": 1,
                    "structural_events": 1,
                    "attribution_hits_at_3": 1,
                    "pre_noise_alerts": 0,
                    "pre_noise_points": 1,
                    "routine_noise_alerts": 0,
                    "routine_noise_points": 1,
                    "pre_noise_alert_rate": 0.0,
                    "routine_noise_alert_rate": 0.0,
                },
            },
        },
        acceptance={
            "all_passed": False,
            "gates": {
                "structural_event_recall_is_one": False,
                "aggregate_routine_noise_alert_rate_within_limit": True,
                "aggregate_pre_noise_alert_rate_within_limit": True,
                "all_topology_strata_within_limits": False,
            },
        },
        limitations=(),
    )

    paths = write_fault_matrix_artifacts(report, tmp_path)
    markdown = paths["report"].read_text()

    assert "unrelated to normal-alert rate" in markdown
    assert "association with multi-worker operation" not in markdown


def _run(fault_kind: str) -> FaultMatrixRun:
    values = _baseline_values(80)
    values[44, 1] += 120.0
    progress = np.linspace(0.1, 1.0, 12)
    if fault_kind in {"worker_crash", "database_lock"}:
        values[60:72, 3] += 80.0 * progress
        values[60:72, 4] = 0.0
        values[60:72, 5] += 4.0 * progress
        values[60:72, 6] = 0.0
        affected = (
            "queue_depth",
            "worker_rate",
            "worker_heartbeat_age_s",
            "db_write_rate",
        )
    else:
        values[60:72, 1] += 30.0
        values[60:72, 2] = 1.0
        values[60:72, 4] = 0.0
        values[60:72, 6] = 0.0
        affected = (
            "request_latency_ms",
            "error_rate",
            "worker_rate",
            "db_write_rate",
        )
    manifest = FaultMatrixCaseManifest(
        case_id=f"{fault_kind}-held-out-01",
        fault_kind=fault_kind,
        point_count=80,
        sample_period_seconds=0.25,
        logical_window_period_nano=1_000_000_000,
        baseline_interval=(0, 36),
        routine_noise_interval=(44, 45),
        structural_interval=(60, 72),
        affected_features=affected,
        requests_per_window=6,
        load_pattern_offsets=(-1, 0, 1, 0),
        routine_noise_delay_ms=120,
        images={"service": "service@sha256:" + "b" * 64},
    )
    manifest_sha256 = hashlib.sha256(
        json.dumps(
            manifest.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    points = tuple(
        MetricPoint(
            metric_name=feature_name,
            unit="1",
            kind=MetricKind.GAUGE,
            temporality=Temporality.UNSPECIFIED,
            monotonic=False,
            time_unix_nano=(point_index + 1) * 1_000_000_000,
            start_time_unix_nano=None,
            flags=0,
            resource_attributes={
                "service.name": "quantis-fault-lab",
                "quantis.application.image.id": "sha256:" + "c" * 64,
                "quantis.application.build_context.sha256": "d" * 64,
                "quantis.experiment.case.id": manifest.case_id,
                "quantis.experiment.fault.kind": manifest.fault_kind,
                "quantis.experiment.manifest.sha256": manifest_sha256,
            },
            point_attributes={},
            scope_name="quantis.fault-matrix",
            scope_version="1",
            number_value=float(values[point_index, feature_index]),
        )
        for point_index in range(len(values))
        for feature_index, feature_name in enumerate(FEATURE_NAMES)
    )
    capture_digest_character = {
        "worker_crash": "a",
        "database_lock": "b",
        "cache_outage": "c",
    }[fault_kind]
    return FaultMatrixRun(
        manifest=manifest,
        capture=TelemetryCapture(
            points,
            capture_digest_character * 64,
            f"memory:{fault_kind}",
            1,
        ),
    )


def _baseline_values(point_count: int) -> np.ndarray:
    load = np.resize(np.asarray([-1.0, 0.0, 1.0, 0.0]), point_count)
    return np.column_stack(
        (
            20.0 + load,
            8.0 + 0.4 * load,
            np.zeros(point_count),
            1.0 + 0.2 * load,
            20.0 + load,
            0.05 + 0.005 * load,
            20.0 + load,
        )
    )


def _feature_spec() -> OtlpFeatureSpec:
    return OtlpFeatureSpec(
        window_period_nano=1_000_000_000,
        features=tuple(
            FeatureDefinition(
                name=name,
                metric_name=name,
                statistic=FeatureStatistic.GAUGE_LAST,
                resource_attributes={
                    "service.name": "quantis-fault-lab"
                },
            )
            for name in FEATURE_NAMES
        ),
    )
