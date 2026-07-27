import numpy as np
import pytest

from quantis_core.fault_lab import (
    FaultLabEvaluationConfig,
    FaultLabManifest,
    evaluate_fault_lab,
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


def test_fault_lab_manifest_produces_independent_phase_and_attribution_truth():
    manifest = FaultLabManifest.from_dict(
        {
            "schema_version": 1,
            "point_count": 12,
            "sample_period_seconds": 0.25,
            "logical_window_period_nano": 1_000_000_000,
            "training_interval": [0, 4],
            "routine_noise_interval": [5, 7],
            "structural_interval": [8, 11],
            "affected_features": ["queue_depth", "worker_rate"],
        }
    )

    np.testing.assert_array_equal(
        manifest.phase_labels(),
        [
            "baseline",
            "baseline",
            "baseline",
            "baseline",
            "validation",
            "routine_noise",
            "routine_noise",
            "recovery",
            "structural",
            "structural",
            "structural",
            "recovery",
        ],
    )
    assert manifest.training_slice == slice(0, 4)
    assert manifest.affected_features == ("queue_depth", "worker_rate")


def test_fault_lab_manifest_rejects_overlapping_or_trained_fault_intervals():
    payload = {
        "schema_version": 1,
        "point_count": 12,
        "sample_period_seconds": 0.25,
        "logical_window_period_nano": 1_000_000_000,
        "training_interval": [0, 6],
        "routine_noise_interval": [5, 7],
        "structural_interval": [8, 11],
        "affected_features": ["queue_depth"],
    }

    with pytest.raises(ValueError, match="training interval must contain baseline"):
        FaultLabManifest.from_dict(payload)

    payload["training_interval"] = [0, 4]
    payload["routine_noise_interval"] = [8, 10]
    payload["structural_interval"] = [5, 7]
    with pytest.raises(ValueError, match="routine noise must precede structural"):
        FaultLabManifest.from_dict(payload)

    payload["training_interval"] = [0, 6]
    payload["routine_noise_interval"] = [8, 10]
    payload["structural_interval"] = [4, 7]
    with pytest.raises(ValueError, match="training interval must contain baseline"):
        FaultLabManifest.from_dict(payload)


def test_fault_lab_evaluation_ignores_isolated_noise_and_detects_stall():
    feature_names = (
        "request_rate",
        "request_latency_ms",
        "queue_depth",
        "worker_rate",
        "worker_heartbeat_age_s",
        "db_write_rate",
    )
    point_count = 80
    load = np.tile(
        np.asarray([-1.0, 0.0, 1.0, 0.0]),
        point_count // 4,
    )
    values = np.column_stack(
        (
            20.0 + load,
            8.0 + 0.4 * load,
            1.0 + 0.2 * load,
            20.0 + load,
            0.05 + 0.005 * load,
            20.0 + load,
        )
    )
    values[44, 1] += 120.0
    structural_progress = np.linspace(0.1, 1.0, 12)
    values[60:72, 2] += 80.0 * structural_progress
    values[60:72, 3] = 0.0
    values[60:72, 4] += 4.0 * structural_progress
    values[60:72, 5] = 0.0
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
            },
            point_attributes={},
            scope_name="quantis.fault-lab",
            scope_version="1",
            number_value=float(values[point_index, feature_index]),
        )
        for point_index in range(point_count)
        for feature_index, feature_name in enumerate(feature_names)
    )
    capture = TelemetryCapture(points, "a" * 64, "memory", 1)
    feature_spec = OtlpFeatureSpec(
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
            for name in feature_names
        ),
    )
    manifest = FaultLabManifest(
        point_count=point_count,
        sample_period_seconds=0.25,
        logical_window_period_nano=1_000_000_000,
        training_interval=(0, 36),
        routine_noise_interval=(44, 45),
        structural_interval=(60, 72),
        affected_features=(
            "queue_depth",
            "worker_rate",
            "worker_heartbeat_age_s",
            "db_write_rate",
        ),
        images={"service": "service@sha256:" + "b" * 64},
    )

    report = evaluate_fault_lab(capture, feature_spec, manifest)

    assert report.acceptance["all_passed"] is True, [
        gate
        for gate, passed in report.acceptance["gates"].items()
        if not passed
    ] + [
        report.detection["routine_noise_alert_rate"],
        report.detection["repaired_isolated_context_cells"],
    ]
    assert report.detection["structural_detected"] is True
    assert report.detection["routine_noise_points"] == 7
    assert report.detection["routine_noise_alert_rate"] <= 0.2
    assert report.detection["detection_delay_windows"] <= 6
    assert report.attribution["hit_at_3"] is True
    assert set(report.attribution["top_features"]) & set(
        manifest.affected_features
    )
    assert report.to_dict()["schema_version"] == 1

    with pytest.raises(ValueError, match="full routine-noise response horizon"):
        evaluate_fault_lab(
            capture,
            feature_spec,
            manifest,
            FaultLabEvaluationConfig(lookback=16),
        )
