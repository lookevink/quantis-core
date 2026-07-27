import json

import numpy as np
import pytest

from quantis_core.otlp import (
    MetricKind,
    MetricPoint,
    TelemetryCapture,
    Temporality,
    read_otlp_capture,
)
from quantis_core.otlp_windowing import (
    FeatureDefinition,
    FeatureStatistic,
    ForwardFillPolicy,
    OtlpFeatureSpec,
    OtlpWindowCompiler,
    OtlpWindowError,
    CompiledTelemetry,
    TelemetryMaterializationError,
    materialize_compiled_telemetry,
)


ONE_SECOND = 1_000_000_000


def semantic_spec():
    return OtlpFeatureSpec(
        window_period_nano=ONE_SECOND,
        features=(
            FeatureDefinition(
                name="temperature",
                metric_name="demo.temperature",
                statistic=FeatureStatistic.GAUGE_LAST,
            ),
            FeatureDefinition(
                name="request_rate",
                metric_name="demo.requests",
                statistic=FeatureStatistic.SUM_RATE,
            ),
            FeatureDefinition(
                name="error_rate",
                metric_name="demo.errors",
                statistic=FeatureStatistic.SUM_RATE,
            ),
            FeatureDefinition(
                name="latency_mean",
                metric_name="demo.latency",
                statistic=FeatureStatistic.HISTOGRAM_MEAN,
            ),
            FeatureDefinition(
                name="latency_count_rate",
                metric_name="demo.latency",
                statistic=FeatureStatistic.HISTOGRAM_COUNT_RATE,
            ),
            FeatureDefinition(
                name="payload_mean",
                metric_name="demo.payload",
                statistic=FeatureStatistic.HISTOGRAM_MEAN,
            ),
        ),
    )


def test_otlp_compiler_applies_temporality_resets_flags_and_missingness():
    capture = read_otlp_capture("tests/fixtures/otlp/semantic-metrics.jsonl")

    compiled = OtlpWindowCompiler(semantic_spec()).compile(capture)

    np.testing.assert_array_equal(
        compiled.window_end_unix_nano,
        np.asarray([ONE_SECOND, 2 * ONE_SECOND, 3 * ONE_SECOND, 4 * ONE_SECOND]),
    )
    assert compiled.feature_names == (
        "temperature",
        "request_rate",
        "error_rate",
        "latency_mean",
        "latency_count_rate",
        "payload_mean",
    )
    np.testing.assert_allclose(
        compiled.values,
        np.asarray(
            [
                [20.0, np.nan, 2.0, 15.0, 2.0, np.nan],
                [21.0, 15.0, 3.0, 25.0, 4.0, 30.0],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [24.0, 5.0, np.nan, np.nan, np.nan, 30.0],
            ]
        ),
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        compiled.reset_mask[:, [1, 5]],
        np.asarray(
            [
                [False, False],
                [False, False],
                [True, True],
                [False, False],
            ]
        ),
    )
    assert compiled.data_quality == {
        "flagged_points": 1,
        "reset_points": 2,
        "missing_cells": 11,
    }
    assert compiled.capture_sha256 == capture.sha256
    assert len(compiled.feature_schema_id) == 64


def test_bounded_forward_fill_is_explicit_and_never_crosses_long_gaps():
    capture = read_otlp_capture("tests/fixtures/otlp/semantic-metrics.jsonl")
    spec = OtlpFeatureSpec(
        window_period_nano=ONE_SECOND,
        features=(
            FeatureDefinition(
                name="temperature",
                metric_name="demo.temperature",
                statistic=FeatureStatistic.GAUGE_LAST,
            ),
        ),
    )
    compiled = OtlpWindowCompiler(spec).compile(capture)

    materialized = materialize_compiled_telemetry(
        compiled, ForwardFillPolicy(max_gap_windows=2)
    )

    np.testing.assert_allclose(materialized.values[:, 0], [20.0, 21.0, 21.0, 24.0])
    np.testing.assert_array_equal(
        materialized.imputed_mask[:, 0], [False, False, True, False]
    )
    with pytest.raises(TelemetryMaterializationError, match="temperature.*window 3"):
        materialize_compiled_telemetry(
            compiled, ForwardFillPolicy(max_gap_windows=0)
        )


def test_versioned_feature_and_compiled_contracts_round_trip_without_nan_json():
    capture = read_otlp_capture("tests/fixtures/otlp/semantic-metrics.jsonl")
    spec = semantic_spec()
    compiled = OtlpWindowCompiler(spec).compile(capture)

    restored_spec = OtlpFeatureSpec.from_dict(spec.to_dict())
    compiled_payload = compiled.to_dict()
    encoded = json.dumps(compiled_payload, allow_nan=False)
    restored_compiled = CompiledTelemetry.from_dict(json.loads(encoded))

    assert spec.to_dict()["schema_version"] == 1
    assert restored_spec.schema_id == spec.schema_id
    assert compiled_payload["schema_version"] == 1
    np.testing.assert_allclose(
        restored_compiled.values, compiled.values, equal_nan=True
    )
    np.testing.assert_array_equal(
        restored_compiled.observed_mask, compiled.observed_mask
    )
    np.testing.assert_array_equal(
        restored_compiled.reset_mask, compiled.reset_mask
    )
    assert restored_compiled.feature_schema_id == compiled.feature_schema_id
    assert restored_compiled.capture_sha256 == compiled.capture_sha256


def test_feature_selection_rejects_multiple_writers_until_resource_is_explicit():
    points = tuple(
        MetricPoint(
            metric_name="demo.temperature",
            unit="Cel",
            kind=MetricKind.GAUGE,
            temporality=Temporality.UNSPECIFIED,
            monotonic=False,
            time_unix_nano=ONE_SECOND,
            start_time_unix_nano=None,
            flags=0,
            resource_attributes={"service.instance.id": instance},
            point_attributes={},
            scope_name="fixture",
            scope_version="1",
            number_value=value,
        )
        for instance, value in (("a", 20.0), ("b", 30.0))
    )
    capture = TelemetryCapture(points, "a" * 64, "memory", 1)
    ambiguous = OtlpFeatureSpec(
        ONE_SECOND,
        (
            FeatureDefinition(
                "temperature",
                "demo.temperature",
                FeatureStatistic.GAUGE_LAST,
            ),
        ),
    )

    with pytest.raises(OtlpWindowError, match="matched 2 metric streams"):
        OtlpWindowCompiler(ambiguous).compile(capture)

    selected = OtlpFeatureSpec(
        ONE_SECOND,
        (
            FeatureDefinition(
                "temperature",
                "demo.temperature",
                FeatureStatistic.GAUGE_LAST,
                resource_attributes={"service.instance.id": "b"},
            ),
        ),
    )
    compiled = OtlpWindowCompiler(selected).compile(capture)
    np.testing.assert_allclose(compiled.values, [[30.0]])


def test_cumulative_rate_differences_int64_before_float_conversion():
    points = tuple(
        MetricPoint(
            metric_name="demo.large_counter",
            unit="{request}",
            kind=MetricKind.SUM,
            temporality=Temporality.CUMULATIVE,
            monotonic=True,
            time_unix_nano=timestamp,
            start_time_unix_nano=0,
            flags=0,
            resource_attributes={},
            point_attributes={},
            scope_name="fixture",
            scope_version="1",
            number_value=value,
        )
        for timestamp, value in (
            (ONE_SECOND, 2**53),
            (2 * ONE_SECOND, 2**53 + 1),
        )
    )
    capture = TelemetryCapture(points, "b" * 64, "memory", 1)
    spec = OtlpFeatureSpec(
        ONE_SECOND,
        (
            FeatureDefinition(
                "request_rate",
                "demo.large_counter",
                FeatureStatistic.SUM_RATE,
            ),
        ),
    )

    compiled = OtlpWindowCompiler(spec).compile(capture)

    np.testing.assert_allclose(compiled.values[:, 0], [np.nan, 1.0], equal_nan=True)


def test_flagged_cumulative_point_still_records_reset_boundary():
    points = tuple(
        MetricPoint(
            metric_name="demo.counter",
            unit="{request}",
            kind=MetricKind.SUM,
            temporality=Temporality.CUMULATIVE,
            monotonic=True,
            time_unix_nano=timestamp,
            start_time_unix_nano=start,
            flags=flags,
            resource_attributes={},
            point_attributes={},
            scope_name="fixture",
            scope_version="1",
            number_value=value,
        )
        for timestamp, start, flags, value in (
            (ONE_SECOND, 0, 0, 10),
            (2 * ONE_SECOND, 0, 0, 20),
            (3 * ONE_SECOND, 2 * ONE_SECOND, 1, 1),
        )
    )
    capture = TelemetryCapture(points, "c" * 64, "memory", 1)
    spec = OtlpFeatureSpec(
        ONE_SECOND,
        (
            FeatureDefinition(
                "request_rate",
                "demo.counter",
                FeatureStatistic.SUM_RATE,
            ),
        ),
    )

    compiled = OtlpWindowCompiler(spec).compile(capture)

    np.testing.assert_array_equal(compiled.reset_mask[:, 0], [False, False, True])
    np.testing.assert_allclose(
        compiled.values[:, 0], [np.nan, 10.0, np.nan], equal_nan=True
    )
    assert compiled.data_quality["flagged_points"] == 1
    assert compiled.data_quality["reset_points"] == 1


def test_cumulative_rate_recovers_across_flagged_point_without_false_reset():
    points = tuple(
        MetricPoint(
            metric_name="demo.counter",
            unit="{request}",
            kind=MetricKind.SUM,
            temporality=Temporality.CUMULATIVE,
            monotonic=True,
            time_unix_nano=timestamp,
            start_time_unix_nano=0,
            flags=flags,
            resource_attributes={},
            point_attributes={},
            scope_name="fixture",
            scope_version="1",
            number_value=value,
        )
        for timestamp, flags, value in (
            (ONE_SECOND, 0, 10),
            (2 * ONE_SECOND, 1, 999),
            (3 * ONE_SECOND, 0, 30),
        )
    )
    capture = TelemetryCapture(points, "d" * 64, "memory", 1)
    spec = OtlpFeatureSpec(
        ONE_SECOND,
        (
            FeatureDefinition(
                "request_rate",
                "demo.counter",
                FeatureStatistic.SUM_RATE,
            ),
        ),
    )

    compiled = OtlpWindowCompiler(spec).compile(capture)

    np.testing.assert_array_equal(compiled.reset_mask[:, 0], [False, False, False])
    np.testing.assert_allclose(
        compiled.values[:, 0], [np.nan, np.nan, 10.0], equal_nan=True
    )


def test_absent_and_explicit_zero_start_times_are_equivalent():
    points = tuple(
        MetricPoint(
            metric_name="demo.counter",
            unit="{request}",
            kind=MetricKind.SUM,
            temporality=Temporality.CUMULATIVE,
            monotonic=True,
            time_unix_nano=timestamp,
            start_time_unix_nano=start,
            flags=0,
            resource_attributes={},
            point_attributes={},
            scope_name="fixture",
            scope_version="1",
            number_value=value,
        )
        for timestamp, start, value in (
            (ONE_SECOND, None, 10),
            (2 * ONE_SECOND, 0, 20),
        )
    )
    capture = TelemetryCapture(points, "e" * 64, "memory", 1)
    spec = OtlpFeatureSpec(
        ONE_SECOND,
        (
            FeatureDefinition(
                "request_rate",
                "demo.counter",
                FeatureStatistic.SUM_RATE,
            ),
        ),
    )

    compiled = OtlpWindowCompiler(spec).compile(capture)

    np.testing.assert_array_equal(compiled.reset_mask[:, 0], [False, False])
    np.testing.assert_allclose(
        compiled.values[:, 0], [np.nan, 10.0], equal_nan=True
    )


def test_histogram_count_rate_does_not_require_optional_sum():
    points = tuple(
        MetricPoint(
            metric_name="demo.payloads",
            unit="{item}",
            kind=MetricKind.HISTOGRAM,
            temporality=Temporality.CUMULATIVE,
            monotonic=False,
            time_unix_nano=timestamp,
            start_time_unix_nano=0,
            flags=0,
            resource_attributes={},
            point_attributes={},
            scope_name="fixture",
            scope_version="1",
            histogram_count=count,
        )
        for timestamp, count in (
            (ONE_SECOND, 10),
            (2 * ONE_SECOND, 15),
        )
    )
    capture = TelemetryCapture(points, "f" * 64, "memory", 1)
    spec = OtlpFeatureSpec(
        ONE_SECOND,
        (
            FeatureDefinition(
                "payload_count_rate",
                "demo.payloads",
                FeatureStatistic.HISTOGRAM_COUNT_RATE,
            ),
        ),
    )

    compiled = OtlpWindowCompiler(spec).compile(capture)

    np.testing.assert_allclose(
        compiled.values[:, 0], [np.nan, 5.0], equal_nan=True
    )


def test_delta_histogram_mean_does_not_require_interval_duration():
    point = MetricPoint(
        metric_name="demo.latency",
        unit="ms",
        kind=MetricKind.HISTOGRAM,
        temporality=Temporality.DELTA,
        monotonic=False,
        time_unix_nano=ONE_SECOND,
        start_time_unix_nano=None,
        flags=0,
        resource_attributes={},
        point_attributes={},
        scope_name="fixture",
        scope_version="1",
        histogram_count=2,
        histogram_sum=30.0,
    )
    capture = TelemetryCapture((point,), "0" * 64, "memory", 1)
    spec = OtlpFeatureSpec(
        ONE_SECOND,
        (
            FeatureDefinition(
                "latency_mean",
                "demo.latency",
                FeatureStatistic.HISTOGRAM_MEAN,
            ),
        ),
    )

    compiled = OtlpWindowCompiler(spec).compile(capture)

    np.testing.assert_allclose(compiled.values[:, 0], [15.0])
