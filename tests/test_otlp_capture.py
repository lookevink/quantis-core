import json

import pytest

from quantis_core.otlp import (
    MetricKind,
    OtlpCaptureError,
    Temporality,
    read_otlp_capture,
)


def test_capture_reader_preserves_otlp_metric_semantics_and_identity():
    capture = read_otlp_capture("tests/fixtures/otlp/basic-metrics.jsonl")

    assert capture.sha256 == (
        "6d8754adbc07271558d4b537547ae5f289d7d26e96a398758e378830f7d6197f"
    )
    assert len(capture.points) == 4

    cpu_points = [
        point
        for point in capture.points
        if point.metric_name == "system.cpu.utilization"
    ]
    first = cpu_points[0]
    assert first.metric_name == "system.cpu.utilization"
    assert first.kind is MetricKind.GAUGE
    assert first.temporality is Temporality.UNSPECIFIED
    assert first.time_unix_nano == 1_000_000_000
    assert first.number_value == 0.35
    assert first.resource_attributes == {
        "service.instance.id": "checkout-1",
        "service.name": "checkout",
    }
    assert first.point_attributes == {"cpu.logical_number": 0}
    assert first.scope_name == "quantis.fixture"
    assert first.scope_version == "1.0.0"

    counter = [
        point
        for point in capture.points
        if point.metric_name == "http.server.request.count"
    ][-1]
    assert counter.metric_name == "http.server.request.count"
    assert counter.kind is MetricKind.SUM
    assert counter.temporality is Temporality.CUMULATIVE
    assert counter.monotonic is True
    assert counter.number_value == 25.0


def test_capture_reader_preserves_int64_values_exactly(tmp_path):
    capture_path = tmp_path / "int64.jsonl"
    capture_path.write_text(
        json.dumps(
            {
                "resourceMetrics": [
                    {
                        "scopeMetrics": [
                            {
                                "metrics": [
                                    {
                                        "name": "demo.large_counter",
                                        "sum": {
                                            "aggregationTemporality": 2,
                                            "isMonotonic": True,
                                            "dataPoints": [
                                                {
                                                    "startTimeUnixNano": "0",
                                                    "timeUnixNano": "1000000000",
                                                    "asInt": str(2**53),
                                                },
                                                {
                                                    "startTimeUnixNano": "0",
                                                    "timeUnixNano": "2000000000",
                                                    "asInt": str(2**53 + 1),
                                                },
                                            ],
                                        },
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        )
        + "\n"
    )

    capture = read_otlp_capture(capture_path)

    assert [point.number_value for point in capture.points] == [
        2**53,
        2**53 + 1,
    ]
    assert all(isinstance(point.number_value, int) for point in capture.points)


def test_capture_reader_wraps_non_object_json_with_line_number(tmp_path):
    capture_path = tmp_path / "malformed.jsonl"
    capture_path.write_text("[]\n")

    with pytest.raises(
        OtlpCaptureError,
        match=r"invalid OTLP JSON at line 1: top-level message must be an object",
    ):
        read_otlp_capture(capture_path)


def test_capture_reader_accepts_flagged_number_point_without_value(tmp_path):
    capture_path = tmp_path / "no-recorded-value.jsonl"
    capture_path.write_text(
        json.dumps(
            {
                "resourceMetrics": [
                    {
                        "scopeMetrics": [
                            {
                                "metrics": [
                                    {
                                        "name": "demo.counter",
                                        "sum": {
                                            "aggregationTemporality": 2,
                                            "isMonotonic": True,
                                            "dataPoints": [
                                                {
                                                    "startTimeUnixNano": "0",
                                                    "timeUnixNano": "1000000000",
                                                    "flags": 1,
                                                }
                                            ],
                                        },
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        )
        + "\n"
    )

    capture = read_otlp_capture(capture_path)

    assert len(capture.points) == 1
    assert capture.points[0].flags == 1
    assert capture.points[0].number_value is None


def test_capture_reader_wraps_malformed_nested_value_with_line_number(tmp_path):
    capture_path = tmp_path / "malformed-nested.jsonl"
    capture_path.write_text('{"resourceMetrics":[1]}\n')

    with pytest.raises(
        OtlpCaptureError,
        match=r"invalid OTLP JSON at line 1:",
    ):
        read_otlp_capture(capture_path)


def test_capture_reader_rejects_multiple_metric_data_kinds(tmp_path):
    capture_path = tmp_path / "invalid-oneof.jsonl"
    capture_path.write_text(
        json.dumps(
            {
                "resourceMetrics": [
                    {
                        "scopeMetrics": [
                            {
                                "metrics": [
                                    {
                                        "name": "demo.invalid",
                                        "gauge": {"dataPoints": []},
                                        "summary": {"dataPoints": []},
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        )
        + "\n"
    )

    with pytest.raises(
        OtlpCaptureError,
        match=r"line 1, metric 'demo.invalid':.*multiple metric kinds",
    ):
        read_otlp_capture(capture_path)


def test_capture_reader_accepts_flagged_histogram_without_payload(tmp_path):
    capture_path = tmp_path / "no-recorded-histogram.jsonl"
    capture_path.write_text(
        json.dumps(
            {
                "resourceMetrics": [
                    {
                        "scopeMetrics": [
                            {
                                "metrics": [
                                    {
                                        "name": "demo.latency",
                                        "histogram": {
                                            "aggregationTemporality": 2,
                                            "dataPoints": [
                                                {
                                                    "startTimeUnixNano": "0",
                                                    "timeUnixNano": "1000000000",
                                                    "flags": 1,
                                                }
                                            ],
                                        },
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        )
        + "\n"
    )

    capture = read_otlp_capture(capture_path)

    assert len(capture.points) == 1
    assert capture.points[0].flags == 1
    assert capture.points[0].histogram_count is None
    assert capture.points[0].histogram_sum is None
