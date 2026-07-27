"""Canonical OTLP JSON capture reader.

The parser intentionally models only the stable metric fields needed by the
replay seam. Unsupported metric kinds fail loudly instead of being dropped.
"""

import hashlib
import json
from dataclasses import dataclass
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union


AttributeValue = Union[
    str,
    int,
    float,
    bool,
    bytes,
    Sequence["AttributeValue"],
    Mapping[str, "AttributeValue"],
]
NO_RECORDED_VALUE = 1


class MetricKind(Enum):
    GAUGE = "gauge"
    SUM = "sum"
    HISTOGRAM = "histogram"


class Temporality(IntEnum):
    UNSPECIFIED = 0
    DELTA = 1
    CUMULATIVE = 2


@dataclass(frozen=True)
class MetricPoint:
    metric_name: str
    unit: str
    kind: MetricKind
    temporality: Temporality
    monotonic: bool
    time_unix_nano: int
    start_time_unix_nano: Optional[int]
    flags: int
    resource_attributes: Mapping[str, AttributeValue]
    point_attributes: Mapping[str, AttributeValue]
    scope_name: str
    scope_version: str
    number_value: Optional[Union[int, float]] = None
    histogram_count: Optional[int] = None
    histogram_sum: Optional[float] = None
    bucket_counts: Tuple[int, ...] = ()
    explicit_bounds: Tuple[float, ...] = ()


@dataclass(frozen=True)
class TelemetryCapture:
    points: Tuple[MetricPoint, ...]
    sha256: str
    source_path: str
    json_message_count: int


class OtlpCaptureError(ValueError):
    """An OTLP JSON capture cannot be interpreted without data loss."""


def read_otlp_capture(path: Union[str, Path]) -> TelemetryCapture:
    """Read newline-delimited OTLP JSON metric export requests."""

    capture_path = Path(path)
    raw = capture_path.read_bytes()
    points: List[MetricPoint] = []
    message_count = 0
    for line_number, raw_line in enumerate(raw.splitlines(), start=1):
        if not raw_line.strip():
            continue
        message_count += 1
        try:
            request = json.loads(raw_line)
            if not isinstance(request, dict):
                raise OtlpCaptureError(
                    f"invalid OTLP JSON at line {line_number}: "
                    "top-level message must be an object"
                )
            points.extend(_request_points(request, line_number))
        except (
            AttributeError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            if isinstance(error, OtlpCaptureError):
                raise
            raise OtlpCaptureError(
                f"invalid OTLP JSON at line {line_number}: {error}"
            ) from error
    if message_count == 0:
        raise OtlpCaptureError("capture contains no OTLP JSON messages")
    canonical_points = tuple(sorted(points, key=_point_sort_key))
    return TelemetryCapture(
        points=canonical_points,
        sha256=hashlib.sha256(raw).hexdigest(),
        source_path=str(capture_path),
        json_message_count=message_count,
    )


def _request_points(request: Mapping[str, Any], line_number: int) -> List[MetricPoint]:
    points: List[MetricPoint] = []
    for resource_metrics in request.get("resourceMetrics", []):
        resource = resource_metrics.get("resource", {})
        resource_attributes = _attributes(resource.get("attributes", []))
        for scope_metrics in resource_metrics.get("scopeMetrics", []):
            scope = scope_metrics.get("scope", {})
            scope_name = str(scope.get("name", ""))
            scope_version = str(scope.get("version", ""))
            for metric in scope_metrics.get("metrics", []):
                metric_name = str(metric.get("name", ""))
                if not metric_name:
                    raise OtlpCaptureError(
                        f"invalid OTLP JSON at line {line_number}: metric name is empty"
                    )
                try:
                    points.extend(
                        _metric_points(
                            metric,
                            resource_attributes,
                            scope_name,
                            scope_version,
                        )
                    )
                except (AttributeError, KeyError, TypeError, ValueError) as error:
                    raise OtlpCaptureError(
                        f"invalid OTLP JSON at line {line_number}, "
                        f"metric {metric_name!r}: {error}"
                    ) from error
    return points


def _metric_points(
    metric: Mapping[str, Any],
    resource_attributes: Mapping[str, AttributeValue],
    scope_name: str,
    scope_version: str,
) -> List[MetricPoint]:
    metric_name = str(metric["name"])
    unit = str(metric.get("unit", ""))
    supported_kinds = ("gauge", "sum", "histogram")
    known_kinds = supported_kinds + ("exponentialHistogram", "summary")
    present_kinds = [name for name in known_kinds if name in metric]
    if len(present_kinds) > 1:
        raise ValueError(f"multiple metric kinds {sorted(present_kinds)}")
    if not present_kinds:
        raise ValueError("missing kind")
    if present_kinds[0] not in supported_kinds:
        raise ValueError(f"unsupported kind {present_kinds[0]!r}")
    kind_name = present_kinds[0]
    data = metric[kind_name]
    kind = MetricKind(kind_name)
    temporality = (
        Temporality(int(data.get("aggregationTemporality", 0)))
        if kind is not MetricKind.GAUGE
        else Temporality.UNSPECIFIED
    )
    monotonic = bool(data.get("isMonotonic", False))
    return [
        _data_point(
            metric_name=metric_name,
            unit=unit,
            kind=kind,
            temporality=temporality,
            monotonic=monotonic,
            data_point=point,
            resource_attributes=resource_attributes,
            scope_name=scope_name,
            scope_version=scope_version,
        )
        for point in data.get("dataPoints", [])
    ]


def _data_point(
    metric_name: str,
    unit: str,
    kind: MetricKind,
    temporality: Temporality,
    monotonic: bool,
    data_point: Mapping[str, Any],
    resource_attributes: Mapping[str, AttributeValue],
    scope_name: str,
    scope_version: str,
) -> MetricPoint:
    flags = int(data_point.get("flags", 0))
    common: Dict[str, Any] = {
        "metric_name": metric_name,
        "unit": unit,
        "kind": kind,
        "temporality": temporality,
        "monotonic": monotonic,
        "time_unix_nano": int(data_point["timeUnixNano"]),
        "start_time_unix_nano": (
            int(data_point["startTimeUnixNano"])
            if "startTimeUnixNano" in data_point
            else None
        ),
        "flags": flags,
        "resource_attributes": dict(resource_attributes),
        "point_attributes": _attributes(data_point.get("attributes", [])),
        "scope_name": scope_name,
        "scope_version": scope_version,
    }
    if kind in (MetricKind.GAUGE, MetricKind.SUM):
        return MetricPoint(
            number_value=_number_value(data_point, flags),
            **common,
        )
    histogram_count = (
        int(data_point["count"]) if "count" in data_point else None
    )
    if histogram_count is None and not flags & NO_RECORDED_VALUE:
        raise ValueError("histogram point is missing count")
    return MetricPoint(
        histogram_count=histogram_count,
        histogram_sum=(
            float(data_point["sum"]) if "sum" in data_point else None
        ),
        bucket_counts=tuple(int(value) for value in data_point.get("bucketCounts", [])),
        explicit_bounds=tuple(
            float(value) for value in data_point.get("explicitBounds", [])
        ),
        **common,
    )


def _number_value(
    data_point: Mapping[str, Any], flags: int
) -> Optional[Union[int, float]]:
    number_fields = [name for name in ("asDouble", "asInt") if name in data_point]
    if not number_fields and flags & NO_RECORDED_VALUE:
        return None
    if len(number_fields) != 1:
        raise ValueError("number point must contain exactly one value")
    field = number_fields[0]
    if field == "asInt":
        return int(data_point[field])
    return float(data_point[field])


def _attributes(
    attributes: Sequence[Mapping[str, Any]]
) -> Mapping[str, AttributeValue]:
    parsed: Dict[str, AttributeValue] = {}
    for attribute in attributes:
        key = str(attribute["key"])
        if key in parsed:
            raise ValueError(f"duplicate attribute key {key!r}")
        parsed[key] = _any_value(attribute["value"])
    return dict(sorted(parsed.items()))


def _any_value(value: Mapping[str, Any]) -> AttributeValue:
    fields = [
        name
        for name in (
            "stringValue",
            "boolValue",
            "intValue",
            "doubleValue",
            "bytesValue",
            "arrayValue",
            "kvlistValue",
        )
        if name in value
    ]
    if len(fields) != 1:
        raise ValueError("AnyValue must contain exactly one value")
    field = fields[0]
    raw = value[field]
    if field == "intValue":
        return int(raw)
    if field == "doubleValue":
        return float(raw)
    if field == "stringValue":
        return str(raw)
    if field == "boolValue":
        return bool(raw)
    if field == "bytesValue":
        return str(raw).encode("ascii")
    if field == "arrayValue":
        return tuple(_any_value(item) for item in raw.get("values", []))
    if field == "kvlistValue":
        return _attributes(raw.get("values", []))
    raise ValueError(f"unsupported AnyValue field {field!r}")


def _point_sort_key(point: MetricPoint) -> Tuple[Any, ...]:
    return (
        point.metric_name,
        json.dumps(point.resource_attributes, sort_keys=True, default=str),
        json.dumps(point.point_attributes, sort_keys=True, default=str),
        point.time_unix_nano,
        point.start_time_unix_nano or 0,
    )
