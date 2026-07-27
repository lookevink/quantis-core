"""Event-time compilation of canonical OTLP metric points."""

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from .otlp import (
    NO_RECORDED_VALUE,
    AttributeValue,
    MetricKind,
    MetricPoint,
    TelemetryCapture,
    Temporality,
)


class FeatureStatistic(Enum):
    GAUGE_LAST = "gauge_last"
    SUM_RATE = "sum_rate"
    HISTOGRAM_MEAN = "histogram_mean"
    HISTOGRAM_COUNT_RATE = "histogram_count_rate"


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    metric_name: str
    statistic: FeatureStatistic
    resource_attributes: Mapping[str, AttributeValue] = field(default_factory=dict)
    point_attributes: Mapping[str, AttributeValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not self.metric_name:
            raise ValueError("feature and metric names cannot be empty")
        _validate_filter_values(self.resource_attributes)
        _validate_filter_values(self.point_attributes)


@dataclass(frozen=True)
class OtlpFeatureSpec:
    window_period_nano: int
    features: Tuple[FeatureDefinition, ...]

    def __post_init__(self) -> None:
        if self.window_period_nano <= 0:
            raise ValueError("window_period_nano must be positive")
        if not self.features:
            raise ValueError("feature specification cannot be empty")
        names = [feature.name for feature in self.features]
        if len(names) != len(set(names)):
            raise ValueError("feature names must be unique")

    @property
    def schema_id(self) -> str:
        payload = self.to_dict()
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "window_period_nano": self.window_period_nano,
            "features": [
                {
                    "name": feature.name,
                    "metric_name": feature.metric_name,
                    "statistic": feature.statistic.value,
                    "resource_attributes": dict(feature.resource_attributes),
                    "point_attributes": dict(feature.point_attributes),
                }
                for feature in self.features
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OtlpFeatureSpec":
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported OtlpFeatureSpec schema_version")
        return cls(
            window_period_nano=int(payload["window_period_nano"]),
            features=tuple(
                FeatureDefinition(
                    name=str(item["name"]),
                    metric_name=str(item["metric_name"]),
                    statistic=FeatureStatistic(str(item["statistic"])),
                    resource_attributes=dict(item.get("resource_attributes", {})),
                    point_attributes=dict(item.get("point_attributes", {})),
                )
                for item in payload["features"]
            ),
        )


@dataclass(frozen=True)
class CompiledTelemetry:
    window_end_unix_nano: NDArray[np.int64]
    feature_names: Tuple[str, ...]
    values: NDArray[np.float64]
    observed_mask: NDArray[np.bool_]
    reset_mask: NDArray[np.bool_]
    feature_schema_id: str
    capture_sha256: str
    data_quality: Mapping[str, int]

    def to_dict(self) -> Dict[str, Any]:
        serialized_values = [
            [
                float(value) if math.isfinite(float(value)) else None
                for value in row
            ]
            for row in self.values
        ]
        return {
            "schema_version": 1,
            "window_end_unix_nano": self.window_end_unix_nano.tolist(),
            "feature_names": list(self.feature_names),
            "values": serialized_values,
            "observed_mask": self.observed_mask.tolist(),
            "reset_mask": self.reset_mask.tolist(),
            "feature_schema_id": self.feature_schema_id,
            "capture_sha256": self.capture_sha256,
            "data_quality": dict(self.data_quality),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CompiledTelemetry":
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported CompiledTelemetry schema_version")
        values = np.asarray(
            [
                [np.nan if value is None else float(value) for value in row]
                for row in payload["values"]
            ],
            dtype=np.float64,
        )
        observed_mask = np.asarray(payload["observed_mask"], dtype=np.bool_)
        reset_mask = np.asarray(payload["reset_mask"], dtype=np.bool_)
        if (
            values.ndim != 2
            or observed_mask.shape != values.shape
            or reset_mask.shape != values.shape
        ):
            raise ValueError("compiled telemetry matrices must have equal 2D shapes")
        if not np.array_equal(observed_mask, np.isfinite(values)):
            raise ValueError("observed_mask must exactly identify finite values")
        feature_names = tuple(str(name) for name in payload["feature_names"])
        window_ends = np.asarray(
            payload["window_end_unix_nano"], dtype=np.int64
        )
        if values.shape != (len(window_ends), len(feature_names)):
            raise ValueError("compiled telemetry axes do not match matrix shape")
        return cls(
            window_end_unix_nano=window_ends,
            feature_names=feature_names,
            values=values,
            observed_mask=observed_mask,
            reset_mask=reset_mask,
            feature_schema_id=str(payload["feature_schema_id"]),
            capture_sha256=str(payload["capture_sha256"]),
            data_quality={
                str(key): int(value)
                for key, value in payload["data_quality"].items()
            },
        )


@dataclass(frozen=True)
class ForwardFillPolicy:
    max_gap_windows: int

    def __post_init__(self) -> None:
        if self.max_gap_windows < 0:
            raise ValueError("max_gap_windows cannot be negative")


@dataclass(frozen=True)
class MaterializedTelemetry:
    window_end_unix_nano: NDArray[np.int64]
    feature_names: Tuple[str, ...]
    values: NDArray[np.float64]
    observed_mask: NDArray[np.bool_]
    imputed_mask: NDArray[np.bool_]
    feature_schema_id: str
    capture_sha256: str


class OtlpWindowError(ValueError):
    """OTLP points cannot be compiled without ambiguous semantics."""


class TelemetryMaterializationError(ValueError):
    """Explicit missingness policy cannot safely produce model values."""


class OtlpWindowCompiler:
    """Compile selected OTLP streams into deterministic event-time windows."""

    def __init__(self, feature_spec: OtlpFeatureSpec) -> None:
        self.feature_spec = feature_spec

    def compile(self, capture: TelemetryCapture) -> CompiledTelemetry:
        matched_by_feature = [
            _matching_points(capture.points, definition)
            for definition in self.feature_spec.features
        ]
        all_matched = [point for points in matched_by_feature for point in points]
        if not all_matched:
            raise OtlpWindowError("feature specification matched no metric points")
        first_end = min(
            _window_end(point.time_unix_nano, self.feature_spec.window_period_nano)
            for point in all_matched
        )
        last_end = max(
            _window_end(point.time_unix_nano, self.feature_spec.window_period_nano)
            for point in all_matched
        )
        window_ends = np.arange(
            first_end,
            last_end + self.feature_spec.window_period_nano,
            self.feature_spec.window_period_nano,
            dtype=np.int64,
        )
        values = np.full(
            (len(window_ends), len(self.feature_spec.features)),
            np.nan,
            dtype=np.float64,
        )
        reset_mask = np.zeros_like(values, dtype=np.bool_)

        for feature_index, (definition, points) in enumerate(
            zip(self.feature_spec.features, matched_by_feature)
        ):
            observations, reset_times = _feature_observations(definition, points)
            for time_unix_nano, value in observations:
                window_index = (
                    _window_end(
                        time_unix_nano, self.feature_spec.window_period_nano
                    )
                    - first_end
                ) // self.feature_spec.window_period_nano
                values[window_index, feature_index] = value
            for time_unix_nano in reset_times:
                window_index = (
                    _window_end(
                        time_unix_nano, self.feature_spec.window_period_nano
                    )
                    - first_end
                ) // self.feature_spec.window_period_nano
                reset_mask[window_index, feature_index] = True

        observed_mask = np.isfinite(values)
        unique_matched_points = {
            id(point): point for point in all_matched
        }.values()
        flagged_points = sum(
            1
            for point in unique_matched_points
            if point.flags & NO_RECORDED_VALUE
        )
        return CompiledTelemetry(
            window_end_unix_nano=window_ends,
            feature_names=tuple(
                feature.name for feature in self.feature_spec.features
            ),
            values=values,
            observed_mask=observed_mask,
            reset_mask=reset_mask,
            feature_schema_id=self.feature_spec.schema_id,
            capture_sha256=capture.sha256,
            data_quality={
                "flagged_points": flagged_points,
                "reset_points": int(np.count_nonzero(reset_mask)),
                "missing_cells": int(np.count_nonzero(~observed_mask)),
            },
        )


def materialize_compiled_telemetry(
    compiled: CompiledTelemetry, policy: ForwardFillPolicy
) -> MaterializedTelemetry:
    """Apply bounded forward fill while retaining explicit imputation evidence."""

    values = compiled.values.copy()
    imputed = np.zeros_like(compiled.observed_mask, dtype=np.bool_)
    for feature_index, feature_name in enumerate(compiled.feature_names):
        last_value: Optional[float] = None
        gap = 0
        for window_index, window_end in enumerate(compiled.window_end_unix_nano):
            if compiled.reset_mask[window_index, feature_index]:
                last_value = None
                gap = 0
            if compiled.observed_mask[window_index, feature_index]:
                last_value = float(values[window_index, feature_index])
                gap = 0
                continue
            gap += 1
            if last_value is None or gap > policy.max_gap_windows:
                raise TelemetryMaterializationError(
                    f"feature {feature_name!r} is missing at window "
                    f"{int(window_end)} and cannot be forward-filled"
                )
            values[window_index, feature_index] = last_value
            imputed[window_index, feature_index] = True
    return MaterializedTelemetry(
        window_end_unix_nano=compiled.window_end_unix_nano.copy(),
        feature_names=compiled.feature_names,
        values=values,
        observed_mask=compiled.observed_mask.copy(),
        imputed_mask=imputed,
        feature_schema_id=compiled.feature_schema_id,
        capture_sha256=compiled.capture_sha256,
    )


def _matching_points(
    points: Sequence[MetricPoint], feature: FeatureDefinition
) -> List[MetricPoint]:
    matched = [
        point
        for point in points
        if point.metric_name == feature.metric_name
        and _contains(point.resource_attributes, feature.resource_attributes)
        and _contains(point.point_attributes, feature.point_attributes)
    ]
    if not matched:
        raise OtlpWindowError(
            f"feature {feature.name!r} matched no points for "
            f"metric {feature.metric_name!r}"
        )
    stream_keys = {_stream_key(point) for point in matched}
    if len(stream_keys) != 1:
        raise OtlpWindowError(
            f"feature {feature.name!r} matched {len(stream_keys)} metric streams; "
            "add resource or point attribute filters"
        )
    return sorted(matched, key=lambda point: point.time_unix_nano)


def _feature_observations(
    feature: FeatureDefinition, points: Sequence[MetricPoint]
) -> Tuple[List[Tuple[int, float]], List[int]]:
    expected_kind = {
        FeatureStatistic.GAUGE_LAST: MetricKind.GAUGE,
        FeatureStatistic.SUM_RATE: MetricKind.SUM,
        FeatureStatistic.HISTOGRAM_MEAN: MetricKind.HISTOGRAM,
        FeatureStatistic.HISTOGRAM_COUNT_RATE: MetricKind.HISTOGRAM,
    }[feature.statistic]
    if points[0].kind is not expected_kind:
        raise OtlpWindowError(
            f"feature {feature.name!r} statistic {feature.statistic.value!r} "
            f"cannot consume {points[0].kind.value!r}"
        )

    if feature.statistic is FeatureStatistic.GAUGE_LAST:
        return [
            (point.time_unix_nano, _finite_number(point.number_value))
            for point in points
            if not point.flags & NO_RECORDED_VALUE
        ], []
    if points[0].temporality is Temporality.DELTA:
        return _delta_observations(
            feature.statistic,
            [
                point
                for point in points
                if not point.flags & NO_RECORDED_VALUE
            ],
        ), []
    if points[0].temporality is Temporality.CUMULATIVE:
        return _cumulative_observations(feature.statistic, points)
    raise OtlpWindowError(
        f"feature {feature.name!r} has unspecified aggregation temporality"
    )


def _delta_observations(
    statistic: FeatureStatistic, points: Sequence[MetricPoint]
) -> List[Tuple[int, float]]:
    observations: List[Tuple[int, float]] = []
    for point in points:
        value = _point_interval_value(statistic, point, previous=None)
        if statistic in (
            FeatureStatistic.SUM_RATE,
            FeatureStatistic.HISTOGRAM_COUNT_RATE,
        ):
            value /= _duration_seconds(point)
        observations.append((point.time_unix_nano, value))
    return observations


def _cumulative_observations(
    statistic: FeatureStatistic, points: Sequence[MetricPoint]
) -> Tuple[List[Tuple[int, float]], List[int]]:
    observations: List[Tuple[int, float]] = []
    resets: List[int] = []
    previous_stream_point: Optional[MetricPoint] = None
    previous_valid_point: Optional[MetricPoint] = None
    for point in points:
        if (
            previous_stream_point is not None
            and point.time_unix_nano <= previous_stream_point.time_unix_nano
        ):
            raise OtlpWindowError("cumulative points must have increasing timestamps")
        start_time_reset = bool(
            previous_stream_point is not None
            and _normalized_start_time(previous_stream_point)
            != _normalized_start_time(point)
        )
        if start_time_reset:
            resets.append(point.time_unix_nano)
            previous_valid_point = None
        previous_stream_point = point

        if point.flags & NO_RECORDED_VALUE:
            continue
        if previous_valid_point is None:
            previous_valid_point = point
            continue
        value_reset = _values_decreased(
            statistic, previous_valid_point, point
        )
        if value_reset:
            resets.append(point.time_unix_nano)
            previous_valid_point = point
            continue
        elapsed_seconds = (
            point.time_unix_nano - previous_valid_point.time_unix_nano
        ) / 1e9
        if elapsed_seconds <= 0.0:
            raise OtlpWindowError("cumulative points must have increasing timestamps")
        value = _point_interval_value(statistic, point, previous_valid_point)
        if statistic in (
            FeatureStatistic.SUM_RATE,
            FeatureStatistic.HISTOGRAM_COUNT_RATE,
        ):
            value /= elapsed_seconds
        observations.append((point.time_unix_nano, value))
        previous_valid_point = point
    return observations, resets


def _point_interval_value(
    statistic: FeatureStatistic,
    point: MetricPoint,
    previous: Optional[MetricPoint],
) -> float:
    if statistic is FeatureStatistic.SUM_RATE:
        current = _finite_number(point.number_value)
        return current - _finite_number(previous.number_value) if previous else current
    current_count = _histogram_count(point)
    if previous is None:
        interval_count = current_count
    else:
        interval_count = current_count - _histogram_count(previous)
    if statistic is FeatureStatistic.HISTOGRAM_COUNT_RATE:
        return float(interval_count)
    current_sum = _histogram_sum(point)
    interval_sum = (
        current_sum
        if previous is None
        else current_sum - _histogram_sum(previous)
    )
    if interval_count <= 0:
        raise OtlpWindowError("histogram mean requires a positive interval count")
    return interval_sum / interval_count


def _values_decreased(
    statistic: FeatureStatistic,
    previous: MetricPoint,
    current: MetricPoint,
) -> bool:
    if current.kind is MetricKind.SUM:
        return bool(
            current.monotonic
            and _finite_number(current.number_value)
            < _finite_number(previous.number_value)
        )
    count_decreased = _histogram_count(current) < _histogram_count(previous)
    if statistic is FeatureStatistic.HISTOGRAM_COUNT_RATE:
        return count_decreased
    return bool(
        count_decreased
        or _histogram_sum(current) < _histogram_sum(previous)
    )


def _duration_seconds(point: MetricPoint) -> float:
    if point.start_time_unix_nano is None:
        raise OtlpWindowError("delta point is missing startTimeUnixNano")
    duration = (point.time_unix_nano - point.start_time_unix_nano) / 1e9
    if duration <= 0.0:
        raise OtlpWindowError("delta point interval must have positive duration")
    return duration


def _normalized_start_time(point: MetricPoint) -> int:
    return point.start_time_unix_nano or 0


def _finite_number(value: Optional[Union[int, float]]) -> Union[int, float]:
    if value is None or not math.isfinite(value):
        raise OtlpWindowError("metric value must be finite")
    return value


def _histogram_count(point: MetricPoint) -> int:
    if point.histogram_count is None:
        raise OtlpWindowError("histogram point is missing count")
    return point.histogram_count


def _histogram_sum(point: MetricPoint) -> float:
    return _finite_number(point.histogram_sum)


def _window_end(time_unix_nano: int, period: int) -> int:
    return ((time_unix_nano + period - 1) // period) * period


def _contains(
    actual: Mapping[str, AttributeValue],
    expected: Mapping[str, AttributeValue],
) -> bool:
    return all(actual.get(key) == value for key, value in expected.items())


def _stream_key(point: MetricPoint) -> Tuple[Any, ...]:
    return (
        point.metric_name,
        point.unit,
        point.kind.value,
        int(point.temporality),
        point.monotonic,
        json.dumps(point.resource_attributes, sort_keys=True, default=str),
        json.dumps(point.point_attributes, sort_keys=True, default=str),
        point.scope_name,
        point.scope_version,
    )


def _validate_filter_values(values: Mapping[str, AttributeValue]) -> None:
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            raise ValueError("attribute filter keys must be non-empty strings")
        if isinstance(value, bool):
            continue
        if isinstance(value, (str, int)):
            continue
        if isinstance(value, float) and math.isfinite(value):
            continue
        raise ValueError(
            "attribute filters support only finite JSON scalar values"
        )
