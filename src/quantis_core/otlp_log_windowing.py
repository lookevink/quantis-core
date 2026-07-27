"""Deterministic structured-event features from OTLP application logs."""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .otlp import AttributeValue
from .otlp_logs import LogRecord, OtlpLogCapture


@dataclass(frozen=True)
class LogFeatureDefinition:
    """One stable event-count feature; unrestricted bodies are not filters."""

    name: str
    resource_attributes: Mapping[
        str, AttributeValue
    ] = field(default_factory=dict)
    record_attributes: Mapping[
        str, AttributeValue
    ] = field(default_factory=dict)
    minimum_severity_number: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("log feature name cannot be empty")
        if (
            not self.resource_attributes
            and not self.record_attributes
            and self.minimum_severity_number is None
        ):
            raise ValueError(
                "log feature requires an attribute or severity filter"
            )
        if (
            self.minimum_severity_number is not None
            and not 1 <= self.minimum_severity_number <= 24
        ):
            raise ValueError(
                "minimum_severity_number must be in [1, 24]"
            )


@dataclass(frozen=True)
class OtlpLogFeatureSpec:
    """Versioned log vocabulary and logical-window assignment."""

    window_index_attribute: str
    features: Tuple[LogFeatureDefinition, ...]

    def __post_init__(self) -> None:
        if not self.window_index_attribute:
            raise ValueError("window_index_attribute cannot be empty")
        if not self.features:
            raise ValueError("log feature specification cannot be empty")
        names = [feature.name for feature in self.features]
        if len(names) != len(set(names)):
            raise ValueError("log feature names must be unique")

    @property
    def schema_id(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "window_index_attribute": self.window_index_attribute,
            "features": [
                {
                    "name": feature.name,
                    "resource_attributes": dict(
                        feature.resource_attributes
                    ),
                    "record_attributes": dict(
                        feature.record_attributes
                    ),
                    "minimum_severity_number": (
                        feature.minimum_severity_number
                    ),
                }
                for feature in self.features
            ],
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "OtlpLogFeatureSpec":
        if payload.get("schema_version") != 1:
            raise ValueError(
                "unsupported OtlpLogFeatureSpec schema_version"
            )
        return cls(
            window_index_attribute=str(
                payload["window_index_attribute"]
            ),
            features=tuple(
                LogFeatureDefinition(
                    name=str(item["name"]),
                    resource_attributes=dict(
                        item.get("resource_attributes", {})
                    ),
                    record_attributes=dict(
                        item.get("record_attributes", {})
                    ),
                    minimum_severity_number=(
                        int(item["minimum_severity_number"])
                        if item.get("minimum_severity_number")
                        is not None
                        else None
                    ),
                )
                for item in payload["features"]
            ),
        )


@dataclass(frozen=True)
class CompiledLogTelemetry:
    """Dense, zero-materialized event counts for one application run."""

    window_indices: NDArray[np.int64]
    feature_names: Tuple[str, ...]
    values: NDArray[np.float64]
    feature_schema_id: str
    capture_sha256: str
    data_quality: Mapping[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "window_indices": self.window_indices.tolist(),
            "feature_names": list(self.feature_names),
            "values": self.values.tolist(),
            "feature_schema_id": self.feature_schema_id,
            "capture_sha256": self.capture_sha256,
            "data_quality": dict(self.data_quality),
        }


class OtlpLogWindowError(ValueError):
    """Structured logs cannot be assigned to declared model windows."""


class OtlpLogWindowCompiler:
    """Compile a preregistered event vocabulary into dense count windows."""

    def __init__(self, feature_spec: OtlpLogFeatureSpec) -> None:
        self.feature_spec = feature_spec

    def compile(
        self,
        capture: OtlpLogCapture,
        window_count: int,
    ) -> CompiledLogTelemetry:
        if window_count < 1:
            raise ValueError("window_count must be positive")
        values = np.zeros(
            (window_count, len(self.feature_spec.features)),
            dtype=np.float64,
        )
        matched_record_count = 0
        for record in capture.records:
            window_index = self._window_index(record, window_count)
            matched = False
            for feature_index, definition in enumerate(
                self.feature_spec.features
            ):
                if _matches(record, definition):
                    values[window_index, feature_index] += 1.0
                    matched = True
            if matched:
                matched_record_count += 1
        return CompiledLogTelemetry(
            window_indices=np.arange(
                window_count,
                dtype=np.int64,
            ),
            feature_names=tuple(
                definition.name
                for definition in self.feature_spec.features
            ),
            values=values,
            feature_schema_id=self.feature_spec.schema_id,
            capture_sha256=capture.sha256,
            data_quality={
                "record_count": len(capture.records),
                "matched_records": matched_record_count,
                "unmatched_records": (
                    len(capture.records) - matched_record_count
                ),
            },
        )

    def _window_index(
        self,
        record: LogRecord,
        window_count: int,
    ) -> int:
        raw = record.record_attributes.get(
            self.feature_spec.window_index_attribute
        )
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise OtlpLogWindowError(
                "every log record requires an integer logical-window "
                "attribute"
            )
        if not 0 <= raw < window_count:
            raise OtlpLogWindowError(
                f"logical-window index {raw} is outside "
                f"[0, {window_count})"
            )
        return raw


def _matches(
    record: LogRecord,
    definition: LogFeatureDefinition,
) -> bool:
    return (
        _contains(
            record.resource_attributes,
            definition.resource_attributes,
        )
        and _contains(
            record.record_attributes,
            definition.record_attributes,
        )
        and (
            definition.minimum_severity_number is None
            or record.severity_number
            >= definition.minimum_severity_number
        )
    )


def _contains(
    actual: Mapping[str, AttributeValue],
    expected: Mapping[str, AttributeValue],
) -> bool:
    return all(actual.get(key) == value for key, value in expected.items())
