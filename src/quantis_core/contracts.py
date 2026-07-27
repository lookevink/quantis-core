"""Versioned contracts crossing model and evidence seams."""

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple


@dataclass(frozen=True)
class FeatureEvidence:
    name: str
    magnitude: float
    direction: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("feature evidence name cannot be empty")
        if not math.isfinite(self.magnitude) or self.magnitude < 0.0:
            raise ValueError("feature evidence magnitude must be finite and nonnegative")
        if self.direction not in (-1, 0, 1):
            raise ValueError("feature evidence direction must be -1, 0, or 1")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "magnitude": self.magnitude,
            "direction": self.direction,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FeatureEvidence":
        return cls(
            name=str(payload["name"]),
            magnitude=float(payload["magnitude"]),
            direction=int(payload["direction"]),
        )


@dataclass(frozen=True)
class DetectionEvent:
    model_kind: str
    model_version: str
    feature_schema_id: str
    capture_sha256: str
    window_end_unix_nano: int
    score: float
    threshold: float
    alert: bool
    top_features: Tuple[FeatureEvidence, ...]
    data_quality: Mapping[str, int]

    def __post_init__(self) -> None:
        if not self.model_kind:
            raise ValueError("model_kind cannot be empty")
        _validate_sha256("model_version", self.model_version)
        _validate_sha256("feature_schema_id", self.feature_schema_id)
        _validate_sha256("capture_sha256", self.capture_sha256)
        if self.window_end_unix_nano <= 0:
            raise ValueError("window_end_unix_nano must be positive")
        if not math.isfinite(self.score) or not math.isfinite(self.threshold):
            raise ValueError("detection score and threshold must be finite")
        names = [feature.name for feature in self.top_features]
        if len(names) != len(set(names)):
            raise ValueError("top feature names must be unique")
        if self.alert and not self.top_features:
            raise ValueError("alerting events require feature evidence")
        if any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, int)
            or value < 0
            for key, value in self.data_quality.items()
        ):
            raise ValueError("data_quality must contain nonnegative integer counts")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "model_kind": self.model_kind,
            "model_version": self.model_version,
            "feature_schema_id": self.feature_schema_id,
            "capture_sha256": self.capture_sha256,
            "window_end_unix_nano": self.window_end_unix_nano,
            "score": self.score,
            "threshold": self.threshold,
            "alert": self.alert,
            "top_features": [
                feature.to_dict() for feature in self.top_features
            ],
            "data_quality": dict(self.data_quality),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DetectionEvent":
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported DetectionEvent schema_version")
        return cls(
            model_kind=str(payload["model_kind"]),
            model_version=str(payload["model_version"]),
            feature_schema_id=str(payload["feature_schema_id"]),
            capture_sha256=str(payload["capture_sha256"]),
            window_end_unix_nano=int(payload["window_end_unix_nano"]),
            score=float(payload["score"]),
            threshold=float(payload["threshold"]),
            alert=bool(payload["alert"]),
            top_features=tuple(
                FeatureEvidence.from_dict(item) for item in payload["top_features"]
            ),
            data_quality={
                str(key): int(value)
                for key, value in payload["data_quality"].items()
            },
        )


def _validate_sha256(name: str, value: str) -> None:
    if len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be a SHA-256 hex digest") from error
