"""Training-serving-parity preprocessing for model windows."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


MAD_NORMAL_SCALE = 1.4826


@dataclass(frozen=True)
class ModelWindows:
    """Normalized temporal contexts paired with their next observed point."""

    contexts: NDArray[np.float64]
    targets: NDArray[np.float64]
    point_indices: NDArray[np.int64]
    feature_names: Tuple[str, ...]


class WindowCompiler:
    """Fit robust normalization once and compile aligned temporal windows."""

    def __init__(
        self,
        lookback: int,
        location: Optional[NDArray[np.float64]] = None,
        scale: Optional[NDArray[np.float64]] = None,
    ) -> None:
        if lookback < 1:
            raise ValueError("lookback must be positive")
        self.lookback = lookback
        self._location = location
        self._scale = scale

    def fit(self, telemetry: NDArray[np.float64]) -> "WindowCompiler":
        values = _validated_telemetry(telemetry)
        if len(values) <= self.lookback:
            raise ValueError("telemetry must contain more points than lookback")

        location = np.median(values, axis=0)
        scale = MAD_NORMAL_SCALE * np.median(np.abs(values - location), axis=0)
        fallback = np.std(values, axis=0)
        scale = np.where(scale > 1e-12, scale, fallback)
        scale = np.where(scale > 1e-12, scale, 1.0)

        self._location = location.astype(np.float64)
        self._scale = scale.astype(np.float64)
        return self

    def transform(
        self,
        telemetry: NDArray[np.float64],
        feature_names: Optional[Sequence[str]] = None,
    ) -> ModelWindows:
        values = _validated_telemetry(telemetry)
        location, scale = self._fitted_state()
        if values.shape[1] != len(location):
            raise ValueError("telemetry feature count does not match fitted compiler")
        if len(values) <= self.lookback:
            raise ValueError("telemetry must contain more points than lookback")

        normalized = (values - location) / scale
        contexts = np.stack(
            [
                normalized[index - self.lookback : index]
                for index in range(self.lookback, len(normalized))
            ]
        )
        targets = normalized[self.lookback :]
        point_indices = np.arange(self.lookback, len(normalized), dtype=np.int64)
        names = (
            tuple(feature_names)
            if feature_names is not None
            else tuple(f"feature_{index}" for index in range(values.shape[1]))
        )
        if len(names) != values.shape[1]:
            raise ValueError("feature_names must match telemetry feature count")
        return ModelWindows(
            contexts=contexts,
            targets=targets,
            point_indices=point_indices,
            feature_names=names,
        )

    def to_dict(self) -> Dict[str, Any]:
        location, scale = self._fitted_state()
        return {
            "schema_version": 1,
            "lookback": self.lookback,
            "location": location.tolist(),
            "scale": scale.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "WindowCompiler":
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported WindowCompiler schema_version")
        return cls(
            lookback=int(payload["lookback"]),
            location=np.asarray(payload["location"], dtype=np.float64),
            scale=np.asarray(payload["scale"], dtype=np.float64),
        )

    def _fitted_state(
        self,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        if self._location is None or self._scale is None:
            raise RuntimeError("WindowCompiler must be fitted before use")
        return self._location, self._scale


def repair_isolated_context_outliers(
    windows: ModelWindows,
    z_threshold: float,
    consensus_rank: int,
) -> Tuple[ModelWindows, int]:
    """Repair isolated context corruption while retaining correlated drift."""

    if z_threshold <= 0.0:
        raise ValueError("isolated context z threshold must be positive")
    contexts = windows.contexts.copy()
    outliers = np.abs(contexts) > z_threshold
    isolated_rows = (
        np.count_nonzero(outliers, axis=2, keepdims=True)
        < consensus_rank
    )
    repaired = outliers & isolated_rows
    for sample_index, time_index in np.argwhere(
        np.any(repaired, axis=2)
    ):
        row_outliers = repaired[sample_index, time_index]
        consensus_values = contexts[
            sample_index, time_index, ~row_outliers
        ]
        contexts[sample_index, time_index, row_outliers] = float(
            np.median(consensus_values)
        )
    return (
        ModelWindows(
            contexts=contexts,
            targets=windows.targets,
            point_indices=windows.point_indices,
            feature_names=windows.feature_names,
        ),
        int(np.count_nonzero(repaired)),
    )


def _validated_telemetry(telemetry: NDArray[np.float64]) -> NDArray[np.float64]:
    values = np.asarray(telemetry, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("telemetry must be a two-dimensional feature matrix")
    if not np.all(np.isfinite(values)):
        raise ValueError("telemetry values must be finite")
    return values
