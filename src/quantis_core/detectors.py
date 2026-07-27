"""Detectors sharing one observable scoring interface."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, TypeVar

import numpy as np
from numpy.typing import NDArray

from .windowing import ModelWindows


@dataclass(frozen=True)
class DetectionScores:
    """Scalar decisions and feature-level evidence for aligned model windows."""

    scores: NDArray[np.float64]
    feature_evidence: NDArray[np.float64]
    threshold: float
    signed_feature_evidence: Optional[NDArray[np.float64]] = None

    @property
    def alerts(self) -> NDArray[np.bool_]:
        return self.scores > self.threshold


DetectorType = TypeVar("DetectorType", bound="_CalibratedDetector")


class _CalibratedDetector:
    kind = "base"

    def __init__(self, calibration_quantile: float = 0.99) -> None:
        if not 0.5 < calibration_quantile < 1.0:
            raise ValueError("calibration_quantile must be between 0.5 and 1.0")
        self.calibration_quantile = calibration_quantile
        self.threshold: float = float("nan")

    def fit(self: DetectorType, windows: ModelWindows) -> DetectorType:
        _validate_windows(windows)
        self._fit_model(windows)
        training_result = self._score_model(windows)
        self.threshold = float(
            np.quantile(training_result.scores, self.calibration_quantile)
        )
        return self

    def score(self, windows: ModelWindows) -> DetectionScores:
        _validate_windows(windows)
        if not np.isfinite(self.threshold):
            raise RuntimeError("detector must be fitted before scoring")
        result = self._score_model(windows)
        return DetectionScores(
            scores=result.scores,
            feature_evidence=result.feature_evidence,
            threshold=self.threshold,
            signed_feature_evidence=result.signed_feature_evidence,
        )

    def _fit_model(self, windows: ModelWindows) -> None:
        pass

    def _score_model(self, windows: ModelWindows) -> DetectionScores:
        raise NotImplementedError

    def _base_artifact(self) -> Dict[str, Any]:
        if not np.isfinite(self.threshold):
            raise RuntimeError("detector must be fitted before serialization")
        return {
            "schema_version": 1,
            "kind": self.kind,
            "calibration_quantile": self.calibration_quantile,
            "threshold": self.threshold,
        }


class PersistenceDetector(_CalibratedDetector):
    """Predict the next point as the most recent point."""

    kind = "persistence"

    def _score_model(self, windows: ModelWindows) -> DetectionScores:
        residual = np.abs(windows.targets - windows.contexts[:, -1, :])
        scores = np.sqrt(np.mean(np.square(residual), axis=1))
        return DetectionScores(scores, residual, self.threshold)

    def to_dict(self) -> Dict[str, Any]:
        return self._base_artifact()

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PersistenceDetector":
        detector = cls(float(payload["calibration_quantile"]))
        detector.threshold = float(payload["threshold"])
        return detector


class RobustFeatureDetector(_CalibratedDetector):
    """Flag the largest absolute robust-normalized feature value."""

    kind = "robust_feature"

    def _score_model(self, windows: ModelWindows) -> DetectionScores:
        evidence = np.abs(windows.targets)
        scores = np.max(evidence, axis=1)
        return DetectionScores(scores, evidence, self.threshold)

    def to_dict(self) -> Dict[str, Any]:
        return self._base_artifact()

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RobustFeatureDetector":
        detector = cls(float(payload["calibration_quantile"]))
        detector.threshold = float(payload["threshold"])
        return detector


class LatentPredictiveDetector(_CalibratedDetector):
    """Linear prediction in a low-rank target representation.

    The target encoder is a principal subspace fitted on training targets. A
    ridge predictor maps the full context window into that latent space. Scoring
    observes latent prediction error, so deviations orthogonal to the learned
    shared subspace are suppressed.
    """

    kind = "linear_latent_predictive"

    def __init__(
        self,
        latent_dimension: int = 2,
        ridge: float = 1e-2,
        calibration_quantile: float = 0.99,
    ) -> None:
        super().__init__(calibration_quantile)
        if latent_dimension < 1:
            raise ValueError("latent_dimension must be positive")
        if ridge < 0.0:
            raise ValueError("ridge cannot be negative")
        self.latent_dimension = latent_dimension
        self.ridge = ridge
        self._target_center: NDArray[np.float64] = np.asarray([])
        self._components: NDArray[np.float64] = np.asarray([[]])
        self._coefficients: NDArray[np.float64] = np.asarray([[]])
        self._context_shape = (0, 0)

    def _fit_model(self, windows: ModelWindows) -> None:
        sample_count, lookback, feature_count = windows.contexts.shape
        latent_dimension = min(self.latent_dimension, feature_count, sample_count)
        self._context_shape = (lookback, feature_count)
        self._target_center = np.mean(windows.targets, axis=0)
        centered_targets = windows.targets - self._target_center
        _, _, right_vectors = np.linalg.svd(centered_targets, full_matrices=False)
        self._components = right_vectors[:latent_dimension]
        latent_targets = centered_targets @ self._components.T

        design = _design_matrix(windows.contexts)
        penalty = np.eye(design.shape[1], dtype=np.float64) * self.ridge
        penalty[-1, -1] = 0.0
        self._coefficients = np.linalg.solve(
            design.T @ design + penalty,
            design.T @ latent_targets,
        )

    def _score_model(self, windows: ModelWindows) -> DetectionScores:
        if windows.contexts.shape[1:] != self._context_shape:
            raise ValueError("window shape does not match fitted latent detector")
        predicted = _design_matrix(windows.contexts) @ self._coefficients
        observed = (windows.targets - self._target_center) @ self._components.T
        latent_residual = observed - predicted
        scores = np.sqrt(np.mean(np.square(latent_residual), axis=1))
        evidence = np.abs(latent_residual @ self._components)
        return DetectionScores(scores, evidence, self.threshold)

    def to_dict(self) -> Dict[str, Any]:
        artifact = self._base_artifact()
        artifact.update(
            {
                "latent_dimension": self.latent_dimension,
                "ridge": self.ridge,
                "target_center": self._target_center.tolist(),
                "components": self._components.tolist(),
                "coefficients": self._coefficients.tolist(),
                "context_shape": list(self._context_shape),
            }
        )
        return artifact

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "LatentPredictiveDetector":
        detector = cls(
            latent_dimension=int(payload["latent_dimension"]),
            ridge=float(payload["ridge"]),
            calibration_quantile=float(payload["calibration_quantile"]),
        )
        _restore_latent_state(detector, payload)
        return detector


class CoherentLatentPredictiveDetector(LatentPredictiveDetector):
    """Require multi-feature disagreement with a decoded latent prediction.

    A low-rank latent predictor captures shared normal dynamics. Its prediction is
    decoded back to feature space, where the ``consensus_rank``-largest residual
    is scored. With rank two, one isolated corrupt feature cannot raise the
    anomaly score by itself.
    """

    kind = "coherent_latent_predictive"

    def __init__(
        self,
        latent_dimension: int = 2,
        ridge: float = 1e-2,
        calibration_quantile: float = 0.99,
        consensus_rank: int = 2,
    ) -> None:
        super().__init__(latent_dimension, ridge, calibration_quantile)
        if consensus_rank < 1:
            raise ValueError("consensus_rank must be positive")
        self.consensus_rank = consensus_rank
        self._residual_scale: NDArray[np.float64] = np.asarray([])

    def _fit_model(self, windows: ModelWindows) -> None:
        if self.consensus_rank > windows.targets.shape[1]:
            raise ValueError("consensus_rank cannot exceed the feature count")
        super()._fit_model(windows)
        training_residual = np.abs(self._decoded_difference(windows))
        residual_scale = np.median(training_residual, axis=0)
        self._residual_scale = np.where(
            residual_scale > 1e-6, residual_scale, 1.0
        )

    def _score_model(self, windows: ModelWindows) -> DetectionScores:
        if windows.contexts.shape[1:] != self._context_shape:
            raise ValueError("window shape does not match fitted latent detector")
        signed_evidence = self._decoded_difference(windows) / self._residual_scale
        evidence = np.abs(signed_evidence)
        sorted_evidence = np.sort(evidence, axis=1)
        scores = sorted_evidence[:, -self.consensus_rank]
        return DetectionScores(
            scores,
            evidence,
            self.threshold,
            signed_feature_evidence=signed_evidence,
        )

    def _decoded_difference(
        self, windows: ModelWindows
    ) -> NDArray[np.float64]:
        predicted_latent = _design_matrix(windows.contexts) @ self._coefficients
        predicted_features = (
            predicted_latent @ self._components + self._target_center
        )
        return windows.targets - predicted_features

    def to_dict(self) -> Dict[str, Any]:
        artifact = super().to_dict()
        artifact["consensus_rank"] = self.consensus_rank
        artifact["residual_scale"] = self._residual_scale.tolist()
        return artifact

    @classmethod
    def from_dict(
        cls, payload: Dict[str, Any]
    ) -> "CoherentLatentPredictiveDetector":
        detector = cls(
            latent_dimension=int(payload["latent_dimension"]),
            ridge=float(payload["ridge"]),
            calibration_quantile=float(payload["calibration_quantile"]),
            consensus_rank=int(payload["consensus_rank"]),
        )
        _restore_latent_state(detector, payload)
        detector._residual_scale = np.asarray(
            payload["residual_scale"], dtype=np.float64
        )
        return detector


class DemandConditionedCoherentDetector(
    CoherentLatentPredictiveDetector
):
    """Coherent detector retaining sensitivity to stable completion ratios."""

    kind = "demand_conditioned_coherent_predictive"

    def __init__(
        self,
        latent_dimension: int = 1,
        ridge: float = 1e-2,
        calibration_quantile: float = 0.98,
        consensus_rank: int = 2,
        residual_scale_floor: float = 1e-3,
    ) -> None:
        super().__init__(
            latent_dimension=latent_dimension,
            ridge=ridge,
            calibration_quantile=calibration_quantile,
            consensus_rank=consensus_rank,
        )
        if residual_scale_floor <= 0.0:
            raise ValueError("residual_scale_floor must be positive")
        self.residual_scale_floor = residual_scale_floor

    def _fit_model(self, windows: ModelWindows) -> None:
        super()._fit_model(windows)
        training_residual = np.abs(self._decoded_difference(windows))
        self._residual_scale = np.maximum(
            np.median(training_residual, axis=0),
            self.residual_scale_floor,
        )

    def to_dict(self) -> Dict[str, Any]:
        artifact = super().to_dict()
        artifact["residual_scale_floor"] = self.residual_scale_floor
        return artifact

    @classmethod
    def from_dict(
        cls, payload: Dict[str, Any]
    ) -> "DemandConditionedCoherentDetector":
        detector = cls(
            latent_dimension=int(payload["latent_dimension"]),
            ridge=float(payload["ridge"]),
            calibration_quantile=float(payload["calibration_quantile"]),
            consensus_rank=int(payload["consensus_rank"]),
            residual_scale_floor=float(payload["residual_scale_floor"]),
        )
        _restore_latent_state(detector, payload)
        detector._residual_scale = np.asarray(
            payload["residual_scale"], dtype=np.float64
        )
        return detector


def _restore_latent_state(
    detector: LatentPredictiveDetector, payload: Dict[str, Any]
) -> None:
    detector.threshold = float(payload["threshold"])
    detector._target_center = np.asarray(
        payload["target_center"], dtype=np.float64
    )
    detector._components = np.asarray(payload["components"], dtype=np.float64)
    detector._coefficients = np.asarray(
        payload["coefficients"], dtype=np.float64
    )
    context_shape = payload["context_shape"]
    detector._context_shape = (int(context_shape[0]), int(context_shape[1]))


def detector_from_dict(payload: Dict[str, Any]) -> _CalibratedDetector:
    """Restore a fitted detector from a versioned artifact."""

    if payload.get("schema_version") != 1:
        raise ValueError("unsupported detector schema_version")
    kind = str(payload.get("kind"))
    if kind == PersistenceDetector.kind:
        return PersistenceDetector.from_dict(payload)
    if kind == RobustFeatureDetector.kind:
        return RobustFeatureDetector.from_dict(payload)
    if kind == LatentPredictiveDetector.kind:
        return LatentPredictiveDetector.from_dict(payload)
    if kind == CoherentLatentPredictiveDetector.kind:
        return CoherentLatentPredictiveDetector.from_dict(payload)
    if kind == DemandConditionedCoherentDetector.kind:
        return DemandConditionedCoherentDetector.from_dict(payload)
    raise ValueError(f"unsupported detector kind: {kind}")


def _design_matrix(contexts: NDArray[np.float64]) -> NDArray[np.float64]:
    flattened = contexts.reshape(len(contexts), -1)
    return np.column_stack((flattened, np.ones(len(flattened), dtype=np.float64)))


def _validate_windows(windows: ModelWindows) -> None:
    if windows.contexts.ndim != 3 or windows.targets.ndim != 2:
        raise ValueError("windows must contain 3D contexts and 2D targets")
    if len(windows.contexts) != len(windows.targets):
        raise ValueError("window contexts and targets must have equal length")
    if windows.contexts.shape[2] != windows.targets.shape[1]:
        raise ValueError("context and target feature counts must match")
    if len(windows.targets) == 0:
        raise ValueError("detector requires at least one window")
    if not np.all(np.isfinite(windows.contexts)) or not np.all(
        np.isfinite(windows.targets)
    ):
        raise ValueError("windows must be finite")
