"""A two-encoder JEPA for aligned metrics and structured application logs."""

from typing import Any, Dict, Tuple

import numpy as np
from numpy.typing import NDArray

from .detectors import DetectionScores, _orthonormal_columns
from .multimodal_corpus import MultimodalModelWindows
from .windowing import ModelWindows


class MultimodalJepaWorldModelDetector:
    """Predict one joint future latent from separate metric and log encoders."""

    kind = "multimodal_jepa_world_model_v0"

    def __init__(
        self,
        metric_latent_dimension: int = 3,
        log_latent_dimension: int = 2,
        epochs: int = 200,
        learning_rate: float = 2e-2,
        ema_decay: float = 0.98,
        weight_decay: float = 1e-4,
        calibration_quantile: float = 0.98,
        seed: int = 0,
    ) -> None:
        if metric_latent_dimension < 1:
            raise ValueError(
                "metric_latent_dimension must be positive"
            )
        if log_latent_dimension < 1:
            raise ValueError("log_latent_dimension must be positive")
        if epochs < 1:
            raise ValueError("epochs must be positive")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        if weight_decay < 0.0:
            raise ValueError("weight_decay cannot be negative")
        if not 0.5 < calibration_quantile < 1.0:
            raise ValueError(
                "calibration_quantile must be between 0.5 and 1.0"
            )
        self.metric_latent_dimension = metric_latent_dimension
        self.log_latent_dimension = log_latent_dimension
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.ema_decay = ema_decay
        self.weight_decay = weight_decay
        self.calibration_quantile = calibration_quantile
        self.seed = seed
        self.threshold = float("nan")
        self.training_losses: Tuple[float, ...] = ()
        self._metric_context_shape = (0, 0)
        self._log_context_shape = (0, 0)
        self._metric_feature_names: Tuple[str, ...] = ()
        self._log_feature_names: Tuple[str, ...] = ()
        self._metric_encoder_weights = _empty_matrix()
        self._metric_encoder_bias = _empty_vector()
        self._metric_target_weights = _empty_matrix()
        self._metric_target_bias = _empty_vector()
        self._log_encoder_weights = _empty_matrix()
        self._log_encoder_bias = _empty_vector()
        self._log_target_weights = _empty_matrix()
        self._log_target_bias = _empty_vector()
        self._predictor_weights = _empty_matrix()
        self._predictor_bias = _empty_vector()
        self._metric_feature_scale = _empty_vector()
        self._log_feature_scale = _empty_vector()

    def fit(
        self,
        windows: MultimodalModelWindows,
    ) -> "MultimodalJepaWorldModelDetector":
        _validate_multimodal_windows(windows)
        self._fit_model(windows)
        result = self._score_model(windows)
        self.threshold = float(
            np.quantile(
                result.scores,
                self.calibration_quantile,
            )
        )
        return self

    def score(
        self,
        windows: MultimodalModelWindows,
    ) -> DetectionScores:
        _validate_multimodal_windows(windows)
        if not np.isfinite(self.threshold):
            raise RuntimeError("detector must be fitted before scoring")
        result = self._score_model(windows)
        return DetectionScores(
            scores=result.scores,
            feature_evidence=result.feature_evidence,
            threshold=self.threshold,
            signed_feature_evidence=(
                result.signed_feature_evidence
            ),
        )

    def _fit_model(
        self,
        windows: MultimodalModelWindows,
    ) -> None:
        sample_count, lookback, metric_feature_count = (
            windows.metric.contexts.shape
        )
        log_feature_count = windows.logs.contexts.shape[2]
        if self.metric_latent_dimension > metric_feature_count:
            raise ValueError(
                "metric latent dimension cannot exceed metric features"
            )
        if self.log_latent_dimension > log_feature_count:
            raise ValueError(
                "log latent dimension cannot exceed log features"
            )
        self._metric_context_shape = (
            lookback,
            metric_feature_count,
        )
        self._log_context_shape = (
            lookback,
            log_feature_count,
        )
        self._metric_feature_names = windows.metric.feature_names
        self._log_feature_names = windows.logs.feature_names
        generator = np.random.default_rng(self.seed)
        self._metric_encoder_weights = _orthonormal_columns(
            generator.normal(
                0.0,
                1.0 / np.sqrt(metric_feature_count),
                size=(
                    metric_feature_count,
                    self.metric_latent_dimension,
                ),
            )
        )
        self._log_encoder_weights = _orthonormal_columns(
            generator.normal(
                0.0,
                1.0 / np.sqrt(log_feature_count),
                size=(
                    log_feature_count,
                    self.log_latent_dimension,
                ),
            )
        )
        self._metric_encoder_bias = np.zeros(
            self.metric_latent_dimension,
            dtype=np.float64,
        )
        self._log_encoder_bias = np.zeros(
            self.log_latent_dimension,
            dtype=np.float64,
        )
        self._metric_target_weights = (
            self._metric_encoder_weights.copy()
        )
        self._metric_target_bias = (
            self._metric_encoder_bias.copy()
        )
        self._log_target_weights = (
            self._log_encoder_weights.copy()
        )
        self._log_target_bias = self._log_encoder_bias.copy()
        joint_dimension = (
            self.metric_latent_dimension
            + self.log_latent_dimension
        )
        self._predictor_weights = generator.normal(
            0.0,
            1.0 / np.sqrt(lookback * joint_dimension),
            size=(
                lookback * joint_dimension,
                joint_dimension,
            ),
        )
        self._predictor_bias = np.zeros(
            joint_dimension,
            dtype=np.float64,
        )

        losses = []
        normalizer = float(sample_count * joint_dimension)
        for _ in range(self.epochs):
            metric_context, log_context = self._encoded_context(
                windows
            )
            joint_context = np.concatenate(
                (metric_context, log_context),
                axis=2,
            )
            flattened = joint_context.reshape(sample_count, -1)
            predicted = (
                flattened @ self._predictor_weights
                + self._predictor_bias
            )
            observed = self._encoded_targets(windows)
            residual = predicted - observed
            losses.append(float(np.mean(np.square(residual))))

            prediction_gradient = 2.0 * residual / normalizer
            predictor_weights_before = (
                self._predictor_weights.copy()
            )
            predictor_gradient = (
                flattened.T @ prediction_gradient
                + self.weight_decay * self._predictor_weights
            )
            predictor_bias_gradient = np.sum(
                prediction_gradient,
                axis=0,
            )
            joint_context_gradient = (
                prediction_gradient
                @ predictor_weights_before.T
            ).reshape(joint_context.shape)
            metric_context_gradient = joint_context_gradient[
                :, :, : self.metric_latent_dimension
            ]
            log_context_gradient = joint_context_gradient[
                :, :, self.metric_latent_dimension :
            ]
            metric_pre_activation_gradient = (
                metric_context_gradient
                * (1.0 - np.square(metric_context))
            )
            log_pre_activation_gradient = (
                log_context_gradient
                * (1.0 - np.square(log_context))
            )
            metric_encoder_gradient = (
                np.einsum(
                    "nlf,nld->fd",
                    windows.metric.contexts,
                    metric_pre_activation_gradient,
                )
                + self.weight_decay
                * self._metric_encoder_weights
            )
            log_encoder_gradient = (
                np.einsum(
                    "nlf,nld->fd",
                    windows.logs.contexts,
                    log_pre_activation_gradient,
                )
                + self.weight_decay * self._log_encoder_weights
            )

            self._predictor_weights -= (
                self.learning_rate * predictor_gradient
            )
            self._predictor_bias -= (
                self.learning_rate * predictor_bias_gradient
            )
            self._metric_encoder_weights = _orthonormal_columns(
                self._metric_encoder_weights
                - self.learning_rate * metric_encoder_gradient
            )
            self._log_encoder_weights = _orthonormal_columns(
                self._log_encoder_weights
                - self.learning_rate * log_encoder_gradient
            )
            self._metric_encoder_bias -= self.learning_rate * np.sum(
                metric_pre_activation_gradient,
                axis=(0, 1),
            )
            self._log_encoder_bias -= self.learning_rate * np.sum(
                log_pre_activation_gradient,
                axis=(0, 1),
            )
            self._update_target_encoders()
        self.training_losses = tuple(losses)
        metric_evidence, log_evidence = self._raw_feature_evidence(
            windows
        )
        self._metric_feature_scale = np.maximum(
            np.median(np.abs(metric_evidence), axis=0),
            1e-3,
        )
        self._log_feature_scale = np.maximum(
            np.median(np.abs(log_evidence), axis=0),
            1e-3,
        )

    def _score_model(
        self,
        windows: MultimodalModelWindows,
    ) -> DetectionScores:
        self._validate_fitted_schema(windows)
        latent_residual, _ = self._latent_difference(windows)
        metric_evidence, log_evidence = self._raw_feature_evidence(
            windows
        )
        signed_evidence = np.concatenate(
            (
                metric_evidence / self._metric_feature_scale,
                log_evidence / self._log_feature_scale,
            ),
            axis=1,
        )
        return DetectionScores(
            scores=np.sqrt(
                np.mean(np.square(latent_residual), axis=1)
            ),
            feature_evidence=np.abs(signed_evidence),
            threshold=self.threshold,
            signed_feature_evidence=signed_evidence,
        )

    def _encoded_context(
        self,
        windows: MultimodalModelWindows,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        return (
            np.tanh(
                windows.metric.contexts
                @ self._metric_encoder_weights
                + self._metric_encoder_bias
            ),
            np.tanh(
                windows.logs.contexts @ self._log_encoder_weights
                + self._log_encoder_bias
            ),
        )

    def _encoded_targets(
        self,
        windows: MultimodalModelWindows,
    ) -> NDArray[np.float64]:
        return np.concatenate(
            (
                np.tanh(
                    windows.metric.targets
                    @ self._metric_target_weights
                    + self._metric_target_bias
                ),
                np.tanh(
                    windows.logs.targets
                    @ self._log_target_weights
                    + self._log_target_bias
                ),
            ),
            axis=1,
        )

    def _latent_difference(
        self,
        windows: MultimodalModelWindows,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        metric_context, log_context = self._encoded_context(windows)
        joint_context = np.concatenate(
            (metric_context, log_context),
            axis=2,
        )
        predicted = (
            joint_context.reshape(len(joint_context), -1)
            @ self._predictor_weights
            + self._predictor_bias
        )
        observed = self._encoded_targets(windows)
        return observed - predicted, observed

    def _raw_feature_evidence(
        self,
        windows: MultimodalModelWindows,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        latent_residual, observed = self._latent_difference(windows)
        metric_residual = latent_residual[
            :, : self.metric_latent_dimension
        ]
        log_residual = latent_residual[
            :, self.metric_latent_dimension :
        ]
        metric_observed = observed[
            :, : self.metric_latent_dimension
        ]
        log_observed = observed[
            :, self.metric_latent_dimension :
        ]
        return (
            (
                metric_residual
                * (1.0 - np.square(metric_observed))
            )
            @ self._metric_target_weights.T,
            (
                log_residual
                * (1.0 - np.square(log_observed))
            )
            @ self._log_target_weights.T,
        )

    def _update_target_encoders(self) -> None:
        online_weight = 1.0 - self.ema_decay
        self._metric_target_weights = (
            self.ema_decay * self._metric_target_weights
            + online_weight * self._metric_encoder_weights
        )
        self._metric_target_bias = (
            self.ema_decay * self._metric_target_bias
            + online_weight * self._metric_encoder_bias
        )
        self._log_target_weights = (
            self.ema_decay * self._log_target_weights
            + online_weight * self._log_encoder_weights
        )
        self._log_target_bias = (
            self.ema_decay * self._log_target_bias
            + online_weight * self._log_encoder_bias
        )

    def _validate_fitted_schema(
        self,
        windows: MultimodalModelWindows,
    ) -> None:
        if (
            windows.metric.contexts.shape[1:]
            != self._metric_context_shape
            or windows.logs.contexts.shape[1:]
            != self._log_context_shape
        ):
            raise ValueError(
                "window shapes do not match fitted multimodal detector"
            )
        if (
            windows.metric.feature_names
            != self._metric_feature_names
            or windows.logs.feature_names != self._log_feature_names
        ):
            raise ValueError(
                "window features do not match fitted multimodal detector"
            )

    def to_dict(self) -> Dict[str, Any]:
        if not np.isfinite(self.threshold):
            raise RuntimeError(
                "detector must be fitted before serialization"
            )
        return {
            "schema_version": 1,
            "kind": self.kind,
            "metric_latent_dimension": (
                self.metric_latent_dimension
            ),
            "log_latent_dimension": self.log_latent_dimension,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "ema_decay": self.ema_decay,
            "weight_decay": self.weight_decay,
            "calibration_quantile": self.calibration_quantile,
            "seed": self.seed,
            "threshold": self.threshold,
            "training_losses": list(self.training_losses),
            "metric_context_shape": list(
                self._metric_context_shape
            ),
            "log_context_shape": list(self._log_context_shape),
            "metric_feature_names": list(
                self._metric_feature_names
            ),
            "log_feature_names": list(self._log_feature_names),
            "metric_encoder_weights": (
                self._metric_encoder_weights.tolist()
            ),
            "metric_encoder_bias": (
                self._metric_encoder_bias.tolist()
            ),
            "metric_target_weights": (
                self._metric_target_weights.tolist()
            ),
            "metric_target_bias": (
                self._metric_target_bias.tolist()
            ),
            "log_encoder_weights": (
                self._log_encoder_weights.tolist()
            ),
            "log_encoder_bias": self._log_encoder_bias.tolist(),
            "log_target_weights": (
                self._log_target_weights.tolist()
            ),
            "log_target_bias": self._log_target_bias.tolist(),
            "predictor_weights": self._predictor_weights.tolist(),
            "predictor_bias": self._predictor_bias.tolist(),
            "metric_feature_scale": (
                self._metric_feature_scale.tolist()
            ),
            "log_feature_scale": self._log_feature_scale.tolist(),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Dict[str, Any],
    ) -> "MultimodalJepaWorldModelDetector":
        if (
            payload.get("schema_version") != 1
            or payload.get("kind") != cls.kind
        ):
            raise ValueError(
                "unsupported multimodal JEPA detector artifact"
            )
        detector = cls(
            metric_latent_dimension=int(
                payload["metric_latent_dimension"]
            ),
            log_latent_dimension=int(
                payload["log_latent_dimension"]
            ),
            epochs=int(payload["epochs"]),
            learning_rate=float(payload["learning_rate"]),
            ema_decay=float(payload["ema_decay"]),
            weight_decay=float(payload["weight_decay"]),
            calibration_quantile=float(
                payload["calibration_quantile"]
            ),
            seed=int(payload["seed"]),
        )
        detector.threshold = float(payload["threshold"])
        detector.training_losses = tuple(
            float(value) for value in payload["training_losses"]
        )
        detector._metric_context_shape = _shape(
            payload["metric_context_shape"]
        )
        detector._log_context_shape = _shape(
            payload["log_context_shape"]
        )
        detector._metric_feature_names = tuple(
            str(value)
            for value in payload["metric_feature_names"]
        )
        detector._log_feature_names = tuple(
            str(value) for value in payload["log_feature_names"]
        )
        detector._metric_encoder_weights = _matrix(
            payload["metric_encoder_weights"]
        )
        detector._metric_encoder_bias = _vector(
            payload["metric_encoder_bias"]
        )
        detector._metric_target_weights = _matrix(
            payload["metric_target_weights"]
        )
        detector._metric_target_bias = _vector(
            payload["metric_target_bias"]
        )
        detector._log_encoder_weights = _matrix(
            payload["log_encoder_weights"]
        )
        detector._log_encoder_bias = _vector(
            payload["log_encoder_bias"]
        )
        detector._log_target_weights = _matrix(
            payload["log_target_weights"]
        )
        detector._log_target_bias = _vector(
            payload["log_target_bias"]
        )
        detector._predictor_weights = _matrix(
            payload["predictor_weights"]
        )
        detector._predictor_bias = _vector(
            payload["predictor_bias"]
        )
        detector._metric_feature_scale = _vector(
            payload["metric_feature_scale"]
        )
        detector._log_feature_scale = _vector(
            payload["log_feature_scale"]
        )
        return detector


def _validate_multimodal_windows(
    windows: MultimodalModelWindows,
) -> None:
    _validate_channel(windows.metric)
    _validate_channel(windows.logs)


def _validate_channel(windows: ModelWindows) -> None:
    if windows.contexts.ndim != 3 or windows.targets.ndim != 2:
        raise ValueError(
            "windows must contain 3D contexts and 2D targets"
        )
    if len(windows.contexts) != len(windows.targets):
        raise ValueError(
            "window contexts and targets must have equal length"
        )
    if windows.contexts.shape[2] != windows.targets.shape[1]:
        raise ValueError(
            "context and target feature counts must match"
        )
    if not len(windows.targets):
        raise ValueError("detector requires at least one window")
    if not np.all(np.isfinite(windows.contexts)) or not np.all(
        np.isfinite(windows.targets)
    ):
        raise ValueError("windows must be finite")


def _empty_matrix() -> NDArray[np.float64]:
    return np.asarray([[]], dtype=np.float64)


def _empty_vector() -> NDArray[np.float64]:
    return np.asarray([], dtype=np.float64)


def _matrix(values: Any) -> NDArray[np.float64]:
    return np.asarray(values, dtype=np.float64)


def _vector(values: Any) -> NDArray[np.float64]:
    return np.asarray(values, dtype=np.float64)


def _shape(values: Any) -> Tuple[int, int]:
    return int(values[0]), int(values[1])
