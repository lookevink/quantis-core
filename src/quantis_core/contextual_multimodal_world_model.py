"""Conditioned contextual JEPA for metrics and structured application logs."""

from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .contextual_multimodal_corpus import (
    ContextualMultimodalModelWindows,
)
from .detectors import DetectionScores, _orthonormal_columns


SUPPORTED_LOSSES = frozenset({"huber", "l1", "mse"})
AUXILIARY_OBJECTIVES = (
    "metric_to_metric",
    "log_to_log",
    "metric_to_log",
    "log_to_metric",
)


class ContextualMultimodalJepaWorldModelDetector:
    """Predict contextual future embeddings with observable controls."""

    kind = "contextual_multimodal_jepa_world_model_v1"

    def __init__(
        self,
        metric_latent_dimension: int = 3,
        log_latent_dimension: int = 1,
        pretraining_epochs: int = 200,
        predictor_refinement_epochs: int = 100,
        learning_rate: float = 2e-2,
        ema_decay: float = 0.98,
        weight_decay: float = 1e-4,
        loss: str = "huber",
        huber_delta: float = 1.0,
        auxiliary_loss_weight: float = 0.2,
        rollout_loss_weight: float = 0.2,
        modality_mask_probability: float = 0.0,
        log_self_loss_multiplier: float = 1.0,
        cross_modal_loss_multiplier: float = 1.0,
        calibration_quantile: float = 0.98,
        seed: int = 0,
    ) -> None:
        if metric_latent_dimension < 0:
            raise ValueError(
                "metric_latent_dimension cannot be negative"
            )
        if log_latent_dimension < 0:
            raise ValueError(
                "log_latent_dimension cannot be negative"
            )
        if metric_latent_dimension + log_latent_dimension < 1:
            raise ValueError(
                "at least one modality must have a latent dimension"
            )
        if pretraining_epochs < 1:
            raise ValueError("pretraining_epochs must be positive")
        if predictor_refinement_epochs < 0:
            raise ValueError(
                "predictor_refinement_epochs cannot be negative"
            )
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        if weight_decay < 0.0:
            raise ValueError("weight_decay cannot be negative")
        if loss not in SUPPORTED_LOSSES:
            raise ValueError(
                f"loss must be one of {sorted(SUPPORTED_LOSSES)}"
            )
        if huber_delta <= 0.0:
            raise ValueError("huber_delta must be positive")
        if auxiliary_loss_weight < 0.0:
            raise ValueError(
                "auxiliary_loss_weight cannot be negative"
            )
        if rollout_loss_weight < 0.0:
            raise ValueError(
                "rollout_loss_weight cannot be negative"
            )
        if not 0.0 <= modality_mask_probability < 0.5:
            raise ValueError(
                "modality_mask_probability must be in [0, 0.5)"
            )
        if log_self_loss_multiplier < 0.0:
            raise ValueError(
                "log_self_loss_multiplier cannot be negative"
            )
        if cross_modal_loss_multiplier < 0.0:
            raise ValueError(
                "cross_modal_loss_multiplier cannot be negative"
            )
        if not 0.5 < calibration_quantile < 1.0:
            raise ValueError(
                "calibration_quantile must be between 0.5 and 1.0"
            )
        self.metric_latent_dimension = metric_latent_dimension
        self.log_latent_dimension = log_latent_dimension
        self.pretraining_epochs = pretraining_epochs
        self.predictor_refinement_epochs = (
            predictor_refinement_epochs
        )
        self.learning_rate = learning_rate
        self.ema_decay = ema_decay
        self.weight_decay = weight_decay
        self.loss = loss
        self.huber_delta = huber_delta
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.rollout_loss_weight = rollout_loss_weight
        self.modality_mask_probability = (
            modality_mask_probability
        )
        self.log_self_loss_multiplier = log_self_loss_multiplier
        self.cross_modal_loss_multiplier = (
            cross_modal_loss_multiplier
        )
        self.calibration_quantile = calibration_quantile
        self.seed = seed
        self.threshold = float("nan")
        self.training_losses: Tuple[float, ...] = ()
        self.diagnostics: Mapping[str, Any] = {}
        self._context_shape = (0, 0, 0)
        self._target_block_size = 0
        self._horizons: Tuple[int, ...] = ()
        self._metric_feature_names: Tuple[str, ...] = ()
        self._log_feature_names: Tuple[str, ...] = ()
        self._control_feature_names: Tuple[str, ...] = ()
        self._metric_encoder_weights = _empty_matrix()
        self._metric_encoder_bias = _empty_vector()
        self._metric_target_weights = _empty_matrix()
        self._metric_target_bias = _empty_vector()
        self._log_encoder_weights = _empty_matrix()
        self._log_encoder_bias = _empty_vector()
        self._log_target_weights = _empty_matrix()
        self._log_target_bias = _empty_vector()
        self._predictor_input_weights = _empty_matrix()
        self._predictor_hidden_bias = _empty_vector()
        self._predictor_output_weights = _empty_matrix()
        self._predictor_output_bias = _empty_vector()
        self._head_weights: Dict[str, NDArray[np.float64]] = {}
        self._head_biases: Dict[str, NDArray[np.float64]] = {}
        self._metric_energy_scale = float("nan")
        self._log_energy_scale = float("nan")
        self._metric_feature_scale = _empty_vector()
        self._log_feature_scale = _empty_vector()
        self._last_auxiliary_losses: Dict[str, float] = {}

    def fit(
        self,
        windows: ContextualMultimodalModelWindows,
    ) -> "ContextualMultimodalJepaWorldModelDetector":
        self._initialize(windows)
        losses = []
        total_epochs = (
            self.pretraining_epochs
            + self.predictor_refinement_epochs
        )
        for epoch in range(total_epochs):
            update_encoders = epoch < self.pretraining_epochs
            losses.append(
                self._train_epoch(
                    windows,
                    update_encoders,
                    epoch,
                )
            )
        self.training_losses = tuple(losses)
        (
            metric_energy,
            log_energy,
            metric_evidence,
            log_evidence,
        ) = self._raw_score_components(
            windows,
            include_metric_context=True,
            include_log_context=True,
        )
        self._metric_energy_scale = max(
            float(np.median(metric_energy)),
            1e-3,
        )
        self._log_energy_scale = max(
            float(np.median(log_energy)),
            1e-3,
        )
        self._metric_feature_scale = np.maximum(
            np.median(np.abs(metric_evidence), axis=0),
            1e-3,
        )
        self._log_feature_scale = np.maximum(
            np.median(np.abs(log_evidence), axis=0),
            1e-3,
        )
        training_scores = self._score_model(
            windows,
            include_metric_context=True,
            include_log_context=True,
        )
        self.threshold = float(
            np.quantile(
                training_scores.scores,
                self.calibration_quantile,
            )
        )
        self.diagnostics = self._representation_diagnostics(windows)
        return self

    def score(
        self,
        windows: ContextualMultimodalModelWindows,
    ) -> DetectionScores:
        return self.score_with_context(
            windows,
            include_metric_context=True,
            include_log_context=True,
        )

    def score_with_context(
        self,
        windows: ContextualMultimodalModelWindows,
        *,
        include_metric_context: bool,
        include_log_context: bool,
    ) -> DetectionScores:
        """Score an explicit modality-dropout ablation."""

        self._validate_fitted(windows)
        if not np.isfinite(self.threshold):
            raise RuntimeError("detector must be fitted before scoring")
        return self._score_model(
            windows,
            include_metric_context,
            include_log_context,
        )

    def _initialize(
        self,
        windows: ContextualMultimodalModelWindows,
    ) -> None:
        _validate_windows(windows)
        _, lookback, metric_features = (
            windows.metric_contexts.shape
        )
        log_features = windows.log_contexts.shape[2]
        block_size = windows.target_block_size
        metric_block_features = block_size * metric_features
        log_block_features = block_size * log_features
        if self.metric_latent_dimension > metric_block_features:
            raise ValueError(
                "metric latent dimension cannot exceed block features"
            )
        if self.log_latent_dimension > log_block_features:
            raise ValueError(
                "log latent dimension cannot exceed block features"
            )
        if lookback % block_size != 0:
            raise ValueError(
                "context lookback must contain complete target-sized blocks"
            )
        patch_count = lookback // block_size
        self._context_shape = (
            lookback,
            metric_features,
            log_features,
        )
        self._target_block_size = block_size
        self._horizons = windows.horizons
        self._metric_feature_names = windows.metric_feature_names
        self._log_feature_names = windows.log_feature_names
        self._control_feature_names = (
            windows.control_feature_names
        )
        generator = np.random.default_rng(self.seed)
        self._metric_encoder_weights = (
            _orthonormal_columns(
                generator.normal(
                    0.0,
                    1.0 / np.sqrt(metric_block_features),
                    size=(
                        metric_block_features,
                        self.metric_latent_dimension,
                    ),
                )
            )
            if self.metric_latent_dimension > 0
            else np.empty(
                (metric_block_features, 0),
                dtype=np.float64,
            )
        )
        self._log_encoder_weights = (
            _orthonormal_columns(
                generator.normal(
                    0.0,
                    1.0 / np.sqrt(log_block_features),
                    size=(
                        log_block_features,
                        self.log_latent_dimension,
                    ),
                )
            )
            if self.log_latent_dimension > 0
            else np.empty(
                (log_block_features, 0),
                dtype=np.float64,
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
        self._log_target_weights = self._log_encoder_weights.copy()
        self._log_target_bias = self._log_encoder_bias.copy()
        joint_dimension = (
            self.metric_latent_dimension
            + self.log_latent_dimension
        )
        condition_dimension = (
            block_size * len(windows.control_feature_names)
            + len(windows.horizons)
        )
        predictor_input_dimension = (
            patch_count * joint_dimension + condition_dimension
        )
        hidden_dimension = max(8, 2 * joint_dimension)
        self._predictor_input_weights = generator.normal(
            0.0,
            1.0 / np.sqrt(predictor_input_dimension),
            size=(predictor_input_dimension, hidden_dimension),
        )
        self._predictor_hidden_bias = np.zeros(
            hidden_dimension,
            dtype=np.float64,
        )
        self._predictor_output_weights = generator.normal(
            0.0,
            1.0 / np.sqrt(hidden_dimension),
            size=(hidden_dimension, joint_dimension),
        )
        self._predictor_output_bias = np.zeros(
            joint_dimension,
            dtype=np.float64,
        )
        metric_input_dimension = (
            patch_count * self.metric_latent_dimension
            + condition_dimension
        )
        log_input_dimension = (
            patch_count * self.log_latent_dimension
            + condition_dimension
        )
        head_shapes = {}
        if self.metric_latent_dimension > 0:
            head_shapes["metric_to_metric"] = (
                metric_input_dimension,
                self.metric_latent_dimension,
            )
        if self.log_latent_dimension > 0:
            head_shapes["log_to_log"] = (
                log_input_dimension,
                self.log_latent_dimension,
            )
        if (
            self.metric_latent_dimension > 0
            and self.log_latent_dimension > 0
        ):
            head_shapes["metric_to_log"] = (
                metric_input_dimension,
                self.log_latent_dimension,
            )
            head_shapes["log_to_metric"] = (
                log_input_dimension,
                self.metric_latent_dimension,
            )
        self._head_weights = {
            name: generator.normal(
                0.0,
                1.0 / np.sqrt(input_dimension),
                size=(input_dimension, output_dimension),
            )
            for name, (
                input_dimension,
                output_dimension,
            ) in head_shapes.items()
        }
        self._head_biases = {
            name: np.zeros(
                output_dimension,
                dtype=np.float64,
            )
            for name, (_, output_dimension) in head_shapes.items()
        }

    def _train_epoch(
        self,
        windows: ContextualMultimodalModelWindows,
        update_encoders: bool,
        epoch_index: int,
    ) -> float:
        (
            metric_patches,
            log_patches,
            metric_context,
            log_context,
            metric_targets,
            log_targets,
            conditioning,
        ) = self._representations(windows)
        sample_count = len(metric_context)
        horizon_count = len(windows.horizons)
        patch_count = metric_context.shape[1]
        joint_dimension = (
            self.metric_latent_dimension
            + self.log_latent_dimension
        )
        metric_direct = np.broadcast_to(
            metric_context[:, None, :, :],
            (
                sample_count,
                horizon_count,
                patch_count,
                self.metric_latent_dimension,
            ),
        )
        log_direct = np.broadcast_to(
            log_context[:, None, :, :],
            (
                sample_count,
                horizon_count,
                patch_count,
                self.log_latent_dimension,
            ),
        )
        metric_direct_mask, log_direct_mask = (
            self._context_modality_masks(
                sample_count,
                epoch_index,
            )
        )
        metric_direct = metric_direct * metric_direct_mask
        log_direct = log_direct * log_direct_mask
        joint_direct = np.concatenate(
            (metric_direct, log_direct),
            axis=3,
        )
        targets = np.concatenate(
            (metric_targets, log_targets),
            axis=2,
        )
        direct_input = np.concatenate(
            (
                joint_direct.reshape(
                    sample_count * horizon_count,
                    patch_count * joint_dimension,
                ),
                conditioning.reshape(
                    sample_count * horizon_count,
                    -1,
                ),
            ),
            axis=1,
        )
        target_flat = targets.reshape(
            sample_count * horizon_count,
            joint_dimension,
        )
        (
            predicted,
            hidden,
        ) = self._predict(direct_input)
        main_loss, prediction_gradient = _loss_and_gradient(
            predicted - target_flat,
            self.loss,
            self.huber_delta,
        )
        (
            input_weight_gradient,
            hidden_bias_gradient,
            output_weight_gradient,
            output_bias_gradient,
            direct_input_gradient,
        ) = self._predictor_gradients(
            direct_input,
            hidden,
            prediction_gradient,
        )
        joint_context_gradient = direct_input_gradient[
            :, : patch_count * joint_dimension
        ].reshape(
            sample_count,
            horizon_count,
            patch_count,
            joint_dimension,
        )

        auxiliary_loss = 0.0
        metric_context_gradient = np.zeros_like(metric_direct)
        log_context_gradient = np.zeros_like(log_direct)
        head_inputs = {
            "metric": np.concatenate(
                (
                    metric_direct.reshape(
                        sample_count * horizon_count,
                        -1,
                    ),
                    conditioning.reshape(
                        sample_count * horizon_count,
                        -1,
                    ),
                ),
                axis=1,
            ),
            "log": np.concatenate(
                (
                    log_direct.reshape(
                        sample_count * horizon_count,
                        -1,
                    ),
                    conditioning.reshape(
                        sample_count * horizon_count,
                        -1,
                    ),
                ),
                axis=1,
            ),
        }
        head_targets = {
            "metric": metric_targets.reshape(
                sample_count * horizon_count,
                self.metric_latent_dimension,
            ),
            "log": log_targets.reshape(
                sample_count * horizon_count,
                self.log_latent_dimension,
            ),
        }
        head_spec = {
            "metric_to_metric": ("metric", "metric"),
            "log_to_log": ("log", "log"),
            "metric_to_log": ("metric", "log"),
            "log_to_metric": ("log", "metric"),
        }
        auxiliary_losses: Dict[str, float] = {}
        for name in self._active_auxiliary_objectives():
            source, target = head_spec[name]
            head_input = head_inputs[source]
            weights_before = self._head_weights[name].copy()
            head_prediction = (
                head_input @ weights_before
                + self._head_biases[name]
            )
            head_loss, head_gradient = _loss_and_gradient(
                head_prediction - head_targets[target],
                self.loss,
                self.huber_delta,
            )
            auxiliary_losses[name] = head_loss
            objective_multiplier = (
                self._auxiliary_objective_multiplier(name)
            )
            auxiliary_loss += objective_multiplier * head_loss
            scaled_gradient = (
                self.auxiliary_loss_weight
                * objective_multiplier
                * head_gradient
            )
            head_weight_gradient = (
                head_input.T @ scaled_gradient
                + self.weight_decay * weights_before
            )
            head_bias_gradient = np.sum(
                scaled_gradient,
                axis=0,
            )
            head_input_gradient = (
                scaled_gradient @ weights_before.T
            )
            if source == "metric":
                metric_context_gradient += (
                    head_input_gradient[
                        :,
                        : patch_count
                        * self.metric_latent_dimension,
                    ].reshape(metric_direct.shape)
                    * metric_direct_mask
                )
            else:
                log_context_gradient += (
                    head_input_gradient[
                        :,
                        : patch_count
                        * self.log_latent_dimension,
                    ].reshape(log_direct.shape)
                    * log_direct_mask
                )
            self._head_weights[name] -= (
                self.learning_rate * head_weight_gradient
            )
            self._head_biases[name] -= (
                self.learning_rate * head_bias_gradient
            )
        self._last_auxiliary_losses = auxiliary_losses

        rollout_loss = 0.0
        rollout_context_gradient = np.zeros(
            (
                sample_count,
                patch_count,
                joint_dimension,
            ),
            dtype=np.float64,
        )
        rollout_indices = self._rollout_indices()
        if (
            rollout_indices is not None
            and self.rollout_loss_weight > 0.0
        ):
            first_index, second_index = rollout_indices
            predicted_by_horizon = predicted.reshape(
                sample_count,
                horizon_count,
                joint_dimension,
            )
            metric_context_mask = metric_direct_mask[:, 0, :, :]
            log_context_mask = log_direct_mask[:, 0, :, :]
            joint_context = np.concatenate(
                (
                    metric_context * metric_context_mask,
                    log_context * log_context_mask,
                ),
                axis=2,
            )
            rolled_context = np.concatenate(
                (
                    joint_context[:, 1:, :],
                    predicted_by_horizon[
                        :, first_index, :
                    ][:, None, :].copy(),
                ),
                axis=1,
            )
            rollout_input = np.concatenate(
                (
                    rolled_context.reshape(sample_count, -1),
                    conditioning[:, second_index, :],
                ),
                axis=1,
            )
            rollout_prediction, rollout_hidden = self._predict(
                rollout_input
            )
            (
                rollout_loss,
                rollout_prediction_gradient,
            ) = _loss_and_gradient(
                rollout_prediction
                - targets[:, second_index, :],
                self.loss,
                self.huber_delta,
            )
            rollout_prediction_gradient *= (
                self.rollout_loss_weight
            )
            (
                rollout_input_weight_gradient,
                rollout_hidden_bias_gradient,
                rollout_output_weight_gradient,
                rollout_output_bias_gradient,
                rollout_input_gradient,
            ) = self._predictor_gradients(
                rollout_input,
                rollout_hidden,
                rollout_prediction_gradient,
            )
            input_weight_gradient += (
                rollout_input_weight_gradient
            )
            hidden_bias_gradient += rollout_hidden_bias_gradient
            output_weight_gradient += (
                rollout_output_weight_gradient
            )
            output_bias_gradient += rollout_output_bias_gradient
            rolled_gradient = rollout_input_gradient[
                :, : patch_count * joint_dimension
            ].reshape(
                sample_count,
                patch_count,
                joint_dimension,
            )
            rollout_context_gradient[:, 1:, :] = (
                rolled_gradient[:, :-1, :]
            )

        metric_context_gradient += joint_context_gradient[
            :, :, :, : self.metric_latent_dimension
        ] * metric_direct_mask
        log_context_gradient += joint_context_gradient[
            :, :, :, self.metric_latent_dimension :
        ] * log_direct_mask
        metric_base_gradient = np.sum(
            metric_context_gradient,
            axis=1,
        )
        log_base_gradient = np.sum(
            log_context_gradient,
            axis=1,
        )
        metric_base_gradient += rollout_context_gradient[
            :, :, : self.metric_latent_dimension
        ] * metric_direct_mask[:, 0, :, :]
        log_base_gradient += rollout_context_gradient[
            :, :, self.metric_latent_dimension :
        ] * log_direct_mask[:, 0, :, :]

        self._predictor_input_weights -= self.learning_rate * (
            input_weight_gradient
            + self.weight_decay
            * self._predictor_input_weights
        )
        self._predictor_hidden_bias -= (
            self.learning_rate * hidden_bias_gradient
        )
        self._predictor_output_weights -= self.learning_rate * (
            output_weight_gradient
            + self.weight_decay
            * self._predictor_output_weights
        )
        self._predictor_output_bias -= (
            self.learning_rate * output_bias_gradient
        )
        if update_encoders:
            metric_pre_activation_gradient = (
                metric_base_gradient
                * (1.0 - np.square(metric_context))
            )
            log_pre_activation_gradient = (
                log_base_gradient
                * (1.0 - np.square(log_context))
            )
            metric_encoder_gradient = (
                np.einsum(
                    "npf,npd->fd",
                    metric_patches,
                    metric_pre_activation_gradient,
                )
                + self.weight_decay
                * self._metric_encoder_weights
            )
            log_encoder_gradient = (
                np.einsum(
                    "npf,npd->fd",
                    log_patches,
                    log_pre_activation_gradient,
                )
                + self.weight_decay * self._log_encoder_weights
            )
            if self.metric_latent_dimension > 0:
                self._metric_encoder_weights = (
                    _orthonormal_columns(
                        self._metric_encoder_weights
                        - self.learning_rate
                        * metric_encoder_gradient
                    )
                )
            if self.log_latent_dimension > 0:
                self._log_encoder_weights = (
                    _orthonormal_columns(
                        self._log_encoder_weights
                        - self.learning_rate
                        * log_encoder_gradient
                    )
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
        return (
            main_loss
            + self.auxiliary_loss_weight * auxiliary_loss
            + self.rollout_loss_weight * rollout_loss
        )

    def _active_auxiliary_objectives(self) -> Tuple[str, ...]:
        return tuple(
            name
            for name in AUXILIARY_OBJECTIVES
            if name in self._head_weights
        )

    def _auxiliary_objective_multiplier(self, name: str) -> float:
        if name == "log_to_log":
            return self.log_self_loss_multiplier
        if name in ("metric_to_log", "log_to_metric"):
            return self.cross_modal_loss_multiplier
        return 1.0

    def _context_modality_masks(
        self,
        sample_count: int,
        epoch_index: int,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        shape = (sample_count, 1, 1, 1)
        metric_mask = np.ones(shape, dtype=np.float64)
        log_mask = np.ones(shape, dtype=np.float64)
        if (
            self.modality_mask_probability == 0.0
            or self.metric_latent_dimension == 0
            or self.log_latent_dimension == 0
        ):
            return metric_mask, log_mask
        generator = np.random.default_rng(
            self.seed + 10_007 * (epoch_index + 1)
        )
        selection = generator.random(sample_count)
        probability = self.modality_mask_probability
        metric_mask[selection < probability] = 0.0
        log_mask[
            (selection >= probability)
            & (selection < 2.0 * probability)
        ] = 0.0
        return metric_mask, log_mask

    def _representations(
        self,
        windows: ContextualMultimodalModelWindows,
    ) -> Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        block_size = windows.target_block_size
        metric_patches = _temporal_patches(
            windows.metric_contexts,
            block_size,
        )
        log_patches = _temporal_patches(
            windows.log_contexts,
            block_size,
        )
        metric_context = np.tanh(
            metric_patches @ self._metric_encoder_weights
            + self._metric_encoder_bias
        )
        log_context = np.tanh(
            log_patches @ self._log_encoder_weights
            + self._log_encoder_bias
        )
        metric_targets = np.tanh(
            windows.metric_target_blocks.reshape(
                len(windows.metric_target_blocks),
                len(windows.horizons),
                -1,
            )
            @ self._metric_target_weights
            + self._metric_target_bias
        )
        log_targets = np.tanh(
            windows.log_target_blocks.reshape(
                len(windows.log_target_blocks),
                len(windows.horizons),
                -1,
            )
            @ self._log_target_weights
            + self._log_target_bias
        )
        horizon_tokens = np.broadcast_to(
            np.eye(len(windows.horizons), dtype=np.float64)[
                None, :, :
            ],
            (
                len(windows.metric_contexts),
                len(windows.horizons),
                len(windows.horizons),
            ),
        )
        conditioning = np.concatenate(
            (
                windows.target_controls.reshape(
                    len(windows.target_controls),
                    len(windows.horizons),
                    -1,
                ),
                horizon_tokens,
            ),
            axis=2,
        )
        return (
            metric_patches,
            log_patches,
            metric_context,
            log_context,
            metric_targets,
            log_targets,
            conditioning,
        )

    def _predict(
        self,
        values: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        hidden = np.tanh(
            values @ self._predictor_input_weights
            + self._predictor_hidden_bias
        )
        return (
            hidden @ self._predictor_output_weights
            + self._predictor_output_bias,
            hidden,
        )

    def _predictor_gradients(
        self,
        predictor_input: NDArray[np.float64],
        hidden: NDArray[np.float64],
        prediction_gradient: NDArray[np.float64],
    ) -> Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        output_weights_before = (
            self._predictor_output_weights.copy()
        )
        input_weights_before = (
            self._predictor_input_weights.copy()
        )
        output_weight_gradient = (
            hidden.T @ prediction_gradient
        )
        output_bias_gradient = np.sum(
            prediction_gradient,
            axis=0,
        )
        hidden_gradient = (
            prediction_gradient @ output_weights_before.T
        ) * (1.0 - np.square(hidden))
        input_weight_gradient = (
            predictor_input.T @ hidden_gradient
        )
        hidden_bias_gradient = np.sum(
            hidden_gradient,
            axis=0,
        )
        input_gradient = hidden_gradient @ input_weights_before.T
        return (
            input_weight_gradient,
            hidden_bias_gradient,
            output_weight_gradient,
            output_bias_gradient,
            input_gradient,
        )

    def _rollout_indices(self) -> Optional[Tuple[int, int]]:
        second_horizon = 1 + self._target_block_size
        if 1 not in self._horizons or second_horizon not in self._horizons:
            return None
        return (
            self._horizons.index(1),
            self._horizons.index(second_horizon),
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

    def _score_model(
        self,
        windows: ContextualMultimodalModelWindows,
        include_metric_context: bool,
        include_log_context: bool,
    ) -> DetectionScores:
        (
            metric_energy,
            log_energy,
            metric_evidence,
            log_evidence,
        ) = self._raw_score_components(
            windows,
            include_metric_context,
            include_log_context,
        )
        evidence_groups = []
        energy_groups = []
        if self.metric_latent_dimension > 0:
            evidence_groups.append(
                metric_evidence / self._metric_feature_scale
            )
            energy_groups.append(
                metric_energy / self._metric_energy_scale
            )
        if self.log_latent_dimension > 0:
            evidence_groups.append(
                log_evidence / self._log_feature_scale
            )
            energy_groups.append(
                log_energy / self._log_energy_scale
            )
        signed_evidence = np.concatenate(
            evidence_groups,
            axis=1,
        )
        calibrated_energy = np.column_stack(energy_groups)
        return DetectionScores(
            scores=np.sqrt(
                np.mean(
                    np.square(calibrated_energy),
                    axis=1,
                )
            ),
            feature_evidence=np.abs(signed_evidence),
            threshold=self.threshold,
            signed_feature_evidence=signed_evidence,
        )

    def _raw_score_components(
        self,
        windows: ContextualMultimodalModelWindows,
        include_metric_context: bool,
        include_log_context: bool,
    ) -> Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        (
            _,
            _,
            metric_context,
            log_context,
            metric_targets,
            log_targets,
            conditioning,
        ) = self._representations(windows)
        if not include_metric_context:
            metric_context = np.zeros_like(metric_context)
        if not include_log_context:
            log_context = np.zeros_like(log_context)
        horizon_index = self._horizons.index(1)
        joint_context = np.concatenate(
            (metric_context, log_context),
            axis=2,
        )
        predictor_input = np.concatenate(
            (
                joint_context.reshape(
                    len(joint_context),
                    -1,
                ),
                conditioning[:, horizon_index, :],
            ),
            axis=1,
        )
        predicted, _ = self._predict(predictor_input)
        observed = np.concatenate(
            (
                metric_targets[:, horizon_index, :],
                log_targets[:, horizon_index, :],
            ),
            axis=1,
        )
        residual = observed - predicted
        metric_residual = residual[
            :, : self.metric_latent_dimension
        ]
        log_residual = residual[
            :, self.metric_latent_dimension :
        ]
        metric_observed = observed[
            :, : self.metric_latent_dimension
        ]
        log_observed = observed[
            :, self.metric_latent_dimension :
        ]
        metric_block_evidence = (
            metric_residual
            * (1.0 - np.square(metric_observed))
        ) @ self._metric_target_weights.T
        log_block_evidence = (
            log_residual
            * (1.0 - np.square(log_observed))
        ) @ self._log_target_weights.T
        metric_energy = (
            np.sqrt(
                np.mean(np.square(metric_residual), axis=1)
            )
            if self.metric_latent_dimension > 0
            else np.zeros(len(residual), dtype=np.float64)
        )
        log_energy = (
            np.sqrt(
                np.mean(np.square(log_residual), axis=1)
            )
            if self.log_latent_dimension > 0
            else np.zeros(len(residual), dtype=np.float64)
        )
        return (
            metric_energy,
            log_energy,
            np.mean(
                metric_block_evidence.reshape(
                    len(metric_block_evidence),
                    self._target_block_size,
                    -1,
                ),
                axis=1,
            ),
            np.mean(
                log_block_evidence.reshape(
                    len(log_block_evidence),
                    self._target_block_size,
                    -1,
                ),
                axis=1,
            ),
        )

    def _representation_diagnostics(
        self,
        windows: ContextualMultimodalModelWindows,
    ) -> Mapping[str, Any]:
        (
            _,
            _,
            metric_context,
            log_context,
            metric_targets,
            log_targets,
            _,
        ) = self._representations(windows)
        target_count = (
            len(windows.metric_contexts) * len(windows.horizons)
        )
        metric_flat = metric_targets.reshape(
            target_count,
            self.metric_latent_dimension,
        )
        log_flat = log_targets.reshape(
            target_count,
            self.log_latent_dimension,
        )
        return {
            "metric_effective_rank": _effective_rank(
                metric_flat
            ),
            "log_effective_rank": _effective_rank(
                log_flat
            ),
            "metric_target_variance": _target_variance(
                metric_flat
            ),
            "metric_target_covariance": _target_covariance(
                metric_flat
            ),
            "log_target_variance": _target_variance(log_flat),
            "log_target_covariance": _target_covariance(log_flat),
            "metric_tanh_saturation": float(
                np.mean(np.abs(metric_context) > 0.98)
                if metric_context.size
                else 0.0
            ),
            "log_tanh_saturation": float(
                np.mean(np.abs(log_context) > 0.98)
                if log_context.size
                else 0.0
            ),
            "online_target_distance": {
                "metric": float(
                    np.linalg.norm(
                        self._metric_encoder_weights
                        - self._metric_target_weights
                    )
                ),
                "logs": float(
                    np.linalg.norm(
                        self._log_encoder_weights
                        - self._log_target_weights
                    )
                ),
            },
            "ema_update_half_life_epochs": _ema_half_life(
                self.ema_decay
            ),
            "auxiliary_losses": dict(
                self._last_auxiliary_losses
            ),
            "frozen_latent_probes": self._frozen_latent_probes(
                windows,
                metric_targets,
                log_targets,
            ),
            "modality_energy_scales": {
                "metric": self._metric_energy_scale,
                "logs": self._log_energy_scale,
            },
        }

    def _frozen_latent_probes(
        self,
        windows: ContextualMultimodalModelWindows,
        metric_targets: NDArray[np.float64],
        log_targets: NDArray[np.float64],
    ) -> Mapping[str, Any]:
        horizon_index = self._horizons.index(1)
        embeddings = np.concatenate(
            (
                metric_targets[:, horizon_index, :],
                log_targets[:, horizon_index, :],
            ),
            axis=1,
        )
        metric_blocks = windows.metric_target_blocks[
            :, horizon_index, :, :
        ]
        log_blocks = windows.log_target_blocks[
            :, horizon_index, :, :
        ]
        probes = {
            "checkout_completion_ratio": _feature_probe_values(
                log_blocks,
                windows.log_feature_names,
                "checkout_completion_ratio",
            ),
            "checkout_backlog_delta_ratio": (
                _feature_probe_values(
                    log_blocks,
                    windows.log_feature_names,
                    "checkout_backlog_delta_ratio",
                )
            ),
            "request_latency_ms": _feature_probe_values(
                metric_blocks,
                windows.metric_feature_names,
                "request_latency_ms",
            ),
            "queue_depth": _feature_probe_values(
                metric_blocks,
                windows.metric_feature_names,
                "queue_depth",
            ),
            "request_latency_bucket": _feature_probe_values(
                metric_blocks,
                windows.metric_feature_names,
                "request_latency_ms",
            ),
            "queue_depth_bucket": _feature_probe_values(
                metric_blocks,
                windows.metric_feature_names,
                "queue_depth",
            ),
            "worker_completion_ratio": _feature_probe_values(
                metric_blocks,
                windows.metric_feature_names,
                "worker_completion_ratio",
            ),
            "queue_transition_direction": (
                _transition_probe_values(
                    metric_blocks,
                    windows.metric_feature_names,
                    "queue_depth",
                )
            ),
        }
        for log_feature_name in (
            "queue_backlog_low_transition_rate",
            "queue_backlog_elevated_transition_rate",
            "queue_backlog_high_transition_rate",
            "database_latency_fast_ratio",
            "database_latency_normal_ratio",
            "database_latency_slow_ratio",
            "worker_busy_transition_rate",
            "worker_idle_transition_rate",
        ):
            if log_feature_name in windows.log_feature_names:
                probes[log_feature_name] = _feature_probe_values(
                    log_blocks,
                    windows.log_feature_names,
                    log_feature_name,
                )
        results = {}
        for name, values in probes.items():
            results[name] = (
                _bucket_probe(embeddings, values)
                if name.endswith("_bucket")
                else _linear_probe(embeddings, values)
            )
        return results

    def _validate_fitted(
        self,
        windows: ContextualMultimodalModelWindows,
    ) -> None:
        _validate_windows(windows)
        if (
            windows.metric_contexts.shape[1:]
            != self._context_shape[:2]
            or windows.log_contexts.shape[1:]
            != (
                self._context_shape[0],
                self._context_shape[2],
            )
            or windows.target_block_size
            != self._target_block_size
            or windows.horizons != self._horizons
            or windows.metric_feature_names
            != self._metric_feature_names
            or windows.log_feature_names
            != self._log_feature_names
            or windows.control_feature_names
            != self._control_feature_names
        ):
            raise ValueError(
                "windows do not match fitted contextual detector schema"
            )

    def to_dict(self) -> Dict[str, Any]:
        if not np.isfinite(self.threshold):
            raise RuntimeError(
                "detector must be fitted before serialization"
            )
        artifact = {
            "schema_version": 1,
            "kind": self.kind,
            "metric_latent_dimension": (
                self.metric_latent_dimension
            ),
            "log_latent_dimension": self.log_latent_dimension,
            "pretraining_epochs": self.pretraining_epochs,
            "predictor_refinement_epochs": (
                self.predictor_refinement_epochs
            ),
            "learning_rate": self.learning_rate,
            "ema_decay": self.ema_decay,
            "weight_decay": self.weight_decay,
            "loss": self.loss,
            "huber_delta": self.huber_delta,
            "auxiliary_loss_weight": self.auxiliary_loss_weight,
            "rollout_loss_weight": self.rollout_loss_weight,
            "calibration_quantile": self.calibration_quantile,
            "seed": self.seed,
            "threshold": self.threshold,
            "training_losses": list(self.training_losses),
            "diagnostics": dict(self.diagnostics),
            "context_shape": list(self._context_shape),
            "target_block_size": self._target_block_size,
            "horizons": list(self._horizons),
            "metric_feature_names": list(
                self._metric_feature_names
            ),
            "log_feature_names": list(self._log_feature_names),
            "control_feature_names": list(
                self._control_feature_names
            ),
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
            "predictor_input_weights": (
                self._predictor_input_weights.tolist()
            ),
            "predictor_hidden_bias": (
                self._predictor_hidden_bias.tolist()
            ),
            "predictor_output_weights": (
                self._predictor_output_weights.tolist()
            ),
            "predictor_output_bias": (
                self._predictor_output_bias.tolist()
            ),
            "head_weights": {
                name: values.tolist()
                for name, values in self._head_weights.items()
            },
            "head_biases": {
                name: values.tolist()
                for name, values in self._head_biases.items()
            },
            "metric_energy_scale": self._metric_energy_scale,
            "log_energy_scale": self._log_energy_scale,
            "metric_feature_scale": (
                self._metric_feature_scale.tolist()
            ),
            "log_feature_scale": self._log_feature_scale.tolist(),
            "training_protocol": self._training_protocol(),
        }
        if self.modality_mask_probability != 0.0:
            artifact["modality_mask_probability"] = (
                self.modality_mask_probability
            )
        if self.log_self_loss_multiplier != 1.0:
            artifact["log_self_loss_multiplier"] = (
                self.log_self_loss_multiplier
            )
        if self.cross_modal_loss_multiplier != 1.0:
            artifact["cross_modal_loss_multiplier"] = (
                self.cross_modal_loss_multiplier
            )
        return artifact

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "ContextualMultimodalJepaWorldModelDetector":
        if (
            payload.get("schema_version") != 1
            or payload.get("kind") != cls.kind
        ):
            raise ValueError(
                "unsupported contextual multimodal JEPA artifact"
            )
        detector = cls(
            metric_latent_dimension=int(
                payload["metric_latent_dimension"]
            ),
            log_latent_dimension=int(
                payload["log_latent_dimension"]
            ),
            pretraining_epochs=int(
                payload["pretraining_epochs"]
            ),
            predictor_refinement_epochs=int(
                payload["predictor_refinement_epochs"]
            ),
            learning_rate=float(payload["learning_rate"]),
            ema_decay=float(payload["ema_decay"]),
            weight_decay=float(payload["weight_decay"]),
            loss=str(payload["loss"]),
            huber_delta=float(payload["huber_delta"]),
            auxiliary_loss_weight=float(
                payload["auxiliary_loss_weight"]
            ),
            rollout_loss_weight=float(
                payload["rollout_loss_weight"]
            ),
            modality_mask_probability=float(
                payload.get("modality_mask_probability", 0.0)
            ),
            log_self_loss_multiplier=float(
                payload.get("log_self_loss_multiplier", 1.0)
            ),
            cross_modal_loss_multiplier=float(
                payload.get("cross_modal_loss_multiplier", 1.0)
            ),
            calibration_quantile=float(
                payload["calibration_quantile"]
            ),
            seed=int(payload["seed"]),
        )
        detector.threshold = float(payload["threshold"])
        detector.training_losses = tuple(
            float(value) for value in payload["training_losses"]
        )
        detector.diagnostics = dict(payload["diagnostics"])
        context_shape = payload["context_shape"]
        detector._context_shape = (
            int(context_shape[0]),
            int(context_shape[1]),
            int(context_shape[2]),
        )
        detector._target_block_size = int(
            payload["target_block_size"]
        )
        detector._horizons = tuple(
            int(value) for value in payload["horizons"]
        )
        detector._metric_feature_names = tuple(
            str(value) for value in payload["metric_feature_names"]
        )
        detector._log_feature_names = tuple(
            str(value) for value in payload["log_feature_names"]
        )
        detector._control_feature_names = tuple(
            str(value) for value in payload["control_feature_names"]
        )
        detector._metric_encoder_weights = _array(
            payload["metric_encoder_weights"]
        )
        detector._metric_encoder_bias = _array(
            payload["metric_encoder_bias"]
        )
        detector._metric_target_weights = _array(
            payload["metric_target_weights"]
        )
        detector._metric_target_bias = _array(
            payload["metric_target_bias"]
        )
        detector._log_encoder_weights = _array(
            payload["log_encoder_weights"]
        )
        detector._log_encoder_bias = _array(
            payload["log_encoder_bias"]
        )
        detector._log_target_weights = _array(
            payload["log_target_weights"]
        )
        detector._log_target_bias = _array(
            payload["log_target_bias"]
        )
        detector._predictor_input_weights = _array(
            payload["predictor_input_weights"]
        )
        detector._predictor_hidden_bias = _array(
            payload["predictor_hidden_bias"]
        )
        detector._predictor_output_weights = _array(
            payload["predictor_output_weights"]
        )
        detector._predictor_output_bias = _array(
            payload["predictor_output_bias"]
        )
        detector._head_weights = {
            str(name): _array(values)
            for name, values in dict(
                payload["head_weights"]
            ).items()
        }
        detector._head_biases = {
            str(name): _array(values)
            for name, values in dict(
                payload["head_biases"]
            ).items()
        }
        detector._metric_energy_scale = float(
            payload["metric_energy_scale"]
        )
        detector._log_energy_scale = float(
            payload["log_energy_scale"]
        )
        detector._metric_feature_scale = _array(
            payload["metric_feature_scale"]
        )
        detector._log_feature_scale = _array(
            payload["log_feature_scale"]
        )
        auxiliary = detector.diagnostics.get(
            "auxiliary_losses",
            {},
        )
        detector._last_auxiliary_losses = {
            str(name): float(value)
            for name, value in dict(auxiliary).items()
        }
        detector._validate_serialized_state()
        if dict(payload["training_protocol"]) != (
            detector._training_protocol()
        ):
            raise ValueError(
                "serialized training protocol is inconsistent"
            )
        return detector

    def _validate_serialized_state(self) -> None:
        lookback, metric_features, log_features = (
            self._context_shape
        )
        block_size = self._target_block_size
        if (
            lookback <= 0
            or metric_features != len(self._metric_feature_names)
            or log_features != len(self._log_feature_names)
            or block_size <= 0
            or lookback % block_size != 0
            or not self._horizons
            or 1 not in self._horizons
            or len(set(self._horizons)) != len(self._horizons)
            or any(horizon <= 0 for horizon in self._horizons)
        ):
            raise ValueError(
                "serialized contextual model schema is inconsistent"
            )
        metric_block_features = block_size * metric_features
        log_block_features = block_size * log_features
        joint_dimension = (
            self.metric_latent_dimension
            + self.log_latent_dimension
        )
        patch_count = lookback // block_size
        condition_dimension = (
            block_size * len(self._control_feature_names)
            + len(self._horizons)
        )
        hidden_dimension = max(8, 2 * joint_dimension)
        expected_shapes = {
            "metric_encoder_weights": (
                self._metric_encoder_weights,
                (
                    metric_block_features,
                    self.metric_latent_dimension,
                ),
            ),
            "metric_encoder_bias": (
                self._metric_encoder_bias,
                (self.metric_latent_dimension,),
            ),
            "metric_target_weights": (
                self._metric_target_weights,
                (
                    metric_block_features,
                    self.metric_latent_dimension,
                ),
            ),
            "metric_target_bias": (
                self._metric_target_bias,
                (self.metric_latent_dimension,),
            ),
            "log_encoder_weights": (
                self._log_encoder_weights,
                (
                    log_block_features,
                    self.log_latent_dimension,
                ),
            ),
            "log_encoder_bias": (
                self._log_encoder_bias,
                (self.log_latent_dimension,),
            ),
            "log_target_weights": (
                self._log_target_weights,
                (
                    log_block_features,
                    self.log_latent_dimension,
                ),
            ),
            "log_target_bias": (
                self._log_target_bias,
                (self.log_latent_dimension,),
            ),
            "predictor_input_weights": (
                self._predictor_input_weights,
                (
                    patch_count * joint_dimension
                    + condition_dimension,
                    hidden_dimension,
                ),
            ),
            "predictor_hidden_bias": (
                self._predictor_hidden_bias,
                (hidden_dimension,),
            ),
            "predictor_output_weights": (
                self._predictor_output_weights,
                (hidden_dimension, joint_dimension),
            ),
            "predictor_output_bias": (
                self._predictor_output_bias,
                (joint_dimension,),
            ),
            "metric_feature_scale": (
                self._metric_feature_scale,
                (metric_features,),
            ),
            "log_feature_scale": (
                self._log_feature_scale,
                (log_features,),
            ),
        }
        for name, (values, expected_shape) in expected_shapes.items():
            if (
                values.shape != expected_shape
                or not np.all(np.isfinite(values))
            ):
                raise ValueError(
                    f"serialized {name} is inconsistent"
                )
        metric_input_dimension = (
            patch_count * self.metric_latent_dimension
            + condition_dimension
        )
        log_input_dimension = (
            patch_count * self.log_latent_dimension
            + condition_dimension
        )
        expected_head_shapes: Dict[str, Tuple[int, int]] = {}
        if self.metric_latent_dimension > 0:
            expected_head_shapes["metric_to_metric"] = (
                metric_input_dimension,
                self.metric_latent_dimension,
            )
        if self.log_latent_dimension > 0:
            expected_head_shapes["log_to_log"] = (
                log_input_dimension,
                self.log_latent_dimension,
            )
        if (
            self.metric_latent_dimension > 0
            and self.log_latent_dimension > 0
        ):
            expected_head_shapes["metric_to_log"] = (
                metric_input_dimension,
                self.log_latent_dimension,
            )
            expected_head_shapes["log_to_metric"] = (
                log_input_dimension,
                self.metric_latent_dimension,
            )
        if (
            set(self._head_weights) != set(expected_head_shapes)
            or set(self._head_biases) != set(expected_head_shapes)
        ):
            raise ValueError(
                "serialized auxiliary heads are incomplete"
            )
        for name, expected_shape in expected_head_shapes.items():
            weights = self._head_weights[name]
            biases = self._head_biases[name]
            if (
                weights.shape != expected_shape
                or biases.shape != (expected_shape[1],)
                or not np.all(np.isfinite(weights))
                or not np.all(np.isfinite(biases))
            ):
                raise ValueError(
                    f"serialized {name} head is inconsistent"
                )
        if (
            not np.isfinite(self.threshold)
            or self.threshold < 0.0
            or not np.isfinite(self._metric_energy_scale)
            or self._metric_energy_scale <= 0.0
            or not np.isfinite(self._log_energy_scale)
            or self._log_energy_scale <= 0.0
            or len(self.training_losses)
            != (
                self.pretraining_epochs
                + self.predictor_refinement_epochs
            )
            or not np.all(np.isfinite(self.training_losses))
        ):
            raise ValueError(
                "serialized contextual model calibration is invalid"
            )

    def _training_protocol(self) -> Dict[str, Any]:
        rollout_indices = self._rollout_indices()
        rollout: Dict[str, Any]
        if rollout_indices is None:
            rollout = {"steps": 0}
        else:
            _, second_index = rollout_indices
            rollout = {
                "steps": 2,
                "first_horizon": 1,
                "second_horizon": self._horizons[second_index],
                "intermediate_prediction": "stop_gradient",
            }
        protocol = {
            "target_encoder_update": (
                "ema_during_pretraining_only"
            ),
            "encoder_refinement": "frozen",
            "loss": self.loss,
            "target_horizons": list(self._horizons),
            "target_block_size": self._target_block_size,
            "context_patch_stride": self._target_block_size,
            "scoring_horizon": 1,
            "rollout": rollout,
            "predictor_conditioning": (
                list(self._control_feature_names)
                + ["target_horizon"]
            ),
            "auxiliary_objectives": list(
                self._active_auxiliary_objectives()
            ),
        }
        if self.modality_mask_probability != 0.0:
            protocol["context_modality_masking"] = {
                "kind": (
                    "deterministic_single_modality_dropout"
                ),
                "probability_per_available_modality": (
                    self.modality_mask_probability
                ),
                "seed": self.seed,
            }
        if (
            self.log_self_loss_multiplier != 1.0
            or self.cross_modal_loss_multiplier != 1.0
        ):
            protocol["auxiliary_objective_multipliers"] = {
                name: self._auxiliary_objective_multiplier(name)
                for name in self._active_auxiliary_objectives()
            }
        return protocol


def _validate_windows(
    windows: ContextualMultimodalModelWindows,
) -> None:
    if len(windows.metric_contexts) < 2:
        raise ValueError(
            "contextual JEPA requires at least two samples"
        )
    if 1 not in windows.horizons:
        raise ValueError(
            "contextual JEPA requires horizon 1 for scoring"
        )


def _temporal_patches(
    contexts: NDArray[np.float64],
    block_size: int,
) -> NDArray[np.float64]:
    return np.stack(
        [
            contexts[:, start : start + block_size, :].reshape(
                len(contexts),
                -1,
            )
            for start in range(0, contexts.shape[1], block_size)
        ],
        axis=1,
    )


def _loss_and_gradient(
    residual: NDArray[np.float64],
    loss: str,
    huber_delta: float,
) -> Tuple[float, NDArray[np.float64]]:
    normalizer = float(residual.size)
    absolute = np.abs(residual)
    if loss == "mse":
        return (
            float(np.mean(np.square(residual))),
            2.0 * residual / normalizer,
        )
    if loss == "l1":
        return (
            float(np.mean(absolute)),
            np.sign(residual) / normalizer,
        )
    quadratic = absolute <= huber_delta
    losses = np.where(
        quadratic,
        0.5 * np.square(residual),
        huber_delta * (absolute - 0.5 * huber_delta),
    )
    gradient = np.where(
        quadratic,
        residual,
        huber_delta * np.sign(residual),
    )
    return float(np.mean(losses)), gradient / normalizer


def _effective_rank(values: NDArray[np.float64]) -> float:
    if values.shape[1] == 0:
        return 0.0
    centered = values - np.mean(values, axis=0)
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues = np.maximum(
        np.linalg.eigvalsh(covariance),
        0.0,
    )
    denominator = float(np.sum(np.square(eigenvalues)))
    if denominator <= 1e-18:
        return 0.0
    return float(np.square(np.sum(eigenvalues)) / denominator)


def _target_variance(
    values: NDArray[np.float64],
) -> list[float]:
    if values.shape[1] == 0:
        return []
    return [
        float(value)
        for value in np.var(values, axis=0).tolist()
    ]


def _target_covariance(
    values: NDArray[np.float64],
) -> list[list[float]]:
    if values.shape[1] == 0:
        return []
    centered = values - np.mean(values, axis=0)
    covariance = (
        centered.T @ centered / max(len(centered) - 1, 1)
    )
    return [
        [float(value) for value in row]
        for row in covariance.tolist()
    ]


def _ema_half_life(ema_decay: float) -> float:
    if ema_decay == 0.0:
        return 0.0
    return float(np.log(0.5) / np.log(ema_decay))


def _feature_probe_values(
    blocks: NDArray[np.float64],
    feature_names: Tuple[str, ...],
    feature_name: str,
) -> Optional[NDArray[np.float64]]:
    if feature_name not in feature_names:
        return None
    feature_index = feature_names.index(feature_name)
    return np.asarray(
        np.mean(blocks[:, :, feature_index], axis=1),
        dtype=np.float64,
    )


def _transition_probe_values(
    blocks: NDArray[np.float64],
    feature_names: Tuple[str, ...],
    feature_name: str,
) -> Optional[NDArray[np.float64]]:
    if feature_name not in feature_names:
        return None
    feature_index = feature_names.index(feature_name)
    return (
        blocks[:, -1, feature_index]
        - blocks[:, 0, feature_index]
    )


def _linear_probe(
    embeddings: NDArray[np.float64],
    values: Optional[NDArray[np.float64]],
) -> Mapping[str, Any]:
    if values is None:
        return {"status": "feature_unavailable"}
    centered_values = values - np.mean(values)
    total_variation = float(np.sum(np.square(centered_values)))
    if total_variation <= 1e-12:
        return {"status": "constant_target"}
    design = np.column_stack(
        (
            np.ones(len(embeddings), dtype=np.float64),
            embeddings,
        )
    )
    regularization = np.eye(
        design.shape[1],
        dtype=np.float64,
    ) * 1e-3
    regularization[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + regularization,
        design.T @ values,
    )
    residual = values - design @ coefficients
    r_squared = 1.0 - (
        float(np.sum(np.square(residual))) / total_variation
    )
    return {
        "status": "completed",
        "r_squared": r_squared,
        "ridge": 1e-3,
    }


def _bucket_probe(
    embeddings: NDArray[np.float64],
    values: Optional[NDArray[np.float64]],
) -> Mapping[str, Any]:
    if values is None:
        return {"status": "feature_unavailable"}
    boundaries = np.unique(
        np.quantile(values, (1.0 / 3.0, 2.0 / 3.0))
    )
    if len(boundaries) < 2:
        return {"status": "insufficient_bucket_variation"}
    labels = np.digitize(values, boundaries).astype(np.int64)
    class_count = int(np.max(labels)) + 1
    targets = np.eye(class_count, dtype=np.float64)[labels]
    design = np.column_stack(
        (
            np.ones(len(embeddings), dtype=np.float64),
            embeddings,
        )
    )
    regularization = np.eye(
        design.shape[1],
        dtype=np.float64,
    ) * 1e-3
    regularization[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + regularization,
        design.T @ targets,
    )
    predictions = np.argmax(design @ coefficients, axis=1)
    return {
        "status": "completed",
        "training_accuracy": float(np.mean(predictions == labels)),
        "bucket_boundaries": [
            float(value) for value in boundaries.tolist()
        ],
        "ridge": 1e-3,
    }


def _array(values: Any) -> NDArray[np.float64]:
    return np.asarray(values, dtype=np.float64)


def _empty_matrix() -> NDArray[np.float64]:
    return np.empty((0, 0), dtype=np.float64)


def _empty_vector() -> NDArray[np.float64]:
    return np.empty(0, dtype=np.float64)
