"""Learned EMA graph-JEPA over subsystem-owned temporal patches."""

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .graph_telemetry import (
    DeclaredTelemetryGraph,
    GraphStateWindows,
)


GRAPH_CONTEXT_SCOPES = (
    "entity_local",
    "one_hop",
    "all_entities",
)


@dataclass(frozen=True)
class GraphEmaJepaConfig:
    """Frozen optimization and subsystem representation widths."""

    entity_latent_dimensions: Mapping[str, int]
    context_scope: str = "one_hop"
    context_entity_overrides: Optional[
        Mapping[str, Tuple[str, ...]]
    ] = None
    epochs: int = 60
    learning_rate: float = 5e-3
    ema_decay: float = 0.98
    weight_decay: float = 1e-4
    batch_size: int = 1024
    seed: int = 89

    def __post_init__(self) -> None:
        if self.context_scope not in GRAPH_CONTEXT_SCOPES:
            raise ValueError("unsupported learned graph context scope")
        if (
            self.epochs < 1
            or self.learning_rate <= 0.0
            or not 0.0 <= self.ema_decay < 1.0
            or self.weight_decay < 0.0
            or self.batch_size < 1
        ):
            raise ValueError(
                "invalid learned graph JEPA optimization config"
            )
        if not self.entity_latent_dimensions or any(
            not entity_id
            or isinstance(width, bool)
            or width < 0
            for entity_id, width in self.entity_latent_dimensions.items()
        ):
            raise ValueError(
                "learned graph JEPA widths must be nonnegative"
            )
        if self.context_entity_overrides is not None and any(
            not entity_id
            or not context_ids
            or len(set(context_ids)) != len(context_ids)
            for entity_id, context_ids in (
                self.context_entity_overrides.items()
            )
        ):
            raise ValueError(
                "learned graph context overrides are invalid"
            )

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "entity_latent_dimensions": {
                key: int(value)
                for key, value in sorted(
                    self.entity_latent_dimensions.items()
                )
            },
            "context_scope": self.context_scope,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "ema_decay": self.ema_decay,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "seed": self.seed,
        }
        if self.context_entity_overrides is not None:
            payload["context_entity_overrides"] = {
                key: list(value)
                for key, value in sorted(
                    self.context_entity_overrides.items()
                )
            }
        return payload


@dataclass(frozen=True)
class LearnedGraphJepaPrediction:
    """Latent predictions and decoded normalized operational targets."""

    predicted_tokens: Mapping[str, NDArray[np.float64]]
    target_tokens: Mapping[str, NDArray[np.float64]]
    decoded_target_blocks: NDArray[np.float64]
    reconstructed_target_blocks: NDArray[np.float64]


class LearnedGraphJepaWorldModel:
    """Graph-scoped predictors with learned online and EMA target encoders."""

    kind = "learned_ema_graph_jepa_world_model_v1"

    def __init__(self, config: GraphEmaJepaConfig) -> None:
        self.config = config
        self.training_losses: Tuple[float, ...] = ()
        self._graph: Optional[DeclaredTelemetryGraph] = None
        self._entity_ids: Tuple[str, ...] = ()
        self._local_feature_keys: Tuple[Tuple[str, ...], ...] = ()
        self._control_feature_names: Tuple[str, ...] = ()
        self._horizons: Tuple[int, ...] = ()
        self._target_block_size = 0
        self._lookback = 0
        self._online: Dict[str, NDArray[np.float64]] = {}
        self._target: Dict[str, NDArray[np.float64]] = {}
        self._predictors: Dict[str, NDArray[np.float64]] = {}
        self._context_entities: Dict[str, Tuple[str, ...]] = {}

    def fit(
        self, windows: GraphStateWindows
    ) -> "LearnedGraphJepaWorldModel":
        """Optimize latent future prediction with stop-gradient EMA targets."""

        self._register_schema(windows)
        generator = np.random.default_rng(self.config.seed)
        local = _local_channels(windows)
        context_patches = {
            entity_id: _context_patches(
                local[entity_id][0],
                windows.target_block_size,
            )
            for entity_id in local
        }
        target_patches = {
            entity_id: local[entity_id][1].reshape(
                len(windows.contexts),
                len(windows.horizons),
                -1,
            )
            for entity_id in local
        }
        self._online = {}
        self._target = {}
        self._predictors = {}
        self._context_entities = {}
        for entity_id, patches in context_patches.items():
            width = self.config.entity_latent_dimensions[
                entity_id
            ]
            if width > patches.shape[-1]:
                raise ValueError(
                    f"latent width exceeds local patch: {entity_id}"
                )
            initial = generator.normal(
                0.0,
                1.0 / np.sqrt(patches.shape[-1]),
                size=(patches.shape[-1], width),
            )
            online = _orthonormal_columns(initial)
            self._online[entity_id] = online
            self._target[entity_id] = online.copy()
        for entity_id in self._online:
            context_entities = self._context_entity_ids(
                entity_id
            )
            self._context_entities[entity_id] = context_entities
            design_width = (
                context_patches[entity_id].shape[1]
                * sum(
                    self._online[source].shape[1]
                    for source in context_entities
                )
                + windows.target_block_size
                * len(windows.control_feature_names)
            )
            self._predictors[entity_id] = generator.normal(
                0.0,
                1.0 / np.sqrt(design_width),
                size=(
                    design_width,
                    self._online[entity_id].shape[1],
                ),
            )

        sample_indices = np.repeat(
            np.arange(len(windows.contexts), dtype=np.int64),
            len(windows.horizons),
        )
        horizon_indices = np.tile(
            np.arange(len(windows.horizons), dtype=np.int64),
            len(windows.contexts),
        )
        row_count = len(sample_indices)
        losses = []
        for _ in range(self.config.epochs):
            order = generator.permutation(row_count)
            epoch_loss = 0.0
            epoch_values = 0
            for start in range(0, row_count, self.config.batch_size):
                selection = order[
                    start : start + self.config.batch_size
                ]
                batch_samples = sample_indices[selection]
                batch_horizons = horizon_indices[selection]
                online_tokens = {
                    entity_id: np.tanh(
                        patches[batch_samples]
                        @ self._online[entity_id]
                    )
                    for entity_id, patches in context_patches.items()
                }
                encoder_gradients = {
                    entity_id: np.zeros_like(weights)
                    for entity_id, weights in self._online.items()
                }
                for target_id, predictor in self._predictors.items():
                    context_ids = self._context_entities[target_id]
                    context_parts = [
                        online_tokens[source].reshape(
                            len(selection), -1
                        )
                        for source in context_ids
                    ]
                    controls = windows.target_controls[
                        batch_samples, batch_horizons
                    ].reshape(len(selection), -1)
                    design = np.concatenate(
                        context_parts + [controls], axis=1
                    )
                    target_raw = target_patches[target_id][
                        batch_samples, batch_horizons
                    ]
                    target_tokens = np.tanh(
                        target_raw @ self._target[target_id]
                    )
                    predicted = design @ predictor
                    residual = predicted - target_tokens
                    epoch_loss += float(np.sum(np.square(residual)))
                    epoch_values += residual.size
                    gradient = (
                        2.0
                        * residual
                        / (
                            len(selection)
                            * max(1, residual.shape[1])
                            * len(self._predictors)
                        )
                    )
                    predictor_before = predictor.copy()
                    self._predictors[target_id] = (
                        predictor
                        - self.config.learning_rate
                        * (
                            design.T @ gradient
                            + self.config.weight_decay * predictor
                        )
                    )
                    design_gradient = gradient @ predictor_before.T
                    offset = 0
                    for source in context_ids:
                        tokens = online_tokens[source]
                        width = tokens.shape[-1]
                        size = tokens.shape[1] * width
                        token_gradient = design_gradient[
                            :, offset : offset + size
                        ].reshape(tokens.shape)
                        preactivation_gradient = token_gradient * (
                            1.0 - np.square(tokens)
                        )
                        source_patches = context_patches[source][
                            batch_samples
                        ]
                        encoder_gradients[source] += np.einsum(
                            "npi,npj->ij",
                            source_patches,
                            preactivation_gradient,
                        )
                        offset += size
                for entity_id, gradient in encoder_gradients.items():
                    self._online[entity_id] -= (
                        self.config.learning_rate
                        * (
                            gradient
                            + self.config.weight_decay
                            * self._online[entity_id]
                        )
                    )
            for entity_id in self._online:
                self._online[entity_id] = _orthonormal_columns(
                    self._online[entity_id]
                )
                target = (
                    self.config.ema_decay * self._target[entity_id]
                    + (1.0 - self.config.ema_decay)
                    * self._online[entity_id]
                )
                self._target[entity_id] = _orthonormal_columns(
                    target
                )
            losses.append(epoch_loss / max(epoch_values, 1))
        self.training_losses = tuple(losses)
        return self

    def predict(
        self, windows: GraphStateWindows
    ) -> LearnedGraphJepaPrediction:
        """Predict and decode each graph-owned future-state block."""

        self._validate_schema(windows)
        local = _local_channels(windows)
        context_patches = {
            entity_id: _context_patches(
                local[entity_id][0],
                windows.target_block_size,
            )
            for entity_id in self._online
        }
        online_tokens = {
            entity_id: np.tanh(
                patches @ self._online[entity_id]
            )
            for entity_id, patches in context_patches.items()
        }
        decoded = np.zeros_like(windows.target_blocks)
        reconstructed = np.zeros_like(windows.target_blocks)
        predictions: Dict[str, NDArray[np.float64]] = {}
        targets: Dict[str, NDArray[np.float64]] = {}
        for entity_id, predictor in self._predictors.items():
            entity_position = windows.entity_ids.index(entity_id)
            mask = windows.observation_mask[entity_position]
            context_parts = [
                np.repeat(
                    online_tokens[source][:, None, :, :],
                    len(windows.horizons),
                    axis=1,
                ).reshape(
                    len(windows.contexts)
                    * len(windows.horizons),
                    -1,
                )
                for source in self._context_entities[entity_id]
            ]
            controls = windows.target_controls.reshape(
                len(windows.contexts) * len(windows.horizons),
                -1,
            )
            design = np.concatenate(
                context_parts + [controls], axis=1
            )
            predicted = (design @ predictor).reshape(
                len(windows.contexts),
                len(windows.horizons),
                -1,
            )
            raw = local[entity_id][1].reshape(
                len(windows.contexts),
                len(windows.horizons),
                -1,
            )
            target = np.tanh(raw @ self._target[entity_id])
            predictions[entity_id] = predicted
            targets[entity_id] = target
            decoded_values = _decode(
                predicted.reshape(-1, predicted.shape[-1]),
                self._target[entity_id],
            ).reshape(
                len(windows.contexts),
                len(windows.horizons),
                windows.target_block_size,
                int(np.count_nonzero(mask)),
            )
            reconstructed_values = _decode(
                target.reshape(-1, target.shape[-1]),
                self._target[entity_id],
            ).reshape(decoded_values.shape)
            decoded[:, :, :, entity_position, mask] = decoded_values
            reconstructed[
                :, :, :, entity_position, mask
            ] = reconstructed_values
        return LearnedGraphJepaPrediction(
            predicted_tokens=predictions,
            target_tokens=targets,
            decoded_target_blocks=decoded,
            reconstructed_target_blocks=reconstructed,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize online and separately identifiable EMA encoders."""

        if self._graph is None:
            raise RuntimeError("learned graph JEPA is not fitted")
        return {
            "schema_version": 1,
            "kind": self.kind,
            "config": self.config.to_dict(),
            "graph": self._graph.to_dict(),
            "schema": {
                "entity_ids": list(self._entity_ids),
                "local_feature_keys": [
                    list(keys) for keys in self._local_feature_keys
                ],
                "control_feature_names": list(
                    self._control_feature_names
                ),
                "horizons": list(self._horizons),
                "target_block_size": self._target_block_size,
                "lookback": self._lookback,
            },
            "training_losses": list(self.training_losses),
            "online_encoders": {
                key: value.tolist()
                for key, value in self._online.items()
            },
            "ema_target_encoders": {
                key: value.tolist()
                for key, value in self._target.items()
            },
            "predictors": {
                key: {
                    "context_entity_ids": list(
                        self._context_entities[key]
                    ),
                    "weights": value.tolist(),
                }
                for key, value in self._predictors.items()
            },
        }

    def _register_schema(self, windows: GraphStateWindows) -> None:
        if windows.contexts.shape[1] % windows.target_block_size:
            raise ValueError(
                "graph context must divide into target-sized patches"
            )
        configured = set(self.config.entity_latent_dimensions)
        if configured != set(windows.entity_ids):
            raise ValueError(
                "learned graph widths must cover every entity"
            )
        observed = {
            entity_id
            for position, entity_id in enumerate(windows.entity_ids)
            if np.any(windows.observation_mask[position])
        }
        active = {
            entity_id
            for entity_id, width in (
                self.config.entity_latent_dimensions.items()
            )
            if width > 0
        }
        if observed != active:
            raise ValueError(
                "learned graph active widths differ from observations"
            )
        if self.config.context_entity_overrides is not None:
            overrides = self.config.context_entity_overrides
            if set(overrides) != observed:
                raise ValueError(
                    "learned graph context overrides must cover "
                    "observed entities"
                )
            for entity_id, context_ids in overrides.items():
                if (
                    entity_id not in context_ids
                    or not set(context_ids) <= observed
                ):
                    raise ValueError(
                        "learned graph context override is invalid"
                    )
        self._graph = windows.graph
        self._entity_ids = windows.entity_ids
        self._local_feature_keys = windows.local_feature_keys
        self._control_feature_names = windows.control_feature_names
        self._horizons = windows.horizons
        self._target_block_size = windows.target_block_size
        self._lookback = windows.contexts.shape[1]

    def _validate_schema(self, windows: GraphStateWindows) -> None:
        if (
            self._graph is None
            or windows.graph.to_dict() != self._graph.to_dict()
            or windows.entity_ids != self._entity_ids
            or windows.local_feature_keys
            != self._local_feature_keys
            or windows.control_feature_names
            != self._control_feature_names
            or windows.horizons != self._horizons
            or windows.target_block_size
            != self._target_block_size
            or windows.contexts.shape[1] != self._lookback
        ):
            raise ValueError(
                "learned graph JEPA tensor schema changed"
            )

    def _context_entity_ids(
        self, entity_id: str
    ) -> Tuple[str, ...]:
        assert self._graph is not None
        active = set(self._online)
        selected: Tuple[str, ...]
        if self.config.context_entity_overrides is not None:
            selected = tuple(
                self.config.context_entity_overrides[entity_id]
            )
        elif self.config.context_scope == "entity_local":
            selected = (entity_id,)
        elif self.config.context_scope == "all_entities":
            selected = self._entity_ids
        else:
            selected = self._graph.neighboring_entity_ids(
                entity_id
            )
        return tuple(
            candidate
            for candidate in selected
            if candidate in active
        )


def evaluate_learned_graph_jepa(
    model: LearnedGraphJepaWorldModel,
    windows: GraphStateWindows,
) -> Mapping[str, Any]:
    """Score raw prediction, reconstruction, and active compression."""

    prediction = model.predict(windows)
    feature_errors: Dict[str, float] = {}
    reconstruction_errors: Dict[str, float] = {}
    entity_errors: Dict[str, list[float]] = {}
    for entity_position, entity_id in enumerate(windows.entity_ids):
        for slot_position, feature_key in enumerate(
            windows.local_feature_keys[entity_position]
        ):
            target = windows.target_blocks[
                :, :, :, entity_position, slot_position
            ].reshape(-1)
            variance = float(np.var(target))
            if variance <= 1e-12:
                continue
            decoded = prediction.decoded_target_blocks[
                :, :, :, entity_position, slot_position
            ].reshape(-1)
            reconstructed = (
                prediction.reconstructed_target_blocks[
                    :, :, :, entity_position, slot_position
                ].reshape(-1)
            )
            error = float(
                np.mean(np.square(decoded - target)) / variance
            )
            reconstruction = float(
                np.mean(np.square(reconstructed - target))
                / variance
            )
            feature_errors[feature_key] = error
            reconstruction_errors[feature_key] = reconstruction
            entity_errors.setdefault(entity_id, []).append(error)
    active_raw_context_values = (
        windows.contexts.shape[1]
        * int(np.count_nonzero(windows.observation_mask))
    )
    latent_values = (
        windows.contexts.shape[1]
        // windows.target_block_size
        * sum(model.config.entity_latent_dimensions.values())
    )
    return {
        "mean_normalized_mse": float(
            np.mean(tuple(feature_errors.values()))
        ),
        "mean_reconstruction_normalized_mse": float(
            np.mean(tuple(reconstruction_errors.values()))
        ),
        "feature_normalized_mse": feature_errors,
        "feature_reconstruction_normalized_mse": (
            reconstruction_errors
        ),
        "entity_normalized_mse": {
            entity_id: float(np.mean(values))
            for entity_id, values in entity_errors.items()
        },
        "compression": {
            "active_raw_context_values": (
                active_raw_context_values
            ),
            "latent_values": latent_values,
            "ratio": (
                active_raw_context_values / latent_values
                if latent_values
                else 0.0
            ),
        },
        "training": {
            "initial_loss": model.training_losses[0],
            "final_loss": model.training_losses[-1],
            "loss_decreased": (
                model.training_losses[-1]
                < model.training_losses[0]
            ),
        },
    }


def _local_channels(
    windows: GraphStateWindows,
) -> Mapping[
    str, Tuple[NDArray[np.float64], NDArray[np.float64]]
]:
    result = {}
    for position, entity_id in enumerate(windows.entity_ids):
        mask = windows.observation_mask[position]
        if np.any(mask):
            result[entity_id] = (
                windows.contexts[:, :, position, mask],
                windows.target_blocks[:, :, :, position, mask],
            )
    return result


def _context_patches(
    contexts: NDArray[np.float64],
    patch_size: int,
) -> NDArray[np.float64]:
    return contexts.reshape(
        len(contexts),
        contexts.shape[1] // patch_size,
        patch_size * contexts.shape[2],
    )


def _orthonormal_columns(
    values: NDArray[np.float64],
) -> NDArray[np.float64]:
    left, _, right = np.linalg.svd(
        values, full_matrices=False
    )
    return np.asarray(left @ right, dtype=np.float64)


def _decode(
    tokens: NDArray[np.float64],
    encoder: NDArray[np.float64],
) -> NDArray[np.float64]:
    clipped = np.clip(tokens, -0.999999, 0.999999)
    return np.arctanh(clipped) @ encoder.T
