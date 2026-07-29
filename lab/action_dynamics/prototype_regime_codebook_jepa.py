"""Retained reproduction runner for the soft regime-codebook JEPA prototype.

Run:
    .venv/bin/python lab/action_dynamics/prototype_regime_codebook_jepa.py

This remains non-production experiment code. Keep it with the immutable result
artifact, and use a fresh ``--output`` directory for every rerun.
"""

import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import io
import json
from pathlib import Path
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn

from quantis_core.action_conditioned_dynamics import (
    ActionConditionedWindows,
    TrajectoryDistribution,
)
from quantis_core.edge_dynamics.data import (
    PreparedAttributionQueries,
    PreparedEdgeDynamicsData,
    load_edge_dynamics_cache,
    partition_worker_topology,
    source_artifact_manifest_sha256,
    subset_attribution_queries,
    topology_transfer_cache_address,
    validate_topology_transfer_cache,
)
from quantis_core.edge_dynamics.evaluation import (
    conformal_sequential_detection,
    forecast_objective,
    score_edge_model,
)
from quantis_core.edge_dynamics.jepa_evaluation import (
    action_conditioning_sanity,
)
from quantis_core.edge_dynamics.models import (
    ContractiveLowRankDynamics,
    LowRankConfig,
)
from quantis_core.graph_telemetry import DeclaredTelemetryGraph


@dataclass(frozen=True)
class PrototypeConfig:
    """Frozen choices for the retained non-production tracer."""

    latent_dimension: int = 16
    code_count: int = 32
    temperature: float = 0.25
    hidden_dimension: int = 256
    epochs: int = 40
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    ema_decay: float = 0.996
    future_state_weight: float = 1.0
    latent_weight: float = 0.2
    assignment_weight: float = 0.2
    target_reconstruction_weight: float = 0.2
    context_reconstruction_weight: float = 0.1
    balance_weight: float = 0.05
    sharpness_weight: float = 0.01
    ridge: float = 1e-3
    seed: int = 127


class PrototypeRegimeNetwork(nn.Module):
    """Fine/coarse entity encoder with an optional shared soft codebook."""

    def __init__(
        self,
        *,
        entity_count: int,
        feature_count: int,
        control_count: int,
        action_count: int,
        horizon: int,
        observation_mask: NDArray[np.bool_],
        config: PrototypeConfig,
        use_codebook: bool,
    ) -> None:
        super().__init__()
        self.entity_count = entity_count
        self.feature_count = feature_count
        self.control_count = control_count
        self.action_count = action_count
        self.horizon = horizon
        self.config = config
        self.use_codebook = use_codebook
        latent = config.latent_dimension
        self.state_encoder = nn.Linear(feature_count, latent)
        self.entity_embedding = nn.Parameter(
            torch.empty(entity_count, latent)
        )
        self.fine_projection = nn.Linear(4 * latent, latent)
        self.coarse_projection = nn.Linear(latent, latent)
        self.trend_projection = nn.Linear(latent, latent)
        self.context_norm = nn.LayerNorm(latent)
        condition_width = (
            entity_count * latent
            + horizon * control_count
            + horizon * entity_count * action_count
        )
        self.predictor = nn.Sequential(
            nn.Linear(condition_width, config.hidden_dimension),
            nn.GELU(),
            nn.Linear(
                config.hidden_dimension,
                horizon * entity_count * latent,
            ),
        )
        self.decoder = nn.Linear(latent, feature_count)
        self.target_state_encoder = copy.deepcopy(self.state_encoder)
        for parameter in self.target_state_encoder.parameters():
            parameter.requires_grad_(False)
        self.target_entity_embedding = nn.Parameter(
            torch.empty(entity_count, latent),
            requires_grad=False,
        )
        if use_codebook:
            self.codebook = nn.Parameter(
                torch.empty(config.code_count, latent)
            )
        else:
            self.register_parameter("codebook", None)
        self.register_buffer(
            "observation_mask",
            torch.from_numpy(observation_mask.astype(np.float32)),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.config.seed)
        with torch.no_grad():
            self.entity_embedding.normal_(
                0.0, 0.05, generator=generator
            )
            self.target_entity_embedding.copy_(self.entity_embedding)
            if self.codebook is not None:
                self.codebook.normal_(0.0, 0.2, generator=generator)

    def encode_context(self, histories: torch.Tensor) -> torch.Tensor:
        points = self.state_encoder(histories)
        fine = points[:, -4:].permute(0, 2, 1, 3).reshape(
            len(points), self.entity_count, -1
        )
        fine = self.fine_projection(fine)
        coarse = self.coarse_projection(points.mean(dim=1))
        trend = self.trend_projection(points[:, -1] - points[:, 0])
        return self.context_norm(
            fine
            + coarse
            + trend
            + self.entity_embedding[None, :, :]
        )

    def encode_targets(self, future: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return (
                self.target_state_encoder(future)
                + self.target_entity_embedding[None, None, :, :]
            )

    def assignments(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.codebook is None:
            raise ValueError("continuous prototype has no codebook")
        codebook_shape = (1,) * (tokens.ndim - 1) + tuple(
            self.codebook.shape
        )
        squared_distance = torch.mean(
            torch.square(
                tokens.unsqueeze(-2)
                - self.codebook.reshape(codebook_shape)
            ),
            dim=-1,
        )
        return torch.softmax(
            -squared_distance / self.config.temperature, dim=-1
        )

    def predict_tokens(
        self,
        histories: torch.Tensor,
        future_controls: torch.Tensor,
        future_actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        context = self.encode_context(histories)
        conditioned = torch.cat(
            (
                context.reshape(len(context), -1),
                future_controls.reshape(len(context), -1),
                future_actions.reshape(len(context), -1),
            ),
            dim=1,
        )
        predicted = self.predictor(conditioned).reshape(
            len(context),
            self.horizon,
            self.entity_count,
            self.config.latent_dimension,
        )
        return context, predicted

    def forward(
        self,
        histories: torch.Tensor,
        future_controls: torch.Tensor,
        future_actions: torch.Tensor,
        future_states: torch.Tensor,
    ) -> Mapping[str, torch.Tensor]:
        context, predicted_tokens = self.predict_tokens(
            histories, future_controls, future_actions
        )
        target_tokens = self.encode_targets(future_states)
        if self.codebook is not None:
            target_probabilities = self.assignments(target_tokens)
            predicted_probabilities = self.assignments(predicted_tokens)
            target_representation = torch.einsum(
                "bhec,cd->bhed",
                target_probabilities,
                self.codebook,
            )
            predicted_representation = torch.einsum(
                "bhec,cd->bhed",
                predicted_probabilities,
                self.codebook,
            )
        else:
            target_probabilities = torch.empty(
                0, device=histories.device
            )
            predicted_probabilities = torch.empty(
                0, device=histories.device
            )
            target_representation = target_tokens
            predicted_representation = predicted_tokens
        predicted_states = (
            self.decoder(predicted_representation)
            * self.observation_mask[None, None, :, :]
        )
        target_reconstruction = (
            self.decoder(target_representation)
            * self.observation_mask[None, None, :, :]
        )
        context_reconstruction = (
            self.decoder(context)
            * self.observation_mask[None, :, :]
        )
        return {
            "context": context,
            "predicted_tokens": predicted_tokens,
            "target_tokens": target_tokens,
            "predicted_probabilities": predicted_probabilities,
            "target_probabilities": target_probabilities,
            "predicted_representation": predicted_representation,
            "target_representation": target_representation,
            "predicted_states": predicted_states,
            "target_reconstruction": target_reconstruction,
            "context_reconstruction": context_reconstruction,
        }

    def ema_update(self) -> None:
        decay = self.config.ema_decay
        with torch.no_grad():
            for target, online in zip(
                self.target_state_encoder.parameters(),
                self.state_encoder.parameters(),
            ):
                target.mul_(decay).add_(online, alpha=1.0 - decay)
            self.target_entity_embedding.mul_(decay).add_(
                self.entity_embedding, alpha=1.0 - decay
            )


class PrototypeTorchDynamics:
    """Expose a trained prototype through the common rollout seam."""

    def __init__(
        self,
        *,
        network: PrototypeRegimeNetwork,
        residual_variance: NDArray[np.float64],
        kind: str,
    ) -> None:
        self.network = network.eval()
        self.residual_variance = np.asarray(
            residual_variance, dtype=np.float64
        )
        self.kind = kind

    def fit(
        self, windows: ActionConditionedWindows
    ) -> "PrototypeTorchDynamics":
        del windows
        return self

    def rollout(
        self,
        histories: NDArray[Any],
        future_controls: NDArray[Any],
        future_actions: NDArray[Any],
        graph: DeclaredTelemetryGraph,
    ) -> TrajectoryDistribution:
        if tuple(graph.entity_ids) != tuple(
            entity.entity_id for entity in graph.entities
        ):
            raise ValueError("prototype graph identity is invalid")
        batches = []
        with torch.no_grad():
            for start in range(0, len(histories), 512):
                stop = min(start + 512, len(histories))
                _, predicted = self.network.predict_tokens(
                    _tensor(histories[start:stop]),
                    _tensor(future_controls[start:stop]),
                    _tensor(future_actions[start:stop]),
                )
                if self.network.codebook is not None:
                    probabilities = self.network.assignments(predicted)
                    predicted = torch.einsum(
                        "bhec,cd->bhed",
                        probabilities,
                        self.network.codebook,
                    )
                state = (
                    self.network.decoder(predicted)
                    * self.network.observation_mask[None, None, :, :]
                )
                batches.append(state.cpu().numpy())
        mean = np.concatenate(batches, axis=0).astype(np.float64)
        variance = np.broadcast_to(
            self.residual_variance[None, ...], mean.shape
        ).copy()
        return TrajectoryDistribution(mean=mean, variance=variance)

    @property
    def parameter_count(self) -> int:
        return int(
            sum(
                parameter.numel()
                for parameter in self.network.parameters()
            )
        )

    def to_dict(self) -> Mapping[str, Any]:
        payload = io.BytesIO()
        torch.save(self.network.state_dict(), payload)
        raw = payload.getvalue()
        return {
            "schema_version": 1,
            "kind": self.kind,
            "config": asdict(self.network.config),
            "use_codebook": self.network.use_codebook,
            "parameter_count": self.parameter_count,
            "state_sha256": hashlib.sha256(raw).hexdigest(),
            "state_size_bytes": len(raw),
            "residual_variance": self.residual_variance.tolist(),
        }


class PrototypeSwitchingRegimeDynamics:
    """K-means regime assignments with a conditioned ridge future model."""

    kind = "prototype_switching_regime_ridge"

    def __init__(
        self,
        *,
        code_count: int,
        projection_width: int,
        ridge: float,
        seed: int,
        observation_mask: NDArray[np.bool_],
    ) -> None:
        self.code_count = code_count
        self.projection_width = projection_width
        self.ridge = ridge
        self.seed = seed
        self.observation_mask = observation_mask
        self.center: Optional[NDArray[np.float64]] = None
        self.components: Optional[NDArray[np.float64]] = None
        self.centroids: Optional[NDArray[np.float64]] = None
        self.coefficients: Optional[NDArray[np.float64]] = None
        self.variance: Optional[NDArray[np.float64]] = None
        self.horizon = 0
        self.entity_count = 0
        self.feature_count = 0
        self.control_count = 0
        self.action_count = 0

    def fit(
        self, windows: ActionConditionedWindows
    ) -> "PrototypeSwitchingRegimeDynamics":
        summary = _observable_summary(windows.histories)
        center, components = _randomized_pca_fit(
            summary,
            self.projection_width,
            self.seed,
        )
        projected = (summary - center) @ components.T
        centroids, assignments = _fit_kmeans(
            projected,
            self.code_count,
            self.seed + 1,
        )
        sample_count = len(projected)
        self.horizon = windows.future_states.shape[1]
        self.entity_count = windows.future_states.shape[2]
        self.feature_count = windows.future_states.shape[3]
        self.control_count = windows.future_controls.shape[2]
        self.action_count = windows.future_actions.shape[3]
        one_hot = np.eye(self.code_count, dtype=np.float64)[assignments]
        repeated_projection = np.repeat(
            projected[:, None, :], self.horizon, axis=1
        )
        repeated_codes = np.repeat(
            one_hot[:, None, :], self.horizon, axis=1
        )
        horizon_codes = np.broadcast_to(
            np.eye(self.horizon, dtype=np.float64)[None, :, :],
            (sample_count, self.horizon, self.horizon),
        )
        design = np.concatenate(
            (
                repeated_projection,
                repeated_codes,
                windows.future_controls,
                windows.future_actions.reshape(
                    sample_count, self.horizon, -1
                ),
                horizon_codes,
                np.ones(
                    (sample_count, self.horizon, 1),
                    dtype=np.float64,
                ),
            ),
            axis=2,
        ).reshape(sample_count * self.horizon, -1)
        target = windows.future_states.reshape(
            sample_count * self.horizon, -1
        )
        penalty = np.eye(design.shape[1], dtype=np.float64)
        penalty[-1, -1] = 0.0
        coefficients = np.linalg.solve(
            design.T @ design + self.ridge * penalty,
            design.T @ target,
        )
        residual = (target - design @ coefficients).reshape(
            sample_count,
            self.horizon,
            self.entity_count,
            self.feature_count,
        )
        self.center = center
        self.components = components
        self.centroids = centroids
        self.coefficients = coefficients
        self.variance = np.maximum(
            np.mean(np.square(residual), axis=0), 1e-3
        )
        return self

    def rollout(
        self,
        histories: NDArray[Any],
        future_controls: NDArray[Any],
        future_actions: NDArray[Any],
        graph: DeclaredTelemetryGraph,
    ) -> TrajectoryDistribution:
        del graph
        (
            center,
            components,
            centroids,
            coefficients,
            variance,
        ) = self._fitted()
        summary = _observable_summary(histories)
        projected = (summary - center) @ components.T
        assignments = _nearest_centroid(projected, centroids)
        sample_count = len(projected)
        one_hot = np.eye(self.code_count, dtype=np.float64)[assignments]
        design = np.concatenate(
            (
                np.repeat(
                    projected[:, None, :], self.horizon, axis=1
                ),
                np.repeat(one_hot[:, None, :], self.horizon, axis=1),
                np.asarray(future_controls, dtype=np.float64),
                np.asarray(future_actions, dtype=np.float64).reshape(
                    sample_count, self.horizon, -1
                ),
                np.broadcast_to(
                    np.eye(self.horizon, dtype=np.float64)[None, :, :],
                    (sample_count, self.horizon, self.horizon),
                ),
                np.ones(
                    (sample_count, self.horizon, 1), dtype=np.float64
                ),
            ),
            axis=2,
        ).reshape(sample_count * self.horizon, -1)
        mean = (design @ coefficients).reshape(
            sample_count,
            self.horizon,
            self.entity_count,
            self.feature_count,
        )
        mean *= self.observation_mask[None, None, :, :]
        return TrajectoryDistribution(
            mean=mean,
            variance=np.broadcast_to(
                variance[None, ...], mean.shape
            ).copy(),
        )

    @property
    def parameter_count(self) -> int:
        values = self._fitted()
        return int(sum(value.size for value in values))

    def to_dict(self) -> Mapping[str, Any]:
        values = self._fitted()
        digest = hashlib.sha256()
        for value in values:
            digest.update(np.ascontiguousarray(value).tobytes())
        return {
            "schema_version": 1,
            "kind": self.kind,
            "code_count": self.code_count,
            "projection_width": self.projection_width,
            "ridge": self.ridge,
            "parameter_count": self.parameter_count,
            "state_sha256": digest.hexdigest(),
        }

    def _fitted(
        self,
    ) -> Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        if (
            self.center is None
            or self.components is None
            or self.centroids is None
            or self.coefficients is None
            or self.variance is None
        ):
            raise ValueError("switching prototype is not fitted")
        return (
            self.center,
            self.components,
            self.centroids,
            self.coefficients,
            self.variance,
        )


def _tensor(values: NDArray[Any]) -> torch.Tensor:
    return torch.from_numpy(np.asarray(values, dtype=np.float32))


def _observation_mask(
    fit: ActionConditionedWindows,
) -> NDArray[np.bool_]:
    entity_positions = {
        name: position
        for position, name in enumerate(fit.entity_names)
    }
    feature_positions = {
        name: position
        for position, name in enumerate(fit.state_feature_names)
    }
    mask = np.zeros(
        (len(fit.entity_names), len(fit.state_feature_names)),
        dtype=np.bool_,
    )
    for binding in fit.graph.bindings:
        modality, feature_name = binding.feature_key.split(".", 1)
        if modality == "metric":
            mask[
                entity_positions[binding.entity_id],
                feature_positions[feature_name],
            ] = True
    values = np.concatenate(
        (fit.histories, fit.future_states), axis=1
    )
    spread = np.std(values, axis=(0, 1))
    for feature_name in (
        "log_event_count",
        "log_error_count",
        "trace_span_count",
        "trace_error_count",
    ):
        position = feature_positions[feature_name]
        mask[:, position] = spread[:, position] > 1e-2
    return mask


def _masked_mse(
    prediction: torch.Tensor,
    observed: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    while mask.ndim < prediction.ndim:
        mask = mask.unsqueeze(0)
    return torch.sum(torch.square(prediction - observed) * mask) / (
        torch.sum(mask)
        * np.prod(prediction.shape[: prediction.ndim - 2])
    )


def _train_network(
    *,
    fit: ActionConditionedWindows,
    config: PrototypeConfig,
    observation_mask: NDArray[np.bool_],
    use_codebook: bool,
) -> Tuple[PrototypeTorchDynamics, Mapping[str, Any]]:
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    network = PrototypeRegimeNetwork(
        entity_count=len(fit.entity_names),
        feature_count=len(fit.state_feature_names),
        control_count=len(fit.control_feature_names),
        action_count=len(fit.action_feature_names),
        horizon=fit.future_states.shape[1],
        observation_mask=observation_mask,
        config=config,
        use_codebook=use_codebook,
    )
    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter in network.parameters()
            if parameter.requires_grad
        ],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    rng = np.random.default_rng(config.seed)
    epoch_rows = []
    started = time.perf_counter()
    network.train()
    for epoch in range(config.epochs):
        losses: Dict[str, list[float]] = {}
        for start in range(0, len(fit.histories), config.batch_size):
            if start == 0:
                order = rng.permutation(len(fit.histories))
            positions = order[start : start + config.batch_size]
            histories = _tensor(fit.histories[positions])
            controls = _tensor(fit.future_controls[positions])
            actions = _tensor(fit.future_actions[positions])
            future = _tensor(fit.future_states[positions])
            output = network(histories, controls, actions, future)
            mask = network.observation_mask
            future_state_loss = _masked_mse(
                output["predicted_states"], future, mask
            )
            latent_loss = torch.mean(
                torch.abs(
                    output["predicted_representation"]
                    - output["target_representation"].detach()
                )
            )
            target_reconstruction_loss = _masked_mse(
                output["target_reconstruction"], future, mask
            )
            context_reconstruction_loss = _masked_mse(
                output["context_reconstruction"],
                histories[:, -1],
                mask,
            )
            if use_codebook:
                target_probabilities = output["target_probabilities"]
                predicted_probabilities = output[
                    "predicted_probabilities"
                ]
                assignment_loss = torch.mean(
                    -torch.sum(
                        target_probabilities.detach()
                        * torch.log(
                            predicted_probabilities.clamp(min=1e-8)
                        ),
                        dim=-1,
                    )
                )
                marginal = target_probabilities.mean(dim=(0, 1, 2))
                balance_loss = torch.sum(
                    marginal
                    * torch.log(
                        (
                            marginal * config.code_count
                        ).clamp(min=1e-8)
                    )
                )
                sharpness_loss = torch.mean(
                    -torch.sum(
                        target_probabilities
                        * torch.log(
                            target_probabilities.clamp(min=1e-8)
                        ),
                        dim=-1,
                    )
                )
            else:
                assignment_loss = torch.zeros(())
                balance_loss = torch.zeros(())
                sharpness_loss = torch.zeros(())
            total = (
                config.future_state_weight * future_state_loss
                + config.latent_weight * latent_loss
                + config.assignment_weight * assignment_loss
                + config.target_reconstruction_weight
                * target_reconstruction_loss
                + config.context_reconstruction_weight
                * context_reconstruction_loss
                + config.balance_weight * balance_loss
                + config.sharpness_weight * sharpness_loss
            )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 5.0)
            optimizer.step()
            network.ema_update()
            for name, value in {
                "total": total,
                "future_state": future_state_loss,
                "latent": latent_loss,
                "assignment": assignment_loss,
                "target_reconstruction": target_reconstruction_loss,
                "context_reconstruction": context_reconstruction_loss,
                "balance": balance_loss,
                "sharpness": sharpness_loss,
            }.items():
                losses.setdefault(name, []).append(
                    float(value.detach().cpu())
                )
        epoch_row = {
            "epoch": epoch + 1,
            **{
                name: float(np.mean(values))
                for name, values in losses.items()
            },
        }
        epoch_rows.append(epoch_row)
        if epoch in (0, 9, 19, 29, config.epochs - 1):
            _print_state(
                "training",
                {
                    "model": (
                        "regime_codebook"
                        if use_codebook
                        else "continuous_null"
                    ),
                    **epoch_row,
                },
            )
    training_seconds = time.perf_counter() - started
    network.eval()
    fit_prediction = _predict_network(
        network,
        fit.histories,
        fit.future_controls,
        fit.future_actions,
    )
    residual_variance = np.maximum(
        np.mean(
            np.square(fit_prediction - fit.future_states), axis=0
        ),
        1e-3,
    )
    model = PrototypeTorchDynamics(
        network=network,
        residual_variance=residual_variance,
        kind=(
            "prototype_soft_regime_codebook_jepa"
            if use_codebook
            else "prototype_continuous_jepa_null"
        ),
    )
    parity = _restoration_parity(
        model,
        fit.histories[:8],
        fit.future_controls[:8],
        fit.future_actions[:8],
        fit.graph,
    )
    return model, {
        "training_seconds": training_seconds,
        "epochs": epoch_rows,
        "restoration_parity": parity,
    }


def _predict_network(
    network: PrototypeRegimeNetwork,
    histories: NDArray[Any],
    controls: NDArray[Any],
    actions: NDArray[Any],
) -> NDArray[np.float64]:
    model = PrototypeTorchDynamics(
        network=network,
        residual_variance=np.ones(
            (
                network.horizon,
                network.entity_count,
                network.feature_count,
            ),
            dtype=np.float64,
        ),
        kind="prototype_prediction_adapter",
    )
    graph = _GRAPH_HOLDER["graph"]
    return model.rollout(histories, controls, actions, graph).mean


def _restoration_parity(
    model: PrototypeTorchDynamics,
    histories: NDArray[Any],
    controls: NDArray[Any],
    actions: NDArray[Any],
    graph: DeclaredTelemetryGraph,
) -> Mapping[str, Any]:
    original = model.rollout(histories, controls, actions, graph).mean
    restored_network = PrototypeRegimeNetwork(
        entity_count=model.network.entity_count,
        feature_count=model.network.feature_count,
        control_count=model.network.control_count,
        action_count=model.network.action_count,
        horizon=model.network.horizon,
        observation_mask=model.network.observation_mask.cpu().numpy().astype(
            np.bool_
        ),
        config=model.network.config,
        use_codebook=model.network.use_codebook,
    )
    payload = io.BytesIO()
    torch.save(model.network.state_dict(), payload)
    payload.seek(0)
    restored_network.load_state_dict(
        torch.load(payload, map_location="cpu", weights_only=True)
    )
    restored = PrototypeTorchDynamics(
        network=restored_network,
        residual_variance=model.residual_variance.copy(),
        kind=model.kind,
    ).rollout(histories, controls, actions, graph).mean
    return {
        "maximum_absolute_difference": float(
            np.max(np.abs(original - restored))
        ),
        "exact": bool(np.array_equal(original, restored)),
        "finite": bool(
            np.all(np.isfinite(original))
            and np.all(np.isfinite(restored))
        ),
    }


def _encode_network(
    model: PrototypeTorchDynamics,
    histories: NDArray[Any],
) -> Tuple[NDArray[np.float64], Optional[NDArray[np.float64]]]:
    token_batches = []
    probability_batches = []
    with torch.no_grad():
        for start in range(0, len(histories), 512):
            context = model.network.encode_context(
                _tensor(histories[start : start + 512])
            )
            if model.network.codebook is not None:
                probabilities = model.network.assignments(context)
                context = torch.einsum(
                    "bec,cd->bed",
                    probabilities,
                    model.network.codebook,
                )
                probability_batches.append(
                    probabilities.cpu().numpy()
                )
            token_batches.append(context.cpu().numpy())
    return (
        np.concatenate(token_batches, axis=0).astype(np.float64),
        (
            np.concatenate(probability_batches, axis=0).astype(
                np.float64
            )
            if probability_batches
            else None
        ),
    )


def _probe_scores(
    train_representation: NDArray[np.float64],
    train_states: NDArray[np.float64],
    evaluation_representation: NDArray[np.float64],
    evaluation_states: NDArray[np.float64],
    observation_mask: NDArray[np.bool_],
    entity_names: Sequence[str],
    *,
    ridge: float,
) -> Mapping[str, Any]:
    entity_rows = {}
    squared_errors = []
    for entity_position, entity_name in enumerate(entity_names):
        mask = observation_mask[entity_position]
        if not np.any(mask):
            entity_rows[entity_name] = {
                "observed_feature_count": 0,
                "nrmse": None,
            }
            continue
        training = np.concatenate(
            (
                train_representation[:, entity_position],
                np.ones((len(train_representation), 1)),
            ),
            axis=1,
        )
        evaluation = np.concatenate(
            (
                evaluation_representation[:, entity_position],
                np.ones((len(evaluation_representation), 1)),
            ),
            axis=1,
        )
        penalty = np.eye(training.shape[1], dtype=np.float64)
        penalty[-1, -1] = 0.0
        coefficients = np.linalg.solve(
            training.T @ training + ridge * penalty,
            training.T @ train_states[:, entity_position, :][:, mask],
        )
        residual = (
            evaluation @ coefficients
            - evaluation_states[:, entity_position, :][:, mask]
        )
        entity_error = np.square(residual).reshape(-1)
        squared_errors.append(entity_error)
        entity_rows[entity_name] = {
            "observed_feature_count": int(np.sum(mask)),
            "nrmse": float(np.sqrt(np.mean(entity_error))),
        }
    return {
        "aggregate_nrmse": float(
            np.sqrt(np.mean(np.concatenate(squared_errors)))
        ),
        "entities": entity_rows,
    }


def _pca_representations(
    fit_histories: NDArray[Any],
    evaluation_histories: NDArray[Any],
    width: int,
    seed: int,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    train_tokens = []
    evaluation_tokens = []
    for entity_position in range(fit_histories.shape[2]):
        training = np.asarray(
            fit_histories[:, :, entity_position, :], dtype=np.float64
        ).reshape(len(fit_histories), -1)
        evaluation = np.asarray(
            evaluation_histories[:, :, entity_position, :],
            dtype=np.float64,
        ).reshape(len(evaluation_histories), -1)
        center, components = _randomized_pca_fit(
            training, width, seed + entity_position
        )
        train_tokens.append((training - center) @ components.T)
        evaluation_tokens.append((evaluation - center) @ components.T)
    return (
        np.stack(train_tokens, axis=1),
        np.stack(evaluation_tokens, axis=1),
    )


def _code_usage(
    probabilities: NDArray[np.float64],
    entity_names: Sequence[str],
) -> Mapping[str, Any]:
    marginal = np.mean(probabilities, axis=(0, 1))
    entropy = -np.sum(
        probabilities
        * np.log(np.maximum(probabilities, 1e-12)),
        axis=-1,
    )
    entity_rows = {}
    for position, name in enumerate(entity_names):
        entity_marginal = np.mean(probabilities[:, position], axis=0)
        entity_rows[name] = {
            "active_above_half_percent": int(
                np.sum(entity_marginal > 0.005)
            ),
            "perplexity": float(
                np.exp(
                    -np.sum(
                        entity_marginal
                        * np.log(
                            np.maximum(entity_marginal, 1e-12)
                        )
                    )
                )
            ),
        }
    return {
        "active_above_half_percent": int(
            np.sum(marginal > 0.005)
        ),
        "marginal_perplexity": float(
            np.exp(
                -np.sum(marginal * np.log(np.maximum(marginal, 1e-12)))
            )
        ),
        "mean_assignment_entropy": float(np.mean(entropy)),
        "maximum_assignment_probability": float(
            np.mean(np.max(probabilities, axis=-1))
        ),
        "marginal_probabilities": marginal.tolist(),
        "entities": entity_rows,
    }


def _observable_summary(histories: NDArray[Any]) -> NDArray[np.float64]:
    values = np.asarray(histories, dtype=np.float64)
    return np.concatenate(
        (
            values[:, -1].reshape(len(values), -1),
            np.mean(values[:, -4:], axis=1).reshape(len(values), -1),
            (values[:, -1] - values[:, 0]).reshape(len(values), -1),
        ),
        axis=1,
    )


def _randomized_pca_fit(
    values: NDArray[np.float64], width: int, seed: int
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    center = np.mean(values, axis=0)
    centered = values - center
    rank = min(width + 8, centered.shape[1], centered.shape[0])
    rng = np.random.default_rng(seed)
    projection = rng.normal(size=(centered.shape[1], rank))
    q, _ = np.linalg.qr(centered @ projection, mode="reduced")
    _, _, right = np.linalg.svd(q.T @ centered, full_matrices=False)
    return center, right[:width]


def _fit_kmeans(
    values: NDArray[np.float64], code_count: int, seed: int
) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
    rng = np.random.default_rng(seed)
    centroids = values[
        rng.choice(len(values), size=code_count, replace=False)
    ].copy()
    assignments = np.zeros(len(values), dtype=np.int64)
    for _ in range(25):
        updated = _nearest_centroid(values, centroids)
        if np.array_equal(updated, assignments):
            break
        assignments = updated
        for code in range(code_count):
            selected = values[assignments == code]
            if len(selected):
                centroids[code] = np.mean(selected, axis=0)
            else:
                centroids[code] = values[rng.integers(len(values))]
    return centroids, assignments


def _nearest_centroid(
    values: NDArray[np.float64],
    centroids: NDArray[np.float64],
) -> NDArray[np.int64]:
    squared = (
        np.sum(np.square(values), axis=1, keepdims=True)
        + np.sum(np.square(centroids), axis=1)[None, :]
        - 2.0 * values @ centroids.T
    )
    return np.argmin(squared, axis=1).astype(np.int64)


def _partition_queries(
    prepared: PreparedEdgeDynamicsData,
    *,
    held_out_normalized_value: float,
) -> Tuple[PreparedAttributionQueries, PreparedAttributionQueries]:
    control_names = prepared.windows["fit"].control_feature_names
    position = control_names.index("worker_replicas")
    values = prepared.attribution_queries.future_controls[
        :, 0, position
    ]
    transfer = np.isclose(values, held_out_normalized_value)
    return (
        subset_attribution_queries(
            prepared.attribution_queries, ~transfer
        ),
        subset_attribution_queries(
            prepared.attribution_queries, transfer
        ),
    )


def _print_state(stage: str, payload: Mapping[str, Any]) -> None:
    print(
        json.dumps(
            {"stage": stage, **dict(payload)},
            sort_keys=True,
        ),
        flush=True,
    )


def _score_candidate(
    *,
    model: Any,
    selection: ActionConditionedWindows,
    in_distribution: ActionConditionedWindows,
    transfer: ActionConditionedWindows,
    in_distribution_queries: PreparedAttributionQueries,
    transfer_queries: PreparedAttributionQueries,
    calibration: ActionConditionedWindows,
    seed: int,
) -> Mapping[str, Any]:
    return {
        "selection": dict(forecast_objective(model, selection)),
        "in_distribution": score_edge_model(
            model, in_distribution, in_distribution_queries
        ).to_dict(),
        "transfer": score_edge_model(
            model, transfer, transfer_queries
        ).to_dict(),
        "action_sanity": action_conditioning_sanity(
            model, transfer, seed=seed
        ),
        "alert_policy": conformal_sequential_detection(
            model=model,
            calibration=calibration,
            evaluation=transfer,
        ),
    }


def _assess(
    *,
    raw: Mapping[str, Any],
    continuous: Mapping[str, Any],
    codebook: Mapping[str, Any],
    code_usage: Mapping[str, Any],
    continuous_probe: Mapping[str, Any],
    codebook_probe: Mapping[str, Any],
    restoration: Mapping[str, Any],
) -> Mapping[str, Any]:
    raw_transfer = raw["transfer"]
    continuous_transfer = continuous["transfer"]
    codebook_transfer = codebook["transfer"]
    safety = {
        "at_least_eight_active_codes": (
            code_usage["active_above_half_percent"] >= 8
        ),
        "marginal_perplexity_at_least_eight": (
            code_usage["marginal_perplexity"] >= 8.0
        ),
        "every_observed_entity_has_multiple_codes": all(
            row["active_above_half_percent"] >= 2
            for row in code_usage["entities"].values()
        ),
        "probe_no_worse_than_continuous": (
            codebook_probe["aggregate_nrmse"]
            <= continuous_probe["aggregate_nrmse"]
        ),
        "action_mse_within_five_percent_of_raw": (
            codebook_transfer["normalized_mse_action_overlap"]
            <= 1.05
            * raw_transfer["normalized_mse_action_overlap"]
        ),
        "overall_mse_within_five_percent_of_raw": (
            codebook_transfer["normalized_mse_overall"]
            <= 1.05 * raw_transfer["normalized_mse_overall"]
        ),
        "restoration_finite_and_exact": (
            restoration["finite"] and restoration["exact"]
        ),
    }
    predictive = {
        "downstream_effect_improves_ten_percent": (
            codebook_transfer["downstream_effect_mse"]
            <= 0.90 * raw_transfer["downstream_effect_mse"]
        ),
        "attribution_at_least_ninety_five_percent": (
            codebook_transfer["action_and_target_hit_at_1"] >= 0.95
        ),
        "no_action_specificity_is_one": (
            codebook_transfer["no_action_specificity"] == 1.0
        ),
    }
    code_sanity = codebook["action_sanity"]
    continuous_sanity = continuous["action_sanity"]
    investigation = {
        "attribution_at_least_ninety_five_percent": (
            codebook_transfer["action_and_target_hit_at_1"] >= 0.95
        ),
        "no_action_specificity_is_one": (
            codebook_transfer["no_action_specificity"] == 1.0
        ),
        "action_sanity_at_least_eighty_percent": (
            code_sanity["correct_action_beats_both_fraction"] >= 0.80
        ),
        "improves_over_continuous_null": (
            codebook_transfer["action_and_target_hit_at_1"]
            > continuous_transfer["action_and_target_hit_at_1"]
            or code_sanity["correct_action_beats_both_fraction"]
            > continuous_sanity["correct_action_beats_both_fraction"]
        ),
    }
    code_alert = codebook["alert_policy"]
    continuous_alert = continuous["alert_policy"]
    code_delay = code_alert["median_sequential_detection_delay_transitions"]
    continuous_delay = continuous_alert[
        "median_sequential_detection_delay_transitions"
    ]
    alert = {
        "control_false_alarm_at_most_five_percent": (
            code_alert["evaluation_control_sequential_false_alarm_rate"]
            <= 0.05
        ),
        "treatment_detection_at_least_eighty_percent": (
            code_alert["evaluation_treatment_sequential_detection_rate"]
            >= 0.80
        ),
        "median_delay_at_most_ten": (
            code_delay is not None and code_delay <= 10.0
        ),
        "improves_over_continuous_null": (
            code_alert["evaluation_treatment_sequential_detection_rate"]
            > continuous_alert[
                "evaluation_treatment_sequential_detection_rate"
            ]
            or (
                code_delay is not None
                and continuous_delay is not None
                and code_delay < continuous_delay
            )
        ),
    }
    safety_passed = all(safety.values())
    lane_passes = {
        "predictive": all(predictive.values()),
        "investigation": all(investigation.values()),
        "alert": all(alert.values()),
    }
    return {
        "safety": safety,
        "value_lanes": {
            "predictive": predictive,
            "investigation": investigation,
            "alert": alert,
        },
        "safety_passed": safety_passed,
        "lane_passes": lane_passes,
        "decision": (
            "advance_to_durable_implementation"
            if safety_passed and any(lane_passes.values())
            else "reject_prototype_recipe"
        ),
    }


_GRAPH_HOLDER: Dict[str, DeclaredTelemetryGraph] = {}


def run_prototype(
    *,
    corpus_directory: Path,
    cache_root: Path,
    output_directory: Path,
    epochs: int,
) -> Mapping[str, Any]:
    if output_directory.exists():
        raise FileExistsError(
            f"refusing to overwrite prototype output: {output_directory}"
        )
    output_directory.mkdir(parents=True)
    source_manifest = source_artifact_manifest_sha256(
        corpus_directory
    )
    cache_directory = cache_root / topology_transfer_cache_address(
        source_manifest
    )
    prepared = load_edge_dynamics_cache(cache_directory)
    validate_topology_transfer_cache(prepared, corpus_directory)
    _GRAPH_HOLDER["graph"] = prepared.graph
    partitions = {
        role: partition_worker_topology(windows)
        for role, windows in prepared.windows.items()
    }
    held_out_value = partitions[
        "fit"
    ].held_out_normalized_value
    fit = partitions["fit"].in_distribution
    selection = partitions["selection"].in_distribution
    calibration = partitions["calibration"].in_distribution
    in_distribution = partitions["evaluation"].in_distribution
    transfer = partitions["evaluation"].held_out
    in_distribution_queries, transfer_queries = _partition_queries(
        prepared, held_out_normalized_value=held_out_value
    )
    mask = _observation_mask(fit)
    config = PrototypeConfig(epochs=epochs)
    _print_state(
        "loaded",
        {
            "source_manifest": source_manifest,
            "cache": str(cache_directory),
            "fit_windows": len(fit.histories),
            "selection_windows": len(selection.histories),
            "calibration_windows": len(calibration.histories),
            "in_distribution_windows": len(in_distribution.histories),
            "transfer_windows": len(transfer.histories),
            "observed_entity_feature_slots": int(np.sum(mask)),
            "config": asdict(config),
        },
    )

    raw_started = time.perf_counter()
    raw_model = ContractiveLowRankDynamics(
        LowRankConfig(rank=32)
    ).fit(fit)
    raw_training_seconds = time.perf_counter() - raw_started
    _print_state(
        "fitted",
        {
            "model": "raw_low_rank",
            "training_seconds": raw_training_seconds,
        },
    )

    switching_started = time.perf_counter()
    switching_model = PrototypeSwitchingRegimeDynamics(
        code_count=config.code_count,
        projection_width=32,
        ridge=config.ridge,
        seed=config.seed,
        observation_mask=mask,
    ).fit(fit)
    switching_training_seconds = time.perf_counter() - switching_started
    _print_state(
        "fitted",
        {
            "model": "switching_regime_ridge",
            "training_seconds": switching_training_seconds,
        },
    )

    continuous_model, continuous_training = _train_network(
        fit=fit,
        config=config,
        observation_mask=mask,
        use_codebook=False,
    )
    codebook_model, codebook_training = _train_network(
        fit=fit,
        config=config,
        observation_mask=mask,
        use_codebook=True,
    )

    model_scores = {}
    for name, model in {
        "raw_low_rank": raw_model,
        "switching_regime_ridge": switching_model,
        "continuous_null": continuous_model,
        "regime_codebook": codebook_model,
    }.items():
        _print_state("scoring", {"model": name})
        model_scores[name] = _score_candidate(
            model=model,
            selection=selection,
            in_distribution=in_distribution,
            transfer=transfer,
            in_distribution_queries=in_distribution_queries,
            transfer_queries=transfer_queries,
            calibration=calibration,
            seed=config.seed + 401,
        )

    fit_continuous, _ = _encode_network(
        continuous_model, fit.histories
    )
    transfer_continuous, _ = _encode_network(
        continuous_model, transfer.histories
    )
    fit_codebook, fit_probabilities = _encode_network(
        codebook_model, fit.histories
    )
    transfer_codebook, transfer_probabilities = _encode_network(
        codebook_model, transfer.histories
    )
    if fit_probabilities is None or transfer_probabilities is None:
        raise ValueError("codebook prototype did not emit probabilities")
    fit_pca, transfer_pca = _pca_representations(
        fit.histories,
        transfer.histories,
        config.latent_dimension,
        config.seed + 77,
    )
    train_current = fit.histories[:, -1]
    transfer_current = transfer.histories[:, -1]
    probes = {
        "continuous_null": _probe_scores(
            fit_continuous,
            train_current,
            transfer_continuous,
            transfer_current,
            mask,
            fit.entity_names,
            ridge=config.ridge,
        ),
        "regime_codebook": _probe_scores(
            fit_codebook,
            train_current,
            transfer_codebook,
            transfer_current,
            mask,
            fit.entity_names,
            ridge=config.ridge,
        ),
        "matched_pca": _probe_scores(
            fit_pca,
            train_current,
            transfer_pca,
            transfer_current,
            mask,
            fit.entity_names,
            ridge=config.ridge,
        ),
    }
    code_usage = _code_usage(
        transfer_probabilities, fit.entity_names
    )
    assessment = _assess(
        raw=model_scores["raw_low_rank"],
        continuous=model_scores["continuous_null"],
        codebook=model_scores["regime_codebook"],
        code_usage=code_usage,
        continuous_probe=probes["continuous_null"],
        codebook_probe=probes["regime_codebook"],
        restoration=codebook_training["restoration_parity"],
    )
    result = {
        "schema_version": 1,
        "kind": "prototype_soft_regime_codebook_jepa_result_v1",
        "evidence_boundary": (
            "throwaway prototype on open development evidence"
        ),
        "source_corpus_sha256": prepared.source_corpus_sha256,
        "source_artifact_manifest_sha256": (
            prepared.source_artifact_manifest_sha256
        ),
        "preprocessing_protocol": prepared.preprocessing_protocol,
        "held_out_topology_normalized_value": held_out_value,
        "config": asdict(config),
        "role_counts": {
            "fit_pairs": len(set(fit.matched_pair_ids)),
            "selection_pairs": len(set(selection.matched_pair_ids)),
            "calibration_pairs": len(set(calibration.matched_pair_ids)),
            "in_distribution_pairs": len(
                set(in_distribution.matched_pair_ids)
            ),
            "transfer_pairs": len(set(transfer.matched_pair_ids)),
        },
        "training": {
            "raw_low_rank_seconds": raw_training_seconds,
            "switching_regime_seconds": switching_training_seconds,
            "continuous_null": continuous_training,
            "regime_codebook": codebook_training,
        },
        "model_scores": model_scores,
        "observable_state_probes": probes,
        "code_usage": code_usage,
        "assessment": assessment,
        "limitations": [
            "the source corpus and evaluation roles were already open",
            "the neural candidates use one deterministic seed",
            "the direct horizon predictor is a prototype, not an edge runtime",
            "the switching control is k-means plus ridge, not a full HMM",
            "the codebook uses normalized structured state, not raw log text",
        ],
    }
    result_path = output_directory / "prototype-result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _print_state(
        "complete",
        {
            "output": str(result_path),
            "assessment": assessment,
            "transfer_summary": {
                name: {
                    "action_mse": values["transfer"][
                        "normalized_mse_action_overlap"
                    ],
                    "overall_mse": values["transfer"][
                        "normalized_mse_overall"
                    ],
                    "downstream_effect_mse": values["transfer"][
                        "downstream_effect_mse"
                    ],
                    "attribution": values["transfer"][
                        "action_and_target_hit_at_1"
                    ],
                    "specificity": values["transfer"][
                        "no_action_specificity"
                    ],
                }
                for name, values in model_scores.items()
            },
            "probe_summary": {
                name: row["aggregate_nrmse"]
                for name, row in probes.items()
            },
            "code_usage": {
                "active_codes": code_usage[
                    "active_above_half_percent"
                ],
                "perplexity": code_usage["marginal_perplexity"],
                "mean_assignment_entropy": code_usage[
                    "mean_assignment_entropy"
                ],
            },
        },
    )
    return result


def main(arguments: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("artifacts/action-dynamics/development-v1"),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(
            "artifacts/action-dynamics/edge-preprocessing-v1"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/action-dynamics/"
            "prototype-regime-codebook-jepa-v1"
        ),
    )
    parser.add_argument("--epochs", type=int, default=40)
    parsed = parser.parse_args(arguments)
    torch.set_num_threads(min(8, max(1, torch.get_num_threads())))
    run_prototype(
        corpus_directory=parsed.corpus,
        cache_root=parsed.cache_root,
        output_directory=parsed.output,
        epochs=parsed.epochs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
