"""Node-preserving JEPA with low-rank action-conditioned latent dynamics.

PyTorch is imported lazily so NumPy-only corpus ingestion remains available
without the optional training dependency.
"""

import copy
from dataclasses import asdict, dataclass
import importlib
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ..action_conditioned_dynamics import (
    ActionConditionedWindows,
    TrajectoryDistribution,
)
from ..graph_telemetry import DeclaredTelemetryGraph
from .models import validate_edge_rollout


_OBJECTIVES = ("jepa", "supervised")


@dataclass(frozen=True)
class ActionConditionedJepaConfig:
    """Architecture and optimization choices for one development model."""

    node_latent_dimension: int = 16
    transition_rank: int = 32
    epochs: int = 60
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    ema_decay: float = 0.996
    maximum_spectral_radius: float = 0.98
    mask_time_fraction: float = 0.6
    mask_entity_fraction: float = 0.5
    latent_prediction_weight: float = 1.0
    reconstruction_weight: float = 0.2
    context_reconstruction_weight: float = 0.1
    variance_weight: float = 0.05
    covariance_weight: float = 0.01
    variance_floor: float = 1e-4
    objective: str = "jepa"
    device: str = "auto"
    seed: int = 89

    def __post_init__(self) -> None:
        integers = (
            self.node_latent_dimension,
            self.transition_rank,
            self.epochs,
            self.batch_size,
        )
        weights = (
            self.learning_rate,
            self.latent_prediction_weight,
            self.reconstruction_weight,
            self.context_reconstruction_weight,
            self.variance_weight,
            self.covariance_weight,
            self.variance_floor,
        )
        if any(
            isinstance(value, bool) or value < 1 for value in integers
        ):
            raise ValueError("JEPA integer controls must be positive")
        if (
            self.weight_decay < 0.0
            or any(value < 0.0 for value in weights)
            or self.learning_rate <= 0.0
            or self.variance_floor <= 0.0
            or not 0.0 <= self.ema_decay < 1.0
            or not 0.0 < self.maximum_spectral_radius <= 1.0
            or not 0.0 <= self.mask_time_fraction < 1.0
            or not 0.0 <= self.mask_entity_fraction < 1.0
            or self.objective not in _OBJECTIVES
            or self.device not in ("auto", "cpu", "mps")
        ):
            raise ValueError("JEPA numeric or categorical controls are invalid")
        if (
            self.objective == "jepa"
            and self.latent_prediction_weight <= 0.0
        ):
            raise ValueError("JEPA objective requires latent prediction")

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-compatible configuration."""

        return dict(asdict(self))

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "ActionConditionedJepaConfig":
        """Restore a configuration from serialized values."""

        return cls(
            node_latent_dimension=int(
                payload["node_latent_dimension"]
            ),
            transition_rank=int(payload["transition_rank"]),
            epochs=int(payload["epochs"]),
            batch_size=int(payload["batch_size"]),
            learning_rate=float(payload["learning_rate"]),
            weight_decay=float(payload["weight_decay"]),
            ema_decay=float(payload["ema_decay"]),
            maximum_spectral_radius=float(
                payload["maximum_spectral_radius"]
            ),
            mask_time_fraction=float(payload["mask_time_fraction"]),
            mask_entity_fraction=float(
                payload["mask_entity_fraction"]
            ),
            latent_prediction_weight=float(
                payload["latent_prediction_weight"]
            ),
            reconstruction_weight=float(
                payload["reconstruction_weight"]
            ),
            context_reconstruction_weight=float(
                payload["context_reconstruction_weight"]
            ),
            variance_weight=float(payload["variance_weight"]),
            covariance_weight=float(payload["covariance_weight"]),
            variance_floor=float(payload["variance_floor"]),
            objective=str(payload["objective"]),
            device=str(payload["device"]),
            seed=int(payload["seed"]),
        )


class ActionConditionedJepaDynamics:
    """EMA JEPA representation with a low-rank global latent transition."""

    kind = "action_conditioned_jepa_low_rank_dynamics_v1"

    def __init__(
        self,
        config: ActionConditionedJepaConfig = (
            ActionConditionedJepaConfig()
        ),
    ) -> None:
        self.config = config
        self.device = "uninitialized"
        self.training_metrics: Tuple[Mapping[str, float], ...] = ()
        self._network: Any = None
        self._graph: Optional[DeclaredTelemetryGraph] = None
        self._state_shape: Optional[Tuple[int, int]] = None
        self._control_count = 0
        self._action_shape: Optional[Tuple[int, int]] = None
        self._residual_variance: Optional[
            NDArray[np.float64]
        ] = None
        self._spectral_radius = 0.0

    def fit(
        self, windows: ActionConditionedWindows
    ) -> "ActionConditionedJepaDynamics":
        """Fit masked latent and decoded future prediction."""

        torch = _require_torch()
        self.device = _select_device(torch, self.config.device)
        _seed_torch(torch, self.config.seed)
        if self.device == "cpu":
            torch.set_num_threads(
                max(1, min(4, int(torch.get_num_threads())))
            )
        self._register_schema(windows)
        schema = self._schema()
        latent_width = (
            schema["entity_count"] * self.config.node_latent_dimension
        )
        if self.config.transition_rank > latent_width:
            raise ValueError("transition rank exceeds total latent width")
        self._network = _build_network(torch, self.config, schema)
        self._network.to(self.device)
        optimizer = torch.optim.AdamW(
            self._network.trainable_parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        generator = np.random.default_rng(self.config.seed)
        sample_count = len(windows.histories)
        metrics: List[Mapping[str, float]] = []
        for _ in range(self.config.epochs):
            order = generator.permutation(sample_count)
            sums = {
                "total": 0.0,
                "latent": 0.0,
                "reconstruction": 0.0,
                "context_reconstruction": 0.0,
                "variance": 0.0,
                "covariance": 0.0,
            }
            batches = 0
            self._network.train()
            for start in range(0, sample_count, self.config.batch_size):
                selection = order[
                    start : start + self.config.batch_size
                ]
                batch = _window_batch(
                    torch, windows, selection, self.device
                )
                visible = _sample_visible_mask(
                    generator=generator,
                    batch_size=len(selection),
                    time_count=windows.histories.shape[1],
                    entity_count=windows.histories.shape[2],
                    time_fraction=(
                        self.config.mask_time_fraction
                        if self.config.objective == "jepa"
                        else 0.0
                    ),
                    entity_fraction=(
                        self.config.mask_entity_fraction
                        if self.config.objective == "jepa"
                        else 0.0
                    ),
                )
                visible_tensor = torch.as_tensor(
                    visible,
                    dtype=torch.bool,
                    device=self.device,
                )
                optimizer.zero_grad(set_to_none=True)
                output = self._network.forward_training(
                    batch, visible_tensor
                )
                components = _loss_components(
                    torch, output, batch, self.config
                )
                components["total"].backward()
                optimizer.step()
                self._network.project_transition(
                    self.config.maximum_spectral_radius
                )
                self._network.update_target(self.config.ema_decay)
                for name in sums:
                    sums[name] += float(
                        components[name].detach().cpu()
                    )
                batches += 1
            epoch = {
                name: value / float(max(batches, 1))
                for name, value in sums.items()
            }
            metrics.append(epoch)
        self.training_metrics = tuple(metrics)
        self._spectral_radius = self._network.spectral_radius()
        self._residual_variance = self._estimate_residual_variance(
            windows
        )
        return self

    def rollout(
        self,
        histories: NDArray[Any],
        future_controls: NDArray[Any],
        future_actions: NDArray[Any],
        graph: DeclaredTelemetryGraph,
    ) -> TrajectoryDistribution:
        """Predict normalized future observations through latent rollout."""

        (
            fitted_graph,
            state_shape,
            control_count,
            action_shape,
            variance,
        ) = self._fitted_values()
        history = np.asarray(histories, dtype=np.float64)
        controls = np.asarray(future_controls, dtype=np.float64)
        actions = np.asarray(future_actions, dtype=np.float64)
        validate_edge_rollout(
            history,
            controls,
            actions,
            graph,
            fitted_graph,
            state_shape,
            control_count,
            action_shape,
        )
        means = self._predict_means(history, controls, actions)
        variances = np.broadcast_to(variance, means.shape).copy()
        return TrajectoryDistribution(mean=means, variance=variances)

    def encode_histories(
        self,
        histories: NDArray[Any],
        graph: DeclaredTelemetryGraph,
    ) -> NDArray[np.float64]:
        """Return one explicit token per graph entity."""

        torch = _require_torch()
        (
            fitted_graph,
            state_shape,
            _,
            _,
            _,
        ) = self._fitted_values()
        values = np.asarray(histories, dtype=np.float64)
        if (
            graph.to_dict() != fitted_graph.to_dict()
            or values.ndim != 4
            or values.shape[2:] != state_shape
            or not np.all(np.isfinite(values))
        ):
            raise ValueError("history embeddings do not match fitted schema")
        parts: List[NDArray[np.float64]] = []
        self._network.eval()
        with torch.no_grad():
            for start in range(0, len(values), self.config.batch_size):
                batch = torch.as_tensor(
                    values[start : start + self.config.batch_size],
                    dtype=torch.float32,
                    device=self.device,
                )
                encoded = self._network.encode_histories(batch)
                parts.append(
                    np.asarray(
                        encoded.detach().cpu().numpy(),
                        dtype=np.float64,
                    )
                )
        return np.concatenate(parts)

    @property
    def parameter_count(self) -> int:
        """Return inference-time scalar parameters."""

        self._fitted_values()
        return int(
            sum(
                parameter.numel()
                for name, parameter in self._network.named_parameters()
                if not name.startswith("target.")
            )
        )

    @property
    def spectral_radius(self) -> float:
        """Return the fitted latent transition spectral radius."""

        self._fitted_values()
        return self._spectral_radius

    def to_dict(self) -> Dict[str, Any]:
        """Serialize training evidence and CPU model parameters."""

        (
            graph,
            state_shape,
            control_count,
            action_shape,
            variance,
        ) = self._fitted_values()
        state = {
            name: tensor.detach().cpu().tolist()
            for name, tensor in self._network.state_dict().items()
        }
        return {
            "schema_version": 1,
            "kind": self.kind,
            "config": self.config.to_dict(),
            "graph": graph.to_dict(),
            "state_shape": list(state_shape),
            "control_count": control_count,
            "action_shape": list(action_shape),
            "spectral_radius": self.spectral_radius,
            "parameter_count": self.parameter_count,
            "training_metrics": [
                dict(values) for values in self.training_metrics
            ],
            "residual_variance": variance.tolist(),
            "state": state,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "ActionConditionedJepaDynamics":
        """Restore a fitted model artifact."""

        if (
            payload.get("schema_version") != 1
            or payload.get("kind") != cls.kind
        ):
            raise ValueError("unsupported action-conditioned JEPA artifact")
        config_payload = payload.get("config")
        graph_payload = payload.get("graph")
        state_payload = payload.get("state")
        if (
            not isinstance(config_payload, Mapping)
            or not isinstance(graph_payload, Mapping)
            or not isinstance(state_payload, Mapping)
        ):
            raise ValueError("action-conditioned JEPA artifact is malformed")
        torch = _require_torch()
        model = cls(
            ActionConditionedJepaConfig.from_dict(config_payload)
        )
        model.device = _select_device(torch, model.config.device)
        model._graph = DeclaredTelemetryGraph.from_dict(
            dict(graph_payload)
        )
        state_shape = tuple(
            int(value) for value in payload["state_shape"]
        )
        if len(state_shape) != 2:
            raise ValueError("JEPA state shape is malformed")
        model._state_shape = (state_shape[0], state_shape[1])
        model._control_count = int(payload["control_count"])
        action_shape = tuple(
            int(value) for value in payload["action_shape"]
        )
        if len(action_shape) != 2:
            raise ValueError("JEPA action shape is malformed")
        model._action_shape = (action_shape[0], action_shape[1])
        model._residual_variance = np.asarray(
            payload["residual_variance"], dtype=np.float64
        )
        schema = model._schema()
        model._network = _build_network(
            torch, model.config, schema
        )
        expected = model._network.state_dict()
        restored = {
            name: torch.as_tensor(
                state_payload[name], dtype=tensor.dtype
            )
            for name, tensor in expected.items()
        }
        model._network.load_state_dict(restored)
        model._network.to(model.device)
        model._spectral_radius = float(payload["spectral_radius"])
        raw_metrics = payload.get("training_metrics", ())
        if not isinstance(raw_metrics, (list, tuple)):
            raise ValueError("JEPA training metrics are malformed")
        model.training_metrics = tuple(
            {
                str(key): float(value)
                for key, value in metric.items()
            }
            for metric in raw_metrics
            if isinstance(metric, Mapping)
        )
        return model

    def _register_schema(
        self, windows: ActionConditionedWindows
    ) -> None:
        if len(windows.histories) < 2:
            raise ValueError("JEPA fit requires at least two samples")
        self._graph = windows.graph
        self._state_shape = (
            len(windows.entity_names),
            len(windows.state_feature_names),
        )
        self._control_count = len(windows.control_feature_names)
        self._action_shape = (
            len(windows.entity_names),
            len(windows.action_feature_names),
        )

    def _schema(self) -> Dict[str, int]:
        if self._state_shape is None or self._action_shape is None:
            raise ValueError("JEPA schema is not registered")
        return {
            "entity_count": self._state_shape[0],
            "feature_count": self._state_shape[1],
            "control_count": self._control_count,
            "action_count": self._action_shape[1],
        }

    def _fitted_values(
        self,
    ) -> Tuple[
        DeclaredTelemetryGraph,
        Tuple[int, int],
        int,
        Tuple[int, int],
        NDArray[np.float64],
    ]:
        if (
            self._network is None
            or self._graph is None
            or self._state_shape is None
            or self._action_shape is None
            or self._residual_variance is None
        ):
            raise ValueError("action-conditioned JEPA is not fitted")
        return (
            self._graph,
            self._state_shape,
            self._control_count,
            self._action_shape,
            self._residual_variance,
        )

    def _predict_means(
        self,
        histories: NDArray[np.float64],
        controls: NDArray[np.float64],
        actions: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        torch = _require_torch()
        parts: List[NDArray[np.float64]] = []
        self._network.eval()
        with torch.no_grad():
            for start in range(
                0, len(histories), self.config.batch_size
            ):
                end = start + self.config.batch_size
                prediction = self._network.forward_prediction(
                    torch.as_tensor(
                        histories[start:end],
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    torch.as_tensor(
                        controls[start:end],
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    torch.as_tensor(
                        actions[start:end],
                        dtype=torch.float32,
                        device=self.device,
                    ),
                )
                parts.append(
                    np.asarray(
                        prediction.detach().cpu().numpy(),
                        dtype=np.float64,
                    )
                )
        return np.concatenate(parts)

    def _estimate_residual_variance(
        self, windows: ActionConditionedWindows
    ) -> NDArray[np.float64]:
        prediction = self._predict_means(
            np.asarray(windows.histories, dtype=np.float64),
            np.asarray(windows.future_controls, dtype=np.float64),
            np.asarray(windows.future_actions, dtype=np.float64),
        )
        residual = (
            np.asarray(windows.future_states, dtype=np.float64)
            - prediction
        )
        return np.asarray(
            np.maximum(
                np.mean(np.square(residual), axis=(0, 1)),
                self.config.variance_floor,
            ),
            dtype=np.float64,
        )


def _require_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError as error:
        raise ImportError(
            "ActionConditionedJepaDynamics requires the optional "
            "PyTorch training extra"
        ) from error


def _select_device(torch: Any, requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    available = bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )
    if requested == "mps" and not available:
        raise RuntimeError("MPS was requested but is unavailable")
    return "mps" if available else "cpu"


def _seed_torch(torch: Any, seed: int) -> None:
    torch.manual_seed(seed)
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)


def _window_batch(
    torch: Any,
    windows: ActionConditionedWindows,
    selection: NDArray[np.int64],
    device: str,
) -> Mapping[str, Any]:
    return {
        "histories": torch.as_tensor(
            np.asarray(windows.histories[selection]),
            dtype=torch.float32,
            device=device,
        ),
        "future_states": torch.as_tensor(
            np.asarray(windows.future_states[selection]),
            dtype=torch.float32,
            device=device,
        ),
        "future_controls": torch.as_tensor(
            np.asarray(windows.future_controls[selection]),
            dtype=torch.float32,
            device=device,
        ),
        "future_actions": torch.as_tensor(
            np.asarray(windows.future_actions[selection]),
            dtype=torch.float32,
            device=device,
        ),
    }


def _sample_visible_mask(
    *,
    generator: np.random.Generator,
    batch_size: int,
    time_count: int,
    entity_count: int,
    time_fraction: float,
    entity_fraction: float,
) -> NDArray[np.bool_]:
    visible = np.ones(
        (batch_size, time_count, entity_count), dtype=np.bool_
    )
    if time_fraction <= 0.0 or entity_fraction <= 0.0:
        return visible
    time_span = max(1, int(round(time_count * time_fraction)))
    entity_span = max(1, int(round(entity_count * entity_fraction)))
    time_span = min(time_span, time_count - 1)
    entity_span = min(entity_span, entity_count)
    for sample in range(batch_size):
        time_start = int(
            generator.integers(0, time_count - time_span + 1)
        )
        entities = generator.choice(
            entity_count, size=entity_span, replace=False
        )
        visible[
            sample,
            time_start : time_start + time_span,
            entities,
        ] = False
    return visible


def _loss_components(
    torch: Any,
    output: Mapping[str, Any],
    batch: Mapping[str, Any],
    config: ActionConditionedJepaConfig,
) -> Mapping[str, Any]:
    latent = torch.nn.functional.smooth_l1_loss(
        output["predicted_latents"].contiguous(),
        output["target_latents"].contiguous(),
    )
    reconstruction = torch.mean(
        torch.square(
            output["decoded_future"] - batch["future_states"]
        )
    )
    context_reconstruction = torch.mean(
        torch.square(
            output["decoded_context"] - batch["histories"][:, -1]
        )
    )
    tokens = output["context_tokens"].reshape(
        -1, output["context_tokens"].shape[-1]
    )
    centered = tokens - tokens.mean(dim=0, keepdim=True)
    standard_deviation = torch.sqrt(
        torch.mean(torch.square(centered), dim=0) + 1e-4
    )
    variance = torch.mean(torch.relu(1.0 - standard_deviation))
    covariance_matrix = (
        centered.T @ centered / float(max(len(tokens) - 1, 1))
    )
    covariance = torch.mean(
        torch.square(
            covariance_matrix
            - torch.diag(torch.diag(covariance_matrix))
        )
    )
    latent_weight = (
        config.latent_prediction_weight
        if config.objective == "jepa"
        else 0.0
    )
    reconstruction_weight = (
        config.reconstruction_weight
        if config.objective == "jepa"
        else 1.0
    )
    total = (
        latent_weight * latent
        + reconstruction_weight * reconstruction
        + config.context_reconstruction_weight
        * context_reconstruction
        + config.variance_weight * variance
        + config.covariance_weight * covariance
    )
    return {
        "total": total,
        "latent": latent,
        "reconstruction": reconstruction,
        "context_reconstruction": context_reconstruction,
        "variance": variance,
        "covariance": covariance,
    }


def _build_network(
    torch: Any,
    config: ActionConditionedJepaConfig,
    schema: Mapping[str, int],
) -> Any:
    nn = torch.nn
    node_latent = config.node_latent_dimension
    total_latent = schema["entity_count"] * node_latent
    action_width = schema["entity_count"] * schema["action_count"]
    exogenous_width = schema["control_count"] + action_width

    class NodeSequenceEncoder(nn.Module):  # type: ignore[misc,name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.input_projection = nn.Linear(
                schema["feature_count"], node_latent
            )
            self.entity_embedding = nn.Embedding(
                schema["entity_count"], node_latent
            )
            self.mask_embedding = nn.Parameter(
                torch.zeros(node_latent)
            )
            self.temporal = nn.GRU(
                node_latent, node_latent, batch_first=True
            )
            self.output_norm = nn.LayerNorm(node_latent)

        def forward(self, values: Any, visible: Any) -> Any:
            batch, time, entities, _ = values.shape
            entity_ids = torch.arange(
                entities, dtype=torch.long, device=values.device
            )
            projected = (
                self.input_projection(values)
                + self.entity_embedding(entity_ids)[None, None]
            )
            projected = torch.where(
                visible[..., None],
                projected,
                self.mask_embedding[None, None, None, :]
                + self.entity_embedding(entity_ids)[None, None],
            )
            sequence = projected.permute(0, 2, 1, 3).reshape(
                batch * entities, time, node_latent
            )
            encoded, _ = self.temporal(sequence)
            return self.output_norm(
                encoded.reshape(
                    batch, entities, time, node_latent
                ).permute(0, 2, 1, 3)
            )

    class Network(nn.Module):  # type: ignore[misc,name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.online = NodeSequenceEncoder()
            self.target = copy.deepcopy(self.online)
            for parameter in self.target.parameters():
                parameter.requires_grad_(False)
            transition_scale = 1.0 / np.sqrt(float(total_latent))
            self.transition_left = nn.Parameter(
                torch.randn(
                    total_latent, config.transition_rank
                )
                * transition_scale
            )
            self.transition_right = nn.Parameter(
                torch.randn(
                    config.transition_rank, total_latent
                )
                * transition_scale
            )
            self.exogenous = nn.Linear(
                exogenous_width, total_latent
            )
            self.decoder = nn.Linear(
                node_latent, schema["feature_count"]
            )

        def trainable_parameters(self) -> Any:
            return (
                parameter
                for parameter in self.parameters()
                if parameter.requires_grad
            )

        def encode_histories(self, histories: Any) -> Any:
            visible = torch.ones(
                histories.shape[:-1],
                dtype=torch.bool,
                device=histories.device,
            )
            return self.online(histories, visible)[:, -1]

        def _roll(
            self,
            current_tokens: Any,
            controls: Any,
            actions: Any,
        ) -> Any:
            batch = current_tokens.shape[0]
            current = current_tokens.reshape(batch, total_latent)
            transition = (
                self.transition_left @ self.transition_right
            )
            predictions = []
            for step in range(controls.shape[1]):
                condition = torch.cat(
                    (
                        controls[:, step],
                        actions[:, step].reshape(batch, action_width),
                    ),
                    dim=1,
                )
                current = (
                    current @ transition + self.exogenous(condition)
                )
                predictions.append(
                    current.reshape(
                        batch, schema["entity_count"], node_latent
                    )
                )
            return torch.stack(predictions, dim=1)

        def forward_training(
            self, batch: Mapping[str, Any], visible: Any
        ) -> Mapping[str, Any]:
            online = self.online(batch["histories"], visible)
            context = online[:, -1]
            predicted = self._roll(
                context,
                batch["future_controls"],
                batch["future_actions"],
            )
            with torch.no_grad():
                target_values = torch.cat(
                    (
                        batch["histories"][:, -1:],
                        batch["future_states"],
                    ),
                    dim=1,
                )
                target_visible = torch.ones(
                    target_values.shape[:-1],
                    dtype=torch.bool,
                    device=target_values.device,
                )
                target = self.target(
                    target_values, target_visible
                )[:, 1:]
            return {
                "predicted_latents": predicted,
                "target_latents": target,
                "decoded_future": self.decoder(predicted),
                "decoded_context": self.decoder(context),
                "context_tokens": context,
            }

        def forward_prediction(
            self, histories: Any, controls: Any, actions: Any
        ) -> Any:
            tokens = self.encode_histories(histories)
            return self.decoder(
                self._roll(tokens, controls, actions)
            )

        def update_target(self, decay: float) -> None:
            with torch.no_grad():
                for online, target in zip(
                    self.online.parameters(),
                    self.target.parameters(),
                ):
                    target.mul_(decay).add_(
                        online, alpha=1.0 - decay
                    )

        def spectral_radius(self) -> float:
            transition = (
                self.transition_left @ self.transition_right
            )
            values = np.asarray(
                transition.detach().cpu().numpy(),
                dtype=np.float64,
            )
            return float(
                np.max(np.abs(np.linalg.eigvals(values)))
            )

        def project_transition(self, maximum: float) -> None:
            radius = self.spectral_radius()
            if radius > maximum:
                with torch.no_grad():
                    self.transition_left.mul_(maximum / radius)

    return Network()
