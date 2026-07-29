"""Edge Causal-JEPA whole-entity observability intervention."""

import copy
import importlib
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ..action_conditioned_dynamics import ActionConditionedWindows
from ..graph_telemetry import DeclaredTelemetryGraph
from .complete_lejepa import PairBlockedAnchorSchedule


CAUSAL_JEPA_OBJECTIVES = (
    "causal_entity_mask",
    "coordinate_time_mask",
    "prediction_only",
)


@dataclass(frozen=True)
class CausalJepaConfig:
    """Frozen controls for one parameter-matched predictor."""

    objective: str = "causal_entity_mask"
    width: int = 32
    transformer_depth: int = 2
    attention_heads: int = 4
    mlp_width: int = 128
    history_size: int = 6
    future_size: int = 10
    masked_entity_count: int = 2
    pretrain_steps: int = 1200
    checkpoint_interval: int = 200
    learning_rate: float = 5e-4
    expected_pair_count: int = 40
    seed: int = 18018
    device: str = "cpu"

    def __post_init__(self) -> None:
        integers = (
            self.width,
            self.transformer_depth,
            self.attention_heads,
            self.mlp_width,
            self.history_size,
            self.future_size,
            self.masked_entity_count,
            self.pretrain_steps,
            self.checkpoint_interval,
            self.expected_pair_count,
        )
        if (
            self.objective not in CAUSAL_JEPA_OBJECTIVES
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
                for value in integers
            )
            or self.width % self.attention_heads
            or self.history_size != 6
            or self.future_size != 10
            or self.masked_entity_count != 2
            or self.learning_rate <= 0.0
            or isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.device != "cpu"
        ):
            raise ValueError("Causal-JEPA configuration is invalid")


@dataclass(frozen=True)
class MaskedHistoryCompletion:
    """One-entity-at-a-time history completion evidence."""

    predictions: NDArray[np.float64]
    targets: NDArray[np.float64]
    ownership_mask: NDArray[np.bool_]
    entity_ids: Tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            self.predictions.shape != self.targets.shape
            or self.predictions.ndim != 4
            or self.predictions.shape[1] != len(self.entity_ids)
            or self.ownership_mask.shape
            != (len(self.entity_ids), self.predictions.shape[-1])
            or not np.all(np.isfinite(self.predictions))
            or not np.all(np.isfinite(self.targets))
        ):
            raise ValueError("masked history completion is invalid")


def causal_mask_plan(
    objective: str,
    *,
    step: int,
    entity_count: int,
    history_size: int = 6,
    masked_entity_count: int = 2,
    seed: int = 18018,
) -> NDArray[np.bool_]:
    """Return a deterministic matched-budget history mask."""

    if (
        objective not in CAUSAL_JEPA_OBJECTIVES
        or isinstance(step, bool)
        or not isinstance(step, int)
        or step < 0
        or entity_count <= masked_entity_count
        or history_size < 2
        or masked_entity_count < 1
    ):
        raise ValueError("Causal-JEPA mask request is invalid")
    mask = np.zeros((history_size, entity_count), dtype=np.bool_)
    if objective == "prediction_only":
        return mask
    generator = np.random.default_rng(
        np.random.SeedSequence((seed, step))
    )
    budget = masked_entity_count * (history_size - 1)
    if objective == "causal_entity_mask":
        entities = generator.choice(
            entity_count, size=masked_entity_count, replace=False
        )
        mask[1:, entities] = True
    else:
        positions = generator.choice(
            (history_size - 1) * entity_count,
            size=budget,
            replace=False,
        )
        local_time, entity = np.divmod(positions, entity_count)
        mask[local_time + 1, entity] = True
    if np.any(mask[0]) or int(np.sum(mask)) != budget:
        raise RuntimeError("Causal-JEPA mask budget differs")
    return mask


class CausalJepaModel:
    """Restorable frozen-slot Causal-JEPA predictor."""

    kind = "causal_jepa_entity_intervention"
    schema_version = 1

    def __init__(
        self, config: CausalJepaConfig = CausalJepaConfig()
    ) -> None:
        self.config = config
        self._graph: Optional[DeclaredTelemetryGraph] = None
        self._feature_names: Tuple[str, ...] = ()
        self._control_names: Tuple[str, ...] = ()
        self._action_names: Tuple[str, ...] = ()
        self._ownership_mask: Optional[NDArray[np.bool_]] = None
        self._center: Optional[NDArray[np.float64]] = None
        self._scale: Optional[NDArray[np.float64]] = None
        self._projection: Optional[NDArray[np.float64]] = None
        self._condition_dimension: Optional[int] = None
        self._network: Any = None
        self._checkpoints: Tuple[Tuple[int, Mapping[str, Any]], ...] = ()
        self._training_metrics: Tuple[Mapping[str, float], ...] = ()
        self._selection_metrics: Tuple[Mapping[str, float], ...] = ()
        self._selected_step: Optional[int] = None

    @property
    def training_parameter_count(self) -> int:
        *_, network = self._fitted_values()
        return int(sum(value.numel() for value in network.parameters()))

    @property
    def selected_step(self) -> Optional[int]:
        return self._selected_step

    @property
    def training_metrics(self) -> Tuple[Mapping[str, float], ...]:
        return tuple(dict(row) for row in self._training_metrics)

    @property
    def selection_metrics(self) -> Tuple[Mapping[str, float], ...]:
        return tuple(dict(row) for row in self._selection_metrics)

    def fit(self, windows: ActionConditionedWindows) -> "CausalJepaModel":
        """Fit checkpoint candidates over pair-blocked anchors."""

        if len(set(windows.matched_pair_ids)) != self.config.expected_pair_count:
            raise ValueError("Causal-JEPA fitting pair count differs")
        _validate_windows(windows, self.config)
        torch = _require_torch()
        _seed_torch(torch, self.config.seed)
        ownership = _fit_owned_feature_mask(windows)
        full = np.concatenate((windows.histories, windows.future_states), axis=1)
        center, scale = _fit_normalizer(full, ownership)
        projection = _frozen_projection(
            len(windows.state_feature_names),
            self.config.width,
            seed=self.config.seed + 7,
        )
        condition_dimension = (
            len(windows.control_feature_names)
            + len(windows.entity_names) * len(windows.action_feature_names)
        )
        network = _build_network(
            torch,
            config=self.config,
            entity_count=len(windows.entity_names),
            condition_dimension=condition_dimension,
        )
        optimizer = torch.optim.Adam(
            network.parameters(), lr=self.config.learning_rate
        )
        schedule = PairBlockedAnchorSchedule(
            windows, seed=self.config.seed + 1
        )
        metrics: List[Mapping[str, float]] = []
        checkpoints = []
        network.train()
        for step in range(self.config.pretrain_steps):
            anchor = schedule.batch(step)
            history, future, conditions = _training_arrays(
                windows,
                anchor.indices,
                ownership,
                center,
                scale,
                self.config,
            )
            mask = causal_mask_plan(
                self.config.objective,
                step=step,
                entity_count=len(windows.entity_names),
                history_size=self.config.history_size,
                masked_entity_count=self.config.masked_entity_count,
                seed=self.config.seed + 2,
            )
            optimizer.zero_grad(set_to_none=True)
            losses = _objective_loss(
                torch,
                network,
                history,
                future,
                conditions,
                projection,
                mask,
            )
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()
            completed = step + 1
            if (
                completed % self.config.checkpoint_interval == 0
                or completed == self.config.pretrain_steps
            ):
                row = {
                    "step": float(completed),
                    "total": float(losses["total"].detach()),
                    "history": float(losses["history"].detach()),
                    "future": float(losses["future"].detach()),
                }
                if not np.all(np.isfinite(list(row.values()))):
                    raise RuntimeError("Causal-JEPA training became non-finite")
                metrics.append(row)
                checkpoints.append(
                    (
                        completed,
                        {
                            name: value.detach().cpu().clone()
                            for name, value in network.state_dict().items()
                        },
                    )
                )
        self._graph = windows.graph
        self._feature_names = windows.state_feature_names
        self._control_names = windows.control_feature_names
        self._action_names = windows.action_feature_names
        self._ownership_mask = ownership
        self._center = center
        self._scale = scale
        self._projection = projection
        self._condition_dimension = condition_dimension
        self._network = network
        self._checkpoints = tuple(checkpoints)
        self._training_metrics = tuple(metrics)
        return self

    def select(
        self, windows: ActionConditionedWindows
    ) -> "CausalJepaModel":
        """Choose one checkpoint with selection-role latent loss only."""

        (
            graph,
            features,
            controls,
            actions,
            ownership,
            center,
            scale,
            projection,
            _,
            network,
        ) = self._fitted_values()
        if (
            windows.graph.to_dict() != graph.to_dict()
            or windows.state_feature_names != features
            or windows.control_feature_names != controls
            or windows.action_feature_names != actions
            or not self._checkpoints
        ):
            raise ValueError("Causal-JEPA selection schema differs")
        _validate_windows(windows, self.config)
        torch = _require_torch()
        schedule = PairBlockedAnchorSchedule(
            windows, seed=self.config.seed + 3
        )
        evaluation_steps = min(10, len(schedule.transitions))
        rows = []
        best_key: Optional[Tuple[float, int]] = None
        best_state = None
        best_step = None
        for checkpoint_step, state in self._checkpoints:
            network.load_state_dict(state)
            network.eval()
            accumulated = {"total": 0.0, "history": 0.0, "future": 0.0}
            with torch.no_grad():
                for local_step in range(evaluation_steps):
                    anchor = schedule.batch(local_step)
                    history, future, conditions = _training_arrays(
                        windows,
                        anchor.indices,
                        ownership,
                        center,
                        scale,
                        self.config,
                    )
                    mask = causal_mask_plan(
                        self.config.objective,
                        step=local_step,
                        entity_count=len(windows.entity_names),
                        history_size=self.config.history_size,
                        masked_entity_count=self.config.masked_entity_count,
                        seed=self.config.seed + 4,
                    )
                    losses = _objective_loss(
                        torch,
                        network,
                        history,
                        future,
                        conditions,
                        projection,
                        mask,
                    )
                    for name in accumulated:
                        accumulated[name] += float(losses[name])
            row = {
                "step": float(checkpoint_step),
                **{
                    name: value / evaluation_steps
                    for name, value in accumulated.items()
                },
            }
            if not np.all(np.isfinite(list(row.values()))):
                raise RuntimeError(
                    "Causal-JEPA selection became non-finite"
                )
            rows.append(row)
            key = (row["total"], checkpoint_step)
            if best_key is None or key < best_key:
                best_key = key
                best_state = copy.deepcopy(state)
                best_step = checkpoint_step
        if best_state is None or best_step is None:
            raise RuntimeError("Causal-JEPA selected no checkpoint")
        network.load_state_dict(best_state)
        network.eval()
        for parameter in network.parameters():
            parameter.requires_grad_(False)
        self._selected_step = best_step
        self._selection_metrics = tuple(rows)
        self._checkpoints = ()
        return self

    def predict(
        self,
        histories: NDArray[np.float64],
        future_controls: NDArray[np.float64],
        future_actions: NDArray[np.float64],
        graph: DeclaredTelemetryGraph,
    ) -> NDArray[np.float64]:
        """Predict ten observable future states from causal public inputs."""

        values, controls, actions = self._validate_public_inputs(
            histories, future_controls, future_actions, graph
        )
        (
            _,
            _,
            _,
            _,
            ownership,
            center,
            scale,
            projection,
            _,
            network,
        ) = self._selected_values()
        normalized = _normalize_states(values, ownership, center, scale)
        history = normalized[:, -self.config.history_size :]
        conditions = _conditions(controls, actions)
        mask = np.zeros(
            (self.config.history_size, len(graph.entities)),
            dtype=np.bool_,
        )
        predictions = []
        torch = _require_torch()
        with torch.no_grad():
            for start in range(0, len(history), 128):
                latent = network(
                    torch.as_tensor(
                        history[start : start + 128] @ projection,
                        dtype=torch.float32,
                    ),
                    torch.as_tensor(
                        conditions[start : start + 128],
                        dtype=torch.float32,
                    ),
                    torch.as_tensor(mask),
                )
                decoded = (
                    latent[:, self.config.history_size :].cpu().numpy()
                    @ projection.T
                )
                predictions.append(decoded)
        normalized_prediction = np.concatenate(predictions)
        raw = normalized_prediction * scale[None, None] + center[None, None]
        return np.asarray(
            np.where(ownership[None, None], raw, center[None, None]),
            dtype=np.float64,
        )

    def complete_masked_histories(
        self, windows: ActionConditionedWindows
    ) -> MaskedHistoryCompletion:
        """Reconstruct each entity trajectory from all other entities."""

        (
            graph,
            _,
            _,
            _,
            ownership,
            center,
            scale,
            projection,
            _,
            network,
        ) = self._selected_values()
        self._validate_public_inputs(
            windows.histories,
            windows.future_controls,
            windows.future_actions,
            windows.graph,
        )
        normalized = _normalize_states(
            windows.histories, ownership, center, scale
        )[:, -self.config.history_size :]
        conditions = _conditions(
            windows.future_controls, windows.future_actions
        )
        torch = _require_torch()
        entity_predictions = []
        with torch.no_grad():
            for entity in range(len(graph.entities)):
                mask = np.zeros(
                    (self.config.history_size, len(graph.entities)),
                    dtype=np.bool_,
                )
                mask[1:, entity] = True
                parts = []
                for start in range(0, len(normalized), 128):
                    latent = network(
                        torch.as_tensor(
                            normalized[start : start + 128] @ projection,
                            dtype=torch.float32,
                        ),
                        torch.as_tensor(
                            conditions[start : start + 128],
                            dtype=torch.float32,
                        ),
                        torch.as_tensor(mask),
                    )
                    decoded = (
                        latent[:, 1 : self.config.history_size, entity]
                        .cpu()
                        .numpy()
                        @ projection.T
                    )
                    parts.append(decoded)
                local = np.concatenate(parts)
                raw = local * scale[entity] + center[entity]
                entity_predictions.append(raw)
        predictions = np.stack(entity_predictions, axis=1)
        targets = np.stack(
            [
                windows.histories[
                    :, -self.config.history_size + 1 :, entity
                ]
                for entity in range(len(graph.entities))
            ],
            axis=1,
        )
        return MaskedHistoryCompletion(
            predictions=np.asarray(predictions, dtype=np.float64),
            targets=np.asarray(targets, dtype=np.float64),
            ownership_mask=ownership.copy(),
            entity_ids=graph.entity_ids,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe selected model."""

        (
            graph,
            features,
            controls,
            actions,
            ownership,
            center,
            scale,
            projection,
            condition_dimension,
            network,
        ) = self._selected_values()
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "config": asdict(self.config),
            "graph": graph.to_dict(),
            "feature_names": list(features),
            "control_names": list(controls),
            "action_names": list(actions),
            "ownership_mask": ownership.astype(int).tolist(),
            "center": center.tolist(),
            "scale": scale.tolist(),
            "projection": projection.tolist(),
            "condition_dimension": condition_dimension,
            "state_dict": _state_dict_to_payload(network.state_dict()),
            "selected_step": self._selected_step,
            "training_metrics": [dict(row) for row in self._training_metrics],
            "selection_metrics": [dict(row) for row in self._selection_metrics],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CausalJepaModel":
        """Restore and validate a selected model."""

        if (
            payload.get("schema_version") != cls.schema_version
            or payload.get("kind") != cls.kind
        ):
            raise ValueError("Causal-JEPA schema is invalid")
        config = CausalJepaConfig(**dict(payload["config"]))
        graph = DeclaredTelemetryGraph.from_dict(dict(payload["graph"]))
        features = tuple(str(value) for value in payload["feature_names"])
        controls = tuple(str(value) for value in payload["control_names"])
        actions = tuple(str(value) for value in payload["action_names"])
        ownership = np.asarray(payload["ownership_mask"], dtype=np.bool_)
        center = np.asarray(payload["center"], dtype=np.float64)
        scale = np.asarray(payload["scale"], dtype=np.float64)
        projection = np.asarray(payload["projection"], dtype=np.float64)
        condition_dimension = payload["condition_dimension"]
        expected = (len(graph.entities), len(features))
        if (
            ownership.shape != expected
            or center.shape != expected
            or scale.shape != expected
            or projection.shape != (len(features), config.width)
            or not np.all(np.isfinite(center))
            or not np.all(np.isfinite(scale))
            or not np.all(np.isfinite(projection))
            or np.any(scale <= 0.0)
            or not np.allclose(
                projection @ projection.T,
                np.eye(len(features)),
                atol=1e-6,
            )
            or isinstance(condition_dimension, bool)
            or not isinstance(condition_dimension, int)
            or condition_dimension
            != len(controls) + len(graph.entities) * len(actions)
        ):
            raise ValueError("Causal-JEPA fitted schema is invalid")
        torch = _require_torch()
        network = _build_network(
            torch,
            config=config,
            entity_count=len(graph.entities),
            condition_dimension=condition_dimension,
        )
        network.load_state_dict(
            _state_dict_from_payload(torch, dict(payload["state_dict"])),
            strict=True,
        )
        selected_step = payload.get("selected_step")
        if (
            isinstance(selected_step, bool)
            or not isinstance(selected_step, int)
            or selected_step < 1
        ):
            raise ValueError("Causal-JEPA selected step is invalid")
        for parameter in network.parameters():
            parameter.requires_grad_(False)
        result = cls(config)
        result._graph = graph
        result._feature_names = features
        result._control_names = controls
        result._action_names = actions
        result._ownership_mask = ownership
        result._center = center
        result._scale = scale
        result._projection = projection
        result._condition_dimension = condition_dimension
        result._network = network.eval()
        result._selected_step = selected_step
        result._training_metrics = _metric_rows(
            payload.get("training_metrics", ())
        )
        result._selection_metrics = _metric_rows(
            payload.get("selection_metrics", ())
        )
        return result

    def _validate_public_inputs(
        self,
        histories: NDArray[np.float64],
        future_controls: NDArray[np.float64],
        future_actions: NDArray[np.float64],
        graph: DeclaredTelemetryGraph,
    ) -> Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        graph_, features, controls, actions, *_ = self._selected_values()
        values = np.asarray(histories, dtype=np.float64)
        control_values = np.asarray(future_controls, dtype=np.float64)
        action_values = np.asarray(future_actions, dtype=np.float64)
        if (
            graph.to_dict() != graph_.to_dict()
            or values.ndim != 4
            or values.shape[1:]
            != (20, len(graph.entities), len(features))
            or control_values.shape
            != (len(values), self.config.future_size, len(controls))
            or action_values.shape
            != (
                len(values),
                self.config.future_size,
                len(graph.entities),
                len(actions),
            )
            or not all(
                np.all(np.isfinite(item))
                for item in (values, control_values, action_values)
            )
        ):
            raise ValueError("Causal-JEPA public inputs are invalid")
        return values, control_values, action_values

    def _fitted_values(self) -> Tuple[Any, ...]:
        if (
            self._graph is None
            or not self._feature_names
            or not self._control_names
            or not self._action_names
            or self._ownership_mask is None
            or self._center is None
            or self._scale is None
            or self._projection is None
            or self._condition_dimension is None
            or self._network is None
        ):
            raise ValueError("Causal-JEPA model is not fitted")
        return (
            self._graph,
            self._feature_names,
            self._control_names,
            self._action_names,
            self._ownership_mask,
            self._center,
            self._scale,
            self._projection,
            self._condition_dimension,
            self._network,
        )

    def _selected_values(self) -> Tuple[Any, ...]:
        values = self._fitted_values()
        if self._selected_step is None:
            raise ValueError("Causal-JEPA model is not selected")
        return values


def _build_network(
    torch: Any,
    *,
    config: CausalJepaConfig,
    entity_count: int,
    condition_dimension: int,
) -> Any:
    nn = torch.nn

    class Network(nn.Module):  # type: ignore[misc,name-defined]
        def __init__(self) -> None:
            super().__init__()
            total_time = config.history_size + config.future_size
            self.mask_token = nn.Parameter(torch.zeros(config.width))
            nn.init.trunc_normal_(self.mask_token, std=0.02)
            self.time_embedding = nn.Parameter(
                torch.randn(total_time, config.width) * 0.02
            )
            self.entity_embedding = nn.Parameter(
                torch.randn(entity_count, config.width) * 0.02
            )
            self.condition_embedding = nn.Parameter(
                torch.randn(config.width) * 0.02
            )
            self.identity_projector = nn.Linear(
                config.width, config.width
            )
            self.condition_projector = nn.Linear(
                condition_dimension, config.width
            )
            layer = nn.TransformerEncoderLayer(
                d_model=config.width,
                nhead=config.attention_heads,
                dim_feedforward=config.mlp_width,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                layer, num_layers=config.transformer_depth
            )
            self.output = nn.Linear(config.width, config.width)

        def forward(
            self, history: Any, conditions: Any, mask: Any
        ) -> Any:
            batch = len(history)
            total_time = config.history_size + config.future_size
            anchor = self.identity_projector(
                history[:, 0] + self.entity_embedding
            )
            query = (
                self.mask_token.reshape(1, 1, 1, -1)
                + self.time_embedding.reshape(1, total_time, 1, -1)
                + anchor.unsqueeze(1)
            ).expand(batch, total_time, entity_count, config.width)
            state_input = query.clone()
            visible_history = (
                history
                + self.entity_embedding.reshape(1, 1, entity_count, -1)
                + self.time_embedding[
                    : config.history_size
                ].reshape(1, config.history_size, 1, -1)
            )
            state_input[:, : config.history_size] = torch.where(
                mask.reshape(
                    1, config.history_size, entity_count, 1
                ),
                query[:, : config.history_size],
                visible_history,
            )
            state_tokens = state_input.flatten(1, 2)
            condition_tokens = (
                self.condition_projector(conditions)
                + self.condition_embedding
                + self.time_embedding[
                    config.history_size :
                ].reshape(1, config.future_size, -1)
            )
            output = self.transformer(
                torch.cat((state_tokens, condition_tokens), dim=1)
            )
            state_output = self.output(
                output[:, : total_time * entity_count]
            )
            return state_output.reshape(
                batch, total_time, entity_count, config.width
            )

    return Network()


def _objective_loss(
    torch: Any,
    network: Any,
    history: NDArray[np.float64],
    future: NDArray[np.float64],
    conditions: NDArray[np.float64],
    projection: NDArray[np.float64],
    mask: NDArray[np.bool_],
) -> Mapping[str, Any]:
    history_latent = torch.as_tensor(
        history @ projection, dtype=torch.float32
    )
    future_latent = torch.as_tensor(
        future @ projection, dtype=torch.float32
    )
    prediction = network(
        history_latent,
        torch.as_tensor(conditions, dtype=torch.float32),
        torch.as_tensor(mask),
    )
    history_prediction = prediction[:, : history.shape[1]]
    future_prediction = prediction[:, history.shape[1] :]
    future_loss = torch.nn.functional.mse_loss(
        future_prediction, future_latent
    )
    if np.any(mask):
        history_loss = torch.nn.functional.mse_loss(
            history_prediction[:, mask], history_latent[:, mask]
        )
    else:
        history_loss = future_loss * 0.0
    return {
        "total": history_loss + future_loss,
        "history": history_loss,
        "future": future_loss,
    }


def _training_arrays(
    windows: ActionConditionedWindows,
    indices: NDArray[np.int64],
    ownership: NDArray[np.bool_],
    center: NDArray[np.float64],
    scale: NDArray[np.float64],
    config: CausalJepaConfig,
) -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    history = _normalize_states(
        windows.histories[indices], ownership, center, scale
    )[:, -config.history_size :]
    future = _normalize_states(
        windows.future_states[indices], ownership, center, scale
    )
    return (
        history,
        future,
        _conditions(
            windows.future_controls[indices],
            windows.future_actions[indices],
        ),
    )


def _conditions(
    controls: NDArray[np.float64],
    actions: NDArray[np.float64],
) -> NDArray[np.float64]:
    return np.concatenate(
        (controls, actions.reshape(len(actions), actions.shape[1], -1)),
        axis=2,
    )


def _validate_windows(
    windows: ActionConditionedWindows, config: CausalJepaConfig
) -> None:
    if (
        windows.histories.shape[1] != 20
        or windows.future_states.shape[1] != config.future_size
        or len(windows.entity_names) <= config.masked_entity_count
    ):
        raise ValueError("Causal-JEPA windows are incompatible")


def _frozen_projection(
    feature_count: int, width: int, *, seed: int
) -> NDArray[np.float64]:
    if width < feature_count:
        raise ValueError("Causal-JEPA width is below state dimension")
    generator = np.random.default_rng(seed)
    q, _ = np.linalg.qr(
        generator.normal(size=(width, feature_count)), mode="reduced"
    )
    projection = q.T
    return np.asarray(projection, dtype=np.float64)


def _fit_normalizer(
    full: NDArray[np.float64],
    ownership: NDArray[np.bool_],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    center = np.mean(full, axis=(0, 1))
    scale = np.std(full, axis=(0, 1))
    center = np.where(ownership, center, 0.0)
    scale = np.where(ownership, np.maximum(scale, 1e-6), 1.0)
    return np.asarray(center), np.asarray(scale)


def _normalize_states(
    values: NDArray[np.float64],
    ownership: NDArray[np.bool_],
    center: NDArray[np.float64],
    scale: NDArray[np.float64],
) -> NDArray[np.float64]:
    return np.asarray(
        ((values - center) / scale) * ownership[None, None],
        dtype=np.float64,
    )


def _fit_owned_feature_mask(
    windows: ActionConditionedWindows,
) -> NDArray[np.bool_]:
    entity_positions = {
        entity: index for index, entity in enumerate(windows.entity_names)
    }
    feature_positions = {
        feature: index
        for index, feature in enumerate(windows.state_feature_names)
    }
    mask = np.zeros(
        (len(windows.entity_names), len(windows.state_feature_names)),
        dtype=np.bool_,
    )
    for key, entity in windows.graph.binding_map().items():
        feature = key.split(".", 1)[-1]
        if entity in entity_positions and feature in feature_positions:
            mask[entity_positions[entity], feature_positions[feature]] = True
    mask |= np.ptp(windows.histories, axis=(0, 1)) > 1e-9
    if not np.any(mask):
        raise ValueError("Causal-JEPA schema has no observations")
    return mask


def _state_dict_to_payload(
    state_dict: Mapping[str, Any],
) -> Mapping[str, Any]:
    return {
        name: {
            "shape": list(value.shape),
            "values": value.detach().cpu().numpy().tolist(),
        }
        for name, value in state_dict.items()
    }


def _state_dict_from_payload(
    torch: Any, payload: Mapping[str, Any]
) -> Mapping[str, Any]:
    result = {}
    for name, raw in payload.items():
        value = dict(raw)
        array = np.asarray(value["values"])
        shape = tuple(int(item) for item in value["shape"])
        if array.shape != shape:
            raise ValueError("Causal-JEPA state tensor shape differs")
        if array.dtype.kind not in ("i", "u", "b") and not np.all(
            np.isfinite(array)
        ):
            raise ValueError("Causal-JEPA state tensor is non-finite")
        result[str(name)] = (
            torch.as_tensor(array)
            if array.dtype.kind in ("i", "u", "b")
            else torch.as_tensor(array, dtype=torch.float32)
        )
    return result


def _metric_rows(values: Any) -> Tuple[Mapping[str, float], ...]:
    rows = []
    for raw in values:
        row = {str(key): float(value) for key, value in dict(raw).items()}
        if not np.all(np.isfinite(list(row.values()))):
            raise ValueError("Causal-JEPA metric row is non-finite")
        rows.append(row)
    return tuple(rows)


def _seed_torch(torch: Any, seed: int) -> None:
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def _require_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError as error:
        raise RuntimeError(
            "Causal-JEPA fitting requires optional training dependencies"
        ) from error
