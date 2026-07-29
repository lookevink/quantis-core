"""Matched-twin latent intervention-effect prediction for telemetry.

The public model predicts only an observable treatment-minus-control
correction. It never consumes the matched control trajectory at inference.
"""

import copy
import importlib
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from ..action_conditioned_dynamics import (
    ActionConditionedWindows,
    TrajectoryDistribution,
)
from ..graph_telemetry import DeclaredTelemetryGraph
from .complete_lejepa import fit_owned_feature_mask
from .models import ContractiveLowRankDynamics


PAIR_EFFECT_OBJECTIVES = (
    "pair_effect_jepa",
    "supervised_pair_effect",
    "deranged_pair_jepa",
)


@dataclass(frozen=True)
class PairEffectJepaConfig:
    """Frozen controls for one capacity-matched paired-effect cell."""

    objective: str = "pair_effect_jepa"
    width: int = 16
    hidden_width: int = 64
    pretrain_steps: int = 800
    checkpoint_interval: int = 100
    learning_rate: float = 5e-4
    weight_decay: float = 1e-3
    ema_decay: float = 0.996
    latent_weight: float = 0.2
    expected_pair_count: int = 40
    seed: int = 23021
    device: str = "cpu"

    def __post_init__(self) -> None:
        integers = (
            self.width,
            self.hidden_width,
            self.pretrain_steps,
            self.checkpoint_interval,
            self.expected_pair_count,
        )
        if (
            self.objective not in PAIR_EFFECT_OBJECTIVES
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
                for value in integers
            )
            or self.learning_rate <= 0.0
            or self.weight_decay < 0.0
            or not 0.0 <= self.ema_decay < 1.0
            or self.latent_weight < 0.0
            or isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.device != "cpu"
        ):
            raise ValueError("PairEffect-JEPA configuration is invalid")

    @property
    def effective_latent_weight(self) -> float:
        """Return the cell-specific latent-effect weight."""

        if self.objective == "supervised_pair_effect":
            return 0.0
        return self.latent_weight


@dataclass(frozen=True)
class _MatchedRows:
    treatment: NDArray[np.int64]
    control: NDArray[np.int64]
    pair_ids: Tuple[str, ...]
    rows_by_pair: Tuple[NDArray[np.int64], ...]
    deranged_control: NDArray[np.int64]


class PairEffectJepaModel:
    """Restorable matched-effect predictor with a causal public seam."""

    kind = "pair_effect_jepa"
    schema_version = 1

    def __init__(
        self, config: PairEffectJepaConfig = PairEffectJepaConfig()
    ) -> None:
        self.config = config
        self._graph: Optional[DeclaredTelemetryGraph] = None
        self._feature_names: Tuple[str, ...] = ()
        self._control_names: Tuple[str, ...] = ()
        self._action_names: Tuple[str, ...] = ()
        self._ownership_mask: Optional[NDArray[np.bool_]] = None
        self._network: Any = None
        self._checkpoints: Tuple[Tuple[int, Mapping[str, Any]], ...] = ()
        self._training_metrics: Tuple[Mapping[str, float], ...] = ()
        self._selection_metrics: Tuple[Mapping[str, float], ...] = ()
        self._selected_step: Optional[int] = None

    @property
    def training_parameter_count(self) -> int:
        """Return all learned and EMA parameters retained during fitting."""

        *_, network = self._fitted_values()
        return int(sum(value.numel() for value in network.parameters()))

    @property
    def inference_parameter_count(self) -> int:
        """Return parameters used by public effect inference."""

        *_, network = self._fitted_values()
        modules: Sequence[Any] = (
            network.online_encoder,
            network.predictor,
            network.decoder,
        )
        return int(
            sum(
                value.numel()
                for module in modules
                for value in module.parameters()
            )
            + network.horizon_embedding.numel()
        )

    @property
    def selected_step(self) -> Optional[int]:
        return self._selected_step

    @property
    def training_metrics(self) -> Tuple[Mapping[str, float], ...]:
        return tuple(dict(row) for row in self._training_metrics)

    @property
    def selection_metrics(self) -> Tuple[Mapping[str, float], ...]:
        return tuple(dict(row) for row in self._selection_metrics)

    def fit(
        self, windows: ActionConditionedWindows
    ) -> "PairEffectJepaModel":
        """Fit paired-effect checkpoint candidates on complete pairs."""

        _validate_windows(windows, self.config.expected_pair_count)
        torch = _require_torch()
        _seed_torch(torch, self.config.seed)
        ownership = fit_owned_feature_mask(windows)
        matched = _match_rows(windows)
        condition_dimension = (
            len(windows.control_feature_names)
            + len(windows.entity_names) * len(windows.action_feature_names)
        )
        network = _build_network(
            torch,
            config=self.config,
            entity_count=len(windows.entity_names),
            feature_count=len(windows.state_feature_names),
            condition_dimension=condition_dimension,
            horizon=len(windows.future_states[0]),
        )
        optimized = [
            parameter
            for name, parameter in network.named_parameters()
            if not name.startswith("target_encoder.")
        ]
        optimizer = torch.optim.AdamW(
            optimized,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.pretrain_steps
        )
        metrics: List[Mapping[str, float]] = []
        checkpoints = []
        network.train()
        for step in range(self.config.pretrain_steps):
            chosen = np.asarray(
                [
                    rows[
                        (step + pair_position) % len(rows)
                    ]
                    for pair_position, rows in enumerate(
                        matched.rows_by_pair
                    )
                ],
                dtype=np.int64,
            )
            treatment_indices = matched.treatment[chosen]
            if self.config.objective == "deranged_pair_jepa":
                control_indices = matched.deranged_control[chosen]
            else:
                control_indices = matched.control[chosen]
            batch = _training_arrays(
                windows,
                treatment_indices=treatment_indices,
                control_indices=control_indices,
                ownership=ownership,
            )
            optimizer.zero_grad(set_to_none=True)
            losses = _objective_loss(
                torch,
                network,
                batch,
                ownership=ownership,
                latent_weight=self.config.effective_latent_weight,
            )
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(optimized, 1.0)
            optimizer.step()
            _update_target_encoder(
                network,
                decay=self.config.ema_decay,
            )
            scheduler.step()
            completed = step + 1
            if (
                completed % self.config.checkpoint_interval == 0
                or completed == self.config.pretrain_steps
            ):
                row = {
                    "step": float(completed),
                    "total": float(losses["total"].detach()),
                    "observable_effect": float(
                        losses["observable_effect"].detach()
                    ),
                    "latent_effect": float(
                        losses["latent_effect"].detach()
                    ),
                    "zero_effect": float(
                        losses["zero_effect"].detach()
                    ),
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                }
                if not np.all(np.isfinite(list(row.values()))):
                    raise RuntimeError(
                        "PairEffect-JEPA training became non-finite"
                    )
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
        self._network = network
        self._checkpoints = tuple(checkpoints)
        self._training_metrics = tuple(metrics)
        return self

    def select(
        self, windows: ActionConditionedWindows
    ) -> "PairEffectJepaModel":
        """Select a checkpoint by true matched observable-effect MSE."""

        (
            graph,
            features,
            controls,
            actions,
            ownership,
            network,
        ) = self._fitted_values()
        _validate_schema(
            windows,
            graph=graph,
            features=features,
            controls=controls,
            actions=actions,
        )
        matched = _match_rows(windows, require_derangement=False)
        torch = _require_torch()
        rows = []
        best_key: Optional[Tuple[float, int]] = None
        best_state = None
        best_step = None
        for checkpoint_step, state in self._checkpoints:
            network.load_state_dict(state)
            network.eval()
            prediction = _predict_effect_batches(
                torch,
                network,
                windows.histories[matched.treatment],
                windows.future_controls[matched.treatment],
                windows.future_actions[matched.treatment],
                ownership,
            )
            target = (
                windows.future_states[matched.treatment]
                - windows.future_states[matched.control]
            )
            value = _masked_mse(prediction, target, ownership)
            row = {
                "step": float(checkpoint_step),
                "observable_effect_mse": value,
            }
            if not np.all(np.isfinite(list(row.values()))):
                raise RuntimeError(
                    "PairEffect-JEPA selection became non-finite"
                )
            rows.append(row)
            key = (value, checkpoint_step)
            if best_key is None or key < best_key:
                best_key = key
                best_state = copy.deepcopy(state)
                best_step = checkpoint_step
        if best_state is None or best_step is None:
            raise RuntimeError("PairEffect-JEPA selected no checkpoint")
        network.load_state_dict(best_state)
        network.eval()
        for parameter in network.parameters():
            parameter.requires_grad_(False)
        self._selected_step = best_step
        self._selection_metrics = tuple(rows)
        self._checkpoints = ()
        return self

    def predict_effect(
        self,
        histories: NDArray[np.float64],
        future_controls: NDArray[np.float64],
        future_actions: NDArray[np.float64],
        graph: DeclaredTelemetryGraph,
    ) -> NDArray[np.float64]:
        """Predict an observable intervention effect from causal inputs."""

        (
            graph_,
            _,
            controls,
            actions,
            ownership,
            network,
        ) = self._selected_values()
        history = np.asarray(histories, dtype=np.float64)
        control = np.asarray(future_controls, dtype=np.float64)
        action = np.asarray(future_actions, dtype=np.float64)
        expected_history = (
            20,
            len(graph_.entities),
            len(self._feature_names),
        )
        expected_future = (
            len(network.horizon_embedding),
            len(graph_.entities),
        )
        if (
            graph.to_dict() != graph_.to_dict()
            or history.ndim != 4
            or history.shape[1:] != expected_history
            or control.shape
            != (
                len(history),
                expected_future[0],
                len(controls),
            )
            or action.shape
            != (
                len(history),
                expected_future[0],
                expected_future[1],
                len(actions),
            )
            or not all(
                np.all(np.isfinite(value))
                for value in (history, control, action)
            )
        ):
            raise ValueError("PairEffect-JEPA public inputs are invalid")
        torch = _require_torch()
        return _predict_effect_batches(
            torch, network, history, control, action, ownership
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe selected model."""

        (
            graph,
            features,
            controls,
            actions,
            ownership,
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
            "state_dict": _state_dict_to_payload(network.state_dict()),
            "selected_step": self._selected_step,
            "training_metrics": [
                dict(row) for row in self._training_metrics
            ],
            "selection_metrics": [
                dict(row) for row in self._selection_metrics
            ],
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "PairEffectJepaModel":
        """Restore and validate a selected paired-effect model."""

        if (
            payload.get("schema_version") != cls.schema_version
            or payload.get("kind") != cls.kind
        ):
            raise ValueError("PairEffect-JEPA model schema is invalid")
        config = PairEffectJepaConfig(**dict(payload["config"]))
        graph = DeclaredTelemetryGraph.from_dict(dict(payload["graph"]))
        features = tuple(str(value) for value in payload["feature_names"])
        controls = tuple(str(value) for value in payload["control_names"])
        actions = tuple(str(value) for value in payload["action_names"])
        ownership = np.asarray(payload["ownership_mask"], dtype=np.bool_)
        expected = (len(graph.entities), len(features))
        if (
            not features
            or not controls
            or not actions
            or ownership.shape != expected
            or not np.any(ownership)
        ):
            raise ValueError("PairEffect-JEPA fitted schema is invalid")
        torch = _require_torch()
        network = _build_network(
            torch,
            config=config,
            entity_count=len(graph.entities),
            feature_count=len(features),
            condition_dimension=(
                len(controls) + len(graph.entities) * len(actions)
            ),
            horizon=10,
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
            raise ValueError("PairEffect-JEPA selected step is invalid")
        for parameter in network.parameters():
            parameter.requires_grad_(False)
        result = cls(config)
        result._graph = graph
        result._feature_names = features
        result._control_names = controls
        result._action_names = actions
        result._ownership_mask = ownership
        result._network = network.eval()
        result._selected_step = selected_step
        result._training_metrics = _metric_rows(
            payload.get("training_metrics", ())
        )
        result._selection_metrics = _metric_rows(
            payload.get("selection_metrics", ())
        )
        return result

    def _fitted_values(self) -> Tuple[Any, ...]:
        if (
            self._graph is None
            or not self._feature_names
            or not self._control_names
            or not self._action_names
            or self._ownership_mask is None
            or self._network is None
        ):
            raise ValueError("PairEffect-JEPA model is not fitted")
        return (
            self._graph,
            self._feature_names,
            self._control_names,
            self._action_names,
            self._ownership_mask,
            self._network,
        )

    def _selected_values(self) -> Tuple[Any, ...]:
        values = self._fitted_values()
        if self._selected_step is None:
            raise ValueError("PairEffect-JEPA model is not selected")
        return values


class PairEffectCorrectedDynamics:
    """Compose a frozen raw no-action rollout with one effect predictor."""

    kind = "pair_effect_corrected_dynamics"
    schema_version = 1

    def __init__(
        self,
        raw_model: ContractiveLowRankDynamics,
        effect_model: PairEffectJepaModel,
    ) -> None:
        self.raw_model = raw_model
        self.effect_model = effect_model

    @property
    def parameter_count(self) -> int:
        """Return all parameters used by the composed predictive core."""

        return (
            self.raw_model.parameter_count
            + self.effect_model.inference_parameter_count
        )

    def rollout(
        self,
        histories: NDArray[Any],
        future_controls: NDArray[Any],
        future_actions: NDArray[Any],
        graph: DeclaredTelemetryGraph,
    ) -> TrajectoryDistribution:
        """Return raw no-action rollout plus the predicted paired effect."""

        actions = np.asarray(future_actions, dtype=np.float64)
        no_action = np.zeros_like(actions)
        no_action[..., 0] = 1.0
        raw = self.raw_model.rollout(
            histories, future_controls, no_action, graph
        )
        effect = self.effect_model.predict_effect(
            np.asarray(histories, dtype=np.float64),
            np.asarray(future_controls, dtype=np.float64),
            actions,
            graph,
        )
        return TrajectoryDistribution(
            mean=np.asarray(raw.mean + effect, dtype=np.float64),
            variance=raw.variance.copy(),
        )

    def to_dict(self) -> Mapping[str, Any]:
        """Return a JSON-safe composed predictive core."""

        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "raw_model": self.raw_model.to_dict(),
            "effect_model": self.effect_model.to_dict(),
            "parameter_count": self.parameter_count,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "PairEffectCorrectedDynamics":
        """Restore the frozen raw and paired-effect paths."""

        if (
            payload.get("schema_version") != cls.schema_version
            or payload.get("kind") != cls.kind
            or not isinstance(payload.get("raw_model"), Mapping)
            or not isinstance(payload.get("effect_model"), Mapping)
        ):
            raise ValueError(
                "PairEffect-JEPA composed artifact is invalid"
            )
        result = cls(
            ContractiveLowRankDynamics.from_dict(
                dict(payload["raw_model"])
            ),
            PairEffectJepaModel.from_dict(dict(payload["effect_model"])),
        )
        if payload.get("parameter_count") != result.parameter_count:
            raise ValueError(
                "PairEffect-JEPA composed parameter count differs"
            )
        return result


def _build_network(
    torch: Any,
    *,
    config: PairEffectJepaConfig,
    entity_count: int,
    feature_count: int,
    condition_dimension: int,
    horizon: int,
) -> Any:
    nn = torch.nn

    class EntityEncoder(nn.Module):  # type: ignore[misc,name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.input = nn.Linear(feature_count, config.hidden_width)
            self.entity_embedding = nn.Embedding(
                entity_count, config.hidden_width
            )
            self.hidden = nn.Linear(
                config.hidden_width, config.hidden_width
            )
            self.output = nn.Linear(config.hidden_width, config.width)

        def forward(self, values: Any) -> Any:
            hidden = self.input(values)
            shape = [1] * (hidden.ndim - 2) + [
                entity_count,
                config.hidden_width,
            ]
            hidden = hidden + self.entity_embedding.weight.reshape(shape)
            hidden = torch.nn.functional.silu(hidden)
            hidden = hidden + torch.nn.functional.silu(self.hidden(hidden))
            return self.output(hidden)

    class Network(nn.Module):  # type: ignore[misc,name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.online_encoder = EntityEncoder()
            self.target_encoder = copy.deepcopy(self.online_encoder)
            for parameter in self.target_encoder.parameters():
                parameter.requires_grad_(False)
            self.horizon_embedding = nn.Parameter(
                torch.empty(horizon, config.width)
            )
            nn.init.normal_(self.horizon_embedding, std=0.02)
            predictor_input = (
                entity_count * config.width
                + condition_dimension
                + config.width
            )
            self.predictor = nn.Sequential(
                nn.Linear(predictor_input, config.hidden_width),
                nn.SiLU(),
                nn.Linear(config.hidden_width, config.hidden_width),
                nn.SiLU(),
                nn.Linear(
                    config.hidden_width, entity_count * config.width
                ),
            )
            self.decoder = nn.Linear(config.width, feature_count)

        def predict(self, current: Any, condition: Any) -> Any:
            batch = len(current)
            current_flat = current.flatten(1)
            repeated = current_flat[:, None].expand(
                batch, horizon, current_flat.shape[1]
            )
            horizon_values = self.horizon_embedding[None].expand(
                batch, horizon, config.width
            )
            values = torch.cat(
                (repeated, condition, horizon_values), dim=-1
            )
            return self.predictor(values).reshape(
                batch, horizon, entity_count, config.width
            )

    return Network()


def _objective_loss(
    torch: Any,
    network: Any,
    batch: Mapping[str, Any],
    *,
    ownership: NDArray[np.bool_],
    latent_weight: float,
) -> Mapping[str, Any]:
    history = torch.as_tensor(batch["history"], dtype=torch.float32)
    treatment_future = torch.as_tensor(
        batch["treatment_future"], dtype=torch.float32
    )
    control_future = torch.as_tensor(
        batch["control_future"], dtype=torch.float32
    )
    condition = torch.as_tensor(
        batch["condition"], dtype=torch.float32
    )
    no_action_condition = torch.as_tensor(
        batch["no_action_condition"], dtype=torch.float32
    )
    mask = torch.as_tensor(ownership, dtype=torch.float32)
    current = network.online_encoder(history[:, -1])
    predicted_latent = network.predict(current, condition)
    predicted_effect = network.decoder(predicted_latent) * mask
    observable_target = (treatment_future - control_future) * mask
    observable_loss = _torch_masked_mse(
        torch, predicted_effect, observable_target, mask
    )
    with torch.no_grad():
        target_latent = (
            network.target_encoder(treatment_future)
            - network.target_encoder(control_future)
        )
    latent_loss = torch.nn.functional.l1_loss(
        predicted_latent, target_latent
    )
    zero_latent = network.predict(current, no_action_condition)
    zero_effect = network.decoder(zero_latent) * mask
    zero_loss = _torch_masked_mse(
        torch, zero_effect, torch.zeros_like(zero_effect), mask
    )
    return {
        "total": observable_loss + zero_loss + latent_weight * latent_loss,
        "observable_effect": observable_loss,
        "latent_effect": latent_loss,
        "zero_effect": zero_loss,
    }


def _torch_masked_mse(
    torch: Any, predicted: Any, target: Any, mask: Any
) -> Any:
    squared = torch.square(predicted - target) * mask
    denominator = len(predicted) * predicted.shape[1] * torch.sum(mask)
    return torch.sum(squared) / denominator


def _training_arrays(
    windows: ActionConditionedWindows,
    *,
    treatment_indices: NDArray[np.int64],
    control_indices: NDArray[np.int64],
    ownership: NDArray[np.bool_],
) -> Mapping[str, NDArray[np.float64]]:
    actions = windows.future_actions[treatment_indices]
    controls = windows.future_controls[treatment_indices]
    no_action = np.zeros_like(actions)
    no_action[..., 0] = 1.0
    history_mask = ownership[None, None]
    future_mask = ownership[None, None]
    return {
        "history": np.asarray(
            windows.histories[treatment_indices] * history_mask,
            dtype=np.float64,
        ),
        "treatment_future": np.asarray(
            windows.future_states[treatment_indices] * future_mask,
            dtype=np.float64,
        ),
        "control_future": np.asarray(
            windows.future_states[control_indices] * future_mask,
            dtype=np.float64,
        ),
        "condition": _condition(controls, actions),
        "no_action_condition": _condition(controls, no_action),
    }


def _condition(
    controls: NDArray[np.float64],
    actions: NDArray[np.float64],
) -> NDArray[np.float64]:
    return np.asarray(
        np.concatenate(
            (
                controls,
                actions.reshape(len(actions), actions.shape[1], -1),
            ),
            axis=2,
        ),
        dtype=np.float64,
    )


def _predict_effect_batches(
    torch: Any,
    network: Any,
    histories: NDArray[np.float64],
    controls: NDArray[np.float64],
    actions: NDArray[np.float64],
    ownership: NDArray[np.bool_],
) -> NDArray[np.float64]:
    mask = torch.as_tensor(ownership, dtype=torch.float32)
    parts = []
    network.eval()
    with torch.no_grad():
        for start in range(0, len(histories), 256):
            stop = start + 256
            history = torch.as_tensor(
                histories[start:stop] * ownership[None, None],
                dtype=torch.float32,
            )
            condition = torch.as_tensor(
                _condition(controls[start:stop], actions[start:stop]),
                dtype=torch.float32,
            )
            current = network.online_encoder(history[:, -1])
            latent = network.predict(current, condition)
            effect = network.decoder(latent) * mask
            parts.append(effect.cpu().numpy())
    result = np.asarray(np.concatenate(parts), dtype=np.float64)
    active = np.any(actions[..., 1] > 0.5, axis=(1, 2))
    return np.asarray(
        result * active[:, None, None, None], dtype=np.float64
    )


def _update_target_encoder(network: Any, *, decay: float) -> None:
    for target, online in zip(
        network.target_encoder.parameters(),
        network.online_encoder.parameters(),
    ):
        target.data.mul_(decay).add_(online.data, alpha=1.0 - decay)


def _match_rows(
    windows: ActionConditionedWindows,
    *,
    require_derangement: bool = True,
) -> _MatchedRows:
    pair_names = tuple(sorted(set(windows.matched_pair_ids)))
    treatment_rows: List[int] = []
    control_rows: List[int] = []
    paired_names: List[str] = []
    pair_positions: Dict[str, List[int]] = {name: [] for name in pair_names}
    trajectory_by_pair: Dict[str, Dict[str, List[int]]] = {
        name: {} for name in pair_names
    }
    for row, (pair, trajectory) in enumerate(
        zip(windows.matched_pair_ids, windows.trajectory_ids)
    ):
        trajectory_by_pair[pair].setdefault(trajectory, []).append(row)
    arm_by_pair: Dict[str, Tuple[str, str]] = {}
    for pair in pair_names:
        trajectories = trajectory_by_pair[pair]
        treatments = [
            trajectory
            for trajectory, rows in trajectories.items()
            if np.any(windows.future_actions[rows, ..., 1] > 0.5)
        ]
        controls = [
            trajectory
            for trajectory in trajectories
            if trajectory not in treatments
        ]
        if len(treatments) != 1 or len(controls) != 1:
            raise ValueError(
                "PairEffect-JEPA requires one treatment and one control"
            )
        arm_by_pair[pair] = (treatments[0], controls[0])
        treatment_index = {
            int(windows.transition_indices[row]): row
            for row in trajectories[treatments[0]]
        }
        control_index = {
            int(windows.transition_indices[row]): row
            for row in trajectories[controls[0]]
        }
        transitions = sorted(set(treatment_index) & set(control_index))
        if not transitions:
            raise ValueError("PairEffect-JEPA pair has no aligned rows")
        for transition in transitions:
            position = len(treatment_rows)
            treatment_rows.append(treatment_index[transition])
            control_rows.append(control_index[transition])
            paired_names.append(pair)
            pair_positions[pair].append(position)
    treatment = np.asarray(treatment_rows, dtype=np.int64)
    control = np.asarray(control_rows, dtype=np.int64)
    donor_by_pair = (
        _deranged_pair_map(windows, pair_names, arm_by_pair)
        if require_derangement
        else {pair: pair for pair in pair_names}
    )
    control_lookup = {
        (
            pair,
            int(windows.transition_indices[control[position]]),
        ): int(control[position])
        for position, pair in enumerate(paired_names)
    }
    deranged = []
    for position, pair in enumerate(paired_names):
        transition = int(windows.transition_indices[treatment[position]])
        donor = donor_by_pair[pair]
        donor_row = control_lookup.get((donor, transition))
        if donor_row is None:
            raise ValueError(
                "PairEffect-JEPA derangement lost transition alignment"
            )
        deranged.append(donor_row)
    return _MatchedRows(
        treatment=treatment,
        control=control,
        pair_ids=tuple(paired_names),
        rows_by_pair=tuple(
            np.asarray(pair_positions[pair], dtype=np.int64)
            for pair in pair_names
        ),
        deranged_control=np.asarray(deranged, dtype=np.int64),
    )


def _deranged_pair_map(
    windows: ActionConditionedWindows,
    pair_names: Tuple[str, ...],
    arm_by_pair: Mapping[str, Tuple[str, str]],
) -> Mapping[str, str]:
    try:
        worker = windows.control_feature_names.index("worker_replicas")
    except ValueError:
        worker = 0
    groups: Dict[Tuple[int, float], List[str]] = {}
    for pair in pair_names:
        treatment_trajectory = arm_by_pair[pair][0]
        rows = [
            row
            for row, trajectory in enumerate(windows.trajectory_ids)
            if trajectory == treatment_trajectory
        ]
        action = windows.future_actions[rows]
        kind_values = np.any(action[..., 2:] > 0.5, axis=(0, 1, 2))
        active_kinds = np.flatnonzero(kind_values)
        kind = int(active_kinds[0]) if len(active_kinds) else -1
        topology = float(windows.future_controls[rows[0], 0, worker])
        groups.setdefault((kind, topology), []).append(pair)
    result: Dict[str, str] = {}
    for members in groups.values():
        ordered = sorted(members)
        if len(ordered) < 2:
            raise ValueError(
                "PairEffect-JEPA derangement cell needs at least two pairs"
            )
        for position, pair in enumerate(ordered):
            result[pair] = ordered[(position + 1) % len(ordered)]
    if any(pair == donor for pair, donor in result.items()):
        raise ValueError("PairEffect-JEPA derangement retained a twin")
    return result


def _masked_mse(
    predicted: NDArray[np.float64],
    target: NDArray[np.float64],
    ownership: NDArray[np.bool_],
) -> float:
    return float(
        np.mean(
            np.square(
                np.asarray(predicted, dtype=np.float64)
                - np.asarray(target, dtype=np.float64)
            )[..., ownership]
        )
    )


def _validate_windows(
    windows: ActionConditionedWindows, expected_pair_count: int
) -> None:
    if (
        len(set(windows.matched_pair_ids)) != expected_pair_count
        or windows.histories.shape[1] != 20
        or windows.future_states.shape[1] != 10
        or windows.action_feature_names[0] != "no_action"
        or "applicable" not in windows.action_feature_names
    ):
        raise ValueError("PairEffect-JEPA fitting windows are invalid")


def _validate_schema(
    windows: ActionConditionedWindows,
    *,
    graph: DeclaredTelemetryGraph,
    features: Tuple[str, ...],
    controls: Tuple[str, ...],
    actions: Tuple[str, ...],
) -> None:
    if (
        windows.graph.to_dict() != graph.to_dict()
        or windows.state_feature_names != features
        or windows.control_feature_names != controls
        or windows.action_feature_names != actions
        or windows.histories.shape[1] != 20
        or windows.future_states.shape[1] != 10
    ):
        raise ValueError("PairEffect-JEPA selection schema differs")


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
            raise ValueError("PairEffect-JEPA state tensor shape differs")
        if array.dtype.kind not in ("i", "u", "b") and not np.all(
            np.isfinite(array)
        ):
            raise ValueError("PairEffect-JEPA state tensor is non-finite")
        result[str(name)] = (
            torch.as_tensor(array)
            if array.dtype.kind in ("i", "u", "b")
            else torch.as_tensor(array, dtype=torch.float32)
        )
    return result


def _metric_rows(values: Any) -> Tuple[Mapping[str, float], ...]:
    rows = []
    for raw in values:
        row = {
            str(key): float(value) for key, value in dict(raw).items()
        }
        if not np.all(np.isfinite(list(row.values()))):
            raise ValueError("PairEffect-JEPA metric row is non-finite")
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
            "PairEffect-JEPA fitting requires training dependencies"
        ) from error
