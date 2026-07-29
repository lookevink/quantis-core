"""Hard-sufficiency JEPA residual grounded by operational witnesses."""

import copy
import hashlib
import importlib
import json
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


CONTRACT_JEPA_OBJECTIVES = (
    "task_grounded_contract_jepa",
    "supervised_task_contract",
    "ungrounded_contract_jepa",
)


@dataclass(frozen=True)
class TaskGroundedContractConfig:
    """Frozen controls for one equal-capacity contract cell."""

    objective: str = "task_grounded_contract_jepa"
    width: int = 16
    hidden_width: int = 64
    pretrain_steps: int = 800
    checkpoint_interval: int = 100
    learning_rate: float = 5e-4
    weight_decay: float = 1e-3
    ema_decay: float = 0.996
    latent_weight: float = 0.2
    state_weight: float = 0.1
    paired_effect_weight: float = 1.0
    effect_score_weight: float = 0.2
    correction_bound_multiplier: float = 3.0
    expected_pair_count: int = 40
    seed: int = 24021
    device: str = "cpu"

    def __post_init__(self) -> None:
        integers = (
            self.width,
            self.hidden_width,
            self.pretrain_steps,
            self.checkpoint_interval,
            self.expected_pair_count,
        )
        weights = (
            self.learning_rate,
            self.latent_weight,
            self.state_weight,
            self.paired_effect_weight,
            self.effect_score_weight,
            self.correction_bound_multiplier,
        )
        if (
            self.objective not in CONTRACT_JEPA_OBJECTIVES
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
                for value in integers
            )
            or self.weight_decay < 0.0
            or any(value < 0.0 for value in weights)
            or self.learning_rate <= 0.0
            or not 0.0 <= self.ema_decay < 1.0
            or self.correction_bound_multiplier <= 0.0
            or isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.device != "cpu"
        ):
            raise ValueError(
                "task-grounded Contract-JEPA configuration is invalid"
            )

    @property
    def effective_latent_weight(self) -> float:
        if self.objective == "supervised_task_contract":
            return 0.0
        return self.latent_weight

    @property
    def effective_state_weight(self) -> float:
        if self.objective == "ungrounded_contract_jepa":
            return 0.0
        return self.state_weight

    @property
    def effective_paired_effect_weight(self) -> float:
        if self.objective == "ungrounded_contract_jepa":
            return 0.0
        return self.paired_effect_weight

    @property
    def effective_effect_score_weight(self) -> float:
        if self.objective == "ungrounded_contract_jepa":
            return 0.0
        return self.effect_score_weight


@dataclass(frozen=True)
class ContractEncodedTelemetry:
    """Structurally sufficient raw state plus learned entity tokens."""

    raw_current_state: NDArray[np.float64]
    learned_tokens: NDArray[np.float64]
    entity_ids: Tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            self.raw_current_state.ndim != 3
            or self.learned_tokens.ndim != 3
            or self.raw_current_state.shape[:2]
            != self.learned_tokens.shape[:2]
            or self.raw_current_state.shape[1] != len(self.entity_ids)
            or not np.all(np.isfinite(self.raw_current_state))
            or not np.all(np.isfinite(self.learned_tokens))
        ):
            raise ValueError("Contract-JEPA encoded telemetry is invalid")


@dataclass(frozen=True)
class _MatchedRows:
    treatment: NDArray[np.int64]
    control: NDArray[np.int64]
    rows_by_pair: Tuple[NDArray[np.int64], ...]


class TaskGroundedContractJepa:
    """Bounded residual branch with state and intervention witnesses."""

    kind = "task_grounded_contract_jepa"
    schema_version = 1

    def __init__(
        self,
        config: TaskGroundedContractConfig = (
            TaskGroundedContractConfig()
        ),
    ) -> None:
        self.config = config
        self._graph: Optional[DeclaredTelemetryGraph] = None
        self._feature_names: Tuple[str, ...] = ()
        self._control_names: Tuple[str, ...] = ()
        self._action_names: Tuple[str, ...] = ()
        self._ownership_mask: Optional[NDArray[np.bool_]] = None
        self._correction_scale: Optional[NDArray[np.float64]] = None
        self._baseline_sha256 = ""
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
    def inference_parameter_count(self) -> int:
        *_, network = self._fitted_values()
        modules: Sequence[Any] = (
            network.online_encoder,
            network.predictor,
            network.correction_decoder,
            network.state_decoder,
            network.effect_score_head,
        )
        return int(
            sum(
                parameter.numel()
                for module in modules
                for parameter in module.parameters()
            )
            + network.horizon_embedding.numel()
        )

    @property
    def correction_bound(self) -> NDArray[np.float64]:
        *_, scale, _, _ = self._fitted_values()
        return np.asarray(
            self.config.correction_bound_multiplier * scale,
            dtype=np.float64,
        )

    @property
    def baseline_sha256(self) -> str:
        self._fitted_values()
        return self._baseline_sha256

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
        self,
        windows: ActionConditionedWindows,
        baseline: ContractiveLowRankDynamics,
    ) -> "TaskGroundedContractJepa":
        """Fit a bounded branch without mutating the raw predictive core."""

        _validate_windows(windows, self.config.expected_pair_count)
        baseline_payload = baseline.to_dict()
        baseline_hash = _artifact_sha256(baseline_payload)
        raw_prediction = baseline.rollout(
            windows.histories,
            windows.future_controls,
            windows.future_actions,
            windows.graph,
        ).mean
        residual = windows.future_states - raw_prediction
        ownership = fit_owned_feature_mask(windows)
        scale = np.std(residual, axis=(0, 1))
        scale = np.where(
            ownership, np.maximum(scale, 1e-3), 0.0
        ).astype(np.float64)
        matched = _match_rows(windows)
        torch = _require_torch()
        _seed_torch(torch, self.config.seed)
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
            local = np.asarray(
                [
                    rows[(step + position) % len(rows)]
                    for position, rows in enumerate(
                        matched.rows_by_pair
                    )
                ],
                dtype=np.int64,
            )
            treatment = matched.treatment[local]
            control = matched.control[local]
            batch = _training_arrays(
                windows,
                raw_prediction,
                treatment=treatment,
                control=control,
                ownership=ownership,
            )
            optimizer.zero_grad(set_to_none=True)
            losses = _objective_loss(
                torch,
                network,
                batch,
                ownership=ownership,
                correction_bound=(
                    self.config.correction_bound_multiplier * scale
                ),
                config=self.config,
            )
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(optimized, 1.0)
            optimizer.step()
            _update_target_encoder(
                network, decay=self.config.ema_decay
            )
            scheduler.step()
            completed = step + 1
            if (
                completed % self.config.checkpoint_interval == 0
                or completed == self.config.pretrain_steps
            ):
                row = {
                    "step": float(completed),
                    **{
                        name: float(value.detach())
                        for name, value in losses.items()
                    },
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                }
                if not np.all(np.isfinite(list(row.values()))):
                    raise RuntimeError(
                        "task-grounded Contract-JEPA became non-finite"
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
        if _artifact_sha256(baseline.to_dict()) != baseline_hash:
            raise RuntimeError(
                "Contract-JEPA fitting mutated the raw baseline"
            )
        self._graph = windows.graph
        self._feature_names = windows.state_feature_names
        self._control_names = windows.control_feature_names
        self._action_names = windows.action_feature_names
        self._ownership_mask = ownership
        self._correction_scale = scale
        self._baseline_sha256 = baseline_hash
        self._network = network
        self._checkpoints = tuple(checkpoints)
        self._training_metrics = tuple(metrics)
        return self

    def select(
        self,
        windows: ActionConditionedWindows,
        baseline: ContractiveLowRankDynamics,
    ) -> "TaskGroundedContractJepa":
        """Select a checkpoint using its frozen selection objective."""

        (
            graph,
            features,
            controls,
            actions,
            ownership,
            scale,
            _,
            network,
        ) = self._fitted_values()
        _validate_schema(
            windows,
            graph=graph,
            features=features,
            controls=controls,
            actions=actions,
        )
        if _artifact_sha256(baseline.to_dict()) != self._baseline_sha256:
            raise ValueError("Contract-JEPA selection baseline differs")
        raw_prediction = baseline.rollout(
            windows.histories,
            windows.future_controls,
            windows.future_actions,
            windows.graph,
        ).mean
        matched = _match_rows(windows)
        torch = _require_torch()
        rows = []
        best_key: Optional[Tuple[float, int]] = None
        best_state = None
        best_step = None
        for step, state in self._checkpoints:
            network.load_state_dict(state)
            network.eval()
            batch = _training_arrays(
                windows,
                raw_prediction,
                treatment=matched.treatment,
                control=matched.control,
                ownership=ownership,
            )
            with torch.no_grad():
                losses = _objective_loss(
                    torch,
                    network,
                    batch,
                    ownership=ownership,
                    correction_bound=(
                        self.config.correction_bound_multiplier * scale
                    ),
                    config=self.config,
                )
            objective = (
                float(losses["residual"])
                + self.config.effective_paired_effect_weight
                * float(losses["paired_effect"])
                + self.config.effective_effect_score_weight
                * float(losses["effect_score"])
                + self.config.effective_state_weight
                * float(losses["state"])
            )
            row = {
                "step": float(step),
                "selection_objective": objective,
                **{
                    name: float(value)
                    for name, value in losses.items()
                    if name != "total"
                },
            }
            if not np.all(np.isfinite(list(row.values()))):
                raise RuntimeError(
                    "task-grounded Contract-JEPA selection is non-finite"
                )
            rows.append(row)
            key = (objective, step)
            if best_key is None or key < best_key:
                best_key = key
                best_state = copy.deepcopy(state)
                best_step = step
        if best_state is None or best_step is None:
            raise RuntimeError("Contract-JEPA selected no checkpoint")
        network.load_state_dict(best_state)
        network.eval()
        for parameter in network.parameters():
            parameter.requires_grad_(False)
        self._selected_step = best_step
        self._selection_metrics = tuple(rows)
        self._checkpoints = ()
        return self

    def encode_contract(
        self,
        histories: NDArray[np.float64],
        graph: DeclaredTelemetryGraph,
    ) -> ContractEncodedTelemetry:
        """Return raw current state unchanged plus learned tokens."""

        graph_, _, _, _, ownership, _, _, network = (
            self._selected_values()
        )
        history = _validate_histories(
            histories,
            graph,
            fitted_graph=graph_,
            feature_count=len(self._feature_names),
        )
        torch = _require_torch()
        with torch.no_grad():
            tokens = network.online_encoder(
                torch.as_tensor(
                    history[:, -1] * ownership[None],
                    dtype=torch.float32,
                )
            ).cpu().numpy()
        return ContractEncodedTelemetry(
            raw_current_state=history[:, -1].copy(),
            learned_tokens=np.asarray(tokens, dtype=np.float64),
            entity_ids=graph_.entity_ids,
        )

    def predict_contract(
        self,
        histories: NDArray[np.float64],
        future_controls: NDArray[np.float64],
        future_actions: NDArray[np.float64],
        graph: DeclaredTelemetryGraph,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Predict bounded correction and non-negative effect witness."""

        (
            graph_,
            _,
            controls,
            actions,
            ownership,
            scale,
            _,
            network,
        ) = self._selected_values()
        history = _validate_histories(
            histories,
            graph,
            fitted_graph=graph_,
            feature_count=len(self._feature_names),
        )
        control = np.asarray(future_controls, dtype=np.float64)
        action = np.asarray(future_actions, dtype=np.float64)
        horizon = len(network.horizon_embedding)
        if (
            control.shape != (len(history), horizon, len(controls))
            or action.shape
            != (
                len(history),
                horizon,
                len(graph_.entities),
                len(actions),
            )
            or not np.all(np.isfinite(control))
            or not np.all(np.isfinite(action))
        ):
            raise ValueError("Contract-JEPA public conditions are invalid")
        return _predict_batches(
            _require_torch(),
            network,
            history,
            control,
            action,
            ownership,
            self.config.correction_bound_multiplier * scale,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe selected residual contract."""

        (
            graph,
            features,
            controls,
            actions,
            ownership,
            scale,
            _,
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
            "correction_scale": scale.tolist(),
            "baseline_sha256": self._baseline_sha256,
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
    ) -> "TaskGroundedContractJepa":
        """Restore and validate one selected residual contract."""

        if (
            payload.get("schema_version") != cls.schema_version
            or payload.get("kind") != cls.kind
        ):
            raise ValueError("Contract-JEPA model schema is invalid")
        config = TaskGroundedContractConfig(**dict(payload["config"]))
        graph = DeclaredTelemetryGraph.from_dict(dict(payload["graph"]))
        features = tuple(str(value) for value in payload["feature_names"])
        controls = tuple(str(value) for value in payload["control_names"])
        actions = tuple(str(value) for value in payload["action_names"])
        ownership = np.asarray(payload["ownership_mask"], dtype=np.bool_)
        scale = np.asarray(
            payload["correction_scale"], dtype=np.float64
        )
        expected = (len(graph.entities), len(features))
        baseline_hash = str(payload["baseline_sha256"])
        if (
            not features
            or not controls
            or not actions
            or ownership.shape != expected
            or scale.shape != expected
            or not np.any(ownership)
            or np.any(scale < 0.0)
            or not np.all(np.isfinite(scale))
            or len(baseline_hash) != 64
        ):
            raise ValueError("Contract-JEPA fitted schema is invalid")
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
        selected = payload.get("selected_step")
        if (
            isinstance(selected, bool)
            or not isinstance(selected, int)
            or selected < 1
        ):
            raise ValueError("Contract-JEPA selected step is invalid")
        for parameter in network.parameters():
            parameter.requires_grad_(False)
        result = cls(config)
        result._graph = graph
        result._feature_names = features
        result._control_names = controls
        result._action_names = actions
        result._ownership_mask = ownership
        result._correction_scale = scale
        result._baseline_sha256 = baseline_hash
        result._network = network.eval()
        result._selected_step = selected
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
            or self._correction_scale is None
            or not self._baseline_sha256
            or self._network is None
        ):
            raise ValueError("Contract-JEPA model is not fitted")
        return (
            self._graph,
            self._feature_names,
            self._control_names,
            self._action_names,
            self._ownership_mask,
            self._correction_scale,
            self._baseline_sha256,
            self._network,
        )

    def _selected_values(self) -> Tuple[Any, ...]:
        values = self._fitted_values()
        if self._selected_step is None:
            raise ValueError("Contract-JEPA model is not selected")
        return values


class TaskGroundedContractDynamics:
    """Compose immutable raw dynamics with one bounded selected correction."""

    kind = "task_grounded_contract_dynamics"
    schema_version = 1

    def __init__(
        self,
        baseline: ContractiveLowRankDynamics,
        branch: TaskGroundedContractJepa,
        *,
        gain: float = 0.0,
    ) -> None:
        if _artifact_sha256(baseline.to_dict()) != branch.baseline_sha256:
            raise ValueError("Contract-JEPA baseline identity differs")
        self.baseline = baseline
        self.branch = branch
        self._gain = _validate_gain(gain)

    @property
    def selected_gain(self) -> float:
        return self._gain

    @property
    def parameter_count(self) -> int:
        return (
            self.baseline.parameter_count
            + self.branch.inference_parameter_count
        )

    def set_gain(self, value: float) -> "TaskGroundedContractDynamics":
        self._gain = _validate_gain(value)
        return self

    def rollout(
        self,
        histories: NDArray[Any],
        future_controls: NDArray[Any],
        future_actions: NDArray[Any],
        graph: DeclaredTelemetryGraph,
    ) -> TrajectoryDistribution:
        """Return exact raw output at zero or bounded composition."""

        raw = self.baseline.rollout(
            histories, future_controls, future_actions, graph
        )
        if self._gain == 0.0:
            return raw
        correction, _ = self.branch.predict_contract(
            np.asarray(histories, dtype=np.float64),
            np.asarray(future_controls, dtype=np.float64),
            np.asarray(future_actions, dtype=np.float64),
            graph,
        )
        return TrajectoryDistribution(
            mean=np.asarray(
                raw.mean + self._gain * correction, dtype=np.float64
            ),
            variance=raw.variance.copy(),
        )

    def witness_scores(
        self,
        histories: NDArray[Any],
        future_controls: NDArray[Any],
        future_actions: NDArray[Any],
        graph: DeclaredTelemetryGraph,
    ) -> NDArray[np.float64]:
        """Return public non-negative per-horizon effect scores."""

        _, score = self.branch.predict_contract(
            np.asarray(histories, dtype=np.float64),
            np.asarray(future_controls, dtype=np.float64),
            np.asarray(future_actions, dtype=np.float64),
            graph,
        )
        return score

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "baseline": self.baseline.to_dict(),
            "branch": self.branch.to_dict(),
            "gain": self._gain,
            "parameter_count": self.parameter_count,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "TaskGroundedContractDynamics":
        if (
            payload.get("schema_version") != cls.schema_version
            or payload.get("kind") != cls.kind
            or not isinstance(payload.get("baseline"), Mapping)
            or not isinstance(payload.get("branch"), Mapping)
        ):
            raise ValueError("Contract-JEPA composed artifact is invalid")
        result = cls(
            ContractiveLowRankDynamics.from_dict(
                dict(payload["baseline"])
            ),
            TaskGroundedContractJepa.from_dict(
                dict(payload["branch"])
            ),
            gain=float(payload["gain"]),
        )
        if payload.get("parameter_count") != result.parameter_count:
            raise ValueError("Contract-JEPA parameter count differs")
        return result


def _build_network(
    torch: Any,
    *,
    config: TaskGroundedContractConfig,
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
            input_width = (
                entity_count * config.width
                + condition_dimension
                + config.width
            )
            self.predictor = nn.Sequential(
                nn.Linear(input_width, config.hidden_width),
                nn.SiLU(),
                nn.Linear(config.hidden_width, config.hidden_width),
                nn.SiLU(),
                nn.Linear(
                    config.hidden_width, entity_count * config.width
                ),
            )
            self.correction_decoder = nn.Linear(
                config.width, feature_count
            )
            self.state_decoder = nn.Linear(config.width, feature_count)
            self.effect_score_head = nn.Sequential(
                nn.Linear(entity_count * config.width, config.hidden_width),
                nn.SiLU(),
                nn.Linear(config.hidden_width, 1),
                nn.Softplus(),
            )

        def predict(self, current: Any, condition: Any) -> Any:
            batch = len(current)
            flat = current.flatten(1)
            repeated = flat[:, None].expand(
                batch, horizon, flat.shape[1]
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
    batch: Mapping[str, NDArray[np.float64]],
    *,
    ownership: NDArray[np.bool_],
    correction_bound: NDArray[np.float64],
    config: TaskGroundedContractConfig,
) -> Mapping[str, Any]:
    histories = torch.as_tensor(
        batch["histories"], dtype=torch.float32
    )
    future = torch.as_tensor(batch["future"], dtype=torch.float32)
    raw = torch.as_tensor(batch["raw"], dtype=torch.float32)
    condition = torch.as_tensor(
        batch["condition"], dtype=torch.float32
    )
    mask = torch.as_tensor(ownership, dtype=torch.float32)
    bound = torch.as_tensor(
        correction_bound, dtype=torch.float32
    )
    current = network.online_encoder(histories[:, -1])
    predicted_latent = network.predict(current, condition)
    correction = (
        torch.tanh(network.correction_decoder(predicted_latent))
        * bound
        * mask
    )
    residual_target = (future - raw) * mask
    residual_loss = _torch_masked_mse(
        torch, correction, residual_target, mask
    )
    with torch.no_grad():
        target_latent = network.target_encoder(future)
    latent_loss = torch.nn.functional.l1_loss(
        predicted_latent, target_latent
    )
    decoded_current = network.state_decoder(current) * mask
    state_loss = _torch_masked_mse(
        torch, decoded_current, histories[:, -1] * mask, mask
    )
    pair_count = len(histories) // 2
    corrected = raw + correction
    predicted_effect = (
        corrected[:pair_count] - corrected[pair_count:]
    )
    observed_effect = future[:pair_count] - future[pair_count:]
    paired_effect_loss = _torch_masked_mse(
        torch, predicted_effect, observed_effect, mask
    )
    effect_target = torch.sqrt(
        torch.sum(torch.square(observed_effect) * mask, dim=(2, 3))
        / torch.sum(mask)
        + 1e-12
    )
    treatment_score = network.effect_score_head(
        predicted_latent[:pair_count].flatten(2)
    ).squeeze(-1)
    control_score = network.effect_score_head(
        predicted_latent[pair_count:].flatten(2)
    ).squeeze(-1)
    effect_score_loss = (
        torch.nn.functional.mse_loss(treatment_score, effect_target)
        + torch.nn.functional.mse_loss(
            control_score, torch.zeros_like(control_score)
        )
    )
    total = (
        residual_loss
        + config.effective_latent_weight * latent_loss
        + config.effective_state_weight * state_loss
        + config.effective_paired_effect_weight * paired_effect_loss
        + config.effective_effect_score_weight * effect_score_loss
    )
    return {
        "total": total,
        "residual": residual_loss,
        "latent": latent_loss,
        "state": state_loss,
        "paired_effect": paired_effect_loss,
        "effect_score": effect_score_loss,
    }


def _torch_masked_mse(
    torch: Any, predicted: Any, target: Any, mask: Any
) -> Any:
    squared = torch.square(predicted - target) * mask
    leading = int(np.prod(predicted.shape[:-2]))
    return torch.sum(squared) / (leading * torch.sum(mask))


def _training_arrays(
    windows: ActionConditionedWindows,
    raw_prediction: NDArray[np.float64],
    *,
    treatment: NDArray[np.int64],
    control: NDArray[np.int64],
    ownership: NDArray[np.bool_],
) -> Mapping[str, NDArray[np.float64]]:
    indices = np.concatenate((treatment, control))
    histories = windows.histories[indices] * ownership[None, None]
    future = windows.future_states[indices] * ownership[None, None]
    controls = windows.future_controls[indices]
    actions = windows.future_actions[indices]
    return {
        "histories": np.asarray(histories, dtype=np.float64),
        "future": np.asarray(future, dtype=np.float64),
        "raw": np.asarray(
            raw_prediction[indices] * ownership[None, None],
            dtype=np.float64,
        ),
        "condition": _condition(controls, actions),
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


def _predict_batches(
    torch: Any,
    network: Any,
    histories: NDArray[np.float64],
    controls: NDArray[np.float64],
    actions: NDArray[np.float64],
    ownership: NDArray[np.bool_],
    correction_bound: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    mask = torch.as_tensor(ownership, dtype=torch.float32)
    bound = torch.as_tensor(
        correction_bound, dtype=torch.float32
    )
    correction_parts = []
    score_parts = []
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
            predicted = network.predict(current, condition)
            correction = (
                torch.tanh(network.correction_decoder(predicted))
                * bound
                * mask
            )
            score = network.effect_score_head(
                predicted.flatten(2)
            ).squeeze(-1)
            correction_parts.append(correction.cpu().numpy())
            score_parts.append(score.cpu().numpy())
    return (
        np.asarray(np.concatenate(correction_parts), dtype=np.float64),
        np.asarray(np.concatenate(score_parts), dtype=np.float64),
    )


def _match_rows(windows: ActionConditionedWindows) -> _MatchedRows:
    pair_names = tuple(sorted(set(windows.matched_pair_ids)))
    treatment_rows: List[int] = []
    control_rows: List[int] = []
    positions: Dict[str, List[int]] = {pair: [] for pair in pair_names}
    trajectories: Dict[str, Dict[str, List[int]]] = {
        pair: {} for pair in pair_names
    }
    for row, (pair, trajectory) in enumerate(
        zip(windows.matched_pair_ids, windows.trajectory_ids)
    ):
        trajectories[pair].setdefault(trajectory, []).append(row)
    for pair in pair_names:
        arms = trajectories[pair]
        treatment = [
            name
            for name, rows in arms.items()
            if np.any(windows.future_actions[rows, ..., 1] > 0.5)
        ]
        control = [name for name in arms if name not in treatment]
        if len(treatment) != 1 or len(control) != 1:
            raise ValueError(
                "Contract-JEPA requires one treatment and one control"
            )
        treatment_index = {
            int(windows.transition_indices[row]): row
            for row in arms[treatment[0]]
        }
        control_index = {
            int(windows.transition_indices[row]): row
            for row in arms[control[0]]
        }
        transitions = sorted(set(treatment_index) & set(control_index))
        if not transitions:
            raise ValueError("Contract-JEPA pair has no aligned rows")
        for transition in transitions:
            position = len(treatment_rows)
            treatment_rows.append(treatment_index[transition])
            control_rows.append(control_index[transition])
            positions[pair].append(position)
    return _MatchedRows(
        treatment=np.asarray(treatment_rows, dtype=np.int64),
        control=np.asarray(control_rows, dtype=np.int64),
        rows_by_pair=tuple(
            np.asarray(positions[pair], dtype=np.int64)
            for pair in pair_names
        ),
    )


def _update_target_encoder(network: Any, *, decay: float) -> None:
    for target, online in zip(
        network.target_encoder.parameters(),
        network.online_encoder.parameters(),
    ):
        target.data.mul_(decay).add_(online.data, alpha=1.0 - decay)


def _validate_histories(
    histories: NDArray[np.float64],
    graph: DeclaredTelemetryGraph,
    *,
    fitted_graph: DeclaredTelemetryGraph,
    feature_count: int,
) -> NDArray[np.float64]:
    values = np.asarray(histories, dtype=np.float64)
    if (
        graph.to_dict() != fitted_graph.to_dict()
        or values.shape[1:]
        != (20, len(fitted_graph.entities), feature_count)
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("Contract-JEPA histories are invalid")
    return values


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
        raise ValueError("Contract-JEPA fitting windows are invalid")


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
        raise ValueError("Contract-JEPA selection schema differs")


def _validate_gain(value: float) -> float:
    gain = float(value)
    if not np.isfinite(gain) or not 0.0 <= gain <= 1.0:
        raise ValueError("Contract-JEPA gain must be in [0, 1]")
    return gain


def _artifact_sha256(payload: Mapping[str, Any]) -> str:
    data = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


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
            raise ValueError("Contract-JEPA state tensor shape differs")
        if array.dtype.kind not in ("i", "u", "b") and not np.all(
            np.isfinite(array)
        ):
            raise ValueError("Contract-JEPA state tensor is non-finite")
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
            raise ValueError("Contract-JEPA metric row is non-finite")
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
            "Contract-JEPA fitting requires training dependencies"
        ) from error
