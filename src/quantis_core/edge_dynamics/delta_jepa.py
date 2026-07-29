"""Edge-sized Delta-JEPA action-displacement representation.

Future state, control, and action tensors are fitting-only inputs. Public
encoding accepts current histories and the declared graph. The offline
``diagnose_intervals`` seam is intentionally separate because it consumes
future observations to test the learned action-displacement mechanism.
"""

import copy
import importlib
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ..action_conditioned_dynamics import ActionConditionedWindows
from ..graph_telemetry import DeclaredTelemetryGraph
from .complete_lejepa import PairBlockedAnchorSchedule


DELTA_JEPA_OBJECTIVES = (
    "delta_jepa",
    "endpoint_concat",
    "prediction_only",
)


@dataclass(frozen=True)
class DeltaJepaConfig:
    """Frozen controls for one capacity-matched Delta-JEPA cell."""

    objective: str = "delta_jepa"
    width: int = 16
    hidden_width: int = 64
    decoder_hidden_width: int = 64
    decoder_layers: int = 2
    decoder_heads: int = 4
    action_horizon: int = 5
    pretrain_steps: int = 1600
    checkpoint_interval: int = 200
    learning_rate: float = 5e-5
    weight_decay: float = 1e-3
    expected_pair_count: int = 40
    seed: int = 16016
    device: str = "cpu"

    def __post_init__(self) -> None:
        integers = (
            self.width,
            self.hidden_width,
            self.decoder_hidden_width,
            self.decoder_layers,
            self.decoder_heads,
            self.action_horizon,
            self.pretrain_steps,
            self.checkpoint_interval,
            self.expected_pair_count,
        )
        if (
            self.objective not in DELTA_JEPA_OBJECTIVES
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
                for value in integers
            )
            or self.decoder_hidden_width % self.decoder_heads
            or self.action_horizon != 5
            or self.learning_rate <= 0.0
            or self.weight_decay < 0.0
            or isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.device != "cpu"
        ):
            raise ValueError("Delta-JEPA configuration is invalid")

    @property
    def action_weight(self) -> float:
        """Return the paper-matched LDAD coefficient for this cell."""

        return 0.0 if self.objective == "prediction_only" else 10.0


@dataclass(frozen=True)
class DeltaEncodedTelemetry:
    """Entity-preserving current state used by frozen downstream probes."""

    tokens: NDArray[np.float64]
    entity_ids: Tuple[str, ...]
    ownership_mask: NDArray[np.bool_]

    def __post_init__(self) -> None:
        if (
            self.tokens.ndim != 3
            or self.tokens.shape[1] != len(self.entity_ids)
            or self.ownership_mask.shape[0] != len(self.entity_ids)
            or not np.all(np.isfinite(self.tokens))
        ):
            raise ValueError("Delta-JEPA encoded telemetry is invalid")


@dataclass(frozen=True)
class DeltaIntervalDiagnostics:
    """Offline five-step displacement evidence."""

    decoder_inputs: NDArray[np.float64]
    displacements: NDArray[np.float64]
    predicted_actions: NDArray[np.float64]
    target_actions: NDArray[np.float64]
    state_changes: NDArray[np.float64]
    treatment_mask: NDArray[np.bool_]
    pair_ids: Tuple[str, ...]

    def __post_init__(self) -> None:
        count = len(self.pair_ids)
        if (
            self.decoder_inputs.ndim != 2
            or self.displacements.ndim != 3
            or self.predicted_actions.ndim != 4
            or self.target_actions.shape != self.predicted_actions.shape
            or self.state_changes.ndim != 3
            or self.treatment_mask.shape != (count,)
            or any(
                len(value) != count
                for value in (
                    self.decoder_inputs,
                    self.displacements,
                    self.predicted_actions,
                    self.target_actions,
                    self.state_changes,
                )
            )
            or not all(
                np.all(np.isfinite(value))
                for value in (
                    self.decoder_inputs,
                    self.displacements,
                    self.predicted_actions,
                    self.target_actions,
                    self.state_changes,
                )
            )
        ):
            raise ValueError("Delta-JEPA interval diagnostics are invalid")


def action_decoder_input(
    start: NDArray[np.float64],
    end: NDArray[np.float64],
    *,
    objective: str,
) -> NDArray[np.float64]:
    """Construct the exact capacity-matched LDAD or endpoint input."""

    before = np.asarray(start, dtype=np.float64)
    after = np.asarray(end, dtype=np.float64)
    if (
        objective not in DELTA_JEPA_OBJECTIVES
        or before.shape != after.shape
        or before.ndim < 2
        or not np.all(np.isfinite(before))
        or not np.all(np.isfinite(after))
    ):
        raise ValueError("Delta-JEPA decoder endpoints are invalid")
    flat_before = before.reshape(len(before), -1)
    flat_after = after.reshape(len(after), -1)
    if objective == "endpoint_concat":
        return np.concatenate((flat_before, flat_after), axis=1)
    displacement = flat_after - flat_before
    return np.concatenate((displacement, displacement), axis=1)


class DeltaJepaModel:
    """Restorable Delta-JEPA or exact-capacity null representation."""

    kind = "delta_jepa_representation"
    schema_version = 1

    def __init__(self, config: DeltaJepaConfig = DeltaJepaConfig()) -> None:
        self.config = config
        self._graph: Optional[DeclaredTelemetryGraph] = None
        self._feature_names: Tuple[str, ...] = ()
        self._control_names: Tuple[str, ...] = ()
        self._action_names: Tuple[str, ...] = ()
        self._ownership_mask: Optional[NDArray[np.bool_]] = None
        self._center: Optional[NDArray[np.float64]] = None
        self._scale: Optional[NDArray[np.float64]] = None
        self._condition_dimension: Optional[int] = None
        self._compact_action_dimension: Optional[int] = None
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
        return int(sum(value.numel() for value in network.encoder.parameters()))

    @property
    def selected_step(self) -> Optional[int]:
        return self._selected_step

    @property
    def training_metrics(self) -> Tuple[Mapping[str, float], ...]:
        return tuple(dict(row) for row in self._training_metrics)

    @property
    def selection_metrics(self) -> Tuple[Mapping[str, float], ...]:
        return tuple(dict(row) for row in self._selection_metrics)

    def fit(self, windows: ActionConditionedWindows) -> "DeltaJepaModel":
        """Fit checkpoint candidates using one anchor per matched pair."""

        if len(set(windows.matched_pair_ids)) != self.config.expected_pair_count:
            raise ValueError("Delta-JEPA fitting pair count differs")
        _validate_window_horizon(windows)
        if windows.action_feature_names[0] != "no_action":
            raise ValueError("Delta-JEPA expects no_action at position zero")
        torch = _require_torch()
        _seed_torch(torch, self.config.seed)
        ownership = _fit_owned_feature_mask(windows)
        full = np.concatenate((windows.histories, windows.future_states), axis=1)
        center, scale = _fit_normalizer(full, ownership)
        condition_dimension = (
            len(windows.control_feature_names)
            + len(windows.entity_names) * len(windows.action_feature_names)
        )
        compact_action_dimension = (
            len(windows.entity_names) * (len(windows.action_feature_names) - 1)
        )
        network = _build_network(
            torch,
            config=self.config,
            entity_count=len(windows.entity_names),
            feature_count=len(windows.state_feature_names),
            condition_dimension=condition_dimension,
            compact_action_dimension=compact_action_dimension,
        )
        optimizer = torch.optim.AdamW(
            network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.pretrain_steps
        )
        schedule = PairBlockedAnchorSchedule(windows, seed=self.config.seed + 1)
        metrics: List[Mapping[str, float]] = []
        checkpoints = []
        network.train()
        for step in range(self.config.pretrain_steps):
            anchor = schedule.batch(step)
            normalized, conditions, compact_actions = _training_batch(
                windows, anchor.indices, ownership, center, scale
            )
            optimizer.zero_grad(set_to_none=True)
            losses = _objective_loss(
                torch,
                network,
                torch.as_tensor(normalized, dtype=torch.float32),
                torch.as_tensor(conditions, dtype=torch.float32),
                torch.as_tensor(compact_actions, dtype=torch.float32),
                config=self.config,
            )
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            completed = step + 1
            if (
                completed % self.config.checkpoint_interval == 0
                or completed == self.config.pretrain_steps
            ):
                row = {
                    "step": float(completed),
                    "total": float(losses["total"].detach()),
                    "prediction": float(losses["prediction"].detach()),
                    "action": float(losses["action"].detach()),
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                }
                if not np.all(np.isfinite(list(row.values()))):
                    raise RuntimeError("Delta-JEPA training became non-finite")
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
        self._condition_dimension = condition_dimension
        self._compact_action_dimension = compact_action_dimension
        self._network = network
        self._checkpoints = tuple(checkpoints)
        self._training_metrics = tuple(metrics)
        return self

    def select(self, windows: ActionConditionedWindows) -> "DeltaJepaModel":
        """Select a checkpoint by this cell's own frozen two-term loss."""

        (
            graph,
            features,
            controls,
            actions,
            ownership,
            center,
            scale,
            _,
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
            raise ValueError("Delta-JEPA selection schema differs")
        _validate_window_horizon(windows)
        torch = _require_torch()
        schedule = PairBlockedAnchorSchedule(windows, seed=self.config.seed + 3)
        evaluation_steps = min(10, len(schedule.transitions))
        rows = []
        best_key: Optional[Tuple[float, int]] = None
        best_state = None
        best_step = None
        for checkpoint_step, state in self._checkpoints:
            network.load_state_dict(state)
            network.eval()
            weighted = {"total": 0.0, "prediction": 0.0, "action": 0.0}
            with torch.no_grad():
                for local_step in range(evaluation_steps):
                    anchor = schedule.batch(local_step)
                    normalized, conditions, compact_actions = _training_batch(
                        windows, anchor.indices, ownership, center, scale
                    )
                    losses = _objective_loss(
                        torch,
                        network,
                        torch.as_tensor(normalized, dtype=torch.float32),
                        torch.as_tensor(conditions, dtype=torch.float32),
                        torch.as_tensor(compact_actions, dtype=torch.float32),
                        config=self.config,
                    )
                    for name in weighted:
                        weighted[name] += float(losses[name])
            row = {
                "step": float(checkpoint_step),
                **{
                    name: value / float(evaluation_steps)
                    for name, value in weighted.items()
                },
            }
            if not np.all(np.isfinite(list(row.values()))):
                raise RuntimeError("Delta-JEPA selection became non-finite")
            rows.append(row)
            key = (row["total"], checkpoint_step)
            if best_key is None or key < best_key:
                best_key = key
                best_state = copy.deepcopy(state)
                best_step = checkpoint_step
        if best_state is None or best_step is None:
            raise RuntimeError("Delta-JEPA selected no checkpoint")
        network.load_state_dict(best_state)
        network.eval()
        for parameter in network.parameters():
            parameter.requires_grad_(False)
        self._selected_step = best_step
        self._selection_metrics = tuple(rows)
        self._checkpoints = ()
        return self

    def encode(
        self,
        histories: NDArray[np.float64],
        graph: DeclaredTelemetryGraph,
    ) -> DeltaEncodedTelemetry:
        """Encode current histories without future or action inputs."""

        torch = _require_torch()
        (
            _,
            _,
            _,
            _,
            ownership,
            center,
            scale,
            _,
            _,
            network,
        ) = self._selected_values()
        values = self._validate_histories(histories, graph)
        normalized = _normalize_states(values, ownership, center, scale)
        parts = []
        with torch.no_grad():
            for start in range(0, len(normalized), 256):
                tokens = network.encode(
                    torch.as_tensor(
                        normalized[start : start + 256, -1],
                        dtype=torch.float32,
                    )
                )
                parts.append(tokens.cpu().numpy())
        return DeltaEncodedTelemetry(
            tokens=np.asarray(np.concatenate(parts), dtype=np.float64),
            entity_ids=graph.entity_ids,
            ownership_mask=ownership.copy(),
        )

    def diagnose_intervals(
        self, windows: ActionConditionedWindows
    ) -> DeltaIntervalDiagnostics:
        """Decode actions from held interval endpoints for offline evidence."""

        (
            graph,
            features,
            controls,
            actions,
            ownership,
            center,
            scale,
            _,
            _,
            network,
        ) = self._selected_values()
        if (
            windows.graph.to_dict() != graph.to_dict()
            or windows.state_feature_names != features
            or windows.control_feature_names != controls
            or windows.action_feature_names != actions
        ):
            raise ValueError("Delta-JEPA diagnostic schema differs")
        _validate_window_horizon(windows)
        full = np.concatenate((windows.histories, windows.future_states), axis=1)
        normalized = _normalize_states(full, ownership, center, scale)
        torch = _require_torch()
        with torch.no_grad():
            encoded = network.encode(
                torch.as_tensor(normalized, dtype=torch.float32)
            ).cpu().numpy()
        starts = np.concatenate((encoded[:, 19:20], encoded[:, 24:25]), axis=1)
        ends = np.concatenate((encoded[:, 24:25], encoded[:, 29:30]), axis=1)
        start_flat = starts.reshape(-1, starts.shape[2], starts.shape[3])
        end_flat = ends.reshape(-1, ends.shape[2], ends.shape[3])
        decoder_inputs = action_decoder_input(
            start_flat, end_flat, objective=self.config.objective
        )
        with torch.no_grad():
            predicted = network.decode_actions(
                torch.as_tensor(decoder_inputs, dtype=torch.float32)
            ).cpu().numpy()
        entity_count = len(windows.entity_names)
        compact_count = len(windows.action_feature_names) - 1
        predicted = predicted.reshape(
            -1, self.config.action_horizon, entity_count, compact_count
        )
        target = windows.future_actions[..., 1:]
        target = target.reshape(
            len(target), 2, self.config.action_horizon, entity_count, compact_count
        ).reshape(predicted.shape)
        start_states = np.concatenate(
            (windows.histories[:, -1:, :, :], windows.future_states[:, 4:5]),
            axis=1,
        )
        end_states = np.concatenate(
            (windows.future_states[:, 4:5], windows.future_states[:, 9:10]),
            axis=1,
        )
        state_changes = (end_states - start_states).reshape(
            -1, len(windows.entity_names), len(windows.state_feature_names)
        )
        try:
            applicable = windows.action_feature_names.index("applicable") - 1
        except ValueError as error:
            raise ValueError(
                "Delta-JEPA diagnostics require applicable action"
            ) from error
        treatment = np.any(target[..., applicable] > 0.5, axis=(1, 2))
        return DeltaIntervalDiagnostics(
            decoder_inputs=np.asarray(decoder_inputs, dtype=np.float64),
            displacements=np.asarray(end_flat - start_flat, dtype=np.float64),
            predicted_actions=np.asarray(predicted, dtype=np.float64),
            target_actions=np.asarray(target, dtype=np.float64),
            state_changes=np.asarray(state_changes, dtype=np.float64),
            treatment_mask=np.asarray(treatment, dtype=np.bool_),
            pair_ids=tuple(
                pair
                for pair in windows.matched_pair_ids
                for _ in range(2)
            ),
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
            condition_dimension,
            compact_action_dimension,
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
            "condition_dimension": condition_dimension,
            "compact_action_dimension": compact_action_dimension,
            "state_dict": _state_dict_to_payload(network.state_dict()),
            "selected_step": self._selected_step,
            "training_metrics": [dict(row) for row in self._training_metrics],
            "selection_metrics": [dict(row) for row in self._selection_metrics],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DeltaJepaModel":
        """Restore and validate a selected Delta-JEPA model."""

        if (
            payload.get("schema_version") != cls.schema_version
            or payload.get("kind") != cls.kind
        ):
            raise ValueError("Delta-JEPA model schema is invalid")
        config = DeltaJepaConfig(**dict(payload["config"]))
        graph = DeclaredTelemetryGraph.from_dict(dict(payload["graph"]))
        features = tuple(str(value) for value in payload["feature_names"])
        controls = tuple(str(value) for value in payload["control_names"])
        actions = tuple(str(value) for value in payload["action_names"])
        ownership = np.asarray(payload["ownership_mask"], dtype=np.bool_)
        center = np.asarray(payload["center"], dtype=np.float64)
        scale = np.asarray(payload["scale"], dtype=np.float64)
        condition_dimension = payload["condition_dimension"]
        compact_action_dimension = payload["compact_action_dimension"]
        expected = (len(graph.entities), len(features))
        if (
            ownership.shape != expected
            or center.shape != expected
            or scale.shape != expected
            or not np.any(ownership)
            or np.any(scale <= 0.0)
            or not np.all(np.isfinite(center))
            or not np.all(np.isfinite(scale))
            or isinstance(condition_dimension, bool)
            or not isinstance(condition_dimension, int)
            or condition_dimension
            != len(controls) + len(graph.entities) * len(actions)
            or isinstance(compact_action_dimension, bool)
            or not isinstance(compact_action_dimension, int)
            or compact_action_dimension
            != len(graph.entities) * (len(actions) - 1)
        ):
            raise ValueError("Delta-JEPA fitted schema is invalid")
        torch = _require_torch()
        network = _build_network(
            torch,
            config=config,
            entity_count=len(graph.entities),
            feature_count=len(features),
            condition_dimension=condition_dimension,
            compact_action_dimension=compact_action_dimension,
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
            raise ValueError("Delta-JEPA selected step is invalid")
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
        result._condition_dimension = condition_dimension
        result._compact_action_dimension = compact_action_dimension
        result._network = network.eval()
        result._selected_step = selected_step
        result._training_metrics = _metric_rows(
            payload.get("training_metrics", ())
        )
        result._selection_metrics = _metric_rows(
            payload.get("selection_metrics", ())
        )
        return result

    def _validate_histories(
        self,
        histories: NDArray[np.float64],
        graph: DeclaredTelemetryGraph,
    ) -> NDArray[np.float64]:
        graph_, features, *_ = self._selected_values()
        values = np.asarray(histories, dtype=np.float64)
        if (
            graph.to_dict() != graph_.to_dict()
            or values.ndim != 4
            or values.shape[1:] != (
                20,
                len(graph_.entities),
                len(features),
            )
            or not np.all(np.isfinite(values))
        ):
            raise ValueError("Delta-JEPA public inputs are invalid")
        return values

    def _fitted_values(self) -> Tuple[Any, ...]:
        if (
            self._graph is None
            or not self._feature_names
            or not self._control_names
            or not self._action_names
            or self._ownership_mask is None
            or self._center is None
            or self._scale is None
            or self._condition_dimension is None
            or self._compact_action_dimension is None
            or self._network is None
        ):
            raise ValueError("Delta-JEPA model is not fitted")
        return (
            self._graph,
            self._feature_names,
            self._control_names,
            self._action_names,
            self._ownership_mask,
            self._center,
            self._scale,
            self._condition_dimension,
            self._compact_action_dimension,
            self._network,
        )

    def _selected_values(self) -> Tuple[Any, ...]:
        values = self._fitted_values()
        if self._selected_step is None:
            raise ValueError("Delta-JEPA model is not selected")
        return values


def _build_network(
    torch: Any,
    *,
    config: DeltaJepaConfig,
    entity_count: int,
    feature_count: int,
    condition_dimension: int,
    compact_action_dimension: int,
) -> Any:
    nn = torch.nn

    class Encoder(nn.Module):  # type: ignore[misc,name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.input_fc = nn.Linear(feature_count, config.hidden_width)
            self.entity_embedding = nn.Embedding(
                entity_count, config.hidden_width
            )
            self.hidden_fc = nn.Linear(
                config.hidden_width, config.hidden_width
            )
            self.output_fc = nn.Linear(config.hidden_width, config.width)

        def forward(self, values: Any) -> Any:
            if (
                values.ndim not in (3, 4)
                or values.shape[-2:] != (entity_count, feature_count)
            ):
                raise ValueError("Delta-JEPA encoder tensor is misaligned")
            hidden = self.input_fc(values)
            embedding_shape = [1] * (hidden.ndim - 2) + [
                entity_count,
                config.hidden_width,
            ]
            hidden = hidden + self.entity_embedding.weight.reshape(
                embedding_shape
            )
            hidden = torch.nn.functional.silu(hidden)
            hidden = hidden + torch.nn.functional.silu(
                self.hidden_fc(hidden)
            )
            return self.output_fc(hidden)

    class ActionDecoder(nn.Module):  # type: ignore[misc,name-defined]
        def __init__(self) -> None:
            super().__init__()
            input_dimension = 2 * entity_count * config.width
            self.queries = nn.Parameter(
                torch.empty(
                    config.action_horizon, config.decoder_hidden_width
                )
            )
            nn.init.normal_(self.queries, std=0.02)
            self.modulation = nn.Linear(
                input_dimension, 2 * config.decoder_hidden_width
            )
            layer = nn.TransformerEncoderLayer(
                d_model=config.decoder_hidden_width,
                nhead=config.decoder_heads,
                dim_feedforward=2 * config.decoder_hidden_width,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                layer, num_layers=config.decoder_layers
            )
            self.output = nn.Linear(
                config.decoder_hidden_width, compact_action_dimension
            )

        def forward(self, decoder_input: Any) -> Any:
            if (
                decoder_input.ndim != 2
                or decoder_input.shape[1]
                != 2 * entity_count * config.width
            ):
                raise ValueError("Delta-JEPA decoder input is misaligned")
            shift, scale = self.modulation(decoder_input).chunk(2, dim=-1)
            queries = self.queries[None] * (1.0 + scale[:, None])
            queries = queries + shift[:, None]
            return self.output(self.transformer(queries))

    class Network(nn.Module):  # type: ignore[misc,name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.encoder = Encoder()
            flat_width = entity_count * config.width
            self.predictor = nn.Sequential(
                nn.Linear(flat_width + condition_dimension, config.hidden_width),
                nn.SiLU(),
                nn.Linear(config.hidden_width, config.hidden_width),
                nn.SiLU(),
                nn.Linear(config.hidden_width, flat_width),
            )
            self.action_decoder = ActionDecoder()

        def encode(self, values: Any) -> Any:
            return self.encoder(values)

        def predict(self, current: Any, condition: Any) -> Any:
            if (
                current.ndim != 3
                or current.shape[1:] != (entity_count, config.width)
                or condition.ndim != 2
                or condition.shape != (len(current), condition_dimension)
            ):
                raise ValueError("Delta-JEPA predictor tensor is misaligned")
            return self.predictor(
                torch.cat((current.flatten(1), condition), dim=1)
            ).reshape(len(current), entity_count, config.width)

        def decode_actions(self, decoder_input: Any) -> Any:
            return self.action_decoder(decoder_input)

    return Network()


def _objective_loss(
    torch: Any,
    network: Any,
    full: Any,
    conditions: Any,
    compact_actions: Any,
    *,
    config: DeltaJepaConfig,
) -> Mapping[str, Any]:
    encoded = network.encode(full)
    prediction = network.predict(
        encoded[:, 19:29].flatten(0, 1),
        conditions.flatten(0, 1),
    )
    prediction_loss = torch.nn.functional.mse_loss(
        prediction, encoded[:, 20:30].flatten(0, 1)
    )
    starts = torch.cat((encoded[:, 19:20], encoded[:, 24:25]), dim=1)
    ends = torch.cat((encoded[:, 24:25], encoded[:, 29:30]), dim=1)
    starts = starts.flatten(0, 1)
    ends = ends.flatten(0, 1)
    if config.objective == "endpoint_concat":
        decoder_input = torch.cat(
            (starts.flatten(1), ends.flatten(1)), dim=1
        )
    else:
        displacement = ends.flatten(1) - starts.flatten(1)
        decoder_input = torch.cat((displacement, displacement), dim=1)
    decoded = network.decode_actions(decoder_input)
    action_loss = torch.nn.functional.mse_loss(
        decoded, compact_actions.flatten(0, 1)
    )
    total = prediction_loss + config.action_weight * action_loss
    return {
        "total": total,
        "prediction": prediction_loss,
        "action": action_loss,
    }


def _training_batch(
    windows: ActionConditionedWindows,
    indices: NDArray[np.int64],
    ownership: NDArray[np.bool_],
    center: NDArray[np.float64],
    scale: NDArray[np.float64],
) -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    full = np.concatenate(
        (windows.histories[indices], windows.future_states[indices]), axis=1
    )
    conditions = np.concatenate(
        (
            windows.future_controls[indices],
            windows.future_actions[indices].reshape(
                len(indices), windows.future_actions.shape[1], -1
            ),
        ),
        axis=2,
    )
    compact = windows.future_actions[indices, ..., 1:].reshape(
        len(indices),
        2,
        5,
        len(windows.entity_names)
        * (len(windows.action_feature_names) - 1),
    )
    return (
        _normalize_states(full, ownership, center, scale),
        np.asarray(conditions, dtype=np.float64),
        np.asarray(compact, dtype=np.float64),
    )


def _validate_window_horizon(windows: ActionConditionedWindows) -> None:
    if windows.histories.shape[1] != 20 or windows.future_states.shape[1] != 10:
        raise ValueError("Delta-JEPA requires 20+10 timestep windows")


def _fit_normalizer(
    full: NDArray[np.float64],
    ownership: NDArray[np.bool_],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    center = np.mean(full, axis=(0, 1))
    scale = np.std(full, axis=(0, 1))
    center = np.where(ownership, center, 0.0)
    scale = np.where(ownership, np.maximum(scale, 1e-6), 1.0)
    return np.asarray(center, dtype=np.float64), np.asarray(
        scale, dtype=np.float64
    )


def _normalize_states(
    values: NDArray[np.float64],
    ownership: NDArray[np.bool_],
    center: NDArray[np.float64],
    scale: NDArray[np.float64],
) -> NDArray[np.float64]:
    normalized = (np.asarray(values, dtype=np.float64) - center) / scale
    return np.asarray(normalized * ownership[None, None], dtype=np.float64)


def _fit_owned_feature_mask(
    windows: ActionConditionedWindows,
) -> NDArray[np.bool_]:
    entity_positions = {
        entity_id: position
        for position, entity_id in enumerate(windows.entity_names)
    }
    feature_positions = {
        name: position
        for position, name in enumerate(windows.state_feature_names)
    }
    mask = np.zeros(
        (len(windows.entity_names), len(windows.state_feature_names)),
        dtype=np.bool_,
    )
    for feature_key, entity_id in windows.graph.binding_map().items():
        feature_name = feature_key.split(".", 1)[-1]
        if entity_id in entity_positions and feature_name in feature_positions:
            mask[
                entity_positions[entity_id],
                feature_positions[feature_name],
            ] = True
    mask |= np.ptp(windows.histories, axis=(0, 1)) > 1e-9
    if not np.any(mask):
        raise ValueError("Delta-JEPA schema has no observations")
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
            raise ValueError("Delta-JEPA state tensor shape differs")
        if array.dtype.kind not in ("i", "u", "b") and not np.all(
            np.isfinite(array)
        ):
            raise ValueError("Delta-JEPA state tensor is non-finite")
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
            raise ValueError("Delta-JEPA metric row is non-finite")
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
            "Delta-JEPA fitting requires optional training dependencies"
        ) from error
