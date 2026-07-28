"""Compact temporal and hybrid graph models for edge development."""

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Protocol, Tuple

import numpy as np
from numpy.typing import NDArray

from ..action_conditioned_dynamics import (
    ActionConditionedWindows,
    TrajectoryDistribution,
)
from ..action_dynamics_real_corpus import DenseActionVarxDynamics
from ..graph_telemetry import DeclaredTelemetryGraph


class EdgeDynamicsModel(Protocol):
    """Common rollout seam for adjacent model experiments."""

    kind: str

    def fit(self, windows: ActionConditionedWindows) -> "EdgeDynamicsModel":
        """Fit from normalized development windows."""

    def rollout(
        self,
        histories: NDArray[Any],
        future_controls: NDArray[Any],
        future_actions: NDArray[Any],
        graph: DeclaredTelemetryGraph,
    ) -> TrajectoryDistribution:
        """Predict one normalized future trajectory."""

    def to_dict(self) -> Mapping[str, Any]:
        """Return a JSON-safe fitted model artifact."""

    @property
    def parameter_count(self) -> int:
        """Return the number of stored scalar model parameters."""


class DenseVarxAdapter:
    """Expose the frozen dense VARX through the candidate-model seam."""

    kind = "action_conditioned_dense_varx"

    def __init__(
        self, *, ridge: float = 1e-3, variance_floor: float = 1e-4
    ) -> None:
        self.model = DenseActionVarxDynamics(
            ridge=ridge, variance_floor=variance_floor
        )

    def fit(
        self, windows: ActionConditionedWindows
    ) -> "DenseVarxAdapter":
        self.model.fit(windows)
        return self

    def rollout(
        self,
        histories: NDArray[Any],
        future_controls: NDArray[Any],
        future_actions: NDArray[Any],
        graph: DeclaredTelemetryGraph,
    ) -> TrajectoryDistribution:
        return self.model.rollout(
            np.asarray(histories, dtype=np.float64),
            np.asarray(future_controls, dtype=np.float64),
            np.asarray(future_actions, dtype=np.float64),
            graph,
        )

    @property
    def parameter_count(self) -> int:
        state = dict(self.model.to_dict()["state"])
        coefficients = np.asarray(state["coefficients"])
        variance = np.asarray(state["residual_variance"])
        return int(coefficients.size + variance.size)

    def to_dict(self) -> Mapping[str, Any]:
        artifact = dict(self.model.to_dict())
        artifact["parameter_count"] = self.parameter_count
        return artifact


@dataclass(frozen=True)
class EchoStateConfig:
    """Small deterministic reservoir choices."""

    reservoir_size: int = 32
    spectral_radius: float = 0.85
    input_scale: float = 0.08
    leak_rate: float = 0.5
    connectivity: float = 0.1
    ridge: float = 1e-2
    variance_floor: float = 1e-4
    seed: int = 17

    def __post_init__(self) -> None:
        if (
            self.reservoir_size < 2
            or not 0.0 < self.spectral_radius < 1.0
            or not 0.0 < self.input_scale <= 1.0
            or not 0.0 < self.leak_rate <= 1.0
            or not 0.0 < self.connectivity <= 1.0
            or self.ridge <= 0.0
            or self.variance_floor <= 0.0
        ):
            raise ValueError("echo-state configuration is invalid")


class EchoStateActionDynamics:
    """Action-conditioned reservoir with a ridge readout."""

    kind = "echo_state_action_dynamics"

    def __init__(self, config: EchoStateConfig = EchoStateConfig()) -> None:
        self.config = config
        self._graph: Optional[DeclaredTelemetryGraph] = None
        self._state_shape: Optional[Tuple[int, int]] = None
        self._control_count = 0
        self._action_shape: Optional[Tuple[int, int]] = None
        self._input_weights: Optional[NDArray[np.float64]] = None
        self._reservoir_weights: Optional[NDArray[np.float64]] = None
        self._readout: Optional[NDArray[np.float64]] = None
        self._residual_variance: Optional[NDArray[np.float64]] = None

    def fit(
        self, windows: ActionConditionedWindows
    ) -> "EchoStateActionDynamics":
        sample_count = len(windows.histories)
        if sample_count < 2:
            raise ValueError("echo-state fit requires at least two samples")
        entity_count = len(windows.entity_names)
        feature_count = len(windows.state_feature_names)
        control_count = len(windows.control_feature_names)
        action_count = len(windows.action_feature_names)
        state_width = entity_count * feature_count
        action_width = entity_count * action_count
        input_width = state_width + control_count + action_width
        rng = np.random.default_rng(self.config.seed)
        input_weights = rng.uniform(
            -self.config.input_scale,
            self.config.input_scale,
            size=(input_width + 1, self.config.reservoir_size),
        )
        raw_reservoir = rng.normal(
            0.0,
            1.0,
            size=(
                self.config.reservoir_size,
                self.config.reservoir_size,
            ),
        )
        keep = rng.random(raw_reservoir.shape) < self.config.connectivity
        raw_reservoir *= keep
        radius = float(
            np.max(np.abs(np.linalg.eigvals(raw_reservoir)))
        )
        if radius <= 1e-12:
            raise ValueError("sampled echo-state reservoir is degenerate")
        reservoir_weights = (
            raw_reservoir * self.config.spectral_radius / radius
        )
        histories = np.asarray(windows.histories, dtype=np.float64)
        state = self._warm_reservoir(
            histories,
            input_weights,
            reservoir_weights,
            control_count,
            action_width,
        )
        current = histories[:, -1].reshape(sample_count, state_width)
        controls = np.asarray(
            windows.future_controls[:, 0], dtype=np.float64
        )
        actions = np.asarray(
            windows.future_actions[:, 0], dtype=np.float64
        ).reshape(sample_count, action_width)
        conditioned = np.concatenate(
            (current, controls, actions), axis=1
        )
        state = self._advance(
            state, conditioned, input_weights, reservoir_weights
        )
        design = np.concatenate(
            (
                current,
                controls,
                actions,
                state,
                np.ones((sample_count, 1), dtype=np.float64),
            ),
            axis=1,
        )
        target = np.asarray(
            windows.future_states[:, 0], dtype=np.float64
        ).reshape(sample_count, state_width)
        penalty = np.eye(design.shape[1], dtype=np.float64)
        penalty[-1, -1] = 0.0
        readout = np.linalg.solve(
            design.T @ design + self.config.ridge * penalty,
            design.T @ target,
        )
        residual = target - design @ readout
        self._graph = windows.graph
        self._state_shape = (entity_count, feature_count)
        self._control_count = control_count
        self._action_shape = (entity_count, action_count)
        self._input_weights = input_weights
        self._reservoir_weights = reservoir_weights
        self._readout = readout
        self._residual_variance = np.maximum(
            np.mean(np.square(residual), axis=0),
            self.config.variance_floor,
        ).reshape(entity_count, feature_count)
        return self

    def rollout(
        self,
        histories: NDArray[Any],
        future_controls: NDArray[Any],
        future_actions: NDArray[Any],
        graph: DeclaredTelemetryGraph,
    ) -> TrajectoryDistribution:
        (
            fitted_graph,
            state_shape,
            control_count,
            action_shape,
            input_weights,
            reservoir_weights,
            readout,
            residual_variance,
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
        sample_count = len(history)
        state_width = state_shape[0] * state_shape[1]
        action_width = action_shape[0] * action_shape[1]
        reservoir = self._warm_reservoir(
            history,
            input_weights,
            reservoir_weights,
            control_count,
            action_width,
        )
        current = history[:, -1].reshape(sample_count, state_width)
        means = np.empty(
            (sample_count, controls.shape[1], *state_shape),
            dtype=np.float64,
        )
        for step in range(controls.shape[1]):
            action = actions[:, step].reshape(
                sample_count, action_width
            )
            conditioned = np.concatenate(
                (current, controls[:, step], action), axis=1
            )
            reservoir = self._advance(
                reservoir,
                conditioned,
                input_weights,
                reservoir_weights,
            )
            design = np.concatenate(
                (
                    current,
                    controls[:, step],
                    action,
                    reservoir,
                    np.ones((sample_count, 1), dtype=np.float64),
                ),
                axis=1,
            )
            current = design @ readout
            means[:, step] = current.reshape(sample_count, *state_shape)
        variances = np.broadcast_to(
            residual_variance, means.shape
        ).copy()
        return TrajectoryDistribution(mean=means, variance=variances)

    @property
    def parameter_count(self) -> int:
        values = self._fitted_values()
        return int(
            values[4].size
            + values[5].size
            + values[6].size
            + values[7].size
        )

    def to_dict(self) -> Mapping[str, Any]:
        (
            graph,
            state_shape,
            control_count,
            action_shape,
            input_weights,
            reservoir_weights,
            readout,
            residual_variance,
        ) = self._fitted_values()
        return {
            "schema_version": 1,
            "kind": self.kind,
            "config": {
                "reservoir_size": self.config.reservoir_size,
                "spectral_radius": self.config.spectral_radius,
                "input_scale": self.config.input_scale,
                "leak_rate": self.config.leak_rate,
                "connectivity": self.config.connectivity,
                "ridge": self.config.ridge,
                "variance_floor": self.config.variance_floor,
                "seed": self.config.seed,
            },
            "graph": graph.to_dict(),
            "state_shape": list(state_shape),
            "control_count": control_count,
            "action_shape": list(action_shape),
            "parameter_count": self.parameter_count,
            "state": {
                "input_weights": input_weights.tolist(),
                "reservoir_weights": reservoir_weights.tolist(),
                "readout": readout.tolist(),
                "residual_variance": residual_variance.tolist(),
            },
        }

    def _warm_reservoir(
        self,
        histories: NDArray[np.float64],
        input_weights: NDArray[np.float64],
        reservoir_weights: NDArray[np.float64],
        control_count: int,
        action_width: int,
    ) -> NDArray[np.float64]:
        sample_count = len(histories)
        state = np.zeros(
            (sample_count, self.config.reservoir_size),
            dtype=np.float64,
        )
        zeros = np.zeros(
            (sample_count, control_count + action_width),
            dtype=np.float64,
        )
        for step in range(histories.shape[1]):
            input_values = np.concatenate(
                (
                    histories[:, step].reshape(sample_count, -1),
                    zeros,
                ),
                axis=1,
            )
            state = self._advance(
                state,
                input_values,
                input_weights,
                reservoir_weights,
            )
        return state

    def _advance(
        self,
        state: NDArray[np.float64],
        input_values: NDArray[np.float64],
        input_weights: NDArray[np.float64],
        reservoir_weights: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        augmented = np.concatenate(
            (
                input_values,
                np.ones((len(input_values), 1), dtype=np.float64),
            ),
            axis=1,
        )
        proposed = np.tanh(
            augmented @ input_weights + state @ reservoir_weights
        )
        return (
            (1.0 - self.config.leak_rate) * state
            + self.config.leak_rate * proposed
        )

    def _fitted_values(
        self,
    ) -> Tuple[
        DeclaredTelemetryGraph,
        Tuple[int, int],
        int,
        Tuple[int, int],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        if (
            self._graph is None
            or self._state_shape is None
            or self._action_shape is None
            or self._input_weights is None
            or self._reservoir_weights is None
            or self._readout is None
            or self._residual_variance is None
        ):
            raise ValueError("echo-state model is not fitted")
        return (
            self._graph,
            self._state_shape,
            self._control_count,
            self._action_shape,
            self._input_weights,
            self._reservoir_weights,
            self._readout,
            self._residual_variance,
        )


@dataclass(frozen=True)
class LowRankConfig:
    """Stable low-rank linear transition choices."""

    rank: int = 24
    maximum_spectral_radius: float = 0.98
    ridge: float = 1e-3
    variance_floor: float = 1e-4

    def __post_init__(self) -> None:
        if (
            self.rank < 1
            or not 0.0 < self.maximum_spectral_radius <= 1.0
            or self.ridge <= 0.0
            or self.variance_floor <= 0.0
        ):
            raise ValueError("low-rank configuration is invalid")


class ContractiveLowRankDynamics:
    """Dense global VARX compressed into a contractive transition."""

    kind = "contractive_low_rank_action_dynamics"

    def __init__(self, config: LowRankConfig = LowRankConfig()) -> None:
        self.config = config
        self._graph: Optional[DeclaredTelemetryGraph] = None
        self._state_shape: Optional[Tuple[int, int]] = None
        self._control_count = 0
        self._action_shape: Optional[Tuple[int, int]] = None
        self._left: Optional[NDArray[np.float64]] = None
        self._right: Optional[NDArray[np.float64]] = None
        self._exogenous: Optional[NDArray[np.float64]] = None
        self._residual_variance: Optional[NDArray[np.float64]] = None
        self._spectral_radius = 0.0

    def fit(
        self, windows: ActionConditionedWindows
    ) -> "ContractiveLowRankDynamics":
        sample_count = len(windows.histories)
        entity_count = len(windows.entity_names)
        feature_count = len(windows.state_feature_names)
        control_count = len(windows.control_feature_names)
        action_count = len(windows.action_feature_names)
        state_width = entity_count * feature_count
        action_width = entity_count * action_count
        if sample_count < 2 or self.config.rank > state_width:
            raise ValueError("low-rank fit dimensions are invalid")
        state = np.asarray(
            windows.histories[:, -1], dtype=np.float64
        ).reshape(sample_count, state_width)
        controls = np.asarray(
            windows.future_controls[:, 0], dtype=np.float64
        )
        actions = np.asarray(
            windows.future_actions[:, 0], dtype=np.float64
        ).reshape(sample_count, action_width)
        exogenous_design = np.concatenate(
            (
                controls,
                actions,
                np.ones((sample_count, 1), dtype=np.float64),
            ),
            axis=1,
        )
        design = np.concatenate((state, exogenous_design), axis=1)
        target = np.asarray(
            windows.future_states[:, 0], dtype=np.float64
        ).reshape(sample_count, state_width)
        penalty = np.eye(design.shape[1], dtype=np.float64)
        penalty[-1, -1] = 0.0
        coefficients = np.linalg.solve(
            design.T @ design + self.config.ridge * penalty,
            design.T @ target,
        )
        transition = coefficients[:state_width]
        left_vectors, singular_values, right_vectors = np.linalg.svd(
            transition, full_matrices=False
        )
        rank = self.config.rank
        left = left_vectors[:, :rank] * singular_values[:rank]
        right = right_vectors[:rank]
        compressed = left @ right
        spectral_radius = float(
            np.max(np.abs(np.linalg.eigvals(compressed)))
        )
        if spectral_radius > self.config.maximum_spectral_radius:
            scale = self.config.maximum_spectral_radius / spectral_radius
            left *= scale
            compressed = left @ right
            spectral_radius = float(
                np.max(np.abs(np.linalg.eigvals(compressed)))
            )
        exogenous = coefficients[state_width:]
        residual = target - state @ compressed - exogenous_design @ exogenous
        self._graph = windows.graph
        self._state_shape = (entity_count, feature_count)
        self._control_count = control_count
        self._action_shape = (entity_count, action_count)
        self._left = left
        self._right = right
        self._exogenous = exogenous
        self._residual_variance = np.maximum(
            np.mean(np.square(residual), axis=0),
            self.config.variance_floor,
        ).reshape(entity_count, feature_count)
        self._spectral_radius = spectral_radius
        return self

    def one_step(
        self,
        current: NDArray[np.float64],
        controls: NDArray[np.float64],
        actions: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Predict one step for fitted-shape normalized arrays."""

        (
            _,
            state_shape,
            _,
            action_shape,
            left,
            right,
            exogenous,
            _,
        ) = self._fitted_values()
        sample_count = len(current)
        state = current.reshape(sample_count, -1)
        action = actions.reshape(
            sample_count, action_shape[0] * action_shape[1]
        )
        exogenous_design = np.concatenate(
            (
                controls,
                action,
                np.ones((sample_count, 1), dtype=np.float64),
            ),
            axis=1,
        )
        prediction = np.asarray(
            state @ left @ right + exogenous_design @ exogenous,
            dtype=np.float64,
        )
        return prediction.reshape(sample_count, *state_shape)

    def rollout(
        self,
        histories: NDArray[Any],
        future_controls: NDArray[Any],
        future_actions: NDArray[Any],
        graph: DeclaredTelemetryGraph,
    ) -> TrajectoryDistribution:
        (
            fitted_graph,
            state_shape,
            control_count,
            action_shape,
            _,
            _,
            _,
            residual_variance,
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
        current = history[:, -1]
        means = np.empty(
            (len(history), controls.shape[1], *state_shape),
            dtype=np.float64,
        )
        for step in range(controls.shape[1]):
            current = self.one_step(
                current, controls[:, step], actions[:, step]
            )
            means[:, step] = current
        variances = np.broadcast_to(
            residual_variance, means.shape
        ).copy()
        return TrajectoryDistribution(mean=means, variance=variances)

    @property
    def spectral_radius(self) -> float:
        self._fitted_values()
        return self._spectral_radius

    def validate_rollout_inputs(
        self,
        histories: NDArray[Any],
        future_controls: NDArray[Any],
        future_actions: NDArray[Any],
        graph: DeclaredTelemetryGraph,
    ) -> None:
        """Validate inputs against the fitted global schema."""

        values = self._fitted_values()
        validate_edge_rollout(
            np.asarray(histories, dtype=np.float64),
            np.asarray(future_controls, dtype=np.float64),
            np.asarray(future_actions, dtype=np.float64),
            graph,
            values[0],
            values[1],
            values[2],
            values[3],
        )

    @property
    def parameter_count(self) -> int:
        values = self._fitted_values()
        return int(
            values[4].size
            + values[5].size
            + values[6].size
            + values[7].size
        )

    def to_dict(self) -> Mapping[str, Any]:
        (
            graph,
            state_shape,
            control_count,
            action_shape,
            left,
            right,
            exogenous,
            residual_variance,
        ) = self._fitted_values()
        return {
            "schema_version": 1,
            "kind": self.kind,
            "config": {
                "rank": self.config.rank,
                "maximum_spectral_radius": (
                    self.config.maximum_spectral_radius
                ),
                "ridge": self.config.ridge,
                "variance_floor": self.config.variance_floor,
            },
            "graph": graph.to_dict(),
            "state_shape": list(state_shape),
            "control_count": control_count,
            "action_shape": list(action_shape),
            "spectral_radius": self.spectral_radius,
            "parameter_count": self.parameter_count,
            "state": {
                "left": left.tolist(),
                "right": right.tolist(),
                "exogenous": exogenous.tolist(),
                "residual_variance": residual_variance.tolist(),
            },
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "ContractiveLowRankDynamics":
        """Restore a fitted low-rank model artifact."""

        if (
            payload.get("schema_version") != 1
            or payload.get("kind") != cls.kind
        ):
            raise ValueError("unsupported low-rank model artifact")
        config_payload = payload.get("config")
        graph_payload = payload.get("graph")
        state_payload = payload.get("state")
        if (
            not isinstance(config_payload, Mapping)
            or not isinstance(graph_payload, Mapping)
            or not isinstance(state_payload, Mapping)
        ):
            raise ValueError("low-rank model artifact is malformed")
        model = cls(
            LowRankConfig(
                rank=int(config_payload["rank"]),
                maximum_spectral_radius=float(
                    config_payload["maximum_spectral_radius"]
                ),
                ridge=float(config_payload["ridge"]),
                variance_floor=float(
                    config_payload["variance_floor"]
                ),
            )
        )
        model._graph = DeclaredTelemetryGraph.from_dict(
            dict(graph_payload)
        )
        state_shape = tuple(int(value) for value in payload["state_shape"])
        action_shape = tuple(
            int(value) for value in payload["action_shape"]
        )
        if len(state_shape) != 2 or len(action_shape) != 2:
            raise ValueError("low-rank model artifact shape is malformed")
        model._state_shape = (state_shape[0], state_shape[1])
        model._control_count = int(payload["control_count"])
        model._action_shape = (action_shape[0], action_shape[1])
        model._left = np.asarray(
            state_payload["left"], dtype=np.float64
        )
        model._right = np.asarray(
            state_payload["right"], dtype=np.float64
        )
        model._exogenous = np.asarray(
            state_payload["exogenous"], dtype=np.float64
        )
        model._residual_variance = np.asarray(
            state_payload["residual_variance"], dtype=np.float64
        )
        model._spectral_radius = float(payload["spectral_radius"])
        model._fitted_values()
        return model

    def _fitted_values(
        self,
    ) -> Tuple[
        DeclaredTelemetryGraph,
        Tuple[int, int],
        int,
        Tuple[int, int],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        if (
            self._graph is None
            or self._state_shape is None
            or self._action_shape is None
            or self._left is None
            or self._right is None
            or self._exogenous is None
            or self._residual_variance is None
        ):
            raise ValueError("low-rank model is not fitted")
        return (
            self._graph,
            self._state_shape,
            self._control_count,
            self._action_shape,
            self._left,
            self._right,
            self._exogenous,
            self._residual_variance,
        )


@dataclass(frozen=True)
class GraphResidualConfig:
    """Bounded one-hop residual choices."""

    global_config: LowRankConfig = LowRankConfig()
    ridge: float = 1e-2
    residual_gain: float = 0.1
    correction_clip: float = 0.5
    variance_floor: float = 1e-4

    def __post_init__(self) -> None:
        if (
            self.ridge <= 0.0
            or not 0.0 < self.residual_gain <= 1.0
            or self.correction_clip <= 0.0
            or self.variance_floor <= 0.0
        ):
            raise ValueError("graph residual configuration is invalid")


class BoundedGraphResidualDynamics:
    """Contractive global transition plus clipped local corrections."""

    kind = "bounded_graph_residual_action_dynamics"

    def __init__(
        self, config: GraphResidualConfig = GraphResidualConfig()
    ) -> None:
        self.config = config
        self.global_model = ContractiveLowRankDynamics(
            config.global_config
        )
        self._coefficients: Optional[
            Tuple[NDArray[np.float64], ...]
        ] = None
        self._neighbor_positions: Optional[Tuple[Tuple[int, ...], ...]] = None
        self._residual_variance: Optional[NDArray[np.float64]] = None

    def fit(
        self, windows: ActionConditionedWindows
    ) -> "BoundedGraphResidualDynamics":
        self.global_model.fit(windows)
        current = np.asarray(
            windows.histories[:, -1], dtype=np.float64
        )
        controls = np.asarray(
            windows.future_controls[:, 0], dtype=np.float64
        )
        actions = np.asarray(
            windows.future_actions[:, 0], dtype=np.float64
        )
        target = np.asarray(
            windows.future_states[:, 0], dtype=np.float64
        )
        global_prediction = self.global_model.one_step(
            current, controls, actions
        )
        residual_target = target - global_prediction
        entity_positions = {
            entity_id: position
            for position, entity_id in enumerate(windows.entity_names)
        }
        neighbor_positions = tuple(
            tuple(
                entity_positions[neighbor]
                for neighbor in windows.graph.neighboring_entity_ids(
                    entity_id
                )
            )
            for entity_id in windows.entity_names
        )
        coefficients = []
        residual_predictions = []
        sample_count = len(current)
        for entity_position, neighbors in enumerate(neighbor_positions):
            local_state = current[:, neighbors].reshape(sample_count, -1)
            local_action = actions[:, entity_position]
            design = np.concatenate(
                (
                    local_state,
                    controls,
                    local_action,
                    np.ones((sample_count, 1), dtype=np.float64),
                ),
                axis=1,
            )
            penalty = np.eye(design.shape[1], dtype=np.float64)
            penalty[-1, -1] = 0.0
            coefficient = np.linalg.solve(
                design.T @ design + self.config.ridge * penalty,
                design.T @ residual_target[:, entity_position],
            )
            raw = design @ coefficient
            correction = self._bound(raw)
            coefficients.append(coefficient)
            residual_predictions.append(correction)
        correction_tensor = np.stack(residual_predictions, axis=1)
        residual = target - global_prediction - correction_tensor
        self._coefficients = tuple(coefficients)
        self._neighbor_positions = neighbor_positions
        self._residual_variance = np.maximum(
            np.mean(np.square(residual), axis=0),
            self.config.variance_floor,
        )
        return self

    def rollout(
        self,
        histories: NDArray[Any],
        future_controls: NDArray[Any],
        future_actions: NDArray[Any],
        graph: DeclaredTelemetryGraph,
    ) -> TrajectoryDistribution:
        coefficients, neighbors, residual_variance = self._fitted_values()
        history = np.asarray(histories, dtype=np.float64)
        controls = np.asarray(future_controls, dtype=np.float64)
        actions = np.asarray(future_actions, dtype=np.float64)
        self.global_model.validate_rollout_inputs(
            history, controls, actions, graph
        )
        current = history[:, -1]
        means = np.empty(
            (
                len(history),
                controls.shape[1],
                *current.shape[1:],
            ),
            dtype=np.float64,
        )
        for step in range(controls.shape[1]):
            global_next = self.global_model.one_step(
                current, controls[:, step], actions[:, step]
            )
            corrections = []
            for entity_position, local_neighbors in enumerate(neighbors):
                local_state = current[:, local_neighbors].reshape(
                    len(current), -1
                )
                design = np.concatenate(
                    (
                        local_state,
                        controls[:, step],
                        actions[:, step, entity_position],
                        np.ones((len(current), 1), dtype=np.float64),
                    ),
                    axis=1,
                )
                corrections.append(
                    self._bound(design @ coefficients[entity_position])
                )
            current = global_next + np.stack(corrections, axis=1)
            means[:, step] = current
        variances = np.broadcast_to(
            residual_variance, means.shape
        ).copy()
        return TrajectoryDistribution(mean=means, variance=variances)

    def _bound(
        self, raw: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return np.clip(
            self.config.residual_gain * raw,
            -self.config.correction_clip,
            self.config.correction_clip,
        )

    @property
    def parameter_count(self) -> int:
        coefficients, _, variance = self._fitted_values()
        return (
            self.global_model.parameter_count
            + sum(coefficient.size for coefficient in coefficients)
            + variance.size
        )

    def to_dict(self) -> Mapping[str, Any]:
        coefficients, neighbors, residual_variance = self._fitted_values()
        return {
            "schema_version": 1,
            "kind": self.kind,
            "config": {
                "ridge": self.config.ridge,
                "residual_gain": self.config.residual_gain,
                "correction_clip": self.config.correction_clip,
                "variance_floor": self.config.variance_floor,
            },
            "global_model": self.global_model.to_dict(),
            "parameter_count": self.parameter_count,
            "state": {
                "neighbor_positions": [
                    list(values) for values in neighbors
                ],
                "coefficients": [
                    coefficient.tolist()
                    for coefficient in coefficients
                ],
                "residual_variance": residual_variance.tolist(),
            },
        }

    def _fitted_values(
        self,
    ) -> Tuple[
        Tuple[NDArray[np.float64], ...],
        Tuple[Tuple[int, ...], ...],
        NDArray[np.float64],
    ]:
        if (
            self._coefficients is None
            or self._neighbor_positions is None
            or self._residual_variance is None
        ):
            raise ValueError("graph residual model is not fitted")
        return (
            self._coefficients,
            self._neighbor_positions,
            self._residual_variance,
        )


class MaskedInputDynamics:
    """Train and score one model with selected state inputs removed."""

    kind = "masked_input_dynamics"

    def __init__(
        self,
        model: EdgeDynamicsModel,
        masked_feature_positions: Tuple[int, ...],
    ) -> None:
        if not masked_feature_positions:
            raise ValueError("masked input model needs feature positions")
        self.model = model
        self.masked_feature_positions = masked_feature_positions

    def fit(
        self, windows: ActionConditionedWindows
    ) -> "MaskedInputDynamics":
        self.model.fit(_masked_windows(windows, self.masked_feature_positions))
        return self

    def rollout(
        self,
        histories: NDArray[Any],
        future_controls: NDArray[Any],
        future_actions: NDArray[Any],
        graph: DeclaredTelemetryGraph,
    ) -> TrajectoryDistribution:
        masked = np.array(histories, copy=True)
        masked[..., self.masked_feature_positions] = 0.0
        return self.model.rollout(
            masked, future_controls, future_actions, graph
        )

    @property
    def parameter_count(self) -> int:
        return self.model.parameter_count

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": 1,
            "kind": self.kind,
            "masked_feature_positions": list(
                self.masked_feature_positions
            ),
            "model": self.model.to_dict(),
            "parameter_count": self.parameter_count,
        }


def _masked_windows(
    windows: ActionConditionedWindows,
    positions: Tuple[int, ...],
) -> ActionConditionedWindows:
    histories = np.array(windows.histories, copy=True)
    histories[..., positions] = 0.0
    return ActionConditionedWindows(
        histories=histories,
        future_states=windows.future_states,
        future_controls=windows.future_controls,
        future_actions=windows.future_actions,
        trajectory_ids=windows.trajectory_ids,
        matched_pair_ids=windows.matched_pair_ids,
        transition_indices=windows.transition_indices,
        entity_names=windows.entity_names,
        state_feature_names=windows.state_feature_names,
        control_feature_names=windows.control_feature_names,
        action_feature_names=windows.action_feature_names,
        graph=windows.graph,
    )


def validate_edge_rollout(
    history: NDArray[np.float64],
    controls: NDArray[np.float64],
    actions: NDArray[np.float64],
    graph: DeclaredTelemetryGraph,
    fitted_graph: DeclaredTelemetryGraph,
    state_shape: Tuple[int, int],
    control_count: int,
    action_shape: Tuple[int, int],
    expected_horizon: Optional[int] = None,
) -> None:
    if (
        graph.to_dict() != fitted_graph.to_dict()
        or history.ndim != 4
        or controls.ndim != 3
        or actions.ndim != 4
        or history.shape[0] != controls.shape[0]
        or history.shape[0] != actions.shape[0]
        or controls.shape[1] != actions.shape[1]
        or (
            expected_horizon is not None
            and controls.shape[1] != expected_horizon
        )
        or history.shape[2:] != state_shape
        or controls.shape[2] != control_count
        or actions.shape[2:] != action_shape
        or not np.all(np.isfinite(history))
        or not np.all(np.isfinite(controls))
        or not np.all(np.isfinite(actions))
    ):
        raise ValueError("edge model rollout inputs are invalid")
