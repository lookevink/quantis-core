"""A small direct-horizon causal temporal convolution for edge studies."""

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from ..action_conditioned_dynamics import (
    ActionConditionedWindows,
    TrajectoryDistribution,
)
from ..graph_telemetry import DeclaredTelemetryGraph
from .models import validate_edge_rollout


@dataclass(frozen=True)
class TemporalConvConfig:
    """Frozen compact TCN training and capacity choices."""

    hidden_channels: int = 32
    kernel_size: int = 3
    epochs: int = 12
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    variance_floor: float = 1e-4
    seed: int = 23

    def __post_init__(self) -> None:
        if (
            self.hidden_channels < 4
            or self.kernel_size < 2
            or self.epochs < 1
            or self.batch_size < 1
            or self.learning_rate <= 0.0
            or self.weight_decay < 0.0
            or self.variance_floor <= 0.0
        ):
            raise ValueError("temporal convolution configuration is invalid")


class _DirectHorizonTcn(nn.Module):
    def __init__(
        self,
        *,
        state_width: int,
        condition_width: int,
        horizon: int,
        hidden_channels: int,
        kernel_size: int,
    ) -> None:
        super().__init__()
        self._padding_one = kernel_size - 1
        self._padding_two = 2 * (kernel_size - 1)
        self.depthwise = nn.Conv1d(
            state_width,
            state_width,
            kernel_size,
            padding=self._padding_one,
            groups=state_width,
        )
        self.pointwise = nn.Conv1d(
            state_width, hidden_channels, kernel_size=1
        )
        self.dilated = nn.Conv1d(
            hidden_channels,
            hidden_channels,
            kernel_size,
            padding=self._padding_two,
            dilation=2,
        )
        self.condition = nn.Linear(condition_width, hidden_channels)
        self.output = nn.Linear(
            hidden_channels, horizon * state_width
        )
        self.horizon = horizon
        self.state_width = state_width

    def forward(self, history: Tensor, condition: Tensor) -> Tensor:
        length = history.shape[2]
        encoded = torch.relu(self.depthwise(history)[..., :length])
        encoded = torch.relu(self.pointwise(encoded))
        encoded = torch.relu(self.dilated(encoded)[..., :length])
        hidden = torch.relu(
            encoded[..., -1] + self.condition(condition)
        )
        delta = self.output(hidden).reshape(
            len(history), self.horizon, self.state_width
        )
        persistence = history[..., -1].unsqueeze(1)
        return cast(Tensor, persistence + delta)


class DirectTemporalConvDynamics:
    """Depthwise causal Conv1D with a direct ten-step output head."""

    kind = "direct_temporal_convolution_action_dynamics"

    def __init__(
        self, config: TemporalConvConfig = TemporalConvConfig()
    ) -> None:
        self.config = config
        self._graph: Optional[DeclaredTelemetryGraph] = None
        self._state_shape: Optional[Tuple[int, int]] = None
        self._control_count = 0
        self._action_shape: Optional[Tuple[int, int]] = None
        self._horizon = 0
        self._module: Optional[_DirectHorizonTcn] = None
        self._residual_variance: Optional[NDArray[np.float64]] = None
        self._training_history: Tuple[Mapping[str, float], ...] = ()

    def fit(
        self,
        windows: ActionConditionedWindows,
        selection: Optional[ActionConditionedWindows] = None,
    ) -> "DirectTemporalConvDynamics":
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
        state_shape = (
            len(windows.entity_names),
            len(windows.state_feature_names),
        )
        action_shape = (
            len(windows.entity_names),
            len(windows.action_feature_names),
        )
        state_width = state_shape[0] * state_shape[1]
        condition_width = windows.future_controls.shape[1] * (
            len(windows.control_feature_names)
            + action_shape[0] * action_shape[1]
        )
        horizon = windows.future_states.shape[1]
        module = _DirectHorizonTcn(
            state_width=state_width,
            condition_width=condition_width,
            horizon=horizon,
            hidden_channels=self.config.hidden_channels,
            kernel_size=self.config.kernel_size,
        )
        optimizer = torch.optim.AdamW(
            module.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        loader = DataLoader(
            TensorDataset(*_tensors(windows)),
            batch_size=self.config.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.config.seed),
        )
        best_state: Optional[Dict[str, Tensor]] = None
        best_selection = float("inf")
        history = []
        for epoch in range(self.config.epochs):
            module.train()
            loss_sum = 0.0
            seen = 0
            for batch_history, batch_condition, batch_target in loader:
                optimizer.zero_grad()
                prediction = module(batch_history, batch_condition)
                loss = torch.mean(
                    torch.square(prediction - batch_target)
                )
                loss.backward()  # type: ignore[no-untyped-call]
                optimizer.step()
                loss_sum += float(loss.detach()) * len(batch_history)
                seen += len(batch_history)
            selection_loss = (
                _dataset_loss(module, selection, self.config.batch_size)
                if selection is not None
                else loss_sum / seen
            )
            history.append(
                {
                    "epoch": float(epoch + 1),
                    "fit_mse": loss_sum / seen,
                    "selection_mse": selection_loss,
                }
            )
            if selection_loss < best_selection:
                best_selection = selection_loss
                best_state = {
                    key: value.detach().clone()
                    for key, value in module.state_dict().items()
                }
        if best_state is None:
            raise ValueError("temporal convolution did not train")
        module.load_state_dict(best_state)
        fit_prediction = _predict(
            module, windows, self.config.batch_size
        )
        residual = (
            np.asarray(windows.future_states, dtype=np.float64)
            - fit_prediction
        )
        self._graph = windows.graph
        self._state_shape = state_shape
        self._control_count = len(windows.control_feature_names)
        self._action_shape = action_shape
        self._horizon = horizon
        self._module = module.eval()
        self._residual_variance = np.maximum(
            np.mean(np.square(residual), axis=(0, 1)),
            self.config.variance_floor,
        )
        self._training_history = tuple(history)
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
            horizon,
            module,
            residual_variance,
        ) = self._fitted_values()
        history = np.asarray(histories, dtype=np.float32)
        controls = np.asarray(future_controls, dtype=np.float32)
        actions = np.asarray(future_actions, dtype=np.float32)
        validate_edge_rollout(
            np.asarray(history, dtype=np.float64),
            np.asarray(controls, dtype=np.float64),
            np.asarray(actions, dtype=np.float64),
            graph,
            fitted_graph,
            state_shape,
            control_count,
            action_shape,
            expected_horizon=horizon,
        )
        history_tensor = torch.from_numpy(
            history.reshape(len(history), history.shape[1], -1)
        ).permute(0, 2, 1)
        condition_tensor = torch.from_numpy(
            np.concatenate(
                (
                    controls,
                    actions.reshape(
                        len(actions), horizon, -1
                    ),
                ),
                axis=2,
            ).reshape(len(actions), -1)
        )
        with torch.no_grad():
            mean = (
                module(history_tensor, condition_tensor)
                .numpy()
                .reshape(len(history), horizon, *state_shape)
                .astype(np.float64)
            )
        variance = np.broadcast_to(
            residual_variance, mean.shape
        ).copy()
        return TrajectoryDistribution(mean=mean, variance=variance)

    @property
    def parameter_count(self) -> int:
        module = self._fitted_values()[5]
        return sum(parameter.numel() for parameter in module.parameters())

    def to_dict(self) -> Mapping[str, Any]:
        (
            graph,
            state_shape,
            control_count,
            action_shape,
            horizon,
            module,
            residual_variance,
        ) = self._fitted_values()
        return {
            "schema_version": 1,
            "kind": self.kind,
            "config": {
                "hidden_channels": self.config.hidden_channels,
                "kernel_size": self.config.kernel_size,
                "epochs": self.config.epochs,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
                "variance_floor": self.config.variance_floor,
                "seed": self.config.seed,
            },
            "graph": graph.to_dict(),
            "state_shape": list(state_shape),
            "control_count": control_count,
            "action_shape": list(action_shape),
            "horizon": horizon,
            "parameter_count": self.parameter_count,
            "training_history": [
                dict(values) for values in self._training_history
            ],
            "state": {
                "weights": {
                    key: value.detach().numpy().tolist()
                    for key, value in module.state_dict().items()
                },
                "residual_variance": residual_variance.tolist(),
            },
        }

    def _fitted_values(
        self,
    ) -> Tuple[
        DeclaredTelemetryGraph,
        Tuple[int, int],
        int,
        Tuple[int, int],
        int,
        _DirectHorizonTcn,
        NDArray[np.float64],
    ]:
        if (
            self._graph is None
            or self._state_shape is None
            or self._action_shape is None
            or self._module is None
            or self._residual_variance is None
        ):
            raise ValueError("temporal convolution is not fitted")
        return (
            self._graph,
            self._state_shape,
            self._control_count,
            self._action_shape,
            self._horizon,
            self._module,
            self._residual_variance,
        )


def _tensors(
    windows: ActionConditionedWindows,
) -> Tuple[Tensor, Tensor, Tensor]:
    history = torch.from_numpy(
        np.asarray(windows.histories, dtype=np.float32).reshape(
            len(windows.histories), windows.histories.shape[1], -1
        )
    ).permute(0, 2, 1)
    condition = torch.from_numpy(
        np.concatenate(
            (
                windows.future_controls,
                windows.future_actions.reshape(
                    len(windows.future_actions),
                    windows.future_actions.shape[1],
                    -1,
                ),
            ),
            axis=2,
        )
        .reshape(len(windows.future_actions), -1)
        .astype(np.float32)
    )
    target = torch.from_numpy(
        np.asarray(windows.future_states, dtype=np.float32).reshape(
            len(windows.future_states),
            windows.future_states.shape[1],
            -1,
        )
    )
    return history, condition, target


def _dataset_loss(
    module: _DirectHorizonTcn,
    windows: ActionConditionedWindows,
    batch_size: int,
) -> float:
    prediction = _predict(module, windows, batch_size)
    return float(
        np.mean(
            np.square(
                prediction
                - np.asarray(windows.future_states, dtype=np.float64)
            )
        )
    )


def _predict(
    module: _DirectHorizonTcn,
    windows: ActionConditionedWindows,
    batch_size: int,
) -> NDArray[np.float64]:
    module.eval()
    history, condition, _ = _tensors(windows)
    predictions = []
    with torch.no_grad():
        for start in range(0, len(history), batch_size):
            predictions.append(
                module(
                    history[start : start + batch_size],
                    condition[start : start + batch_size],
                ).numpy()
            )
    values = np.asarray(
        np.concatenate(predictions, axis=0),
        dtype=np.float64,
    )
    return values.reshape(
        len(windows.histories),
        windows.future_states.shape[1],
        len(windows.entity_names),
        len(windows.state_feature_names),
    )
