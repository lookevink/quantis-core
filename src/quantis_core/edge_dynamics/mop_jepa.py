"""Hard-assigned mixture-of-predictors JEPA for edge telemetry tracers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ..action_conditioned_dynamics import (
    ActionConditionedWindows,
    MAX_MIXTURE_COMPONENT_COUNT,
    MixtureTrajectoryDistribution,
)


MOP_OBJECTIVES = ("mop_jepa", "dense_jepa", "supervised_hard_wta")


@dataclass(frozen=True)
class HardCosineAssignment:
    """Differentiable winning-head loss plus its discrete assignment."""

    distances: Any
    winner_indices: Any
    loss: Any


def hard_cosine_assignment(
    candidate_latents: Any, target_latents: Any
) -> HardCosineAssignment:
    """Choose one complete-trajectory head by mean cosine distance."""

    import torch

    if (
        candidate_latents.ndim != target_latents.ndim + 1
        or candidate_latents.shape[0] != target_latents.shape[0]
        or candidate_latents.shape[2:] != target_latents.shape[1:]
        or candidate_latents.shape[1] < 1
        or not bool(torch.all(torch.isfinite(candidate_latents)))
        or not bool(torch.all(torch.isfinite(target_latents)))
    ):
        raise ValueError("hard cosine assignment tensors are invalid")
    distances = torch.mean(
        1.0
        - torch.sum(
            candidate_latents * target_latents[:, None],
            dim=-1,
        ),
        dim=tuple(range(2, candidate_latents.ndim - 1)),
    )
    winners = torch.argmin(distances, dim=1)
    loss = torch.mean(
        distances[torch.arange(len(distances)), winners]
    )
    return HardCosineAssignment(
        distances=distances,
        winner_indices=winners,
        loss=loss,
    )


@dataclass(frozen=True)
class MopJepaConfig:
    """Frozen compact hard-assignment recipe."""

    objective: str = "mop_jepa"
    head_count: int = 8
    state_latent_width: int = 12
    context_width: int = 16
    predictor_width: int = 128
    epochs: int = 40
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    ema_decay: float = 0.996
    route_weight: float = 1.0
    balance_weight: float = 0.0
    target_reconstruction_weight: float = 0.10
    context_reconstruction_weight: float = 0.05
    variance_floor: float = 1e-4
    variance_pseudocount: float = 16.0
    seed: int = 19019

    def __post_init__(self) -> None:
        integers = (
            self.head_count,
            self.state_latent_width,
            self.context_width,
            self.predictor_width,
            self.epochs,
            self.batch_size,
        )
        if (
            self.objective not in MOP_OBJECTIVES
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
                for value in integers
            )
            or self.head_count > MAX_MIXTURE_COMPONENT_COUNT
            or (self.objective == "dense_jepa" and self.head_count != 1)
            or (
                self.objective != "dense_jepa"
                and self.head_count < 2
            )
            or self.learning_rate <= 0.0
            or self.weight_decay < 0.0
            or not 0.0 <= self.ema_decay < 1.0
            or self.route_weight < 0.0
            or self.balance_weight != 0.0
            or self.target_reconstruction_weight < 0.0
            or self.context_reconstruction_weight < 0.0
            or self.variance_floor <= 0.0
            or self.variance_pseudocount < 0.0
            or isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
        ):
            raise ValueError("MoP-JEPA configuration is invalid")


class MopJepaModel:
    """One-pass hard-assigned latent successor mixture."""

    kind = "edge_mop_jepa_v1"

    def __init__(self, config: MopJepaConfig) -> None:
        self.config = config
        self.training_metrics: Tuple[Mapping[str, float], ...] = ()
        self._network: Any = None
        self._shape: Optional[Tuple[int, int, int, int, int]] = None
        self._component_variance: Optional[NDArray[np.float64]] = None
        self._calibration_assignment_count: Optional[
            NDArray[np.int64]
        ] = None

    def fit(self, windows: ActionConditionedWindows) -> "MopJepaModel":
        """Fit encoders, predictor heads, and context-only router."""

        import torch

        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        self._shape = _window_shape(windows)
        self._network = _build_network(torch, self.config, self._shape)
        optimizer = torch.optim.AdamW(
            [
                parameter
                for parameter in self._network.parameters()
                if parameter.requires_grad
            ],
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        generator = np.random.default_rng(self.config.seed)
        rows = []
        for epoch in range(self.config.epochs):
            order = generator.permutation(len(windows.histories))
            totals = {
                "total": 0.0,
                "winner": 0.0,
                "route": 0.0,
                "balance": 0.0,
                "target_reconstruction": 0.0,
                "context_reconstruction": 0.0,
                "winner_effective_heads": 0.0,
            }
            batches = 0
            self._network.train()
            for start in range(0, len(order), self.config.batch_size):
                selected = order[start : start + self.config.batch_size]
                histories, controls, actions, future = _torch_batch(
                    torch, windows, selected
                )
                optimizer.zero_grad(set_to_none=True)
                output = self._network(
                    histories, controls, actions, future
                )
                losses = _training_losses(
                    torch, output, future, self.config
                )
                losses["total"].backward()
                optimizer.step()
                self._network.update_target(self.config.ema_decay)
                for name in totals:
                    totals[name] += float(losses[name].detach())
                batches += 1
            row = {
                name: value / float(batches)
                for name, value in totals.items()
            }
            row["epoch"] = float(epoch + 1)
            rows.append(row)
            if epoch in (0, 9, 19, 29, 39):
                print(
                    f"{self.config.objective}/K={self.config.head_count} "
                    f"epoch={epoch + 1} total={row['total']:.6f} "
                    f"winner={row['winner']:.6f} "
                    f"effective={row['winner_effective_heads']:.3f}",
                    flush=True,
                )
        self.training_metrics = tuple(rows)
        return self

    def calibrate(
        self, windows: ActionConditionedWindows
    ) -> "MopJepaModel":
        """Fit observable per-head variances on a disjoint role."""

        import torch

        network, shape = self._fitted()
        if _window_shape(windows) != shape:
            raise ValueError("MoP-JEPA calibration schema differs")
        residuals: list[list[NDArray[np.float32]]] = [
            [] for _ in range(self.config.head_count)
        ]
        all_selected: list[NDArray[np.float32]] = []
        network.eval()
        with torch.no_grad():
            for start in range(0, len(windows.histories), 128):
                selected = np.arange(
                    start, min(start + 128, len(windows.histories))
                )
                histories, controls, actions, future = _torch_batch(
                    torch, windows, selected
                )
                output = network(histories, controls, actions, future)
                winner = _winner_indices(
                    torch, output, future, self.config
                ).detach().numpy()
                means = output["component_mean"].detach().numpy()
                target = future.detach().numpy()
                chosen = means[np.arange(len(means)), winner] - target
                all_selected.append(np.square(chosen))
                for head in range(self.config.head_count):
                    rows = np.flatnonzero(winner == head)
                    if len(rows):
                        residuals[head].append(
                            np.square(means[rows, head] - target[rows])
                        )
        global_variance = np.mean(
            np.concatenate(all_selected, axis=0), axis=0
        )
        calibrated = []
        counts = []
        pseudo = self.config.variance_pseudocount
        for parts in residuals:
            count = int(sum(len(part) for part in parts))
            counts.append(count)
            if count:
                empirical = np.mean(
                    np.concatenate(parts, axis=0), axis=0
                )
                variance = (
                    count * empirical + pseudo * global_variance
                ) / (count + pseudo)
            else:
                variance = global_variance
            calibrated.append(
                np.maximum(variance, self.config.variance_floor)
            )
        self._component_variance = np.asarray(
            calibrated, dtype=np.float64
        )
        self._calibration_assignment_count = np.asarray(
            counts, dtype=np.int64
        )
        return self

    def rollout(
        self,
        histories: NDArray[Any],
        future_controls: NDArray[Any],
        future_actions: NDArray[Any],
    ) -> MixtureTrajectoryDistribution:
        """Emit every candidate and router probability without a future."""

        import torch

        network, shape = self._fitted()
        if self._component_variance is None:
            raise ValueError("MoP-JEPA model is not calibrated")
        values = np.asarray(histories, dtype=np.float32)
        controls = np.asarray(future_controls, dtype=np.float32)
        actions = np.asarray(future_actions, dtype=np.float32)
        _validate_inference(values, controls, actions, shape)
        means = []
        weights = []
        network.eval()
        with torch.no_grad():
            for start in range(0, len(values), self.config.batch_size):
                output = network.predict(
                    torch.as_tensor(values[start : start + self.config.batch_size]),
                    torch.as_tensor(controls[start : start + self.config.batch_size]),
                    torch.as_tensor(actions[start : start + self.config.batch_size]),
                )
                means.append(output["component_mean"].detach().numpy())
                weights.append(output["weight"].detach().numpy())
        component_mean = np.concatenate(means).astype(np.float64)
        weight = np.concatenate(weights).astype(np.float64)
        variance = np.broadcast_to(
            self._component_variance[None],
            component_mean.shape,
        ).copy()
        return MixtureTrajectoryDistribution(
            component_mean=component_mean,
            component_variance=variance,
            weight=weight,
        )

    @property
    def training_parameter_count(self) -> int:
        network, _ = self._fitted()
        return int(
            sum(
                parameter.numel()
                for name, parameter in network.named_parameters()
                if not name.startswith("target_encoder.")
            )
        )

    @property
    def calibration_assignment_count(self) -> NDArray[np.int64]:
        if self._calibration_assignment_count is None:
            raise ValueError("MoP-JEPA model is not calibrated")
        return self._calibration_assignment_count.copy()

    def save(self, directory: Path, name: str) -> int:
        """Store compact metadata and a compressed numeric sidecar."""

        network, shape = self._fitted()
        if (
            self._component_variance is None
            or self._calibration_assignment_count is None
        ):
            raise ValueError("cannot serialize uncalibrated MoP-JEPA")
        directory.mkdir(parents=True, exist_ok=True)
        metadata = directory / f"{name}.json"
        sidecar = directory / f"{name}.npz"
        metadata.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": self.kind,
                    "config": asdict(self.config),
                    "shape": list(shape),
                    "training_metrics": [
                        dict(row) for row in self.training_metrics
                    ],
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
        state = {
            f"state__{key}": value.detach().numpy()
            for key, value in network.state_dict().items()
        }
        np.savez_compressed(
            sidecar,
            **state,
            component_variance=self._component_variance,
            calibration_assignment_count=(
                self._calibration_assignment_count
            ),
        )
        return metadata.stat().st_size + sidecar.stat().st_size

    @classmethod
    def load(cls, directory: Path, name: str) -> "MopJepaModel":
        """Restore a fitted and calibrated model."""

        import torch

        metadata = json.loads((directory / f"{name}.json").read_text())
        if (
            metadata.get("schema_version") != 1
            or metadata.get("kind") != cls.kind
        ):
            raise ValueError("unsupported MoP-JEPA artifact")
        model = cls(MopJepaConfig(**metadata["config"]))
        raw_shape = tuple(int(value) for value in metadata["shape"])
        if len(raw_shape) != 5:
            raise ValueError("MoP-JEPA artifact shape is invalid")
        model._shape = (
            raw_shape[0],
            raw_shape[1],
            raw_shape[2],
            raw_shape[3],
            raw_shape[4],
        )
        model._network = _build_network(torch, model.config, model._shape)
        with np.load(
            directory / f"{name}.npz", allow_pickle=False
        ) as stored:
            state = {
                key: torch.as_tensor(stored[f"state__{key}"])
                for key in model._network.state_dict()
            }
            model._component_variance = np.asarray(
                stored["component_variance"], dtype=np.float64
            )
            model._calibration_assignment_count = np.asarray(
                stored["calibration_assignment_count"], dtype=np.int64
            )
        model._network.load_state_dict(state)
        model.training_metrics = tuple(metadata["training_metrics"])
        return model

    def _fitted(self) -> Tuple[Any, Tuple[int, int, int, int, int]]:
        if self._network is None or self._shape is None:
            raise ValueError("MoP-JEPA model is not fitted")
        return self._network, self._shape


@dataclass(frozen=True)
class ContextFreeCodebookConfig:
    """Frozen input-agnostic successor codebook."""

    component_count: int = 8
    iterations: int = 20
    variance_floor: float = 1e-4
    variance_pseudocount: float = 16.0
    def __post_init__(self) -> None:
        if (
            self.component_count < 2
            or self.component_count > MAX_MIXTURE_COMPONENT_COUNT
            or self.iterations < 1
            or self.variance_floor <= 0.0
            or self.variance_pseudocount < 0.0
        ):
            raise ValueError("context-free codebook config is invalid")


class ContextFreeTrajectoryCodebook:
    """Deterministic fitting-role future codebook."""

    kind = "edge_context_free_trajectory_codebook_v1"

    def __init__(self, config: ContextFreeCodebookConfig) -> None:
        self.config = config
        self._shape: Optional[Tuple[int, int, int]] = None
        self._centers: Optional[NDArray[np.float64]] = None
        self._variance: Optional[NDArray[np.float64]] = None
        self._weight: Optional[NDArray[np.float64]] = None

    def fit(
        self, windows: ActionConditionedWindows
    ) -> "ContextFreeTrajectoryCodebook":
        values = np.asarray(windows.future_states, dtype=np.float32)
        flat = values.reshape(len(values), -1)
        if len(flat) < self.config.component_count:
            raise ValueError("too few futures for codebook")
        mean = np.mean(flat, axis=0, dtype=np.float64).astype(np.float32)
        centers = [flat[int(np.argmax(np.sum(np.square(flat - mean), axis=1)))]]
        closest = np.sum(np.square(flat - centers[0]), axis=1)
        for _ in range(1, self.config.component_count):
            index = int(np.argmax(closest))
            centers.append(flat[index])
            distance = np.sum(np.square(flat - flat[index]), axis=1)
            closest = np.minimum(closest, distance)
        matrix = np.stack(centers)
        assignment = np.zeros(len(flat), dtype=np.int64)
        for _ in range(self.config.iterations):
            distance = _squared_distance_matrix(flat, matrix)
            assignment = np.argmin(distance, axis=1)
            updated = matrix.copy()
            for component in range(self.config.component_count):
                selected = flat[assignment == component]
                if len(selected):
                    updated[component] = np.mean(selected, axis=0)
            if np.array_equal(updated, matrix):
                break
            matrix = updated
        assignment = np.argmin(
            _squared_distance_matrix(flat, matrix), axis=1
        )
        counts = np.bincount(
            assignment, minlength=self.config.component_count
        ).astype(np.float64)
        weights = np.maximum(counts, 1.0)
        self._shape = (
            int(values.shape[1]),
            int(values.shape[2]),
            int(values.shape[3]),
        )
        self._centers = matrix.reshape(
            (self.config.component_count,) + self._shape
        ).astype(np.float64)
        self._weight = weights / np.sum(weights)
        return self

    def calibrate(
        self, windows: ActionConditionedWindows
    ) -> "ContextFreeTrajectoryCodebook":
        centers, _, shape = self._fitted()
        values = np.asarray(windows.future_states, dtype=np.float64)
        if tuple(values.shape[1:]) != shape:
            raise ValueError("codebook calibration schema differs")
        squared = np.square(values[:, None] - centers[None])
        assignment = np.argmin(
            np.mean(squared, axis=(2, 3, 4)), axis=1
        )
        selected = squared[np.arange(len(values)), assignment]
        global_variance = np.mean(selected, axis=0)
        pseudo = self.config.variance_pseudocount
        variances = []
        for component in range(self.config.component_count):
            rows = np.flatnonzero(assignment == component)
            if len(rows):
                empirical = np.mean(squared[rows, component], axis=0)
                variance = (
                    len(rows) * empirical + pseudo * global_variance
                ) / (len(rows) + pseudo)
            else:
                variance = global_variance
            variances.append(
                np.maximum(variance, self.config.variance_floor)
            )
        self._variance = np.asarray(variances, dtype=np.float64)
        return self

    def rollout(
        self,
        histories: NDArray[Any],
        future_controls: NDArray[Any],
        future_actions: NDArray[Any],
    ) -> MixtureTrajectoryDistribution:
        centers, weight, _ = self._fitted()
        if self._variance is None:
            raise ValueError("codebook is not calibrated")
        count = len(np.asarray(histories))
        if (
            len(np.asarray(future_controls)) != count
            or len(np.asarray(future_actions)) != count
        ):
            raise ValueError("codebook input batch differs")
        return MixtureTrajectoryDistribution(
            component_mean=np.broadcast_to(
                centers[None], (count,) + centers.shape
            ).copy(),
            component_variance=np.broadcast_to(
                self._variance[None], (count,) + self._variance.shape
            ).copy(),
            weight=np.broadcast_to(
                weight[None], (count, len(weight))
            ).copy(),
        )

    @property
    def parameter_count(self) -> int:
        centers, weight, _ = self._fitted()
        return int(centers.size + weight.size)

    def save(self, directory: Path, name: str) -> int:
        centers, weight, shape = self._fitted()
        if self._variance is None:
            raise ValueError("cannot serialize uncalibrated codebook")
        directory.mkdir(parents=True, exist_ok=True)
        metadata = directory / f"{name}.json"
        sidecar = directory / f"{name}.npz"
        metadata.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": self.kind,
                    "config": asdict(self.config),
                    "shape": list(shape),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        np.savez_compressed(
            sidecar,
            centers=centers,
            variance=self._variance,
            weight=weight,
        )
        return metadata.stat().st_size + sidecar.stat().st_size

    @classmethod
    def load(
        cls, directory: Path, name: str
    ) -> "ContextFreeTrajectoryCodebook":
        metadata = json.loads((directory / f"{name}.json").read_text())
        if (
            metadata.get("schema_version") != 1
            or metadata.get("kind") != cls.kind
        ):
            raise ValueError("unsupported codebook artifact")
        model = cls(ContextFreeCodebookConfig(**metadata["config"]))
        raw_shape = tuple(int(value) for value in metadata["shape"])
        if len(raw_shape) != 3:
            raise ValueError("codebook artifact shape is invalid")
        model._shape = (raw_shape[0], raw_shape[1], raw_shape[2])
        with np.load(
            directory / f"{name}.npz", allow_pickle=False
        ) as stored:
            model._centers = np.asarray(
                stored["centers"], dtype=np.float64
            )
            model._variance = np.asarray(
                stored["variance"], dtype=np.float64
            )
            model._weight = np.asarray(
                stored["weight"], dtype=np.float64
            )
        return model

    def _fitted(
        self,
    ) -> Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        Tuple[int, int, int],
    ]:
        if (
            self._centers is None
            or self._weight is None
            or self._shape is None
        ):
            raise ValueError("context-free codebook is not fitted")
        return self._centers, self._weight, self._shape


def _build_network(
    torch: Any,
    config: MopJepaConfig,
    shape: Tuple[int, int, int, int, int],
) -> Any:
    entity_count, feature_count, horizon, control_count, action_count = shape
    latent_width = config.state_latent_width
    exogenous_width = horizon * (
        control_count + entity_count * action_count
    )
    predictor_input = entity_count * config.context_width + exogenous_width
    latent_output = (
        config.head_count * horizon * entity_count * latent_width
    )

    class Network(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.state_encoder = torch.nn.Linear(
                feature_count, latent_width
            )
            self.target_encoder = torch.nn.Linear(
                feature_count, latent_width
            )
            self.target_encoder.load_state_dict(
                self.state_encoder.state_dict()
            )
            for parameter in self.target_encoder.parameters():
                parameter.requires_grad_(False)
            self.context_projector = torch.nn.Linear(
                3 * latent_width, config.context_width
            )
            self.predictor_hidden = torch.nn.Linear(
                predictor_input, config.predictor_width
            )
            self.predictor_output = torch.nn.Linear(
                config.predictor_width, latent_output
            )
            self.router = torch.nn.Linear(
                config.predictor_width, config.head_count
            )
            self.decoder = torch.nn.Linear(
                latent_width, feature_count
            )

        @staticmethod
        def unit(values: Any) -> Any:
            return torch.nn.functional.normalize(
                values, p=2.0, dim=-1, eps=1e-8
            )

        def encode_states(self, values: Any) -> Any:
            return self.unit(
                torch.nn.functional.gelu(self.state_encoder(values))
            )

        def encode_context(self, histories: Any) -> Any:
            encoded = self.encode_states(histories)
            summary = torch.cat(
                (
                    encoded[:, -1],
                    torch.mean(encoded, dim=1),
                    encoded[:, -1] - encoded[:, 0],
                ),
                dim=-1,
            )
            return torch.nn.functional.gelu(
                self.context_projector(summary)
            )

        def predict(
            self, histories: Any, controls: Any, actions: Any
        ) -> Mapping[str, Any]:
            context = self.encode_context(histories)
            combined = torch.cat(
                (
                    context.flatten(1),
                    controls.flatten(1),
                    actions.flatten(1),
                ),
                dim=1,
            )
            hidden = torch.nn.functional.gelu(
                self.predictor_hidden(combined)
            )
            latent = self.predictor_output(hidden).reshape(
                len(histories),
                config.head_count,
                horizon,
                entity_count,
                latent_width,
            )
            latent = self.unit(latent)
            weight = torch.softmax(self.router(hidden), dim=1)
            weight = torch.clamp(weight, min=1e-9)
            weight = weight / torch.sum(weight, dim=1, keepdim=True)
            return {
                "context": context,
                "component_latent": latent,
                "component_mean": self.decoder(latent),
                "weight": weight,
            }

        def forward(
            self,
            histories: Any,
            controls: Any,
            actions: Any,
            future: Any,
        ) -> Mapping[str, Any]:
            output = dict(self.predict(histories, controls, actions))
            with torch.no_grad():
                target = self.unit(
                    torch.nn.functional.gelu(
                        self.target_encoder(future)
                    )
                )
            output["target_latent"] = target
            output["target_reconstruction"] = self.decoder(target)
            output["context_reconstruction"] = self.decoder(
                self.encode_states(histories)[:, -1]
            )
            output["current_state"] = histories[:, -1]
            return output

        def update_target(self, decay: float) -> None:
            with torch.no_grad():
                for online, target in zip(
                    self.state_encoder.parameters(),
                    self.target_encoder.parameters(),
                ):
                    target.mul_(decay).add_(
                        online, alpha=1.0 - decay
                    )

    return Network()


def _winner_indices(
    torch: Any,
    output: Mapping[str, Any],
    future: Any,
    config: MopJepaConfig,
) -> Any:
    return torch.argmin(
        _head_distance(torch, output, future, config), dim=1
    )


def _training_losses(
    torch: Any,
    output: Mapping[str, Any],
    future: Any,
    config: MopJepaConfig,
) -> Mapping[str, Any]:
    distance = _head_distance(torch, output, future, config)
    winner = torch.argmin(distance, dim=1)
    winner_loss = torch.mean(
        distance[torch.arange(len(distance)), winner]
    )
    route = torch.nn.functional.nll_loss(
        torch.log(torch.clamp(output["weight"], min=1e-9)),
        winner,
    )
    usage = torch.mean(
        torch.nn.functional.one_hot(
            winner, num_classes=config.head_count
        ).to(torch.float32),
        dim=0,
    )
    positive = usage > 0
    balance = torch.sum(
        usage[positive]
        * torch.log(config.head_count * usage[positive])
    )
    entropy = -torch.sum(
        usage[positive] * torch.log(usage[positive])
    )
    effective = torch.exp(entropy)
    target_reconstruction = torch.mean(
        torch.square(output["target_reconstruction"] - future)
    )
    context_reconstruction = torch.mean(
        torch.square(
            output["context_reconstruction"]
            - output["current_state"]
        )
    )
    total = (
        winner_loss
        + config.route_weight * route
        + config.balance_weight * balance
        + config.target_reconstruction_weight * target_reconstruction
        + config.context_reconstruction_weight * context_reconstruction
    )
    return {
        "total": total,
        "winner": winner_loss,
        "route": route,
        "balance": balance,
        "target_reconstruction": target_reconstruction,
        "context_reconstruction": context_reconstruction,
        "winner_effective_heads": effective,
    }


def _head_distance(
    torch: Any,
    output: Mapping[str, Any],
    future: Any,
    config: MopJepaConfig,
) -> Any:
    if config.objective == "supervised_hard_wta":
        return torch.mean(
            torch.square(output["component_mean"] - future[:, None]),
            dim=(2, 3, 4),
        )
    return hard_cosine_assignment(
        output["component_latent"], output["target_latent"]
    ).distances


def _torch_batch(
    torch: Any,
    windows: ActionConditionedWindows,
    indices: NDArray[np.int64],
) -> Tuple[Any, Any, Any, Any]:
    return (
        torch.as_tensor(windows.histories[indices], dtype=torch.float32),
        torch.as_tensor(
            windows.future_controls[indices], dtype=torch.float32
        ),
        torch.as_tensor(
            windows.future_actions[indices], dtype=torch.float32
        ),
        torch.as_tensor(
            windows.future_states[indices], dtype=torch.float32
        ),
    )


def _window_shape(
    windows: ActionConditionedWindows,
) -> Tuple[int, int, int, int, int]:
    return (
        windows.histories.shape[2],
        windows.histories.shape[3],
        windows.future_states.shape[1],
        windows.future_controls.shape[2],
        windows.future_actions.shape[3],
    )


def _validate_inference(
    histories: NDArray[np.float32],
    controls: NDArray[np.float32],
    actions: NDArray[np.float32],
    shape: Tuple[int, int, int, int, int],
) -> None:
    entity_count, feature_count, horizon, control_count, action_count = shape
    if (
        histories.ndim != 4
        or histories.shape[2:] != (entity_count, feature_count)
        or controls.shape != (len(histories), horizon, control_count)
        or actions.shape
        != (len(histories), horizon, entity_count, action_count)
        or not np.all(np.isfinite(histories))
        or not np.all(np.isfinite(controls))
        or not np.all(np.isfinite(actions))
    ):
        raise ValueError("MoP-JEPA inference inputs are invalid")


def _squared_distance_matrix(
    values: NDArray[np.float32], centers: NDArray[np.float32]
) -> NDArray[np.float32]:
    return np.asarray(
        np.maximum(
            np.sum(np.square(values), axis=1, keepdims=True)
            + np.sum(np.square(centers), axis=1)[None]
            - 2.0 * values @ centers.T,
            0.0,
        ),
        dtype=np.float32,
    )
