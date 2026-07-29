"""Edge LeWorldModel and bounded latent-geometry variants."""

import copy
import importlib
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ..action_conditioned_dynamics import ActionConditionedWindows
from ..graph_telemetry import DeclaredTelemetryGraph
from .action_conditioned_jepa import (
    sketched_isotropic_gaussian_regularization,
)
from .complete_lejepa import PairBlockedAnchorSchedule


LEWORLD_GEOMETRY_OBJECTIVES = (
    "lewm_ambient",
    "sub_jepa",
    "rectified_lp",
    "ker_jepa",
    "sphere_jepa",
    "sphere_mmd",
    "prediction_only",
)
SPHERICAL_OBJECTIVES = ("sphere_jepa", "sphere_mmd")


@dataclass(frozen=True)
class LeWorldGeometryConfig:
    """Frozen controls for a parameter-matched geometry cell."""

    objective: str = "lewm_ambient"
    width: int = 32
    hidden_width: int = 64
    history_size: int = 3
    pretrain_steps: int = 800
    checkpoint_interval: int = 100
    learning_rate: float = 5e-5
    weight_decay: float = 1e-3
    regularizer_weight: float = 0.09
    sketch_dimension: int = 256
    knot_count: int = 17
    subspace_count: int = 8
    expected_pair_count: int = 40
    seed: int = 17017
    device: str = "cpu"

    def __post_init__(self) -> None:
        integers = (
            self.width,
            self.hidden_width,
            self.history_size,
            self.pretrain_steps,
            self.checkpoint_interval,
            self.sketch_dimension,
            self.knot_count,
            self.subspace_count,
            self.expected_pair_count,
        )
        if (
            self.objective not in LEWORLD_GEOMETRY_OBJECTIVES
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
                for value in integers
            )
            or self.width % self.subspace_count
            or self.history_size != 3
            or self.learning_rate <= 0.0
            or self.weight_decay < 0.0
            or self.regularizer_weight != 0.09
            or self.knot_count < 2
            or isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.device != "cpu"
        ):
            raise ValueError("LeWorldModel geometry configuration is invalid")


@dataclass(frozen=True)
class LeWorldEncodedTelemetry:
    """Current entity tokens plus causal temporal scene tokens."""

    tokens: NDArray[np.float64]
    scene_history: NDArray[np.float64]
    entity_ids: Tuple[str, ...]
    ownership_mask: NDArray[np.bool_]

    def __post_init__(self) -> None:
        if (
            self.tokens.ndim != 3
            or self.scene_history.ndim != 3
            or len(self.tokens) != len(self.scene_history)
            or self.tokens.shape[1] != len(self.entity_ids)
            or self.tokens.shape[2] != self.scene_history.shape[2]
            or self.ownership_mask.shape[0] != len(self.entity_ids)
            or not np.all(np.isfinite(self.tokens))
            or not np.all(np.isfinite(self.scene_history))
        ):
            raise ValueError("LeWorldModel geometry encoding is invalid")


def gaussian_prior_mmd(
    embeddings: Any, *, bandwidth_squared: Optional[float] = None
) -> Any:
    """Exact RBF MMD to N(0,I), up to finite-sample diagonal bias."""

    if embeddings.ndim < 2 or embeddings.shape[-2] < 2:
        raise ValueError("Gaussian-prior MMD embeddings are invalid")
    torch = _require_torch()
    dimension = embeddings.shape[-1]
    bandwidth = (
        float(dimension)
        if bandwidth_squared is None
        else float(bandwidth_squared)
    )
    if not bandwidth > 0.0:
        raise ValueError("Gaussian-prior MMD bandwidth is invalid")
    difference = embeddings.unsqueeze(-2) - embeddings.unsqueeze(-3)
    empirical = torch.exp(
        -difference.square().sum(dim=-1) / (2.0 * bandwidth)
    ).mean(dim=(-2, -1))
    cross_scale = (bandwidth / (bandwidth + 1.0)) ** (
        dimension / 2.0
    )
    cross = cross_scale * torch.exp(
        -embeddings.square().sum(dim=-1)
        / (2.0 * (bandwidth + 1.0))
    ).mean(dim=-1)
    prior = (bandwidth / (bandwidth + 2.0)) ** (dimension / 2.0)
    return (empirical - 2.0 * cross + prior).mean()


def sphere_heat_uniformity(
    embeddings: Any, *, temperature: Optional[float] = None
) -> Any:
    """Deterministic heat-kernel MMD term against sphere-uniform measure."""

    if embeddings.ndim < 2 or embeddings.shape[-2] < 2:
        raise ValueError("sphere heat embeddings are invalid")
    torch = _require_torch()
    dimension = embeddings.shape[-1]
    heat = 5.0 / float(dimension) if temperature is None else float(temperature)
    if not heat > 0.0:
        raise ValueError("sphere heat temperature is invalid")
    normalized = torch.nn.functional.normalize(
        embeddings, p=2, dim=-1, eps=1e-8
    )
    similarities = normalized @ normalized.transpose(-1, -2)
    count = similarities.shape[-1]
    off_diagonal = ~torch.eye(
        count, dtype=torch.bool, device=similarities.device
    )
    return torch.exp(
        (similarities[..., off_diagonal] - 1.0) / heat
    ).mean()


def sliced_wasserstein_distance(
    samples: Any, targets: Any, directions: Any
) -> Any:
    """Two-sample sliced squared 2-Wasserstein distance."""

    if (
        samples.ndim < 2
        or targets.shape != samples.shape
        or directions.ndim != 2
        or directions.shape[0] != samples.shape[-1]
    ):
        raise ValueError("sliced Wasserstein arrays are invalid")
    projected_samples = samples @ directions
    projected_targets = targets @ directions
    sorted_samples = projected_samples.sort(dim=-2).values
    sorted_targets = projected_targets.sort(dim=-2).values
    return (sorted_samples - sorted_targets).square().mean()


class LeWorldGeometryModel:
    """Restorable exact LeWorldModel core with one frozen geometry."""

    kind = "leworld_geometry_representation"
    schema_version = 1

    def __init__(
        self, config: LeWorldGeometryConfig = LeWorldGeometryConfig()
    ) -> None:
        self.config = config
        self._graph: Optional[DeclaredTelemetryGraph] = None
        self._feature_names: Tuple[str, ...] = ()
        self._control_names: Tuple[str, ...] = ()
        self._action_names: Tuple[str, ...] = ()
        self._ownership_mask: Optional[NDArray[np.bool_]] = None
        self._center: Optional[NDArray[np.float64]] = None
        self._scale: Optional[NDArray[np.float64]] = None
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

    def fit(
        self, windows: ActionConditionedWindows
    ) -> "LeWorldGeometryModel":
        """Fit checkpoint candidates with one anchor per matched pair."""

        if len(set(windows.matched_pair_ids)) != self.config.expected_pair_count:
            raise ValueError("LeWorldModel fitting pair count differs")
        _validate_windows(windows)
        torch = _require_torch()
        _seed_torch(torch, self.config.seed)
        ownership = _fit_owned_feature_mask(windows)
        full = np.concatenate((windows.histories, windows.future_states), axis=1)
        center, scale = _fit_normalizer(full, ownership)
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
            observed_entities=np.any(ownership, axis=1),
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
            values, conditions = _training_batch(
                windows, anchor.indices, ownership, center, scale
            )
            generator = torch.Generator(device="cpu").manual_seed(
                self.config.seed + 10_000 + step
            )
            optimizer.zero_grad(set_to_none=True)
            losses = _objective_loss(
                torch,
                network,
                torch.as_tensor(values, dtype=torch.float32),
                torch.as_tensor(conditions, dtype=torch.float32),
                config=self.config,
                generator=generator,
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
                    "regularizer": float(losses["regularizer"].detach()),
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                }
                if not np.all(np.isfinite(list(row.values()))):
                    raise RuntimeError(
                        "LeWorldModel geometry training became non-finite"
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
        self._center = center
        self._scale = scale
        self._condition_dimension = condition_dimension
        self._network = network
        self._checkpoints = tuple(checkpoints)
        self._training_metrics = tuple(metrics)
        return self

    def select(
        self, windows: ActionConditionedWindows
    ) -> "LeWorldGeometryModel":
        """Select this cell by its own selection-role objective."""

        (
            graph,
            features,
            controls,
            actions,
            ownership,
            center,
            scale,
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
            raise ValueError("LeWorldModel selection schema differs")
        _validate_windows(windows)
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
            weighted = {"total": 0.0, "prediction": 0.0, "regularizer": 0.0}
            with torch.no_grad():
                for local_step in range(evaluation_steps):
                    anchor = schedule.batch(local_step)
                    values, conditions = _training_batch(
                        windows, anchor.indices, ownership, center, scale
                    )
                    generator = torch.Generator(device="cpu").manual_seed(
                        self.config.seed + 20_000 + local_step
                    )
                    losses = _objective_loss(
                        torch,
                        network,
                        torch.as_tensor(values, dtype=torch.float32),
                        torch.as_tensor(conditions, dtype=torch.float32),
                        config=self.config,
                        generator=generator,
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
                raise RuntimeError(
                    "LeWorldModel geometry selection became non-finite"
                )
            rows.append(row)
            key = (row["total"], checkpoint_step)
            if best_key is None or key < best_key:
                best_key = key
                best_state = copy.deepcopy(state)
                best_step = checkpoint_step
        if best_state is None or best_step is None:
            raise RuntimeError("LeWorldModel selected no checkpoint")
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
    ) -> LeWorldEncodedTelemetry:
        """Encode histories without future/control/action tensors."""

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
            network,
        ) = self._selected_values()
        values = self._validate_histories(histories, graph)
        normalized = _normalize_states(values, ownership, center, scale)
        entity_parts = []
        scene_parts = []
        with torch.no_grad():
            for start in range(0, len(normalized), 256):
                entity, scene = network.encode(
                    torch.as_tensor(
                        normalized[start : start + 256],
                        dtype=torch.float32,
                    )
                )
                entity_parts.append(entity[:, -1].cpu().numpy())
                scene_parts.append(scene.cpu().numpy())
        return LeWorldEncodedTelemetry(
            tokens=np.asarray(
                np.concatenate(entity_parts), dtype=np.float64
            ),
            scene_history=np.asarray(
                np.concatenate(scene_parts), dtype=np.float64
            ),
            entity_ids=graph.entity_ids,
            ownership_mask=ownership.copy(),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe selected cell."""

        (
            graph,
            features,
            controls,
            actions,
            ownership,
            center,
            scale,
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
            "condition_dimension": condition_dimension,
            "state_dict": _state_dict_to_payload(network.state_dict()),
            "selected_step": self._selected_step,
            "training_metrics": [dict(row) for row in self._training_metrics],
            "selection_metrics": [dict(row) for row in self._selection_metrics],
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "LeWorldGeometryModel":
        """Restore and validate a selected geometry cell."""

        if (
            payload.get("schema_version") != cls.schema_version
            or payload.get("kind") != cls.kind
        ):
            raise ValueError("LeWorldModel geometry schema is invalid")
        config = LeWorldGeometryConfig(**dict(payload["config"]))
        graph = DeclaredTelemetryGraph.from_dict(dict(payload["graph"]))
        features = tuple(str(value) for value in payload["feature_names"])
        controls = tuple(str(value) for value in payload["control_names"])
        actions = tuple(str(value) for value in payload["action_names"])
        ownership = np.asarray(payload["ownership_mask"], dtype=np.bool_)
        center = np.asarray(payload["center"], dtype=np.float64)
        scale = np.asarray(payload["scale"], dtype=np.float64)
        condition_dimension = payload["condition_dimension"]
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
        ):
            raise ValueError("LeWorldModel fitted schema is invalid")
        torch = _require_torch()
        network = _build_network(
            torch,
            config=config,
            entity_count=len(graph.entities),
            feature_count=len(features),
            condition_dimension=condition_dimension,
            observed_entities=np.any(ownership, axis=1),
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
            raise ValueError("LeWorldModel selected step is invalid")
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
            raise ValueError("LeWorldModel public inputs are invalid")
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
            or self._network is None
        ):
            raise ValueError("LeWorldModel geometry cell is not fitted")
        return (
            self._graph,
            self._feature_names,
            self._control_names,
            self._action_names,
            self._ownership_mask,
            self._center,
            self._scale,
            self._condition_dimension,
            self._network,
        )

    def _selected_values(self) -> Tuple[Any, ...]:
        values = self._fitted_values()
        if self._selected_step is None:
            raise ValueError("LeWorldModel geometry cell is not selected")
        return values


def _build_network(
    torch: Any,
    *,
    config: LeWorldGeometryConfig,
    entity_count: int,
    feature_count: int,
    condition_dimension: int,
    observed_entities: NDArray[np.bool_],
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

        def forward(self, values: Any) -> Tuple[Any, Any]:
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
            entity = _geometry_transform(
                torch, self.output_fc(hidden), config.objective
            )
            scene = entity[..., observed_entities, :].mean(dim=-2)
            if config.objective in SPHERICAL_OBJECTIVES:
                scene = torch.nn.functional.normalize(
                    scene, p=2, dim=-1, eps=1e-8
                )
            return entity, scene

    class Network(nn.Module):  # type: ignore[misc,name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.encoder = Encoder()
            predictor_input = (
                config.history_size * config.width + condition_dimension
            )
            self.predictor = nn.Sequential(
                nn.Linear(predictor_input, config.hidden_width),
                nn.SiLU(),
                nn.Linear(config.hidden_width, config.hidden_width),
                nn.SiLU(),
                nn.Linear(config.hidden_width, config.width),
            )
            matrices = []
            subspace_width = config.width // config.subspace_count
            for _ in range(config.subspace_count):
                q, _ = torch.linalg.qr(
                    torch.randn(config.width, subspace_width),
                    mode="reduced",
                )
                matrices.append(q.T)
            self.register_buffer(
                "subspace_matrices", torch.stack(matrices)
            )

        def encode(self, values: Any) -> Tuple[Any, Any]:
            entity, scene = self.encoder(values)
            return entity, scene

        def predict(self, context: Any, condition: Any) -> Any:
            prediction = self.predictor(
                torch.cat((context.flatten(1), condition), dim=1)
            )
            return _geometry_transform(
                torch, prediction, config.objective
            )

    return Network()


def _geometry_transform(torch: Any, values: Any, objective: str) -> Any:
    if objective == "rectified_lp":
        return torch.nn.functional.relu(values)
    if objective in SPHERICAL_OBJECTIVES:
        return torch.nn.functional.normalize(
            values, p=2, dim=-1, eps=1e-8
        )
    return values


def _objective_loss(
    torch: Any,
    network: Any,
    full: Any,
    conditions: Any,
    *,
    config: LeWorldGeometryConfig,
    generator: Any,
) -> Mapping[str, Any]:
    _, scene = network.encode(full)
    contexts = []
    targets = []
    for offset in range(10):
        target_index = 20 + offset
        contexts.append(
            scene[
                :,
                target_index - config.history_size : target_index,
            ]
        )
        targets.append(scene[:, target_index])
    context = torch.stack(contexts, dim=1).flatten(0, 1)
    target = torch.stack(targets, dim=1).flatten(0, 1)
    prediction = network.predict(context, conditions.flatten(0, 1))
    prediction_loss = torch.nn.functional.mse_loss(prediction, target)
    regularizer = _geometry_regularizer(
        torch,
        scene,
        network.subspace_matrices,
        config=config,
        generator=generator,
    )
    return {
        "total": prediction_loss + config.regularizer_weight * regularizer,
        "prediction": prediction_loss,
        "regularizer": regularizer,
    }


def _geometry_regularizer(
    torch: Any,
    scene: Any,
    subspace_matrices: Any,
    *,
    config: LeWorldGeometryConfig,
    generator: Any,
) -> Any:
    time_first = scene.transpose(0, 1)
    if config.objective == "prediction_only":
        return scene.sum() * 0.0
    if config.objective == "lewm_ambient":
        return sketched_isotropic_gaussian_regularization(
            time_first,
            generator=generator,
            sketch_dimension=config.sketch_dimension,
            knot_count=config.knot_count,
        )
    if config.objective == "sub_jepa":
        projected = torch.einsum(
            "tbd,ked->ktbe", time_first, subspace_matrices
        )
        return _multi_subspace_sigreg(
            torch,
            projected,
            generator=generator,
            sketch_dimension=config.sketch_dimension,
            knot_count=config.knot_count,
        )
    if config.objective == "rectified_lp":
        targets = torch.nn.functional.relu(
            torch.randn(
                scene.shape,
                dtype=scene.dtype,
                device="cpu",
                generator=generator,
            ).to(scene.device)
        )
        directions = _random_directions(
            torch,
            config.width,
            config.sketch_dimension,
            generator=generator,
            dtype=scene.dtype,
            device=scene.device,
        )
        return sliced_wasserstein_distance(
            scene.transpose(0, 1), targets.transpose(0, 1), directions
        )
    if config.objective == "ker_jepa":
        return gaussian_prior_mmd(time_first)
    if config.objective == "sphere_jepa":
        targets = torch.randn(
            scene.shape,
            dtype=scene.dtype,
            device="cpu",
            generator=generator,
        ).to(scene.device)
        targets = torch.nn.functional.normalize(
            targets, p=2, dim=-1, eps=1e-8
        )
        directions = _random_directions(
            torch,
            config.width,
            config.sketch_dimension,
            generator=generator,
            dtype=scene.dtype,
            device=scene.device,
        )
        return sliced_wasserstein_distance(
            scene.transpose(0, 1), targets.transpose(0, 1), directions
        )
    if config.objective == "sphere_mmd":
        return sphere_heat_uniformity(time_first)
    raise ValueError("unknown LeWorldModel geometry objective")


def _multi_subspace_sigreg(
    torch: Any,
    embeddings: Any,
    *,
    generator: Any,
    sketch_dimension: int,
    knot_count: int,
) -> Any:
    subspace_count, _, sample_count, dimension = embeddings.shape
    directions = torch.randn(
        subspace_count,
        dimension,
        sketch_dimension,
        dtype=embeddings.dtype,
        device="cpu",
        generator=generator,
    ).to(embeddings.device)
    directions = directions / directions.norm(
        p=2, dim=1, keepdim=True
    )
    knots = torch.linspace(
        0.0,
        3.0,
        knot_count,
        dtype=embeddings.dtype,
        device=embeddings.device,
    )
    delta = 3.0 / float(knot_count - 1)
    weights = torch.full(
        (knot_count,),
        2.0 * delta,
        dtype=embeddings.dtype,
        device=embeddings.device,
    )
    weights[[0, -1]] = delta
    phi = torch.exp(-knots.square() / 2.0)
    projected = torch.einsum(
        "ktbd,kdp->ktbp", embeddings, directions
    ).unsqueeze(-1) * knots
    error = (projected.cos().mean(dim=2) - phi).square()
    error = error + projected.sin().mean(dim=2).square()
    statistic = (error @ (weights * phi)) * sample_count
    return statistic.mean()


def _random_directions(
    torch: Any,
    dimension: int,
    count: int,
    *,
    generator: Any,
    dtype: Any,
    device: Any,
) -> Any:
    directions = torch.randn(
        dimension,
        count,
        dtype=dtype,
        device="cpu",
        generator=generator,
    ).to(device)
    return directions / directions.norm(p=2, dim=0, keepdim=True)


def _training_batch(
    windows: ActionConditionedWindows,
    indices: NDArray[np.int64],
    ownership: NDArray[np.bool_],
    center: NDArray[np.float64],
    scale: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
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
    return _normalize_states(full, ownership, center, scale), np.asarray(
        conditions, dtype=np.float64
    )


def _validate_windows(windows: ActionConditionedWindows) -> None:
    if windows.histories.shape[1] != 20 or windows.future_states.shape[1] != 10:
        raise ValueError("LeWorldModel requires 20+10 timestep windows")


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
        raise ValueError("LeWorldModel schema has no observations")
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
            raise ValueError("LeWorldModel state tensor shape differs")
        if array.dtype.kind not in ("i", "u", "b") and not np.all(
            np.isfinite(array)
        ):
            raise ValueError("LeWorldModel state tensor is non-finite")
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
            raise ValueError("LeWorldModel metric row is non-finite")
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
            "LeWorldModel fitting requires optional training dependencies"
        ) from error
