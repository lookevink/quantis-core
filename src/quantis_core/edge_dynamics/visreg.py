"""Exact VISReg scale-shape regularization for telemetry histories."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from ..action_conditioned_dynamics import ActionConditionedWindows
from ..graph_telemetry import DeclaredTelemetryGraph
from .complete_lejepa import (
    EncodedTelemetry,
    PairBlockedAnchorSchedule,
    TelemetryViewSchedule,
    build_telemetry_backbone,
    fit_owned_feature_mask,
)


_OBJECTIVES = ("detached_visreg", "no_detach_visreg")


@dataclass(frozen=True)
class VisregConfig:
    """Frozen architecture, optimizer, and explicit randomness controls."""

    objective: str = "detached_visreg"
    width: int = 64
    block_count: int = 2
    head_count: int = 4
    feedforward_width: int = 128
    projector_width: int = 256
    steps: int = 1600
    expected_pair_count: int = 40
    projection_count: int = 256
    prediction_weight: float = 0.4
    regularization_weight: float = 0.6
    learning_rate: float = 5e-4
    weight_decay: float = 5e-2
    warmup_steps: int = 80
    minimum_learning_rate: float = 5e-7
    initialization_seed: int = 509
    anchor_seed: int = 1509
    view_seed: int = 2509
    direction_seed: int = 3509
    preprocessing_protocol: str = (
        "action_conditioned_jepa_topology_transfer_v1"
    )

    def __post_init__(self) -> None:
        integer_values = (
            self.width,
            self.block_count,
            self.head_count,
            self.feedforward_width,
            self.projector_width,
            self.steps,
            self.expected_pair_count,
            self.projection_count,
            self.warmup_steps,
        )
        if (
            self.objective not in _OBJECTIVES
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
                for value in integer_values
            )
            or self.width % self.head_count
        ):
            raise ValueError("VISReg architecture controls are invalid")
        if not (
            self.prediction_weight > 0.0
            and self.regularization_weight > 0.0
            and math.isclose(
                self.prediction_weight + self.regularization_weight,
                1.0,
            )
            and self.learning_rate > 0.0
            and self.minimum_learning_rate > 0.0
            and self.minimum_learning_rate <= self.learning_rate
            and self.weight_decay >= 0.0
        ):
            raise ValueError("VISReg numeric controls are invalid")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (
                self.initialization_seed,
                self.anchor_seed,
                self.view_seed,
                self.direction_seed,
            )
        ):
            raise ValueError("VISReg seeds must be integers")
        if not self.preprocessing_protocol:
            raise ValueError("VISReg preprocessing identity is required")

    def learning_rate_at(self, step: int) -> float:
        """Return the frozen linear-warmup/cosine learning rate."""

        _validate_step(step, self.steps)
        warmup = min(self.warmup_steps, self.steps)
        if step < warmup:
            return self.learning_rate * float(step + 1) / float(warmup)
        remaining = self.steps - warmup
        if remaining <= 1:
            return self.minimum_learning_rate
        progress = float(step - warmup) / float(remaining - 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.minimum_learning_rate + (
            self.learning_rate - self.minimum_learning_rate
        ) * cosine

    def to_dict(self) -> Dict[str, Any]:
        return dict(asdict(self))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "VisregConfig":
        if set(payload) != set(asdict(cls())):
            raise ValueError("VISReg config schema is invalid")
        return cls(**dict(payload))


class VisregDirectionSchedule:
    """One explicit CPU float32 Gaussian direction draw per step."""

    def __init__(
        self, *, width: int, projection_count: int, seed: int
    ) -> None:
        if (
            isinstance(width, bool)
            or not isinstance(width, int)
            or width < 1
            or isinstance(projection_count, bool)
            or not isinstance(projection_count, int)
            or projection_count < 1
            or isinstance(seed, bool)
            or not isinstance(seed, int)
        ):
            raise ValueError("VISReg direction controls are invalid")
        torch = _require_torch()
        self.width = width
        self.projection_count = projection_count
        self.seed = seed
        self._generator = torch.Generator(device="cpu").manual_seed(seed)
        self._initial_state: NDArray[np.uint8] = (
            self._generator.get_state().cpu().numpy().astype(np.uint8)
        )
        self._draw_count = 0

    @property
    def initial_state(self) -> NDArray[np.uint8]:
        return self._initial_state.copy()

    @property
    def final_state(self) -> NDArray[np.uint8]:
        result: NDArray[np.uint8] = (
            self._generator.get_state().cpu().numpy().astype(np.uint8)
        )
        return result

    @property
    def draw_count(self) -> int:
        return self._draw_count

    def draw(self) -> Any:
        """Draw and column-normalize one contiguous float32 matrix."""

        torch = _require_torch()
        raw = torch.randn(
            (self.width, self.projection_count),
            generator=self._generator,
            dtype=torch.float32,
            device="cpu",
        )
        directions = raw / torch.linalg.vector_norm(
            raw, dim=0, keepdim=True
        )
        self._draw_count += 1
        return directions

    @classmethod
    def replay(
        cls,
        *,
        width: int,
        projection_count: int,
        seed: int,
        step: int,
    ) -> Any:
        """Reconstruct the direction matrix consumed at a given step."""

        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("VISReg replay step is invalid")
        schedule = cls(
            width=width, projection_count=projection_count, seed=seed
        )
        result = None
        for _ in range(step + 1):
            result = schedule.draw()
        return result


@dataclass(frozen=True)
class VisregLoss:
    """Literal VISReg components and reconstruction receipts."""

    regularization: Any
    scale: Any
    shape: Any
    center: Any
    means: Any
    standard_deviations: Any
    quantiles: Any
    sorted_projections: Any


def visreg_loss(
    embeddings: Any,
    directions: Any,
    *,
    detach_shape: bool,
) -> VisregLoss:
    """Return the exact clean-room VISReg regularizer."""

    torch = _require_torch()
    if (
        embeddings.ndim != 3
        or embeddings.shape[1] < 2
        or embeddings.shape[2] < 1
        or embeddings.dtype != torch.float32
        or directions.ndim != 2
        or directions.shape[0] != embeddings.shape[2]
        or directions.shape[1] < 1
        or directions.dtype != torch.float32
        or embeddings.device.type != "cpu"
        or directions.device.type != "cpu"
    ):
        raise ValueError("VISReg loss inputs are invalid")
    means = embeddings.mean(dim=1)
    centered = embeddings - means[:, None]
    standard_deviations = (
        torch.linalg.vector_norm(centered, dim=1)
        / math.sqrt(float(embeddings.shape[1]))
    ).clamp_min(1e-6)
    denominator = (
        standard_deviations.detach()
        if detach_shape
        else standard_deviations
    )
    normalized = centered / denominator[:, None]
    sorted_projections = torch.sort(
        normalized @ directions, dim=1
    ).values
    indices = torch.arange(
        1,
        embeddings.shape[1] + 1,
        dtype=torch.float32,
        device="cpu",
    )
    quantiles = math.sqrt(2.0) * torch.erfinv(
        2.0 * indices / float(embeddings.shape[1] + 1) - 1.0
    )
    shape = (
        sorted_projections - quantiles[None, :, None]
    ).square().mean()
    scale = (standard_deviations - 1.0).square().mean()
    center = means.square().mean()
    return VisregLoss(
        regularization=scale + shape + center,
        scale=scale,
        shape=shape,
        center=center,
        means=means,
        standard_deviations=standard_deviations,
        quantiles=quantiles,
        sorted_projections=sorted_projections,
    )


class VisregRepresentation:
    """Shared-encoder VISReg representation with training-only projector."""

    kind = "visreg_telemetry_representation"
    schema_version = 1

    def __init__(self, config: VisregConfig = VisregConfig()) -> None:
        self.config = config
        self._graph: Optional[DeclaredTelemetryGraph] = None
        self._feature_names: Optional[Tuple[str, ...]] = None
        self._ownership_mask: Optional[NDArray[np.bool_]] = None
        self._varying_entity_mask: Optional[NDArray[np.bool_]] = None
        self._network: Any = None
        self._projector: Any = None
        self._direction_initial_state: Optional[
            NDArray[np.uint8]
        ] = None
        self._direction_final_state: Optional[NDArray[np.uint8]] = None
        self._direction_draw_count = 0
        self._training_evidence: Optional[
            Mapping[str, NDArray[Any]]
        ] = None
        self._inference_only = False

    @property
    def inference_parameter_count(self) -> int:
        _, _, _, network = self._encoder_values()
        return int(
            sum(parameter.numel() for parameter in network.parameters())
        )

    @property
    def training_parameter_count(self) -> int:
        _, _, _, network = self._training_values()
        assert self._projector is not None
        return int(
            sum(parameter.numel() for parameter in network.parameters())
            + sum(
                parameter.numel()
                for parameter in self._projector.parameters()
            )
        )

    @property
    def network_sha256(self) -> str:
        _, _, _, network = self._encoder_values()
        return _module_sha256(network)

    @property
    def projector_sha256(self) -> str:
        self._training_values()
        assert self._projector is not None
        return _module_sha256(self._projector)

    @property
    def direction_initial_state(self) -> NDArray[np.uint8]:
        self._training_values()
        assert self._direction_initial_state is not None
        return self._direction_initial_state.copy()

    @property
    def direction_final_state(self) -> NDArray[np.uint8]:
        self._training_values()
        assert self._direction_final_state is not None
        return self._direction_final_state.copy()

    @property
    def direction_draw_count(self) -> int:
        self._training_values()
        return self._direction_draw_count

    @property
    def training_evidence(self) -> Mapping[str, NDArray[Any]]:
        self._training_values()
        if self._training_evidence is None:
            raise RuntimeError("VISReg training evidence is unavailable")
        return {
            name: values.copy()
            for name, values in self._training_evidence.items()
        }

    def fit(
        self, windows: ActionConditionedWindows
    ) -> "VisregRepresentation":
        """Fit the frozen deterministic CPU VISReg representation."""

        if len(set(windows.matched_pair_ids)) != self.config.expected_pair_count:
            raise ValueError("VISReg fit pair count differs from its contract")
        if windows.histories.shape[1] != 20:
            raise ValueError("VISReg requires 20-point current histories")
        torch = _require_torch()
        ownership = fit_owned_feature_mask(windows)
        varying_entities = np.any(
            (
                np.ptp(windows.histories, axis=(0, 1)) > 1e-9
            )
            & ownership,
            axis=1,
        )
        anchors = PairBlockedAnchorSchedule(
            windows, seed=self.config.anchor_seed
        )
        views = TelemetryViewSchedule(
            graph=windows.graph,
            ownership_mask=ownership,
            varying_entity_mask=varying_entities,
            seed=self.config.view_seed,
        )
        rng_state = torch.random.get_rng_state()
        try:
            initialization_generator = torch.Generator(
                device="cpu"
            ).manual_seed(self.config.initialization_seed)
            torch.random.set_rng_state(
                initialization_generator.get_state()
            )
            network = build_telemetry_backbone(
                torch,
                feature_count=windows.histories.shape[-1],
                graph=windows.graph,
                config=self.config,
            )
            projector = torch.nn.Sequential(
                torch.nn.Linear(
                    self.config.width, self.config.projector_width
                ),
                torch.nn.GELU(),
                torch.nn.Linear(
                    self.config.projector_width, self.config.width
                ),
            )
        finally:
            torch.random.set_rng_state(rng_state)
        optimizer = torch.optim.AdamW(
            list(network.parameters()) + list(projector.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        direction_schedule = VisregDirectionSchedule(
            width=self.config.width,
            projection_count=self.config.projection_count,
            seed=self.config.direction_seed,
        )
        pair_count = len(anchors.pair_ids)
        view_count = 8
        entity_count = len(windows.graph.entities)
        evidence: Dict[str, NDArray[Any]] = {
            "embeddings": np.empty(
                (
                    self.config.steps,
                    view_count,
                    pair_count,
                    self.config.width,
                ),
                dtype=np.float32,
            ),
            "directions": np.empty(
                (
                    self.config.steps,
                    self.config.width,
                    self.config.projection_count,
                ),
                dtype=np.float32,
            ),
            "direction_sha256": np.empty(
                (self.config.steps, 32), dtype=np.uint8
            ),
            "gaussian_quantiles": np.empty(
                pair_count, dtype=np.float32
            ),
            "gaussian_quantile_sha256": np.empty(
                32, dtype=np.uint8
            ),
            "sorted_projection_sha256": np.empty(
                (self.config.steps, 32), dtype=np.uint8
            ),
            "anchor_indices": np.empty(
                (self.config.steps, pair_count), dtype=np.int64
            ),
            "anchor_arm_ids": np.empty(
                (self.config.steps, pair_count), dtype=np.int64
            ),
            "anchor_transitions": np.empty(
                (self.config.steps, pair_count), dtype=np.int64
            ),
            "view_visible": np.empty(
                (
                    self.config.steps,
                    view_count,
                    20,
                    entity_count,
                ),
                dtype=np.bool_,
            ),
            "view_present": np.empty(
                (
                    self.config.steps,
                    view_count,
                    20,
                    entity_count,
                ),
                dtype=np.bool_,
            ),
            "regularizer_gradient_step0": np.empty(
                (view_count, pair_count, self.config.width),
                dtype=np.float32,
            ),
        }
        for name in (
            "learning_rate",
            "loss",
            "invariance",
            "regularization",
            "scale",
            "shape",
            "center",
            "gradient_norm",
            "direction_norm_max_error",
            "sorted_projection_mean",
            "sorted_projection_std",
            "sorted_projection_min",
            "sorted_projection_max",
        ):
            evidence[name] = np.empty(self.config.steps, dtype=np.float64)
        positions = np.arange(
            20 * len(windows.graph.entities), dtype=np.int64
        )
        network.train()
        projector.train()
        for step in range(self.config.steps):
            learning_rate = self.config.learning_rate_at(step)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            anchor = anchors.batch(step)
            view_batch = views.batch(
                windows.histories[anchor.indices], step=step
            )
            if (
                len(view_batch.view_names) != view_count
                or len(anchor.indices) != pair_count
            ):
                raise RuntimeError("VISReg independent axes changed")
            pooled = []
            for view_position in range(view_count):
                visible = torch.as_tensor(
                    view_batch.visible_tokens[view_position],
                    dtype=torch.bool,
                )
                hidden = network(
                    torch.as_tensor(
                        view_batch.values[view_position],
                        dtype=torch.float32,
                    ),
                    visible,
                    torch.as_tensor(
                        view_batch.present_tokens[view_position],
                        dtype=torch.bool,
                    ),
                    positions,
                )
                selected = visible.reshape(pair_count, -1)
                pooled.append(
                    (
                        hidden
                        * selected.to(hidden.dtype).unsqueeze(-1)
                    ).sum(dim=1)
                    / selected.sum(dim=1)
                    .clamp_min(1)
                    .unsqueeze(-1)
                )
            embeddings = torch.stack(
                [projector(value) for value in pooled], dim=0
            )
            directions = direction_schedule.draw()
            components = visreg_loss(
                embeddings,
                directions,
                detach_shape=(
                    self.config.objective == "detached_visreg"
                ),
            )
            global_centroid = embeddings[:2].mean(dim=0)
            invariance = (
                embeddings - global_centroid[None]
            ).square().mean()
            loss = (
                self.config.prediction_weight * invariance
                + self.config.regularization_weight
                * components.regularization
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("VISReg training became non-finite")
            if step == 0:
                evidence["regularizer_gradient_step0"][:] = (
                    torch.autograd.grad(
                        components.regularization,
                        embeddings,
                        retain_graph=True,
                    )[0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = _gradient_norm(
                list(network.parameters())
                + list(projector.parameters())
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise RuntimeError("VISReg gradient became non-finite")
            optimizer.step()
            embeddings_array = (
                embeddings.detach().cpu().numpy().astype(np.float32)
            )
            directions_array = (
                directions.detach().cpu().numpy().astype(np.float32)
            )
            evidence["embeddings"][step] = embeddings_array
            evidence["directions"][step] = directions_array
            evidence["direction_sha256"][step] = np.frombuffer(
                hashlib.sha256(
                    directions_array.tobytes(order="C")
                ).digest(),
                dtype=np.uint8,
            )
            quantiles_array = (
                components.quantiles.detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            sorted_array = (
                components.sorted_projections.detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            evidence["gaussian_quantiles"][:] = quantiles_array
            evidence["gaussian_quantile_sha256"][:] = np.frombuffer(
                hashlib.sha256(
                    quantiles_array.tobytes(order="C")
                ).digest(),
                dtype=np.uint8,
            )
            evidence["sorted_projection_sha256"][
                step
            ] = np.frombuffer(
                hashlib.sha256(
                    sorted_array.tobytes(order="C")
                ).digest(),
                dtype=np.uint8,
            )
            evidence["anchor_indices"][step] = anchor.indices
            evidence["anchor_arm_ids"][step] = anchor.arm_ids
            evidence["anchor_transitions"][
                step
            ] = anchor.transition_indices
            evidence["view_visible"][step] = (
                view_batch.visible_tokens[:, 0]
            )
            evidence["view_present"][step] = (
                view_batch.present_tokens[:, 0]
            )
            evidence["learning_rate"][step] = learning_rate
            evidence["loss"][step] = float(loss.detach())
            evidence["invariance"][step] = float(invariance.detach())
            evidence["regularization"][step] = float(
                components.regularization.detach()
            )
            evidence["scale"][step] = float(components.scale.detach())
            evidence["shape"][step] = float(components.shape.detach())
            evidence["center"][step] = float(components.center.detach())
            evidence["gradient_norm"][step] = float(gradient_norm)
            evidence["direction_norm_max_error"][step] = float(
                torch.max(
                    torch.abs(
                        torch.linalg.vector_norm(
                            directions, dim=0
                        )
                        - 1.0
                    )
                )
            )
            evidence["sorted_projection_mean"][step] = float(
                np.mean(sorted_array, dtype=np.float64)
            )
            evidence["sorted_projection_std"][step] = float(
                np.std(sorted_array, dtype=np.float64)
            )
            evidence["sorted_projection_min"][step] = float(
                np.min(sorted_array)
            )
            evidence["sorted_projection_max"][step] = float(
                np.max(sorted_array)
            )
        self._graph = windows.graph
        self._feature_names = windows.state_feature_names
        self._ownership_mask = ownership.copy()
        self._varying_entity_mask = varying_entities.copy()
        self._network = network.eval()
        self._projector = projector.eval()
        self._direction_initial_state = (
            direction_schedule.initial_state
        )
        self._direction_final_state = direction_schedule.final_state
        self._direction_draw_count = direction_schedule.draw_count
        self._training_evidence = evidence
        self._inference_only = False
        return self

    def encode(
        self,
        histories: NDArray[np.float64],
        graph: DeclaredTelemetryGraph,
    ) -> EncodedTelemetry:
        """Encode complete histories as ordered anchor-time entity tokens."""

        fitted_graph, feature_names, ownership, network = (
            self._encoder_values()
        )
        source = np.asarray(histories, dtype=np.float64)
        if (
            graph.to_dict() != fitted_graph.to_dict()
            or source.ndim != 4
            or source.shape[1:]
            != (
                20,
                len(fitted_graph.entities),
                len(feature_names),
            )
            or not np.all(np.isfinite(source))
        ):
            raise ValueError("VISReg encoding input is invalid")
        torch = _require_torch()
        values = np.where(ownership[None, None], source, 0.0)
        visible = np.ones(values.shape[:-1], dtype=np.bool_)
        with torch.no_grad():
            hidden = network(
                torch.as_tensor(values, dtype=torch.float32),
                torch.as_tensor(visible, dtype=torch.bool),
                torch.as_tensor(visible, dtype=torch.bool),
                np.arange(20 * len(fitted_graph.entities)),
            )
            tokens = (
                hidden.reshape(
                    len(source),
                    20,
                    len(fitted_graph.entities),
                    self.config.width,
                )[:, -1]
                .cpu()
                .numpy()
                .astype(np.float64)
            )
        return EncodedTelemetry(
            tokens=tokens,
            entity_ids=fitted_graph.entity_ids,
            ownership_mask=ownership.copy(),
            observation_mask=ownership.copy(),
            content_sha256=self._content_sha256(),
            graph_sha256=_canonical_sha256(fitted_graph.to_dict()),
            state_schema_sha256=_canonical_sha256(
                {"feature_names": list(feature_names)}
            ),
            preprocessing_sha256=_canonical_sha256(
                {"protocol": self.config.preprocessing_protocol}
            ),
            encoder_sha256=self._content_sha256(),
        )

    def diagnose_views(
        self,
        histories: NDArray[np.float64],
        graph: DeclaredTelemetryGraph,
        *,
        step: int,
    ) -> NDArray[np.float64]:
        """Return all eight projector embeddings for a fixed diagnostic."""

        fitted_graph, _, ownership, network = self._training_values()
        if (
            self._projector is None
            or self._varying_entity_mask is None
            or graph.to_dict() != fitted_graph.to_dict()
        ):
            raise ValueError("VISReg diagnostic state is unavailable")
        source = np.asarray(histories, dtype=np.float64)
        view_batch = TelemetryViewSchedule(
            graph=graph,
            ownership_mask=ownership,
            varying_entity_mask=self._varying_entity_mask,
            seed=self.config.view_seed,
        ).batch(source, step=step)
        torch = _require_torch()
        positions = np.arange(20 * len(graph.entities), dtype=np.int64)
        embeddings = []
        with torch.no_grad():
            for view_position in range(len(view_batch.view_names)):
                visible = torch.as_tensor(
                    view_batch.visible_tokens[view_position],
                    dtype=torch.bool,
                )
                hidden = network(
                    torch.as_tensor(
                        view_batch.values[view_position],
                        dtype=torch.float32,
                    ),
                    visible,
                    torch.as_tensor(
                        view_batch.present_tokens[view_position],
                        dtype=torch.bool,
                    ),
                    positions,
                )
                selected = visible.reshape(len(source), -1)
                pooled = (
                    hidden * selected[..., None].to(hidden.dtype)
                ).sum(dim=1) / selected.sum(
                    dim=1
                ).clamp_min(1).unsqueeze(-1)
                embeddings.append(self._projector(pooled))
        result: NDArray[np.float64] = (
            torch.stack(embeddings)
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the complete fitted training and inference state."""

        graph, feature_names, ownership, network = self._training_values()
        assert self._projector is not None
        assert self._varying_entity_mask is not None
        assert self._direction_initial_state is not None
        assert self._direction_final_state is not None
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "config": self.config.to_dict(),
            "graph": graph.to_dict(),
            "feature_names": list(feature_names),
            "ownership_mask": ownership.astype(int).tolist(),
            "varying_entity_mask": (
                self._varying_entity_mask.astype(int).tolist()
            ),
            "network_state": _module_state(network),
            "projector_state": _module_state(self._projector),
            "direction_initial_state": (
                self._direction_initial_state.astype(int).tolist()
            ),
            "direction_final_state": (
                self._direction_final_state.astype(int).tolist()
            ),
            "direction_draw_count": self._direction_draw_count,
            "network_sha256": self.network_sha256,
            "projector_sha256": self.projector_sha256,
            "training_parameter_count": self.training_parameter_count,
            "inference_parameter_count": self.inference_parameter_count,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "VisregRepresentation":
        """Restore a full model without consuming ambient random state."""

        expected_keys = {
            "schema_version",
            "kind",
            "config",
            "graph",
            "feature_names",
            "ownership_mask",
            "varying_entity_mask",
            "network_state",
            "projector_state",
            "direction_initial_state",
            "direction_final_state",
            "direction_draw_count",
            "network_sha256",
            "projector_sha256",
            "training_parameter_count",
            "inference_parameter_count",
        }
        if set(payload) != expected_keys:
            raise ValueError("VISReg full artifact schema is invalid")
        model = cls._restore_encoder(payload)
        torch = _require_torch()
        rng_state = torch.random.get_rng_state()
        try:
            projector = torch.nn.Sequential(
                torch.nn.Linear(
                    model.config.width, model.config.projector_width
                ),
                torch.nn.GELU(),
                torch.nn.Linear(
                    model.config.projector_width, model.config.width
                ),
            )
        finally:
            torch.random.set_rng_state(rng_state)
        _restore_module(
            projector, dict(payload["projector_state"]), "projector"
        )
        model._projector = projector.eval()
        model._varying_entity_mask = np.asarray(
            payload["varying_entity_mask"], dtype=np.bool_
        )
        model._direction_initial_state = np.asarray(
            payload["direction_initial_state"], dtype=np.uint8
        )
        model._direction_final_state = np.asarray(
            payload["direction_final_state"], dtype=np.uint8
        )
        model._direction_draw_count = int(
            payload["direction_draw_count"]
        )
        model._inference_only = False
        graph, _, _, _ = model._encoder_values()
        if (
            model._varying_entity_mask.shape
            != (len(graph.entities),)
            or model._direction_draw_count != model.config.steps
            or model.network_sha256 != str(payload["network_sha256"])
            or model.projector_sha256
            != str(payload["projector_sha256"])
            or model.training_parameter_count
            != int(payload["training_parameter_count"])
            or model.inference_parameter_count
            != int(payload["inference_parameter_count"])
        ):
            raise ValueError("VISReg full artifact identity differs")
        return model

    def to_inference_dict(self) -> Dict[str, Any]:
        """Serialize only the state reachable by public edge inference."""

        graph, feature_names, ownership, network = self._encoder_values()
        return {
            "schema_version": self.schema_version,
            "kind": self.kind + "_inference",
            "config": {
                "width": self.config.width,
                "block_count": self.config.block_count,
                "head_count": self.config.head_count,
                "feedforward_width": self.config.feedforward_width,
                "preprocessing_protocol": (
                    self.config.preprocessing_protocol
                ),
            },
            "graph": graph.to_dict(),
            "feature_names": list(feature_names),
            "ownership_mask": ownership.astype(int).tolist(),
            "network_state": _module_state(network),
            "inference_parameter_count": self.inference_parameter_count,
        }

    @classmethod
    def from_inference_dict(
        cls, payload: Mapping[str, Any]
    ) -> "VisregRepresentation":
        """Restore the strict causal inference-only bundle."""

        expected_keys = {
            "schema_version",
            "kind",
            "config",
            "graph",
            "feature_names",
            "ownership_mask",
            "network_state",
            "inference_parameter_count",
        }
        if (
            set(payload) != expected_keys
            or payload.get("schema_version") != cls.schema_version
            or payload.get("kind") != cls.kind + "_inference"
        ):
            raise ValueError("unsupported VISReg inference artifact")
        raw_config = dict(payload["config"])
        expected_config_keys = {
            "width",
            "block_count",
            "head_count",
            "feedforward_width",
            "preprocessing_protocol",
        }
        if set(raw_config) != expected_config_keys:
            raise ValueError("VISReg inference config schema is invalid")
        if any(
            isinstance(raw_config[name], bool)
            or not isinstance(raw_config[name], int)
            for name in (
                "width",
                "block_count",
                "head_count",
                "feedforward_width",
            )
        ) or not isinstance(
            raw_config["preprocessing_protocol"], str
        ):
            raise ValueError("VISReg inference config values are invalid")
        config = VisregConfig(
            width=raw_config["width"],
            block_count=raw_config["block_count"],
            head_count=raw_config["head_count"],
            feedforward_width=raw_config["feedforward_width"],
            preprocessing_protocol=raw_config[
                "preprocessing_protocol"
            ],
        )
        adjusted = dict(payload)
        adjusted["kind"] = cls.kind
        adjusted["config"] = config.to_dict()
        model = cls._restore_encoder(adjusted)
        model._inference_only = True
        if model.inference_parameter_count != int(
            payload["inference_parameter_count"]
        ):
            raise ValueError("VISReg inference capacity mismatch")
        return model

    @classmethod
    def _restore_encoder(
        cls, payload: Mapping[str, Any]
    ) -> "VisregRepresentation":
        if (
            payload.get("schema_version") != cls.schema_version
            or payload.get("kind") != cls.kind
        ):
            raise ValueError("unsupported VISReg artifact")
        config = VisregConfig.from_dict(dict(payload["config"]))
        graph = DeclaredTelemetryGraph.from_dict(dict(payload["graph"]))
        feature_names = tuple(str(value) for value in payload["feature_names"])
        ownership = np.asarray(
            payload["ownership_mask"], dtype=np.bool_
        )
        if ownership.shape != (len(graph.entities), len(feature_names)):
            raise ValueError("VISReg ownership schema differs")
        torch = _require_torch()
        rng_state = torch.random.get_rng_state()
        try:
            network = build_telemetry_backbone(
                torch,
                feature_count=len(feature_names),
                graph=graph,
                config=config,
            )
        finally:
            torch.random.set_rng_state(rng_state)
        _restore_module(network, dict(payload["network_state"]), "network")
        model = cls(config)
        model._graph = graph
        model._feature_names = feature_names
        model._ownership_mask = ownership
        model._network = network.eval()
        return model

    def _content_sha256(self) -> str:
        graph, feature_names, ownership, network = self._encoder_values()
        return _canonical_sha256(
            {
                "config": self.config.to_dict(),
                "graph": graph.to_dict(),
                "feature_names": list(feature_names),
                "ownership_mask": ownership.astype(int).tolist(),
                "network_state": _module_state(network),
            }
        )

    def _encoder_values(
        self,
    ) -> Tuple[
        DeclaredTelemetryGraph,
        Tuple[str, ...],
        NDArray[np.bool_],
        Any,
    ]:
        if (
            self._graph is None
            or self._feature_names is None
            or self._ownership_mask is None
            or self._network is None
        ):
            raise RuntimeError("VISReg representation is not fitted")
        return (
            self._graph,
            self._feature_names,
            self._ownership_mask,
            self._network,
        )

    def _training_values(
        self,
    ) -> Tuple[
        DeclaredTelemetryGraph,
        Tuple[str, ...],
        NDArray[np.bool_],
        Any,
    ]:
        values = self._encoder_values()
        if self._inference_only or self._projector is None:
            raise RuntimeError("VISReg inference bundle has no training state")
        return values


def assess_visreg_gates(
    *,
    forecast_scores: Mapping[str, Mapping[str, Mapping[str, float]]],
    raw_scores: Mapping[str, Mapping[str, float]],
    mechanism_gates: Mapping[str, bool],
    attribution: Mapping[str, Mapping[str, float]],
    action_sanity: Mapping[str, Mapping[str, float]],
    restoration_max_abs: Mapping[str, float],
    protocol_checks: Mapping[str, bool],
    parameter_counts: Mapping[str, Mapping[str, int]],
    transfer_pair_errors: Mapping[str, Mapping[str, float]],
    state_probe: Mapping[str, Mapping[str, Any]],
    varying_entity_ids: Sequence[str],
    deployed_bundle_bytes: int,
    median_latency_ms: float,
) -> Dict[str, Any]:
    """Purely recompute the frozen VISReg promotion decision."""

    candidate_name = "detached_visreg"
    candidate = forecast_scores[candidate_name]
    control_names = tuple(
        name for name in forecast_scores if name != candidate_name
    )
    best_selection_control = min(
        control_names,
        key=lambda name: forecast_scores[name]["selection"][
            "downstream_effect_mse"
        ],
    )
    best_transfer_control = min(
        control_names,
        key=lambda name: forecast_scores[name]["transfer_evaluation"][
            "downstream_effect_mse"
        ],
    )
    required_protocols = (
        "evidence_arrays_are_finite",
        "role_contract_recomputes",
        "all_schedules_recompute",
        "objective_recomputes",
        "mode_enforcement_recomputes",
        "capacity_recomputes",
        "public_inference_is_causal",
        "copied_source_assessor_recomputes",
        "copied_prior_controls_match",
        "selection_only_ridge_choice_recomputes",
        "selection_safety_status_recomputes",
        "bundle_size_recomputes",
        "latency_recomputes",
        "state_probe_recomputes",
    )
    safety = {
        name: bool(protocol_checks.get(name, False))
        for name in required_protocols
    }
    counts = list(parameter_counts.values())
    candidate_probe = state_probe[candidate_name]
    pca_probe = state_probe["matched_pca"]
    safety.update(
        {
            "capacity_is_matched": bool(counts)
            and all(value == counts[0] for value in counts[1:]),
            "restoration_within_1e_6": all(
                np.isfinite(value) and value <= 1e-6
                for value in restoration_max_abs.values()
            ),
            "state_probe_aggregate_within_1_05_pca": (
                float(candidate_probe["aggregate_nrmse"])
                <= 1.05 * float(pca_probe["aggregate_nrmse"])
            ),
            "state_probe_entities_within_1_15_pca": all(
                float(candidate_probe["entity_nrmse"][entity_id])
                <= 1.15
                * float(pca_probe["entity_nrmse"][entity_id])
                for entity_id in varying_entity_ids
            ),
            "selection_overall_within_1_05_raw": (
                candidate["selection"]["overall_mse"]
                <= 1.05 * raw_scores["selection"]["overall_mse"]
            ),
            "selection_action_within_1_05_raw": (
                candidate["selection"]["action_overlap_mse"]
                <= 1.05
                * raw_scores["selection"]["action_overlap_mse"]
            ),
            "transfer_overall_within_1_05_raw": (
                candidate["transfer_evaluation"]["overall_mse"]
                <= 1.05
                * raw_scores["transfer_evaluation"]["overall_mse"]
            ),
            "transfer_action_within_1_05_raw": (
                candidate["transfer_evaluation"][
                    "action_overlap_mse"
                ]
                <= 1.05
                * raw_scores["transfer_evaluation"][
                    "action_overlap_mse"
                ]
            ),
            "action_and_target_hit_at_1": (
                attribution[candidate_name][
                    "action_and_target_hit_at_1"
                ]
                >= 0.95
            ),
            "no_action_specificity": (
                attribution[candidate_name]["no_action_specificity"] == 1.0
            ),
            "correct_action_sanity": (
                action_sanity[candidate_name][
                    "correct_action_beats_both_fraction"
                ]
                >= 0.80
            ),
            "deployed_bundle_within_16_mib": (
                isinstance(deployed_bundle_bytes, int)
                and not isinstance(deployed_bundle_bytes, bool)
                and 0 < deployed_bundle_bytes <= 16 * 1024 * 1024
            ),
            "latency_is_recorded": (
                np.isfinite(median_latency_ms)
                and median_latency_ms > 0.0
            ),
        }
    )
    candidate_pair = transfer_pair_errors[candidate_name]
    control_pair = transfer_pair_errors[best_transfer_control]
    common_pairs = sorted(set(candidate_pair) & set(control_pair))
    pair_win_fraction = float(
        np.mean(
            [
                candidate_pair[pair] < control_pair[pair]
                for pair in common_pairs
            ]
        )
    )
    value = {
        "selection_effect_is_best": (
            candidate["selection"]["downstream_effect_mse"]
            < forecast_scores[best_selection_control]["selection"][
                "downstream_effect_mse"
            ]
        ),
        "transfer_effect_improves_best_control_and_raw_by_5_percent": (
            candidate["transfer_evaluation"]["downstream_effect_mse"]
            <= 0.95
            * min(
                forecast_scores[best_transfer_control][
                    "transfer_evaluation"
                ]["downstream_effect_mse"],
                raw_scores["transfer_evaluation"][
                    "downstream_effect_mse"
                ],
            )
        ),
        "transfer_pair_win_fraction": pair_win_fraction >= 0.60,
    }
    passed = bool(
        all(safety.values())
        and all(bool(value) for value in mechanism_gates.values())
        and all(value.values())
    )
    return {
        "schema_version": 1,
        "experiment": "visreg_telemetry_tracer_v1",
        "safety_gates": safety,
        "mechanism_gates": {
            str(name): bool(value)
            for name, value in mechanism_gates.items()
        },
        "value_gates": value,
        "best_selection_control": best_selection_control,
        "best_transfer_control": best_transfer_control,
        "transfer_pair_errors": {
            str(name): {
                str(pair): float(error)
                for pair, error in pair_values.items()
            }
            for name, pair_values in transfer_pair_errors.items()
        },
        "candidate_pair_win_fraction": pair_win_fraction,
        "passed": passed,
        "decision": (
            "advance_visreg_recipe" if passed else "reject_visreg_recipe"
        ),
    }


def _gradient_norm(parameters: Any) -> Any:
    torch = _require_torch()
    squared = [
        torch.sum(parameter.grad.detach().square())
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not squared:
        return torch.tensor(0.0)
    return torch.sqrt(torch.stack(squared).sum())


def _module_state(module: Any) -> Dict[str, Any]:
    return {
        name: tensor.detach().cpu().tolist()
        for name, tensor in module.state_dict().items()
    }


def _module_sha256(module: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _restore_module(
    module: Any, payload: Mapping[str, Any], label: str
) -> None:
    torch = _require_torch()
    expected = module.state_dict()
    if set(payload) != set(expected):
        raise ValueError(f"VISReg {label} tensor names do not match")
    module.load_state_dict(
        {
            name: torch.as_tensor(payload[name], dtype=tensor.dtype)
            for name, tensor in expected.items()
        }
    )


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _validate_step(step: int, steps: int) -> None:
    if (
        isinstance(step, bool)
        or not isinstance(step, int)
        or step < 0
        or step >= steps
    ):
        raise ValueError("VISReg step is outside its schedule")


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("VISReg requires the torch extra") from error
    return torch
