"""Complete predictor-free multi-view LeJEPA representation primitives.

This module is intentionally separate from the action-conditioned EMA JEPA
predictor.  Its public seams implement the frozen representation contract:
pair-blocked anchors, semantic telemetry views, the exact LeJEPA loss, and
restorable entity-preserving representations.
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from ..action_conditioned_dynamics import ActionConditionedWindows
from ..graph_telemetry import DeclaredTelemetryGraph
from .action_conditioned_jepa import (
    sketched_isotropic_gaussian_regularization,
)


@dataclass(frozen=True)
class PairBlockedAnchorBatch:
    """One independent anchor per matched pair."""

    indices: NDArray[np.int64]
    pair_ids: Tuple[str, ...]
    arm_ids: NDArray[np.int64]
    transition_indices: NDArray[np.int64]

    def __post_init__(self) -> None:
        count = len(self.pair_ids)
        if (
            self.indices.shape != (count,)
            or self.arm_ids.shape != (count,)
            or self.transition_indices.shape != (count,)
            or len(set(self.pair_ids)) != count
            or not np.all(np.isin(self.arm_ids, (0, 1)))
        ):
            raise ValueError("pair-blocked anchor batch is invalid")


class PairBlockedAnchorSchedule:
    """Deterministic arm-balanced schedule over every pair transition."""

    def __init__(
        self, windows: ActionConditionedWindows, *, seed: int
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("pair-blocked schedule seed must be an integer")
        self.seed = seed
        self.pair_ids = tuple(sorted(set(windows.matched_pair_ids)))
        if not self.pair_ids:
            raise ValueError("pair-blocked schedule requires matched pairs")
        self._indices: Dict[Tuple[str, int, int], int] = {}
        transitions_by_pair: Dict[str, Tuple[int, ...]] = {}
        for pair_id in self.pair_ids:
            rows = np.flatnonzero(
                np.asarray(windows.matched_pair_ids) == pair_id
            )
            trajectories = tuple(
                sorted({windows.trajectory_ids[int(row)] for row in rows})
            )
            if len(trajectories) != 2:
                raise ValueError(
                    "each matched pair must contain exactly two trajectories"
                )
            arm_transitions = []
            for arm_id, trajectory_id in enumerate(trajectories):
                arm_rows = [
                    int(row)
                    for row in rows
                    if windows.trajectory_ids[int(row)] == trajectory_id
                ]
                transition_map = {
                    int(windows.transition_indices[row]): row
                    for row in arm_rows
                }
                if len(transition_map) != len(arm_rows):
                    raise ValueError(
                        "trajectory transitions must be unique"
                    )
                arm_transitions.append(tuple(sorted(transition_map)))
                for transition, row in transition_map.items():
                    self._indices[(pair_id, arm_id, transition)] = row
            if arm_transitions[0] != arm_transitions[1]:
                raise ValueError("matched arms must align on transitions")
            transitions_by_pair[pair_id] = arm_transitions[0]
        transition_sets = set(transitions_by_pair.values())
        if len(transition_sets) != 1:
            raise ValueError(
                "all pair-blocked schedule transitions must align"
            )
        self.transitions = next(iter(transition_sets))
        if not self.transitions:
            raise ValueError("pair-blocked schedule has no transitions")

    def batch(self, step: int) -> PairBlockedAnchorBatch:
        """Return the step's one-anchor-per-pair independent batch."""

        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("pair-blocked schedule step is invalid")
        transition_count = len(self.transitions)
        cycle = step // transition_count
        cycle_position = step % transition_count
        generator = np.random.default_rng(
            np.random.SeedSequence((self.seed, cycle))
        )
        indices = []
        arm_ids = []
        selected_transitions = []
        for pair_position, pair_id in enumerate(self.pair_ids):
            permutation = generator.permutation(self.transitions)
            transition = int(permutation[cycle_position])
            arm_id = (step + pair_position) % 2
            indices.append(
                self._indices[(pair_id, arm_id, transition)]
            )
            arm_ids.append(arm_id)
            selected_transitions.append(transition)
        return PairBlockedAnchorBatch(
            indices=np.asarray(indices, dtype=np.int64),
            pair_ids=self.pair_ids,
            arm_ids=np.asarray(arm_ids, dtype=np.int64),
            transition_indices=np.asarray(
                selected_transitions, dtype=np.int64
            ),
        )


@dataclass(frozen=True)
class TelemetryViewBatch:
    """Eight aligned semantic views of the same independent anchors."""

    values: NDArray[np.float64]
    visible_tokens: NDArray[np.bool_]
    present_tokens: NDArray[np.bool_]
    view_names: Tuple[str, ...]
    local_roots: Tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            self.values.ndim != 5
            or self.visible_tokens.shape != self.values.shape[:-1]
            or self.present_tokens.shape != self.visible_tokens.shape
            or np.any(self.visible_tokens & ~self.present_tokens)
            or self.values.shape[0] != len(self.view_names)
            or self.values.shape[0] != 2 + len(self.local_roots)
            or not np.all(np.isfinite(self.values))
        ):
            raise ValueError("telemetry views do not align")


def fit_owned_feature_mask(
    windows: ActionConditionedWindows,
) -> NDArray[np.bool_]:
    """Return declared and fit-observed entity/feature ownership."""

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
        if (
            entity_id in entity_positions
            and feature_name in feature_positions
        ):
            mask[
                entity_positions[entity_id],
                feature_positions[feature_name],
            ] = True
    ranges = np.ptp(windows.histories, axis=(0, 1))
    mask |= ranges > 1e-9
    if not np.any(mask):
        raise ValueError("telemetry schema has no owned observations")
    return mask


class TelemetryViewSchedule:
    """Seeded fixed-layout telemetry view generator."""

    def __init__(
        self,
        *,
        graph: DeclaredTelemetryGraph,
        ownership_mask: NDArray[np.bool_],
        varying_entity_mask: NDArray[np.bool_],
        seed: int,
    ) -> None:
        ownership = np.asarray(ownership_mask, dtype=np.bool_)
        varying = np.asarray(varying_entity_mask, dtype=np.bool_)
        if (
            ownership.ndim != 2
            or ownership.shape[0] != len(graph.entities)
            or varying.shape != (len(graph.entities),)
            or isinstance(seed, bool)
            or not isinstance(seed, int)
        ):
            raise ValueError("telemetry view schedule schema is invalid")
        roots = tuple(int(value) for value in np.flatnonzero(varying))
        if len(roots) != 6:
            raise ValueError(
                "frozen telemetry view layout requires six varying roots"
            )
        self.graph = graph
        self.ownership_mask = ownership.copy()
        self.roots = roots
        self.seed = seed
        self._local_entities = tuple(
            self._connected_block(root) for root in roots
        )

    def batch(
        self, histories: NDArray[np.float64], *, step: int
    ) -> TelemetryViewBatch:
        """Copy anchors into the frozen two-global/six-local layout."""

        source = np.asarray(histories, dtype=np.float64)
        if (
            source.ndim != 4
            or source.shape[1] != 20
            or source.shape[2:] != self.ownership_mask.shape
            or isinstance(step, bool)
            or not isinstance(step, int)
            or step < 0
            or not np.all(np.isfinite(source))
        ):
            raise ValueError("telemetry view anchors are invalid")
        view_count = 2 + len(self.roots)
        values = np.zeros((view_count,) + source.shape, dtype=np.float64)
        visible = np.zeros(
            (view_count,) + source.shape[:-1], dtype=np.bool_
        )
        present = np.zeros_like(visible)
        owned_values = np.where(
            self.ownership_mask[None, None], source, 0.0
        )
        values[0] = owned_values
        visible[0] = True
        present[0] = True
        values[1, :, 4:] = owned_values[:, 4:]
        visible[1, :, 4:] = True
        present[1, :, 4:] = True
        generator = np.random.default_rng(
            np.random.SeedSequence((self.seed, step))
        )
        observed_entities = np.flatnonzero(
            np.any(self.ownership_mask, axis=1)
        )
        for view_position, time_start in ((0, 0), (1, 4)):
            candidates = np.asarray(
                [
                    (time_position, int(entity_position))
                    for time_position in range(time_start, 20)
                    for entity_position in observed_entities
                ],
                dtype=np.int64,
            )
            mask_count = max(1, int(round(0.10 * len(candidates))))
            selected = generator.choice(
                len(candidates), size=mask_count, replace=False
            )
            for time_position, entity_position in candidates[selected]:
                visible[
                    view_position, :, time_position, entity_position
                ] = False
                values[
                    view_position, :, time_position, entity_position
                ] = 0.0
        for local_position, entities in enumerate(
            self._local_entities, start=2
        ):
            for entity_position in entities:
                values[
                    local_position, :, 10:, entity_position
                ] = owned_values[:, 10:, entity_position]
                visible[
                    local_position, :, 10:, entity_position
                ] = True
                present[
                    local_position, :, 10:, entity_position
                ] = True
        return TelemetryViewBatch(
            values=values,
            visible_tokens=visible,
            present_tokens=present,
            view_names=("global_a", "global_b")
            + tuple(
                f"local_{self.graph.entity_ids[root]}"
                for root in self.roots
            ),
            local_roots=self.roots,
        )

    def _connected_block(self, root: int) -> NDArray[np.int64]:
        adjacency = _entity_adjacency(self.graph)
        selected = [root]
        frontier = list(np.flatnonzero(adjacency[root]))
        while len(selected) < 3 and frontier:
            candidate = int(frontier.pop(0))
            if candidate in selected:
                continue
            selected.append(candidate)
            for neighbor in np.flatnonzero(adjacency[candidate]):
                value = int(neighbor)
                if value not in selected and value not in frontier:
                    frontier.append(value)
        if len(selected) != 3:
            raise ValueError(
                "each local telemetry root needs a connected 3-entity block"
            )
        return np.asarray(sorted(selected), dtype=np.int64)


@dataclass(frozen=True)
class CompleteLejepaLoss:
    """Exact scalar loss and its two reportable components."""

    loss: Any
    sigreg: Any
    invariance: Any


def complete_lejepa_loss(
    embeddings: Any,
    *,
    generator: Any,
    sketch_dimension: int = 1024,
    knot_count: int = 17,
    sigreg_weight: float = 0.05,
    invariance_weight: float = 0.95,
) -> CompleteLejepaLoss:
    """Return the exact predictor-free LeJEPA objective for ``(V,N,D)``."""

    if embeddings.ndim != 3 or embeddings.size(0) < 2:
        raise ValueError("complete LeJEPA embeddings must have shape (V,N,D)")
    global_mean = embeddings[:2].mean(dim=0)
    invariance = ((embeddings - global_mean) ** 2).mean()
    sigreg = sketched_isotropic_gaussian_regularization(
        embeddings,
        generator=generator,
        sketch_dimension=sketch_dimension,
        knot_count=knot_count,
    )
    return CompleteLejepaLoss(
        loss=sigreg_weight * sigreg + invariance_weight * invariance,
        sigreg=sigreg,
        invariance=invariance,
    )


_OBJECTIVES = (
    "lejepa",
    "invariance_only",
    "sigreg_only",
    "masked_autoencoder",
)
COMPLETE_LEJEPA_REPRESENTATION_NAMES = (
    "complete_lejepa",
    "invariance_only",
    "sigreg_only",
    "masked_autoencoder",
    "matched_pca",
)


@dataclass(frozen=True)
class CompleteLejepaConfig:
    """Frozen architecture, optimizer, objective, and randomness controls."""

    objective: str = "lejepa"
    width: int = 64
    block_count: int = 2
    head_count: int = 4
    feedforward_width: int = 128
    projector_width: int = 256
    steps: int = 1600
    expected_pair_count: int = 40
    learning_rate: float = 5e-4
    weight_decay: float = 5e-2
    warmup_steps: int = 80
    minimum_learning_rate: float = 5e-7
    sketch_dimension: int = 1024
    knot_count: int = 17
    initialization_seed: int = 509
    anchor_seed: int = 1509
    view_seed: int = 2509
    sigreg_seed: int = 3509
    decoder_seed: int = 4509
    preprocessing_protocol: str = (
        "action_conditioned_jepa_topology_transfer_v1"
    )

    def __post_init__(self) -> None:
        if self.objective not in _OBJECTIVES:
            raise ValueError("unsupported complete LeJEPA objective")
        integer_values = (
            self.width,
            self.block_count,
            self.head_count,
            self.feedforward_width,
            self.projector_width,
            self.steps,
            self.expected_pair_count,
            self.warmup_steps,
            self.sketch_dimension,
            self.knot_count,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in integer_values
        ):
            raise ValueError("complete LeJEPA integer controls are invalid")
        if self.width % self.head_count:
            raise ValueError("complete LeJEPA heads must divide width")
        if self.knot_count < 2:
            raise ValueError("complete LeJEPA requires at least two knots")
        if not (
            self.learning_rate > 0.0
            and self.weight_decay >= 0.0
            and self.minimum_learning_rate > 0.0
            and self.minimum_learning_rate <= self.learning_rate
        ):
            raise ValueError("complete LeJEPA optimizer controls are invalid")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (
                self.initialization_seed,
                self.anchor_seed,
                self.view_seed,
                self.sigreg_seed,
                self.decoder_seed,
            )
        ):
            raise ValueError("complete LeJEPA seeds must be integers")
        if not self.preprocessing_protocol:
            raise ValueError(
                "complete LeJEPA preprocessing identity cannot be empty"
            )

    def to_dict(self) -> Dict[str, Any]:
        return dict(asdict(self))

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "CompleteLejepaConfig":
        if set(payload) != set(asdict(cls())):
            raise ValueError("complete LeJEPA config schema is invalid")
        return cls(**dict(payload))


@dataclass(frozen=True)
class EncodedTelemetry:
    """Ordered frozen entity tokens plus their semantic identity."""

    tokens: NDArray[np.float64]
    entity_ids: Tuple[str, ...]
    ownership_mask: NDArray[np.bool_]
    observation_mask: NDArray[np.bool_]
    content_sha256: str
    graph_sha256: str
    state_schema_sha256: str
    preprocessing_sha256: str
    encoder_sha256: str

    def __post_init__(self) -> None:
        if (
            self.tokens.ndim != 3
            or self.tokens.shape[1] != len(self.entity_ids)
            or self.ownership_mask.shape[:1] != (len(self.entity_ids),)
            or self.observation_mask.shape != self.ownership_mask.shape
            or not np.all(np.isfinite(self.tokens))
            or any(
                len(value) != 64
                for value in (
                    self.content_sha256,
                    self.graph_sha256,
                    self.state_schema_sha256,
                    self.preprocessing_sha256,
                    self.encoder_sha256,
                )
            )
        ):
            raise ValueError("encoded telemetry is invalid")


class CompleteLejepaRepresentation:
    """Fitted entity-preserving complete LeJEPA representation."""

    kind = "complete_multi_view_lejepa_representation"
    schema_version = 1

    def __init__(
        self, config: CompleteLejepaConfig = CompleteLejepaConfig()
    ) -> None:
        self.config = config
        self._graph: Optional[DeclaredTelemetryGraph] = None
        self._feature_names: Optional[Tuple[str, ...]] = None
        self._ownership_mask: Optional[NDArray[np.bool_]] = None
        self._network: Any = None
        self._projector: Any = None
        self._decoder: Any = None
        self._sigreg_generator_state: Optional[Sequence[int]] = None
        self._training_metrics: Tuple[Mapping[str, float], ...] = ()

    @property
    def training_metrics(self) -> Tuple[Mapping[str, float], ...]:
        self._fitted_values()
        return self._training_metrics

    @property
    def inference_parameter_count(self) -> int:
        self._fitted_values()
        return int(
            sum(parameter.numel() for parameter in self._network.parameters())
        )

    @property
    def training_only_parameter_count(self) -> int:
        self._fitted_values()
        modules = [self._projector]
        if self._decoder is not None:
            modules.append(self._decoder)
        return int(
            sum(
                parameter.numel()
                for module in modules
                for parameter in module.parameters()
            )
        )

    def fit(
        self, windows: ActionConditionedWindows
    ) -> "CompleteLejepaRepresentation":
        """Fit the final deterministic CPU representation."""

        torch = _require_torch()
        if len(set(windows.matched_pair_ids)) != self.config.expected_pair_count:
            raise ValueError(
                "complete LeJEPA fit pair count differs from its contract"
            )
        if windows.histories.shape[1] != 20:
            raise ValueError("complete LeJEPA requires 20-point contexts")
        ownership = fit_owned_feature_mask(windows)
        view_schedule = TelemetryViewSchedule(
            graph=windows.graph,
            ownership_mask=ownership,
            varying_entity_mask=np.any(
                np.ptp(windows.histories, axis=(0, 1)) > 1e-9, axis=1
            ),
            seed=self.config.view_seed,
        )
        anchor_schedule = PairBlockedAnchorSchedule(
            windows, seed=self.config.anchor_seed
        )
        torch.manual_seed(self.config.initialization_seed)
        network = _build_backbone(
            torch,
            feature_count=windows.histories.shape[-1],
            graph=windows.graph,
            config=self.config,
        )
        projector = torch.nn.Sequential(
            torch.nn.Linear(self.config.width, self.config.projector_width),
            torch.nn.GELU(),
            torch.nn.Linear(self.config.projector_width, self.config.width),
        )
        decoder = None
        if self.config.objective == "masked_autoencoder":
            state = torch.random.get_rng_state()
            torch.manual_seed(self.config.decoder_seed)
            decoder = torch.nn.Sequential(
                torch.nn.Linear(self.config.width, self.config.width),
                torch.nn.GELU(),
                torch.nn.Linear(
                    self.config.width, windows.histories.shape[-1]
                ),
            )
            torch.random.set_rng_state(state)
        trainable_modules = [network]
        if decoder is None:
            trainable_modules.append(projector)
        else:
            trainable_modules.append(decoder)
        parameters = [
            parameter
            for module in trainable_modules
            for parameter in module.parameters()
        ]
        optimizer = torch.optim.AdamW(
            parameters,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        sigreg_generator = torch.Generator(device="cpu").manual_seed(
            self.config.sigreg_seed
        )
        metrics = []
        network.train()
        projector.train()
        if decoder is not None:
            decoder.train()
        for step in range(self.config.steps):
            learning_rate = _learning_rate(self.config, step)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            anchors = anchor_schedule.batch(step)
            source = windows.histories[anchors.indices]
            views = view_schedule.batch(source, step=step)
            optimizer.zero_grad(set_to_none=True)
            pooled = []
            hidden_by_view = []
            for view_position in range(len(views.view_names)):
                present = views.present_tokens[view_position]
                positions = np.arange(
                    20 * len(windows.graph.entities), dtype=np.int64
                )
                values = torch.as_tensor(
                    views.values[view_position], dtype=torch.float32
                )
                visible = torch.as_tensor(
                    views.visible_tokens[view_position], dtype=torch.bool
                )
                present_tensor = torch.as_tensor(
                    present, dtype=torch.bool
                )
                hidden = network(
                    values, visible, present_tensor, positions
                )
                selected_visible = visible.reshape(
                    len(source), -1
                )[:, positions]
                denominator = selected_visible.sum(dim=1).clamp_min(1)
                view_pooled = (
                    hidden
                    * selected_visible.to(hidden.dtype).unsqueeze(-1)
                ).sum(dim=1) / denominator.unsqueeze(-1)
                pooled.append(view_pooled)
                hidden_by_view.append(hidden)
            if decoder is None:
                embeddings = torch.stack(
                    [projector(value) for value in pooled], dim=0
                )
                if self.config.objective == "lejepa":
                    components = complete_lejepa_loss(
                        embeddings,
                        generator=sigreg_generator,
                        sketch_dimension=self.config.sketch_dimension,
                        knot_count=self.config.knot_count,
                    )
                    loss = components.loss
                    sigreg = components.sigreg
                    invariance = components.invariance
                else:
                    global_mean = embeddings[:2].mean(dim=0)
                    invariance = (
                        (embeddings - global_mean) ** 2
                    ).mean()
                    if self.config.objective == "sigreg_only":
                        sigreg = sketched_isotropic_gaussian_regularization(
                            embeddings,
                            generator=sigreg_generator,
                            sketch_dimension=self.config.sketch_dimension,
                            knot_count=self.config.knot_count,
                        )
                        loss = 0.05 * sigreg
                    else:
                        sigreg = embeddings.new_zeros(())
                        loss = 0.95 * invariance
                reconstruction = embeddings.new_zeros(())
            else:
                target = torch.as_tensor(
                    np.where(
                        ownership[None, None], source, 0.0
                    ).reshape(len(source), -1, source.shape[-1]),
                    dtype=torch.float32,
                )
                owned = torch.as_tensor(
                    np.broadcast_to(
                        ownership[None, None],
                        source.shape,
                    ).reshape(len(source), -1, source.shape[-1]),
                    dtype=torch.bool,
                )
                reconstruction_losses = []
                for hidden in hidden_by_view:
                    prediction = decoder(hidden)
                    reconstruction_losses.append(
                        ((prediction[owned] - target[owned]) ** 2).mean()
                    )
                reconstruction = torch.stack(
                    reconstruction_losses
                ).mean()
                loss = reconstruction
                sigreg = loss.new_zeros(())
                invariance = loss.new_zeros(())
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("complete LeJEPA training became non-finite")
            loss.backward()
            optimizer.step()
            metrics.append(
                {
                    "step": float(step + 1),
                    "learning_rate": float(learning_rate),
                    "loss": float(loss.detach()),
                    "sigreg": float(sigreg.detach()),
                    "invariance": float(invariance.detach()),
                    "reconstruction": float(reconstruction.detach()),
                    "independent_samples": float(len(source)),
                    "view_count": float(len(views.view_names)),
                }
            )
        self._graph = windows.graph
        self._feature_names = windows.state_feature_names
        self._ownership_mask = ownership.copy()
        self._network = network.eval()
        self._projector = projector.eval()
        self._decoder = decoder.eval() if decoder is not None else None
        self._sigreg_generator_state = (
            sigreg_generator.get_state().tolist()
        )
        self._training_metrics = tuple(metrics)
        return self

    def encode(
        self,
        histories: NDArray[np.float64],
        graph: DeclaredTelemetryGraph,
    ) -> EncodedTelemetry:
        """Encode complete, unaugmented contexts as anchor-time tokens."""

        (
            fitted_graph,
            feature_names,
            ownership,
            network,
        ) = self._fitted_values()
        source = np.asarray(histories, dtype=np.float64)
        if (
            graph.to_dict() != fitted_graph.to_dict()
            or source.ndim != 4
            or source.shape[1:] != (
                20,
                len(fitted_graph.entities),
                len(feature_names),
            )
            or not np.all(np.isfinite(source))
        ):
            raise ValueError("complete LeJEPA encoding input is invalid")
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

    def diagnose_training_views(
        self,
        histories: NDArray[np.float64],
        graph: DeclaredTelemetryGraph,
        *,
        step: int,
    ) -> NDArray[np.float64]:
        """Return projector embeddings for fixed post-fit view diagnostics."""

        fitted_graph, _, ownership, network = self._fitted_values()
        if graph.to_dict() != fitted_graph.to_dict():
            raise ValueError("diagnostic graph differs from fitted graph")
        views = TelemetryViewSchedule(
            graph=graph,
            ownership_mask=ownership,
            varying_entity_mask=np.any(ownership, axis=1),
            seed=self.config.view_seed,
        ).batch(np.asarray(histories, dtype=np.float64), step=step)
        torch = _require_torch()
        embeddings = []
        positions = np.arange(20 * len(graph.entities), dtype=np.int64)
        with torch.no_grad():
            for view_position in range(len(views.view_names)):
                visible = torch.as_tensor(
                    views.visible_tokens[view_position], dtype=torch.bool
                )
                hidden = network(
                    torch.as_tensor(
                        views.values[view_position], dtype=torch.float32
                    ),
                    visible,
                    torch.as_tensor(
                        views.present_tokens[view_position],
                        dtype=torch.bool,
                    ),
                    positions,
                )
                selected_visible = visible.reshape(len(histories), -1)
                pooled = (
                    hidden
                    * selected_visible.to(hidden.dtype).unsqueeze(-1)
                ).sum(dim=1) / selected_visible.sum(
                    dim=1
                ).clamp_min(1).unsqueeze(-1)
                embeddings.append(self._projector(pooled))
        return (  # type: ignore[no-any-return]
            torch.stack(embeddings)
            .cpu()
            .numpy()
            .astype(np.float64)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the restorable public encoder and training evidence."""

        graph, feature_names, ownership, network = self._fitted_values()
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "config": self.config.to_dict(),
            "graph": graph.to_dict(),
            "feature_names": list(feature_names),
            "ownership_mask": ownership.astype(int).tolist(),
            "network_state": _module_state(network),
            "projector_state": _module_state(self._projector),
            "decoder_state": (
                _module_state(self._decoder)
                if self._decoder is not None
                else None
            ),
            "sigreg_generator_state": list(
                self._sigreg_generator_state or ()
            ),
            "training_metrics": [
                dict(row) for row in self._training_metrics
            ],
            "inference_parameter_count": self.inference_parameter_count,
            "training_only_parameter_count": (
                self.training_only_parameter_count
            ),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "CompleteLejepaRepresentation":
        """Restore a fitted public encoder artifact."""

        if (
            payload.get("schema_version") != cls.schema_version
            or payload.get("kind") != cls.kind
        ):
            raise ValueError("unsupported complete LeJEPA artifact")
        config = CompleteLejepaConfig.from_dict(
            dict(payload["config"])
        )
        graph = DeclaredTelemetryGraph.from_dict(dict(payload["graph"]))
        feature_names = tuple(str(value) for value in payload["feature_names"])
        ownership = np.asarray(
            payload["ownership_mask"], dtype=np.bool_
        )
        torch = _require_torch()
        model = cls(config)
        network = _build_backbone(
            torch,
            feature_count=len(feature_names),
            graph=graph,
            config=config,
        )
        _restore_module(network, dict(payload["network_state"]))
        projector = torch.nn.Sequential(
            torch.nn.Linear(config.width, config.projector_width),
            torch.nn.GELU(),
            torch.nn.Linear(config.projector_width, config.width),
        )
        _restore_module(projector, dict(payload["projector_state"]))
        decoder = None
        if payload.get("decoder_state") is not None:
            decoder = torch.nn.Sequential(
                torch.nn.Linear(config.width, config.width),
                torch.nn.GELU(),
                torch.nn.Linear(config.width, len(feature_names)),
            )
            _restore_module(decoder, dict(payload["decoder_state"]))
        model._graph = graph
        model._feature_names = feature_names
        model._ownership_mask = ownership
        model._network = network.eval()
        model._projector = projector.eval()
        model._decoder = decoder.eval() if decoder is not None else None
        model._sigreg_generator_state = tuple(
            int(value) for value in payload["sigreg_generator_state"]
        )
        model._training_metrics = tuple(
            {
                str(key): float(value)
                for key, value in dict(row).items()
            }
            for row in payload["training_metrics"]
        )
        if (
            model.inference_parameter_count
            != int(payload["inference_parameter_count"])
            or model.training_only_parameter_count
            != int(payload["training_only_parameter_count"])
        ):
            raise ValueError("complete LeJEPA parameter identity mismatch")
        return model

    def _content_sha256(self) -> str:
        artifact = self.to_dict()
        return _canonical_sha256(
            {
                "config": artifact["config"],
                "graph": artifact["graph"],
                "feature_names": artifact["feature_names"],
                "ownership_mask": artifact["ownership_mask"],
                "network_state": artifact["network_state"],
            }
        )

    def _fitted_values(
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
            or self._projector is None
        ):
            raise RuntimeError("complete LeJEPA representation is not fitted")
        return (
            self._graph,
            self._feature_names,
            self._ownership_mask,
            self._network,
        )


class EntityPcaRepresentation:
    """Fit-only deterministic entity-preserving width-matched PCA."""

    kind = "entity_preserving_pca_representation"
    schema_version = 1

    def __init__(self, *, width: int = 64) -> None:
        if isinstance(width, bool) or not isinstance(width, int) or width < 1:
            raise ValueError("entity PCA width must be positive")
        self.width = width
        self._graph: Optional[DeclaredTelemetryGraph] = None
        self._feature_names: Optional[Tuple[str, ...]] = None
        self._ownership_mask: Optional[NDArray[np.bool_]] = None
        self._centers: Optional[Tuple[NDArray[np.float64], ...]] = None
        self._components: Optional[Tuple[NDArray[np.float64], ...]] = None

    def fit(
        self, windows: ActionConditionedWindows
    ) -> "EntityPcaRepresentation":
        ownership = fit_owned_feature_mask(windows)
        centers = []
        components = []
        for entity_position in range(len(windows.entity_names)):
            mask = ownership[entity_position]
            values = windows.histories[
                :, :, entity_position, :
            ][:, :, mask].reshape(len(windows.histories), -1)
            center = np.mean(values, axis=0)
            if values.shape[1]:
                _, _, right = np.linalg.svd(
                    values - center, full_matrices=False
                )
                component_count = min(self.width, len(right))
                local_components = right[:component_count].copy()
                _orient_components(local_components)
            else:
                local_components = np.zeros((0, 0), dtype=np.float64)
            centers.append(center)
            components.append(local_components)
        self._graph = windows.graph
        self._feature_names = windows.state_feature_names
        self._ownership_mask = ownership
        self._centers = tuple(centers)
        self._components = tuple(components)
        return self

    def encode(
        self,
        histories: NDArray[np.float64],
        graph: DeclaredTelemetryGraph,
    ) -> EncodedTelemetry:
        graph_, feature_names, ownership, centers, components = (
            self._fitted_values()
        )
        values = np.asarray(histories, dtype=np.float64)
        if (
            graph.to_dict() != graph_.to_dict()
            or values.ndim != 4
            or values.shape[1:] != (
                20,
                len(graph_.entities),
                len(feature_names),
            )
        ):
            raise ValueError("entity PCA encoding input is invalid")
        tokens = np.zeros(
            (len(values), len(graph_.entities), self.width),
            dtype=np.float64,
        )
        for entity_position in range(len(graph_.entities)):
            local = values[
                :, :, entity_position, :
            ][:, :, ownership[entity_position]].reshape(len(values), -1)
            local_components = components[entity_position]
            component_count = len(local_components)
            if component_count:
                tokens[:, entity_position, :component_count] = (
                    local - centers[entity_position]
                ) @ local_components.T
        return EncodedTelemetry(
            tokens=tokens,
            entity_ids=graph_.entity_ids,
            ownership_mask=ownership.copy(),
            observation_mask=ownership.copy(),
            content_sha256=_canonical_sha256(self.to_dict()),
            graph_sha256=_canonical_sha256(graph_.to_dict()),
            state_schema_sha256=_canonical_sha256(
                {"feature_names": list(feature_names)}
            ),
            preprocessing_sha256=_canonical_sha256(
                {
                    "protocol": (
                        "action_conditioned_jepa_topology_transfer_v1"
                    )
                }
            ),
            encoder_sha256=_canonical_sha256(self.to_dict()),
        )

    def to_dict(self) -> Dict[str, Any]:
        graph, feature_names, ownership, centers, components = (
            self._fitted_values()
        )
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "width": self.width,
            "graph": graph.to_dict(),
            "feature_names": list(feature_names),
            "ownership_mask": ownership.astype(int).tolist(),
            "centers": [value.tolist() for value in centers],
            "components": [value.tolist() for value in components],
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "EntityPcaRepresentation":
        if (
            payload.get("schema_version") != cls.schema_version
            or payload.get("kind") != cls.kind
        ):
            raise ValueError("unsupported entity PCA artifact")
        model = cls(width=int(payload["width"]))
        model._graph = DeclaredTelemetryGraph.from_dict(
            dict(payload["graph"])
        )
        model._feature_names = tuple(
            str(value) for value in payload["feature_names"]
        )
        model._ownership_mask = np.asarray(
            payload["ownership_mask"], dtype=np.bool_
        )
        model._centers = tuple(
            np.asarray(value, dtype=np.float64)
            for value in payload["centers"]
        )
        restored_components = []
        for position, value in enumerate(payload["components"]):
            feature_count = len(model._centers[position])
            array = np.asarray(value, dtype=np.float64)
            restored_components.append(
                array.reshape(-1, feature_count)
                if feature_count
                else np.zeros((0, 0), dtype=np.float64)
            )
        model._components = tuple(restored_components)
        model._fitted_values()
        return model

    def _fitted_values(
        self,
    ) -> Tuple[
        DeclaredTelemetryGraph,
        Tuple[str, ...],
        NDArray[np.bool_],
        Tuple[NDArray[np.float64], ...],
        Tuple[NDArray[np.float64], ...],
    ]:
        if (
            self._graph is None
            or self._feature_names is None
            or self._ownership_mask is None
            or self._centers is None
            or self._components is None
        ):
            raise RuntimeError("entity PCA representation is not fitted")
        return (
            self._graph,
            self._feature_names,
            self._ownership_mask,
            self._centers,
            self._components,
        )


class ReducedRankActionProbe:
    """Fit-only intercept-bearing reduced-rank ridge evaluation probe."""

    kind = "reduced_rank_action_representation_probe"
    schema_version = 1

    def __init__(self, *, rank: int = 32, ridge: float = 1e-3) -> None:
        if (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank < 1
            or isinstance(ridge, bool)
            or not isinstance(ridge, (int, float))
            or not float(ridge) > 0.0
        ):
            raise ValueError("reduced-rank action probe controls are invalid")
        self.rank = rank
        self.ridge = float(ridge)
        self._input_shape: Optional[Tuple[int, int, int, int]] = None
        self._target_shape: Optional[Tuple[int, int, int]] = None
        self._input_mask: Optional[NDArray[np.bool_]] = None
        self._target_mask: Optional[NDArray[np.bool_]] = None
        self._input_center: Optional[NDArray[np.float64]] = None
        self._input_scale: Optional[NDArray[np.float64]] = None
        self._target_center: Optional[NDArray[np.float64]] = None
        self._target_scale: Optional[NDArray[np.float64]] = None
        self._coefficients: Optional[NDArray[np.float64]] = None
        self._fitted_rank = 0

    @property
    def fitted_rank(self) -> int:
        self._fitted_values()
        return self._fitted_rank

    def fit(
        self,
        representations: NDArray[np.float64],
        future_controls: NDArray[np.float64],
        future_actions: NDArray[np.float64],
        future_states: NDArray[np.float64],
    ) -> "ReducedRankActionProbe":
        """Fit all transforms and the rank-constrained map on one role."""

        tokens = np.asarray(representations, dtype=np.float64)
        controls = np.asarray(future_controls, dtype=np.float64)
        actions = np.asarray(future_actions, dtype=np.float64)
        targets = np.asarray(future_states, dtype=np.float64)
        sample_count = len(tokens)
        if (
            tokens.ndim != 3
            or controls.ndim != 3
            or actions.ndim != 4
            or targets.ndim != 4
            or any(
                len(value) != sample_count
                for value in (controls, actions, targets)
            )
            or not all(
                np.all(np.isfinite(value))
                for value in (tokens, controls, actions, targets)
            )
        ):
            raise ValueError("reduced-rank action probe arrays do not align")
        raw_input = np.concatenate(
            (
                tokens.reshape(sample_count, -1),
                controls.reshape(sample_count, -1),
                actions.reshape(sample_count, -1),
            ),
            axis=1,
        )
        raw_target = targets.reshape(sample_count, -1)
        input_center = np.mean(raw_input, axis=0)
        input_scale = np.std(raw_input, axis=0)
        input_mask = input_scale > 1e-12
        target_center = np.mean(raw_target, axis=0)
        target_scale = np.std(raw_target, axis=0)
        target_mask = target_scale > 1e-12
        normalized_input = (
            raw_input[:, input_mask] - input_center[input_mask]
        ) / input_scale[input_mask]
        normalized_target = (
            raw_target[:, target_mask] - target_center[target_mask]
        ) / target_scale[target_mask]
        if normalized_input.shape[1] and normalized_target.shape[1]:
            gram = normalized_input.T @ normalized_input
            coefficients = np.linalg.solve(
                gram
                + self.ridge
                * np.eye(gram.shape[0], dtype=np.float64),
                normalized_input.T @ normalized_target,
            )
            fitted = normalized_input @ coefficients
            _, singular_values, right = np.linalg.svd(
                fitted, full_matrices=False
            )
            numerical_rank = int(
                np.sum(
                    singular_values
                    > (
                        singular_values[0] * 1e-10
                        if len(singular_values)
                        else 0.0
                    )
                )
            )
            fitted_rank = min(self.rank, numerical_rank)
            if fitted_rank:
                basis = right[:fitted_rank].T
                coefficients = coefficients @ basis @ basis.T
            else:
                coefficients = np.zeros_like(coefficients)
        else:
            coefficients = np.zeros(
                (normalized_input.shape[1], normalized_target.shape[1]),
                dtype=np.float64,
            )
            fitted_rank = 0
        self._input_shape = (
            tokens.shape[1],
            tokens.shape[2],
            controls.shape[1] * controls.shape[2],
            actions.shape[1] * actions.shape[2] * actions.shape[3],
        )
        self._target_shape = (
            int(targets.shape[1]),
            int(targets.shape[2]),
            int(targets.shape[3]),
        )
        self._input_mask = input_mask
        self._target_mask = target_mask
        self._input_center = input_center
        self._input_scale = input_scale
        self._target_center = target_center
        self._target_scale = target_scale
        self._coefficients = coefficients
        self._fitted_rank = fitted_rank
        return self

    def predict(
        self,
        representations: NDArray[np.float64],
        future_controls: NDArray[np.float64],
        future_actions: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Predict the complete future observable trajectory."""

        (
            input_shape,
            target_shape,
            input_mask,
            target_mask,
            input_center,
            input_scale,
            target_center,
            target_scale,
            coefficients,
        ) = self._fitted_values()
        tokens = np.asarray(representations, dtype=np.float64)
        controls = np.asarray(future_controls, dtype=np.float64)
        actions = np.asarray(future_actions, dtype=np.float64)
        sample_count = len(tokens)
        if (
            tokens.shape[1:] != input_shape[:2]
            or controls.ndim != 3
            or controls.shape[0] != sample_count
            or int(np.prod(controls.shape[1:])) != input_shape[2]
            or actions.ndim != 4
            or actions.shape[0] != sample_count
            or int(np.prod(actions.shape[1:])) != input_shape[3]
        ):
            raise ValueError("reduced-rank action probe input is invalid")
        raw = np.concatenate(
            (
                tokens.reshape(sample_count, -1),
                controls.reshape(sample_count, -1),
                actions.reshape(sample_count, -1),
            ),
            axis=1,
        )
        normalized = (
            raw[:, input_mask] - input_center[input_mask]
        ) / input_scale[input_mask]
        output = np.broadcast_to(
            target_center, (sample_count, len(target_center))
        ).copy()
        if np.any(target_mask):
            output[:, target_mask] = (
                normalized @ coefficients
            ) * target_scale[target_mask] + target_center[target_mask]
        return output.reshape((sample_count,) + target_shape)

    def to_dict(self) -> Dict[str, Any]:
        values = self._fitted_values()
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "rank": self.rank,
            "ridge": self.ridge,
            "fitted_rank": self._fitted_rank,
            "input_shape": list(values[0]),
            "target_shape": list(values[1]),
            "input_mask": values[2].astype(int).tolist(),
            "target_mask": values[3].astype(int).tolist(),
            "input_center": values[4].tolist(),
            "input_scale": values[5].tolist(),
            "target_center": values[6].tolist(),
            "target_scale": values[7].tolist(),
            "coefficients": values[8].tolist(),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "ReducedRankActionProbe":
        if (
            payload.get("schema_version") != cls.schema_version
            or payload.get("kind") != cls.kind
        ):
            raise ValueError("unsupported reduced-rank action probe")
        model = cls(
            rank=int(payload["rank"]), ridge=float(payload["ridge"])
        )
        restored_input_shape = tuple(
            int(value) for value in payload["input_shape"]
        )
        restored_target_shape = tuple(
            int(value) for value in payload["target_shape"]
        )
        if len(restored_input_shape) != 4 or len(restored_target_shape) != 3:
            raise ValueError("reduced-rank action probe shapes are invalid")
        model._input_shape = (
            restored_input_shape[0],
            restored_input_shape[1],
            restored_input_shape[2],
            restored_input_shape[3],
        )
        model._target_shape = (
            restored_target_shape[0],
            restored_target_shape[1],
            restored_target_shape[2],
        )
        model._input_mask = np.asarray(
            payload["input_mask"], dtype=np.bool_
        )
        model._target_mask = np.asarray(
            payload["target_mask"], dtype=np.bool_
        )
        model._input_center = np.asarray(
            payload["input_center"], dtype=np.float64
        )
        model._input_scale = np.asarray(
            payload["input_scale"], dtype=np.float64
        )
        model._target_center = np.asarray(
            payload["target_center"], dtype=np.float64
        )
        model._target_scale = np.asarray(
            payload["target_scale"], dtype=np.float64
        )
        coefficient_rows = int(np.sum(model._input_mask))
        coefficient_columns = int(np.sum(model._target_mask))
        model._coefficients = np.asarray(
            payload["coefficients"], dtype=np.float64
        ).reshape(coefficient_rows, coefficient_columns)
        model._fitted_rank = int(payload["fitted_rank"])
        model._fitted_values()
        if model._fitted_rank > model.rank:
            raise ValueError("restored probe rank exceeds its contract")
        return model

    def _fitted_values(
        self,
    ) -> Tuple[
        Tuple[int, int, int, int],
        Tuple[int, int, int],
        NDArray[np.bool_],
        NDArray[np.bool_],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        values = (
            self._input_shape,
            self._target_shape,
            self._input_mask,
            self._target_mask,
            self._input_center,
            self._input_scale,
            self._target_center,
            self._target_scale,
            self._coefficients,
        )
        if any(value is None for value in values):
            raise RuntimeError("reduced-rank action probe is not fitted")
        return values  # type: ignore[return-value]


def assess_complete_lejepa_gates(
    *,
    forecast_scores: Mapping[str, Mapping[str, Mapping[str, float]]],
    raw_scores: Mapping[str, Mapping[str, float]],
    state_probes: Mapping[str, Mapping[str, Any]],
    attribution: Mapping[str, Mapping[str, float]],
    action_sanity: Mapping[str, Mapping[str, float]],
    restoration_parity: Mapping[str, bool],
    ridge_curves: Mapping[str, Sequence[Mapping[str, Any]]],
    selected_ridges: Mapping[str, float],
    transfer_pair_errors: Mapping[str, Mapping[str, float]],
    protocol_checks: Mapping[str, bool],
) -> Mapping[str, Any]:
    """Purely recompute the frozen safety and representation-value gates."""

    names = COMPLETE_LEJEPA_REPRESENTATION_NAMES
    if (
        set(forecast_scores) != set(names)
        or set(state_probes) != set(names)
        or set(attribution) != set(names)
        or set(action_sanity) != set(names)
        or set(restoration_parity) != set(names)
        or set(ridge_curves) != set(names)
        or set(selected_ridges) != set(names)
        or set(transfer_pair_errors) != set(names)
    ):
        raise ValueError("complete LeJEPA assessment evidence is incomplete")
    candidate = forecast_scores["complete_lejepa"]
    transfer = candidate["transfer_evaluation"]
    raw_transfer = raw_scores["transfer_evaluation"]
    pca_state = state_probes["matched_pca"]
    candidate_state = state_probes["complete_lejepa"]
    varying_entities = [
        name
        for name, row in candidate_state["entities"].items()
        if row["nrmse"] is not None
    ]
    selected_rows = {
        name: min(
            (
                [row for row in ridge_curves[name] if row["raw_safe"]]
                or list(ridge_curves[name])
            ),
            key=lambda row: (
                float(row["downstream_effect_mse"]),
                float(row["ridge"]),
            ),
        )
        for name in names
    }
    selection_verified = all(
        float(selected_ridges[name])
        == float(selected_rows[name]["ridge"])
        for name in names
    )
    safety = {
        "pair_blocked_schedule_is_valid": bool(
            protocol_checks.get("pair_blocked_schedule_is_valid", False)
        ),
        "telemetry_view_schedule_is_valid": bool(
            protocol_checks.get("telemetry_view_schedule_is_valid", False)
        ),
        "evidence_arrays_are_finite": bool(
            protocol_checks.get("evidence_arrays_are_finite", False)
        ),
        "all_public_outputs_restore": all(restoration_parity.values()),
        "selection_only_ridge_choice_recomputes": selection_verified,
        "every_representation_has_raw_safe_ridge": all(
            any(bool(row["raw_safe"]) for row in ridge_curves[name])
            for name in names
        ),
        "aggregate_state_probe_within_1_05_pca": (
            float(candidate_state["aggregate_nrmse"])
            <= 1.05 * float(pca_state["aggregate_nrmse"])
        ),
        "every_entity_state_probe_within_1_15_pca": all(
            float(candidate_state["entities"][name]["nrmse"])
            <= 1.15 * float(pca_state["entities"][name]["nrmse"])
            for name in varying_entities
        ),
        "overall_mse_within_1_05_raw": (
            transfer["overall_mse"] <= 1.05 * raw_transfer["overall_mse"]
        ),
        "action_overlap_mse_within_1_05_raw": (
            transfer["action_overlap_mse"]
            <= 1.05 * raw_transfer["action_overlap_mse"]
        ),
        "downstream_effect_mse_within_1_05_raw": (
            transfer["downstream_effect_mse"]
            <= 1.05 * raw_transfer["downstream_effect_mse"]
        ),
        "action_and_target_hit_at_1_at_least_0_95": (
            attribution["complete_lejepa"]["action_and_target_hit_at_1"]
            >= 0.95
        ),
        "no_action_specificity_is_1": (
            attribution["complete_lejepa"]["no_action_specificity"] == 1.0
        ),
        "correct_action_beats_both_at_least_0_80": (
            action_sanity["complete_lejepa"][
                "correct_action_beats_both_fraction"
            ]
            >= 0.80
        ),
    }
    controls = names[1:]
    selection_candidate = candidate["selection"][
        "downstream_effect_mse"
    ]
    best_control = min(
        controls,
        key=lambda name: forecast_scores[name][
            "transfer_evaluation"
        ]["downstream_effect_mse"],
    )
    best_transfer = forecast_scores[best_control][
        "transfer_evaluation"
    ]["downstream_effect_mse"]
    common_pairs = sorted(
        set(transfer_pair_errors["complete_lejepa"])
        & set(transfer_pair_errors[best_control])
    )
    if not common_pairs:
        raise ValueError("complete LeJEPA pair evidence is empty")
    win_fraction = float(
        np.mean(
            [
                transfer_pair_errors["complete_lejepa"][pair]
                < transfer_pair_errors[best_control][pair]
                for pair in common_pairs
            ]
        )
    )
    value = {
        "selection_strictly_best_of_all_controls": all(
            selection_candidate
            < forecast_scores[name]["selection"][
                "downstream_effect_mse"
            ]
            for name in controls
        ),
        "transfer_improves_best_control_by_5_percent": (
            transfer["downstream_effect_mse"] <= 0.95 * best_transfer
        ),
        "per_pair_win_fraction_at_least_0_60": win_fraction >= 0.60,
        "evaluation_did_not_select_configuration": True,
    }
    passed = all(safety.values()) and all(value.values())
    return {
        "safety_gates": safety,
        "value_gates": value,
        "safety_passed": all(safety.values()),
        "value_passed": all(value.values()),
        "passed": passed,
        "decision": (
            "advance_to_fixed_seed_representation_robustness"
            if passed
            else "reject_exact_complete_multi_view_lejepa_recipe"
        ),
        "best_transfer_control": best_control,
        "candidate_pair_win_fraction": win_fraction,
    }


def build_complete_lejepa_backbone(
    *,
    feature_count: int,
    graph: DeclaredTelemetryGraph,
    config: CompleteLejepaConfig,
) -> Any:
    """Build the shared graph-biased telemetry transformer."""

    return _build_backbone(
        _require_torch(),
        feature_count=feature_count,
        graph=graph,
        config=config,
    )


def build_telemetry_backbone(
    torch: Any,
    *,
    feature_count: int,
    graph: DeclaredTelemetryGraph,
    config: Any,
) -> Any:
    """Build the shared entity-preserving telemetry view encoder."""

    return _build_backbone(
        torch,
        feature_count=feature_count,
        graph=graph,
        config=config,
    )


def _build_backbone(
    torch: Any,
    *,
    feature_count: int,
    graph: DeclaredTelemetryGraph,
    config: Any,
) -> Any:
    """Build the fixed pre-norm graph-biased telemetry transformer."""

    entity_count = len(graph.entities)
    kind_ids = torch.as_tensor(
        [0 if entity.kind == "node" else 1 for entity in graph.entities],
        dtype=torch.long,
    )
    relation_categories, distances = _graph_pair_metadata(graph)

    class _Block(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.norm1 = torch.nn.LayerNorm(
                config.width, eps=1e-5
            )
            self.qkv = torch.nn.Linear(
                config.width, 3 * config.width
            )
            self.output = torch.nn.Linear(config.width, config.width)
            self.norm2 = torch.nn.LayerNorm(
                config.width, eps=1e-5
            )
            self.feedforward = torch.nn.Sequential(
                torch.nn.Linear(
                    config.width, config.feedforward_width
                ),
                torch.nn.GELU(),
                torch.nn.Linear(
                    config.feedforward_width, config.width
                ),
            )

        def forward(self, values: Any, bias: Any) -> Any:
            batch, token_count, _ = values.shape
            normalized = self.norm1(values)
            qkv = self.qkv(normalized).reshape(
                batch,
                token_count,
                3,
                config.head_count,
                config.width // config.head_count,
            )
            query, key, value = qkv.unbind(dim=2)
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            scores = (
                query @ key.transpose(-1, -2)
            ) / math.sqrt(config.width // config.head_count)
            attention = torch.softmax(scores + bias[None], dim=-1)
            attended = (
                attention @ value
            ).transpose(1, 2).reshape(batch, token_count, config.width)
            values = values + self.output(attended)
            return values + self.feedforward(self.norm2(values))

    class _Backbone(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.feature_weight = torch.nn.Parameter(
                torch.empty(entity_count, feature_count, config.width)
            )
            self.feature_bias = torch.nn.Parameter(
                torch.zeros(entity_count, config.width)
            )
            torch.nn.init.xavier_uniform_(self.feature_weight)
            self.mask_embedding = torch.nn.Parameter(
                torch.zeros(entity_count, config.width)
            )
            torch.nn.init.normal_(self.mask_embedding, std=0.02)
            self.time_embedding = torch.nn.Embedding(20, config.width)
            self.entity_embedding = torch.nn.Embedding(
                entity_count, config.width
            )
            self.kind_embedding = torch.nn.Embedding(2, config.width)
            self.presence_embedding = torch.nn.Embedding(2, config.width)
            self.relation_bias = torch.nn.Parameter(
                torch.zeros(
                    int(np.max(relation_categories)) + 1,
                    config.head_count,
                )
            )
            self.distance_bias = torch.nn.Parameter(
                torch.zeros(
                    int(np.max(distances)) + 1, config.head_count
                )
            )
            self.blocks = torch.nn.ModuleList(
                [_Block() for _ in range(config.block_count)]
            )
            self.register_buffer("kind_ids", kind_ids)
            self.register_buffer(
                "relation_categories",
                torch.as_tensor(relation_categories, dtype=torch.long),
            )
            self.register_buffer(
                "graph_distances",
                torch.as_tensor(distances, dtype=torch.long),
            )

        def full_position_embeddings(self) -> Any:
            time_ids = torch.arange(20).repeat_interleave(entity_count)
            entity_ids = torch.arange(entity_count).repeat(20)
            return (
                self.time_embedding(time_ids)
                + self.entity_embedding(entity_ids)
                + self.kind_embedding(self.kind_ids[entity_ids])
            )

        def forward(
            self,
            values: Any,
            visible: Any,
            present: Any,
            positions: Any,
        ) -> Any:
            projected = torch.einsum(
                "btef,efw->btew", values, self.feature_weight
            ) + self.feature_bias[None, None]
            masked = self.mask_embedding[None, None].expand_as(projected)
            token_values = torch.where(
                visible[..., None], projected, masked
            )
            flat = token_values.reshape(
                len(values), 20 * entity_count, config.width
            )
            position_tensor = torch.as_tensor(
                positions, dtype=torch.long, device=flat.device
            )
            entity_ids = position_tensor % entity_count
            time_ids = position_tensor // entity_count
            selected_visible = visible.reshape(
                len(values), -1
            )[:, position_tensor]
            selected_present = present.reshape(
                len(values), -1
            )[:, position_tensor]
            hidden = (
                flat[:, position_tensor]
                + self.time_embedding(time_ids)[None]
                + self.entity_embedding(entity_ids)[None]
                + self.kind_embedding(self.kind_ids[entity_ids])[None]
                + self.presence_embedding(
                    selected_present.to(torch.long)
                )
            )
            relations = self.relation_categories[
                entity_ids[:, None], entity_ids[None, :]
            ]
            graph_distances = self.graph_distances[
                entity_ids[:, None], entity_ids[None, :]
            ]
            bias = (
                self.relation_bias[relations]
                + self.distance_bias[graph_distances]
            ).permute(2, 0, 1)
            for block in self.blocks:
                hidden = block(hidden, bias)
            return hidden

    return _Backbone()


def _graph_pair_metadata(
    graph: DeclaredTelemetryGraph,
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
    adjacency = _entity_adjacency(graph)
    entity_count = len(graph.entities)
    relation_types = {
        value: position + 2
        for position, value in enumerate(
            dict.fromkeys(
                entity.entity_type
                for entity in graph.entities
                if entity.kind == "edge"
            )
        )
    }
    categories = np.ones((entity_count, entity_count), dtype=np.int64)
    np.fill_diagonal(categories, 0)
    for edge_position, entity in enumerate(graph.entities):
        if entity.kind != "edge":
            continue
        category = relation_types[entity.entity_type]
        for node_position in np.flatnonzero(adjacency[edge_position]):
            if int(node_position) == edge_position:
                continue
            categories[edge_position, node_position] = category
            categories[node_position, edge_position] = category
    distances = np.full(
        (entity_count, entity_count), entity_count, dtype=np.int64
    )
    np.fill_diagonal(distances, 0)
    distances[adjacency & ~np.eye(entity_count, dtype=np.bool_)] = 1
    for middle in range(entity_count):
        distances = np.minimum(
            distances,
            distances[:, middle, None] + distances[None, middle, :],
        )
    distances = np.minimum(distances, entity_count)
    return categories, distances


def _learning_rate(config: CompleteLejepaConfig, step: int) -> float:
    warmup = min(config.warmup_steps, config.steps)
    if step < warmup:
        return config.learning_rate * float(step + 1) / float(warmup)
    remaining = config.steps - warmup
    if remaining <= 1:
        return config.minimum_learning_rate
    progress = float(step - warmup) / float(remaining - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.minimum_learning_rate + (
        config.learning_rate - config.minimum_learning_rate
    ) * cosine


def _module_state(module: Any) -> Dict[str, Any]:
    return {
        name: tensor.detach().cpu().tolist()
        for name, tensor in module.state_dict().items()
    }


def _restore_module(module: Any, payload: Mapping[str, Any]) -> None:
    torch = _require_torch()
    expected = module.state_dict()
    if set(payload) != set(expected):
        raise ValueError("complete LeJEPA tensor names do not match")
    restored = {
        name: torch.as_tensor(payload[name], dtype=tensor.dtype)
        for name, tensor in expected.items()
    }
    module.load_state_dict(restored)


def _orient_components(components: NDArray[np.float64]) -> None:
    for component in components:
        pivot = int(np.argmax(np.abs(component)))
        if component[pivot] < 0.0:
            component *= -1.0


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "complete LeJEPA representation requires the torch extra"
        ) from error
    return torch


def _entity_adjacency(
    graph: DeclaredTelemetryGraph,
) -> NDArray[np.bool_]:
    positions = {
        entity_id: position
        for position, entity_id in enumerate(graph.entity_ids)
    }
    adjacency = np.eye(len(graph.entities), dtype=np.bool_)
    for edge_position, entity in enumerate(graph.entities):
        if entity.kind != "edge":
            continue
        assert entity.source is not None
        assert entity.target is not None
        for node_id in (entity.source, entity.target):
            node_position = positions[node_id]
            adjacency[edge_position, node_position] = True
            adjacency[node_position, edge_position] = True
    return adjacency
