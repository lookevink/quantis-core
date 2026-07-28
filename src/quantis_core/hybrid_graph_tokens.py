"""NumPy-only graph tokenization and multi-mask sampling."""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .graph_telemetry import GraphStateWindows


@dataclass(frozen=True)
class AlignedEventFeatures:
    """Dense event features already aligned to graph window coordinates."""

    contexts: NDArray[np.float64]
    target_blocks: NDArray[np.float64]
    observation_mask: NDArray[np.bool_]
    feature_names: Tuple[str, ...]

    def __post_init__(self) -> None:
        if self.contexts.ndim != 4 or self.target_blocks.ndim != 5:
            raise ValueError("event features have invalid ranks")
        feature_count = self.contexts.shape[-1]
        if (
            feature_count < 1
            or self.target_blocks.shape[-1] != feature_count
            or self.observation_mask.ndim != 2
            or self.observation_mask.shape[-1] != feature_count
            or len(self.feature_names) != feature_count
            or len(set(self.feature_names)) != feature_count
            or any(not name for name in self.feature_names)
        ):
            raise ValueError("event feature metadata does not align")
        if not (
            np.all(np.isfinite(self.contexts))
            and np.all(np.isfinite(self.target_blocks))
        ):
            raise ValueError("event features must be finite")


@dataclass(frozen=True)
class HybridGraphTokens:
    """Validated fine/coarse observations and declared graph identities."""

    fine_context: NDArray[np.float64]
    fine_targets: NDArray[np.float64]
    coarse_context: NDArray[np.float64]
    coarse_targets: NDArray[np.float64]
    target_controls: NDArray[np.float64]
    control_feature_names: Tuple[str, ...]
    feature_mask: NDArray[np.bool_]
    entity_ids: NDArray[np.int64]
    entity_names: Tuple[str, ...]
    kind_ids: NDArray[np.int64]
    kind_names: Tuple[str, ...]
    entity_type_ids: NDArray[np.int64]
    entity_type_names: Tuple[str, ...]
    relation_ids: NDArray[np.int64]
    relation_names: Tuple[str, ...]
    typed_adjacency: NDArray[np.bool_]
    feature_names: Tuple[str, ...]
    local_feature_keys: Tuple[Tuple[str, ...], ...]
    point_indices: NDArray[np.int64]
    horizons: Tuple[int, ...]
    coarse_factor: int

    def __post_init__(self) -> None:
        sample_count, _, entity_count, feature_count = (
            self.fine_context.shape
        )
        if (
            self.fine_context.ndim != 4
            or self.fine_targets.ndim != 5
            or self.coarse_context.ndim != 4
            or self.coarse_targets.ndim != 5
        ):
            raise ValueError("hybrid graph token tensors have invalid ranks")
        if (
            self.fine_targets.shape[0] != sample_count
            or self.fine_targets.shape[-2:] != (
                entity_count,
                feature_count,
            )
            or self.coarse_context.shape[0] != sample_count
            or self.coarse_context.shape[-2:] != (
                entity_count,
                feature_count,
            )
            or self.coarse_targets.shape[:2]
            != self.fine_targets.shape[:2]
            or self.coarse_targets.shape[-2:] != (
                entity_count,
                feature_count,
            )
        ):
            raise ValueError("hybrid graph token tensors do not align")
        if (
            self.target_controls.shape[:2] != self.fine_targets.shape[:2]
            or len(self.control_feature_names)
            != self.target_controls.shape[-1]
            or len(set(self.control_feature_names))
            != len(self.control_feature_names)
            or any(not name for name in self.control_feature_names)
        ):
            raise ValueError("hybrid target controls do not align")
        if self.feature_mask.shape != (entity_count, feature_count):
            raise ValueError("hybrid feature mask does not align")
        if (
            self.entity_ids.shape != (entity_count,)
            or self.kind_ids.shape != (entity_count,)
            or self.entity_type_ids.shape != (entity_count,)
            or self.relation_ids.shape != (entity_count,)
            or len(self.entity_names) != entity_count
            or len(self.local_feature_keys) != entity_count
        ):
            raise ValueError("hybrid entity metadata does not align")
        if self.typed_adjacency.shape != (
            len(self.relation_names),
            entity_count,
            entity_count,
        ):
            raise ValueError("hybrid typed adjacency does not align")
        if (
            len(self.feature_names) != feature_count
            or self.point_indices.shape != (sample_count,)
            or len(self.horizons) != self.fine_targets.shape[1]
            or self.coarse_factor < 1
        ):
            raise ValueError("hybrid token metadata does not align")
        if any(
            not np.all(np.isfinite(array))
            for array in (
                self.fine_context,
                self.fine_targets,
                self.coarse_context,
                self.coarse_targets,
                self.target_controls,
            )
        ):
            raise ValueError("hybrid graph token values must be finite")


@dataclass(frozen=True)
class MultiMaskConfig:
    """Reproducible entity/time mask sampling controls."""

    mask_count: int = 2
    target_coverage: float = 0.75
    seed: int = 89

    def __post_init__(self) -> None:
        if isinstance(self.mask_count, bool) or self.mask_count < 1:
            raise ValueError("multi-mask mask count must be positive")
        if not 0.0 < self.target_coverage < 1.0:
            raise ValueError(
                "multi-mask target coverage must be between zero and one"
            )


@dataclass(frozen=True)
class MultiMaskBatch:
    """Visible context and held-out targets for every sampled mask."""

    context_masks: NDArray[np.bool_]
    target_masks: NDArray[np.bool_]

    def __post_init__(self) -> None:
        if (
            self.context_masks.ndim != 4
            or self.context_masks.shape != self.target_masks.shape
            or np.any(
                np.logical_and(
                    self.context_masks,
                    self.target_masks,
                )
            )
            or not np.all(
                np.logical_or(
                    self.context_masks,
                    self.target_masks,
                )
            )
        ):
            raise ValueError("multi-mask context and targets do not align")


def compile_hybrid_graph_tokens(
    windows: GraphStateWindows,
    *,
    coarse_factor: int = 4,
    aligned_event_features: Optional[AlignedEventFeatures] = None,
) -> HybridGraphTokens:
    """Compile graph windows into copied fine and coarse token tensors."""

    if isinstance(coarse_factor, bool) or coarse_factor < 1:
        raise ValueError("coarse factor must be positive")
    fine_context = windows.contexts.copy()
    fine_targets = windows.target_blocks.copy()
    feature_mask = windows.observation_mask.copy()
    event_feature_names: Tuple[str, ...] = ()
    if aligned_event_features is not None:
        _validate_event_alignment(windows, aligned_event_features)
        fine_context = np.concatenate(
            (fine_context, aligned_event_features.contexts),
            axis=-1,
        )
        fine_targets = np.concatenate(
            (fine_targets, aligned_event_features.target_blocks),
            axis=-1,
        )
        feature_mask = np.concatenate(
            (feature_mask, aligned_event_features.observation_mask),
            axis=-1,
        )
        event_feature_names = aligned_event_features.feature_names

    graph = windows.graph
    entity_count = len(graph.entities)
    kind_names = ("node", "edge")
    kind_positions = {
        name: position for position, name in enumerate(kind_names)
    }
    entity_type_names = _ordered_unique(
        tuple(entity.entity_type for entity in graph.entities)
    )
    entity_type_positions = {
        name: position for position, name in enumerate(entity_type_names)
    }
    relation_names = _ordered_unique(
        tuple(
            entity.entity_type
            for entity in graph.entities
            if entity.kind == "edge"
        )
    )
    relation_positions = {
        name: position for position, name in enumerate(relation_names)
    }
    entity_positions = {
        entity.entity_id: position
        for position, entity in enumerate(graph.entities)
    }
    relation_ids = np.full(entity_count, -1, dtype=np.int64)
    typed_adjacency = np.zeros(
        (len(relation_names), entity_count, entity_count),
        dtype=np.bool_,
    )
    for entity_position, entity in enumerate(graph.entities):
        if entity.kind != "edge":
            continue
        relation_id = relation_positions[entity.entity_type]
        relation_ids[entity_position] = relation_id
        assert entity.source is not None
        assert entity.target is not None
        source_position = entity_positions[entity.source]
        target_position = entity_positions[entity.target]
        typed_adjacency[
            relation_id, source_position, entity_position
        ] = True
        typed_adjacency[
            relation_id, entity_position, target_position
        ] = True

    return HybridGraphTokens(
        fine_context=fine_context,
        fine_targets=fine_targets,
        coarse_context=_pool_temporal(
            fine_context,
            temporal_axis=1,
            factor=coarse_factor,
        ),
        coarse_targets=_pool_temporal(
            fine_targets,
            temporal_axis=2,
            factor=coarse_factor,
        ),
        target_controls=windows.target_controls.copy(),
        control_feature_names=windows.control_feature_names,
        feature_mask=feature_mask,
        entity_ids=np.arange(entity_count, dtype=np.int64),
        entity_names=windows.entity_ids,
        kind_ids=np.asarray(
            [
                kind_positions[entity.kind]
                for entity in graph.entities
            ],
            dtype=np.int64,
        ),
        kind_names=kind_names,
        entity_type_ids=np.asarray(
            [
                entity_type_positions[entity.entity_type]
                for entity in graph.entities
            ],
            dtype=np.int64,
        ),
        entity_type_names=entity_type_names,
        relation_ids=relation_ids,
        relation_names=relation_names,
        typed_adjacency=typed_adjacency,
        feature_names=tuple(
            f"graph.slot.{position}"
            for position in range(windows.contexts.shape[-1])
        )
        + event_feature_names,
        local_feature_keys=windows.local_feature_keys,
        point_indices=windows.point_indices.copy(),
        horizons=windows.horizons,
        coarse_factor=coarse_factor,
    )


def sample_multi_masks(
    tokens: HybridGraphTokens,
    config: MultiMaskConfig = MultiMaskConfig(),
) -> MultiMaskBatch:
    """Sample contiguous temporal spans over connected topology blocks."""

    sample_count, time_count, entity_count, _ = (
        tokens.fine_context.shape
    )
    time_span, entity_span = _nearest_mask_dimensions(
        time_count,
        entity_count,
        config.target_coverage,
    )
    generator = np.random.default_rng(config.seed)
    weak_adjacency = np.any(tokens.typed_adjacency, axis=0)
    weak_adjacency = np.logical_or(
        weak_adjacency,
        weak_adjacency.T,
    )
    target_masks = np.zeros(
        (
            config.mask_count,
            sample_count,
            time_count,
            entity_count,
        ),
        dtype=np.bool_,
    )
    for mask_position in range(config.mask_count):
        for sample_position in range(sample_count):
            time_start = int(
                generator.integers(0, time_count - time_span + 1)
            )
            entities = _sample_connected_entities(
                weak_adjacency,
                entity_span,
                generator,
            )
            target_masks[
                mask_position,
                sample_position,
                time_start : time_start + time_span,
                entities,
            ] = True
    return MultiMaskBatch(
        context_masks=np.logical_not(target_masks),
        target_masks=target_masks,
    )


def _validate_event_alignment(
    windows: GraphStateWindows,
    event_features: AlignedEventFeatures,
) -> None:
    if event_features.contexts.shape[:3] != windows.contexts.shape[:3]:
        raise ValueError("event contexts do not align with graph windows")
    if (
        event_features.target_blocks.shape[:4]
        != windows.target_blocks.shape[:4]
    ):
        raise ValueError("event targets do not align with graph windows")
    if (
        event_features.observation_mask.shape[0]
        != windows.observation_mask.shape[0]
    ):
        raise ValueError("event observation mask does not align")


def _pool_temporal(
    values: NDArray[np.float64],
    *,
    temporal_axis: int,
    factor: int,
) -> NDArray[np.float64]:
    pooled = [
        np.mean(
            np.take(
                values,
                np.arange(start, min(start + factor, values.shape[temporal_axis])),
                axis=temporal_axis,
            ),
            axis=temporal_axis,
        )
        for start in range(0, values.shape[temporal_axis], factor)
    ]
    return np.stack(pooled, axis=temporal_axis)


def _ordered_unique(values: Tuple[str, ...]) -> Tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _nearest_mask_dimensions(
    time_count: int,
    entity_count: int,
    target_coverage: float,
) -> Tuple[int, int]:
    if time_count < 1 or entity_count < 1:
        raise ValueError("cannot mask empty graph tokens")
    candidates = [
        (time_span, entity_span)
        for time_span in range(1, time_count + 1)
        for entity_span in range(1, entity_count + 1)
        if time_span * entity_span < time_count * entity_count
    ]
    if not candidates:
        raise ValueError("multi-mask sampling requires at least two tokens")
    return min(
        candidates,
        key=lambda candidate: (
            abs(
                candidate[0]
                * candidate[1]
                / (time_count * entity_count)
                - target_coverage
            ),
            abs(candidate[0] / time_count - target_coverage**0.5),
            -candidate[1],
        ),
    )


def _sample_connected_entities(
    adjacency: NDArray[np.bool_],
    entity_count: int,
    generator: np.random.Generator,
) -> NDArray[np.int64]:
    total_entities = adjacency.shape[0]
    start = int(generator.integers(0, total_entities))
    selected = [start]
    selected_set = {start}
    frontier = [
        int(value)
        for value in np.flatnonzero(adjacency[start])
    ]
    while len(selected) < entity_count:
        if frontier:
            frontier_position = int(
                generator.integers(0, len(frontier))
            )
            candidate = frontier.pop(frontier_position)
        else:
            remaining = [
                value
                for value in range(total_entities)
                if value not in selected_set
            ]
            candidate = remaining[
                int(generator.integers(0, len(remaining)))
            ]
        if candidate in selected_set:
            continue
        selected.append(candidate)
        selected_set.add(candidate)
        for neighbor in np.flatnonzero(adjacency[candidate]):
            value = int(neighbor)
            if value not in selected_set and value not in frontier:
                frontier.append(value)
    return np.asarray(selected, dtype=np.int64)
