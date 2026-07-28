"""Declared graph ownership for contextual metrics and structured logs."""

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .contextual_multimodal_corpus import (
    ContextualMultimodalModelWindows,
)


ENTITY_KINDS = ("node", "edge")


@dataclass(frozen=True)
class GraphEntity:
    """One declared node or directed relationship in the lab graph."""

    entity_id: str
    kind: str
    entity_type: str
    source: Optional[str] = None
    target: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.entity_id:
            raise ValueError("graph entity id cannot be empty")
        if self.kind not in ENTITY_KINDS:
            raise ValueError("graph entity kind must be node or edge")
        if self.kind == "node":
            if self.source is not None or self.target is not None:
                raise ValueError("graph nodes cannot declare endpoints")
        elif not self.source or not self.target:
            raise ValueError("graph edges require source and target nodes")


@dataclass(frozen=True)
class TelemetryBinding:
    """Assign one semantic observation to exactly one graph entity."""

    feature_key: str
    entity_id: str

    def __post_init__(self) -> None:
        if not self.feature_key or "." not in self.feature_key:
            raise ValueError(
                "graph feature keys require a modality prefix"
            )
        if not self.entity_id:
            raise ValueError("graph telemetry binding requires an entity")


@dataclass(frozen=True)
class DeclaredTelemetryGraph:
    """Known operational topology and observation ownership."""

    entities: Tuple[GraphEntity, ...]
    bindings: Tuple[TelemetryBinding, ...]

    def __post_init__(self) -> None:
        entity_ids = tuple(entity.entity_id for entity in self.entities)
        if not entity_ids or len(set(entity_ids)) != len(entity_ids):
            raise ValueError("graph entity ids must be unique and nonempty")
        nodes = {
            entity.entity_id
            for entity in self.entities
            if entity.kind == "node"
        }
        for entity in self.entities:
            if entity.kind == "edge" and (
                entity.source not in nodes or entity.target not in nodes
            ):
                raise ValueError(
                    "graph edge endpoints must reference declared nodes"
                )
        feature_keys = tuple(
            binding.feature_key for binding in self.bindings
        )
        if len(set(feature_keys)) != len(feature_keys):
            raise ValueError("graph telemetry bindings must be unique")
        unknown_entities = {
            binding.entity_id
            for binding in self.bindings
            if binding.entity_id not in set(entity_ids)
        }
        if unknown_entities:
            raise ValueError(
                "graph telemetry bindings reference unknown entities: "
                f"{sorted(unknown_entities)}"
            )

    @property
    def entity_ids(self) -> Tuple[str, ...]:
        return tuple(entity.entity_id for entity in self.entities)

    def binding_map(self) -> Mapping[str, str]:
        return {
            binding.feature_key: binding.entity_id
            for binding in self.bindings
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "declared_telemetry_graph",
            "entities": [
                {
                    "entity_id": entity.entity_id,
                    "kind": entity.kind,
                    "entity_type": entity.entity_type,
                    **(
                        {
                            "source": entity.source,
                            "target": entity.target,
                        }
                        if entity.kind == "edge"
                        else {}
                    ),
                }
                for entity in self.entities
            ],
            "bindings": [
                {
                    "feature_key": binding.feature_key,
                    "entity_id": binding.entity_id,
                }
                for binding in self.bindings
            ],
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "DeclaredTelemetryGraph":
        if (
            payload.get("schema_version") != 1
            or payload.get("kind") != "declared_telemetry_graph"
        ):
            raise ValueError("unsupported declared telemetry graph")
        entities = []
        for raw in payload["entities"]:
            item = dict(raw)
            entities.append(
                GraphEntity(
                    entity_id=str(item["entity_id"]),
                    kind=str(item["kind"]),
                    entity_type=str(item["entity_type"]),
                    source=(
                        str(item["source"])
                        if item.get("source") is not None
                        else None
                    ),
                    target=(
                        str(item["target"])
                        if item.get("target") is not None
                        else None
                    ),
                )
            )
        return cls(
            entities=tuple(entities),
            bindings=tuple(
                TelemetryBinding(
                    feature_key=str(dict(raw)["feature_key"]),
                    entity_id=str(dict(raw)["entity_id"]),
                )
                for raw in payload["bindings"]
            ),
        )

    def neighboring_entity_ids(
        self, entity_id: str
    ) -> Tuple[str, ...]:
        """Return the entity plus incident nodes and relationships."""

        if entity_id not in set(self.entity_ids):
            raise ValueError(f"unknown graph entity: {entity_id}")
        entity_by_id = {
            entity.entity_id: entity for entity in self.entities
        }
        selected = {entity_id}
        selected_entity = entity_by_id[entity_id]
        if selected_entity.kind == "edge":
            assert selected_entity.source is not None
            assert selected_entity.target is not None
            selected.update(
                (selected_entity.source, selected_entity.target)
            )
        else:
            for entity in self.entities:
                if entity.kind == "edge" and entity_id in (
                    entity.source,
                    entity.target,
                ):
                    selected.add(entity.entity_id)
                    assert entity.source is not None
                    assert entity.target is not None
                    selected.update((entity.source, entity.target))
        return tuple(
            candidate
            for candidate in self.entity_ids
            if candidate in selected
        )


@dataclass(frozen=True)
class GraphStateWindows:
    """Padded node/edge observations with stable ownership metadata."""

    contexts: NDArray[np.float64]
    target_blocks: NDArray[np.float64]
    target_controls: NDArray[np.float64]
    point_indices: NDArray[np.int64]
    observation_mask: NDArray[np.bool_]
    entity_ids: Tuple[str, ...]
    entity_kinds: Tuple[str, ...]
    local_feature_keys: Tuple[Tuple[str, ...], ...]
    control_feature_names: Tuple[str, ...]
    horizons: Tuple[int, ...]
    target_block_size: int
    graph: DeclaredTelemetryGraph

    def __post_init__(self) -> None:
        sample_count = len(self.contexts)
        entity_count = len(self.entity_ids)
        slot_count = self.contexts.shape[3]
        if (
            self.contexts.ndim != 4
            or self.target_blocks.ndim != 5
            or self.target_controls.ndim != 4
        ):
            raise ValueError("graph state windows have invalid ranks")
        if self.contexts.shape[2:] != (entity_count, slot_count):
            raise ValueError("graph context entities do not align")
        if self.target_blocks.shape != (
            sample_count,
            len(self.horizons),
            self.target_block_size,
            entity_count,
            slot_count,
        ):
            raise ValueError("graph target blocks do not align")
        if self.target_controls.shape[:3] != (
            sample_count,
            len(self.horizons),
            self.target_block_size,
        ):
            raise ValueError("graph target controls do not align")
        if self.point_indices.shape != (sample_count,):
            raise ValueError("graph point indices do not align")
        if self.observation_mask.shape != (
            entity_count,
            slot_count,
        ):
            raise ValueError("graph observation mask does not align")
        if (
            len(self.entity_kinds) != entity_count
            or len(self.local_feature_keys) != entity_count
            or self.entity_ids != self.graph.entity_ids
        ):
            raise ValueError("graph entity metadata does not align")
        if any(
            not np.all(np.isfinite(array))
            for array in (
                self.contexts,
                self.target_blocks,
                self.target_controls,
            )
        ):
            raise ValueError("graph state windows must be finite")

    def feature_position(self, feature_key: str) -> Tuple[int, int]:
        for entity_position, feature_keys in enumerate(
            self.local_feature_keys
        ):
            if feature_key in feature_keys:
                return (
                    entity_position,
                    feature_keys.index(feature_key),
                )
        raise ValueError(f"unknown graph feature: {feature_key}")


def quantis_checkout_graph() -> DeclaredTelemetryGraph:
    """Return the declared checkout topology and current feature ownership."""

    entities = (
        GraphEntity("api", "node", "service"),
        GraphEntity("checkout_queue", "node", "stateful_resource"),
        GraphEntity("worker_pool", "node", "service_pool"),
        GraphEntity("redis", "node", "dependency"),
        GraphEntity("postgresql", "node", "dependency"),
        GraphEntity(
            "api_enqueues_queue",
            "edge",
            "enqueue",
            "api",
            "checkout_queue",
        ),
        GraphEntity(
            "queue_dequeues_to_worker",
            "edge",
            "dequeue",
            "checkout_queue",
            "worker_pool",
        ),
        GraphEntity(
            "queue_hosted_on_redis",
            "edge",
            "hosted_on",
            "checkout_queue",
            "redis",
        ),
        GraphEntity(
            "worker_writes_postgresql",
            "edge",
            "database_write",
            "worker_pool",
            "postgresql",
        ),
    )
    ownership = {
        "metric.request_latency_ms": "api",
        "metric.error_rate": "api",
        "metric.queue_depth": "checkout_queue",
        "metric.worker_completion_ratio": "worker_pool",
        "metric.worker_heartbeat_age_s": "worker_pool",
        "metric.db_write_completion_ratio": "postgresql",
        "log.checkout_completion_ratio": "worker_pool",
        "log.checkout_backlog_delta_ratio": "api_enqueues_queue",
        "log.checkout_rejection_rate": "api",
        "log.queue_pressure_transition_rate": "checkout_queue",
        "log.queue_high_transition_rate": "checkout_queue",
        "log.postgresql_latency_pressure_ratio": (
            "worker_writes_postgresql"
        ),
        "log.postgresql_slow_or_error_ratio": (
            "worker_writes_postgresql"
        ),
        "log.worker_activation_rate": "worker_pool",
        "log.redis_latency_pressure_rate": "redis",
        "log.redis_slow_or_error_rate": "redis",
        "log.checkout_queue_wait_pressure_ratio": (
            "queue_dequeues_to_worker"
        ),
        "log.checkout_queue_wait_slow_ratio": (
            "queue_dequeues_to_worker"
        ),
    }
    return DeclaredTelemetryGraph(
        entities=entities,
        bindings=tuple(
            TelemetryBinding(feature_key, entity_id)
            for feature_key, entity_id in ownership.items()
        ),
    )


def compile_graph_state_windows(
    windows: ContextualMultimodalModelWindows,
    graph: DeclaredTelemetryGraph,
) -> GraphStateWindows:
    """Attach every contextual observation to its declared graph owner."""

    metric_keys = tuple(
        f"metric.{name}" for name in windows.metric_feature_names
    )
    log_keys = tuple(
        f"log.{name}" for name in windows.log_feature_names
    )
    feature_keys = metric_keys + log_keys
    if len(set(feature_keys)) != len(feature_keys):
        raise ValueError("contextual telemetry feature keys must be unique")
    binding_map = graph.binding_map()
    unbound = set(feature_keys) - set(binding_map)
    unexpected = set(binding_map) - set(feature_keys)
    if unbound:
        raise ValueError(
            f"unbound telemetry features: {sorted(unbound)}"
        )
    if unexpected:
        raise ValueError(
            "graph bindings are absent from contextual telemetry: "
            f"{sorted(unexpected)}"
        )

    entity_ids = graph.entity_ids
    features_by_entity: Dict[str, list[str]] = {
        entity_id: [] for entity_id in entity_ids
    }
    for feature_key in feature_keys:
        features_by_entity[binding_map[feature_key]].append(feature_key)
    local_feature_keys = tuple(
        tuple(features_by_entity[entity_id])
        for entity_id in entity_ids
    )
    slot_count = max(
        1, max(len(features) for features in local_feature_keys)
    )
    observation_mask = np.zeros(
        (len(entity_ids), slot_count), dtype=np.bool_
    )
    contexts = np.zeros(
        (
            len(windows.metric_contexts),
            windows.metric_contexts.shape[1],
            len(entity_ids),
            slot_count,
        ),
        dtype=np.float64,
    )
    target_blocks = np.zeros(
        (
            len(windows.metric_target_blocks),
            len(windows.horizons),
            windows.target_block_size,
            len(entity_ids),
            slot_count,
        ),
        dtype=np.float64,
    )
    context_values = np.concatenate(
        (windows.metric_contexts, windows.log_contexts),
        axis=2,
    )
    target_values = np.concatenate(
        (
            windows.metric_target_blocks,
            windows.log_target_blocks,
        ),
        axis=3,
    )
    entity_positions = {
        entity_id: position
        for position, entity_id in enumerate(entity_ids)
    }
    slot_positions: Dict[str, int] = {
        entity_id: 0 for entity_id in entity_ids
    }
    for feature_position, feature_key in enumerate(feature_keys):
        entity_id = binding_map[feature_key]
        entity_position = entity_positions[entity_id]
        slot_position = slot_positions[entity_id]
        contexts[:, :, entity_position, slot_position] = (
            context_values[:, :, feature_position]
        )
        target_blocks[:, :, :, entity_position, slot_position] = (
            target_values[:, :, :, feature_position]
        )
        observation_mask[entity_position, slot_position] = True
        slot_positions[entity_id] += 1
    return GraphStateWindows(
        contexts=contexts,
        target_blocks=target_blocks,
        target_controls=windows.target_controls.copy(),
        point_indices=windows.point_indices.copy(),
        observation_mask=observation_mask,
        entity_ids=entity_ids,
        entity_kinds=tuple(
            entity.kind for entity in graph.entities
        ),
        local_feature_keys=local_feature_keys,
        control_feature_names=windows.control_feature_names,
        horizons=windows.horizons,
        target_block_size=windows.target_block_size,
        graph=graph,
    )
