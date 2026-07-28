"""Operational-state preprocessing and caching for the graph corpus."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from .contextual_multimodal_corpus import (
    DEPENDENCY_LOG_FEATURE_NAMES,
)
from .graph_telemetry import (
    DeclaredTelemetryGraph,
    GraphStateWindows,
    TelemetryBinding,
    quantis_checkout_graph,
)


OBSERVABILITY_RAW_FEATURE_NAMES = (
    "request_rate",
    "request_latency_ms",
    "error_rate",
    "api_inflight_current",
    "api_inflight_peak",
    "api_concurrency_mean",
    "queue_depth",
    "queue_oldest_age_ms",
    "enqueue_event_age_ms",
    "dequeue_event_age_ms",
    "queue_residence_mean_ms",
    "worker_rate",
    "worker_heartbeat_age_s",
    "worker_active_count",
    "worker_busy_count",
    "worker_busy_age_max_ms",
    "worker_busy_fraction",
    "worker_processing_latency_ms",
    "redis_enqueue_latency_ms",
    "redis_enqueue_error_rate",
    "redis_dequeue_latency_ms",
    "redis_dequeue_error_rate",
    "db_write_rate",
    "postgresql_write_latency_ms",
    "postgresql_write_error_rate",
    "postgresql_write_event_age_ms",
)
OBSERVABILITY_METRIC_FEATURE_NAMES = (
    "request_latency_ms",
    "error_rate",
    "api_inflight_current",
    "api_inflight_peak",
    "api_concurrency_mean",
    "queue_depth",
    "queue_oldest_age_ms",
    "enqueue_event_age_ms",
    "dequeue_event_age_ms",
    "queue_residence_mean_ms",
    "worker_completion_ratio",
    "worker_heartbeat_age_s",
    "worker_active_ratio",
    "worker_busy_ratio",
    "worker_busy_age_max_ms",
    "worker_busy_fraction",
    "worker_processing_latency_ms",
    "redis_enqueue_latency_ms",
    "redis_enqueue_error_rate",
    "redis_dequeue_latency_ms",
    "redis_dequeue_error_rate",
    "db_write_completion_ratio",
    "postgresql_write_latency_ms",
    "postgresql_write_error_rate",
    "postgresql_write_event_age_ms",
)
OBSERVABILITY_CONTROL_FEATURE_NAMES = (
    "request_demand",
    "worker_replicas",
)


@dataclass(frozen=True)
class OperationalStateTelemetry:
    """Finite endogenous state plus explicitly separated controls."""

    values: NDArray[np.float64]
    feature_names: Tuple[str, ...]
    controls: NDArray[np.float64]
    control_feature_names: Tuple[str, ...]


class OperationalStateTransformer:
    """Convert raw lab gauges into graph-owned endogenous state."""

    kind = "observability_rich_operational_state_v1"

    def transform(
        self,
        values: NDArray[np.float64],
        feature_names: Sequence[str],
        *,
        request_demand: NDArray[np.float64],
        worker_replicas: int,
    ) -> OperationalStateTelemetry:
        telemetry = np.asarray(values, dtype=np.float64)
        demand = np.asarray(request_demand, dtype=np.float64)
        names = tuple(feature_names)
        if (
            telemetry.ndim != 2
            or telemetry.shape[1] != len(names)
            or demand.shape != (len(telemetry),)
        ):
            raise ValueError(
                "operational values, names, and demand must align"
            )
        if len(set(names)) != len(names):
            raise ValueError(
                "operational feature names must be unique"
            )
        missing = set(OBSERVABILITY_RAW_FEATURE_NAMES) - set(names)
        unexpected = set(names) - set(
            OBSERVABILITY_RAW_FEATURE_NAMES
        )
        if missing:
            raise ValueError(
                f"operational state features are missing: "
                f"{sorted(missing)}"
            )
        if unexpected:
            raise ValueError(
                "operational state features are unexpected: "
                f"{sorted(unexpected)}"
            )
        if (
            not np.all(np.isfinite(telemetry))
            or not np.all(np.isfinite(demand))
        ):
            raise ValueError("operational state must be finite")
        if np.any(telemetry < 0.0):
            raise ValueError("operational state cannot be negative")
        if np.any(demand <= 0.0):
            raise ValueError(
                "operational request demand must be positive"
            )
        if (
            isinstance(worker_replicas, bool)
            or worker_replicas < 1
        ):
            raise ValueError(
                "operational worker replicas must be positive"
            )
        position = {
            name: index for index, name in enumerate(names)
        }

        def column(name: str) -> NDArray[np.float64]:
            return telemetry[:, position[name]]

        semantic = np.column_stack(
            (
                column("request_latency_ms"),
                column("error_rate"),
                column("api_inflight_current"),
                column("api_inflight_peak"),
                column("api_concurrency_mean"),
                column("queue_depth"),
                column("queue_oldest_age_ms"),
                column("enqueue_event_age_ms"),
                column("dequeue_event_age_ms"),
                column("queue_residence_mean_ms"),
                column("worker_rate") / demand,
                column("worker_heartbeat_age_s"),
                column("worker_active_count") / worker_replicas,
                column("worker_busy_count") / worker_replicas,
                column("worker_busy_age_max_ms"),
                column("worker_busy_fraction"),
                column("worker_processing_latency_ms"),
                column("redis_enqueue_latency_ms"),
                column("redis_enqueue_error_rate"),
                column("redis_dequeue_latency_ms"),
                column("redis_dequeue_error_rate"),
                column("db_write_rate") / demand,
                column("postgresql_write_latency_ms"),
                column("postgresql_write_error_rate"),
                column("postgresql_write_event_age_ms"),
            )
        )
        controls = np.column_stack(
            (
                demand,
                np.full(
                    len(demand),
                    float(worker_replicas),
                    dtype=np.float64,
                ),
            )
        )
        return OperationalStateTelemetry(
            values=np.asarray(semantic, dtype=np.float64),
            feature_names=OBSERVABILITY_METRIC_FEATURE_NAMES,
            controls=np.asarray(controls, dtype=np.float64),
            control_feature_names=(
                OBSERVABILITY_CONTROL_FEATURE_NAMES
            ),
        )


def quantis_checkout_observability_graph() -> DeclaredTelemetryGraph:
    """Return the fixed topology with observability-rich ownership."""

    ownership = {
        "metric.request_latency_ms": "api",
        "metric.error_rate": "api",
        "metric.api_inflight_current": "api",
        "metric.api_inflight_peak": "api",
        "metric.api_concurrency_mean": "api",
        "metric.queue_depth": "checkout_queue",
        "metric.queue_oldest_age_ms": "checkout_queue",
        "metric.enqueue_event_age_ms": "checkout_queue",
        "metric.dequeue_event_age_ms": "checkout_queue",
        "metric.queue_residence_mean_ms": "checkout_queue",
        "metric.worker_completion_ratio": "worker_pool",
        "metric.worker_heartbeat_age_s": "worker_pool",
        "metric.worker_active_ratio": "worker_pool",
        "metric.worker_busy_ratio": "worker_pool",
        "metric.worker_busy_age_max_ms": "worker_pool",
        "metric.worker_busy_fraction": "worker_pool",
        "metric.worker_processing_latency_ms": "worker_pool",
        "metric.redis_enqueue_latency_ms": (
            "api_enqueues_queue"
        ),
        "metric.redis_enqueue_error_rate": (
            "api_enqueues_queue"
        ),
        "metric.redis_dequeue_latency_ms": (
            "queue_dequeues_to_worker"
        ),
        "metric.redis_dequeue_error_rate": (
            "queue_dequeues_to_worker"
        ),
        "metric.db_write_completion_ratio": "postgresql",
        "metric.postgresql_write_latency_ms": (
            "worker_writes_postgresql"
        ),
        "metric.postgresql_write_error_rate": (
            "worker_writes_postgresql"
        ),
        "metric.postgresql_write_event_age_ms": "postgresql",
        "log.checkout_completion_ratio": "worker_pool",
        "log.checkout_backlog_delta_ratio": (
            "api_enqueues_queue"
        ),
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
    metric_keys = {
        f"metric.{name}"
        for name in OBSERVABILITY_METRIC_FEATURE_NAMES
    }
    log_keys = {
        f"log.{name}" for name in DEPENDENCY_LOG_FEATURE_NAMES
    }
    if set(ownership) != metric_keys | log_keys:
        raise AssertionError(
            "observability graph ownership is incomplete"
        )
    return DeclaredTelemetryGraph(
        entities=quantis_checkout_graph().entities,
        bindings=tuple(
            TelemetryBinding(feature_key, entity_id)
            for feature_key, entity_id in ownership.items()
        ),
    )


@dataclass(frozen=True)
class ObservabilityGraphCorpus:
    """Frozen graph tensors and the identities that produced each sample."""

    training: GraphStateWindows
    validation: GraphStateWindows
    training_case_ids: Tuple[str, ...]
    validation_case_ids: Tuple[str, ...]
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        if len(self.training_case_ids) != len(
            self.training.contexts
        ):
            raise ValueError(
                "training graph samples and case ids do not align"
            )
        if len(self.validation_case_ids) != len(
            self.validation.contexts
        ):
            raise ValueError(
                "validation graph samples and case ids do not align"
            )
        if (
            self.training.graph.to_dict()
            != self.validation.graph.to_dict()
        ):
            raise ValueError(
                "training and validation graph schemas differ"
            )
        if (
            self.training.local_feature_keys
            != self.validation.local_feature_keys
            or self.training.horizons != self.validation.horizons
            or self.training.target_block_size
            != self.validation.target_block_size
        ):
            raise ValueError(
                "training and validation tensor schemas differ"
            )
        _canonical_json_bytes(self.provenance)


def write_observability_graph_cache(
    corpus: ObservabilityGraphCorpus,
    root: Union[str, Path],
) -> Path:
    """Write immutable tensors below a semantic content-addressed key."""

    arrays = _corpus_arrays(corpus)
    array_sha256 = {
        name: _array_sha256(value)
        for name, value in arrays.items()
    }
    semantic = {
        "schema_version": 1,
        "kind": "observability_graph_corpus_cache",
        "training": _windows_metadata(corpus.training),
        "validation": _windows_metadata(corpus.validation),
        "training_case_ids": list(corpus.training_case_ids),
        "validation_case_ids": list(corpus.validation_case_ids),
        "provenance": dict(corpus.provenance),
        "array_sha256": array_sha256,
    }
    cache_key = hashlib.sha256(
        _canonical_json_bytes(semantic)
    ).hexdigest()
    cache_directory = Path(root) / cache_key
    metadata_path = cache_directory / "metadata.json"
    tensors_path = cache_directory / "tensors.npz"
    if cache_directory.exists():
        loaded = load_observability_graph_cache(cache_directory)
        if _cache_semantic_payload(loaded) != semantic:
            raise ValueError(
                "content-addressed graph cache already differs"
            )
        return cache_directory
    cache_directory.mkdir(parents=True)
    np.savez_compressed(tensors_path, **arrays)
    metadata = {
        **semantic,
        "cache_key": cache_key,
        "tensor_archive_sha256": hashlib.sha256(
            tensors_path.read_bytes()
        ).hexdigest(),
    }
    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    return cache_directory


def load_observability_graph_cache(
    directory: Union[str, Path],
) -> ObservabilityGraphCorpus:
    """Load a cache only after verifying its address and every array."""

    cache_directory = Path(directory)
    metadata_path = cache_directory / "metadata.json"
    tensors_path = cache_directory / "tensors.npz"
    metadata = json.loads(metadata_path.read_text())
    if (
        metadata.get("schema_version") != 1
        or metadata.get("kind")
        != "observability_graph_corpus_cache"
    ):
        raise ValueError("unsupported observability graph cache")
    archive_sha256 = hashlib.sha256(
        tensors_path.read_bytes()
    ).hexdigest()
    if archive_sha256 != metadata.get(
        "tensor_archive_sha256"
    ):
        raise ValueError(
            "observability graph tensor archive hash changed"
        )
    semantic = {
        key: value
        for key, value in metadata.items()
        if key not in ("cache_key", "tensor_archive_sha256")
    }
    cache_key = hashlib.sha256(
        _canonical_json_bytes(semantic)
    ).hexdigest()
    if (
        cache_key != metadata.get("cache_key")
        or cache_key != cache_directory.name
    ):
        raise ValueError(
            "observability graph cache address changed"
        )
    with np.load(tensors_path, allow_pickle=False) as archive:
        arrays = {
            name: np.asarray(archive[name])
            for name in archive.files
        }
    expected_names = set(metadata["array_sha256"])
    if set(arrays) != expected_names:
        raise ValueError(
            "observability graph cache arrays changed"
        )
    for name, expected_sha256 in metadata[
        "array_sha256"
    ].items():
        if _array_sha256(arrays[name]) != expected_sha256:
            raise ValueError(
                f"observability graph array hash changed: {name}"
            )
    corpus = ObservabilityGraphCorpus(
        training=_restore_windows(
            arrays, "training", metadata["training"]
        ),
        validation=_restore_windows(
            arrays, "validation", metadata["validation"]
        ),
        training_case_ids=tuple(metadata["training_case_ids"]),
        validation_case_ids=tuple(
            metadata["validation_case_ids"]
        ),
        provenance=dict(metadata["provenance"]),
    )
    if _cache_semantic_payload(corpus) != semantic:
        raise ValueError(
            "observability graph cache metadata changed"
        )
    return corpus


def _corpus_arrays(
    corpus: ObservabilityGraphCorpus,
) -> Dict[str, NDArray[Any]]:
    arrays: Dict[str, NDArray[Any]] = {}
    for split_name, windows in (
        ("training", corpus.training),
        ("validation", corpus.validation),
    ):
        arrays[f"{split_name}_contexts"] = windows.contexts
        arrays[f"{split_name}_target_blocks"] = (
            windows.target_blocks
        )
        arrays[f"{split_name}_target_controls"] = (
            windows.target_controls
        )
        arrays[f"{split_name}_point_indices"] = (
            windows.point_indices
        )
        arrays[f"{split_name}_observation_mask"] = (
            windows.observation_mask
        )
    return arrays


def _windows_metadata(
    windows: GraphStateWindows,
) -> Dict[str, Any]:
    return {
        "entity_ids": list(windows.entity_ids),
        "entity_kinds": list(windows.entity_kinds),
        "local_feature_keys": [
            list(keys) for keys in windows.local_feature_keys
        ],
        "control_feature_names": list(
            windows.control_feature_names
        ),
        "horizons": list(windows.horizons),
        "target_block_size": windows.target_block_size,
        "graph": windows.graph.to_dict(),
        "shapes": {
            "contexts": list(windows.contexts.shape),
            "target_blocks": list(windows.target_blocks.shape),
            "target_controls": list(
                windows.target_controls.shape
            ),
            "point_indices": list(windows.point_indices.shape),
            "observation_mask": list(
                windows.observation_mask.shape
            ),
        },
    }


def _restore_windows(
    arrays: Mapping[str, NDArray[Any]],
    split_name: str,
    metadata: Mapping[str, Any],
) -> GraphStateWindows:
    names = (
        "contexts",
        "target_blocks",
        "target_controls",
        "point_indices",
        "observation_mask",
    )
    selected = {
        name: arrays[f"{split_name}_{name}"]
        for name in names
    }
    for name, value in selected.items():
        if list(value.shape) != metadata["shapes"][name]:
            raise ValueError(
                f"observability graph tensor shape changed: "
                f"{split_name}_{name}"
            )
    return GraphStateWindows(
        contexts=np.asarray(
            selected["contexts"], dtype=np.float64
        ),
        target_blocks=np.asarray(
            selected["target_blocks"], dtype=np.float64
        ),
        target_controls=np.asarray(
            selected["target_controls"], dtype=np.float64
        ),
        point_indices=np.asarray(
            selected["point_indices"], dtype=np.int64
        ),
        observation_mask=np.asarray(
            selected["observation_mask"], dtype=np.bool_
        ),
        entity_ids=tuple(metadata["entity_ids"]),
        entity_kinds=tuple(metadata["entity_kinds"]),
        local_feature_keys=tuple(
            tuple(keys) for keys in metadata["local_feature_keys"]
        ),
        control_feature_names=tuple(
            metadata["control_feature_names"]
        ),
        horizons=tuple(int(value) for value in metadata["horizons"]),
        target_block_size=int(metadata["target_block_size"]),
        graph=DeclaredTelemetryGraph.from_dict(metadata["graph"]),
    )


def _array_sha256(array: NDArray[Any]) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(
        json.dumps(list(contiguous.shape)).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _cache_semantic_payload(
    corpus: ObservabilityGraphCorpus,
) -> Dict[str, Any]:
    arrays = _corpus_arrays(corpus)
    return {
        "schema_version": 1,
        "kind": "observability_graph_corpus_cache",
        "training": _windows_metadata(corpus.training),
        "validation": _windows_metadata(corpus.validation),
        "training_case_ids": list(corpus.training_case_ids),
        "validation_case_ids": list(
            corpus.validation_case_ids
        ),
        "provenance": dict(corpus.provenance),
        "array_sha256": {
            name: _array_sha256(value)
            for name, value in arrays.items()
        },
    }


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError(
            "observability graph provenance must be canonical JSON"
        ) from error
