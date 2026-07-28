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
    CONTROL_FEATURE_NAMES,
    ContextualMultimodalModelWindows,
    DependencyResidualLogTransformer,
)
from .demand_conditioning import canonical_request_schedule
from .fault_matrix import FaultMatrixRun
from .graph_telemetry import (
    DeclaredTelemetryGraph,
    GraphStateWindows,
    TelemetryBinding,
    compile_graph_state_windows,
    quantis_checkout_graph,
)
from .multimodal_corpus import _metric_event_time_boundaries
from .otlp import TelemetryCapture
from .otlp_log_windowing import (
    OtlpLogFeatureSpec,
    OtlpLogWindowCompiler,
)
from .otlp_logs import OtlpLogCapture
from .otlp_windowing import OtlpFeatureSpec, OtlpWindowCompiler
from .telemetry_corpus import TelemetryCorpusSplitSpec
from .windowing import MAD_NORMAL_SCALE


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
class _SemanticRun:
    metrics: NDArray[np.float64]
    logs: NDArray[np.float64]
    controls: NDArray[np.float64]


def compile_observability_graph_corpus(
    runs: Sequence[FaultMatrixRun],
    log_captures: Mapping[str, OtlpLogCapture],
    metric_spec: OtlpFeatureSpec,
    log_spec: OtlpLogFeatureSpec,
    split_spec: TelemetryCorpusSplitSpec,
    *,
    horizons: Tuple[int, ...] = (1, 5, 10),
    target_block_size: int = 2,
    protocol: Mapping[str, Any],
) -> "ObservabilityGraphCorpus":
    """Compile raw captures into normalized, graph-owned future blocks."""

    if (
        not horizons
        or tuple(sorted(set(horizons))) != horizons
        or any(horizon < 1 for horizon in horizons)
        or target_block_size < 1
    ):
        raise ValueError("invalid observability graph temporal design")
    run_by_case_id = {
        run.manifest.case_id: run for run in runs
    }
    if len(run_by_case_id) != len(runs):
        raise ValueError("duplicate observability graph case id")
    selected_case_ids = (
        split_spec.training_case_ids
        + split_spec.validation_case_ids
    )
    missing = set(selected_case_ids) - set(run_by_case_id)
    missing_logs = set(selected_case_ids) - set(log_captures)
    if missing or missing_logs:
        raise ValueError(
            "observability graph corpus inputs are incomplete: "
            f"metrics={sorted(missing)}, logs={sorted(missing_logs)}"
        )
    training_schedules = _schedule_set(
        run_by_case_id, split_spec.training_case_ids
    )
    validation_schedules = _schedule_set(
        run_by_case_id, split_spec.validation_case_ids
    )
    if training_schedules & validation_schedules:
        raise ValueError(
            "observability graph train and validation schedules overlap"
        )

    metric_compiler = OtlpWindowCompiler(metric_spec)
    log_compiler = OtlpLogWindowCompiler(log_spec)
    metric_transformer = OperationalStateTransformer()
    log_transformer = DependencyResidualLogTransformer()
    semantic_by_case_id: Dict[str, _SemanticRun] = {}
    run_provenance: Dict[str, Any] = {}
    application_builds = set()
    queue_sizes = set()
    for case_id in selected_case_ids:
        run = run_by_case_id[case_id]
        capture = run.capture
        log_capture = log_captures[case_id]
        manifest_sha256 = hashlib.sha256(
            _canonical_json_bytes(run.manifest.to_dict())
        ).hexdigest()
        _validate_run_identity(
            run, log_capture, manifest_sha256
        )
        compiled_metrics = metric_compiler.compile(capture)
        if (
            len(compiled_metrics.values)
            != run.manifest.point_count
            or compiled_metrics.data_quality["missing_cells"] != 0
            or compiled_metrics.feature_names
            != OBSERVABILITY_RAW_FEATURE_NAMES
        ):
            raise ValueError(
                f"{case_id} does not contain complete operational state"
            )
        boundaries = _metric_event_time_boundaries(run)
        if boundaries is None:
            compiled_logs = log_compiler.compile(
                log_capture, run.manifest.point_count
            )
            log_assignment = "declared_logical_window"
        else:
            run_start, window_ends, drain_end = boundaries
            compiled_logs = log_compiler.compile(
                log_capture,
                run.manifest.point_count,
                run_start_unix_nano=run_start,
                window_end_unix_nano=window_ends,
                drain_end_unix_nano=drain_end,
            )
            log_assignment = "event_time_metric_boundaries"
        raw_metrics = compiled_metrics.values[
            run.manifest.baseline_slice
        ]
        raw_logs = compiled_logs.values[
            run.manifest.baseline_slice
        ]
        demand = _request_demand(run, len(raw_metrics))
        metric_state = metric_transformer.transform(
            raw_metrics,
            compiled_metrics.feature_names,
            request_demand=demand,
            worker_replicas=run.manifest.worker_replicas,
        )
        log_state = log_transformer.transform(
            raw_logs,
            compiled_logs.feature_names,
            demand,
        )
        if log_state.feature_names != DEPENDENCY_LOG_FEATURE_NAMES:
            raise ValueError(
                "observability graph semantic log schema changed"
            )
        semantic_by_case_id[case_id] = _SemanticRun(
            metrics=metric_state.values,
            logs=log_state.values,
            controls=metric_state.controls,
        )
        build = _application_build(capture)
        queue_size = _application_queue_size(capture)
        application_builds.add(build)
        queue_sizes.add(queue_size)
        run_provenance[case_id] = {
            "capture_sha256": capture.sha256,
            "log_capture_sha256": log_capture.sha256,
            "manifest_sha256": manifest_sha256,
            "worker_replicas": run.manifest.worker_replicas,
            "canonical_request_schedule": list(
                canonical_request_schedule(
                    run.manifest.requests_per_window,
                    run.manifest.load_pattern_offsets,
                )
            ),
            "metric_data_quality": dict(
                compiled_metrics.data_quality
            ),
            "log_data_quality": dict(
                compiled_logs.data_quality
            ),
            "log_window_assignment": log_assignment,
        }
    if len(application_builds) != 1 or len(queue_sizes) != 1:
        raise ValueError(
            "observability graph application provenance differs"
        )
    application_image_id, build_sha256 = next(
        iter(application_builds)
    )
    queue_size = next(iter(queue_sizes))
    if (
        split_spec.expected_application_api_request_queue_size
        is not None
        and queue_size
        != split_spec.expected_application_api_request_queue_size
    ):
        raise ValueError(
            "observability graph API queue size differs from split"
        )

    training_ids = split_spec.training_case_ids
    metric_normalizer = _fit_normalizer(
        np.concatenate(
            [
                semantic_by_case_id[case_id].metrics
                for case_id in training_ids
            ]
        )
    )
    log_normalizer = _fit_normalizer(
        np.concatenate(
            [
                semantic_by_case_id[case_id].logs
                for case_id in training_ids
            ]
        )
    )
    control_normalizer = _fit_normalizer(
        np.concatenate(
            [
                semantic_by_case_id[case_id].controls
                for case_id in training_ids
            ]
        )
    )
    normalized = {
        case_id: _SemanticRun(
            metrics=_normalize(
                values.metrics, metric_normalizer
            ),
            logs=_normalize(values.logs, log_normalizer),
            controls=_normalize(
                values.controls, control_normalizer
            ),
        )
        for case_id, values in semantic_by_case_id.items()
    }
    graph = quantis_checkout_observability_graph()
    training, training_window_case_ids = _compile_graph_split(
        training_ids,
        normalized,
        split_spec.lookback,
        horizons,
        target_block_size,
        graph,
    )
    validation, validation_window_case_ids = _compile_graph_split(
        split_spec.validation_case_ids,
        normalized,
        split_spec.lookback,
        horizons,
        target_block_size,
        graph,
    )
    return ObservabilityGraphCorpus(
        training=training,
        validation=validation,
        training_case_ids=training_window_case_ids,
        validation_case_ids=validation_window_case_ids,
        provenance={
            "schema_version": 1,
            "kind": "observability_graph_compilation",
            "protocol": dict(protocol),
            "metric_feature_spec": metric_spec.to_dict(),
            "log_feature_spec": log_spec.to_dict(),
            "split_spec": split_spec.to_dict(),
            "metric_normalizer": metric_normalizer,
            "log_normalizer": log_normalizer,
            "control_normalizer": control_normalizer,
            "application_image_id": application_image_id,
            "application_build_context_sha256": build_sha256,
            "application_api_request_queue_size": queue_size,
            "preprocessing_fitted_on_training_only": True,
            "context_crosses_run_boundary": False,
            "target_crosses_run_boundary": False,
            "runs": run_provenance,
        },
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


def _schedule_set(
    runs: Mapping[str, FaultMatrixRun],
    case_ids: Sequence[str],
) -> set[Tuple[int, ...]]:
    return {
        canonical_request_schedule(
            runs[case_id].manifest.requests_per_window,
            runs[case_id].manifest.load_pattern_offsets,
        )
        for case_id in case_ids
    }


def _request_demand(
    run: FaultMatrixRun,
    point_count: int,
) -> NDArray[np.float64]:
    start, stop = run.manifest.baseline_interval
    if stop - start != point_count:
        raise ValueError(
            f"{run.manifest.case_id} baseline does not cover the run"
        )
    schedule = canonical_request_schedule(
        run.manifest.requests_per_window,
        run.manifest.load_pattern_offsets,
    )
    return np.asarray(
        [
            schedule[index % len(schedule)]
            for index in range(start, stop)
        ],
        dtype=np.float64,
    )


def _validate_run_identity(
    run: FaultMatrixRun,
    log_capture: OtlpLogCapture,
    manifest_sha256: str,
) -> None:
    metric_identity = {
        (
            point.resource_attributes.get(
                "quantis.experiment.case.id"
            ),
            point.resource_attributes.get(
                "quantis.experiment.fault.kind"
            ),
            point.resource_attributes.get(
                "quantis.experiment.manifest.sha256"
            ),
        )
        for point in run.capture.points
    }
    log_identity = {
        (
            record.resource_attributes.get(
                "quantis.experiment.case.id"
            ),
            record.resource_attributes.get(
                "quantis.experiment.fault.kind"
            ),
            record.resource_attributes.get(
                "quantis.experiment.manifest.sha256"
            ),
        )
        for record in log_capture.records
    }
    expected = {
        (
            run.manifest.case_id,
            run.manifest.fault_kind,
            manifest_sha256,
        )
    }
    if metric_identity != expected or log_identity != expected:
        raise ValueError(
            f"{run.manifest.case_id} capture identity changed"
        )


def _application_build(
    capture: TelemetryCapture,
) -> Tuple[str, str]:
    values = {
        (
            point.resource_attributes.get(
                "quantis.application.image.id"
            ),
            point.resource_attributes.get(
                "quantis.application.build_context.sha256"
            ),
        )
        for point in capture.points
    }
    if len(values) != 1:
        raise ValueError("application build provenance is ambiguous")
    image_id, build_sha256 = next(iter(values))
    if (
        not isinstance(image_id, str)
        or not image_id.startswith("sha256:")
        or len(image_id) != 71
        or not isinstance(build_sha256, str)
        or len(build_sha256) != 64
    ):
        raise ValueError("application build provenance is invalid")
    return image_id, build_sha256


def _application_queue_size(capture: TelemetryCapture) -> int:
    values = {
        point.resource_attributes.get(
            "quantis.application.api.request_queue_size"
        )
        for point in capture.points
    }
    raw = next(iter(values)) if len(values) == 1 else None
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
        raise ValueError(
            "application API queue size provenance is invalid"
        )
    return raw


def _fit_normalizer(
    values: NDArray[np.float64],
) -> Dict[str, Any]:
    location = np.median(values, axis=0)
    scale = MAD_NORMAL_SCALE * np.median(
        np.abs(values - location), axis=0
    )
    fallback = np.std(values, axis=0)
    scale = np.where(scale > 1e-12, scale, fallback)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return {
        "schema_version": 1,
        "kind": "robust_location_scale",
        "location": location.tolist(),
        "scale": scale.tolist(),
    }


def _normalize(
    values: NDArray[np.float64],
    artifact: Mapping[str, Any],
) -> NDArray[np.float64]:
    location = np.asarray(
        artifact["location"], dtype=np.float64
    )
    scale = np.asarray(artifact["scale"], dtype=np.float64)
    return (values - location) / scale


def _compile_graph_split(
    case_ids: Tuple[str, ...],
    values_by_case_id: Mapping[str, _SemanticRun],
    lookback: int,
    horizons: Tuple[int, ...],
    target_block_size: int,
    graph: DeclaredTelemetryGraph,
) -> Tuple[GraphStateWindows, Tuple[str, ...]]:
    groups = tuple(
        _compile_semantic_windows(
            values_by_case_id[case_id],
            lookback,
            horizons,
            target_block_size,
        )
        for case_id in case_ids
    )
    contextual = _combine_contextual_windows(groups)
    window_case_ids = tuple(
        case_id
        for case_id, group in zip(case_ids, groups)
        for _ in range(len(group.point_indices))
    )
    return (
        compile_graph_state_windows(contextual, graph),
        window_case_ids,
    )


def _compile_semantic_windows(
    values: _SemanticRun,
    lookback: int,
    horizons: Tuple[int, ...],
    target_block_size: int,
) -> ContextualMultimodalModelWindows:
    last_context_end = (
        len(values.metrics)
        - horizons[-1]
        - target_block_size
        + 1
    )
    if last_context_end < lookback:
        raise ValueError(
            "observability graph run is too short for temporal design"
        )
    context_ends = range(lookback, last_context_end + 1)
    metric_contexts = np.stack(
        [
            values.metrics[end - lookback : end]
            for end in context_ends
        ]
    )
    log_contexts = np.stack(
        [
            values.logs[end - lookback : end]
            for end in context_ends
        ]
    )

    def future_blocks(
        channel: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return np.stack(
            [
                np.stack(
                    [
                        channel[
                            end + horizon - 1 :
                            end
                            + horizon
                            - 1
                            + target_block_size
                        ]
                        for horizon in horizons
                    ]
                )
                for end in context_ends
            ]
        )

    return ContextualMultimodalModelWindows(
        metric_contexts=metric_contexts,
        log_contexts=log_contexts,
        metric_target_blocks=future_blocks(values.metrics),
        log_target_blocks=future_blocks(values.logs),
        target_controls=future_blocks(values.controls),
        point_indices=np.asarray(
            list(context_ends), dtype=np.int64
        ),
        metric_feature_names=OBSERVABILITY_METRIC_FEATURE_NAMES,
        log_feature_names=DEPENDENCY_LOG_FEATURE_NAMES,
        control_feature_names=CONTROL_FEATURE_NAMES,
        horizons=horizons,
        target_block_size=target_block_size,
    )


def _combine_contextual_windows(
    groups: Sequence[ContextualMultimodalModelWindows],
) -> ContextualMultimodalModelWindows:
    if not groups:
        raise ValueError(
            "cannot combine an empty observability graph split"
        )
    first = groups[0]
    return ContextualMultimodalModelWindows(
        metric_contexts=np.concatenate(
            [group.metric_contexts for group in groups]
        ),
        log_contexts=np.concatenate(
            [group.log_contexts for group in groups]
        ),
        metric_target_blocks=np.concatenate(
            [group.metric_target_blocks for group in groups]
        ),
        log_target_blocks=np.concatenate(
            [group.log_target_blocks for group in groups]
        ),
        target_controls=np.concatenate(
            [group.target_controls for group in groups]
        ),
        point_indices=np.concatenate(
            [group.point_indices for group in groups]
        ),
        metric_feature_names=first.metric_feature_names,
        log_feature_names=first.log_feature_names,
        control_feature_names=first.control_feature_names,
        horizons=first.horizons,
        target_block_size=first.target_block_size,
    )


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
