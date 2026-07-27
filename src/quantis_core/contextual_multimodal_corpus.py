"""Demand-aware temporal blocks for contextual multimodal JEPA training."""

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .demand_conditioning import canonical_request_schedule
from .fault_matrix import FaultMatrixRun
from .multimodal_corpus import (
    MultimodalTelemetryCorpus,
    MultimodalTelemetryCorpusSplit,
)
from .windowing import MAD_NORMAL_SCALE


LOG_FEATURE_NAMES = (
    "checkout_completion_ratio",
    "checkout_backlog_delta_ratio",
    "checkout_rejection_rate",
    "application_error_event_rate",
)
CONTROL_FEATURE_NAMES = (
    "request_demand",
    "worker_replicas",
)
REQUIRED_LOG_FEATURE_NAMES = (
    "checkout_accepted_count",
    "checkout_rejected_count",
    "checkout_completed_count",
    "error_event_count",
)


@dataclass(frozen=True)
class SemanticLogTelemetry:
    """Finite demand-relative log features with stable names."""

    values: NDArray[np.float64]
    feature_names: Tuple[str, ...]


class DemandResidualLogTransformer:
    """Remove absolute request volume from bounded application event counts."""

    kind = "demand_residual_application_logs"

    def transform(
        self,
        values: NDArray[np.float64],
        feature_names: Sequence[str],
        request_demand: NDArray[np.float64],
    ) -> SemanticLogTelemetry:
        logs = np.asarray(values, dtype=np.float64)
        demand = np.asarray(request_demand, dtype=np.float64)
        names = tuple(feature_names)
        if (
            logs.ndim != 2
            or logs.shape[1] != len(names)
            or demand.shape != (len(logs),)
        ):
            raise ValueError(
                "log values, feature names, and request demand "
                "must align"
            )
        if (
            not np.all(np.isfinite(logs))
            or not np.all(np.isfinite(demand))
        ):
            raise ValueError("log values and request demand must be finite")
        if np.any(logs < 0.0):
            raise ValueError("application event counts cannot be negative")
        if np.any(demand <= 0.0):
            raise ValueError("request demand must be positive")
        missing = set(REQUIRED_LOG_FEATURE_NAMES) - set(names)
        if missing:
            raise ValueError(
                f"application log features are missing: {sorted(missing)}"
            )
        positions = {
            name: names.index(name) for name in REQUIRED_LOG_FEATURE_NAMES
        }
        accepted = logs[
            :, positions["checkout_accepted_count"]
        ]
        rejected = logs[
            :, positions["checkout_rejected_count"]
        ]
        completed = logs[
            :, positions["checkout_completed_count"]
        ]
        errors = logs[:, positions["error_event_count"]]
        accepted_denominator = np.maximum(accepted, 1.0)
        transformed = np.column_stack(
            (
                completed / accepted_denominator,
                (accepted - completed) / demand,
                rejected / demand,
                errors / demand,
            )
        )
        return SemanticLogTelemetry(
            values=transformed.astype(np.float64),
            feature_names=LOG_FEATURE_NAMES,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": self.kind,
            "features": list(LOG_FEATURE_NAMES),
        }


@dataclass(frozen=True)
class ContextualMultimodalModelWindows:
    """Run-isolated contexts, future blocks, and exogenous controls."""

    metric_contexts: NDArray[np.float64]
    log_contexts: NDArray[np.float64]
    metric_target_blocks: NDArray[np.float64]
    log_target_blocks: NDArray[np.float64]
    target_controls: NDArray[np.float64]
    point_indices: NDArray[np.int64]
    metric_feature_names: Tuple[str, ...]
    log_feature_names: Tuple[str, ...]
    control_feature_names: Tuple[str, ...]
    horizons: Tuple[int, ...]
    target_block_size: int

    def __post_init__(self) -> None:
        sample_count = len(self.metric_contexts)
        horizon_count = len(self.horizons)
        if (
            self.metric_contexts.ndim != 3
            or self.log_contexts.ndim != 3
            or self.metric_target_blocks.ndim != 4
            or self.log_target_blocks.ndim != 4
            or self.target_controls.ndim != 4
        ):
            raise ValueError(
                "contextual multimodal windows have invalid ranks"
            )
        if (
            len(self.log_contexts) != sample_count
            or len(self.metric_target_blocks) != sample_count
            or len(self.log_target_blocks) != sample_count
            or len(self.target_controls) != sample_count
            or self.point_indices.shape != (sample_count,)
        ):
            raise ValueError(
                "contextual multimodal window samples must align"
            )
        expected_target_prefix = (
            sample_count,
            horizon_count,
            self.target_block_size,
        )
        if (
            self.metric_target_blocks.shape[:3]
            != expected_target_prefix
            or self.log_target_blocks.shape[:3]
            != expected_target_prefix
            or self.target_controls.shape[:3]
            != expected_target_prefix
        ):
            raise ValueError(
                "target blocks and controls must align with horizons"
            )
        if (
            self.metric_contexts.shape[2]
            != len(self.metric_feature_names)
            or self.log_contexts.shape[2]
            != len(self.log_feature_names)
            or self.metric_target_blocks.shape[3]
            != len(self.metric_feature_names)
            or self.log_target_blocks.shape[3]
            != len(self.log_feature_names)
            or self.target_controls.shape[3]
            != len(self.control_feature_names)
        ):
            raise ValueError(
                "contextual feature names must match tensor columns"
            )
        arrays = (
            self.metric_contexts,
            self.log_contexts,
            self.metric_target_blocks,
            self.log_target_blocks,
            self.target_controls,
        )
        if any(not np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("contextual multimodal windows must be finite")


@dataclass(frozen=True)
class ContextualMultimodalTelemetryCorpusSplit:
    """One contextual split and the source run of every sample."""

    windows: ContextualMultimodalModelWindows
    case_ids: Tuple[str, ...]
    window_case_ids: Tuple[str, ...]


@dataclass(frozen=True)
class ContextualMultimodalTelemetryCorpus:
    """Contextual development tensors and fitted preprocessing."""

    training: ContextualMultimodalTelemetryCorpusSplit
    validation: ContextualMultimodalTelemetryCorpusSplit
    base_corpus_metadata: Mapping[str, Any]
    preprocessing: Mapping[str, Any]
    protocol: Mapping[str, Any]

    def metadata_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "contextual_multimodal_telemetry_corpus",
            "base_corpus": dict(self.base_corpus_metadata),
            "preprocessing": dict(self.preprocessing),
            "protocol": dict(self.protocol),
        }


@dataclass(frozen=True)
class _RunValues:
    metric: NDArray[np.float64]
    logs: NDArray[np.float64]
    controls: NDArray[np.float64]


def compile_contextual_multimodal_telemetry_corpus(
    base: MultimodalTelemetryCorpus,
    runs: Sequence[FaultMatrixRun],
    horizons: Tuple[int, ...] = (1, 3, 6),
    target_block_size: int = 2,
) -> ContextualMultimodalTelemetryCorpus:
    """Compile demand-relative logs and contextual future target blocks."""

    _validate_temporal_design(horizons, target_block_size)
    runs_by_case_id = {
        run.manifest.case_id: run for run in runs
    }
    selected_case_ids = (
        base.training.case_ids + base.validation.case_ids
    )
    missing = set(selected_case_ids) - set(runs_by_case_id)
    if missing:
        raise ValueError(
            f"contextual corpus manifests are missing: {sorted(missing)}"
        )
    log_location, log_scale = _normalizer_state(
        base.log_window_compiler_artifact
    )
    reconstructed: Dict[str, _RunValues] = {}
    transformer = DemandResidualLogTransformer()
    for split in (base.training, base.validation):
        for case_id in split.case_ids:
            run = runs_by_case_id[case_id]
            metric, normalized_logs = _reconstruct_run(split, case_id)
            raw_logs = normalized_logs * log_scale + log_location
            demand = _request_demand(run, len(metric))
            semantic_logs = transformer.transform(
                raw_logs,
                split.windows.logs.feature_names,
                demand,
            )
            controls = np.column_stack(
                (
                    demand,
                    np.full(
                        len(demand),
                        float(run.manifest.worker_replicas),
                    ),
                )
            )
            reconstructed[case_id] = _RunValues(
                metric=metric,
                logs=semantic_logs.values,
                controls=controls,
            )

    training_ids = base.training.case_ids
    log_normalizer = _fit_normalizer(
        np.concatenate(
            [reconstructed[case_id].logs for case_id in training_ids],
            axis=0,
        )
    )
    control_normalizer = _fit_normalizer(
        np.concatenate(
            [
                reconstructed[case_id].controls
                for case_id in training_ids
            ],
            axis=0,
        )
    )
    normalized = {
        case_id: _RunValues(
            metric=values.metric,
            logs=_normalize(values.logs, log_normalizer),
            controls=_normalize(
                values.controls,
                control_normalizer,
            ),
        )
        for case_id, values in reconstructed.items()
    }
    training = _compile_split(
        base.training.case_ids,
        normalized,
        base.training.windows.metric.feature_names,
        base.training.windows.metric.contexts.shape[1],
        horizons,
        target_block_size,
    )
    validation = _compile_split(
        base.validation.case_ids,
        normalized,
        base.validation.windows.metric.feature_names,
        base.validation.windows.metric.contexts.shape[1],
        horizons,
        target_block_size,
    )
    return ContextualMultimodalTelemetryCorpus(
        training=training,
        validation=validation,
        base_corpus_metadata=base.metadata_dict(),
        preprocessing={
            "metrics": {
                "source": "base_demand_conditioned_metric_corpus",
                "conditioner": base.metric_corpus_metadata[
                    "conditioner"
                ],
                "normalizer": base.metric_corpus_metadata[
                    "window_compiler"
                ],
            },
            "logs": {
                "source": "bounded_otlp_application_event_counts",
                "transformer": transformer.to_dict(),
                "normalizer": log_normalizer,
            },
            "controls": {
                "semantics": (
                    "observed exogenous values aligned to each "
                    "target block"
                ),
                "normalizer": control_normalizer,
                "feature_names": list(CONTROL_FEATURE_NAMES),
            },
        },
        protocol={
            "model_selection_status": "development_only",
            "training_case_ids": list(base.training.case_ids),
            "validation_case_ids": list(base.validation.case_ids),
            "target_horizons": list(horizons),
            "target_block_size": target_block_size,
            "context_crosses_run_boundary": False,
            "target_crosses_run_boundary": False,
            "preprocessing_fitted_on_training_only": True,
            "validation_status": "previously_exposed_diagnostic_only",
        },
    )


def subset_contextual_windows(
    windows: ContextualMultimodalModelWindows,
    selection: NDArray[np.bool_],
) -> ContextualMultimodalModelWindows:
    """Select samples without changing their registered tensor semantics."""

    mask = np.asarray(selection, dtype=np.bool_)
    if mask.shape != (len(windows.metric_contexts),):
        raise ValueError("contextual window selection must match samples")
    return ContextualMultimodalModelWindows(
        metric_contexts=windows.metric_contexts[mask],
        log_contexts=windows.log_contexts[mask],
        metric_target_blocks=windows.metric_target_blocks[mask],
        log_target_blocks=windows.log_target_blocks[mask],
        target_controls=windows.target_controls[mask],
        point_indices=windows.point_indices[mask],
        metric_feature_names=windows.metric_feature_names,
        log_feature_names=windows.log_feature_names,
        control_feature_names=windows.control_feature_names,
        horizons=windows.horizons,
        target_block_size=windows.target_block_size,
    )


def _validate_temporal_design(
    horizons: Tuple[int, ...],
    target_block_size: int,
) -> None:
    if (
        not horizons
        or any(horizon < 1 for horizon in horizons)
        or tuple(sorted(set(horizons))) != horizons
    ):
        raise ValueError(
            "target horizons must be positive, unique, and increasing"
        )
    if target_block_size < 1:
        raise ValueError("target_block_size must be positive")


def _reconstruct_run(
    split: MultimodalTelemetryCorpusSplit,
    case_id: str,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    selection = np.asarray(
        [value == case_id for value in split.window_case_ids],
        dtype=np.bool_,
    )
    metric = split.windows.metric
    logs = split.windows.logs
    if not np.any(selection):
        raise ValueError(f"{case_id} has no model windows")
    point_indices = metric.point_indices[selection]
    lookback = metric.contexts.shape[1]
    expected_indices = np.arange(
        lookback,
        lookback + len(point_indices),
        dtype=np.int64,
    )
    if not np.array_equal(point_indices, expected_indices):
        raise ValueError(
            f"{case_id} model windows are not a complete run sequence"
        )
    first_row = int(np.flatnonzero(selection)[0])
    metric_values = np.concatenate(
        (
            metric.contexts[first_row],
            metric.targets[selection],
        ),
        axis=0,
    )
    log_values = np.concatenate(
        (
            logs.contexts[first_row],
            logs.targets[selection],
        ),
        axis=0,
    )
    return metric_values, log_values


def _request_demand(
    run: FaultMatrixRun,
    point_count: int,
) -> NDArray[np.float64]:
    start, stop = run.manifest.baseline_interval
    if stop - start != point_count:
        raise ValueError(
            f"{run.manifest.case_id} normal interval does not match "
            "reconstructed points"
        )
    schedule = canonical_request_schedule(
        run.manifest.requests_per_window,
        run.manifest.load_pattern_offsets,
    )
    return np.asarray(
        [
            schedule[point_index % len(schedule)]
            for point_index in range(start, stop)
        ],
        dtype=np.float64,
    )


def _compile_split(
    case_ids: Tuple[str, ...],
    values_by_case_id: Mapping[str, _RunValues],
    metric_feature_names: Tuple[str, ...],
    lookback: int,
    horizons: Tuple[int, ...],
    target_block_size: int,
) -> ContextualMultimodalTelemetryCorpusSplit:
    window_groups = [
        _compile_run_windows(
            values_by_case_id[case_id],
            metric_feature_names,
            lookback,
            horizons,
            target_block_size,
        )
        for case_id in case_ids
    ]
    combined = _combine_windows(window_groups)
    window_case_ids = tuple(
        case_id
        for case_id, windows in zip(case_ids, window_groups)
        for _ in range(len(windows.point_indices))
    )
    return ContextualMultimodalTelemetryCorpusSplit(
        windows=combined,
        case_ids=case_ids,
        window_case_ids=window_case_ids,
    )


def _compile_run_windows(
    values: _RunValues,
    metric_feature_names: Tuple[str, ...],
    lookback: int,
    horizons: Tuple[int, ...],
    target_block_size: int,
) -> ContextualMultimodalModelWindows:
    last_context_end = (
        len(values.metric)
        - horizons[-1]
        - target_block_size
        + 1
    )
    if last_context_end < lookback:
        raise ValueError(
            "run is too short for contextual target design"
        )
    context_ends = range(lookback, last_context_end + 1)
    metric_contexts = np.stack(
        [
            values.metric[context_end - lookback : context_end]
            for context_end in context_ends
        ]
    )
    log_contexts = np.stack(
        [
            values.logs[context_end - lookback : context_end]
            for context_end in context_ends
        ]
    )
    metric_targets = _future_blocks(
        values.metric,
        context_ends,
        horizons,
        target_block_size,
    )
    log_targets = _future_blocks(
        values.logs,
        context_ends,
        horizons,
        target_block_size,
    )
    target_controls = _future_blocks(
        values.controls,
        context_ends,
        horizons,
        target_block_size,
    )
    return ContextualMultimodalModelWindows(
        metric_contexts=metric_contexts,
        log_contexts=log_contexts,
        metric_target_blocks=metric_targets,
        log_target_blocks=log_targets,
        target_controls=target_controls,
        point_indices=np.asarray(
            list(context_ends),
            dtype=np.int64,
        ),
        metric_feature_names=metric_feature_names,
        log_feature_names=LOG_FEATURE_NAMES,
        control_feature_names=CONTROL_FEATURE_NAMES,
        horizons=horizons,
        target_block_size=target_block_size,
    )


def _future_blocks(
    values: NDArray[np.float64],
    context_ends: range,
    horizons: Tuple[int, ...],
    target_block_size: int,
) -> NDArray[np.float64]:
    return np.stack(
        [
            np.stack(
                [
                    values[
                        context_end + horizon - 1 :
                        context_end
                        + horizon
                        - 1
                        + target_block_size
                    ]
                    for horizon in horizons
                ]
            )
            for context_end in context_ends
        ]
    )


def _combine_windows(
    groups: Sequence[ContextualMultimodalModelWindows],
) -> ContextualMultimodalModelWindows:
    if not groups:
        raise ValueError("cannot combine an empty contextual corpus")
    first = groups[0]
    return ContextualMultimodalModelWindows(
        metric_contexts=np.concatenate(
            [group.metric_contexts for group in groups],
            axis=0,
        ),
        log_contexts=np.concatenate(
            [group.log_contexts for group in groups],
            axis=0,
        ),
        metric_target_blocks=np.concatenate(
            [group.metric_target_blocks for group in groups],
            axis=0,
        ),
        log_target_blocks=np.concatenate(
            [group.log_target_blocks for group in groups],
            axis=0,
        ),
        target_controls=np.concatenate(
            [group.target_controls for group in groups],
            axis=0,
        ),
        point_indices=np.concatenate(
            [group.point_indices for group in groups],
            axis=0,
        ),
        metric_feature_names=first.metric_feature_names,
        log_feature_names=first.log_feature_names,
        control_feature_names=first.control_feature_names,
        horizons=first.horizons,
        target_block_size=first.target_block_size,
    )


def _fit_normalizer(
    values: NDArray[np.float64],
) -> Dict[str, Any]:
    location = np.median(values, axis=0)
    scale = (
        MAD_NORMAL_SCALE
        * np.median(np.abs(values - location), axis=0)
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


def _normalizer_state(
    artifact: Mapping[str, Any],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    return (
        np.asarray(artifact["location"], dtype=np.float64),
        np.asarray(artifact["scale"], dtype=np.float64),
    )


def _normalize(
    values: NDArray[np.float64],
    artifact: Mapping[str, Any],
) -> NDArray[np.float64]:
    location, scale = _normalizer_state(artifact)
    return (values - location) / scale
