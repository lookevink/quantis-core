"""Align structured application-log events to graph telemetry windows."""

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .hybrid_graph_tokens import AlignedEventFeatures
from .observability_graph_corpus import ObservabilityGraphCorpus
from .otlp_logs import LogRecord, OtlpLogCapture
from .structured_events import (
    CompiledStructuredEvents,
    StructuredEventVocabulary,
)


@dataclass(frozen=True)
class HybridEventCorpus:
    """Training-fitted structured events aligned to both graph splits."""

    training: AlignedEventFeatures
    validation: AlignedEventFeatures
    vocabulary: StructuredEventVocabulary
    data_quality: Mapping[str, Any]

    def __post_init__(self) -> None:
        if (
            self.training.feature_names
            != self.validation.feature_names
            or self.training.observation_mask.shape
            != self.validation.observation_mask.shape
            or not self.data_quality
        ):
            raise ValueError("hybrid event corpus splits do not align")


@dataclass(frozen=True)
class _NumericScale:
    center: NDArray[np.float64]
    scale: NDArray[np.float64]


def compile_hybrid_event_corpus(
    corpus: ObservabilityGraphCorpus,
    log_captures: Mapping[str, OtlpLogCapture],
    *,
    logical_window_attribute: str,
    numeric_attribute_names: Sequence[str] = (),
    service_to_entity: Mapping[str, str],
    event_entity_overrides: Optional[Mapping[str, str]] = None,
    service_event_entity_overrides: Optional[
        Mapping[str, str]
    ] = None,
) -> HybridEventCorpus:
    """Compile raw logs without fitting any preprocessing on validation."""

    if not logical_window_attribute:
        raise ValueError("logical window attribute cannot be empty")
    training_case_ids = _ordered_unique(
        corpus.training_case_ids
    )
    validation_case_ids = _ordered_unique(
        corpus.validation_case_ids
    )
    required_case_ids = training_case_ids + validation_case_ids
    missing = set(required_case_ids) - set(log_captures)
    if missing:
        raise ValueError(
            f"missing application-log captures: {sorted(missing)}"
        )
    vocabulary = StructuredEventVocabulary.fit(
        (log_captures[case_id] for case_id in training_case_ids),
        numeric_attribute_names=numeric_attribute_names,
    )
    compiled = {
        case_id: vocabulary.compile(log_captures[case_id])
        for case_id in required_case_ids
    }
    entity_positions = {
        entity_id: position
        for position, entity_id in enumerate(
            corpus.training.entity_ids
        )
    }
    overrides = (
        {} if event_entity_overrides is None
        else event_entity_overrides
    )
    service_overrides = (
        {}
        if service_event_entity_overrides is None
        else service_event_entity_overrides
    )
    _validate_entity_mappings(
        entity_positions,
        service_to_entity,
        overrides,
        service_overrides,
    )

    requirements = _point_requirements(corpus)
    raw_by_case: Dict[str, NDArray[np.float64]] = {}
    applicable = np.zeros(
        (
            len(entity_positions),
            _feature_count(vocabulary),
        ),
        dtype=np.bool_,
    )
    training_numeric_values = []
    trace_link_count = 0
    record_count = 0
    validation_unknown_count = 0
    for case_id in required_case_ids:
        capture = log_captures[case_id]
        structured = compiled[case_id]
        is_training = case_id in training_case_ids
        raw, case_applicable, numeric_observations = (
            _aggregate_case(
                capture.records,
                structured,
                vocabulary,
                point_count=requirements[case_id],
                logical_window_attribute=(
                    logical_window_attribute
                ),
                entity_positions=entity_positions,
                service_to_entity=service_to_entity,
                event_entity_overrides=overrides,
                service_event_entity_overrides=(
                    service_overrides
                ),
            )
        )
        raw_by_case[case_id] = raw
        if is_training:
            applicable |= case_applicable
            training_numeric_values.extend(numeric_observations)
        else:
            validation_unknown_count += int(
                np.count_nonzero(structured.template_ids == 0)
            )
        trace_link_count += sum(
            bool(trace_id or span_id)
            for trace_id, span_id in zip(
                structured.trace_ids,
                structured.span_ids,
            )
        )
        record_count += len(capture.records)

    numeric_scale = _fit_numeric_scale(
        training_numeric_values,
        len(vocabulary.numeric_attribute_names),
    )
    transformed = {
        case_id: _transform_features(
            raw,
            len(vocabulary.templates),
            numeric_scale,
        )
        for case_id, raw in raw_by_case.items()
    }
    feature_names = _feature_names(vocabulary)
    training = _align_split(
        corpus.training.contexts.shape[1],
        corpus.training.horizons,
        corpus.training.target_block_size,
        corpus.training.point_indices,
        corpus.training_case_ids,
        transformed,
        applicable,
        feature_names,
    )
    validation = _align_split(
        corpus.validation.contexts.shape[1],
        corpus.validation.horizons,
        corpus.validation.target_block_size,
        corpus.validation.point_indices,
        corpus.validation_case_ids,
        transformed,
        applicable,
        feature_names,
    )
    return HybridEventCorpus(
        training=training,
        validation=validation,
        vocabulary=vocabulary,
        data_quality={
            "record_count": record_count,
            "training_record_count": sum(
                len(log_captures[case_id].records)
                for case_id in training_case_ids
            ),
            "validation_record_count": sum(
                len(log_captures[case_id].records)
                for case_id in validation_case_ids
            ),
            "training_template_count": len(
                vocabulary.templates
            ),
            "validation_unknown_event_count": (
                validation_unknown_count
            ),
            "trace_link_count": trace_link_count,
            "trace_link_coverage": (
                trace_link_count / record_count
                if record_count
                else 0.0
            ),
            "numeric_centers": numeric_scale.center.tolist(),
            "numeric_scales": numeric_scale.scale.tolist(),
            "preprocessing_fitted_on_training_only": True,
            "context_crosses_run_boundary": False,
            "target_crosses_run_boundary": False,
        },
    )


def _aggregate_case(
    records: Tuple[LogRecord, ...],
    structured: CompiledStructuredEvents,
    vocabulary: StructuredEventVocabulary,
    *,
    point_count: int,
    logical_window_attribute: str,
    entity_positions: Mapping[str, int],
    service_to_entity: Mapping[str, str],
    event_entity_overrides: Mapping[str, str],
    service_event_entity_overrides: Mapping[str, str],
) -> Tuple[
    NDArray[np.float64],
    NDArray[np.bool_],
    list[Tuple[int, float]],
]:
    template_count = len(vocabulary.templates)
    feature_count = _feature_count(vocabulary)
    values = np.zeros(
        (point_count, len(entity_positions), feature_count),
        dtype=np.float64,
    )
    applicable = np.zeros(
        (len(entity_positions), feature_count),
        dtype=np.bool_,
    )
    numeric_observations: list[Tuple[int, float]] = []
    severity_count = np.zeros(
        (point_count, len(entity_positions)), dtype=np.float64
    )
    delta_count = severity_count.copy()
    numeric_counts = np.zeros(
        (
            point_count,
            len(entity_positions),
            len(vocabulary.numeric_attribute_names),
        ),
        dtype=np.float64,
    )
    for index, (record, service_name) in enumerate(
        zip(records, structured.service_names)
    ):
        point = _logical_window(
            record, logical_window_attribute
        )
        if point >= point_count:
            raise ValueError(
                f"logical window {point} exceeds graph run length "
                f"{point_count}"
            )
        template_id = int(structured.template_ids[index])
        event_name = _event_name(vocabulary, template_id)
        entity_id = _resolve_entity(
            service_name,
            event_name,
            service_to_entity,
            event_entity_overrides,
            service_event_entity_overrides,
        )
        entity = entity_positions[entity_id]
        values[point, entity, template_id] += 1.0
        total_position = template_count + 1
        values[point, entity, total_position] += 1.0
        severity_position = total_position + 1
        values[
            point, entity, severity_position
        ] += float(structured.severity_numbers[index])
        severity_count[point, entity] += 1.0
        delta_position = severity_position + 1
        values[
            point, entity, delta_position
        ] += float(structured.delta_seconds[index])
        delta_count[point, entity] += 1.0
        trace_position = delta_position + 1
        if structured.trace_ids[index] or structured.span_ids[index]:
            values[point, entity, trace_position] += 1.0
        applicable[entity, template_id] = True
        applicable[
            entity, total_position : trace_position + 1
        ] = True
        for numeric_index in range(
            len(vocabulary.numeric_attribute_names)
        ):
            if not structured.numeric_mask[
                index, numeric_index
            ]:
                continue
            numeric_value = float(
                structured.numeric_values[
                    index, numeric_index
                ]
            )
            mean_position = (
                trace_position + 1 + numeric_index * 2
            )
            ratio_position = mean_position + 1
            values[
                point, entity, mean_position
            ] += numeric_value
            numeric_counts[
                point, entity, numeric_index
            ] += 1.0
            applicable[
                entity, mean_position : ratio_position + 1
            ] = True
            numeric_observations.append(
                (numeric_index, numeric_value)
            )
    total_position = template_count + 1
    severity_position = total_position + 1
    delta_position = severity_position + 1
    trace_position = delta_position + 1
    np.divide(
        values[..., severity_position],
        severity_count,
        out=values[..., severity_position],
        where=severity_count > 0,
    )
    np.divide(
        values[..., delta_position],
        delta_count,
        out=values[..., delta_position],
        where=delta_count > 0,
    )
    np.divide(
        values[..., trace_position],
        severity_count,
        out=values[..., trace_position],
        where=severity_count > 0,
    )
    for numeric_index in range(
        len(vocabulary.numeric_attribute_names)
    ):
        mean_position = (
            trace_position + 1 + numeric_index * 2
        )
        ratio_position = mean_position + 1
        counts = numeric_counts[..., numeric_index]
        np.divide(
            values[..., mean_position],
            counts,
            out=values[..., mean_position],
            where=counts > 0,
        )
        np.divide(
            counts,
            severity_count,
            out=values[..., ratio_position],
            where=severity_count > 0,
        )
    event_entities = np.any(
        applicable[:, : template_count + 1], axis=1
    )
    applicable[event_entities, 0] = True
    return values, applicable, numeric_observations


def _transform_features(
    values: NDArray[np.float64],
    template_count: int,
    numeric_scale: _NumericScale,
) -> NDArray[np.float64]:
    transformed = values.copy()
    total_position = template_count + 1
    transformed[..., : total_position + 1] = np.log1p(
        transformed[..., : total_position + 1]
    )
    severity_position = total_position + 1
    transformed[..., severity_position] /= 24.0
    delta_position = severity_position + 1
    transformed[..., delta_position] = np.log1p(
        transformed[..., delta_position]
    )
    trace_position = delta_position + 1
    for numeric_index in range(len(numeric_scale.center)):
        mean_position = (
            trace_position + 1 + numeric_index * 2
        )
        ratio_position = mean_position + 1
        observed = transformed[..., ratio_position] > 0.0
        transformed[..., mean_position][observed] = np.clip(
            (
                transformed[..., mean_position][observed]
                - numeric_scale.center[numeric_index]
            )
            / numeric_scale.scale[numeric_index],
            -8.0,
            8.0,
        )
    return transformed


def _fit_numeric_scale(
    observations: Sequence[Tuple[int, float]],
    numeric_count: int,
) -> _NumericScale:
    center = np.zeros(numeric_count, dtype=np.float64)
    scale = np.ones(numeric_count, dtype=np.float64)
    for numeric_index in range(numeric_count):
        selected = np.asarray(
            [
                value
                for index, value in observations
                if index == numeric_index
            ],
            dtype=np.float64,
        )
        if not len(selected):
            continue
        center[numeric_index] = float(np.median(selected))
        lower, upper = np.percentile(selected, (25.0, 75.0))
        robust = float(upper - lower)
        scale[numeric_index] = (
            robust
            if robust > 1e-9
            else max(float(np.max(np.abs(selected))), 1.0)
        )
    return _NumericScale(center=center, scale=scale)


def _align_split(
    lookback: int,
    horizons: Tuple[int, ...],
    target_block_size: int,
    point_indices: NDArray[np.int64],
    case_ids: Tuple[str, ...],
    values_by_case: Mapping[str, NDArray[np.float64]],
    observation_mask: NDArray[np.bool_],
    feature_names: Tuple[str, ...],
) -> AlignedEventFeatures:
    contexts = np.stack(
        [
            values_by_case[case_id][
                int(point) - lookback : int(point)
            ]
            for point, case_id in zip(point_indices, case_ids)
        ]
    )
    target_blocks = np.stack(
        [
            np.stack(
                [
                    values_by_case[case_id][
                        int(point) + horizon - 1 :
                        int(point)
                        + horizon
                        - 1
                        + target_block_size
                    ]
                    for horizon in horizons
                ]
            )
            for point, case_id in zip(point_indices, case_ids)
        ]
    )
    return AlignedEventFeatures(
        contexts=contexts,
        target_blocks=target_blocks,
        observation_mask=observation_mask.copy(),
        feature_names=feature_names,
    )


def _point_requirements(
    corpus: ObservabilityGraphCorpus,
) -> Dict[str, int]:
    required: Dict[str, int] = {}
    for windows, case_ids in (
        (corpus.training, corpus.training_case_ids),
        (corpus.validation, corpus.validation_case_ids),
    ):
        extension = (
            windows.horizons[-1]
            - 1
            + windows.target_block_size
        )
        for point, case_id in zip(
            windows.point_indices, case_ids
        ):
            required[case_id] = max(
                required.get(case_id, 0),
                int(point) + extension,
            )
    return required


def _logical_window(
    record: LogRecord,
    attribute_name: str,
) -> int:
    raw = record.record_attributes.get(attribute_name)
    if isinstance(raw, bool):
        raise ValueError("logical window must be a non-negative integer")
    if isinstance(raw, Integral):
        result = int(raw)
    elif isinstance(raw, Real) and float(raw).is_integer():
        result = int(raw)
    elif isinstance(raw, str):
        try:
            result = int(raw)
        except ValueError as error:
            raise ValueError(
                "logical window must be a non-negative integer"
            ) from error
    else:
        raise ValueError(
            "logical window must be a non-negative integer"
        )
    if result < 0:
        raise ValueError(
            "logical window must be a non-negative integer"
        )
    return result


def _event_name(
    vocabulary: StructuredEventVocabulary,
    template_id: int,
) -> str:
    if template_id == 0:
        return "<unknown>"
    return vocabulary.templates[template_id - 1]


def _resolve_entity(
    service_name: str,
    event_name: str,
    service_to_entity: Mapping[str, str],
    event_entity_overrides: Mapping[str, str],
    service_event_entity_overrides: Mapping[str, str],
) -> str:
    combined = f"{service_name}|{event_name}"
    if combined in service_event_entity_overrides:
        return service_event_entity_overrides[combined]
    if event_name.startswith("event:"):
        alias = event_name.split(":", 1)[1]
        combined_alias = f"{service_name}|{alias}"
        if combined_alias in service_event_entity_overrides:
            return service_event_entity_overrides[combined_alias]
    if event_name in event_entity_overrides:
        return event_entity_overrides[event_name]
    if (
        event_name.startswith("event:")
        and event_name.split(":", 1)[1] in event_entity_overrides
    ):
        return event_entity_overrides[
            event_name.split(":", 1)[1]
        ]
    if service_name in service_to_entity:
        return service_to_entity[service_name]
    raise ValueError(
        "unmapped service/event: "
        f"service={service_name!r}, event={event_name!r}"
    )


def _validate_entity_mappings(
    entity_positions: Mapping[str, int],
    *mappings: Mapping[str, str],
) -> None:
    unknown = {
        entity_id
        for mapping in mappings
        for entity_id in mapping.values()
        if entity_id not in entity_positions
    }
    if unknown:
        raise ValueError(
            f"event mappings reference unknown entities: "
            f"{sorted(unknown)}"
        )


def _feature_count(
    vocabulary: StructuredEventVocabulary,
) -> int:
    return (
        len(vocabulary.templates)
        + 1
        + 4
        + 2 * len(vocabulary.numeric_attribute_names)
    )


def _feature_names(
    vocabulary: StructuredEventVocabulary,
) -> Tuple[str, ...]:
    templates = ("<unknown>",) + vocabulary.templates
    numeric = tuple(
        value
        for name in vocabulary.numeric_attribute_names
        for value in (
            f"event.numeric.{name}.mean",
            f"event.numeric.{name}.observed_ratio",
        )
    )
    return tuple(
        f"event.template.{template}.log_count"
        for template in templates
    ) + (
        "event.total.log_count",
        "event.severity.mean",
        "event.related_delta.log_mean_seconds",
        "event.trace_link.ratio",
    ) + numeric


def _ordered_unique(values: Tuple[str, ...]) -> Tuple[str, ...]:
    return tuple(dict.fromkeys(values))
