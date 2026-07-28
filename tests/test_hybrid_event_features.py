from dataclasses import replace
from typing import Any, Mapping, Optional, Tuple

import numpy as np

from quantis_core.graph_telemetry import (
    DeclaredTelemetryGraph,
    GraphEntity,
    GraphStateWindows,
    TelemetryBinding,
)
from quantis_core.hybrid_event_features import (
    HybridEventCorpus,
    compile_hybrid_event_corpus,
)
from quantis_core.observability_graph_corpus import (
    ObservabilityGraphCorpus,
)
from quantis_core.otlp_logs import (
    LogRecord,
    OtlpLogCapture,
)


def test_compiles_training_fitted_events_into_graph_windows() -> None:
    corpus = _corpus()
    captures = {
        "train": _capture(
            "train",
            (
                _record(1, 0, "checkout.accepted", "api", 200),
                _record(2, 1, "checkout.completed", "worker", None),
                _record(3, 2, "checkout.accepted", "api", 201),
            ),
        ),
        "validation": _capture(
            "validation",
            (
                _record(1, 0, "checkout.accepted", "api", 503),
                _record(2, 1, "new.validation.event", "api", None),
                _record(3, 2, "checkout.completed", "worker", None),
            ),
        ),
    }

    compiled = compile_hybrid_event_corpus(
        corpus,
        captures,
        logical_window_attribute="logical.window",
        numeric_attribute_names=("http.response.status_code",),
        service_to_entity={
            "api": "api",
            "worker": "worker_pool",
        },
        event_entity_overrides={
            "checkout.completed": "worker_pool",
        },
    )

    assert isinstance(compiled, HybridEventCorpus)
    assert compiled.vocabulary.templates == (
        "event:checkout.accepted",
        "event:checkout.completed",
    )
    assert compiled.training.contexts.shape == (1, 2, 2, 9)
    assert compiled.training.target_blocks.shape == (1, 1, 1, 2, 9)
    accepted = compiled.training.feature_names.index(
        "event.template.event:checkout.accepted.log_count"
    )
    completed = compiled.training.feature_names.index(
        "event.template.event:checkout.completed.log_count"
    )
    assert compiled.training.contexts[0, 0, 0, accepted] != 0.0
    assert compiled.training.contexts[0, 1, 1, completed] != 0.0
    assert compiled.data_quality["validation_unknown_event_count"] == 1
    assert compiled.data_quality["trace_link_coverage"] == 0.0
    assert compiled.data_quality["preprocessing_fitted_on_training_only"]


def test_named_event_alias_never_captures_same_text_body_template() -> None:
    corpus = _corpus()
    named_event = _record(
        1,
        0,
        "checkout.completed",
        "api",
        None,
    )
    body_template = replace(
        _record(
            2,
            1,
            "ignored.event.name",
            "api",
            None,
        ),
        body="checkout.completed",
        record_attributes={"logical.window": "1"},
    )
    capture = _capture(
        "collision",
        (named_event, body_template),
    )

    compiled = compile_hybrid_event_corpus(
        corpus,
        {"train": capture, "validation": capture},
        logical_window_attribute="logical.window",
        service_to_entity={"api": "api"},
        event_entity_overrides={
            "checkout.completed": "worker_pool",
        },
    )

    assert compiled.vocabulary.templates == (
        "body:checkout.completed",
        "event:checkout.completed",
    )
    body_feature = compiled.training.feature_names.index(
        "event.template.body:checkout.completed.log_count"
    )
    named_feature = compiled.training.feature_names.index(
        "event.template.event:checkout.completed.log_count"
    )
    api = corpus.training.entity_ids.index("api")
    worker = corpus.training.entity_ids.index("worker_pool")
    assert compiled.training.contexts[0, 0, worker, named_feature] > 0.0
    assert compiled.training.contexts[0, 1, api, body_feature] > 0.0
    assert compiled.training.contexts[0, 1, worker, body_feature] == 0.0


def test_aligns_each_case_without_crossing_run_boundaries() -> None:
    corpus = _corpus()
    second_validation = replace(
        corpus.validation,
        contexts=np.concatenate(
            (corpus.validation.contexts, corpus.validation.contexts)
        ),
        target_blocks=np.concatenate(
            (
                corpus.validation.target_blocks,
                corpus.validation.target_blocks,
            )
        ),
        target_controls=np.concatenate(
            (
                corpus.validation.target_controls,
                corpus.validation.target_controls,
            )
        ),
        point_indices=np.asarray([2, 2], dtype=np.int64),
    )
    corpus = replace(
        corpus,
        validation=second_validation,
        validation_case_ids=("validation", "validation-2"),
    )
    captures = {
        "train": _capture(
            "train",
            (_record(1, 0, "event.a", "api", None),),
        ),
        "validation": _capture(
            "validation",
            (_record(1, 0, "event.a", "api", None),),
        ),
        "validation-2": _capture(
            "validation-2",
            (_record(1, 1, "event.a", "api", None),),
        ),
    }

    compiled = compile_hybrid_event_corpus(
        corpus,
        captures,
        logical_window_attribute="logical.window",
        service_to_entity={"api": "api"},
    )

    count_position = compiled.validation.feature_names.index(
        "event.total.log_count"
    )
    first = compiled.validation.contexts[0, :, 0, count_position]
    second = compiled.validation.contexts[1, :, 0, count_position]
    assert not np.array_equal(first, second)


def test_rejects_unmapped_services_and_invalid_logical_windows() -> None:
    corpus = _corpus()
    valid = _capture(
        "valid",
        (_record(1, 0, "event.a", "api", None),),
    )
    unmapped = _capture(
        "unmapped",
        (_record(1, 0, "event.a", "other", None),),
    )
    bad_record = replace(
        _record(1, 0, "event.a", "api", None),
        record_attributes={
            "event.name": "event.a",
            "logical.window": "not-an-integer",
        },
    )
    invalid = OtlpLogCapture((bad_record,), "c" * 64, "invalid", 1)

    try:
        compile_hybrid_event_corpus(
            corpus,
            {"train": unmapped, "validation": valid},
            logical_window_attribute="logical.window",
            service_to_entity={"api": "api"},
        )
    except ValueError as error:
        assert "unmapped service/event" in str(error)
    else:
        raise AssertionError("unmapped services must be rejected")

    try:
        compile_hybrid_event_corpus(
            corpus,
            {"train": invalid, "validation": valid},
            logical_window_attribute="logical.window",
            service_to_entity={"api": "api"},
        )
    except ValueError as error:
        assert "logical window" in str(error)
    else:
        raise AssertionError("invalid logical windows must be rejected")


def _corpus() -> ObservabilityGraphCorpus:
    graph = DeclaredTelemetryGraph(
        entities=(
            GraphEntity("api", "node", "service"),
            GraphEntity("worker_pool", "node", "service_pool"),
        ),
        bindings=(
            TelemetryBinding("api.value", "api"),
            TelemetryBinding("worker.value", "worker_pool"),
        ),
    )
    windows = GraphStateWindows(
        contexts=np.zeros((1, 2, 2, 1), dtype=np.float64),
        target_blocks=np.zeros(
            (1, 1, 1, 2, 1), dtype=np.float64
        ),
        target_controls=np.zeros(
            (1, 1, 1, 1), dtype=np.float64
        ),
        point_indices=np.asarray([2], dtype=np.int64),
        observation_mask=np.ones((2, 1), dtype=np.bool_),
        entity_ids=graph.entity_ids,
        entity_kinds=("node", "node"),
        local_feature_keys=(("api.value",), ("worker.value",)),
        control_feature_names=("request_demand",),
        horizons=(1,),
        target_block_size=1,
        graph=graph,
    )
    return ObservabilityGraphCorpus(
        training=windows,
        validation=windows,
        training_case_ids=("train",),
        validation_case_ids=("validation",),
        provenance={"split_spec": {"lookback": 2}},
    )


def _capture(
    name: str,
    records: Tuple[LogRecord, ...],
) -> OtlpLogCapture:
    return OtlpLogCapture(records, "a" * 64, name, 1)


def _record(
    time: int,
    window: int,
    event_name: str,
    service: str,
    status: Optional[int],
) -> LogRecord:
    attributes: dict[str, Any] = {
        "event.name": event_name,
        "logical.window": str(window),
    }
    if status is not None:
        attributes["http.response.status_code"] = status
    resource_attributes: Mapping[str, Any] = {
        "service.name": service,
    }
    return LogRecord(
        time_unix_nano=time * 1_000_000_000,
        observed_time_unix_nano=None,
        severity_number=9,
        severity_text="INFO",
        body=event_name,
        resource_attributes=resource_attributes,
        record_attributes=attributes,
        scope_name="test",
        scope_version="",
        trace_id="",
        span_id="",
        flags=0,
        dropped_attributes_count=0,
    )
