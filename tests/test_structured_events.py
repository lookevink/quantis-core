import hashlib
import json
from dataclasses import replace
from typing import Mapping, Optional

import numpy as np
import pytest

from quantis_core.otlp import AttributeValue
from quantis_core.otlp_logs import LogRecord, OtlpLogCapture
from quantis_core.structured_events import (
    CompiledStructuredEvents,
    StructuredEventVocabulary,
)


def _record(
    *,
    time_unix_nano: int,
    body: Optional[AttributeValue] = "fallback body",
    event_name: Optional[AttributeValue] = "checkout.completed",
    service_namespace: str = "",
    service_name: str = "api",
    service_instance_id: str = "api-1",
    severity_number: int = 9,
    severity_text: str = "INFO",
    trace_id: str = "",
    span_id: str = "",
    attributes: Optional[Mapping[str, AttributeValue]] = None,
) -> LogRecord:
    record_attributes = (
        {} if attributes is None else dict(attributes)
    )
    if event_name is not None:
        record_attributes["event.name"] = event_name
    return LogRecord(
        time_unix_nano=time_unix_nano,
        observed_time_unix_nano=None,
        severity_number=severity_number,
        severity_text=severity_text,
        body=body,
        resource_attributes={
            "service.namespace": service_namespace,
            "service.name": service_name,
            "service.instance.id": service_instance_id,
        },
        record_attributes=record_attributes,
        scope_name="quantis.application",
        scope_version="1.0",
        trace_id=trace_id,
        span_id=span_id,
        flags=0,
        dropped_attributes_count=0,
    )


def _capture(*records: LogRecord, label: str = "capture") -> OtlpLogCapture:
    return OtlpLogCapture(
        records=records,
        sha256=hashlib.sha256(label.encode("utf-8")).hexdigest(),
        source_path=f"memory://{label}",
        json_message_count=1,
    )


def test_vocabulary_is_fit_only_from_training_event_names() -> None:
    training = _capture(
        _record(time_unix_nano=2, event_name="worker.completed"),
        _record(time_unix_nano=1, event_name="checkout.accepted"),
    )

    vocabulary = StructuredEventVocabulary.fit((training,))
    compiled_training = vocabulary.compile(training)
    validation = _capture(
        _record(time_unix_nano=3, event_name="validation.only"),
        label="validation",
    )

    assert vocabulary.templates == (
        "event:checkout.accepted",
        "event:worker.completed",
    )
    assert compiled_training.template_ids.tolist() == [2, 1]
    assert vocabulary.compile(validation).template_ids.tolist() == [0]


def test_body_fallback_normalizes_uuid_hex_numbers_and_whitespace() -> None:
    training = _capture(
        _record(
            time_unix_nano=1,
            event_name=None,
            body=(
                " User 42 opened 550e8400-e29b-41d4-a716-446655440000 "
                "at 0xDEADBEEF in 1.25e-3 seconds "
            ),
        ),
        _record(
            time_unix_nano=2,
            event_name=" ",
            body=(
                "user 900 opened 123e4567-e89b-12d3-a456-426614174000 "
                "at DEADBEEF in 8 seconds"
            ),
        ),
    )

    vocabulary = StructuredEventVocabulary.fit((training,))

    assert vocabulary.templates == (
        "body:user <num> opened <uuid> at <hex> in <num> seconds",
    )
    assert vocabulary.compile(training).template_ids.tolist() == [1, 1]


def test_compilation_preserves_identity_linkage_time_and_typed_parameters(
) -> None:
    capture = _capture(
        _record(
            time_unix_nano=1_000_000_000,
            event_name="request.accepted",
            service_name="api",
            service_instance_id="api-1",
            severity_number=9,
            severity_text="INFO",
            trace_id="trace-a",
            span_id="span-api",
            attributes={"duration_ms": 1.25, "attempt": 1},
        ),
        _record(
            time_unix_nano=1_250_000_000,
            event_name="database.completed",
            service_name="worker",
            service_instance_id="worker-2",
            severity_number=13,
            severity_text="WARN",
            trace_id="trace-a",
            span_id="span-db",
            attributes={"duration_ms": 250, "attempt": True},
        ),
        _record(
            time_unix_nano=2_000_000_000,
            event_name="heartbeat",
            service_name="worker",
            service_instance_id="worker-2",
            trace_id="",
            attributes={"duration_ms": float("nan")},
        ),
    )
    vocabulary = StructuredEventVocabulary.fit(
        (capture,),
        numeric_attribute_names=("attempt", "duration_ms", "missing"),
    )

    with pytest.raises(ValueError, match="finite"):
        vocabulary.compile(capture)

    valid_capture = _capture(*capture.records[:2], label="valid")
    compiled = vocabulary.compile(valid_capture)

    assert compiled.service_names == ("api", "worker")
    assert compiled.service_namespaces == ("", "")
    assert compiled.service_instance_ids == ("api-1", "worker-2")
    assert compiled.severity_numbers.tolist() == [9, 13]
    assert compiled.severity_texts == ("INFO", "WARN")
    assert compiled.trace_ids == ("trace-a", "trace-a")
    assert compiled.span_ids == ("span-api", "span-db")
    assert compiled.event_time_unix_nano.tolist() == [
        1_000_000_000,
        1_250_000_000,
    ]
    assert compiled.delta_seconds.tolist() == [0.0, 0.25]
    assert compiled.numeric_attribute_names == (
        "attempt",
        "duration_ms",
        "missing",
    )
    np.testing.assert_array_equal(
        compiled.numeric_mask,
        [[True, True, False], [False, True, False]],
    )
    np.testing.assert_allclose(
        compiled.numeric_values,
        [[1.0, 1.25, 0.0], [0.0, 250.0, 0.0]],
    )
    assert compiled.capture_sha256 == valid_capture.sha256
    assert compiled.vocabulary_schema_id == vocabulary.schema_id


def test_untraced_event_deltas_are_scoped_to_full_service_identity() -> None:
    capture = _capture(
        _record(
            time_unix_nano=1_000_000_000,
            event_name="heartbeat",
            service_namespace="checkout",
            service_name="worker",
            service_instance_id="worker-1",
        ),
        _record(
            time_unix_nano=1_500_000_000,
            event_name="heartbeat",
            service_namespace="checkout",
            service_name="worker",
            service_instance_id="worker-1",
        ),
        _record(
            time_unix_nano=2_000_000_000,
            event_name="heartbeat",
            service_namespace="billing",
            service_name="worker",
            service_instance_id="worker-1",
        ),
    )

    compiled = StructuredEventVocabulary.fit((capture,)).compile(capture)

    assert compiled.service_namespaces == (
        "checkout",
        "checkout",
        "billing",
    )
    assert compiled.delta_seconds.tolist() == [0.0, 0.5, 0.0]


def test_vocabulary_and_compiled_events_round_trip_deterministically() -> None:
    capture = _capture(
        _record(
            time_unix_nano=10,
            event_name="checkout.completed",
            attributes={"duration_ms": 4.5},
        )
    )
    vocabulary = StructuredEventVocabulary.fit(
        (capture,),
        numeric_attribute_names=("duration_ms",),
    )
    compiled = vocabulary.compile(capture)

    restored_vocabulary = StructuredEventVocabulary.from_dict(
        vocabulary.to_dict()
    )
    restored_compiled = CompiledStructuredEvents.from_dict(
        compiled.to_dict()
    )

    assert restored_vocabulary.to_dict() == vocabulary.to_dict()
    assert restored_vocabulary.schema_id == vocabulary.schema_id
    assert restored_compiled.to_dict() == compiled.to_dict()
    assert json.dumps(
        compiled.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def test_structured_event_artifacts_reject_invalid_metadata_shapes_and_values(
) -> None:
    with pytest.raises(ValueError, match="unique"):
        StructuredEventVocabulary(
            templates=("event:a", "event:a"),
        )
    with pytest.raises(ValueError, match="numeric attribute"):
        StructuredEventVocabulary(
            templates=("event:a",),
            numeric_attribute_names=("duration", "duration"),
        )

    capture = _capture(
        _record(time_unix_nano=10, event_name="a"),
        _record(time_unix_nano=20, event_name="b"),
    )
    compiled = StructuredEventVocabulary.fit((capture,)).compile(capture)

    with pytest.raises(ValueError, match="metadata"):
        replace(compiled, service_names=("api",))
    with pytest.raises(ValueError, match="finite"):
        replace(
            compiled,
            delta_seconds=np.asarray(
                [0.0, np.inf],
                dtype=np.float64,
            ),
        )
    with pytest.raises(ValueError, match="numeric arrays"):
        replace(
            compiled,
            numeric_values=np.zeros((2, 1), dtype=np.float64),
        )
