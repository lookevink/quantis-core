import numpy as np

from quantis_core.otlp_log_windowing import (
    LogFeatureDefinition,
    OtlpLogFeatureSpec,
    OtlpLogWindowCompiler,
)
from quantis_core.otlp_logs import read_otlp_log_capture


def test_log_capture_reader_preserves_structured_otlp_semantics() -> None:
    capture = read_otlp_log_capture(
        "tests/fixtures/otlp/application-logs.jsonl"
    )

    assert len(capture.records) == 2
    assert capture.json_message_count == 1
    assert capture.sha256 == (
        "e0dd02dbbe20cbe5986f27871c5d63cb0284d98dd7a686de5e60c696ba4ed114"
    )

    accepted = capture.records[0]
    assert accepted.time_unix_nano == 1_000_000_100
    assert accepted.observed_time_unix_nano == 1_000_000_200
    assert accepted.severity_number == 9
    assert accepted.severity_text == "INFO"
    assert accepted.body == "checkout accepted"
    assert accepted.record_attributes == {
        "event.name": "checkout.accepted",
        "http.response.status_code": 202,
        "quantis.experiment.window.index": 0,
    }
    assert accepted.resource_attributes == {
        "quantis.experiment.case.id": "log-fixture-01",
        "quantis.experiment.fault.kind": "none",
        "quantis.experiment.manifest.sha256": (
            "manifest-fixture-sha"
        ),
        "service.instance.id": "api-1",
        "service.name": "checkout-api",
    }
    assert accepted.scope_name == "quantis.application"
    assert accepted.scope_version == "1.0.0"
    assert accepted.trace_id == "00112233445566778899aabbccddeeff"
    assert accepted.span_id == "0011223344556677"


def test_log_window_compiler_counts_only_declared_structured_events() -> None:
    capture = read_otlp_log_capture(
        "tests/fixtures/otlp/application-logs.jsonl"
    )
    feature_spec = OtlpLogFeatureSpec(
        window_index_attribute="quantis.experiment.window.index",
        features=(
            LogFeatureDefinition(
                name="checkout_accepted_count",
                record_attributes={"event.name": "checkout.accepted"},
            ),
            LogFeatureDefinition(
                name="checkout_rejected_count",
                record_attributes={"event.name": "checkout.rejected"},
            ),
            LogFeatureDefinition(
                name="error_event_count",
                minimum_severity_number=17,
            ),
        ),
    )

    compiled = OtlpLogWindowCompiler(feature_spec).compile(
        capture,
        window_count=3,
    )

    assert compiled.window_indices.tolist() == [0, 1, 2]
    assert compiled.feature_names == (
        "checkout_accepted_count",
        "checkout_rejected_count",
        "error_event_count",
    )
    np.testing.assert_array_equal(
        compiled.values,
        np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0],
            ]
        ),
    )
    assert compiled.data_quality == {
        "matched_records": 2,
        "record_count": 2,
        "unmatched_records": 0,
    }
