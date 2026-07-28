import hashlib
import json

import numpy as np

from quantis_core.contextual_multimodal_corpus import (
    DemandResidualLogTransformer,
    DependencyResidualLogTransformer,
    compile_contextual_multimodal_telemetry_corpus,
)
from quantis_core.multimodal_corpus import (
    compile_multimodal_telemetry_corpus,
)
from quantis_core.otlp_log_windowing import OtlpLogFeatureSpec
from quantis_core.otlp_logs import LogRecord, OtlpLogCapture
from quantis_core.telemetry_corpus import TelemetryCorpusSplitSpec
from tests.corpus_test_support import (
    FRESH_CASE_IDS,
    fresh_development_runs,
)
from tests.multimodal_test_support import (
    normal_log_captures,
    v2_normal_log_captures,
)


def test_log_transformer_expresses_application_state_against_demand() -> None:
    transformed = DemandResidualLogTransformer().transform(
        np.asarray(
            [
                [10.0, 1.0, 8.0, 2.0],
                [4.0, 0.0, 4.0, 0.0],
            ]
        ),
        (
            "checkout_accepted_count",
            "checkout_rejected_count",
            "checkout_completed_count",
            "error_event_count",
        ),
        np.asarray([10.0, 4.0]),
    )

    assert transformed.feature_names == (
        "checkout_completion_ratio",
        "checkout_backlog_delta_ratio",
        "checkout_rejection_rate",
        "application_error_event_rate",
    )
    np.testing.assert_allclose(
        transformed.values,
        np.asarray(
            [
                [0.8, 0.2, 0.1, 0.2],
                [1.0, 0.0, 0.0, 0.0],
            ]
        ),
    )


def test_log_transformer_retains_bounded_endogenous_state() -> None:
    transformed = DemandResidualLogTransformer().transform(
        np.asarray(
            [
                [
                    10.0,
                    0.0,
                    8.0,
                    0.0,
                    1.0,
                    2.0,
                    1.0,
                    5.0,
                    2.0,
                    1.0,
                    1.0,
                    1.0,
                ]
            ]
        ),
        (
            "checkout_accepted_count",
            "checkout_rejected_count",
            "checkout_completed_count",
            "error_event_count",
            "queue_backlog_low_transition_count",
            "queue_backlog_elevated_transition_count",
            "queue_backlog_high_transition_count",
            "database_latency_fast_count",
            "database_latency_normal_count",
            "database_latency_slow_count",
            "worker_busy_transition_count",
            "worker_idle_transition_count",
        ),
        np.asarray([10.0]),
    )

    assert transformed.feature_names == (
        "checkout_completion_ratio",
        "checkout_backlog_delta_ratio",
        "checkout_rejection_rate",
        "application_error_event_rate",
        "queue_backlog_low_transition_rate",
        "queue_backlog_elevated_transition_rate",
        "queue_backlog_high_transition_rate",
        "database_latency_fast_ratio",
        "database_latency_normal_ratio",
        "database_latency_slow_ratio",
        "worker_busy_transition_rate",
        "worker_idle_transition_rate",
    )
    np.testing.assert_allclose(
        transformed.values,
        np.asarray(
            [
                [
                    0.8,
                    0.2,
                    0.0,
                    0.0,
                    0.1,
                    0.2,
                    0.1,
                    0.625,
                    0.25,
                    0.125,
                    0.1,
                    0.1,
                ]
            ]
        ),
    )


def test_v2_log_transformer_keeps_pressure_without_complement_events() -> None:
    transformed = DependencyResidualLogTransformer().transform(
        np.asarray(
            [
                [
                    10.0,
                    1.0,
                    8.0,
                    2.0,
                    1.0,
                    1.0,
                    3.0,
                    1.0,
                    1.0,
                    2.0,
                    1.0,
                    0.0,
                    4.0,
                    2.0,
                ]
            ]
        ),
        (
            "checkout_accepted_count",
            "checkout_rejected_count",
            "checkout_completed_count",
            "queue_backlog_elevated_transition_count",
            "queue_backlog_high_transition_count",
            "worker_busy_transition_count",
            "redis_latency_elevated_count",
            "redis_latency_slow_count",
            "redis_operation_error_count",
            "postgresql_latency_elevated_count",
            "postgresql_latency_slow_count",
            "postgresql_operation_error_count",
            "checkout_queue_wait_elevated_count",
            "checkout_queue_wait_slow_count",
        ),
        np.asarray([10.0]),
    )

    assert transformed.feature_names == (
        "checkout_completion_ratio",
        "checkout_backlog_delta_ratio",
        "checkout_rejection_rate",
        "queue_pressure_transition_rate",
        "queue_high_transition_rate",
        "postgresql_latency_pressure_ratio",
        "postgresql_slow_or_error_ratio",
        "worker_activation_rate",
        "redis_latency_pressure_rate",
        "redis_slow_or_error_rate",
        "checkout_queue_wait_pressure_ratio",
        "checkout_queue_wait_slow_ratio",
    )
    np.testing.assert_allclose(
        transformed.values,
        np.asarray(
            [
                [
                    0.8,
                    0.2,
                    0.1,
                    0.3,
                    0.1,
                    0.375,
                    0.125,
                    0.1,
                    0.4,
                    0.2,
                    0.75,
                    0.25,
                ]
            ]
        ),
    )
    assert transformed.feature_names.count(
        "database_latency_fast_ratio"
    ) == 0
    assert transformed.feature_names.count(
        "worker_idle_transition_rate"
    ) == 0


def test_contextual_corpus_builds_conditioned_multihorizon_blocks() -> None:
    runs, metric_spec = fresh_development_runs()
    log_spec = OtlpLogFeatureSpec.from_dict(
        json.loads(
            open(
                "lab/fault_matrix/log-feature-spec.json"
            ).read()
        )
    )
    split_spec = TelemetryCorpusSplitSpec(
        training_case_ids=FRESH_CASE_IDS[:2],
        validation_case_ids=(FRESH_CASE_IDS[2],),
        reserved_case_ids=(),
        lookback=6,
    )
    base = compile_multimodal_telemetry_corpus(
        runs,
        normal_log_captures(runs),
        metric_spec,
        log_spec,
        split_spec,
    )

    corpus = compile_contextual_multimodal_telemetry_corpus(
        base,
        runs,
        horizons=(1, 3, 6),
        target_block_size=2,
    )

    assert corpus.metadata_dict()["schema_version"] == 2
    windows = corpus.training.windows
    assert windows.metric_contexts.shape == (48, 6, 6)
    assert windows.log_contexts.shape == (48, 6, 4)
    assert windows.metric_target_blocks.shape == (48, 3, 2, 6)
    assert windows.log_target_blocks.shape == (48, 3, 2, 4)
    assert windows.target_controls.shape == (48, 3, 2, 2)
    assert windows.horizons == (1, 3, 6)
    assert windows.target_block_size == 2
    assert windows.control_feature_names == (
        "request_demand",
        "worker_replicas",
    )
    assert windows.log_feature_names == (
        "checkout_completion_ratio",
        "checkout_backlog_delta_ratio",
        "checkout_rejection_rate",
        "application_error_event_rate",
    )
    assert corpus.training.window_case_ids == (
        (FRESH_CASE_IDS[0],) * 24
        + (FRESH_CASE_IDS[1],) * 24
    )
    assert corpus.protocol["context_crosses_run_boundary"] is False
    assert corpus.protocol["target_crosses_run_boundary"] is False
    assert corpus.protocol["target_horizons"] == [1, 3, 6]
    assert corpus.preprocessing["logs"]["transformer"] == {
        "schema_version": 1,
        "kind": "demand_residual_application_logs",
        "features": list(windows.log_feature_names),
    }


def test_contextual_corpus_compiles_rich_promotion_log_vocabulary() -> None:
    runs, metric_spec = fresh_development_runs()
    log_spec = OtlpLogFeatureSpec.from_dict(
        json.loads(
            open(
                "lab/fault_matrix/"
                "contextual-promotion-log-feature-spec.json"
            ).read()
        )
    )
    base = compile_multimodal_telemetry_corpus(
        runs,
        _rich_normal_log_captures(runs),
        metric_spec,
        log_spec,
        TelemetryCorpusSplitSpec(
            training_case_ids=FRESH_CASE_IDS[:2],
            validation_case_ids=(FRESH_CASE_IDS[2],),
            reserved_case_ids=(),
            lookback=6,
        ),
    )

    corpus = compile_contextual_multimodal_telemetry_corpus(
        base,
        runs,
    )

    assert corpus.training.windows.log_contexts.shape == (
        48,
        6,
        12,
    )
    assert corpus.training.windows.log_feature_names == (
        "checkout_completion_ratio",
        "checkout_backlog_delta_ratio",
        "checkout_rejection_rate",
        "application_error_event_rate",
        "queue_backlog_low_transition_rate",
        "queue_backlog_elevated_transition_rate",
        "queue_backlog_high_transition_rate",
        "database_latency_fast_ratio",
        "database_latency_normal_ratio",
        "database_latency_slow_ratio",
        "worker_busy_transition_rate",
        "worker_idle_transition_rate",
    )
    assert corpus.preprocessing["logs"]["transformer"][
        "features"
    ] == list(corpus.training.windows.log_feature_names)


def test_contextual_corpus_selects_v2_dependency_transformer() -> None:
    runs, metric_spec = fresh_development_runs()
    log_spec = OtlpLogFeatureSpec.from_dict(
        json.loads(
            open(
                "lab/fault_matrix/"
                "contextual-v2-log-feature-spec.json"
            ).read()
        )
    )
    base = compile_multimodal_telemetry_corpus(
        runs,
        v2_normal_log_captures(runs),
        metric_spec,
        log_spec,
        TelemetryCorpusSplitSpec(
            training_case_ids=FRESH_CASE_IDS[:2],
            validation_case_ids=(FRESH_CASE_IDS[2],),
            reserved_case_ids=(),
            lookback=6,
        ),
    )

    corpus = compile_contextual_multimodal_telemetry_corpus(
        base,
        runs,
    )

    assert corpus.training.windows.log_contexts.shape == (
        48,
        6,
        12,
    )
    assert corpus.preprocessing["logs"]["transformer"] == {
        "schema_version": 2,
        "kind": "dependency_residual_application_logs_v2",
        "features": list(
            corpus.training.windows.log_feature_names
        ),
        "routine_success_events_included": False,
        "complementary_state_pairs_included": False,
    }


def _rich_normal_log_captures(runs):
    captures = normal_log_captures(runs)
    enriched = {}
    for run in runs:
        capture = captures[run.manifest.case_id]
        resource = capture.records[0].resource_attributes
        extra_records = tuple(
            LogRecord(
                time_unix_nano=point_index * 100 + event_index + 50,
                observed_time_unix_nano=None,
                severity_number=9,
                severity_text="INFO",
                body=event_name,
                resource_attributes=resource,
                record_attributes={
                    "event.name": event_name,
                    "quantis.experiment.window.index": point_index,
                },
                scope_name="quantis.application",
                scope_version="1.0.0",
                trace_id="",
                span_id="",
                flags=0,
                dropped_attributes_count=0,
            )
            for point_index in range(run.manifest.point_count)
            for event_index, event_name in enumerate(
                (
                    "queue.backlog.low",
                    "database.write.latency.fast",
                    "worker.state.busy",
                    "worker.state.idle",
                )
            )
        )
        enriched[run.manifest.case_id] = OtlpLogCapture(
            records=capture.records + extra_records,
            sha256=hashlib.sha256(
                f"rich:{run.manifest.case_id}".encode()
            ).hexdigest(),
            source_path=f"memory://rich/{run.manifest.case_id}",
            json_message_count=1,
        )
    return enriched
