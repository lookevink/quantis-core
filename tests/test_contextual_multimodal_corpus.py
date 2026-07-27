import json

import numpy as np

from quantis_core.contextual_multimodal_corpus import (
    DemandResidualLogTransformer,
    compile_contextual_multimodal_telemetry_corpus,
)
from quantis_core.multimodal_corpus import (
    compile_multimodal_telemetry_corpus,
)
from quantis_core.otlp_log_windowing import OtlpLogFeatureSpec
from quantis_core.telemetry_corpus import TelemetryCorpusSplitSpec
from tests.corpus_test_support import (
    FRESH_CASE_IDS,
    fresh_development_runs,
)
from tests.multimodal_test_support import normal_log_captures


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
