import json

from quantis_core.otlp_log_windowing import OtlpLogFeatureSpec
from quantis_core.telemetry_corpus import TelemetryCorpusSplitSpec
from quantis_core.multimodal_corpus import (
    compile_multimodal_telemetry_corpus,
)
from tests.corpus_test_support import (
    FRESH_CASE_IDS,
    fresh_development_runs,
)
from tests.multimodal_test_support import normal_log_captures


def test_multimodal_corpus_fits_separate_run_isolated_channels() -> None:
    runs, metric_spec = fresh_development_runs()
    log_captures = normal_log_captures(runs)
    log_spec = OtlpLogFeatureSpec.from_dict(
        json.loads(
            open(
                "lab/fault_matrix/log-feature-spec.json"
            ).read()
        )
    )

    corpus = compile_multimodal_telemetry_corpus(
        runs,
        log_captures,
        metric_spec,
        log_spec,
        TelemetryCorpusSplitSpec(
            training_case_ids=FRESH_CASE_IDS[:2],
            validation_case_ids=(FRESH_CASE_IDS[2],),
            reserved_case_ids=(),
            lookback=6,
        ),
    )

    assert corpus.training.windows.metric.contexts.shape == (
        60,
        6,
        6,
    )
    assert corpus.training.windows.logs.contexts.shape == (
        60,
        6,
        4,
    )
    assert corpus.validation.windows.metric.targets.shape == (30, 6)
    assert corpus.validation.windows.logs.targets.shape == (30, 4)
    assert corpus.training.window_case_ids == (
        (FRESH_CASE_IDS[0],) * 30
        + (FRESH_CASE_IDS[1],) * 30
    )
    assert corpus.training.windows.metric.point_indices.tolist() == (
        list(range(6, 36)) + list(range(6, 36))
    )
    assert corpus.training.windows.logs.point_indices.tolist() == (
        list(range(6, 36)) + list(range(6, 36))
    )
    assert corpus.protocol["context_crosses_run_boundary"] is False
    assert set(corpus.protocol["runs"]) == set(FRESH_CASE_IDS)
    assert corpus.protocol["runs"][FRESH_CASE_IDS[0]][
        "log_capture_sha256"
    ] == log_captures[FRESH_CASE_IDS[0]].sha256
