import json
from dataclasses import replace

import numpy as np
import pytest

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


def test_multimodal_corpus_prefers_metric_event_time_boundaries() -> None:
    source_runs, metric_spec = fresh_development_runs()
    runs = [_with_event_time_boundaries(run) for run in source_runs]
    log_captures = {
        case_id: replace(
            capture,
            records=tuple(
                replace(
                    record,
                    record_attributes={
                        **record.record_attributes,
                        "quantis.experiment.window.index": 0,
                    },
                )
                for record in capture.records
            ),
        )
        for case_id, capture in normal_log_captures(runs).items()
    }
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

    assert corpus.protocol["log_window_assignment"] == (
        "event_time_metric_boundaries"
    )
    np.testing.assert_allclose(
        corpus.training.windows.logs.targets[:, 0],
        0.0,
    )


def test_multimodal_corpus_requires_normal_drain_boundary() -> None:
    source_runs, metric_spec = fresh_development_runs()
    runs = [
        _with_event_time_boundaries(run, include_drain=False)
        for run in source_runs
    ]
    log_spec = OtlpLogFeatureSpec.from_dict(
        json.loads(
            open(
                "lab/fault_matrix/log-feature-spec.json"
            ).read()
        )
    )

    with pytest.raises(ValueError, match="drain boundary"):
        compile_multimodal_telemetry_corpus(
            runs,
            normal_log_captures(runs),
            metric_spec,
            log_spec,
            TelemetryCorpusSplitSpec(
                training_case_ids=FRESH_CASE_IDS[:2],
                validation_case_ids=(FRESH_CASE_IDS[2],),
                reserved_case_ids=(),
                lookback=6,
            ),
        )


def _with_event_time_boundaries(run, include_drain=True):
    point_times = sorted(
        {point.time_unix_nano for point in run.capture.points}
    )
    source_by_time = {
        point_time: next(
            point
            for point in run.capture.points
            if point.time_unix_nano == point_time
        )
        for point_time in point_times
    }
    boundary_points = tuple(
        replace(
            source_by_time[point_time],
            metric_name=(
                "quantis.experiment.window.closed_unix_nano"
            ),
            unit="ns",
            number_value=point_index * 10 + 5,
        )
        for point_index, point_time in enumerate(point_times)
    )
    drain_points = (
        (
            replace(
                boundary_points[-1],
                metric_name=(
                    "quantis.experiment.drain.closed_unix_nano"
                ),
                time_unix_nano=point_times[-1] + 1_000_000_000,
                number_value=(len(point_times) * 10 + 5),
            ),
        )
        if include_drain
        else ()
    )
    return replace(
        run,
        capture=replace(
            run.capture,
            points=(
                tuple(
                    replace(
                        point,
                        resource_attributes={
                            **point.resource_attributes,
                            "quantis.experiment.run.started_unix_nano": 0,
                        },
                    )
                    for point in run.capture.points
                )
                + tuple(
                    replace(
                        point,
                        resource_attributes={
                            **point.resource_attributes,
                            "quantis.experiment.run.started_unix_nano": 0,
                        },
                    )
                    for point in boundary_points
                )
                + tuple(
                    replace(
                        point,
                        resource_attributes={
                            **point.resource_attributes,
                            "quantis.experiment.run.started_unix_nano": 0,
                        },
                    )
                    for point in drain_points
                )
            ),
        ),
    )
