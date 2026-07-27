import json
from dataclasses import replace
from pathlib import Path

import pytest

from quantis_core.fault_matrix import (
    FaultMatrixCaseManifest,
    FaultMatrixRun,
)
from quantis_core.telemetry_corpus import (
    RESERVED_EVIDENCE_CASE_IDS,
    TelemetryCorpusSplitSpec,
    compile_telemetry_corpus,
)
from tests.corpus_test_support import (
    FRESH_CASE_IDS,
    fresh_development_runs,
)


def test_corpus_compiles_normal_windows_without_crossing_runs():
    runs, feature_spec = fresh_development_runs()
    corpus = compile_telemetry_corpus(
        runs,
        feature_spec,
        TelemetryCorpusSplitSpec(
            training_case_ids=FRESH_CASE_IDS[:2],
            validation_case_ids=(FRESH_CASE_IDS[2],),
            reserved_case_ids=(),
            lookback=6,
        ),
    )

    assert corpus.training.case_ids == FRESH_CASE_IDS[:2]
    assert corpus.validation.case_ids == (FRESH_CASE_IDS[2],)
    assert len(corpus.training.windows.targets) == 60
    assert len(corpus.validation.windows.targets) == 30
    assert corpus.training.window_case_ids[:30] == (
        FRESH_CASE_IDS[0],
    ) * 30
    assert corpus.training.window_case_ids[30:] == (
        FRESH_CASE_IDS[1],
    ) * 30
    assert corpus.protocol["training_point_count"] == 72
    assert corpus.protocol["validation_point_count"] == 36
    assert corpus.protocol["context_crosses_run_boundary"] is False
    assert corpus.protocol["training_validation_schedule_overlap"] == []
    assert corpus.protocol["application_image_id"].startswith("sha256:")
    assert len(
        corpus.protocol["application_build_context_sha256"]
    ) == 64
    first_run = corpus.protocol["runs"][FRESH_CASE_IDS[0]]
    assert first_run["topology_id"] == (
        runs[0].manifest.topology_id
    )
    assert first_run["worker_replicas"] == (
        runs[0].manifest.worker_replicas
    )
    assert set(
        corpus.protocol["split_spec"]["reserved_case_ids"]
    ) == RESERVED_EVIDENCE_CASE_IDS
    assert corpus.training.windows.feature_names == (
        "request_latency_ms",
        "error_rate",
        "queue_depth",
        "worker_completion_ratio",
        "worker_heartbeat_age_s",
        "db_write_completion_ratio",
    )


def test_corpus_rejects_unverified_or_mixed_application_builds():
    runs, feature_spec = fresh_development_runs()
    split = TelemetryCorpusSplitSpec(
        training_case_ids=FRESH_CASE_IDS[:2],
        validation_case_ids=(FRESH_CASE_IDS[2],),
        reserved_case_ids=(),
        lookback=6,
    )

    unverified = _with_application_image(
        runs[0],
        "unverified",
    )
    with pytest.raises(ValueError, match="application image provenance"):
        compile_telemetry_corpus(
            [unverified, runs[1], runs[2]],
            feature_spec,
            split,
        )

    mixed = _with_application_image(
        runs[1],
        "sha256:" + "a" * 64,
    )
    with pytest.raises(ValueError, match="same application build"):
        compile_telemetry_corpus(
            [runs[0], mixed, runs[2]],
            feature_spec,
            split,
        )


def test_corpus_rejects_reserved_evidence_and_schedule_leakage():
    runs, feature_spec = fresh_development_runs()

    with pytest.raises(ValueError, match="reserved evidence"):
        TelemetryCorpusSplitSpec(
            training_case_ids=("cache-outage-held-out-01",),
            validation_case_ids=(FRESH_CASE_IDS[2],),
            reserved_case_ids=(),
        )

    repeated_schedule_run = FaultMatrixRun(
        manifest=FaultMatrixCaseManifest.from_dict(
            {
                **runs[0].manifest.to_dict(),
                "case_id": "same-schedule-validation",
            }
        ),
        capture=runs[0].capture,
    )
    with pytest.raises(ValueError, match="canonical request schedules"):
        compile_telemetry_corpus(
            [runs[0], repeated_schedule_run],
            feature_spec,
            TelemetryCorpusSplitSpec(
                training_case_ids=(runs[0].manifest.case_id,),
                validation_case_ids=(
                    repeated_schedule_run.manifest.case_id,
                ),
                reserved_case_ids=(),
            ),
        )


def test_split_rejects_failed_multimodal_corpus_case_ids():
    with pytest.raises(ValueError, match="failed corpus"):
        TelemetryCorpusSplitSpec(
            training_case_ids=(
                "multimodal-normal-f01-w1-47",
            ),
            validation_case_ids=(FRESH_CASE_IDS[2],),
            reserved_case_ids=(),
        )


def test_corpus_enforces_preregistered_api_queue_size():
    runs, feature_spec = fresh_development_runs()
    split = TelemetryCorpusSplitSpec(
        training_case_ids=FRESH_CASE_IDS[:2],
        validation_case_ids=(FRESH_CASE_IDS[2],),
        reserved_case_ids=(),
        expected_application_api_request_queue_size=128,
    )

    with pytest.raises(ValueError, match="expected API request queue"):
        compile_telemetry_corpus(runs, feature_spec, split)

    admitted = compile_telemetry_corpus(
        [
            _with_application_queue_size(run, 128)
            for run in runs
        ],
        feature_spec,
        split,
    )
    assert admitted.protocol[
        "application_api_request_queue_size"
    ] == 128

    with pytest.raises(ValueError, match="expected API request queue"):
        compile_telemetry_corpus(
            [
                _with_application_queue_size(run, 5)
                for run in runs
            ],
            feature_spec,
            split,
        )


def test_reserved_evidence_registry_covers_all_committed_manifests():
    repository = Path(__file__).resolve().parents[1]
    lab = repository / "lab" / "fault_matrix"
    manifest_case_ids = {
        str(json.loads(path.read_text())["case_id"])
        for directory in lab.glob("experiments*")
        for path in directory.glob("*.json")
    }

    assert manifest_case_ids == RESERVED_EVIDENCE_CASE_IDS


def _with_application_image(
    run: FaultMatrixRun,
    image_id: str,
) -> FaultMatrixRun:
    return replace(
        run,
        capture=replace(
            run.capture,
            points=tuple(
                replace(
                    point,
                    resource_attributes={
                        **point.resource_attributes,
                        "quantis.application.image.id": image_id,
                    },
                )
                for point in run.capture.points
            ),
        ),
    )


def _with_application_queue_size(
    run: FaultMatrixRun,
    queue_size: int,
) -> FaultMatrixRun:
    return replace(
        run,
        capture=replace(
            run.capture,
            points=tuple(
                replace(
                    point,
                    resource_attributes={
                        **point.resource_attributes,
                        "quantis.application.api.request_queue_size": (
                            queue_size
                        ),
                    },
                )
                for point in run.capture.points
            ),
        ),
    )
