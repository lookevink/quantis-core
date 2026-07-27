import json
from dataclasses import replace

import numpy as np
import pytest

from quantis_core.contextual_multimodal_corpus import (
    compile_contextual_multimodal_telemetry_corpus,
    subset_contextual_windows,
)
from quantis_core.contextual_multimodal_training import (
    ContextualMultimodalJepaTrainingConfig,
    refit_contextual_fold_preprocessing,
    train_contextual_multimodal_jepa_world_model,
    write_contextual_multimodal_jepa_artifacts,
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


def test_contextual_training_reports_controls_and_diagnostics(
    tmp_path,
) -> None:
    runs, metric_spec = fresh_development_runs()
    log_spec = OtlpLogFeatureSpec.from_dict(
        json.loads(
            open(
                "lab/fault_matrix/log-feature-spec.json"
            ).read()
        )
    )
    base = compile_multimodal_telemetry_corpus(
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
    corpus = compile_contextual_multimodal_telemetry_corpus(
        base,
        runs,
    )

    result = train_contextual_multimodal_jepa_world_model(
        corpus,
        ContextualMultimodalJepaTrainingConfig(
            metric_latent_dimension=2,
            log_latent_dimension=1,
            pretraining_epochs=30,
            predictor_refinement_epochs=10,
            cross_validation_epochs=0,
            seed=59,
        ),
    )

    assert result.model_artifact["kind"] == (
        "contextual_multimodal_jepa_world_model_v1"
    )
    assert result.evidence_mode == "development"
    assert result.metrics_only_model_artifact["kind"] == (
        "contextual_multimodal_jepa_world_model_v1"
    )
    assert result.metrics_only_model_artifact[
        "log_latent_dimension"
    ] == 0
    assert result.capacity_matched_model_artifact[
        "metric_latent_dimension"
    ] == 3
    assert result.capacity_matched_model_artifact[
        "log_latent_dimension"
    ] == 0
    assert result.shuffled_log_model_artifact[
        "control_protocol"
    ]["breaks_metric_log_alignment"] is True
    assert result.log_only_model_artifact["kind"] == (
        "contextual_multimodal_jepa_world_model_v1"
    )
    assert result.log_only_model_artifact[
        "metric_latent_dimension"
    ] == 0
    assert set(result.metrics) == {
        "contextual_multimodal",
        "metrics_only",
        "capacity_matched_metrics_only",
        "shuffled_logs",
        "log_only",
        "modality_dropout",
    }
    assert set(result.metrics["modality_dropout"]) == {
        "metric_context_only",
        "log_context_only",
    }
    assert len(
        result.schedule_transfer["validation_families"]
    ) == 1
    assert set(
        result.schedule_transfer["validation_families"][0]
    ) == {
        "schedule_sha256",
        "case_ids",
        "contextual_multimodal",
        "metrics_only",
        "capacity_matched_metrics_only",
        "shuffled_logs",
    }
    assert result.protocol["validation_use"] == (
        "diagnostic_only"
    )
    assert result.protocol["cross_validation"]["status"] == (
        "disabled"
    )
    assert result.selection["status"] == "not_assessed"

    case_ids = np.asarray(corpus.training.window_case_ids)
    fold_training = subset_contextual_windows(
        corpus.training.windows,
        case_ids == FRESH_CASE_IDS[0],
    )
    fold_held_out = subset_contextual_windows(
        corpus.training.windows,
        case_ids == FRESH_CASE_IDS[1],
    )
    _, _, first_preprocessing = (
        refit_contextual_fold_preprocessing(
            fold_training,
            fold_held_out,
            corpus.preprocessing,
        )
    )
    _, shifted_held_out, second_preprocessing = (
        refit_contextual_fold_preprocessing(
            fold_training,
            replace(
                fold_held_out,
                target_controls=(
                    fold_held_out.target_controls + 100.0
                ),
            ),
            corpus.preprocessing,
        )
    )
    assert first_preprocessing == second_preprocessing
    assert first_preprocessing["held_out_values_used"] is False
    assert np.max(shifted_held_out.target_controls) > 1.0

    paths = write_contextual_multimodal_jepa_artifacts(
        result,
        tmp_path,
    )
    assert set(paths) == {
        "corpus",
        "model",
        "metrics_only_model",
        "capacity_matched_model",
        "shuffled_log_model",
        "log_only_model",
        "development",
        "report",
    }
    report = paths["report"].read_text()
    assert "Contextual conditioned JEPA" in report
    assert "Previously exposed validation" in report
    assert "Primary JEPA references" in report

    with pytest.raises(
        ValueError,
        match="inner contextual training cases",
    ):
        train_contextual_multimodal_jepa_world_model(
            corpus,
            ContextualMultimodalJepaTrainingConfig(
                cross_validation_epochs=0,
                loss="l1",
                seed=73,
            ),
            evidence_mode="promotion_confirmation",
            promotion_protocol=json.loads(
                open(
                    "lab/fault_matrix/"
                    "contextual-jepa-promotion-v1.json"
                ).read()
            ),
        )
