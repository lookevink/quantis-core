import json

import pytest

from quantis_core.multimodal_corpus import (
    compile_multimodal_telemetry_corpus,
)
from quantis_core.multimodal_training import (
    MultimodalJepaTrainingConfig,
    train_multimodal_jepa_world_model,
    write_multimodal_jepa_development_artifacts,
)
from quantis_core.otlp_log_windowing import OtlpLogFeatureSpec
from quantis_core.telemetry_corpus import TelemetryCorpusSplitSpec
from tests.corpus_test_support import (
    FRESH_CASE_IDS,
    fresh_development_runs,
)
from tests.multimodal_test_support import normal_log_captures


def test_multimodal_training_rejects_invalid_promotion_rate() -> None:
    with pytest.raises(
        ValueError,
        match="maximum_validation_alert_rate",
    ):
        MultimodalJepaTrainingConfig(
            maximum_validation_alert_rate=1.01
        )


def test_multimodal_training_reports_preregistered_ablations(
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
    corpus = compile_multimodal_telemetry_corpus(
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

    result = train_multimodal_jepa_world_model(
        corpus,
        MultimodalJepaTrainingConfig(
            metric_latent_dimension=3,
            log_latent_dimension=2,
            epochs=40,
            seed=43,
        ),
    )

    assert result.model_artifact["kind"] == (
        "multimodal_jepa_world_model_v0"
    )
    assert result.metrics_only_model_artifact["kind"] == (
        "jepa_world_model_v0"
    )
    assert result.capacity_matched_model_artifact["kind"] == (
        "jepa_world_model_v0"
    )
    assert result.capacity_matched_model_artifact[
        "latent_dimension"
    ] == 5
    assert result.shuffled_log_model_artifact["kind"] == (
        "multimodal_jepa_world_model_v0"
    )
    assert result.model_artifact["preprocessing"] == {
        "metric": {
            "conditioner": corpus.metric_corpus_metadata[
                "conditioner"
            ],
            "window_compiler": corpus.metric_corpus_metadata[
                "window_compiler"
            ],
        },
        "logs": {
            "feature_spec": corpus.log_feature_spec_artifact,
            "window_compiler": (
                corpus.log_window_compiler_artifact
            ),
        },
    }
    assert result.metrics["multimodal"]["training"][
        "window_count"
    ] == 60
    assert result.metrics["multimodal"]["validation"][
        "window_count"
    ] == 30
    assert result.metrics["metrics_only"]["validation"][
        "window_count"
    ] == 30
    assert result.metrics["capacity_matched_metrics_only"][
        "validation"
    ]["window_count"] == 30
    assert result.metrics["shuffled_logs"]["validation"][
        "window_count"
    ] == 30
    assert result.protocol["training_uses_validation_windows"] is False
    assert result.protocol["log_ablation"] == {
        "method": "deterministic_window_pair_permutation",
        "training_seed": 1044,
        "validation_seed": 2044,
        "preserves_log_context_target_pairs": True,
        "breaks_metric_log_alignment": True,
    }
    assert set(result.promotion["gates"]) == {
        "validation_alert_rate_at_most_maximum",
        "no_worse_than_capacity_matched_metrics_only_alert_rate",
        "no_worse_than_shuffled_logs_alert_rate",
        "no_worse_than_shuffled_logs_latent_loss",
    }
    assert result.promotion["status"] in {"passed", "failed"}

    paths = write_multimodal_jepa_development_artifacts(
        result,
        tmp_path,
    )

    assert set(paths) == {
        "corpus",
        "model",
        "metrics_only_model",
        "capacity_matched_model",
        "shuffled_log_model",
        "development",
        "report",
    }
    assert "Application-log JEPA" in paths["report"].read_text()
    assert "Metrics-only baseline" in paths["report"].read_text()
    assert "Capacity-matched metrics-only baseline" in (
        paths["report"].read_text()
    )
    assert "Shuffled-log ablation" in paths["report"].read_text()
    assert "Promotion gates" in paths["report"].read_text()
