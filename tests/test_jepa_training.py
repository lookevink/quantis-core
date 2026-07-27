import json

from quantis_core.telemetry_corpus import (
    TelemetryCorpusSplitSpec,
    compile_telemetry_corpus,
)
from quantis_core.world_model import (
    JepaTrainingConfig,
    train_jepa_world_model,
    write_jepa_development_artifacts,
)
from tests.corpus_test_support import (
    FRESH_CASE_IDS,
    fresh_development_runs,
)


def test_jepa_training_reports_separate_train_and_validation_evidence(
    tmp_path,
):
    corpus = _development_corpus()
    result = train_jepa_world_model(
        corpus,
        JepaTrainingConfig(
            latent_dimension=3,
            epochs=80,
            learning_rate=0.02,
            ema_decay=0.96,
            seed=29,
        ),
    )

    assert result.model_artifact["kind"] == "jepa_world_model_v0"
    assert result.metrics["training"]["window_count"] == 60
    assert result.metrics["validation"]["window_count"] == 30
    assert result.metrics["training"]["latent_loss_mean"] >= 0.0
    assert result.metrics["validation"]["latent_loss_mean"] >= 0.0
    assert result.protocol["model_selection_status"] == (
        "development_only"
    )
    assert result.protocol["validation_case_ids"] == [
        FRESH_CASE_IDS[2]
    ]

    paths = write_jepa_development_artifacts(result, tmp_path)

    assert set(paths) == {
        "corpus",
        "model",
        "development",
        "report",
    }
    checked = json.loads(paths["development"].read_text())
    assert checked == result.to_dict()
    assert "not confirmation evidence" in paths["report"].read_text()


def _development_corpus():
    runs, feature_spec = fresh_development_runs()
    return compile_telemetry_corpus(
        runs,
        feature_spec,
        TelemetryCorpusSplitSpec(
            training_case_ids=FRESH_CASE_IDS[:2],
            validation_case_ids=(FRESH_CASE_IDS[2],),
            reserved_case_ids=(),
            lookback=6,
        ),
    )
