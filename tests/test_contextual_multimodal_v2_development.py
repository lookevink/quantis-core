import json

import pytest

from quantis_core.contextual_multimodal_corpus import (
    compile_contextual_multimodal_telemetry_corpus,
)
from quantis_core.contextual_multimodal_development import (
    ContextualMultimodalJepaV2Candidate,
    default_contextual_multimodal_jepa_v2_candidates,
    develop_contextual_multimodal_jepa_v2,
    select_contextual_multimodal_jepa_v2_candidate,
    write_contextual_multimodal_jepa_v2_artifacts,
)
from quantis_core.contextual_multimodal_training import (
    ContextualMultimodalJepaTrainingConfig,
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
from tests.multimodal_test_support import (
    v2_normal_log_captures,
)


def test_v2_candidate_sequence_is_fixed_before_development_scoring() -> None:
    candidates = default_contextual_multimodal_jepa_v2_candidates(
        ContextualMultimodalJepaTrainingConfig(
            pretraining_epochs=20,
            predictor_refinement_epochs=10,
            cross_validation_epochs=8,
            seed=89,
        )
    )

    assert tuple(candidate.name for candidate in candidates) == (
        "v2_log_latent_1",
        "v2_log_latent_2",
        "v2_log_latent_3",
        "v2_balanced_masked_log_latent_2",
        "v2_balanced_masked_log_latent_3",
    )
    assert candidates[-1].config.to_dict()[
        "modality_mask_probability"
    ] == 0.15
    assert candidates[-1].config.to_dict()[
        "log_self_loss_multiplier"
    ] == 0.25
    assert candidates[-1].config.to_dict()[
        "cross_modal_loss_multiplier"
    ] == 1.5


def test_v2_selection_uses_only_family_held_out_controls() -> None:
    selection = select_contextual_multimodal_jepa_v2_candidate(
        (
            _assessment(
                "larger_robust_margin",
                contextual=0.04,
                metrics_only=0.06,
                capacity=0.05,
                shuffled=0.07,
                log_rank=1.5,
            ),
            _assessment(
                "lower_rate_but_weak_alignment",
                contextual=0.03,
                metrics_only=0.05,
                capacity=0.04,
                shuffled=0.031,
                log_rank=1.5,
            ),
            _assessment(
                "collapsed",
                contextual=0.02,
                metrics_only=0.06,
                capacity=0.05,
                shuffled=0.07,
                log_rank=0.2,
            ),
        )
    )

    assert selection["status"] == "selected"
    assert selection["selected_candidate"] == "larger_robust_margin"
    assert selection["selection_basis"] == (
        "training_schedule_family_held_out_controls_only"
    )
    assert selection["uses_exposed_validation"] is False
    assert selection["publication_eligible"] is False
    assert selection["leaderboard"][0]["candidate"] == (
        "larger_robust_margin"
    )
    assert selection["leaderboard"][-1]["eligible"] is False


def test_v2_development_writes_candidate_evidence(tmp_path) -> None:
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
    candidates = default_contextual_multimodal_jepa_v2_candidates(
        ContextualMultimodalJepaTrainingConfig(
            metric_latent_dimension=2,
            pretraining_epochs=6,
            predictor_refinement_epochs=2,
            cross_validation_epochs=0,
            seed=97,
        ),
    )
    result = develop_contextual_multimodal_jepa_v2(
        corpus,
        candidates=candidates,
    )

    paths = write_contextual_multimodal_jepa_v2_artifacts(
        result,
        tmp_path,
    )

    assert result.selection["status"] == "failed"
    assert set(paths) == {"development", "selection", "report"}
    assert (
        tmp_path
        / "candidates"
        / "v2_log_latent_1"
        / "model.json"
    ).exists()
    assert "Previously exposed validation" in (
        tmp_path
        / "candidates"
        / "v2_log_latent_1"
        / "report.md"
    ).read_text()
    assert result.to_dict()["protocol"][
        "selection_uses_exposed_validation"
    ] is False
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_contextual_multimodal_jepa_v2_artifacts(
            result,
            tmp_path,
        )


def test_v2_development_rejects_noncanonical_candidate_sequence() -> None:
    candidate = ContextualMultimodalJepaV2Candidate(
        name="quick_v2",
        config=ContextualMultimodalJepaTrainingConfig(
            pretraining_epochs=1,
            predictor_refinement_epochs=1,
            cross_validation_epochs=0,
        ),
    )

    with pytest.raises(ValueError, match="canonical"):
        develop_contextual_multimodal_jepa_v2(
            object(),
            candidates=(candidate,),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("contextual_mean_alert_rate", -0.1),
        ("no_worse_fold_fraction", 1.1),
    ),
)
def test_v2_selection_rejects_invalid_fold_metrics(
    field,
    value,
) -> None:
    assessment = _assessment(
        "invalid",
        contextual=0.04,
        metrics_only=0.06,
        capacity=0.05,
        shuffled=0.07,
        log_rank=1.5,
    )
    assessment["cross_validation"]["summary"][field] = value

    with pytest.raises(ValueError, match="invalid"):
        select_contextual_multimodal_jepa_v2_candidate(
            (assessment,)
        )


def test_v2_selection_rejects_invalid_rank_or_dimension() -> None:
    invalid_rank = _assessment(
        "invalid_rank",
        contextual=0.04,
        metrics_only=0.06,
        capacity=0.05,
        shuffled=0.07,
        log_rank=float("inf"),
    )
    invalid_dimension = _assessment(
        "invalid_dimension",
        contextual=0.04,
        metrics_only=0.06,
        capacity=0.05,
        shuffled=0.07,
        log_rank=1.0,
    )
    invalid_dimension["config"]["log_latent_dimension"] = 0

    with pytest.raises(ValueError, match="invalid"):
        select_contextual_multimodal_jepa_v2_candidate(
            (invalid_rank,)
        )
    with pytest.raises(ValueError, match="invalid"):
        select_contextual_multimodal_jepa_v2_candidate(
            (invalid_dimension,)
        )


def _assessment(
    name,
    *,
    contextual,
    metrics_only,
    capacity,
    shuffled,
    log_rank,
):
    return {
        "name": name,
        "config": {
            "metric_latent_dimension": 3,
            "log_latent_dimension": 3,
        },
        "cross_validation": {
            "status": "completed",
            "uses_exposed_validation": False,
            "summary": {
                "contextual_mean_alert_rate": contextual,
                "metrics_only_mean_alert_rate": metrics_only,
                "capacity_matched_mean_alert_rate": capacity,
                "shuffled_logs_mean_alert_rate": shuffled,
                "no_worse_fold_fraction": 0.75,
            },
        },
        "diagnostics": {
            "metric_effective_rank": 2.0,
            "log_effective_rank": log_rank,
        },
    }
