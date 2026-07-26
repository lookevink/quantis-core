import json

from quantis_core.evaluation import (
    EvaluationConfig,
    run_evaluation,
    write_evaluation_artifacts,
)


def small_config():
    return EvaluationConfig(
        train_seeds=(11, 23, 37),
        test_seeds=(101, 103, 107, 109),
        scenario_length=360,
        lookback=12,
        calibration_quantile=0.99,
    )


def test_evaluation_is_reproducible_apart_from_runtime_measurements():
    first = run_evaluation(small_config())
    second = run_evaluation(small_config())

    first_payload = first.to_dict(include_runtime=False)
    second_payload = second.to_dict(include_runtime=False)

    assert first_payload == second_payload
    assert set(first.detectors) == {
        "persistence",
        "robust_feature",
        "linear_latent_predictive",
        "coherent_latent_predictive",
    }
    assert first.protocol["training_structural_points"] == 0
    assert first.protocol["test_scenario_count"] == 4
    assert first.protocol["attribution_random_hit_at_3"] < 0.8
    assert [item["seed"] for item in first.scenario_manifests["test"]] == [
        101,
        103,
        107,
        109,
    ]


def test_reference_experiment_meets_the_declared_synthetic_gates():
    report = run_evaluation(small_config())
    latent = report.detectors["coherent_latent_predictive"]

    assert report.acceptance["all_passed"] is True
    assert latent.structural_event_recall >= 0.8
    assert latent.routine_noise_alert_rate <= 0.1
    assert latent.attribution_hit_at_3 >= 0.8
    assert (
        latent.attribution_hit_at_3
        > report.protocol["attribution_random_hit_at_3"]
    )
    assert latent.attribution_recall_at_3 >= 0.8
    assert (
        latent.routine_noise_alert_rate
        < report.detectors["persistence"].routine_noise_alert_rate
    )
    assert latent.mean_streaming_scoring_ms_per_point < 1.0


def test_artifacts_capture_results_protocol_models_and_limitations(tmp_path):
    report = run_evaluation(small_config())

    paths = write_evaluation_artifacts(report, tmp_path)

    assert set(paths) == {
        "evaluation",
        "report",
        "scenario_manifest",
        "window_compiler",
        "detector_persistence",
        "detector_robust_feature",
        "detector_linear_latent_predictive",
        "detector_coherent_latent_predictive",
    }
    payload = json.loads(paths["evaluation"].read_text())
    markdown = paths["report"].read_text()
    assert "Infinity" not in paths["evaluation"].read_text()
    assert payload["acceptance"]["all_passed"] is True
    assert payload["limitations"]
    assert "does not establish" in markdown
    assert "linear latent predictive" in markdown.lower()
