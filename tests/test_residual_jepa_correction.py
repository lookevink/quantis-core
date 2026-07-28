import numpy as np
import pytest

from quantis_core.action_dynamics_synthetic import synthetic_action_runs
from quantis_core.action_conditioned_dynamics import ActionTrajectoryCompiler
from quantis_core.edge_dynamics.action_conditioned_jepa import (
    ActionConditionedJepaConfig,
    ActionConditionedJepaDynamics,
)
from quantis_core.edge_dynamics.models import (
    ContractiveLowRankDynamics,
    LowRankConfig,
)
from quantis_core.edge_dynamics.residual_jepa import (
    FrozenBaselineResidualDynamics,
    assess_residual_jepa_development,
    latent_divergence_detection,
    write_residual_jepa_artifacts,
)


def _windows():
    training_runs = synthetic_action_runs(
        20, split="training", seed=310
    )
    validation_runs = synthetic_action_runs(
        10, split="validation", seed=910
    )
    compiler = ActionTrajectoryCompiler(
        context_length=5, rollout_horizon=3
    ).fit(training_runs)
    return (
        compiler.transform(training_runs),
        compiler.transform(validation_runs),
    )


def _residual_model(
    baseline: ContractiveLowRankDynamics,
    *,
    objective: str,
) -> FrozenBaselineResidualDynamics:
    correction = ActionConditionedJepaDynamics(
        ActionConditionedJepaConfig(
            node_latent_dimension=2,
            transition_rank=3,
            epochs=1,
            batch_size=256,
            mask_time_fraction=0.4,
            mask_entity_fraction=0.5,
            context_reconstruction_weight=0.0,
            objective=objective,
            zero_initialize_decoder=True,
            device="cpu",
            seed=13,
        )
    )
    return FrozenBaselineResidualDynamics(
        baseline=baseline,
        correction=correction,
    )


def test_zero_gain_residual_model_exactly_preserves_frozen_baseline() -> None:
    training, validation = _windows()
    baseline = ContractiveLowRankDynamics(
        LowRankConfig(rank=3)
    ).fit(training)
    baseline_before = baseline.to_dict()
    model = _residual_model(
        baseline, objective="supervised"
    ).fit(training)
    model.set_correction_gain(0.0)

    expected = baseline.rollout(
        validation.histories,
        validation.future_controls,
        validation.future_actions,
        validation.graph,
    )
    actual = model.rollout(
        validation.histories,
        validation.future_controls,
        validation.future_actions,
        validation.graph,
    )

    assert np.array_equal(actual.mean, expected.mean)
    assert baseline.to_dict() == baseline_before
    assert (
        model.to_dict()["correction"][
            "initial_maximum_absolute_prediction"
        ]
        == 0.0
    )


def test_residual_jepa_roundtrips_and_exposes_latent_divergence() -> None:
    training, validation = _windows()
    baseline = ContractiveLowRankDynamics(
        LowRankConfig(rank=3)
    ).fit(training)
    model = _residual_model(baseline, objective="jepa").fit(training)
    model.select_correction_gain(validation)

    first = model.rollout(
        validation.histories[:8],
        validation.future_controls[:8],
        validation.future_actions[:8],
        validation.graph,
    )
    divergence = model.latent_divergence(validation)
    restored = FrozenBaselineResidualDynamics.from_dict(model.to_dict())
    second = restored.rollout(
        validation.histories[:8],
        validation.future_controls[:8],
        validation.future_actions[:8],
        validation.graph,
    )

    assert divergence.shape == validation.future_states.shape[:2]
    assert np.all(np.isfinite(divergence))
    assert np.all(divergence >= 0.0)
    assert np.allclose(first.mean, second.mean, atol=1e-6)
    assert model.selected_gain in (0.0, 0.25, 0.5, 0.75, 1.0)
    assert len(model.selection_curve) == 5
    assert (
        model.correction.initial_maximum_absolute_prediction
        == 0.0
    )


def test_residual_jepa_assessment_keeps_confirmation_sealed() -> None:
    baseline = {
        "downstream_effect_mse": 1.0,
        "normalized_mse_action_overlap": 1.0,
        "normalized_mse_overall": 1.0,
    }
    supervised = {"downstream_effect_mse": 0.95}
    jepa = {
        "downstream_effect_mse": 0.8,
        "normalized_mse_action_overlap": 1.01,
        "normalized_mse_overall": 0.99,
        "action_and_target_hit_at_1": 1.0,
        "no_action_specificity": 1.0,
    }
    assessment = assess_residual_jepa_development(
        baseline_transfer=baseline,
        supervised_transfer=supervised,
        jepa_transfer=jepa,
        selected_gain=0.5,
        action_sanity={
            "correct_action_beats_both_fraction": 0.9,
        },
        latent_detection={
            "evaluation_control_trajectory_false_alarm_rate": 0.0,
            "evaluation_treatment_trajectory_detection_rate": 0.9,
            "median_post_onset_detection_delay_transitions": 4.0,
        },
        seed_robustness={
            "seed_count": 1,
            "required_seed_count": 3,
            "passed": False,
        },
    )

    assert assessment["predictive_tracer_gates_passed"] is True
    assert assessment["investigation_gates_passed"] is True
    assert assessment["decision"] == "run_seed_robustness"
    assert assessment["sealed_confirmation"] is False


def test_latent_divergence_calibrates_the_trajectory_alarm_unit() -> None:
    training, validation = _windows()
    baseline = ContractiveLowRankDynamics(
        LowRankConfig(rank=3)
    ).fit(training)
    model = _residual_model(baseline, objective="jepa").fit(training)

    result = latent_divergence_detection(
        model=model,
        calibration=validation,
        evaluation=validation,
        alpha=0.05,
    )

    assert result["calibration_unit"] == "control_trajectory_maximum"
    assert result["calibration_control_trajectory_count"] == 5
    calibration_control_rows = [
        row
        for row in result["calibration_trajectory_rows"]
        if not row["is_treatment"]
    ]
    assert not any(row["any_alarm"] for row in calibration_control_rows)


def test_latent_divergence_detects_signal_only_at_later_horizon() -> None:
    training, validation = _windows()
    calibration_scores = np.zeros(
        training.future_states.shape[:2], dtype=np.float64
    )
    evaluation_scores = np.zeros(
        validation.future_states.shape[:2], dtype=np.float64
    )
    applicable = validation.action_feature_names.index("applicable")
    treatment_at_first_horizon = np.any(
        validation.future_actions[:, 0, :, applicable] > 0.5,
        axis=1,
    )
    evaluation_scores[treatment_at_first_horizon, 2] = 1.0

    class LaterHorizonSignal:
        def latent_divergence(self, windows):
            if windows is training:
                return calibration_scores
            if windows is validation:
                return evaluation_scores
            raise AssertionError("unexpected windows")

    result = latent_divergence_detection(
        model=LaterHorizonSignal(),
        calibration=training,
        evaluation=validation,
        alpha=0.05,
    )

    assert (
        result["evaluation_control_trajectory_false_alarm_rate"]
        == 0.0
    )
    assert (
        result["evaluation_treatment_trajectory_detection_rate"]
        == 1.0
    )
    assert (
        result["median_post_onset_detection_delay_transitions"]
        == 2.0
    )


def test_residual_jepa_artifacts_are_immutable(tmp_path) -> None:
    report = {
        "assessment": {"decision": "reject_this_configuration"},
        "transfer_scores": {},
    }
    output = tmp_path / "residual-evidence"

    manifest = write_residual_jepa_artifacts(
        output_directory=output,
        report=report,
        model_artifacts={"jepa": {"kind": "test-model"}},
    )

    assert manifest["kind"] == "residual_jepa_development_manifest_v1"
    assert "not sealed confirmation" in (
        output / "report.md"
    ).read_text()
    with pytest.raises(FileExistsError):
        write_residual_jepa_artifacts(
            output_directory=output,
            report=report,
            model_artifacts={},
        )
