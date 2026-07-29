import copy
import importlib.util
from pathlib import Path


def _tracer_module():
    path = (
        Path(__file__).parents[1]
        / "lab"
        / "action_dynamics"
        / "prototype_sigreg_lejepa.py"
    )
    specification = importlib.util.spec_from_file_location(
        "prototype_sigreg_lejepa", path
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _assessment_inputs():
    score = {
        "normalized_mse_action_overlap": 1.0,
        "normalized_mse_overall": 1.0,
        "downstream_effect_mse": 1.0,
        "action_and_target_hit_at_1": 1.0,
        "no_action_specificity": 1.0,
    }
    transfer_scores = {
        name: copy.deepcopy(score)
        for name in (
            "raw_low_rank",
            "no_regularizer_jepa",
            "variance_covariance_jepa",
            "sigreg_jepa",
        )
    }
    state_probes = {
        "no_regularizer_jepa": {"aggregate_nrmse": 1.0},
        "variance_covariance_jepa": {"aggregate_nrmse": 1.0},
        "sigreg_jepa": {"aggregate_nrmse": 1.0},
        "matched_pca": {"aggregate_nrmse": 0.9},
    }
    action_sanity = {
        name: {"correct_action_beats_both_fraction": 1.0}
        for name in (
            "no_regularizer_jepa",
            "variance_covariance_jepa",
            "sigreg_jepa",
        )
    }
    detection = {
        "evaluation_control_trajectory_false_alarm_rate": 0.0,
        "evaluation_treatment_trajectory_detection_rate": 0.8,
        "median_post_onset_detection_delay_transitions": 5.0,
    }
    transfer_detection = {
        "no_regularizer_jepa": {
            **detection,
            "evaluation_treatment_trajectory_detection_rate": 0.9,
            "median_post_onset_detection_delay_transitions": 10.0,
        },
        "variance_covariance_jepa": detection,
        "sigreg_jepa": {
            **detection,
            "evaluation_treatment_trajectory_detection_rate": 0.9,
            "median_post_onset_detection_delay_transitions": 8.0,
        },
    }
    return {
        "transfer_scores": transfer_scores,
        "state_probes": state_probes,
        "action_sanity": action_sanity,
        "transfer_detection": transfer_detection,
        "selected_gains": {
            "no_regularizer_jepa": 1.0,
            "variance_covariance_jepa": 1.0,
            "sigreg_jepa": 1.0,
        },
        "parameter_counts": {
            "raw_low_rank": 10,
            "no_regularizer_jepa": 20,
            "variance_covariance_jepa": 20,
            "sigreg_jepa": 20,
        },
        "reported_measurements": {"nested": [0.0, 1.0]},
        "restoration_parity": {
            "raw_low_rank": True,
            "no_regularizer_jepa": True,
            "variance_covariance_jepa": True,
            "sigreg_jepa": True,
        },
    }


def test_sigreg_assessor_allows_mixed_alert_improvements() -> None:
    module = _tracer_module()

    assessment = module.assess_sigreg_jepa_tracer(
        **_assessment_inputs()
    )

    assert assessment["safety_passed"] is True
    assert assessment["value_lanes"]["alert"]["passed"] is True
    assert assessment["decision"] == "run_fixed_seed_robustness"


def test_sigreg_assessor_rejects_nested_nonfinite_evidence() -> None:
    module = _tracer_module()
    inputs = _assessment_inputs()
    inputs["reported_measurements"] = {
        "training": {"sigreg": float("nan")}
    }

    assessment = module.assess_sigreg_jepa_tracer(**inputs)

    assert assessment["safety_gates"]["metrics_are_finite"] is False
    assert assessment["decision"] == "reject_sigreg_residual_recipe"
