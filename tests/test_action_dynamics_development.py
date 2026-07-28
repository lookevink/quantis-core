import numpy as np

from quantis_core.action_conditioned_dynamics import (
    TrajectoryDistribution,
)
from quantis_core.action_dynamics_development import (
    assess_action_dynamics_development,
)


def test_phase_zero_assessment_advances_only_with_forecast_and_attribution() -> None:
    observed = np.zeros((4, 3, 2, 1), dtype=np.float64)
    action_prediction = TrajectoryDistribution(
        mean=np.full_like(observed, 0.1),
        variance=np.ones_like(observed),
    )
    action_agnostic_prediction = TrajectoryDistribution(
        mean=np.full_like(observed, 0.3),
        variance=np.ones_like(observed),
    )
    persistence_prediction = TrajectoryDistribution(
        mean=np.full_like(observed, 0.5),
        variance=np.ones_like(observed),
    )

    assessment = assess_action_dynamics_development(
        action_prediction=action_prediction,
        action_agnostic_prediction=action_agnostic_prediction,
        persistence_prediction=persistence_prediction,
        observed_future=observed,
        predicted_candidate_ids=(
            "fault-a",
            "none",
            "fault-b",
            "none",
        ),
        true_candidate_ids=(
            "fault-a",
            "none",
            "fault-b",
            "none",
        ),
        no_action_candidate_id="none",
        propagation_delay_passed=True,
        minimum_forecast_relative_improvement=0.1,
        minimum_persistence_relative_improvement=0.1,
        minimum_attribution_hit_at_1=0.7,
        minimum_no_action_specificity=0.9,
    )

    assert assessment["status"] == "supported"
    assert assessment["decision"] == "advance_to_instrumented_pilot"
    assert assessment["scores"]["forecast_relative_improvement"] > 0.8
    assert assessment["scores"][
        "persistence_relative_improvement"
    ] > 0.9
    assert assessment["scores"]["attribution_hit_at_1"] == 1.0
    assert assessment["scores"]["no_action_specificity"] == 1.0
    assert all(
        gate["passed"] for gate in assessment["gates"].values()
    )


def test_phase_zero_assessment_stops_when_action_labels_do_not_help() -> None:
    observed = np.zeros((2, 2, 1, 1), dtype=np.float64)
    prediction = TrajectoryDistribution(
        mean=np.ones_like(observed),
        variance=np.ones_like(observed),
    )

    assessment = assess_action_dynamics_development(
        action_prediction=prediction,
        action_agnostic_prediction=prediction,
        persistence_prediction=prediction,
        observed_future=observed,
        predicted_candidate_ids=("fault", "fault"),
        true_candidate_ids=("fault", "none"),
        no_action_candidate_id="none",
        propagation_delay_passed=False,
        minimum_forecast_relative_improvement=0.1,
        minimum_persistence_relative_improvement=0.1,
        minimum_attribution_hit_at_1=0.7,
        minimum_no_action_specificity=0.9,
    )

    assert assessment["status"] == "not_supported"
    assert assessment["decision"] == "stop_or_redesign_phase_zero"
    assert not assessment["gates"][
        "action_conditioning_improves_forecast"
    ]["passed"]
    assert not assessment["gates"]["beats_persistence"]["passed"]
    assert not assessment["gates"][
        "respects_graph_propagation_delay"
    ]["passed"]
    assert not assessment["gates"]["recognizes_no_action"]["passed"]
