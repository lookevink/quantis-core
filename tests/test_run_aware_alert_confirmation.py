import math

import numpy as np
import pytest

from lab.action_dynamics.assess_run_aware_alert_confirmation import (
    assess_run_aware_alert_arrays_independently,
)
from quantis_core.run_aware_alert_confirmation import (
    assign_pair_roles,
    conformal_run_threshold,
    resettable_cusum,
    run_aware_alert_assessment,
)


def _balanced_pair_metadata():
    pair_ids = []
    action_kind_by_pair = {}
    topology_by_pair = {}
    for action in (
        "api_rejection",
        "postgres_lock",
        "redis_dequeue_delay",
        "redis_enqueue_delay",
        "worker_pause",
    ):
        for topology in ("workers-1", "workers-2", "workers-3"):
            for replicate in range(8):
                pair_id = f"{action}:{topology}:{replicate}"
                pair_ids.append(pair_id)
                action_kind_by_pair[pair_id] = action
                topology_by_pair[pair_id] = topology
    return pair_ids, action_kind_by_pair, topology_by_pair


def test_pair_roles_are_cell_balanced_and_frozen() -> None:
    pair_ids, actions, topologies = _balanced_pair_metadata()

    roles = assign_pair_roles(
        pair_ids=pair_ids,
        action_kind_by_pair=actions,
        topology_by_pair=topologies,
    )

    assert tuple(roles.values()).count("score_reference") == 30
    assert tuple(roles.values()).count("threshold_calibration") == 30
    assert tuple(roles.values()).count("sealed_evaluation") == 60
    for action in set(actions.values()):
        action_roles = [
            roles[pair_id]
            for pair_id in pair_ids
            if actions[pair_id] == action
        ]
        assert action_roles.count("score_reference") == 6
        assert action_roles.count("threshold_calibration") == 6
        assert action_roles.count("sealed_evaluation") == 12


def test_conformal_run_threshold_uses_strict_five_percent_rank() -> None:
    maxima = np.arange(30, dtype=np.float64)

    threshold = conformal_run_threshold(maxima, alpha=0.05)

    assert threshold == 29.0


def test_resettable_cusum_can_discard_routine_run_length() -> None:
    increments = np.asarray(
        [2.0, -3.0, -3.0, 1.0, 1.0], dtype=np.float64
    )

    cumulative = resettable_cusum(increments)

    assert np.array_equal(cumulative, [2.0, 0.0, 0.0, 1.0, 2.0])


def test_assessment_confirms_useful_candidate_and_rejects_weak_control() -> None:
    pair_ids, actions, topologies = _balanced_pair_metadata()
    roles = assign_pair_roles(
        pair_ids=pair_ids,
        action_kind_by_pair=actions,
        topology_by_pair=topologies,
    )
    trajectory_ids = []
    window_pair_ids = []
    transition_indices = []
    future_actions = []
    observed = []
    candidate = []
    persistence = []
    for pair_id in pair_ids:
        for treatment in (False, True):
            trajectory = f"{pair_id}:{'t' if treatment else 'c'}"
            for transition in range(12):
                active = treatment and 4 <= transition <= 8
                trajectory_ids.append(trajectory)
                window_pair_ids.append(pair_id)
                transition_indices.append(transition)
                action = np.zeros((1, 1, 2), dtype=np.float64)
                action[..., 0] = not active
                action[..., 1] = active
                future_actions.append(action)
                observed.append([[[0.0]]])
                candidate_error = (
                    10.0
                    if active
                    and roles[pair_id] == "sealed_evaluation"
                    else 0.1
                )
                persistence_error = (
                    10.0
                    if active
                    and roles[pair_id] == "sealed_evaluation"
                    and actions[pair_id] in {
                        "api_rejection",
                        "postgres_lock",
                        "worker_pause",
                    }
                    else 0.1
                )
                candidate.append([[[math.sqrt(candidate_error)]]])
                persistence.append([[[math.sqrt(persistence_error)]]])

    assessment = run_aware_alert_assessment(
        observed=np.asarray(observed, dtype=np.float64),
        candidate_prediction=np.asarray(candidate, dtype=np.float64),
        persistence_prediction=np.asarray(
            persistence, dtype=np.float64
        ),
        future_actions=np.asarray(future_actions, dtype=np.float64),
        trajectory_ids=tuple(trajectory_ids),
        matched_pair_ids=tuple(window_pair_ids),
        transition_indices=np.asarray(
            transition_indices, dtype=np.int64
        ),
        action_kind_by_pair=actions,
        topology_by_pair=topologies,
        alpha=0.05,
        cusum_drift=math.log(4.0),
    )
    independent = assess_run_aware_alert_arrays_independently(
        observed=np.asarray(observed, dtype=np.float64),
        candidate=np.asarray(candidate, dtype=np.float64),
        persistence=np.asarray(persistence, dtype=np.float64),
        future_actions=np.asarray(future_actions, dtype=np.float64),
        trajectory_ids=tuple(trajectory_ids),
        matched_pair_ids=tuple(window_pair_ids),
        transition_indices=np.asarray(
            transition_indices, dtype=np.int64
        ),
        action_kind_by_pair=actions,
        topology_by_pair=topologies,
    )

    assert assessment["status"] == "confirmed"
    assert assessment["decision"] == (
        "confirm_predictive_core_yields_useful_run_aware_warnings"
    )
    assert assessment["candidate"]["control_false_alarm_rate"] == 0.0
    assert assessment["candidate"]["treatment_detection_rate"] == 1.0
    assert (
        assessment["candidate"][
            "median_detection_delay_transitions"
        ]
        == 1.0
    )
    assert assessment["persistence"]["treatment_detection_rate"] == 0.6
    assert independent == assessment


def test_role_assignment_rejects_incomplete_cells() -> None:
    pair_ids, actions, topologies = _balanced_pair_metadata()

    with pytest.raises(ValueError, match="120"):
        assign_pair_roles(
            pair_ids=pair_ids[:-1],
            action_kind_by_pair=actions,
            topology_by_pair=topologies,
        )
