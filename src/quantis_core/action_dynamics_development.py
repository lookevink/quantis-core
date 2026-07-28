"""Pure Phase-0 assessment for action-conditioned graph dynamics."""

from typing import Any, Dict, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .action_conditioned_dynamics import TrajectoryDistribution


_PHASE_ZERO_NO_ACTION_ID = "none"
_PHASE_ZERO_EXPECTED_PROPAGATION_STEPS = (1, 2, 3, 4, 5)
_PHASE_ZERO_PROPAGATION_EFFECT_THRESHOLD = 0.01
_PHASE_ZERO_MINIMUM_FORECAST_RELATIVE_IMPROVEMENT = 0.1
_PHASE_ZERO_MINIMUM_PERSISTENCE_RELATIVE_IMPROVEMENT = 0.1
_PHASE_ZERO_MINIMUM_ATTRIBUTION_HIT_AT_1 = 0.7
_PHASE_ZERO_MINIMUM_NO_ACTION_SPECIFICITY = 0.9


def assess_action_dynamics_development(
    *,
    action_prediction: TrajectoryDistribution,
    action_agnostic_prediction: TrajectoryDistribution,
    persistence_prediction: TrajectoryDistribution,
    observed_future: NDArray[np.float64],
    predicted_candidate_ids: Sequence[str],
    true_candidate_ids: Sequence[str],
    no_action_candidate_id: str,
    propagation_delay_passed: bool,
    minimum_forecast_relative_improvement: float,
    minimum_persistence_relative_improvement: float,
    minimum_attribution_hit_at_1: float,
    minimum_no_action_specificity: float,
) -> Dict[str, Any]:
    """Recompute Phase-0 forecast and inverse-attribution gates."""

    observed = np.asarray(observed_future, dtype=np.float64)
    if (
        observed.shape != action_prediction.mean.shape
        or observed.shape != action_agnostic_prediction.mean.shape
        or observed.shape != persistence_prediction.mean.shape
        or not np.all(np.isfinite(observed))
    ):
        raise ValueError("assessment trajectories do not align")
    predicted = tuple(predicted_candidate_ids)
    truth = tuple(true_candidate_ids)
    if (
        not predicted
        or len(predicted) != len(truth)
        or not no_action_candidate_id
    ):
        raise ValueError("assessment attribution labels do not align")
    if not isinstance(propagation_delay_passed, bool):
        raise ValueError(
            "propagation-delay result must be boolean"
        )
    for threshold in (
        minimum_forecast_relative_improvement,
        minimum_persistence_relative_improvement,
        minimum_attribution_hit_at_1,
        minimum_no_action_specificity,
    ):
        if (
            isinstance(threshold, bool)
            or not np.isfinite(threshold)
            or not 0.0 <= threshold <= 1.0
        ):
            raise ValueError(
                "assessment thresholds must be finite probabilities"
            )

    action_mse = float(
        np.mean(np.square(action_prediction.mean - observed))
    )
    action_agnostic_mse = float(
        np.mean(
            np.square(action_agnostic_prediction.mean - observed)
        )
    )
    persistence_mse = float(
        np.mean(
            np.square(persistence_prediction.mean - observed)
        )
    )
    forecast_relative_improvement = (
        (action_agnostic_mse - action_mse)
        / action_agnostic_mse
        if action_agnostic_mse > 0.0
        else 0.0
    )
    persistence_relative_improvement = (
        (persistence_mse - action_mse) / persistence_mse
        if persistence_mse > 0.0
        else 0.0
    )
    hit_at_1 = float(
        np.mean(
            np.asarray(
                [
                    predicted_id == true_id
                    for predicted_id, true_id in zip(
                        predicted, truth
                    )
                ],
                dtype=np.float64,
            )
        )
    )
    no_action_positions = [
        position
        for position, true_id in enumerate(truth)
        if true_id == no_action_candidate_id
    ]
    if not no_action_positions:
        raise ValueError(
            "assessment requires at least one no-action case"
        )
    no_action_specificity = float(
        np.mean(
            np.asarray(
                [
                    predicted[position]
                    == no_action_candidate_id
                    for position in no_action_positions
                ],
                dtype=np.float64,
            )
        )
    )
    gates = {
        "action_conditioning_improves_forecast": _gate(
            forecast_relative_improvement,
            minimum_forecast_relative_improvement,
        ),
        "beats_persistence": _gate(
            persistence_relative_improvement,
            minimum_persistence_relative_improvement,
        ),
        "attributes_known_actions": _gate(
            hit_at_1, minimum_attribution_hit_at_1
        ),
        "recognizes_no_action": _gate(
            no_action_specificity,
            minimum_no_action_specificity,
        ),
        "respects_graph_propagation_delay": {
            "passed": propagation_delay_passed,
            "observed": propagation_delay_passed,
            "threshold": True,
        },
    }
    supported = all(gate["passed"] for gate in gates.values())
    return {
        "schema_version": 1,
        "kind": "action_dynamics_phase_zero_assessment",
        "status": "supported" if supported else "not_supported",
        "decision": (
            "advance_to_instrumented_pilot"
            if supported
            else "stop_or_redesign_phase_zero"
        ),
        "gates": gates,
        "scores": {
            "action_conditioned_normalized_mse": action_mse,
            "action_agnostic_normalized_mse": (
                action_agnostic_mse
            ),
            "persistence_normalized_mse": persistence_mse,
            "forecast_relative_improvement": (
                forecast_relative_improvement
            ),
            "persistence_relative_improvement": (
                persistence_relative_improvement
            ),
            "attribution_hit_at_1": hit_at_1,
            "no_action_specificity": no_action_specificity,
        },
        "evidence_boundary": (
            "synthetic Phase-0 tracer bullet only; not lab, "
            "confirmation, attribution, or world-model evidence"
        ),
    }


def assess_action_dynamics_evidence(
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    """Recompute the assessment from a portable evidence payload."""

    if (
        set(payload)
        != {
            "schema_version",
            "kind",
            "observed_future",
            "predictions",
            "propagation_case_id",
            "attribution_rows",
        }
        or payload.get("schema_version") != 2
        or payload.get("kind")
        != "action_dynamics_phase_zero_evidence"
    ):
        raise ValueError("unsupported Phase-0 evidence")
    raw_predictions = payload["predictions"]
    if (
        not isinstance(raw_predictions, dict)
        or set(raw_predictions)
        != {
            "action_conditioned",
            "action_agnostic",
            "persistence",
        }
    ):
        raise ValueError(
            "Phase-0 prediction evidence is invalid"
        )
    predictions = dict(raw_predictions)
    observed = np.asarray(
        payload["observed_future"], dtype=np.float64
    )
    propagation_case_id = payload["propagation_case_id"]
    if not isinstance(propagation_case_id, str):
        raise ValueError(
            "Phase-0 propagation case id is invalid"
        )
    attribution_rows = payload["attribution_rows"]
    if (
        not isinstance(attribution_rows, list)
        or not attribution_rows
    ):
        raise ValueError(
            "Phase-0 attribution evidence is invalid"
        )
    predicted_candidate_ids = []
    true_candidate_ids = []
    case_ids = set()
    propagation_delta = None
    for raw_row in attribution_rows:
        if not isinstance(raw_row, dict):
            raise ValueError(
                "Phase-0 attribution row is invalid"
            )
        row = dict(raw_row)
        if set(row) != {
            "case_id",
            "true_candidate_id",
            "observed_future",
            "candidate_ids",
            "candidate_distribution",
        }:
            raise ValueError(
                "Phase-0 attribution row schema is invalid"
            )
        case_id = row["case_id"]
        true_id = row["true_candidate_id"]
        candidate_ids = row["candidate_ids"]
        raw_distribution = row["candidate_distribution"]
        if (
            not isinstance(case_id, str)
            or not isinstance(true_id, str)
            or not isinstance(candidate_ids, list)
            or not candidate_ids
            or any(
                not isinstance(candidate_id, str)
                for candidate_id in candidate_ids
            )
            or len(set(candidate_ids)) != len(candidate_ids)
            or _PHASE_ZERO_NO_ACTION_ID not in candidate_ids
            or not isinstance(raw_distribution, dict)
            or set(raw_distribution) != {"mean", "variance"}
            or case_id in case_ids
        ):
            raise ValueError(
                "Phase-0 attribution row values are invalid"
            )
        candidate_distribution = TrajectoryDistribution(
            mean=np.asarray(
                raw_distribution["mean"], dtype=np.float64
            ),
            variance=np.asarray(
                raw_distribution["variance"], dtype=np.float64
            ),
        )
        row_observed = np.asarray(
            row["observed_future"], dtype=np.float64
        )
        if (
            candidate_distribution.mean.shape[0]
            != len(candidate_ids)
            or row_observed.shape
            != candidate_distribution.mean.shape[1:]
            or not np.all(np.isfinite(row_observed))
        ):
            raise ValueError(
                "Phase-0 attribution trajectories do not align"
            )
        observed_batch = np.repeat(
            row_observed[np.newaxis, ...],
            len(candidate_ids),
            axis=0,
        )
        nll_values = candidate_distribution.negative_log_likelihood(
            observed_batch
        )
        ranked_positions = sorted(
            range(len(candidate_ids)),
            key=lambda position: (
                float(nll_values[position]),
                candidate_ids[position],
            ),
        )
        winner_position = ranked_positions[0]
        case_ids.add(case_id)
        predicted_candidate_ids.append(
            candidate_ids[winner_position]
        )
        true_candidate_ids.append(true_id)
        if case_id == propagation_case_id:
            no_action_position = candidate_ids.index(
                _PHASE_ZERO_NO_ACTION_ID
            )
            propagation_delta = (
                candidate_distribution.mean[winner_position]
                - candidate_distribution.mean[
                    no_action_position
                ]
            )

    if propagation_case_id not in case_ids or propagation_delta is None:
        raise ValueError(
            "Phase-0 propagation case is missing"
        )
    if (
        propagation_delta.shape[1]
        != len(_PHASE_ZERO_EXPECTED_PROPAGATION_STEPS)
    ):
        raise ValueError(
            "Phase-0 propagation evidence is invalid"
        )
    observed_propagation_steps = []
    for entity_position in range(propagation_delta.shape[1]):
        affected = np.flatnonzero(
            np.max(
                np.abs(
                    propagation_delta[:, entity_position, :]
                ),
                axis=1,
            )
            > _PHASE_ZERO_PROPAGATION_EFFECT_THRESHOLD
        )
        observed_propagation_steps.append(
            int(affected[0]) if len(affected) else -1
        )
    propagation_delay_passed = (
        tuple(observed_propagation_steps)
        == _PHASE_ZERO_EXPECTED_PROPAGATION_STEPS
    )

    def distribution(name: str) -> TrajectoryDistribution:
        raw = dict(predictions[name])
        return TrajectoryDistribution(
            mean=np.asarray(raw["mean"], dtype=np.float64),
            variance=np.asarray(
                raw["variance"], dtype=np.float64
            ),
        )

    return assess_action_dynamics_development(
        action_prediction=distribution("action_conditioned"),
        action_agnostic_prediction=distribution(
            "action_agnostic"
        ),
        persistence_prediction=distribution("persistence"),
        observed_future=observed,
        predicted_candidate_ids=tuple(predicted_candidate_ids),
        true_candidate_ids=tuple(true_candidate_ids),
        no_action_candidate_id=_PHASE_ZERO_NO_ACTION_ID,
        propagation_delay_passed=propagation_delay_passed,
        minimum_forecast_relative_improvement=(
            _PHASE_ZERO_MINIMUM_FORECAST_RELATIVE_IMPROVEMENT
        ),
        minimum_persistence_relative_improvement=(
            _PHASE_ZERO_MINIMUM_PERSISTENCE_RELATIVE_IMPROVEMENT
        ),
        minimum_attribution_hit_at_1=(
            _PHASE_ZERO_MINIMUM_ATTRIBUTION_HIT_AT_1
        ),
        minimum_no_action_specificity=(
            _PHASE_ZERO_MINIMUM_NO_ACTION_SPECIFICITY
        ),
    )
def _gate(observed: float, threshold: float) -> Dict[str, Any]:
    return {
        "passed": observed >= threshold,
        "observed": observed,
        "threshold": threshold,
    }
