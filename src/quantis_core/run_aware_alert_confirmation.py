"""Frozen run-aware alert policy and confirmation decision rule."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Dict, Mapping, Sequence, Tuple, cast

import numpy as np
from numpy.typing import NDArray

from .action_dynamics_lab import ActionCollectionProtocol


_ACTIONS = (
    "api_rejection",
    "postgres_lock",
    "redis_dequeue_delay",
    "redis_enqueue_delay",
    "worker_pause",
)
_TOPOLOGIES = ("workers-1", "workers-2", "workers-3")
_ROLES = (
    "score_reference",
    "threshold_calibration",
    "sealed_evaluation",
)


@dataclass(frozen=True)
class RunAwareAlertContract:
    """Validated public view of the sealed alert confirmation contract."""

    payload: Mapping[str, Any]

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "RunAwareAlertContract":
        required = {
            "schema_version",
            "kind",
            "status",
            "base_collection_protocol",
            "generator_seed",
            "evidence_boundary",
            "predictive_core",
            "collection",
            "role_split",
            "policy",
            "decision_gates",
            "execution",
            "claim",
        }
        if (
            set(payload) != required
            or payload.get("schema_version") != 1
            or payload.get("kind")
            != "run_aware_alert_confirmation_contract"
            or payload.get("status") != "frozen_pre_collection"
        ):
            raise ValueError("run-aware alert contract is invalid")
        contract = cls(json.loads(json.dumps(payload, sort_keys=True)))
        contract._validate()
        return contract

    def _validate(self) -> None:
        base = _mapping(self.payload, "base_collection_protocol")
        core = _mapping(self.payload, "predictive_core")
        collection = _mapping(self.payload, "collection")
        split = _mapping(self.payload, "role_split")
        policy = _mapping(self.payload, "policy")
        gates = _mapping(self.payload, "decision_gates")
        execution = _mapping(self.payload, "execution")
        claim = _mapping(self.payload, "claim")
        if (
            base.get("path")
            != "lab/action_dynamics/development-protocol-v1.json"
            or not _is_sha256(base.get("canonical_sha256"))
            or self.payload.get("generator_seed") != 26073079
            or core.get("kind")
            != "contractive_low_rank_action_dynamics"
            or core.get("rank") != 32
            or not _is_sha256(core.get("model_sha256"))
            or not _is_sha256(core.get("confirmation_assessment_sha256"))
            or core.get("confirmation_status") != "confirmed"
            or collection
            != {
                "pair_count": 120,
                "capture_count": 240,
                "parallel_jobs": 6,
                "automatic_retry": False,
                "overwrite": False,
            }
            or split
            != {
                "unit": "matched_pair",
                "stratification": [
                    "action_kind",
                    "worker_topology",
                ],
                "ordering": "lexicographic_matched_pair_id_within_cell",
                "score_reference_positions": [0, 1],
                "threshold_calibration_positions": [2, 3],
                "sealed_evaluation_positions": [4, 5, 6, 7],
                "score_reference_pair_count": 30,
                "threshold_calibration_pair_count": 30,
                "sealed_evaluation_pair_count": 60,
            }
            or policy.get("score")
            != "mean_normalized_one_step_squared_error"
            or policy.get("actions_visible_at_inference") is not False
            or policy.get("tail_probability")
            != "empirical_upper_with_plus_one"
            or policy.get("cusum_update")
            != "max(0,previous-log(p)-log(4))"
            or policy.get("alpha") != 0.05
            or policy.get("threshold_unit")
            != "control_trajectory_cusum_maximum"
            or policy.get("threshold_quantile")
            != "ceil((n+1)*(1-alpha))th_order_statistic"
            or policy.get("crossing") != "strictly_greater"
            or policy.get("alert_latched_once_per_run") is not True
            or gates
            != {
                "control_false_alarm_rate_max": 0.05,
                "unacceptable_control_false_alarm_rate": 0.15,
                "false_alarm_exact_p_value_max": 0.05,
                "treatment_pre_onset_alert_rate_max": 0.05,
                "treatment_detection_rate_min": 0.90,
                "within_active_intervention_detection_rate_min": 0.85,
                "median_detection_delay_transitions_max": 8.0,
                "per_action_family_detection_rate_min": 0.75,
                "candidate_detection_advantage_over_persistence_min": 0.10,
                "candidate_within_active_advantage_over_persistence_min": 0.10,
            }
            or set(execution)
            != {"contract_module", "runner", "independent_assessor"}
            or any(
                not isinstance(value, Mapping)
                or not isinstance(value.get("path"), str)
                or not _is_sha256(value.get("sha256"))
                for value in execution.values()
            )
            or claim.get("pass_decision")
            != "confirm_predictive_core_yields_useful_run_aware_warnings"
            or claim.get("failure_decision")
            != "do_not_confirm_useful_run_aware_warnings"
        ):
            raise ValueError("run-aware alert contract choices drifted")

    @property
    def base_protocol_sha256(self) -> str:
        return str(
            _mapping(
                self.payload, "base_collection_protocol"
            )["canonical_sha256"]
        )

    def materialize_collection_protocol(
        self,
        base_payload: Mapping[str, Any],
        *,
        execution_source_commit: str,
    ) -> ActionCollectionProtocol:
        """Bind the frozen contract into a collector-compatible protocol."""

        if _sha256(base_payload) != self.base_protocol_sha256:
            raise ValueError("base protocol identity differs from contract")
        if not _is_git_commit(execution_source_commit):
            raise ValueError("execution source commit is invalid")
        payload = json.loads(json.dumps(base_payload, sort_keys=True))
        payload["generator_seed"] = int(self.payload["generator_seed"])
        payload["evidence_boundary"] = str(
            self.payload["evidence_boundary"]
        )
        analysis = dict(cast(Mapping[str, Any], payload["analysis"]))
        analysis.update(
            {
                "authoritative_corpus_role": (
                    "sealed_run_aware_alert_confirmation"
                ),
                "model_training_allowed_during_collection": False,
                "model_training_allowed_after_collection": False,
                "alert_policy_selection_allowed_after_collection": False,
                "all_pairs_receive_one_frozen_alert_role": True,
                "transport_split_labels_are_non_analytic": True,
                "execution_source_commit": execution_source_commit,
            }
        )
        payload["analysis"] = analysis
        claim = dict(cast(Mapping[str, Any], payload["claim"]))
        claim.update(
            {
                "supported": self.payload["claim"]["supported"],
                "excluded": self.payload["claim"]["excluded"],
                "failure_outcome": self.payload["claim"][
                    "failure_outcome"
                ],
                "alert_confirmation_contract": self.to_dict(),
                "execution_source_commit": execution_source_commit,
            }
        )
        payload["claim"] = claim
        return ActionCollectionProtocol.from_dict(payload)

    def to_dict(self) -> Dict[str, Any]:
        value = json.loads(json.dumps(self.payload, sort_keys=True))
        if not isinstance(value, dict):
            raise AssertionError("serialized contract changed type")
        return cast(Dict[str, Any], value)


def assign_pair_roles(
    *,
    pair_ids: Sequence[str],
    action_kind_by_pair: Mapping[str, str],
    topology_by_pair: Mapping[str, str],
) -> Mapping[str, str]:
    """Apply the frozen cell-stratified role split."""

    ids = tuple(str(value) for value in pair_ids)
    if len(ids) != 120 or len(set(ids)) != 120:
        raise ValueError("role assignment requires 120 unique pairs")
    if (
        set(action_kind_by_pair) != set(ids)
        or set(topology_by_pair) != set(ids)
    ):
        raise ValueError("pair metadata does not align")
    roles: Dict[str, str] = {}
    for action in _ACTIONS:
        for topology in _TOPOLOGIES:
            cell = sorted(
                pair_id
                for pair_id in ids
                if action_kind_by_pair[pair_id] == action
                and topology_by_pair[pair_id] == topology
            )
            if len(cell) != 8:
                raise ValueError("every action-topology cell needs 8 pairs")
            for position, pair_id in enumerate(cell):
                roles[pair_id] = (
                    "score_reference"
                    if position < 2
                    else (
                        "threshold_calibration"
                        if position < 4
                        else "sealed_evaluation"
                    )
                )
    if (
        len(roles) != 120
        or tuple(roles.values()).count("score_reference") != 30
        or tuple(roles.values()).count("threshold_calibration") != 30
        or tuple(roles.values()).count("sealed_evaluation") != 60
    ):
        raise ValueError("frozen role coverage is incomplete")
    return roles


def conformal_run_threshold(
    control_run_maxima: NDArray[Any], *, alpha: float
) -> float:
    """Return the conservative split-conformal run threshold."""

    maxima = np.asarray(control_run_maxima, dtype=np.float64)
    if (
        maxima.ndim != 1
        or len(maxima) < 1
        or not np.all(np.isfinite(maxima))
        or not 0.0 < alpha < 1.0
    ):
        raise ValueError("run-threshold inputs are invalid")
    rank = int(math.ceil((len(maxima) + 1) * (1.0 - alpha)))
    if rank > len(maxima):
        return math.inf
    return float(np.sort(maxima)[rank - 1])


def resettable_cusum(
    increments: NDArray[Any],
) -> NDArray[np.float64]:
    """Accumulate evidence with negative-drift reset."""

    values = np.asarray(increments, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("CUSUM increments must be one finite vector")
    result = np.empty_like(values)
    cumulative = 0.0
    for index, increment in enumerate(values):
        cumulative = max(0.0, cumulative + float(increment))
        result[index] = cumulative
    return result


def run_aware_alert_assessment(
    *,
    observed: NDArray[Any],
    candidate_prediction: NDArray[Any],
    persistence_prediction: NDArray[Any],
    future_actions: NDArray[Any],
    trajectory_ids: Sequence[str],
    matched_pair_ids: Sequence[str],
    transition_indices: NDArray[Any],
    action_kind_by_pair: Mapping[str, str],
    topology_by_pair: Mapping[str, str],
    alpha: float,
    cusum_drift: float,
) -> Mapping[str, Any]:
    """Recompute both policies and apply the frozen confirmation gates."""

    observed_array = np.asarray(observed, dtype=np.float64)
    candidate_array = np.asarray(
        candidate_prediction, dtype=np.float64
    )
    persistence_array = np.asarray(
        persistence_prediction, dtype=np.float64
    )
    actions = np.asarray(future_actions, dtype=np.float64)
    transitions = np.asarray(transition_indices, dtype=np.int64)
    sample_count = len(observed_array)
    if (
        observed_array.ndim != 4
        or candidate_array.shape != observed_array.shape
        or persistence_array.shape != observed_array.shape
        or actions.ndim != 4
        or len(actions) != sample_count
        or actions.shape[1] < 1
        or actions.shape[-1] < 2
        or len(trajectory_ids) != sample_count
        or len(matched_pair_ids) != sample_count
        or transitions.shape != (sample_count,)
        or any(
            not np.all(np.isfinite(value))
            for value in (
                observed_array,
                candidate_array,
                persistence_array,
                actions,
            )
        )
        or alpha != 0.05
        or not math.isclose(cusum_drift, math.log(4.0))
    ):
        raise ValueError("alert assessment inputs differ from contract")
    pair_ids = tuple(sorted(set(str(value) for value in matched_pair_ids)))
    roles = assign_pair_roles(
        pair_ids=pair_ids,
        action_kind_by_pair=action_kind_by_pair,
        topology_by_pair=topology_by_pair,
    )
    positions = _trajectory_positions(
        trajectory_ids, matched_pair_ids, transitions
    )
    pair_by_trajectory = {
        trajectory_id: str(matched_pair_ids[indices[0]])
        for trajectory_id, indices in positions.items()
    }
    treatment_ids = {
        trajectory_id
        for trajectory_id, indices in positions.items()
        if any(np.any(actions[index, 0, :, 1] > 0.5) for index in indices)
    }
    _validate_twin_coverage(
        positions, pair_by_trajectory, treatment_ids
    )
    scores = {
        "candidate": np.mean(
            np.square(candidate_array - observed_array),
            axis=(1, 2, 3),
        ),
        "persistence": np.mean(
            np.square(persistence_array - observed_array),
            axis=(1, 2, 3),
        ),
    }
    policies = {
        name: _assess_one_policy(
            scores=score,
            positions=positions,
            pair_by_trajectory=pair_by_trajectory,
            treatment_ids=treatment_ids,
            roles=roles,
            actions=actions,
            transitions=transitions,
            action_kind_by_pair=action_kind_by_pair,
            alpha=alpha,
            cusum_drift=cusum_drift,
        )
        for name, score in scores.items()
    }
    candidate = policies["candidate"]
    persistence = policies["persistence"]
    candidate_far = float(candidate["control_false_alarm_rate"])
    gates = {
        "role_coverage_exact": (
            candidate["evaluation_control_count"] == 60
            and candidate["evaluation_treatment_count"] == 60
            and candidate["reference_control_count"] == 30
            and candidate["threshold_calibration_control_count"] == 30
        ),
        "candidate_control_false_alarm_at_most_5_percent": (
            candidate_far <= 0.05
        ),
        "candidate_rejects_15_percent_false_alarm_null": (
            float(candidate["false_alarm_exact_p_value"]) <= 0.05
        ),
        "candidate_pre_onset_alert_at_most_5_percent": (
            float(candidate["treatment_pre_onset_alert_rate"]) <= 0.05
        ),
        "candidate_detection_at_least_90_percent": (
            float(candidate["treatment_detection_rate"]) >= 0.90
        ),
        "candidate_within_active_detection_at_least_85_percent": (
            float(
                candidate[
                    "within_active_intervention_detection_rate"
                ]
            )
            >= 0.85
        ),
        "candidate_median_delay_at_most_8": (
            candidate["median_detection_delay_transitions"] is not None
            and float(candidate["median_detection_delay_transitions"])
            <= 8.0
        ),
        "every_action_family_detection_at_least_75_percent": all(
            float(value) >= 0.75
            for value in cast(
                Mapping[str, float],
                candidate["detection_rate_by_action_family"],
            ).values()
        ),
        "candidate_detection_advantage_over_persistence_at_least_10_points": (
            float(candidate["treatment_detection_rate"])
            - float(persistence["treatment_detection_rate"])
            >= 0.10 - 1e-12
        ),
        "candidate_within_active_advantage_over_persistence_at_least_10_points": (
            float(
                candidate[
                    "within_active_intervention_detection_rate"
                ]
            )
            - float(
                persistence[
                    "within_active_intervention_detection_rate"
                ]
            )
            >= 0.10 - 1e-12
        ),
    }
    passed = all(gates.values())
    return {
        "schema_version": 1,
        "kind": "run_aware_alert_confirmation_assessment",
        "status": "confirmed" if passed else "not_confirmed",
        "decision": (
            "confirm_predictive_core_yields_useful_run_aware_warnings"
            if passed
            else "do_not_confirm_useful_run_aware_warnings"
        ),
        "policy": {
            "score": "mean_normalized_one_step_squared_error",
            "actions_visible_at_inference": False,
            "tail_probability": "empirical_upper_with_plus_one",
            "cusum_drift": cusum_drift,
            "alpha": alpha,
            "threshold_unit": "control_trajectory_cusum_maximum",
            "crossing": "strictly_greater",
            "alert_latched_once_per_run": True,
        },
        "role_counts": {
            role: tuple(roles.values()).count(role) for role in _ROLES
        },
        "candidate": candidate,
        "persistence": persistence,
        "gates": gates,
    }


def _assess_one_policy(
    *,
    scores: NDArray[np.float64],
    positions: Mapping[str, Tuple[int, ...]],
    pair_by_trajectory: Mapping[str, str],
    treatment_ids: set[str],
    roles: Mapping[str, str],
    actions: NDArray[np.float64],
    transitions: NDArray[np.int64],
    action_kind_by_pair: Mapping[str, str],
    alpha: float,
    cusum_drift: float,
) -> Mapping[str, Any]:
    control_ids = set(positions) - treatment_ids
    reference_ids = sorted(
        trajectory_id
        for trajectory_id in control_ids
        if roles[pair_by_trajectory[trajectory_id]]
        == "score_reference"
    )
    calibration_ids = sorted(
        trajectory_id
        for trajectory_id in control_ids
        if roles[pair_by_trajectory[trajectory_id]]
        == "threshold_calibration"
    )
    evaluation_ids = sorted(
        trajectory_id
        for trajectory_id in positions
        if roles[pair_by_trajectory[trajectory_id]]
        == "sealed_evaluation"
    )
    reference = np.concatenate(
        [scores[list(positions[value])] for value in reference_ids]
    )
    p_values = np.asarray(
        [
            (1.0 + float(np.count_nonzero(reference >= score)))
            / (len(reference) + 1.0)
            for score in scores
        ],
        dtype=np.float64,
    )
    increments = -np.log(p_values) - cusum_drift
    traces = {
        trajectory_id: resettable_cusum(
            increments[list(indices)]
        )
        for trajectory_id, indices in positions.items()
    }
    maxima = np.asarray(
        [float(np.max(traces[value])) for value in calibration_ids],
        dtype=np.float64,
    )
    threshold = conformal_run_threshold(maxima, alpha=alpha)
    rows = []
    for trajectory_id in evaluation_ids:
        indices = positions[trajectory_id]
        trace = traces[trajectory_id]
        crossings = [
            int(transitions[index]) + 1
            for index, value in zip(indices, trace)
            if value > threshold
        ]
        first_alert = crossings[0] if crossings else None
        is_treatment = trajectory_id in treatment_ids
        active_indices = [
            index
            for index in indices
            if np.any(actions[index, 0, :, 1] > 0.5)
        ]
        onset = (
            min(int(transitions[index]) for index in active_indices)
            if active_indices
            else None
        )
        stop = (
            max(int(transitions[index]) for index in active_indices)
            if active_indices
            else None
        )
        valid_detection = bool(
            is_treatment
            and first_alert is not None
            and onset is not None
            and first_alert > onset
        )
        rows.append(
            {
                "trajectory_id": trajectory_id,
                "matched_pair_id": pair_by_trajectory[trajectory_id],
                "action_kind": (
                    action_kind_by_pair[pair_by_trajectory[trajectory_id]]
                    if is_treatment
                    else None
                ),
                "is_treatment": is_treatment,
                "onset_transition": onset,
                "stop_transition": stop,
                "first_alert_transition": first_alert,
                "pre_onset_alert": bool(
                    is_treatment
                    and first_alert is not None
                    and onset is not None
                    and first_alert <= onset
                ),
                "detected": valid_detection,
                "detected_while_active": bool(
                    valid_detection
                    and stop is not None
                    and first_alert is not None
                    and first_alert <= stop
                ),
                "detection_delay_transitions": (
                    int(first_alert - onset)
                    if valid_detection
                    and first_alert is not None
                    and onset is not None
                    else None
                ),
            }
        )
    controls = [row for row in rows if not row["is_treatment"]]
    treatments = [row for row in rows if row["is_treatment"]]
    detections = [row for row in treatments if row["detected"]]
    false_alarm_count = sum(
        row["first_alert_transition"] is not None for row in controls
    )
    family_rates = {
        action: float(
            np.mean(
                [
                    bool(row["detected"])
                    for row in treatments
                    if row["action_kind"] == action
                ]
            )
        )
        for action in _ACTIONS
    }
    delays = [
        int(row["detection_delay_transitions"]) for row in detections
    ]
    return {
        "reference_control_count": len(reference_ids),
        "reference_control_window_count": len(reference),
        "threshold_calibration_control_count": len(calibration_ids),
        "threshold": threshold,
        "threshold_calibration_exceedance_count": int(
            np.count_nonzero(maxima > threshold)
        ),
        "evaluation_control_count": len(controls),
        "evaluation_treatment_count": len(treatments),
        "control_false_alarm_count": false_alarm_count,
        "control_false_alarm_rate": (
            false_alarm_count / len(controls)
        ),
        "false_alarm_exact_p_value": _binomial_lower_tail(
            false_alarm_count, len(controls), 0.15
        ),
        "treatment_pre_onset_alert_rate": float(
            np.mean([bool(row["pre_onset_alert"]) for row in treatments])
        ),
        "treatment_detection_rate": len(detections) / len(treatments),
        "within_active_intervention_detection_rate": float(
            np.mean(
                [
                    bool(row["detected_while_active"])
                    for row in treatments
                ]
            )
        ),
        "median_detection_delay_transitions": (
            float(np.median(delays)) if delays else None
        ),
        "detection_rate_by_action_family": family_rates,
        "trajectory_rows": rows,
    }


def _trajectory_positions(
    trajectory_ids: Sequence[str],
    matched_pair_ids: Sequence[str],
    transitions: NDArray[np.int64],
) -> Mapping[str, Tuple[int, ...]]:
    raw: Dict[str, list[int]] = {}
    pair_by_trajectory: Dict[str, str] = {}
    for index, (trajectory_id, pair_id) in enumerate(
        zip(trajectory_ids, matched_pair_ids)
    ):
        trajectory = str(trajectory_id)
        pair = str(pair_id)
        raw.setdefault(trajectory, []).append(index)
        previous = pair_by_trajectory.setdefault(trajectory, pair)
        if previous != pair:
            raise ValueError("trajectory spans multiple matched pairs")
    result = {}
    for trajectory, indices in raw.items():
        ordered = sorted(indices, key=lambda index: int(transitions[index]))
        if len({int(transitions[index]) for index in ordered}) != len(ordered):
            raise ValueError("trajectory transition indices repeat")
        result[trajectory] = tuple(ordered)
    return result


def _validate_twin_coverage(
    positions: Mapping[str, Tuple[int, ...]],
    pair_by_trajectory: Mapping[str, str],
    treatment_ids: set[str],
) -> None:
    if len(positions) != 240 or len(treatment_ids) != 120:
        raise ValueError("assessment requires 120 complete twins")
    trajectories_by_pair: Dict[str, list[str]] = {}
    for trajectory_id, pair_id in pair_by_trajectory.items():
        trajectories_by_pair.setdefault(pair_id, []).append(trajectory_id)
    if (
        len(trajectories_by_pair) != 120
        or any(len(values) != 2 for values in trajectories_by_pair.values())
        or any(
            sum(value in treatment_ids for value in values) != 1
            for values in trajectories_by_pair.values()
        )
    ):
        raise ValueError("assessment requires one control and treatment twin")


def _binomial_lower_tail(successes: int, trials: int, probability: float) -> float:
    if (
        isinstance(successes, bool)
        or isinstance(trials, bool)
        or successes < 0
        or trials < 1
        or successes > trials
        or not 0.0 < probability < 1.0
    ):
        raise ValueError("binomial inputs are invalid")
    return float(
        sum(
            math.comb(trials, count)
            * probability**count
            * (1.0 - probability) ** (trials - count)
            for count in range(successes + 1)
        )
    )


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be an object")
    return value


def _sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_git_commit(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )
