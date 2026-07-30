import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from lab.action_dynamics.assess_run_aware_alert_confirmation import (
    assess_run_aware_alert_arrays_independently,
)
from quantis_core.action_dynamics_lab import (
    prepare_action_collection,
    validate_action_collection_attestation,
)
from quantis_core.run_aware_alert_confirmation import (
    RunAwareAlertContract,
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


def test_v2_contract_changes_only_seed_and_collection_concurrency(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (
            repository
            / "lab"
            / "action_dynamics"
            / "run-aware-alert-confirmation-contract-v2.json"
        ).read_text()
    )
    v1_payload = json.loads(
        (
            repository
            / "lab"
            / "action_dynamics"
            / "run-aware-alert-confirmation-contract-v1.json"
        ).read_text()
    )

    contract = RunAwareAlertContract.from_dict(payload)
    RunAwareAlertContract.from_dict(v1_payload)

    assert contract.payload["generator_seed"] == 26073080
    assert contract.payload["collection"]["parallel_jobs"] == 4
    normalized_v1 = copy.deepcopy(v1_payload)
    normalized_v2 = copy.deepcopy(payload)
    for normalized in (normalized_v1, normalized_v2):
        normalized["schema_version"] = 0
        normalized["evidence_boundary"] = ""
        normalized["generator_seed"] = 0
        normalized["collection"]["parallel_jobs"] = 0
        normalized["execution"]["contract_module"]["sha256"] = ""
        normalized["execution"]["runner"]["sha256"] = ""
    assert normalized_v2 == normalized_v1
    base = json.loads(
        (
            repository
            / "lab"
            / "action_dynamics"
            / "development-protocol-v1.json"
        ).read_text()
    )
    protocol = contract.materialize_collection_protocol(
        base, execution_source_commit="a" * 40
    )
    manifests, assignments = prepare_action_collection(
        protocol,
        image_digests={
            "application": "sha256:" + "1" * 64,
        },
        observation_schema_sha256="2" * 64,
    )
    assert protocol.parallel_jobs == 4
    assert protocol.scheduling["lane_count"] == 4
    assert protocol.scheduling["batch_count"] == 30
    assert protocol.scheduling["pairs_per_batch"] == 4
    assert protocol.scheduling["lane_assignment"].endswith(
        "modulo 4."
    )
    assert protocol.scheduling["batch_assignment"].endswith(
        "divided by 4)."
    )
    assert {assignment.lane for assignment in assignments} == {
        1,
        2,
        3,
        4,
    }
    assert max(
        sum(item.batch == batch for item in assignments)
        for batch in {item.batch for item in assignments}
    ) == 8
    assert max(item.batch for item in assignments) == 30

    prepared = tmp_path / "prepared"
    manifest_directory = prepared / "manifests"
    manifest_directory.mkdir(parents=True)
    protocol_payload = protocol.to_dict()
    plan_payload = {"schema_version": 1, "kind": "test_plan"}
    (prepared / "protocol.json").write_text(
        json.dumps(protocol_payload)
    )
    (prepared / "plan.json").write_text(json.dumps(plan_payload))
    manifest_by_case = {
        item.action_case.case_id: item for item in manifests
    }
    attested_cases = []
    for assignment in assignments:
        manifest_path = (
            manifest_directory / f"{assignment.case_id}.json"
        )
        manifest_path.write_text(
            json.dumps(manifest_by_case[assignment.case_id].to_dict())
        )
        attested_cases.append(
            {
                **assignment.to_dict(),
                "manifest_sha256": hashlib.sha256(
                    manifest_path.read_bytes()
                ).hexdigest(),
            }
        )
    attestation = {
        "schema_version": 1,
        "kind": "action_dynamics_collection_attestation",
        "protocol_sha256": hashlib.sha256(
            json.dumps(
                protocol_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "plan_sha256": hashlib.sha256(
            json.dumps(
                plan_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "case_count": len(assignments),
        "pair_count": len(assignments) // 2,
        "parallel_jobs": 4,
        "application_image_id": "sha256:" + "1" * 64,
        "application_build_context_sha256": "3" * 64,
        "cases": attested_cases,
    }
    validate_action_collection_attestation(
        attestation, prepared, assignments
    )

    drifted = copy.deepcopy(payload)
    drifted["collection"]["parallel_jobs"] = 6
    with pytest.raises(ValueError, match="choices drifted"):
        RunAwareAlertContract.from_dict(drifted)


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
