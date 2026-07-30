"""Independently assess stored run-aware alert confirmation arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from quantis_core.action_conditioned_dynamics import (
    ActionTrajectoryCompiler,
)
from quantis_core.action_dynamics_corpus import (
    load_action_dynamics_development_corpus,
)
from quantis_core.edge_dynamics.models import (
    ContractiveLowRankDynamics,
)
from quantis_core.run_aware_alert_confirmation import (
    RunAwareAlertContract,
)


def assess_stored_run_aware_alert_confirmation(
    *,
    contract_path: Path,
    model_path: Path,
    predictive_confirmation_assessment: Path,
    source_artifact_manifest: Path,
    predictions_directory: Path,
    expected_contract_sha256: str,
    expected_model_sha256: str,
    expected_predictive_confirmation_assessment_sha256: str,
    expected_source_artifact_manifest_sha256: str,
    expected_prediction_manifest_sha256: str,
) -> Mapping[str, Any]:
    """Verify identities and recompute the frozen policy decision."""

    expected = {
        contract_path: expected_contract_sha256,
        model_path: expected_model_sha256,
        predictive_confirmation_assessment: (
            expected_predictive_confirmation_assessment_sha256
        ),
        source_artifact_manifest: (
            expected_source_artifact_manifest_sha256
        ),
        predictions_directory / "prediction-manifest.json": (
            expected_prediction_manifest_sha256
        ),
    }
    if any(_file_sha256(path) != digest for path, digest in expected.items()):
        raise ValueError("frozen alert input hash differs")
    contract = RunAwareAlertContract.from_dict(_read_object(contract_path))
    core = contract.payload["predictive_core"]
    if (
        not isinstance(core, Mapping)
        or core.get("model_sha256") != expected_model_sha256
        or core.get("confirmation_assessment_sha256")
        != expected_predictive_confirmation_assessment_sha256
    ):
        raise ValueError("assessed predictive core differs from contract")
    predictive_assessment = _read_object(
        predictive_confirmation_assessment
    )
    if (
        predictive_assessment.get("status") != "confirmed"
        or predictive_assessment.get("decision")
        != "confirm_learnable_action_dynamics"
    ):
        raise ValueError("predictive core is not confirmed")
    _verify_qualified_source(
        contract=contract,
        contract_path=contract_path,
        source_artifact_manifest=source_artifact_manifest,
    )
    manifest = _read_object(
        predictions_directory / "prediction-manifest.json"
    )
    recorded = manifest.get("sha256")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind")
        != "run_aware_alert_prediction_manifest"
        or manifest.get("contract_sha256")
        != expected_contract_sha256
        or manifest.get("model_sha256") != expected_model_sha256
        or manifest.get(
            "predictive_confirmation_assessment_sha256"
        )
        != expected_predictive_confirmation_assessment_sha256
        or manifest.get("source_artifact_manifest_sha256")
        != expected_source_artifact_manifest_sha256
        or not isinstance(recorded, dict)
    ):
        raise ValueError("alert prediction manifest identity differs")
    for filename, digest in recorded.items():
        path = predictions_directory / str(filename)
        if (
            not isinstance(digest, str)
            or not path.is_file()
            or _file_sha256(path) != digest
        ):
            raise ValueError("stored alert prediction drifted")
    with np.load(
        predictions_directory / "alert-inputs.npz",
        allow_pickle=False,
    ) as arrays:
        observed = arrays["observed"]
        future_actions = arrays["future_actions"]
        trajectory_ids = tuple(
            str(value) for value in arrays["trajectory_ids"]
        )
        matched_pair_ids = tuple(
            str(value) for value in arrays["matched_pair_ids"]
        )
        transition_indices = arrays["transition_indices"]
        action_kind_by_pair = json.loads(
            str(arrays["action_kind_by_pair_json"])
        )
        topology_by_pair = json.loads(
            str(arrays["topology_by_pair_json"])
        )
    predictions = {}
    for name in ("candidate", "persistence"):
        with np.load(
            predictions_directory / f"{name}.npz",
            allow_pickle=False,
        ) as arrays:
            predictions[name] = arrays["prediction"]
    _verify_stored_arrays_against_raw(
        contract=contract,
        contract_path=contract_path,
        model_path=model_path,
        source_artifact_manifest=source_artifact_manifest,
        observed=observed,
        future_actions=future_actions,
        trajectory_ids=trajectory_ids,
        matched_pair_ids=matched_pair_ids,
        transition_indices=transition_indices,
        action_kind_by_pair=action_kind_by_pair,
        topology_by_pair=topology_by_pair,
        candidate_prediction=predictions["candidate"],
        persistence_prediction=predictions["persistence"],
    )
    result = dict(
        assess_run_aware_alert_arrays_independently(
            observed=np.asarray(observed, dtype=np.float64),
            candidate=np.asarray(
                predictions["candidate"], dtype=np.float64
            ),
            persistence=np.asarray(
                predictions["persistence"], dtype=np.float64
            ),
            future_actions=np.asarray(
                future_actions, dtype=np.float64
            ),
            trajectory_ids=trajectory_ids,
            matched_pair_ids=matched_pair_ids,
            transition_indices=np.asarray(
                transition_indices, dtype=np.int64
            ),
            action_kind_by_pair={
                str(key): str(value)
                for key, value in action_kind_by_pair.items()
            },
            topology_by_pair={
                str(key): str(value)
                for key, value in topology_by_pair.items()
            },
        )
    )
    result["identities"] = {
        "contract_sha256": expected_contract_sha256,
        "model_sha256": expected_model_sha256,
        "predictive_confirmation_assessment_sha256": (
            expected_predictive_confirmation_assessment_sha256
        ),
        "source_artifact_manifest_sha256": (
            expected_source_artifact_manifest_sha256
        ),
        "prediction_manifest_sha256": (
            expected_prediction_manifest_sha256
        ),
    }
    return result


def _verify_stored_arrays_against_raw(
    *,
    contract: RunAwareAlertContract,
    contract_path: Path,
    model_path: Path,
    source_artifact_manifest: Path,
    observed: np.ndarray,
    future_actions: np.ndarray,
    trajectory_ids: Sequence[str],
    matched_pair_ids: Sequence[str],
    transition_indices: np.ndarray,
    action_kind_by_pair: Mapping[str, Any],
    topology_by_pair: Mapping[str, Any],
    candidate_prediction: np.ndarray,
    persistence_prediction: np.ndarray,
) -> None:
    source_root = source_artifact_manifest.parent
    manifest = _read_object(source_artifact_manifest)
    recorded = manifest.get("sha256")
    if not isinstance(recorded, dict):
        raise ValueError("raw source manifest is invalid")
    with tempfile.TemporaryDirectory(
        prefix="quantis-alert-assessor-"
    ) as temporary:
        clean_root = Path(temporary)
        (clean_root / "artifact-manifest.json").write_bytes(
            source_artifact_manifest.read_bytes()
        )
        for raw_name in recorded:
            if not isinstance(raw_name, str):
                raise ValueError("raw source path is invalid")
            source = source_root / raw_name
            destination = clean_root / raw_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, destination)
        corpus = load_action_dynamics_development_corpus(clean_root)
    repository = contract_path.resolve().parents[2]
    core = contract.payload["predictive_core"]
    if not isinstance(core, Mapping):
        raise ValueError("predictive-core contract is invalid")
    compiler_path = (
        repository / str(core["compiler_metadata_path"])
    )
    compiler_manifest_path = (
        repository
        / str(core["compiler_artifact_manifest_path"])
    )
    if (
        _file_sha256(compiler_path)
        != core["compiler_metadata_sha256"]
        or _file_sha256(compiler_manifest_path)
        != core["compiler_artifact_manifest_sha256"]
    ):
        raise ValueError("frozen preprocessing artifact drifted")
    compiler_manifest = _read_object(compiler_manifest_path)
    compiler_recorded = compiler_manifest.get("sha256")
    if (
        compiler_manifest.get("schema_version") != 1
        or compiler_manifest.get("kind")
        != "edge_dynamics_preprocessing_manifest"
        or not isinstance(compiler_recorded, dict)
        or compiler_recorded.get("metadata.json")
        != core["compiler_metadata_sha256"]
    ):
        raise ValueError("preprocessing provenance is invalid")
    compiler_metadata = _read_object(compiler_path)
    compiler_payload = compiler_metadata.get("compiler")
    if not isinstance(compiler_payload, Mapping):
        raise ValueError("compiler payload is absent")
    compiler = ActionTrajectoryCompiler.from_dict(compiler_payload)
    windows = compiler.transform(corpus.runs)
    model = ContractiveLowRankDynamics.from_dict(
        _read_object(model_path)
    )
    neutral_actions = np.zeros_like(windows.future_actions)
    neutral_actions[..., 0] = 1.0
    expected_candidate = model.rollout(
        windows.histories,
        windows.future_controls,
        neutral_actions,
        windows.graph,
    ).mean[:, :1]
    expected_persistence = windows.histories[:, -1:, :, :]
    expected_actions = {
        run.manifest.matched_pair_id: (
            run.manifest.actions[0].action_kind
        )
        for run in corpus.runs
        if run.manifest.actions
    }
    expected_topologies = {
        run.manifest.matched_pair_id: run.manifest.topology_id
        for run in corpus.runs
    }
    stored_actions = {
        str(key): str(value)
        for key, value in action_kind_by_pair.items()
    }
    stored_topologies = {
        str(key): str(value) for key, value in topology_by_pair.items()
    }
    if (
        not np.array_equal(observed, windows.future_states[:, :1])
        or not np.array_equal(
            future_actions, windows.future_actions[:, :1]
        )
        or tuple(trajectory_ids) != windows.trajectory_ids
        or tuple(matched_pair_ids) != windows.matched_pair_ids
        or not np.array_equal(
            transition_indices, windows.transition_indices
        )
        or stored_actions != expected_actions
        or stored_topologies != expected_topologies
        or not np.array_equal(
            candidate_prediction, expected_candidate
        )
        or not np.array_equal(
            persistence_prediction, expected_persistence
        )
    ):
        raise ValueError(
            "stored alert arrays do not reproduce from qualified raw data"
        )


def _verify_qualified_source(
    *,
    contract: RunAwareAlertContract,
    contract_path: Path,
    source_artifact_manifest: Path,
) -> None:
    source_root = source_artifact_manifest.parent
    manifest = _read_object(source_artifact_manifest)
    recorded = manifest.get("sha256")
    required = {
        "inputs/protocol.json",
        "data-quality.json",
        "collection-attestation.json",
    }
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "action_dynamics_artifact_manifest"
        or not isinstance(recorded, dict)
        or not required.issubset(recorded)
    ):
        raise ValueError("qualified source manifest is incomplete")
    for raw_name, raw_digest in recorded.items():
        if not isinstance(raw_name, str) or not isinstance(
            raw_digest, str
        ):
            raise ValueError("qualified source manifest entry is invalid")
        path = source_root / raw_name
        if (
            not path.is_file()
            or _file_sha256(path) != raw_digest
        ):
            raise ValueError("qualified source evidence drifted")
    quality = _read_object(source_root / "data-quality.json")
    if (
        quality.get("kind")
        != "action_dynamics_data_quality_assessment"
        or quality.get("status") != "qualified"
    ):
        raise ValueError("source campaign is not qualified")
    protocol = _read_object(source_root / "inputs" / "protocol.json")
    claim = protocol.get("claim")
    source_commit = (
        claim.get("execution_source_commit")
        if isinstance(claim, Mapping)
        else None
    )
    if (
        not isinstance(claim, Mapping)
        or claim.get("alert_confirmation_contract")
        != contract.to_dict()
        or not isinstance(source_commit, str)
    ):
        raise ValueError("source protocol is not bound to alert contract")
    repository = contract_path.resolve().parents[2]
    base = _read_object(
        repository
        / str(
            contract.payload["base_collection_protocol"]["path"]
        )
    )
    expected = contract.materialize_collection_protocol(
        base, execution_source_commit=source_commit
    )
    if protocol != expected.to_dict():
        raise ValueError("source protocol differs from frozen materialization")


def assess_run_aware_alert_arrays_independently(
    *,
    observed: np.ndarray,
    candidate: np.ndarray,
    persistence: np.ndarray,
    future_actions: np.ndarray,
    trajectory_ids: Sequence[str],
    matched_pair_ids: Sequence[str],
    transition_indices: np.ndarray,
    action_kind_by_pair: Mapping[str, str],
    topology_by_pair: Mapping[str, str],
) -> Mapping[str, Any]:
    if (
        observed.ndim != 4
        or candidate.shape != observed.shape
        or persistence.shape != observed.shape
        or future_actions.ndim != 4
        or len(future_actions) != len(observed)
        or future_actions.shape[-1] < 2
        or len(trajectory_ids) != len(observed)
        or len(matched_pair_ids) != len(observed)
        or transition_indices.shape != (len(observed),)
        or any(
            not np.all(np.isfinite(value))
            for value in (
                observed,
                candidate,
                persistence,
                future_actions,
            )
        )
    ):
        raise ValueError("stored alert arrays do not align")
    pair_ids = tuple(sorted(set(matched_pair_ids)))
    roles = _independent_roles(
        pair_ids, action_kind_by_pair, topology_by_pair
    )
    positions: Dict[str, list[int]] = {}
    pair_by_trajectory: Dict[str, str] = {}
    for index, (trajectory_id, pair_id) in enumerate(
        zip(trajectory_ids, matched_pair_ids)
    ):
        positions.setdefault(trajectory_id, []).append(index)
        prior = pair_by_trajectory.setdefault(trajectory_id, pair_id)
        if prior != pair_id:
            raise ValueError("trajectory pair identity changed")
    for indices in positions.values():
        indices.sort(key=lambda index: int(transition_indices[index]))
    treatments = {
        trajectory_id
        for trajectory_id, indices in positions.items()
        if any(
            np.any(future_actions[index, 0, :, 1] > 0.5)
            for index in indices
        )
    }
    pair_trajectories: Dict[str, list[str]] = {}
    for trajectory_id, pair_id in pair_by_trajectory.items():
        pair_trajectories.setdefault(pair_id, []).append(trajectory_id)
    if (
        len(positions) != 240
        or len(treatments) != 120
        or len(pair_trajectories) != 120
        or any(len(values) != 2 for values in pair_trajectories.values())
        or any(
            sum(value in treatments for value in values) != 1
            for values in pair_trajectories.values()
        )
    ):
        raise ValueError("stored alert twins are incomplete")
    policy_results = {}
    for name, prediction in (
        ("candidate", candidate),
        ("persistence", persistence),
    ):
        scores = np.mean(
            np.square(prediction - observed), axis=(1, 2, 3)
        )
        policy_results[name] = _independent_policy_metrics(
            scores=scores,
            positions=positions,
            pair_by_trajectory=pair_by_trajectory,
            treatment_ids=treatments,
            roles=roles,
            actions=future_actions,
            transitions=transition_indices,
            action_kind_by_pair=action_kind_by_pair,
        )
    candidate_result = policy_results["candidate"]
    persistence_result = policy_results["persistence"]
    family_rates = candidate_result[
        "detection_rate_by_action_family"
    ]
    if not isinstance(family_rates, Mapping):
        raise ValueError("candidate family rates are absent")
    gates = {
        "role_coverage_exact": (
            candidate_result["evaluation_control_count"] == 60
            and candidate_result["evaluation_treatment_count"] == 60
            and candidate_result["reference_control_count"] == 30
            and candidate_result[
                "threshold_calibration_control_count"
            ]
            == 30
        ),
        "candidate_control_false_alarm_at_most_5_percent": (
            float(candidate_result["control_false_alarm_rate"]) <= 0.05
        ),
        "candidate_rejects_15_percent_false_alarm_null": (
            float(candidate_result["false_alarm_exact_p_value"]) <= 0.05
        ),
        "candidate_pre_onset_alert_at_most_5_percent": (
            float(candidate_result["treatment_pre_onset_alert_rate"])
            <= 0.05
        ),
        "candidate_detection_at_least_90_percent": (
            float(candidate_result["treatment_detection_rate"]) >= 0.90
        ),
        "candidate_within_active_detection_at_least_85_percent": (
            float(
                candidate_result[
                    "within_active_intervention_detection_rate"
                ]
            )
            >= 0.85
        ),
        "candidate_median_delay_at_most_8": (
            candidate_result["median_detection_delay_transitions"]
            is not None
            and float(
                candidate_result[
                    "median_detection_delay_transitions"
                ]
            )
            <= 8.0
        ),
        "every_action_family_detection_at_least_75_percent": all(
            float(value) >= 0.75 for value in family_rates.values()
        ),
        "candidate_detection_advantage_over_persistence_at_least_10_points": (
            float(candidate_result["treatment_detection_rate"])
            - float(persistence_result["treatment_detection_rate"])
            >= 0.10 - 1e-12
        ),
        "candidate_within_active_advantage_over_persistence_at_least_10_points": (
            float(
                candidate_result[
                    "within_active_intervention_detection_rate"
                ]
            )
            - float(
                persistence_result[
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
            "cusum_drift": math.log(4.0),
            "alpha": 0.05,
            "threshold_unit": "control_trajectory_cusum_maximum",
            "crossing": "strictly_greater",
            "alert_latched_once_per_run": True,
        },
        "role_counts": {
            role: tuple(roles.values()).count(role)
            for role in (
                "score_reference",
                "threshold_calibration",
                "sealed_evaluation",
            )
        },
        "candidate": candidate_result,
        "persistence": persistence_result,
        "gates": gates,
    }


def _independent_roles(
    pair_ids: Sequence[str],
    actions: Mapping[str, str],
    topologies: Mapping[str, str],
) -> Mapping[str, str]:
    action_names = (
        "api_rejection",
        "postgres_lock",
        "redis_dequeue_delay",
        "redis_enqueue_delay",
        "worker_pause",
    )
    topology_names = ("workers-1", "workers-2", "workers-3")
    if (
        len(pair_ids) != 120
        or len(set(pair_ids)) != 120
        or set(actions) != set(pair_ids)
        or set(topologies) != set(pair_ids)
    ):
        raise ValueError("independent role metadata is incomplete")
    roles: Dict[str, str] = {}
    for action in action_names:
        for topology in topology_names:
            cell = sorted(
                pair_id
                for pair_id in pair_ids
                if actions[pair_id] == action
                and topologies[pair_id] == topology
            )
            if len(cell) != 8:
                raise ValueError("independent role cell is incomplete")
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
    return roles


def _independent_policy_metrics(
    *,
    scores: np.ndarray,
    positions: Mapping[str, Sequence[int]],
    pair_by_trajectory: Mapping[str, str],
    treatment_ids: set[str],
    roles: Mapping[str, str],
    actions: np.ndarray,
    transitions: np.ndarray,
    action_kind_by_pair: Mapping[str, str],
) -> Mapping[str, Any]:
    controls = set(positions) - treatment_ids
    reference_ids = sorted(
        trajectory_id
        for trajectory_id in controls
        if roles[pair_by_trajectory[trajectory_id]]
        == "score_reference"
    )
    calibration_ids = sorted(
        trajectory_id
        for trajectory_id in controls
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
    increments = np.asarray(
        [
            -math.log(
                (1.0 + float(np.count_nonzero(reference >= score)))
                / (len(reference) + 1.0)
            )
            - math.log(4.0)
            for score in scores
        ],
        dtype=np.float64,
    )
    traces = {}
    for trajectory_id, indices in positions.items():
        cumulative = 0.0
        values = []
        for index in indices:
            cumulative = max(
                0.0, cumulative + float(increments[index])
            )
            values.append(cumulative)
        traces[trajectory_id] = np.asarray(values, dtype=np.float64)
    maxima = np.asarray(
        [
            float(np.max(traces[trajectory_id]))
            for trajectory_id in calibration_ids
        ],
        dtype=np.float64,
    )
    rank = int(math.ceil((len(maxima) + 1) * 0.95))
    threshold = (
        math.inf
        if rank > len(maxima)
        else float(np.sort(maxima)[rank - 1])
    )
    rows = []
    for trajectory_id in evaluation_ids:
        indices = positions[trajectory_id]
        crossings = [
            int(transitions[index]) + 1
            for index, value in zip(
                indices, traces[trajectory_id]
            )
            if value > threshold
        ]
        first_alert = crossings[0] if crossings else None
        active = [
            index
            for index in indices
            if np.any(actions[index, 0, :, 1] > 0.5)
        ]
        onset = (
            min(int(transitions[index]) for index in active)
            if active
            else None
        )
        stop = (
            max(int(transitions[index]) for index in active)
            if active
            else None
        )
        is_treatment = trajectory_id in treatment_ids
        detected = bool(
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
                    action_kind_by_pair[
                        pair_by_trajectory[trajectory_id]
                    ]
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
                "detected": detected,
                "detected_while_active": bool(
                    detected
                    and stop is not None
                    and first_alert is not None
                    and first_alert <= stop
                ),
                "detection_delay_transitions": (
                    int(first_alert - onset)
                    if detected
                    and first_alert is not None
                    and onset is not None
                    else None
                ),
            }
        )
    control_rows = [row for row in rows if not row["is_treatment"]]
    treatment_rows = [row for row in rows if row["is_treatment"]]
    detected_rows = [row for row in treatment_rows if row["detected"]]
    false_alarms = sum(
        row["first_alert_transition"] is not None
        for row in control_rows
    )
    delays = [
        int(row["detection_delay_transitions"])
        for row in detected_rows
    ]
    family_rates = {
        action: float(
            np.mean(
                [
                    bool(row["detected"])
                    for row in treatment_rows
                    if row["action_kind"] == action
                ]
            )
        )
        for action in (
            "api_rejection",
            "postgres_lock",
            "redis_dequeue_delay",
            "redis_enqueue_delay",
            "worker_pause",
        )
    }
    return {
        "reference_control_count": len(reference_ids),
        "reference_control_window_count": len(reference),
        "threshold_calibration_control_count": len(calibration_ids),
        "threshold": threshold,
        "threshold_calibration_exceedance_count": int(
            np.count_nonzero(maxima > threshold)
        ),
        "evaluation_control_count": len(control_rows),
        "evaluation_treatment_count": len(treatment_rows),
        "control_false_alarm_count": false_alarms,
        "control_false_alarm_rate": false_alarms / len(control_rows),
        "false_alarm_exact_p_value": _binomial_lower_tail(
            false_alarms, len(control_rows), 0.15
        ),
        "treatment_pre_onset_alert_rate": float(
            np.mean(
                [
                    bool(row["pre_onset_alert"])
                    for row in treatment_rows
                ]
            )
        ),
        "treatment_detection_rate": (
            len(detected_rows) / len(treatment_rows)
        ),
        "within_active_intervention_detection_rate": float(
            np.mean(
                [
                    bool(row["detected_while_active"])
                    for row in treatment_rows
                ]
            )
        ),
        "median_detection_delay_transitions": (
            float(np.median(delays)) if delays else None
        ),
        "detection_rate_by_action_family": family_rates,
        "trajectory_rows": rows,
    }


def _binomial_lower_tail(
    successes: int, trials: int, probability: float
) -> float:
    return float(
        sum(
            math.comb(trials, count)
            * probability**count
            * (1.0 - probability) ** (trials - count)
            for count in range(successes + 1)
        )
    )


def _read_object(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(arguments: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--predictive-confirmation-assessment",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--source-artifact-manifest", type=Path, required=True
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--expected-contract-sha256", required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument(
        "--expected-predictive-confirmation-assessment-sha256",
        required=True,
    )
    parser.add_argument(
        "--expected-source-artifact-manifest-sha256", required=True
    )
    parser.add_argument(
        "--expected-prediction-manifest-sha256", required=True
    )
    parsed = parser.parse_args(arguments)
    result = assess_stored_run_aware_alert_confirmation(
        contract_path=parsed.contract,
        model_path=parsed.model,
        predictive_confirmation_assessment=(
            parsed.predictive_confirmation_assessment
        ),
        source_artifact_manifest=parsed.source_artifact_manifest,
        predictions_directory=parsed.predictions,
        expected_contract_sha256=parsed.expected_contract_sha256,
        expected_model_sha256=parsed.expected_model_sha256,
        expected_predictive_confirmation_assessment_sha256=(
            parsed.expected_predictive_confirmation_assessment_sha256
        ),
        expected_source_artifact_manifest_sha256=(
            parsed.expected_source_artifact_manifest_sha256
        ),
        expected_prediction_manifest_sha256=(
            parsed.expected_prediction_manifest_sha256
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") == "confirmed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
