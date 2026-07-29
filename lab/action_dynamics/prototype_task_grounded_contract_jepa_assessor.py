#!/usr/bin/env python3
"""Independent stored-evidence assessor for task-grounded Contract-JEPA."""

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from lab.action_dynamics import (
    prototype_pair_effect_jepa_assessor as shared,
)


CELL_NAMES = (
    "task_grounded_contract_jepa",
    "supervised_task_contract",
    "ungrounded_contract_jepa",
)


def assess_stored_bundle(directory: Path) -> Mapping[str, Any]:
    """Recompute every Contract-JEPA gate from stored evidence."""

    root = Path(directory)
    metadata = _read_json(root / "evidence-metadata.json")
    with np.load(root / "evidence.npz", allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    finite = all(
        np.all(np.isfinite(value))
        for value in arrays.values()
        if value.dtype.kind in ("f", "i", "u")
    )
    ownership = np.asarray(
        metadata["ownership_mask"], dtype=np.bool_
    )
    role_results = {}
    for role in ("selection", "iid_evaluation", "transfer_evaluation"):
        role_meta = dict(metadata["roles"][role])
        pair_ids = tuple(str(value) for value in role_meta["pair_ids"])
        trajectory_ids = tuple(
            str(value) for value in role_meta["trajectory_ids"]
        )
        transitions = np.asarray(
            role_meta["transition_indices"], dtype=np.int64
        )
        target = arrays[f"target__{role}"]
        actions = arrays[f"actions__{role}"]
        scores = {}
        pair_errors = {}
        for name in ("raw", *CELL_NAMES):
            prediction = arrays[f"prediction__{name}__{role}"]
            scores[name] = shared._forecast_scores(
                prediction,
                target,
                actions,
                pair_ids,
                trajectory_ids,
                transitions,
                metadata["graph"],
            )
            pair_errors[name] = shared._downstream_pair_errors(
                prediction,
                target,
                actions,
                pair_ids,
                trajectory_ids,
                transitions,
                metadata["graph"],
            )
        paired_effect_mse = {
            name: float(
                np.mean(
                    np.square(
                        arrays[f"paired_prediction__{name}__{role}"]
                        - arrays[f"paired_target__{role}"]
                    )[..., ownership]
                )
            )
            for name in CELL_NAMES
        }
        witness_mse = {
            name: float(
                np.mean(
                    np.square(
                        arrays[f"paired_witness__{name}__{role}"]
                        - arrays[f"paired_witness_target__{role}"]
                    )
                )
            )
            for name in CELL_NAMES
        }
        role_results[role] = {
            "scores": scores,
            "pair_errors": pair_errors,
            "paired_effect_mse": paired_effect_mse,
            "effect_score_mse": witness_mse,
        }
    attribution = shared._attribution_scores(
        arrays["query_prediction__task_grounded_contract_jepa"],
        arrays["query_observed_future"],
        tuple(str(value) for value in metadata["queries"]["candidate_ids"]),
        tuple(
            str(value)
            for value in metadata["queries"]["candidate_action_kinds"]
        ),
        tuple(
            str(value)
            for value in metadata["queries"][
                "candidate_target_entities"
            ]
        ),
        tuple(
            str(value)
            for value in metadata["queries"]["expected_action_kinds"]
        ),
        tuple(
            str(value)
            for value in metadata["queries"]["expected_target_entities"]
        ),
        ownership,
    )
    action_sanity = shared._action_sanity_scores(
        {
            variant: arrays[
                "action_sanity__task_grounded_contract_jepa__"
                f"{variant}"
            ]
            for variant in ("correct", "no_action", "shuffled")
        },
        arrays["target__transfer_evaluation"],
        arrays["actions__transfer_evaluation"],
        tuple(
            str(value)
            for value in metadata["roles"]["transfer_evaluation"][
                "pair_ids"
            ]
        ),
        ownership,
    )
    detection = {
        name: _calibrated_detection(
            calibration_scores=arrays[f"witness__{name}__calibration"],
            calibration_metadata=metadata["roles"]["calibration"],
            calibration_actions=arrays["actions__calibration"],
            evaluation_scores=arrays[
                f"witness__{name}__transfer_evaluation"
            ],
            evaluation_metadata=metadata["roles"][
                "transfer_evaluation"
            ],
            evaluation_actions=arrays[
                "actions__transfer_evaluation"
            ],
        )
        for name in CELL_NAMES
    }
    transfer = role_results["transfer_evaluation"]
    selection = role_results["selection"]
    candidate = transfer["scores"]["task_grounded_contract_jepa"]
    raw = transfer["scores"]["raw"]
    supervised = transfer["scores"]["supervised_task_contract"]
    ungrounded = transfer["scores"]["ungrounded_contract_jepa"]
    candidate_pair = transfer["pair_errors"][
        "task_grounded_contract_jepa"
    ]
    pair_win_fractions = {}
    for control in (
        "supervised_task_contract",
        "ungrounded_contract_jepa",
    ):
        control_pair = transfer["pair_errors"][control]
        common = sorted(set(candidate_pair) & set(control_pair))
        pair_win_fractions[control] = float(
            np.mean(
                [
                    candidate_pair[pair] < control_pair[pair]
                    for pair in common
                ]
            )
        )
    parameter_counts = dict(metadata["parameter_counts"])
    training_counts = {
        int(dict(value)["training"])
        for value in parameter_counts.values()
    }
    inference_counts = {
        int(dict(value)["inference"])
        for value in parameter_counts.values()
    }
    correction = arrays[
        "correction__task_grounded_contract_jepa__transfer_evaluation"
    ]
    bound = arrays["correction_bound__task_grounded_contract_jepa"]
    safety = {
        "all_evidence_is_finite": finite,
        "capacity_is_matched": (
            len(training_counts) == 1 and len(inference_counts) == 1
        ),
        "raw_hash_unchanged": bool(metadata["raw_hash_unchanged"]),
        "gain_zero_is_exact_raw": bool(metadata["gain_zero_is_exact_raw"]),
        "public_inference_is_causal": bool(
            metadata["public_causality"]
        ),
        "correction_respects_trust_bound": bool(
            np.all(np.abs(correction) <= bound[None, None] + 1e-7)
        ),
        "restoration_max_abs_at_most_1e_6": float(
            metadata["restoration_max_abs"]
        )
        <= 1e-6,
        "transfer_overall_is_raw_safe": float(candidate["overall_mse"])
        <= 1.05 * float(raw["overall_mse"]),
        "transfer_action_overlap_is_raw_safe": float(
            candidate["action_overlap_mse"]
        )
        <= 1.05 * float(raw["action_overlap_mse"]),
        "action_and_target_hit_at_1_at_least_0_95": float(
            attribution["action_and_target_hit_at_1"]
        )
        >= 0.95,
        "no_action_specificity_is_one": float(
            attribution["no_action_specificity"]
        )
        == 1.0,
        "action_sanity_at_least_0_80": float(
            action_sanity["correct_action_beats_both_fraction"]
        )
        >= 0.80,
        "candidate_bundle_at_most_16_mib": int(
            metadata["candidate_bundle_bytes"]
        )
        <= 16 * 1024 * 1024,
        "batch_one_latency_recorded": float(
            metadata["latency"]["task_grounded_contract_jepa"][
                "median_ms"
            ]
        )
        >= 0.0,
    }
    mechanism = {
        "selection_paired_effect_beats_ungrounded_by_10_percent": float(
            selection["paired_effect_mse"][
                "task_grounded_contract_jepa"
            ]
        )
        <= 0.90
        * float(
            selection["paired_effect_mse"]["ungrounded_contract_jepa"]
        ),
        "transfer_effect_score_beats_ungrounded_by_10_percent": float(
            transfer["effect_score_mse"][
                "task_grounded_contract_jepa"
            ]
        )
        <= 0.90
        * float(
            transfer["effect_score_mse"]["ungrounded_contract_jepa"]
        ),
    }
    selected_gains = {
        str(name): float(value)
        for name, value in dict(metadata["selected_gains"]).items()
    }
    value = {
        "selected_nonzero_gain": (
            selected_gains["task_grounded_contract_jepa"] > 0.0
        ),
        "transfer_downstream_beats_raw_by_10_percent": float(
            candidate["downstream_effect_mse"]
        )
        <= 0.90 * float(raw["downstream_effect_mse"]),
        "transfer_downstream_beats_supervised_by_10_percent": float(
            candidate["downstream_effect_mse"]
        )
        <= 0.90 * float(supervised["downstream_effect_mse"]),
        "transfer_downstream_beats_ungrounded_by_10_percent": float(
            candidate["downstream_effect_mse"]
        )
        <= 0.90 * float(ungrounded["downstream_effect_mse"]),
        "pair_wins_supervised_at_least_0_60": (
            pair_win_fractions["supervised_task_contract"] >= 0.60
        ),
        "pair_wins_ungrounded_at_least_0_60": (
            pair_win_fractions["ungrounded_contract_jepa"] >= 0.60
        ),
        "selection_downstream_beats_both_controls": (
            float(
                selection["scores"]["task_grounded_contract_jepa"][
                    "downstream_effect_mse"
                ]
            )
            < float(
                selection["scores"]["supervised_task_contract"][
                    "downstream_effect_mse"
                ]
            )
            and float(
                selection["scores"]["task_grounded_contract_jepa"][
                    "downstream_effect_mse"
                ]
            )
            < float(
                selection["scores"]["ungrounded_contract_jepa"][
                    "downstream_effect_mse"
                ]
            )
        ),
    }
    candidate_detection = detection["task_grounded_contract_jepa"]
    delay = candidate_detection["median_post_onset_delay_transitions"]
    witness = {
        "control_false_alarm_at_most_0_05": float(
            candidate_detection["control_trajectory_false_alarm_rate"]
        )
        <= 0.05,
        "treatment_detection_at_least_0_80": float(
            candidate_detection["treatment_trajectory_detection_rate"]
        )
        >= 0.80,
        "median_delay_at_most_10": (
            delay is not None and float(delay) <= 10.0
        ),
    }
    interpretable = bool(metadata["interpretable"])
    gates_pass = (
        all(safety.values())
        and all(mechanism.values())
        and all(value.values())
        and all(witness.values())
    )
    passed = interpretable and gates_pass
    if not interpretable:
        decision = (
            "non_interpretable_task_grounded_contract_jepa_smoke"
        )
    elif passed:
        decision = "advance_contract_jepa_to_fixed_seed_robustness"
    else:
        decision = "reject_task_grounded_contract_jepa_recipe"
    return {
        "schema_version": 1,
        "kind": "task_grounded_contract_jepa_independent_assessment",
        "eligible_for_advance": interpretable,
        "roles": role_results,
        "selected_gains": selected_gains,
        "gain_curves": metadata["gain_curves"],
        "attribution": attribution,
        "action_sanity": action_sanity,
        "detection": detection,
        "pair_win_fractions": pair_win_fractions,
        "safety_gates": safety,
        "mechanism_gates": mechanism,
        "value_gates": value,
        "witness_gates": witness,
        "safety_passed": all(safety.values()),
        "mechanism_passed": all(mechanism.values()),
        "value_passed": all(value.values()),
        "witness_passed": all(witness.values()),
        "passed": passed,
        "decision": decision,
    }


def verify_stored_assessment(directory: Path) -> None:
    root = Path(directory)
    if shared._canonical_json(
        _read_json(root / "assessment.json")
    ) != shared._canonical_json(assess_stored_bundle(root)):
        raise ValueError("stored Contract-JEPA assessment differs")


def verify_artifact_manifest(directory: Path) -> None:
    root = Path(directory)
    manifest = _read_json(root / "artifact-manifest.json")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "task_grounded_contract_jepa_manifest"
    ):
        raise ValueError("Contract-JEPA artifact manifest is invalid")
    expected = dict(manifest["sha256"])
    actual = {
        path.relative_to(root).as_posix(): _file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "artifact-manifest.json"
    }
    if expected != actual:
        raise ValueError("Contract-JEPA artifact manifest differs")


def _calibrated_detection(
    *,
    calibration_scores: np.ndarray,
    calibration_metadata: Mapping[str, Any],
    calibration_actions: np.ndarray,
    evaluation_scores: np.ndarray,
    evaluation_metadata: Mapping[str, Any],
    evaluation_actions: np.ndarray,
) -> Mapping[str, Any]:
    calibration_rows = _trajectory_rows(
        calibration_scores,
        calibration_metadata,
        calibration_actions,
        threshold=None,
    )
    control_maxima = [
        float(row["maximum_score"])
        for row in calibration_rows
        if not bool(row["is_treatment"])
    ]
    threshold = float(
        np.quantile(control_maxima, 0.95, method="higher")
    )
    rows = _trajectory_rows(
        evaluation_scores,
        evaluation_metadata,
        evaluation_actions,
        threshold=threshold,
    )
    controls = [row for row in rows if not bool(row["is_treatment"])]
    treatments = [row for row in rows if bool(row["is_treatment"])]
    detected = [
        row
        for row in treatments
        if row["post_onset_detection_transition"] is not None
    ]
    delays = [
        int(row["post_onset_detection_transition"])
        - int(row["onset_transition"])
        for row in detected
    ]
    return {
        "threshold": threshold,
        "calibration_control_trajectory_count": len(control_maxima),
        "control_trajectory_false_alarm_rate": float(
            np.mean([bool(row["any_alarm"]) for row in controls])
        ),
        "treatment_trajectory_detection_rate": float(
            len(detected) / len(treatments)
        ),
        "median_post_onset_delay_transitions": (
            float(np.median(delays)) if delays else None
        ),
        "trajectory_rows": rows,
    }


def _trajectory_rows(
    scores: np.ndarray,
    metadata: Mapping[str, Any],
    actions: np.ndarray,
    *,
    threshold: Any,
) -> list[Mapping[str, Any]]:
    trajectory_ids = tuple(
        str(value) for value in metadata["trajectory_ids"]
    )
    transitions = np.asarray(
        metadata["transition_indices"], dtype=np.int64
    )
    groups = {}
    for row, trajectory in enumerate(trajectory_ids):
        groups.setdefault(trajectory, []).append(row)
    result = []
    for trajectory, positions in groups.items():
        local = np.asarray(positions, dtype=np.int64)
        onset_rows = local[
            np.any(actions[local, 0, :, 1] > 0.5, axis=1)
        ]
        onset = (
            int(np.min(transitions[onset_rows]))
            if len(onset_rows)
            else None
        )
        local_scores = scores[local]
        event_times = (
            transitions[local, None]
            + np.arange(1, scores.shape[1] + 1)[None]
        )
        if threshold is None:
            alarms = np.zeros_like(local_scores, dtype=np.bool_)
        else:
            alarms = local_scores > float(threshold)
        eligible = (
            event_times[alarms & (event_times >= onset)]
            if onset is not None
            else np.asarray([], dtype=np.int64)
        )
        result.append(
            {
                "trajectory_id": trajectory,
                "is_treatment": onset is not None,
                "onset_transition": onset,
                "maximum_score": float(np.max(local_scores)),
                "any_alarm": bool(np.any(alarms)),
                "post_onset_detection_transition": (
                    int(np.min(eligible)) if len(eligible) else None
                ),
            }
        )
    return result


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

