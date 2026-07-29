#!/usr/bin/env python3
"""Independent stored-evidence assessor for Error-Certificate-JEPA."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
from numpy.typing import NDArray


CELL_NAMES = (
    "jepa_error_certificate",
    "raw_error_certificate",
    "deranged_jepa_certificate",
)
CONTROL_NAMES = (
    "jepa_error_certificate",
    "raw_error_certificate",
    "deranged_jepa_certificate",
    "constant_conformal",
)


def assess_stored_bundle(directory: Path) -> Mapping[str, Any]:
    """Recompute every frozen gate from stored arrays and role metadata."""

    root = Path(directory)
    metadata = _read_json(root / "evidence-metadata.json")
    with np.load(root / "evidence.npz", allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    finite = all(
        np.all(np.isfinite(value))
        for value in arrays.values()
        if value.dtype.kind in ("f", "i", "u")
    )
    role_results: Dict[str, Mapping[str, Any]] = {}
    for role in (
        "selection",
        "calibration",
        "iid_evaluation",
        "transfer_evaluation",
    ):
        target = arrays[f"realized_error__{role}"]
        actions = arrays[f"actions__{role}"]
        role_meta = dict(metadata["roles"][role])
        cells = {}
        for name in CONTROL_NAMES:
            bound = arrays[f"bound__{name}__{role}"]
            cells[name] = _certificate_scores(
                target, bound, actions, role_meta
            )
            if name in CELL_NAMES:
                cells[name] = {
                    **cells[name],
                    "unadjusted_pinball": _pinball_loss(
                        arrays[f"unadjusted__{name}__{role}"],
                        target,
                        quantile=0.95,
                    ),
                }
        role_results[role] = {"certificates": cells}

    recomputed_adjustments = {}
    calibrated_values_match = {}
    calibration_target = arrays["realized_error__calibration"]
    calibration_actions = arrays["actions__calibration"]
    calibration_meta = dict(metadata["roles"]["calibration"])
    for name in CELL_NAMES:
        unadjusted = arrays[f"unadjusted__{name}__calibration"]
        violations = calibration_target - unadjusted
        maxima = _control_trajectory_maxima(
            violations, calibration_actions, calibration_meta
        )
        adjustment = max(
            0.0,
            float(np.quantile(maxima, 0.95, method="higher")),
        )
        recomputed_adjustments[name] = adjustment
        expected = {
            role: arrays[f"unadjusted__{name}__{role}"] + adjustment
            for role in (
                "selection",
                "calibration",
                "iid_evaluation",
                "transfer_evaluation",
            )
        }
        calibrated_values_match[name] = (
            abs(
                adjustment
                - float(dict(metadata["calibration_adjustments"])[name])
            )
            <= 1e-7
            and all(
                np.allclose(
                    arrays[f"bound__{name}__{role}"],
                    values,
                    atol=1e-6,
                    rtol=0.0,
                )
                for role, values in expected.items()
            )
        )
    constant = float(
        np.quantile(
            _control_trajectory_maxima(
                calibration_target,
                calibration_actions,
                calibration_meta,
            ),
            0.95,
            method="higher",
        )
    )
    constant_matches = (
        abs(constant - float(metadata["constant_conformal_bound"]))
        <= 1e-7
        and all(
            np.allclose(
                arrays[f"bound__constant_conformal__{role}"],
                constant,
                atol=1e-7,
                rtol=0.0,
            )
            for role in role_results
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
    raw_mean = arrays["raw_mean__transfer_evaluation"]
    raw_variance = arrays["raw_variance__transfer_evaluation"]
    raw_exact = all(
        np.array_equal(
            raw_mean,
            arrays[f"wrapper_raw_mean__{name}__transfer_evaluation"],
        )
        and np.array_equal(
            raw_variance,
            arrays[
                f"wrapper_raw_variance__{name}__transfer_evaluation"
            ],
        )
        for name in CELL_NAMES
    )
    nonnegative = all(
        np.all(arrays[f"unadjusted__{name}__{role}"] >= 0.0)
        and np.all(arrays[f"bound__{name}__{role}"] >= 0.0)
        for name in CELL_NAMES
        for role in role_results
    )
    role_audit = dict(metadata["role_use_audit"])
    role_use_valid = (
        role_audit.get("checkpoint_selection") == "selection"
        and role_audit.get("calibration_adjustment") == "calibration"
        and role_audit.get("evaluation_roles")
        == ["iid_evaluation", "transfer_evaluation"]
    )
    safety = {
        "all_evidence_is_finite": finite,
        "capacity_is_matched": (
            len(training_counts) == 1 and len(inference_counts) == 1
        ),
        "raw_hash_unchanged": bool(metadata["raw_hash_unchanged"]),
        "every_cell_returns_exact_raw_distribution": raw_exact,
        "public_inference_is_causal": bool(
            metadata["public_causality"]
        ),
        "restoration_max_abs_at_most_1e_6": float(
            metadata["restoration_max_abs"]
        )
        <= 1e-6,
        "restored_alert_decisions_match": bool(
            metadata["restored_alert_decisions_match"]
        ),
        "bounds_are_finite_and_nonnegative": finite and nonnegative,
        "calibration_recomputes_exactly": all(
            calibrated_values_match.values()
        )
        and constant_matches,
        "selection_and_calibration_roles_are_isolated": role_use_valid,
        "candidate_bundle_at_most_16_mib": int(
            metadata["candidate_bundle_bytes"]
        )
        <= 16 * 1024 * 1024,
        "batch_one_latency_recorded": float(
            dict(metadata["latency"])["median_ms"]
        )
        >= 0.0,
    }
    transfer = role_results["transfer_evaluation"]["certificates"]
    selection = role_results["selection"]["certificates"]
    candidate = transfer["jepa_error_certificate"]
    raw_control = transfer["raw_error_certificate"]
    deranged = transfer["deranged_jepa_certificate"]
    direct = transfer["constant_conformal"]
    coverage = {
        "transfer_control_point_coverage_at_least_0_95": float(
            candidate["control_point_coverage"]
        )
        >= 0.95,
        "transfer_control_simultaneous_coverage_is_one": float(
            candidate["control_simultaneous_coverage"]
        )
        == 1.0,
        "control_trajectory_false_alarm_at_most_0_05": float(
            candidate["control_trajectory_false_alarm_rate"]
        )
        <= 0.05,
        "no_evaluation_data_widens_bounds": role_use_valid,
    }
    both_mechanism_cells_cover = (
        float(candidate["control_point_coverage"]) >= 0.95
        and float(candidate["control_simultaneous_coverage"]) == 1.0
        and float(deranged["control_point_coverage"]) >= 0.95
        and float(deranged["control_simultaneous_coverage"]) == 1.0
    )
    mechanism = {
        "selection_pinball_beats_deranged_by_10_percent": float(
            selection["jepa_error_certificate"][
                "unadjusted_pinball"
            ]
        )
        <= 0.90
        * float(
            selection["deranged_jepa_certificate"][
                "unadjusted_pinball"
            ]
        ),
        "transfer_bound_beats_deranged_by_10_percent_with_coverage": (
            both_mechanism_cells_cover
            and float(candidate["control_mean_bound"])
            <= 0.90 * float(deranged["control_mean_bound"])
        ),
    }
    candidate_delay = candidate[
        "median_post_onset_delay_transitions"
    ]
    same_false_alarm_ceiling = (
        float(candidate["control_trajectory_false_alarm_rate"]) <= 0.05
        and float(raw_control["control_trajectory_false_alarm_rate"])
        <= 0.05
    )
    value = {
        "transfer_bound_beats_raw_learned_by_10_percent": float(
            candidate["control_mean_bound"]
        )
        <= 0.90 * float(raw_control["control_mean_bound"]),
        "transfer_bound_beats_constant_by_10_percent": float(
            candidate["control_mean_bound"]
        )
        <= 0.90 * float(direct["control_mean_bound"]),
        "transfer_treatment_detection_at_least_0_80": float(
            candidate["treatment_trajectory_detection_rate"]
        )
        >= 0.80,
        "median_post_onset_delay_at_most_10": (
            candidate_delay is not None
            and float(candidate_delay) <= 10.0
        ),
        "detection_beats_raw_by_10_points_at_same_false_alarm_ceiling": (
            same_false_alarm_ceiling
            and float(
                candidate["treatment_trajectory_detection_rate"]
            )
            >= float(
                raw_control["treatment_trajectory_detection_rate"]
            )
            + 0.10
        ),
    }
    interpretable = bool(metadata["interpretable"])
    gates_pass = (
        all(safety.values())
        and all(coverage.values())
        and all(mechanism.values())
        and all(value.values())
    )
    passed = interpretable and gates_pass
    if not interpretable:
        decision = "non_interpretable_error_certificate_jepa_smoke"
    elif passed:
        decision = "advance_error_certificate_jepa_to_robustness"
    else:
        decision = "reject_error_certificate_jepa_recipe"
    return {
        "schema_version": 1,
        "kind": "error_certificate_jepa_independent_assessment",
        "eligible_for_advance": interpretable,
        "roles": role_results,
        "recomputed_calibration_adjustments": (
            recomputed_adjustments
        ),
        "recomputed_constant_conformal_bound": constant,
        "safety_gates": safety,
        "coverage_gates": coverage,
        "mechanism_gates": mechanism,
        "value_gates": value,
        "safety_passed": all(safety.values()),
        "coverage_passed": all(coverage.values()),
        "mechanism_passed": all(mechanism.values()),
        "value_passed": all(value.values()),
        "passed": passed,
        "decision": decision,
    }


def verify_stored_assessment(directory: Path) -> None:
    root = Path(directory)
    if _canonical_json(
        _read_json(root / "assessment.json")
    ) != _canonical_json(assess_stored_bundle(root)):
        raise ValueError("stored Error-Certificate-JEPA assessment differs")


def verify_artifact_manifest(directory: Path) -> None:
    root = Path(directory)
    manifest = _read_json(root / "artifact-manifest.json")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind")
        != "error_certificate_jepa_manifest"
    ):
        raise ValueError(
            "Error-Certificate-JEPA artifact manifest is invalid"
        )
    expected = dict(manifest["sha256"])
    actual = {
        path.relative_to(root).as_posix(): _file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "artifact-manifest.json"
    }
    if expected != actual:
        raise ValueError(
            "Error-Certificate-JEPA artifact manifest differs"
        )


def _certificate_scores(
    target: NDArray[Any],
    bound: NDArray[Any],
    actions: NDArray[Any],
    metadata: Mapping[str, Any],
) -> Mapping[str, Any]:
    rows = _trajectory_rows(target, bound, actions, metadata)
    controls = [row for row in rows if not bool(row["is_treatment"])]
    treatments = [row for row in rows if bool(row["is_treatment"])]
    control_positions = _control_positions(actions, metadata)
    control_covered = target[control_positions] <= (
        bound[control_positions] + 1e-12
    )
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
    control_bounds = bound[control_positions]
    return {
        "control_point_coverage": float(np.mean(control_covered)),
        "control_simultaneous_coverage": float(
            np.mean([not bool(row["any_alarm"]) for row in controls])
        ),
        "control_trajectory_false_alarm_rate": float(
            np.mean([bool(row["any_alarm"]) for row in controls])
        ),
        "control_mean_bound": float(np.mean(control_bounds)),
        "control_p95_bound": float(
            np.quantile(control_bounds, 0.95)
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
    target: NDArray[Any],
    bound: NDArray[Any],
    actions: NDArray[Any],
    metadata: Mapping[str, Any],
) -> Sequence[Mapping[str, Any]]:
    trajectory_ids = tuple(
        str(value) for value in metadata["trajectory_ids"]
    )
    transitions = np.asarray(
        metadata["transition_indices"], dtype=np.int64
    )
    groups: Dict[str, list[int]] = {}
    for row, trajectory in enumerate(trajectory_ids):
        groups.setdefault(trajectory, []).append(row)
    result = []
    for trajectory, positions in sorted(groups.items()):
        local = np.asarray(positions, dtype=np.int64)
        onset_rows = local[
            np.any(actions[local, 0, :, 1] > 0.5, axis=1)
        ]
        onset: Optional[int] = (
            int(np.min(transitions[onset_rows]))
            if len(onset_rows)
            else None
        )
        alarms = target[local] > bound[local]
        event_times = transitions[local, None] + np.arange(
            1, target.shape[1] + 1
        )[None]
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
                "any_alarm": bool(np.any(alarms)),
                "post_onset_detection_transition": (
                    int(np.min(eligible)) if len(eligible) else None
                ),
            }
        )
    return result


def _control_positions(
    actions: NDArray[Any], metadata: Mapping[str, Any]
) -> NDArray[np.int64]:
    trajectory_ids = np.asarray(
        [str(value) for value in metadata["trajectory_ids"]]
    )
    controls = []
    for trajectory in sorted(set(trajectory_ids)):
        rows = np.flatnonzero(trajectory_ids == trajectory)
        if not np.any(actions[rows, ..., 1] > 0.5):
            controls.extend(rows.tolist())
    return np.asarray(controls, dtype=np.int64)


def _control_trajectory_maxima(
    values: NDArray[Any],
    actions: NDArray[Any],
    metadata: Mapping[str, Any],
) -> NDArray[np.float64]:
    positions = np.asarray(
        [str(value) for value in metadata["trajectory_ids"]]
    )
    maxima = []
    for trajectory in sorted(set(positions)):
        rows = np.flatnonzero(positions == trajectory)
        if not np.any(actions[rows, ..., 1] > 0.5):
            maxima.append(float(np.max(values[rows])))
    if len(maxima) < 2:
        raise ValueError("stored evidence has too few control trajectories")
    return np.asarray(maxima, dtype=np.float64)


def _pinball_loss(
    prediction: NDArray[Any],
    target: NDArray[Any],
    *,
    quantile: float,
) -> float:
    error = target - prediction
    return float(
        np.mean(np.maximum(quantile * error, (quantile - 1.0) * error))
    )


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


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
