#!/usr/bin/env python3
"""Independent stored-array assessment for the SD-JEPA alert tracer."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from quantis_core.edge_dynamics.hepa_jepa import (
    calibrate_probability_surface,
    fit_logit_calibrator,
    trajectory_alert_threshold,
)
from quantis_core.edge_dynamics.sd_jepa import SD_JEPA_SCORE_NAMES


ASSESSMENT_ROLES = (
    "calibration",
    "evaluation_iid",
    "evaluation_transfer",
)


def assess_stored_bundle(directory: Path) -> Mapping[str, Any]:
    """Recompute the complete decision without loading fitted models."""

    root = Path(directory)
    protocol = _read_json(root / "protocol.json")
    with np.load(root / "evidence.npz", allow_pickle=False) as arrays:
        labels = {
            role: arrays[f"labels__{role}"].astype(np.bool_)
            for role in ASSESSMENT_ROLES
        }
        scores = _read_float_arrays(arrays, "score")
        restored_scores = _read_float_arrays(arrays, "restored_score")
        stored_calibrated = _read_float_arrays(arrays, "calibrated")
        restored_calibrated = _read_float_arrays(
            arrays, "restored_calibrated"
        )
        stored_decisions = _read_bool_arrays(arrays, "alert_decision")
        restored_decisions = _read_bool_arrays(
            arrays, "restored_alert_decision"
        )
        calibration_protocol = dict(protocol["roles"]["calibration"])
        calibration_ids = tuple(
            str(value)
            for value in calibration_protocol["trajectory_ids"]
        )
        calibration_controls = tuple(
            str(value)
            for value in calibration_protocol[
                "control_trajectory_ids"
            ]
        )
        calibrations: Dict[str, Mapping[str, float]] = {}
        calibrated: Dict[str, Dict[str, NDArray[np.float64]]] = {
            role: {} for role in ASSESSMENT_ROLES
        }
        decisions: Dict[str, Dict[str, NDArray[np.bool_]]] = {
            role: {} for role in ASSESSMENT_ROLES
        }
        for name in SD_JEPA_SCORE_NAMES:
            slope, intercept, calibration_brier = fit_logit_calibrator(
                scores["calibration"][name][:, None],
                labels["calibration"][:, None],
            )
            calibration_values = calibrate_probability_surface(
                scores["calibration"][name][:, None],
                slope=slope,
                intercept=intercept,
            )
            threshold = trajectory_alert_threshold(
                calibration_values,
                calibration_ids,
                calibration_controls,
            )
            calibrations[name] = {
                "slope": slope,
                "intercept": intercept,
                "calibration_brier": calibration_brier,
                "alert_threshold": threshold,
            }
            for role in ASSESSMENT_ROLES:
                calibrated[role][name] = (
                    calibrate_probability_surface(
                        scores[role][name][:, None],
                        slope=slope,
                        intercept=intercept,
                    )[:, 0]
                )
                decisions[role][name] = np.asarray(
                    calibrated[role][name] >= threshold,
                    dtype=np.bool_,
                )

        risk_metrics = {
            role: {
                name: _binary_probability_metrics(
                    calibrated[role][name], labels[role]
                )
                for name in SD_JEPA_SCORE_NAMES
            }
            for role in ASSESSMENT_ROLES
        }
        localization = {}
        alert_metrics = {}
        for role in ASSESSMENT_ROLES:
            role_protocol = dict(protocol["roles"][role])
            trajectory_ids = tuple(
                str(value)
                for value in role_protocol["trajectory_ids"]
            )
            transition_indices = arrays[
                f"transition_indices__{role}"
            ].astype(np.int64)
            onsets = {
                str(key): (
                    None if value is None else int(value)
                )
                for key, value in dict(
                    role_protocol["trajectory_onsets"]
                ).items()
            }
            localization[role] = {
                name: _localization_metrics(
                    scores[role][name], labels[role], trajectory_ids
                )
                for name in SD_JEPA_SCORE_NAMES
            }
            alert_metrics[role] = {
                name: _trajectory_alert_metrics(
                    decisions=decisions[role][name],
                    trajectory_ids=trajectory_ids,
                    transition_indices=transition_indices,
                    onsets=onsets,
                )
                for name in SD_JEPA_SCORE_NAMES
            }

        progress = _progress_metrics(
            truth=arrays["progress_truth"],
            trajectory_ids=tuple(
                str(value)
                for value in arrays["progress_trajectory_ids"].astype(str)
            ),
            features={
                "sd_jepa_progression": arrays[
                    "progress_features__sd_jepa_progression"
                ],
                "sd_jepa_content": arrays[
                    "progress_features__sd_jepa_content"
                ],
                "lewm_unsplit_first_two": arrays[
                    "progress_features__lewm_unsplit_first_two"
                ],
            },
        )
        progression_geometry = _progression_geometry(
            arrays["progress_features__sd_jepa_progression"],
            tuple(
                str(value)
                for value in arrays["progress_trajectory_ids"].astype(str)
            ),
        )
        state_retention = _state_retention_metrics(
            truth=arrays["state_truth"],
            scale=arrays["state_scale"],
            varying=arrays["state_varying_mask"].astype(np.bool_),
            predictions={
                "sd_jepa_content": arrays[
                    "state_prediction__sd_jepa_content"
                ],
                "matched_pca": arrays[
                    "state_prediction__matched_pca"
                ],
            },
        )
        restoration = {
            "score_max_abs": _maximum_difference(
                scores, restored_scores
            ),
            "calibrated_max_abs": _maximum_difference(
                calibrated, restored_calibrated
            ),
            "stored_calibrated_max_abs": _maximum_difference(
                calibrated, stored_calibrated
            ),
            "decision_exact": _all_equal(
                decisions, restored_decisions
            ),
            "stored_decision_exact": _all_equal(
                decisions, stored_decisions
            ),
            "scene_representation_max_abs": float(
                np.max(
                    np.abs(
                        arrays["candidate_scene"]
                        - arrays["restored_candidate_scene"]
                    )
                )
            ),
            "entity_representation_max_abs": float(
                np.max(
                    np.abs(
                        arrays["candidate_entity"]
                        - arrays["restored_candidate_entity"]
                    )
                )
            ),
        }
        latency = {
            name: {
                "mean_milliseconds": float(
                    np.mean(arrays[f"latency_samples__{name}"])
                ),
                "p95_milliseconds": float(
                    np.quantile(
                        arrays[f"latency_samples__{name}"], 0.95
                    )
                ),
            }
            for name in SD_JEPA_SCORE_NAMES
        }
        causal_audit = {
            "history_counterfactual_changed": bool(
                not np.array_equal(
                    arrays["audit_histories"],
                    arrays["audit_counterfactual_histories"],
                )
            ),
            "history_output_changed": bool(
                not np.allclose(
                    arrays["audit_original_outputs"],
                    arrays["audit_history_counterfactual_outputs"],
                    atol=1e-12,
                    rtol=0.0,
                )
            ),
            "forbidden_counterfactual_changed": bool(
                not np.array_equal(
                    arrays["audit_forbidden"],
                    arrays["audit_counterfactual_forbidden"],
                )
            ),
            "forbidden_output_unchanged": bool(
                np.array_equal(
                    arrays["audit_original_outputs"],
                    arrays["audit_forbidden_counterfactual_outputs"],
                )
            ),
            "forbidden_keywords_rejected": bool(
                np.all(
                    arrays[
                        "audit_forbidden_keyword_rejections"
                    ].astype(np.bool_)
                )
            ),
        }
        finite_evidence = bool(
            all(
                array.dtype.kind in ("U", "S", "b", "i", "u")
                or np.all(np.isfinite(array))
                for array in (arrays[name] for name in arrays.files)
            )
        )

    training_counts = {
        str(key): int(value)
        for key, value in dict(
            protocol["training_parameter_counts"]
        ).items()
    }
    inference_counts = {
        str(key): int(value)
        for key, value in dict(
            protocol["inference_parameter_counts"]
        ).items()
    }
    capacity = {
        "training_parameter_counts": training_counts,
        "inference_parameter_counts": inference_counts,
        "all_training_equal": len(set(training_counts.values())) == 1,
        "all_inference_equal": len(set(inference_counts.values())) == 1,
    }
    candidate_bundle_bytes = sum(
        (root / name).stat().st_size
        for name in (
            "models.json",
            "calibrators.json",
            "state-probes.json",
            "event-definition.json",
        )
    )
    transfer_localization = localization["evaluation_transfer"]
    transfer_risk = risk_metrics["evaluation_transfer"]
    transfer_alert = alert_metrics["evaluation_transfer"]
    candidate = "sd_jepa_angle"
    references = tuple(
        name for name in SD_JEPA_SCORE_NAMES if name != candidate
    )
    candidate_detection = transfer_alert[candidate][
        "treatment_detection_rate"
    ]
    median_delay = transfer_alert[candidate][
        "median_post_onset_delay_transitions"
    ]
    gates = {
        "finite_evidence": finite_evidence,
        "restoration": bool(
            restoration["score_max_abs"] <= 1e-6
            and restoration["calibrated_max_abs"] <= 1e-6
            and restoration["stored_calibrated_max_abs"] <= 1e-6
            and restoration["scene_representation_max_abs"] <= 1e-6
            and restoration["entity_representation_max_abs"] <= 1e-6
            and restoration["decision_exact"]
            and restoration["stored_decision_exact"]
        ),
        "matched_capacity": bool(
            capacity["all_training_equal"]
            and capacity["all_inference_equal"]
        ),
        "state_safety": bool(
            state_retention["sd_jepa_content"]["aggregate_nrmse"]
            <= 1.05
            * state_retention["matched_pca"]["aggregate_nrmse"]
        ),
        "edge_size": candidate_bundle_bytes <= 16 * 1024 * 1024,
        "causal_input": all(causal_audit.values()),
        "progression_mechanism_lane": bool(
            transfer_localization[candidate]["pooled_auroc"]
            >= transfer_localization["sd_jepa_z_mse"]["pooled_auroc"]
            + 0.05
            and transfer_localization[candidate]["pooled_auroc"]
            >= transfer_localization["lewm_unsplit_angle"][
                "pooled_auroc"
            ]
            + 0.05
            and progress["sd_jepa_progression"]["pooled_r2"]
            >= progress["lewm_unsplit_first_two"]["pooled_r2"] + 0.10
            and progress["sd_jepa_progression"]["pooled_r2"]
            > progress["sd_jepa_content"]["pooled_r2"]
        ),
        "calibrated_alert_lane": bool(
            transfer_risk[candidate]["brier"]
            <= 0.95
            * min(transfer_risk[name]["brier"] for name in references)
            and transfer_alert[candidate][
                "control_trajectory_false_alarm_rate"
            ]
            <= 0.05
            and candidate_detection >= 0.80
            and median_delay is not None
            and float(median_delay) <= 10.0
            and all(
                candidate_detection
                >= transfer_alert[name]["treatment_detection_rate"]
                + 0.10
                for name in references
            )
        ),
    }
    safety = (
        "finite_evidence",
        "restoration",
        "matched_capacity",
        "state_safety",
        "edge_size",
        "causal_input",
    )
    interpretable = bool(protocol["interpretable"])
    passed = bool(
        interpretable
        and all(gates[name] for name in safety)
        and (
            gates["progression_mechanism_lane"]
            or gates["calibrated_alert_lane"]
        )
    )
    decision = (
        "non_interpretable_sd_jepa_smoke"
        if not interpretable
        else (
            "advance_sd_jepa_to_fixed_seed_robustness"
            if passed
            else "reject_sd_jepa_edge_recipe"
        )
    )
    return {
        "schema_version": 1,
        "kind": "sd_jepa_stored_array_assessment",
        "interpretable": interpretable,
        "eligible_for_advance": interpretable,
        "passed": passed,
        "decision": decision,
        "calibrations": calibrations,
        "risk_metrics": risk_metrics,
        "localization": localization,
        "alert_metrics": alert_metrics,
        "progress": progress,
        "progression_geometry": progression_geometry,
        "state_retention": state_retention,
        "restoration": restoration,
        "capacity": capacity,
        "candidate_bundle_bytes": candidate_bundle_bytes,
        "latency": latency,
        "peak_rss_bytes": int(protocol["peak_rss_bytes"]),
        "causal_audit": causal_audit,
        "gates": gates,
    }


def verify_stored_assessment(directory: Path) -> Mapping[str, Any]:
    root = Path(directory)
    recomputed = assess_stored_bundle(root)
    if (root / "assessment.json").read_text() != _pretty_json(recomputed):
        raise ValueError("stored SD-JEPA assessment does not recompute")
    return recomputed


def verify_artifact_manifest(directory: Path) -> None:
    root = Path(directory)
    payload = _read_json(root / "artifact-manifest.json")
    files = dict(payload["files"])
    observed = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name != "artifact-manifest.json"
    }
    if observed != set(files):
        raise ValueError("SD-JEPA artifact manifest file set differs")
    for relative, raw in files.items():
        expected = dict(raw)
        path = root / relative
        if (
            path.stat().st_size != int(expected["bytes"])
            or _file_sha256(path) != str(expected["sha256"])
        ):
            raise ValueError(
                f"SD-JEPA artifact manifest differs: {relative}"
            )


def _read_float_arrays(
    arrays: Any, prefix: str
) -> Mapping[str, Mapping[str, NDArray[np.float64]]]:
    return {
        role: {
            name: np.asarray(
                arrays[f"{prefix}__{role}__{name}"],
                dtype=np.float64,
            )
            for name in SD_JEPA_SCORE_NAMES
        }
        for role in ASSESSMENT_ROLES
    }


def _read_bool_arrays(
    arrays: Any, prefix: str
) -> Mapping[str, Mapping[str, NDArray[np.bool_]]]:
    return {
        role: {
            name: arrays[
                f"{prefix}__{role}__{name}"
            ].astype(np.bool_)
            for name in SD_JEPA_SCORE_NAMES
        }
        for role in ASSESSMENT_ROLES
    }


def _binary_probability_metrics(
    probabilities: NDArray[np.float64],
    labels: NDArray[np.bool_],
) -> Mapping[str, float]:
    values = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.bool_)
    if (
        values.shape != truth.shape
        or values.ndim != 1
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise ValueError("SD-JEPA probability evidence is invalid")
    bins = np.minimum((values * 10.0).astype(np.int64), 9)
    ece = 0.0
    for index in range(10):
        selected = bins == index
        if np.any(selected):
            ece += float(np.mean(selected)) * abs(
                float(np.mean(values[selected]))
                - float(np.mean(truth[selected]))
            )
    return {
        "brier": float(
            np.mean(np.square(values - truth.astype(np.float64)))
        ),
        "ece_10_equal_width_bins": ece,
        "positive_rate": float(np.mean(truth)),
        "auroc": _binary_auroc(values, truth),
    }


def _localization_metrics(
    scores: NDArray[np.float64],
    labels: NDArray[np.bool_],
    trajectory_ids: Tuple[str, ...],
) -> Mapping[str, Any]:
    ids = np.asarray(trajectory_ids, dtype=str)
    rows = {}
    for trajectory_id in sorted(set(trajectory_ids)):
        selected = ids == trajectory_id
        rows[trajectory_id] = _binary_auroc(
            scores[selected], labels[selected]
        )
    finite = [value for value in rows.values() if value is not None]
    return {
        "pooled_auroc": _binary_auroc(scores, labels),
        "per_trajectory_auroc": rows,
        "mean_per_trajectory_auroc": (
            None if not finite else float(np.mean(finite))
        ),
        "evaluable_trajectory_count": len(finite),
    }


def _binary_auroc(
    scores: NDArray[np.float64],
    labels: NDArray[np.bool_],
) -> Optional[float]:
    values = np.asarray(scores, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.bool_)
    positive = int(np.sum(truth))
    negative = len(truth) - positive
    if positive == 0 or negative == 0:
        return None
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    position = 0
    while position < len(values):
        end = position + 1
        while end < len(values) and values[order[end]] == values[order[position]]:
            end += 1
        ranks[order[position:end]] = (
            float(position + 1) + float(end)
        ) / 2.0
        position = end
    rank_sum = float(np.sum(ranks[truth]))
    return (
        rank_sum - positive * (positive + 1) / 2.0
    ) / float(positive * negative)


def _progress_metrics(
    *,
    truth: NDArray[np.float64],
    trajectory_ids: Tuple[str, ...],
    features: Mapping[str, NDArray[np.float64]],
) -> Mapping[str, Any]:
    target = np.asarray(truth, dtype=np.float64)
    ids = np.asarray(trajectory_ids, dtype=str)
    if target.shape != (len(ids),):
        raise ValueError("SD-JEPA progress evidence is invalid")
    result = {}
    for name, raw in features.items():
        values = np.asarray(raw, dtype=np.float64)
        if values.ndim != 2 or len(values) != len(target):
            raise ValueError("SD-JEPA progress feature shape differs")
        predictions = np.empty_like(target)
        per_trajectory = {}
        for trajectory_id in sorted(set(trajectory_ids)):
            selected = ids == trajectory_id
            local_x = values[selected]
            local_y = target[selected]
            design = np.concatenate(
                (local_x, np.ones((len(local_x), 1))), axis=1
            )
            penalty = np.eye(design.shape[1]) * 1e-3
            penalty[-1, -1] = 0.0
            weights = np.linalg.solve(
                design.T @ design + penalty,
                design.T @ local_y,
            )
            local_prediction = design @ weights
            predictions[selected] = local_prediction
            per_trajectory[trajectory_id] = _r2(
                local_y, local_prediction
            )
        result[name] = {
            "pooled_r2": _r2(target, predictions),
            "mean_per_trajectory_r2": float(
                np.mean(list(per_trajectory.values()))
            ),
            "per_trajectory_r2": per_trajectory,
        }
    return result


def _r2(
    truth: NDArray[np.float64], prediction: NDArray[np.float64]
) -> float:
    residual = float(np.sum(np.square(truth - prediction)))
    total = float(np.sum(np.square(truth - np.mean(truth))))
    return 0.0 if total <= 1e-18 else 1.0 - residual / total


def _progression_geometry(
    progression: NDArray[np.float64],
    trajectory_ids: Tuple[str, ...],
) -> Mapping[str, Any]:
    values = np.asarray(progression, dtype=np.float64)
    ids = np.asarray(trajectory_ids, dtype=str)
    angles = np.arctan2(values[:, 1], values[:, 0])
    radii = np.linalg.norm(values, axis=1)
    spans = []
    for trajectory_id in sorted(set(trajectory_ids)):
        local = np.unwrap(angles[ids == trajectory_id])
        spans.append(float(np.max(local) - np.min(local)))
    return {
        "mean_unwrapped_angular_span_radians": float(np.mean(spans)),
        "maximum_unwrapped_angular_span_radians": float(np.max(spans)),
        "mean_radius": float(np.mean(radii)),
        "radius_standard_deviation": float(np.std(radii)),
        "radius_coefficient_of_variation": float(
            np.std(radii) / max(np.mean(radii), 1e-12)
        ),
    }


def _trajectory_alert_metrics(
    *,
    decisions: NDArray[np.bool_],
    trajectory_ids: Tuple[str, ...],
    transition_indices: NDArray[np.int64],
    onsets: Mapping[str, Optional[int]],
) -> Mapping[str, Any]:
    ids = np.asarray(trajectory_ids, dtype=str)
    rows = []
    control_alerts = []
    detections = []
    pre_onset = []
    delays = []
    for trajectory_id in sorted(set(trajectory_ids)):
        positions = np.flatnonzero(ids == trajectory_id)
        order = positions[
            np.argsort(transition_indices[positions], kind="stable")
        ]
        local_decisions = decisions[order]
        local_transitions = transition_indices[order]
        onset = onsets[trajectory_id]
        if onset is None:
            any_alert = bool(np.any(local_decisions))
            control_alerts.append(any_alert)
            rows.append(
                {
                    "trajectory_id": trajectory_id,
                    "is_treatment": False,
                    "onset_transition": None,
                    "any_alert": any_alert,
                    "alert_count": int(np.sum(local_decisions)),
                    "pre_onset_alert": False,
                    "first_post_onset_alert_transition": None,
                    "post_onset_delay_transitions": None,
                }
            )
            continue
        before = local_transitions < onset
        after = local_transitions >= onset
        local_pre = bool(np.any(local_decisions & before))
        post = np.flatnonzero(local_decisions & after)
        detected = bool(len(post))
        delay = (
            None
            if not detected
            else int(local_transitions[int(post[0])] - onset)
        )
        detections.append(detected)
        pre_onset.append(local_pre)
        if delay is not None:
            delays.append(delay)
        rows.append(
            {
                "trajectory_id": trajectory_id,
                "is_treatment": True,
                "onset_transition": onset,
                "any_alert": bool(np.any(local_decisions)),
                "alert_count": int(np.sum(local_decisions)),
                "pre_onset_alert": local_pre,
                "first_post_onset_alert_transition": (
                    None if delay is None else onset + delay
                ),
                "post_onset_delay_transitions": delay,
            }
        )
    return {
        "control_trajectory_count": len(control_alerts),
        "treatment_trajectory_count": len(detections),
        "control_trajectory_false_alarm_rate": (
            float(np.mean(control_alerts)) if control_alerts else 0.0
        ),
        "treatment_detection_rate": (
            float(np.mean(detections)) if detections else 0.0
        ),
        "treatment_pre_onset_alert_rate": (
            float(np.mean(pre_onset)) if pre_onset else 0.0
        ),
        "total_alert_count": int(np.sum(decisions)),
        "alerts_per_logical_run": float(
            np.sum(decisions) / max(1, len(set(trajectory_ids)))
        ),
        "median_post_onset_delay_transitions": (
            None if not delays else float(np.median(delays))
        ),
        "worst_post_onset_delay_transitions": (
            None if not delays else int(max(delays))
        ),
        "trajectory_rows": rows,
    }


def _state_retention_metrics(
    *,
    truth: NDArray[np.float64],
    scale: NDArray[np.float64],
    varying: NDArray[np.bool_],
    predictions: Mapping[str, NDArray[np.float64]],
) -> Mapping[str, Any]:
    target = np.asarray(truth, dtype=np.float64)
    target_scale = np.asarray(scale, dtype=np.float64)
    varying_mask = np.asarray(varying, dtype=np.bool_)
    if (
        target.ndim != 3
        or target_scale.shape != target.shape[1:]
        or varying_mask.shape != target_scale.shape
        or not np.any(varying_mask)
    ):
        raise ValueError("SD-JEPA state evidence is invalid")
    metrics: Dict[str, Any] = {}
    for name, raw in predictions.items():
        prediction = np.asarray(raw, dtype=np.float64)
        if prediction.shape != target.shape:
            raise ValueError("SD-JEPA state prediction shape differs")
        normalized = (prediction - target) / target_scale[None]
        per_entity: Dict[str, Optional[float]] = {}
        for entity in range(target.shape[1]):
            selected = varying_mask[entity]
            per_entity[str(entity)] = (
                None
                if not np.any(selected)
                else float(
                    np.sqrt(
                        np.mean(
                            np.square(normalized[:, entity, selected])
                        )
                    )
                )
            )
        metrics[name] = {
            "aggregate_nrmse": float(
                np.sqrt(np.mean(np.square(normalized[:, varying_mask])))
            ),
            "per_entity_nrmse": per_entity,
        }
    metrics["varying_entity_count"] = int(
        np.sum(np.any(varying_mask, axis=1))
    )
    return metrics


def _maximum_difference(
    left: Mapping[str, Mapping[str, NDArray[np.float64]]],
    right: Mapping[str, Mapping[str, NDArray[np.float64]]],
) -> float:
    return max(
        float(np.max(np.abs(left[role][name] - right[role][name])))
        for role in ASSESSMENT_ROLES
        for name in SD_JEPA_SCORE_NAMES
    )


def _all_equal(
    left: Mapping[str, Mapping[str, NDArray[np.bool_]]],
    right: Mapping[str, Mapping[str, NDArray[np.bool_]]],
) -> bool:
    return all(
        np.array_equal(left[role][name], right[role][name])
        for role in ASSESSMENT_ROLES
        for name in SD_JEPA_SCORE_NAMES
    )


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"SD-JEPA JSON root is not an object: {path}")
    return value


def _pretty_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    arguments = parser.parse_args()
    result = verify_stored_assessment(arguments.artifact)
    verify_artifact_manifest(arguments.artifact)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
