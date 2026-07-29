#!/usr/bin/env python3
"""Independent stored-array assessment for the CF-JEPA alert tracer."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from quantis_core.edge_dynamics.cf_jepa import (
    CF_JEPA_ALERT_MODEL_NAMES,
    CF_JEPA_ASSESSMENT_ROLE_NAMES,
)
from quantis_core.edge_dynamics.hepa_jepa import (
    calibrate_probability_surface,
    fit_logit_calibrator,
    trajectory_alert_threshold,
)


def assess_stored_bundle(directory: Path) -> Mapping[str, Any]:
    """Recompute the complete decision without loading a fitted model."""

    root = Path(directory)
    protocol = _read_json(root / "protocol.json")
    models = _read_json(root / "models.json")
    with np.load(root / "evidence.npz", allow_pickle=False) as arrays:
        labels = {
            role: arrays[f"labels__{role}"].astype(np.bool_)
            for role in CF_JEPA_ASSESSMENT_ROLE_NAMES
        }
        raw_scores = _read_float_arrays(arrays, "score")
        restored_scores = _read_float_arrays(
            arrays, "restored_score"
        )
        stored_calibrated = _read_float_arrays(
            arrays, "calibrated"
        )
        restored_calibrated = _read_float_arrays(
            arrays, "restored_calibrated"
        )
        stored_decisions = _read_bool_arrays(
            arrays, "alert_decision"
        )
        restored_decisions = _read_bool_arrays(
            arrays, "restored_alert_decision"
        )
        calibrations: Dict[str, Mapping[str, float]] = {}
        calibrated: Dict[
            str, Dict[str, NDArray[np.float64]]
        ] = {
            role: {} for role in CF_JEPA_ASSESSMENT_ROLE_NAMES
        }
        decisions: Dict[str, Dict[str, NDArray[np.bool_]]] = {
            role: {} for role in CF_JEPA_ASSESSMENT_ROLE_NAMES
        }
        calibration_role = dict(protocol["roles"]["calibration"])
        calibration_trajectory_ids = tuple(
            str(value)
            for value in calibration_role["trajectory_ids"]
        )
        calibration_control_ids = tuple(
            str(value)
            for value in calibration_role["control_trajectory_ids"]
        )
        for model in CF_JEPA_ALERT_MODEL_NAMES:
            slope, intercept, calibration_brier = fit_logit_calibrator(
                raw_scores["calibration"][model][:, None],
                labels["calibration"][:, None],
            )
            calibration_values = calibrate_probability_surface(
                raw_scores["calibration"][model][:, None],
                slope=slope,
                intercept=intercept,
            )
            threshold = trajectory_alert_threshold(
                calibration_values,
                calibration_trajectory_ids,
                calibration_control_ids,
            )
            calibrations[model] = {
                "slope": slope,
                "intercept": intercept,
                "calibration_brier": calibration_brier,
                "alert_threshold": threshold,
            }
            for role in CF_JEPA_ASSESSMENT_ROLE_NAMES:
                calibrated[role][model] = (
                    calibrate_probability_surface(
                        raw_scores[role][model][:, None],
                        slope=slope,
                        intercept=intercept,
                    )[:, 0]
                )
                decisions[role][model] = np.asarray(
                    calibrated[role][model] >= threshold,
                    dtype=np.bool_,
                )

        risk_metrics = {
            role: {
                model: _binary_probability_metrics(
                    calibrated[role][model], labels[role]
                )
                for model in CF_JEPA_ALERT_MODEL_NAMES
            }
            for role in CF_JEPA_ASSESSMENT_ROLE_NAMES
        }
        alert_metrics = {}
        for role in CF_JEPA_ASSESSMENT_ROLE_NAMES:
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
            alert_metrics[role] = {
                model: _trajectory_alert_metrics(
                    decisions=decisions[role][model],
                    trajectory_ids=trajectory_ids,
                    transition_indices=transition_indices,
                    onsets=onsets,
                )
                for model in CF_JEPA_ALERT_MODEL_NAMES
            }

        restoration = {
            "score_max_abs": _maximum_difference(
                raw_scores, restored_scores
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
            "target_representation_max_abs": float(
                np.max(
                    np.abs(
                        arrays["candidate_target_temporal"]
                        - arrays[
                            "restored_candidate_target_temporal"
                        ]
                    )
                )
            ),
            "online_representation_max_abs": float(
                np.max(
                    np.abs(
                        arrays["candidate_online_temporal"]
                        - arrays[
                            "restored_candidate_online_temporal"
                        ]
                    )
                )
            ),
        }
        geometry = _geometry_metrics(
            target=arrays["candidate_target_temporal"],
            online=arrays["candidate_online_temporal"],
        )
        state_retention = _state_retention_metrics(
            truth=arrays["state_truth"],
            scale=arrays["state_scale"],
            varying=arrays["state_varying_mask"].astype(np.bool_),
            predictions={
                "cf_jepa_target": arrays[
                    "state_prediction__cf_jepa_target"
                ],
                "matched_pca": arrays[
                    "state_prediction__matched_pca"
                ],
            },
        )
        latency = {
            model: {
                "mean_milliseconds": float(
                    np.mean(arrays[f"latency_samples__{model}"])
                ),
                "p95_milliseconds": float(
                    np.quantile(
                        arrays[f"latency_samples__{model}"],
                        0.95,
                    )
                ),
            }
            for model in CF_JEPA_ALERT_MODEL_NAMES
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
        finite_arrays = bool(
            all(
                array.dtype.kind in ("U", "S", "b", "i", "u")
                or np.all(np.isfinite(array))
                for array in (
                    arrays[name] for name in arrays.files
                )
            )
        )

    inference_counts = _inference_parameter_counts(models)
    capacity = {
        "inference_parameter_counts": inference_counts,
        "all_neural_routes_equal": (
            len(set(inference_counts.values())) == 1
        ),
        "training_parameter_counts": dict(
            protocol["training_parameter_counts"]
        ),
    }
    candidate_bundle_bytes = sum(
        (root / name).stat().st_size
        for name in (
            "models.json",
            "alerts.json",
            "state-probes.json",
            "event-definition.json",
        )
    )
    transfer_risk = risk_metrics["evaluation_transfer"]
    transfer_alert = alert_metrics["evaluation_transfer"]
    candidate_brier = transfer_risk["cf_jepa_target"]["brier"]
    reference_briers = [
        transfer_risk[name]["brier"]
        for name in (
            "cf_jepa_online",
            "one_zone_target",
            "masked_latent_target",
            "matched_pca",
        )
    ]
    candidate_detection = transfer_alert[
        "cf_jepa_target"
    ]["treatment_detection_rate"]
    neural_detection_references = [
        transfer_alert[name]["treatment_detection_rate"]
        for name in (
            "cf_jepa_online",
            "one_zone_target",
            "masked_latent_target",
        )
    ]
    median_delay = transfer_alert["cf_jepa_target"][
        "median_post_onset_delay_transitions"
    ]
    gates = {
        "finite_evidence": finite_arrays,
        "restoration": bool(
            restoration["score_max_abs"] <= 1e-6
            and restoration["calibrated_max_abs"] <= 1e-6
            and restoration["stored_calibrated_max_abs"] <= 1e-6
            and restoration["target_representation_max_abs"] <= 1e-6
            and restoration["online_representation_max_abs"] <= 1e-6
            and restoration["decision_exact"]
            and restoration["stored_decision_exact"]
        ),
        "deployed_capacity": capacity["all_neural_routes_equal"],
        "state_safety": bool(
            state_retention["cf_jepa_target"]["aggregate_nrmse"]
            <= 1.05
            * state_retention["matched_pca"]["aggregate_nrmse"]
        ),
        "asymmetric_geometry": bool(
            geometry["target_adjacent_cosine_similarity"]
            > geometry["online_adjacent_cosine_similarity"]
            and geometry["target_effective_rank_90"]
            <= geometry["online_effective_rank_90"]
        ),
        "edge_size": candidate_bundle_bytes <= 16 * 1024 * 1024,
        "causal_input": all(causal_audit.values()),
        "predictive_alert_score_lane": bool(
            candidate_brier <= 0.95 * min(reference_briers)
        ),
        "trajectory_alert_lane": bool(
            transfer_alert["cf_jepa_target"][
                "control_trajectory_false_alarm_rate"
            ]
            <= 0.05
            and candidate_detection >= 0.80
            and median_delay is not None
            and float(median_delay) <= 10.0
            and all(
                candidate_detection >= value + 0.10
                for value in neural_detection_references
            )
        ),
    }
    safety_names = (
        "finite_evidence",
        "restoration",
        "deployed_capacity",
        "state_safety",
        "asymmetric_geometry",
        "edge_size",
        "causal_input",
    )
    interpretable = bool(protocol["interpretable"])
    passed = bool(
        interpretable
        and all(gates[name] for name in safety_names)
        and (
            gates["predictive_alert_score_lane"]
            or gates["trajectory_alert_lane"]
        )
    )
    decision = (
        "non_interpretable_cf_jepa_smoke"
        if not interpretable
        else (
            "advance_cf_jepa_to_fixed_seed_robustness"
            if passed
            else "reject_cf_jepa_edge_recipe"
        )
    )
    return {
        "schema_version": 1,
        "kind": "cf_jepa_stored_array_assessment",
        "interpretable": interpretable,
        "eligible_for_advance": interpretable,
        "passed": passed,
        "decision": decision,
        "calibrations": calibrations,
        "risk_metrics": risk_metrics,
        "alert_metrics": alert_metrics,
        "geometry": geometry,
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
    """Require the checked-in assessment to equal recomputation."""

    root = Path(directory)
    recomputed = assess_stored_bundle(root)
    if (root / "assessment.json").read_text() != _pretty_json(
        recomputed
    ):
        raise ValueError("stored CF-JEPA assessment does not recompute")
    return recomputed


def verify_artifact_manifest(directory: Path) -> None:
    """Verify every retained evidence-bearing file digest."""

    root = Path(directory)
    payload = _read_json(root / "artifact-manifest.json")
    files = dict(payload["files"])
    observed = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name != "artifact-manifest.json"
    }
    if observed != set(files):
        raise ValueError("CF-JEPA artifact manifest file set differs")
    for relative, raw in files.items():
        path = root / relative
        expected = dict(raw)
        if (
            path.stat().st_size != int(expected["bytes"])
            or _file_sha256(path) != str(expected["sha256"])
        ):
            raise ValueError(
                f"CF-JEPA artifact manifest differs: {relative}"
            )


def _read_float_arrays(
    arrays: Any, prefix: str
) -> Mapping[str, Mapping[str, NDArray[np.float64]]]:
    return {
        role: {
            model: np.asarray(
                arrays[f"{prefix}__{role}__{model}"],
                dtype=np.float64,
            )
            for model in CF_JEPA_ALERT_MODEL_NAMES
        }
        for role in CF_JEPA_ASSESSMENT_ROLE_NAMES
    }


def _read_bool_arrays(
    arrays: Any, prefix: str
) -> Mapping[str, Mapping[str, NDArray[np.bool_]]]:
    return {
        role: {
            model: arrays[
                f"{prefix}__{role}__{model}"
            ].astype(np.bool_)
            for model in CF_JEPA_ALERT_MODEL_NAMES
        }
        for role in CF_JEPA_ASSESSMENT_ROLE_NAMES
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
        raise ValueError("CF-JEPA probability inputs are invalid")
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
    treatment_detections = []
    treatment_pre_onset = []
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
        pre_alert = bool(np.any(local_decisions & before))
        post = np.flatnonzero(local_decisions & after)
        detected = bool(len(post))
        delay = (
            None
            if not detected
            else int(local_transitions[int(post[0])] - onset)
        )
        treatment_pre_onset.append(pre_alert)
        treatment_detections.append(detected)
        if delay is not None:
            delays.append(delay)
        rows.append(
            {
                "trajectory_id": trajectory_id,
                "is_treatment": True,
                "onset_transition": onset,
                "any_alert": bool(np.any(local_decisions)),
                "alert_count": int(np.sum(local_decisions)),
                "pre_onset_alert": pre_alert,
                "first_post_onset_alert_transition": (
                    None if delay is None else onset + delay
                ),
                "post_onset_delay_transitions": delay,
            }
        )
    return {
        "control_trajectory_count": len(control_alerts),
        "treatment_trajectory_count": len(treatment_detections),
        "control_trajectory_false_alarm_rate": (
            float(np.mean(control_alerts)) if control_alerts else 0.0
        ),
        "treatment_detection_rate": (
            float(np.mean(treatment_detections))
            if treatment_detections
            else 0.0
        ),
        "treatment_pre_onset_alert_rate": (
            float(np.mean(treatment_pre_onset))
            if treatment_pre_onset
            else 0.0
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


def _geometry_metrics(
    *,
    target: NDArray[np.float64],
    online: NDArray[np.float64],
) -> Mapping[str, Any]:
    return {
        "target_adjacent_cosine_similarity": _adjacent_cosine(target),
        "online_adjacent_cosine_similarity": _adjacent_cosine(online),
        "target_effective_rank_90": _effective_rank(target),
        "online_effective_rank_90": _effective_rank(online),
    }


def _adjacent_cosine(values: NDArray[np.float64]) -> float:
    temporal = np.asarray(values, dtype=np.float64)
    left = temporal[:, :, :-1].reshape(-1, temporal.shape[-1])
    right = temporal[:, :, 1:].reshape(-1, temporal.shape[-1])
    denominator = np.maximum(
        np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1),
        1e-12,
    )
    return float(np.mean(np.sum(left * right, axis=1) / denominator))


def _effective_rank(values: NDArray[np.float64]) -> int:
    matrix = np.asarray(values, dtype=np.float64).reshape(
        -1, values.shape[-1]
    )
    matrix = matrix - np.mean(matrix, axis=0, keepdims=True)
    singular = np.linalg.svd(matrix, compute_uv=False)
    variance = np.square(singular)
    if float(np.sum(variance)) <= 1e-18:
        return 0
    cumulative = np.cumsum(variance) / np.sum(variance)
    return int(np.searchsorted(cumulative, 0.90, side="left") + 1)


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
        raise ValueError("CF-JEPA state evidence is invalid")
    metrics: Dict[str, Any] = {}
    for name, raw in predictions.items():
        prediction = np.asarray(raw, dtype=np.float64)
        if prediction.shape != target.shape:
            raise ValueError("CF-JEPA state prediction shape differs")
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
                            np.square(
                                normalized[:, entity, selected]
                            )
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
        float(np.max(np.abs(left[role][model] - right[role][model])))
        for role in CF_JEPA_ASSESSMENT_ROLE_NAMES
        for model in CF_JEPA_ALERT_MODEL_NAMES
    )


def _all_equal(
    left: Mapping[str, Mapping[str, NDArray[np.bool_]]],
    right: Mapping[str, Mapping[str, NDArray[np.bool_]]],
) -> bool:
    return all(
        np.array_equal(left[role][model], right[role][model])
        for role in CF_JEPA_ASSESSMENT_ROLE_NAMES
        for model in CF_JEPA_ALERT_MODEL_NAMES
    )


def _inference_parameter_counts(
    models: Mapping[str, Any]
) -> Mapping[str, int]:
    counts = {}
    objective_payloads = dict(models["objectives"])
    for model, objective in (
        ("cf_jepa_target", "three_zone"),
        ("cf_jepa_online", "three_zone"),
        ("one_zone_target", "one_zone"),
        ("masked_latent_target", "masked_latent"),
    ):
        state = dict(objective_payloads[objective]["state_dict"])
        counts[model] = sum(
            int(np.prod(dict(raw)["shape"], dtype=np.int64))
            for name, raw in state.items()
            if str(name).startswith("online_encoder.")
        )
    return counts


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"CF-JEPA JSON root is not an object: {path}")
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
