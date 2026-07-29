#!/usr/bin/env python3
"""Fresh-process, stored-array assessor for the Delta-JEPA tracer."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from lab.action_dynamics.prototype_complete_lejepa import (
    _action_sanity_from_predictions,
    _attribution_scores_from_predictions,
    _downstream_pair_errors,
    _forecast_scores,
    _state_probe,
)
from quantis_core.action_conditioned_dynamics import ActionConditionedWindows
from quantis_core.edge_dynamics.data import PreparedAttributionQueries
from quantis_core.graph_telemetry import DeclaredTelemetryGraph


CELL_NAMES = ("delta_jepa", "endpoint_concat", "prediction_only")
EVALUATION_ROLES = ("selection", "iid_evaluation", "transfer_evaluation")


def assess_stored_bundle(directory: Path) -> Dict[str, Any]:
    """Recompute all reported metrics and gates from immutable arrays."""

    root = Path(directory)
    metadata = _read_json(root / "evidence-metadata.json")
    if (
        metadata.get("schema_version") != 1
        or metadata.get("kind") != "delta_jepa_assessment_evidence"
    ):
        raise ValueError("unsupported Delta-JEPA assessment evidence")
    graph = DeclaredTelemetryGraph.from_dict(dict(metadata["graph"]))
    entity_names = tuple(str(value) for value in metadata["entity_names"])
    state_names = tuple(
        str(value) for value in metadata["state_feature_names"]
    )
    control_names = tuple(
        str(value) for value in metadata["control_feature_names"]
    )
    action_names = tuple(
        str(value) for value in metadata["action_feature_names"]
    )
    ownership = np.asarray(metadata["ownership_mask"], dtype=np.bool_)
    with np.load(root / "evidence.npz", allow_pickle=False) as stored:
        if any(
            not np.all(np.isfinite(stored[name]))
            for name in stored.files
        ):
            raise ValueError("Delta-JEPA evidence contains non-finite arrays")
        windows = {
            role: _restore_windows(
                stored,
                role=role,
                identity=dict(metadata["roles"][role]),
                graph=graph,
                entity_names=entity_names,
                state_names=state_names,
                control_names=control_names,
                action_names=action_names,
            )
            for role in ("fit",) + EVALUATION_ROLES
        }
        raw_scores = {
            role: _forecast_scores(
                stored[f"raw_prediction__{role}"], windows[role]
            )
            for role in EVALUATION_ROLES
        }
        forecast_scores = {
            name: {
                role: _forecast_scores(
                    stored[f"prediction__{name}__{role}"],
                    windows[role],
                )
                for role in EVALUATION_ROLES
            }
            for name in CELL_NAMES
        }
        ridge_curves = {}
        selected_ridges = {}
        for name in CELL_NAMES:
            rows = []
            for ridge in metadata["ridge_values"]:
                scores = _forecast_scores(
                    stored[
                        f"ridge_prediction__{name}__{float(ridge):.4g}"
                    ],
                    windows["selection"],
                )
                rows.append(
                    {
                        "ridge": float(ridge),
                        "raw_safe": bool(
                            all(
                                scores[key]
                                <= 1.05 * raw_scores["selection"][key]
                                for key in (
                                    "overall_mse",
                                    "action_overlap_mse",
                                    "downstream_effect_mse",
                                )
                            )
                        ),
                        **scores,
                    }
                )
            ridge_curves[name] = rows
            eligible = [row for row in rows if row["raw_safe"]]
            selected_ridges[name] = float(
                min(
                    eligible if eligible else rows,
                    key=lambda row: (
                        row["downstream_effect_mse"],
                        row["ridge"],
                    ),
                )["ridge"]
            )

        state_probes = {
            name: _state_probe(
                stored[f"representation__{name}__fit"],
                windows["fit"],
                stored[f"representation__{name}__transfer_evaluation"],
                windows["transfer_evaluation"],
                ownership,
            )
            for name in CELL_NAMES
        }
        state_probes["matched_pca"] = _state_probe(
            stored["representation__matched_pca__fit"],
            windows["fit"],
            stored["representation__matched_pca__transfer_evaluation"],
            windows["transfer_evaluation"],
            ownership,
        )

        query_metadata = dict(metadata["queries"])
        queries = PreparedAttributionQueries(
            query_ids=tuple(
                str(value) for value in query_metadata["query_ids"]
            ),
            histories=stored["query_histories"],
            future_controls=stored["query_future_controls"],
            observed_future=stored["query_observed_future"],
            candidate_actions=stored["query_candidate_actions"],
            candidate_ids=tuple(
                str(value) for value in query_metadata["candidate_ids"]
            ),
            candidate_action_kinds=tuple(
                str(value)
                for value in query_metadata["candidate_action_kinds"]
            ),
            candidate_target_entities=tuple(
                str(value)
                for value in query_metadata["candidate_target_entities"]
            ),
            expected_action_kinds=tuple(
                str(value)
                for value in query_metadata["expected_action_kinds"]
            ),
            expected_target_entities=tuple(
                str(value)
                for value in query_metadata["expected_target_entities"]
            ),
            expected_variant_ids=tuple(
                str(value)
                for value in query_metadata["expected_variant_ids"]
            ),
        )
        attribution = {
            name: _attribution_scores_from_predictions(
                stored[f"attribution_prediction__{name}"],
                queries,
                ownership,
            )
            for name in CELL_NAMES
        }
        action_sanity = {
            name: _action_sanity_from_predictions(
                {
                    variant: stored[
                        f"action_sanity__{name}__{variant}"
                    ]
                    for variant in ("correct", "no_action", "shuffled")
                },
                windows["transfer_evaluation"],
                ownership,
            )
            for name in CELL_NAMES
        }
        pair_errors = {
            name: _downstream_pair_errors(
                stored[f"prediction__{name}__transfer_evaluation"],
                windows["transfer_evaluation"],
            )
            for name in CELL_NAMES
        }

        mechanism = {}
        for name in CELL_NAMES:
            fit_diag = _diagnostic_arrays(stored, name, "fit")
            transfer_diag = _diagnostic_arrays(
                stored, name, "transfer_evaluation"
            )
            mechanism[name] = {
                "action_reconstruction": _action_reconstruction(
                    transfer_diag
                ),
                "action_sequence_retrieval": _sequence_retrieval(
                    transfer_diag
                ),
                "delta_to_state_change": _delta_state_probe(
                    fit_diag, transfer_diag, ownership
                ),
                "displacement_geometry": _displacement_geometry(
                    transfer_diag["displacements"]
                ),
            }

        restoration = {
            name: {
                "representation": bool(
                    np.allclose(
                        stored[f"restored_representation__{name}"],
                        stored[
                            f"representation__{name}__transfer_evaluation"
                        ][:8],
                        atol=1e-6,
                        rtol=0.0,
                    )
                ),
                "decoder": bool(
                    np.allclose(
                        stored[f"restored_decoder__{name}"],
                        stored[
                            f"diagnostic__{name}__transfer_evaluation"
                            "__predicted_actions"
                        ][:16],
                        atol=1e-6,
                        rtol=0.0,
                    )
                ),
                "probe": bool(
                    np.allclose(
                        stored[f"restored_probe_prediction__{name}"],
                        stored[
                            f"prediction__{name}__transfer_evaluation"
                        ][:8],
                        atol=1e-6,
                        rtol=0.0,
                    )
                ),
                "attribution": bool(
                    np.allclose(
                        stored[f"restored_attribution_prediction__{name}"],
                        stored[f"attribution_prediction__{name}"],
                        atol=1e-6,
                        rtol=0.0,
                    )
                ),
            }
            for name in CELL_NAMES
        }
        schedule_valid = _schedule_is_pair_blocked(
            root / "anchor-schedule.npz",
            int(metadata["pair_counts"]["fit"]),
        )
        result = _assess(
            interpretable=bool(metadata["interpretable"]),
            forecast_scores=forecast_scores,
            raw_scores=raw_scores,
            ridge_curves=ridge_curves,
            selected_ridges=selected_ridges,
            state_probes=state_probes,
            attribution=attribution,
            action_sanity=action_sanity,
            pair_errors=pair_errors,
            mechanism=mechanism,
            restoration=restoration,
            parameter_counts=dict(metadata["parameter_counts"]),
            candidate_bundle_bytes=int(
                metadata["candidate_bundle_bytes"]
            ),
            public_causality=bool(metadata["public_causality"]),
            schedule_valid=schedule_valid,
        )
        return {
            "schema_version": 1,
            "kind": "delta_jepa_stored_array_assessment",
            "forecast_scores": forecast_scores,
            "raw_low_rank_scores": raw_scores,
            "ridge_curves": ridge_curves,
            "selected_ridges": selected_ridges,
            "state_probes": state_probes,
            "attribution": attribution,
            "action_sanity": action_sanity,
            "mechanism": mechanism,
            "transfer_pair_errors": pair_errors,
            "restoration": restoration,
            **result,
        }


def verify_stored_assessment(directory: Path) -> None:
    """Require the recorded assessment to equal fresh recomputation."""

    root = Path(directory)
    recorded = _read_json(root / "assessment.json")
    actual = assess_stored_bundle(root)
    if recorded != actual:
        raise ValueError("stored Delta-JEPA assessment does not recompute")


def verify_artifact_manifest(directory: Path) -> None:
    """Verify every retained file against the content manifest."""

    root = Path(directory)
    payload = _read_json(root / "artifact-manifest.json")
    files = dict(payload["files"])
    observed = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name != "artifact-manifest.json"
    }
    if observed != set(files):
        raise ValueError("Delta-JEPA artifact manifest file set differs")
    for relative, raw in files.items():
        expected = dict(raw)
        path = root / relative
        if (
            path.stat().st_size != int(expected["bytes"])
            or _file_sha256(path) != str(expected["sha256"])
        ):
            raise ValueError(
                f"Delta-JEPA artifact manifest differs: {relative}"
            )


def _assess(
    *,
    interpretable: bool,
    forecast_scores: Mapping[str, Mapping[str, Mapping[str, float]]],
    raw_scores: Mapping[str, Mapping[str, float]],
    ridge_curves: Mapping[str, Sequence[Mapping[str, Any]]],
    selected_ridges: Mapping[str, float],
    state_probes: Mapping[str, Mapping[str, Any]],
    attribution: Mapping[str, Mapping[str, float]],
    action_sanity: Mapping[str, Mapping[str, float]],
    pair_errors: Mapping[str, Mapping[str, float]],
    mechanism: Mapping[str, Mapping[str, Any]],
    restoration: Mapping[str, Mapping[str, bool]],
    parameter_counts: Mapping[str, Any],
    candidate_bundle_bytes: int,
    public_causality: bool,
    schedule_valid: bool,
) -> Mapping[str, Any]:
    selection_verified = all(
        float(selected_ridges[name])
        == float(
            min(
                (
                    [row for row in ridge_curves[name] if row["raw_safe"]]
                    or list(ridge_curves[name])
                ),
                key=lambda row: (
                    row["downstream_effect_mse"],
                    row["ridge"],
                ),
            )["ridge"]
        )
        for name in CELL_NAMES
    )
    training_counts = {
        int(dict(parameter_counts[name])["training"]) for name in CELL_NAMES
    }
    inference_counts = {
        int(dict(parameter_counts[name])["inference"]) for name in CELL_NAMES
    }
    candidate_transfer = forecast_scores["delta_jepa"][
        "transfer_evaluation"
    ]
    concat_transfer = forecast_scores["endpoint_concat"][
        "transfer_evaluation"
    ]
    raw_transfer = raw_scores["transfer_evaluation"]
    candidate_pairs = pair_errors["delta_jepa"]
    concat_pairs = pair_errors["endpoint_concat"]
    common_pairs = sorted(set(candidate_pairs) & set(concat_pairs))
    win_fraction = float(
        np.mean(
            [
                candidate_pairs[pair] < concat_pairs[pair]
                for pair in common_pairs
            ]
        )
    )
    candidate_mechanism = mechanism["delta_jepa"]
    concat_mechanism = mechanism["endpoint_concat"]
    candidate_reconstruction = candidate_mechanism[
        "action_reconstruction"
    ]["treatment_mse"]
    concat_reconstruction = concat_mechanism[
        "action_reconstruction"
    ]["treatment_mse"]
    candidate_retrieval = candidate_mechanism[
        "action_sequence_retrieval"
    ]["treatment_correct_fraction"]
    concat_retrieval = concat_mechanism[
        "action_sequence_retrieval"
    ]["treatment_correct_fraction"]
    safety = {
        "all_evidence_finite": True,
        "all_public_outputs_restore": all(
            all(rows.values()) for rows in restoration.values()
        ),
        "equal_training_capacity": len(training_counts) == 1,
        "equal_inference_capacity": len(inference_counts) == 1,
        "pair_blocked_schedule_valid": schedule_valid,
        "selection_only_ridge_choice_recomputes": selection_verified,
        "state_probe_within_1_05_pca": (
            state_probes["delta_jepa"]["aggregate_nrmse"]
            <= 1.05 * state_probes["matched_pca"]["aggregate_nrmse"]
        ),
        "overall_mse_within_1_05_raw": (
            candidate_transfer["overall_mse"]
            <= 1.05 * raw_transfer["overall_mse"]
        ),
        "action_overlap_mse_within_1_05_raw": (
            candidate_transfer["action_overlap_mse"]
            <= 1.05 * raw_transfer["action_overlap_mse"]
        ),
        "action_and_target_hit_at_1_at_least_0_95": (
            attribution["delta_jepa"]["action_and_target_hit_at_1"] >= 0.95
        ),
        "no_action_specificity_is_1": (
            attribution["delta_jepa"]["no_action_specificity"] == 1.0
        ),
        "correct_action_beats_both_at_least_0_80": (
            action_sanity["delta_jepa"][
                "correct_action_beats_both_fraction"
            ]
            >= 0.80
        ),
        "public_inference_is_causal": public_causality,
        "candidate_bundle_at_most_16_mib": (
            candidate_bundle_bytes <= 16 * 1024 * 1024
        ),
    }
    mechanism_gates = {
        "treatment_reconstruction_improves_concat_by_10_percent": (
            candidate_reconstruction <= 0.90 * concat_reconstruction
        ),
        "retrieval_improves_concat_by_0_10": (
            candidate_retrieval >= concat_retrieval + 0.10
        ),
    }
    value = {
        "transfer_effect_improves_concat_by_10_percent": (
            candidate_transfer["downstream_effect_mse"]
            <= 0.90 * concat_transfer["downstream_effect_mse"]
        ),
        "transfer_effect_improves_raw_by_10_percent": (
            candidate_transfer["downstream_effect_mse"]
            <= 0.90 * raw_transfer["downstream_effect_mse"]
        ),
        "pair_win_fraction_at_least_0_60": win_fraction >= 0.60,
        "selection_effect_strictly_below_concat": (
            forecast_scores["delta_jepa"]["selection"][
                "downstream_effect_mse"
            ]
            < forecast_scores["endpoint_concat"]["selection"][
                "downstream_effect_mse"
            ]
        ),
    }
    safety_passed = all(safety.values())
    mechanism_passed = all(mechanism_gates.values())
    value_passed = all(value.values())
    eligible = interpretable and safety_passed and mechanism_passed and value_passed
    if not interpretable:
        decision = "non_interpretable_delta_jepa_smoke"
    elif eligible:
        decision = "advance_delta_jepa_to_fixed_seed_robustness"
    else:
        decision = "reject_delta_jepa_edge_recipe"
    return {
        "interpretable": interpretable,
        "safety_gates": safety,
        "mechanism_gates": mechanism_gates,
        "downstream_value_gates": value,
        "candidate_pair_win_fraction": win_fraction,
        "safety_passed": safety_passed,
        "mechanism_passed": mechanism_passed,
        "downstream_value_passed": value_passed,
        "eligible_for_advance": eligible,
        "passed": eligible,
        "decision": decision,
    }


def _restore_windows(
    stored: Any,
    *,
    role: str,
    identity: Mapping[str, Any],
    graph: DeclaredTelemetryGraph,
    entity_names: Tuple[str, ...],
    state_names: Tuple[str, ...],
    control_names: Tuple[str, ...],
    action_names: Tuple[str, ...],
) -> ActionConditionedWindows:
    return ActionConditionedWindows(
        histories=stored[f"histories__{role}"],
        future_states=stored[f"target__{role}"],
        future_controls=stored[f"controls__{role}"],
        future_actions=stored[f"actions__{role}"],
        trajectory_ids=tuple(
            str(value) for value in identity["trajectory_ids"]
        ),
        matched_pair_ids=tuple(
            str(value) for value in identity["matched_pair_ids"]
        ),
        transition_indices=np.asarray(
            identity["transition_indices"], dtype=np.int64
        ),
        entity_names=entity_names,
        state_feature_names=state_names,
        control_feature_names=control_names,
        action_feature_names=action_names,
        graph=graph,
    )


def _diagnostic_arrays(
    stored: Any, name: str, role: str
) -> Mapping[str, NDArray[Any]]:
    return {
        field: stored[f"diagnostic__{name}__{role}__{field}"]
        for field in (
            "displacements",
            "predicted_actions",
            "target_actions",
            "state_changes",
            "treatment_mask",
            "pair_indices",
        )
    }


def _action_reconstruction(
    diagnostic: Mapping[str, NDArray[Any]]
) -> Mapping[str, float]:
    squared = np.square(
        diagnostic["predicted_actions"] - diagnostic["target_actions"]
    )
    treatment = diagnostic["treatment_mask"].astype(np.bool_)
    return {
        "all_mse": float(np.mean(squared)),
        "treatment_mse": float(np.mean(squared[treatment])),
        "treatment_interval_count": int(np.sum(treatment)),
    }


def _sequence_retrieval(
    diagnostic: Mapping[str, NDArray[Any]]
) -> Mapping[str, float]:
    prediction = diagnostic["predicted_actions"].reshape(
        len(diagnostic["predicted_actions"]), -1
    )
    target = diagnostic["target_actions"].reshape(len(prediction), -1)
    no_action = np.zeros_like(target)
    pair_indices = diagnostic["pair_indices"].astype(np.int64)
    unique_pairs = sorted(set(pair_indices.tolist()))
    shuffled = np.empty_like(target)
    for position, pair in enumerate(unique_pairs):
        rows = np.flatnonzero(pair_indices == pair)
        donor = unique_pairs[(position + 1) % len(unique_pairs)]
        donor_rows = np.flatnonzero(pair_indices == donor)
        shuffled[rows] = np.resize(target[donor_rows], (len(rows), target.shape[1]))
    distances = np.stack(
        (
            np.mean(np.square(prediction - target), axis=1),
            np.mean(np.square(prediction - no_action), axis=1),
            np.mean(np.square(prediction - shuffled), axis=1),
        ),
        axis=1,
    )
    correct = np.argmin(distances, axis=1) == 0
    treatment = diagnostic["treatment_mask"].astype(np.bool_)
    return {
        "all_correct_fraction": float(np.mean(correct)),
        "treatment_correct_fraction": float(np.mean(correct[treatment])),
        "mean_correct_distance": float(np.mean(distances[:, 0])),
        "mean_no_action_distance": float(np.mean(distances[:, 1])),
        "mean_shuffled_distance": float(np.mean(distances[:, 2])),
    }


def _delta_state_probe(
    fit: Mapping[str, NDArray[Any]],
    evaluation: Mapping[str, NDArray[Any]],
    ownership: NDArray[np.bool_],
) -> Mapping[str, float]:
    fit_x = fit["displacements"].reshape(len(fit["displacements"]), -1)
    evaluation_x = evaluation["displacements"].reshape(
        len(evaluation["displacements"]), -1
    )
    fit_y = fit["state_changes"][..., ownership]
    evaluation_y = evaluation["state_changes"][..., ownership]
    center = np.mean(fit_x, axis=0)
    scale = np.std(fit_x, axis=0)
    mask = scale > 1e-12
    design = np.column_stack(
        (
            (fit_x[:, mask] - center[mask]) / scale[mask],
            np.ones(len(fit_x)),
        )
    )
    penalty = np.eye(design.shape[1], dtype=np.float64)
    penalty[-1, -1] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + 1e-3 * penalty,
        design.T @ fit_y,
    )
    evaluation_design = np.column_stack(
        (
            (evaluation_x[:, mask] - center[mask]) / scale[mask],
            np.ones(len(evaluation_x)),
        )
    )
    prediction = evaluation_design @ coefficients
    target_scale = np.std(fit_y, axis=0)
    target_scale[target_scale <= 1e-12] = 1.0
    nrmse = float(
        np.sqrt(np.mean(np.square((prediction - evaluation_y) / target_scale)))
    )
    flat_prediction = prediction.reshape(-1)
    flat_target = evaluation_y.reshape(-1)
    correlation = (
        0.0
        if np.std(flat_prediction) <= 1e-12 or np.std(flat_target) <= 1e-12
        else float(np.corrcoef(flat_prediction, flat_target)[0, 1])
    )
    return {"nrmse": nrmse, "pearson": correlation}


def _displacement_geometry(
    displacements: NDArray[np.float64],
) -> Mapping[str, float]:
    flattened = np.asarray(displacements, dtype=np.float64).reshape(
        len(displacements), -1
    )
    variance = np.var(flattened, axis=0)
    singular = np.linalg.svd(
        flattened - np.mean(flattened, axis=0),
        compute_uv=False,
    )
    energy = np.square(singular)
    total = float(np.sum(energy))
    if total <= 1e-18:
        effective_rank = 0.0
    else:
        probabilities = energy[energy > 0.0] / total
        effective_rank = float(
            np.exp(-np.sum(probabilities * np.log(probabilities)))
        )
    return {
        "mean_variance": float(np.mean(variance)),
        "effective_rank": effective_rank,
    }


def _schedule_is_pair_blocked(path: Path, pair_count: int) -> bool:
    with np.load(path, allow_pickle=False) as stored:
        indices = stored["indices"]
        arm_ids = stored["arm_ids"]
        pair_ids = stored["pair_ids"]
    return bool(
        indices.ndim == 2
        and indices.shape[1] == pair_count
        and arm_ids.shape == indices.shape
        and len(pair_ids) == pair_count
        and len(set(str(value) for value in pair_ids)) == pair_count
        and np.all(np.isin(arm_ids, (0, 1)))
    )


def _read_json(path: Path) -> Dict[str, Any]:
    return dict(json.loads(path.read_text()))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(arguments: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parsed = parser.parse_args(arguments)
    verify_artifact_manifest(parsed.directory)
    verify_stored_assessment(parsed.directory)
    print(
        json.dumps(
            assess_stored_bundle(parsed.directory),
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
