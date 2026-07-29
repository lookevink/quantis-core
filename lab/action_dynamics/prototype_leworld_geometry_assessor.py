#!/usr/bin/env python3
"""Fresh-process stored-array assessor for ticket 017."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from lab.action_dynamics.prototype_complete_lejepa import (
    _action_sanity_from_predictions,
    _attribution_scores_from_predictions,
    _downstream_pair_errors,
    _forecast_scores,
    _state_probe,
)
from quantis_core.action_conditioned_dynamics import ActionConditionedWindows
from quantis_core.edge_dynamics.data import PreparedAttributionQueries
from quantis_core.edge_dynamics.leworld_geometry import (
    LEWORLD_GEOMETRY_OBJECTIVES,
)
from quantis_core.graph_telemetry import DeclaredTelemetryGraph


CELL_NAMES = LEWORLD_GEOMETRY_OBJECTIVES
REGULARIZED_CELLS = CELL_NAMES[:-1]
EVALUATION_ROLES = ("selection", "iid_evaluation", "transfer_evaluation")


def assess_stored_bundle(directory: Path) -> Dict[str, Any]:
    """Recompute selection, metrics, and gates from retained arrays."""

    root = Path(directory)
    metadata = _read_json(root / "evidence-metadata.json")
    if (
        metadata.get("schema_version") != 1
        or metadata.get("kind") != "leworld_geometry_assessment_evidence"
    ):
        raise ValueError("unsupported LeWorld geometry evidence")
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
            not np.all(np.isfinite(stored[name])) for name in stored.files
        ):
            raise ValueError("LeWorld geometry evidence is non-finite")
        windows = {
            role: _restore_windows(
                stored,
                role,
                dict(metadata["roles"][role]),
                graph,
                entity_names,
                state_names,
                control_names,
                action_names,
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
                    stored[f"prediction__{name}__{role}"], windows[role]
                )
                for role in EVALUATION_ROLES
            }
            for name in CELL_NAMES
        }
        ridge_curves: Dict[str, Any] = {}
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
                    eligible or rows,
                    key=lambda row: (
                        row["downstream_effect_mse"],
                        row["ridge"],
                    ),
                )["ridge"]
            )

        safe_cells = [
            name
            for name in REGULARIZED_CELLS
            if next(
                row
                for row in ridge_curves[name]
                if row["ridge"] == selected_ridges[name]
            )["raw_safe"]
        ]
        winner_pool = safe_cells or list(REGULARIZED_CELLS)
        winner = min(
            winner_pool,
            key=lambda name: (
                forecast_scores[name]["selection"][
                    "downstream_effect_mse"
                ],
                name,
            ),
        )
        winner_selection_safe = winner in safe_cells

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
        geometry = {
            name: _geometry_diagnostics(
                stored[
                    f"representation__{name}__transfer_evaluation"
                ],
                stored[f"scene_history__{name}__transfer_evaluation"],
            )
            for name in CELL_NAMES
        }
        queries = _restore_queries(stored, dict(metadata["queries"]))
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
                "scene_history": bool(
                    np.allclose(
                        stored[f"restored_scene_history__{name}"],
                        stored[
                            f"scene_history__{name}__transfer_evaluation"
                        ][:8],
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
                        stored[
                            f"restored_attribution_prediction__{name}"
                        ],
                        stored[f"attribution_prediction__{name}"],
                        atol=1e-6,
                        rtol=0.0,
                    )
                ),
            }
            for name in CELL_NAMES
        }

    selection_verified = all(
        float(metadata["selected_ridges_runner"][name])
        == selected_ridges[name]
        for name in CELL_NAMES
    )
    screen_selection_verified = (
        metadata["screen_winner_runner"] == winner
        and bool(metadata["winner_selection_safe_runner"])
        == winner_selection_safe
    )
    parameter_counts = dict(metadata["parameter_counts"])
    equal_training = len(
        {
            int(dict(parameter_counts[name])["training"])
            for name in CELL_NAMES
        }
    ) == 1
    equal_inference = len(
        {
            int(dict(parameter_counts[name])["inference"])
            for name in CELL_NAMES
        }
    ) == 1
    winner_transfer = forecast_scores[winner]["transfer_evaluation"]
    raw_transfer = raw_scores["transfer_evaluation"]
    prediction_transfer = forecast_scores["prediction_only"][
        "transfer_evaluation"
    ]
    common_pairs = sorted(
        set(pair_errors[winner]) & set(pair_errors["prediction_only"])
    )
    win_fraction = float(
        np.mean(
            [
                pair_errors[winner][pair]
                < pair_errors["prediction_only"][pair]
                for pair in common_pairs
            ]
        )
    )
    safety = {
        "all_evidence_finite": True,
        "all_public_outputs_restore": all(
            all(row.values()) for row in restoration.values()
        ),
        "equal_training_capacity": equal_training,
        "equal_inference_capacity": equal_inference,
        "pair_blocked_schedule_valid": _schedule_is_pair_blocked(
            root / "anchor-schedule.npz",
            int(metadata["pair_counts"]["fit"]),
        ),
        "selection_only_ridge_choice_recomputes": selection_verified,
        "selection_only_geometry_choice_recomputes": (
            screen_selection_verified
        ),
        "winner_is_selection_safe": winner_selection_safe,
        "state_probe_within_1_05_pca": (
            state_probes[winner]["aggregate_nrmse"]
            <= 1.05 * state_probes["matched_pca"]["aggregate_nrmse"]
        ),
        "overall_mse_within_1_05_raw": (
            winner_transfer["overall_mse"]
            <= 1.05 * raw_transfer["overall_mse"]
        ),
        "action_overlap_mse_within_1_05_raw": (
            winner_transfer["action_overlap_mse"]
            <= 1.05 * raw_transfer["action_overlap_mse"]
        ),
        "action_and_target_hit_at_1_at_least_0_95": (
            attribution[winner]["action_and_target_hit_at_1"] >= 0.95
        ),
        "no_action_specificity_is_1": (
            attribution[winner]["no_action_specificity"] == 1.0
        ),
        "correct_action_beats_both_at_least_0_80": (
            action_sanity[winner][
                "correct_action_beats_both_fraction"
            ]
            >= 0.80
        ),
        "public_inference_is_causal": bool(
            metadata["public_causality"]
        ),
        "winner_bundle_at_most_16_mib": (
            int(metadata["bundle_bytes"][winner]) <= 16 * 1024 * 1024
        ),
    }
    geometry_gates = {
        "effective_rank_at_least_8": (
            geometry[winner]["effective_rank"] >= 8.0
        ),
        "selection_effect_strictly_below_prediction_only": (
            forecast_scores[winner]["selection"][
                "downstream_effect_mse"
            ]
            < forecast_scores["prediction_only"]["selection"][
                "downstream_effect_mse"
            ]
        ),
        "state_nrmse_no_worse_than_prediction_only": (
            state_probes[winner]["aggregate_nrmse"]
            <= state_probes["prediction_only"]["aggregate_nrmse"]
        ),
    }
    value_gates = {
        "transfer_effect_improves_prediction_only_by_5_percent": (
            winner_transfer["downstream_effect_mse"]
            <= 0.95 * prediction_transfer["downstream_effect_mse"]
        ),
        "transfer_effect_improves_raw_by_5_percent": (
            winner_transfer["downstream_effect_mse"]
            <= 0.95 * raw_transfer["downstream_effect_mse"]
        ),
        "pair_win_fraction_at_least_0_60": win_fraction >= 0.60,
    }
    otherwise_competitive = lambda name: all(
        forecast_scores[name]["transfer_evaluation"][metric]
        <= 1.05 * raw_transfer[metric]
        for metric in (
            "overall_mse",
            "action_overlap_mse",
            "downstream_effect_mse",
        )
    )
    ur_jepa_prerequisite_met = bool(
        geometry["lewm_ambient"]["effective_rank"] < 8.0
        and geometry["sub_jepa"]["effective_rank"] < 8.0
        and otherwise_competitive("lewm_ambient")
        and otherwise_competitive("sub_jepa")
    )
    safety_passed = all(safety.values())
    geometry_passed = all(geometry_gates.values())
    value_passed = all(value_gates.values())
    interpretable = bool(metadata["interpretable"])
    eligible = (
        interpretable and safety_passed and geometry_passed and value_passed
    )
    decision = (
        "non_interpretable_leworld_geometry_smoke"
        if not interpretable
        else (
            "advance_leworld_geometry_to_fixed_seed_robustness"
            if eligible
            else "reject_leworld_geometry_edge_recipe"
        )
    )
    return {
        "schema_version": 1,
        "kind": "leworld_geometry_stored_array_assessment",
        "interpretable": interpretable,
        "screen_winner": winner,
        "winner_selection_safe": winner_selection_safe,
        "forecast_scores": forecast_scores,
        "raw_low_rank_scores": raw_scores,
        "ridge_curves": ridge_curves,
        "selected_ridges": selected_ridges,
        "state_probes": state_probes,
        "geometry": geometry,
        "attribution": attribution,
        "action_sanity": action_sanity,
        "transfer_pair_errors": pair_errors,
        "restoration": restoration,
        "safety_gates": safety,
        "geometry_gates": geometry_gates,
        "downstream_value_gates": value_gates,
        "winner_pair_win_fraction": win_fraction,
        "ur_jepa_prerequisite_met": ur_jepa_prerequisite_met,
        "safety_passed": safety_passed,
        "geometry_passed": geometry_passed,
        "downstream_value_passed": value_passed,
        "eligible_for_advance": eligible,
        "passed": eligible,
        "decision": decision,
    }


def _geometry_diagnostics(
    values: np.ndarray, scene_history: np.ndarray
) -> Mapping[str, float]:
    flattened = np.asarray(values, dtype=np.float64).reshape(len(values), -1)
    centered = flattened - np.mean(flattened, axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = np.square(singular)
    if float(np.sum(energy)) <= 1e-18:
        effective_rank = 0.0
    else:
        probabilities = energy[energy > 0.0] / np.sum(energy)
        effective_rank = float(
            np.exp(-np.sum(probabilities * np.log(probabilities)))
        )
    norms = np.linalg.norm(values, axis=-1)
    mean_norm = float(np.mean(norms))
    differences = np.diff(scene_history, axis=1)
    left = differences[:, :-1]
    right = differences[:, 1:]
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(
        right, axis=-1
    )
    valid = denominator > 1e-12
    straightness = (
        float(np.mean(np.sum(left * right, axis=-1)[valid] / denominator[valid]))
        if np.any(valid)
        else 0.0
    )
    return {
        "mean_variance": float(np.mean(np.var(flattened, axis=0))),
        "effective_rank": effective_rank,
        "zero_fraction": float(np.mean(np.abs(values) <= 1e-8)),
        "mean_norm": mean_norm,
        "norm_coefficient_of_variation": (
            float(np.std(norms) / mean_norm) if mean_norm > 1e-12 else 0.0
        ),
        "temporal_straightness": straightness,
    }


def _restore_windows(
    stored: Any,
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
        trajectory_ids=tuple(identity["trajectory_ids"]),
        matched_pair_ids=tuple(identity["matched_pair_ids"]),
        transition_indices=np.asarray(
            identity["transition_indices"], dtype=np.int64
        ),
        entity_names=entity_names,
        state_feature_names=state_names,
        control_feature_names=control_names,
        action_feature_names=action_names,
        graph=graph,
    )


def _restore_queries(
    stored: Any, metadata: Mapping[str, Any]
) -> PreparedAttributionQueries:
    return PreparedAttributionQueries(
        query_ids=tuple(metadata["query_ids"]),
        histories=stored["query_histories"],
        future_controls=stored["query_future_controls"],
        observed_future=stored["query_observed_future"],
        candidate_actions=stored["query_candidate_actions"],
        candidate_ids=tuple(metadata["candidate_ids"]),
        candidate_action_kinds=tuple(metadata["candidate_action_kinds"]),
        candidate_target_entities=tuple(
            metadata["candidate_target_entities"]
        ),
        expected_action_kinds=tuple(metadata["expected_action_kinds"]),
        expected_target_entities=tuple(
            metadata["expected_target_entities"]
        ),
        expected_variant_ids=tuple(metadata["expected_variant_ids"]),
    )


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


def verify_stored_assessment(directory: Path) -> None:
    if _read_json(Path(directory) / "assessment.json") != assess_stored_bundle(
        directory
    ):
        raise ValueError("stored LeWorld geometry assessment differs")


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
        raise ValueError("LeWorld geometry manifest file set differs")
    for relative, raw in files.items():
        path = root / relative
        if (
            path.stat().st_size != int(raw["bytes"])
            or _file_sha256(path) != str(raw["sha256"])
        ):
            raise ValueError(
                f"LeWorld geometry manifest differs: {relative}"
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
    print(json.dumps(assess_stored_bundle(parsed.directory), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
