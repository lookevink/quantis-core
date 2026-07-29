#!/usr/bin/env python3
"""Fresh-process stored-array assessor for ticket 018."""

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
)
from quantis_core.action_conditioned_dynamics import ActionConditionedWindows
from quantis_core.edge_dynamics.causal_jepa import (
    CAUSAL_JEPA_OBJECTIVES,
)
from quantis_core.edge_dynamics.data import PreparedAttributionQueries
from quantis_core.graph_telemetry import DeclaredTelemetryGraph


CELL_NAMES = CAUSAL_JEPA_OBJECTIVES
CONTROL_NAMES = ("coordinate_time_mask", "prediction_only")
EVALUATION_ROLES = ("selection", "iid_evaluation", "transfer_evaluation")


def assess_stored_bundle(directory: Path) -> Dict[str, Any]:
    """Recompute every metric and gate from immutable evidence."""

    root = Path(directory)
    metadata = _read_json(root / "evidence-metadata.json")
    if (
        metadata.get("schema_version") != 1
        or metadata.get("kind") != "causal_jepa_assessment_evidence"
    ):
        raise ValueError("unsupported Causal-JEPA evidence")
    graph = DeclaredTelemetryGraph.from_dict(dict(metadata["graph"]))
    entity_names = tuple(metadata["entity_names"])
    state_names = tuple(metadata["state_feature_names"])
    control_names = tuple(metadata["control_feature_names"])
    action_names = tuple(metadata["action_feature_names"])
    ownership = np.asarray(metadata["ownership_mask"], dtype=np.bool_)
    with np.load(root / "evidence.npz", allow_pickle=False) as stored:
        if any(
            not np.all(np.isfinite(stored[name])) for name in stored.files
        ):
            raise ValueError("Causal-JEPA evidence is non-finite")
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
        completion = {
            name: _completion_scores(
                stored[f"completion_prediction__{name}"],
                windows["transfer_evaluation"],
                ownership,
            )
            for name in CELL_NAMES
        }
        completion["anchor_persistence"] = _completion_scores(
            np.repeat(
                windows["transfer_evaluation"].histories[
                    :, -6:-5
                ],
                5,
                axis=1,
            ).transpose(0, 2, 1, 3),
            windows["transfer_evaluation"],
            ownership,
        )
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
                "forecast": bool(
                    np.allclose(
                        stored[f"restored_prediction__{name}"],
                        stored[
                            f"prediction__{name}__transfer_evaluation"
                        ][:8],
                        atol=1e-6,
                        rtol=0.0,
                    )
                ),
                "completion": bool(
                    np.allclose(
                        stored[f"restored_completion__{name}"],
                        stored[f"completion_prediction__{name}"][:8],
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

    checkpoint_selection_valid = all(
        int(metadata["selected_steps"][name])
        == int(
            min(
                metadata["selection_metrics"][name],
                key=lambda row: (float(row["total"]), int(row["step"])),
            )["step"]
        )
        for name in CELL_NAMES
    )
    best_control = min(
        CONTROL_NAMES,
        key=lambda name: (
            forecast_scores[name]["selection"]["downstream_effect_mse"],
            name,
        ),
    )
    candidate_transfer = forecast_scores["causal_entity_mask"][
        "transfer_evaluation"
    ]
    control_transfer = forecast_scores[best_control]["transfer_evaluation"]
    raw_transfer = raw_scores["transfer_evaluation"]
    common_pairs = sorted(
        set(pair_errors["causal_entity_mask"]) & set(pair_errors[best_control])
    )
    win_fraction = float(
        np.mean(
            [
                pair_errors["causal_entity_mask"][pair]
                < pair_errors[best_control][pair]
                for pair in common_pairs
            ]
        )
    )
    counts = {
        int(metadata["parameter_counts"][name]) for name in CELL_NAMES
    }
    safety = {
        "all_evidence_finite": True,
        "all_outputs_restore": all(
            all(row.values()) for row in restoration.values()
        ),
        "equal_trainable_capacity": len(counts) == 1,
        "checkpoint_selection_recomputes": checkpoint_selection_valid,
        "pair_blocked_schedule_valid": _anchor_schedule_valid(
            root / "anchor-schedule.npz",
            int(metadata["pair_counts"]["fit"]),
        ),
        "matched_mask_schedules_valid": _mask_schedules_valid(
            root / "mask-schedule.npz"
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
            attribution["causal_entity_mask"][
                "action_and_target_hit_at_1"
            ]
            >= 0.95
        ),
        "no_action_specificity_is_1": (
            attribution["causal_entity_mask"]["no_action_specificity"] == 1.0
        ),
        "correct_action_beats_both_at_least_0_80": (
            action_sanity["causal_entity_mask"][
                "correct_action_beats_both_fraction"
            ]
            >= 0.80
        ),
        "public_inference_is_causal": bool(metadata["public_causality"]),
        "candidate_model_at_most_16_mib": (
            int(metadata["model_bytes"]["causal_entity_mask"])
            <= 16 * 1024 * 1024
        ),
    }
    candidate_completion = completion["causal_entity_mask"]["overall_mse"]
    mechanism = {
        "completion_improves_coordinate_by_10_percent": (
            candidate_completion
            <= 0.90 * completion["coordinate_time_mask"]["overall_mse"]
        ),
        "completion_improves_persistence_by_10_percent": (
            candidate_completion
            <= 0.90 * completion["anchor_persistence"]["overall_mse"]
        ),
    }
    value = {
        "selection_effect_strictly_best_neural": all(
            forecast_scores["causal_entity_mask"]["selection"][
                "downstream_effect_mse"
            ]
            < forecast_scores[name]["selection"]["downstream_effect_mse"]
            for name in CONTROL_NAMES
        ),
        "transfer_effect_improves_best_control_by_5_percent": (
            candidate_transfer["downstream_effect_mse"]
            <= 0.95 * control_transfer["downstream_effect_mse"]
        ),
        "transfer_effect_improves_raw_by_5_percent": (
            candidate_transfer["downstream_effect_mse"]
            <= 0.95 * raw_transfer["downstream_effect_mse"]
        ),
        "pair_win_fraction_at_least_0_60": win_fraction >= 0.60,
    }
    safety_passed = all(safety.values())
    mechanism_passed = all(mechanism.values())
    value_passed = all(value.values())
    interpretable = bool(metadata["interpretable"])
    eligible = (
        interpretable and safety_passed and mechanism_passed and value_passed
    )
    decision = (
        "non_interpretable_causal_jepa_smoke"
        if not interpretable
        else (
            "advance_causal_jepa_to_fixed_seed_robustness"
            if eligible
            else "reject_causal_jepa_edge_recipe"
        )
    )
    return {
        "schema_version": 1,
        "kind": "causal_jepa_stored_array_assessment",
        "interpretable": interpretable,
        "forecast_scores": forecast_scores,
        "raw_low_rank_scores": raw_scores,
        "completion": completion,
        "attribution": attribution,
        "action_sanity": action_sanity,
        "transfer_pair_errors": pair_errors,
        "restoration": restoration,
        "best_neural_control": best_control,
        "candidate_pair_win_fraction": win_fraction,
        "safety_gates": safety,
        "interaction_mechanism_gates": mechanism,
        "downstream_value_gates": value,
        "safety_passed": safety_passed,
        "interaction_mechanism_passed": mechanism_passed,
        "downstream_value_passed": value_passed,
        "eligible_for_advance": eligible,
        "passed": eligible,
        "decision": decision,
    }


def _completion_scores(
    prediction: np.ndarray,
    windows: ActionConditionedWindows,
    ownership: np.ndarray,
) -> Mapping[str, float]:
    target = windows.histories[:, -5:].transpose(0, 2, 1, 3)
    squared = np.square(prediction - target)
    selected = squared[
        np.broadcast_to(
            ownership[None, :, None, :], squared.shape
        )
    ].reshape(len(prediction), -1)
    treatment = np.any(
        windows.future_actions[..., 1] > 0.5, axis=(1, 2)
    )
    return {
        "overall_mse": float(np.mean(selected)),
        "treatment_mse": float(np.mean(selected[treatment])),
    }


def _anchor_schedule_valid(path: Path, pair_count: int) -> bool:
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


def _mask_schedules_valid(path: Path) -> bool:
    with np.load(path, allow_pickle=False) as stored:
        entity = stored["causal_entity_mask"].astype(np.bool_)
        coordinate = stored["coordinate_time_mask"].astype(np.bool_)
        prediction = stored["prediction_only"].astype(np.bool_)
    entity_columns = np.sum(np.all(entity[:, 1:], axis=1), axis=1)
    return bool(
        entity.shape == coordinate.shape == prediction.shape
        and entity.ndim == 3
        and entity.shape[1:] == (6, 7)
        and not np.any(entity[:, 0])
        and not np.any(coordinate[:, 0])
        and np.all(np.sum(entity, axis=(1, 2)) == 10)
        and np.all(np.sum(coordinate, axis=(1, 2)) == 10)
        and np.all(entity_columns == 2)
        and not np.any(prediction)
        and np.any(entity != coordinate)
    )


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


def verify_stored_assessment(directory: Path) -> None:
    if _read_json(Path(directory) / "assessment.json") != assess_stored_bundle(
        directory
    ):
        raise ValueError("stored Causal-JEPA assessment differs")


def verify_artifact_manifest(directory: Path) -> None:
    root = Path(directory)
    files = dict(_read_json(root / "artifact-manifest.json")["files"])
    observed = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name != "artifact-manifest.json"
    }
    if observed != set(files):
        raise ValueError("Causal-JEPA manifest file set differs")
    for relative, raw in files.items():
        path = root / relative
        if (
            path.stat().st_size != int(raw["bytes"])
            or _file_sha256(path) != str(raw["sha256"])
        ):
            raise ValueError(f"Causal-JEPA manifest differs: {relative}")


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
