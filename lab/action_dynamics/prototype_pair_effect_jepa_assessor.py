#!/usr/bin/env python3
"""Independent stored-evidence assessor for PairEffect-JEPA."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np


CELL_NAMES = (
    "pair_effect_jepa",
    "supervised_pair_effect",
    "deranged_pair_jepa",
)


def assess_stored_bundle(directory: Path) -> Mapping[str, Any]:
    """Recompute every PairEffect-JEPA gate from stored arrays."""

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
    roles = {}
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
            scores[name] = _forecast_scores(
                prediction,
                target,
                actions,
                pair_ids,
                trajectory_ids,
                transitions,
                metadata["graph"],
            )
            pair_errors[name] = _downstream_pair_errors(
                prediction,
                target,
                actions,
                pair_ids,
                trajectory_ids,
                transitions,
                metadata["graph"],
            )
        effect_mse = {
            name: float(
                np.mean(
                    np.square(
                        arrays[f"effect_prediction__{name}__{role}"]
                        - arrays[f"effect_target__{role}"]
                    )[..., ownership]
                )
            )
            for name in CELL_NAMES
        }
        roles[role] = {
            "scores": scores,
            "pair_errors": pair_errors,
            "observable_effect_mse": effect_mse,
        }
    attribution = _attribution_scores(
        arrays["query_prediction__pair_effect_jepa"],
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
    action_sanity = _action_sanity_scores(
        {
            variant: arrays[
                f"action_sanity__pair_effect_jepa__{variant}"
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
    transfer = roles["transfer_evaluation"]
    selection = roles["selection"]
    candidate_scores = transfer["scores"]["pair_effect_jepa"]
    raw_scores = transfer["scores"]["raw"]
    supervised_scores = transfer["scores"]["supervised_pair_effect"]
    candidate_pair = transfer["pair_errors"]["pair_effect_jepa"]
    supervised_pair = transfer["pair_errors"]["supervised_pair_effect"]
    common = sorted(set(candidate_pair) & set(supervised_pair))
    win_fraction = float(
        np.mean(
            [
                candidate_pair[pair] < supervised_pair[pair]
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
    safety = {
        "all_evidence_is_finite": finite,
        "capacity_is_matched": (
            len(training_counts) == 1 and len(inference_counts) == 1
        ),
        "restoration_max_abs_at_most_1e_6": float(
            metadata["restoration_max_abs"]
        )
        <= 1e-6,
        "public_inference_is_causal": bool(
            metadata["public_causality"]
        ),
        "no_action_correction_is_zero": float(
            metadata["zero_effect_max_abs"]
        )
        <= 1e-7,
        "transfer_overall_is_raw_safe": float(
            candidate_scores["overall_mse"]
        )
        <= 1.05 * float(raw_scores["overall_mse"]),
        "transfer_action_overlap_is_raw_safe": float(
            candidate_scores["action_overlap_mse"]
        )
        <= 1.05 * float(raw_scores["action_overlap_mse"]),
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
        "batch_one_latency_recorded": (
            float(
                metadata["latency"]["pair_effect_jepa"]["median_ms"]
            )
            >= 0.0
        ),
    }
    mechanism = {
        "selection_effect_mse_beats_deranged_by_10_percent": float(
            selection["observable_effect_mse"]["pair_effect_jepa"]
        )
        <= 0.90
        * float(
            selection["observable_effect_mse"]["deranged_pair_jepa"]
        ),
        "transfer_effect_mse_beats_deranged_by_10_percent": float(
            transfer["observable_effect_mse"]["pair_effect_jepa"]
        )
        <= 0.90
        * float(
            transfer["observable_effect_mse"]["deranged_pair_jepa"]
        ),
    }
    value = {
        "transfer_downstream_beats_raw_by_10_percent": float(
            candidate_scores["downstream_effect_mse"]
        )
        <= 0.90 * float(raw_scores["downstream_effect_mse"]),
        "transfer_downstream_beats_supervised_by_10_percent": float(
            candidate_scores["downstream_effect_mse"]
        )
        <= 0.90
        * float(supervised_scores["downstream_effect_mse"]),
        "transfer_pair_win_fraction_at_least_0_60": win_fraction >= 0.60,
        "selection_downstream_beats_supervised": float(
            selection["scores"]["pair_effect_jepa"][
                "downstream_effect_mse"
            ]
        )
        < float(
            selection["scores"]["supervised_pair_effect"][
                "downstream_effect_mse"
            ]
        ),
    }
    interpretable = bool(metadata["interpretable"])
    gates_pass = (
        all(safety.values())
        and all(mechanism.values())
        and all(value.values())
    )
    passed = interpretable and gates_pass
    if not interpretable:
        decision = "non_interpretable_pair_effect_jepa_smoke"
    elif passed:
        decision = "advance_pair_effect_jepa_to_fixed_seed_robustness"
    else:
        decision = "reject_pair_effect_jepa_recipe"
    return {
        "schema_version": 1,
        "kind": "pair_effect_jepa_independent_assessment",
        "eligible_for_advance": interpretable,
        "roles": roles,
        "attribution": attribution,
        "action_sanity": action_sanity,
        "candidate_pair_win_fraction": win_fraction,
        "safety_gates": safety,
        "mechanism_gates": mechanism,
        "value_gates": value,
        "safety_passed": all(safety.values()),
        "mechanism_passed": all(mechanism.values()),
        "value_passed": all(value.values()),
        "passed": passed,
        "decision": decision,
    }


def verify_stored_assessment(directory: Path) -> None:
    """Require the stored assessment to equal fresh recomputation."""

    root = Path(directory)
    expected = _read_json(root / "assessment.json")
    actual = assess_stored_bundle(root)
    if _canonical_json(expected) != _canonical_json(actual):
        raise ValueError("stored PairEffect-JEPA assessment differs")


def verify_artifact_manifest(directory: Path) -> None:
    """Verify every artifact file bound by the SHA-256 manifest."""

    root = Path(directory)
    manifest = _read_json(root / "artifact-manifest.json")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "pair_effect_jepa_manifest"
    ):
        raise ValueError("PairEffect-JEPA artifact manifest is invalid")
    expected = dict(manifest["sha256"])
    actual = {
        path.relative_to(root).as_posix(): _file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.name != "artifact-manifest.json"
    }
    if expected != actual:
        raise ValueError("PairEffect-JEPA artifact manifest differs")


def _forecast_scores(
    prediction: np.ndarray,
    target: np.ndarray,
    actions: np.ndarray,
    pair_ids: Sequence[str],
    trajectory_ids: Sequence[str],
    transitions: np.ndarray,
    graph: Mapping[str, Any],
) -> Mapping[str, float]:
    squared = np.square(prediction - target)
    row_mse = np.mean(squared, axis=(1, 2, 3))
    active = np.any(actions[..., 1] > 0.5, axis=2)
    action_rows = np.asarray(
        [
            np.mean(squared[index][active[index]])
            if np.any(active[index])
            else np.nan
            for index in range(len(squared))
        ],
        dtype=np.float64,
    )
    pair_errors = _downstream_pair_errors(
        prediction,
        target,
        actions,
        pair_ids,
        trajectory_ids,
        transitions,
        graph,
    )
    return {
        "overall_mse": _pair_balanced(row_mse, pair_ids),
        "action_overlap_mse": _pair_balanced(action_rows, pair_ids),
        "downstream_effect_mse": float(
            np.mean(list(pair_errors.values()))
        ),
    }


def _downstream_pair_errors(
    prediction: np.ndarray,
    target: np.ndarray,
    actions: np.ndarray,
    pair_ids: Sequence[str],
    trajectory_ids: Sequence[str],
    transitions: np.ndarray,
    graph: Mapping[str, Any],
) -> Mapping[str, float]:
    index = {
        (trajectory, int(transition)): position
        for position, (trajectory, transition) in enumerate(
            zip(trajectory_ids, transitions)
        )
    }
    trajectories: Dict[str, list[str]] = {}
    treatment_target: Dict[str, int] = {}
    for row, (pair, trajectory) in enumerate(
        zip(pair_ids, trajectory_ids)
    ):
        if trajectory not in trajectories.setdefault(pair, []):
            trajectories[pair].append(trajectory)
        active = np.argwhere(actions[row, ..., 1] > 0.5)
        if len(active):
            treatment_target[trajectory] = int(active[0, 1])
    rows: Dict[str, float] = {}
    for pair, pair_trajectories in trajectories.items():
        treatment = [
            value
            for value in pair_trajectories
            if value in treatment_target
        ]
        control = [
            value
            for value in pair_trajectories
            if value not in treatment_target
        ]
        if len(treatment) != 1 or len(control) != 1:
            continue
        downstream = _downstream_positions(
            graph, treatment_target[treatment[0]]
        )
        errors = []
        for row, trajectory in enumerate(trajectory_ids):
            if trajectory != treatment[0]:
                continue
            active = np.any(actions[row, ..., 1] > 0.5, axis=1)
            other = index.get(
                (control[0], int(transitions[row]))
            )
            if other is None or not np.any(active) or not downstream:
                continue
            predicted_effect = prediction[row] - prediction[other]
            observed_effect = target[row] - target[other]
            errors.append(
                float(
                    np.mean(
                        np.square(
                            predicted_effect[active][:, downstream]
                            - observed_effect[active][:, downstream]
                        )
                    )
                )
            )
        if errors:
            rows[pair] = float(np.mean(errors))
    if not rows:
        raise ValueError("PairEffect-JEPA has no downstream pair errors")
    return rows


def _downstream_positions(
    graph: Mapping[str, Any], start: int
) -> Tuple[int, ...]:
    entities = tuple(dict(value) for value in graph["entities"])
    names = tuple(str(value["entity_id"]) for value in entities)
    adjacency: Dict[str, list[str]] = {name: [] for name in names}
    for entity in entities:
        if entity["kind"] == "edge":
            adjacency[str(entity["source"])].append(
                str(entity["entity_id"])
            )
            adjacency[str(entity["entity_id"])].append(
                str(entity["target"])
            )
    start_name = names[start]
    discovered = []
    frontier = list(adjacency[start_name])
    while frontier:
        candidate = frontier.pop(0)
        if candidate in discovered or candidate == start_name:
            continue
        discovered.append(candidate)
        frontier.extend(adjacency[candidate])
    return tuple(names.index(value) for value in discovered)


def _pair_balanced(
    values: np.ndarray, pair_ids: Sequence[str]
) -> float:
    pair_array = np.asarray(pair_ids)
    rows = []
    for pair in sorted(set(pair_ids)):
        local = values[pair_array == pair]
        local = local[np.isfinite(local)]
        if len(local):
            rows.append(float(np.mean(local)))
    return float(np.mean(rows))


def _attribution_scores(
    prediction: np.ndarray,
    observed: np.ndarray,
    candidate_ids: Sequence[str],
    candidate_kinds: Sequence[str],
    candidate_targets: Sequence[str],
    expected_kinds: Sequence[str],
    expected_targets: Sequence[str],
    ownership: np.ndarray,
) -> Mapping[str, float]:
    treatment_hits = []
    control_hits = []
    for index in range(len(observed)):
        error = np.mean(
            np.square(prediction[index] - observed[index][None])[
                ..., ownership
            ],
            axis=(1, 2),
        )
        winner = int(np.argmin(error))
        if expected_kinds[index]:
            treatment_hits.append(
                candidate_kinds[winner] == expected_kinds[index]
                and candidate_targets[winner] == expected_targets[index]
            )
        else:
            control_hits.append(candidate_ids[winner] == "no_action")
    return {
        "action_and_target_hit_at_1": float(np.mean(treatment_hits)),
        "no_action_specificity": float(np.mean(control_hits)),
    }


def _action_sanity_scores(
    predictions: Mapping[str, np.ndarray],
    target: np.ndarray,
    actions: np.ndarray,
    pair_ids: Sequence[str],
    ownership: np.ndarray,
) -> Mapping[str, float]:
    pair_array = np.asarray(pair_ids)
    wins = []
    for pair in sorted(set(pair_ids)):
        rows = np.flatnonzero(pair_array == pair)
        rows = rows[
            np.any(actions[rows, ..., 1] > 0.5, axis=(1, 2))
        ]
        if not len(rows):
            continue
        scores = {
            name: float(
                np.mean(
                    np.square(value[rows] - target[rows])[
                        ..., ownership
                    ]
                )
            )
            for name, value in predictions.items()
        }
        wins.append(
            scores["correct"] < scores["no_action"]
            and scores["correct"] < scores["shuffled"]
        )
    return {
        "treatment_pair_count": len(wins),
        "correct_action_beats_both_fraction": float(np.mean(wins)),
    }


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

