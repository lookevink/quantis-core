#!/usr/bin/env python3
"""Fresh-process stored-array assessor for ticket 019."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from lab.action_dynamics.prototype_complete_lejepa import (
    _downstream_pair_errors,
)
from quantis_core.action_conditioned_dynamics import (
    ActionConditionedWindows,
    MixtureTrajectoryDistribution,
)
from quantis_core.graph_telemetry import DeclaredTelemetryGraph


MODEL_NAMES = (
    "mop_jepa",
    "dense_jepa",
    "supervised_hard_wta",
    "context_free_codebook",
    "raw_low_rank",
)
EVALUATION_ROLES = ("selection", "transfer_evaluation")


def assess_stored_bundle(directory: Path) -> Dict[str, Any]:
    """Recompute all MoP-JEPA metrics and gates from immutable arrays."""

    root = Path(directory)
    metadata = _read_json(root / "evidence-metadata.json")
    if (
        metadata.get("schema_version") != 1
        or metadata.get("kind") != "mop_jepa_assessment_evidence"
    ):
        raise ValueError("unsupported MoP-JEPA evidence")
    graph = DeclaredTelemetryGraph.from_dict(dict(metadata["graph"]))
    with np.load(root / "evidence.npz", allow_pickle=False) as stored:
        if any(
            not np.all(np.isfinite(stored[name])) for name in stored.files
        ):
            raise ValueError("MoP-JEPA evidence is non-finite")
        windows = {
            role: _restore_windows(
                stored, role, dict(metadata["roles"][role]), metadata, graph
            )
            for role in ("calibration",) + EVALUATION_ROLES
        }
        raw_calibration = _distribution(
            stored,
            "raw_low_rank",
            "calibration",
            len(windows["calibration"].histories),
        )
        raw_calibration_error = np.sqrt(
            np.mean(
                np.square(
                    raw_calibration.as_trajectory_distribution().mean
                    - windows["calibration"].future_states
                ),
                axis=(1, 2, 3),
            )
        )
        transition_radius = float(
            np.quantile(raw_calibration_error, 0.95)
        )
        metrics = {
            role: {
                name: _distribution_metrics(
                    _distribution(
                        stored,
                        name,
                        role,
                        len(windows[role].histories),
                    ),
                    windows[role],
                    transition_radius,
                    router_gated=(name != "context_free_codebook"),
                )
                for name in MODEL_NAMES
            }
            for role in EVALUATION_ROLES
        }
        shuffled_metrics = {
            role: _distribution_metrics(
                _distribution(
                    stored,
                    "mop_jepa_shuffled",
                    role,
                    len(windows[role].histories),
                ),
                windows[role],
                transition_radius,
                router_gated=True,
            )
            for role in EVALUATION_ROLES
        }
        restoration = {
            name: _restoration_matches(stored, name)
            for name in MODEL_NAMES
        }

    selection = metrics["selection"]
    transfer = metrics["transfer_evaluation"]
    candidate = selection["mop_jepa"]
    candidate_transfer = transfer["mop_jepa"]
    raw = selection["raw_low_rank"]
    raw_transfer = transfer["raw_low_rank"]
    dense = selection["dense_jepa"]
    supervised = selection["supervised_hard_wta"]
    codebook = selection["context_free_codebook"]
    capacities = dict(metadata["parameter_counts"])
    safety = {
        "all_evidence_finite": True,
        "all_outputs_restore_within_1e_6": all(restoration.values()),
        "candidate_matches_supervised_capacity": (
            int(capacities["mop_jepa"])
            == int(capacities["supervised_hard_wta"])
        ),
        "public_inference_is_causal": bool(metadata["public_causality"]),
        "candidate_model_at_most_16_mib": (
            int(metadata["model_bytes"]["mop_jepa"])
            <= 16 * 1024 * 1024
        ),
        "selection_overall_mse_within_1_05_raw": (
            candidate["point_overall_mse"]
            <= 1.05 * raw["point_overall_mse"]
        ),
        "selection_action_mse_within_1_05_raw": (
            candidate["point_action_overlap_mse"]
            <= 1.05 * raw["point_action_overlap_mse"]
        ),
        "transfer_overall_mse_within_1_05_raw": (
            candidate_transfer["point_overall_mse"]
            <= 1.05 * raw_transfer["point_overall_mse"]
        ),
        "transfer_action_mse_within_1_05_raw": (
            candidate_transfer["point_action_overlap_mse"]
            <= 1.05 * raw_transfer["point_action_overlap_mse"]
        ),
    }
    mechanism = {
        "winner_usage_effective_heads_at_least_2": (
            candidate["winner_usage_effective_heads"] >= 2.0
        ),
        "router_effective_heads_at_least_1_5": (
            candidate["router_effective_heads"] >= 1.5
        ),
        "gated_realized_transition_precision_at_least_0_80": (
            candidate["gated_realized_transition_precision"] >= 0.80
        ),
        "oracle_improves_dense_point_by_10_percent": (
            candidate["oracle_mse"]
            <= 0.90 * dense["point_overall_mse"]
        ),
        "gated_oracle_improves_codebook_by_10_percent": (
            candidate["gated_oracle_mse"]
            <= 0.90 * codebook["oracle_mse"]
        ),
        "correct_context_improves_shuffle_by_10_percent": (
            candidate["oracle_mse"]
            <= 0.90 * shuffled_metrics["selection"]["oracle_mse"]
        ),
    }
    candidate_pair = _downstream_pair_errors(
        _distribution_mean(root, "mop_jepa", "transfer_evaluation"),
        windows["transfer_evaluation"],
    )
    raw_pair = _downstream_pair_errors(
        _distribution_mean(root, "raw_low_rank", "transfer_evaluation"),
        windows["transfer_evaluation"],
    )
    common = sorted(set(candidate_pair) & set(raw_pair))
    pair_win_fraction = float(
        np.mean(
            [candidate_pair[pair] < raw_pair[pair] for pair in common]
        )
    )
    value = {
        "selection_nll_improves_dense_by_0_01": (
            candidate["trajectory_nll"]
            <= dense["trajectory_nll"] - 0.01
        ),
        "selection_nll_improves_supervised_wta_by_0_01": (
            candidate["trajectory_nll"]
            <= supervised["trajectory_nll"] - 0.01
        ),
        "transfer_effect_improves_raw_by_5_percent": (
            candidate_transfer["downstream_effect_mse"]
            <= 0.95 * raw_transfer["downstream_effect_mse"]
        ),
        "transfer_pair_win_fraction_at_least_0_60": (
            pair_win_fraction >= 0.60
        ),
    }
    safety_passed = all(safety.values())
    mechanism_passed = all(mechanism.values())
    value_passed = all(value.values())
    interpretable = bool(metadata["interpretable"])
    eligible = (
        interpretable and safety_passed and mechanism_passed and value_passed
    )
    decision = (
        "non_interpretable_mop_jepa_smoke"
        if not interpretable
        else (
            "advance_mop_jepa_to_fixed_seed_robustness"
            if eligible
            else "reject_mop_jepa_edge_recipe"
        )
    )
    return {
        "schema_version": 1,
        "kind": "mop_jepa_stored_array_assessment",
        "interpretable": interpretable,
        "transition_radius": transition_radius,
        "metrics": metrics,
        "shuffled_context_metrics": shuffled_metrics,
        "restoration": restoration,
        "candidate_transfer_pair_win_fraction": pair_win_fraction,
        "safety_gates": safety,
        "hard_assignment_mechanism_gates": mechanism,
        "downstream_value_gates": value,
        "safety_passed": safety_passed,
        "hard_assignment_mechanism_passed": mechanism_passed,
        "downstream_value_passed": value_passed,
        "eligible_for_advance": eligible,
        "passed": eligible,
        "decision": decision,
    }


def _distribution_metrics(
    distribution: MixtureTrajectoryDistribution,
    windows: ActionConditionedWindows,
    transition_radius: float,
    *,
    router_gated: bool,
) -> Mapping[str, float]:
    observed = np.asarray(windows.future_states, dtype=np.float64)
    component_error = np.mean(
        np.square(distribution.component_mean - observed[:, None]),
        axis=(2, 3, 4),
    )
    point = distribution.as_trajectory_distribution().mean
    squared = np.square(point - observed)
    row_mse = np.mean(squared, axis=(1, 2, 3))
    active_steps = np.any(
        windows.future_actions[..., 1] > 0.5, axis=2
    )
    action_mse = np.asarray(
        [
            (
                np.mean(squared[row][active_steps[row]])
                if np.any(active_steps[row])
                else np.nan
            )
            for row in range(len(squared))
        ]
    )
    gated = (
        distribution.weight
        > 0.5 / distribution.component_mean.shape[1]
        if router_gated
        else np.ones(distribution.weight.shape, dtype=np.bool_)
    )
    gated_error = np.where(gated, component_error, np.inf)
    oracle = np.min(component_error, axis=1)
    gated_oracle = np.min(gated_error, axis=1)
    valid = np.sqrt(component_error) <= transition_radius
    precision = float(np.sum(valid & gated) / np.sum(gated))
    winner = np.argmin(component_error, axis=1)
    usage = np.bincount(
        winner, minlength=distribution.component_mean.shape[1]
    ).astype(np.float64)
    usage /= np.sum(usage)
    positive_usage = usage[usage > 0.0]
    winner_effective = float(
        np.exp(-np.sum(positive_usage * np.log(positive_usage)))
    )
    router_effective = float(
        np.mean(
            np.exp(
                -np.sum(
                    distribution.weight
                    * np.log(np.maximum(distribution.weight, 1e-300)),
                    axis=1,
                )
            )
        )
    )
    nll = distribution.negative_log_likelihood(observed)
    return {
        "trajectory_nll": _pair_balanced(
            nll, windows.matched_pair_ids
        ),
        "point_overall_mse": _pair_balanced(
            row_mse, windows.matched_pair_ids
        ),
        "point_action_overlap_mse": _pair_balanced(
            action_mse, windows.matched_pair_ids
        ),
        "downstream_effect_mse": float(
            np.mean(
                tuple(_downstream_pair_errors(point, windows).values())
            )
        ),
        "oracle_mse": _pair_balanced(
            oracle, windows.matched_pair_ids
        ),
        "gated_oracle_mse": _pair_balanced(
            gated_oracle, windows.matched_pair_ids
        ),
        "gated_realized_transition_precision": precision,
        "winner_usage_effective_heads": winner_effective,
        "router_effective_heads": router_effective,
        "mean_active_head_count": float(np.mean(np.sum(gated, axis=1))),
    }


def _distribution(
    stored: Any, name: str, role: str, count: int
) -> MixtureTrajectoryDistribution:
    mean = np.asarray(stored[f"mean__{name}__{role}"], dtype=np.float64)
    weight = np.asarray(
        stored[f"weight__{name}__{role}"], dtype=np.float64
    )
    variance = np.asarray(stored[f"variance__{name}"], dtype=np.float64)
    return MixtureTrajectoryDistribution(
        component_mean=mean,
        component_variance=np.broadcast_to(
            variance[None], (count,) + variance.shape
        ),
        weight=weight,
    )


def _distribution_mean(
    directory: Path, name: str, role: str
) -> np.ndarray:
    with np.load(
        Path(directory) / "evidence.npz", allow_pickle=False
    ) as stored:
        mean = np.asarray(
            stored[f"mean__{name}__{role}"], dtype=np.float64
        )
        weight = np.asarray(
            stored[f"weight__{name}__{role}"], dtype=np.float64
        )
    return np.sum(weight[:, :, None, None, None] * mean, axis=1)


def _restoration_matches(stored: Any, name: str) -> bool:
    return bool(
        np.allclose(
            stored[f"restored_mean__{name}"],
            stored[f"mean__{name}__selection"][:8],
            atol=1e-6,
            rtol=0.0,
        )
        and np.allclose(
            stored[f"restored_weight__{name}"],
            stored[f"weight__{name}__selection"][:8],
            atol=1e-6,
            rtol=0.0,
        )
        and np.allclose(
            stored[f"restored_variance__{name}"],
            stored[f"variance__{name}"],
            atol=1e-7,
            rtol=0.0,
        )
    )


def _pair_balanced(
    values: np.ndarray, pair_ids: Tuple[str, ...]
) -> float:
    grouped: Dict[str, list[float]] = {}
    for value, pair in zip(values, pair_ids):
        if np.isfinite(value):
            grouped.setdefault(pair, []).append(float(value))
    if not grouped or any(not rows for rows in grouped.values()):
        raise ValueError("pair-balanced metric has no finite rows")
    return float(np.mean([np.mean(grouped[key]) for key in sorted(grouped)]))


def _restore_windows(
    stored: Any,
    role: str,
    identity: Mapping[str, Any],
    metadata: Mapping[str, Any],
    graph: DeclaredTelemetryGraph,
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
        entity_names=tuple(metadata["entity_names"]),
        state_feature_names=tuple(metadata["state_feature_names"]),
        control_feature_names=tuple(metadata["control_feature_names"]),
        action_feature_names=tuple(metadata["action_feature_names"]),
        graph=graph,
    )


def verify_stored_assessment(directory: Path) -> None:
    if _read_json(Path(directory) / "assessment.json") != assess_stored_bundle(
        directory
    ):
        raise ValueError("stored MoP-JEPA assessment differs")


def verify_artifact_manifest(directory: Path) -> None:
    root = Path(directory)
    files = dict(_read_json(root / "artifact-manifest.json")["files"])
    observed = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name != "artifact-manifest.json"
    }
    if observed != set(files):
        raise ValueError("MoP-JEPA manifest file set differs")
    for relative, raw in files.items():
        path = root / relative
        if (
            path.stat().st_size != int(raw["bytes"])
            or _file_sha256(path) != str(raw["sha256"])
        ):
            raise ValueError(f"MoP-JEPA manifest differs: {relative}")


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
