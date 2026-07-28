"""Bounded evaluation helpers for action-conditioned JEPA development."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
from numpy.typing import NDArray

from ..action_conditioned_dynamics import ActionConditionedWindows
from .action_conditioned_jepa import ActionConditionedJepaDynamics
from .models import EdgeDynamicsModel


def write_action_conditioned_jepa_artifacts(
    *,
    output_directory: Path,
    report: Mapping[str, Any],
    model_artifacts: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Write one immutable JEPA development evidence bundle."""

    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite JEPA development: {output}"
        )
    output.mkdir(parents=True)
    models = output / "models"
    models.mkdir()
    (output / "results.json").write_text(_pretty_json(report))
    (output / "report.md").write_text(_markdown_report(report))
    for name, artifact in model_artifacts.items():
        (models / f"{name}.json").write_text(
            _pretty_json(artifact)
        )
    hashes = {
        path.relative_to(output).as_posix(): _file_sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "schema_version": 1,
        "kind": (
            "action_conditioned_jepa_low_rank_development_manifest"
        ),
        "sha256": hashes,
    }
    (output / "artifact-manifest.json").write_text(
        _pretty_json(manifest)
    )
    return manifest


def node_token_diagnostics(
    model: ActionConditionedJepaDynamics,
    windows: ActionConditionedWindows,
) -> Mapping[str, Any]:
    """Measure collapse over per-node rather than flattened system tokens."""

    embeddings = model.encode_histories(
        windows.histories, windows.graph
    )
    values = embeddings.reshape(-1, embeddings.shape[-1])
    overall = _latent_diagnostics(values)
    by_entity: Dict[str, Mapping[str, Any]] = {}
    observed_ranks = []
    for position, entity_id in enumerate(windows.entity_names):
        diagnostics = dict(
            _latent_diagnostics(embeddings[:, position])
        )
        observed_variance = float(
            np.max(
                np.var(
                    np.asarray(
                        windows.histories[:, :, position],
                        dtype=np.float64,
                    ),
                    axis=(0, 1),
                )
            )
        )
        observed = observed_variance > 1e-12
        diagnostics["has_varying_observation"] = observed
        diagnostics["maximum_observed_feature_variance"] = (
            observed_variance
        )
        by_entity[entity_id] = diagnostics
        if observed:
            observed_ranks.append(
                float(diagnostics["effective_rank"])
            )
    if not observed_ranks:
        raise ValueError("token diagnostics have no observed entities")
    return {
        **overall,
        "latent_dimension": int(embeddings.shape[-1]),
        "entity_count": int(embeddings.shape[1]),
        "observed_entity_count": len(observed_ranks),
        "minimum_observed_entity_effective_rank": min(
            observed_ranks
        ),
        "by_entity": by_entity,
    }


def action_conditioning_sanity(
    model: EdgeDynamicsModel,
    windows: ActionConditionedWindows,
    *,
    seed: int = 401,
) -> Mapping[str, Any]:
    """Compare correct actions with no-action and whole-pair shuffles."""

    actions = np.asarray(windows.future_actions, dtype=np.float64)
    try:
        no_action_position = windows.action_feature_names.index(
            "no_action"
        )
        applicable_position = windows.action_feature_names.index(
            "applicable"
        )
    except ValueError as error:
        raise ValueError(
            "action sanity requires no_action and applicable features"
        ) from error
    treatment = np.any(
        actions[..., applicable_position] > 0.5, axis=(1, 2)
    )
    if not np.any(treatment):
        raise ValueError("action sanity requires treatment windows")
    no_actions = np.zeros_like(actions)
    no_actions[..., no_action_position] = 1.0
    shuffled_actions = _shuffle_actions_by_pair(
        windows, actions, seed=seed
    )
    observed = np.asarray(windows.future_states, dtype=np.float64)
    correct = model.rollout(
        windows.histories,
        windows.future_controls,
        actions,
        windows.graph,
    ).mean
    absent = model.rollout(
        windows.histories,
        windows.future_controls,
        no_actions,
        windows.graph,
    ).mean
    shuffled = model.rollout(
        windows.histories,
        windows.future_controls,
        shuffled_actions,
        windows.graph,
    ).mean
    correct_error = np.mean(
        np.square(correct - observed), axis=(1, 2, 3)
    )
    absent_error = np.mean(
        np.square(absent - observed), axis=(1, 2, 3)
    )
    shuffled_error = np.mean(
        np.square(shuffled - observed), axis=(1, 2, 3)
    )
    pair_scores: Dict[str, tuple[float, float, float]] = {}
    for pair_id in sorted(set(windows.matched_pair_ids)):
        selector = np.asarray(
            [
                candidate == pair_id
                for candidate in windows.matched_pair_ids
            ],
            dtype=np.bool_,
        ) & treatment
        if np.any(selector):
            pair_scores[pair_id] = (
                float(np.mean(correct_error[selector])),
                float(np.mean(absent_error[selector])),
                float(np.mean(shuffled_error[selector])),
            )
    if not pair_scores:
        raise ValueError("action sanity has no treatment pairs")
    beats_no_action = [
        correct_value < absent_value
        for correct_value, absent_value, _ in pair_scores.values()
    ]
    beats_shuffled = [
        correct_value < shuffled_value
        for correct_value, _, shuffled_value in pair_scores.values()
    ]
    beats_both = [
        no_action and shuffled_value
        for no_action, shuffled_value in zip(
            beats_no_action, beats_shuffled
        )
    ]
    return {
        "schema_version": 1,
        "kind": "whole_pair_action_conditioning_sanity",
        "treatment_pair_count": len(pair_scores),
        "correct_action_mean_mse": float(
            np.mean([values[0] for values in pair_scores.values()])
        ),
        "no_action_mean_mse": float(
            np.mean([values[1] for values in pair_scores.values()])
        ),
        "shuffled_action_mean_mse": float(
            np.mean([values[2] for values in pair_scores.values()])
        ),
        "correct_action_beats_no_action_fraction": float(
            np.mean(beats_no_action)
        ),
        "correct_action_beats_shuffled_fraction": float(
            np.mean(beats_shuffled)
        ),
        "correct_action_beats_both_fraction": float(
            np.mean(beats_both)
        ),
        "pair_scores": {
            pair_id: {
                "correct_action_mse": values[0],
                "no_action_mse": values[1],
                "shuffled_action_mse": values[2],
            }
            for pair_id, values in pair_scores.items()
        },
    }


def assess_action_conditioned_jepa_development(
    *,
    baseline_transfer: Mapping[str, Any],
    jepa_transfer: Mapping[str, Any],
    token_diagnostics: Mapping[str, Any],
    action_sanity: Mapping[str, Any],
    detection: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Apply frozen development gates without making a confirmation claim."""

    baseline_effect = float(
        baseline_transfer["downstream_effect_mse"]
    )
    jepa_effect = float(jepa_transfer["downstream_effect_mse"])
    baseline_action = float(
        baseline_transfer["normalized_mse_action_overlap"]
    )
    jepa_action = float(
        jepa_transfer["normalized_mse_action_overlap"]
    )
    if min(
        baseline_effect,
        jepa_effect,
        baseline_action,
        jepa_action,
    ) < 0.0 or baseline_effect <= 0.0 or baseline_action <= 0.0:
        raise ValueError("development assessment metrics are invalid")
    effect_improvement = 1.0 - jepa_effect / baseline_effect
    action_ratio = jepa_action / baseline_action
    effective_rank = float(token_diagnostics["effective_rank"])
    minimum_entity_rank = float(
        token_diagnostics.get(
            "minimum_observed_entity_effective_rank",
            effective_rank,
        )
    )
    latent_dimension = int(token_diagnostics["latent_dimension"])
    action_hit = float(
        jepa_transfer["action_and_target_hit_at_1"]
    )
    action_fraction = float(
        action_sanity["correct_action_beats_both_fraction"]
    )
    false_alarm = float(
        detection[
            "evaluation_control_sequential_false_alarm_rate"
        ]
    )
    detection_rate = float(
        detection[
            "evaluation_treatment_sequential_detection_rate"
        ]
    )
    raw_delay = detection[
        "median_sequential_detection_delay_transitions"
    ]
    delay = float(raw_delay) if raw_delay is not None else float("inf")
    gates = {
        "downstream_effect_improvement_at_least_10_percent": (
            effect_improvement >= 0.10
        ),
        "action_overlap_mse_within_5_percent": action_ratio <= 1.05,
        "action_and_target_hit_at_1_at_least_90_percent": (
            action_hit >= 0.90
        ),
        "effective_rank_at_least_25_percent": (
            minimum_entity_rank >= 0.25 * latent_dimension
        ),
        "correct_action_beats_both_on_80_percent_of_pairs": (
            action_fraction >= 0.80
        ),
    }
    anomaly_gates = {
        "sequential_control_false_alarm_at_most_5_percent": (
            false_alarm <= 0.05
        ),
        "sequential_treatment_detection_at_least_80_percent": (
            detection_rate >= 0.80
        ),
        "median_sequential_delay_at_most_10": delay <= 10.0,
    }
    predictive_passed = all(gates.values())
    anomaly_passed = all(anomaly_gates.values())
    return {
        "schema_version": 1,
        "kind": "action_conditioned_jepa_low_rank_development_assessment",
        "evidence_boundary": (
            "open development only; fresh sealed matched pairs are "
            "required for confirmation"
        ),
        "observed": {
            "downstream_effect_relative_improvement": effect_improvement,
            "action_overlap_mse_ratio": action_ratio,
            "action_and_target_hit_at_1": action_hit,
            "effective_rank": effective_rank,
            "minimum_observed_entity_effective_rank": (
                minimum_entity_rank
            ),
            "latent_dimension": latent_dimension,
            "correct_action_beats_both_fraction": action_fraction,
            "sequential_control_false_alarm_rate": false_alarm,
            "sequential_treatment_detection_rate": detection_rate,
            "median_sequential_detection_delay": (
                None if not np.isfinite(delay) else delay
            ),
        },
        "predictive_gates": gates,
        "anomaly_gates": anomaly_gates,
        "predictive_development_gates_passed": predictive_passed,
        "anomaly_development_gates_passed": anomaly_passed,
        "decision": (
            "advance_to_sealed_confirmation"
            if predictive_passed
            else "reject_this_configuration"
        ),
        "sealed_confirmation": False,
    }


def _shuffle_actions_by_pair(
    windows: ActionConditionedWindows,
    actions: NDArray[np.float64],
    *,
    seed: int,
) -> NDArray[np.float64]:
    pair_ids = tuple(sorted(set(windows.matched_pair_ids)))
    treatment_trajectories = {
        trajectory_id
        for trajectory_id in set(windows.trajectory_ids)
        if any(
            candidate == trajectory_id
            and np.any(actions[position, ..., 1] > 0.5)
            for position, candidate in enumerate(
                windows.trajectory_ids
            )
        )
    }
    generator = np.random.default_rng(seed)
    shuffled_pairs = list(pair_ids)
    generator.shuffle(shuffled_pairs)
    if len(shuffled_pairs) > 1 and all(
        left == right
        for left, right in zip(pair_ids, shuffled_pairs)
    ):
        shuffled_pairs = shuffled_pairs[1:] + shuffled_pairs[:1]
    source_pair = dict(zip(pair_ids, shuffled_pairs))
    source_index = {
        (
            pair_id,
            int(transition),
            trajectory_id in treatment_trajectories,
        ): position
        for position, (pair_id, trajectory_id, transition) in enumerate(
            zip(
                windows.matched_pair_ids,
                windows.trajectory_ids,
                windows.transition_indices,
            )
        )
    }
    result = np.empty_like(actions)
    for position, (pair_id, trajectory_id, transition) in enumerate(
        zip(
            windows.matched_pair_ids,
            windows.trajectory_ids,
            windows.transition_indices,
        )
    ):
        lookup = (
            source_pair[pair_id],
            int(transition),
            trajectory_id in treatment_trajectories,
        )
        if lookup not in source_index:
            raise ValueError("shuffled action pair windows do not align")
        result[position] = actions[source_index[lookup]]
    return result


def _latent_diagnostics(
    embeddings: NDArray[np.float64],
) -> Mapping[str, Any]:
    values = np.asarray(embeddings, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1] < 1
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("token diagnostics require finite rank-2 values")
    centered = values - np.mean(values, axis=0, keepdims=True)
    covariance = (
        centered.T @ centered / float(max(len(values) - 1, 1))
    )
    eigenvalues = np.maximum(
        np.linalg.eigvalsh(covariance), 0.0
    )
    total = float(np.sum(eigenvalues))
    if total <= 1e-12:
        effective_rank = 0.0
    else:
        probabilities = eigenvalues[eigenvalues > 1e-12] / total
        effective_rank = float(
            np.exp(-np.sum(probabilities * np.log(probabilities)))
        )
    variances = np.diag(covariance)
    off_diagonal = covariance - np.diag(variances)
    return {
        "effective_rank": effective_rank,
        "minimum_dimension_variance": float(np.min(variances)),
        "mean_dimension_variance": float(np.mean(variances)),
        "mean_absolute_off_diagonal_covariance": float(
            np.mean(np.abs(off_diagonal))
        ),
    }


def _markdown_report(report: Mapping[str, Any]) -> str:
    assessment = report.get("assessment", {})
    transfer_scores = report.get("transfer_scores", {})
    lines = [
        "# Action-conditioned JEPA + low-rank development v1",
        "",
        "Open development evidence only. This is not sealed confirmation "
        "or a world-model claim.",
        "",
        "## Decision",
        "",
        f"`{assessment.get('decision', 'unavailable')}`"
        if isinstance(assessment, Mapping)
        else "`unavailable`",
        "",
        "## Held-out-topology scores",
        "",
        "| Model | Action MSE | Overall MSE | Downstream effect MSE | "
        "Hit@1 | Parameters | Latency ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    if isinstance(transfer_scores, Mapping):
        for name, raw in transfer_scores.items():
            if not isinstance(raw, Mapping):
                continue
            lines.append(
                f"| {name} | "
                f"{float(raw['normalized_mse_action_overlap']):.4f} | "
                f"{float(raw['normalized_mse_overall']):.4f} | "
                f"{float(raw['downstream_effect_mse']):.4f} | "
                f"{float(raw['action_and_target_hit_at_1']):.3f} | "
                f"{int(raw['parameter_count'])} | "
                f"{float(raw['batch_one_latency_ms']):.3f} |"
            )
    lines.extend(
        [
            "",
            "## Evidence boundary",
            "",
            "- The source corpus and evaluation data were already open.",
            "- The largest worker topology is a development transfer "
            "diagnostic, not sealed evidence.",
            "- A fresh matched-pair corpus is required for confirmation.",
            "",
        ]
    )
    return "\n".join(lines)


def _pretty_json(payload: Mapping[str, Any]) -> str:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
