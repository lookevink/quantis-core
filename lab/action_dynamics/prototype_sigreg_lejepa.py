"""Retained reproduction runner for the exact SIGReg LeJEPA tracer.

Question: does the exact LeJEPA sketched-isotropic-Gaussian regularizer improve
the strongest entity-preserving residual JEPA without changing its inference
architecture?

This is non-production experiment code. Keep it with its immutable artifact
and use a fresh ``--output`` directory for every rerun.
"""

import argparse
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from quantis_core.action_dynamics_corpus import (
    load_action_dynamics_development_corpus,
)
from quantis_core.edge_dynamics.action_conditioned_jepa import (
    ActionConditionedJepaConfig,
    ActionConditionedJepaDynamics,
)
from quantis_core.edge_dynamics.data import (
    PreparedAttributionQueries,
    PreparedEdgeDynamicsData,
    load_edge_dynamics_cache,
    partition_worker_topology,
    prepare_worker_topology_transfer_data,
    source_artifact_manifest_sha256,
    subset_attribution_queries,
    topology_transfer_cache_address,
    validate_topology_transfer_cache,
    write_edge_dynamics_cache,
)
from quantis_core.edge_dynamics.evaluation import (
    forecast_objective,
    score_edge_model,
)
from quantis_core.edge_dynamics.jepa_evaluation import (
    action_conditioning_sanity,
    node_token_diagnostics,
)
from quantis_core.edge_dynamics.models import (
    ContractiveLowRankDynamics,
    EdgeDynamicsModel,
    LowRankConfig,
)
from quantis_core.edge_dynamics.residual_jepa import (
    FrozenBaselineResidualDynamics,
    artifact_sha256,
    latent_divergence_detection,
)


MODEL_NAMES = (
    "raw_low_rank",
    "no_regularizer_jepa",
    "variance_covariance_jepa",
    "sigreg_jepa",
)


def run_sigreg_jepa_tracer(
    *,
    corpus_directory: Path,
    cache_root: Path,
    output_directory: Path,
    epochs: int = 60,
    batch_size: int = 256,
    device: str = "cpu",
    seed: int = 401,
) -> Mapping[str, Any]:
    """Fit the frozen tracer and write immutable open-development evidence."""

    if output_directory.exists():
        raise FileExistsError(
            f"refusing to overwrite SIGReg JEPA: {output_directory}"
        )
    source_manifest = source_artifact_manifest_sha256(
        corpus_directory
    )
    cache_directory = cache_root / topology_transfer_cache_address(
        source_manifest
    )
    if cache_directory.exists():
        prepared = load_edge_dynamics_cache(cache_directory)
        cache_reused = True
    else:
        corpus = load_action_dynamics_development_corpus(
            corpus_directory
        )
        prepared = prepare_worker_topology_transfer_data(corpus)
        write_edge_dynamics_cache(prepared, cache_directory)
        cache_reused = False
    validate_topology_transfer_cache(prepared, corpus_directory)
    partitions = {
        role: partition_worker_topology(windows)
        for role, windows in prepared.windows.items()
    }
    held_out_values = {
        partition.held_out_normalized_value
        for partition in partitions.values()
    }
    if len(held_out_values) != 1:
        raise ValueError("worker topology holdout drifted across roles")
    held_out_value = next(iter(held_out_values))
    fit = partitions["fit"].in_distribution
    selection = partitions["selection"].in_distribution
    calibration = partitions["calibration"].in_distribution
    iid_evaluation = partitions["evaluation"].in_distribution
    transfer_evaluation = partitions["evaluation"].held_out
    iid_queries, transfer_queries = _partition_queries(
        prepared,
        held_out_normalized_value=held_out_value,
    )
    if len(fit.entity_names) != 7:
        raise ValueError(
            "SIGReg tracer requires the preregistered seven-entity schema"
        )

    training_seconds: Dict[str, float] = {}
    baseline = ContractiveLowRankDynamics(
        LowRankConfig(rank=32)
    )
    started = time.perf_counter()
    baseline.fit(fit)
    training_seconds["raw_low_rank"] = time.perf_counter() - started
    baseline_hash_before = artifact_sha256(baseline.to_dict())

    shared_config = {
        "node_latent_dimension": 16,
        "transition_rank": 32,
        "epochs": epochs,
        "batch_size": batch_size,
        "context_reconstruction_weight": 0.0,
        "zero_initialize_decoder": True,
        "device": device,
        "seed": seed,
    }
    common_jepa_config = {
        **shared_config,
        "mask_time_fraction": 0.3,
        "mask_entity_fraction": 0.25,
        "latent_prediction_weight": 0.2,
        "reconstruction_weight": 1.0,
        "objective": "jepa",
    }
    no_regularizer_config = ActionConditionedJepaConfig(
        **common_jepa_config,
        regularizer="none",
        variance_weight=0.0,
        covariance_weight=0.0,
        sigreg_weight=0.0,
    )
    variance_covariance_config = ActionConditionedJepaConfig(
        **common_jepa_config,
        regularizer="variance_covariance",
        variance_weight=0.01,
        covariance_weight=0.005,
        sigreg_weight=0.0,
    )
    sigreg_config = ActionConditionedJepaConfig(
        **common_jepa_config,
        regularizer="sigreg",
        variance_weight=0.0,
        covariance_weight=0.0,
        sigreg_weight=0.02,
        sigreg_sketch_dimension=256,
        sigreg_knot_count=17,
        sigreg_projection_seed=1401,
    )
    no_regularizer = FrozenBaselineResidualDynamics(
        baseline=baseline,
        correction=ActionConditionedJepaDynamics(
            no_regularizer_config
        ),
    )
    variance_covariance = FrozenBaselineResidualDynamics(
        baseline=baseline,
        correction=ActionConditionedJepaDynamics(
            variance_covariance_config
        ),
    )
    sigreg = FrozenBaselineResidualDynamics(
        baseline=baseline,
        correction=ActionConditionedJepaDynamics(sigreg_config),
    )
    residual_models = {
        "no_regularizer_jepa": no_regularizer,
        "variance_covariance_jepa": variance_covariance,
        "sigreg_jepa": sigreg,
    }
    for name, model in residual_models.items():
        _timed_fit(model, fit, training_seconds, name)
        model.select_correction_gain(selection)
    baseline_hash_after = artifact_sha256(baseline.to_dict())
    if baseline_hash_before != baseline_hash_after:
        raise RuntimeError("shared frozen baseline changed during training")

    models: Dict[str, EdgeDynamicsModel] = {
        "raw_low_rank": baseline,
        **residual_models,
    }
    selection_scores = {
        name: dict(forecast_objective(model, selection))
        for name, model in models.items()
    }
    iid_scores = {
        name: score_edge_model(
            model, iid_evaluation, iid_queries
        ).to_dict()
        for name, model in models.items()
    }
    transfer_scores = {
        name: score_edge_model(
            model, transfer_evaluation, transfer_queries
        ).to_dict()
        for name, model in models.items()
    }
    diagnostics = {
        name: node_token_diagnostics(
            model.correction, transfer_evaluation
        )
        for name, model in residual_models.items()
    }
    action_sanity = {
        name: action_conditioning_sanity(
            model, transfer_evaluation, seed=seed + 312
        )
        for name, model in residual_models.items()
    }
    iid_detection = {
        name: latent_divergence_detection(
            model=model,
            calibration=calibration,
            evaluation=iid_evaluation,
        )
        for name, model in residual_models.items()
    }
    transfer_detection = {
        name: latent_divergence_detection(
            model=model,
            calibration=calibration,
            evaluation=transfer_evaluation,
        )
        for name, model in residual_models.items()
    }
    observable_state_probes = _observable_state_probes(
        fit=fit,
        transfer=transfer_evaluation,
        models=residual_models,
    )
    training_metrics = {
        name: [dict(row) for row in model.correction.training_metrics]
        for name, model in residual_models.items()
    }
    restoration_parity = {
        name: _restoration_parity(
            name, model, transfer_evaluation
        )
        for name, model in models.items()
    }
    reported_measurements = {
        "training_seconds": training_seconds,
        "training_metrics": training_metrics,
        "selection_scores": selection_scores,
        "in_distribution_scores": iid_scores,
        "transfer_scores": transfer_scores,
        "node_token_diagnostics": diagnostics,
        "observable_state_probes": observable_state_probes,
        "action_conditioning_sanity": action_sanity,
        "in_distribution_detection": iid_detection,
        "transfer_detection": transfer_detection,
    }
    assessment = assess_sigreg_jepa_tracer(
        transfer_scores=transfer_scores,
        state_probes=observable_state_probes,
        action_sanity=action_sanity,
        transfer_detection=transfer_detection,
        selected_gains={
            name: model.selected_gain
            for name, model in residual_models.items()
        },
        parameter_counts={
            name: int(score["parameter_count"])
            for name, score in transfer_scores.items()
        },
        reported_measurements=reported_measurements,
        restoration_parity=restoration_parity,
    )
    report: Dict[str, Any] = {
        "schema_version": 1,
        "kind": "sigreg_lejepa_tracer_result_v1",
        "evidence_boundary": (
            "open development only; not sealed confirmation or a "
            "world-model claim"
        ),
        "sigreg_source": {
            "source_preset": "official-minimal-c293d29",
            "source_commit": (
                "c293d291ca87cd4fddee9d3fffe4e914c7272052"
            ),
            "paper": "https://arxiv.org/abs/2511.08544",
            "official_minimal": (
                "https://github.com/galilai-group/lejepa/"
                "blob/c293d291ca87cd4fddee9d3fffe4e914c7272052/"
                "MINIMAL.md"
            ),
            "integration_interval": [0.0, 3.0],
            "quadrature": "17-knot symmetric trapezoidal",
            "sketch_dimension": 256,
            "applied_independently_by_entity": True,
        },
        "implementation_identity": _implementation_identity(),
        "source_corpus_sha256": prepared.source_corpus_sha256,
        "source_artifact_manifest_sha256": (
            prepared.source_artifact_manifest_sha256
        ),
        "preprocessing_cache_address": cache_directory.name,
        "preprocessing_protocol": prepared.preprocessing_protocol,
        "preprocessing_cache_reused": cache_reused,
        "frozen_baseline": {
            "sha256_before_correction_training": baseline_hash_before,
            "sha256_after_correction_training": baseline_hash_after,
            "unchanged": baseline_hash_before == baseline_hash_after,
        },
        "held_out_topology": {
            "control_feature": "worker_replicas",
            "normalized_value": held_out_value,
            "fit_pair_count": len(set(fit.matched_pair_ids)),
            "selection_pair_count": len(
                set(selection.matched_pair_ids)
            ),
            "calibration_pair_count": len(
                set(calibration.matched_pair_ids)
            ),
            "in_distribution_evaluation_pair_count": len(
                set(iid_evaluation.matched_pair_ids)
            ),
            "transfer_evaluation_pair_count": len(
                set(transfer_evaluation.matched_pair_ids)
            ),
        },
        "window_counts": {
            "fit": len(fit.histories),
            "selection": len(selection.histories),
            "calibration": len(calibration.histories),
            "in_distribution_evaluation": len(
                iid_evaluation.histories
            ),
            "transfer_evaluation": len(
                transfer_evaluation.histories
            ),
        },
        "model_configs": {
            "raw_low_rank": LowRankConfig(rank=32).__dict__,
            "no_regularizer_jepa": no_regularizer_config.to_dict(),
            "variance_covariance_jepa": (
                variance_covariance_config.to_dict()
            ),
            "sigreg_jepa": sigreg_config.to_dict(),
        },
        "training_seconds": training_seconds,
        "training_metrics": training_metrics,
        "training_runtime": _training_runtime(residual_models),
        "initial_correction_maximum_absolute_prediction": {
            name: model.correction.initial_maximum_absolute_prediction
            for name, model in residual_models.items()
        },
        "selection_scores": selection_scores,
        "selection_gain_curves": {
            name: [dict(row) for row in model.selection_curve]
            for name, model in residual_models.items()
        },
        "selected_correction_gains": {
            name: model.selected_gain
            for name, model in residual_models.items()
        },
        "in_distribution_scores": iid_scores,
        "transfer_scores": transfer_scores,
        "node_token_diagnostics": diagnostics,
        "observable_state_probes": observable_state_probes,
        "action_conditioning_sanity": action_sanity,
        "latent_divergence_detection": {
            "in_distribution": iid_detection,
            "topology_transfer": transfer_detection,
        },
        "restoration_parity": restoration_parity,
        "assessment": assessment,
        "limitations": [
            "the source corpus and evaluation roles were already open",
            "only one worker topology is held out",
            "the action library contains randomized known interventions",
            "one deterministic training seed is a tracer, not robustness",
            "fresh sealed matched pairs are required for confirmation",
        ],
    }
    _write_sigreg_artifacts(
        output_directory=output_directory,
        report=report,
        model_artifacts={
            name: model.to_dict()
            for name, model in models.items()
        },
    )
    return report


def _observable_state_probes(
    *,
    fit: Any,
    transfer: Any,
    models: Mapping[str, FrozenBaselineResidualDynamics],
) -> Mapping[str, Any]:
    fit_state = np.asarray(fit.histories[:, -1], dtype=np.float64)
    transfer_state = np.asarray(
        transfer.histories[:, -1], dtype=np.float64
    )
    observation_mask = np.var(fit_state, axis=0) > 1e-12
    rows: Dict[str, Any] = {}
    for name, model in models.items():
        fit_tokens = model.correction.encode_histories(
            fit.histories, fit.graph
        )
        transfer_tokens = model.correction.encode_histories(
            transfer.histories, transfer.graph
        )
        rows[name] = _probe_scores(
            fit_tokens,
            fit_state,
            transfer_tokens,
            transfer_state,
            observation_mask,
            fit.entity_names,
            ridge=1e-3,
        )
    fit_pca, transfer_pca = _pca_representations(
        fit.histories,
        transfer.histories,
        width=16,
    )
    rows["matched_pca"] = _probe_scores(
        fit_pca,
        fit_state,
        transfer_pca,
        transfer_state,
        observation_mask,
        fit.entity_names,
        ridge=1e-3,
    )
    return rows


def _probe_scores(
    train_representation: np.ndarray,
    train_states: np.ndarray,
    evaluation_representation: np.ndarray,
    evaluation_states: np.ndarray,
    observation_mask: np.ndarray,
    entity_names: Sequence[str],
    *,
    ridge: float,
) -> Mapping[str, Any]:
    entity_rows: Dict[str, Any] = {}
    normalized_squared_errors = []
    for entity_position, entity_name in enumerate(entity_names):
        mask = observation_mask[entity_position]
        if not np.any(mask):
            entity_rows[entity_name] = {
                "observed_feature_count": 0,
                "nrmse": None,
            }
            continue
        training_token = train_representation[:, entity_position]
        token_mean = np.mean(training_token, axis=0)
        token_scale = np.std(training_token, axis=0)
        token_scale = np.where(token_scale > 1e-12, token_scale, 1.0)
        training = np.concatenate(
            (
                (training_token - token_mean) / token_scale,
                np.ones((len(training_token), 1)),
            ),
            axis=1,
        )
        evaluation = np.concatenate(
            (
                (
                    evaluation_representation[:, entity_position]
                    - token_mean
                )
                / token_scale,
                np.ones((len(evaluation_representation), 1)),
            ),
            axis=1,
        )
        penalty = np.eye(training.shape[1], dtype=np.float64)
        penalty[-1, -1] = 0.0
        target = train_states[:, entity_position, :][:, mask]
        coefficients = np.linalg.solve(
            training.T @ training + ridge * penalty,
            training.T @ target,
        )
        residual = (
            evaluation @ coefficients
            - evaluation_states[:, entity_position, :][:, mask]
        )
        target_scale = np.std(target, axis=0)
        target_scale = np.where(target_scale > 1e-12, target_scale, 1.0)
        normalized = np.square(residual / target_scale).reshape(-1)
        normalized_squared_errors.append(normalized)
        entity_rows[entity_name] = {
            "observed_feature_count": int(np.sum(mask)),
            "nrmse": float(np.sqrt(np.mean(normalized))),
        }
    if not normalized_squared_errors:
        raise ValueError("observable-state probe has no varying features")
    return {
        "aggregate_nrmse": float(
            np.sqrt(np.mean(np.concatenate(normalized_squared_errors)))
        ),
        "entities": entity_rows,
    }


def _pca_representations(
    fit_histories: np.ndarray,
    evaluation_histories: np.ndarray,
    *,
    width: int,
) -> Tuple[np.ndarray, np.ndarray]:
    train_tokens = []
    evaluation_tokens = []
    for entity_position in range(fit_histories.shape[2]):
        training = np.asarray(
            fit_histories[:, :, entity_position, :], dtype=np.float64
        ).reshape(len(fit_histories), -1)
        evaluation = np.asarray(
            evaluation_histories[:, :, entity_position, :],
            dtype=np.float64,
        ).reshape(len(evaluation_histories), -1)
        varying = np.var(training, axis=0) > 1e-12
        local_training = training[:, varying]
        local_evaluation = evaluation[:, varying]
        token_train = np.zeros((len(training), width), dtype=np.float64)
        token_evaluation = np.zeros(
            (len(evaluation), width), dtype=np.float64
        )
        if local_training.shape[1]:
            center = np.mean(local_training, axis=0)
            _, _, right = np.linalg.svd(
                local_training - center, full_matrices=False
            )
            component_count = min(width, len(right))
            components = right[:component_count].copy()
            _orient_components(components)
            token_train[:, :component_count] = (
                local_training - center
            ) @ components.T
            token_evaluation[:, :component_count] = (
                local_evaluation - center
            ) @ components.T
        train_tokens.append(token_train)
        evaluation_tokens.append(token_evaluation)
    return (
        np.stack(train_tokens, axis=1),
        np.stack(evaluation_tokens, axis=1),
    )


def _orient_components(components: np.ndarray) -> None:
    for component in components:
        pivot = int(np.argmax(np.abs(component)))
        if component[pivot] < 0.0:
            component *= -1.0


def assess_sigreg_jepa_tracer(
    *,
    transfer_scores: Mapping[str, Mapping[str, Any]],
    state_probes: Mapping[str, Mapping[str, Any]],
    action_sanity: Mapping[str, Mapping[str, Any]],
    transfer_detection: Mapping[str, Mapping[str, Any]],
    selected_gains: Mapping[str, float],
    parameter_counts: Mapping[str, int],
    reported_measurements: Mapping[str, Any],
    restoration_parity: Mapping[str, bool],
) -> Mapping[str, Any]:
    """Apply the preregistered SIGReg safety and value gates."""

    if set(transfer_scores) != set(MODEL_NAMES):
        raise ValueError("SIGReg assessment model set is incomplete")
    candidate = transfer_scores["sigreg_jepa"]
    raw = transfer_scores["raw_low_rank"]
    current = transfer_scores["variance_covariance_jepa"]
    null = transfer_scores["no_regularizer_jepa"]
    candidate_probe = float(
        state_probes["sigreg_jepa"]["aggregate_nrmse"]
    )
    current_probe = float(
        state_probes["variance_covariance_jepa"]["aggregate_nrmse"]
    )
    pca_probe = float(state_probes["matched_pca"]["aggregate_nrmse"])
    candidate_detection = transfer_detection["sigreg_jepa"]
    current_detection = transfer_detection["variance_covariance_jepa"]
    null_detection = transfer_detection["no_regularizer_jepa"]
    candidate_delay = _finite_delay(candidate_detection)
    current_delay = _finite_delay(current_detection)
    null_delay = _finite_delay(null_detection)
    metrics_finite = _all_reported_numbers_finite(
        reported_measurements
    )
    residual_parameter_counts = {
        parameter_counts[name]
        for name in (
            "no_regularizer_jepa",
            "variance_covariance_jepa",
            "sigreg_jepa",
        )
    }
    safety = {
        "metrics_are_finite": bool(metrics_finite),
        "inference_parameter_count_is_matched": (
            len(residual_parameter_counts) == 1
        ),
        "every_model_restores_public_outputs": all(
            restoration_parity.values()
        ),
        "action_overlap_mse_within_5_percent_of_raw": (
            float(candidate["normalized_mse_action_overlap"])
            <= 1.05 * float(raw["normalized_mse_action_overlap"])
        ),
        "overall_mse_within_5_percent_of_raw": (
            float(candidate["normalized_mse_overall"])
            <= 1.05 * float(raw["normalized_mse_overall"])
        ),
        "action_and_target_hit_at_1_at_least_95_percent": (
            float(candidate["action_and_target_hit_at_1"]) >= 0.95
        ),
        "no_action_specificity_is_100_percent": (
            float(candidate["no_action_specificity"]) == 1.0
        ),
        "correct_action_beats_both_on_80_percent_of_pairs": (
            float(
                action_sanity["sigreg_jepa"][
                    "correct_action_beats_both_fraction"
                ]
            )
            >= 0.80
        ),
    }
    predictive_lane = {
        "downstream_effect_improves_raw_by_10_percent": (
            float(candidate["downstream_effect_mse"])
            <= 0.90 * float(raw["downstream_effect_mse"])
        ),
        "downstream_effect_beats_current_regularizer": (
            float(candidate["downstream_effect_mse"])
            < float(current["downstream_effect_mse"])
        ),
        "downstream_effect_beats_no_regularizer": (
            float(candidate["downstream_effect_mse"])
            < float(null["downstream_effect_mse"])
        ),
        "selected_nonzero_correction_gain": (
            float(selected_gains["sigreg_jepa"]) > 0.0
        ),
    }
    investigation_lane = {
        "state_probe_improves_current_regularizer_by_5_percent": (
            candidate_probe <= 0.95 * current_probe
        ),
        "state_probe_no_worse_than_matched_pca": (
            candidate_probe <= pca_probe
        ),
    }
    candidate_detection_rate = float(
        candidate_detection[
            "evaluation_treatment_trajectory_detection_rate"
        ]
    )
    current_detection_rate = float(
        current_detection[
            "evaluation_treatment_trajectory_detection_rate"
        ]
    )
    null_detection_rate = float(
        null_detection[
            "evaluation_treatment_trajectory_detection_rate"
        ]
    )
    improves_alert_control = all(
        candidate_detection_rate > control_rate
        or (
            candidate_detection_rate == control_rate
            and candidate_delay < control_delay
        )
        for control_rate, control_delay in (
            (current_detection_rate, current_delay),
            (null_detection_rate, null_delay),
        )
    )
    alert_lane = {
        "control_trajectory_false_alarm_at_most_5_percent": (
            float(
                candidate_detection[
                    "evaluation_control_trajectory_false_alarm_rate"
                ]
            )
            <= 0.05
        ),
        "treatment_trajectory_detection_at_least_80_percent": (
            candidate_detection_rate >= 0.80
        ),
        "median_post_onset_delay_at_most_10": candidate_delay <= 10.0,
        "improves_current_and_no_regularizer_alert_control": (
            improves_alert_control
        ),
    }
    lanes = {
        "predictive": {
            "gates": predictive_lane,
            "passed": all(predictive_lane.values()),
        },
        "investigation": {
            "gates": investigation_lane,
            "passed": all(investigation_lane.values()),
        },
        "alert": {
            "gates": alert_lane,
            "passed": all(alert_lane.values()),
        },
    }
    safety_passed = all(safety.values())
    any_lane_passed = any(
        bool(row["passed"]) for row in lanes.values()
    )
    return {
        "schema_version": 1,
        "kind": "sigreg_lejepa_tracer_assessment_v1",
        "safety_gates": safety,
        "safety_passed": safety_passed,
        "value_lanes": lanes,
        "any_value_lane_passed": any_lane_passed,
        "decision": (
            "run_fixed_seed_robustness"
            if safety_passed and any_lane_passed
            else "reject_sigreg_residual_recipe"
        ),
        "sealed_confirmation": False,
    }


def _finite_delay(detection: Mapping[str, Any]) -> float:
    raw = detection["median_post_onset_detection_delay_transitions"]
    return float(raw) if raw is not None else float("inf")


def _all_reported_numbers_finite(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(np.isfinite(float(value)))
    if isinstance(value, Mapping):
        return all(
            _all_reported_numbers_finite(item)
            for item in value.values()
        )
    if isinstance(value, (list, tuple)):
        return all(_all_reported_numbers_finite(item) for item in value)
    if isinstance(value, np.ndarray):
        return bool(np.all(np.isfinite(value)))
    return True


def _restoration_parity(
    name: str,
    model: EdgeDynamicsModel,
    windows: Any,
) -> bool:
    artifact = model.to_dict()
    if name == "raw_low_rank":
        restored: EdgeDynamicsModel = ContractiveLowRankDynamics.from_dict(
            artifact
        )
    else:
        restored = FrozenBaselineResidualDynamics.from_dict(artifact)
    selection = slice(0, min(8, len(windows.histories)))
    first = model.rollout(
        windows.histories[selection],
        windows.future_controls[selection],
        windows.future_actions[selection],
        windows.graph,
    )
    second = restored.rollout(
        windows.histories[selection],
        windows.future_controls[selection],
        windows.future_actions[selection],
        windows.graph,
    )
    return bool(
        np.allclose(first.mean, second.mean, atol=1e-6)
        and np.allclose(first.variance, second.variance, atol=1e-9)
    )


def _write_sigreg_artifacts(
    *,
    output_directory: Path,
    report: Mapping[str, Any],
    model_artifacts: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    if output_directory.exists():
        raise FileExistsError(
            f"refusing to overwrite SIGReg results: {output_directory}"
        )
    output_directory.mkdir(parents=True)
    models = output_directory / "models"
    models.mkdir()
    (output_directory / "prototype-result.json").write_text(
        _pretty_json(report)
    )
    (output_directory / "report.md").write_text(
        _markdown_report(report)
    )
    for name, artifact in model_artifacts.items():
        (models / f"{name}.json").write_text(_pretty_json(artifact))
    hashes = {
        path.relative_to(output_directory).as_posix(): _file_sha256(path)
        for path in sorted(output_directory.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "schema_version": 1,
        "kind": "sigreg_lejepa_tracer_manifest_v1",
        "sha256": hashes,
    }
    (output_directory / "artifact-manifest.json").write_text(
        _pretty_json(manifest)
    )
    return manifest


def _implementation_identity() -> Mapping[str, Any]:
    repository = Path(__file__).resolve().parents[2]
    relative_paths = (
        "lab/action_dynamics/prototype_sigreg_lejepa.py",
        "src/quantis_core/edge_dynamics/action_conditioned_jepa.py",
        "docs/specs/sigreg-lejepa-prototype-v1.md",
        "docs/research/lejepa-sigreg-primary-source-notes.md",
        "pyproject.toml",
    )
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        text=True,
    ).strip()
    if len(revision) != 40:
        raise ValueError("repository revision identity is invalid")
    return {
        "repository_revision": revision,
        "sha256": {
            relative: _file_sha256(repository / relative)
            for relative in relative_paths
        },
    }


def _pretty_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _markdown_report(report: Mapping[str, Any]) -> str:
    assessment = report["assessment"]
    scores = report["transfer_scores"]
    probes = report["observable_state_probes"]
    lines = [
        "# Exact SIGReg LeJEPA tracer v1",
        "",
        "Open development evidence only. This is not sealed confirmation.",
        "",
        f"Decision: **{assessment['decision']}**.",
        "",
        "## Held-out-topology measurements",
        "",
        "| Model | Action MSE | Overall MSE | Effect MSE | Hit@1 | "
        "State-probe NRMSE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in MODEL_NAMES:
        score = scores[name]
        probe = probes.get(name)
        probe_value = (
            f"{float(probe['aggregate_nrmse']):.4f}"
            if isinstance(probe, Mapping)
            else "—"
        )
        lines.append(
            f"| {name} | "
            f"{float(score['normalized_mse_action_overlap']):.4f} | "
            f"{float(score['normalized_mse_overall']):.4f} | "
            f"{float(score['downstream_effect_mse']):.4f} | "
            f"{float(score['action_and_target_hit_at_1']):.1%} | "
            f"{probe_value} |"
        )
    lines.append(
        "| matched_pca | — | — | — | — | "
        f"{float(probes['matched_pca']['aggregate_nrmse']):.4f} |"
    )
    lines.extend(
        [
            "",
            "The candidate changes only the training regularizer. The three "
            "neural variants have the same inference architecture.",
            "",
        ]
    )
    return "\n".join(lines)


def _timed_fit(
    model: FrozenBaselineResidualDynamics,
    fit: Any,
    timings: Dict[str, float],
    name: str,
) -> None:
    started = time.perf_counter()
    model.fit(fit)
    timings[name] = time.perf_counter() - started


def _training_runtime(
    models: Mapping[str, FrozenBaselineResidualDynamics],
) -> Mapping[str, Any]:
    import torch

    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch_version": str(torch.__version__),
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "requested_devices": {
            name: model.correction.config.device
            for name, model in models.items()
        },
        "resolved_devices": {
            name: model.correction.device
            for name, model in models.items()
        },
    }


def _partition_queries(
    prepared: PreparedEdgeDynamicsData,
    *,
    held_out_normalized_value: float,
) -> tuple[PreparedAttributionQueries, PreparedAttributionQueries]:
    control_names = prepared.windows["fit"].control_feature_names
    try:
        position = control_names.index("worker_replicas")
    except ValueError as error:
        raise ValueError(
            "attribution query split requires worker_replicas"
        ) from error
    values = prepared.attribution_queries.future_controls[:, 0, position]
    if not np.allclose(
        prepared.attribution_queries.future_controls[..., position],
        values[:, None],
    ):
        raise ValueError(
            "worker topology must be constant within attribution query"
        )
    transfer = np.isclose(values, held_out_normalized_value)
    return (
        subset_attribution_queries(
            prepared.attribution_queries, ~transfer
        ),
        subset_attribution_queries(
            prepared.attribution_queries, transfer
        ),
    )


def main(arguments: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("artifacts/action-dynamics/development-v1"),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(
            "artifacts/action-dynamics/edge-preprocessing-v1"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/action-dynamics/"
            "prototype-sigreg-lejepa-v1"
        ),
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps"),
        default="cpu",
    )
    parser.add_argument("--seed", type=int, default=401)
    parsed = parser.parse_args(arguments)
    result = run_sigreg_jepa_tracer(
        corpus_directory=parsed.corpus,
        cache_root=parsed.cache_root,
        output_directory=parsed.output,
        epochs=parsed.epochs,
        batch_size=parsed.batch_size,
        device=parsed.device,
        seed=parsed.seed,
    )
    print(
        json.dumps(
            {
                "assessment": result["assessment"],
                "selected_correction_gains": result[
                    "selected_correction_gains"
                ],
                "transfer_scores": result["transfer_scores"],
                "training_seconds": result["training_seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
