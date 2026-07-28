"""Held-out frozen-state probes for contextual metrics + logs JEPA."""

import re
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .contextual_multimodal_corpus import (
    ContextualMultimodalModelWindows,
)
from .contextual_multimodal_world_model import (
    ContextualMultimodalJepaWorldModelDetector,
)


REQUIRED_MODEL_NAMES = (
    "contextual_multimodal",
    "metrics_only",
    "capacity_matched_metrics_only",
    "shuffled_logs",
)


def evaluate_frozen_context_transfer(
    models: Mapping[
        str, ContextualMultimodalJepaWorldModelDetector
    ],
    training: ContextualMultimodalModelWindows,
    validation: ContextualMultimodalModelWindows,
    *,
    training_window_case_ids: Sequence[str],
    validation_window_case_ids: Sequence[str],
    shuffled_training: ContextualMultimodalModelWindows,
    shuffled_validation: ContextualMultimodalModelWindows,
    target_names: Sequence[str],
    ridge: float,
    pca_dimension: int,
) -> Mapping[str, Any]:
    """Fit probes on training families and score untouched families."""

    if set(models) != set(REQUIRED_MODEL_NAMES):
        raise ValueError(
            "frozen transfer requires all preregistered model controls"
        )
    if ridge <= 0.0:
        raise ValueError("frozen transfer ridge must be positive")
    if (
        len(training_window_case_ids)
        != len(training.metric_contexts)
        or len(validation_window_case_ids)
        != len(validation.metric_contexts)
    ):
        raise ValueError(
            "frozen transfer case identities must align with windows"
        )
    raw_training = _raw_context(training)
    raw_validation = _raw_context(validation)
    pca_training, pca_validation = _pca_contexts(
        raw_training,
        raw_validation,
        pca_dimension,
    )
    representations = {
        "contextual_multimodal": (
            models["contextual_multimodal"]
            .encode_context(training)
            .reshape(len(training.metric_contexts), -1),
            models["contextual_multimodal"]
            .encode_context(validation)
            .reshape(len(validation.metric_contexts), -1),
        ),
        "metrics_only": (
            models["metrics_only"]
            .encode_context(training)
            .reshape(len(training.metric_contexts), -1),
            models["metrics_only"]
            .encode_context(validation)
            .reshape(len(validation.metric_contexts), -1),
        ),
        "capacity_matched_metrics_only": (
            models["capacity_matched_metrics_only"]
            .encode_context(training)
            .reshape(len(training.metric_contexts), -1),
            models["capacity_matched_metrics_only"]
            .encode_context(validation)
            .reshape(len(validation.metric_contexts), -1),
        ),
        "shuffled_logs": (
            models["shuffled_logs"]
            .encode_context(shuffled_training)
            .reshape(len(training.metric_contexts), -1),
            models["shuffled_logs"]
            .encode_context(shuffled_validation)
            .reshape(len(validation.metric_contexts), -1),
        ),
        "raw_context_ridge": (raw_training, raw_validation),
        f"pca_{pca_dimension}_context_ridge": (
            pca_training,
            pca_validation,
        ),
    }
    targets = _target_columns(training, validation, target_names)
    family_ids = tuple(
        _family_id(case_id)
        for case_id in validation_window_case_ids
    )
    evaluated: Dict[str, Any] = {}
    for name, (training_context, validation_context) in (
        representations.items()
    ):
        evaluated[name] = _evaluate_representation(
            training_context,
            validation_context,
            training,
            validation,
            targets,
            family_ids,
            ridge,
        )
    return {
        "schema_version": 1,
        "kind": "frozen_context_future_state_ridge",
        "fit_split": "training_schedule_families_only",
        "evaluation_split": "untouched_validation_schedule_families",
        "ridge": ridge,
        "target_block_reduction": "mean",
        "targets": list(target_names),
        "representations": evaluated,
    }


def _evaluate_representation(
    training_context: NDArray[np.float64],
    validation_context: NDArray[np.float64],
    training: ContextualMultimodalModelWindows,
    validation: ContextualMultimodalModelWindows,
    targets: Mapping[
        str, Tuple[NDArray[np.float64], NDArray[np.float64]]
    ],
    validation_family_ids: Sequence[str],
    ridge: float,
) -> Mapping[str, Any]:
    training_design = _conditioned_design(
        training_context,
        training,
    )
    validation_design = _conditioned_design(
        validation_context,
        validation,
    )
    horizon_count = len(validation.horizons)
    repeated_families = np.repeat(
        np.asarray(validation_family_ids, dtype=object),
        horizon_count,
    )
    target_results: Dict[str, Any] = {}
    completed_errors = []
    completed_r_squared = []
    for target_name, (training_target, validation_target) in (
        targets.items()
    ):
        training_variance = float(np.var(training_target))
        if training_variance <= 1e-12:
            target_results[target_name] = {
                "status": "insufficient_training_variation"
            }
            continue
        prediction = _ridge_predict(
            training_design,
            training_target,
            validation_design,
            ridge,
        )
        squared_error = np.square(prediction - validation_target)
        normalized_mse = float(
            np.mean(squared_error) / training_variance
        )
        validation_variance = float(np.var(validation_target))
        r_squared = (
            float(
                1.0
                - np.mean(squared_error) / validation_variance
            )
            if validation_variance > 1e-12
            else None
        )
        family_normalized_mse = {
            family: float(
                np.mean(
                    squared_error[repeated_families == family]
                )
                / training_variance
            )
            for family in sorted(set(repeated_families))
        }
        target_results[target_name] = {
            "status": "completed",
            "training_variance": training_variance,
            "validation_normalized_mse": normalized_mse,
            "validation_r_squared": r_squared,
            "family_normalized_mse": family_normalized_mse,
        }
        completed_errors.append(normalized_mse)
        if r_squared is not None:
            completed_r_squared.append(r_squared)
    return {
        "context_dimension": int(training_context.shape[1]),
        "completed_target_count": len(completed_errors),
        "mean_validation_normalized_mse": (
            float(np.mean(completed_errors))
            if completed_errors
            else None
        ),
        "mean_validation_r_squared": (
            float(np.mean(completed_r_squared))
            if completed_r_squared
            else None
        ),
        "targets": target_results,
    }


def _conditioned_design(
    context: NDArray[np.float64],
    windows: ContextualMultimodalModelWindows,
) -> NDArray[np.float64]:
    sample_count = len(context)
    horizon_count = len(windows.horizons)
    controls = np.mean(windows.target_controls, axis=2)
    horizon_one_hot = np.broadcast_to(
        np.eye(horizon_count, dtype=np.float64)[None, :, :],
        (sample_count, horizon_count, horizon_count),
    )
    repeated_context = np.broadcast_to(
        context[:, None, :],
        (sample_count, horizon_count, context.shape[1]),
    )
    return np.asarray(
        np.concatenate(
            (repeated_context, controls, horizon_one_hot),
            axis=2,
        ).reshape(sample_count * horizon_count, -1),
        dtype=np.float64,
    )


def _target_columns(
    training: ContextualMultimodalModelWindows,
    validation: ContextualMultimodalModelWindows,
    target_names: Sequence[str],
) -> Mapping[
    str, Tuple[NDArray[np.float64], NDArray[np.float64]]
]:
    targets: Dict[
        str, Tuple[NDArray[np.float64], NDArray[np.float64]]
    ] = {}
    for target_name in target_names:
        try:
            modality, feature_name = target_name.split(".", 1)
        except ValueError as error:
            raise ValueError(
                f"invalid frozen transfer target: {target_name}"
            ) from error
        if modality == "metric":
            names = training.metric_feature_names
            training_blocks = training.metric_target_blocks
            validation_blocks = validation.metric_target_blocks
        elif modality == "log":
            names = training.log_feature_names
            training_blocks = training.log_target_blocks
            validation_blocks = validation.log_target_blocks
        else:
            raise ValueError(
                f"invalid frozen transfer modality: {modality}"
            )
        if names != (
            validation.metric_feature_names
            if modality == "metric"
            else validation.log_feature_names
        ):
            raise ValueError(
                "frozen transfer feature names differ across splits"
            )
        try:
            position = names.index(feature_name)
        except ValueError as error:
            raise ValueError(
                f"missing frozen transfer target: {target_name}"
            ) from error
        targets[target_name] = (
            np.mean(
                training_blocks[:, :, :, position],
                axis=2,
            ).reshape(-1),
            np.mean(
                validation_blocks[:, :, :, position],
                axis=2,
            ).reshape(-1),
        )
    return targets


def _raw_context(
    windows: ContextualMultimodalModelWindows,
) -> NDArray[np.float64]:
    return np.concatenate(
        (windows.metric_contexts, windows.log_contexts),
        axis=2,
    ).reshape(len(windows.metric_contexts), -1)


def _pca_contexts(
    training: NDArray[np.float64],
    validation: NDArray[np.float64],
    dimension: int,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    if dimension < 1 or dimension > training.shape[1]:
        raise ValueError("PCA context dimension is invalid")
    location = np.mean(training, axis=0)
    centered = training - location
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    components = right[:dimension].T
    return (
        centered @ components,
        (validation - location) @ components,
    )


def _ridge_predict(
    training: NDArray[np.float64],
    target: NDArray[np.float64],
    validation: NDArray[np.float64],
    ridge: float,
) -> NDArray[np.float64]:
    location = np.mean(training, axis=0)
    scale = np.std(training, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    normalized_training = (training - location) / scale
    normalized_validation = (validation - location) / scale
    design = np.column_stack(
        (normalized_training, np.ones(len(training)))
    )
    validation_design = np.column_stack(
        (normalized_validation, np.ones(len(validation)))
    )
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(
        design.T @ design + penalty,
        design.T @ target,
    )
    return validation_design @ weights


def _family_id(case_id: str) -> str:
    match = re.search(r"-f([0-9]{2})-", case_id)
    if match is None:
        raise ValueError(
            f"cannot derive schedule family from case: {case_id}"
        )
    return f"f{match.group(1)}"
