"""NumPy-only evaluation helpers for hybrid telemetry JEPA experiments."""

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import numpy as np
from numpy.typing import NDArray

from .hybrid_graph_tokens import HybridGraphTokens


@dataclass(frozen=True)
class NormalizedTargetBlockScore:
    """Variance-normalized target error summarized by graph entity."""

    mean_normalized_mse: float
    entity_normalized_mse: Mapping[str, Optional[float]]
    scored_channel_count: int
    constant_channel_count: int


def score_normalized_target_blocks(
    training: HybridGraphTokens,
    evaluation: HybridGraphTokens,
    predictions: NDArray[np.float64],
    *,
    minimum_training_variance: float = 1e-12,
    channel_mask: Optional[NDArray[np.bool_]] = None,
) -> NormalizedTargetBlockScore:
    """Score targets with per-channel scales fitted on training only."""

    if (
        not np.isfinite(minimum_training_variance)
        or minimum_training_variance <= 0.0
    ):
        raise ValueError("minimum training variance must be finite and positive")
    _validate_compatible_schemas(training, evaluation)
    if predictions.shape != evaluation.fine_targets.shape:
        raise ValueError("prediction shape does not match evaluation targets")
    if not np.all(np.isfinite(predictions)):
        raise ValueError("predictions must be finite")
    if not (
        np.all(np.isfinite(training.fine_targets))
        and np.all(np.isfinite(evaluation.fine_targets))
    ):
        raise ValueError("target blocks must be finite")
    if len(training.fine_targets) < 1:
        raise ValueError("training targets cannot be empty")
    if (
        channel_mask is not None
        and channel_mask.shape != training.feature_mask.shape
    ):
        raise ValueError(
            "channel mask must have entity-by-feature shape"
        )

    training_variance = np.var(
        training.fine_targets,
        axis=(0, 1, 2),
    )
    evaluation_mse = np.mean(
        np.square(predictions - evaluation.fine_targets),
        axis=(0, 1, 2),
    )
    observed = (
        training.feature_mask
        if channel_mask is None
        else np.logical_and(training.feature_mask, channel_mask)
    )
    varying = np.logical_and(
        observed,
        training_variance > minimum_training_variance,
    )
    scored_channel_count = int(np.count_nonzero(varying))
    if scored_channel_count == 0:
        raise ValueError("no varying observed target channels to score")

    normalized_errors = np.zeros_like(
        training_variance,
        dtype=np.float64,
    )
    normalized_errors[varying] = (
        evaluation_mse[varying] / training_variance[varying]
    )
    entity_errors: Dict[str, Optional[float]] = {}
    for entity_position, entity_name in enumerate(
        training.entity_names
    ):
        entity_channels = varying[entity_position]
        entity_errors[entity_name] = (
            float(
                np.mean(
                    normalized_errors[
                        entity_position,
                        entity_channels,
                    ]
                )
            )
            if np.any(entity_channels)
            else None
        )
    return NormalizedTargetBlockScore(
        mean_normalized_mse=float(np.mean(normalized_errors[varying])),
        entity_normalized_mse=entity_errors,
        scored_channel_count=scored_channel_count,
        constant_channel_count=int(
            np.count_nonzero(
                np.logical_and(observed, np.logical_not(varying))
            )
        ),
    )


def embedding_effective_rank_fraction(
    embeddings: NDArray[np.float64],
) -> float:
    """Return covariance participation rank divided by embedding width."""

    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a rank-2 array")
    if embeddings.shape[0] < 1 or embeddings.shape[1] < 1:
        raise ValueError("embeddings cannot have an empty dimension")
    if not np.all(np.isfinite(embeddings)):
        raise ValueError("embeddings must be finite")

    centered = embeddings - np.mean(embeddings, axis=0)
    singular_values = np.linalg.svd(
        centered,
        compute_uv=False,
    )
    eigenvalue_scale = np.square(singular_values)
    squared_scale_sum = float(np.sum(np.square(eigenvalue_scale)))
    if squared_scale_sum <= 1e-18:
        return 0.0
    effective_rank = (
        float(np.square(np.sum(eigenvalue_scale)))
        / squared_scale_sum
    )
    return float(effective_rank / embeddings.shape[1])


def _validate_compatible_schemas(
    training: HybridGraphTokens,
    evaluation: HybridGraphTokens,
) -> None:
    scalars_match = (
        training.entity_names == evaluation.entity_names
        and training.kind_names == evaluation.kind_names
        and training.entity_type_names == evaluation.entity_type_names
        and training.relation_names == evaluation.relation_names
        and training.feature_names == evaluation.feature_names
        and training.local_feature_keys == evaluation.local_feature_keys
        and training.horizons == evaluation.horizons
        and training.fine_targets.shape[1:]
        == evaluation.fine_targets.shape[1:]
    )
    arrays_match = all(
        np.array_equal(training_value, evaluation_value)
        for training_value, evaluation_value in (
            (training.entity_ids, evaluation.entity_ids),
            (training.kind_ids, evaluation.kind_ids),
            (training.entity_type_ids, evaluation.entity_type_ids),
            (training.relation_ids, evaluation.relation_ids),
            (training.typed_adjacency, evaluation.typed_adjacency),
            (training.feature_mask, evaluation.feature_mask),
        )
    )
    if not scalars_match or not arrays_match:
        raise ValueError("training and evaluation token schemas do not match")
