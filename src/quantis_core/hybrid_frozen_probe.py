"""Training-fitted NumPy baselines for hybrid graph representations."""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from .hybrid_graph_tokens import HybridGraphTokens


PROBE_MODES = ("all_entities", "one_hop", "entity_local")


@dataclass(frozen=True)
class _FrozenTokenSchema:
    entity_names: Tuple[str, ...]
    kind_ids: NDArray[np.int64]
    entity_type_ids: NDArray[np.int64]
    relation_ids: NDArray[np.int64]
    typed_adjacency: NDArray[np.bool_]
    feature_mask: NDArray[np.bool_]
    feature_names: Tuple[str, ...]
    local_feature_keys: Tuple[Tuple[str, ...], ...]
    horizons: Tuple[int, ...]
    context_shape: Tuple[int, int, int]
    target_shape: Tuple[int, int, int, int]
    control_shape: Tuple[int, int, int]


@dataclass(frozen=True)
class PerEntityPca:
    """Frozen, training-centered PCA projection for every graph entity."""

    width: int
    means: Tuple[NDArray[np.float64], ...]
    components: Tuple[NDArray[np.float64], ...]
    schema: _FrozenTokenSchema

    def transform(
        self,
        tokens: HybridGraphTokens,
    ) -> NDArray[np.float64]:
        """Apply training-fitted projections without refitting."""

        _validate_schema(self.schema, tokens)
        _validate_context(tokens)
        sample_count = len(tokens.fine_context)
        entity_count = len(tokens.entity_names)
        representation = np.zeros(
            (sample_count, entity_count, self.width),
            dtype=np.float64,
        )
        for entity_position in range(entity_count):
            active = tokens.feature_mask[entity_position]
            local = tokens.fine_context[
                :, :, entity_position, :
            ][..., active].reshape(sample_count, -1)
            components = self.components[entity_position]
            if len(components) == 0:
                continue
            representation[
                :, entity_position, : len(components)
            ] = (
                local - self.means[entity_position]
            ) @ components.T
        return representation


@dataclass(frozen=True)
class FrozenRidgeFutureProbe:
    """Training-fitted ridge maps from graph embeddings to future state."""

    mode: str
    ridge: float
    representation_width: int
    weights: Tuple[NDArray[np.float64], ...]
    design_means: Tuple[NDArray[np.float64], ...]
    design_scales: Tuple[NDArray[np.float64], ...]
    schema: _FrozenTokenSchema

    def predict(
        self,
        tokens: HybridGraphTokens,
        representations: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Apply stored ridge weights to a compatible split."""

        _validate_schema(self.schema, tokens)
        _validate_representations(
            tokens,
            representations,
            expected_width=self.representation_width,
        )
        if not np.all(np.isfinite(tokens.target_controls)):
            raise ValueError("target controls must be finite")
        prediction = np.zeros_like(
            tokens.fine_targets,
            dtype=np.float64,
        )
        for entity_position, weights in enumerate(self.weights):
            active = tokens.feature_mask[entity_position]
            if not np.any(active):
                continue
            design = _probe_design(
                tokens,
                representations,
                self.mode,
                entity_position,
            )
            design = _standardize_probe_design(
                design,
                self.design_means[entity_position],
                self.design_scales[entity_position],
            )
            local_prediction = design @ weights
            prediction[
                :, :, :, entity_position, :
            ][..., active] = local_prediction.reshape(
                tokens.fine_targets.shape[:3]
                + (int(np.count_nonzero(active)),)
            )
        return prediction


def fit_per_entity_pca(
    training: HybridGraphTokens,
    *,
    width: int,
) -> PerEntityPca:
    """Fit deterministic local PCA projections on training context only."""

    if isinstance(width, bool) or width < 1:
        raise ValueError("per-entity PCA width must be positive")
    _validate_context(training)
    if len(training.fine_context) < 1:
        raise ValueError("PCA training context cannot be empty")

    means = []
    components = []
    sample_count = len(training.fine_context)
    for entity_position in range(len(training.entity_names)):
        active = training.feature_mask[entity_position]
        local = training.fine_context[
            :, :, entity_position, :
        ][..., active].reshape(sample_count, -1)
        mean = np.mean(local, axis=0)
        centered = local - mean
        if centered.shape[1] == 0:
            local_components = np.zeros((0, 0), dtype=np.float64)
        else:
            _, singular_values, right_vectors = np.linalg.svd(
                centered,
                full_matrices=False,
            )
            if len(singular_values) == 0:
                component_count = 0
            else:
                tolerance = (
                    max(centered.shape)
                    * np.finfo(np.float64).eps
                    * float(singular_values[0])
                )
                component_count = min(
                    width,
                    int(np.count_nonzero(singular_values > tolerance)),
                )
            local_components = right_vectors[:component_count].copy()
            _orient_components(local_components)
        means.append(mean)
        components.append(local_components)
    return PerEntityPca(
        width=width,
        means=tuple(means),
        components=tuple(components),
        schema=_schema_from_tokens(training),
    )


def raw_context_representation(
    tokens: HybridGraphTokens,
) -> NDArray[np.float64]:
    """Flatten observed local context and pad entities to one width."""

    _validate_context(tokens)
    sample_count, time_count, entity_count, _ = (
        tokens.fine_context.shape
    )
    local_widths = tuple(
        time_count * int(np.count_nonzero(tokens.feature_mask[position]))
        for position in range(entity_count)
    )
    maximum_width = max(local_widths, default=0)
    if maximum_width == 0:
        raise ValueError(
            "raw context representation requires an observed channel"
        )
    representation = np.zeros(
        (sample_count, entity_count, maximum_width),
        dtype=np.float64,
    )
    for entity_position, local_width in enumerate(local_widths):
        if local_width == 0:
            continue
        active = tokens.feature_mask[entity_position]
        representation[
            :, entity_position, :local_width
        ] = tokens.fine_context[
            :, :, entity_position, :
        ][..., active].reshape(sample_count, local_width)
    return representation


def fit_frozen_ridge_future_probe(
    training: HybridGraphTokens,
    representations: NDArray[np.float64],
    *,
    mode: str = "all_entities",
    ridge: float = 1e-6,
) -> FrozenRidgeFutureProbe:
    """Fit observed-channel future probes on the training split only."""

    if mode not in PROBE_MODES:
        raise ValueError("unsupported frozen probe mode")
    if not np.isfinite(ridge) or ridge <= 0.0:
        raise ValueError("frozen probe ridge must be finite and positive")
    _validate_representations(training, representations)
    if not (
        np.all(np.isfinite(training.fine_targets))
        and np.all(np.isfinite(training.target_controls))
    ):
        raise ValueError("probe training targets and controls must be finite")

    entity_count = len(training.entity_names)
    if mode == "all_entities":
        design = _probe_design(
            training,
            representations,
            mode,
            0,
        )
        mean, scale = _fit_probe_design_standardization(design)
        design = _standardize_probe_design(design, mean, scale)
        target_parts = []
        target_widths = []
        for entity_position in range(entity_count):
            active = training.feature_mask[entity_position]
            target_width = int(np.count_nonzero(active))
            target_widths.append(target_width)
            if target_width:
                target_parts.append(
                    training.fine_targets[
                        :, :, :, entity_position, :
                    ][..., active].reshape(len(design), target_width)
                )
        shared_weights = (
            _ridge_fit(
                design,
                np.concatenate(target_parts, axis=1),
                ridge,
            )
            if target_parts
            else np.zeros((design.shape[1], 0), dtype=np.float64)
        )
        weights = []
        start = 0
        for target_width in target_widths:
            weights.append(
                shared_weights[:, start : start + target_width].copy()
            )
            start += target_width
        design_means = [mean.copy() for _ in range(entity_count)]
        design_scales = [scale.copy() for _ in range(entity_count)]
    else:
        weights = []
        design_means = []
        design_scales = []
        for entity_position in range(entity_count):
            active = training.feature_mask[entity_position]
            design = _probe_design(
                training,
                representations,
                mode,
                entity_position,
            )
            mean, scale = _fit_probe_design_standardization(design)
            design = _standardize_probe_design(design, mean, scale)
            design_means.append(mean)
            design_scales.append(scale)
            if not np.any(active):
                weights.append(
                    np.zeros((design.shape[1], 0), dtype=np.float64)
                )
                continue
            targets = training.fine_targets[
                :, :, :, entity_position, :
            ][..., active].reshape(
                len(design),
                int(np.count_nonzero(active)),
            )
            weights.append(_ridge_fit(design, targets, ridge))
    return FrozenRidgeFutureProbe(
        mode=mode,
        ridge=ridge,
        representation_width=representations.shape[2],
        weights=tuple(weights),
        design_means=tuple(design_means),
        design_scales=tuple(design_scales),
        schema=_schema_from_tokens(training),
    )


def _probe_design(
    tokens: HybridGraphTokens,
    representations: NDArray[np.float64],
    mode: str,
    entity_position: int,
) -> NDArray[np.float64]:
    sample_count, horizon_count, block_count, _ = (
        tokens.target_controls.shape
    )
    if mode == "all_entities":
        base = representations.reshape(sample_count, -1)
    elif mode == "entity_local":
        base = representations[:, entity_position, :]
    else:
        entity_positions = _one_hop_entity_positions(
            tokens,
            entity_position,
        )
        base = representations[:, entity_positions, :].reshape(
            sample_count,
            -1,
        )
    expanded = np.broadcast_to(
        base[:, None, None, :],
        (
            sample_count,
            horizon_count,
            block_count,
            base.shape[1],
        ),
    )
    horizon_positions = np.broadcast_to(
        np.eye(horizon_count, dtype=np.float64)[
            None, :, None, :
        ],
        (
            sample_count,
            horizon_count,
            block_count,
            horizon_count,
        ),
    )
    block_positions = np.broadcast_to(
        np.eye(block_count, dtype=np.float64)[
            None, None, :, :
        ],
        (
            sample_count,
            horizon_count,
            block_count,
            block_count,
        ),
    )
    return np.concatenate(
        (
            expanded,
            tokens.target_controls,
            horizon_positions,
            block_positions,
            np.ones(
                (
                    sample_count,
                    horizon_count,
                    block_count,
                    1,
                ),
                dtype=np.float64,
            ),
        ),
        axis=-1,
    ).reshape(sample_count * horizon_count * block_count, -1)


def _one_hop_entity_positions(
    tokens: HybridGraphTokens,
    entity_position: int,
) -> NDArray[np.int64]:
    adjacency = np.any(tokens.typed_adjacency, axis=0)
    adjacency = np.logical_or(adjacency, adjacency.T)
    selected = adjacency[entity_position].copy()
    selected[entity_position] = True
    return np.flatnonzero(selected).astype(np.int64)


def _ridge_fit(
    design: NDArray[np.float64],
    targets: NDArray[np.float64],
    ridge: float,
) -> NDArray[np.float64]:
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge
    penalty[-1, -1] = 0.0
    return np.linalg.solve(
        design.T @ design + penalty,
        design.T @ targets,
    )


def _fit_probe_design_standardization(
    design: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    non_bias = design[:, :-1]
    mean = np.mean(non_bias, axis=0)
    scale = np.std(non_bias, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return mean, scale


def _standardize_probe_design(
    design: NDArray[np.float64],
    mean: NDArray[np.float64],
    scale: NDArray[np.float64],
) -> NDArray[np.float64]:
    standardized = design.copy()
    standardized[:, :-1] = (standardized[:, :-1] - mean) / scale
    return standardized


def _orient_components(components: NDArray[np.float64]) -> None:
    for component in components:
        pivot = int(np.argmax(np.abs(component)))
        if component[pivot] < 0.0:
            component *= -1.0


def _validate_context(tokens: HybridGraphTokens) -> None:
    if (
        tokens.fine_context.ndim != 4
        or tokens.feature_mask.shape
        != tokens.fine_context.shape[2:]
        or len(tokens.entity_names) != tokens.fine_context.shape[2]
    ):
        raise ValueError("hybrid graph context schema does not align")
    if not np.all(np.isfinite(tokens.fine_context)):
        raise ValueError("hybrid graph context must be finite")


def _validate_representations(
    tokens: HybridGraphTokens,
    representations: NDArray[np.float64],
    *,
    expected_width: int = -1,
) -> None:
    expected_prefix = (
        len(tokens.fine_context),
        len(tokens.entity_names),
    )
    if (
        representations.ndim != 3
        or representations.shape[:2] != expected_prefix
        or representations.shape[2] < 1
        or (
            expected_width >= 0
            and representations.shape[2] != expected_width
        )
    ):
        raise ValueError(
            "probe representation shape must be sample-by-entity-by-width"
        )
    if not np.all(np.isfinite(representations)):
        raise ValueError("probe representations must be finite")


def _schema_from_tokens(
    tokens: HybridGraphTokens,
) -> _FrozenTokenSchema:
    context_shape = tokens.fine_context.shape
    target_shape = tokens.fine_targets.shape
    control_shape = tokens.target_controls.shape
    return _FrozenTokenSchema(
        entity_names=tokens.entity_names,
        kind_ids=tokens.kind_ids.copy(),
        entity_type_ids=tokens.entity_type_ids.copy(),
        relation_ids=tokens.relation_ids.copy(),
        typed_adjacency=tokens.typed_adjacency.copy(),
        feature_mask=tokens.feature_mask.copy(),
        feature_names=tokens.feature_names,
        local_feature_keys=tokens.local_feature_keys,
        horizons=tokens.horizons,
        context_shape=(
            context_shape[1],
            context_shape[2],
            context_shape[3],
        ),
        target_shape=(
            target_shape[1],
            target_shape[2],
            target_shape[3],
            target_shape[4],
        ),
        control_shape=(
            control_shape[1],
            control_shape[2],
            control_shape[3],
        ),
    )


def _validate_schema(
    schema: _FrozenTokenSchema,
    tokens: HybridGraphTokens,
) -> None:
    scalar_metadata_matches = (
        schema.entity_names == tokens.entity_names
        and schema.feature_names == tokens.feature_names
        and schema.local_feature_keys == tokens.local_feature_keys
        and schema.horizons == tokens.horizons
        and schema.context_shape == tokens.fine_context.shape[1:]
        and schema.target_shape == tokens.fine_targets.shape[1:]
        and schema.control_shape == tokens.target_controls.shape[1:]
    )
    array_metadata_matches = all(
        np.array_equal(reference, candidate)
        for reference, candidate in (
            (schema.kind_ids, tokens.kind_ids),
            (schema.entity_type_ids, tokens.entity_type_ids),
            (schema.relation_ids, tokens.relation_ids),
            (schema.typed_adjacency, tokens.typed_adjacency),
            (schema.feature_mask, tokens.feature_mask),
        )
    )
    if not scalar_metadata_matches or not array_metadata_matches:
        raise ValueError("hybrid frozen model token schema mismatch")
