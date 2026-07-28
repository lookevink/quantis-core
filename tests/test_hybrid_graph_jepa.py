import json
from dataclasses import replace

import numpy as np
import pytest

from quantis_core.hybrid_graph_jepa import (
    HybridGraphJepa,
    HybridJepaConfig,
    latent_diagnostics,
    pool_context_mask,
)
from quantis_core.hybrid_graph_tokens import (
    HybridGraphTokens,
    MultiMaskBatch,
)


def test_hybrid_jepa_fits_deterministically_and_decodes_local_state() -> None:
    tokens = _tokens()
    config = HybridJepaConfig(
        latent_dimension=4,
        attention_heads=2,
        transformer_layers=1,
        epochs=3,
        batch_size=4,
        learning_rate=0.01,
        device="cpu",
        seed=17,
    )

    first = HybridGraphJepa(config).fit(tokens)
    second = HybridGraphJepa(config).fit(tokens)
    prediction = first.predict(tokens)
    restored = HybridGraphJepa.from_dict(first.to_dict())
    restored_prediction = restored.predict(tokens)

    assert prediction.reconstructed_targets.shape == tokens.fine_targets.shape
    assert prediction.reconstructed_contexts.shape == tokens.fine_context.shape
    assert prediction.predicted_latents.shape == (
        len(tokens.fine_context),
        len(tokens.horizons),
        len(tokens.entity_ids),
        config.latent_dimension,
    )
    assert prediction.validation_embeddings.shape == (
        len(tokens.fine_context),
        len(tokens.entity_ids) * config.latent_dimension,
    )
    assert np.isfinite(prediction.reconstructed_targets).all()
    assert first.training_losses == second.training_losses
    assert first.training_losses[-1] <= first.training_losses[0] * 1.25
    assert json.dumps(first.to_dict(), sort_keys=True)
    assert first.to_dict()["training_topology"] == (
        tokens.typed_adjacency.tolist()
    )
    np.testing.assert_allclose(
        restored_prediction.predicted_latents,
        prediction.predicted_latents,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        first.decode_context_embeddings(
            prediction.validation_embeddings
        ),
        prediction.reconstructed_contexts,
        atol=1e-6,
    )


def test_latent_diagnostics_identify_collapsed_representations() -> None:
    collapsed = np.ones((12, 5), dtype=np.float64)
    healthy = np.eye(5, dtype=np.float64)

    collapsed_result = latent_diagnostics(collapsed)
    healthy_result = latent_diagnostics(healthy)

    assert collapsed_result["effective_rank"] == 0.0
    assert collapsed_result["minimum_dimension_variance"] == 0.0
    # Centering five one-hot rows removes the shared mean direction.
    assert healthy_result["effective_rank"] > 3.9
    assert len(healthy_result["per_dimension_variance"]) == 5
    assert healthy_result["mean_absolute_off_diagonal_covariance"] > 0.0


def test_declared_topology_changes_hybrid_predictions() -> None:
    tokens = _tokens()
    model = HybridGraphJepa(
        HybridJepaConfig(
            latent_dimension=4,
            attention_heads=2,
            epochs=1,
            batch_size=6,
            device="cpu",
            seed=23,
        )
    ).fit(tokens)
    disconnected = replace(
        tokens,
        typed_adjacency=np.zeros_like(tokens.typed_adjacency),
    )

    connected_prediction = model.predict(tokens).predicted_latents
    with pytest.raises(ValueError, match="topology"):
        model.predict(disconnected)
    disconnected_prediction = model.predict(
        disconnected,
        allow_topology_ablation=True,
    ).predicted_latents

    assert not np.allclose(
        connected_prediction,
        disconnected_prediction,
        atol=1e-8,
    )


def test_target_controls_condition_future_predictions() -> None:
    tokens = _tokens()
    model = HybridGraphJepa(
        HybridJepaConfig(
            latent_dimension=4,
            attention_heads=2,
            epochs=1,
            batch_size=6,
            device="cpu",
            seed=29,
        )
    ).fit(tokens)
    changed_controls = replace(
        tokens,
        target_controls=np.ones_like(tokens.target_controls),
    )

    baseline = model.predict(tokens).predicted_latents
    conditioned = model.predict(changed_controls).predicted_latents

    assert not np.allclose(baseline, conditioned, atol=1e-8)


def test_fine_visibility_is_conservatively_pooled_for_coarse_tokens() -> None:
    fine = np.asarray(
        [
            [
                [True, True],
                [True, False],
                [True, True],
                [False, False],
                [True, True],
            ]
        ],
        dtype=np.bool_,
    )

    pooled = pool_context_mask(fine, coarse_factor=2)

    np.testing.assert_array_equal(
        pooled,
        np.asarray(
            [[[True, False], [False, False], [True, True]]],
            dtype=np.bool_,
        ),
    )


def test_fit_accepts_declared_masks_and_rejects_misaligned_masks() -> None:
    tokens = _tokens()
    target = np.zeros(
        (2,) + tokens.fine_context.shape[:-1],
        dtype=np.bool_,
    )
    target[:, :, 1:3, 1] = True
    masks = MultiMaskBatch(
        context_masks=np.logical_not(target),
        target_masks=target,
    )
    model = HybridGraphJepa(
        HybridJepaConfig(
            latent_dimension=4,
            attention_heads=2,
            epochs=1,
            batch_size=6,
            device="cpu",
            seed=31,
        )
    ).fit(tokens, masks=masks)
    invalid_target = target[:, :-1]
    invalid = MultiMaskBatch(
        context_masks=np.logical_not(invalid_target),
        target_masks=invalid_target,
    )

    assert model.training_losses
    with pytest.raises(ValueError, match="mask"):
        HybridGraphJepa(model.config).fit(tokens, masks=invalid)


def test_prediction_rejects_semantically_different_token_schema() -> None:
    tokens = _tokens()
    model = HybridGraphJepa(
        HybridJepaConfig(
            latent_dimension=4,
            attention_heads=2,
            epochs=1,
            batch_size=6,
            device="cpu",
            seed=37,
        )
    ).fit(tokens)
    renamed = replace(
        tokens,
        feature_names=("renamed",) + tokens.feature_names[1:],
    )

    with pytest.raises(ValueError, match="semantic"):
        model.predict(renamed)
    renamed_controls = replace(
        tokens,
        control_feature_names=("renamed_control",),
    )
    with pytest.raises(ValueError, match="semantic"):
        model.predict(renamed_controls)


def _tokens() -> HybridGraphTokens:
    generator = np.random.default_rng(3)
    samples, lookback, horizons, block, entities, features = (
        12,
        5,
        2,
        2,
        3,
        4,
    )
    context = generator.normal(
        size=(samples, lookback, entities, features)
    )
    targets = np.empty(
        (samples, horizons, block, entities, features),
        dtype=np.float64,
    )
    for horizon in range(horizons):
        targets[:, horizon] = (
            context[:, -1:, :, :]
            + float(horizon + 1) * 0.05
        )
    adjacency = np.zeros((1, entities, entities), dtype=np.bool_)
    adjacency[0, 0, 1] = True
    adjacency[0, 1, 2] = True
    return HybridGraphTokens(
        fine_context=context,
        fine_targets=targets,
        coarse_context=context[:, ::2],
        coarse_targets=targets[:, :, ::2],
        target_controls=np.zeros(
            (samples, horizons, block, 1),
            dtype=np.float64,
        ),
        control_feature_names=("request_demand",),
        feature_mask=np.ones((entities, features), dtype=np.bool_),
        entity_ids=np.arange(entities, dtype=np.int64),
        entity_names=("api", "queue", "worker"),
        kind_ids=np.asarray((0, 1, 0), dtype=np.int64),
        kind_names=("node", "edge"),
        entity_type_ids=np.asarray((0, 1, 2), dtype=np.int64),
        entity_type_names=("service", "queue", "worker"),
        relation_ids=np.asarray((-1, 0, -1), dtype=np.int64),
        relation_names=("dispatch",),
        typed_adjacency=adjacency,
        feature_names=tuple(f"feature.{index}" for index in range(features)),
        local_feature_keys=tuple(
            (f"feature.{index}",)
            for index in range(entities)
        ),
        point_indices=np.arange(samples, dtype=np.int64),
        horizons=(1, 2),
        coarse_factor=2,
    )
