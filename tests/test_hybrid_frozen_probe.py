from dataclasses import replace
from typing import Optional

import numpy as np
import pytest
from numpy.typing import NDArray

from quantis_core.graph_telemetry import (
    DeclaredTelemetryGraph,
    GraphEntity,
    GraphStateWindows,
    TelemetryBinding,
)
from quantis_core.hybrid_frozen_probe import (
    fit_frozen_ridge_future_probe,
    fit_per_entity_pca,
    raw_context_representation,
)
from quantis_core.hybrid_graph_tokens import (
    HybridGraphTokens,
    compile_hybrid_graph_tokens,
)


def test_per_entity_pca_is_deterministic_masked_and_padded() -> None:
    contexts = np.zeros((3, 2, 3, 2), dtype=np.float64)
    contexts[:, :, :, 0] = np.array(
        [
            [[0.0, 0.0, 2.0], [0.0, 1.0, 0.0]],
            [[1.0, 1.0, 1.0], [1.0, 2.0, 1.0]],
            [[2.0, 2.0, 0.0], [2.0, 3.0, 2.0]],
        ]
    )
    contexts[:, :, :, 1] = 1000.0
    training = _tokens(contexts=contexts)

    first = fit_per_entity_pca(training, width=3)
    second = fit_per_entity_pca(training, width=3)
    representation = first.transform(training)
    noisy_unobserved = training.fine_context.copy()
    noisy_unobserved[..., 1] = -987654.0
    unaffected = first.transform(
        replace(training, fine_context=noisy_unobserved)
    )

    assert representation.shape == (3, 3, 3)
    np.testing.assert_allclose(representation, second.transform(training))
    np.testing.assert_allclose(representation, unaffected)
    np.testing.assert_array_equal(representation[..., 1:], 0.0)
    assert first.width == 3


@pytest.mark.parametrize("mode", ["all_entities", "entity_local"])
def test_frozen_ridge_probe_predicts_observed_targets_in_both_modes(
    mode: str,
) -> None:
    generator = np.random.default_rng(41)
    training_representations = generator.normal(size=(20, 3, 1))
    training_controls = generator.normal(size=(20, 1, 1, 1))
    coefficients = np.array([2.0, -1.5, 0.75])
    intercepts = np.array([1.0, -2.0, 0.5])
    training_targets = _probe_targets(
        training_representations,
        training_controls,
        coefficients,
        intercepts,
    )
    training = _tokens(
        contexts=np.zeros((20, 2, 3, 2)),
        targets=training_targets,
        controls=training_controls,
    )
    probe = fit_frozen_ridge_future_probe(
        training,
        training_representations,
        mode=mode,
        ridge=1e-9,
    )

    evaluation_representations = np.array(
        [[[0.25], [-1.0], [2.0]], [[-0.5], [0.5], [1.5]]]
    )
    evaluation_controls = np.array([0.2, -0.4]).reshape(2, 1, 1, 1)
    expected = _probe_targets(
        evaluation_representations,
        evaluation_controls,
        coefficients,
        intercepts,
    )
    deliberately_wrong_targets = expected.copy()
    deliberately_wrong_targets[..., 0] += 1000.0
    evaluation = _tokens(
        contexts=np.zeros((2, 2, 3, 2)),
        targets=deliberately_wrong_targets,
        controls=evaluation_controls,
    )

    prediction = probe.predict(
        evaluation,
        evaluation_representations,
    )

    assert prediction.shape == evaluation.fine_targets.shape
    np.testing.assert_allclose(
        prediction[..., 0],
        expected[..., 0],
        atol=1e-7,
    )
    np.testing.assert_array_equal(prediction[..., 1], 0.0)


def test_probe_standardization_is_fitted_once_on_training_design() -> None:
    training_representations = np.zeros((4, 3, 2))
    training_representations[:, 0, 0] = [10.0, 12.0, 14.0, 16.0]
    training_representations[:, 0, 1] = 5.0
    training_controls = np.array([1.0, -1.0, 2.0, -2.0]).reshape(
        4, 1, 1, 1
    )
    training_targets = np.zeros((4, 1, 1, 3, 2))
    training_targets[:, 0, 0, 0, 0] = (
        2.0 * training_representations[:, 0, 0]
        + 3.0 * training_controls[:, 0, 0, 0]
        + 1.0
    )
    training = _tokens(
        contexts=np.zeros((4, 2, 3, 2)),
        targets=training_targets,
        controls=training_controls,
    )

    probe = fit_frozen_ridge_future_probe(
        training,
        training_representations,
        mode="entity_local",
        ridge=1e-9,
    )

    np.testing.assert_allclose(
        probe.design_means[0],
        [13.0, 5.0, 0.0, 1.0, 1.0],
    )
    np.testing.assert_allclose(
        probe.design_scales[0],
        [np.sqrt(5.0), 1.0, np.sqrt(2.5), 1.0, 1.0],
    )

    evaluation_representations = np.zeros((2, 3, 2))
    evaluation_representations[:, 0, 0] = [100.0, 200.0]
    evaluation_representations[:, 0, 1] = 5.0
    evaluation_controls = np.array([10.0, -10.0]).reshape(
        2, 1, 1, 1
    )
    evaluation = _tokens(
        contexts=np.zeros((2, 2, 3, 2)),
        controls=evaluation_controls,
    )

    prediction = probe.predict(
        evaluation,
        evaluation_representations,
    )

    np.testing.assert_allclose(
        prediction[:, 0, 0, 0, 0],
        [231.0, 371.0],
        atol=1e-6,
    )


def test_one_hop_probe_design_follows_declared_typed_adjacency() -> None:
    representations = np.array(
        [
            [[0.0], [10.0], [20.0]],
            [[2.0], [12.0], [22.0]],
            [[4.0], [14.0], [24.0]],
        ]
    )
    controls = np.array([1.0, 2.0, 3.0]).reshape(3, 1, 1, 1)
    training = _tokens(
        contexts=np.zeros((3, 2, 3, 2)),
        controls=controls,
    )

    probe = fit_frozen_ridge_future_probe(
        training,
        representations,
        mode="one_hop",
    )

    assert probe.mode == "one_hop"
    np.testing.assert_array_equal(
        probe.design_means[0],
        [2.0, 22.0, 2.0, 1.0, 1.0],
    )
    np.testing.assert_array_equal(
        probe.design_means[1],
        [12.0, 22.0, 2.0, 1.0, 1.0],
    )
    np.testing.assert_array_equal(
        probe.design_means[2],
        [2.0, 12.0, 22.0, 2.0, 1.0, 1.0],
    )


def test_probe_can_fit_distinct_horizon_and_block_positions() -> None:
    sample_count = 4
    targets = np.zeros((sample_count, 2, 2, 3, 2))
    expected = np.array([[1.0, 2.0], [11.0, 12.0]])
    targets[:, :, :, 0, 0] = expected[None, :, :]
    controls = np.zeros((sample_count, 2, 2, 1))
    training = _tokens(
        contexts=np.zeros((sample_count, 2, 3, 2)),
        targets=targets,
        controls=controls,
    )
    representations = np.zeros((sample_count, 3, 1))

    probe = fit_frozen_ridge_future_probe(
        training,
        representations,
        mode="all_entities",
        ridge=1e-9,
    )
    prediction = probe.predict(training, representations)

    np.testing.assert_allclose(
        prediction[0, :, :, 0, 0],
        expected,
        atol=1e-7,
    )


def test_raw_context_representation_flattens_observed_channels_and_pads() -> None:
    contexts = np.arange(2 * 2 * 3 * 2, dtype=np.float64).reshape(
        2, 2, 3, 2
    )
    tokens = _tokens(contexts=contexts)

    representation = raw_context_representation(tokens)

    assert representation.shape == (2, 3, 2)
    for entity_position in range(3):
        np.testing.assert_array_equal(
            representation[:, entity_position],
            contexts[:, :, entity_position, 0],
        )

    tokens.fine_context[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="context must be finite"):
        raw_context_representation(tokens)


def test_frozen_models_reject_invalid_data_and_schema_drift() -> None:
    training = _tokens(contexts=np.zeros((3, 2, 3, 2)))

    with pytest.raises(ValueError, match="PCA width"):
        fit_per_entity_pca(training, width=0)

    pca = fit_per_entity_pca(training, width=2)
    training.fine_context[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="context must be finite"):
        pca.transform(training)
    training.fine_context[0, 0, 0, 0] = 0.0

    representations = np.zeros((3, 3, 1))
    with pytest.raises(ValueError, match="probe mode"):
        fit_frozen_ridge_future_probe(
            training,
            representations,
            mode="unknown",
        )
    with pytest.raises(ValueError, match="representation shape"):
        fit_frozen_ridge_future_probe(
            training,
            np.zeros((3, 2, 1)),
        )

    probe = fit_frozen_ridge_future_probe(
        training,
        representations,
    )
    mismatched = replace(
        training,
        entity_names=("renamed", "database", "writes"),
    )
    with pytest.raises(ValueError, match="schema"):
        probe.predict(mismatched, representations)
    bad_representations = representations.copy()
    bad_representations[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        probe.predict(training, bad_representations)


def _probe_targets(
    representations: NDArray[np.float64],
    controls: NDArray[np.float64],
    coefficients: NDArray[np.float64],
    intercepts: NDArray[np.float64],
) -> NDArray[np.float64]:
    sample_count, entity_count, _ = representations.shape
    targets = np.full(
        (sample_count, 1, 1, entity_count, 2),
        99.0,
        dtype=np.float64,
    )
    targets[:, 0, 0, :, 0] = (
        representations[:, :, 0] * coefficients[None, :]
        + controls[:, 0, 0, 0, None]
        + intercepts[None, :]
    )
    return targets


def _tokens(
    *,
    contexts: NDArray[np.float64],
    targets: Optional[NDArray[np.float64]] = None,
    controls: Optional[NDArray[np.float64]] = None,
) -> HybridGraphTokens:
    sample_count = len(contexts)
    if targets is None:
        targets = np.zeros((sample_count, 1, 1, 3, 2))
    if controls is None:
        controls = np.zeros((sample_count, 1, 1, 1))
    graph = DeclaredTelemetryGraph(
        entities=(
            GraphEntity("frontend", "node", "service"),
            GraphEntity("database", "node", "dependency"),
            GraphEntity(
                "writes",
                "edge",
                "database_write",
                "frontend",
                "database",
            ),
        ),
        bindings=(
            TelemetryBinding("metric.latency", "frontend"),
            TelemetryBinding("metric.connections", "database"),
            TelemetryBinding("log.write", "writes"),
        ),
    )
    windows = GraphStateWindows(
        contexts=contexts,
        target_blocks=targets,
        target_controls=controls,
        point_indices=np.arange(sample_count, dtype=np.int64),
        observation_mask=np.array(
            [[True, False], [True, False], [True, False]]
        ),
        entity_ids=("frontend", "database", "writes"),
        entity_kinds=("node", "node", "edge"),
        local_feature_keys=(
            ("metric.latency",),
            ("metric.connections",),
            ("log.write",),
        ),
        control_feature_names=("load",),
        horizons=tuple(range(1, targets.shape[1] + 1)),
        target_block_size=targets.shape[2],
        graph=graph,
    )
    return compile_hybrid_graph_tokens(windows, coarse_factor=1)
