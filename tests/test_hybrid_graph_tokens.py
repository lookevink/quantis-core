import numpy as np
import pytest
from numpy.typing import NDArray

from quantis_core.graph_telemetry import (
    DeclaredTelemetryGraph,
    GraphEntity,
    GraphStateWindows,
    TelemetryBinding,
)
from quantis_core.hybrid_graph_tokens import (
    AlignedEventFeatures,
    MultiMaskConfig,
    compile_hybrid_graph_tokens,
    sample_multi_masks,
)


def test_compiler_emits_fine_coarse_and_typed_graph_tokens_without_aliasing() -> None:
    windows = _graph_windows()
    event_contexts = np.full((1, 4, 3, 1), 100.0)
    event_targets = np.full((1, 1, 2, 3, 1), 200.0)
    event_features = AlignedEventFeatures(
        contexts=event_contexts,
        target_blocks=event_targets,
        observation_mask=np.array([[True], [False], [True]]),
        feature_names=("event.count",),
    )
    original_contexts = windows.contexts.copy()
    original_events = event_contexts.copy()

    tokens = compile_hybrid_graph_tokens(
        windows,
        coarse_factor=2,
        aligned_event_features=event_features,
    )

    assert tokens.fine_context.shape == (1, 4, 3, 3)
    assert tokens.fine_targets.shape == (1, 1, 2, 3, 3)
    np.testing.assert_array_equal(
        tokens.fine_context[..., :2], windows.contexts
    )
    np.testing.assert_array_equal(
        tokens.fine_context[..., 2:], event_contexts
    )
    np.testing.assert_array_equal(
        tokens.coarse_context,
        np.mean(tokens.fine_context.reshape(1, 2, 2, 3, 3), axis=2),
    )
    np.testing.assert_array_equal(
        tokens.coarse_targets,
        np.mean(
            tokens.fine_targets.reshape(1, 1, 1, 2, 3, 3),
            axis=3,
        ),
    )
    np.testing.assert_array_equal(tokens.entity_ids, [0, 1, 2])
    assert tokens.entity_names == ("frontend", "database", "writes")
    np.testing.assert_array_equal(tokens.kind_ids, [0, 0, 1])
    assert tokens.kind_names == ("node", "edge")
    np.testing.assert_array_equal(tokens.relation_ids, [-1, -1, 0])
    assert tokens.relation_names == ("database_write",)
    assert tokens.typed_adjacency.shape == (1, 3, 3)
    assert tokens.typed_adjacency[0, 0, 2]
    assert tokens.typed_adjacency[0, 2, 1]
    assert np.count_nonzero(tokens.typed_adjacency) == 2
    np.testing.assert_array_equal(
        tokens.feature_mask,
        [
            [True, False, True],
            [True, True, False],
            [True, False, True],
        ],
    )
    assert tokens.feature_names[-1] == "event.count"

    tokens.fine_context[0, 0, 0, 0] = -999.0
    tokens.fine_context[0, 0, 0, 2] = -999.0
    np.testing.assert_array_equal(windows.contexts, original_contexts)
    np.testing.assert_array_equal(event_contexts, original_events)


def test_compiler_rejects_misaligned_or_nonfinite_event_features() -> None:
    windows = _graph_windows()

    with pytest.raises(ValueError, match="event contexts do not align"):
        compile_hybrid_graph_tokens(
            windows,
            aligned_event_features=AlignedEventFeatures(
                contexts=np.zeros((2, 4, 3, 1)),
                target_blocks=np.zeros((1, 1, 2, 3, 1)),
                observation_mask=np.ones((3, 1), dtype=np.bool_),
                feature_names=("event.count",),
            ),
        )

    bad_contexts = np.zeros((1, 4, 3, 1))
    bad_contexts[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="event features must be finite"):
        AlignedEventFeatures(
            contexts=bad_contexts,
            target_blocks=np.zeros((1, 1, 2, 3, 1)),
            observation_mask=np.ones((3, 1), dtype=np.bool_),
            feature_names=("event.count",),
        )


def test_seeded_multi_masks_are_reproducible_contiguous_and_nonmutating() -> None:
    tokens = compile_hybrid_graph_tokens(_graph_windows())
    original = tokens.fine_context.copy()
    config = MultiMaskConfig(
        mask_count=3,
        target_coverage=0.5,
        seed=17,
    )

    first = sample_multi_masks(tokens, config)
    second = sample_multi_masks(tokens, config)
    different = sample_multi_masks(
        tokens,
        MultiMaskConfig(mask_count=3, target_coverage=0.5, seed=23),
    )

    assert first.target_masks.shape == (3, 1, 4, 3)
    np.testing.assert_array_equal(first.target_masks, second.target_masks)
    assert not np.array_equal(
        first.target_masks, different.target_masks
    )
    np.testing.assert_array_equal(
        first.context_masks, np.logical_not(first.target_masks)
    )
    np.testing.assert_array_equal(tokens.fine_context, original)
    for target_mask in first.target_masks[:, 0]:
        assert abs(float(np.mean(target_mask)) - 0.5) <= 0.1
        selected_times = np.flatnonzero(np.any(target_mask, axis=1))
        np.testing.assert_array_equal(
            selected_times,
            np.arange(selected_times[0], selected_times[-1] + 1),
        )
        selected_entities = np.flatnonzero(np.any(target_mask, axis=0))
        assert _is_connected(selected_entities, tokens.typed_adjacency)


def test_multi_mask_configuration_rejects_degenerate_masks() -> None:
    with pytest.raises(ValueError, match="mask count"):
        MultiMaskConfig(mask_count=0)
    with pytest.raises(ValueError, match="target coverage"):
        MultiMaskConfig(target_coverage=1.0)


def _graph_windows() -> GraphStateWindows:
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
            TelemetryBinding("log.slow_write", "database"),
            TelemetryBinding("log.write", "writes"),
        ),
    )
    contexts = np.arange(1 * 4 * 3 * 2, dtype=np.float64).reshape(
        1, 4, 3, 2
    )
    targets = np.arange(1 * 1 * 2 * 3 * 2, dtype=np.float64).reshape(
        1, 1, 2, 3, 2
    )
    return GraphStateWindows(
        contexts=contexts,
        target_blocks=targets,
        target_controls=np.zeros((1, 1, 2, 1), dtype=np.float64),
        point_indices=np.array([4], dtype=np.int64),
        observation_mask=np.array(
            [[True, False], [True, True], [True, False]],
            dtype=np.bool_,
        ),
        entity_ids=("frontend", "database", "writes"),
        entity_kinds=("node", "node", "edge"),
        local_feature_keys=(
            ("metric.latency",),
            ("metric.connections", "log.slow_write"),
            ("log.write",),
        ),
        control_feature_names=("load",),
        horizons=(1,),
        target_block_size=2,
        graph=graph,
    )


def _is_connected(
    selected: NDArray[np.int64],
    typed_adjacency: NDArray[np.bool_],
) -> bool:
    if len(selected) <= 1:
        return True
    adjacency = np.any(typed_adjacency, axis=0)
    adjacency = np.logical_or(adjacency, adjacency.T)
    reachable = {int(selected[0])}
    pending = [int(selected[0])]
    selected_set = {int(value) for value in selected}
    while pending:
        node = pending.pop()
        for neighbor in np.flatnonzero(adjacency[node]):
            value = int(neighbor)
            if value in selected_set and value not in reachable:
                reachable.add(value)
                pending.append(value)
    return reachable == selected_set
