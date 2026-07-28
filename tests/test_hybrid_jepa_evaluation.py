from dataclasses import replace

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
    HybridGraphTokens,
    compile_hybrid_graph_tokens,
)
from quantis_core.hybrid_jepa_evaluation import (
    embedding_effective_rank_fraction,
    score_normalized_target_blocks,
)


def test_target_score_uses_training_variance_and_reports_each_entity() -> None:
    training = compile_hybrid_graph_tokens(
        _tokens_source(
            np.array(
                [[0.0, 0.0, 0.0], [2.0, 4.0, 1.0]]
            ).reshape(2, 1, 1, 3, 1)
        )
    )
    evaluation = compile_hybrid_graph_tokens(
        _tokens_source(
            np.array([1.0, 2.0, 0.5]).reshape(1, 1, 1, 3, 1)
        )
    )
    predictions = np.array([2.0, 4.0, 1.5]).reshape(1, 1, 1, 3, 1)

    score = score_normalized_target_blocks(
        training,
        evaluation,
        predictions,
    )

    assert score.mean_normalized_mse == pytest.approx(2.0)
    assert score.entity_normalized_mse == {
        "frontend": pytest.approx(1.0),
        "database": pytest.approx(1.0),
        "writes": pytest.approx(4.0),
    }
    assert score.scored_channel_count == 3
    assert score.constant_channel_count == 0


def test_target_score_skips_constant_observed_channels() -> None:
    training_targets = np.array(
        [[0.0, 4.0, 1.0], [2.0, 4.0, 1.0]]
    ).reshape(2, 1, 1, 3, 1)
    training = compile_hybrid_graph_tokens(
        _tokens_source(training_targets)
    )
    evaluation = compile_hybrid_graph_tokens(
        _tokens_source(
            np.array([1.0, 9.0, 7.0]).reshape(1, 1, 1, 3, 1)
        )
    )

    score = score_normalized_target_blocks(
        training,
        evaluation,
        evaluation.fine_targets.copy(),
    )

    assert score.mean_normalized_mse == 0.0
    assert score.entity_normalized_mse == {
        "frontend": 0.0,
        "database": None,
        "writes": None,
    }
    assert score.scored_channel_count == 1
    assert score.constant_channel_count == 2


def test_target_score_can_select_a_subset_of_observed_channels() -> None:
    training = compile_hybrid_graph_tokens(
        _tokens_source(
            np.array(
                [[0.0, 0.0, 0.0], [2.0, 4.0, 1.0]]
            ).reshape(2, 1, 1, 3, 1)
        )
    )
    evaluation = compile_hybrid_graph_tokens(
        _tokens_source(
            np.array([1.0, 2.0, 0.5]).reshape(1, 1, 1, 3, 1)
        )
    )
    predictions = np.array([2.0, 4.0, 1.5]).reshape(1, 1, 1, 3, 1)

    score = score_normalized_target_blocks(
        training,
        evaluation,
        predictions,
        channel_mask=np.array([[True], [False], [False]]),
    )

    assert score.mean_normalized_mse == pytest.approx(1.0)
    assert score.entity_normalized_mse == {
        "frontend": pytest.approx(1.0),
        "database": None,
        "writes": None,
    }
    assert score.scored_channel_count == 1
    assert score.constant_channel_count == 0

    with pytest.raises(ValueError, match="channel mask"):
        score_normalized_target_blocks(
            training,
            evaluation,
            predictions,
            channel_mask=np.ones((3, 2), dtype=np.bool_),
        )


def test_effective_rank_fraction_detects_full_rank_and_collapse() -> None:
    full_rank = np.array(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
    )
    collapsed = np.array(
        [[1.0, 1.0], [-1.0, -1.0], [2.0, 2.0], [-2.0, -2.0]]
    )

    assert embedding_effective_rank_fraction(full_rank) == pytest.approx(
        1.0
    )
    assert embedding_effective_rank_fraction(collapsed) == pytest.approx(
        0.5
    )
    assert embedding_effective_rank_fraction(np.ones((3, 2))) == 0.0


def test_evaluation_rejects_shape_nonfinite_and_schema_mismatch() -> None:
    training = compile_hybrid_graph_tokens(
        _tokens_source(
            np.array(
                [[0.0, 0.0, 0.0], [2.0, 4.0, 1.0]]
            ).reshape(2, 1, 1, 3, 1)
        )
    )
    evaluation = compile_hybrid_graph_tokens(
        _tokens_source(
            np.array([1.0, 2.0, 0.5]).reshape(1, 1, 1, 3, 1)
        )
    )

    with pytest.raises(ValueError, match="prediction shape"):
        score_normalized_target_blocks(
            training,
            evaluation,
            np.zeros((1, 1, 1, 3, 2)),
        )

    nonfinite = evaluation.fine_targets.copy()
    nonfinite[0, 0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="predictions must be finite"):
        score_normalized_target_blocks(
            training,
            evaluation,
            nonfinite,
        )

    mismatched = replace(
        evaluation,
        entity_names=("renamed", "database", "writes"),
    )
    with pytest.raises(ValueError, match="token schemas do not match"):
        score_normalized_target_blocks(
            training,
            mismatched,
            mismatched.fine_targets,
        )

    with pytest.raises(ValueError, match="rank-2"):
        embedding_effective_rank_fraction(np.zeros((2, 2, 1)))
    bad_embedding = np.zeros((2, 2))
    bad_embedding[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        embedding_effective_rank_fraction(bad_embedding)


def _tokens_source(
    target_blocks: NDArray[np.float64],
) -> GraphStateWindows:
    sample_count = target_blocks.shape[0]
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
    return GraphStateWindows(
        contexts=np.zeros((sample_count, 2, 3, 1)),
        target_blocks=target_blocks,
        target_controls=np.zeros((sample_count, 1, 1, 1)),
        point_indices=np.arange(sample_count, dtype=np.int64),
        observation_mask=np.ones((3, 1), dtype=np.bool_),
        entity_ids=("frontend", "database", "writes"),
        entity_kinds=("node", "node", "edge"),
        local_feature_keys=(
            ("metric.latency",),
            ("metric.connections",),
            ("log.write",),
        ),
        control_feature_names=("load",),
        horizons=(1,),
        target_block_size=1,
        graph=graph,
    )
