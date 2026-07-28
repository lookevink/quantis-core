import json

import numpy as np

from quantis_core.contextual_multimodal_corpus import (
    DEPENDENCY_LOG_FEATURE_NAMES,
    ContextualMultimodalModelWindows,
)
from quantis_core.graph_telemetry import (
    compile_graph_state_windows,
)
from quantis_core.learned_graph_jepa import (
    GraphEmaJepaConfig,
    LearnedGraphJepaWorldModel,
    evaluate_learned_graph_jepa,
)
from quantis_core.observability_graph_corpus import (
    OBSERVABILITY_METRIC_FEATURE_NAMES,
    quantis_checkout_observability_graph,
)


def test_learned_graph_jepa_is_deterministic_and_serializes_ema_target() -> None:
    windows = _windows()
    widths = {
        entity_id: (
            0
            if entity_id == "queue_hosted_on_redis"
            else 1
        )
        for entity_id in windows.entity_ids
    }
    config = GraphEmaJepaConfig(
        entity_latent_dimensions=widths,
        context_scope="one_hop",
        epochs=12,
        learning_rate=0.002,
        batch_size=17,
        seed=31,
    )

    first = LearnedGraphJepaWorldModel(config).fit(windows)
    second = LearnedGraphJepaWorldModel(config).fit(windows)
    prediction = first.predict(windows)
    assessment = evaluate_learned_graph_jepa(first, windows)

    assert json.dumps(
        first.to_dict(), sort_keys=True
    ) == json.dumps(second.to_dict(), sort_keys=True)
    assert first.to_dict()["ema_target_encoders"]
    assert first.to_dict()["online_encoders"]
    assert prediction.decoded_target_blocks.shape == (
        windows.target_blocks.shape
    )
    assert np.isfinite(prediction.decoded_target_blocks).all()
    assert assessment["compression"]["ratio"] > 1.0
    assert len(first.training_losses) == 12


def _windows():
    generator = np.random.default_rng(7)
    sample_count = 24
    lookback = 4
    horizon_count = 2
    block_size = 2
    metric_count = len(OBSERVABILITY_METRIC_FEATURE_NAMES)
    log_count = len(DEPENDENCY_LOG_FEATURE_NAMES)
    metric_contexts = generator.normal(
        0.0,
        0.3,
        size=(sample_count, lookback, metric_count),
    )
    log_contexts = generator.normal(
        0.0,
        0.3,
        size=(sample_count, lookback, log_count),
    )
    metric_targets = np.repeat(
        metric_contexts[:, -1:, :],
        horizon_count * block_size,
        axis=1,
    ).reshape(
        sample_count,
        horizon_count,
        block_size,
        metric_count,
    )
    log_targets = np.repeat(
        log_contexts[:, -1:, :],
        horizon_count * block_size,
        axis=1,
    ).reshape(
        sample_count,
        horizon_count,
        block_size,
        log_count,
    )
    contextual = ContextualMultimodalModelWindows(
        metric_contexts=metric_contexts,
        log_contexts=log_contexts,
        metric_target_blocks=metric_targets,
        log_target_blocks=log_targets,
        target_controls=np.zeros(
            (sample_count, horizon_count, block_size, 2),
            dtype=np.float64,
        ),
        point_indices=np.arange(sample_count, dtype=np.int64),
        metric_feature_names=OBSERVABILITY_METRIC_FEATURE_NAMES,
        log_feature_names=DEPENDENCY_LOG_FEATURE_NAMES,
        control_feature_names=(
            "request_demand",
            "worker_replicas",
        ),
        horizons=(1, 3),
        target_block_size=block_size,
    )
    return compile_graph_state_windows(
        contextual, quantis_checkout_observability_graph()
    )
