import numpy as np
import pytest

from quantis_core.contextual_multimodal_corpus import (
    ContextualMultimodalModelWindows,
)
from quantis_core.graph_telemetry import (
    compile_graph_state_windows,
    quantis_checkout_graph,
)


def test_declared_graph_compiler_preserves_owned_node_and_edge_features() -> None:
    windows = _contextual_windows()

    graph_windows = compile_graph_state_windows(
        windows,
        quantis_checkout_graph(),
    )

    assert graph_windows.contexts.shape == (2, 2, 9, 4)
    assert graph_windows.target_blocks.shape == (2, 2, 1, 9, 4)
    assert graph_windows.entity_ids == (
        "api",
        "checkout_queue",
        "worker_pool",
        "redis",
        "postgresql",
        "api_enqueues_queue",
        "queue_dequeues_to_worker",
        "queue_hosted_on_redis",
        "worker_writes_postgresql",
    )
    api_position, api_slot = graph_windows.feature_position(
        "metric.request_latency_ms"
    )
    edge_position, edge_slot = graph_windows.feature_position(
        "log.checkout_backlog_delta_ratio"
    )
    assert graph_windows.entity_ids[api_position] == "api"
    assert (
        graph_windows.entity_ids[edge_position]
        == "api_enqueues_queue"
    )
    assert graph_windows.contexts[0, 0, api_position, api_slot] == 0.0
    assert graph_windows.contexts[0, 0, edge_position, edge_slot] == 1.0
    assert not np.any(
        graph_windows.observation_mask[
            graph_windows.entity_ids.index("queue_hosted_on_redis")
        ]
    )


def test_declared_graph_compiler_rejects_an_unowned_feature() -> None:
    windows = _contextual_windows(
        metric_feature_names=(
            "request_latency_ms",
            "error_rate",
            "queue_depth",
            "worker_completion_ratio",
            "worker_heartbeat_age_s",
            "db_write_completion_ratio",
            "mystery_state",
        )
    )

    with pytest.raises(ValueError, match="unbound telemetry features"):
        compile_graph_state_windows(windows, quantis_checkout_graph())


def _contextual_windows(
    metric_feature_names: tuple[str, ...] = (
        "request_latency_ms",
        "error_rate",
        "queue_depth",
        "worker_completion_ratio",
        "worker_heartbeat_age_s",
        "db_write_completion_ratio",
    ),
) -> ContextualMultimodalModelWindows:
    sample_count = 2
    lookback = 2
    horizon_count = 2
    block_size = 1
    log_feature_names = (
        "checkout_completion_ratio",
        "checkout_backlog_delta_ratio",
        "checkout_rejection_rate",
        "queue_pressure_transition_rate",
        "queue_high_transition_rate",
        "postgresql_latency_pressure_ratio",
        "postgresql_slow_or_error_ratio",
        "worker_activation_rate",
        "redis_latency_pressure_rate",
        "redis_slow_or_error_rate",
        "checkout_queue_wait_pressure_ratio",
        "checkout_queue_wait_slow_ratio",
    )
    metric_values = np.arange(
        sample_count * lookback * len(metric_feature_names),
        dtype=np.float64,
    ).reshape(sample_count, lookback, len(metric_feature_names))
    log_values = np.arange(
        sample_count * lookback * len(log_feature_names),
        dtype=np.float64,
    ).reshape(sample_count, lookback, len(log_feature_names))
    return ContextualMultimodalModelWindows(
        metric_contexts=metric_values,
        log_contexts=log_values,
        metric_target_blocks=np.arange(
            sample_count
            * horizon_count
            * block_size
            * len(metric_feature_names),
            dtype=np.float64,
        ).reshape(
            sample_count,
            horizon_count,
            block_size,
            len(metric_feature_names),
        ),
        log_target_blocks=np.arange(
            sample_count
            * horizon_count
            * block_size
            * len(log_feature_names),
            dtype=np.float64,
        ).reshape(
            sample_count,
            horizon_count,
            block_size,
            len(log_feature_names),
        ),
        target_controls=np.zeros(
            (sample_count, horizon_count, block_size, 2),
            dtype=np.float64,
        ),
        point_indices=np.arange(sample_count, dtype=np.int64),
        metric_feature_names=metric_feature_names,
        log_feature_names=log_feature_names,
        control_feature_names=("request_demand", "worker_replicas"),
        horizons=(1, 3),
        target_block_size=block_size,
    )
