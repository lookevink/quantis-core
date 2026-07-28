import numpy as np

from quantis_core.contextual_multimodal_corpus import (
    ContextualMultimodalModelWindows,
)
from quantis_core.graph_observability import (
    evaluate_graph_observability,
)
from quantis_core.graph_telemetry import (
    compile_graph_state_windows,
    quantis_checkout_graph,
)


def test_graph_observability_gate_passes_predictable_held_out_state() -> None:
    training = _predictable_windows(60, seed=11)
    validation = _predictable_windows(30, seed=17)
    graph = quantis_checkout_graph()

    assessment = evaluate_graph_observability(
        compile_graph_state_windows(training, graph),
        compile_graph_state_windows(validation, graph),
        training_window_case_ids=tuple(
            f"pilot-f{index % 3 + 1:02d}-w1"
            for index in range(60)
        ),
        validation_window_case_ids=tuple(
            f"pilot-f{index % 3 + 4:02d}-w1"
            for index in range(30)
        ),
        ridge=1e-3,
    )

    assert assessment["status"] == "supported"
    assert assessment["decision"] == "train_graph_jepa"
    assert all(
        gate["passed"] for gate in assessment["gates"].values()
    )
    assert (
        assessment["representations"]["one_hop_graph_ridge"][
            "mean_validation_normalized_mse"
        ]
        < assessment["representations"]["persistence"][
            "mean_validation_normalized_mse"
        ]
    )
    assert set(assessment["validation_families"]) == {
        "f04",
        "f05",
        "f06",
    }


def _predictable_windows(
    sample_count: int,
    *,
    seed: int,
) -> ContextualMultimodalModelWindows:
    generator = np.random.default_rng(seed)
    metric_names = (
        "request_latency_ms",
        "error_rate",
        "queue_depth",
        "worker_completion_ratio",
        "worker_heartbeat_age_s",
        "db_write_completion_ratio",
    )
    log_names = (
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
    metric_contexts = generator.normal(
        size=(sample_count, 3, len(metric_names))
    )
    log_contexts = generator.normal(
        size=(sample_count, 3, len(log_names))
    )
    horizon_offsets = np.asarray((0.2, -0.3), dtype=np.float64)
    metric_targets = (
        2.0 * metric_contexts[:, -1, :][:, None, None, :]
        + horizon_offsets[None, :, None, None]
    )
    log_targets = (
        2.0 * log_contexts[:, -1, :][:, None, None, :]
        + horizon_offsets[None, :, None, None]
    )
    return ContextualMultimodalModelWindows(
        metric_contexts=metric_contexts,
        log_contexts=log_contexts,
        metric_target_blocks=metric_targets,
        log_target_blocks=log_targets,
        target_controls=np.zeros(
            (sample_count, 2, 1, 2), dtype=np.float64
        ),
        point_indices=np.arange(sample_count, dtype=np.int64),
        metric_feature_names=metric_names,
        log_feature_names=log_names,
        control_feature_names=("request_demand", "worker_replicas"),
        horizons=(1, 3),
        target_block_size=1,
    )
