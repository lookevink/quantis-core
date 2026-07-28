import numpy as np

from quantis_core.contextual_multimodal_corpus import (
    ContextualMultimodalModelWindows,
)
from quantis_core.graph_jepa import (
    GraphJepaTrainingConfig,
    LinearGraphJepaWorldModel,
    evaluate_linear_graph_jepa,
)
from quantis_core.graph_telemetry import (
    compile_graph_state_windows,
    quantis_checkout_graph,
)


def test_linear_graph_jepa_predicts_entity_tokens_and_roundtrips() -> None:
    graph = quantis_checkout_graph()
    training = compile_graph_state_windows(
        _predictable_windows(80, seed=23), graph
    )
    validation = compile_graph_state_windows(
        _predictable_windows(20, seed=29), graph
    )
    model = LinearGraphJepaWorldModel(
        GraphJepaTrainingConfig(
            latent_dimension=4,
            ridge=1e-3,
            context_scope="one_hop",
        )
    ).fit(training)

    prediction = model.predict(validation)
    restored_prediction = LinearGraphJepaWorldModel.from_dict(
        model.to_dict()
    ).predict(validation)

    assert prediction.predicted_tokens.shape == (20, 2, 9, 4)
    assert prediction.target_tokens.shape == (20, 2, 9, 4)
    assert prediction.decoded_target_blocks.shape == (
        20,
        2,
        1,
        9,
        4,
    )
    np.testing.assert_allclose(
        restored_prediction.predicted_tokens,
        prediction.predicted_tokens,
    )
    np.testing.assert_allclose(
        restored_prediction.decoded_target_blocks,
        prediction.decoded_target_blocks,
    )
    observed = np.broadcast_to(
        validation.observation_mask[None, None, None, :, :],
        prediction.decoded_target_blocks.shape,
    )
    assert float(
        np.mean(
            np.square(
                prediction.decoded_target_blocks[observed]
                - validation.target_blocks[observed]
            )
        )
    ) < 1e-6

    models = {
        scope: LinearGraphJepaWorldModel(
            GraphJepaTrainingConfig(
                latent_dimension=4,
                ridge=1e-3,
                context_scope=scope,
            )
        ).fit(training)
        for scope in (
            "entity_local",
            "one_hop",
            "all_entities",
        )
    }
    assessment = evaluate_linear_graph_jepa(
        models,
        training,
        validation,
        validation_window_case_ids=tuple(
            f"pilot-f{index % 2 + 4:02d}-w1"
            for index in range(20)
        ),
    )
    assert assessment["status"] == "supported"
    assert assessment["decision"] == "collect_observability_rich_corpus"
    assert assessment["compression"]["context_ratio"] > 0.0
    assert all(
        gate["passed"] for gate in assessment["gates"].values()
    )


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
    metric_targets[:, :, :, 0] = (
        2.0
        * metric_contexts[:, -1, 2][:, None, None]
        + horizon_offsets[None, :, None]
    )
    return ContextualMultimodalModelWindows(
        metric_contexts=metric_contexts,
        log_contexts=log_contexts,
        metric_target_blocks=metric_targets,
        log_target_blocks=(
            2.0 * log_contexts[:, -1, :][:, None, None, :]
            + horizon_offsets[None, :, None, None]
        ),
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
