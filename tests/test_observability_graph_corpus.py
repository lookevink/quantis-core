import numpy as np
import pytest

from quantis_core.contextual_multimodal_corpus import (
    DEPENDENCY_LOG_FEATURE_NAMES,
    ContextualMultimodalModelWindows,
)
from quantis_core.graph_telemetry import (
    compile_graph_state_windows,
)
from quantis_core.observability_graph_corpus import (
    OBSERVABILITY_METRIC_FEATURE_NAMES,
    OBSERVABILITY_RAW_FEATURE_NAMES,
    ObservabilityGraphCorpus,
    OperationalStateTransformer,
    load_observability_graph_cache,
    quantis_checkout_observability_graph,
    write_observability_graph_cache,
)


def test_operational_state_transformer_separates_controls_and_owners() -> None:
    values = np.asarray(
        [
            [
                10.0,
                4.0,
                0.1,
                2.0,
                5.0,
                1.5,
                7.0,
                20.0,
                3.0,
                4.0,
                12.0,
                8.0,
                0.02,
                2.0,
                1.0,
                9.0,
                0.25,
                6.0,
                0.8,
                0.0,
                1.1,
                0.2,
                7.0,
                2.5,
                0.0,
                5.0,
            ]
        ],
        dtype=np.float64,
    )

    state = OperationalStateTransformer().transform(
        values,
        OBSERVABILITY_RAW_FEATURE_NAMES,
        request_demand=np.asarray([10.0]),
        worker_replicas=4,
    )

    assert state.control_feature_names == (
        "request_demand",
        "worker_replicas",
    )
    np.testing.assert_allclose(state.controls, [[10.0, 4.0]])
    feature = {
        name: state.values[0, position]
        for position, name in enumerate(state.feature_names)
    }
    assert feature["worker_completion_ratio"] == 0.8
    assert feature["worker_active_ratio"] == 0.5
    assert feature["worker_busy_ratio"] == 0.25
    assert feature["db_write_completion_ratio"] == 0.7
    assert "request_rate" not in feature


def test_operational_state_transformer_requires_complete_finite_state() -> None:
    values = np.ones(
        (1, len(OBSERVABILITY_RAW_FEATURE_NAMES) - 1),
        dtype=np.float64,
    )

    with pytest.raises(ValueError, match="missing"):
        OperationalStateTransformer().transform(
            values,
            OBSERVABILITY_RAW_FEATURE_NAMES[:-1],
            request_demand=np.asarray([1.0]),
            worker_replicas=1,
        )


def test_observability_graph_owns_every_semantic_feature() -> None:
    graph = quantis_checkout_observability_graph()
    bindings = graph.binding_map()

    assert {
        key
        for key in bindings
        if key.startswith("metric.")
    } == {
        f"metric.{name}"
        for name in OBSERVABILITY_METRIC_FEATURE_NAMES
    }
    assert (
        bindings["metric.api_inflight_peak"] == "api"
    )
    assert (
        bindings["metric.queue_oldest_age_ms"]
        == "checkout_queue"
    )
    assert (
        bindings["metric.redis_enqueue_latency_ms"]
        == "api_enqueues_queue"
    )
    assert (
        bindings["metric.postgresql_write_latency_ms"]
        == "worker_writes_postgresql"
    )


def test_observability_graph_cache_round_trips_and_rejects_tampering(
    tmp_path,
) -> None:
    graph = quantis_checkout_observability_graph()
    training = compile_graph_state_windows(
        _model_windows(3), graph
    )
    validation = compile_graph_state_windows(
        _model_windows(2), graph
    )
    corpus = ObservabilityGraphCorpus(
        training=training,
        validation=validation,
        training_case_ids=("train-a", "train-a", "train-b"),
        validation_case_ids=("validation-a", "validation-b"),
        provenance={
            "protocol_sha256": "a" * 64,
            "capture_sha256": ["b" * 64, "c" * 64],
        },
    )

    cache_directory = write_observability_graph_cache(
        corpus, tmp_path
    )
    restored = load_observability_graph_cache(cache_directory)

    assert cache_directory.name == restored.provenance.get(
        "cache_key", cache_directory.name
    )
    assert restored.training_case_ids == corpus.training_case_ids
    assert (
        restored.training.local_feature_keys
        == corpus.training.local_feature_keys
    )
    np.testing.assert_array_equal(
        restored.validation.target_blocks,
        corpus.validation.target_blocks,
    )

    tensor_path = cache_directory / "tensors.npz"
    tensor_path.write_bytes(tensor_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="archive hash changed"):
        load_observability_graph_cache(cache_directory)


def _model_windows(
    sample_count: int,
) -> ContextualMultimodalModelWindows:
    lookback = 2
    horizon_count = 3
    target_block_size = 2
    metric_count = len(OBSERVABILITY_METRIC_FEATURE_NAMES)
    log_count = len(DEPENDENCY_LOG_FEATURE_NAMES)
    return ContextualMultimodalModelWindows(
        metric_contexts=np.arange(
            sample_count * lookback * metric_count,
            dtype=np.float64,
        ).reshape(sample_count, lookback, metric_count),
        log_contexts=np.arange(
            sample_count * lookback * log_count,
            dtype=np.float64,
        ).reshape(sample_count, lookback, log_count),
        metric_target_blocks=np.arange(
            sample_count
            * horizon_count
            * target_block_size
            * metric_count,
            dtype=np.float64,
        ).reshape(
            sample_count,
            horizon_count,
            target_block_size,
            metric_count,
        ),
        log_target_blocks=np.arange(
            sample_count
            * horizon_count
            * target_block_size
            * log_count,
            dtype=np.float64,
        ).reshape(
            sample_count,
            horizon_count,
            target_block_size,
            log_count,
        ),
        target_controls=np.zeros(
            (
                sample_count,
                horizon_count,
                target_block_size,
                2,
            ),
            dtype=np.float64,
        ),
        point_indices=np.arange(sample_count, dtype=np.int64),
        metric_feature_names=OBSERVABILITY_METRIC_FEATURE_NAMES,
        log_feature_names=DEPENDENCY_LOG_FEATURE_NAMES,
        control_feature_names=(
            "request_demand",
            "worker_replicas",
        ),
        horizons=(1, 5, 10),
        target_block_size=target_block_size,
    )
