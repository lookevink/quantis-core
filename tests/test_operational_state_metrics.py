import importlib.util
from pathlib import Path
from types import ModuleType

from quantis_core.observability_graph_corpus import (
    OBSERVABILITY_RAW_FEATURE_NAMES,
)


def test_operational_window_metrics_have_complete_worked_values() -> None:
    metrics = _operational_metrics()
    previous = {
        name: 0
        for name in metrics.CUMULATIVE_COUNTER_NAMES
    }
    current = dict(previous)
    current.update(
        {
            "api_requests": 10,
            "api_latency_us": 40_000,
            "api_errors": 1,
            "api_busy_us": 150_000,
            "worker_processed": 8,
            "worker_processing_us": 48_000,
            "worker_busy_us": 48_000,
            "redis_enqueue_count": 10,
            "redis_enqueue_latency_us": 8_000,
            "redis_dequeue_count": 12,
            "redis_dequeue_latency_us": 13_200,
            "redis_dequeue_errors": 2,
            "queue_residence_count": 8,
            "queue_residence_us": 96_000,
            "postgresql_write_count": 8,
            "postgresql_write_latency_us": 20_000,
        }
    )
    gauges = metrics.OperationalGaugeSnapshot(
        api_inflight_current=2,
        api_inflight_peak=5,
        queue_depth=7,
        queue_oldest_age_ms=20.0,
        enqueue_event_age_ms=3.0,
        dequeue_event_age_ms=4.0,
        worker_heartbeat_age_s=0.02,
        worker_active_count=2,
        worker_busy_count=1,
        worker_busy_age_max_ms=9.0,
        postgresql_write_event_age_ms=5.0,
        database_row_count=7,
    )

    values = metrics.compile_operational_metric_window(
        previous,
        current,
        previous_database_row_count=0,
        gauges=gauges,
        period_seconds=0.1,
        worker_replicas=4,
    )

    assert tuple(values) == OBSERVABILITY_RAW_FEATURE_NAMES
    assert values["request_rate"] == 100.0
    assert values["request_latency_ms"] == 4.0
    assert values["error_rate"] == 0.1
    assert values["api_concurrency_mean"] == 1.5
    assert values["worker_busy_fraction"] == 0.12
    assert values["worker_processing_latency_ms"] == 6.0
    assert values["redis_enqueue_latency_ms"] == 0.8
    assert values["redis_dequeue_error_rate"] == 2.0 / 12.0
    assert values["queue_residence_mean_ms"] == 12.0
    assert values["db_write_rate"] == 70.0
    assert values["postgresql_write_latency_ms"] == 2.5


def _operational_metrics() -> ModuleType:
    repository = Path(__file__).resolve().parents[1]
    path = (
        repository
        / "lab"
        / "graph_jepa"
        / "operational_state.py"
    )
    spec = importlib.util.spec_from_file_location(
        "quantis_test_operational_state", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
