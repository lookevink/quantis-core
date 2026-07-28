"""Pure window-level operational metric calculations for the graph lab."""

from dataclasses import dataclass
from typing import Dict, Mapping


RAW_FEATURE_NAMES = (
    "request_rate",
    "request_latency_ms",
    "error_rate",
    "api_inflight_current",
    "api_inflight_peak",
    "api_concurrency_mean",
    "queue_depth",
    "queue_oldest_age_ms",
    "enqueue_event_age_ms",
    "dequeue_event_age_ms",
    "queue_residence_mean_ms",
    "worker_rate",
    "worker_heartbeat_age_s",
    "worker_active_count",
    "worker_busy_count",
    "worker_busy_age_max_ms",
    "worker_busy_fraction",
    "worker_processing_latency_ms",
    "redis_enqueue_latency_ms",
    "redis_enqueue_error_rate",
    "redis_dequeue_latency_ms",
    "redis_dequeue_error_rate",
    "db_write_rate",
    "postgresql_write_latency_ms",
    "postgresql_write_error_rate",
    "postgresql_write_event_age_ms",
)
CUMULATIVE_COUNTER_NAMES = (
    "api_requests",
    "api_latency_us",
    "api_errors",
    "api_busy_us",
    "worker_processed",
    "worker_processing_us",
    "worker_busy_us",
    "redis_enqueue_count",
    "redis_enqueue_latency_us",
    "redis_enqueue_errors",
    "redis_dequeue_count",
    "redis_dequeue_latency_us",
    "redis_dequeue_errors",
    "queue_residence_count",
    "queue_residence_us",
    "postgresql_write_count",
    "postgresql_write_latency_us",
    "postgresql_write_errors",
    "application_log_emit_errors",
)


@dataclass(frozen=True)
class OperationalGaugeSnapshot:
    """Instantaneous state sampled at one window boundary."""

    api_inflight_current: int
    api_inflight_peak: int
    queue_depth: int
    queue_oldest_age_ms: float
    enqueue_event_age_ms: float
    dequeue_event_age_ms: float
    worker_heartbeat_age_s: float
    worker_active_count: int
    worker_busy_count: int
    worker_busy_age_max_ms: float
    postgresql_write_event_age_ms: float
    database_row_count: int

    def __post_init__(self) -> None:
        for value in self.__dict__.values():
            if isinstance(value, bool) or value < 0:
                raise ValueError(
                    "operational gauges must be nonnegative"
                )


def compile_operational_metric_window(
    previous_counters: Mapping[str, int],
    current_counters: Mapping[str, int],
    *,
    previous_database_row_count: int,
    gauges: OperationalGaugeSnapshot,
    period_seconds: float,
    worker_replicas: int,
) -> Dict[str, float]:
    """Return one complete raw metric row from independent observations."""

    if period_seconds <= 0.0:
        raise ValueError("operational metric period must be positive")
    if (
        isinstance(worker_replicas, bool)
        or worker_replicas < 1
    ):
        raise ValueError(
            "operational worker replicas must be positive"
        )
    if previous_database_row_count < 0:
        raise ValueError(
            "previous database row count must be nonnegative"
        )
    missing = set(CUMULATIVE_COUNTER_NAMES) - (
        set(previous_counters) & set(current_counters)
    )
    if missing:
        raise ValueError(
            f"operational counters are missing: {sorted(missing)}"
        )

    def delta(name: str) -> int:
        previous = previous_counters[name]
        current = current_counters[name]
        if (
            isinstance(previous, bool)
            or isinstance(current, bool)
            or previous < 0
            or current < previous
        ):
            raise ValueError(
                f"operational counter is not monotonic: {name}"
            )
        return current - previous

    requests = delta("api_requests")
    workers = delta("worker_processed")
    enqueue_count = delta("redis_enqueue_count")
    dequeue_count = delta("redis_dequeue_count")
    residence_count = delta("queue_residence_count")
    postgresql_count = delta("postgresql_write_count")
    database_rows = (
        gauges.database_row_count - previous_database_row_count
    )
    if database_rows < 0:
        raise ValueError("database row count is not monotonic")

    def mean_milliseconds(sum_name: str, count: int) -> float:
        return (
            delta(sum_name) / count / 1_000.0
            if count
            else 0.0
        )

    values = {
        "request_rate": requests / period_seconds,
        "request_latency_ms": mean_milliseconds(
            "api_latency_us", requests
        ),
        "error_rate": (
            delta("api_errors") / requests if requests else 0.0
        ),
        "api_inflight_current": float(
            gauges.api_inflight_current
        ),
        "api_inflight_peak": float(gauges.api_inflight_peak),
        "api_concurrency_mean": (
            delta("api_busy_us")
            / (period_seconds * 1_000_000.0)
        ),
        "queue_depth": float(gauges.queue_depth),
        "queue_oldest_age_ms": gauges.queue_oldest_age_ms,
        "enqueue_event_age_ms": gauges.enqueue_event_age_ms,
        "dequeue_event_age_ms": gauges.dequeue_event_age_ms,
        "queue_residence_mean_ms": mean_milliseconds(
            "queue_residence_us", residence_count
        ),
        "worker_rate": workers / period_seconds,
        "worker_heartbeat_age_s": (
            gauges.worker_heartbeat_age_s
        ),
        "worker_active_count": float(
            gauges.worker_active_count
        ),
        "worker_busy_count": float(gauges.worker_busy_count),
        "worker_busy_age_max_ms": (
            gauges.worker_busy_age_max_ms
        ),
        "worker_busy_fraction": (
            delta("worker_busy_us")
            / (
                period_seconds
                * worker_replicas
                * 1_000_000.0
            )
        ),
        "worker_processing_latency_ms": mean_milliseconds(
            "worker_processing_us", workers
        ),
        "redis_enqueue_latency_ms": mean_milliseconds(
            "redis_enqueue_latency_us", enqueue_count
        ),
        "redis_enqueue_error_rate": (
            delta("redis_enqueue_errors") / enqueue_count
            if enqueue_count
            else 0.0
        ),
        "redis_dequeue_latency_ms": mean_milliseconds(
            "redis_dequeue_latency_us", dequeue_count
        ),
        "redis_dequeue_error_rate": (
            delta("redis_dequeue_errors") / dequeue_count
            if dequeue_count
            else 0.0
        ),
        "db_write_rate": database_rows / period_seconds,
        "postgresql_write_latency_ms": mean_milliseconds(
            "postgresql_write_latency_us", postgresql_count
        ),
        "postgresql_write_error_rate": (
            delta("postgresql_write_errors") / postgresql_count
            if postgresql_count
            else 0.0
        ),
        "postgresql_write_event_age_ms": (
            gauges.postgresql_write_event_age_ms
        ),
    }
    if tuple(values) != RAW_FEATURE_NAMES:
        raise AssertionError(
            "operational metric output schema changed"
        )
    return values
