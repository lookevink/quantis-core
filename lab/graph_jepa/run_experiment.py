"""Observability-rich runner adapter for the frozen fault-matrix driver."""

import importlib.util
import json
import sys
import time
from types import ModuleType
from typing import Dict, Mapping, Optional

import redis

from operational_state import (
    CUMULATIVE_COUNTER_NAMES,
    RAW_FEATURE_NAMES,
    OperationalGaugeSnapshot,
    compile_operational_metric_window,
)


API_INFLIGHT_CURRENT = "quantis:api:inflight:current"
API_INFLIGHT_PEAK = "quantis:api:inflight:peak"
WORKER_BUSY = "quantis:worker:busy"
LAST_ENQUEUE_UNIX_NANO = "quantis:event:last_enqueue"
LAST_DEQUEUE_UNIX_NANO = "quantis:event:last_dequeue"
LAST_POSTGRESQL_WRITE_UNIX_NANO = "quantis:event:last_postgresql_write"


def _load_legacy_runner() -> ModuleType:
    path = "/app/fault_matrix_run_experiment.py"
    spec = importlib.util.spec_from_file_location(
        "quantis_graph_jepa_legacy_runner", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen fault-matrix runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


legacy = _load_legacy_runner()


def _counters(redis_client: redis.Redis) -> Dict[str, int]:
    raw = redis_client.hgetall(legacy.COUNTERS)
    result = {
        name: int(raw.get(name, 0))
        for name in CUMULATIVE_COUNTER_NAMES
    }
    worker_processing_us = int(
        raw.get("worker_db_latency_us", 0)
    )
    result["worker_processing_us"] = worker_processing_us
    result["worker_busy_us"] = worker_processing_us
    return result


def _event_age_ms(
    redis_client: redis.Redis,
    key: str,
    *,
    now_unix_nano: int,
    default_age_ms: float,
) -> float:
    timestamp = redis_client.get(key)
    if timestamp is None:
        return default_age_ms
    return max(
        0.0,
        (now_unix_nano - int(timestamp)) / 1_000_000.0,
    )


def _queue_oldest_age_ms(
    redis_client: redis.Redis,
    *,
    now_unix_nano: int,
) -> float:
    raw = redis_client.lindex(legacy.QUEUE, 0)
    if raw is None:
        return 0.0
    try:
        payload = json.loads(raw)
        enqueued_unix_nano = int(payload["enqueued_unix_nano"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return 0.0
    return max(
        0.0,
        (now_unix_nano - enqueued_unix_nano) / 1_000_000.0,
    )


def _busy_gauges(
    redis_client: redis.Redis,
    *,
    now_unix_nano: int,
) -> tuple[int, float]:
    raw = redis_client.hvals(WORKER_BUSY)
    ages_ms = [
        max(0.0, (now_unix_nano - int(value)) / 1_000_000.0)
        for value in raw
    ]
    return len(raw), max(ages_ms, default=0.0)


def _sample(
    redis_client: redis.Redis,
    period: float,
    previous_counters: Mapping[str, int],
    current_counters: Mapping[str, int],
    previous_db_rows: int,
    current_db_rows: int,
) -> Dict[str, float]:
    now_unix_nano = time.time_ns()
    current_inflight = int(
        redis_client.get(API_INFLIGHT_CURRENT) or 0
    )
    peak_inflight = int(
        redis_client.get(API_INFLIGHT_PEAK) or current_inflight
    )
    redis_client.set(API_INFLIGHT_PEAK, current_inflight)
    heartbeat = redis_client.get(legacy.WORKER_HEARTBEAT)
    heartbeat_age = (
        max(0.0, time.time() - float(heartbeat))
        if heartbeat is not None
        else period
    )
    busy_count, busy_age_max_ms = _busy_gauges(
        redis_client, now_unix_nano=now_unix_nano
    )
    default_event_age_ms = period * 1_000.0
    gauges = OperationalGaugeSnapshot(
        api_inflight_current=current_inflight,
        api_inflight_peak=peak_inflight,
        queue_depth=int(redis_client.llen(legacy.QUEUE)),
        queue_oldest_age_ms=_queue_oldest_age_ms(
            redis_client, now_unix_nano=now_unix_nano
        ),
        enqueue_event_age_ms=_event_age_ms(
            redis_client,
            LAST_ENQUEUE_UNIX_NANO,
            now_unix_nano=now_unix_nano,
            default_age_ms=default_event_age_ms,
        ),
        dequeue_event_age_ms=_event_age_ms(
            redis_client,
            LAST_DEQUEUE_UNIX_NANO,
            now_unix_nano=now_unix_nano,
            default_age_ms=default_event_age_ms,
        ),
        worker_heartbeat_age_s=heartbeat_age,
        worker_active_count=legacy._active_worker_count(redis_client),
        worker_busy_count=busy_count,
        worker_busy_age_max_ms=busy_age_max_ms,
        postgresql_write_event_age_ms=_event_age_ms(
            redis_client,
            LAST_POSTGRESQL_WRITE_UNIX_NANO,
            now_unix_nano=now_unix_nano,
            default_age_ms=default_event_age_ms,
        ),
        database_row_count=current_db_rows,
    )
    return compile_operational_metric_window(
        previous_counters,
        current_counters,
        previous_database_row_count=previous_db_rows,
        gauges=gauges,
        period_seconds=period,
        worker_replicas=legacy.WORKER_REPLICAS,
    )


setattr(legacy, "FEATURE_NAMES", RAW_FEATURE_NAMES)
setattr(legacy, "_counters", _counters)
setattr(legacy, "_sample", _sample)


if __name__ == "__main__":
    legacy.main()
