"""Observability-rich adapter around the frozen fault-matrix application."""

import importlib.util
import os
import sys
import time
from contextlib import AbstractContextManager
from types import ModuleType
from typing import Any, Callable, Mapping, Optional, Type

import redis


COUNTERS = "quantis:counters"
API_INFLIGHT_CURRENT = "quantis:api:inflight:current"
API_INFLIGHT_PEAK = "quantis:api:inflight:peak"
WORKER_BUSY = "quantis:worker:busy"
LAST_ENQUEUE_UNIX_NANO = "quantis:event:last_enqueue"
LAST_DEQUEUE_UNIX_NANO = "quantis:event:last_dequeue"
LAST_POSTGRESQL_WRITE_UNIX_NANO = "quantis:event:last_postgresql_write"
INFLIGHT_SCRIPT = """
local current = redis.call('INCRBY', KEYS[1], ARGV[1])
if current < 0 then
  redis.call('SET', KEYS[1], 0)
  current = 0
end
local peak = tonumber(redis.call('GET', KEYS[2]) or '0')
if current > peak then
  redis.call('SET', KEYS[2], current)
end
return current
""".strip()


def _load_legacy_service() -> ModuleType:
    path = "/app/fault_matrix_service.py"
    spec = importlib.util.spec_from_file_location(
        "quantis_graph_jepa_legacy_service", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen fault-matrix service")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


legacy = _load_legacy_service()
_legacy_record_api_counters = legacy._record_api_counters
_legacy_enqueue = legacy.enqueue_with_queue_transition
_legacy_dequeue = legacy.dequeue_with_queue_transition
_legacy_queue_wait_event = legacy.checkout_queue_wait_event
_legacy_dependency_events = legacy.dependency_operation_events
_legacy_database = legacy._database
_legacy_do_post = legacy.CheckoutHandler.do_POST


def _counter_client() -> redis.Redis:
    return redis.Redis.from_url(
        legacy.REDIS_URL, decode_responses=True
    )


def _increment_inflight(delta: int) -> None:
    try:
        _counter_client().eval(
            INFLIGHT_SCRIPT,
            2,
            API_INFLIGHT_CURRENT,
            API_INFLIGHT_PEAK,
            delta,
        )
    except redis.RedisError:
        pass


def _observed_do_post(handler: Any) -> None:
    _increment_inflight(1)
    try:
        _legacy_do_post(handler)
    finally:
        _increment_inflight(-1)


def _record_api_counters(
    redis_client: redis.Redis,
    *,
    latency_us: int,
    failed: bool,
) -> None:
    _legacy_record_api_counters(
        redis_client,
        latency_us=latency_us,
        failed=failed,
    )
    try:
        redis_client.hincrby(
            COUNTERS, "api_busy_us", latency_us
        )
    except redis.RedisError:
        pass


def _observed_enqueue(
    redis_client: redis.Redis,
    *,
    queue_key: str,
    state_key: str,
    payload: str,
) -> Any:
    started = time.perf_counter_ns()
    failed = False
    try:
        result = _legacy_enqueue(
            redis_client,
            queue_key=queue_key,
            state_key=state_key,
            payload=payload,
        )
        redis_client.set(
            LAST_ENQUEUE_UNIX_NANO, str(time.time_ns())
        )
        return result
    except Exception:
        failed = True
        raise
    finally:
        _record_operation(
            redis_client,
            prefix="redis_enqueue",
            latency_us=max(
                1,
                (time.perf_counter_ns() - started) // 1_000,
            ),
            failed=failed,
        )


def _observed_dequeue(
    redis_client: redis.Redis,
    *,
    queue_key: str,
    state_key: str,
) -> Any:
    started = time.perf_counter_ns()
    failed = False
    try:
        result = _legacy_dequeue(
            redis_client,
            queue_key=queue_key,
            state_key=state_key,
        )
        if result[0] is not None:
            redis_client.set(
                LAST_DEQUEUE_UNIX_NANO, str(time.time_ns())
            )
        return result
    except Exception:
        failed = True
        raise
    finally:
        _record_operation(
            redis_client,
            prefix="redis_dequeue",
            latency_us=max(
                1,
                (time.perf_counter_ns() - started) // 1_000,
            ),
            failed=failed,
        )


def _observed_queue_wait_event(
    *,
    enqueued_unix_nano: int,
    dequeued_unix_nano: int,
    origin_window_index: int,
) -> Any:
    wait_us = max(
        0, (dequeued_unix_nano - enqueued_unix_nano) // 1_000
    )
    try:
        client = _counter_client()
        pipeline = client.pipeline()
        pipeline.hincrby(COUNTERS, "queue_residence_count", 1)
        pipeline.hincrby(
            COUNTERS, "queue_residence_us", wait_us
        )
        pipeline.execute()
    except redis.RedisError:
        pass
    return _legacy_queue_wait_event(
        enqueued_unix_nano=enqueued_unix_nano,
        dequeued_unix_nano=dequeued_unix_nano,
        origin_window_index=origin_window_index,
    )


def _observed_dependency_events(
    *,
    dependency_name: str,
    latency_us: int,
    origin_window_index: int,
    failed: bool = False,
    pool_wait_us: Optional[int] = None,
    timestamp_unix_nano: Optional[int] = None,
) -> Any:
    if dependency_name == "postgresql":
        client = _counter_client()
        _record_operation(
            client,
            prefix="postgresql_write",
            latency_us=latency_us,
            failed=failed,
        )
        if not failed:
            try:
                client.set(
                    LAST_POSTGRESQL_WRITE_UNIX_NANO,
                    str(
                        timestamp_unix_nano
                        if timestamp_unix_nano is not None
                        else time.time_ns()
                    ),
                )
            except redis.RedisError:
                pass
    return _legacy_dependency_events(
        dependency_name=dependency_name,
        latency_us=latency_us,
        origin_window_index=origin_window_index,
        failed=failed,
        pool_wait_us=pool_wait_us,
        timestamp_unix_nano=timestamp_unix_nano,
    )


def _record_operation(
    redis_client: redis.Redis,
    *,
    prefix: str,
    latency_us: int,
    failed: bool,
) -> None:
    try:
        pipeline = redis_client.pipeline()
        pipeline.hincrby(COUNTERS, f"{prefix}_count", 1)
        pipeline.hincrby(
            COUNTERS, f"{prefix}_latency_us", latency_us
        )
        if failed:
            pipeline.hincrby(
                COUNTERS, f"{prefix}_errors", 1
            )
        pipeline.execute()
    except redis.RedisError:
        pass


class _ObservedTransaction(AbstractContextManager[Any]):
    def __init__(
        self,
        transaction: AbstractContextManager[Any],
        worker_id: str,
    ) -> None:
        self._transaction = transaction
        self._worker_id = worker_id

    def __enter__(self) -> Any:
        try:
            _counter_client().hset(
                WORKER_BUSY,
                self._worker_id,
                str(time.time_ns()),
            )
        except redis.RedisError:
            pass
        return self._transaction.__enter__()

    def __exit__(
        self,
        exception_type: Optional[Type[BaseException]],
        exception: Optional[BaseException],
        traceback: Any,
    ) -> Optional[bool]:
        try:
            return self._transaction.__exit__(
                exception_type, exception, traceback
            )
        finally:
            try:
                _counter_client().hdel(
                    WORKER_BUSY, self._worker_id
                )
            except redis.RedisError:
                pass


class _ObservedConnection:
    def __init__(self, connection: Any) -> None:
        self._connection = connection
        self._worker_id = os.environ.get(
            "HOSTNAME", f"pid-{os.getpid()}"
        )

    def transaction(self) -> _ObservedTransaction:
        return _ObservedTransaction(
            self._connection.transaction(),
            self._worker_id,
        )

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        return self._connection.execute(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._connection, name)


def _observed_database() -> _ObservedConnection:
    return _ObservedConnection(_legacy_database())


setattr(legacy.CheckoutHandler, "do_POST", _observed_do_post)
setattr(legacy, "_record_api_counters", _record_api_counters)
setattr(
    legacy,
    "enqueue_with_queue_transition",
    _observed_enqueue,
)
setattr(
    legacy,
    "dequeue_with_queue_transition",
    _observed_dequeue,
)
setattr(
    legacy,
    "checkout_queue_wait_event",
    _observed_queue_wait_event,
)
setattr(
    legacy,
    "dependency_operation_events",
    _observed_dependency_events,
)
setattr(legacy, "_database", _observed_database)


if __name__ == "__main__":
    legacy.main()
