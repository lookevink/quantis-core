"""API and worker processes exposing three externally controlled fault modes."""

import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Mapping, NoReturn, Optional, Sequence
from urllib.parse import parse_qs, urlparse

import psycopg
import redis

from application_logging import (
    ApplicationEvent,
    QueueTransition,
    checkout_queue_wait_event,
    dependency_operation_events,
    dequeue_with_queue_transition,
    emit_application_events,
    enqueue_with_queue_transition,
    experiment_identity_from_manifest,
)


REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://quantis:quantis@postgres:5432/quantis",
)
EXPERIMENT_PATH = os.environ.get(
    "EXPERIMENT_PATH",
    "/experiments/worker-crash.json",
)
QUEUE = "quantis:checkout:queue"
COUNTERS = "quantis:counters"
WORKER_HEARTBEAT = "quantis:worker:heartbeat"
WORKER_INSTANCES = "quantis:worker:instances"
WORKER_CRASH = "quantis:fault:worker_crash"
CACHE_OUTAGE = "quantis:fault:cache_outage"
QUEUE_BACKLOG_STATE = "quantis:queue:backlog:state"
DATABASE_ADVISORY_LOCK = 424242
APPLICATION_LOG_EMIT_ERRORS = "application_log_emit_errors"
WORKER_IDLE_TRANSITION_SECONDS = 0.02
API_REQUEST_QUEUE_SIZE = int(
    os.environ.get("QUANTIS_API_REQUEST_QUEUE_SIZE", "5")
)
if API_REQUEST_QUEUE_SIZE < 1:
    raise ValueError("QUANTIS_API_REQUEST_QUEUE_SIZE must be positive")


class QuantisThreadingHTTPServer(ThreadingHTTPServer):
    """Lab server with room for the largest declared request burst."""

    request_queue_size = API_REQUEST_QUEUE_SIZE


def _redis() -> redis.Redis:
    client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    for _ in range(120):
        try:
            client.ping()
            return client
        except redis.RedisError:
            time.sleep(0.25)
    raise RuntimeError("Redis did not become ready")


def _database() -> psycopg.Connection:
    for _ in range(120):
        try:
            return psycopg.connect(DATABASE_URL, autocommit=True)
        except psycopg.OperationalError:
            time.sleep(0.25)
    raise RuntimeError("PostgreSQL did not become ready")


class CheckoutHandler(BaseHTTPRequestHandler):
    redis_client = _redis()

    def do_GET(self) -> None:
        if urlparse(self.path).path != "/health":
            self.send_error(404)
            return
        self._json_response(200, {"status": "ok"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/checkout":
            self.send_error(404)
            return
        query = parse_qs(parsed.query)
        experiment = _experiment_identity(query)
        window_index = int(query["window_index"][0])
        started = time.perf_counter_ns()
        status = 202
        queue_transition: Optional[QueueTransition] = None
        dependency_events: list[ApplicationEvent] = []
        try:
            delay_ms = float(
                query.get("delay_ms", ["0"])[0]
            )
            if delay_ms > 0.0:
                time.sleep(delay_ms / 1000.0)
            cache_outage = _observe_dependency_call(
                lambda: self.redis_client.get(CACHE_OUTAGE),
                dependency_name="redis",
                origin_window_index=window_index,
                events=dependency_events,
            )
            if cache_outage == "1":
                time.sleep(0.03)
                status = 503
            else:
                payload = json.dumps(
                    {
                        "created_unix_nano": time.time_ns(),
                        "experiment": experiment,
                        "window_index": window_index,
                    },
                    separators=(",", ":"),
                )
                queue_transition = _observe_dependency_call(
                    lambda: enqueue_with_queue_transition(
                        self.redis_client,
                        queue_key=QUEUE,
                        state_key=QUEUE_BACKLOG_STATE,
                        payload=payload,
                    ),
                    dependency_name="redis",
                    origin_window_index=window_index,
                    events=dependency_events,
                )
        except Exception:
            status = 500
        latency_us = max(
            1, (time.perf_counter_ns() - started) // 1_000
        )
        event_names = (
            (
                "checkout.accepted"
                if status == 202
                else "checkout.rejected"
            )
        ,)
        _emit_checkout_events(
            redis_client=self.redis_client,
            service_name="quantis-fault-matrix-api",
            event_names=event_names,
            queue_transition=queue_transition,
            additional_events=dependency_events,
            status=status,
            experiment=experiment,
            window_index=window_index,
        )
        _record_api_counters(
            self.redis_client,
            latency_us=latency_us,
            failed=status != 202,
        )
        self._json_response(status, {"accepted": status == 202})

    def _json_response(self, status: int, payload: object) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


def run_api() -> NoReturn:
    server = QuantisThreadingHTTPServer(
        ("0.0.0.0", 8080),
        CheckoutHandler,
    )
    print("api ready", flush=True)
    server.serve_forever()
    raise AssertionError("unreachable")


def run_worker() -> NoReturn:
    redis_client = _redis()
    database = _database()
    database.execute(
        """
        CREATE TABLE IF NOT EXISTS completed_checkout (
            id BIGSERIAL PRIMARY KEY,
            created_unix_nano BIGINT NOT NULL,
            completed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )
    worker_id = os.environ.get("HOSTNAME", f"pid-{os.getpid()}")
    worker_busy = False
    last_activity = 0.0
    last_experiment = experiment_identity_from_manifest(
        EXPERIMENT_PATH
    )
    last_window_index = 0
    print("worker ready", flush=True)
    while True:
        redis_phase_started = time.perf_counter_ns()
        try:
            if redis_client.get(WORKER_CRASH) == "1":
                print(
                    "worker fault: exiting with status 17",
                    flush=True,
                )
                os._exit(17)
            redis_client.set(
                WORKER_HEARTBEAT,
                repr(time.time()),
            )
            now = time.time()
            redis_client.zadd(
                WORKER_INSTANCES,
                {worker_id: now},
            )
            redis_client.zremrangebyscore(
                WORKER_INSTANCES,
                "-inf",
                now - 2.0,
            )
            dequeue_started = time.perf_counter_ns()
            payload, queue_transition = dequeue_with_queue_transition(
                redis_client,
                queue_key=QUEUE,
                state_key=QUEUE_BACKLOG_STATE,
            )
        except Exception:
            failed_unix_nano = time.time_ns()
            _emit_checkout_events(
                redis_client=redis_client,
                service_name="quantis-fault-matrix-worker",
                event_names=(),
                additional_events=dependency_operation_events(
                    dependency_name="redis",
                    latency_us=max(
                        1,
                        (
                            time.perf_counter_ns()
                            - redis_phase_started
                        )
                        // 1_000,
                    ),
                    origin_window_index=last_window_index,
                    failed=True,
                    timestamp_unix_nano=failed_unix_nano,
                ),
                status=500,
                experiment=last_experiment,
                window_index=last_window_index,
            )
            raise
        dequeued_unix_nano = time.time_ns()
        redis_latency_us = max(
            1,
            (time.perf_counter_ns() - dequeue_started) // 1_000,
        )
        if payload is None:
            if (
                worker_busy
                and time.monotonic() - last_activity
                >= WORKER_IDLE_TRANSITION_SECONDS
            ):
                _emit_checkout_events(
                    redis_client=redis_client,
                    service_name="quantis-fault-matrix-worker",
                    event_names=("worker.state.idle",),
                    status=200,
                    experiment=last_experiment,
                    window_index=last_window_index,
                )
                worker_busy = False
            time.sleep(0.005)
            continue
        started = time.perf_counter_ns()
        item = json.loads(payload)
        experiment = {
            str(key): str(value)
            for key, value in item["experiment"].items()
        }
        window_index = int(item["window_index"])
        queue_wait_event = checkout_queue_wait_event(
            enqueued_unix_nano=int(item["enqueued_unix_nano"]),
            dequeued_unix_nano=dequeued_unix_nano,
            origin_window_index=window_index,
        )
        dependency_events = list(
            dependency_operation_events(
                dependency_name="redis",
                latency_us=redis_latency_us,
                origin_window_index=window_index,
                timestamp_unix_nano=dequeued_unix_nano,
            )
        )
        if queue_wait_event is not None:
            dependency_events.append(queue_wait_event)
        database_started = time.perf_counter_ns()
        try:
            with database.transaction():
                database.execute(
                    "SELECT pg_advisory_xact_lock(%s)",
                    (DATABASE_ADVISORY_LOCK,),
                )
                database.execute(
                    "INSERT INTO completed_checkout "
                    "(created_unix_nano) VALUES (%s)",
                    (int(item["created_unix_nano"]),),
                )
        except Exception:
            dependency_events.extend(
                dependency_operation_events(
                    dependency_name="postgresql",
                    latency_us=max(
                        1,
                        (
                            time.perf_counter_ns()
                            - database_started
                        )
                        // 1_000,
                    ),
                    origin_window_index=window_index,
                    failed=True,
                    timestamp_unix_nano=time.time_ns(),
                )
            )
            _emit_checkout_events(
                redis_client=redis_client,
                service_name="quantis-fault-matrix-worker",
                event_names=(),
                additional_events=dependency_events,
                status=500,
                experiment=experiment,
                window_index=window_index,
            )
            raise
        latency_us = max(1, (time.perf_counter_ns() - started) // 1_000)
        database_latency_us = max(
            1,
            (time.perf_counter_ns() - database_started) // 1_000,
        )
        database_completed_unix_nano = time.time_ns()
        dependency_events.extend(
            dependency_operation_events(
                dependency_name="postgresql",
                latency_us=database_latency_us,
                origin_window_index=window_index,
                timestamp_unix_nano=database_completed_unix_nano,
            )
        )
        event_names = [
            "checkout.completed",
        ]
        if not worker_busy:
            event_names.append("worker.state.busy")
        _emit_checkout_events(
            redis_client=redis_client,
            service_name="quantis-fault-matrix-worker",
            event_names=event_names,
            queue_transition=queue_transition,
            additional_events=dependency_events,
            status=200,
            experiment=experiment,
            window_index=window_index,
        )
        worker_busy = True
        last_activity = time.monotonic()
        last_experiment = experiment
        last_window_index = window_index
        pipeline = redis_client.pipeline()
        pipeline.hincrby(COUNTERS, "worker_processed", 1)
        pipeline.hincrby(COUNTERS, "worker_db_latency_us", latency_us)
        try:
            pipeline.execute()
        except redis.RedisError:
            pass


def _experiment_identity(
    query: Mapping[str, list[str]],
) -> dict[str, str]:
    keys = (
        "case_id",
        "fault_kind",
        "manifest_sha256",
        "topology_id",
    )
    try:
        return {key: query[key][0] for key in keys}
    except (KeyError, IndexError) as error:
        raise ValueError(
            "checkout requires complete experiment identity"
        ) from error


def _record_api_counters(
    redis_client: redis.Redis,
    *,
    latency_us: int,
    failed: bool,
) -> None:
    """Record lab metrics without blocking already-emitted OTLP logs."""

    try:
        pipeline = redis_client.pipeline()
        pipeline.hincrby(COUNTERS, "api_requests", 1)
        pipeline.hincrby(COUNTERS, "api_latency_us", latency_us)
        if failed:
            pipeline.hincrby(COUNTERS, "api_errors", 1)
        pipeline.execute()
    except redis.RedisError:
        pass


def _emit_checkout_events(
    *,
    redis_client: redis.Redis,
    service_name: str,
    event_names: Sequence[str],
    queue_transition: Optional[QueueTransition] = None,
    additional_events: Sequence[ApplicationEvent] = (),
    status: int,
    experiment: Mapping[str, str],
    window_index: int,
) -> None:
    try:
        events = [
            ApplicationEvent(
                event_name=event_name,
                severity_number=(
                    9 if status < 500 else 17
                ),
                severity_text=(
                    "INFO" if status < 500 else "ERROR"
                ),
                body=event_name.replace(".", " "),
                attributes={
                    "quantis.experiment.origin.window.index": (
                        window_index
                    ),
                    "http.response.status_code": status,
                },
            )
            for event_name in event_names
        ]
        events.extend(additional_events)
        if queue_transition is not None:
            events.append(
                ApplicationEvent(
                    event_name=queue_transition.event_name,
                    severity_number=9,
                    severity_text="INFO",
                    body=queue_transition.event_name.replace(".", " "),
                    attributes={
                        "quantis.experiment.origin.window.index": (
                            window_index
                        ),
                        "http.response.status_code": status,
                    },
                    timestamp_unix_nano=(
                        queue_transition.timestamp_unix_nano
                    ),
                )
            )
        emit_application_events(
            service_name=service_name,
            service_instance_id=os.environ.get(
                "HOSTNAME",
                f"pid-{os.getpid()}",
            ),
            experiment=experiment,
            events=tuple(events),
        )
    except Exception:
        try:
            redis_client.hincrby(
                COUNTERS,
                APPLICATION_LOG_EMIT_ERRORS,
                1,
            )
        except redis.RedisError:
            pass


def _observe_dependency_call(
    operation: Callable[[], Any],
    *,
    dependency_name: str,
    origin_window_index: int,
    events: list[ApplicationEvent],
) -> Any:
    started = time.perf_counter_ns()
    failed = False
    try:
        return operation()
    except Exception:
        failed = True
        raise
    finally:
        completed_unix_nano = time.time_ns()
        events.extend(
            dependency_operation_events(
                dependency_name=dependency_name,
                latency_us=max(
                    1,
                    (time.perf_counter_ns() - started) // 1_000,
                ),
                origin_window_index=origin_window_index,
                failed=failed,
                timestamp_unix_nano=completed_unix_nano,
            )
        )


def main() -> NoReturn:
    mode = sys.argv[1] if len(sys.argv) > 1 else "api"
    if mode == "api":
        return run_api()
    if mode == "worker":
        return run_worker()
    raise SystemExit(f"unknown service mode {mode!r}")


if __name__ == "__main__":
    main()
