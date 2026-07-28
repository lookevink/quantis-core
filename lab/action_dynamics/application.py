"""Instrumented checkout API and worker for action-conditioned captures."""

import hashlib
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping, NoReturn, Optional
from urllib.parse import parse_qs, urlparse

import psycopg  # type: ignore[import-not-found]
import redis  # type: ignore[import-not-found]

from application_telemetry import (
    ApplicationEvent,
    ObservationIdentity,
    OtlpTelemetryClient,
    Span,
    TraceContext,
)
from interventions import (
    API_REJECTION,
    DEQUEUE_DELAY_MS,
    ENQUEUE_DELAY_MS,
    PAUSED_WORKERS,
)


REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://quantis:quantis@postgres:5432/quantis",
)
OTLP_LOGS_ENDPOINT = os.environ.get(
    "OTLP_LOGS_ENDPOINT", "http://collector:4318/v1/logs"
)
OTLP_TRACES_ENDPOINT = os.environ.get(
    "OTLP_TRACES_ENDPOINT", "http://collector:4318/v1/traces"
)
EXPERIMENT_PATH = os.environ.get(
    "EXPERIMENT_PATH", "/experiments/placeholder.json"
)
API_REQUEST_QUEUE_SIZE = int(
    os.environ.get("QUANTIS_API_REQUEST_QUEUE_SIZE", "128")
)
QUEUE = "quantis:checkout:queue"
COUNTERS = "quantis:counters"
WORKER_HEARTBEAT = "quantis:worker:heartbeat"
WORKER_INSTANCES = "quantis:worker:instances"
WORKER_BUSY = "quantis:worker:busy"
POSTGRESQL_WRITE_BUSY = "quantis:postgresql:write_busy"
API_INFLIGHT_CURRENT = "quantis:api:inflight:current"
API_INFLIGHT_PEAK = "quantis:api:inflight:peak"
LAST_ENQUEUE_UNIX_NANO = "quantis:event:last_enqueue"
LAST_DEQUEUE_UNIX_NANO = "quantis:event:last_dequeue"
LAST_POSTGRESQL_WRITE_UNIX_NANO = (
    "quantis:event:last_postgresql_write"
)
DATABASE_ADVISORY_LOCK = 424242
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


def _load_identity() -> ObservationIdentity:
    manifest_path = Path(EXPERIMENT_PATH)
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError("lab capture manifest must be an object")
    action_case = manifest.get("action_case")
    if not isinstance(action_case, dict):
        raise ValueError("lab capture manifest has no action case")
    return ObservationIdentity(
        case_id=str(action_case["case_id"]),
        manifest_sha256=hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        topology_id=str(action_case["topology_id"]),
    )


IDENTITY = _load_identity()
TELEMETRY = OtlpTelemetryClient(
    logs_endpoint=OTLP_LOGS_ENDPOINT,
    traces_endpoint=OTLP_TRACES_ENDPOINT,
)


def _redis() -> redis.Redis:
    client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    for _ in range(120):
        try:
            client.ping()
            return client
        except redis.RedisError:
            time.sleep(0.25)
    raise RuntimeError("Redis did not become ready")


def _database() -> psycopg.Connection[Any]:
    for _ in range(120):
        try:
            return psycopg.connect(DATABASE_URL, autocommit=True)
        except psycopg.OperationalError:
            time.sleep(0.25)
    raise RuntimeError("PostgreSQL did not become ready")


def _service_instance_id() -> str:
    return os.environ.get("HOSTNAME", f"pid-{os.getpid()}")


def _increment_inflight(
    redis_client: redis.Redis, delta: int
) -> None:
    redis_client.eval(
        INFLIGHT_SCRIPT,
        2,
        API_INFLIGHT_CURRENT,
        API_INFLIGHT_PEAK,
        delta,
    )


class QuantisThreadingHTTPServer(ThreadingHTTPServer):
    """HTTP server sized for the declared concurrent request schedule."""

    request_queue_size = API_REQUEST_QUEUE_SIZE


class CheckoutHandler(BaseHTTPRequestHandler):
    """Accept checkout work and propagate one W3C context into Redis."""

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
        try:
            window_index = int(query["window_index"][0])
            request_index = int(query["request_index"][0])
            request_token = query["request_token"][0]
        except (KeyError, IndexError, ValueError) as error:
            raise ValueError(
                "checkout requires opaque workload coordinates"
            ) from error
        root = TraceContext.root()
        request_started_unix_nano = time.time_ns()
        request_started = time.perf_counter_ns()
        status = 202
        enqueue_span: Optional[Span] = None
        _increment_inflight(self.redis_client, 1)
        try:
            if _should_reject(self.redis_client, request_token):
                status = 503
            else:
                enqueue_context = root.child()
                enqueue_started_unix_nano = time.time_ns()
                enqueue_started = time.perf_counter_ns()
                failed = False
                try:
                    delay_ms = float(
                        self.redis_client.get(ENQUEUE_DELAY_MS) or 0.0
                    )
                    if delay_ms > 0.0:
                        time.sleep(delay_ms / 1_000.0)
                    enqueued_unix_nano = time.time_ns()
                    payload = json.dumps(
                        {
                            "created_unix_nano": (
                                request_started_unix_nano
                            ),
                            "enqueued_unix_nano": (
                                enqueued_unix_nano
                            ),
                            "origin_window_index": window_index,
                            "request_index": request_index,
                            "traceparent": (
                                enqueue_context.to_traceparent()
                            ),
                        },
                        separators=(",", ":"),
                    )
                    self.redis_client.rpush(QUEUE, payload)
                    self.redis_client.set(
                        LAST_ENQUEUE_UNIX_NANO,
                        str(enqueued_unix_nano),
                    )
                except Exception:
                    failed = True
                    status = 500
                    raise
                finally:
                    enqueue_completed_unix_nano = time.time_ns()
                    latency_us = max(
                        1,
                        (
                            time.perf_counter_ns()
                            - enqueue_started
                        )
                        // 1_000,
                    )
                    _record_operation(
                        self.redis_client,
                        "redis_enqueue",
                        latency_us,
                        failed,
                    )
                    enqueue_span = Span(
                        name="redis.enqueue",
                        graph_entity_id="api_enqueues_queue",
                        context=enqueue_context,
                        parent_span_id=root.span_id,
                        start_unix_nano=(
                            enqueue_started_unix_nano
                        ),
                        end_unix_nano=(
                            enqueue_completed_unix_nano
                        ),
                        kind=3,
                        status_code=2 if failed else 1,
                        attributes={
                            "quantis.experiment.origin.window.index": (
                                window_index
                            )
                        },
                    )
        except Exception:
            status = 500
        finally:
            _increment_inflight(self.redis_client, -1)
        request_completed_unix_nano = time.time_ns()
        latency_us = max(
            1,
            (time.perf_counter_ns() - request_started) // 1_000,
        )
        _record_api(
            self.redis_client,
            latency_us=latency_us,
            failed=status != 202,
        )
        root_span = Span(
            name="api.admission",
            graph_entity_id="api",
            context=root,
            parent_span_id="",
            start_unix_nano=request_started_unix_nano,
            end_unix_nano=request_completed_unix_nano,
            kind=2,
            status_code=2 if status >= 500 else 1,
            attributes={
                "quantis.experiment.origin.window.index": (
                    window_index
                ),
                "http.response.status_code": status,
            },
        )
        event_name = (
            "checkout.accepted"
            if status == 202
            else "checkout.rejected"
        )
        try:
            TELEMETRY.emit_logs(
                service_name="quantis-action-api",
                service_instance_id=_service_instance_id(),
                identity=IDENTITY,
                events=(
                    ApplicationEvent(
                        event_name=event_name,
                        body=event_name.replace(".", " "),
                        timestamp_unix_nano=(
                            request_completed_unix_nano
                        ),
                        severity_number=(
                            9 if status < 500 else 17
                        ),
                        severity_text=(
                            "INFO" if status < 500 else "ERROR"
                        ),
                        attributes={
                            "quantis.experiment.origin.window.index": (
                                window_index
                            ),
                            "http.response.status_code": status,
                        },
                        trace=root,
                    ),
                ),
            )
            TELEMETRY.emit_spans(
                service_name="quantis-action-api",
                service_instance_id=_service_instance_id(),
                identity=IDENTITY,
                spans=tuple(
                    span
                    for span in (root_span, enqueue_span)
                    if span is not None
                ),
            )
        except Exception:
            self.redis_client.hincrby(
                COUNTERS, "telemetry_emit_errors", 1
            )
        self._json_response(
            status, {"accepted": status == 202}
        )

    def _json_response(
        self, status: int, payload: Mapping[str, Any]
    ) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode(
            "utf-8"
        )
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        del format, args


def run_api() -> NoReturn:
    server = QuantisThreadingHTTPServer(
        ("0.0.0.0", 8080), CheckoutHandler
    )
    print("action api ready", flush=True)
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
    worker_id = _service_instance_id()
    print("action worker ready", flush=True)
    while True:
        now = time.time()
        redis_client.set(WORKER_HEARTBEAT, repr(now))
        redis_client.zadd(WORKER_INSTANCES, {worker_id: now})
        redis_client.zremrangebyscore(
            WORKER_INSTANCES, "-inf", now - 2.0
        )
        if redis_client.sismember(PAUSED_WORKERS, worker_id):
            time.sleep(0.005)
            continue
        dequeue_started_unix_nano = time.time_ns()
        dequeue_started = time.perf_counter_ns()
        payload = redis_client.lpop(QUEUE)
        if payload is None:
            time.sleep(0.005)
            continue
        lpop_unix_nano = time.time_ns()
        delay_ms = float(
            redis_client.get(DEQUEUE_DELAY_MS) or 0.0
        )
        if delay_ms > 0.0:
            time.sleep(delay_ms / 1_000.0)
        dequeue_completed_unix_nano = time.time_ns()
        redis_client.set(
            LAST_DEQUEUE_UNIX_NANO,
            str(dequeue_completed_unix_nano),
        )
        dequeue_latency_us = max(
            1,
            (time.perf_counter_ns() - dequeue_started) // 1_000,
        )
        _record_operation(
            redis_client,
            "redis_dequeue",
            dequeue_latency_us,
            False,
        )
        item = json.loads(payload)
        origin_window_index = int(item["origin_window_index"])
        enqueued_unix_nano = int(item["enqueued_unix_nano"])
        upstream = TraceContext.from_traceparent(
            str(item["traceparent"])
        )
        residence = upstream.child()
        dequeue = residence.child()
        processing = dequeue.child()
        postgres = processing.child()
        residence_us = max(
            0, (lpop_unix_nano - enqueued_unix_nano) // 1_000
        )
        pipeline = redis_client.pipeline()
        pipeline.hincrby(COUNTERS, "queue_residence_count", 1)
        pipeline.hincrby(
            COUNTERS, "queue_residence_us", residence_us
        )
        pipeline.hset(WORKER_BUSY, worker_id, str(time.time_ns()))
        pipeline.execute()
        processing_started_unix_nano = time.time_ns()
        processing_started = time.perf_counter_ns()
        database_started_unix_nano = time.time_ns()
        database_started = time.perf_counter_ns()
        redis_client.hset(
            POSTGRESQL_WRITE_BUSY,
            worker_id,
            str(database_started_unix_nano),
        )
        failed = False
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
            failed = True
            raise
        finally:
            database_completed_unix_nano = time.time_ns()
            database_latency_us = max(
                1,
                (
                    time.perf_counter_ns() - database_started
                )
                // 1_000,
            )
            redis_client.hdel(POSTGRESQL_WRITE_BUSY, worker_id)
            _record_operation(
                redis_client,
                "postgresql_write",
                database_latency_us,
                failed,
            )
            if not failed:
                redis_client.set(
                    LAST_POSTGRESQL_WRITE_UNIX_NANO,
                    str(database_completed_unix_nano),
                )
        processing_completed_unix_nano = time.time_ns()
        processing_us = max(
            1,
            (time.perf_counter_ns() - processing_started)
            // 1_000,
        )
        pipeline = redis_client.pipeline()
        pipeline.hdel(WORKER_BUSY, worker_id)
        pipeline.hincrby(COUNTERS, "worker_processed", 1)
        pipeline.hincrby(
            COUNTERS, "worker_processing_us", processing_us
        )
        pipeline.hincrby(
            COUNTERS, "worker_busy_us", processing_us
        )
        pipeline.execute()
        spans = (
            Span(
                name="queue.residence",
                graph_entity_id="checkout_queue",
                context=residence,
                parent_span_id=upstream.span_id,
                start_unix_nano=enqueued_unix_nano,
                end_unix_nano=lpop_unix_nano,
                kind=5,
                attributes={
                    "quantis.experiment.origin.window.index": (
                        origin_window_index
                    )
                },
            ),
            Span(
                name="redis.dequeue",
                graph_entity_id="queue_dequeues_to_worker",
                context=dequeue,
                parent_span_id=residence.span_id,
                start_unix_nano=dequeue_started_unix_nano,
                end_unix_nano=dequeue_completed_unix_nano,
                kind=4,
                attributes={
                    "quantis.experiment.origin.window.index": (
                        origin_window_index
                    )
                },
            ),
            Span(
                name="worker.processing",
                graph_entity_id="worker_pool",
                context=processing,
                parent_span_id=dequeue.span_id,
                start_unix_nano=processing_started_unix_nano,
                end_unix_nano=processing_completed_unix_nano,
                kind=1,
                attributes={
                    "quantis.experiment.origin.window.index": (
                        origin_window_index
                    )
                },
            ),
            Span(
                name="postgresql.write",
                graph_entity_id="worker_writes_postgresql",
                context=postgres,
                parent_span_id=processing.span_id,
                start_unix_nano=database_started_unix_nano,
                end_unix_nano=database_completed_unix_nano,
                kind=3,
                status_code=2 if failed else 1,
                attributes={
                    "quantis.experiment.origin.window.index": (
                        origin_window_index
                    )
                },
            ),
        )
        try:
            TELEMETRY.emit_logs(
                service_name="quantis-action-worker",
                service_instance_id=worker_id,
                identity=IDENTITY,
                events=(
                    ApplicationEvent(
                        event_name="checkout.completed",
                        body="checkout completed",
                        timestamp_unix_nano=(
                            processing_completed_unix_nano
                        ),
                        attributes={
                            "quantis.experiment.origin.window.index": (
                                origin_window_index
                            )
                        },
                        trace=processing,
                    ),
                ),
            )
            TELEMETRY.emit_spans(
                service_name="quantis-action-worker",
                service_instance_id=worker_id,
                identity=IDENTITY,
                spans=spans,
            )
        except Exception:
            redis_client.hincrby(
                COUNTERS, "telemetry_emit_errors", 1
            )


def _should_reject(
    redis_client: redis.Redis, request_token: str
) -> bool:
    raw = redis_client.get(API_REJECTION)
    if raw is None:
        return False
    control = json.loads(raw)
    probability = float(control["probability"])
    seed = int(control["seed"])
    digest = hashlib.sha256(
        f"{seed}:{request_token}".encode("utf-8")
    ).digest()
    draw = int.from_bytes(digest[:8], "big") / float(2**64)
    return draw < probability


def _record_api(
    redis_client: redis.Redis,
    *,
    latency_us: int,
    failed: bool,
) -> None:
    pipeline = redis_client.pipeline()
    pipeline.hincrby(COUNTERS, "api_requests", 1)
    pipeline.hincrby(COUNTERS, "api_latency_us", latency_us)
    pipeline.hincrby(COUNTERS, "api_busy_us", latency_us)
    if failed:
        pipeline.hincrby(COUNTERS, "api_errors", 1)
    pipeline.execute()


def _record_operation(
    redis_client: redis.Redis,
    prefix: str,
    latency_us: int,
    failed: bool,
) -> None:
    pipeline = redis_client.pipeline()
    pipeline.hincrby(COUNTERS, f"{prefix}_count", 1)
    pipeline.hincrby(
        COUNTERS, f"{prefix}_latency_us", latency_us
    )
    if failed:
        pipeline.hincrby(COUNTERS, f"{prefix}_errors", 1)
    pipeline.execute()


def main() -> NoReturn:
    mode = sys.argv[1] if len(sys.argv) > 1 else "api"
    if mode == "api":
        run_api()
    if mode == "worker":
        run_worker()
    raise SystemExit(f"unknown service mode {mode!r}")


if __name__ == "__main__":
    main()
