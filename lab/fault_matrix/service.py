"""API and worker processes exposing three externally controlled fault modes."""

import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Mapping, NoReturn
from urllib.parse import parse_qs, urlparse

import psycopg
import redis

from application_logging import emit_application_event


REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://quantis:quantis@postgres:5432/quantis",
)
QUEUE = "quantis:checkout:queue"
COUNTERS = "quantis:counters"
WORKER_HEARTBEAT = "quantis:worker:heartbeat"
WORKER_INSTANCES = "quantis:worker:instances"
WORKER_CRASH = "quantis:fault:worker_crash"
CACHE_OUTAGE = "quantis:fault:cache_outage"
DATABASE_ADVISORY_LOCK = 424242
APPLICATION_LOG_EMIT_ERRORS = "application_log_emit_errors"
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
        try:
            delay_ms = float(
                query.get("delay_ms", ["0"])[0]
            )
            if delay_ms > 0.0:
                time.sleep(delay_ms / 1000.0)
            if self.redis_client.get(CACHE_OUTAGE) == "1":
                time.sleep(0.03)
                status = 503
                self.redis_client.hincrby(COUNTERS, "api_errors", 1)
            else:
                payload = json.dumps(
                    {
                        "created_unix_nano": time.time_ns(),
                        "experiment": experiment,
                        "window_index": window_index,
                    },
                    separators=(",", ":"),
                )
                self.redis_client.rpush(QUEUE, payload)
        except Exception:
            status = 500
            self.redis_client.hincrby(COUNTERS, "api_errors", 1)
        finally:
            latency_us = max(
                1, (time.perf_counter_ns() - started) // 1_000
            )
            pipeline = self.redis_client.pipeline()
            pipeline.hincrby(COUNTERS, "api_requests", 1)
            pipeline.hincrby(COUNTERS, "api_latency_us", latency_us)
            pipeline.execute()
        _emit_checkout_event(
            redis_client=self.redis_client,
            service_name="quantis-fault-matrix-api",
            event_name=(
                "checkout.accepted"
                if status == 202
                else "checkout.rejected"
            ),
            status=status,
            experiment=experiment,
            window_index=window_index,
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
    print("worker ready", flush=True)
    while True:
        if redis_client.get(WORKER_CRASH) == "1":
            print("worker fault: exiting with status 17", flush=True)
            os._exit(17)
        redis_client.set(WORKER_HEARTBEAT, repr(time.time()))
        now = time.time()
        redis_client.zadd(WORKER_INSTANCES, {worker_id: now})
        redis_client.zremrangebyscore(
            WORKER_INSTANCES, "-inf", now - 2.0
        )
        payload = redis_client.lpop(QUEUE)
        if payload is None:
            time.sleep(0.005)
            continue
        started = time.perf_counter_ns()
        item = json.loads(payload)
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
        latency_us = max(1, (time.perf_counter_ns() - started) // 1_000)
        _emit_checkout_event(
            redis_client=redis_client,
            service_name="quantis-fault-matrix-worker",
            event_name="checkout.completed",
            status=200,
            experiment={
                str(key): str(value)
                for key, value in item["experiment"].items()
            },
            window_index=int(item["window_index"]),
        )
        pipeline = redis_client.pipeline()
        pipeline.hincrby(COUNTERS, "worker_processed", 1)
        pipeline.hincrby(COUNTERS, "worker_db_latency_us", latency_us)
        pipeline.execute()


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


def _emit_checkout_event(
    *,
    redis_client: redis.Redis,
    service_name: str,
    event_name: str,
    status: int,
    experiment: Mapping[str, str],
    window_index: int,
) -> None:
    try:
        emit_application_event(
            service_name=service_name,
            service_instance_id=os.environ.get(
                "HOSTNAME",
                f"pid-{os.getpid()}",
            ),
            event_name=event_name,
            severity_number=9 if status < 500 else 17,
            severity_text="INFO" if status < 500 else "ERROR",
            body=event_name.replace(".", " "),
            experiment=experiment,
            attributes={
                "quantis.experiment.origin.window.index": (
                    window_index
                ),
                "http.response.status_code": status,
            },
        )
    except Exception:
        redis_client.hincrby(
            COUNTERS,
            APPLICATION_LOG_EMIT_ERRORS,
            1,
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
