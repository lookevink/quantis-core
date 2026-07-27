"""Minimal API and worker processes for the instrumented fault lab."""

import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import NoReturn
from urllib.parse import parse_qs, urlparse

import psycopg
import redis


REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://quantis:quantis@postgres:5432/quantis",
)
QUEUE = "quantis:checkout:queue"
COUNTERS = "quantis:counters"
WORKER_HEARTBEAT = "quantis:worker:heartbeat"
WORKER_PAUSED = "quantis:fault:worker_paused"


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
        started = time.perf_counter_ns()
        status = 202
        try:
            delay_ms = float(parse_qs(parsed.query).get("delay_ms", ["0"])[0])
            if delay_ms > 0.0:
                time.sleep(delay_ms / 1000.0)
            payload = json.dumps(
                {"created_unix_nano": time.time_ns()},
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
    server = ThreadingHTTPServer(("0.0.0.0", 8080), CheckoutHandler)
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
    print("worker ready", flush=True)
    while True:
        if redis_client.get(WORKER_PAUSED) == "1":
            time.sleep(0.02)
            continue
        redis_client.set(WORKER_HEARTBEAT, repr(time.time()))
        payload = redis_client.lpop(QUEUE)
        if payload is None:
            time.sleep(0.005)
            continue
        started = time.perf_counter_ns()
        item = json.loads(payload)
        database.execute(
            "INSERT INTO completed_checkout (created_unix_nano) VALUES (%s)",
            (int(item["created_unix_nano"]),),
        )
        latency_us = max(1, (time.perf_counter_ns() - started) // 1_000)
        pipeline = redis_client.pipeline()
        pipeline.hincrby(COUNTERS, "worker_processed", 1)
        pipeline.hincrby(COUNTERS, "worker_db_latency_us", latency_us)
        pipeline.execute()


def main() -> NoReturn:
    mode = sys.argv[1] if len(sys.argv) > 1 else "api"
    if mode == "api":
        return run_api()
    if mode == "worker":
        return run_worker()
    raise SystemExit(f"unknown service mode {mode!r}")


if __name__ == "__main__":
    main()
