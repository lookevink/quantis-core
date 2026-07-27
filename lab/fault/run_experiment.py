"""Drive real workload, inject a worker stall, and export observed OTLP gauges."""

import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Mapping, Tuple

import psycopg
import redis


REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://quantis:quantis@postgres:5432/quantis",
)
API_URL = os.environ.get("API_URL", "http://api:8080")
OTLP_ENDPOINT = os.environ.get(
    "OTLP_ENDPOINT", "http://collector:4318/v1/metrics"
)
COLLECTOR_HEALTH_URL = os.environ.get(
    "COLLECTOR_HEALTH_URL", "http://collector:13133/"
)
APPLICATION_IMAGE_ID = os.environ.get(
    "APPLICATION_IMAGE_ID", "unverified"
)
APPLICATION_BUILD_CONTEXT_SHA256 = os.environ.get(
    "APPLICATION_BUILD_CONTEXT_SHA256", "unverified"
)
QUEUE = "quantis:checkout:queue"
COUNTERS = "quantis:counters"
WORKER_HEARTBEAT = "quantis:worker:heartbeat"
WORKER_PAUSED = "quantis:fault:worker_paused"
FEATURE_NAMES = (
    "request_rate",
    "request_latency_ms",
    "error_rate",
    "queue_depth",
    "worker_rate",
    "worker_heartbeat_age_s",
    "db_write_rate",
    "db_probe_latency_ms",
)


def main() -> None:
    experiment = json.loads(
        Path("/app/experiment.json").read_text()
    )
    point_count = int(experiment["point_count"])
    period = float(experiment["sample_period_seconds"])
    requests_per_window = int(experiment["requests_per_window"])
    load_pattern = tuple(
        int(offset) for offset in experiment["load_pattern_offsets"]
    )
    noise_start = int(experiment["routine_noise_interval"][0])
    fault_start, fault_stop = (
        int(value) for value in experiment["structural_interval"]
    )
    noise_delay_ms = int(experiment["routine_noise_delay_ms"])

    redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    database = psycopg.connect(DATABASE_URL, autocommit=True)
    _wait_until_ready(redis_client, database)
    redis_client.delete(
        QUEUE, COUNTERS, WORKER_PAUSED, WORKER_HEARTBEAT
    )
    database.execute("TRUNCATE completed_checkout RESTART IDENTITY")
    time.sleep(0.25)
    previous_counters = _counters(redis_client)
    previous_db_rows = _database_row_count(database)[0]

    with ThreadPoolExecutor(max_workers=requests_per_window) as executor:
        for point_index in range(point_count):
            window_started = time.monotonic()
            request_count = (
                requests_per_window
                + load_pattern[point_index % len(load_pattern)]
            )
            if point_index == fault_start:
                redis_client.set(WORKER_PAUSED, "1")
            if point_index == fault_stop:
                redis_client.delete(WORKER_PAUSED)

            futures = []
            for request_index in range(request_count):
                delay = (
                    noise_delay_ms
                    if point_index == noise_start and request_index == 0
                    else 0
                )
                futures.append(
                    executor.submit(_checkout, delay)
                )
            for future in futures:
                future.result()

            remaining = period - (time.monotonic() - window_started)
            if remaining > 0.0:
                time.sleep(remaining)
            current_counters = _counters(redis_client)
            db_rows, db_probe_latency_ms = _database_row_count(database)
            values = _sample(
                redis_client=redis_client,
                period=period,
                previous_counters=previous_counters,
                current_counters=current_counters,
                previous_db_rows=previous_db_rows,
                current_db_rows=db_rows,
                db_probe_latency_ms=db_probe_latency_ms,
            )
            _emit(point_index, values)
            previous_counters = current_counters
            previous_db_rows = db_rows
            phase = _phase(point_index, experiment)
            print(
                f"point={point_index:03d} phase={phase:<13} "
                f"queue={values['queue_depth']:.0f} "
                f"worker_rate={values['worker_rate']:.1f}",
                flush=True,
            )
    redis_client.delete(WORKER_PAUSED)
    print(f"emitted {point_count} observed telemetry windows", flush=True)


def _wait_until_ready(
    redis_client: redis.Redis, database: psycopg.Connection
) -> None:
    for _ in range(120):
        try:
            redis_client.ping()
            database.execute("SELECT 1")
            with urllib.request.urlopen(f"{API_URL}/health", timeout=1) as response:
                if response.status != 200:
                    continue
            with urllib.request.urlopen(
                COLLECTOR_HEALTH_URL, timeout=1
            ) as response:
                if response.status != 200:
                    continue
            return
        except Exception:
            time.sleep(0.25)
    raise RuntimeError("fault-lab services did not become ready")


def _checkout(delay_ms: int) -> None:
    request = urllib.request.Request(
        f"{API_URL}/checkout?delay_ms={delay_ms}",
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=3) as response:
        if response.status != 202:
            raise RuntimeError(f"checkout returned HTTP {response.status}")


def _counters(redis_client: redis.Redis) -> Dict[str, int]:
    raw = redis_client.hgetall(COUNTERS)
    return {
        name: int(raw.get(name, 0))
        for name in (
            "api_requests",
            "api_latency_us",
            "api_errors",
            "worker_processed",
            "worker_db_latency_us",
        )
    }


def _database_row_count(
    database: psycopg.Connection,
) -> Tuple[int, float]:
    started = time.perf_counter_ns()
    row = database.execute(
        "SELECT COUNT(*) FROM completed_checkout"
    ).fetchone()
    latency_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    if row is None:
        raise RuntimeError("database count query returned no row")
    return int(row[0]), latency_ms


def _sample(
    redis_client: redis.Redis,
    period: float,
    previous_counters: Mapping[str, int],
    current_counters: Mapping[str, int],
    previous_db_rows: int,
    current_db_rows: int,
    db_probe_latency_ms: float,
) -> Dict[str, float]:
    requests = current_counters["api_requests"] - previous_counters["api_requests"]
    latency_us = (
        current_counters["api_latency_us"]
        - previous_counters["api_latency_us"]
    )
    errors = current_counters["api_errors"] - previous_counters["api_errors"]
    workers = (
        current_counters["worker_processed"]
        - previous_counters["worker_processed"]
    )
    heartbeat = redis_client.get(WORKER_HEARTBEAT)
    heartbeat_age = (
        max(0.0, time.time() - float(heartbeat))
        if heartbeat is not None
        else period
    )
    return {
        "request_rate": requests / period,
        "request_latency_ms": (
            latency_us / requests / 1_000.0 if requests else 0.0
        ),
        "error_rate": errors / requests if requests else 0.0,
        "queue_depth": float(redis_client.llen(QUEUE)),
        "worker_rate": workers / period,
        "worker_heartbeat_age_s": heartbeat_age,
        "db_write_rate": (current_db_rows - previous_db_rows) / period,
        "db_probe_latency_ms": db_probe_latency_ms,
    }


def _emit(point_index: int, values: Mapping[str, float]) -> None:
    timestamp = str((point_index + 1) * 1_000_000_000)
    metrics = [
        {
            "name": name,
            "unit": "1",
            "gauge": {
                "dataPoints": [
                    {
                        "timeUnixNano": timestamp,
                        "asDouble": float(values[name]),
                    }
                ]
            },
        }
        for name in FEATURE_NAMES
    ]
    body = json.dumps(
        {
            "resourceMetrics": [
                {
                    "resource": {
                        "attributes": [
                            {
                                "key": "service.name",
                                "value": {
                                    "stringValue": "quantis-fault-lab"
                                },
                            },
                            {
                                "key": "quantis.application.image.id",
                                "value": {
                                    "stringValue": APPLICATION_IMAGE_ID
                                },
                            },
                            {
                                "key": "quantis.application.build_context.sha256",
                                "value": {
                                    "stringValue": (
                                        APPLICATION_BUILD_CONTEXT_SHA256
                                    )
                                },
                            }
                        ]
                    },
                    "scopeMetrics": [
                        {
                            "scope": {
                                "name": "quantis.fault-lab",
                                "version": "1.0.0",
                            },
                            "metrics": metrics,
                        }
                    ],
                }
            ]
        },
        separators=(",", ":"),
    ).encode("utf-8")
    request = urllib.request.Request(
        OTLP_ENDPOINT,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=3) as response:
        if response.status != 200:
            raise RuntimeError(f"collector returned HTTP {response.status}")


def _phase(point_index: int, experiment: Mapping[str, object]) -> str:
    noise_start, noise_stop = (
        int(value) for value in experiment["routine_noise_interval"]
    )
    fault_start, fault_stop = (
        int(value) for value in experiment["structural_interval"]
    )
    training_stop = int(experiment["training_interval"][1])
    if point_index < training_stop:
        return "baseline"
    if noise_start <= point_index < noise_stop:
        return "routine_noise"
    if fault_start <= point_index < fault_stop:
        return "structural"
    if point_index < noise_start:
        return "validation"
    return "recovery"


if __name__ == "__main__":
    main()
