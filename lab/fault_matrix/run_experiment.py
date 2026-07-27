"""Drive one declared fault case and export only observed OTLP gauges."""

import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

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
EXPERIMENT_PATH = os.environ.get(
    "EXPERIMENT_PATH", "/experiments/worker-crash.json"
)
APPLICATION_IMAGE_ID = os.environ.get(
    "APPLICATION_IMAGE_ID", "unverified"
)
APPLICATION_BUILD_CONTEXT_SHA256 = os.environ.get(
    "APPLICATION_BUILD_CONTEXT_SHA256", "unverified"
)
WORKER_REPLICAS = int(os.environ.get("WORKER_REPLICAS", "1"))
QUEUE = "quantis:checkout:queue"
COUNTERS = "quantis:counters"
WORKER_HEARTBEAT = "quantis:worker:heartbeat"
WORKER_INSTANCES = "quantis:worker:instances"
WORKER_CRASH = "quantis:fault:worker_crash"
CACHE_OUTAGE = "quantis:fault:cache_outage"
DATABASE_ADVISORY_LOCK = 424242
FEATURE_NAMES = (
    "request_rate",
    "request_latency_ms",
    "error_rate",
    "queue_depth",
    "worker_rate",
    "worker_heartbeat_age_s",
    "db_write_rate",
)
NORMAL_TELEMETRY_KIND = "none"


def main() -> None:
    experiment = json.loads(Path(EXPERIMENT_PATH).read_text())
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
    fault_kind = str(experiment["fault_kind"])
    case_id = str(experiment["case_id"])
    topology_id = str(
        experiment.get("topology_id", "legacy-single-worker")
    )
    declared_worker_replicas = int(
        experiment.get("worker_replicas", 1)
    )
    if declared_worker_replicas != WORKER_REPLICAS:
        raise ValueError(
            "manifest worker_replicas does not match runner topology"
        )
    manifest_sha256 = hashlib.sha256(
        json.dumps(
            experiment, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()

    redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    database = psycopg.connect(DATABASE_URL, autocommit=True)
    observed_worker_replicas = _wait_until_ready(
        redis_client, database, declared_worker_replicas
    )
    redis_client.delete(
        QUEUE,
        COUNTERS,
        WORKER_CRASH,
        CACHE_OUTAGE,
        WORKER_HEARTBEAT,
    )
    database.execute("TRUNCATE completed_checkout RESTART IDENTITY")
    time.sleep(0.25)
    previous_counters = _counters(redis_client)
    previous_db_rows = _database_row_count(database)[0]
    lock_connection: Optional[psycopg.Connection] = None

    maximum_requests = requests_per_window + max(load_pattern)
    try:
        with ThreadPoolExecutor(max_workers=maximum_requests) as executor:
            for point_index in range(point_count):
                if (
                    point_index < fault_start
                    and _active_worker_count(redis_client)
                    != declared_worker_replicas
                ):
                    raise RuntimeError(
                        "observed worker replica count changed before fault"
                    )
                window_started = time.monotonic()
                request_count = (
                    requests_per_window
                    + load_pattern[point_index % len(load_pattern)]
                )
                if point_index == fault_start:
                    lock_connection = _start_fault(
                        fault_kind, redis_client
                    )
                if point_index == fault_stop:
                    _stop_fault(
                        fault_kind, redis_client, lock_connection
                    )
                    lock_connection = None

                futures = []
                for request_index in range(request_count):
                    delay = (
                        noise_delay_ms
                        if (
                            point_index == noise_start
                            and request_index == 0
                        )
                        else 0
                    )
                    expected_status = (
                        503
                        if (
                            fault_kind == "cache_outage"
                            and fault_start <= point_index < fault_stop
                        )
                        else 202
                    )
                    futures.append(
                        executor.submit(
                            _checkout,
                            delay,
                            expected_status,
                            point_index,
                            case_id,
                            fault_kind,
                            manifest_sha256,
                            topology_id,
                        )
                    )
                for future in futures:
                    future.result()

                remaining = period - (
                    time.monotonic() - window_started
                )
                if remaining > 0.0:
                    time.sleep(remaining)
                current_counters = _counters(redis_client)
                db_rows, _ = _database_row_count(database)
                values = _sample(
                    redis_client=redis_client,
                    period=period,
                    previous_counters=previous_counters,
                    current_counters=current_counters,
                    previous_db_rows=previous_db_rows,
                    current_db_rows=db_rows,
                )
                _emit(
                    point_index,
                    values,
                    case_id,
                    fault_kind,
                    manifest_sha256,
                    topology_id,
                    observed_worker_replicas,
                )
                previous_counters = current_counters
                previous_db_rows = db_rows
                print(
                    f"case={experiment['case_id']} point={point_index:03d} "
                    f"phase={_phase(point_index, experiment):<13} "
                    f"queue={values['queue_depth']:.0f} "
                    f"errors={values['error_rate']:.2f} "
                    f"worker_rate={values['worker_rate']:.1f}",
                    flush=True,
                )
        if fault_kind == NORMAL_TELEMETRY_KIND:
            _wait_for_normal_completion(redis_client)
        log_emit_errors = _counters(redis_client)[
            "application_log_emit_errors"
        ]
        if log_emit_errors:
            raise RuntimeError(
                f"application log emission failed {log_emit_errors} times"
            )
    finally:
        _stop_fault(fault_kind, redis_client, lock_connection)
    print(f"emitted {point_count} observed telemetry windows", flush=True)


def _start_fault(
    fault_kind: str, redis_client: redis.Redis
) -> Optional[psycopg.Connection]:
    if fault_kind == "worker_crash":
        redis_client.set(WORKER_CRASH, "1")
        return None
    if fault_kind == "cache_outage":
        redis_client.set(CACHE_OUTAGE, "1")
        return None
    if fault_kind == "database_lock":
        connection = psycopg.connect(DATABASE_URL, autocommit=True)
        connection.execute(
            "SELECT pg_advisory_lock(%s)",
            (DATABASE_ADVISORY_LOCK,),
        )
        return connection
    raise ValueError(f"unsupported fault kind: {fault_kind}")


def _stop_fault(
    fault_kind: str,
    redis_client: redis.Redis,
    lock_connection: Optional[psycopg.Connection],
) -> None:
    if fault_kind == NORMAL_TELEMETRY_KIND:
        return
    redis_client.delete(WORKER_CRASH, CACHE_OUTAGE)
    if fault_kind == "database_lock" and lock_connection is not None:
        lock_connection.execute(
            "SELECT pg_advisory_unlock(%s)",
            (DATABASE_ADVISORY_LOCK,),
        )
        lock_connection.close()


def _wait_until_ready(
    redis_client: redis.Redis,
    database: psycopg.Connection,
    expected_worker_replicas: int,
) -> int:
    for _ in range(120):
        try:
            redis_client.ping()
            database.execute("SELECT 1")
            with urllib.request.urlopen(
                f"{API_URL}/health", timeout=1
            ) as response:
                if response.status != 200:
                    continue
            with urllib.request.urlopen(
                COLLECTOR_HEALTH_URL, timeout=1
            ) as response:
                if response.status != 200:
                    continue
            observed_worker_replicas = _active_worker_count(redis_client)
            if observed_worker_replicas != expected_worker_replicas:
                time.sleep(0.25)
                continue
            return observed_worker_replicas
        except Exception:
            time.sleep(0.25)
    raise RuntimeError("fault-matrix services did not become ready")


def _active_worker_count(redis_client: redis.Redis) -> int:
    now = time.time()
    redis_client.zremrangebyscore(
        WORKER_INSTANCES, "-inf", now - 2.0
    )
    return int(redis_client.zcard(WORKER_INSTANCES))


def _wait_for_normal_completion(
    redis_client: redis.Redis,
) -> None:
    for _ in range(600):
        counters = _counters(redis_client)
        expected_completions = (
            counters["api_requests"] - counters["api_errors"]
        )
        if (
            counters["worker_processed"] >= expected_completions
            and redis_client.llen(QUEUE) == 0
        ):
            return
        time.sleep(0.05)
    raise RuntimeError(
        "normal run did not drain application work and logs"
    )


def _checkout(
    delay_ms: int,
    expected_status: int,
    point_index: int,
    case_id: str,
    fault_kind: str,
    manifest_sha256: str,
    topology_id: str,
) -> None:
    query = urllib.parse.urlencode(
        {
            "delay_ms": delay_ms,
            "window_index": point_index,
            "case_id": case_id,
            "fault_kind": fault_kind,
            "manifest_sha256": manifest_sha256,
            "topology_id": topology_id,
        }
    )
    request = urllib.request.Request(
        f"{API_URL}/checkout?{query}",
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            status = response.status
    except urllib.error.HTTPError as error:
        status = error.code
    if status != expected_status:
        raise RuntimeError(
            f"checkout returned HTTP {status}; expected {expected_status}"
        )


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
            "application_log_emit_errors",
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
) -> Dict[str, float]:
    requests = (
        current_counters["api_requests"]
        - previous_counters["api_requests"]
    )
    latency_us = (
        current_counters["api_latency_us"]
        - previous_counters["api_latency_us"]
    )
    errors = (
        current_counters["api_errors"]
        - previous_counters["api_errors"]
    )
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
    }


def _emit(
    point_index: int,
    values: Mapping[str, float],
    case_id: str,
    fault_kind: str,
    manifest_sha256: str,
    topology_id: str,
    worker_replicas: int,
) -> None:
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
                            },
                            {
                                "key": "quantis.experiment.case.id",
                                "value": {"stringValue": case_id},
                            },
                            {
                                "key": "quantis.experiment.fault.kind",
                                "value": {"stringValue": fault_kind},
                            },
                            {
                                "key": "quantis.experiment.manifest.sha256",
                                "value": {
                                    "stringValue": manifest_sha256
                                },
                            },
                            {
                                "key": "quantis.experiment.topology.id",
                                "value": {"stringValue": topology_id},
                            },
                            {
                                "key": "quantis.experiment.worker.replicas.observed",
                                "value": {"intValue": worker_replicas},
                            },
                        ]
                    },
                    "scopeMetrics": [
                        {
                            "scope": {
                                "name": "quantis.fault-matrix",
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
            raise RuntimeError(
                f"collector returned HTTP {response.status}"
            )


def _phase(point_index: int, experiment: Mapping[str, object]) -> str:
    if experiment["fault_kind"] == NORMAL_TELEMETRY_KIND:
        return "normal"
    noise_start, noise_stop = (
        int(value) for value in experiment["routine_noise_interval"]
    )
    fault_start, fault_stop = (
        int(value) for value in experiment["structural_interval"]
    )
    baseline_stop = int(experiment["baseline_interval"][1])
    if point_index < baseline_stop:
        return "baseline"
    if noise_start <= point_index < noise_stop:
        return "routine_noise"
    if fault_start <= point_index < fault_stop:
        return "structural"
    if point_index < noise_start:
        return "pre_noise"
    return "recovery"


if __name__ == "__main__":
    main()
