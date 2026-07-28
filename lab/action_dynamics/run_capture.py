"""Drive one action-conditioned capture with transition-aligned commands."""

import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

from application_telemetry import ObservationIdentity
from interventions import (
    API_REJECTION,
    DEQUEUE_DELAY_MS,
    ENQUEUE_DELAY_MS,
    PAUSED_WORKERS,
    ReversibleInterventionController,
)


REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://quantis:quantis@postgres:5432/quantis",
)
API_URL = os.environ.get("API_URL", "http://api:8080")
OTLP_METRICS_ENDPOINT = os.environ.get(
    "OTLP_METRICS_ENDPOINT", "http://collector:4318/v1/metrics"
)
OTLP_ACTIONS_ENDPOINT = os.environ.get(
    "OTLP_ACTIONS_ENDPOINT", "http://collector:4319/v1/logs"
)
COLLECTOR_HEALTH_URL = os.environ.get(
    "COLLECTOR_HEALTH_URL", "http://collector:13133/"
)
EXPERIMENT_PATH = os.environ.get(
    "EXPERIMENT_PATH", "/experiments/placeholder.json"
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
WORKER_BUSY = "quantis:worker:busy"
POSTGRESQL_WRITE_BUSY = "quantis:postgresql:write_busy"
API_INFLIGHT_CURRENT = "quantis:api:inflight:current"
API_INFLIGHT_PEAK = "quantis:api:inflight:peak"
LAST_ENQUEUE_UNIX_NANO = "quantis:event:last_enqueue"
LAST_DEQUEUE_UNIX_NANO = "quantis:event:last_dequeue"
LAST_POSTGRESQL_WRITE_UNIX_NANO = (
    "quantis:event:last_postgresql_write"
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
    "telemetry_emit_errors",
)
FEATURE_NAMES = (
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
    "postgresql_write_busy_age_max_ms",
)
EVIDENCE_METRIC_NAMES = (
    "quantis.experiment.request_count",
    "quantis.experiment.error_count",
)


@dataclass(frozen=True)
class ScheduledActionCommand:
    """A command at transition t that affects observed state t+1."""

    logical_index: int
    phase: str
    affected_state_index: int
    action: Mapping[str, Any]


def scheduled_action_commands(
    action_case: Mapping[str, Any],
) -> Tuple[ScheduledActionCommand, ...]:
    """Return the exact transition-aligned command schedule."""

    point_count = action_case.get("point_count")
    actions = action_case.get("actions")
    if (
        isinstance(point_count, bool)
        or not isinstance(point_count, int)
        or point_count < 2
        or not isinstance(actions, list)
    ):
        raise ValueError("action case trajectory is invalid")
    commands: list[ScheduledActionCommand] = []
    for raw in actions:
        if not isinstance(raw, dict):
            raise ValueError("action case contains an invalid action")
        start = raw.get("start_index")
        stop = raw.get("stop_index")
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(stop, bool)
            or not isinstance(stop, int)
            or not 0 <= start < stop
            or stop + 1 >= point_count
        ):
            raise ValueError(
                "lab action must leave an observed recovery state"
            )
        commands.extend(
            (
                ScheduledActionCommand(
                    logical_index=start,
                    phase="start",
                    affected_state_index=start + 1,
                    action=raw,
                ),
                ScheduledActionCommand(
                    logical_index=stop,
                    phase="stop",
                    affected_state_index=stop + 1,
                    action=raw,
                ),
            )
        )
    return tuple(
        sorted(
            commands,
            key=lambda command: (
                command.logical_index,
                0 if command.phase == "stop" else 1,
            ),
        )
    )


def main() -> None:
    import psycopg  # type: ignore[import-not-found]
    import redis  # type: ignore[import-not-found]

    manifest_path = Path(EXPERIMENT_PATH)
    manifest = _load_manifest(manifest_path)
    action_case = _action_case(manifest)
    point_count = int(action_case["point_count"])
    sample_period_seconds = float(
        manifest["sample_period_seconds"]
    )
    request_schedule = tuple(
        int(value) for value in manifest["request_schedule"]
    )
    if (
        sample_period_seconds <= 0.0
        or len(request_schedule) < point_count - 1
        or min(request_schedule[: point_count - 1]) < 0
    ):
        raise ValueError("lab workload schedule is invalid")
    if int(action_case["worker_replicas"]) != WORKER_REPLICAS:
        raise ValueError(
            "manifest worker replicas differ from runtime topology"
        )
    manifest_sha256 = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    identity = ObservationIdentity(
        case_id=str(action_case["case_id"]),
        manifest_sha256=manifest_sha256,
        topology_id=str(action_case["topology_id"]),
    )
    workload_seed = int(action_case["workload_seed"])
    intervention_seed = int(action_case["intervention_seed"])
    redis_client = redis.Redis.from_url(
        REDIS_URL, decode_responses=True
    )
    database = psycopg.connect(DATABASE_URL, autocommit=True)
    _wait_until_ready(redis_client, database)
    _reset(redis_client, database)
    controller = ReversibleInterventionController(
        redis_client=redis_client,
        database_lock_factory=lambda: psycopg.connect(
            DATABASE_URL, autocommit=True
        ),
        action_endpoint=OTLP_ACTIONS_ENDPOINT,
        case_id=identity.case_id,
        manifest_sha256=identity.manifest_sha256,
        topology_id=identity.topology_id,
        intervention_seed=intervention_seed,
    )
    commands_by_transition: Dict[
        int, list[ScheduledActionCommand]
    ] = {}
    for command in scheduled_action_commands(action_case):
        commands_by_transition.setdefault(
            command.logical_index, []
        ).append(command)

    previous_counters = _counters(redis_client)
    previous_db_rows = _database_row_count(database)
    run_started_unix_nano = time.time_ns()
    controller.emit_run_boundary("started")
    try:
        initial = _sample(
            redis_client,
            database,
            previous_counters,
            previous_counters,
            previous_db_rows,
            previous_db_rows,
            sample_period_seconds,
        )
        _emit_metrics(
            0,
            initial,
            identity,
            run_started_unix_nano,
            time.time_ns(),
        )
        maximum_requests = max(request_schedule[: point_count - 1])
        with ThreadPoolExecutor(
            max_workers=maximum_requests
        ) as executor:
            for transition_index in range(point_count - 1):
                for command in commands_by_transition.get(
                    transition_index, ()
                ):
                    controller.command(
                        command.action,
                        command.phase,
                        command.logical_index,
                    )
                started = time.monotonic()
                request_count = request_schedule[transition_index]
                futures = [
                    executor.submit(
                        _checkout,
                        transition_index + 1,
                        request_index,
                        (
                            f"{workload_seed}:"
                            f"{transition_index}:"
                            f"{request_index}"
                        ),
                    )
                    for request_index in range(request_count)
                ]
                for future in futures:
                    future.result()
                remaining = sample_period_seconds - (
                    time.monotonic() - started
                )
                if remaining > 0.0:
                    time.sleep(remaining)
                current_counters = _counters(redis_client)
                current_db_rows = _database_row_count(database)
                values = _sample(
                    redis_client,
                    database,
                    previous_counters,
                    current_counters,
                    previous_db_rows,
                    current_db_rows,
                    sample_period_seconds,
                )
                _emit_metrics(
                    transition_index + 1,
                    values,
                    identity,
                    run_started_unix_nano,
                    time.time_ns(),
                )
                previous_counters = current_counters
                previous_db_rows = current_db_rows
                print(
                    json.dumps(
                        {
                            "case_id": identity.case_id,
                            "transition_index": transition_index,
                            "state_index": transition_index + 1,
                            "queue_depth": values["queue_depth"],
                            "error_rate": values["error_rate"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        errors = _counters(redis_client)["telemetry_emit_errors"]
        if errors:
            raise RuntimeError(
                f"application telemetry failed {errors} times"
            )
    finally:
        controller.close()
        controller.emit_run_boundary("closed")
        database.close()


def _load_manifest(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("lab capture manifest must be an object")
    required = {
        "action_case",
        "sample_period_seconds",
        "request_schedule",
    }
    if not required <= set(payload):
        raise ValueError("lab capture manifest is incomplete")
    return payload


def _action_case(
    manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    value = manifest["action_case"]
    if not isinstance(value, dict):
        raise ValueError("embedded action case must be an object")
    return value


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _wait_until_ready(
    redis_client: Any,
    database: Any,
) -> None:
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
            if _active_worker_count(redis_client) == WORKER_REPLICAS:
                return
        except Exception:
            pass
        time.sleep(0.25)
    raise RuntimeError("action-dynamics services did not become ready")


def _reset(
    redis_client: Any,
    database: Any,
) -> None:
    redis_client.delete(
        QUEUE,
        COUNTERS,
        WORKER_HEARTBEAT,
        WORKER_INSTANCES,
        WORKER_BUSY,
        POSTGRESQL_WRITE_BUSY,
        API_INFLIGHT_CURRENT,
        API_INFLIGHT_PEAK,
        LAST_ENQUEUE_UNIX_NANO,
        LAST_DEQUEUE_UNIX_NANO,
        LAST_POSTGRESQL_WRITE_UNIX_NANO,
        PAUSED_WORKERS,
        ENQUEUE_DELAY_MS,
        DEQUEUE_DELAY_MS,
        API_REJECTION,
    )
    database.execute(
        """
        CREATE TABLE IF NOT EXISTS completed_checkout (
            id BIGSERIAL PRIMARY KEY,
            created_unix_nano BIGINT NOT NULL,
            completed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )
    database.execute("TRUNCATE completed_checkout RESTART IDENTITY")
    for _ in range(120):
        if _active_worker_count(redis_client) == WORKER_REPLICAS:
            return
        time.sleep(0.05)
    raise RuntimeError("workers did not re-register after reset")


def _checkout(
    window_index: int, request_index: int, request_token: str
) -> None:
    query = urllib.parse.urlencode(
        {
            "window_index": window_index,
            "request_index": request_index,
            "request_token": request_token,
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
            if response.status not in {202, 503}:
                raise RuntimeError(
                    f"checkout returned HTTP {response.status}"
                )
    except urllib.error.HTTPError as error:
        if error.code != 503:
            raise


def _counters(redis_client: Any) -> Dict[str, int]:
    raw = redis_client.hgetall(COUNTERS)
    return {
        name: int(raw.get(name, 0))
        for name in CUMULATIVE_COUNTER_NAMES
    }


def _database_row_count(
    database: Any,
) -> int:
    row = database.execute(
        "SELECT COUNT(*) FROM completed_checkout"
    ).fetchone()
    if row is None:
        raise RuntimeError("database count query returned no row")
    return int(row[0])


def _active_worker_count(redis_client: Any) -> int:
    now = time.time()
    redis_client.zremrangebyscore(
        WORKER_INSTANCES, "-inf", now - 2.0
    )
    return int(redis_client.zcard(WORKER_INSTANCES))


def _sample(
    redis_client: Any,
    database: Any,
    previous: Mapping[str, int],
    current: Mapping[str, int],
    previous_db_rows: int,
    current_db_rows: int,
    period: float,
) -> Dict[str, float]:
    del database
    now = time.time_ns()

    def delta(name: str) -> int:
        value = current[name] - previous[name]
        if value < 0:
            raise RuntimeError(f"counter regressed: {name}")
        return value

    def mean_ms(sum_name: str, count: int) -> float:
        return delta(sum_name) / count / 1_000.0 if count else 0.0

    def event_age(key: str) -> float:
        value = redis_client.get(key)
        return (
            max(0.0, (now - int(value)) / 1_000_000.0)
            if value is not None
            else period * 1_000.0
        )

    def busy_age(key: str) -> Tuple[int, float]:
        values = redis_client.hvals(key)
        ages = [
            max(0.0, (now - int(value)) / 1_000_000.0)
            for value in values
        ]
        return len(values), max(ages, default=0.0)

    requests = delta("api_requests")
    workers = delta("worker_processed")
    enqueue_count = delta("redis_enqueue_count")
    dequeue_count = delta("redis_dequeue_count")
    residence_count = delta("queue_residence_count")
    postgres_count = delta("postgresql_write_count")
    heartbeat = redis_client.get(WORKER_HEARTBEAT)
    worker_busy_count, worker_busy_age = busy_age(WORKER_BUSY)
    _, postgresql_busy_age = busy_age(POSTGRESQL_WRITE_BUSY)
    oldest = redis_client.lindex(QUEUE, 0)
    oldest_age = 0.0
    if oldest is not None:
        try:
            oldest_payload = json.loads(oldest)
            oldest_age = max(
                0.0,
                (
                    now
                    - int(oldest_payload["enqueued_unix_nano"])
                )
                / 1_000_000.0,
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            oldest_age = 0.0
    values = {
        "request_rate": requests / period,
        "request_latency_ms": mean_ms(
            "api_latency_us", requests
        ),
        "error_rate": (
            delta("api_errors") / requests if requests else 0.0
        ),
        "api_inflight_current": float(
            redis_client.get(API_INFLIGHT_CURRENT) or 0
        ),
        "api_inflight_peak": float(
            redis_client.get(API_INFLIGHT_PEAK) or 0
        ),
        "api_concurrency_mean": (
            delta("api_busy_us") / (period * 1_000_000.0)
        ),
        "queue_depth": float(redis_client.llen(QUEUE)),
        "queue_oldest_age_ms": oldest_age,
        "enqueue_event_age_ms": event_age(
            LAST_ENQUEUE_UNIX_NANO
        ),
        "dequeue_event_age_ms": event_age(
            LAST_DEQUEUE_UNIX_NANO
        ),
        "queue_residence_mean_ms": mean_ms(
            "queue_residence_us", residence_count
        ),
        "worker_rate": workers / period,
        "worker_heartbeat_age_s": (
            max(0.0, time.time() - float(heartbeat))
            if heartbeat is not None
            else period
        ),
        "worker_active_count": float(
            _active_worker_count(redis_client)
        ),
        "worker_busy_count": float(worker_busy_count),
        "worker_busy_age_max_ms": worker_busy_age,
        "worker_busy_fraction": (
            delta("worker_busy_us")
            / (period * WORKER_REPLICAS * 1_000_000.0)
        ),
        "worker_processing_latency_ms": mean_ms(
            "worker_processing_us", workers
        ),
        "redis_enqueue_latency_ms": mean_ms(
            "redis_enqueue_latency_us", enqueue_count
        ),
        "redis_enqueue_error_rate": (
            delta("redis_enqueue_errors") / enqueue_count
            if enqueue_count
            else 0.0
        ),
        "redis_dequeue_latency_ms": mean_ms(
            "redis_dequeue_latency_us", dequeue_count
        ),
        "redis_dequeue_error_rate": (
            delta("redis_dequeue_errors") / dequeue_count
            if dequeue_count
            else 0.0
        ),
        "db_write_rate": (
            current_db_rows - previous_db_rows
        )
        / period,
        "postgresql_write_latency_ms": mean_ms(
            "postgresql_write_latency_us", postgres_count
        ),
        "postgresql_write_error_rate": (
            delta("postgresql_write_errors") / postgres_count
            if postgres_count
            else 0.0
        ),
        "postgresql_write_event_age_ms": event_age(
            LAST_POSTGRESQL_WRITE_UNIX_NANO
        ),
        "postgresql_write_busy_age_max_ms": postgresql_busy_age,
        "quantis.experiment.request_count": float(requests),
        "quantis.experiment.error_count": float(
            delta("api_errors")
        ),
    }
    if tuple(values) != (*FEATURE_NAMES, *EVIDENCE_METRIC_NAMES):
        raise AssertionError("runtime metric schema changed")
    redis_client.set(
        API_INFLIGHT_PEAK,
        redis_client.get(API_INFLIGHT_CURRENT) or 0,
    )
    return values


def _emit_metrics(
    state_index: int,
    values: Mapping[str, float],
    identity: ObservationIdentity,
    run_started_unix_nano: int,
    window_closed_unix_nano: int,
) -> None:
    timestamp = str((state_index + 1) * 1_000_000_000)
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
        for name in (*FEATURE_NAMES, *EVIDENCE_METRIC_NAMES)
    ]
    metrics.append(
        {
            "name": "quantis.experiment.window.closed_unix_nano",
            "unit": "ns",
            "gauge": {
                "dataPoints": [
                    {
                        "timeUnixNano": timestamp,
                        "asInt": str(window_closed_unix_nano),
                    }
                ]
            },
        }
    )
    payload = {
        "resourceMetrics": [
            {
                "resource": {
                    "attributes": _attributes(
                        {
                            "service.name": "quantis-action-lab",
                            **identity.attributes(),
                            "quantis.application.image.id": (
                                APPLICATION_IMAGE_ID
                            ),
                            "quantis.application.build_context.sha256": (
                                APPLICATION_BUILD_CONTEXT_SHA256
                            ),
                            "quantis.experiment.run.started_unix_nano": (
                                run_started_unix_nano
                            ),
                            "quantis.experiment.worker.replicas.observed": (
                                WORKER_REPLICAS
                            ),
                        }
                    )
                },
                "scopeMetrics": [
                    {
                        "scope": {
                            "name": "quantis.action-runtime",
                            "version": "1.0.0",
                        },
                        "metrics": metrics,
                    }
                ],
            }
        ]
    }
    request = urllib.request.Request(
        OTLP_METRICS_ENDPOINT,
        data=json.dumps(
            payload, separators=(",", ":"), allow_nan=False
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=3) as response:
        if response.status != 200:
            raise RuntimeError(
                f"collector returned HTTP {response.status}"
            )


def _attributes(
    values: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    for key, value in sorted(values.items()):
        if isinstance(value, int):
            encoded: Mapping[str, Any] = {"intValue": str(value)}
        else:
            encoded = {"stringValue": str(value)}
        result.append({"key": key, "value": encoded})
    return result


if __name__ == "__main__":
    main()
