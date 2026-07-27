"""Small structured OTLP Logs emitter for the instrumented lab application."""

import json
import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Mapping, Optional, Protocol, Sequence, Tuple, Union


OTLP_LOGS_ENDPOINT = os.environ.get(
    "OTLP_LOGS_ENDPOINT",
    "http://collector:4318/v1/logs",
)
Attribute = Union[str, int, float, bool]
QUEUE_ENQUEUE_TRANSITION_SCRIPT = """
redis.call('RPUSH', KEYS[1], ARGV[1])
local depth = redis.call('LLEN', KEYS[1])
local state
if depth <= 2 then
  state = 'queue.backlog.low'
elseif depth <= 8 then
  state = 'queue.backlog.elevated'
else
  state = 'queue.backlog.high'
end
local previous = redis.call('GET', KEYS[2])
if previous == state then
  return {false, false, false}
end
redis.call('SET', KEYS[2], state)
local timestamp = redis.call('TIME')
return {state, timestamp[1], timestamp[2]}
""".strip()
QUEUE_DEQUEUE_TRANSITION_SCRIPT = """
local payload = redis.call('LPOP', KEYS[1])
if not payload then
  return {false, false, false, false}
end
local depth = redis.call('LLEN', KEYS[1])
local state
if depth <= 2 then
  state = 'queue.backlog.low'
elseif depth <= 8 then
  state = 'queue.backlog.elevated'
else
  state = 'queue.backlog.high'
end
local previous = redis.call('GET', KEYS[2])
if previous == state then
  return {payload, false, false, false}
end
redis.call('SET', KEYS[2], state)
local timestamp = redis.call('TIME')
return {payload, state, timestamp[1], timestamp[2]}
""".strip()


class RedisScriptClient(Protocol):
    """Minimal atomic-script interface used by backlog transitions."""

    def eval(
        self,
        script: str,
        numkeys: int,
        *keys_and_arguments: str,
    ) -> object:
        ...


@dataclass(frozen=True)
class ApplicationEvent:
    """One event from the lab's finite application-state vocabulary."""

    event_name: str
    severity_number: int
    severity_text: str
    body: str
    attributes: Mapping[str, Attribute]
    timestamp_unix_nano: Optional[int] = None


@dataclass(frozen=True)
class QueueTransition:
    """One transition observed atomically with a queue mutation."""

    event_name: str
    timestamp_unix_nano: int


def queue_backlog_event_name(queue_depth: int) -> str:
    """Map queue depth to one preregistered, bounded state."""

    if queue_depth < 0:
        raise ValueError("queue_depth cannot be negative")
    if queue_depth <= 2:
        return "queue.backlog.low"
    if queue_depth <= 8:
        return "queue.backlog.elevated"
    return "queue.backlog.high"


def enqueue_with_queue_transition(
    redis_client: RedisScriptClient,
    *,
    queue_key: str,
    state_key: str,
    payload: str,
) -> Optional[QueueTransition]:
    """Atomically enqueue and advance the bounded backlog state."""

    raw = redis_client.eval(
        QUEUE_ENQUEUE_TRANSITION_SCRIPT,
        2,
        queue_key,
        state_key,
        payload,
    )
    return _parse_queue_transition(raw)


def dequeue_with_queue_transition(
    redis_client: RedisScriptClient,
    *,
    queue_key: str,
    state_key: str,
) -> Tuple[Optional[str], Optional[QueueTransition]]:
    """Atomically dequeue and advance the bounded backlog state."""

    raw = redis_client.eval(
        QUEUE_DEQUEUE_TRANSITION_SCRIPT,
        2,
        queue_key,
        state_key,
    )
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise TypeError("queue dequeue script returned invalid result")
    payload = _optional_text(raw[0])
    if payload is None:
        return None, None
    return payload, _parse_queue_transition(raw[1:])


def database_latency_event_name(latency_us: int) -> str:
    """Map database-write latency to one preregistered bucket."""

    if latency_us < 0:
        raise ValueError("latency_us cannot be negative")
    if latency_us < 2_000:
        return "database.write.latency.fast"
    if latency_us < 10_000:
        return "database.write.latency.normal"
    return "database.write.latency.slow"


def emit_application_event(
    *,
    service_name: str,
    service_instance_id: str,
    event_name: str,
    severity_number: int,
    severity_text: str,
    body: str,
    experiment: Mapping[str, str],
    attributes: Mapping[str, Attribute],
) -> None:
    """Emit one bounded-vocabulary event without arbitrary payload fields."""

    emit_application_events(
        service_name=service_name,
        service_instance_id=service_instance_id,
        experiment=experiment,
        events=(
            ApplicationEvent(
                event_name=event_name,
                severity_number=severity_number,
                severity_text=severity_text,
                body=body,
                attributes=attributes,
            ),
        ),
    )


def emit_application_events(
    *,
    service_name: str,
    service_instance_id: str,
    experiment: Mapping[str, str],
    events: Sequence[ApplicationEvent],
) -> None:
    """Emit several related bounded events in one OTLP request."""

    if not events:
        raise ValueError("application event batch cannot be empty")
    resource_attributes: dict[str, Attribute] = {
        "service.name": service_name,
        "service.instance.id": service_instance_id,
        "quantis.experiment.case.id": experiment["case_id"],
        "quantis.experiment.fault.kind": experiment["fault_kind"],
        "quantis.experiment.manifest.sha256": (
            experiment["manifest_sha256"]
        ),
        "quantis.experiment.topology.id": experiment["topology_id"],
    }
    batch_timestamp = time.time_ns()
    payload = {
        "resourceLogs": [
            {
                "resource": {
                    "attributes": _attributes(resource_attributes)
                },
                "scopeLogs": [
                    {
                        "scope": {
                            "name": "quantis.application",
                            "version": "1.0.0",
                        },
                        "logRecords": [
                            {
                                "timeUnixNano": str(
                                    event.timestamp_unix_nano
                                    if event.timestamp_unix_nano
                                    is not None
                                    else batch_timestamp + event_index
                                ),
                                "observedTimeUnixNano": str(
                                    batch_timestamp + event_index
                                ),
                                "severityNumber": (
                                    event.severity_number
                                ),
                                "severityText": event.severity_text,
                                "body": {
                                    "stringValue": event.body
                                },
                                "attributes": _attributes(
                                    {
                                        "event.name": (
                                            event.event_name
                                        ),
                                        **event.attributes,
                                    }
                                ),
                            }
                            for event_index, event in enumerate(events)
                        ],
                    }
                ],
            }
        ]
    }
    request = urllib.request.Request(
        OTLP_LOGS_ENDPOINT,
        data=json.dumps(
            payload,
            separators=(",", ":"),
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
    attributes: Mapping[str, Attribute],
) -> list[dict[str, object]]:
    return [
        {
            "key": key,
            "value": _any_value(value),
        }
        for key, value in sorted(attributes.items())
    ]


def _any_value(value: Attribute) -> dict[str, object]:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    return {"stringValue": value}


def _parse_queue_transition(
    raw: object,
) -> Optional[QueueTransition]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise TypeError("queue transition script returned invalid result")
    event_name = _optional_text(raw[0])
    if event_name is None:
        return None
    seconds = _required_int(raw[1])
    microseconds = _required_int(raw[2])
    return QueueTransition(
        event_name=event_name,
        timestamp_unix_nano=(
            seconds * 1_000_000_000 + microseconds * 1_000
        ),
    )


def _optional_text(value: object) -> Optional[str]:
    if value is None or value is False:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    raise TypeError("queue script returned invalid text")


def _required_int(value: object) -> int:
    if isinstance(value, bytes):
        value = value.decode("ascii")
    if isinstance(value, str):
        return int(value)
    if isinstance(value, int):
        return value
    raise TypeError("queue script returned invalid timestamp")
