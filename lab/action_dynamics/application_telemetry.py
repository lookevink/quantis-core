"""Raw OTLP JSON telemetry for the versioned action-dynamics lab."""

import json
import re
import secrets
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence, Union


Attribute = Union[str, int, float, bool]
JsonPoster = Callable[[str, Mapping[str, Any]], None]
_TRACEPARENT = re.compile(
    r"^00-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$"
)
_FORBIDDEN_OBSERVATION_ATTRIBUTE_PARTS = (
    "action.",
    "fault.kind",
    "matched_pair",
)


@dataclass(frozen=True)
class ObservationIdentity:
    """Opaque capture identity allowed on observation telemetry."""

    case_id: str
    manifest_sha256: str
    topology_id: str

    def __post_init__(self) -> None:
        if not self.case_id or not self.topology_id:
            raise ValueError("observation identity fields cannot be empty")
        if (
            len(self.manifest_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.manifest_sha256
            )
        ):
            raise ValueError(
                "observation manifest identity must be a SHA-256 digest"
            )

    def attributes(self) -> Mapping[str, Attribute]:
        """Return the only experiment identity allowed on observations."""

        return {
            "quantis.experiment.case.id": self.case_id,
            "quantis.experiment.manifest.sha256": (
                self.manifest_sha256
            ),
            "quantis.experiment.topology.id": self.topology_id,
        }


@dataclass(frozen=True)
class TraceContext:
    """A W3C trace context represented in OTLP hexadecimal form."""

    trace_id: str
    span_id: str
    trace_flags: str = "01"

    def __post_init__(self) -> None:
        candidate = (
            f"00-{self.trace_id}-{self.span_id}-{self.trace_flags}"
        )
        if _TRACEPARENT.fullmatch(candidate) is None:
            raise ValueError("trace context is not valid W3C traceparent")
        if (
            self.trace_id == "0" * 32
            or self.span_id == "0" * 16
        ):
            raise ValueError("trace and span ids cannot be all zero")

    @classmethod
    def root(cls) -> "TraceContext":
        """Create a sampled root context."""

        return cls(
            trace_id=secrets.token_hex(16),
            span_id=secrets.token_hex(8),
        )

    @classmethod
    def from_traceparent(cls, value: str) -> "TraceContext":
        """Parse the exact W3C version used by the lab."""

        match = _TRACEPARENT.fullmatch(value)
        if match is None:
            raise ValueError("invalid W3C traceparent")
        return cls(
            trace_id=match.group(1),
            span_id=match.group(2),
            trace_flags=match.group(3),
        )

    def child(
        self, *, span_id: Optional[str] = None
    ) -> "TraceContext":
        """Return a child in the same trace."""

        return TraceContext(
            trace_id=self.trace_id,
            span_id=(
                span_id if span_id is not None else secrets.token_hex(8)
            ),
            trace_flags=self.trace_flags,
        )

    def to_traceparent(self) -> str:
        """Serialize this context for propagation through Redis."""

        return (
            f"00-{self.trace_id}-{self.span_id}-{self.trace_flags}"
        )


@dataclass(frozen=True)
class ApplicationEvent:
    """One bounded application event linked to its current span."""

    event_name: str
    body: str
    timestamp_unix_nano: int
    attributes: Mapping[str, Attribute] = field(default_factory=dict)
    trace: Optional[TraceContext] = None
    severity_number: int = 9
    severity_text: str = "INFO"

    def __post_init__(self) -> None:
        if (
            not self.event_name
            or not self.body
            or self.timestamp_unix_nano < 0
        ):
            raise ValueError("application event is invalid")
        _validate_observation_attributes(self.attributes)


@dataclass(frozen=True)
class Span:
    """One graph-owned OTLP span."""

    name: str
    graph_entity_id: str
    context: TraceContext
    parent_span_id: str
    start_unix_nano: int
    end_unix_nano: int
    kind: int
    attributes: Mapping[str, Attribute] = field(default_factory=dict)
    status_code: int = 1

    def __post_init__(self) -> None:
        if (
            not self.name
            or not self.graph_entity_id
            or self.start_unix_nano < 0
            or self.end_unix_nano < self.start_unix_nano
            or self.kind not in {1, 2, 3, 4, 5}
            or self.status_code not in {0, 1, 2}
        ):
            raise ValueError("span is invalid")
        if self.parent_span_id and (
            len(self.parent_span_id) != 16
            or any(
                character not in "0123456789abcdef"
                for character in self.parent_span_id
            )
        ):
            raise ValueError("span parent id is invalid")
        _validate_observation_attributes(self.attributes)


class OtlpTelemetryClient:
    """Emit application logs and traces to the observation receiver."""

    def __init__(
        self,
        *,
        logs_endpoint: str,
        traces_endpoint: str,
        post_json: Optional[JsonPoster] = None,
    ) -> None:
        if not logs_endpoint or not traces_endpoint:
            raise ValueError("OTLP endpoints cannot be empty")
        self.logs_endpoint = logs_endpoint
        self.traces_endpoint = traces_endpoint
        self._post_json = (
            _post_json if post_json is None else post_json
        )

    def emit_logs(
        self,
        *,
        service_name: str,
        service_instance_id: str,
        identity: ObservationIdentity,
        events: Sequence[ApplicationEvent],
    ) -> None:
        """Emit a nonempty batch of observation-only log records."""

        if not events:
            raise ValueError("application log batch cannot be empty")
        resource = _resource(
            service_name, service_instance_id, identity
        )
        records = []
        for event in events:
            record: dict[str, Any] = {
                "timeUnixNano": str(event.timestamp_unix_nano),
                "observedTimeUnixNano": str(
                    event.timestamp_unix_nano
                ),
                "severityNumber": event.severity_number,
                "severityText": event.severity_text,
                "body": {"stringValue": event.body},
                "attributes": _attributes(
                    {
                        "event.name": event.event_name,
                        **event.attributes,
                    }
                ),
            }
            if event.trace is not None:
                record["traceId"] = event.trace.trace_id
                record["spanId"] = event.trace.span_id
                record["flags"] = int(
                    event.trace.trace_flags, 16
                )
            records.append(record)
        self._post_json(
            self.logs_endpoint,
            {
                "resourceLogs": [
                    {
                        "resource": resource,
                        "scopeLogs": [
                            {
                                "scope": {
                                    "name": "quantis.application",
                                    "version": "3.0.0",
                                },
                                "logRecords": records,
                            }
                        ],
                    }
                ]
            },
        )

    def emit_spans(
        self,
        *,
        service_name: str,
        service_instance_id: str,
        identity: ObservationIdentity,
        spans: Sequence[Span],
    ) -> None:
        """Emit a nonempty batch of graph-owned spans."""

        if not spans:
            raise ValueError("span batch cannot be empty")
        resource = _resource(
            service_name, service_instance_id, identity
        )
        self._post_json(
            self.traces_endpoint,
            {
                "resourceSpans": [
                    {
                        "resource": resource,
                        "scopeSpans": [
                            {
                                "scope": {
                                    "name": "quantis.application",
                                    "version": "3.0.0",
                                },
                                "spans": [
                                    {
                                        "traceId": span.context.trace_id,
                                        "spanId": span.context.span_id,
                                        **(
                                            {
                                                "parentSpanId": (
                                                    span.parent_span_id
                                                )
                                            }
                                            if span.parent_span_id
                                            else {}
                                        ),
                                        "name": span.name,
                                        "kind": span.kind,
                                        "startTimeUnixNano": str(
                                            span.start_unix_nano
                                        ),
                                        "endTimeUnixNano": str(
                                            span.end_unix_nano
                                        ),
                                        "attributes": _attributes(
                                            {
                                                "quantis.graph.entity.id": (
                                                    span.graph_entity_id
                                                ),
                                                **span.attributes,
                                            }
                                        ),
                                        "status": {
                                            "code": span.status_code
                                        },
                                        "flags": int(
                                            span.context.trace_flags, 16
                                        ),
                                    }
                                    for span in spans
                                ],
                            }
                        ],
                    }
                ]
            },
        )


def _resource(
    service_name: str,
    service_instance_id: str,
    identity: ObservationIdentity,
) -> Mapping[str, Any]:
    if not service_name or not service_instance_id:
        raise ValueError("service identity cannot be empty")
    return {
        "attributes": _attributes(
            {
                "service.name": service_name,
                "service.instance.id": service_instance_id,
                **identity.attributes(),
            }
        )
    }


def _validate_observation_attributes(
    attributes: Mapping[str, Attribute],
) -> None:
    for key in attributes:
        lowered = key.lower()
        if any(
            part in lowered
            for part in _FORBIDDEN_OBSERVATION_ATTRIBUTE_PARTS
        ):
            raise ValueError(
                f"observation attribute leaks intervention truth: {key}"
            )


def _attributes(
    attributes: Mapping[str, Attribute],
) -> list[Mapping[str, Any]]:
    return [
        {"key": key, "value": _any_value(value)}
        for key, value in sorted(attributes.items())
    ]


def _any_value(value: Attribute) -> Mapping[str, Any]:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    return {"stringValue": value}


def _post_json(endpoint: str, payload: Mapping[str, Any]) -> None:
    body = json.dumps(
        payload, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=3) as response:
        if response.status != 200:
            raise RuntimeError(
                f"collector returned HTTP {response.status}"
            )
