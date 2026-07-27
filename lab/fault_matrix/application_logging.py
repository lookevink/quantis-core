"""Small structured OTLP Logs emitter for the instrumented lab application."""

import json
import os
import time
import urllib.request
from typing import Mapping, Union


OTLP_LOGS_ENDPOINT = os.environ.get(
    "OTLP_LOGS_ENDPOINT",
    "http://collector:4318/v1/logs",
)
Attribute = Union[str, int, float, bool]


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
    record_attributes: dict[str, Attribute] = {
        "event.name": event_name,
        **attributes,
    }
    timestamp = str(time.time_ns())
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
                                "timeUnixNano": timestamp,
                                "observedTimeUnixNano": timestamp,
                                "severityNumber": severity_number,
                                "severityText": severity_text,
                                "body": {"stringValue": body},
                                "attributes": _attributes(
                                    record_attributes
                                ),
                            }
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
