"""Lossless reader for the stable OTLP application-log fields Quantis uses."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional, Tuple, Union

from .otlp import AttributeValue, _any_value, _attributes


@dataclass(frozen=True)
class LogRecord:
    """One structured OTLP log record with its resource and scope identity."""

    time_unix_nano: int
    observed_time_unix_nano: Optional[int]
    severity_number: int
    severity_text: str
    body: Optional[AttributeValue]
    resource_attributes: Mapping[str, AttributeValue]
    record_attributes: Mapping[str, AttributeValue]
    scope_name: str
    scope_version: str
    trace_id: str
    span_id: str
    flags: int
    dropped_attributes_count: int


@dataclass(frozen=True)
class OtlpLogCapture:
    """Parsed records plus immutable source identity."""

    records: Tuple[LogRecord, ...]
    sha256: str
    source_path: str
    json_message_count: int


class OtlpLogCaptureError(ValueError):
    """An OTLP Logs capture cannot be interpreted without data loss."""


def read_otlp_log_capture(
    path: Union[str, Path],
) -> OtlpLogCapture:
    """Read newline-delimited OTLP JSON Logs export requests."""

    capture_path = Path(path)
    raw = capture_path.read_bytes()
    records: List[LogRecord] = []
    message_count = 0
    for line_number, raw_line in enumerate(raw.splitlines(), start=1):
        if not raw_line.strip():
            continue
        message_count += 1
        try:
            request = json.loads(raw_line)
            if not isinstance(request, dict):
                raise OtlpLogCaptureError(
                    f"invalid OTLP Logs JSON at line {line_number}: "
                    "top-level message must be an object"
                )
            records.extend(_request_records(request))
        except (
            AttributeError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            if isinstance(error, OtlpLogCaptureError):
                raise
            raise OtlpLogCaptureError(
                f"invalid OTLP Logs JSON at line {line_number}: {error}"
            ) from error
    if not records:
        raise OtlpLogCaptureError(
            "capture contains no OTLP JSON log records"
        )
    return OtlpLogCapture(
        records=tuple(sorted(records, key=_record_sort_key)),
        sha256=hashlib.sha256(raw).hexdigest(),
        source_path=str(capture_path),
        json_message_count=message_count,
    )


def _request_records(
    request: Mapping[str, Any],
) -> List[LogRecord]:
    records = []
    for resource_logs in request.get("resourceLogs", []):
        resource = resource_logs.get("resource", {})
        resource_attributes = _attributes(
            resource.get("attributes", [])
        )
        for scope_logs in resource_logs.get("scopeLogs", []):
            scope = scope_logs.get("scope", {})
            scope_name = str(scope.get("name", ""))
            scope_version = str(scope.get("version", ""))
            for record in scope_logs.get("logRecords", []):
                records.append(
                    _log_record(
                        record,
                        resource_attributes,
                        scope_name,
                        scope_version,
                    )
                )
    return records


def _log_record(
    record: Mapping[str, Any],
    resource_attributes: Mapping[str, AttributeValue],
    scope_name: str,
    scope_version: str,
) -> LogRecord:
    body = (
        _any_value(record["body"])
        if "body" in record
        else None
    )
    return LogRecord(
        time_unix_nano=int(record["timeUnixNano"]),
        observed_time_unix_nano=(
            int(record["observedTimeUnixNano"])
            if "observedTimeUnixNano" in record
            else None
        ),
        severity_number=int(record.get("severityNumber", 0)),
        severity_text=str(record.get("severityText", "")),
        body=body,
        resource_attributes=resource_attributes,
        record_attributes=_attributes(record.get("attributes", [])),
        scope_name=scope_name,
        scope_version=scope_version,
        trace_id=str(record.get("traceId", "")),
        span_id=str(record.get("spanId", "")),
        flags=int(record.get("flags", 0)),
        dropped_attributes_count=int(
            record.get("droppedAttributesCount", 0)
        ),
    )


def _record_sort_key(record: LogRecord) -> Tuple[Any, ...]:
    return (
        record.time_unix_nano,
        json.dumps(
            record.resource_attributes,
            sort_keys=True,
            default=str,
        ),
        json.dumps(
            record.record_attributes,
            sort_keys=True,
            default=str,
        ),
        record.severity_number,
        str(record.body),
    )
