import hashlib
import json
from pathlib import Path

from quantis_core.otlp_logs import LogRecord, OtlpLogCapture


def normal_log_captures(runs):
    return {
        run.manifest.case_id: normal_log_capture(run)
        for run in runs
    }


def normal_log_capture(run) -> OtlpLogCapture:
    manifest = run.manifest
    manifest_sha256 = hashlib.sha256(
        json.dumps(
            manifest.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    resource = {
        "service.name": "quantis-fault-matrix",
        "quantis.experiment.case.id": manifest.case_id,
        "quantis.experiment.fault.kind": manifest.fault_kind,
        "quantis.experiment.manifest.sha256": manifest_sha256,
    }
    records = tuple(
        LogRecord(
            time_unix_nano=point_index * 10 + event_index,
            observed_time_unix_nano=None,
            severity_number=9,
            severity_text="INFO",
            body=event_name,
            resource_attributes=resource,
            record_attributes={
                "event.name": event_name,
                "quantis.experiment.window.index": point_index,
            },
            scope_name="quantis.application",
            scope_version="1.0.0",
            trace_id="",
            span_id="",
            flags=0,
            dropped_attributes_count=0,
        )
        for point_index in range(manifest.point_count)
        for event_index, event_name in enumerate(
            ("checkout.accepted", "checkout.completed")
        )
    )
    digest = hashlib.sha256(
        f"logs:{manifest.case_id}".encode("utf-8")
    ).hexdigest()
    return OtlpLogCapture(
        records=records,
        sha256=digest,
        source_path=f"memory://{manifest.case_id}",
        json_message_count=1,
    )


def v2_normal_log_captures(runs):
    captures = normal_log_captures(runs)
    enriched = {}
    event_names = (
        "queue.backlog.elevated",
        "queue.backlog.high",
        "worker.state.busy",
        "dependency.redis.latency.elevated",
        "dependency.redis.latency.slow",
        "dependency.redis.operation.error",
        "dependency.postgresql.latency.elevated",
        "dependency.postgresql.latency.slow",
        "dependency.postgresql.operation.error",
        "checkout.queue_wait.elevated",
        "checkout.queue_wait.slow",
    )
    for run in runs:
        capture = captures[run.manifest.case_id]
        resource = capture.records[0].resource_attributes
        extra_records = tuple(
            LogRecord(
                time_unix_nano=point_index * 100 + event_index + 50,
                observed_time_unix_nano=None,
                severity_number=(
                    17 if event_name.endswith(".error") else 13
                ),
                severity_text=(
                    "ERROR"
                    if event_name.endswith(".error")
                    else "WARN"
                ),
                body=event_name,
                resource_attributes=resource,
                record_attributes={
                    "event.name": event_name,
                    "quantis.experiment.window.index": point_index,
                },
                scope_name="quantis.application",
                scope_version="2.0.0",
                trace_id="",
                span_id="",
                flags=0,
                dropped_attributes_count=0,
            )
            for point_index in range(run.manifest.point_count)
            for event_index, event_name in enumerate(event_names)
        )
        enriched[run.manifest.case_id] = OtlpLogCapture(
            records=capture.records + extra_records,
            sha256=hashlib.sha256(
                f"v2:{run.manifest.case_id}".encode()
            ).hexdigest(),
            source_path=f"memory://v2/{run.manifest.case_id}",
            json_message_count=1,
        )
    return enriched


def write_normal_log_captures(
    captures_directory: Path,
    manifests_directory: Path,
) -> None:
    for manifest_path in sorted(manifests_directory.glob("*.json")):
        manifest = json.loads(manifest_path.read_text())
        case_id = str(manifest["case_id"])
        manifest_sha256 = hashlib.sha256(
            json.dumps(
                manifest,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        resource_attributes = [
            _otlp_attribute("service.name", "quantis-fault-matrix"),
            _otlp_attribute(
                "quantis.experiment.case.id",
                case_id,
            ),
            _otlp_attribute(
                "quantis.experiment.fault.kind",
                str(manifest["fault_kind"]),
            ),
            _otlp_attribute(
                "quantis.experiment.manifest.sha256",
                manifest_sha256,
            ),
        ]
        records = [
            {
                "timeUnixNano": str(
                    point_index * 10 + event_index + 1
                ),
                "severityNumber": 9,
                "severityText": "INFO",
                "body": {"stringValue": event_name},
                "attributes": [
                    _otlp_attribute("event.name", event_name),
                    _otlp_attribute(
                        "quantis.experiment.window.index",
                        point_index,
                    ),
                ],
            }
            for point_index in range(int(manifest["point_count"]))
            for event_index, event_name in enumerate(
                ("checkout.accepted", "checkout.completed")
            )
        ]
        message = {
            "resourceLogs": [
                {
                    "resource": {
                        "attributes": resource_attributes,
                    },
                    "scopeLogs": [
                        {
                            "scope": {
                                "name": "quantis.application",
                                "version": "1.0.0",
                            },
                            "logRecords": records,
                        }
                    ],
                }
            ]
        }
        (
            captures_directory
            / case_id
            / "collector-logs.jsonl"
        ).write_text(
            json.dumps(
                message,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )


def _otlp_attribute(key: str, value):
    encoded = (
        {"intValue": str(value)}
        if isinstance(value, int)
        else {"stringValue": str(value)}
    )
    return {"key": key, "value": encoded}
