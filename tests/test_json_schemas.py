import json
from pathlib import Path

from jsonschema import Draft202012Validator


def load_json(path):
    return json.loads(Path(path).read_text())


def test_versioned_contract_artifacts_validate_against_their_json_schemas():
    feature_schema = load_json(
        "contracts/schemas/otlp-feature-spec.schema.json"
    )
    compiled_schema = load_json(
        "contracts/schemas/compiled-telemetry.schema.json"
    )
    detection_schema = load_json(
        "contracts/schemas/detection-event.schema.json"
    )
    fault_manifest_schema = load_json(
        "contracts/schemas/fault-lab-manifest.schema.json"
    )
    fault_report_schema = load_json(
        "contracts/schemas/fault-lab-report.schema.json"
    )
    for schema in (
        feature_schema,
        compiled_schema,
        detection_schema,
        fault_manifest_schema,
        fault_report_schema,
    ):
        Draft202012Validator.check_schema(schema)

    Draft202012Validator(feature_schema).validate(
        load_json("lab/otel/scenario-feature-spec.json")
    )
    Draft202012Validator(compiled_schema).validate(
        load_json("artifacts/otlp-replay/compiled-telemetry.json")
    )
    detection_payload = load_json(
        "artifacts/otlp-replay/detection-events.json"
    )
    for event in detection_payload["events"]:
        Draft202012Validator(detection_schema).validate(event)
    Draft202012Validator(fault_manifest_schema).validate(
        load_json("lab/fault/experiment.json")
    )
    Draft202012Validator(fault_report_schema).validate(
        load_json("artifacts/fault-lab/verification.json")
    )
