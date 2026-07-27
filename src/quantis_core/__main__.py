"""Command-line entry point for reproducible Quantis experiments."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Optional, Sequence

from .evaluation import (
    EvaluationConfig,
    run_evaluation,
    write_evaluation_artifacts,
)
from .fault_lab import (
    FaultLabManifest,
    evaluate_fault_lab,
    write_fault_lab_artifacts,
)
from .otlp import read_otlp_capture
from .otlp_windowing import OtlpFeatureSpec, OtlpWindowCompiler


def main(arguments: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m quantis_core")
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("evaluate", help="run the synthetic thesis test")
    evaluate.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evaluation"),
        help="directory for reports and fitted artifacts",
    )
    evaluate.add_argument(
        "--quick",
        action="store_true",
        help="use four held-out scenarios for a fast CI evaluation",
    )
    replay = commands.add_parser(
        "replay-otlp", help="compile an OTLP JSON capture into model features"
    )
    replay.add_argument("--capture", type=Path, required=True)
    replay.add_argument("--feature-spec", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)
    fault_lab = commands.add_parser(
        "evaluate-fault-lab",
        help="evaluate an instrumented API/worker OTLP capture",
    )
    fault_lab.add_argument("--capture", type=Path, required=True)
    fault_lab.add_argument("--feature-spec", type=Path, required=True)
    fault_lab.add_argument("--manifest", type=Path, required=True)
    fault_lab.add_argument("--output", type=Path, required=True)
    parsed = parser.parse_args(arguments)

    if parsed.command == "evaluate":
        config = (
            EvaluationConfig(
                train_seeds=(11, 23, 37),
                test_seeds=(101, 103, 107, 109),
                scenario_length=360,
            )
            if parsed.quick
            else EvaluationConfig()
        )
        report = run_evaluation(config)
        paths = write_evaluation_artifacts(report, parsed.output)
        status = "PASS" if report.acceptance["all_passed"] else "FAIL"
        print(f"Acceptance: {status}")
        print(f"Report: {paths['report']}")
        return 0 if report.acceptance["all_passed"] else 1
    if parsed.command == "replay-otlp":
        capture = read_otlp_capture(parsed.capture)
        feature_spec = OtlpFeatureSpec.from_dict(
            json.loads(parsed.feature_spec.read_text())
        )
        compiled = OtlpWindowCompiler(feature_spec).compile(capture)
        compiled_payload = compiled.to_dict()
        encoded = json.dumps(
            compiled_payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n"
        parsed.output.mkdir(parents=True, exist_ok=True)
        (parsed.output / "compiled-telemetry.json").write_text(encoded)
        summary = {
            "schema_version": 1,
            "capture_sha256": capture.sha256,
            "feature_schema_id": compiled.feature_schema_id,
            "compiled_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
            "window_count": len(compiled.window_end_unix_nano),
            "feature_count": len(compiled.feature_names),
            "data_quality": dict(compiled.data_quality),
        }
        (parsed.output / "replay.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        print("OTLP replay: PASS")
        print(f"Compiled telemetry: {parsed.output / 'compiled-telemetry.json'}")
        return 0
    if parsed.command == "evaluate-fault-lab":
        capture = read_otlp_capture(parsed.capture)
        feature_spec = OtlpFeatureSpec.from_dict(
            json.loads(parsed.feature_spec.read_text())
        )
        manifest = FaultLabManifest.from_dict(
            json.loads(parsed.manifest.read_text())
        )
        fault_report = evaluate_fault_lab(capture, feature_spec, manifest)
        fault_paths = write_fault_lab_artifacts(
            fault_report, parsed.output
        )
        status = (
            "PASS" if fault_report.acceptance["all_passed"] else "FAIL"
        )
        print(f"Fault-lab acceptance: {status}")
        print(f"Report: {fault_paths['report']}")
        return 0 if fault_report.acceptance["all_passed"] else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
