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
from .demand_conditioning import (
    train_demand_conditioned_model,
    write_demand_conditioned_model,
)
from .fault_lab import (
    FaultLabManifest,
    evaluate_fault_lab,
    write_fault_lab_artifacts,
)
from .fault_matrix import (
    FaultMatrixCaseManifest,
    FaultMatrixRun,
    evaluate_fault_matrix,
    evaluate_demand_conditioned_fault_matrix,
    write_fault_matrix_artifacts,
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
    fault_matrix = commands.add_parser(
        "evaluate-fault-matrix",
        help="score held-out fault captures with frozen model artifacts",
    )
    fault_matrix.add_argument(
        "--captures-directory", type=Path, required=True
    )
    fault_matrix.add_argument(
        "--manifests-directory", type=Path, required=True
    )
    fault_matrix.add_argument("--feature-spec", type=Path, required=True)
    fault_matrix.add_argument(
        "--window-compiler", type=Path, required=True
    )
    fault_matrix.add_argument("--detector", type=Path, required=True)
    fault_matrix.add_argument(
        "--window-compiler-file-sha256", required=True
    )
    fault_matrix.add_argument("--detector-file-sha256", required=True)
    fault_matrix.add_argument("--output", type=Path, required=True)
    train_v2 = commands.add_parser(
        "train-demand-conditioned-v2",
        help="fit v2 from fault-free intervals across development captures",
    )
    train_v2.add_argument(
        "--captures-directory", type=Path, required=True
    )
    train_v2.add_argument(
        "--manifests-directory", type=Path, required=True
    )
    train_v2.add_argument("--feature-spec", type=Path, required=True)
    train_v2.add_argument("--output", type=Path, required=True)
    evaluate_v2 = commands.add_parser(
        "evaluate-demand-conditioned-matrix",
        help="score fault captures with one frozen demand-conditioned model",
    )
    evaluate_v2.add_argument(
        "--captures-directory", type=Path, required=True
    )
    evaluate_v2.add_argument(
        "--manifests-directory", type=Path, required=True
    )
    evaluate_v2.add_argument("--feature-spec", type=Path, required=True)
    evaluate_v2.add_argument("--model", type=Path, required=True)
    evaluate_v2.add_argument("--model-file-sha256", required=True)
    evaluate_v2.add_argument("--confirmation-protocol", type=Path)
    evaluate_v2.add_argument("--preregistered-git-commit")
    evaluate_v2.add_argument("--output", type=Path, required=True)
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
    if parsed.command == "evaluate-fault-matrix":
        compiler_bytes = parsed.window_compiler.read_bytes()
        detector_bytes = parsed.detector.read_bytes()
        compiler_file_sha256 = hashlib.sha256(
            compiler_bytes
        ).hexdigest()
        detector_file_sha256 = hashlib.sha256(
            detector_bytes
        ).hexdigest()
        if (
            compiler_file_sha256
            != parsed.window_compiler_file_sha256
            or detector_file_sha256 != parsed.detector_file_sha256
        ):
            raise ValueError(
                "frozen artifact file changed after capture began"
            )
        matrix_feature_spec = OtlpFeatureSpec.from_dict(
            json.loads(parsed.feature_spec.read_text())
        )
        matrix_report = evaluate_fault_matrix(
            _load_fault_matrix_runs(
                parsed.captures_directory,
                parsed.manifests_directory,
            ),
            matrix_feature_spec,
            compiler_bytes,
            detector_bytes,
        )
        matrix_paths = write_fault_matrix_artifacts(
            matrix_report, parsed.output
        )
        matrix_status = (
            "PASS" if matrix_report.acceptance["all_passed"] else "FAIL"
        )
        print(f"Fault-matrix acceptance: {matrix_status}")
        print(f"Report: {matrix_paths['report']}")
        return 0 if matrix_report.acceptance["all_passed"] else 1
    if parsed.command == "train-demand-conditioned-v2":
        training_feature_spec = OtlpFeatureSpec.from_dict(
            json.loads(parsed.feature_spec.read_text())
        )
        training_runs = _load_fault_matrix_runs(
            parsed.captures_directory,
            parsed.manifests_directory,
        )
        v2_model = train_demand_conditioned_model(
            training_runs, training_feature_spec
        )
        v2_paths = write_demand_conditioned_model(
            v2_model, parsed.output
        )
        print("Demand-conditioned v2 training: PASS")
        print(f"Model: {v2_paths['model']}")
        return 0
    if parsed.command == "evaluate-demand-conditioned-matrix":
        v2_model_bytes = parsed.model.read_bytes()
        if hashlib.sha256(v2_model_bytes).hexdigest() != (
            parsed.model_file_sha256
        ):
            raise ValueError(
                "demand-conditioned model changed after evaluation began"
            )
        v2_feature_spec = OtlpFeatureSpec.from_dict(
            json.loads(parsed.feature_spec.read_text())
        )
        v2_report = evaluate_demand_conditioned_fault_matrix(
            _load_fault_matrix_runs(
                parsed.captures_directory,
                parsed.manifests_directory,
            ),
            v2_feature_spec,
            v2_model_bytes,
            (
                parsed.confirmation_protocol.read_bytes()
                if parsed.confirmation_protocol is not None
                else None
            ),
            parsed.preregistered_git_commit,
        )
        v2_paths = write_fault_matrix_artifacts(
            v2_report, parsed.output
        )
        v2_status = (
            "PASS" if v2_report.acceptance["all_passed"] else "FAIL"
        )
        print(f"Demand-conditioned matrix acceptance: {v2_status}")
        print(f"Report: {v2_paths['report']}")
        return 0 if v2_report.acceptance["all_passed"] else 1
    return 2


def _load_fault_matrix_runs(
    captures_directory: Path,
    manifests_directory: Path,
) -> Sequence[FaultMatrixRun]:
    runs = []
    for manifest_path in sorted(manifests_directory.glob("*.json")):
        manifest = FaultMatrixCaseManifest.from_dict(
            json.loads(manifest_path.read_text())
        )
        runs.append(
            FaultMatrixRun(
                manifest=manifest,
                capture=read_otlp_capture(
                    captures_directory
                    / manifest.case_id
                    / "collector-output.jsonl"
                ),
            )
        )
    return runs


if __name__ == "__main__":
    raise SystemExit(main())
