"""Command-line entry point for reproducible Quantis experiments."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Optional, Sequence

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
from .otlp_log_windowing import OtlpLogFeatureSpec
from .otlp_logs import OtlpLogCapture, read_otlp_log_capture
from .otlp_windowing import OtlpFeatureSpec, OtlpWindowCompiler
from .multimodal_corpus import (
    compile_multimodal_telemetry_corpus,
)
from .multimodal_training import (
    MultimodalJepaTrainingConfig,
    train_multimodal_jepa_world_model,
    write_multimodal_jepa_development_artifacts,
)
from .contextual_multimodal_corpus import (
    compile_contextual_multimodal_telemetry_corpus,
)
from .contextual_multimodal_training import (
    ContextualMultimodalJepaTrainingConfig,
    train_contextual_multimodal_jepa_world_model,
    write_contextual_multimodal_jepa_artifacts,
)
from .contextual_multimodal_promotion import (
    assess_contextual_multimodal_promotion,
    write_contextual_multimodal_promotion_assessment,
)
from .contextual_confirmation import (
    assess_contextual_confirmation,
    write_contextual_confirmation_assessment,
)
from .contextual_multimodal_development import (
    default_contextual_multimodal_jepa_v2_candidates,
    develop_contextual_multimodal_jepa_v2,
    write_contextual_multimodal_jepa_v2_artifacts,
)
from .telemetry_corpus import (
    TelemetryCorpusSplitSpec,
    compile_telemetry_corpus,
)
from .world_model import (
    JepaTrainingConfig,
    train_jepa_world_model,
    write_jepa_development_artifacts,
)


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
    train_jepa = commands.add_parser(
        "train-jepa-world-model",
        help="compile a run-isolated corpus and fit JEPA v0",
    )
    train_jepa.add_argument(
        "--captures-directory",
        type=Path,
        required=True,
    )
    train_jepa.add_argument(
        "--manifests-directory",
        type=Path,
        required=True,
    )
    train_jepa.add_argument(
        "--feature-spec",
        type=Path,
        required=True,
    )
    train_jepa.add_argument(
        "--split-spec",
        type=Path,
        required=True,
    )
    train_jepa.add_argument(
        "--latent-dimension",
        type=int,
        default=4,
    )
    train_jepa.add_argument("--epochs", type=int, default=200)
    train_jepa.add_argument(
        "--learning-rate",
        type=float,
        default=2e-2,
    )
    train_jepa.add_argument(
        "--ema-decay",
        type=float,
        default=0.98,
    )
    train_jepa.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )
    train_jepa.add_argument(
        "--calibration-quantile",
        type=float,
        default=0.98,
    )
    train_jepa.add_argument("--seed", type=int, default=0)
    train_jepa.add_argument("--output", type=Path, required=True)
    train_multimodal = commands.add_parser(
        "train-multimodal-jepa-world-model",
        help="fit separate metric and application-log JEPA encoders",
    )
    train_multimodal.add_argument(
        "--captures-directory",
        type=Path,
        required=True,
    )
    train_multimodal.add_argument(
        "--manifests-directory",
        type=Path,
        required=True,
    )
    train_multimodal.add_argument(
        "--metric-feature-spec",
        type=Path,
        required=True,
    )
    train_multimodal.add_argument(
        "--log-feature-spec",
        type=Path,
        required=True,
    )
    train_multimodal.add_argument(
        "--split-spec",
        type=Path,
        required=True,
    )
    train_multimodal.add_argument(
        "--metric-latent-dimension",
        type=int,
        default=3,
    )
    train_multimodal.add_argument(
        "--log-latent-dimension",
        type=int,
        default=2,
    )
    train_multimodal.add_argument(
        "--epochs",
        type=int,
        default=200,
    )
    train_multimodal.add_argument(
        "--learning-rate",
        type=float,
        default=2e-2,
    )
    train_multimodal.add_argument(
        "--ema-decay",
        type=float,
        default=0.98,
    )
    train_multimodal.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )
    train_multimodal.add_argument(
        "--calibration-quantile",
        type=float,
        default=0.98,
    )
    train_multimodal.add_argument(
        "--maximum-validation-alert-rate",
        type=float,
        default=0.10,
    )
    train_multimodal.add_argument("--seed", type=int, default=0)
    train_multimodal.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    train_contextual = commands.add_parser(
        "train-contextual-multimodal-jepa-world-model",
        help=(
            "fit a demand-conditioned contextual metrics and logs JEPA"
        ),
    )
    train_contextual.add_argument(
        "--captures-directory",
        type=Path,
        required=True,
    )
    train_contextual.add_argument(
        "--manifests-directory",
        type=Path,
        required=True,
    )
    train_contextual.add_argument(
        "--metric-feature-spec",
        type=Path,
        required=True,
    )
    train_contextual.add_argument(
        "--log-feature-spec",
        type=Path,
        required=True,
    )
    train_contextual.add_argument(
        "--split-spec",
        type=Path,
        required=True,
    )
    train_contextual.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[1, 3, 6],
    )
    train_contextual.add_argument(
        "--target-block-size",
        type=int,
        default=2,
    )
    train_contextual.add_argument(
        "--metric-latent-dimension",
        type=int,
        default=3,
    )
    train_contextual.add_argument(
        "--log-latent-dimension",
        type=int,
        default=1,
    )
    train_contextual.add_argument(
        "--pretraining-epochs",
        type=int,
        default=200,
    )
    train_contextual.add_argument(
        "--predictor-refinement-epochs",
        type=int,
        default=100,
    )
    train_contextual.add_argument(
        "--cross-validation-epochs",
        type=int,
        default=40,
    )
    train_contextual.add_argument(
        "--learning-rate",
        type=float,
        default=2e-2,
    )
    train_contextual.add_argument(
        "--ema-decay",
        type=float,
        default=0.98,
    )
    train_contextual.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )
    train_contextual.add_argument(
        "--loss",
        choices=("huber", "l1", "mse"),
        default="huber",
    )
    train_contextual.add_argument(
        "--huber-delta",
        type=float,
        default=1.0,
    )
    train_contextual.add_argument(
        "--auxiliary-loss-weight",
        type=float,
        default=0.2,
    )
    train_contextual.add_argument(
        "--rollout-loss-weight",
        type=float,
        default=0.2,
    )
    train_contextual.add_argument(
        "--calibration-quantile",
        type=float,
        default=0.98,
    )
    train_contextual.add_argument("--seed", type=int, default=0)
    train_contextual.add_argument(
        "--evidence-mode",
        choices=("development", "promotion_confirmation"),
        default="development",
    )
    train_contextual.add_argument(
        "--promotion-protocol",
        type=Path,
    )
    train_contextual.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    develop_contextual_v2 = commands.add_parser(
        "develop-contextual-multimodal-jepa-v2",
        help=(
            "train and select the fixed dependency-log JEPA v2 "
            "candidate sequence"
        ),
    )
    develop_contextual_v2.add_argument(
        "--captures-directory",
        type=Path,
        required=True,
    )
    develop_contextual_v2.add_argument(
        "--manifests-directory",
        type=Path,
        required=True,
    )
    develop_contextual_v2.add_argument(
        "--metric-feature-spec",
        type=Path,
        required=True,
    )
    develop_contextual_v2.add_argument(
        "--log-feature-spec",
        type=Path,
        required=True,
    )
    develop_contextual_v2.add_argument(
        "--split-spec",
        type=Path,
        required=True,
    )
    develop_contextual_v2.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[1, 3, 6],
    )
    develop_contextual_v2.add_argument(
        "--target-block-size",
        type=int,
        default=2,
    )
    develop_contextual_v2.add_argument(
        "--metric-latent-dimension",
        type=int,
        default=3,
    )
    develop_contextual_v2.add_argument(
        "--pretraining-epochs",
        type=int,
        default=200,
    )
    develop_contextual_v2.add_argument(
        "--predictor-refinement-epochs",
        type=int,
        default=100,
    )
    develop_contextual_v2.add_argument(
        "--cross-validation-epochs",
        type=int,
        default=40,
    )
    develop_contextual_v2.add_argument(
        "--learning-rate",
        type=float,
        default=2e-2,
    )
    develop_contextual_v2.add_argument(
        "--ema-decay",
        type=float,
        default=0.98,
    )
    develop_contextual_v2.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )
    develop_contextual_v2.add_argument(
        "--loss",
        choices=("huber", "l1", "mse"),
        default="huber",
    )
    develop_contextual_v2.add_argument(
        "--huber-delta",
        type=float,
        default=1.0,
    )
    develop_contextual_v2.add_argument(
        "--auxiliary-loss-weight",
        type=float,
        default=0.2,
    )
    develop_contextual_v2.add_argument(
        "--rollout-loss-weight",
        type=float,
        default=0.2,
    )
    develop_contextual_v2.add_argument(
        "--calibration-quantile",
        type=float,
        default=0.98,
    )
    develop_contextual_v2.add_argument(
        "--seed",
        type=int,
        default=89,
    )
    develop_contextual_v2.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    assess_contextual = commands.add_parser(
        "assess-contextual-multimodal-jepa-promotion",
        help=(
            "apply frozen promotion gates to untouched contextual "
            "JEPA evidence"
        ),
    )
    assess_contextual.add_argument(
        "--training-result",
        type=Path,
        required=True,
    )
    assess_contextual.add_argument(
        "--promotion-protocol",
        type=Path,
        required=True,
    )
    assess_contextual.add_argument(
        "--repository",
        type=Path,
        required=True,
    )
    assess_contextual.add_argument(
        "--preregistered-git-commit",
        required=True,
    )
    assess_contextual.add_argument(
        "--repeat-training-result",
        type=Path,
        required=True,
    )
    assess_contextual.add_argument(
        "--training-attestation",
        type=Path,
        required=True,
    )
    assess_contextual.add_argument(
        "--repeat-training-attestation",
        type=Path,
        required=True,
    )
    assess_contextual.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    assess_confirmation = commands.add_parser(
        "assess-contextual-confirmation-v2",
        help=(
            "verify and assess the frozen multi-seed contextual JEPA "
            "confirmation"
        ),
    )
    assess_confirmation.add_argument(
        "--training-result",
        type=Path,
        action="append",
        required=True,
    )
    assess_confirmation.add_argument(
        "--training-attestation",
        type=Path,
        action="append",
        required=True,
    )
    assess_confirmation.add_argument(
        "--repeat-training-result",
        type=Path,
        required=True,
    )
    assess_confirmation.add_argument(
        "--collection-attestation",
        type=Path,
        required=True,
    )
    assess_confirmation.add_argument(
        "--repeat-training-attestation",
        type=Path,
        required=True,
    )
    assess_confirmation.add_argument(
        "--confirmation-protocol",
        type=Path,
        required=True,
    )
    assess_confirmation.add_argument(
        "--repository",
        type=Path,
        required=True,
    )
    assess_confirmation.add_argument(
        "--preregistered-git-commit",
        required=True,
    )
    assess_confirmation.add_argument(
        "--output",
        type=Path,
        required=True,
    )
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
    if parsed.command == "train-jepa-world-model":
        jepa_feature_spec = OtlpFeatureSpec.from_dict(
            json.loads(parsed.feature_spec.read_text())
        )
        split_spec = TelemetryCorpusSplitSpec.from_dict(
            json.loads(parsed.split_spec.read_text())
        )
        corpus = compile_telemetry_corpus(
            _load_fault_matrix_runs(
                parsed.captures_directory,
                parsed.manifests_directory,
            ),
            jepa_feature_spec,
            split_spec,
        )
        result = train_jepa_world_model(
            corpus,
            JepaTrainingConfig(
                latent_dimension=parsed.latent_dimension,
                epochs=parsed.epochs,
                learning_rate=parsed.learning_rate,
                ema_decay=parsed.ema_decay,
                weight_decay=parsed.weight_decay,
                calibration_quantile=(
                    parsed.calibration_quantile
                ),
                seed=parsed.seed,
            ),
        )
        jepa_paths = write_jepa_development_artifacts(
            result,
            parsed.output,
        )
        print("JEPA development training: PASS")
        print(f"Model: {jepa_paths['model']}")
        print(f"Report: {jepa_paths['report']}")
        return 0
    if parsed.command == "train-multimodal-jepa-world-model":
        multimodal_metric_spec = OtlpFeatureSpec.from_dict(
            json.loads(parsed.metric_feature_spec.read_text())
        )
        multimodal_log_spec = OtlpLogFeatureSpec.from_dict(
            json.loads(parsed.log_feature_spec.read_text())
        )
        multimodal_split_spec = TelemetryCorpusSplitSpec.from_dict(
            json.loads(parsed.split_spec.read_text())
        )
        multimodal_runs = _load_fault_matrix_runs(
            parsed.captures_directory,
            parsed.manifests_directory,
        )
        multimodal_corpus = compile_multimodal_telemetry_corpus(
            multimodal_runs,
            _load_log_captures(
                parsed.captures_directory,
                multimodal_runs,
            ),
            multimodal_metric_spec,
            multimodal_log_spec,
            multimodal_split_spec,
        )
        multimodal_result = train_multimodal_jepa_world_model(
            multimodal_corpus,
            MultimodalJepaTrainingConfig(
                metric_latent_dimension=(
                    parsed.metric_latent_dimension
                ),
                log_latent_dimension=parsed.log_latent_dimension,
                epochs=parsed.epochs,
                learning_rate=parsed.learning_rate,
                ema_decay=parsed.ema_decay,
                weight_decay=parsed.weight_decay,
                calibration_quantile=(
                    parsed.calibration_quantile
                ),
                maximum_validation_alert_rate=(
                    parsed.maximum_validation_alert_rate
                ),
                seed=parsed.seed,
            ),
        )
        multimodal_paths = (
            write_multimodal_jepa_development_artifacts(
                multimodal_result,
                parsed.output,
            )
        )
        print("Multimodal JEPA development training: PASS")
        print(
            "Promotion gates: "
            f"{multimodal_result.promotion['status'].upper()}"
        )
        print(f"Model: {multimodal_paths['model']}")
        print(f"Report: {multimodal_paths['report']}")
        return 0
    if (
        parsed.command
        == "train-contextual-multimodal-jepa-world-model"
    ):
        contextual_metric_spec = OtlpFeatureSpec.from_dict(
            json.loads(parsed.metric_feature_spec.read_text())
        )
        contextual_log_spec = OtlpLogFeatureSpec.from_dict(
            json.loads(parsed.log_feature_spec.read_text())
        )
        contextual_split_spec = TelemetryCorpusSplitSpec.from_dict(
            json.loads(parsed.split_spec.read_text())
        )
        contextual_runs = _load_fault_matrix_runs(
            parsed.captures_directory,
            parsed.manifests_directory,
        )
        base_contextual_corpus = (
            compile_multimodal_telemetry_corpus(
                contextual_runs,
                _load_log_captures(
                    parsed.captures_directory,
                    contextual_runs,
                ),
                contextual_metric_spec,
                contextual_log_spec,
                contextual_split_spec,
            )
        )
        contextual_corpus = (
            compile_contextual_multimodal_telemetry_corpus(
                base_contextual_corpus,
                contextual_runs,
                horizons=tuple(parsed.horizons),
                target_block_size=parsed.target_block_size,
            )
        )
        contextual_result = (
            train_contextual_multimodal_jepa_world_model(
                contextual_corpus,
                ContextualMultimodalJepaTrainingConfig(
                    metric_latent_dimension=(
                        parsed.metric_latent_dimension
                    ),
                    log_latent_dimension=(
                        parsed.log_latent_dimension
                    ),
                    pretraining_epochs=(
                        parsed.pretraining_epochs
                    ),
                    predictor_refinement_epochs=(
                        parsed.predictor_refinement_epochs
                    ),
                    cross_validation_epochs=(
                        parsed.cross_validation_epochs
                    ),
                    learning_rate=parsed.learning_rate,
                    ema_decay=parsed.ema_decay,
                    weight_decay=parsed.weight_decay,
                    loss=parsed.loss,
                    huber_delta=parsed.huber_delta,
                    auxiliary_loss_weight=(
                        parsed.auxiliary_loss_weight
                    ),
                    rollout_loss_weight=(
                        parsed.rollout_loss_weight
                    ),
                    calibration_quantile=(
                        parsed.calibration_quantile
                    ),
                    seed=parsed.seed,
                ),
                evidence_mode=parsed.evidence_mode,
                promotion_protocol=(
                    json.loads(
                        parsed.promotion_protocol.read_text()
                    )
                    if parsed.promotion_protocol is not None
                    else None
                ),
            )
        )
        contextual_paths = (
            write_contextual_multimodal_jepa_artifacts(
                contextual_result,
                parsed.output,
            )
        )
        print("Contextual multimodal JEPA training: PASS")
        print(
            "Development selection: "
            f"{contextual_result.selection['status'].upper()}"
        )
        print(f"Model: {contextual_paths['model']}")
        print(f"Report: {contextual_paths['report']}")
        return 0
    if parsed.command == "develop-contextual-multimodal-jepa-v2":
        contextual_metric_spec = OtlpFeatureSpec.from_dict(
            json.loads(parsed.metric_feature_spec.read_text())
        )
        contextual_log_spec = OtlpLogFeatureSpec.from_dict(
            json.loads(parsed.log_feature_spec.read_text())
        )
        contextual_split_spec = TelemetryCorpusSplitSpec.from_dict(
            json.loads(parsed.split_spec.read_text())
        )
        contextual_runs = _load_fault_matrix_runs(
            parsed.captures_directory,
            parsed.manifests_directory,
        )
        base_contextual_corpus = (
            compile_multimodal_telemetry_corpus(
                contextual_runs,
                _load_log_captures(
                    parsed.captures_directory,
                    contextual_runs,
                ),
                contextual_metric_spec,
                contextual_log_spec,
                contextual_split_spec,
            )
        )
        contextual_corpus = (
            compile_contextual_multimodal_telemetry_corpus(
                base_contextual_corpus,
                contextual_runs,
                horizons=tuple(parsed.horizons),
                target_block_size=parsed.target_block_size,
            )
        )
        base_config = ContextualMultimodalJepaTrainingConfig(
            metric_latent_dimension=(
                parsed.metric_latent_dimension
            ),
            pretraining_epochs=parsed.pretraining_epochs,
            predictor_refinement_epochs=(
                parsed.predictor_refinement_epochs
            ),
            cross_validation_epochs=(
                parsed.cross_validation_epochs
            ),
            learning_rate=parsed.learning_rate,
            ema_decay=parsed.ema_decay,
            weight_decay=parsed.weight_decay,
            loss=parsed.loss,
            huber_delta=parsed.huber_delta,
            auxiliary_loss_weight=parsed.auxiliary_loss_weight,
            rollout_loss_weight=parsed.rollout_loss_weight,
            calibration_quantile=parsed.calibration_quantile,
            seed=parsed.seed,
        )
        v2_result = develop_contextual_multimodal_jepa_v2(
            contextual_corpus,
            default_contextual_multimodal_jepa_v2_candidates(
                base_config
            ),
        )
        v2_paths = (
            write_contextual_multimodal_jepa_v2_artifacts(
                v2_result,
                parsed.output,
            )
        )
        print("Contextual multimodal JEPA v2 development: PASS")
        print(
            "Candidate selection: "
            f"{v2_result.selection['status'].upper()}"
        )
        selected = v2_result.selection["selected_candidate"]
        print(f"Selected candidate: {selected or 'none'}")
        print(f"Report: {v2_paths['report']}")
        return 0
    if (
        parsed.command
        == "assess-contextual-multimodal-jepa-promotion"
    ):
        promotion_assessment = (
            assess_contextual_multimodal_promotion(
                parsed.training_result,
                parsed.promotion_protocol,
                repeat_training_result_path=(
                    parsed.repeat_training_result
                ),
                training_attestation_path=(
                    parsed.training_attestation
                ),
                repeat_training_attestation_path=(
                    parsed.repeat_training_attestation
                ),
                repository=parsed.repository,
                preregistered_git_commit=(
                    parsed.preregistered_git_commit
                ),
            )
        )
        promotion_paths = (
            write_contextual_multimodal_promotion_assessment(
                promotion_assessment,
                parsed.output,
            )
        )
        print(
            "Contextual multimodal JEPA promotion: "
            f"{promotion_assessment['status'].upper()}"
        )
        print(f"Report: {promotion_paths['report']}")
        return (
            0
            if promotion_assessment["status"] == "passed"
            else 1
        )
    if parsed.command == "assess-contextual-confirmation-v2":
        confirmation_assessment = assess_contextual_confirmation(
            parsed.training_result,
            parsed.training_attestation,
            collection_attestation_path=(
                parsed.collection_attestation
            ),
            repeat_training_result_path=(
                parsed.repeat_training_result
            ),
            repeat_training_attestation_path=(
                parsed.repeat_training_attestation
            ),
            confirmation_protocol_path=(
                parsed.confirmation_protocol
            ),
            repository=parsed.repository,
            preregistered_git_commit=(
                parsed.preregistered_git_commit
            ),
        )
        confirmation_paths = (
            write_contextual_confirmation_assessment(
                confirmation_assessment,
                parsed.output,
            )
        )
        print(
            "Contextual multimodal JEPA confirmation: "
            f"{confirmation_assessment['status'].upper()}"
        )
        print(f"Report: {confirmation_paths['report']}")
        return (
            0
            if confirmation_assessment["publication_ready"]
            else 1
        )
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


def _load_log_captures(
    captures_directory: Path,
    runs: Sequence[FaultMatrixRun],
) -> Mapping[str, OtlpLogCapture]:
    return {
        run.manifest.case_id: read_otlp_log_capture(
            captures_directory
            / run.manifest.case_id
            / "collector-logs.jsonl"
        )
        for run in runs
    }


if __name__ == "__main__":
    raise SystemExit(main())
