"""Evaluate declared graph state on the preserved confirmation corpus."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Optional, Sequence

from quantis_core.contextual_multimodal_corpus import (
    compile_contextual_multimodal_telemetry_corpus,
)
from quantis_core.fault_matrix import (
    FaultMatrixCaseManifest,
    FaultMatrixRun,
)
from quantis_core.graph_observability import (
    evaluate_graph_observability,
    write_graph_observability_assessment,
)
from quantis_core.graph_telemetry import (
    GraphStateWindows,
    compile_graph_state_windows,
    quantis_checkout_graph,
)
from quantis_core.multimodal_corpus import (
    compile_multimodal_telemetry_corpus,
)
from quantis_core.otlp import read_otlp_capture
from quantis_core.otlp_logs import read_otlp_log_capture
from quantis_core.otlp_log_windowing import OtlpLogFeatureSpec
from quantis_core.otlp_windowing import OtlpFeatureSpec
from quantis_core.telemetry_corpus import TelemetryCorpusSplitSpec


def run_graph_observability_pilot(
    *,
    captures_directory: Path,
    manifests_directory: Path,
    metric_feature_spec_path: Path,
    log_feature_spec_path: Path,
    split_spec_path: Path,
    source_assessment_path: Path,
    output_directory: Path,
) -> Mapping[str, Path]:
    """Compile, graph-bind, and score the inspected development corpus."""

    if output_directory.exists():
        raise FileExistsError(
            "refusing to overwrite graph observability pilot: "
            f"{output_directory}"
        )
    training, validation, case_count = compile_preserved_graph_corpus(
        captures_directory=captures_directory,
        manifests_directory=manifests_directory,
        metric_feature_spec_path=metric_feature_spec_path,
        log_feature_spec_path=log_feature_spec_path,
        split_spec_path=split_spec_path,
    )
    assessment = dict(
        evaluate_graph_observability(
            training[0],
            validation[0],
            training_window_case_ids=training[1],
            validation_window_case_ids=validation[1],
            ridge=1e-3,
        )
    )
    assessment["declared_graph"] = training[0].graph.to_dict()
    assessment["source_confirmation_assessment_sha256"] = (
        hashlib.sha256(source_assessment_path.read_bytes()).hexdigest()
    )
    assessment["source_case_count"] = case_count
    return write_graph_observability_assessment(
        assessment, output_directory
    )


def compile_preserved_graph_corpus(
    *,
    captures_directory: Path,
    manifests_directory: Path,
    metric_feature_spec_path: Path,
    log_feature_spec_path: Path,
    split_spec_path: Path,
) -> tuple[
    tuple[GraphStateWindows, tuple[str, ...]],
    tuple[GraphStateWindows, tuple[str, ...]],
    int,
]:
    """Compile the preserved contextual corpus into declared graph tensors."""

    manifests = []
    runs = []
    log_captures = {}
    for manifest_path in sorted(manifests_directory.glob("*.json")):
        manifest = FaultMatrixCaseManifest.from_dict(
            json.loads(manifest_path.read_text())
        )
        manifests.append(manifest)
        case_directory = captures_directory / manifest.case_id
        runs.append(
            FaultMatrixRun(
                manifest=manifest,
                capture=read_otlp_capture(
                    case_directory / "collector-output.jsonl"
                ),
            )
        )
        log_captures[manifest.case_id] = read_otlp_log_capture(
            case_directory / "collector-logs.jsonl"
        )
    metric_spec = OtlpFeatureSpec.from_dict(
        json.loads(metric_feature_spec_path.read_text())
    )
    log_spec = OtlpLogFeatureSpec.from_dict(
        json.loads(log_feature_spec_path.read_text())
    )
    split_spec = TelemetryCorpusSplitSpec.from_dict(
        json.loads(split_spec_path.read_text())
    )
    base = compile_multimodal_telemetry_corpus(
        runs,
        log_captures,
        metric_spec,
        log_spec,
        split_spec,
    )
    contextual = compile_contextual_multimodal_telemetry_corpus(
        base,
        runs,
        horizons=(1, 3, 6),
        target_block_size=2,
    )
    graph = quantis_checkout_graph()
    training = compile_graph_state_windows(
        contextual.training.windows, graph
    )
    validation = compile_graph_state_windows(
        contextual.validation.windows, graph
    )
    return (
        (training, contextual.training.window_case_ids),
        (validation, contextual.validation.window_case_ids),
        len(manifests),
    )


def main(arguments: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="run the graph JEPA raw-observability pilot"
    )
    parser.add_argument(
        "--captures-directory", type=Path, required=True
    )
    parser.add_argument(
        "--manifests-directory", type=Path, required=True
    )
    parser.add_argument(
        "--metric-feature-spec", type=Path, required=True
    )
    parser.add_argument(
        "--log-feature-spec", type=Path, required=True
    )
    parser.add_argument("--split-spec", type=Path, required=True)
    parser.add_argument(
        "--source-assessment", type=Path, required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parsed = parser.parse_args(arguments)
    paths = run_graph_observability_pilot(
        captures_directory=parsed.captures_directory,
        manifests_directory=parsed.manifests_directory,
        metric_feature_spec_path=parsed.metric_feature_spec,
        log_feature_spec_path=parsed.log_feature_spec,
        split_spec_path=parsed.split_spec,
        source_assessment_path=parsed.source_assessment,
        output_directory=parsed.output,
    )
    print(f"Graph observability report: {paths['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
