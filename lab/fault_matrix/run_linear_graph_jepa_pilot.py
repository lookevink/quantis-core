"""Train the linear graph-JEPA tracer on the preserved graph corpus."""

import argparse
import hashlib
from pathlib import Path
from typing import Mapping, Optional, Sequence

from quantis_core.graph_jepa import (
    GraphJepaTrainingConfig,
    LinearGraphJepaWorldModel,
    evaluate_linear_graph_jepa,
    write_linear_graph_jepa_artifacts,
)

from run_graph_observability_pilot import (
    compile_preserved_graph_corpus,
)


def run_linear_graph_jepa_pilot(
    *,
    captures_directory: Path,
    manifests_directory: Path,
    metric_feature_spec_path: Path,
    log_feature_spec_path: Path,
    split_spec_path: Path,
    source_observability_assessment_path: Path,
    output_directory: Path,
) -> Mapping[str, Path]:
    """Fit equal-encoder graph-scope controls and evaluate held-out state."""

    training, validation, case_count = compile_preserved_graph_corpus(
        captures_directory=captures_directory,
        manifests_directory=manifests_directory,
        metric_feature_spec_path=metric_feature_spec_path,
        log_feature_spec_path=log_feature_spec_path,
        split_spec_path=split_spec_path,
    )
    models = {
        scope: LinearGraphJepaWorldModel(
            GraphJepaTrainingConfig(
                latent_dimension=2,
                ridge=1e-3,
                context_scope=scope,
            )
        ).fit(training[0])
        for scope in (
            "entity_local",
            "one_hop",
            "all_entities",
        )
    }
    assessment = dict(
        evaluate_linear_graph_jepa(
            models,
            training[0],
            validation[0],
            validation_window_case_ids=validation[1],
        )
    )
    assessment["source_case_count"] = case_count
    assessment[
        "source_observability_assessment_sha256"
    ] = hashlib.sha256(
        source_observability_assessment_path.read_bytes()
    ).hexdigest()
    return write_linear_graph_jepa_artifacts(
        models, assessment, output_directory
    )


def main(arguments: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="run the linear graph-JEPA development tracer"
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
        "--source-observability-assessment",
        type=Path,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parsed = parser.parse_args(arguments)
    paths = run_linear_graph_jepa_pilot(
        captures_directory=parsed.captures_directory,
        manifests_directory=parsed.manifests_directory,
        metric_feature_spec_path=parsed.metric_feature_spec,
        log_feature_spec_path=parsed.log_feature_spec,
        split_spec_path=parsed.split_spec,
        source_observability_assessment_path=(
            parsed.source_observability_assessment
        ),
        output_directory=parsed.output,
    )
    print(f"Linear graph-JEPA report: {paths['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
