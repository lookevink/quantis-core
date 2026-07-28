"""Development-only latent-width sweep for the linear graph-JEPA tracer."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from quantis_core.graph_jepa import (
    GraphJepaTrainingConfig,
    LinearGraphJepaWorldModel,
    evaluate_linear_graph_jepa,
)

from run_graph_observability_pilot import (
    compile_preserved_graph_corpus,
)


WIDTHS = (1, 2, 3, 4, 6, 8)


def run_graph_jepa_width_sweep(
    *,
    captures_directory: Path,
    manifests_directory: Path,
    metric_feature_spec_path: Path,
    log_feature_spec_path: Path,
    split_spec_path: Path,
    output_directory: Path,
) -> Mapping[str, Path]:
    """Fit every width on one compiled corpus and select without promotion."""

    if output_directory.exists():
        raise FileExistsError(
            f"refusing to overwrite graph JEPA width sweep: "
            f"{output_directory}"
        )
    training, validation, case_count = compile_preserved_graph_corpus(
        captures_directory=captures_directory,
        manifests_directory=manifests_directory,
        metric_feature_spec_path=metric_feature_spec_path,
        log_feature_spec_path=log_feature_spec_path,
        split_spec_path=split_spec_path,
    )
    candidates: Dict[str, Mapping[str, Any]] = {}
    for width in WIDTHS:
        models = {
            scope: LinearGraphJepaWorldModel(
                GraphJepaTrainingConfig(
                    latent_dimension=width,
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
        candidates[str(width)] = evaluate_linear_graph_jepa(
            models,
            training[0],
            validation[0],
            validation_window_case_ids=validation[1],
        )
    supported = [
        width
        for width in WIDTHS
        if candidates[str(width)]["status"] == "supported"
    ]
    selected_width = supported[0] if supported else None
    result = {
        "schema_version": 1,
        "kind": "linear_graph_jepa_width_sweep",
        "status": "selected" if selected_width else "not_supported",
        "selected_width": selected_width,
        "selection_rule": (
            "smallest width passing every frozen development gate"
        ),
        "source_case_count": case_count,
        "evidence_boundary": (
            "post-confirmation development on already inspected "
            "fault-free schedule families"
        ),
        "candidates": candidates,
    }
    output_directory.mkdir(parents=True)
    result_path = output_directory / "width-sweep.json"
    result_path.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    report_path = output_directory / "report.md"
    report_path.write_text(_report(result))
    return {"result": result_path, "report": report_path}


def _report(result: Mapping[str, Any]) -> str:
    lines = [
        "# Linear graph-JEPA latent-width sweep",
        "",
        f"Result: **{str(result['status']).upper()}**",
        "",
        f"Selected width: `{result['selected_width']}`",
        "",
        "| width | context ratio | PCA reconstruction | one-hop prediction | local prediction | all-entity prediction | status |",
        "|---:|---:|---:|---:|---:|---:|:---|",
    ]
    for width in WIDTHS:
        candidate = dict(result["candidates"][str(width)])
        compression = dict(candidate["compression"])
        representations = dict(candidate["representations"])
        lines.append(
            "| "
            f"{width} | "
            f"{float(compression['context_ratio']):.3f}:1 | "
            f"{_error(representations, 'pca_target_reconstruction'):.4f} | "
            f"{_error(representations, 'one_hop'):.4f} | "
            f"{_error(representations, 'entity_local'):.4f} | "
            f"{_error(representations, 'all_entities'):.4f} | "
            f"{candidate['status']} |"
        )
    lines.extend(
        (
            "",
            "Evidence boundary: "
            f"{result['evidence_boundary']}.",
            "",
        )
    )
    return "\n".join(lines)


def _error(
    representations: Mapping[str, Any],
    name: str,
) -> float:
    return float(
        dict(representations[name])[
            "mean_validation_normalized_mse"
        ]
    )


def main(arguments: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="sweep linear graph-JEPA entity latent widths"
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
    parser.add_argument("--output", type=Path, required=True)
    parsed = parser.parse_args(arguments)
    paths = run_graph_jepa_width_sweep(
        captures_directory=parsed.captures_directory,
        manifests_directory=parsed.manifests_directory,
        metric_feature_spec_path=parsed.metric_feature_spec,
        log_feature_spec_path=parsed.log_feature_spec,
        split_spec_path=parsed.split_spec,
        output_directory=parsed.output,
    )
    print(f"Graph-JEPA width sweep: {paths['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
