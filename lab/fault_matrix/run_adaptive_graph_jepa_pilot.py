"""Training-only per-entity width selection for the graph-JEPA tracer."""

import argparse
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

from quantis_core.graph_jepa import (
    GraphJepaTrainingConfig,
    LinearGraphJepaWorldModel,
    evaluate_linear_graph_jepa,
    write_linear_graph_jepa_artifacts,
)
from quantis_core.graph_telemetry import GraphStateWindows

from run_graph_jepa_width_sweep import WIDTHS
from run_graph_observability_pilot import (
    compile_preserved_graph_corpus,
)


def run_adaptive_graph_jepa_pilot(
    *,
    captures_directory: Path,
    manifests_directory: Path,
    metric_feature_spec_path: Path,
    log_feature_spec_path: Path,
    split_spec_path: Path,
    output_directory: Path,
) -> Mapping[str, Path]:
    """Select widths on training reconstruction, then evaluate once."""

    training, validation, case_count = compile_preserved_graph_corpus(
        captures_directory=captures_directory,
        manifests_directory=manifests_directory,
        metric_feature_spec_path=metric_feature_spec_path,
        log_feature_spec_path=log_feature_spec_path,
        split_spec_path=split_spec_path,
    )
    reconstruction_by_width = {}
    for width in WIDTHS:
        model = LinearGraphJepaWorldModel(
            GraphJepaTrainingConfig(
                latent_dimension=width,
                ridge=1e-3,
                context_scope="one_hop",
            )
        ).fit(training[0])
        reconstruction_by_width[width] = (
            _training_entity_reconstruction(model, training[0])
        )
    observed_entities = tuple(
        entity_id
        for position, entity_id in enumerate(training[0].entity_ids)
        if np.any(training[0].observation_mask[position])
    )
    entity_widths: Dict[str, int] = {}
    for entity_id in observed_entities:
        passing = [
            width
            for width in WIDTHS
            if reconstruction_by_width[width][entity_id] <= 0.1
        ]
        if not passing:
            raise RuntimeError(
                "no candidate width preserves training state for "
                f"{entity_id}"
            )
        entity_widths[entity_id] = passing[0]
    maximum_width = max(entity_widths.values())
    models = {
        scope: LinearGraphJepaWorldModel(
            GraphJepaTrainingConfig(
                latent_dimension=maximum_width,
                ridge=1e-3,
                context_scope=scope,
                entity_latent_dimensions=entity_widths,
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
    compression = dict(assessment["compression"])
    gates = dict(assessment["gates"])
    gates["active_graph_context_is_compressed"] = {
        "observed_ratio": float(compression["context_ratio"]),
        "minimum_exclusive": 1.0,
        "passed": float(compression["context_ratio"]) > 1.0,
    }
    supported = all(
        bool(dict(gate)["passed"]) for gate in gates.values()
    )
    assessment["gates"] = gates
    assessment["status"] = (
        "supported" if supported else "not_supported"
    )
    assessment["decision"] = (
        "collect_observability_rich_corpus"
        if supported
        else "improve_graph_representation_before_collection"
    )
    assessment["adaptive_width_selection"] = {
        "fit_split": "training_schedule_families_only",
        "maximum_training_reconstruction_normalized_mse": 0.1,
        "entity_latent_dimensions": entity_widths,
        "training_reconstruction_by_width": {
            str(width): values
            for width, values in reconstruction_by_width.items()
        },
    }
    assessment["source_case_count"] = case_count
    return write_linear_graph_jepa_artifacts(
        models, assessment, output_directory
    )


def _training_entity_reconstruction(
    model: LinearGraphJepaWorldModel,
    training: GraphStateWindows,
) -> Mapping[str, float]:
    prediction = model.predict(training)
    errors: Dict[str, float] = {}
    for entity_position, entity_id in enumerate(
        training.entity_ids
    ):
        feature_errors = []
        for slot_position, _ in enumerate(
            training.local_feature_keys[entity_position]
        ):
            target = training.target_blocks[
                :, :, :, entity_position, slot_position
            ].reshape(-1)
            variance = float(np.var(target))
            if variance <= 1e-12:
                continue
            reconstruction = prediction.reconstructed_target_blocks[
                :, :, :, entity_position, slot_position
            ].reshape(-1)
            feature_errors.append(
                float(
                    np.mean(np.square(reconstruction - target))
                    / variance
                )
            )
        if feature_errors:
            errors[entity_id] = float(np.mean(feature_errors))
    return errors


def main(arguments: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="run adaptive-width linear graph-JEPA development"
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
    paths = run_adaptive_graph_jepa_pilot(
        captures_directory=parsed.captures_directory,
        manifests_directory=parsed.manifests_directory,
        metric_feature_spec_path=parsed.metric_feature_spec,
        log_feature_spec_path=parsed.log_feature_spec,
        split_spec_path=parsed.split_spec,
        output_directory=parsed.output,
    )
    print(f"Adaptive graph-JEPA report: {paths['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
