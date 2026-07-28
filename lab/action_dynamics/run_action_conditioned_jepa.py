"""Run the open action-conditioned JEPA + low-rank transfer experiment."""

import argparse
import json
from pathlib import Path
import time
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from quantis_core.edge_dynamics.action_conditioned_jepa import (
    ActionConditionedJepaConfig,
    ActionConditionedJepaDynamics,
)
from quantis_core.edge_dynamics.data import (
    PreparedAttributionQueries,
    PreparedEdgeDynamicsData,
    load_edge_dynamics_cache,
    partition_worker_topology,
    source_artifact_manifest_sha256,
    subset_attribution_queries,
    validate_edge_cache_source,
)
from quantis_core.edge_dynamics.evaluation import (
    conformal_sequential_detection,
    forecast_objective,
    score_edge_model,
)
from quantis_core.edge_dynamics.jepa_evaluation import (
    action_conditioning_sanity,
    assess_action_conditioned_jepa_development,
    node_token_diagnostics,
    write_action_conditioned_jepa_artifacts,
)
from quantis_core.edge_dynamics.models import (
    ContractiveLowRankDynamics,
    EdgeDynamicsModel,
    LowRankConfig,
)


def run_action_conditioned_jepa_development(
    *,
    corpus_directory: Path,
    cache_root: Path,
    output_directory: Path,
    epochs: int = 60,
    batch_size: int = 256,
    device: str = "auto",
    seed: int = 89,
) -> Mapping[str, Any]:
    """Train all controls and write immutable open-development evidence."""

    if output_directory.exists():
        raise FileExistsError(
            f"refusing to overwrite JEPA results: {output_directory}"
        )
    source_manifest = source_artifact_manifest_sha256(
        corpus_directory
    )
    cache_directory = cache_root / source_manifest
    prepared = load_edge_dynamics_cache(cache_directory)
    validate_edge_cache_source(prepared, corpus_directory)
    partitions = {
        role: partition_worker_topology(windows)
        for role, windows in prepared.windows.items()
    }
    held_out_values = {
        partition.held_out_normalized_value
        for partition in partitions.values()
    }
    if len(held_out_values) != 1:
        raise ValueError("worker topology holdout drifted across roles")
    held_out_value = next(iter(held_out_values))
    fit = partitions["fit"].in_distribution
    selection = partitions["selection"].in_distribution
    calibration = partitions["calibration"].in_distribution
    in_distribution_evaluation = partitions[
        "evaluation"
    ].in_distribution
    transfer_evaluation = partitions["evaluation"].held_out
    in_distribution_queries, transfer_queries = (
        _partition_queries(
            prepared,
            held_out_normalized_value=held_out_value,
        )
    )

    models: Dict[str, EdgeDynamicsModel] = {}
    training_seconds: Dict[str, float] = {}
    baseline = ContractiveLowRankDynamics(
        LowRankConfig(rank=32)
    )
    models["raw_low_rank"] = _timed_fit(
        baseline, fit, training_seconds, "raw_low_rank"
    )
    supervised_config = ActionConditionedJepaConfig(
        node_latent_dimension=16,
        transition_rank=32,
        epochs=epochs,
        batch_size=batch_size,
        mask_time_fraction=0.0,
        mask_entity_fraction=0.0,
        objective="supervised",
        device=device,
        seed=seed,
    )
    supervised = ActionConditionedJepaDynamics(
        supervised_config
    )
    models["supervised_latent_low_rank"] = _timed_fit(
        supervised,
        fit,
        training_seconds,
        "supervised_latent_low_rank",
    )
    jepa_config = ActionConditionedJepaConfig(
        node_latent_dimension=16,
        transition_rank=32,
        epochs=epochs,
        batch_size=batch_size,
        objective="jepa",
        device=device,
        seed=seed,
    )
    jepa = ActionConditionedJepaDynamics(jepa_config)
    models["jepa_latent_low_rank"] = _timed_fit(
        jepa,
        fit,
        training_seconds,
        "jepa_latent_low_rank",
    )

    selection_scores = {
        name: dict(forecast_objective(model, selection))
        for name, model in models.items()
    }
    in_distribution_scores = {
        name: score_edge_model(
            model,
            in_distribution_evaluation,
            in_distribution_queries,
        ).to_dict()
        for name, model in models.items()
    }
    transfer_scores = {
        name: score_edge_model(
            model,
            transfer_evaluation,
            transfer_queries,
        ).to_dict()
        for name, model in models.items()
    }
    diagnostics = node_token_diagnostics(
        jepa, transfer_evaluation
    )
    action_sanity = action_conditioning_sanity(
        jepa, transfer_evaluation, seed=seed + 312
    )
    detection = conformal_sequential_detection(
        model=jepa,
        calibration=calibration,
        evaluation=transfer_evaluation,
    )
    assessment = assess_action_conditioned_jepa_development(
        baseline_transfer=transfer_scores["raw_low_rank"],
        jepa_transfer=transfer_scores["jepa_latent_low_rank"],
        token_diagnostics=diagnostics,
        action_sanity=action_sanity,
        detection=detection,
    )
    report: Dict[str, Any] = {
        "schema_version": 1,
        "kind": (
            "action_conditioned_jepa_low_rank_development_result_v1"
        ),
        "evidence_boundary": (
            "open development only; not sealed confirmation or a "
            "world-model claim"
        ),
        "source_corpus_sha256": prepared.source_corpus_sha256,
        "source_artifact_manifest_sha256": (
            prepared.source_artifact_manifest_sha256
        ),
        "preprocessing_cache_address": cache_directory.name,
        "held_out_topology": {
            "control_feature": "worker_replicas",
            "normalized_value": held_out_value,
            "fit_pair_count": len(set(fit.matched_pair_ids)),
            "selection_pair_count": len(
                set(selection.matched_pair_ids)
            ),
            "calibration_pair_count": len(
                set(calibration.matched_pair_ids)
            ),
            "in_distribution_evaluation_pair_count": len(
                set(in_distribution_evaluation.matched_pair_ids)
            ),
            "transfer_evaluation_pair_count": len(
                set(transfer_evaluation.matched_pair_ids)
            ),
        },
        "window_counts": {
            "fit": len(fit.histories),
            "selection": len(selection.histories),
            "calibration": len(calibration.histories),
            "in_distribution_evaluation": len(
                in_distribution_evaluation.histories
            ),
            "transfer_evaluation": len(
                transfer_evaluation.histories
            ),
        },
        "model_configs": {
            "raw_low_rank": LowRankConfig(rank=32).__dict__,
            "supervised_latent_low_rank": (
                supervised_config.to_dict()
            ),
            "jepa_latent_low_rank": jepa_config.to_dict(),
        },
        "training_seconds": training_seconds,
        "selection_scores": selection_scores,
        "in_distribution_scores": in_distribution_scores,
        "transfer_scores": transfer_scores,
        "node_token_diagnostics": diagnostics,
        "action_conditioning_sanity": action_sanity,
        "conformal_sequential_detection": detection,
        "assessment": assessment,
        "limitations": [
            "the source corpus and its evaluation data were already open",
            "only one worker topology is held out",
            "the application log vocabulary contains three templates",
            "one deterministic training seed is a tracer, not robustness",
            "fresh sealed matched pairs are required for confirmation",
        ],
    }
    write_action_conditioned_jepa_artifacts(
        output_directory=output_directory,
        report=report,
        model_artifacts={
            name: model.to_dict()
            for name, model in models.items()
        },
    )
    return report


def _timed_fit(
    model: EdgeDynamicsModel,
    fit: Any,
    timings: Dict[str, float],
    name: str,
) -> EdgeDynamicsModel:
    started = time.perf_counter()
    fitted = model.fit(fit)
    timings[name] = time.perf_counter() - started
    return fitted


def _partition_queries(
    prepared: PreparedEdgeDynamicsData,
    *,
    held_out_normalized_value: float,
) -> tuple[PreparedAttributionQueries, PreparedAttributionQueries]:
    control_names = prepared.windows["fit"].control_feature_names
    try:
        position = control_names.index("worker_replicas")
    except ValueError as error:
        raise ValueError(
            "attribution query split requires worker_replicas"
        ) from error
    values = prepared.attribution_queries.future_controls[
        :, 0, position
    ]
    if not np.allclose(
        prepared.attribution_queries.future_controls[..., position],
        values[:, None],
    ):
        raise ValueError(
            "worker topology must be constant within attribution query"
        )
    transfer = np.isclose(values, held_out_normalized_value)
    return (
        subset_attribution_queries(
            prepared.attribution_queries, ~transfer
        ),
        subset_attribution_queries(
            prepared.attribution_queries, transfer
        ),
    )


def main(arguments: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("artifacts/action-dynamics/development-v1"),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(
            "artifacts/action-dynamics/edge-preprocessing-v1"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/action-dynamics/"
            "action-conditioned-jepa-low-rank-development-v1"
        ),
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps"),
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=89)
    parsed = parser.parse_args(arguments)
    result = run_action_conditioned_jepa_development(
        corpus_directory=parsed.corpus,
        cache_root=parsed.cache_root,
        output_directory=parsed.output,
        epochs=parsed.epochs,
        batch_size=parsed.batch_size,
        device=parsed.device,
        seed=parsed.seed,
    )
    print(
        json.dumps(
            {
                "assessment": result["assessment"],
                "transfer_scores": result["transfer_scores"],
                "training_seconds": result["training_seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
