"""Run frozen low-rank plus residual-JEPA topology transfer development."""

import argparse
import json
import platform
from pathlib import Path
import time
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from quantis_core.action_dynamics_corpus import (
    load_action_dynamics_development_corpus,
)
from quantis_core.edge_dynamics.action_conditioned_jepa import (
    ActionConditionedJepaConfig,
    ActionConditionedJepaDynamics,
)
from quantis_core.edge_dynamics.data import (
    PreparedAttributionQueries,
    PreparedEdgeDynamicsData,
    load_edge_dynamics_cache,
    partition_worker_topology,
    prepare_worker_topology_transfer_data,
    source_artifact_manifest_sha256,
    subset_attribution_queries,
    topology_transfer_cache_address,
    validate_topology_transfer_cache,
    write_edge_dynamics_cache,
)
from quantis_core.edge_dynamics.evaluation import (
    forecast_objective,
    score_edge_model,
)
from quantis_core.edge_dynamics.jepa_evaluation import (
    action_conditioning_sanity,
    node_token_diagnostics,
)
from quantis_core.edge_dynamics.models import (
    ContractiveLowRankDynamics,
    EdgeDynamicsModel,
    LowRankConfig,
)
from quantis_core.edge_dynamics.residual_jepa import (
    FrozenBaselineResidualDynamics,
    assess_residual_jepa_development,
    latent_divergence_detection,
    write_residual_jepa_artifacts,
)


def run_residual_jepa_development(
    *,
    corpus_directory: Path,
    cache_root: Path,
    output_directory: Path,
    epochs: int = 60,
    batch_size: int = 256,
    device: str = "auto",
    seed: int = 113,
) -> Mapping[str, Any]:
    """Fit controls and write immutable open-development evidence."""

    if output_directory.exists():
        raise FileExistsError(
            f"refusing to overwrite residual JEPA: {output_directory}"
        )
    source_manifest = source_artifact_manifest_sha256(
        corpus_directory
    )
    cache_directory = cache_root / topology_transfer_cache_address(
        source_manifest
    )
    if cache_directory.exists():
        prepared = load_edge_dynamics_cache(cache_directory)
        cache_reused = True
    else:
        corpus = load_action_dynamics_development_corpus(
            corpus_directory
        )
        prepared = prepare_worker_topology_transfer_data(corpus)
        write_edge_dynamics_cache(prepared, cache_directory)
        cache_reused = False
    validate_topology_transfer_cache(prepared, corpus_directory)
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
    iid_evaluation = partitions["evaluation"].in_distribution
    transfer_evaluation = partitions["evaluation"].held_out
    iid_queries, transfer_queries = _partition_queries(
        prepared,
        held_out_normalized_value=held_out_value,
    )

    training_seconds: Dict[str, float] = {}
    baseline = ContractiveLowRankDynamics(
        LowRankConfig(rank=32)
    )
    started = time.perf_counter()
    baseline.fit(fit)
    training_seconds["raw_low_rank"] = time.perf_counter() - started
    baseline_hash_before = _artifact_hash(baseline.to_dict())

    shared_config = {
        "node_latent_dimension": 16,
        "transition_rank": 32,
        "epochs": epochs,
        "batch_size": batch_size,
        "context_reconstruction_weight": 0.0,
        "variance_weight": 0.01,
        "covariance_weight": 0.005,
        "zero_initialize_decoder": True,
        "device": device,
        "seed": seed,
    }
    supervised_config = ActionConditionedJepaConfig(
        **shared_config,
        mask_time_fraction=0.0,
        mask_entity_fraction=0.0,
        objective="supervised",
    )
    jepa_config = ActionConditionedJepaConfig(
        **shared_config,
        mask_time_fraction=0.3,
        mask_entity_fraction=0.25,
        latent_prediction_weight=0.2,
        reconstruction_weight=1.0,
        objective="jepa",
    )
    supervised = FrozenBaselineResidualDynamics(
        baseline=baseline,
        correction=ActionConditionedJepaDynamics(supervised_config),
    )
    jepa = FrozenBaselineResidualDynamics(
        baseline=baseline,
        correction=ActionConditionedJepaDynamics(jepa_config),
    )
    _timed_fit(
        supervised,
        fit,
        training_seconds,
        "supervised_residual_correction",
    )
    _timed_fit(
        jepa,
        fit,
        training_seconds,
        "jepa_residual_correction",
    )
    baseline_hash_after = _artifact_hash(baseline.to_dict())
    if baseline_hash_before != baseline_hash_after:
        raise RuntimeError("shared frozen baseline changed during training")
    supervised.select_correction_gain(selection)
    jepa.select_correction_gain(selection)

    models: Dict[str, EdgeDynamicsModel] = {
        "raw_low_rank": baseline,
        "supervised_residual_correction": supervised,
        "jepa_residual_correction": jepa,
    }
    selection_scores = {
        name: dict(forecast_objective(model, selection))
        for name, model in models.items()
    }
    iid_scores = {
        name: score_edge_model(
            model, iid_evaluation, iid_queries
        ).to_dict()
        for name, model in models.items()
    }
    transfer_scores = {
        name: score_edge_model(
            model, transfer_evaluation, transfer_queries
        ).to_dict()
        for name, model in models.items()
    }
    diagnostics = node_token_diagnostics(
        jepa.correction, transfer_evaluation
    )
    action_sanity = action_conditioning_sanity(
        jepa, transfer_evaluation, seed=seed + 312
    )
    iid_detection = latent_divergence_detection(
        model=jepa,
        calibration=calibration,
        evaluation=iid_evaluation,
    )
    transfer_detection = latent_divergence_detection(
        model=jepa,
        calibration=calibration,
        evaluation=transfer_evaluation,
    )
    assessment = assess_residual_jepa_development(
        baseline_transfer=transfer_scores["raw_low_rank"],
        supervised_transfer=transfer_scores[
            "supervised_residual_correction"
        ],
        jepa_transfer=transfer_scores["jepa_residual_correction"],
        selected_gain=jepa.selected_gain,
        action_sanity=action_sanity,
        latent_detection=transfer_detection,
        seed_robustness={
            "seed_count": 1,
            "required_seed_count": 3,
            "passed": False,
        },
    )
    report: Dict[str, Any] = {
        "schema_version": 1,
        "kind": "residual_jepa_correction_development_result_v1",
        "evidence_boundary": (
            "open development only; not sealed confirmation or a "
            "world-model claim"
        ),
        "source_corpus_sha256": prepared.source_corpus_sha256,
        "source_artifact_manifest_sha256": (
            prepared.source_artifact_manifest_sha256
        ),
        "preprocessing_cache_address": cache_directory.name,
        "preprocessing_protocol": prepared.preprocessing_protocol,
        "preprocessing_cache_reused": cache_reused,
        "frozen_baseline": {
            "sha256_before_correction_training": baseline_hash_before,
            "sha256_after_correction_training": baseline_hash_after,
            "unchanged": baseline_hash_before == baseline_hash_after,
        },
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
                set(iid_evaluation.matched_pair_ids)
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
                iid_evaluation.histories
            ),
            "transfer_evaluation": len(
                transfer_evaluation.histories
            ),
        },
        "model_configs": {
            "raw_low_rank": LowRankConfig(rank=32).__dict__,
            "supervised_residual_correction": (
                supervised_config.to_dict()
            ),
            "jepa_residual_correction": jepa_config.to_dict(),
        },
        "training_seconds": training_seconds,
        "training_runtime": _training_runtime(
            jepa=jepa.correction,
            supervised=supervised.correction,
        ),
        "initial_correction_maximum_absolute_prediction": {
            "supervised": (
                supervised.correction
                .initial_maximum_absolute_prediction
            ),
            "jepa": (
                jepa.correction.initial_maximum_absolute_prediction
            ),
        },
        "selection_scores": selection_scores,
        "selection_gain_curves": {
            "supervised": [
                dict(row) for row in supervised.selection_curve
            ],
            "jepa": [dict(row) for row in jepa.selection_curve],
        },
        "selected_correction_gains": {
            "supervised": supervised.selected_gain,
            "jepa": jepa.selected_gain,
        },
        "in_distribution_scores": iid_scores,
        "transfer_scores": transfer_scores,
        "node_token_diagnostics": diagnostics,
        "action_conditioning_sanity": action_sanity,
        "latent_divergence_detection": {
            "in_distribution": iid_detection,
            "topology_transfer": transfer_detection,
        },
        "assessment": assessment,
        "limitations": [
            "the source corpus and evaluation roles were already open",
            "only one worker topology is held out",
            "the action library contains randomized known interventions",
            "one deterministic training seed is a tracer, not robustness",
            "fresh sealed matched pairs are required for confirmation",
        ],
    }
    write_residual_jepa_artifacts(
        output_directory=output_directory,
        report=report,
        model_artifacts={
            name: model.to_dict()
            for name, model in models.items()
        },
    )
    return report


def _timed_fit(
    model: FrozenBaselineResidualDynamics,
    fit: Any,
    timings: Dict[str, float],
    name: str,
) -> None:
    started = time.perf_counter()
    model.fit(fit)
    timings[name] = time.perf_counter() - started


def _artifact_hash(artifact: Mapping[str, Any]) -> str:
    import hashlib

    payload = json.dumps(
        artifact,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _training_runtime(
    *,
    jepa: ActionConditionedJepaDynamics,
    supervised: ActionConditionedJepaDynamics,
) -> Mapping[str, Any]:
    import torch

    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch_version": str(torch.__version__),
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "jepa_requested_device": jepa.config.device,
        "jepa_resolved_device": jepa.device,
        "supervised_requested_device": supervised.config.device,
        "supervised_resolved_device": supervised.device,
    }


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
    values = prepared.attribution_queries.future_controls[:, 0, position]
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
            "residual-jepa-correction-development-v1"
        ),
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps"),
        default="mps",
    )
    parser.add_argument("--seed", type=int, default=113)
    parsed = parser.parse_args(arguments)
    result = run_residual_jepa_development(
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
                "selected_correction_gains": result[
                    "selected_correction_gains"
                ],
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
