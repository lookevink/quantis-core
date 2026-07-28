"""Run the sequential edge-dynamics development-v1 tournament."""

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from quantis_core.action_dynamics_corpus import (
    load_action_dynamics_development_corpus,
)
from quantis_core.edge_dynamics.data import (
    PreparedEdgeDynamicsData,
    load_edge_dynamics_cache,
    prepare_edge_dynamics_data,
    source_artifact_manifest_sha256,
    validate_edge_cache_source,
    write_edge_dynamics_cache,
)
from quantis_core.edge_dynamics.evaluation import (
    audit_streaming_log_templates,
    benchmark_event_sketch,
    conformal_sequential_detection,
    forecast_objective,
    persistence_scores,
    score_edge_model,
    sketch_event_predictor_effect,
    write_edge_experiment_artifacts,
)
from quantis_core.edge_dynamics.models import (
    BoundedGraphResidualDynamics,
    ContractiveLowRankDynamics,
    DenseVarxAdapter,
    EchoStateActionDynamics,
    EchoStateConfig,
    EdgeDynamicsModel,
    GraphResidualConfig,
    LowRankConfig,
    MaskedInputDynamics,
)
from quantis_core.edge_dynamics.temporal_convolution import (
    DirectTemporalConvDynamics,
    TemporalConvConfig,
)


ModelFactory = Callable[[], EdgeDynamicsModel]


def run_edge_experiments(
    *,
    corpus_directory: Path,
    cache_directory: Path,
    output_directory: Path,
) -> Mapping[str, Any]:
    """Preprocess once, run every candidate, and write bounded evidence."""

    if output_directory.exists():
        raise FileExistsError(
            f"refusing to overwrite edge results: {output_directory}"
        )
    source_manifest_sha256 = source_artifact_manifest_sha256(
        corpus_directory
    )
    addressed_cache = cache_directory / source_manifest_sha256
    if addressed_cache.exists():
        prepared = load_edge_dynamics_cache(addressed_cache)
        validate_edge_cache_source(prepared, corpus_directory)
        cache_reused = True
    else:
        corpus = load_action_dynamics_development_corpus(
            corpus_directory
        )
        prepared = prepare_edge_dynamics_data(corpus)
        validate_edge_cache_source(prepared, corpus_directory)
        write_edge_dynamics_cache(prepared, addressed_cache)
        cache_reused = False

    fit = prepared.windows["fit"]
    selection = prepared.windows["selection"]
    evaluation = prepared.windows["evaluation"]
    model_artifacts: Dict[str, Mapping[str, Any]] = {}
    selected_models: Dict[str, EdgeDynamicsModel] = {}
    selection_scores: Dict[str, Mapping[str, Any]] = {}
    evaluation_scores: Dict[str, Mapping[str, Any]] = {}
    factories: Dict[str, ModelFactory] = {}

    dense = DenseVarxAdapter().fit(fit)
    _record_model(
        name="dense_varx",
        model=dense,
        factory=lambda: DenseVarxAdapter(),
        prepared=prepared,
        selection_scores=selection_scores,
        evaluation_scores=evaluation_scores,
        selected_models=selected_models,
        factories=factories,
        model_artifacts=model_artifacts,
    )

    echo_model, echo_config, echo_candidates = _select_echo_state(
        fit, selection
    )
    _record_model(
        name="echo_state",
        model=echo_model,
        factory=lambda: EchoStateActionDynamics(echo_config),
        prepared=prepared,
        selection_scores=selection_scores,
        evaluation_scores=evaluation_scores,
        selected_models=selected_models,
        factories=factories,
        model_artifacts=model_artifacts,
    )

    tcn_config = TemporalConvConfig()
    tcn = DirectTemporalConvDynamics(tcn_config).fit(fit, selection)
    _record_model(
        name="temporal_convolution",
        model=tcn,
        factory=lambda: DirectTemporalConvDynamics(tcn_config),
        prepared=prepared,
        selection_scores=selection_scores,
        evaluation_scores=evaluation_scores,
        selected_models=selected_models,
        factories=factories,
        model_artifacts=model_artifacts,
    )

    low_rank_model, low_rank_config, low_rank_candidates = (
        _select_low_rank(fit, selection)
    )
    _record_model(
        name="contractive_low_rank",
        model=low_rank_model,
        factory=lambda: ContractiveLowRankDynamics(low_rank_config),
        prepared=prepared,
        selection_scores=selection_scores,
        evaluation_scores=evaluation_scores,
        selected_models=selected_models,
        factories=factories,
        model_artifacts=model_artifacts,
    )

    graph_model, graph_config, graph_candidates = _select_graph_residual(
        fit, selection, low_rank_config
    )
    _record_model(
        name="bounded_graph_residual",
        model=graph_model,
        factory=lambda: BoundedGraphResidualDynamics(graph_config),
        prepared=prepared,
        selection_scores=selection_scores,
        evaluation_scores=evaluation_scores,
        selected_models=selected_models,
        factories=factories,
        model_artifacts=model_artifacts,
    )

    candidate_names = (
        "echo_state",
        "temporal_convolution",
        "contractive_low_rank",
        "bounded_graph_residual",
    )
    selected_name = min(
        candidate_names,
        key=lambda name: float(
            selection_scores[name][
                "normalized_mse_action_overlap"
            ]
        ),
    )
    selected_model = selected_models[selected_name]

    event_positions = tuple(
        index
        for index, feature_name in enumerate(
            fit.state_feature_names
        )
        if feature_name
        in {
            "log_event_count",
            "log_error_count",
            "trace_span_count",
            "trace_error_count",
        }
    )
    metric_positions = tuple(
        index
        for index in range(len(fit.state_feature_names))
        if index not in event_positions
    )
    masked_model = MaskedInputDynamics(
        factories[selected_name](), event_positions
    ).fit(fit)
    structured_feature_ablation = {
        "selected_model": selected_name,
        "metrics_plus_events": dict(
            forecast_objective(
                selected_model,
                evaluation,
                state_feature_positions=metric_positions,
            )
        ),
        "metrics_only_inputs": dict(
            forecast_objective(
                masked_model,
                evaluation,
                state_feature_positions=metric_positions,
            )
        ),
        "event_feature_names": [
            fit.state_feature_names[index]
            for index in event_positions
        ],
    }
    model_artifacts[
        f"{selected_name}_metrics_only_inputs"
    ] = masked_model.to_dict()

    detection = conformal_sequential_detection(
        model=selected_model,
        calibration=prepared.windows["calibration"],
        evaluation=evaluation,
    )
    log_templates = audit_streaming_log_templates(corpus_directory)
    sketch = {
        **benchmark_event_sketch(
            evaluation, prepared.compiler_artifact
        ),
        "predictor_effect": sketch_event_predictor_effect(
            model=selected_model,
            windows=evaluation,
            compiler_artifact=prepared.compiler_artifact,
        ),
    }
    report: Dict[str, Any] = {
        "schema_version": 1,
        "kind": "edge_dynamics_development_v1_result",
        "evidence_boundary": (
            "open development only; not sealed confirmation or a "
            "world-model claim"
        ),
        "source_corpus_sha256": prepared.source_corpus_sha256,
        "source_artifact_manifest_sha256": (
            prepared.source_artifact_manifest_sha256
        ),
        "preprocessing_cache_address": addressed_cache.name,
        "cache_reused": cache_reused,
        "pair_roles": prepared.roles.to_dict(),
        "window_counts": {
            role: len(windows.histories)
            for role, windows in prepared.windows.items()
        },
        "candidate_selection": {
            "echo_state": echo_candidates,
            "contractive_low_rank": low_rank_candidates,
            "bounded_graph_residual": graph_candidates,
        },
        "selection_scores": selection_scores,
        "selected_model": selected_name,
        "evaluation_scores": evaluation_scores,
        "persistence": dict(persistence_scores(evaluation)),
        "structured_feature_ablation": structured_feature_ablation,
        "conformal_sequential_detection": detection,
        "streaming_log_templates": log_templates,
        "event_sketch": sketch,
        "limitations": [
            "the evaluation split has already informed redesign",
            "the action library and stack topology are closed and fixed",
            "the log corpus contains only three application messages",
            "a fresh sealed corpus is required for confirmation",
        ],
    }
    write_edge_experiment_artifacts(
        output_directory=output_directory,
        report=report,
        model_artifacts=model_artifacts,
    )
    return report


def _record_model(
    *,
    name: str,
    model: EdgeDynamicsModel,
    factory: ModelFactory,
    prepared: PreparedEdgeDynamicsData,
    selection_scores: Dict[str, Mapping[str, Any]],
    evaluation_scores: Dict[str, Mapping[str, Any]],
    selected_models: Dict[str, EdgeDynamicsModel],
    factories: Dict[str, ModelFactory],
    model_artifacts: Dict[str, Mapping[str, Any]],
) -> None:
    selection_scores[name] = dict(
        forecast_objective(model, prepared.windows["selection"])
    )
    evaluation_scores[name] = score_edge_model(
        model,
        prepared.windows["evaluation"],
        prepared.attribution_queries,
    ).to_dict()
    selected_models[name] = model
    factories[name] = factory
    model_artifacts[name] = model.to_dict()


def _select_echo_state(
    fit: Any, selection: Any
) -> Tuple[
    EchoStateActionDynamics,
    EchoStateConfig,
    Sequence[Mapping[str, Any]],
]:
    candidates = []
    selected_model: Optional[EchoStateActionDynamics] = None
    selected_config: Optional[EchoStateConfig] = None
    selected_score = float("inf")
    for reservoir_size in (16, 32):
        config = EchoStateConfig(reservoir_size=reservoir_size)
        model = EchoStateActionDynamics(config).fit(fit)
        scores = dict(forecast_objective(model, selection))
        candidates.append(
            {
                "reservoir_size": reservoir_size,
                **scores,
            }
        )
        score = float(scores["normalized_mse_action_overlap"])
        if score < selected_score:
            selected_model = model
            selected_config = config
            selected_score = score
    if selected_model is None or selected_config is None:
        raise ValueError("echo-state selection failed")
    return selected_model, selected_config, candidates


def _select_low_rank(
    fit: Any, selection: Any
) -> Tuple[
    ContractiveLowRankDynamics,
    LowRankConfig,
    Sequence[Mapping[str, Any]],
]:
    candidates = []
    selected_model: Optional[ContractiveLowRankDynamics] = None
    selected_config: Optional[LowRankConfig] = None
    selected_score = float("inf")
    for rank in (8, 16, 24, 32):
        config = LowRankConfig(rank=rank)
        model = ContractiveLowRankDynamics(config).fit(fit)
        scores = dict(forecast_objective(model, selection))
        candidates.append(
            {
                "rank": rank,
                "spectral_radius": model.spectral_radius,
                **scores,
            }
        )
        score = float(scores["normalized_mse_action_overlap"])
        if score < selected_score:
            selected_model = model
            selected_config = config
            selected_score = score
    if selected_model is None or selected_config is None:
        raise ValueError("low-rank selection failed")
    return selected_model, selected_config, candidates


def _select_graph_residual(
    fit: Any,
    selection: Any,
    global_config: LowRankConfig,
) -> Tuple[
    BoundedGraphResidualDynamics,
    GraphResidualConfig,
    Sequence[Mapping[str, Any]],
]:
    candidates = []
    selected_model: Optional[BoundedGraphResidualDynamics] = None
    selected_config: Optional[GraphResidualConfig] = None
    selected_score = float("inf")
    for residual_gain in (0.05, 0.1, 0.2):
        config = GraphResidualConfig(
            global_config=global_config,
            residual_gain=residual_gain,
        )
        model = BoundedGraphResidualDynamics(config).fit(fit)
        scores = dict(forecast_objective(model, selection))
        candidates.append(
            {
                "residual_gain": residual_gain,
                **scores,
            }
        )
        score = float(scores["normalized_mse_action_overlap"])
        if score < selected_score:
            selected_model = model
            selected_config = config
            selected_score = score
    if selected_model is None or selected_config is None:
        raise ValueError("graph residual selection failed")
    return selected_model, selected_config, candidates


def main(arguments: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("artifacts/action-dynamics/development-v1"),
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(
            "artifacts/action-dynamics/edge-preprocessing-v1"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/action-dynamics/edge-development-v1"
        ),
    )
    parsed = parser.parse_args(arguments)
    result = run_edge_experiments(
        corpus_directory=parsed.corpus,
        cache_directory=parsed.cache,
        output_directory=parsed.output,
    )
    summary = {
        "selected_model": result["selected_model"],
        "evaluation_scores": result["evaluation_scores"],
        "persistence": result["persistence"],
        "structured_feature_ablation": result[
            "structured_feature_ablation"
        ],
        "conformal_sequential_detection": {
            key: value
            for key, value in dict(
                result["conformal_sequential_detection"]
            ).items()
            if key != "trajectory_rows"
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
