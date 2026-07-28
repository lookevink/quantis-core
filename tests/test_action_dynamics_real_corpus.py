import math
from dataclasses import replace

import numpy as np
import pytest

from quantis_core.action_conditioned_dynamics import (
    ActionConditionedCaseManifest,
    ActionConditionedRun,
    InterventionAction,
)
from quantis_core.action_dynamics_real_corpus import (
    AttributionCandidatePlan,
    AttributionQuery,
    FROZEN_ACTION_CANDIDATE_GRID,
    RealCorpusRun,
    RealCorpusStudyConfig,
    assess_development_gates,
    build_development_validation_queries,
    train_and_evaluate_real_corpus,
    write_real_corpus_study_artifacts,
)
from quantis_core.action_dynamics_synthetic import (
    causal_chain_graph,
    synthetic_action_runs,
)
from quantis_core.graph_telemetry import (
    DeclaredTelemetryGraph,
    GraphEntity,
)


def test_real_corpus_study_rejects_pilot_evidence() -> None:
    training = synthetic_action_runs(
        4, split="training", seed=100
    )
    validation = synthetic_action_runs(
        2, split="validation", seed=900
    )
    runs = tuple(
        RealCorpusRun(run, corpus_role="development")
        for run in training
    ) + tuple(
        RealCorpusRun(run, corpus_role="instrumentation_pilot")
        for run in validation
    )

    with pytest.raises(ValueError, match="development corpus only"):
        train_and_evaluate_real_corpus(
            runs=runs,
            graph=causal_chain_graph(),
            queries=(),
            config=RealCorpusStudyConfig(),
        )


def test_development_config_and_candidate_queries_are_frozen() -> None:
    with pytest.raises(ValueError, match="frozen development-v1"):
        RealCorpusStudyConfig(context_length=19)
    validation = synthetic_action_runs(
        60, split="validation", seed=800
    )
    allowed_magnitude = {
        "worker_pause": 1.0,
        "postgres_lock": 1.0,
        "redis_enqueue_delay": 20.0,
        "redis_dequeue_delay": 20.0,
        "api_rejection": 0.25,
    }
    allowed_target = {
        "worker_pause": "worker_pool",
        "postgres_lock": "worker_writes_postgresql",
        "redis_enqueue_delay": "api_enqueues_queue",
        "redis_dequeue_delay": "queue_dequeues_to_worker",
        "api_rejection": "api",
    }
    graph = _development_graph()
    validation = tuple(
        (
            replace(
                run,
                graph=graph,
                observations=np.pad(
                    run.observations,
                    ((0, 0), (0, 2), (0, 0)),
                ),
                manifest=replace(
                    run.manifest,
                    actions=(
                        replace(
                            run.manifest.actions[0],
                            target_entity=allowed_target[
                                run.manifest.actions[0].action_kind
                            ],
                            stop_index=(
                                run.manifest.actions[0].start_index
                                + (
                                    20
                                    if run.manifest.actions[
                                        0
                                    ].action_kind
                                    == "api_rejection"
                                    else 8
                                )
                            ),
                            magnitude=allowed_magnitude[
                                run.manifest.actions[0].action_kind
                            ],
                        ),
                    ),
                ),
            )
            if run.manifest.actions
            else replace(
                run,
                graph=graph,
                observations=np.pad(
                    run.observations,
                    ((0, 0), (0, 2), (0, 0)),
                ),
            )
        )
        for run in validation
    )

    queries = build_development_validation_queries(
        validation, graph
    )

    assert len(queries) == 60
    assert len(FROZEN_ACTION_CANDIDATE_GRID) == 107
    assert len(queries[0].candidates) == 108
    assert queries[0].query_id.endswith(":treatment")
    assert queries[1].query_id.endswith(":control")
    assert queries[0].candidates[0].candidate_id == "no_action"
    assert queries[0].transition_index == (
        validation[0].manifest.actions[0].start_index - 1
    )
    assert queries[0].expected_variant_candidate_id is not None
    assert queries[1].expected_variant_candidate_id is None
    assert queries == build_development_validation_queries(
        tuple(reversed(validation)), graph
    )


def test_development_assessment_separates_the_graph_claim() -> None:
    graph_blocked = assess_development_gates(
        action_overlap_mse=0.8,
        action_agnostic_overlap_mse=1.0,
        persistence_overlap_mse=1.0,
        graph_downstream_mse=0.96,
        dense_downstream_mse=1.0,
        action_location_hit_at_1=0.8,
        no_action_specificity=0.95,
    )

    assert graph_blocked.action_conditioning_supported is True
    assert graph_blocked.attribution_supported is True
    assert graph_blocked.graph_topology_supported is False
    assert graph_blocked.graph_claim_blocked is True
    assert graph_blocked.decision == (
        "publish_action_conditioning_without_graph_claim"
    )

    all_pass = assess_development_gates(
        action_overlap_mse=0.8,
        action_agnostic_overlap_mse=1.0,
        persistence_overlap_mse=1.0,
        graph_downstream_mse=0.94,
        dense_downstream_mse=1.0,
        action_location_hit_at_1=0.8,
        no_action_specificity=0.95,
    )
    assert all_pass.decision == "advance_to_sealed_confirmation"


def test_study_trains_four_models_and_scores_validation_queries(
    tmp_path,
) -> None:
    training = _late_action_runs(30, split="training", seed=200)
    validation = _late_action_runs(4, split="validation", seed=800)
    treatment = validation[0]
    true_action = treatment.manifest.actions[0]
    wrong_action = InterventionAction(
        action_id="wrong-location",
        action_kind=true_action.action_kind,
        target_entity="sink",
        start_index=true_action.start_index,
        stop_index=true_action.stop_index,
        magnitude=true_action.magnitude,
    )
    candidates = (
        AttributionCandidatePlan(
            candidate_id="true",
            actions=(true_action,),
        ),
        AttributionCandidatePlan(
            candidate_id="wrong-location",
            actions=(wrong_action,),
        ),
        AttributionCandidatePlan(candidate_id="none", actions=()),
    )
    queries = (
        AttributionQuery(
            query_id="action-source",
            validation_case_id=treatment.manifest.case_id,
            transition_index=true_action.start_index - 1,
            candidates=candidates,
            no_action_candidate_id="none",
            expected_action_kind=true_action.action_kind,
            expected_target_entity=true_action.target_entity,
            expected_variant_candidate_id="true",
        ),
        AttributionQuery(
            query_id="nominal-source",
            validation_case_id=validation[1].manifest.case_id,
            transition_index=true_action.start_index - 1,
            candidates=candidates,
            no_action_candidate_id="none",
            expected_action_kind=None,
            expected_target_entity=None,
        ),
    )
    admitted = tuple(
        RealCorpusRun(run, corpus_role="development")
        for run in training + validation
    )

    result = train_and_evaluate_real_corpus(
        runs=admitted,
        graph=causal_chain_graph(),
        queries=queries,
        config=RealCorpusStudyConfig(),
    )

    assert set(result.forecast_metrics) == {
        "action_conditioned_graph_varx",
        "action_agnostic_graph_varx",
        "action_conditioned_dense_varx",
        "persistence",
    }
    assert all(
        math.isfinite(metric.normalized_mse_overall)
        and metric.normalized_mse_overall >= 0.0
        and math.isfinite(metric.normalized_mse_action_overlap)
        and metric.normalized_mse_action_overlap >= 0.0
        for metric in result.forecast_metrics.values()
    )
    assert result.attribution.action_family_hit_at_1 == 1.0
    assert result.attribution.action_location_hit_at_1 == 1.0
    assert result.attribution.no_action_specificity == 1.0
    assert result.attribution.query_count == 2
    assert result.training_run_ids == tuple(
        run.manifest.case_id for run in training
    )
    assert result.validation_run_ids == tuple(
        run.manifest.case_id for run in validation
    )
    assert set(result.model_artifacts) == {
        "action_conditioned_graph_varx",
        "action_agnostic_graph_varx",
        "action_conditioned_dense_varx",
    }
    report = result.to_dict()
    assert report["evaluation_boundary"][
        "forecast_run_ids"
    ] == list(result.validation_run_ids)
    assert report["evaluation_boundary"][
        "attribution_query_ids"
    ] == ["action-source", "nominal-source"]
    assert set(result.forecast_panels) == {
        "all_forecast_states",
        "action_overlap",
        "target_entity_intervention_effect",
        "downstream_entity_intervention_effect",
        "recovery",
    }
    assert result.assessment.to_dict()["thresholds"][
        "action_location_hit_at_1_min"
    ] == 0.70
    assert result.forecast_panels[
        "downstream_entity_intervention_effect"
    ].comparison_kind == "paired_treatment_minus_control"
    written = write_real_corpus_study_artifacts(
        result, tmp_path / "study"
    )
    assert written.manifest_path.is_file()
    assert len(written.manifest_sha256) == 64
    assert (tmp_path / "study" / "study.json").is_file()
    assert (
        tmp_path
        / "study"
        / "models"
        / "action_conditioned_graph_varx.json"
    ).is_file()
    with pytest.raises(FileExistsError):
        write_real_corpus_study_artifacts(
            result, tmp_path / "study"
        )


def _development_graph() -> DeclaredTelemetryGraph:
    entity_ids = (
        "api",
        "api_enqueues_queue",
        "checkout_queue",
        "queue_dequeues_to_worker",
        "worker_pool",
        "worker_writes_postgresql",
        "postgresql",
    )
    nodes = {"api", "checkout_queue", "worker_pool", "postgresql"}
    edges = {
        "api_enqueues_queue": ("api", "checkout_queue"),
        "queue_dequeues_to_worker": (
            "checkout_queue",
            "worker_pool",
        ),
        "worker_writes_postgresql": (
            "worker_pool",
            "postgresql",
        ),
    }
    return DeclaredTelemetryGraph(
        entities=tuple(
            (
                GraphEntity(entity_id, "node", "service")
                if entity_id in nodes
                else GraphEntity(
                    entity_id,
                    "edge",
                    "dependency",
                    edges[entity_id][0],
                    edges[entity_id][1],
                )
            )
            for entity_id in entity_ids
        ),
        bindings=(),
    )


def _late_action_runs(
    count: int, *, split: str, seed: int
) -> tuple[ActionConditionedRun, ...]:
    library = (
        ("api_rejection", "source"),
        ("redis_enqueue_delay", "source_to_middle"),
        ("worker_pause", "middle"),
        ("redis_dequeue_delay", "middle_to_sink"),
        ("postgres_lock", "sink"),
    )
    runs = []
    for pair_index in range(count // 2):
        rng = np.random.default_rng(seed + pair_index)
        controls = (
            1.0
            + 0.2
            * np.sin(
                np.arange(48, dtype=np.float64) * 0.4
                + pair_index
            )
        ).reshape(48, 1)
        noise = rng.normal(0.0, 0.005, size=(47, 5))
        kind, target = library[pair_index % len(library)]
        target_position = causal_chain_graph().entity_ids.index(target)
        for with_action in (True, False):
            states = np.zeros((48, 5, 1), dtype=np.float64)
            states[0, :, 0] = rng.normal(0.0, 0.03, size=5)
            for point in range(47):
                current = states[point, :, 0]
                following = states[point + 1, :, 0]
                following[0] = (
                    0.55 * current[0] + 0.12 * controls[point, 0]
                )
                following[1] = (
                    0.50 * current[1] + 0.45 * current[0]
                )
                following[2] = (
                    0.55 * current[2] + 0.40 * current[1]
                )
                following[3] = (
                    0.50 * current[3] + 0.45 * current[2]
                )
                following[4] = (
                    0.55 * current[4] + 0.40 * current[3]
                )
                if with_action and 22 <= point < 30:
                    following[target_position] += 0.7
                following += noise[point]
            pair_id = f"{split}-late-{pair_index:03d}"
            case_id = (
                f"{pair_id}-action"
                if with_action
                else f"{pair_id}-control"
            )
            actions = (
                (
                    InterventionAction(
                        action_id=f"{case_id}-action",
                        action_kind=kind,
                        target_entity=target,
                        start_index=22,
                        stop_index=30,
                        magnitude=0.5,
                    ),
                )
                if with_action
                else ()
            )
            runs.append(
                ActionConditionedRun(
                    manifest=ActionConditionedCaseManifest(
                        case_id=case_id,
                        matched_pair_id=pair_id,
                        split=split,
                        point_count=48,
                        logical_window_period_nano=1,
                        topology_id="chain",
                        worker_replicas=1 + pair_index % 3,
                        workload_seed=seed + pair_index,
                        intervention_seed=seed * 10 + pair_index,
                        actions=actions,
                    ),
                    graph=causal_chain_graph(),
                    observations=states,
                    controls=controls,
                    state_feature_names=("pressure",),
                    control_feature_names=("request_demand",),
                )
            )
    return tuple(runs)
