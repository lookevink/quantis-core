import numpy as np
import pytest

from quantis_core.action_conditioned_dynamics import ActionConditionedWindows
from quantis_core.edge_dynamics.retrieval_jepa import (
    EpisodeRetrievalConfig,
    EpisodeRetrievalRepresentation,
    OwnedStateRidgeProbe,
    PcaRetrievalRepresentation,
    RawTelemetryRetrievalRepresentation,
    assess_retrieval_jepa,
    compile_retrieval_episodes,
    exact_retrieval,
    fit_empirical_abstention,
)
from quantis_core.graph_telemetry import (
    DeclaredTelemetryGraph,
    GraphEntity,
    TelemetryBinding,
)


def test_retrieval_episode_compiler_keeps_queries_causal_and_pairs_atomic() -> None:
    windows = _tiny_retrieval_windows(pair_count=4)
    histories_before = windows.histories.copy()
    future_before = windows.future_states.copy()

    episodes = compile_retrieval_episodes(windows)

    assert episodes.contexts.shape == (8, 20, 7, 2)
    assert episodes.evidence.shape == (8, 10, 7, 2)
    assert episodes.pair_ids == tuple(
        value
        for pair in range(4)
        for value in (f"pair{pair}", f"pair{pair}")
    )
    assert episodes.is_treatment.tolist() == [True, False] * 4
    assert episodes.action_and_target_labels == tuple(
        value
        for _ in range(4)
        for value in ("worker_pause@e4", "no_action")
    )
    assert episodes.transition_indices.tolist() == [1, 1] * 4
    assert len(set(episodes.episode_ids)) == 8
    assert all("#transition=1" in value for value in episodes.evidence_refs)
    np.testing.assert_array_equal(episodes.contexts, histories_before[[1, 4, 7, 10, 13, 16, 19, 22]])
    np.testing.assert_array_equal(
        episodes.evidence, future_before[[1, 4, 7, 10, 13, 16, 19, 22]]
    )
    np.testing.assert_array_equal(windows.histories, histories_before)
    np.testing.assert_array_equal(windows.future_states, future_before)
    ambiguous_actions = windows.future_actions.copy()
    ambiguous_actions[1, 0, 3, 1] = 1.0
    ambiguous_actions[1, 0, 3, 2] = 1.0
    ambiguous_actions[1, 0, 3, 8] = 1.0
    ambiguous_actions[1, 0, 3, 11] = 0.5
    ambiguous = ActionConditionedWindows(
        histories=windows.histories,
        future_states=windows.future_states,
        future_controls=windows.future_controls,
        future_actions=ambiguous_actions,
        trajectory_ids=windows.trajectory_ids,
        matched_pair_ids=windows.matched_pair_ids,
        transition_indices=windows.transition_indices,
        entity_names=windows.entity_names,
        state_feature_names=windows.state_feature_names,
        control_feature_names=windows.control_feature_names,
        action_feature_names=windows.action_feature_names,
        graph=windows.graph,
    )
    with pytest.raises(ValueError, match="multiple target"):
        compile_retrieval_episodes(ambiguous)


def test_episode_predictive_retriever_encodes_normalized_vectors_and_restores() -> None:
    windows = _tiny_retrieval_windows(pair_count=4)
    config = EpisodeRetrievalConfig(
        objective="episode_predictive_jepa",
        steps=1,
        expected_pair_count=4,
        seed=9019,
    )
    model = EpisodeRetrievalRepresentation(config).fit(windows)
    episodes = compile_retrieval_episodes(windows)

    query = model.encode_queries(episodes.contexts, windows.graph)
    bank = model.encode_evidence(
        episodes.contexts, episodes.evidence, windows.graph
    )
    restored = EpisodeRetrievalRepresentation.from_dict(model.to_dict())

    assert query.vectors.shape == (8, 64)
    assert bank.vectors.shape == (8, 64)
    np.testing.assert_allclose(
        np.linalg.norm(query.vectors, axis=1), 1.0, atol=1e-6
    )
    np.testing.assert_allclose(
        np.linalg.norm(bank.vectors, axis=1), 1.0, atol=1e-6
    )
    assert np.all(np.isfinite(query.vectors))
    assert model.inference_parameter_count < 500_000
    np.testing.assert_allclose(
        query.vectors,
        restored.encode_queries(episodes.contexts, windows.graph).vectors,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        bank.vectors,
        restored.encode_evidence(
            episodes.contexts, episodes.evidence, windows.graph
        ).vectors,
        atol=1e-6,
    )


def test_raw_and_pca_controls_share_the_causal_query_evidence_space() -> None:
    windows = _tiny_retrieval_windows(pair_count=4)
    episodes = compile_retrieval_episodes(windows)

    raw = RawTelemetryRetrievalRepresentation().fit(windows)
    pca = PcaRetrievalRepresentation(width=64).fit(windows)
    raw_query = raw.encode_queries(episodes.contexts, windows.graph)
    raw_evidence = raw.encode_evidence(
        episodes.contexts, episodes.evidence, windows.graph
    )
    pca_query = pca.encode_queries(episodes.contexts, windows.graph)
    pca_evidence = pca.encode_evidence(
        episodes.contexts, episodes.evidence, windows.graph
    )

    assert raw_query.vectors.shape == (8, 20)
    assert raw_evidence.vectors.shape == (8, 20)
    assert pca_query.vectors.shape == (8, 64)
    assert pca_evidence.vectors.shape == (8, 64)
    np.testing.assert_allclose(
        raw_query.vectors,
        RawTelemetryRetrievalRepresentation.from_dict(
            raw.to_dict()
        ).encode_queries(episodes.contexts, windows.graph).vectors,
    )
    np.testing.assert_allclose(
        pca_evidence.vectors,
        PcaRetrievalRepresentation.from_dict(
            pca.to_dict()
        )
        .encode_evidence(
            episodes.contexts, episodes.evidence, windows.graph
        )
        .vectors,
    )
    probe = OwnedStateRidgeProbe(ridge=1e-3).fit(
        raw_query.vectors,
        episodes.contexts,
        raw.ownership_mask,
    )
    restored_probe = OwnedStateRidgeProbe.from_dict(probe.to_dict())
    np.testing.assert_allclose(
        probe.predict(raw_query.vectors),
        restored_probe.predict(raw_query.vectors),
    )
    assert probe.target_scale.shape == (2,)
    assert probe.target_varying_mask.shape == (2,)


def test_exact_retrieval_calibrates_abstention_and_assessor_recomputes_reject() -> None:
    bank_ids = ("a0", "a1", "b0", "b1")
    bank_labels = ("a@e0", "a@e0", "b@e1", "b@e1")
    bank = np.eye(4, dtype=np.float64)
    queries = np.asarray(
        (
            (0.90, 0.40, 0.10, 0.00),
            (0.45, 0.44, 0.20, 0.10),
            (0.10, 0.00, 0.90, 0.40),
            (0.42, 0.41, 0.30, 0.20),
        ),
        dtype=np.float64,
    )
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    result = exact_retrieval(
        queries,
        bank,
        bank_episode_ids=bank_ids,
        bank_labels=bank_labels,
        k=3,
    )
    labels = ("a@e0", "no_action", "b@e1", "no_action")
    treatment = np.asarray((True, False, True, False))

    policy = fit_empirical_abstention(
        result,
        query_labels=labels,
        is_treatment=treatment,
    )

    assert policy.accept(result.class_margins).tolist() == [
        True,
        False,
        True,
        False,
    ]
    model_names = (
        "episode_predictive_jepa",
        "raw_telemetry",
        "pca_64",
        "deranged_target_jepa",
        "cpc_infonce",
        "supervised_retriever",
    )
    similarities = {
        role: {
            name: result.similarities.copy()
            for name in model_names
        }
        for role in (
            "calibration",
            "selection_iid",
            "selection_transfer",
            "evaluation_iid",
            "evaluation_transfer",
        )
    }
    assessment = assess_retrieval_jepa(
        gallery_episode_ids=bank_ids,
        gallery_labels=bank_labels,
        similarities=similarities,
        query_labels={
            role: labels
            for role in similarities
        },
        is_treatment={
            role: treatment
            for role in similarities
        },
        pair_ids={
            role: ("p0", "p0", "p1", "p1")
            for role in similarities
        },
        bank_vectors={name: bank for name in model_names},
        restored_bank_vectors={name: bank for name in model_names},
        state_truth=np.zeros((4, 2), dtype=np.float64),
        state_scale=np.ones(2, dtype=np.float64),
        state_varying_mask=np.ones(2, dtype=np.bool_),
        state_predictions={
            name: np.zeros((4, 2), dtype=np.float64)
            for name in model_names
        },
        original_query_vectors={
            role: {name: queries for name in model_names}
            for role in similarities
        },
        restored_query_vectors={
            role: {name: queries for name in model_names}
            for role in similarities
        },
        protocol_checks={
            "role_pairs_are_disjoint": True,
            "query_future_is_excluded": True,
            "action_and_identifiers_are_excluded": True,
            "bank_membership_is_equal_and_immutable": True,
            "episode_counts_match_contract": True,
        },
        edge_metrics={
            name: {
                "online_parameter_count": 1,
                "serialized_model_bytes": 1,
                "query_latency_median_ms": 1.0,
                "search_latency_median_ms": 1.0,
            }
            for name in model_names
        },
    )

    assert not assessment["passed"]
    assert not assessment["value_gates"][
        "selection_beats_best_non_supervised_by_0_05"
    ]
    assert assessment["decision"] == (
        "reject_episode_predictive_retrieval_jepa_recipe"
    )
    candidate_metrics = assessment["metrics"]["evaluation_transfer"][
        "episode_predictive_jepa"
    ]
    assert candidate_metrics["first_relevant_ranks"] == [1, 1]
    assert set(candidate_metrics["per_action"]) == {"a@e0", "b@e1"}
    assert candidate_metrics["risk_coverage_curve"][-1][
        "treatment_coverage"
    ] == 0.0


def _tiny_retrieval_windows(*, pair_count: int) -> ActionConditionedWindows:
    graph = DeclaredTelemetryGraph(
        entities=tuple(
            GraphEntity(f"e{index}", "node", "service")
            for index in range(7)
        ),
        bindings=(
            TelemetryBinding("metric.f0", "e0"),
            TelemetryBinding("metric.f1", "e4"),
        ),
    )
    action_names = (
        "no_action",
        "applicable",
        "kind:worker_pause",
        "kind:postgres_lock",
        "kind:redis_enqueue_delay",
        "kind:redis_dequeue_delay",
        "kind:api_rejection",
        "phase:start",
        "phase:active",
        "phase:stop",
        "magnitude",
        "elapsed_fraction",
        "remaining_fraction",
    )
    histories = []
    futures = []
    actions = []
    trajectory_ids = []
    pair_ids = []
    transitions = []
    for pair in range(pair_count):
        for arm in range(2):
            for transition in range(3):
                context = np.zeros((20, 7, 2), dtype=np.float64)
                context[:, 0, 0] = np.arange(20) + pair
                context[:, 4, 1] = (
                    np.arange(20) + 100 * arm + 10 * transition
                )
                evidence = np.zeros((10, 7, 2), dtype=np.float64)
                evidence[:, 0, 0] = np.arange(10) + pair
                evidence[:, 4, 1] = (
                    np.arange(10) + 100 * arm + 10 * transition
                )
                future_action = np.zeros(
                    (10, 7, len(action_names)), dtype=np.float64
                )
                if arm == 0 and transition == 1:
                    future_action[0, 4, 1] = 1.0
                    future_action[0, 4, 2] = 1.0
                    future_action[0, 4, 8] = 1.0
                    future_action[0, 4, 10] = 1.0
                    future_action[0, 4, 11] = 0.5
                    future_action[0, 4, 12] = 0.5
                histories.append(context)
                futures.append(evidence)
                actions.append(future_action)
                trajectory_ids.append(f"pair{pair}-arm{arm}")
                pair_ids.append(f"pair{pair}")
                transitions.append(transition)
    sample_count = len(histories)
    return ActionConditionedWindows(
        histories=np.asarray(histories),
        future_states=np.asarray(futures),
        future_controls=np.zeros(
            (sample_count, 10, 1), dtype=np.float64
        ),
        future_actions=np.asarray(actions),
        trajectory_ids=tuple(trajectory_ids),
        matched_pair_ids=tuple(pair_ids),
        transition_indices=np.asarray(transitions, dtype=np.int64),
        entity_names=graph.entity_ids,
        state_feature_names=("f0", "f1"),
        control_feature_names=("worker_replicas",),
        action_feature_names=action_names,
        graph=graph,
    )
