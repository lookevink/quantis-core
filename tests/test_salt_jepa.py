import numpy as np

from lab.action_dynamics.prototype_salt_jepa import (
    FROZEN_CACHE,
    run_experiment,
)
from lab.action_dynamics.prototype_salt_jepa_assessor import (
    verify_stored_assessment,
)
from quantis_core.action_conditioned_dynamics import (
    ActionConditionedWindows,
)
from quantis_core.edge_dynamics.complete_lejepa import (
    fit_owned_feature_mask,
)
from quantis_core.edge_dynamics.salt_jepa import (
    SaltJepaConfig,
    SaltMaskedTelemetry,
    SaltJepaRepresentation,
    SaltMaskSchedule,
    SaltTargetSchedule,
    assess_salt_jepa_gates,
)
from quantis_core.graph_telemetry import (
    DeclaredTelemetryGraph,
    GraphEntity,
    TelemetryBinding,
)


def test_salt_mask_schedule_is_deterministic_semantic_and_nonmutating() -> None:
    windows = _tiny_windows(pair_count=4, transition_count=5)
    ownership = fit_owned_feature_mask(windows)
    source = windows.histories[:3].copy()
    before = source.copy()
    schedule = SaltMaskSchedule(
        graph=windows.graph,
        ownership_mask=ownership,
        seed=28023,
    )

    first = schedule.batch(source, step=7)
    second = schedule.batch(source, step=7)

    np.testing.assert_array_equal(source, before)
    np.testing.assert_array_equal(first.values, second.values)
    np.testing.assert_array_equal(
        first.visible_tokens, second.visible_tokens
    )
    np.testing.assert_array_equal(
        first.block_rectangles, second.block_rectangles
    )
    np.testing.assert_array_equal(first.fill_order, second.fill_order)
    np.testing.assert_array_equal(first.target_tokens, ~first.visible_tokens)
    assert first.values.shape == source.shape
    assert first.visible_tokens.shape == source.shape[:-1]
    assert np.all(np.sum(first.target_tokens, axis=(1, 2)) == 126)
    observed_entities = np.flatnonzero(np.any(ownership, axis=1))
    assert np.all(first.visible_tokens[:, -1, observed_entities])
    assert np.all(first.values[~first.visible_tokens] == 0.0)
    assert np.all(np.isfinite(first.values))
    expected_values = np.where(
        first.visible_tokens[..., None],
        np.where(ownership[None, None], source, 0.0),
        0.0,
    )
    np.testing.assert_array_equal(first.values, expected_values)
    _assert_mask_provenance(first, windows.graph, observed_entities)


def test_salt_target_schedule_deranges_pairs_without_fixed_points() -> None:
    pair_ids = ("p0", "p1", "p2", "p3")

    aligned = SaltTargetSchedule("aligned").indices(pair_ids, step=3)
    first = SaltTargetSchedule("deranged").indices(pair_ids, step=3)
    second = SaltTargetSchedule("deranged").indices(pair_ids, step=3)

    np.testing.assert_array_equal(aligned, np.arange(4))
    np.testing.assert_array_equal(first, second)
    assert sorted(first.tolist()) == [0, 1, 2, 3]
    assert np.all(first != np.arange(4))
    with np.testing.assert_raises(ValueError):
        SaltTargetSchedule("deranged").indices(("only",), step=0)


def test_salt_representation_freezes_teacher_and_restores_public_outputs() -> None:
    import torch

    windows = _tiny_windows(pair_count=4, transition_count=5)
    config = SaltJepaConfig(
        alignment="aligned",
        teacher_steps=1,
        student_steps=1,
        expected_pair_count=4,
    )

    torch.manual_seed(9182)
    rng_before = torch.random.get_rng_state().clone()
    model = SaltJepaRepresentation(config).fit(windows)
    np.testing.assert_array_equal(
        torch.random.get_rng_state().numpy(), rng_before.numpy()
    )
    student = model.encode(windows.histories[:3], windows.graph)
    teacher = model.encode_teacher(
        windows.histories[:3], windows.graph
    )
    diagnostic = model.diagnose_masked_prediction(
        windows.histories[:3], windows.graph, step=2
    )
    restored = SaltJepaRepresentation.from_dict(model.to_dict())

    assert model.teacher_unchanged_during_student
    assert student.tokens.shape == (3, 7, 64)
    assert teacher.tokens.shape == student.tokens.shape
    assert diagnostic.predicted_tokens.shape == (3, 140, 64)
    assert diagnostic.target_tokens.shape == diagnostic.predicted_tokens.shape
    assert diagnostic.target_mask.shape == (3, 140)
    assert np.isfinite(diagnostic.l1)
    assert model.inference_parameter_count > 0
    assert model.training_only_parameter_count > 0
    np.testing.assert_allclose(
        student.tokens,
        restored.encode(windows.histories[:3], windows.graph).tokens,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        teacher.tokens,
        restored.encode_teacher(
            windows.histories[:3], windows.graph
        ).tokens,
        atol=1e-7,
    )
    restored_diagnostic = restored.diagnose_masked_prediction(
        windows.histories[:3], windows.graph, step=2
    )
    np.testing.assert_allclose(
        diagnostic.predicted_tokens,
        restored_diagnostic.predicted_tokens,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        diagnostic.target_tokens,
        restored_diagnostic.target_tokens,
        atol=1e-7,
    )
    np.testing.assert_array_equal(
        diagnostic.target_mask, restored_diagnostic.target_mask
    )


def test_salt_assessment_recomputes_failed_raw_safety() -> None:
    names = (
        "salt_jepa",
        "deranged_salt_jepa",
        "reconstructive_teacher",
        "matched_pca",
    )
    scores = {
        name: {
            role: {
                "overall_mse": (
                    2.0
                    if name == "salt_jepa"
                    and role == "transfer_evaluation"
                    else 1.0
                ),
                "action_overlap_mse": 1.0,
                "downstream_effect_mse": (
                    0.5 if name == "salt_jepa" else 1.0
                ),
            }
            for role in ("selection", "transfer_evaluation")
        }
        for name in names
    }
    state = {
        name: {
            "transfer_evaluation": {
                "aggregate_nrmse": 1.0,
                "entities": {"e0": {"nrmse": 1.0}},
            }
        }
        for name in names
    }

    assessment = assess_salt_jepa_gates(
        forecast_scores=scores,
        raw_scores={
            role: {
                "overall_mse": 1.0,
                "action_overlap_mse": 1.0,
                "downstream_effect_mse": 1.0,
            }
            for role in ("selection", "transfer_evaluation")
        },
        masked_latent_l1={
            "salt_jepa": {
                "selection": 0.5,
                "transfer_evaluation": 0.5,
            },
            "deranged_salt_jepa": {
                "selection": 1.0,
                "transfer_evaluation": 1.0,
            },
        },
        state_probes=state,
        attribution={
            name: {
                "action_and_target_hit_at_1": 1.0,
                "no_action_specificity": 1.0,
            }
            for name in names
        },
        action_sanity={
            name: {"correct_action_beats_both_fraction": 1.0}
            for name in names
        },
        restoration_max_abs={name: 0.0 for name in names},
        protocol_checks={
            "evidence_arrays_are_finite": True,
            "pair_and_trajectory_roles_are_disjoint": True,
            "public_inference_is_causal": True,
            "mask_schedule_is_valid": True,
            "selection_only_ridge_choice_recomputes": True,
            "selection_safety_status_recomputes": True,
            "capacity_metadata_recomputes": True,
            "teacher_metadata_recomputes": True,
            "causality_metadata_recomputes": True,
            "deployed_bundle_metadata_recomputes": True,
            "latency_metadata_recomputes": True,
        },
        parameter_counts={
            "salt_jepa": {"training": 10, "inference": 5},
            "deranged_salt_jepa": {"training": 10, "inference": 5},
        },
        teacher_unchanged={
            "salt_jepa": True,
            "deranged_salt_jepa": True,
        },
        transfer_pair_errors={
            name: {"p0": 0.5 if name == "salt_jepa" else 1.0}
            for name in names
        },
        deployed_bundle_bytes=1024,
        median_latency_ms=0.5,
    )

    assert not assessment["passed"]
    assert not assessment["safety_gates"][
        "transfer_overall_within_1_05_raw"
    ]
    assert assessment["mechanism_gates"]["masked_latent_advantage"]
    assert assessment["decision"] == "reject_salt_jepa_telemetry_recipe"


def test_salt_smoke_artifact_reassesses_from_stored_arrays(tmp_path) -> None:
    output = tmp_path / "salt-smoke"

    run_experiment(
        cache_directory=FROZEN_CACHE,
        output_directory=output,
        teacher_steps=1,
        student_steps=1,
        latency_repetitions=1,
        allow_noninterpretable_smoke=True,
    )

    assessment = verify_stored_assessment(output)
    assert assessment["decision"] == "non_interpretable_salt_jepa_smoke"
    assert all(assessment["protocol_checks"].values())
    assert (output / "artifact-manifest.json").is_file()
    assert (output / "reproduction-source").is_dir()


def _assert_mask_provenance(
    masked: SaltMaskedTelemetry,
    graph: DeclaredTelemetryGraph,
    observed_entities: np.ndarray,
) -> None:
    target = np.asarray(masked.target_tokens, dtype=np.bool_)
    rectangles = np.asarray(masked.block_rectangles, dtype=np.int64)
    fill_order = np.asarray(masked.fill_order, dtype=np.int64)
    protected = np.zeros(target.shape[1:], dtype=np.bool_)
    protected[-1, observed_entities] = True
    for sample_position in range(len(target)):
        reconstructed = np.zeros(target.shape[1:], dtype=np.bool_)
        for start, duration, first, second, third in rectangles[
            sample_position
        ]:
            if start < 0:
                assert np.all(
                    np.asarray(
                        (start, duration, first, second, third)
                    )
                    == -1
                )
                continue
            entities = (int(first), int(second), int(third))
            assert 0 <= start < start + duration <= 20
            assert len(set(entities)) == 3
            reached = {entities[0]}
            while True:
                expanded = reached | {
                    entity
                    for entity in entities
                    if any(
                        graph.entity_ids[neighbor]
                        in graph.neighboring_entity_ids(
                            graph.entity_ids[entity]
                        )
                        for neighbor in reached
                    )
                }
                if expanded == reached:
                    break
                reached = expanded
            assert reached == set(entities)
            proposal = np.zeros_like(reconstructed)
            proposal[start : start + duration, list(entities)] = True
            proposal[protected] = False
            reconstructed |= proposal
        for flat_position in fill_order[sample_position]:
            if flat_position < 0:
                continue
            time_position, entity_position = np.unravel_index(
                int(flat_position), reconstructed.shape
            )
            assert not reconstructed[time_position, entity_position]
            assert not protected[time_position, entity_position]
            temporal_neighbor = (
                time_position > 0
                and reconstructed[time_position - 1, entity_position]
            ) or (
                time_position + 1 < reconstructed.shape[0]
                and reconstructed[time_position + 1, entity_position]
            )
            graph_neighbor = any(
                reconstructed[
                    time_position,
                    graph.entity_ids.index(neighbor_id),
                ]
                for neighbor_id in graph.neighboring_entity_ids(
                    graph.entity_ids[entity_position]
                )
            )
            assert temporal_neighbor or graph_neighbor
            reconstructed[time_position, entity_position] = True
        np.testing.assert_array_equal(reconstructed, target[sample_position])


def _tiny_windows(
    *, pair_count: int, transition_count: int
) -> ActionConditionedWindows:
    graph = DeclaredTelemetryGraph(
        entities=(
            GraphEntity("e0", "node", "service"),
            GraphEntity("e1", "edge", "relation", "e0", "e2"),
            GraphEntity("e2", "node", "service"),
            GraphEntity("e3", "edge", "relation", "e2", "e4"),
            GraphEntity("e4", "node", "service"),
            GraphEntity("e5", "edge", "relation", "e4", "e6"),
            GraphEntity("e6", "node", "service"),
        ),
        bindings=tuple(
            TelemetryBinding(
                f"metric.declared{index}", f"e{index}"
            )
            for index in range(6)
        ),
    )
    rows = []
    trajectory_ids = []
    pair_ids = []
    transitions = []
    for pair in range(pair_count):
        for arm in range(2):
            for transition in range(transition_count):
                values = np.zeros((20, 7, 2), dtype=np.float64)
                for entity in range(6):
                    values[:, entity, entity % 2] = (
                        np.arange(20) + pair + arm + transition + entity
                    )
                rows.append(values)
                trajectory_ids.append(f"pair{pair}-arm{arm}")
                pair_ids.append(f"pair{pair}")
                transitions.append(transition)
    sample_count = len(rows)
    return ActionConditionedWindows(
        histories=np.asarray(rows),
        future_states=np.zeros(
            (sample_count, 10, 7, 2), dtype=np.float64
        ),
        future_controls=np.zeros(
            (sample_count, 10, 1), dtype=np.float64
        ),
        future_actions=np.zeros(
            (sample_count, 10, 7, 1), dtype=np.float64
        ),
        trajectory_ids=tuple(trajectory_ids),
        matched_pair_ids=tuple(pair_ids),
        transition_indices=np.asarray(transitions, dtype=np.int64),
        entity_names=graph.entity_ids,
        state_feature_names=("f0", "f1"),
        control_feature_names=("control",),
        action_feature_names=("action",),
        graph=graph,
    )
