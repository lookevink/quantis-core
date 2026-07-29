import numpy as np

from quantis_core.action_conditioned_dynamics import (
    ActionConditionedWindows,
)
from quantis_core.edge_dynamics.complete_lejepa import (
    CompleteLejepaConfig,
    CompleteLejepaRepresentation,
    EntityPcaRepresentation,
    PairBlockedAnchorSchedule,
    ReducedRankActionProbe,
    assess_complete_lejepa_gates,
    TelemetryViewSchedule,
    complete_lejepa_loss,
    fit_owned_feature_mask,
)
from quantis_core.graph_telemetry import (
    DeclaredTelemetryGraph,
    GraphEntity,
    TelemetryBinding,
)


def test_pair_blocked_schedule_cycles_transitions_and_balances_arms() -> None:
    windows = _tiny_windows(pair_count=4, transition_count=5)
    schedule = PairBlockedAnchorSchedule(windows, seed=1509)

    batches = [schedule.batch(step) for step in range(10)]

    for batch in batches:
        assert batch.indices.shape == (4,)
        assert len(set(batch.pair_ids)) == 4
        assert sum(batch.arm_ids) == 2
    for pair_id in sorted(set(windows.matched_pair_ids)):
        observed = {
            int(batch.transition_indices[batch.pair_ids.index(pair_id)])
            for batch in batches[:5]
        }
        assert observed == set(range(5))
        arms = [
            int(batch.arm_ids[batch.pair_ids.index(pair_id)])
            for batch in batches
        ]
        assert sum(arms) == 5
        assert all(left != right for left, right in zip(arms, arms[1:]))


def test_telemetry_views_are_seeded_owned_connected_and_nonmutating() -> None:
    windows = _tiny_windows(pair_count=4, transition_count=5)
    anchors = PairBlockedAnchorSchedule(windows, seed=1509).batch(0)
    histories = windows.histories[anchors.indices]
    before = histories.copy()
    ownership = fit_owned_feature_mask(windows)
    schedule = TelemetryViewSchedule(
        graph=windows.graph,
        ownership_mask=ownership,
        varying_entity_mask=np.any(ownership, axis=1),
        seed=2509,
    )

    first = schedule.batch(histories, step=3)
    second = schedule.batch(histories, step=3)

    np.testing.assert_array_equal(histories, before)
    np.testing.assert_array_equal(first.values, second.values)
    np.testing.assert_array_equal(first.visible_tokens, second.visible_tokens)
    assert first.values.shape == (8, 4, 20, 7, 2)
    assert first.visible_tokens.shape == (8, 4, 20, 7)
    assert first.view_names == (
        "global_a",
        "global_b",
        "local_e0",
        "local_e1",
        "local_e2",
        "local_e3",
        "local_e4",
        "local_e5",
    )
    assert not np.any(first.visible_tokens[1, :, :4])
    assert not np.any(first.visible_tokens[2:, :, :10])
    assert np.all(np.sum(first.visible_tokens[2:], axis=(2, 3)) == 30)
    for view_position, root in enumerate(range(6), start=2):
        visible_entities = np.flatnonzero(
            np.any(first.visible_tokens[view_position, 0], axis=0)
        )
        assert root in visible_entities
        assert len(visible_entities) == 3
        assert _is_connected(windows.graph, visible_entities)
    masked_or_visible = np.logical_or(
        first.visible_tokens[..., None], ~ownership[None, None, None]
    )
    assert np.all(first.values[~masked_or_visible] == 0.0)


def test_complete_lejepa_loss_matches_literal_reference_and_has_gradients() -> None:
    import torch

    values = torch.arange(48, dtype=torch.float64).reshape(3, 4, 4) / 17
    values.requires_grad_(True)
    generator = torch.Generator(device="cpu").manual_seed(3509)
    reference_generator = torch.Generator(device="cpu").manual_seed(3509)

    result = complete_lejepa_loss(
        values,
        generator=generator,
        sketch_dimension=7,
        knot_count=5,
    )
    directions = torch.randn(
        4, 7, dtype=torch.float64, generator=reference_generator
    )
    directions = directions / directions.norm(dim=0, keepdim=True)
    knots = torch.linspace(0.0, 3.0, 5, dtype=torch.float64)
    delta = 3.0 / 4.0
    quadrature = torch.full((5,), 2.0 * delta, dtype=torch.float64)
    quadrature[[0, -1]] = delta
    gaussian = torch.exp(-(knots**2) / 2.0)
    projected = (values @ directions).unsqueeze(-1) * knots
    error = (
        (projected.cos().mean(dim=-3) - gaussian) ** 2
        + projected.sin().mean(dim=-3) ** 2
    )
    sigreg = ((error @ (quadrature * gaussian)) * 4).mean()
    global_mean = values[:2].mean(dim=0)
    invariance = ((values - global_mean) ** 2).mean()
    expected = 0.05 * sigreg + 0.95 * invariance

    torch.testing.assert_close(result.loss, expected)
    torch.testing.assert_close(result.sigreg, sigreg)
    torch.testing.assert_close(result.invariance, invariance)
    result.loss.backward()
    assert values.grad is not None
    assert torch.all(torch.isfinite(values.grad))
    assert torch.count_nonzero(values.grad) > 0
    assert torch.equal(
        generator.get_state(), reference_generator.get_state()
    )


def test_complete_lejepa_encodes_ordered_tokens_and_restores_outputs() -> None:
    windows = _tiny_windows(pair_count=4, transition_count=5)
    config = CompleteLejepaConfig(
        objective="lejepa",
        steps=1,
        expected_pair_count=4,
        sketch_dimension=7,
    )

    model = CompleteLejepaRepresentation(config).fit(windows)
    encoded = model.encode(windows.histories[:3], windows.graph)
    restored = CompleteLejepaRepresentation.from_dict(model.to_dict())
    restored_encoded = restored.encode(
        windows.histories[:3], windows.graph
    )

    assert encoded.tokens.shape == (3, 7, 64)
    assert encoded.entity_ids == windows.entity_names
    np.testing.assert_array_equal(
        encoded.ownership_mask, fit_owned_feature_mask(windows)
    )
    assert np.all(np.isfinite(encoded.tokens))
    np.testing.assert_allclose(
        encoded.tokens, restored_encoded.tokens, atol=1e-7
    )
    assert model.inference_parameter_count == (
        CompleteLejepaRepresentation(
            CompleteLejepaConfig(
                objective="masked_autoencoder",
                steps=1,
                expected_pair_count=4,
                sketch_dimension=7,
            )
        )
        .fit(windows)
        .inference_parameter_count
    )


def test_entity_pca_is_fit_only_deterministic_and_width_matched() -> None:
    fit = _tiny_windows(pair_count=4, transition_count=5)
    shifted = fit.histories[:2].copy()
    shifted[:, :, :6] += 10_000.0

    first = EntityPcaRepresentation(width=64).fit(fit)
    before = first.to_dict()
    encoded = first.encode(shifted, fit.graph)
    second = EntityPcaRepresentation(width=64).fit(fit)
    restored = EntityPcaRepresentation.from_dict(first.to_dict())

    assert first.to_dict() == before
    assert encoded.tokens.shape == (2, 7, 64)
    np.testing.assert_allclose(
        first.encode(fit.histories[:3], fit.graph).tokens,
        second.encode(fit.histories[:3], fit.graph).tokens,
    )
    np.testing.assert_allclose(
        encoded.tokens, restored.encode(shifted, fit.graph).tokens
    )


def test_reduced_rank_action_probe_restores_and_respects_fitted_rank() -> None:
    windows = _tiny_windows(pair_count=4, transition_count=5)
    representation = EntityPcaRepresentation(width=64).fit(windows)
    tokens = representation.encode(
        windows.histories, windows.graph
    ).tokens
    probe = ReducedRankActionProbe(rank=3, ridge=1e-3).fit(
        tokens,
        windows.future_controls,
        windows.future_actions,
        windows.future_states,
    )

    prediction = probe.predict(
        tokens[:3],
        windows.future_controls[:3],
        windows.future_actions[:3],
    )
    restored = ReducedRankActionProbe.from_dict(probe.to_dict())

    assert prediction.shape == windows.future_states[:3].shape
    assert probe.fitted_rank <= 3
    np.testing.assert_allclose(
        prediction,
        restored.predict(
            tokens[:3],
            windows.future_controls[:3],
            windows.future_actions[:3],
        ),
    )


def test_complete_lejepa_assessment_recomputes_a_failed_gate() -> None:
    names = (
        "complete_lejepa",
        "invariance_only",
        "sigreg_only",
        "masked_autoencoder",
        "matched_pca",
    )
    forecast = {
        name: {
            "selection": {
                "overall_mse": 1.0,
                "action_overlap_mse": 1.0,
                "downstream_effect_mse": (
                    0.5 if name == "complete_lejepa" else 1.0
                ),
            },
            "transfer_evaluation": {
                "overall_mse": (
                    2.0 if name == "complete_lejepa" else 1.0
                ),
                "action_overlap_mse": 1.0,
                "downstream_effect_mse": (
                    0.5 if name == "complete_lejepa" else 1.0
                ),
            },
        }
        for name in names
    }
    state = {
        name: {
            "aggregate_nrmse": 1.0,
            "entities": {"e0": {"nrmse": 1.0}},
        }
        for name in names
    }
    attribution = {
        name: {
            "action_and_target_hit_at_1": 1.0,
            "no_action_specificity": 1.0,
        }
        for name in names
    }
    sanity = {
        name: {"correct_action_beats_both_fraction": 1.0}
        for name in names
    }
    curves = {
        name: [
            {
                "ridge": 1e-3,
                "raw_safe": True,
                "downstream_effect_mse": 1.0,
            }
        ]
        for name in names
    }

    assessment = assess_complete_lejepa_gates(
        forecast_scores=forecast,
        raw_scores={
            "transfer_evaluation": {
                "overall_mse": 1.0,
                "action_overlap_mse": 1.0,
                "downstream_effect_mse": 1.0,
            }
        },
        state_probes=state,
        attribution=attribution,
        action_sanity=sanity,
        restoration_parity={name: True for name in names},
        ridge_curves=curves,
        selected_ridges={name: 1e-3 for name in names},
        transfer_pair_errors={
            name: {"pair": 0.5 if name == "complete_lejepa" else 1.0}
            for name in names
        },
        protocol_checks={
            "pair_blocked_schedule_is_valid": True,
            "telemetry_view_schedule_is_valid": True,
            "evidence_arrays_are_finite": True,
        },
    )

    assert not assessment["passed"]
    assert not assessment["safety_gates"][
        "overall_mse_within_1_05_raw"
    ]
    assert assessment["decision"] == (
        "reject_exact_complete_multi_view_lejepa_recipe"
    )


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


def _is_connected(
    graph: DeclaredTelemetryGraph, positions: np.ndarray
) -> bool:
    names = {graph.entity_ids[int(position)] for position in positions}
    reached = {next(iter(names))}
    while True:
        expanded = set(reached)
        for name in reached:
            expanded.update(graph.neighboring_entity_ids(name))
        expanded.intersection_update(names)
        if expanded == reached:
            return reached == names
        reached = expanded
