from pathlib import Path

import numpy as np
import pytest

from quantis_core.action_conditioned_dynamics import (
    ActionConditionedWindows,
)
from quantis_core.edge_dynamics.visreg import (
    VisregConfig,
    VisregDirectionSchedule,
    VisregRepresentation,
    assess_visreg_gates,
    visreg_loss,
)
from quantis_core.graph_telemetry import (
    DeclaredTelemetryGraph,
    GraphEntity,
    TelemetryBinding,
)


def test_visreg_loss_matches_literal_float32_formula() -> None:
    import torch

    embeddings = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 1.0], [2.0, -1.0]],
            [[0.0, 1.0], [2.0, 4.0], [-1.0, 3.0]],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    raw_directions = torch.tensor(
        [[1.0, 1.0, -2.0], [0.0, 1.0, 1.0]],
        dtype=torch.float32,
    )
    directions = raw_directions / torch.linalg.vector_norm(
        raw_directions, dim=0, keepdim=True
    )

    result = visreg_loss(
        embeddings, directions, detach_shape=True
    )

    mean = embeddings.mean(dim=1)
    centered = embeddings - mean[:, None]
    std = torch.linalg.vector_norm(centered, dim=1) / np.sqrt(3.0)
    std = std.clamp_min(1e-6)
    quantiles = np.sqrt(2.0) * torch.erfinv(
        2.0
        * torch.arange(1, 4, dtype=torch.float32)
        / 4.0
        - 1.0
    )
    projected = torch.sort(
        (centered / std.detach()[:, None]) @ directions, dim=1
    ).values
    shape = (
        projected - quantiles[None, :, None]
    ).square().mean()
    scale = (std - 1.0).square().mean()
    center = mean.square().mean()

    torch.testing.assert_close(result.quantiles, quantiles)
    torch.testing.assert_close(result.shape, shape)
    torch.testing.assert_close(result.scale, scale)
    torch.testing.assert_close(result.center, center)
    torch.testing.assert_close(
        result.regularization, shape + scale + center
    )
    result.regularization.backward()
    assert embeddings.grad is not None
    assert bool(torch.isfinite(embeddings.grad).all())


def test_visreg_detach_changes_only_gradient_semantics() -> None:
    import torch

    generator = torch.Generator(device="cpu").manual_seed(19)
    base = torch.randn(
        (3, 5, 4), generator=generator, dtype=torch.float32
    )
    raw = torch.randn(
        (4, 7), generator=generator, dtype=torch.float32
    )
    directions = raw / torch.linalg.vector_norm(
        raw, dim=0, keepdim=True
    )
    detached_input = base.clone().requires_grad_(True)
    attached_input = base.clone().requires_grad_(True)

    detached = visreg_loss(
        detached_input, directions, detach_shape=True
    )
    attached = visreg_loss(
        attached_input, directions, detach_shape=False
    )
    detached_gradient = torch.autograd.grad(
        detached.regularization, detached_input
    )[0]
    attached_gradient = torch.autograd.grad(
        attached.regularization, attached_input
    )[0]

    torch.testing.assert_close(
        detached.regularization, attached.regularization
    )
    torch.testing.assert_close(detached.shape, attached.shape)
    assert float(
        torch.max(torch.abs(detached_gradient - attached_gradient))
    ) > 1e-7


def test_visreg_direction_schedule_is_explicit_and_replayable() -> None:
    import torch

    torch.manual_seed(7331)
    ambient_before = torch.random.get_rng_state()
    schedule = VisregDirectionSchedule(
        width=5, projection_count=11, seed=3509
    )
    initial_state = schedule.initial_state
    first = schedule.draw()
    second = schedule.draw()
    ambient_after = torch.random.get_rng_state()

    assert schedule.draw_count == 2
    assert initial_state.dtype == np.uint8
    assert not np.array_equal(initial_state, schedule.final_state)
    np.testing.assert_array_equal(
        first.numpy(),
        VisregDirectionSchedule.replay(
            width=5,
            projection_count=11,
            seed=3509,
            step=0,
        ).numpy(),
    )
    np.testing.assert_array_equal(
        second.numpy(),
        VisregDirectionSchedule.replay(
            width=5,
            projection_count=11,
            seed=3509,
            step=1,
        ).numpy(),
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(first, dim=0),
        torch.ones(11),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_array_equal(
        ambient_before.numpy(), ambient_after.numpy()
    )


def test_visreg_representation_is_causal_restorable_and_matched() -> None:
    import torch

    windows = _tiny_windows(pair_count=4, transition_count=5)
    common = dict(
        steps=2,
        expected_pair_count=4,
        width=8,
        head_count=2,
        feedforward_width=16,
        projector_width=16,
        projection_count=8,
        warmup_steps=1,
    )
    detached = VisregRepresentation(
        VisregConfig(objective="detached_visreg", **common)
    )
    attached = VisregRepresentation(
        VisregConfig(objective="no_detach_visreg", **common)
    )

    torch.manual_seed(8871)
    before = torch.random.get_rng_state()
    mps_before = (
        torch.mps.get_rng_state()
        if torch.backends.mps.is_available()
        else None
    )
    detached.fit(windows)
    after_detached = torch.random.get_rng_state()
    mps_after_detached = (
        torch.mps.get_rng_state()
        if torch.backends.mps.is_available()
        else None
    )
    attached.fit(windows)
    after_attached = torch.random.get_rng_state()
    mps_after_attached = (
        torch.mps.get_rng_state()
        if torch.backends.mps.is_available()
        else None
    )

    encoded = detached.encode(windows.histories[:3], windows.graph)
    restored = VisregRepresentation.from_dict(detached.to_dict())
    replay = restored.encode(windows.histories[:3], windows.graph)
    inference_payload = detached.to_inference_dict()
    inference = VisregRepresentation.from_inference_dict(
        inference_payload
    )
    inference_values = inference.encode(
        windows.histories[:3], windows.graph
    )

    np.testing.assert_array_equal(before.numpy(), after_detached.numpy())
    np.testing.assert_array_equal(
        after_detached.numpy(), after_attached.numpy()
    )
    if mps_before is not None:
        assert mps_after_detached is not None
        assert mps_after_attached is not None
        np.testing.assert_array_equal(
            mps_before.cpu().numpy(),
            mps_after_detached.cpu().numpy(),
        )
        np.testing.assert_array_equal(
            mps_after_detached.cpu().numpy(),
            mps_after_attached.cpu().numpy(),
        )
    np.testing.assert_allclose(encoded.tokens, replay.tokens, atol=1e-7)
    np.testing.assert_allclose(
        encoded.tokens, inference_values.tokens, atol=1e-7
    )
    assert detached.training_parameter_count == (
        attached.training_parameter_count
    )
    assert detached.inference_parameter_count == (
        attached.inference_parameter_count
    )
    assert detached.network_sha256 != attached.network_sha256
    assert detached.projector_sha256 != attached.projector_sha256
    assert np.max(
        np.abs(
            encoded.tokens
            - attached.encode(
                windows.histories[:3], windows.graph
            ).tokens
        )
    ) > 1e-6
    assert set(inference_payload) == {
        "schema_version",
        "kind",
        "config",
        "graph",
        "feature_names",
        "ownership_mask",
        "network_state",
        "inference_parameter_count",
    }
    assert "projector_state" not in inference_payload
    assert "direction_final_state" not in inference_payload
    assert set(inference_payload["config"]) == {
        "width",
        "block_count",
        "head_count",
        "feedforward_width",
        "preprocessing_protocol",
    }
    assert "objective" not in inference_payload["config"]
    assert "direction_seed" not in inference_payload["config"]
    assert detached.training_evidence["embeddings"].shape == (
        2,
        8,
        4,
        8,
    )
    assert detached.training_evidence["directions"].shape == (
        2,
        8,
        8,
    )
    assert detached.training_evidence[
        "sorted_projection_sha256"
    ].shape == (2, 32)
    assert detached.training_evidence["gaussian_quantiles"].shape == (
        4,
    )
    assert detached.direction_draw_count == 2
    with pytest.raises(RuntimeError):
        _ = inference.training_parameter_count


def test_visreg_assessment_rejects_failed_mechanism() -> None:
    names = (
        "detached_visreg",
        "no_detach_visreg",
        "complete_lejepa",
        "invariance_only",
        "sigreg_only",
        "masked_autoencoder",
        "matched_pca",
    )
    scores = {
        name: {
            role: {
                "overall_mse": 1.0,
                "action_overlap_mse": 1.0,
                "downstream_effect_mse": (
                    0.8 if name == "detached_visreg" else 1.0
                ),
            }
            for role in ("selection", "transfer_evaluation")
        }
        for name in names
    }
    assessment = assess_visreg_gates(
        forecast_scores=scores,
        raw_scores={
            role: {
                "overall_mse": 1.0,
                "action_overlap_mse": 1.0,
                "downstream_effect_mse": 1.0,
            }
            for role in ("selection", "transfer_evaluation")
        },
        mechanism_gates={
            "exact_math_and_rng": True,
            "candidate_noncollapsed": True,
            "collapse_gradient_beats_sigreg": False,
            "detached_shape_beats_no_detach": True,
        },
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
            name: True
            for name in (
                "evidence_arrays_are_finite",
                "role_contract_recomputes",
                "all_schedules_recompute",
                "objective_recomputes",
                "mode_enforcement_recomputes",
                "capacity_recomputes",
                "public_inference_is_causal",
                "copied_source_assessor_recomputes",
                "copied_prior_controls_match",
                "selection_only_ridge_choice_recomputes",
                "selection_safety_status_recomputes",
                "bundle_size_recomputes",
                "latency_recomputes",
                "state_probe_recomputes",
            )
        },
        parameter_counts={
            name: {"training": 10, "inference": 5}
            for name in ("detached_visreg", "no_detach_visreg")
        },
        transfer_pair_errors={
            name: {
                "pair-0": 0.8 if name == "detached_visreg" else 1.0
            }
            for name in names
        },
        state_probe={
            "detached_visreg": {
                "aggregate_nrmse": 1.0,
                "entity_nrmse": {"e0": 1.0},
            },
            "matched_pca": {
                "aggregate_nrmse": 1.0,
                "entity_nrmse": {"e0": 1.0},
            },
        },
        varying_entity_ids=("e0",),
        deployed_bundle_bytes=1024,
        median_latency_ms=1.0,
    )

    assert not assessment["passed"]
    assert not assessment["mechanism_gates"][
        "collapse_gradient_beats_sigreg"
    ]
    assert assessment["decision"] == "reject_visreg_recipe"


def test_visreg_smoke_artifact_reassesses(tmp_path: Path) -> None:
    from lab.action_dynamics.prototype_visreg import (
        FROZEN_CACHE,
        FROZEN_PRIOR_CONTROL,
        run_experiment,
    )
    from lab.action_dynamics.prototype_visreg_assessor import (
        verify_stored_assessment,
    )

    output = tmp_path / "visreg-smoke"
    run_experiment(
        cache_directory=FROZEN_CACHE,
        prior_control_directory=FROZEN_PRIOR_CONTROL,
        output_directory=output,
        steps=1,
        latency_repetitions=1,
        allow_noninterpretable_smoke=True,
    )
    assessment = verify_stored_assessment(output)

    assert assessment["decision"] == "non_interpretable_visreg_smoke"
    assert all(assessment["protocol_checks"].values())


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
    rows = pair_count * 2 * transition_count
    histories = np.zeros((rows, 20, 7, 6), dtype=np.float64)
    future_states = np.zeros((rows, 3, 7, 6), dtype=np.float64)
    future_controls = np.zeros((rows, 3, 1), dtype=np.float64)
    future_actions = np.zeros((rows, 3, 7, 1), dtype=np.float64)
    pair_ids = []
    trajectory_ids = []
    transitions = []
    row = 0
    for pair in range(pair_count):
        for arm in range(2):
            for transition in range(transition_count):
                base = pair + arm * 0.5 + transition * 0.1
                for entity in range(6):
                    histories[row, :, entity, entity] = (
                        base + np.arange(20) * 0.01
                    )
                    future_states[row, :, entity, entity] = (
                        base + 0.2 + np.arange(3) * 0.01
                    )
                pair_ids.append(f"pair-{pair}")
                trajectory_ids.append(f"pair-{pair}-arm-{arm}")
                transitions.append(transition)
                row += 1
    return ActionConditionedWindows(
        histories=histories,
        future_states=future_states,
        future_controls=future_controls,
        future_actions=future_actions,
        graph=graph,
        entity_names=graph.entity_ids,
        state_feature_names=tuple(
            f"declared{index}" for index in range(6)
        ),
        control_feature_names=("control",),
        action_feature_names=("action",),
        matched_pair_ids=tuple(pair_ids),
        trajectory_ids=tuple(trajectory_ids),
        transition_indices=np.asarray(transitions, dtype=np.int64),
    )
