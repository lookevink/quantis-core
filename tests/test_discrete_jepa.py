import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from lab.action_dynamics.prototype_discrete_jepa import (
    FROZEN_CACHE,
    run_experiment,
)
from lab.action_dynamics.prototype_discrete_jepa_assessor import (
    _role_identifiers_are_disjoint,
    _transition_metrics,
    verify_stored_assessment,
)
from quantis_core.action_conditioned_dynamics import (
    ActionConditionedWindows,
)
from quantis_core.edge_dynamics.discrete_jepa import (
    DiscreteJepaConfig,
    DiscreteJepaRepresentation,
    DiscreteMaskSchedule,
    assess_discrete_jepa_gates,
    discrete_jepa_losses,
)
from quantis_core.graph_telemetry import (
    DeclaredTelemetryGraph,
    GraphEntity,
    TelemetryBinding,
)


def test_discrete_mask_schedule_is_deterministic_and_bounded() -> None:
    schedule = DiscreteMaskSchedule(
        entity_count=7, patch_count=5, seed=25025
    )

    first = schedule.batch(step=3, batch_size=40)
    replay = schedule.batch(step=3, batch_size=40)
    other = schedule.batch(step=4, batch_size=40)

    np.testing.assert_array_equal(first, replay)
    assert not np.array_equal(first, other)
    assert first.shape == (40, 7, 5)
    masked_fraction = np.mean(first, axis=(1, 2))
    assert np.all(masked_fraction >= 0.40)
    assert np.all(masked_fraction <= 0.60)
    assert np.all(np.sum(~first, axis=2) >= 1)


def test_discrete_complementary_loss_is_literal_and_stops_targets() -> None:
    import torch

    s2p = torch.arange(24, dtype=torch.float64).reshape(
        2, 2, 3, 2
    ).requires_grad_()
    p2s = torch.arange(8, dtype=torch.float64).reshape(
        2, 2, 2
    ).requires_grad_()
    p2p = (s2p.detach() / 7).requires_grad_()
    target_patch = (
        torch.arange(24, dtype=torch.float64).reshape(2, 2, 3, 2)
        / 11
    ).requires_grad_()
    target_semantic = (
        torch.arange(8, dtype=torch.float64).reshape(2, 2, 2) / 13
    ).requires_grad_()
    mask = torch.tensor(
        [
            [[True, False, True], [False, True, False]],
            [[False, True, False], [True, False, True]],
        ]
    )
    commitment = (s2p.square().mean() * 0.25)

    result = discrete_jepa_losses(
        s2p_prediction=s2p,
        p2s_prediction=p2s,
        p2p_prediction=p2p,
        target_patch=target_patch,
        target_semantic=target_semantic,
        mask=mask,
        commitment=commitment,
        objective="discrete_complete",
    )
    expanded = mask[..., None].expand_as(s2p)
    expected_s2p = (
        (s2p - target_patch.detach()).square()[expanded].mean()
    )
    expected_p2s = (
        p2s - target_semantic.detach()
    ).square().mean()
    expected_p2p = (
        (p2p - target_patch.detach()).square()[expanded].mean()
    )

    torch.testing.assert_close(result.s2p, expected_s2p)
    torch.testing.assert_close(result.p2s, expected_p2s)
    torch.testing.assert_close(result.p2p, expected_p2p)
    torch.testing.assert_close(
        result.total,
        expected_s2p + expected_p2s + expected_p2p + commitment,
    )
    result.total.backward()
    assert s2p.grad is not None
    assert p2s.grad is not None
    assert p2p.grad is not None
    assert target_patch.grad is None
    assert target_semantic.grad is None


def test_discrete_representation_is_hard_restorable_and_capacity_matched() -> None:
    import torch

    windows = _tiny_windows(pair_count=4, transition_count=5)
    configs = [
        DiscreteJepaConfig(
            objective=objective,
            width=8,
            depth=1,
            head_count=2,
            feedforward_width=16,
            code_count=8,
            steps=1,
            warmup_steps=1,
            expected_pair_count=4,
        )
        for objective in (
            "discrete_complete",
            "continuous_complete",
            "discrete_p2p_only",
        )
    ]
    torch.manual_seed(250250)
    before = torch.random.get_rng_state().clone()
    models = [
        DiscreteJepaRepresentation(config).fit(windows)
        for config in configs
    ]
    np.testing.assert_array_equal(
        torch.random.get_rng_state().numpy(), before.numpy()
    )
    assert len(
        {model.training_parameter_count for model in models}
    ) == 1
    assert len(
        {model.inference_parameter_count for model in models}
    ) == 1

    model = models[0]
    encoded = model.encode(windows.histories[:6], windows.graph)
    restored = DiscreteJepaRepresentation.from_dict(model.to_dict())
    inference_payload = model.to_inference_dict()
    inference = DiscreteJepaRepresentation.from_inference_dict(
        inference_payload
    )
    restored_encoded = restored.encode(
        windows.histories[:6], windows.graph
    )
    inference_encoded = inference.encode(
        windows.histories[:6], windows.graph
    )
    diagnostic = model.diagnose(
        windows.histories[:3], windows.graph, mask_seed=90025
    )
    altered = windows.histories[:3].copy()
    for row, entity, patch in np.argwhere(diagnostic.mask):
        altered[
            row,
            patch * 4 : (patch + 1) * 4,
            entity,
            :,
        ] += 10_000.0
    altered_diagnostic = model.diagnose(
        altered, windows.graph, mask_seed=90025
    )

    assert encoded.tokens.shape == (6, 7, 8)
    assert encoded.indices is not None
    assert encoded.indices.shape == (6, 7)
    assert np.all(encoded.indices >= 0)
    assert np.all(encoded.indices < 8)
    np.testing.assert_allclose(
        encoded.tokens, restored_encoded.tokens, atol=1e-7
    )
    np.testing.assert_array_equal(
        encoded.indices, restored_encoded.indices
    )
    np.testing.assert_allclose(
        encoded.tokens, inference_encoded.tokens, atol=1e-7
    )
    assert "target_state" not in inference_payload
    assert "predictor_state" not in inference_payload
    np.testing.assert_allclose(
        diagnostic.s2p_prediction,
        altered_diagnostic.s2p_prediction,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        diagnostic.p2s_prediction,
        altered_diagnostic.p2s_prediction,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        diagnostic.p2p_prediction,
        altered_diagnostic.p2p_prediction,
        atol=1e-7,
    )


def test_discrete_assessment_rejects_failed_value() -> None:
    names = (
        "discrete_complete",
        "continuous_complete",
        "discrete_p2p_only",
        "matched_pca",
    )
    scores = {
        name: {
            role: {
                "overall_mse": 1.0,
                "action_overlap_mse": 1.0,
                "downstream_effect_mse": (
                    1.2 if name == "discrete_complete" else 1.0
                ),
            }
            for role in ("selection", "transfer_evaluation")
        }
        for name in names
    }
    assessment = assess_discrete_jepa_gates(
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
            "noncollapsed_code_usage": True,
            "complementary_heads_are_learned": True,
            "p2p_is_preserved": True,
            "next_code_advantage": True,
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
            key: True
            for key in (
                "evidence_arrays_are_finite",
                "pair_and_trajectory_roles_are_disjoint",
                "capacity_recomputes",
                "public_inference_is_causal",
                "anchor_schedule_recomputes",
                "mask_schedule_recomputes",
                "selection_only_ridge_choice_recomputes",
                "selection_safety_status_recomputes",
                "bundle_size_recomputes",
                "latency_recomputes",
            )
        },
        parameter_counts={
            name: {"training": 10, "inference": 5}
            for name in names[:3]
        },
        transfer_pair_errors={
            name: {"p0": 1.2 if name == "discrete_complete" else 1.0}
            for name in names
        },
        deployed_bundle_bytes=1024,
        median_latency_ms=1.0,
    )

    assert not assessment["passed"]
    assert not assessment["value_gates"]["selection_effect_is_best"]
    assert assessment["decision"] == "reject_discrete_jepa_recipe"
    assert assessment["transfer_pair_errors"][
        "discrete_complete"
    ] == {"p0": 1.2}


def test_discrete_transition_metrics_report_every_entity() -> None:
    fit = _tiny_windows(pair_count=4, transition_count=5)
    evaluation = _tiny_windows(pair_count=4, transition_count=5)
    fit_indices = np.zeros((len(fit.histories), 7), dtype=np.int64)
    evaluation_indices = np.zeros(
        (len(evaluation.histories), 7), dtype=np.int64
    )

    metrics = _transition_metrics(
        fit_indices=fit_indices,
        fit=fit,
        evaluation_indices={"selection": evaluation_indices},
        evaluations={"selection": evaluation},
        code_count=64,
    )

    assert metrics["selection"]["overall"] == 1.0
    assert metrics["selection"]["per_entity"] == [1.0] * 7


def test_discrete_role_contract_enforces_frozen_pair_counts() -> None:
    expected = {
        "fit": 40,
        "selection": 10,
        "calibration": 10,
        "iid_evaluation": 20,
        "transfer_evaluation": 10,
    }
    metadata = {
        "roles": {
            role: {
                "pair_ids": [
                    f"{role}-pair-{index}" for index in range(count)
                ],
                "trajectory_ids": [
                    f"{role}-trajectory-{index}"
                    for index in range(count)
                ],
            }
            for role, count in expected.items()
        }
    }

    assert _role_identifiers_are_disjoint(metadata)
    metadata["roles"]["selection"]["pair_ids"].pop()
    assert not _role_identifiers_are_disjoint(metadata)


def test_discrete_smoke_artifact_reassesses(
    tmp_path: "Path",
) -> None:
    output = tmp_path / "discrete-jepa-smoke"
    run_experiment(
        cache_directory=FROZEN_CACHE,
        output_directory=output,
        steps=1,
        latency_repetitions=1,
        allow_noninterpretable_smoke=True,
    )

    assessment = verify_stored_assessment(output)
    assert (
        assessment["decision"]
        == "non_interpretable_discrete_jepa_smoke"
    )
    assert all(assessment["protocol_checks"].values())
    assert assessment["safety_gates"][
        "deployed_bundle_within_16_mib"
    ]
    assert (output / "artifact-manifest.json").is_file()
    assert (output / "reproduction-source").is_dir()
    reproduction = output / "reproduction-source"
    copied_environment = dict(os.environ)
    copied_environment["PYTHONPATH"] = os.pathsep.join(
        (
            str(reproduction / "src"),
            str(reproduction / "lab/action_dynamics"),
        )
    )
    copied_assessment = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            (
                "import runpy,sys;"
                "sys.path[:0]=sys.argv[1:4];"
                "sys.argv=sys.argv[4:];"
                "runpy.run_path(sys.argv[0],run_name='__main__')"
            ),
            str(reproduction / "src"),
            str(reproduction / "lab/action_dynamics"),
            str(Path(np.__file__).resolve().parents[1]),
            str(
                reproduction
                / "lab/action_dynamics/"
                "prototype_discrete_jepa_assessor.py"
            ),
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=copied_environment,
        cwd=tmp_path,
    )
    assert (
        "non_interpretable_discrete_jepa_smoke"
        in copied_assessment.stdout
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
                        np.arange(20)
                        + pair
                        + arm
                        + transition
                        + entity
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
            (sample_count, 10, 7, 2), dtype=np.float64
        ),
        trajectory_ids=tuple(trajectory_ids),
        matched_pair_ids=tuple(pair_ids),
        transition_indices=np.asarray(transitions, dtype=np.int64),
        entity_names=graph.entity_ids,
        state_feature_names=("f0", "f1"),
        control_feature_names=("control",),
        action_feature_names=("none", "treatment"),
        graph=graph,
    )
