import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from quantis_core.action_conditioned_dynamics import (
    ActionConditionedWindows,
)
from quantis_core.edge_dynamics.peira import (
    PeiraConfig,
    PeiraRepresentation,
    PeiraSchedule,
    assess_peira_gates,
    peira_auxiliary_step,
)
from quantis_core.graph_telemetry import (
    DeclaredTelemetryGraph,
    GraphEntity,
    TelemetryBinding,
)


def test_peira_auxiliary_step_matches_literal_formula() -> None:
    import torch

    first = torch.tensor(
        [[1.0, 2.0], [3.0, 1.0], [2.0, -1.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    second = torch.tensor(
        [[2.0, 0.0], [1.0, 4.0], [-1.0, 3.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    zeros = torch.zeros((2, 2), dtype=torch.float64)

    result = peira_auxiliary_step(
        first,
        second,
        running_signal=zeros,
        running_noise=zeros,
        regularization=0.1,
        eta=0.8,
    )

    batch_signal = (
        first.detach().T @ second.detach()
        + second.detach().T @ first.detach()
    ) / 3
    batch_noise = (
        first.detach().T @ first.detach()
        + second.detach().T @ second.detach()
    ) / 3
    signal = 0.8 * batch_signal
    noise = 0.8 * batch_noise
    inverse = torch.linalg.solve(
        noise + 0.1 * torch.eye(2, dtype=torch.float64),
        torch.eye(2, dtype=torch.float64),
    )
    predictor = signal @ inverse
    residual_first = first @ predictor.T - second
    residual_second = second @ predictor.T - first
    expected = 0.5 * (
        torch.sum(
            first * (residual_first @ inverse.T), dim=1
        ).mean()
        + torch.sum(
            second * (residual_second @ inverse.T), dim=1
        ).mean()
    ) + 0.05 * (
        torch.sum(first.square(), dim=1).mean()
        + torch.sum(second.square(), dim=1).mean()
    )

    torch.testing.assert_close(result.loss, expected)
    torch.testing.assert_close(result.signal, signal)
    torch.testing.assert_close(result.noise, noise)
    torch.testing.assert_close(result.predictor, predictor)
    torch.testing.assert_close(result.inverse, inverse)
    result.loss.backward()
    assert first.grad is not None
    assert second.grad is not None
    assert result.signal.grad is None
    assert result.noise.grad is None


def test_peira_schedule_is_deterministic_and_deranged() -> None:
    schedule = PeiraSchedule(
        steps=5,
        eta_initial=0.9,
        eta_final=0.5,
        derangement_seed=26326,
    )

    assert schedule.eta(0) == 0.9
    assert schedule.eta(4) == 0.5
    first = schedule.derangement(2, 8)
    replay = schedule.derangement(2, 8)

    np.testing.assert_array_equal(first, replay)
    np.testing.assert_array_equal(np.sort(first), np.arange(8))
    assert np.all(first != np.arange(8))


def test_peira_representation_is_causal_restorable_and_matched() -> None:
    import torch

    windows = _tiny_windows(pair_count=4, transition_count=5)
    aligned = PeiraRepresentation(
        PeiraConfig(
            objective="aligned_peira",
            steps=2,
            expected_pair_count=4,
            width=8,
            head_count=2,
            feedforward_width=16,
            projector_width=16,
            warmup_steps=1,
        )
    )
    deranged = PeiraRepresentation(
        PeiraConfig(
            objective="deranged_peira",
            steps=2,
            expected_pair_count=4,
            width=8,
            head_count=2,
            feedforward_width=16,
            projector_width=16,
            warmup_steps=1,
        )
    )
    torch.manual_seed(8871)
    before = torch.random.get_rng_state()
    aligned.fit(windows)
    after = torch.random.get_rng_state()
    deranged.fit(windows)
    after_deranged = torch.random.get_rng_state()

    encoded = aligned.encode(windows.histories[:3], windows.graph)
    restored = PeiraRepresentation.from_dict(aligned.to_dict())
    replay = restored.encode(windows.histories[:3], windows.graph)
    inference = PeiraRepresentation.from_inference_dict(
        aligned.to_inference_dict()
    )
    inference_values = inference.encode(
        windows.histories[:3], windows.graph
    )
    malformed_kind = aligned.to_inference_dict()
    malformed_kind["kind"] = "not_peira"
    with pytest.raises(ValueError):
        PeiraRepresentation.from_inference_dict(malformed_kind)
    leaked_training_state = aligned.to_inference_dict()
    leaked_training_state["projector_state"] = {}
    with pytest.raises(ValueError):
        PeiraRepresentation.from_inference_dict(leaked_training_state)

    np.testing.assert_array_equal(before.numpy(), after.numpy())
    np.testing.assert_array_equal(after.numpy(), after_deranged.numpy())
    np.testing.assert_allclose(encoded.tokens, replay.tokens, atol=1e-7)
    np.testing.assert_allclose(
        encoded.tokens, inference_values.tokens, atol=1e-7
    )
    assert aligned.training_parameter_count == (
        deranged.training_parameter_count
    )
    assert aligned.inference_parameter_count == (
        deranged.inference_parameter_count
    )
    aligned_signal, aligned_noise = aligned.final_moments
    deranged_signal, deranged_noise = deranged.final_moments
    assert not np.allclose(aligned_signal, deranged_signal)
    assert not np.allclose(aligned_noise, deranged_noise)
    assert not np.allclose(
        encoded.tokens,
        deranged.encode(windows.histories[:3], windows.graph).tokens,
    )
    assert "projector_state" not in aligned.to_inference_dict()
    assert "running_signal" not in aligned.to_inference_dict()
    assert aligned.training_evidence["running_signal"].shape == (
        2,
        8,
        8,
    )


def test_peira_assessment_rejects_failed_mechanism() -> None:
    names = (
        "aligned_peira",
        "deranged_peira",
        "complete_lejepa",
        "masked_autoencoder",
        "matched_pca",
    )
    scores = {
        name: {
            role: {
                "overall_mse": 1.0,
                "action_overlap_mse": 1.0,
                "downstream_effect_mse": (
                    0.8 if name == "aligned_peira" else 1.0
                ),
            }
            for role in ("selection", "transfer_evaluation")
        }
        for name in names
    }
    assessment = assess_peira_gates(
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
            "matrix_numerics": True,
            "noncollapsed": False,
            "trace_objective_advantage": True,
            "eigenvector_alignment_advantage": True,
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
                "role_contract_recomputes",
                "capacity_recomputes",
                "public_inference_is_causal",
                "all_schedules_recompute",
                "training_moments_recompute",
                "final_operators_recompute",
                "varying_entity_mask_recomputes",
                "copied_source_assessor_recomputes",
                "selection_only_ridge_choice_recomputes",
                "selection_safety_status_recomputes",
                "bundle_size_recomputes",
                "latency_recomputes",
                "copied_prior_controls_match",
            )
        },
        parameter_counts={
            name: {"training": 10, "inference": 5}
            for name in ("aligned_peira", "deranged_peira")
        },
        transfer_pair_errors={
            name: {
                "pair-0": 0.8 if name == "aligned_peira" else 1.0
            }
            for name in names
        },
        deployed_bundle_bytes=1024,
        median_latency_ms=1.0,
    )

    assert not assessment["passed"]
    assert not assessment["mechanism_gates"]["noncollapsed"]
    assert assessment["decision"] == "reject_peira_recipe"


def test_peira_moment_replay_accepts_float32_loss_rounding() -> None:
    from lab.action_dynamics.prototype_peira_assessor import (
        PEIRA_NAMES,
        recompute_peira_training_moments,
    )

    configs = {
        name: PeiraConfig(
            objective=name,
            steps=1,
            width=8,
            head_count=2,
            feedforward_width=16,
            projector_width=16,
            warmup_steps=1,
        ).to_dict()
        for name in PEIRA_NAMES
    }
    arrays = {}
    for name in PEIRA_NAMES:
        for field in (
            "batch_signal",
            "batch_noise",
            "running_signal",
            "running_noise",
        ):
            arrays[f"training__{name}__{field}"] = np.zeros(
                (1, 8, 8), dtype=np.float64
            )
        arrays[f"training__{name}__eta"] = np.ones(
            1, dtype=np.float64
        )
        for field in (
            "auxiliary_value",
            "trace_objective",
            "trace_predictor",
            "symmetry_error",
            "solve_residual",
        ):
            arrays[f"training__{name}__{field}"] = np.zeros(
                1, dtype=np.float64
            )
        arrays[f"training__{name}__loss"] = np.asarray(
            [2.2e-7], dtype=np.float64
        )
        arrays[f"training__{name}__condition_number"] = np.ones(
            1, dtype=np.float64
        )

    assert recompute_peira_training_moments(
        {"configs": configs}, arrays
    )


def test_peira_smoke_artifact_reassesses(tmp_path: Path) -> None:
    from lab.action_dynamics.prototype_peira import (
        FROZEN_CACHE,
        FROZEN_PRIOR_CONTROL,
        run_experiment,
    )
    from lab.action_dynamics.prototype_peira_assessor import (
        verify_stored_assessment,
    )

    output = tmp_path / "peira-smoke"
    run_experiment(
        cache_directory=FROZEN_CACHE,
        prior_control_directory=FROZEN_PRIOR_CONTROL,
        output_directory=output,
        steps=1,
        latency_repetitions=1,
        allow_noninterpretable_smoke=True,
    )
    assessment = verify_stored_assessment(output)

    assert assessment["decision"] == "non_interpretable_peira_smoke"
    assert all(assessment["protocol_checks"].values())
    reproduction = output / "reproduction-source"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (
            str(reproduction / "src"),
            str(reproduction / "lab/action_dynamics"),
        )
    )
    copied = subprocess.run(
        [
            sys.executable,
            "-I",
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
                / "lab/action_dynamics/prototype_peira_assessor.py"
            ),
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
        cwd=tmp_path,
    )
    assert "non_interpretable_peira_smoke" in copied.stdout


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
                        base + 0.2 + np.arange(3)[:, None] * 0.01
                    ).reshape(3)
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
