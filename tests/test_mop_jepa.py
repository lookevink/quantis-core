from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from lab.action_dynamics.prototype_mop_jepa import (
    FROZEN_CACHE,
    run_experiment,
)
from lab.action_dynamics.prototype_mop_jepa_assessor import (
    assess_stored_bundle,
    verify_artifact_manifest,
    verify_stored_assessment,
)
from quantis_core.edge_dynamics.mop_jepa import (
    ContextFreeCodebookConfig,
    ContextFreeTrajectoryCodebook,
    MopJepaConfig,
    MopJepaModel,
    hard_cosine_assignment,
)
from tests.test_sd_jepa import tiny_action_conditioned_windows


def test_hard_cosine_assignment_updates_only_winner() -> None:
    torch = pytest.importorskip("torch")
    candidates = torch.tensor(
        [[[[[0.8, 0.2]]], [[[0.0, 1.0]]], [[[-1.0, 0.0]]]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    target = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float32)

    assignment = hard_cosine_assignment(candidates, target)
    assignment.loss.backward()

    assert assignment.winner_indices.tolist() == [0]
    assert torch.any(candidates.grad[0, 0] != 0.0)
    assert torch.all(candidates.grad[0, 1:] == 0.0)


def test_mop_cells_restore_match_capacity_and_are_causal(
    tmp_path: Path,
) -> None:
    fitting = tiny_action_conditioned_windows(
        pair_count=4, transition_count=3
    )
    calibration = tiny_action_conditioned_windows(
        pair_count=2,
        transition_count=3,
        pair_prefix="calibration",
    )
    base = MopJepaConfig(
        head_count=2,
        state_latent_width=4,
        context_width=4,
        predictor_width=8,
        epochs=2,
        batch_size=8,
    )
    candidate = MopJepaModel(base).fit(fitting).calibrate(calibration)
    supervised = MopJepaModel(
        replace(base, objective="supervised_hard_wta")
    ).fit(fitting).calibrate(calibration)

    assert (
        candidate.training_parameter_count
        == supervised.training_parameter_count
    )
    expected = candidate.rollout(
        calibration.histories,
        calibration.future_controls,
        calibration.future_actions,
    )
    candidate.save(tmp_path, "candidate")
    restored = MopJepaModel.load(tmp_path, "candidate")
    actual = restored.rollout(
        calibration.histories,
        calibration.future_controls,
        calibration.future_actions,
    )
    np.testing.assert_allclose(
        expected.component_mean, actual.component_mean, atol=1e-6
    )
    np.testing.assert_allclose(
        expected.component_variance,
        actual.component_variance,
        atol=1e-7,
    )
    np.testing.assert_allclose(expected.weight, actual.weight, atol=1e-7)
    assert int(np.sum(candidate.calibration_assignment_count)) == len(
        calibration.histories
    )
    assert 1.0 <= candidate.training_metrics[-1][
        "winner_effective_heads"
    ] <= 2.0
    with pytest.raises(TypeError):
        candidate.rollout(  # type: ignore[call-arg]
            calibration.histories,
            calibration.future_controls,
            calibration.future_actions,
            future_states=calibration.future_states,
        )


def test_context_free_codebook_ignores_context_and_restores(
    tmp_path: Path,
) -> None:
    fitting = tiny_action_conditioned_windows(
        pair_count=4, transition_count=3
    )
    calibration = tiny_action_conditioned_windows(
        pair_count=2,
        transition_count=3,
        pair_prefix="calibration",
    )
    model = ContextFreeTrajectoryCodebook(
        ContextFreeCodebookConfig(component_count=3, iterations=2)
    ).fit(fitting).calibrate(calibration)
    first = model.rollout(
        calibration.histories,
        calibration.future_controls,
        calibration.future_actions,
    )
    second = model.rollout(
        calibration.histories[::-1],
        calibration.future_controls[::-1],
        calibration.future_actions[::-1],
    )
    np.testing.assert_array_equal(
        first.component_mean, second.component_mean
    )
    np.testing.assert_array_equal(first.weight, second.weight)

    model.save(tmp_path, "codebook")
    restored = ContextFreeTrajectoryCodebook.load(tmp_path, "codebook")
    actual = restored.rollout(
        calibration.histories,
        calibration.future_controls,
        calibration.future_actions,
    )
    np.testing.assert_array_equal(
        first.component_mean, actual.component_mean
    )


def test_mop_jepa_smoke_reassesses_from_stored_arrays(
    tmp_path: Path,
) -> None:
    output = tmp_path / "artifact"
    run_experiment(
        cache_directory=FROZEN_CACHE,
        output_directory=output,
        epochs=1,
        codebook_iterations=1,
        latency_repetitions=1,
        allow_noninterpretable_smoke=True,
        evaluation_pair_limit=1,
    )

    assessment = assess_stored_bundle(output)
    assert assessment["eligible_for_advance"] is False
    assert assessment["decision"] == "non_interpretable_mop_jepa_smoke"
    verify_stored_assessment(output)
    verify_artifact_manifest(output)
