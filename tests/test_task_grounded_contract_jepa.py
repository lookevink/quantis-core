import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from lab.action_dynamics.prototype_task_grounded_contract_jepa import (
    FROZEN_CACHE,
    run_experiment,
)
from lab.action_dynamics.prototype_task_grounded_contract_jepa_assessor import (
    assess_stored_bundle,
    verify_artifact_manifest,
    verify_stored_assessment,
)
from quantis_core.edge_dynamics.models import (
    ContractiveLowRankDynamics,
    LowRankConfig,
)
from quantis_core.edge_dynamics.task_grounded_contract_jepa import (
    CONTRACT_JEPA_OBJECTIVES,
    TaskGroundedContractConfig,
    TaskGroundedContractDynamics,
    TaskGroundedContractJepa,
)
from tests.test_sd_jepa import tiny_action_conditioned_windows


def test_contract_cells_are_bounded_sufficient_restorable_and_matched() -> None:
    fit = tiny_action_conditioned_windows(
        pair_count=4, transition_count=6
    )
    selection = tiny_action_conditioned_windows(
        pair_count=2,
        transition_count=6,
        pair_prefix="selection",
    )
    raw = ContractiveLowRankDynamics(LowRankConfig(rank=8)).fit(fit)
    base = TaskGroundedContractConfig(
        width=8,
        hidden_width=16,
        pretrain_steps=2,
        checkpoint_interval=1,
        expected_pair_count=4,
    )
    models = {}
    for objective in CONTRACT_JEPA_OBJECTIVES:
        model = TaskGroundedContractJepa(
            replace(base, objective=objective)
        )
        model.fit(fit, raw).select(selection, raw)
        models[objective] = model

    assert len(
        {model.training_parameter_count for model in models.values()}
    ) == 1
    assert len(
        {model.inference_parameter_count for model in models.values()}
    ) == 1
    for contract in models.values():
        for row in contract.selection_metrics:
            assert row["selection_objective"] == pytest.approx(
                row["residual"]
                + base.paired_effect_weight * row["paired_effect"]
            )

    model = models["task_grounded_contract_jepa"]
    encoded = model.encode_contract(fit.histories[:3], fit.graph)
    np.testing.assert_array_equal(
        encoded.raw_current_state, fit.histories[:3, -1]
    )
    assert encoded.learned_tokens.shape == (3, 7, 8)

    correction, score = model.predict_contract(
        selection.histories[:3],
        selection.future_controls[:3],
        selection.future_actions[:3],
        selection.graph,
    )
    assert correction.shape == selection.future_states[:3].shape
    assert score.shape == selection.future_states[:3].shape[:2]
    assert np.all(score >= 0.0)
    assert np.all(
        np.abs(correction)
        <= model.correction_bound[None, None] + 1e-8
    )

    restored = TaskGroundedContractJepa.from_dict(model.to_dict())
    restored_correction, restored_score = restored.predict_contract(
        selection.histories[:3],
        selection.future_controls[:3],
        selection.future_actions[:3],
        selection.graph,
    )
    np.testing.assert_allclose(
        correction, restored_correction, atol=1e-7
    )
    np.testing.assert_allclose(score, restored_score, atol=1e-7)

    with pytest.raises(TypeError):
        model.predict_contract(  # type: ignore[call-arg]
            selection.histories[:3],
            selection.future_controls[:3],
            selection.future_actions[:3],
            selection.graph,
            future_states=selection.future_states[:3],
        )

    corrupted = copy.deepcopy(model.to_dict())
    corrupted["state_dict"]["online_encoder.input.weight"]["values"][0][0] = (
        float("nan")
    )
    with pytest.raises(ValueError, match="non-finite"):
        TaskGroundedContractJepa.from_dict(corrupted)


def test_contract_gain_zero_is_raw_and_composed_model_restores() -> None:
    fit = tiny_action_conditioned_windows(
        pair_count=4, transition_count=6
    )
    selection = tiny_action_conditioned_windows(
        pair_count=2,
        transition_count=6,
        pair_prefix="selection",
    )
    raw = ContractiveLowRankDynamics(LowRankConfig(rank=8)).fit(fit)
    branch = TaskGroundedContractJepa(
        TaskGroundedContractConfig(
            width=8,
            hidden_width=16,
            pretrain_steps=2,
            checkpoint_interval=1,
            expected_pair_count=4,
        )
    ).fit(fit, raw).select(selection, raw)
    composed = TaskGroundedContractDynamics(raw, branch, gain=0.0)
    expected = raw.rollout(
        selection.histories[:3],
        selection.future_controls[:3],
        selection.future_actions[:3],
        selection.graph,
    )
    actual = composed.rollout(
        selection.histories[:3],
        selection.future_controls[:3],
        selection.future_actions[:3],
        selection.graph,
    )
    np.testing.assert_array_equal(expected.mean, actual.mean)
    np.testing.assert_array_equal(expected.variance, actual.variance)

    composed.set_gain(0.5)
    restored = TaskGroundedContractDynamics.from_dict(composed.to_dict())
    changed = composed.rollout(
        selection.histories[:3],
        selection.future_controls[:3],
        selection.future_actions[:3],
        selection.graph,
    )
    replay = restored.rollout(
        selection.histories[:3],
        selection.future_controls[:3],
        selection.future_actions[:3],
        selection.graph,
    )
    np.testing.assert_allclose(changed.mean, replay.mean, atol=1e-7)
    np.testing.assert_allclose(changed.variance, replay.variance, atol=1e-12)


def test_contract_smoke_artifact_reassesses_from_stored_evidence(
    tmp_path: Path,
) -> None:
    output = tmp_path / "artifact"
    run_experiment(
        cache_directory=FROZEN_CACHE,
        output_directory=output,
        pretrain_steps=1,
        latency_repetitions=1,
        allow_noninterpretable_smoke=True,
        expected_pair_count=40,
    )

    assessment = assess_stored_bundle(output)
    assert assessment["eligible_for_advance"] is False
    assert assessment["passed"] is False
    assert (
        assessment["decision"]
        == "non_interpretable_task_grounded_contract_jepa_smoke"
    )
    assert assessment["safety_gates"]["restoration_arrays_match"] is True
    assert (
        assessment["safety_gates"][
            "restored_alert_decisions_match"
        ]
        is True
    )
    verify_stored_assessment(output)
    verify_artifact_manifest(output)
