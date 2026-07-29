import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from lab.action_dynamics.prototype_pair_effect_jepa import (
    FROZEN_CACHE,
    run_experiment,
)
from lab.action_dynamics.prototype_pair_effect_jepa_assessor import (
    assess_stored_bundle,
    verify_artifact_manifest,
    verify_stored_assessment,
)
from quantis_core.edge_dynamics.pair_effect_jepa import (
    PAIR_EFFECT_OBJECTIVES,
    PairEffectCorrectedDynamics,
    PairEffectJepaConfig,
    PairEffectJepaModel,
)
from quantis_core.edge_dynamics.models import (
    ContractiveLowRankDynamics,
    LowRankConfig,
)
from tests.test_sd_jepa import tiny_action_conditioned_windows


def test_pair_effect_cells_restore_match_capacity_and_zero_no_action() -> None:
    fit = tiny_action_conditioned_windows(
        pair_count=4, transition_count=6
    )
    selection = tiny_action_conditioned_windows(
        pair_count=2,
        transition_count=6,
        pair_prefix="selection",
    )
    base = PairEffectJepaConfig(
        width=8,
        hidden_width=16,
        pretrain_steps=2,
        checkpoint_interval=1,
        expected_pair_count=4,
    )
    models = {}
    for objective in PAIR_EFFECT_OBJECTIVES:
        model = PairEffectJepaModel(replace(base, objective=objective))
        model.fit(fit).select(selection)
        models[objective] = model

    assert len(
        {model.training_parameter_count for model in models.values()}
    ) == 1
    assert len(
        {model.inference_parameter_count for model in models.values()}
    ) == 1

    model = models["pair_effect_jepa"]
    treatment = np.asarray(
        [
            np.any(actions[..., 1] > 0.5)
            for actions in selection.future_actions
        ],
        dtype=np.bool_,
    )
    row = int(np.flatnonzero(treatment)[0])
    effect = model.predict_effect(
        selection.histories[row : row + 1],
        selection.future_controls[row : row + 1],
        selection.future_actions[row : row + 1],
        selection.graph,
    )
    assert effect.shape == selection.future_states[row : row + 1].shape
    assert np.all(np.isfinite(effect))

    no_action = np.zeros_like(selection.future_actions[row : row + 1])
    no_action[..., 0] = 1.0
    zero = model.predict_effect(
        selection.histories[row : row + 1],
        selection.future_controls[row : row + 1],
        no_action,
        selection.graph,
    )
    np.testing.assert_allclose(zero, 0.0, atol=1e-12)

    restored = PairEffectJepaModel.from_dict(model.to_dict())
    restored_effect = restored.predict_effect(
        selection.histories[row : row + 1],
        selection.future_controls[row : row + 1],
        selection.future_actions[row : row + 1],
        selection.graph,
    )
    np.testing.assert_allclose(effect, restored_effect, atol=1e-7)

    with pytest.raises(TypeError):
        model.predict_effect(  # type: ignore[call-arg]
            selection.histories[row : row + 1],
            selection.future_controls[row : row + 1],
            selection.future_actions[row : row + 1],
            selection.graph,
            future_states=selection.future_states[row : row + 1],
        )

    corrupted = copy.deepcopy(model.to_dict())
    corrupted["state_dict"]["online_encoder.input.weight"]["values"][0][0] = (
        float("nan")
    )
    with pytest.raises(ValueError, match="non-finite"):
        PairEffectJepaModel.from_dict(corrupted)


def test_pair_effect_composition_preserves_raw_path_and_restores() -> None:
    fit = tiny_action_conditioned_windows(
        pair_count=4, transition_count=6
    )
    selection = tiny_action_conditioned_windows(
        pair_count=2,
        transition_count=6,
        pair_prefix="selection",
    )
    effect = PairEffectJepaModel(
        PairEffectJepaConfig(
            width=8,
            hidden_width=16,
            pretrain_steps=2,
            checkpoint_interval=1,
            expected_pair_count=4,
        )
    ).fit(fit).select(selection)
    raw = ContractiveLowRankDynamics(LowRankConfig(rank=8)).fit(fit)
    composed = PairEffectCorrectedDynamics(raw, effect)

    rows = slice(0, 3)
    actual = composed.rollout(
        selection.histories[rows],
        selection.future_controls[rows],
        selection.future_actions[rows],
        selection.graph,
    )
    restored = PairEffectCorrectedDynamics.from_dict(composed.to_dict())
    replay = restored.rollout(
        selection.histories[rows],
        selection.future_controls[rows],
        selection.future_actions[rows],
        selection.graph,
    )
    np.testing.assert_allclose(actual.mean, replay.mean, atol=1e-7)
    np.testing.assert_allclose(actual.variance, replay.variance, atol=1e-12)
    assert composed.parameter_count == (
        raw.parameter_count + effect.inference_parameter_count
    )


def test_pair_effect_smoke_artifact_reassesses_from_stored_evidence(
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
        == "non_interpretable_pair_effect_jepa_smoke"
    )
    assert assessment["safety_gates"]["restoration_arrays_match"] is True
    verify_stored_assessment(output)
    verify_artifact_manifest(output)
