import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from lab.action_dynamics.prototype_delta_jepa import (
    FROZEN_CACHE,
    run_experiment,
)
from lab.action_dynamics.prototype_delta_jepa_assessor import (
    assess_stored_bundle,
    verify_artifact_manifest,
    verify_stored_assessment,
)
from quantis_core.edge_dynamics.complete_lejepa import (
    PairBlockedAnchorSchedule,
)
from quantis_core.edge_dynamics.delta_jepa import (
    DeltaJepaConfig,
    DeltaJepaModel,
    action_decoder_input,
)
from tests.test_sd_jepa import _tiny_windows


def test_delta_jepa_displacement_input_removes_endpoint_translation() -> None:
    rng = np.random.default_rng(16016)
    start = rng.normal(size=(5, 7, 8))
    end = start + rng.normal(size=start.shape)
    translation = rng.normal(size=(1, 7, 8))

    original = action_decoder_input(start, end, objective="delta_jepa")
    translated = action_decoder_input(
        start + translation,
        end + translation,
        objective="delta_jepa",
    )
    np.testing.assert_allclose(original, translated, atol=1e-12)

    endpoint_original = action_decoder_input(
        start, end, objective="endpoint_concat"
    )
    endpoint_translated = action_decoder_input(
        start + translation,
        end + translation,
        objective="endpoint_concat",
    )
    assert original.shape == endpoint_original.shape
    assert not np.allclose(endpoint_original, endpoint_translated)


def test_delta_jepa_cells_match_capacity_restore_and_are_causal() -> None:
    fit = _tiny_windows(pair_count=4, transition_count=6)
    selection = _tiny_windows(
        pair_count=2, transition_count=6, pair_prefix="selection"
    )
    base = DeltaJepaConfig(
        width=8,
        hidden_width=16,
        decoder_hidden_width=16,
        decoder_heads=4,
        pretrain_steps=2,
        checkpoint_interval=1,
        expected_pair_count=4,
    )
    models = {}
    for objective in (
        "delta_jepa",
        "endpoint_concat",
        "prediction_only",
    ):
        model = DeltaJepaModel(replace(base, objective=objective))
        model.fit(fit).select(selection)
        models[objective] = model

    assert len(
        {model.training_parameter_count for model in models.values()}
    ) == 1
    assert len(
        {model.inference_parameter_count for model in models.values()}
    ) == 1

    model = models["delta_jepa"]
    encoded = model.encode(fit.histories[:3], fit.graph)
    restored = DeltaJepaModel.from_dict(model.to_dict())
    restored_encoded = restored.encode(fit.histories[:3], fit.graph)
    np.testing.assert_allclose(
        encoded.tokens, restored_encoded.tokens, atol=1e-7
    )
    assert encoded.tokens.shape == (3, 7, 8)

    diagnosed = model.diagnose_intervals(selection)
    restored_diagnosed = restored.diagnose_intervals(selection)
    np.testing.assert_allclose(
        diagnosed.predicted_actions,
        restored_diagnosed.predicted_actions,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        diagnosed.displacements,
        restored_diagnosed.displacements,
        atol=1e-7,
    )
    assert diagnosed.predicted_actions.shape[:3] == (
        2 * len(selection.histories),
        5,
        7,
    )

    with pytest.raises(TypeError):
        model.encode(  # type: ignore[call-arg]
            fit.histories[:3],
            fit.graph,
            future_states=fit.future_states[:3],
        )

    corrupted = copy.deepcopy(model.to_dict())
    corrupted["state_dict"]["encoder.input_fc.weight"]["values"][0][0] = (
        float("nan")
    )
    with pytest.raises(ValueError, match="non-finite"):
        DeltaJepaModel.from_dict(corrupted)


def test_delta_jepa_pair_schedule_is_pair_atomic() -> None:
    windows = _tiny_windows(pair_count=4, transition_count=6)
    schedule = PairBlockedAnchorSchedule(windows, seed=16017)
    first = schedule.batch(0)
    assert len(first.indices) == 4
    assert len(set(first.pair_ids)) == 4
    assert set(first.arm_ids.tolist()) == {0, 1}


def test_delta_jepa_smoke_artifact_reassesses_from_stored_arrays(
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
        == "non_interpretable_delta_jepa_smoke"
    )
    verify_stored_assessment(output)
    verify_artifact_manifest(output)
