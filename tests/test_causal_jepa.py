import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from lab.action_dynamics.prototype_causal_jepa import (
    FROZEN_CACHE,
    run_experiment,
)
from lab.action_dynamics.prototype_causal_jepa_assessor import (
    assess_stored_bundle,
    verify_artifact_manifest,
    verify_stored_assessment,
)
from quantis_core.edge_dynamics.causal_jepa import (
    CAUSAL_JEPA_OBJECTIVES,
    CausalJepaConfig,
    CausalJepaModel,
    causal_mask_plan,
)
from tests.test_sd_jepa import _tiny_windows


def test_causal_jepa_masks_match_budget_and_intervention_shape() -> None:
    entity = causal_mask_plan(
        "causal_entity_mask", step=3, entity_count=7
    )
    coordinate = causal_mask_plan(
        "coordinate_time_mask", step=3, entity_count=7
    )
    prediction = causal_mask_plan(
        "prediction_only", step=3, entity_count=7
    )

    assert entity.shape == coordinate.shape == prediction.shape == (6, 7)
    assert not np.any(entity[0])
    assert not np.any(coordinate[0])
    assert np.sum(entity) == np.sum(coordinate) == 10
    assert np.sum(np.all(entity[1:], axis=0)) == 2
    assert np.sum(np.all(coordinate[1:], axis=0)) < 2
    assert not np.any(prediction)
    np.testing.assert_array_equal(
        entity,
        causal_mask_plan(
            "causal_entity_mask", step=3, entity_count=7
        ),
    )
    assert not np.array_equal(
        entity,
        causal_mask_plan(
            "causal_entity_mask", step=4, entity_count=7
        ),
    )


def test_causal_jepa_cells_restore_match_capacity_and_are_causal() -> None:
    fitting = _tiny_windows(pair_count=4, transition_count=6)
    selection = _tiny_windows(
        pair_count=2, transition_count=6, pair_prefix="selection"
    )
    base = CausalJepaConfig(
        width=8,
        transformer_depth=1,
        attention_heads=2,
        mlp_width=16,
        pretrain_steps=2,
        checkpoint_interval=1,
        expected_pair_count=4,
    )
    models = {}
    for objective in CAUSAL_JEPA_OBJECTIVES:
        model = CausalJepaModel(replace(base, objective=objective))
        model.fit(fitting).select(selection)
        models[objective] = model

    assert len(
        {model.training_parameter_count for model in models.values()}
    ) == 1
    for model in models.values():
        expected = model.predict(
            fitting.histories[:3],
            fitting.future_controls[:3],
            fitting.future_actions[:3],
            fitting.graph,
        )
        completion = model.complete_masked_histories(fitting)
        restored = CausalJepaModel.from_dict(model.to_dict())
        actual = restored.predict(
            fitting.histories[:3],
            fitting.future_controls[:3],
            fitting.future_actions[:3],
            fitting.graph,
        )
        restored_completion = restored.complete_masked_histories(fitting)
        np.testing.assert_allclose(expected, actual, atol=1e-6)
        np.testing.assert_allclose(
            completion.predictions,
            restored_completion.predictions,
            atol=1e-6,
        )
        assert expected.shape == (3, 10, 7, 3)
        assert completion.predictions.shape == (48, 7, 5, 3)

    with pytest.raises(TypeError):
        models["causal_entity_mask"].predict(  # type: ignore[call-arg]
            fitting.histories[:3],
            fitting.future_controls[:3],
            fitting.future_actions[:3],
            fitting.graph,
            future_states=fitting.future_states[:3],
        )

    corrupted = copy.deepcopy(models["causal_entity_mask"].to_dict())
    corrupted["state_dict"]["output.weight"]["values"][0][0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        CausalJepaModel.from_dict(corrupted)


def test_causal_jepa_smoke_reassesses_from_stored_arrays(
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
    assert assessment["decision"] == "non_interpretable_causal_jepa_smoke"
    verify_stored_assessment(output)
    verify_artifact_manifest(output)
