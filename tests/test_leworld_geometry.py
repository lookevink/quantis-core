import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from lab.action_dynamics.prototype_leworld_geometry import (
    FROZEN_CACHE,
    run_experiment,
)
from lab.action_dynamics.prototype_leworld_geometry_assessor import (
    assess_stored_bundle,
    verify_artifact_manifest,
    verify_stored_assessment,
)
from quantis_core.edge_dynamics.leworld_geometry import (
    LEWORLD_GEOMETRY_OBJECTIVES,
    SPHERICAL_OBJECTIVES,
    LeWorldGeometryConfig,
    LeWorldGeometryModel,
    gaussian_prior_mmd,
    sliced_wasserstein_distance,
    sphere_heat_uniformity,
)
from tests.test_sd_jepa import _tiny_windows


def test_leworld_geometry_regularizers_prefer_reference_geometry() -> None:
    generator = torch.Generator().manual_seed(17)
    gaussian = torch.randn(512, 8, generator=generator)
    collapsed = torch.zeros_like(gaussian)
    assert gaussian_prior_mmd(gaussian) < gaussian_prior_mmd(collapsed)

    sphere = torch.nn.functional.normalize(
        torch.randn(512, 8, generator=generator), dim=-1
    )
    sphere_collapsed = torch.zeros_like(sphere)
    sphere_collapsed[:, 0] = 1.0
    assert sphere_heat_uniformity(sphere) < sphere_heat_uniformity(
        sphere_collapsed
    )

    directions = torch.nn.functional.normalize(
        torch.randn(8, 32, generator=generator), dim=0
    )
    assert sliced_wasserstein_distance(
        gaussian, gaussian, directions
    ).item() == pytest.approx(0.0)
    assert sliced_wasserstein_distance(
        gaussian + 2.0, gaussian, directions
    ) > 0.0


def test_leworld_geometry_cells_restore_and_match_capacity() -> None:
    fitting = _tiny_windows(pair_count=4, transition_count=6)
    selection = _tiny_windows(
        pair_count=2, transition_count=6, pair_prefix="selection"
    )
    base = LeWorldGeometryConfig(
        width=8,
        hidden_width=16,
        pretrain_steps=2,
        checkpoint_interval=1,
        sketch_dimension=8,
        knot_count=5,
        subspace_count=2,
        expected_pair_count=4,
    )
    models = {}
    for objective in LEWORLD_GEOMETRY_OBJECTIVES:
        model = LeWorldGeometryModel(replace(base, objective=objective))
        model.fit(fitting).select(selection)
        models[objective] = model

    assert len(
        {model.training_parameter_count for model in models.values()}
    ) == 1
    assert len(
        {model.inference_parameter_count for model in models.values()}
    ) == 1

    for objective, model in models.items():
        encoded = model.encode(fitting.histories[:3], fitting.graph)
        restored = LeWorldGeometryModel.from_dict(model.to_dict())
        restored_encoded = restored.encode(
            fitting.histories[:3], fitting.graph
        )
        np.testing.assert_allclose(
            encoded.tokens, restored_encoded.tokens, atol=1e-7
        )
        np.testing.assert_allclose(
            encoded.scene_history,
            restored_encoded.scene_history,
            atol=1e-7,
        )
        assert encoded.tokens.shape == (3, 7, 8)
        assert encoded.scene_history.shape == (3, 20, 8)
        if objective == "rectified_lp":
            assert np.all(encoded.tokens >= 0.0)
        if objective in SPHERICAL_OBJECTIVES:
            np.testing.assert_allclose(
                np.linalg.norm(encoded.tokens, axis=-1),
                1.0,
                atol=1e-6,
            )

    with pytest.raises(TypeError):
        models["lewm_ambient"].encode(  # type: ignore[call-arg]
            fitting.histories[:3],
            fitting.graph,
            future_states=fitting.future_states[:3],
        )

    corrupted = copy.deepcopy(models["lewm_ambient"].to_dict())
    corrupted["state_dict"]["encoder.input_fc.weight"]["values"][0][0] = (
        float("nan")
    )
    with pytest.raises(ValueError, match="non-finite"):
        LeWorldGeometryModel.from_dict(corrupted)


def test_leworld_geometry_smoke_reassesses_from_stored_arrays(
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
        == "non_interpretable_leworld_geometry_smoke"
    )
    verify_stored_assessment(output)
    verify_artifact_manifest(output)
