import numpy as np

from quantis_core.edge_dynamics.action_conditioned_jepa import (
    ActionConditionedJepaConfig,
    ActionConditionedJepaDynamics,
    sketched_isotropic_gaussian_regularization,
)
from quantis_core.action_conditioned_dynamics import (
    ActionTrajectoryCompiler,
)
from quantis_core.action_dynamics_synthetic import synthetic_action_runs


def test_sigreg_matches_the_official_collapsed_input_statistic() -> None:
    import torch

    embeddings = torch.zeros((2, 4, 3), dtype=torch.float64)
    generator = torch.Generator(device="cpu").manual_seed(5)

    loss = sketched_isotropic_gaussian_regularization(
        embeddings,
        generator=generator,
        sketch_dimension=5,
        knot_count=17,
    )

    np.testing.assert_allclose(
        float(loss),
        1.6081903189266582,
        rtol=1e-12,
        atol=1e-12,
    )


def test_sigreg_has_finite_gradients_for_entity_preserving_tokens() -> None:
    import torch

    value_generator = torch.Generator(device="cpu").manual_seed(17)
    sketch_generator = torch.Generator(device="cpu").manual_seed(29)
    embeddings = torch.randn(
        (3, 8, 4),
        dtype=torch.float64,
        generator=value_generator,
        requires_grad=True,
    )

    loss = sketched_isotropic_gaussian_regularization(
        embeddings,
        generator=sketch_generator,
        sketch_dimension=11,
        knot_count=17,
    )
    loss.backward()

    assert loss.ndim == 0
    assert embeddings.grad is not None
    assert torch.all(torch.isfinite(embeddings.grad))


def test_sigreg_uses_the_batch_axis_with_shared_entity_sketches() -> None:
    import torch

    embeddings = torch.tensor(
        [
            [
                [-1.0, 0.0, 1.0],
                [0.5, -0.5, 0.25],
                [1.5, 0.75, -1.0],
                [-0.25, 1.25, 0.5],
            ],
            [
                [0.1, 0.2, 0.3],
                [-0.4, -0.2, 0.0],
                [0.9, -1.1, 0.7],
                [1.2, 0.4, -0.8],
            ],
        ],
        dtype=torch.float64,
    )
    generator = torch.Generator(device="cpu").manual_seed(23)

    loss = sketched_isotropic_gaussian_regularization(
        embeddings,
        generator=generator,
        sketch_dimension=5,
        knot_count=17,
    )

    np.testing.assert_allclose(
        float(loss),
        0.542579829731339,
        rtol=1e-12,
        atol=1e-12,
    )


def test_sigreg_sketch_sequence_is_explicit_and_restartable() -> None:
    import torch

    values = torch.arange(96, dtype=torch.float64).reshape(3, 8, 4) / 20
    first_generator = torch.Generator(device="cpu").manual_seed(53)
    second_generator = torch.Generator(device="cpu").manual_seed(53)

    first_sequence = [
        float(
            sketched_isotropic_gaussian_regularization(
                values,
                generator=first_generator,
                sketch_dimension=11,
            )
        )
        for _ in range(2)
    ]
    second_sequence = [
        float(
            sketched_isotropic_gaussian_regularization(
                values,
                generator=second_generator,
                sketch_dimension=11,
            )
        )
        for _ in range(2)
    ]

    np.testing.assert_allclose(first_sequence, second_sequence)
    assert first_sequence[0] != first_sequence[1]


def test_action_conditioned_jepa_config_roundtrips_sigreg() -> None:
    config = ActionConditionedJepaConfig(
        regularizer="sigreg",
        sigreg_weight=0.02,
        sigreg_sketch_dimension=256,
        sigreg_knot_count=17,
        sigreg_projection_seed=1401,
        device="cpu",
    )

    restored = ActionConditionedJepaConfig.from_dict(config.to_dict())

    assert restored == config


def test_action_conditioned_jepa_fits_and_restores_with_sigreg() -> None:
    runs = synthetic_action_runs(8, split="training", seed=1400)
    windows = ActionTrajectoryCompiler(
        context_length=5, rollout_horizon=3
    ).fit(runs).transform(runs)
    config = ActionConditionedJepaConfig(
        node_latent_dimension=2,
        transition_rank=3,
        epochs=1,
        batch_size=256,
        regularizer="sigreg",
        sigreg_weight=0.02,
        sigreg_sketch_dimension=11,
        sigreg_knot_count=17,
        sigreg_projection_seed=1401,
        device="cpu",
        seed=19,
    )

    model = ActionConditionedJepaDynamics(config).fit(windows)
    restored = ActionConditionedJepaDynamics.from_dict(model.to_dict())
    first = model.rollout(
        windows.histories[:4],
        windows.future_controls[:4],
        windows.future_actions[:4],
        windows.graph,
    )
    second = restored.rollout(
        windows.histories[:4],
        windows.future_controls[:4],
        windows.future_actions[:4],
        windows.graph,
    )

    assert np.isfinite(model.training_metrics[-1]["sigreg"])
    assert model.training_metrics[-1]["sigreg"] > 0.0
    np.testing.assert_allclose(first.mean, second.mean, atol=1e-6)
    np.testing.assert_allclose(first.variance, second.variance)
