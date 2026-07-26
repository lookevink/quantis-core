import numpy as np

from quantis_core.scenarios import Phase, ScenarioSpec, generate_scenario


def test_scenario_generation_is_reproducible_and_self_describing():
    spec = ScenarioSpec(seed=17, length=240)

    first = generate_scenario(spec)
    second = generate_scenario(spec)
    different_seed = generate_scenario(ScenarioSpec(seed=18, length=240))

    np.testing.assert_array_equal(first.telemetry, second.telemetry)
    np.testing.assert_array_equal(first.phases, second.phases)
    np.testing.assert_array_equal(first.affected_features, second.affected_features)
    assert first.manifest == second.manifest
    assert not np.array_equal(first.telemetry, different_seed.telemetry)

    assert first.telemetry.shape == (240, len(first.feature_names))
    assert len(first.feature_names) >= 10
    assert first.affected_features.shape == first.telemetry.shape
    assert set(np.unique(first.phases)) == {
        Phase.NORMAL.value,
        Phase.ROUTINE_NOISE.value,
        Phase.STRUCTURAL.value,
    }
    assert first.manifest["seed"] == 17
    assert first.manifest["feature_names"] == list(first.feature_names)


def test_annotations_distinguish_isolated_noise_from_correlated_drift():
    scenario = generate_scenario(ScenarioSpec(seed=29, length=300))
    noise_rows = scenario.phases == Phase.ROUTINE_NOISE.value
    structural_rows = scenario.phases == Phase.STRUCTURAL.value

    noise_affected_counts = scenario.affected_features[noise_rows].sum(axis=1)
    structural_affected_counts = scenario.affected_features[structural_rows].sum(axis=1)

    assert noise_rows.sum() >= 8
    assert np.all(noise_affected_counts == 1)
    assert structural_rows.sum() >= 30
    assert np.all(structural_affected_counts == 3)
