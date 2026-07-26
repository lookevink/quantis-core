import numpy as np

from quantis_core.windowing import WindowCompiler


def test_window_compiler_aligns_history_with_the_next_target():
    telemetry = np.asarray(
        [
            [10.0, 100.0],
            [20.0, 200.0],
            [30.0, 300.0],
            [40.0, 400.0],
            [50.0, 500.0],
        ]
    )
    compiler = WindowCompiler(lookback=3).fit(telemetry)

    windows = compiler.transform(telemetry, feature_names=("a", "b"))

    assert windows.contexts.shape == (2, 3, 2)
    assert windows.targets.shape == (2, 2)
    np.testing.assert_array_equal(windows.point_indices, np.asarray([3, 4]))
    np.testing.assert_allclose(windows.contexts[0, -1], [0.0, 0.0])
    np.testing.assert_allclose(windows.targets[0], [0.67449076, 0.67449076])
    assert windows.feature_names == ("a", "b")


def test_fitted_preprocessing_round_trips_without_changing_windows():
    training = np.asarray(
        [
            [1.0, 20.0],
            [2.0, 21.0],
            [3.0, 22.0],
            [4.0, 23.0],
            [1000.0, 24.0],
        ]
    )
    compiler = WindowCompiler(lookback=2).fit(training)
    restored = WindowCompiler.from_dict(compiler.to_dict())

    original_windows = compiler.transform(training)
    restored_windows = restored.transform(training)

    np.testing.assert_allclose(original_windows.contexts, restored_windows.contexts)
    np.testing.assert_allclose(original_windows.targets, restored_windows.targets)
    assert restored.to_dict() == compiler.to_dict()
    assert compiler.to_dict()["location"] == [3.0, 22.0]


def test_window_compiler_rejects_non_finite_telemetry():
    compiler = WindowCompiler(lookback=2)

    try:
        compiler.fit(np.asarray([[1.0], [np.nan], [3.0]]))
    except ValueError as error:
        assert "finite" in str(error)
    else:
        raise AssertionError("non-finite telemetry should be rejected")
