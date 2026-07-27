import json

import numpy as np
import pytest

from quantis_core.detectors import (
    JepaWorldModelDetector,
    detector_from_dict,
)
from quantis_core.windowing import ModelWindows, WindowCompiler


def test_jepa_training_is_deterministic_and_artifact_roundtrips():
    training, normal, _ = _world_model_windows()

    first = JepaWorldModelDetector(
        latent_dimension=2,
        epochs=160,
        learning_rate=0.03,
        ema_decay=0.96,
        calibration_quantile=0.95,
        seed=17,
    ).fit(training)
    second = JepaWorldModelDetector(
        latent_dimension=2,
        epochs=160,
        learning_rate=0.03,
        ema_decay=0.96,
        calibration_quantile=0.95,
        seed=17,
    ).fit(training)

    first_bytes = json.dumps(
        first.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    second_bytes = json.dumps(
        second.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    assert first_bytes == second_bytes
    assert len(first.training_losses) == 160
    assert first.training_losses[-1] < first.training_losses[0]

    expected = first.score(normal)
    restored = detector_from_dict(first.to_dict())
    actual = restored.score(normal)

    np.testing.assert_allclose(actual.scores, expected.scores)
    np.testing.assert_allclose(
        actual.feature_evidence,
        expected.feature_evidence,
    )
    np.testing.assert_allclose(
        actual.signed_feature_evidence,
        expected.signed_feature_evidence,
    )
    assert actual.threshold == expected.threshold


def test_jepa_rejects_reordered_feature_schema():
    training, normal, _ = _world_model_windows()
    detector = JepaWorldModelDetector(
        latent_dimension=2,
        epochs=20,
        seed=31,
    ).fit(training)
    reordered = ModelWindows(
        contexts=normal.contexts[:, :, ::-1],
        targets=normal.targets[:, ::-1],
        point_indices=normal.point_indices,
        feature_names=normal.feature_names[::-1],
    )

    with pytest.raises(ValueError, match="window features"):
        detector.score(reordered)


def test_jepa_scores_correlated_future_drift_above_normal_dynamics():
    training, normal, drift = _world_model_windows()
    detector = JepaWorldModelDetector(
        latent_dimension=2,
        epochs=200,
        learning_rate=0.03,
        ema_decay=0.96,
        calibration_quantile=0.95,
        seed=23,
    ).fit(training)

    normal_scores = detector.score(normal)
    drift_scores = detector.score(drift)

    assert np.median(drift_scores.scores[-20:]) > (
        3.0 * np.median(normal_scores.scores)
    )
    assert np.mean(drift_scores.alerts[-20:]) > 0.8
    assert drift_scores.feature_evidence.shape == drift.targets.shape


def _world_model_windows():
    time = np.arange(180, dtype=np.float64)
    values = np.column_stack(
        (
            np.sin(time / 8.0),
            0.7 * np.sin(time / 8.0 + 0.3),
            np.cos(time / 11.0),
        )
    )
    compiler = WindowCompiler(lookback=6).fit(values[:100])
    training = compiler.transform(
        values[:100],
        ("queue", "worker", "database"),
    )
    normal = compiler.transform(
        values[100:],
        ("queue", "worker", "database"),
    )
    drift_values = values[100:].copy()
    drift_values[45:] += np.asarray([4.0, -3.0, 2.5])
    drift = compiler.transform(
        drift_values,
        ("queue", "worker", "database"),
    )
    return training, normal, drift
