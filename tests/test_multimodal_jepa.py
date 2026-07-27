import json

import numpy as np

from quantis_core.multimodal_corpus import MultimodalModelWindows
from quantis_core.multimodal_world_model import (
    MultimodalJepaWorldModelDetector,
)
from quantis_core.windowing import WindowCompiler


def test_multimodal_jepa_is_deterministic_and_roundtrips() -> None:
    training, validation = _multimodal_windows()

    first = MultimodalJepaWorldModelDetector(
        metric_latent_dimension=2,
        log_latent_dimension=2,
        epochs=120,
        learning_rate=0.025,
        ema_decay=0.96,
        calibration_quantile=0.95,
        seed=41,
    ).fit(training)
    second = MultimodalJepaWorldModelDetector(
        metric_latent_dimension=2,
        log_latent_dimension=2,
        epochs=120,
        learning_rate=0.025,
        ema_decay=0.96,
        calibration_quantile=0.95,
        seed=41,
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
    assert first.training_losses[-1] < first.training_losses[0]

    expected = first.score(validation)
    log_shifted = MultimodalModelWindows(
        metric=validation.metric,
        logs=type(validation.logs)(
            contexts=validation.logs.contexts + 2.0,
            targets=validation.logs.targets + 2.0,
            point_indices=validation.logs.point_indices,
            feature_names=validation.logs.feature_names,
        ),
    )
    shifted = first.score(log_shifted)
    restored = MultimodalJepaWorldModelDetector.from_dict(
        first.to_dict()
    )
    actual = restored.score(validation)

    np.testing.assert_allclose(actual.scores, expected.scores)
    np.testing.assert_allclose(
        actual.feature_evidence,
        expected.feature_evidence,
    )
    np.testing.assert_allclose(
        actual.signed_feature_evidence,
        expected.signed_feature_evidence,
    )
    assert actual.feature_evidence.shape == (
        len(validation.metric.targets),
        5,
    )
    assert np.max(np.abs(shifted.scores - expected.scores)) > 1e-6
    assert actual.threshold == expected.threshold


def _multimodal_windows() -> tuple[
    MultimodalModelWindows,
    MultimodalModelWindows,
]:
    time = np.arange(180, dtype=np.float64)
    metric_values = np.column_stack(
        (
            np.sin(time / 8.0),
            np.cos(time / 11.0),
            0.6 * np.sin(time / 8.0 + 0.2),
        )
    )
    log_values = np.column_stack(
        (
            2.0 + np.sin(time / 8.0),
            2.0 + np.cos(time / 11.0),
        )
    )
    metric_compiler = WindowCompiler(6).fit(metric_values[:100])
    log_compiler = WindowCompiler(6).fit(log_values[:100])

    def windows(start: int, stop: int) -> MultimodalModelWindows:
        return MultimodalModelWindows(
            metric=metric_compiler.transform(
                metric_values[start:stop],
                ("queue", "worker", "database"),
            ),
            logs=log_compiler.transform(
                log_values[start:stop],
                ("accepted", "completed"),
            ),
        )

    return windows(0, 100), windows(100, 180)
