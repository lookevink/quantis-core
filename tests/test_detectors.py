import numpy as np

from quantis_core.detectors import (
    CoherentLatentPredictiveDetector,
    LatentPredictiveDetector,
    PersistenceDetector,
    RobustFeatureDetector,
    detector_from_dict,
)
from quantis_core.windowing import ModelWindows


def windows(contexts, targets):
    context_values = np.asarray(contexts, dtype=np.float64)
    target_values = np.asarray(targets, dtype=np.float64)
    return ModelWindows(
        contexts=context_values,
        targets=target_values,
        point_indices=np.arange(len(target_values), dtype=np.int64),
        feature_names=tuple(
            f"feature_{index}" for index in range(target_values.shape[1])
        ),
    )


def test_persistence_detector_scores_next_point_prediction_error():
    training = windows(
        contexts=[[[0.0, 0.0]], [[1.0, 1.0]], [[2.0, 2.0]]],
        targets=[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
    )
    detector = PersistenceDetector(calibration_quantile=0.9).fit(training)
    evaluated = windows(contexts=[[[0.0, 0.0]]], targets=[[3.0, 4.0]])

    result = detector.score(evaluated)

    np.testing.assert_allclose(result.scores, [np.sqrt(12.5)])
    np.testing.assert_allclose(result.feature_evidence, [[3.0, 4.0]])
    assert result.threshold == 1.0


def test_featurewise_detector_exposes_largest_robust_deviation():
    training = windows(
        contexts=[[[0.0, 0.0]]] * 5,
        targets=[
            [-1.0, 0.0],
            [0.0, -0.5],
            [0.0, 0.0],
            [0.0, 0.5],
            [1.0, 0.0],
        ],
    )
    detector = RobustFeatureDetector(calibration_quantile=0.8).fit(training)

    result = detector.score(
        windows(contexts=[[[0.0, 0.0]]], targets=[[2.0, -3.0]])
    )

    np.testing.assert_allclose(result.scores, [3.0])
    np.testing.assert_allclose(result.feature_evidence, [[2.0, 3.0]])
    assert result.alerts.tolist() == [True]


def test_latent_detector_suppresses_orthogonal_noise_but_scores_shared_shift():
    axis = np.linspace(-1.0, 1.0, 40)
    targets = np.column_stack((axis, axis))
    contexts = targets[:, None, :] - 0.05
    training = windows(contexts=contexts, targets=targets)
    detector = LatentPredictiveDetector(
        latent_dimension=1,
        ridge=1e-3,
        calibration_quantile=0.95,
    ).fit(training)
    evaluated = windows(
        contexts=[[[-0.05, -0.05]], [[-0.05, -0.05]]],
        targets=[[3.0, -3.0], [3.0, 3.0]],
    )

    result = detector.score(evaluated)

    assert result.scores[0] < result.scores[1] * 0.01
    assert result.alerts.tolist() == [False, True]


def test_fitted_detector_artifact_round_trip_preserves_scores():
    axis = np.linspace(-1.0, 1.0, 30)
    training = windows(
        contexts=np.column_stack((axis - 0.1, axis - 0.1))[:, None, :],
        targets=np.column_stack((axis, axis)),
    )
    fitted = LatentPredictiveDetector(latent_dimension=1).fit(training)
    restored = detector_from_dict(fitted.to_dict())
    evaluated = windows(
        contexts=[[[0.2, 0.2]], [[0.7, 0.7]]],
        targets=[[0.3, 0.3], [2.0, 2.0]],
    )

    expected = fitted.score(evaluated)
    actual = restored.score(evaluated)

    np.testing.assert_allclose(actual.scores, expected.scores)
    np.testing.assert_allclose(actual.feature_evidence, expected.feature_evidence)
    assert actual.threshold == expected.threshold


def test_coherent_latent_detector_requires_multiple_features_to_disagree():
    axis = np.linspace(-1.0, 1.0, 40)
    training = windows(
        contexts=np.column_stack((axis - 0.05, axis - 0.05))[:, None, :],
        targets=np.column_stack((axis, axis)),
    )
    detector = CoherentLatentPredictiveDetector(
        latent_dimension=1,
        consensus_rank=2,
        calibration_quantile=0.95,
    ).fit(training)
    evaluated = windows(
        contexts=[[[-0.05, -0.05]], [[-0.05, -0.05]]],
        targets=[[3.0, 0.0], [3.0, 3.0]],
    )

    result = detector.score(evaluated)

    assert result.scores[0] < result.scores[1] * 0.01
    assert result.alerts.tolist() == [False, True]
    assert result.feature_evidence[1, 1] > result.feature_evidence[0, 1] * 100


def test_coherent_detector_artifact_preserves_residual_calibration():
    axis = np.linspace(-1.0, 1.0, 40)
    training = windows(
        contexts=np.column_stack((axis - 0.05, axis - 0.05))[:, None, :],
        targets=np.column_stack((axis, axis)),
    )
    fitted = CoherentLatentPredictiveDetector(
        latent_dimension=1,
        consensus_rank=2,
    ).fit(training)
    restored = detector_from_dict(fitted.to_dict())
    evaluated = windows(
        contexts=[[[-0.05, -0.05]], [[-0.05, -0.05]]],
        targets=[[3.0, 0.0], [3.0, 3.0]],
    )

    expected = fitted.score(evaluated)
    actual = restored.score(evaluated)

    np.testing.assert_allclose(actual.scores, expected.scores)
    np.testing.assert_allclose(actual.feature_evidence, expected.feature_evidence)
    assert expected.signed_feature_evidence is not None
    assert actual.signed_feature_evidence is not None
    np.testing.assert_allclose(
        actual.signed_feature_evidence,
        expected.signed_feature_evidence,
    )
