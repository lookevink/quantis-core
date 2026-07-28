import json

import numpy as np
import pytest

from quantis_core.contextual_multimodal_corpus import (
    ContextualMultimodalModelWindows,
)
from quantis_core.contextual_multimodal_world_model import (
    ContextualMultimodalJepaWorldModelDetector,
)
from quantis_core.contextual_representation_transfer import (
    evaluate_frozen_context_transfer,
)


def test_contextual_jepa_is_conditioned_staged_and_roundtrips() -> None:
    training, validation = _contextual_windows()
    arguments = {
        "metric_latent_dimension": 2,
        "log_latent_dimension": 1,
        "pretraining_epochs": 70,
        "predictor_refinement_epochs": 20,
        "learning_rate": 0.02,
        "ema_decay": 0.96,
        "loss": "huber",
        "auxiliary_loss_weight": 0.2,
        "rollout_loss_weight": 0.2,
        "calibration_quantile": 0.95,
        "seed": 53,
    }
    first = ContextualMultimodalJepaWorldModelDetector(
        **arguments
    ).fit(training)
    second = ContextualMultimodalJepaWorldModelDetector(
        **arguments
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
    assert first.to_dict()["kind"] == (
        "contextual_multimodal_jepa_world_model_v1"
    )
    assert first.to_dict()["training_protocol"] == {
        "target_encoder_update": "ema_during_pretraining_only",
        "encoder_refinement": "frozen",
        "loss": "huber",
        "target_horizons": [1, 3, 6],
        "target_block_size": 2,
        "context_patch_stride": 2,
        "scoring_horizon": 1,
        "rollout": {
            "steps": 2,
            "first_horizon": 1,
            "second_horizon": 3,
            "intermediate_prediction": "stop_gradient",
        },
        "predictor_conditioning": [
            "request_demand",
            "worker_replicas",
            "target_horizon",
        ],
        "auxiliary_objectives": [
            "metric_to_metric",
            "log_to_log",
            "metric_to_log",
            "log_to_metric",
        ],
    }
    assert set(first.diagnostics) >= {
        "metric_effective_rank",
        "log_effective_rank",
        "metric_tanh_saturation",
        "log_tanh_saturation",
        "online_target_distance",
        "auxiliary_losses",
        "frozen_latent_probes",
        "metric_target_variance",
        "metric_target_covariance",
        "log_target_variance",
        "log_target_covariance",
        "ema_update_half_life_epochs",
    }
    assert set(first.diagnostics["frozen_latent_probes"]) == {
        "checkout_completion_ratio",
        "checkout_backlog_delta_ratio",
        "request_latency_ms",
        "queue_depth",
        "request_latency_bucket",
        "queue_depth_bucket",
        "worker_completion_ratio",
        "queue_transition_direction",
    }

    expected = first.score(validation)
    without_logs = first.score_with_context(
        validation,
        include_metric_context=True,
        include_log_context=False,
    )
    restored = (
        ContextualMultimodalJepaWorldModelDetector.from_dict(
            first.to_dict()
        )
    )
    actual = restored.score(validation)
    encoded = first.encode_context(validation)
    restored_encoded = restored.encode_context(validation)

    np.testing.assert_allclose(actual.scores, expected.scores)
    np.testing.assert_allclose(restored_encoded, encoded)
    assert encoded.shape == (
        len(validation.metric_contexts),
        3,
        3,
    )
    np.testing.assert_allclose(
        actual.feature_evidence,
        expected.feature_evidence,
    )
    np.testing.assert_allclose(
        actual.signed_feature_evidence,
        expected.signed_feature_evidence,
    )
    assert actual.feature_evidence.shape == (
        len(validation.metric_contexts),
        5,
    )
    assert np.max(
        np.abs(without_logs.scores - expected.scores)
    ) > 1e-6
    assert actual.threshold == expected.threshold

    truncated = first.to_dict()
    truncated["metric_encoder_weights"].pop()
    with pytest.raises(ValueError, match="metric_encoder_weights"):
        ContextualMultimodalJepaWorldModelDetector.from_dict(
            truncated
        )


def test_contextual_jepa_v2_recipe_masks_and_balances_modalities() -> None:
    training, validation = _contextual_windows()
    detector = ContextualMultimodalJepaWorldModelDetector(
        metric_latent_dimension=2,
        log_latent_dimension=2,
        pretraining_epochs=12,
        predictor_refinement_epochs=4,
        modality_mask_probability=0.15,
        log_self_loss_multiplier=0.25,
        cross_modal_loss_multiplier=1.5,
        seed=71,
    ).fit(training)

    artifact = detector.to_dict()
    assert artifact["training_protocol"][
        "context_modality_masking"
    ] == {
        "kind": "deterministic_single_modality_dropout",
        "probability_per_available_modality": 0.15,
        "seed": 71,
    }
    assert artifact["training_protocol"][
        "auxiliary_objective_multipliers"
    ] == {
        "metric_to_metric": 1.0,
        "log_to_log": 0.25,
        "metric_to_log": 1.5,
        "log_to_metric": 1.5,
    }
    restored = ContextualMultimodalJepaWorldModelDetector.from_dict(
        artifact
    )
    unmasked = ContextualMultimodalJepaWorldModelDetector(
        metric_latent_dimension=2,
        log_latent_dimension=2,
        pretraining_epochs=12,
        predictor_refinement_epochs=4,
        seed=71,
    ).fit(training)
    np.testing.assert_allclose(
        restored.score(validation).scores,
        detector.score(validation).scores,
    )
    assert np.max(
        np.abs(
            detector.score(validation).scores
            - unmasked.score(validation).scores
        )
    ) > 1e-6


def test_frozen_context_transfer_uses_held_out_families() -> None:
    training, validation = _contextual_windows()

    def detector(metric_dimension: int, log_dimension: int, seed: int):
        return ContextualMultimodalJepaWorldModelDetector(
            metric_latent_dimension=metric_dimension,
            log_latent_dimension=log_dimension,
            pretraining_epochs=5,
            predictor_refinement_epochs=2,
            seed=seed,
        ).fit(training)

    models = {
        "contextual_multimodal": detector(2, 1, 3),
        "metrics_only": detector(2, 0, 5),
        "capacity_matched_metrics_only": detector(3, 0, 7),
        "shuffled_logs": detector(2, 1, 11),
    }
    validation_cases = tuple(
        "confirmation-f13-w1-173"
        if index < len(validation.metric_contexts) // 2
        else "confirmation-f14-w1-173"
        for index in range(len(validation.metric_contexts))
    )
    transfer = evaluate_frozen_context_transfer(
        models,
        training,
        validation,
        training_window_case_ids=tuple(
            "confirmation-f01-w1-173"
            for _ in range(len(training.metric_contexts))
        ),
        validation_window_case_ids=validation_cases,
        shuffled_training=training,
        shuffled_validation=validation,
        target_names=(
            "metric.queue",
            "metric.worker",
            "log.completion",
            "log.backlog",
        ),
        ridge=0.001,
        pca_dimension=6,
    )

    assert transfer["fit_split"] == (
        "training_schedule_families_only"
    )
    assert set(transfer["representations"]) == {
        "contextual_multimodal",
        "metrics_only",
        "capacity_matched_metrics_only",
        "shuffled_logs",
        "raw_context_ridge",
        "pca_6_context_ridge",
    }
    contextual = transfer["representations"][
        "contextual_multimodal"
    ]
    assert contextual["context_dimension"] == 9
    assert contextual["completed_target_count"] == 4
    assert set(
        contextual["targets"]["metric.queue"][
            "family_normalized_mse"
        ]
    ) == {"f13", "f14"}


def _contextual_windows() -> tuple[
    ContextualMultimodalModelWindows,
    ContextualMultimodalModelWindows,
]:
    time = np.arange(220, dtype=np.float64)
    demand = 5.0 + np.sin(time / 9.0)
    metric_values = np.column_stack(
        (
            np.sin(time / 8.0),
            np.cos(time / 11.0),
            0.6 * np.sin(time / 8.0 + 0.2),
        )
    )
    log_values = np.column_stack(
        (
            0.95 + 0.02 * np.sin(time / 13.0),
            0.02 * np.cos(time / 17.0),
        )
    )
    controls = np.column_stack(
        (demand, np.ones_like(demand))
    )

    def windows(start: int, stop: int) -> ContextualMultimodalModelWindows:
        metric = metric_values[start:stop]
        logs = log_values[start:stop]
        control = controls[start:stop]
        lookback = 6
        horizons = (1, 3, 6)
        block_size = 2
        context_ends = range(
            lookback,
            len(metric) - horizons[-1] - block_size + 2,
        )
        return ContextualMultimodalModelWindows(
            metric_contexts=np.stack(
                [
                    metric[index - lookback : index]
                    for index in context_ends
                ]
            ),
            log_contexts=np.stack(
                [
                    logs[index - lookback : index]
                    for index in context_ends
                ]
            ),
            metric_target_blocks=np.stack(
                [
                    np.stack(
                        [
                            metric[
                                index + horizon - 1 :
                                index + horizon + block_size - 1
                            ]
                            for horizon in horizons
                        ]
                    )
                    for index in context_ends
                ]
            ),
            log_target_blocks=np.stack(
                [
                    np.stack(
                        [
                            logs[
                                index + horizon - 1 :
                                index + horizon + block_size - 1
                            ]
                            for horizon in horizons
                        ]
                    )
                    for index in context_ends
                ]
            ),
            target_controls=np.stack(
                [
                    np.stack(
                        [
                            control[
                                index + horizon - 1 :
                                index + horizon + block_size - 1
                            ]
                            for horizon in horizons
                        ]
                    )
                    for index in context_ends
                ]
            ),
            point_indices=np.asarray(
                list(context_ends),
                dtype=np.int64,
            ),
            metric_feature_names=(
                "queue",
                "worker",
                "database",
            ),
            log_feature_names=("completion", "backlog"),
            control_feature_names=(
                "request_demand",
                "worker_replicas",
            ),
            horizons=horizons,
            target_block_size=block_size,
        )

    return windows(0, 130), windows(130, 220)
