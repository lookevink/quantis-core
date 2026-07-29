from pathlib import Path

import numpy as np
import pytest

from lab.action_dynamics.prototype_hepa_jepa import (
    _write_manifest,
    run_experiment,
)
from lab.action_dynamics.prototype_hepa_jepa_assessor import (
    _verify_manifest,
)
from quantis_core.action_conditioned_dynamics import (
    ActionConditionedWindows,
)
from quantis_core.edge_dynamics.hepa_jepa import (
    HEPA_ASSESSMENT_ROLE_NAMES,
    HEPA_MODEL_NAMES,
    HepaConfig,
    HepaEventDefinition,
    HepaJepaModel,
    assess_hepa_tracer,
    calibrate_probability_surface,
    fit_logit_calibrator,
    survival_cdf,
    trajectory_alert_threshold,
)
from quantis_core.graph_telemetry import (
    DeclaredTelemetryGraph,
    GraphEntity,
    TelemetryBinding,
)


def test_survival_cdf_and_calibration_are_monotone() -> None:
    hazards = np.asarray(
        [[0.1, 0.2, 0.3], [0.8, 0.01, 0.02]], dtype=np.float64
    )

    probabilities = survival_cdf(hazards)
    calibrated = calibrate_probability_surface(
        probabilities, slope=1.5, intercept=-0.4
    )

    np.testing.assert_allclose(
        probabilities,
        np.asarray([[0.1, 0.28, 0.496], [0.8, 0.802, 0.80596]]),
    )
    assert np.all(np.diff(probabilities, axis=1) >= 0.0)
    assert np.all(np.diff(calibrated, axis=1) >= 0.0)
    assert np.all((calibrated >= 0.0) & (calibrated <= 1.0))


def test_event_definition_is_action_blind_restorable_and_trajectory_scaled() -> None:
    windows = _tiny_windows(pair_count=4, transition_count=6)

    definition = HepaEventDefinition.fit(windows)
    restored = HepaEventDefinition.from_dict(definition.to_dict())
    labels = definition.labels(windows)

    assert definition.control_trajectory_count == 4
    assert definition.threshold > 0.0
    assert labels.shape == (len(windows.histories), 3)
    np.testing.assert_array_equal(labels, restored.labels(windows))
    np.testing.assert_allclose(
        definition.transition_scores(windows),
        restored.transition_scores(windows),
    )


def test_hepa_public_outputs_restore_exactly_and_null_capacity_matches() -> None:
    fit = _tiny_windows(pair_count=4, transition_count=6)
    selection = _tiny_windows(
        pair_count=2, transition_count=6, pair_prefix="selection"
    )
    event = HepaEventDefinition.fit(fit)
    shared = dict(
        width=16,
        block_count=2,
        head_count=4,
        feedforward_width=32,
        alert_horizon=3,
        stage1_steps=2,
        stage2_steps=2,
        checkpoint_interval=1,
        batch_size=8,
        expected_pair_count=4,
        seed=12012,
    )

    treatment = HepaJepaModel(HepaConfig(**shared)).fit(
        fit, event
    ).select(selection, event).fit_calibration(selection, event)
    null = HepaJepaModel(
        HepaConfig(objective="horizon_deranged", **shared)
    ).fit(fit, event).select(
        selection, event
    ).fit_calibration(selection, event)
    restored = HepaJepaModel.from_dict(treatment.to_dict())

    encoded = treatment.encode(fit.histories[:3], fit.graph)
    restored_encoded = restored.encode(fit.histories[:3], fit.graph)
    probabilities = treatment.predict_event_cdf(
        fit.histories[:3], fit.graph
    )
    restored_probabilities = restored.predict_event_cdf(
        fit.histories[:3], fit.graph
    )
    calibrated = treatment.calibrated_event_cdf(
        fit.histories[:3], fit.graph
    )
    restored_calibrated = restored.calibrated_event_cdf(
        fit.histories[:3], fit.graph
    )
    decisions = treatment.alert_decisions(
        fit.histories[:3], fit.graph
    )
    restored_decisions = restored.alert_decisions(
        fit.histories[:3], fit.graph
    )

    assert encoded.tokens.shape == (3, 7, 16)
    assert encoded.entity_ids == fit.entity_names
    assert treatment.inference_parameter_count == (
        null.inference_parameter_count
    )
    assert treatment.stage1_target_alignment == "aligned"
    assert null.stage1_target_alignment == "whole_pair_deranged"
    assert np.all(np.diff(probabilities, axis=1) >= -1e-12)
    np.testing.assert_allclose(
        encoded.tokens, restored_encoded.tokens, atol=1e-7
    )
    np.testing.assert_allclose(
        probabilities, restored_probabilities, atol=1e-7
    )
    np.testing.assert_allclose(
        calibrated, restored_calibrated, atol=1e-7
    )
    np.testing.assert_array_equal(decisions, restored_decisions)


def test_assessment_recomputes_calibration_and_public_alert_seam() -> None:
    ids = ("c0", "c1", "t0", "t1")
    onsets = {"c0": None, "c1": None, "t0": 0, "t1": 0}
    labels = np.asarray(
        [
            [False, False, False],
            [False, False, False],
            [False, True, True],
            [False, True, True],
        ],
        dtype=np.bool_,
    )
    discriminating = np.asarray(
        [
            [0.02, 0.04, 0.06],
            [0.03, 0.05, 0.07],
            [0.10, 0.80, 0.92],
            [0.12, 0.82, 0.94],
        ],
        dtype=np.float64,
    )
    non_discriminating = np.asarray(
        [
            [0.02, 0.04, 0.06],
            [0.03, 0.05, 0.07],
            [0.03, 0.05, 0.07],
            [0.02, 0.04, 0.06],
        ],
        dtype=np.float64,
    )
    probability_surfaces = {
        role: {
            model: (
                non_discriminating.copy()
                if role != "calibration"
                and model == "horizon_deranged"
                else discriminating.copy()
            )
            for model in HEPA_MODEL_NAMES
        }
        for role in HEPA_ASSESSMENT_ROLE_NAMES
    }
    calibrated_surfaces = {
        role: {} for role in HEPA_ASSESSMENT_ROLE_NAMES
    }
    decisions = {
        role: {} for role in HEPA_ASSESSMENT_ROLE_NAMES
    }
    calibrations = {}
    for model in HEPA_MODEL_NAMES:
        slope, intercept, brier = fit_logit_calibrator(
            probability_surfaces["calibration"][model], labels
        )
        for role in HEPA_ASSESSMENT_ROLE_NAMES:
            calibrated_surfaces[role][model] = (
                calibrate_probability_surface(
                    probability_surfaces[role][model],
                    slope=slope,
                    intercept=intercept,
                )
            )
        threshold = trajectory_alert_threshold(
            calibrated_surfaces["calibration"][model],
            ids,
            ("c0", "c1"),
        )
        calibrations[model] = {
            "slope": slope,
            "intercept": intercept,
            "calibration_brier": brier,
            "alert_threshold": threshold,
        }
        for role in HEPA_ASSESSMENT_ROLE_NAMES:
            decisions[role][model] = (
                calibrated_surfaces[role][model][:, -1]
                > threshold
            )
    truth = np.asarray(
        [[[1.0], [2.0]], [[2.0], [3.0]]], dtype=np.float64
    )
    assessment = assess_hepa_tracer(
        probability_surfaces=probability_surfaces,
        restored_probability_surfaces=probability_surfaces,
        stored_calibrated_surfaces=calibrated_surfaces,
        restored_calibrated_surfaces=calibrated_surfaces,
        stored_alert_decisions=decisions,
        restored_alert_decisions=decisions,
        stored_model_calibrations=calibrations,
        restored_model_calibrations=calibrations,
        labels={
            role: labels.copy()
            for role in HEPA_ASSESSMENT_ROLE_NAMES
        },
        trajectory_ids={
            role: ids for role in HEPA_ASSESSMENT_ROLE_NAMES
        },
        transition_indices={
            role: np.ones(4, dtype=np.int64)
            for role in HEPA_ASSESSMENT_ROLE_NAMES
        },
        trajectory_onsets={
            role: onsets
            for role in HEPA_ASSESSMENT_ROLE_NAMES
        },
        candidate_tokens=np.zeros((2, 2, 2), dtype=np.float64),
        restored_candidate_tokens=np.zeros(
            (2, 2, 2), dtype=np.float64
        ),
        state_truth=truth,
        state_scale=np.ones((2, 1), dtype=np.float64),
        state_varying_mask=np.ones((2, 1), dtype=np.bool_),
        state_predictions={
            "hepa": truth.copy(),
            "matched_pca": truth + 1.0,
        },
        inference_parameter_counts={
            "hepa": 10,
            "horizon_deranged": 10,
            "supervised_scratch": 10,
        },
        protocol_checks={
            "only_target_alignment_differs": True,
            "pair_atomic_derangement": True,
        },
        edge_metrics={
            model: {
                "inference_parameter_count": 10.0,
                "serialized_candidate_sidecars_bytes": 100.0,
                "batch_one_cpu_latency_ms": 1.0,
                "peak_rss_bytes": 1024.0,
                "latency_repetitions": 3.0,
            }
            for model in HEPA_MODEL_NAMES
        },
        raw_effect_scores={
            role: np.asarray([0.0, 0.0, 2.0, 2.0])
            for role in HEPA_ASSESSMENT_ROLE_NAMES
        },
        event_threshold=1.0,
    )

    assert assessment["passed"] is True
    assert assessment["gates"][
        "restoration_reproduces_all_public_outputs"
    ] is True


def test_runner_refuses_to_overwrite_existing_output(
    tmp_path: Path,
) -> None:
    output = tmp_path / "retained"
    output.mkdir()

    with pytest.raises(FileExistsError):
        run_experiment(
            cache_directory=tmp_path / "missing-cache",
            output_directory=output,
            stage1_steps=1,
            stage2_steps=1,
            latency_repetitions=1,
            allow_noninterpretable_smoke=True,
        )


def test_manifest_detects_stored_content_tampering(
    tmp_path: Path,
) -> None:
    payload = tmp_path / "payload.json"
    payload.write_text('{"value": 1}\n')
    _write_manifest(tmp_path)
    _verify_manifest(tmp_path)

    payload.write_text('{"value": 2}\n')

    with pytest.raises(ValueError, match="content identity"):
        _verify_manifest(tmp_path)


def _tiny_windows(
    *,
    pair_count: int,
    transition_count: int,
    pair_prefix: str = "pair",
) -> ActionConditionedWindows:
    rng = np.random.default_rng(881)
    entities = tuple(f"e{index}" for index in range(7))
    features = ("latency", "queue")
    graph = DeclaredTelemetryGraph(
        entities=(
            GraphEntity("e0", "node", "service"),
            GraphEntity("e1", "edge", "relation", "e0", "e2"),
            GraphEntity("e2", "node", "service"),
            GraphEntity("e3", "edge", "relation", "e2", "e4"),
            GraphEntity("e4", "node", "service"),
            GraphEntity("e5", "edge", "relation", "e4", "e6"),
            GraphEntity("e6", "node", "service"),
        ),
        bindings=tuple(
            TelemetryBinding(
                feature_key=f"{entity}.{feature}",
                entity_id=entity,
            )
            for entity in entities
            for feature in features
        ),
    )
    histories = []
    future_states = []
    future_controls = []
    future_actions = []
    trajectory_ids = []
    pair_ids = []
    transitions = []
    action_names = (
        "no_action",
        "applicable",
        "kind:worker_pause",
        "phase:start",
        "phase:active",
        "phase:stop",
        "magnitude",
        "elapsed_fraction",
        "remaining_fraction",
    )
    for pair in range(pair_count):
        base = rng.normal(scale=0.05, size=(29, 7, 2)).cumsum(axis=0)
        for arm in ("control", "treatment"):
            trajectory = base.copy()
            if arm == "treatment":
                trajectory[22:, 3:, 0] += np.linspace(
                    0.0, 3.0, len(trajectory) - 22
                )[:, None]
            actions = np.zeros((28, 7, len(action_names)))
            actions[..., 0] = 1.0
            if arm == "treatment":
                actions[21:25, 2, 0] = 0.0
                actions[21:25, 2, 1] = 1.0
                actions[21:25, 2, 2] = 1.0
                actions[21, 2, 3] = 1.0
                actions[22:25, 2, 4] = 1.0
                actions[25, 2, 5] = 1.0
                actions[21:26, 2, 6] = 1.0
            for offset in range(transition_count):
                transition = 19 + offset
                histories.append(trajectory[offset : offset + 20])
                future_states.append(
                    trajectory[transition + 1 : transition + 4]
                )
                future_controls.append(np.zeros((3, 2)))
                future_actions.append(actions[transition : transition + 3])
                trajectory_ids.append(
                    f"{pair_prefix}-{pair}-{arm}"
                )
                pair_ids.append(f"{pair_prefix}-{pair}")
                transitions.append(transition)
    return ActionConditionedWindows(
        histories=np.asarray(histories, dtype=np.float64),
        future_states=np.asarray(future_states, dtype=np.float64),
        future_controls=np.asarray(future_controls, dtype=np.float64),
        future_actions=np.asarray(future_actions, dtype=np.float64),
        trajectory_ids=tuple(trajectory_ids),
        matched_pair_ids=tuple(pair_ids),
        transition_indices=np.asarray(transitions, dtype=np.int64),
        entity_names=entities,
        state_feature_names=features,
        control_feature_names=("request_demand", "worker_replicas"),
        action_feature_names=action_names,
        graph=graph,
    )
