import numpy as np

from quantis_core.action_conditioned_dynamics import (
    ActionConditionedWindows,
)
from quantis_core.edge_dynamics.hepa_jepa import (
    HepaConfig,
    HepaEventDefinition,
    HepaJepaModel,
    calibrate_probability_surface,
    survival_cdf,
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
    ).select(selection, event)
    null = HepaJepaModel(
        HepaConfig(objective="horizon_deranged", **shared)
    ).fit(fit, event).select(selection, event)
    restored = HepaJepaModel.from_dict(treatment.to_dict())

    encoded = treatment.encode(fit.histories[:3], fit.graph)
    restored_encoded = restored.encode(fit.histories[:3], fit.graph)
    probabilities = treatment.predict_event_cdf(
        fit.histories[:3], fit.graph
    )
    restored_probabilities = restored.predict_event_cdf(
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
