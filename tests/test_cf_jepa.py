import copy
from dataclasses import replace

import numpy as np
import pytest

from quantis_core.action_conditioned_dynamics import (
    ActionConditionedWindows,
)
from quantis_core.edge_dynamics.cf_jepa import (
    CfGaussianAlert,
    CfJepaConfig,
    CfJepaModel,
    cf_forward_zones,
    sample_cf_crop,
)
from quantis_core.edge_dynamics.hepa_jepa import HepaEventDefinition
from quantis_core.graph_telemetry import (
    DeclaredTelemetryGraph,
    GraphEntity,
    TelemetryBinding,
)


def test_cf_crop_reserves_three_ordered_forward_zones() -> None:
    generator = np.random.default_rng(14014)
    for _ in range(100):
        start, end = sample_cf_crop(
            30,
            crop_min=0.6,
            crop_max=0.8,
            generator=generator,
        )
        zones = cf_forward_zones(end, 30)

        assert 0 <= start < end <= 27
        assert 18 <= end - start <= 23
        assert zones[0][0] == end
        assert zones[-1][1] == 30
        assert all(zone_end > zone_start for zone_start, zone_end in zones)
        assert all(
            left[1] == right[0]
            for left, right in zip(zones, zones[1:])
        )


def test_cf_jepa_objectives_restore_and_share_deployed_capacity() -> None:
    fit = _tiny_windows(pair_count=4, transition_count=6)
    selection = _tiny_windows(
        pair_count=2,
        transition_count=6,
        pair_prefix="selection",
    )
    base = CfJepaConfig(
        width=8,
        hidden_width=16,
        depth=1,
        pretrain_steps=2,
        checkpoint_interval=1,
        batch_size=8,
        crop_count=2,
        expected_pair_count=4,
    )
    models = {}
    for objective in ("three_zone", "one_zone", "masked_latent"):
        model = CfJepaModel(replace(base, objective=objective))
        model.fit(fit).select(selection)
        models[objective] = model

    assert len(
        {model.inference_parameter_count for model in models.values()}
    ) == 1
    assert (
        models["three_zone"].training_parameter_count
        != models["one_zone"].training_parameter_count
    )

    histories = fit.histories[:3]
    candidate = models["three_zone"]
    target = candidate.encode(histories, fit.graph, route="target")
    online = candidate.encode(histories, fit.graph, route="online")
    restored = CfJepaModel.from_dict(candidate.to_dict())
    restored_target = restored.encode(
        histories, fit.graph, route="target"
    )
    restored_online = restored.encode(
        histories, fit.graph, route="online"
    )

    assert target.tokens.shape == (3, 7, 8)
    assert target.temporal_tokens.shape == (3, 7, 20, 8)
    assert target.entity_ids == fit.graph.entity_ids
    np.testing.assert_allclose(
        target.tokens, restored_target.tokens, atol=1e-7
    )
    np.testing.assert_allclose(
        target.temporal_tokens,
        restored_target.temporal_tokens,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        online.temporal_tokens,
        restored_online.temporal_tokens,
        atol=1e-7,
    )
    assert not np.array_equal(
        target.temporal_tokens, online.temporal_tokens
    )
    with pytest.raises(TypeError):
        candidate.encode(  # type: ignore[call-arg]
            histories,
            fit.graph,
            route="target",
            future_states=fit.future_states[:3],
        )

    corrupted = copy.deepcopy(candidate.to_dict())
    corrupted["state_dict"]["target_encoder.input_fc.weight"][
        "values"
    ][0][0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        CfJepaModel.from_dict(corrupted)


def test_cf_gaussian_alert_restores_scores_and_calibration() -> None:
    fit = _tiny_windows(pair_count=4, transition_count=6)
    calibration = _tiny_windows(
        pair_count=3,
        transition_count=6,
        pair_prefix="calibration",
    )
    config = CfJepaConfig(
        width=8,
        hidden_width=16,
        depth=1,
        pretrain_steps=2,
        checkpoint_interval=1,
        batch_size=8,
        crop_count=2,
        expected_pair_count=4,
    )
    model = CfJepaModel(config).fit(fit).select(fit)
    event = HepaEventDefinition.fit(fit)
    alert = (
        CfGaussianAlert(route="target")
        .fit(model, fit)
        .fit_calibration(model, calibration, event)
    )
    restored_alert = CfGaussianAlert.from_dict(alert.to_dict())

    scores = alert.score(model, calibration.histories, calibration.graph)
    restored_scores = restored_alert.score(
        model, calibration.histories, calibration.graph
    )
    probabilities = alert.calibrated_risk(
        model, calibration.histories, calibration.graph
    )
    decisions = alert.alert_decisions(
        model, calibration.histories, calibration.graph
    )

    np.testing.assert_allclose(scores, restored_scores, atol=1e-9)
    assert scores.shape == (len(calibration.histories),)
    assert np.all(np.isfinite(scores))
    assert np.all((0.0 <= probabilities) & (probabilities <= 1.0))
    assert decisions.dtype == np.bool_
    with pytest.raises(ValueError, match="not fitted"):
        CfGaussianAlert(route="online").fit(
            CfJepaModel(config), fit
        )


def _tiny_windows(
    *,
    pair_count: int,
    transition_count: int,
    pair_prefix: str = "pair",
) -> ActionConditionedWindows:
    rng = np.random.default_rng(914)
    entities = tuple(f"e{index}" for index in range(7))
    features = ("latency", "queue", "utilization")
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
        base = rng.normal(scale=0.05, size=(36, 7, 3)).cumsum(
            axis=0
        )
        for arm in ("control", "treatment"):
            trajectory = base.copy()
            if arm == "treatment":
                trajectory[22:, 3:, 0] += np.linspace(
                    0.0, 3.0, len(trajectory) - 22
                )[:, None]
            actions = np.zeros((35, 7, len(action_names)))
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
                    trajectory[transition + 1 : transition + 11]
                )
                controls = np.zeros((10, 2))
                controls[:, 1] = float(pair == pair_count - 1)
                future_controls.append(controls)
                future_actions.append(
                    actions[transition : transition + 10]
                )
                trajectory_ids.append(
                    f"{pair_prefix}-{pair}-{arm}"
                )
                pair_ids.append(f"{pair_prefix}-{pair}")
                transitions.append(transition)
    return ActionConditionedWindows(
        histories=np.asarray(histories, dtype=np.float64),
        future_states=np.asarray(future_states, dtype=np.float64),
        future_controls=np.asarray(
            future_controls, dtype=np.float64
        ),
        future_actions=np.asarray(
            future_actions, dtype=np.float64
        ),
        trajectory_ids=tuple(trajectory_ids),
        matched_pair_ids=tuple(pair_ids),
        transition_indices=np.asarray(transitions, dtype=np.int64),
        entity_names=entities,
        state_feature_names=features,
        control_feature_names=(
            "request_demand",
            "worker_replicas",
        ),
        action_feature_names=action_names,
        graph=graph,
    )
