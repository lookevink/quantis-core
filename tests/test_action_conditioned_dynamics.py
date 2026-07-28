import copy
from dataclasses import replace

import numpy as np
import pytest

from quantis_core.action_conditioned_dynamics import (
    ActionConditionedCaseManifest,
    ActionConditionedRun,
    ActionConditionedWindows,
    ActionTrajectoryCompiler,
    GraphVarxConfig,
    GraphVarxDynamics,
    InterventionAction,
    RolloutCandidate,
    rank_action_candidates,
    validate_matched_action_pairs,
)
from quantis_core.graph_telemetry import (
    DeclaredTelemetryGraph,
    GraphEntity,
)
from quantis_core.action_dynamics_synthetic import (
    causal_chain_graph,
    synthetic_action_runs,
)


def test_action_manifest_round_trips_a_reversible_intervention() -> None:
    manifest = ActionConditionedCaseManifest(
        case_id="train-pair-001-action",
        matched_pair_id="train-pair-001",
        split="training",
        point_count=72,
        logical_window_period_nano=1_000_000_000,
        topology_id="workers-2",
        worker_replicas=2,
        workload_seed=11,
        intervention_seed=29,
        actions=(
            InterventionAction(
                action_id="pause-001",
                action_kind="worker_pause",
                target_entity="worker_pool",
                start_index=28,
                stop_index=42,
                magnitude=0.5,
            ),
        ),
    )

    restored = ActionConditionedCaseManifest.from_dict(
        manifest.to_dict()
    )

    assert restored == manifest
    assert manifest.to_dict()["schema_version"] == 3
    assert manifest.to_dict()["actions"][0][
        "parameter_schema_version"
    ] == 1
    assert manifest.to_dict()["actions"][0][
        "effect_direction"
    ] == "increase"
    assert len(manifest.canonical_sha256()) == 64
    assert manifest.actions[0].duration == 14


def test_action_manifest_rejects_an_action_outside_the_trajectory() -> None:
    with pytest.raises(ValueError, match="action interval"):
        ActionConditionedCaseManifest(
            case_id="invalid",
            matched_pair_id="pair",
            split="training",
            point_count=20,
            logical_window_period_nano=1_000_000_000,
            topology_id="workers-1",
            worker_replicas=1,
            workload_seed=1,
            intervention_seed=2,
            actions=(
                InterventionAction(
                    action_id="late",
                    action_kind="api_rejection",
                    target_entity="api",
                    start_index=18,
                    stop_index=21,
                    magnitude=1.0,
                ),
            ),
        )


def test_action_manifest_parser_rejects_unknown_or_nonfinite_evidence() -> None:
    manifest = synthetic_action_runs(
        2, split="training", seed=8
    )[0].manifest
    unexpected = manifest.to_dict()
    unexpected["unreviewed_truth"] = "leak"
    nonfinite = manifest.to_dict()
    nonfinite["actions"][0]["magnitude"] = float("inf")

    with pytest.raises(ValueError, match="manifest schema"):
        ActionConditionedCaseManifest.from_dict(unexpected)
    with pytest.raises(ValueError, match="magnitude"):
        ActionConditionedCaseManifest.from_dict(nonfinite)


def test_manifest_rejects_overlapping_phase_zero_actions() -> None:
    first = InterventionAction(
        action_id="first",
        action_kind="worker_pause",
        target_entity="worker_pool",
        start_index=4,
        stop_index=10,
        magnitude=0.5,
    )
    second = InterventionAction(
        action_id="second",
        action_kind="api_rejection",
        target_entity="api",
        start_index=9,
        stop_index=12,
        magnitude=0.5,
    )

    with pytest.raises(ValueError, match="overlap"):
        ActionConditionedCaseManifest(
            case_id="overlap",
            matched_pair_id="overlap-pair",
            split="training",
            point_count=20,
            logical_window_period_nano=1_000_000_000,
            topology_id="workers-1",
            worker_replicas=1,
            workload_seed=1,
            intervention_seed=2,
            actions=(first, second),
        )


def test_matched_pair_validator_rejects_split_or_control_drift() -> None:
    treatment, control = synthetic_action_runs(
        2, split="training", seed=18
    )

    summary = validate_matched_action_pairs((treatment, control))

    assert summary["pair_count"] == 1
    assert summary["run_count"] == 2
    wrong_split = replace(
        control,
        manifest=replace(control.manifest, split="validation"),
    )
    with pytest.raises(ValueError, match="matched pair"):
        validate_matched_action_pairs((treatment, wrong_split))
    wrong_controls = replace(
        control, controls=control.controls + 1.0
    )
    with pytest.raises(ValueError, match="control schedule"):
        validate_matched_action_pairs((treatment, wrong_controls))
    ineffective = replace(
        treatment, observations=control.observations.copy()
    )
    with pytest.raises(ValueError, match="raw effect"):
        validate_matched_action_pairs((ineffective, control))
    nonrecovering_values = treatment.observations.copy()
    target = treatment.graph.entity_ids.index(
        treatment.manifest.actions[0].target_entity
    )
    nonrecovering_values[-3:, target, 0] += 1.0
    nonrecovering = replace(
        treatment, observations=nonrecovering_values
    )
    with pytest.raises(ValueError, match="recovery"):
        validate_matched_action_pairs((nonrecovering, control))


def test_compiler_keeps_state_controls_and_actions_separate() -> None:
    training = _run(
        case_id="train-action",
        pair_id="train-pair",
        split="training",
        offset=0.0,
        with_action=True,
    )
    validation = _run(
        case_id="validation-action",
        pair_id="validation-pair",
        split="validation",
        offset=100.0,
        with_action=True,
    )
    compiler = ActionTrajectoryCompiler(
        context_length=2,
        rollout_horizon=2,
    ).fit((training,))

    training_windows = compiler.transform((training,))
    validation_windows = compiler.transform((validation,))

    assert training_windows.histories.shape == (4, 2, 3, 1)
    assert training_windows.future_states.shape == (4, 2, 3, 1)
    assert training_windows.future_controls.shape == (4, 2, 1)
    assert training_windows.future_actions.shape == (4, 2, 3, 13)
    assert set(training_windows.trajectory_ids) == {"train-action"}
    assert set(training_windows.matched_pair_ids) == {"train-pair"}
    transition_two = np.flatnonzero(
        training_windows.transition_indices == 2
    )[0]
    worker = training_windows.entity_names.index("worker_pool")
    pause = training_windows.action_feature_names.index(
        "kind:worker_pause"
    )
    no_action = training_windows.action_feature_names.index(
        "no_action"
    )
    assert (
        training_windows.future_actions[
            transition_two, 0, worker, pause
        ]
        == 1.0
    )
    assert (
        training_windows.future_actions[
            transition_two, 0, worker, no_action
        ]
        == 0.0
    )
    assert np.mean(validation_windows.histories) > 10.0
    assert "split" not in training_windows.state_feature_names
    assert "action_kind" not in training_windows.state_feature_names
    stop_phase = training_windows.action_feature_names.index(
        "phase:stop"
    )
    transition_four = np.flatnonzero(
        training_windows.transition_indices == 4
    )[0]
    assert (
        training_windows.future_actions[
            transition_four, 0, worker, stop_phase
        ]
        == 1.0
    )


def test_windows_reject_mismatched_future_horizons() -> None:
    compiler = ActionTrajectoryCompiler(
        context_length=2,
        rollout_horizon=2,
    ).fit(
        synthetic_action_runs(2, split="training", seed=9)
    )
    windows = compiler.transform(
        synthetic_action_runs(2, split="training", seed=9)
    )

    with pytest.raises(ValueError, match="horizon"):
        replace(
            windows,
            future_controls=windows.future_controls[:, :1],
        )


def test_compiler_artifact_restores_the_same_held_out_windows() -> None:
    training_runs = synthetic_action_runs(
        8, split="training", seed=70
    )
    validation_runs = synthetic_action_runs(
        4, split="validation", seed=170
    )
    compiler = ActionTrajectoryCompiler(
        context_length=4,
        rollout_horizon=5,
    ).fit(training_runs)
    expected = compiler.transform(validation_runs)

    restored = ActionTrajectoryCompiler.from_dict(
        compiler.to_dict()
    )
    actual = restored.transform(validation_runs)

    np.testing.assert_array_equal(
        actual.histories, expected.histories
    )
    np.testing.assert_array_equal(
        actual.future_controls, expected.future_controls
    )
    np.testing.assert_array_equal(
        actual.future_actions, expected.future_actions
    )
    assert actual.state_feature_names == expected.state_feature_names
    assert actual.graph.to_dict() == expected.graph.to_dict()


def test_graph_varx_rolls_out_actions_and_restores_identically() -> None:
    training_runs = synthetic_action_runs(
        24, split="training", seed=100
    )
    validation_runs = synthetic_action_runs(
        8, split="validation", seed=900
    )
    compiler = ActionTrajectoryCompiler(
        context_length=4,
        rollout_horizon=6,
    ).fit(training_runs)
    training = compiler.transform(training_runs)
    validation = compiler.transform(validation_runs)
    action_model = GraphVarxDynamics(
        GraphVarxConfig(ridge=1e-3, include_actions=True)
    ).fit(training)
    action_agnostic = GraphVarxDynamics(
        GraphVarxConfig(ridge=1e-3, include_actions=False)
    ).fit(training)

    predicted = action_model.rollout(
        validation.histories,
        validation.future_controls,
        validation.future_actions,
        validation.graph,
    )
    control = action_agnostic.rollout(
        validation.histories,
        validation.future_controls,
        validation.future_actions,
        validation.graph,
    )
    action_error = np.mean(
        np.square(predicted.mean - validation.future_states)
    )
    control_error = np.mean(
        np.square(control.mean - validation.future_states)
    )
    restored = GraphVarxDynamics.from_dict(action_model.to_dict())
    restored_prediction = restored.rollout(
        validation.histories,
        validation.future_controls,
        validation.future_actions,
        validation.graph,
    )

    assert action_error < 0.6 * control_error
    np.testing.assert_allclose(
        restored_prediction.mean, predicted.mean
    )
    np.testing.assert_allclose(
        restored_prediction.variance, predicted.variance
    )
    with pytest.raises(ValueError, match="graph"):
        action_model.rollout(
            validation.histories,
            validation.future_controls,
            validation.future_actions,
            _graph(),
        )


def test_varx_restore_rejects_semantic_or_topology_state_corruption() -> None:
    training_runs = synthetic_action_runs(
        8, split="training", seed=310
    )
    compiler = ActionTrajectoryCompiler(
        context_length=4,
        rollout_horizon=5,
    ).fit(training_runs)
    model = GraphVarxDynamics(GraphVarxConfig()).fit(
        compiler.transform(training_runs)
    )
    action_schema = copy.deepcopy(model.to_dict())
    action_schema["semantic_schema"]["action_feature_names"][0] = (
        "corrupted"
    )
    topology_state = copy.deepcopy(model.to_dict())
    topology_state["state"]["source_positions"][1] = [1]
    state_schema = copy.deepcopy(model.to_dict())
    state_schema["semantic_schema"]["state_feature_names"][0] = (
        "renamed"
    )
    config = copy.deepcopy(model.to_dict())
    config["config"]["include_actions"] = "false"

    with pytest.raises(ValueError, match="action schema"):
        GraphVarxDynamics.from_dict(action_schema)
    with pytest.raises(ValueError, match="topology"):
        GraphVarxDynamics.from_dict(topology_state)
    with pytest.raises(ValueError, match="semantic schema hash"):
        GraphVarxDynamics.from_dict(state_schema)
    with pytest.raises(ValueError, match="configuration"):
        GraphVarxDynamics.from_dict(config)


def test_windows_reject_entity_names_that_disagree_with_graph_order() -> None:
    runs = synthetic_action_runs(
        4, split="training", seed=1_000
    )
    windows = ActionTrajectoryCompiler(
        context_length=4,
        rollout_horizon=8,
    ).fit(runs).transform(runs)

    with pytest.raises(
        ValueError, match="schemas do not align"
    ):
        replace(
            windows,
            entity_names=tuple(reversed(windows.entity_names)),
        )


def test_synthetic_matched_pair_differs_only_after_the_action() -> None:
    treatment, control = synthetic_action_runs(
        2, split="training", seed=50
    )
    action = treatment.manifest.actions[0]

    assert treatment.manifest.matched_pair_id == (
        control.manifest.matched_pair_id
    )
    np.testing.assert_array_equal(
        treatment.controls, control.controls
    )
    np.testing.assert_array_equal(
        treatment.observations[: action.start_index + 1],
        control.observations[: action.start_index + 1],
    )
    assert not np.array_equal(
        treatment.observations[action.start_index + 1 :],
        control.observations[action.start_index + 1 :],
    )


def test_candidate_ranking_recovers_action_target_and_no_action() -> None:
    training_runs = synthetic_action_runs(
        30, split="training", seed=200
    )
    validation_runs = synthetic_action_runs(
        4, split="validation", seed=800
    )
    compiler = ActionTrajectoryCompiler(
        context_length=4,
        rollout_horizon=8,
    ).fit(training_runs)
    training = compiler.transform(training_runs)
    validation = compiler.transform(validation_runs)
    model = GraphVarxDynamics(
        GraphVarxConfig(ridge=1e-3, include_actions=True)
    ).fit(training)
    action_index = _window_index(
        validation, "validation-pair-000-action", 9
    )
    true_actions = validation.future_actions[action_index]
    no_actions = np.zeros_like(true_actions)
    no_actions[:, :, 0] = 1.0
    wrong_target = no_actions.copy()
    source = validation.entity_names.index("source")
    sink = validation.entity_names.index("sink")
    wrong_target[:, sink] = true_actions[:, source]
    candidates = (
        RolloutCandidate("true:api_rejection@source", true_actions),
        RolloutCandidate("wrong:api_rejection@sink", wrong_target),
        RolloutCandidate("none", no_actions),
    )

    attributed = rank_action_candidates(
        model=model,
        history=validation.histories[action_index],
        future_controls=validation.future_controls[action_index],
        observed_future=validation.future_states[action_index],
        candidates=candidates,
        graph=validation.graph,
        no_action_candidate_id="none",
    )
    nominal_index = _window_index(
        validation, "validation-pair-000-control", 9
    )
    nominal = rank_action_candidates(
        model=model,
        history=validation.histories[nominal_index],
        future_controls=validation.future_controls[nominal_index],
        observed_future=validation.future_states[nominal_index],
        candidates=(candidates[0], candidates[2]),
        graph=validation.graph,
        no_action_candidate_id="none",
    )

    assert attributed.ranked_candidate_ids[0] == (
        "true:api_rejection@source"
    )
    first_effect_steps = [
        int(
            np.flatnonzero(
                np.abs(
                    attributed.counterfactual_delta[
                        :, entity_position, 0
                    ]
                )
                > 0.01
            )[0]
        )
        for entity_position in range(
            len(validation.entity_names)
        )
    ]
    assert first_effect_steps == [1, 2, 3, 4, 5]
    assert attributed.per_entity_effect[sink] > 0.0
    assert nominal.ranked_candidate_ids[0] == "none"


def _run(
    *,
    case_id: str,
    pair_id: str,
    split: str,
    offset: float,
    with_action: bool,
) -> ActionConditionedRun:
    point_count = 7
    actions = (
        (
            InterventionAction(
                action_id=f"{case_id}-pause",
                action_kind="worker_pause",
                target_entity="worker_pool",
                start_index=2,
                stop_index=4,
                magnitude=0.5,
            ),
        )
        if with_action
        else ()
    )
    manifest = ActionConditionedCaseManifest(
        case_id=case_id,
        matched_pair_id=pair_id,
        split=split,
        point_count=point_count,
        logical_window_period_nano=1_000_000_000,
        topology_id="synthetic-chain",
        worker_replicas=1,
        workload_seed=7,
        intervention_seed=13,
        actions=actions,
    )
    observations = (
        np.arange(point_count * 3, dtype=np.float64)
        .reshape(point_count, 3, 1)
        + offset
    )
    return ActionConditionedRun(
        manifest=manifest,
        graph=_graph(),
        observations=observations,
        controls=np.arange(point_count, dtype=np.float64).reshape(
            point_count, 1
        ),
        state_feature_names=("load",),
        control_feature_names=("request_demand",),
    )


def _window_index(
    windows: ActionConditionedWindows,
    trajectory_id: str,
    transition_index: int,
) -> int:
    return next(
        index
        for index, (candidate_id, candidate_transition) in enumerate(
            zip(
                windows.trajectory_ids,
                windows.transition_indices,
            )
        )
        if candidate_id == trajectory_id
        and candidate_transition == transition_index
    )


def _graph() -> DeclaredTelemetryGraph:
    return DeclaredTelemetryGraph(
        entities=(
            GraphEntity("api", "node", "service"),
            GraphEntity("worker_pool", "node", "service_pool"),
            GraphEntity(
                "api_to_worker",
                "edge",
                "request",
                "api",
                "worker_pool",
            ),
        ),
        bindings=(),
    )
