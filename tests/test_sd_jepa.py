import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from lab.action_dynamics.prototype_sd_jepa import (
    FROZEN_CACHE,
    run_experiment,
)
from lab.action_dynamics.prototype_sd_jepa_assessor import (
    assess_stored_bundle,
    verify_artifact_manifest,
    verify_stored_assessment,
)
from quantis_core.action_conditioned_dynamics import ActionConditionedWindows
from quantis_core.edge_dynamics.hepa_jepa import HepaEventDefinition
from quantis_core.edge_dynamics.sd_jepa import (
    SdJepaConfig,
    SdJepaModel,
    SdScoreCalibrator,
    cosine_margin_triplet_loss,
)
from quantis_core.graph_telemetry import (
    DeclaredTelemetryGraph,
    GraphEntity,
    TelemetryBinding,
)


def test_sd_jepa_triplet_matches_released_middle_adjacent_sampler() -> None:
    embeddings = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
            [[0.0, 1.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
            [[-1.0, 0.0], [-1.0, 0.0], [0.0, -1.0], [1.0, 0.0]],
        ]
    )
    actual = cosine_margin_triplet_loss(
        embeddings, margin=0.2, negative_shift=1
    )
    anchor = embeddings[:, 2]
    positive = embeddings[:, 3]
    negative = torch.roll(anchor, shifts=1, dims=0)
    expected = torch.relu(
        torch.nn.functional.cosine_similarity(anchor, negative)
        - torch.nn.functional.cosine_similarity(anchor, positive)
        + 0.2
    ).mean()

    torch.testing.assert_close(actual, expected)


def test_sd_jepa_cells_restore_and_match_capacity() -> None:
    fit = _tiny_windows(pair_count=4, transition_count=6)
    selection = _tiny_windows(
        pair_count=2, transition_count=6, pair_prefix="selection"
    )
    base = SdJepaConfig(
        width=8,
        hidden_width=16,
        pretrain_steps=2,
        checkpoint_interval=1,
        sigreg_sketch_dimension=8,
        expected_pair_count=4,
    )
    models = {}
    for objective in ("sd_jepa", "lewm_unsplit", "a2_full"):
        model = SdJepaModel(replace(base, objective=objective))
        model.fit(fit).select(selection)
        models[objective] = model

    assert len(
        {model.training_parameter_count for model in models.values()}
    ) == 1
    assert len(
        {model.inference_parameter_count for model in models.values()}
    ) == 1

    model = models["sd_jepa"]
    encoded = model.encode(fit.histories[:3], fit.graph)
    restored = SdJepaModel.from_dict(model.to_dict())
    restored_encoded = restored.encode(fit.histories[:3], fit.graph)
    np.testing.assert_allclose(
        encoded.entity_tokens, restored_encoded.entity_tokens, atol=1e-7
    )
    np.testing.assert_allclose(
        encoded.scene_tokens, restored_encoded.scene_tokens, atol=1e-7
    )
    assert encoded.entity_tokens.shape == (3, 20, 7, 8)
    assert encoded.scene_tokens.shape == (3, 20, 8)
    assert encoded.current_content_tokens.shape == (3, 7, 6)
    for kind in ("angle", "z_mse"):
        score = model.raw_score(
            fit.histories[:3], fit.graph, kind=kind
        )
        restored_score = restored.raw_score(
            fit.histories[:3], fit.graph, kind=kind
        )
        np.testing.assert_allclose(score, restored_score, atol=1e-7)
        assert np.all((0.0 <= score) & (score <= 1.0))

    with pytest.raises(TypeError):
        model.encode(  # type: ignore[call-arg]
            fit.histories[:3],
            fit.graph,
            future_states=fit.future_states[:3],
        )

    corrupted = copy.deepcopy(model.to_dict())
    corrupted["state_dict"]["encoder.input_fc.weight"]["values"][0][0] = (
        float("nan")
    )
    with pytest.raises(ValueError, match="non-finite"):
        SdJepaModel.from_dict(corrupted)


def test_sd_jepa_score_calibrator_restores() -> None:
    fit = _tiny_windows(pair_count=4, transition_count=6)
    calibration = _tiny_windows(
        pair_count=3, transition_count=6, pair_prefix="calibration"
    )
    model = (
        SdJepaModel(
            SdJepaConfig(
                width=8,
                hidden_width=16,
                pretrain_steps=2,
                checkpoint_interval=1,
                sigreg_sketch_dimension=8,
                expected_pair_count=4,
            )
        )
        .fit(fit)
        .select(calibration)
    )
    event = HepaEventDefinition.fit(fit)
    calibrator = SdScoreCalibrator(score_name="sd_jepa_angle").fit(
        model, calibration, event
    )
    restored = SdScoreCalibrator.from_dict(calibrator.to_dict())

    expected = calibrator.calibrated_risk(
        model, calibration.histories, calibration.graph
    )
    actual = restored.calibrated_risk(
        model, calibration.histories, calibration.graph
    )
    np.testing.assert_allclose(expected, actual, atol=1e-12)
    assert np.all((0.0 <= actual) & (actual <= 1.0))
    assert calibrator.alert_decisions(
        model, calibration.histories, calibration.graph
    ).dtype == np.bool_


def test_sd_jepa_smoke_artifact_reassesses_from_stored_arrays(
    tmp_path: Path,
) -> None:
    output = tmp_path / "artifact"
    run_experiment(
        cache_directory=FROZEN_CACHE,
        output_directory=output,
        pretrain_steps=1,
        latency_repetitions=1,
        allow_noninterpretable_smoke=True,
        expected_pair_count=40,
    )

    assessment = assess_stored_bundle(output)
    assert assessment["eligible_for_advance"] is False
    assert assessment["passed"] is False
    assert assessment["decision"] == "non_interpretable_sd_jepa_smoke"
    verify_stored_assessment(output)
    verify_artifact_manifest(output)


def tiny_action_conditioned_windows(
    *,
    pair_count: int,
    transition_count: int,
    pair_prefix: str = "pair",
) -> ActionConditionedWindows:
    rng = np.random.default_rng(15015)
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
    histories = []
    futures = []
    controls = []
    actions = []
    trajectory_ids = []
    pair_ids = []
    transitions = []
    for pair in range(pair_count):
        base = rng.normal(scale=0.05, size=(36, 7, 3)).cumsum(axis=0)
        for arm in ("control", "treatment"):
            trajectory = base.copy()
            action = np.zeros((35, 7, len(action_names)))
            action[..., 0] = 1.0
            if arm == "treatment":
                trajectory[22:, 3:, 0] += np.linspace(
                    0.0, 3.0, len(trajectory) - 22
                )[:, None]
                action[21:25, 2, 0] = 0.0
                action[21:25, 2, 1] = 1.0
                action[21:25, 2, 2] = 1.0
            for offset in range(transition_count):
                transition = 19 + offset
                histories.append(trajectory[offset : offset + 20])
                futures.append(
                    trajectory[transition + 1 : transition + 11]
                )
                control = np.zeros((10, 2))
                control[:, 1] = float(pair == pair_count - 1)
                controls.append(control)
                actions.append(action[transition : transition + 10])
                trajectory_ids.append(f"{pair_prefix}-{pair}-{arm}")
                pair_ids.append(f"{pair_prefix}-{pair}")
                transitions.append(transition)
    return ActionConditionedWindows(
        histories=np.asarray(histories, dtype=np.float64),
        future_states=np.asarray(futures, dtype=np.float64),
        future_controls=np.asarray(controls, dtype=np.float64),
        future_actions=np.asarray(actions, dtype=np.float64),
        trajectory_ids=tuple(trajectory_ids),
        matched_pair_ids=tuple(pair_ids),
        transition_indices=np.asarray(transitions, dtype=np.int64),
        entity_names=entities,
        state_feature_names=features,
        control_feature_names=("request_demand", "worker_replicas"),
        action_feature_names=action_names,
        graph=graph,
    )


_tiny_windows = tiny_action_conditioned_windows
