import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from lab.action_dynamics.prototype_sc_jepa_interaction import (
    run_experiment,
    write_artifact_manifest,
)
from lab.action_dynamics.prototype_sc_jepa_interaction_assessor import (
    assess_stored_bundle,
    verify_artifact_manifest,
    verify_stored_assessment,
)
from quantis_core.action_conditioned_dynamics import (
    ActionConditionedWindows,
)
from quantis_core.edge_dynamics.hepa_jepa import (
    HepaEventDefinition,
    calibrate_probability_surface,
    fit_logit_calibrator,
    trajectory_alert_threshold,
)
from quantis_core.edge_dynamics.data import (
    EdgePairRoles,
    PreparedAttributionQueries,
    PreparedEdgeDynamicsData,
    write_edge_dynamics_cache,
)
from quantis_core.edge_dynamics.sc_jepa import (
    SC_JEPA_ASSESSMENT_MODEL_NAMES,
    SC_JEPA_ASSESSMENT_ROLE_NAMES,
    SC_JEPA_CELL_NAMES,
    ScEncodedTelemetry,
    ScJepaConfig,
    ScJepaModel,
    assess_sc_jepa_interaction,
    sc_jepa_views,
)
from quantis_core.graph_telemetry import (
    DeclaredTelemetryGraph,
    GraphEntity,
    TelemetryBinding,
)


def test_sc_jepa_views_freeze_fine_and_coarse_time_axes() -> None:
    histories = np.arange(20, dtype=np.float64).reshape(
        1, 20, 1, 1
    )
    future = np.arange(20, 30, dtype=np.float64).reshape(
        1, 10, 1, 1
    )

    views = sc_jepa_views(histories, future)

    assert views.context_fine.shape == (1, 1, 5, 2, 1)
    assert views.future_fine.shape == (1, 1, 5, 2, 1)
    assert views.future_coarse.shape == (1, 1, 1, 2, 1)
    np.testing.assert_array_equal(
        views.context_fine[0, 0, :, :, 0],
        np.arange(10, 20).reshape(5, 2),
    )
    np.testing.assert_array_equal(
        views.future_coarse[0, 0, 0, :, 0],
        np.asarray([22.0, 27.0]),
    )


def test_factorial_cells_match_capacity_and_restore_public_outputs() -> None:
    fit = _tiny_windows(pair_count=4, transition_count=6)
    selection = _tiny_windows(
        pair_count=2,
        transition_count=6,
        pair_prefix="selection",
    )
    event = HepaEventDefinition.fit(fit)
    base = ScJepaConfig(
        width=16,
        code_count=16,
        head_count=4,
        feedforward_width=32,
        pretrain_steps=2,
        alert_steps=2,
        checkpoint_interval=1,
        alert_checkpoint_interval=1,
        batch_size=8,
        expected_pair_count=4,
        seed=13013,
    )
    models = {}
    for name, use_codebook, multi_resolution in (
        ("continuous_single", False, False),
        ("continuous_multi", False, True),
        ("codebook_single", True, False),
        ("codebook_multi", True, True),
    ):
        model = ScJepaModel(
            replace(
                base,
                use_codebook=use_codebook,
                multi_resolution=multi_resolution,
            )
        )
        model.fit(fit).select(selection)
        models[name] = model

    assert len(
        {
            model.training_parameter_count
            for model in models.values()
        }
    ) == 1
    assert len(
        {
            model.inference_parameter_count
            for model in models.values()
        }
    ) == 1

    candidate = models["codebook_multi"]
    candidate.fit_alert_head(fit, event).select_alert_head(
        selection, event
    ).fit_calibration(selection, event)
    restored = ScJepaModel.from_dict(candidate.to_dict())
    histories = fit.histories[:3]

    encoded = candidate.encode(histories, fit.graph)
    restored_encoded = restored.encode(histories, fit.graph)
    risks = candidate.predict_risk(histories, fit.graph)
    restored_risks = restored.predict_risk(histories, fit.graph)
    calibrated = candidate.calibrated_risk(histories, fit.graph)
    restored_calibrated = restored.calibrated_risk(
        histories, fit.graph
    )

    assert encoded.tokens.shape == (3, 7, 16)
    assert encoded.patch_values.shape == (3, 7, 5, 16)
    assert encoded.code_probabilities is not None
    assert restored_encoded.code_probabilities is not None
    np.testing.assert_allclose(
        np.sum(encoded.code_probabilities, axis=-1),
        1.0,
        atol=1e-5,
    )
    with pytest.raises(ValueError, match="do not align"):
        ScEncodedTelemetry(
            tokens=encoded.tokens,
            patch_values=encoded.patch_values,
            entity_ids=encoded.entity_ids,
            ownership_mask=encoded.ownership_mask,
            code_probabilities=encoded.code_probabilities * 2.0,
        )
    np.testing.assert_allclose(
        encoded.tokens, restored_encoded.tokens, atol=1e-7
    )
    np.testing.assert_allclose(
        encoded.patch_values,
        restored_encoded.patch_values,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        encoded.code_probabilities,
        restored_encoded.code_probabilities,
        atol=1e-7,
    )
    np.testing.assert_allclose(risks, restored_risks, atol=1e-7)
    np.testing.assert_allclose(
        calibrated, restored_calibrated, atol=1e-7
    )
    np.testing.assert_array_equal(
        candidate.alert_decisions(histories, fit.graph),
        restored.alert_decisions(histories, fit.graph),
    )
    with pytest.raises(TypeError):
        candidate.predict_risk(  # type: ignore[call-arg]
            histories,
            fit.graph,
            future_states=fit.future_states[:3],
        )

    corrupted = copy.deepcopy(candidate.to_dict())
    corrupted["state_dict"][
        "online_encoder.entity_embedding.weight"
    ]["values"][0][0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        ScJepaModel.from_dict(corrupted)


def test_assessment_requires_a_positive_factorial_interaction() -> None:
    ids = ("c0", "c1", "t0", "t1")
    onsets = {"c0": None, "c1": None, "t0": 0, "t1": 0}
    labels = np.asarray([False, False, True, True])
    calibration_risk = np.asarray([0.05, 0.08, 0.85, 0.90])
    reference_risk = np.asarray([0.05, 0.08, 0.07, 0.06])
    candidate_risk = np.asarray([0.05, 0.08, 0.85, 0.90])
    risks = {
        role: {
            model: (
                calibration_risk.copy()
                if role == "calibration"
                else (
                    candidate_risk.copy()
                    if model == "codebook_multi"
                    else reference_risk.copy()
                )
            )
            for model in SC_JEPA_ASSESSMENT_MODEL_NAMES
        }
        for role in SC_JEPA_ASSESSMENT_ROLE_NAMES
    }
    calibrated = {
        role: {}
        for role in SC_JEPA_ASSESSMENT_ROLE_NAMES
    }
    decisions = {
        role: {}
        for role in SC_JEPA_ASSESSMENT_ROLE_NAMES
    }
    calibrations = {}
    for model in SC_JEPA_ASSESSMENT_MODEL_NAMES:
        slope, intercept, brier = fit_logit_calibrator(
            risks["calibration"][model][:, None],
            labels[:, None],
        )
        for role in SC_JEPA_ASSESSMENT_ROLE_NAMES:
            calibrated[role][model] = (
                calibrate_probability_surface(
                    risks[role][model][:, None],
                    slope=slope,
                    intercept=intercept,
                )[:, 0]
            )
        threshold = trajectory_alert_threshold(
            calibrated["calibration"][model][:, None],
            ids,
            ("c0", "c1"),
        )
        calibrations[model] = {
            "slope": slope,
            "intercept": intercept,
            "calibration_brier": brier,
            "alert_threshold": threshold,
        }
        for role in SC_JEPA_ASSESSMENT_ROLE_NAMES:
            decisions[role][model] = (
                calibrated[role][model] > threshold
            )
    codes = np.zeros((4, 2, 2, 16), dtype=np.float64)
    codes.reshape(-1, 16)[np.arange(16), np.arange(16)] = 1.0
    tokens = np.arange(32, dtype=np.float64).reshape(4, 2, 4)
    truth = np.arange(8, dtype=np.float64).reshape(4, 2, 1)
    assessment_arguments = dict(
        risks=risks,
        restored_risks=risks,
        stored_calibrated_risks=calibrated,
        restored_calibrated_risks=calibrated,
        stored_alert_decisions=decisions,
        restored_alert_decisions=decisions,
        stored_calibrations=calibrations,
        restored_calibrations=calibrations,
        labels={
            role: labels.copy()
            for role in SC_JEPA_ASSESSMENT_ROLE_NAMES
        },
        trajectory_ids={
            role: ids for role in SC_JEPA_ASSESSMENT_ROLE_NAMES
        },
        transition_indices={
            role: np.ones(4, dtype=np.int64)
            for role in SC_JEPA_ASSESSMENT_ROLE_NAMES
        },
        trajectory_onsets={
            role: onsets
            for role in SC_JEPA_ASSESSMENT_ROLE_NAMES
        },
        representation_tokens={
            name: tokens.copy() for name in SC_JEPA_CELL_NAMES
        },
        restored_representation_tokens={
            name: tokens.copy() for name in SC_JEPA_CELL_NAMES
        },
        representation_patch_values={
            name: tokens[:, :, None, :].copy()
            for name in SC_JEPA_CELL_NAMES
        },
        restored_representation_patch_values={
            name: tokens[:, :, None, :].copy()
            for name in SC_JEPA_CELL_NAMES
        },
        representation_code_probabilities={
            name: (
                codes.copy() if name.startswith("codebook_") else None
            )
            for name in SC_JEPA_CELL_NAMES
        },
        restored_representation_code_probabilities={
            name: (
                codes.copy() if name.startswith("codebook_") else None
            )
            for name in SC_JEPA_CELL_NAMES
        },
        state_truth=truth,
        state_scale=np.ones((2, 1), dtype=np.float64),
        state_varying_mask=np.ones((2, 1), dtype=np.bool_),
        state_predictions={
            "codebook_multi": truth.copy(),
            "matched_pca": truth + 1.0,
        },
        training_parameter_counts={
            name: 100 for name in SC_JEPA_CELL_NAMES
        },
        inference_parameter_counts={
            name: 50 for name in SC_JEPA_CELL_NAMES
        },
        protocol_checks={
            "derived_protocol": True,
            "frozen_interpretable_contract": True,
        },
        edge_metrics={
            model: {
                "inference_parameter_count": 50.0,
                "serialized_candidate_sidecars_bytes": 100.0,
                "batch_one_cpu_latency_ms": 1.0,
                "batch_one_cpu_p95_latency_ms": 2.0,
                "peak_rss_bytes": 1024.0,
                "latency_repetitions": 3.0,
            }
            for model in SC_JEPA_ASSESSMENT_MODEL_NAMES
        },
    )
    assessment = assess_sc_jepa_interaction(**assessment_arguments)

    assert assessment["passed"] is True
    assert assessment["interactions"][
        "held_transfer_brier"
    ] > 0.0
    assert assessment["interactions"][
        "held_transfer_detection"
    ] > 0.0
    assert (
        assessment["alert_metrics"]["evaluation_transfer"][
            "codebook_multi"
        ]["alerts_per_logical_run"]
        >= 0.0
    )

    negative_risks = copy.deepcopy(risks)
    negative_calibrated = copy.deepcopy(calibrated)
    negative_decisions = copy.deepcopy(decisions)
    negative_risks["evaluation_transfer"]["codebook_multi"] = (
        reference_risk.copy()
    )
    candidate_calibration = calibrations["codebook_multi"]
    negative_calibrated["evaluation_transfer"]["codebook_multi"] = (
        calibrate_probability_surface(
            reference_risk[:, None],
            slope=candidate_calibration["slope"],
            intercept=candidate_calibration["intercept"],
        )[:, 0]
    )
    negative_decisions["evaluation_transfer"]["codebook_multi"] = (
        negative_calibrated["evaluation_transfer"]["codebook_multi"]
        > candidate_calibration["alert_threshold"]
    )
    negative_arguments = {
        **assessment_arguments,
        "risks": negative_risks,
        "restored_risks": copy.deepcopy(negative_risks),
        "stored_calibrated_risks": negative_calibrated,
        "restored_calibrated_risks": copy.deepcopy(
            negative_calibrated
        ),
        "stored_alert_decisions": negative_decisions,
        "restored_alert_decisions": copy.deepcopy(
            negative_decisions
        ),
    }
    negative = assess_sc_jepa_interaction(**negative_arguments)

    assert negative["interactions"]["held_transfer_brier"] == pytest.approx(
        0.0, abs=1e-12
    )
    assert negative["interactions"][
        "held_transfer_detection"
    ] == pytest.approx(0.0, abs=1e-12)
    assert negative["passed"] is False


def test_sc_jepa_runner_refuses_existing_output(
    tmp_path: Path,
) -> None:
    output = tmp_path / "retained"
    output.mkdir()

    with pytest.raises(FileExistsError):
        run_experiment(
            cache_directory=tmp_path / "missing-cache",
            output_directory=output,
            pretrain_steps=1,
            alert_steps=1,
            latency_repetitions=1,
            allow_noninterpretable_smoke=True,
        )


def test_sc_jepa_manifest_detects_tampering(tmp_path: Path) -> None:
    payload = tmp_path / "payload.json"
    payload.write_text('{"value": 1}\n')
    write_artifact_manifest(tmp_path)
    verify_artifact_manifest(tmp_path)

    payload.write_text('{"value": 2}\n')

    with pytest.raises(ValueError, match="content identity"):
        verify_artifact_manifest(tmp_path)


def test_stored_smoke_cannot_advance_and_recomputes_assessment(
    tmp_path: Path,
) -> None:
    cache = _write_tiny_edge_cache(tmp_path / "cache")
    output = tmp_path / "smoke"

    run_experiment(
        cache_directory=cache,
        output_directory=output,
        pretrain_steps=1,
        alert_steps=1,
        latency_repetitions=1,
        allow_noninterpretable_smoke=True,
        expected_pair_count=2,
    )

    assessment = assess_stored_bundle(output)
    assert assessment["eligible_for_advance"] is False
    assert assessment["passed"] is False
    assert assessment["decision"] == "non_interpretable_sc_jepa_smoke"

    stored = output / "assessment.json"
    stored.write_text(stored.read_text().replace(
        '"passed": false', '"passed": true'
    ))
    write_artifact_manifest(output)
    with pytest.raises(ValueError, match="does not recompute"):
        verify_stored_assessment(output)


def _tiny_windows(
    *,
    pair_count: int,
    transition_count: int,
    pair_prefix: str = "pair",
) -> ActionConditionedWindows:
    rng = np.random.default_rng(913)
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


def _write_tiny_edge_cache(output: Path) -> Path:
    windows = {
        role: _tiny_windows(
            pair_count=3,
            transition_count=6,
            pair_prefix=role,
        )
        for role in ("fit", "selection", "calibration", "evaluation")
    }
    fit = windows["fit"]
    query = PreparedAttributionQueries(
        query_ids=("query",),
        histories=fit.histories[:1].astype(np.float32),
        future_controls=fit.future_controls[:1].astype(np.float32),
        observed_future=fit.future_states[:1].astype(np.float32),
        candidate_actions=fit.future_actions[:1, None].astype(
            np.float32
        ),
        candidate_ids=("candidate",),
        candidate_action_kinds=("worker_pause",),
        candidate_target_entities=("e2",),
        expected_action_kinds=("worker_pause",),
        expected_target_entities=("e2",),
        expected_variant_ids=("variant",),
    )
    roles = EdgePairRoles(
        fit_pair_ids=tuple(sorted(set(fit.matched_pair_ids))),
        selection_pair_ids=tuple(
            sorted(set(windows["selection"].matched_pair_ids))
        ),
        calibration_pair_ids=tuple(
            sorted(set(windows["calibration"].matched_pair_ids))
        ),
        evaluation_pair_ids=tuple(
            sorted(set(windows["evaluation"].matched_pair_ids))
        ),
    )
    compiler = {
        "schema_version": 1,
        "kind": "action_trajectory_compiler",
        "context_length": 20,
        "rollout_horizon": 10,
        "semantic_schema": {
            "graph": fit.graph.to_dict(),
            "state_feature_names": list(fit.state_feature_names),
            "control_feature_names": list(fit.control_feature_names),
            "action_feature_names": list(fit.action_feature_names),
        },
        "semantic_schema_sha256": fit.semantic_schema_sha256,
    }
    write_edge_dynamics_cache(
        PreparedEdgeDynamicsData(
            source_corpus_sha256="a" * 64,
            source_artifact_manifest_sha256="b" * 64,
            roles=roles,
            compiler_artifact=compiler,
            windows=windows,
            attribution_queries=query,
        ),
        output,
    )
    return output
