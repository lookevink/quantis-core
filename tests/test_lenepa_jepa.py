from pathlib import Path

import numpy as np

from lab.action_dynamics.prototype_lenepa_jepa import (
    FROZEN_CACHE,
    run_experiment,
)
from lab.action_dynamics.prototype_lenepa_jepa_assessor import (
    FROZEN_PREPROCESSING_PROTOCOL,
    FROZEN_SOURCE_ARTIFACT_MANIFEST_SHA256,
    FROZEN_SOURCE_CORPUS_SHA256,
    _recompute_interpretable,
    verify_stored_assessment,
)
from quantis_core.action_conditioned_dynamics import (
    ActionConditionedWindows,
)
from quantis_core.edge_dynamics.complete_lejepa import (
    fit_owned_feature_mask,
)
from quantis_core.edge_dynamics.action_conditioned_jepa import (
    sketched_isotropic_gaussian_regularization,
)
from quantis_core.edge_dynamics.lenepa_jepa import (
    LenepaConfig,
    LenepaRepresentation,
    assess_lenepa_gates,
    lenepa_objective,
)
from quantis_core.graph_telemetry import (
    DeclaredTelemetryGraph,
    GraphEntity,
    TelemetryBinding,
)


def test_lenepa_objective_matches_literal_projected_reference() -> None:
    import torch

    layer_zero = (
        torch.arange(96, dtype=torch.float64).reshape(3, 4, 8) / 41
    ).requires_grad_(True)
    layer_final = (
        layer_zero.detach().flip(1).clone().requires_grad_(True)
    )
    projector = torch.nn.Linear(
        8, 8, bias=False, dtype=torch.float64
    )
    with torch.no_grad():
        projector.weight.copy_(torch.eye(8, dtype=torch.float64))
    generator = torch.Generator(device="cpu").manual_seed(24024)
    reference_generator = torch.Generator(device="cpu").manual_seed(24024)

    result = lenepa_objective(
        layer_zero,
        layer_final,
        objective="projected_lenepa",
        projector=projector,
        generator=generator,
        sketch_dimension=7,
        knot_count=5,
        sigreg_weight=20.0,
    )
    projected_input = projector(layer_zero)
    projected_output = projector(layer_final)
    prediction = projected_output[:, :-1]
    target = projected_input[:, 1:]
    prediction_mse = ((prediction - target) ** 2).mean()
    temporal_sigreg = sketched_isotropic_gaussian_regularization(
        torch.cat((projected_input, projected_output), dim=0),
        generator=reference_generator,
        sketch_dimension=7,
        knot_count=5,
    )

    torch.testing.assert_close(result.prediction_mse, prediction_mse)
    torch.testing.assert_close(result.temporal_sigreg, temporal_sigreg)
    torch.testing.assert_close(
        result.loss, prediction_mse + 20.0 * temporal_sigreg
    )
    result.loss.backward()
    assert layer_final.grad is not None
    assert layer_zero.grad is not None
    assert projector.weight.grad is not None
    assert torch.all(torch.isfinite(layer_final.grad))
    assert torch.all(torch.isfinite(layer_zero.grad))
    assert torch.all(torch.isfinite(projector.weight.grad))
    assert torch.equal(
        generator.get_state(), reference_generator.get_state()
    )


def test_lenepa_representation_is_causal_restorable_and_capacity_matched() -> None:
    import torch

    windows = _tiny_windows(pair_count=4, transition_count=5)
    configs = [
        LenepaConfig(
            objective=objective,
            width=8,
            depth=2,
            head_count=2,
            feedforward_width=16,
            projector_hidden_width=16,
            projector_width=8,
            steps=1,
            warmup_steps=1,
            expected_pair_count=4,
            sketch_dimension=7,
        )
        for objective in (
            "projected_lenepa",
            "unprojected_lenepa",
            "projected_sigreg_only",
        )
    ]
    torch.manual_seed(72024)
    before = torch.random.get_rng_state().clone()
    models = [LenepaRepresentation(config).fit(windows) for config in configs]
    np.testing.assert_array_equal(
        torch.random.get_rng_state().numpy(), before.numpy()
    )

    inference_counts = {
        model.inference_parameter_count for model in models
    }
    training_counts = {model.training_parameter_count for model in models}
    assert len(inference_counts) == 1
    assert len(training_counts) == 1

    model = models[0]
    encoded = model.encode(windows.histories[:3], windows.graph)
    sequence = model.encode_sequence(
        windows.histories[:3], windows.graph
    )
    diagnostic = model.diagnose_next_latent(
        windows.histories[:3], windows.graph
    )
    altered = windows.histories[:3].copy()
    altered[:, 10:] += 10_000.0
    altered_sequence = model.encode_sequence(altered, windows.graph)
    restored = LenepaRepresentation.from_dict(model.to_dict())
    inference_payload = model.to_inference_dict()
    inference = LenepaRepresentation.from_inference_dict(
        inference_payload
    )

    assert encoded.tokens.shape == (3, 7, 8)
    np.testing.assert_array_equal(
        encoded.observation_mask, fit_owned_feature_mask(windows)
    )
    assert sequence.shape == (3, 20, 8)
    assert diagnostic.input_tokens.shape == (3, 20, 8)
    assert diagnostic.output_tokens.shape == (3, 20, 8)
    assert diagnostic.predicted_tokens.shape == (3, 19, 8)
    assert diagnostic.target_tokens.shape == (3, 19, 8)
    assert np.isfinite(diagnostic.cosine_error)
    assert 0.0 <= diagnostic.retrieval_hit_at_1 <= 1.0
    np.testing.assert_allclose(
        sequence[:, :10], altered_sequence[:, :10], atol=1e-7
    )
    np.testing.assert_allclose(
        encoded.tokens,
        restored.encode(
            windows.histories[:3], windows.graph
        ).tokens,
        atol=1e-7,
    )
    assert "projector_state" not in inference_payload
    np.testing.assert_allclose(
        encoded.tokens,
        inference.encode(
            windows.histories[:3], windows.graph
        ).tokens,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        sequence,
        restored.encode_sequence(
            windows.histories[:3], windows.graph
        ),
        atol=1e-7,
    )
    restored_diagnostic = restored.diagnose_next_latent(
        windows.histories[:3], windows.graph
    )
    np.testing.assert_allclose(
        diagnostic.input_tokens,
        restored_diagnostic.input_tokens,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        diagnostic.output_tokens,
        restored_diagnostic.output_tokens,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        diagnostic.predicted_tokens,
        restored_diagnostic.predicted_tokens,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        diagnostic.target_tokens,
        restored_diagnostic.target_tokens,
        atol=1e-7,
    )


def test_lenepa_assessment_rejects_failed_raw_safety() -> None:
    names = (
        "projected_lenepa",
        "unprojected_lenepa",
        "projected_sigreg_only",
        "matched_pca",
    )
    scores = {
        name: {
            role: {
                "overall_mse": (
                    2.0
                    if name == "projected_lenepa"
                    and role == "transfer_evaluation"
                    else 1.0
                ),
                "action_overlap_mse": 1.0,
                "downstream_effect_mse": (
                    0.5 if name == "projected_lenepa" else 1.0
                ),
            }
            for role in ("selection", "transfer_evaluation")
        }
        for name in names
    }
    mechanism = {
        "projected_lenepa": {
            role: {"cosine_error": 0.5, "retrieval_hit_at_1": 0.8}
            for role in ("selection", "transfer_evaluation")
        },
        "unprojected_lenepa": {
            role: {"cosine_error": 1.0, "retrieval_hit_at_1": 0.6}
            for role in ("selection", "transfer_evaluation")
        },
    }

    assessment = assess_lenepa_gates(
        forecast_scores=scores,
        raw_scores={
            role: {
                "overall_mse": 1.0,
                "action_overlap_mse": 1.0,
                "downstream_effect_mse": 1.0,
            }
            for role in ("selection", "transfer_evaluation")
        },
        mechanism=mechanism,
        attribution={
            name: {
                "action_and_target_hit_at_1": 1.0,
                "no_action_specificity": 1.0,
            }
            for name in names
        },
        action_sanity={
            name: {"correct_action_beats_both_fraction": 1.0}
            for name in names
        },
        restoration_max_abs={name: 0.0 for name in names},
        protocol_checks={
            key: True
            for key in (
                "evidence_arrays_are_finite",
                "pair_and_trajectory_roles_are_disjoint",
                "capacity_recomputes",
                "public_inference_is_causal",
                "prefix_invariance_recomputes",
                "anchor_schedule_recomputes",
                "selection_only_ridge_choice_recomputes",
                "selection_safety_status_recomputes",
                "bundle_size_recomputes",
                "latency_recomputes",
                "mechanism_history_coverage_recomputes",
                "diagnostic_shift_consistency_recomputes",
            )
        },
        parameter_counts={
            name: {"training": 10, "inference": 5}
            for name in names[:3]
        },
        transfer_pair_errors={
            name: {"p0": 0.5 if name == "projected_lenepa" else 1.0}
            for name in names
        },
        deployed_bundle_bytes=1024,
        median_latency_ms=0.5,
    )

    assert not assessment["passed"]
    assert not assessment["safety_gates"][
        "transfer_overall_within_1_05_raw"
    ]
    assert assessment["mechanism_gates"]["projected_prediction_advantage"]
    assert assessment["decision"] == "reject_lenepa_telemetry_recipe"


def test_lenepa_nonfrozen_role_cannot_be_promoted() -> None:
    metadata = {
        "interpretable": False,
        "source_corpus_sha256": FROZEN_SOURCE_CORPUS_SHA256,
        "source_artifact_manifest_sha256": (
            FROZEN_SOURCE_ARTIFACT_MANIFEST_SHA256
        ),
        "preprocessing_protocol": FROZEN_PREPROCESSING_PROTOCOL,
    }

    assert not _recompute_interpretable(
        metadata=metadata,
        frozen_controls=True,
        latency_repetitions=100,
        mechanism_history_coverage=True,
    )
    metadata["interpretable"] = True
    assert _recompute_interpretable(
        metadata=metadata,
        frozen_controls=True,
        latency_repetitions=100,
        mechanism_history_coverage=True,
    )


def test_lenepa_smoke_artifact_reassesses_from_stored_arrays(
    tmp_path: Path,
) -> None:
    output = tmp_path / "lenepa-smoke"
    run_experiment(
        cache_directory=FROZEN_CACHE,
        output_directory=output,
        steps=1,
        latency_repetitions=1,
        allow_noninterpretable_smoke=True,
    )

    assessment = verify_stored_assessment(output)
    assert assessment["decision"] == "non_interpretable_lenepa_smoke"
    assert all(assessment["protocol_checks"].values())
    assert not assessment["eligible_for_advance"]
    assert assessment["safety_gates"][
        "deployed_bundle_within_16_mib"
    ]
    assert (output / "artifact-manifest.json").is_file()
    assert (output / "reproduction-source").is_dir()


def _tiny_windows(
    *, pair_count: int, transition_count: int
) -> ActionConditionedWindows:
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
                f"metric.declared{index}", f"e{index}"
            )
            for index in range(6)
        ),
    )
    rows = []
    trajectory_ids = []
    pair_ids = []
    transitions = []
    for pair in range(pair_count):
        for arm in range(2):
            for transition in range(transition_count):
                values = np.zeros((20, 7, 2), dtype=np.float64)
                for entity in range(6):
                    values[:, entity, entity % 2] = (
                        np.arange(20) + pair + arm + transition + entity
                    )
                rows.append(values)
                trajectory_ids.append(f"pair{pair}-arm{arm}")
                pair_ids.append(f"pair{pair}")
                transitions.append(transition)
    sample_count = len(rows)
    return ActionConditionedWindows(
        histories=np.asarray(rows),
        future_states=np.zeros(
            (sample_count, 10, 7, 2), dtype=np.float64
        ),
        future_controls=np.zeros(
            (sample_count, 10, 1), dtype=np.float64
        ),
        future_actions=np.zeros(
            (sample_count, 10, 7, 1), dtype=np.float64
        ),
        trajectory_ids=tuple(trajectory_ids),
        matched_pair_ids=tuple(pair_ids),
        transition_indices=np.asarray(transitions, dtype=np.int64),
        entity_names=graph.entity_ids,
        state_feature_names=("f0", "f1"),
        control_feature_names=("control",),
        action_feature_names=("action",),
        graph=graph,
    )
