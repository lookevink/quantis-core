import numpy as np
import pytest

from quantis_core.action_conditioned_dynamics import (
    ActionConditionedWindows,
    ActionTrajectoryCompiler,
)
from quantis_core.action_dynamics_synthetic import synthetic_action_runs
from quantis_core.edge_dynamics.action_conditioned_jepa import (
    ActionConditionedJepaConfig,
    ActionConditionedJepaDynamics,
)
from quantis_core.edge_dynamics.data import (
    partition_worker_topology,
)
from quantis_core.edge_dynamics.jepa_evaluation import (
    action_conditioning_sanity,
    assess_action_conditioned_jepa_development,
    write_action_conditioned_jepa_artifacts,
)
from quantis_core.edge_dynamics.evaluation import (
    conformal_sequential_detection,
)
from quantis_core.edge_dynamics.models import (
    ContractiveLowRankDynamics,
    LowRankConfig,
)


def _windows():
    training_runs = synthetic_action_runs(
        20, split="training", seed=100
    )
    validation_runs = synthetic_action_runs(
        10, split="validation", seed=900
    )
    compiler = ActionTrajectoryCompiler(
        context_length=5, rollout_horizon=3
    ).fit(training_runs)
    return (
        compiler.transform(training_runs),
        compiler.transform(validation_runs),
    )


def test_action_conditioned_jepa_preserves_node_tokens_and_roundtrips() -> None:
    training, validation = _windows()
    config = ActionConditionedJepaConfig(
        node_latent_dimension=2,
        transition_rank=3,
        epochs=1,
        batch_size=256,
        device="cpu",
        seed=7,
    )
    model = ActionConditionedJepaDynamics(config).fit(training)

    first = model.rollout(
        validation.histories[:8],
        validation.future_controls[:8],
        validation.future_actions[:8],
        validation.graph,
    )
    embeddings = model.encode_histories(
        validation.histories[:8], validation.graph
    )
    restored = ActionConditionedJepaDynamics.from_dict(
        model.to_dict()
    )
    second = restored.rollout(
        validation.histories[:8],
        validation.future_controls[:8],
        validation.future_actions[:8],
        validation.graph,
    )
    detection = conformal_sequential_detection(
        model=model,
        calibration=validation,
        evaluation=validation,
        alpha=0.1,
    )

    assert first.mean.shape == validation.future_states[:8].shape
    assert embeddings.shape == (8, len(validation.entity_names), 2)
    assert np.all(np.isfinite(first.mean))
    assert np.all(first.variance > 0.0)
    assert np.allclose(first.mean, second.mean, atol=1e-6)
    assert model.parameter_count > 0
    assert model.spectral_radius <= 0.98 + 1e-6
    assert model.to_dict()["kind"] == (
        "action_conditioned_jepa_low_rank_dynamics_v1"
    )
    assert (
        0.0
        <= detection[
            "evaluation_control_sequential_false_alarm_rate"
        ]
        <= 1.0
    )


def test_worker_topology_partition_holds_out_complete_pairs() -> None:
    training, _ = _windows()
    pair_ids = tuple(sorted(set(training.matched_pair_ids)))
    topology_by_pair = {
        pair_id: float(position % 3)
        for position, pair_id in enumerate(pair_ids)
    }
    controls = np.concatenate(
        (
            training.future_controls,
            np.asarray(
                [
                    np.full(
                        (training.future_controls.shape[1], 1),
                        topology_by_pair[pair_id],
                    )
                    for pair_id in training.matched_pair_ids
                ],
                dtype=np.float64,
            ),
        ),
        axis=2,
    )
    with_topology = ActionConditionedWindows(
        histories=training.histories,
        future_states=training.future_states,
        future_controls=controls,
        future_actions=training.future_actions,
        trajectory_ids=training.trajectory_ids,
        matched_pair_ids=training.matched_pair_ids,
        transition_indices=training.transition_indices,
        entity_names=training.entity_names,
        state_feature_names=training.state_feature_names,
        control_feature_names=(
            *training.control_feature_names,
            "worker_replicas",
        ),
        action_feature_names=training.action_feature_names,
        graph=training.graph,
    )

    partition = partition_worker_topology(with_topology)

    fit_pairs = set(partition.in_distribution.matched_pair_ids)
    held_out_pairs = set(partition.held_out.matched_pair_ids)
    assert fit_pairs.isdisjoint(held_out_pairs)
    assert fit_pairs | held_out_pairs == set(pair_ids)
    assert partition.held_out_normalized_value == 2.0
    assert np.all(
        partition.held_out.future_controls[..., -1] == 2.0
    )


def test_jepa_development_assessment_keeps_confirmation_sealed() -> None:
    assessment = assess_action_conditioned_jepa_development(
        baseline_transfer={
            "downstream_effect_mse": 1.0,
            "normalized_mse_action_overlap": 1.0,
        },
        jepa_transfer={
            "downstream_effect_mse": 0.8,
            "normalized_mse_action_overlap": 1.02,
            "action_and_target_hit_at_1": 0.95,
        },
        token_diagnostics={
            "effective_rank": 5.0,
            "latent_dimension": 16,
        },
        action_sanity={
            "correct_action_beats_both_fraction": 0.9,
        },
        detection={
            "evaluation_control_sequential_false_alarm_rate": 0.0,
            "evaluation_treatment_sequential_detection_rate": 0.85,
            "median_sequential_detection_delay_transitions": 8.0,
        },
        seed_robustness={
            "seed_count": 3,
            "required_seed_count": 3,
            "passed": True,
        },
    )

    assert assessment["predictive_development_gates_passed"] is True
    assert assessment["anomaly_development_gates_passed"] is True
    assert assessment["decision"] == "advance_to_sealed_confirmation"
    assert assessment["sealed_confirmation"] is False


def test_jepa_development_artifacts_are_bounded_and_non_overwriting(
    tmp_path,
) -> None:
    scores = {
        "normalized_mse_action_overlap": 0.4,
        "normalized_mse_overall": 0.2,
        "downstream_effect_mse": 0.1,
        "action_and_target_hit_at_1": 0.9,
        "parameter_count": 12,
        "batch_one_latency_ms": 0.3,
    }
    report = {
        "assessment": {"decision": "reject_this_configuration"},
        "transfer_scores": {"jepa_latent_low_rank": scores},
    }
    output = tmp_path / "evidence"

    manifest = write_action_conditioned_jepa_artifacts(
        output_directory=output,
        report=report,
        model_artifacts={"jepa": {"kind": "test-model"}},
    )

    assert manifest["kind"] == (
        "action_conditioned_jepa_low_rank_development_manifest"
    )
    assert "not sealed confirmation" in (
        output / "report.md"
    ).read_text()
    with pytest.raises(FileExistsError):
        write_action_conditioned_jepa_artifacts(
            output_directory=output,
            report=report,
            model_artifacts={},
        )


def test_action_sanity_shuffles_complete_treatment_trajectories() -> None:
    training, validation = _windows()
    model = ContractiveLowRankDynamics(
        LowRankConfig(rank=3)
    ).fit(training)

    result = action_conditioning_sanity(
        model, validation, seed=19
    )

    assert result["treatment_pair_count"] == 5
    assert (
        0.0
        <= result["correct_action_beats_both_fraction"]
        <= 1.0
    )
