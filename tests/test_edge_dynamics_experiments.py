import json

import numpy as np

from quantis_core.action_conditioned_dynamics import (
    ActionTrajectoryCompiler,
)
from quantis_core.action_dynamics_synthetic import synthetic_action_runs
from quantis_core.edge_dynamics.evaluation import (
    CountMinSketch,
    audit_streaming_log_templates,
    conformal_sequential_detection,
    forecast_objective,
)
from quantis_core.edge_dynamics.models import (
    BoundedGraphResidualDynamics,
    ContractiveLowRankDynamics,
    EchoStateActionDynamics,
    EchoStateConfig,
    GraphResidualConfig,
    LowRankConfig,
)
from quantis_core.edge_dynamics.temporal_convolution import (
    DirectTemporalConvDynamics,
    TemporalConvConfig,
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


def test_compact_dynamics_models_share_the_rollout_seam() -> None:
    training, validation = _windows()
    models = (
        EchoStateActionDynamics(
            EchoStateConfig(reservoir_size=8, seed=4)
        ),
        ContractiveLowRankDynamics(
            LowRankConfig(rank=3, maximum_spectral_radius=0.9)
        ),
        BoundedGraphResidualDynamics(
            GraphResidualConfig(
                global_config=LowRankConfig(
                    rank=3, maximum_spectral_radius=0.9
                ),
                residual_gain=0.1,
            )
        ),
    )

    for model in models:
        model.fit(training)
        distribution = model.rollout(
            validation.histories,
            validation.future_controls,
            validation.future_actions,
            validation.graph,
        )
        assert distribution.mean.shape == validation.future_states.shape
        assert np.all(np.isfinite(distribution.mean))
        assert np.all(distribution.variance > 0.0)
        assert model.parameter_count > 0
        scores = forecast_objective(model, validation)
        assert scores["normalized_mse_action_overlap"] >= 0.0

    low_rank = models[1]
    assert isinstance(low_rank, ContractiveLowRankDynamics)
    assert low_rank.spectral_radius <= 0.9 + 1e-12


def test_temporal_convolution_predicts_the_horizon_directly() -> None:
    training, validation = _windows()
    model = DirectTemporalConvDynamics(
        TemporalConvConfig(
            hidden_channels=4,
            epochs=1,
            batch_size=64,
            seed=8,
        )
    ).fit(training)

    distribution = model.rollout(
        validation.histories[:7],
        validation.future_controls[:7],
        validation.future_actions[:7],
        validation.graph,
    )

    assert distribution.mean.shape == validation.future_states[:7].shape
    assert np.all(np.isfinite(distribution.mean))
    assert model.parameter_count > 0
    assert model.to_dict()["horizon"] == 3


def test_count_min_sketch_never_underestimates_positive_updates() -> None:
    sketch = CountMinSketch(width=8, depth=3, seed=2)
    exact = {"api:accepted": 10.0, "worker:completed": 7.0}
    for key, value in exact.items():
        sketch.update(key, value)

    assert all(
        sketch.estimate(key) >= value
        for key, value in exact.items()
    )
    assert sketch.storage_bytes == 8 * 3 * 8


def test_conformal_detector_hides_actions_and_reports_both_alarm_modes() -> None:
    training, validation = _windows()
    model = ContractiveLowRankDynamics(
        LowRankConfig(rank=3)
    ).fit(training)

    result = conformal_sequential_detection(
        model=model,
        calibration=validation,
        evaluation=validation,
        alpha=0.1,
    )

    assert 0.0 <= result["evaluation_control_point_alarm_rate"] <= 1.0
    assert (
        0.0
        <= result["evaluation_control_sequential_false_alarm_rate"]
        <= 1.0
    )
    assert 0.0 <= result["evaluation_treatment_point_detection_rate"] <= 1.0
    assert (
        0.0
        <= result["evaluation_treatment_sequential_detection_rate"]
        <= 1.0
    )


def test_streaming_template_audit_masks_variable_tokens(tmp_path) -> None:
    cases = tmp_path / "cases" / "case-a"
    cases.mkdir(parents=True)
    payloads = (
        {"body": {"stringValue": "checkout accepted 123"}},
        {"body": {"stringValue": "checkout accepted 456"}},
        {"body": {"stringValue": "worker completed"}},
    )
    (cases / "collector-logs.jsonl").write_text(
        "\n".join(
            json.dumps(payload, separators=(",", ":"))
            for payload in payloads
        )
        + "\n"
    )

    audit = audit_streaming_log_templates(tmp_path)

    assert audit["message_count"] == 3
    assert audit["template_count"] == 2
    assert audit["template_counts"] == {
        "checkout accepted <*>": 2,
        "worker completed": 1,
    }
