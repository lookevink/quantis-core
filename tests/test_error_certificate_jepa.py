import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from lab.action_dynamics.prototype_error_certificate_jepa import (
    FROZEN_CACHE,
    run_experiment,
)
from lab.action_dynamics.prototype_error_certificate_jepa_assessor import (
    assess_stored_bundle,
    verify_artifact_manifest,
    verify_stored_assessment,
)
from quantis_core.edge_dynamics.error_certificate_jepa import (
    ERROR_CERTIFICATE_OBJECTIVES,
    CertifiedRawDynamics,
    ErrorCertificateJepa,
    ErrorCertificateJepaConfig,
    _training_arrays,
    realized_raw_error,
)
from quantis_core.edge_dynamics.models import (
    ContractiveLowRankDynamics,
    LowRankConfig,
)
from tests.test_sd_jepa import tiny_action_conditioned_windows


def test_only_deranged_cell_receives_deranged_latent_targets() -> None:
    windows = tiny_action_conditioned_windows(
        pair_count=2, transition_count=6
    )
    raw_prediction = np.zeros_like(windows.future_states)
    ownership = np.ones(
        windows.future_states.shape[2:], dtype=np.bool_
    )
    error_target = realized_raw_error(
        raw_prediction, windows.future_states, ownership
    )
    indices = np.asarray([0, 1], dtype=np.int64)
    donor = np.asarray([1, 0], dtype=np.int64)

    aligned = _training_arrays(
        windows,
        raw_prediction,
        error_target,
        indices=indices,
        donor=donor,
        ownership=ownership,
        derange=False,
    )
    deranged = _training_arrays(
        windows,
        raw_prediction,
        error_target,
        indices=indices,
        donor=donor,
        ownership=ownership,
        derange=True,
    )

    np.testing.assert_array_equal(
        aligned["latent_future"], windows.future_states[indices]
    )
    np.testing.assert_array_equal(
        deranged["latent_future"], windows.future_states[donor]
    )


def test_error_certificate_cells_restore_calibrate_and_match_capacity() -> None:
    fit = tiny_action_conditioned_windows(
        pair_count=4, transition_count=6
    )
    selection = tiny_action_conditioned_windows(
        pair_count=2,
        transition_count=6,
        pair_prefix="selection",
    )
    calibration = tiny_action_conditioned_windows(
        pair_count=2,
        transition_count=6,
        pair_prefix="calibration",
    )
    raw = ContractiveLowRankDynamics(LowRankConfig(rank=8)).fit(fit)
    base = ErrorCertificateJepaConfig(
        width=8,
        hidden_width=16,
        pretrain_steps=2,
        checkpoint_interval=1,
        batch_size=8,
        expected_pair_count=4,
    )
    models = {}
    for objective in ERROR_CERTIFICATE_OBJECTIVES:
        model = ErrorCertificateJepa(replace(base, objective=objective))
        model.fit(fit, raw).select(selection, raw).calibrate(
            calibration, raw
        )
        models[objective] = model

    assert len(
        {model.training_parameter_count for model in models.values()}
    ) == 1
    assert len(
        {model.inference_parameter_count for model in models.values()}
    ) == 1

    model = models["jepa_error_certificate"]
    raw_prediction = raw.rollout(
        selection.histories[:3],
        selection.future_controls[:3],
        selection.future_actions[:3],
        selection.graph,
    ).mean
    unadjusted = model.predict_unadjusted(
        selection.histories[:3],
        selection.future_controls[:3],
        selection.future_actions[:3],
        raw_prediction,
        selection.graph,
    )
    bound = model.predict_bound(
        selection.histories[:3],
        selection.future_controls[:3],
        selection.future_actions[:3],
        raw_prediction,
        selection.graph,
    )
    assert unadjusted.shape == selection.future_states[:3].shape[:2]
    assert np.all(unadjusted >= 0.0)
    assert np.all(bound >= unadjusted)
    assert model.calibration_adjustment >= 0.0

    restored = ErrorCertificateJepa.from_dict(model.to_dict())
    replay = restored.predict_bound(
        selection.histories[:3],
        selection.future_controls[:3],
        selection.future_actions[:3],
        raw_prediction,
        selection.graph,
    )
    np.testing.assert_allclose(bound, replay, atol=1e-7)

    with pytest.raises(TypeError):
        model.predict_bound(  # type: ignore[call-arg]
            selection.histories[:3],
            selection.future_controls[:3],
            selection.future_actions[:3],
            raw_prediction,
            selection.graph,
            future_states=selection.future_states[:3],
        )

    corrupted = copy.deepcopy(model.to_dict())
    corrupted["state_dict"]["online_encoder.input.weight"]["values"][0][0] = (
        float("nan")
    )
    with pytest.raises(ValueError, match="non-finite"):
        ErrorCertificateJepa.from_dict(corrupted)


def test_certified_raw_dynamics_never_changes_raw_distribution() -> None:
    fit = tiny_action_conditioned_windows(
        pair_count=4, transition_count=6
    )
    selection = tiny_action_conditioned_windows(
        pair_count=2,
        transition_count=6,
        pair_prefix="selection",
    )
    raw = ContractiveLowRankDynamics(LowRankConfig(rank=8)).fit(fit)
    certificate = ErrorCertificateJepa(
        ErrorCertificateJepaConfig(
            width=8,
            hidden_width=16,
            pretrain_steps=2,
            checkpoint_interval=1,
            batch_size=8,
            expected_pair_count=4,
        )
    ).fit(fit, raw).select(selection, raw).calibrate(selection, raw)
    certified = CertifiedRawDynamics(raw, certificate)
    expected = raw.rollout(
        selection.histories[:3],
        selection.future_controls[:3],
        selection.future_actions[:3],
        selection.graph,
    )
    actual = certified.forecast_with_certificate(
        selection.histories[:3],
        selection.future_controls[:3],
        selection.future_actions[:3],
        selection.graph,
    )
    np.testing.assert_array_equal(expected.mean, actual.distribution.mean)
    np.testing.assert_array_equal(
        expected.variance, actual.distribution.variance
    )
    assert np.all(actual.error_bound >= 0.0)

    restored = CertifiedRawDynamics.from_dict(certified.to_dict())
    replay = restored.forecast_with_certificate(
        selection.histories[:3],
        selection.future_controls[:3],
        selection.future_actions[:3],
        selection.graph,
    )
    np.testing.assert_array_equal(
        actual.distribution.mean, replay.distribution.mean
    )
    np.testing.assert_allclose(
        actual.error_bound, replay.error_bound, atol=1e-7
    )


def test_error_certificate_smoke_reassesses_from_stored_evidence(
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
    assert (
        assessment["safety_gates"][
            "calibration_recomputes_exactly"
        ]
        is True
    )
    assert (
        assessment["decision"]
        == "non_interpretable_error_certificate_jepa_smoke"
    )
    verify_stored_assessment(output)
    verify_artifact_manifest(output)
