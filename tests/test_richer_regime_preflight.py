import numpy as np

from quantis_core.richer_regime_preflight import (
    RicherRegimeFitEvidence,
    assess_richer_regime_fit_preflight,
)


def test_preflight_routes_techniques_from_observed_mechanisms() -> None:
    generator = np.random.default_rng(7)
    row_count = 600
    metric_context = generator.normal(size=(row_count, 4))
    event_context = generator.normal(size=(row_count, 2))
    families = tuple(
        ("steady", "ramp_or_burst", "periodic_or_multiphase")[
            index % 3
        ]
        for index in range(row_count)
    )
    family_effect = np.asarray(
        [
            {
                "steady": 0.0,
                "ramp_or_burst": 1.5,
                "periodic_or_multiphase": -1.5,
            }[family]
            for family in families
        ]
    )
    noise_scale = np.asarray(
        [
            {
                "steady": 0.1,
                "ramp_or_burst": 0.8,
                "periodic_or_multiphase": 0.3,
            }[family]
            for family in families
        ]
    )
    target = (
        metric_context @ generator.normal(size=(4, 3))
        + event_context @ generator.normal(size=(2, 3))
        + family_effect[:, None]
        + generator.normal(size=(row_count, 3))
        * noise_scale[:, None]
    )
    evidence = RicherRegimeFitEvidence(
        metric_context=metric_context,
        event_context=event_context,
        targets=target,
        demand=generator.uniform(4.0, 18.0, size=row_count),
        topology=generator.integers(1, 4, size=row_count).astype(float),
        workload_families=families,
        replicates=np.asarray(
            [0] * (row_count // 2) + [1] * (row_count // 2)
        ),
        regime_classification_accuracy=1.0,
    )

    assessment = assess_richer_regime_fit_preflight(evidence)

    assert assessment["status"] == "qualified"
    assert assessment["recommendations"][
        "contextual_multimodal_jepa"
    ] == "retry"
    assert assessment["recommendations"][
        "error_certificate_jepa"
    ] == "retry"
    assert assessment["measurements"]["contextual_mse_ratio"] < 1.0
    assert assessment["measurements"][
        "heteroscedastic_variance_ratio"
    ] > 1.5


def test_preflight_rejects_role_overlap() -> None:
    evidence = RicherRegimeFitEvidence(
        metric_context=np.ones((4, 1)),
        event_context=np.ones((4, 1)),
        targets=np.ones((4, 1)),
        demand=np.ones(4),
        topology=np.ones(4),
        workload_families=("steady",) * 4,
        replicates=np.asarray([0, 0, 0, 0]),
        regime_classification_accuracy=1.0,
    )

    try:
        assess_richer_regime_fit_preflight(evidence)
    except ValueError as error:
        assert "replicates 0 and 1" in str(error)
    else:
        raise AssertionError("missing probe replicate was accepted")
