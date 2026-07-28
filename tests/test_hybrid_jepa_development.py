from quantis_core.hybrid_jepa_development import (
    assess_hybrid_jepa_development,
)


def test_hybrid_development_advances_only_when_every_gate_passes() -> None:
    assessment = assess_hybrid_jepa_development(
        declared_topology_normalized_mse=0.44,
        shuffled_topology_normalized_mse=0.53,
        local_context_normalized_mse=0.48,
        equal_width_pca_normalized_mse=0.49,
        raw_context_normalized_mse=0.47,
        state_reconstruction_normalized_mse=0.08,
        effective_rank_fraction=0.62,
        trace_link_coverage=0.0,
        maximum_state_reconstruction_normalized_mse=0.1,
        minimum_effective_rank_fraction=0.25,
        minimum_topology_relative_improvement=0.01,
    )

    assert assessment["status"] == "supported"
    assert assessment["decision"] == "collect_intervention_data"
    assert all(
        gate["passed"] for gate in assessment["gates"].values()
    )
    assert assessment["data_quality"]["trace_link_coverage"] == 0.0
    assert "development-only" in assessment["evidence_boundary"]


def test_hybrid_development_rejects_a_collapsed_or_uncompetitive_latent() -> None:
    assessment = assess_hybrid_jepa_development(
        declared_topology_normalized_mse=0.61,
        shuffled_topology_normalized_mse=0.59,
        local_context_normalized_mse=0.57,
        equal_width_pca_normalized_mse=0.49,
        raw_context_normalized_mse=0.50,
        state_reconstruction_normalized_mse=0.18,
        effective_rank_fraction=0.12,
        trace_link_coverage=0.0,
        maximum_state_reconstruction_normalized_mse=0.1,
        minimum_effective_rank_fraction=0.25,
        minimum_topology_relative_improvement=0.01,
    )

    assert assessment["status"] == "not_supported"
    assert assessment["decision"] == "stop_or_redesign_nominal_jepa"
    assert not assessment["gates"]["beats_equal_width_pca"]["passed"]
    assert not assessment["gates"]["beats_raw_context"]["passed"]
    assert not assessment["gates"]["declared_topology_beats_shuffled"][
        "passed"
    ]
    assert not assessment["gates"][
        "declared_topology_beats_local_context"
    ]["passed"]
    assert not assessment["gates"]["state_is_recoverable"]["passed"]
    assert not assessment["gates"]["latent_has_effective_rank"]["passed"]
