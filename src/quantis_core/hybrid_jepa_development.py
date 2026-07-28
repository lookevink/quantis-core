"""Development-only gates for the hybrid telemetry JEPA."""

from typing import Any, Dict

import numpy as np


def assess_hybrid_jepa_development(
    *,
    declared_topology_normalized_mse: float,
    shuffled_topology_normalized_mse: float,
    local_context_normalized_mse: float,
    equal_width_pca_normalized_mse: float,
    raw_context_normalized_mse: float,
    state_reconstruction_normalized_mse: float,
    effective_rank_fraction: float,
    trace_link_coverage: float,
    maximum_state_reconstruction_normalized_mse: float,
    minimum_effective_rank_fraction: float,
    minimum_topology_relative_improvement: float,
) -> Dict[str, Any]:
    """Apply frozen advancement gates without making a confirmation claim."""

    values = (
        declared_topology_normalized_mse,
        shuffled_topology_normalized_mse,
        local_context_normalized_mse,
        equal_width_pca_normalized_mse,
        raw_context_normalized_mse,
        state_reconstruction_normalized_mse,
        effective_rank_fraction,
        trace_link_coverage,
        maximum_state_reconstruction_normalized_mse,
        minimum_effective_rank_fraction,
        minimum_topology_relative_improvement,
    )
    if not all(np.isfinite(value) for value in values):
        raise ValueError("hybrid JEPA assessment values must be finite")
    if (
        min(values[:6]) < 0.0
        or not 0.0 <= effective_rank_fraction <= 1.0
        or not 0.0 <= trace_link_coverage <= 1.0
        or maximum_state_reconstruction_normalized_mse < 0.0
        or not 0.0 <= minimum_effective_rank_fraction <= 1.0
        or not 0.0
        <= minimum_topology_relative_improvement
        < 1.0
    ):
        raise ValueError("hybrid JEPA assessment values are invalid")

    gates = {
        "beats_equal_width_pca": _gate(
            declared_topology_normalized_mse
            < equal_width_pca_normalized_mse,
            declared_topology_normalized_mse,
            equal_width_pca_normalized_mse,
        ),
        "beats_raw_context": _gate(
            declared_topology_normalized_mse
            < raw_context_normalized_mse,
            declared_topology_normalized_mse,
            raw_context_normalized_mse,
        ),
        "declared_topology_beats_shuffled": _gate(
            (
                shuffled_topology_normalized_mse
                - declared_topology_normalized_mse
            )
            / max(shuffled_topology_normalized_mse, 1e-12)
            >= minimum_topology_relative_improvement,
            (
                shuffled_topology_normalized_mse
                - declared_topology_normalized_mse
            )
            / max(shuffled_topology_normalized_mse, 1e-12),
            minimum_topology_relative_improvement,
        ),
        "declared_topology_beats_local_context": _gate(
            declared_topology_normalized_mse
            < local_context_normalized_mse,
            declared_topology_normalized_mse,
            local_context_normalized_mse,
        ),
        "state_is_recoverable": _gate(
            state_reconstruction_normalized_mse
            <= maximum_state_reconstruction_normalized_mse,
            state_reconstruction_normalized_mse,
            maximum_state_reconstruction_normalized_mse,
        ),
        "latent_has_effective_rank": _gate(
            effective_rank_fraction
            >= minimum_effective_rank_fraction,
            effective_rank_fraction,
            minimum_effective_rank_fraction,
        ),
    }
    supported = all(gate["passed"] for gate in gates.values())
    return {
        "schema_version": 1,
        "kind": "hybrid_telemetry_jepa_development_assessment_v1",
        "status": "supported" if supported else "not_supported",
        "decision": (
            "collect_intervention_data"
            if supported
            else "stop_or_redesign_nominal_jepa"
        ),
        "gates": gates,
        "scores": {
            "declared_topology_normalized_mse": (
                declared_topology_normalized_mse
            ),
            "shuffled_topology_normalized_mse": (
                shuffled_topology_normalized_mse
            ),
            "equal_width_pca_normalized_mse": (
                equal_width_pca_normalized_mse
            ),
            "local_context_normalized_mse": (
                local_context_normalized_mse
            ),
            "raw_context_normalized_mse": (
                raw_context_normalized_mse
            ),
            "state_reconstruction_normalized_mse": (
                state_reconstruction_normalized_mse
            ),
            "effective_rank_fraction": effective_rank_fraction,
        },
        "data_quality": {
            "trace_link_coverage": trace_link_coverage,
            "trace_supervision_available": trace_link_coverage > 0.0,
        },
        "evidence_boundary": (
            "development-only reuse of an opened nominal corpus; "
            "not confirmation, causal, fault-localization, or "
            "world-model evidence"
        ),
    }


def _gate(
    passed: bool,
    observed: float,
    threshold_or_control: float,
) -> Dict[str, Any]:
    return {
        "passed": bool(passed),
        "observed": float(observed),
        "threshold_or_control": float(threshold_or_control),
    }
