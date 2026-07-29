import hashlib
import json

import numpy as np
import pytest

from quantis_core.action_conditioned_dynamics import (
    ActionConditionedWindows,
)
from quantis_core.edge_dynamics.complete_lejepa import (
    CompleteLejepaConfig,
    CompleteLejepaRepresentation,
)
from quantis_core.edge_dynamics.jepa_score import (
    ExactJepaScorer,
    assess_jepa_score_gates,
    jepa_score_from_jacobian,
)
from quantis_core.graph_telemetry import (
    DeclaredTelemetryGraph,
    GraphEntity,
    TelemetryBinding,
)


def test_jepa_score_matches_literal_clipped_log_singular_values() -> None:
    import torch

    jacobian = torch.zeros((3, 2, 4), dtype=torch.float32)
    jacobian[0, 0, 0] = 4.0
    jacobian[1, 0, 1] = 2.0
    jacobian[2, 0, 2] = 0.5
    jacobian[0, 1, 0] = 3.0
    jacobian[1, 1, 1] = 1.0

    result = jepa_score_from_jacobian(jacobian, epsilon=1e-6)
    literal = torch.linalg.svdvals(
        jacobian.permute(1, 0, 2)
    ).clamp_min(1e-6)

    torch.testing.assert_close(result.singular_values, literal)
    torch.testing.assert_close(
        result.jepa_score, literal.log().sum(dim=1)
    )
    torch.testing.assert_close(
        result.anomaly_score, -literal.log().sum(dim=1)
    )
    assert result.clipped_count.tolist() == [0, 1]


def test_exact_scorer_is_batch_separable_restorable_and_rng_isolated() -> None:
    import torch

    windows = _tiny_windows(pair_count=4, transition_count=3)
    fitted = CompleteLejepaRepresentation(
        CompleteLejepaConfig(
            objective="lejepa",
            steps=1,
            expected_pair_count=4,
            width=8,
            head_count=2,
            feedforward_width=16,
            projector_width=16,
            sketch_dimension=7,
            warmup_steps=1,
        )
    ).fit(windows)
    torch.manual_seed(9917)
    before = torch.random.get_rng_state()
    payload = fitted.to_dict()
    raw_payload = json.dumps(payload, indent=2).encode("utf-8")
    scorer = ExactJepaScorer.from_model_json_bytes(raw_payload)
    together = scorer.score(windows.histories[:2], windows.graph)
    after = torch.random.get_rng_state()
    restored = ExactJepaScorer.from_dict(scorer.to_dict())
    after_restore = torch.random.get_rng_state()
    separate = [
        restored.score(windows.histories[index : index + 1], windows.graph)
        for index in range(2)
    ]

    np.testing.assert_array_equal(before.numpy(), after.numpy())
    np.testing.assert_array_equal(after.numpy(), after_restore.numpy())
    np.testing.assert_allclose(
        together.jepa_score,
        np.concatenate([value.jepa_score for value in separate]),
        atol=2e-4,
    )
    np.testing.assert_allclose(
        together.singular_values,
        np.concatenate([value.singular_values for value in separate]),
        atol=2e-5,
    )
    np.testing.assert_allclose(
        together.projector_embeddings,
        restored.projector_embeddings(
            windows.histories[:2], windows.graph
        ),
        atol=1e-7,
    )
    assert scorer.parameter_count == (
        fitted.inference_parameter_count
        + fitted.training_only_parameter_count
    )
    assert scorer.source_model_sha256 == restored.source_model_sha256
    assert scorer.source_model_file_sha256 == hashlib.sha256(
        raw_payload
    ).hexdigest()
    assert scorer.source_model_payload_sha256 != (
        scorer.source_model_file_sha256
    )
    assert "training_metrics" not in scorer.to_dict()[
        "strict_model_payload"
    ]
    assert "decoder_state" not in scorer.to_dict()[
        "strict_model_payload"
    ]
    assert "learning_rate" not in scorer.to_dict()[
        "strict_model_payload"
    ]["inference_config"]
    assert np.all(np.isfinite(together.anomaly_score))
    assert np.all(together.singular_values >= 0.0)
    np.testing.assert_array_equal(
        together.unowned_jacobian_max_abs, np.zeros(2)
    )


def test_exact_scorer_rejects_schema_and_nonfinite_inputs() -> None:
    windows = _tiny_windows(pair_count=4, transition_count=2)
    fitted = CompleteLejepaRepresentation(
        CompleteLejepaConfig(
            steps=1,
            expected_pair_count=4,
            width=8,
            head_count=2,
            feedforward_width=16,
            projector_width=16,
            sketch_dimension=7,
            warmup_steps=1,
        )
    ).fit(windows)
    raw_model = json.dumps(fitted.to_dict()).encode()
    scorer = ExactJepaScorer.from_model_json_bytes(raw_model)
    bad = windows.histories[:1].copy()
    bad[0, 0, 0, 0] = np.nan

    with pytest.raises(ValueError, match="input"):
        scorer.score(bad, windows.graph)
    with pytest.raises(ValueError, match="model"):
        ExactJepaScorer.from_model_json_bytes(
            json.dumps(
                {**fitted.to_dict(), "kind": "wrong"}
            ).encode()
        )

    bundle = scorer.to_dict()
    changed_epsilon = {**bundle, "epsilon": 1.0}
    with pytest.raises(ValueError, match="epsilon"):
        ExactJepaScorer.from_dict(changed_epsilon)
    changed_view = json.loads(json.dumps(bundle))
    changed_view["visible_tokens"][0][0] = 1 - int(
        changed_view["visible_tokens"][0][0]
    )
    with pytest.raises(ValueError, match="view"):
        ExactJepaScorer.from_dict(changed_view)
    changed_state = json.loads(json.dumps(bundle))
    changed_state["strict_model_payload"]["network_state"][
        "feature_weight"
    ][0][0][0] += 1.0
    with pytest.raises(ValueError, match="identity"):
        ExactJepaScorer.from_dict(changed_state)


def test_jepa_score_assessment_recomputes_failed_value_gate() -> None:
    protocols = {
        name: True
        for name in (
            "source_identities_recompute",
            "role_contract_recomputes",
            "fixed_anchors_recompute",
            "action_blind_sampling_recomputes",
            "model_restoration_recomputes",
            "exact_score_recomputes",
            "batch_and_literal_parity_recompute",
            "latency_contract_recomputes",
            "evidence_arrays_are_finite",
            "calibration_isolation_recomputes",
            "alert_metrics_recompute",
            "evaluation_has_no_selection_authority",
            "source_snapshots_and_manifest_verify",
        )
    }
    candidate = {
        "iid_evaluation": {
            "control_trajectory_false_alarm_rate": 0.0,
            "treatment_detection_rate": 0.9,
            "treatment_pre_onset_alert_rate": 0.0,
            "median_post_onset_delay_transitions": 10.0,
        },
        "transfer_evaluation": {
            "control_trajectory_false_alarm_rate": 0.0,
            "treatment_detection_rate": 0.9,
            "treatment_pre_onset_alert_rate": 0.0,
            "median_post_onset_delay_transitions": 10.0,
        },
    }
    raw = {
        "iid_evaluation": {
            "control_trajectory_false_alarm_rate": 0.0,
            "treatment_detection_rate": 0.8,
            "treatment_pre_onset_alert_rate": 0.0,
            "median_post_onset_delay_transitions": 30.0,
        },
        "transfer_evaluation": {
            "control_trajectory_false_alarm_rate": 0.0,
            "treatment_detection_rate": 0.8,
            "treatment_pre_onset_alert_rate": 0.0,
            "median_post_onset_delay_transitions": 30.0,
        },
    }

    passed = assess_jepa_score_gates(
        interpretable=True,
        protocol_checks=protocols,
        candidate_metrics=candidate,
        raw_metrics=raw,
        selection_pair_win_fraction=0.7,
        median_latency_ms=80.0,
        p95_latency_ms=110.0,
        bundle_bytes=4_000_000,
        parameter_count=116_848,
    )
    failed = assess_jepa_score_gates(
        interpretable=True,
        protocol_checks=protocols,
        candidate_metrics={
            **candidate,
            "transfer_evaluation": {
                **candidate["transfer_evaluation"],
                "treatment_detection_rate": 0.7,
            },
        },
        raw_metrics=raw,
        selection_pair_win_fraction=0.7,
        median_latency_ms=80.0,
        p95_latency_ms=110.0,
        bundle_bytes=4_000_000,
        parameter_count=116_848,
    )
    latency_contract_failed = assess_jepa_score_gates(
        interpretable=True,
        protocol_checks={
            **protocols,
            "latency_contract_recomputes": False,
        },
        candidate_metrics=candidate,
        raw_metrics=raw,
        selection_pair_win_fraction=0.7,
        median_latency_ms=80.0,
        p95_latency_ms=110.0,
        bundle_bytes=4_000_000,
        parameter_count=116_848,
    )

    assert passed["passed"]
    assert passed["decision"] == (
        "advance_exact_jepa_score_to_fixed_seed_robustness"
    )
    assert not failed["passed"]
    assert not failed["value_gates"]["transfer_detection_at_least_0_80"]
    assert failed["decision"] == "reject_exact_jepa_score_edge_alert_recipe"
    assert not latency_contract_failed["passed"]
    assert not latency_contract_failed["protocol_passed"]
    smoke = assess_jepa_score_gates(
        interpretable=False,
        protocol_checks=protocols,
        candidate_metrics=candidate,
        raw_metrics=raw,
        selection_pair_win_fraction=0.7,
        median_latency_ms=80.0,
        p95_latency_ms=110.0,
        bundle_bytes=4_000_000,
        parameter_count=116_848,
    )
    assert not smoke["passed"]
    assert smoke["scientific_gates_passed"]
    assert smoke["decision"] == "non_interpretable_jepa_score_smoke"


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
