import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

from quantis_core.otlp_log_windowing import OtlpLogFeatureSpec
from quantis_core.otlp_windowing import OtlpFeatureSpec


FROZEN_PROMOTION_PROTOCOL_SHA256 = (
    "f17f46e56629e4cd6c861fb3eb39a1d1"
    "56d20baa73568161cd1b7352485f209b"
)


def test_contextual_promotion_protocol_freezes_hypothesis_and_gates():
    repository = Path(__file__).resolve().parents[1]
    lab = repository / "lab" / "fault_matrix"
    protocol = json.loads(
        (lab / "contextual-jepa-promotion-v1.json").read_text()
    )
    feature_spec = OtlpLogFeatureSpec.from_dict(
        json.loads(
            (
                lab
                / "contextual-promotion-log-feature-spec.json"
            ).read_text()
        )
    )
    metric_feature_spec = OtlpFeatureSpec.from_dict(
        json.loads((lab / "feature-spec.json").read_text())
    )

    assert protocol["training_config"] == {
        "metric_latent_dimension": 3,
        "log_latent_dimension": 1,
        "pretraining_epochs": 200,
        "predictor_refinement_epochs": 100,
        "cross_validation_epochs": 0,
        "learning_rate": 0.02,
        "ema_decay": 0.98,
        "weight_decay": 0.0001,
        "loss": "l1",
        "huber_delta": 1.0,
        "auxiliary_loss_weight": 0.2,
        "rollout_loss_weight": 0.2,
        "calibration_quantile": 0.98,
        "seed": 73,
    }
    assert protocol["gates"] == {
        "maximum_validation_alert_rate": 0.03,
        "maximum_schedule_family_alert_rate": 0.05,
        "minimum_no_worse_schedule_family_fraction": 0.8,
        "minimum_metric_effective_rank": 1.5,
        "minimum_log_effective_rank": 0.5,
    }
    assert protocol["shuffled_log_control"] == {
        "kind": "shuffled_demand_residual_log_alignment",
        "training_seed": 1074,
        "validation_seed": 2074,
        "preserves_log_context_target_blocks": True,
        "breaks_metric_log_alignment": True,
        "keeps_exogenous_controls_metric_aligned": True,
    }
    assert protocol["training_runtime"] == {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
    }
    assert protocol["metric_vocabulary"] == {
        "feature_schema_id": metric_feature_spec.schema_id,
        "feature_spec_sha256": metric_feature_spec.schema_id,
        "raw_feature_names": [
            feature.name for feature in metric_feature_spec.features
        ],
        "semantic_feature_names": [
            "request_latency_ms",
            "error_rate",
            "queue_depth",
            "worker_completion_ratio",
            "worker_heartbeat_age_s",
            "db_write_completion_ratio",
        ],
    }
    assert tuple(
        feature.name for feature in feature_spec.features
    ) == (
        "checkout_accepted_count",
        "checkout_rejected_count",
        "checkout_completed_count",
        "error_event_count",
        "queue_backlog_low_transition_count",
        "queue_backlog_elevated_transition_count",
        "queue_backlog_high_transition_count",
        "database_latency_fast_count",
        "database_latency_normal_count",
        "database_latency_slow_count",
        "worker_busy_transition_count",
        "worker_idle_transition_count",
    )
    assert len(protocol["training_case_ids"]) == 24
    assert len(protocol["validation_case_ids"]) == 6
    assert set(protocol["training_case_ids"]).isdisjoint(
        protocol["validation_case_ids"]
    )


def test_contextual_promotion_preparer_matches_protocol(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    protocol = json.loads(
        (
            repository
            / "lab"
            / "fault_matrix"
            / "contextual-jepa-promotion-v1.json"
        ).read_text()
    )
    completed = subprocess.run(
        [
            sys.executable,
            "lab/fault_matrix/prepare_contextual_promotion_corpus.py",
            "--output",
            str(tmp_path),
        ],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    split = json.loads((tmp_path / "split.json").read_text())

    assert split["training_case_ids"] == protocol["training_case_ids"]
    assert split["validation_case_ids"] == (
        protocol["validation_case_ids"]
    )


def test_contextual_promotion_v1_protocol_remains_frozen_during_v2():
    repository = Path(__file__).resolve().parents[1]
    protocol = json.loads(
        (
            repository
            / "lab"
            / "fault_matrix"
            / "contextual-jepa-promotion-v1.json"
        ).read_text()
    )
    protocol_path = (
        repository
        / "lab"
        / "fault_matrix"
        / "contextual-jepa-promotion-v1.json"
    )
    assert hashlib.sha256(protocol_path.read_bytes()).hexdigest() == (
        FROZEN_PROMOTION_PROTOCOL_SHA256
    )
    current_build_context = subprocess.run(
        [
            sys.executable,
            "lab/fault_matrix/hash_build_context.py",
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert current_build_context != protocol["corpus"][
        "application_build_context_sha256"
    ]


def test_contextual_promotion_frozen_hash_manifest_is_well_formed():
    repository = Path(__file__).resolve().parents[1]
    protocol = json.loads(
        (
            repository
            / "lab"
            / "fault_matrix"
            / "contextual-jepa-promotion-v1.json"
        ).read_text()
    )

    assert protocol["frozen_files"]
    # Later versioned modules must not be retroactively added to this
    # historical, commit-bound v1 preregistration.
    expected_source_dependencies = {
        str(path.relative_to(repository))
        for path in (
            repository / "src" / "quantis_core"
        ).glob("*.py")
        if path.name
        not in {
            "action_conditioned_dynamics.py",
            "action_dynamics_corpus.py",
            "action_dynamics_development.py",
            "action_dynamics_lab.py",
            "action_dynamics_real_corpus.py",
            "action_dynamics_synthetic.py",
                "contextual_multimodal_development.py",
                "contextual_confirmation.py",
                "contextual_representation_transfer.py",
                "cross_stack_corpus.py",
                "graph_jepa.py",
            "graph_observability.py",
            "graph_telemetry.py",
            "hybrid_event_features.py",
            "hybrid_frozen_probe.py",
            "hybrid_graph_jepa.py",
            "hybrid_graph_tokens.py",
            "hybrid_jepa_development.py",
            "hybrid_jepa_evaluation.py",
                "learned_graph_jepa.py",
                    "mprm_jepa.py",
                    "observability_graph_corpus.py",
                    "richer_regime_corpus.py",
                    "richer_regime_preflight.py",
                    "richer_regime_retry.py",
                "structured_events.py",
        }
    } | {
        "src/quantis_core/py.typed",
        "pyproject.toml",
    }
    assert expected_source_dependencies <= set(
        protocol["frozen_files"]
    )
    assert all(
        (repository / relative_path).exists()
        for relative_path in protocol["frozen_files"]
    )
    assert all(
        len(digest) == 64
        and set(digest) <= set("0123456789abcdef")
        for digest in protocol["frozen_files"].values()
    )
