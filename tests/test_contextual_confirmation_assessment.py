import copy
import hashlib
import json
import uuid
from pathlib import Path

import pytest

from quantis_core.contextual_confirmation import (
    assess_confirmation_results,
    plan_parallel_confirmation_collection,
    validate_confirmation_collection_attestation,
)


def _protocol():
    return json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "lab"
            / "fault_matrix"
            / "contextual-jepa-confirmation-v2.json"
        ).read_text()
    )


def _score(rate):
    return {
        "window_count": 1000,
        "alerts": round(1000 * rate),
        "alert_rate": rate,
    }


def _result(seed, *, contextual=0.01, comparator=0.03):
    families = []
    transfer_representations = {}
    family_errors = {
        f"f{family:02d}": 0.45
        for family in range(13, 25)
    }
    for family in range(13, 25):
        families.append(
            {
                "schedule_sha256": f"{family:064x}",
                "case_ids": [
                    f"confirmation-f{family:02d}-w1-173"
                ],
                "contextual_multimodal": _score(contextual),
                "metrics_only": _score(comparator),
                "capacity_matched_metrics_only": _score(
                    comparator
                ),
                "shuffled_logs": _score(comparator + 0.01),
            }
        )
    for name, mean_error in (
        ("contextual_multimodal", 0.45),
        ("metrics_only", 0.70),
        ("capacity_matched_metrics_only", 0.65),
        ("shuffled_logs", 0.75),
        ("raw_context_ridge", 0.40),
        ("pca_12_context_ridge", 0.60),
    ):
        transfer_representations[name] = {
            "context_dimension": 12,
            "completed_target_count": 8,
            "mean_validation_normalized_mse": mean_error,
            "mean_validation_r_squared": 0.3,
            "targets": {
                "metric.request_latency_ms": {
                    "status": "completed",
                    "training_variance": 1.0,
                    "validation_normalized_mse": mean_error,
                    "validation_r_squared": 0.3,
                    "family_normalized_mse": {
                        family: (
                            value
                            + mean_error
                            - 0.45
                        )
                        for family, value in family_errors.items()
                    },
                }
            },
        }
    return {
        "config": {"seed": seed},
        "metrics": {
            "contextual_multimodal": {
                "validation": _score(contextual)
            },
            "metrics_only": {
                "validation": _score(comparator)
            },
            "capacity_matched_metrics_only": {
                "validation": _score(comparator)
            },
            "shuffled_logs": {
                "validation": _score(comparator + 0.01)
            },
        },
        "schedule_transfer": {
            "validation_families": families
        },
        "model": {
            "diagnostics": {
                "metric_effective_rank": 1.8,
                "log_effective_rank": 1.0,
            }
        },
        "representation_transfer": {
            "representations": transfer_representations
        },
    }


def test_confirmation_supports_claim_and_advances_world_model():
    protocol = _protocol()
    results = [
        _result(seed) for seed in protocol["training_seeds"]
    ]

    assessment = assess_confirmation_results(
        results,
        protocol,
        determinism_verified=True,
    )

    assert assessment["publication_ready"] is True
    assert assessment["claim_supported"] is True
    assert assessment["status"] == "supported"
    assert all(
        gate["passed"] for gate in assessment["gates"].values()
    )
    assert assessment["next_step"]["decision"] == (
        "action_conditioned_intervention_corpus"
    )


def test_confirmation_preserves_publishable_negative_result():
    protocol = _protocol()
    results = [
        _result(
            seed,
            contextual=0.03,
            comparator=0.02,
        )
        for seed in protocol["training_seeds"]
    ]

    assessment = assess_confirmation_results(
        results,
        protocol,
        determinism_verified=True,
    )

    assert assessment["publication_ready"] is True
    assert assessment["claim_supported"] is False
    assert assessment["status"] == "not_supported"
    assert assessment["next_step"]["decision"] == (
        "repair_log_alignment_before_dynamics"
    )


def test_confirmation_rejects_missing_preregistered_seed():
    protocol = _protocol()
    results = [
        _result(seed) for seed in protocol["training_seeds"][:-1]
    ]

    with pytest.raises(ValueError, match="training seed coverage"):
        assess_confirmation_results(
            results,
            protocol,
            determinism_verified=True,
        )


def test_confirmation_requires_deterministic_primary_repeat():
    protocol = _protocol()
    results = [
        _result(seed) for seed in protocol["training_seeds"]
    ]

    assessment = assess_confirmation_results(
        results,
        protocol,
        determinism_verified=False,
    )

    assert assessment["publication_ready"] is False
    assert assessment["claim_supported"] is False
    assert assessment["gates"]["deterministic_primary_repeat"][
        "passed"
    ] is False


def test_confirmation_detects_probe_control_failure():
    protocol = _protocol()
    results = [
        _result(seed) for seed in protocol["training_seeds"]
    ]
    failed = copy.deepcopy(results)
    for result in failed:
        result["representation_transfer"]["representations"][
            "shuffled_logs"
        ] = copy.deepcopy(
            result["representation_transfer"]["representations"][
                "contextual_multimodal"
            ]
        )

    assessment = assess_confirmation_results(
        failed,
        protocol,
        determinism_verified=True,
    )

    assert assessment["claim_supported"] is False
    assert assessment["gates"][
        "frozen_state_better_than_shuffled_logs"
    ]["passed"] is False


def test_confirmation_requires_predictive_nontrivial_compression():
    protocol = _protocol()
    results = [
        _result(seed) for seed in protocol["training_seeds"]
    ]
    for result in results:
        representations = result["representation_transfer"][
            "representations"
        ]
        contextual = representations["contextual_multimodal"]
        contextual["mean_validation_normalized_mse"] = 1.2
        for target in contextual["targets"].values():
            target["validation_normalized_mse"] = 1.2
            target["family_normalized_mse"] = {
                family: 1.2
                for family in target["family_normalized_mse"]
            }
        representations["raw_context_ridge"][
            "mean_validation_normalized_mse"
        ] = 1.1
        representations["pca_12_context_ridge"][
            "mean_validation_normalized_mse"
        ] = 1.3

    assessment = assess_confirmation_results(
        results,
        protocol,
        determinism_verified=True,
    )

    assert assessment["claim_supported"] is False
    assert assessment["gates"][
        "compressed_state_predicts_better_than_mean"
    ]["passed"] is False
    assert assessment["next_step"]["decision"] == (
        "improve_state_observability_before_dynamics"
    )


def test_confirmation_requires_advantage_over_pca_compression():
    protocol = _protocol()
    results = [
        _result(seed) for seed in protocol["training_seeds"]
    ]
    failed = copy.deepcopy(results)
    for result in failed:
        representations = result["representation_transfer"][
            "representations"
        ]
        representations["pca_12_context_ridge"] = copy.deepcopy(
            representations["contextual_multimodal"]
        )

    assessment = assess_confirmation_results(
        failed,
        protocol,
        determinism_verified=True,
    )

    assert assessment["claim_supported"] is False
    assert assessment["gates"][
        "frozen_state_better_than_pca_context"
    ]["passed"] is False


def test_confirmation_validates_parallel_collection_attestation():
    protocol = _protocol()
    plans = plan_parallel_confirmation_collection(protocol)
    cases = []
    for plan in plans:
        started = plan.batch * 10_000 + plan.lane * 10
        cases.append(
            {
                "case_id": plan.case_id,
                "family": plan.family,
                "worker_replicas": plan.worker_replicas,
                "split": plan.split,
                "batch": plan.batch,
                "lane": plan.lane,
                "compose_project": f"lane-{plan.lane}",
                "started_unix_nano": started,
                "completed_unix_nano": started + 100,
            }
        )
    protocol_sha256 = hashlib.sha256(
        json.dumps(
            protocol,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    attestation = {
        "schema_version": 1,
        "kind": (
            "contextual_multimodal_jepa_confirmation_"
            "collection_attestation"
        ),
        "execution_id": str(uuid.uuid4()),
        "started_unix_nano": 1,
        "completed_unix_nano": 250_000,
        "parallel_jobs": 3,
        "batch_count": 24,
        "case_count": 72,
        "application_build_context_sha256": protocol["corpus"][
            "application_build_context_sha256"
        ],
        "application_image_id": "sha256:confirmation-test",
        "protocol_sha256": protocol_sha256,
        "cases": cases,
    }

    validate_confirmation_collection_attestation(
        attestation,
        protocol,
    )
    invalid = copy.deepcopy(attestation)
    invalid["cases"][0]["lane"] = 2
    with pytest.raises(ValueError, match="lane plan"):
        validate_confirmation_collection_attestation(
            invalid,
            protocol,
        )
