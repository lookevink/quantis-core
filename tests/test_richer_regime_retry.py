import copy
import json
from pathlib import Path

import pytest

from quantis_core.richer_regime_retry import (
    RicherRegimeRetryProtocol,
    assess_richer_regime_plan,
    build_richer_regime_plan,
    materialize_workload_schedule,
    prepare_richer_regime_shard,
    validate_richer_regime_plan,
    zero_event_upper_bound,
)


def _protocol() -> RicherRegimeRetryProtocol:
    repository = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (
            repository
            / "lab"
            / "action_dynamics"
            / "richer-regime-retry-protocol-v1.json"
        ).read_text()
    )
    return RicherRegimeRetryProtocol.from_dict(payload)


def test_retry_plan_freezes_roles_and_family_replication() -> None:
    protocol = _protocol()

    plan = build_richer_regime_plan(protocol)
    assessment = assess_richer_regime_plan(protocol, plan)

    assert plan == build_richer_regime_plan(protocol)
    assert len(plan["pairs"]) == 495
    assert len({pair["pair_id"] for pair in plan["pairs"]}) == 495
    assert assessment["status"] == "qualified"
    assert assessment["pair_counts_by_role"] == {
        "fit": 90,
        "selection": 45,
        "calibration": 180,
        "evaluation": 180,
    }
    assert assessment["evaluation_controls_by_workload_family"] == {
        "periodic_or_multiphase": 60,
        "ramp_or_burst": 60,
        "steady": 60,
    }
    assert assessment["gates"] == {
        "complete_factorial": True,
        "deterministic_role_ownership": True,
        "evaluation_family_false_positive_resolution": True,
        "calibration_family_false_positive_resolution": True,
        "pair_identity_unique": True,
        "workload_family_coverage": True,
    }


def test_retry_plan_rejects_post_hoc_role_drift() -> None:
    protocol = _protocol()
    plan = copy.deepcopy(build_richer_regime_plan(protocol))
    plan["pairs"][0]["corpus_role"] = "evaluation"

    with pytest.raises(ValueError, match="deterministic generator"):
        validate_richer_regime_plan(protocol, plan)


def test_regime_schedules_are_distinct_but_reproducible() -> None:
    protocol = _protocol()
    schedules = {
        family: materialize_workload_schedule(
            protocol,
            workload_family=family,
            workload_seed=12345,
            action_kind="worker_pause",
        )
        for family in protocol.workload_families
    }

    assert all(len(schedule) == 108 for schedule in schedules.values())
    assert all(min(schedule) >= 1 for schedule in schedules.values())
    assert len({schedule for schedule in schedules.values()}) == 3
    assert schedules["steady"] == materialize_workload_schedule(
        protocol,
        workload_family="steady",
        workload_seed=12345,
        action_kind="worker_pause",
    )
    ramp = schedules["ramp_or_burst"]
    assert max(ramp[36:48]) > max(ramp[:12])
    periodic = schedules["periodic_or_multiphase"]
    assert sum(periodic[12:24]) > sum(periodic[:12])


def test_action_specific_schedule_constraints_survive_regimes() -> None:
    protocol = _protocol()

    rejection = materialize_workload_schedule(
        protocol,
        workload_family="ramp_or_burst",
        workload_seed=9,
        action_kind="api_rejection",
    )
    enqueue = materialize_workload_schedule(
        protocol,
        workload_family="periodic_or_multiphase",
        workload_seed=9,
        action_kind="redis_enqueue_delay",
    )

    assert rejection == (12,) * 108
    assert enqueue[84:92] == (0,) * 8
    assert enqueue[92:107] == (8,) * 15


def test_false_positive_resolution_uses_one_sided_exact_bound() -> None:
    assert zero_event_upper_bound(58, confidence=0.95) > 0.05
    assert zero_event_upper_bound(59, confidence=0.95) < 0.05


def test_prepared_shard_is_pair_atomic_and_collector_compatible() -> None:
    protocol = _protocol()
    plan = build_richer_regime_plan(protocol)

    prepared = prepare_richer_regime_shard(
        protocol,
        plan,
        corpus_role="evaluation",
        workload_family="steady",
        action_library=_action_library(),
        image_digests={"application": "sha256:" + "a" * 64},
        observation_schema_sha256="b" * 64,
        application_build_context_sha256="c" * 64,
    )

    assert prepared["summary"]["pair_count"] == 60
    assert prepared["summary"]["capture_count"] == 120
    assert prepared["protocol"]["collection"] == {
        "parallel_jobs": 6,
        "pair_count": 60,
        "expected_capture_count": 120,
        "overwrite": False,
    }
    assert len(prepared["manifests"]) == 120
    assert len(prepared["plan"]["assignments"]) == 120
    pair_manifests: dict[str, list[dict[str, object]]] = {}
    for manifest in prepared["manifests"].values():
        pair_manifests.setdefault(
            str(manifest["action_case"]["matched_pair_id"]), []
        ).append(manifest)
        assert manifest["retry_corpus_role"] == "evaluation"
        assert manifest["workload_family"] == "steady"
        assert manifest["action_case"]["split"] == "confirmation"
    assert all(
        len(twins) == 2
        and twins[0]["request_schedule"] == twins[1]["request_schedule"]
        and sum(bool(twin["action_case"]["actions"]) for twin in twins)
        == 1
        for twins in pair_manifests.values()
    )


def _action_library() -> dict[str, object]:
    repository = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (
            repository
            / "lab"
            / "action_dynamics"
            / "development-protocol-v1.json"
        ).read_text()
    )
    return payload["action_library"]
