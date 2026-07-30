import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from quantis_core.action_conditioned_dynamics import (
    ActionConditionedCaseManifest,
    InterventionAction,
)
from quantis_core.action_dynamics_lab import (
    ActionCollectionProtocol,
    LabActionCaptureManifest,
    assess_action_pair_metric_series,
)
from quantis_core.richer_regime_retry import WORKLOAD_FAMILIES


REPOSITORY = Path(__file__).resolve().parents[1]
LAB = REPOSITORY / "lab" / "action_dynamics"
sys.path.insert(0, str(LAB))
SPEC = importlib.util.spec_from_file_location(
    "audit_richer_regime_validity",
    LAB / "audit_richer_regime_validity.py",
)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_pair_audit_uses_protocol_effect_and_recovery_ratios() -> None:
    protocol = ActionCollectionProtocol.from_dict(
        json.loads(
            (LAB / "development-protocol-v1.json").read_text()
        )
    )
    treatment, control = _dequeue_manifests()
    control_values = [0.0] * 108
    valid_values = [0.0] * 108
    valid_values[29:39] = [20.0] * 10
    nonrecovering_values = list(valid_values)
    nonrecovering_values[-8:] = [20.0] * 8
    valid = assess_action_pair_metric_series(
        protocol,
        (treatment, control),
        {
            treatment.action_case.case_id: {
                "redis_dequeue_latency_ms": valid_values
            },
            control.action_case.case_id: {
                "redis_dequeue_latency_ms": control_values
            },
        },
    )
    ineffective = assess_action_pair_metric_series(
        protocol,
        (treatment, control),
        {
            treatment.action_case.case_id: {
                "redis_dequeue_latency_ms": control_values
            },
            control.action_case.case_id: {
                "redis_dequeue_latency_ms": control_values
            },
        },
    )
    nonrecovering = assess_action_pair_metric_series(
        protocol,
        (treatment, control),
        {
            treatment.action_case.case_id: {
                "redis_dequeue_latency_ms": nonrecovering_values
            },
            control.action_case.case_id: {
                "redis_dequeue_latency_ms": control_values
            },
        },
    )
    summary = audit.summarize_action_pair_assessments(
        (
            {**valid[0], "pair_id": "valid"},
            {**ineffective[0], "pair_id": "ineffective"},
            {**nonrecovering[0], "pair_id": "nonrecovering"},
        )
    )

    assert valid[0]["raw_effect_passed"] is True
    assert valid[0]["recovery_passed"] is True
    assert ineffective[0]["raw_effect_passed"] is False
    assert nonrecovering[0]["recovery_ratio"] == 1.0
    assert summary["failed_pair_count"] == 2
    assert summary["failure_counts_by_reason"] == {
        "raw_effect_passed": 1,
        "recovery_passed": 1,
    }
    extra_control = replace(
        control,
        action_case=replace(
            control.action_case, case_id="extra-control"
        ),
    )
    with pytest.raises(ValueError, match="identity"):
        assess_action_pair_metric_series(
            protocol,
            (treatment, control, extra_control),
            {
                treatment.action_case.case_id: {
                    "redis_dequeue_latency_ms": valid_values
                },
                control.action_case.case_id: {
                    "redis_dequeue_latency_ms": control_values
                },
                extra_control.action_case.case_id: {
                    "redis_dequeue_latency_ms": control_values
                },
            },
        )


def test_amendment_binding_and_source_hashes_are_auditable(
    tmp_path: Path,
) -> None:
    fit = tmp_path / "fit-campaign"
    selection = tmp_path / "selection-campaign"
    amendment_path = tmp_path / "amendment.json"
    action_protocol_path = tmp_path / "action-protocol.json"
    _write(action_protocol_path, {"protocol": "action"})
    _write(fit / "campaign" / "protocol.json", {"protocol": 1})
    _write(fit / "campaign" / "plan.json", {"plan": 1})
    _write(
        fit / "selection" / "steady" / "inputs" / "plan.json",
        {"failed": 1},
    )
    _write(fit / "fit" / "source.json", {"fit": 1})
    for family in WORKLOAD_FAMILIES:
        _write(
            selection
            / "selection"
            / family
            / "inputs"
            / "manifests"
            / "case.json",
            {"case": family},
        )
    _write(selection / "campaign" / "protocol.json", {"protocol": 1})
    amendment = {
        "parent_campaign_protocol_file_sha256": _sha256(
            fit / "campaign" / "protocol.json"
        ),
        "parent_campaign_plan_file_sha256": _sha256(
            fit / "campaign" / "plan.json"
        ),
        "failed_execution_plan_file_sha256": _sha256(
            fit / "selection" / "steady" / "inputs" / "plan.json"
        ),
    }
    _write(amendment_path, amendment)

    unbound = audit.audit_collection_amendment(
        fit_campaign=fit,
        selection_campaign=selection,
        amendment_path=amendment_path,
    )
    before = audit.build_consumed_source_manifest(
        fit_campaign=fit,
        selection_campaign=selection,
        amendment_path=amendment_path,
        action_protocol_path=action_protocol_path,
    )

    assert unbound["referenced_hashes_valid"] is True
    assert unbound["selection_manifests_bind_amendment"] is False
    amendment_sha256 = _sha256(amendment_path)
    for family in WORKLOAD_FAMILIES:
        _write(
            selection
            / "selection"
            / family
            / "inputs"
            / "manifests"
            / "case.json",
            {
                "case": family,
                "collection_amendment_sha256": amendment_sha256,
            },
        )
    bound = audit.audit_collection_amendment(
        fit_campaign=fit,
        selection_campaign=selection,
        amendment_path=amendment_path,
    )
    after = audit.build_consumed_source_manifest(
        fit_campaign=fit,
        selection_campaign=selection,
        amendment_path=amendment_path,
        action_protocol_path=action_protocol_path,
    )

    assert bound["selection_manifests_bind_amendment"] is True
    assert before["sha256"] != after["sha256"]


def test_validity_audit_verifier_rejects_tampering(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "audit"
    _write(
        bundle / "validity-audit.json",
        {
            "kind": "richer_regime_retained_validity_audit",
            "scientific_status": "inconclusive_methodology_failure",
            "stored_model_decision_admissible": False,
        },
    )
    _write(bundle / "source-content-manifest.json", {"sha256": {}})
    (bundle / "report.md").write_text("inconclusive\n")
    members = {
        path.relative_to(bundle).as_posix(): _sha256(path)
        for path in bundle.iterdir()
    }
    _write(
        bundle / "artifact-manifest.json",
        {
            "kind": "richer_regime_validity_audit_artifact_manifest",
            "sha256": members,
        },
    )
    manifest_sha256 = _sha256(bundle / "artifact-manifest.json")

    verified = audit.verify_validity_audit(
        bundle,
        expected_manifest_sha256=manifest_sha256,
    )

    assert verified["stored_model_decision_admissible"] is False
    (bundle / "report.md").write_text("tampered\n")
    with pytest.raises(ValueError, match="member hashes"):
        audit.verify_validity_audit(
            bundle,
            expected_manifest_sha256=manifest_sha256,
        )


def test_role_completeness_rejects_a_missing_pair() -> None:
    complete = {
        family: {f"{family}-{index}" for index in range(2)}
        for family in WORKLOAD_FAMILIES
    }

    qualified = audit.assess_role_pair_completeness(
        complete, expected_pairs_per_family=2
    )
    complete["steady"].pop()
    incomplete = audit.assess_role_pair_completeness(
        complete, expected_pairs_per_family=2
    )

    assert qualified["corpus_complete"] is True
    assert incomplete["corpus_complete"] is False


def _dequeue_manifests() -> tuple[
    LabActionCaptureManifest, LabActionCaptureManifest
]:
    action = InterventionAction(
        action_id="dequeue-action",
        action_kind="redis_dequeue_delay",
        target_entity="queue_dequeues_to_worker",
        start_index=28,
        stop_index=38,
        magnitude=20.0,
        magnitude_unit="milliseconds",
        effect_feature="redis_dequeue_latency_ms",
        effect_direction="increase",
        minimum_effect=10.0,
        recovery_tolerance=0.3,
    )
    shared = {
        "split": "training",
        "point_count": 108,
        "logical_window_period_nano": 250_000_000,
        "topology_id": "workers-1",
        "worker_replicas": 1,
        "workload_seed": 1,
        "intervention_seed": 2,
    }
    treatment_case = ActionConditionedCaseManifest(
        case_id="treatment",
        matched_pair_id="pair",
        actions=(action,),
        **shared,
    )
    control_case = ActionConditionedCaseManifest(
        case_id="control",
        matched_pair_id="pair",
        actions=(),
        **shared,
    )

    def manifest(
        case: ActionConditionedCaseManifest,
    ) -> LabActionCaptureManifest:
        return LabActionCaptureManifest(
            action_case=case,
            sample_period_seconds=0.25,
            request_schedule=(8,) * 108,
            api_request_queue_size=128,
            image_digests={
                "application": "sha256:" + "a" * 64
            },
            observation_schema_sha256="b" * 64,
            protocol_sha256="c" * 64,
            prepared_plan_sha256="d" * 64,
            graph_observation_schema_sha256="b" * 64,
            corpus_role="development",
        )

    return (
        manifest(treatment_case),
        manifest(control_case),
    )


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
