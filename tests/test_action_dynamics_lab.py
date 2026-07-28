import hashlib
import json
from pathlib import Path
from typing import Optional

import pytest

from quantis_core.action_conditioned_dynamics import ACTION_KINDS
from quantis_core.action_dynamics_lab import (
    ACTION_LAB_FEATURE_NAMES,
    ActionCollectionProtocol,
    LabActionCaptureManifest,
    assess_prepared_action_collection,
    load_prepared_action_collection,
    prepare_action_collection,
    write_action_collection_assessment,
    write_prepared_action_collection,
)

_IMAGES = {
    "application": "sha256:" + "a" * 64,
    "redis": "redis@sha256:" + "b" * 64,
}
_OBSERVATION_SCHEMA = "c" * 64
_BUILD_CONTEXT_SHA256 = "b" * 64
_EVIDENCE_METRIC_NAMES = (
    "quantis.experiment.request_count",
    "quantis.experiment.error_count",
)


def test_lab_manifest_round_trip_is_strict() -> None:
    protocol = _protocol("smoke")
    manifests, _ = prepare_action_collection(
        protocol,
        image_digests=_IMAGES,
        observation_schema_sha256=_OBSERVATION_SCHEMA,
    )
    manifest = manifests[0]

    assert LabActionCaptureManifest.from_dict(
        manifest.to_dict()
    ) == manifest
    assert len(manifest.request_schedule) == 84
    assert all(6 <= value <= 10 for value in manifest.request_schedule)
    assert len(manifest.canonical_sha256()) == 64
    assert manifest.action_case.split == "validation"
    assert manifest.corpus_role == "smoke"
    assert manifest.protocol_sha256 == protocol.canonical_sha256()
    assert manifest.graph_observation_schema_sha256 == (
        manifest.observation_schema_sha256
    )

    invalid = manifest.to_dict()
    invalid["unexpected"] = True
    with pytest.raises(ValueError, match="schema"):
        LabActionCaptureManifest.from_dict(invalid)


def test_pilot_plan_is_complete_deterministic_factorial() -> None:
    protocol = _protocol("instrumentation_pilot")
    first_manifests, first_assignments = prepare_action_collection(
        protocol,
        image_digests=_IMAGES,
        observation_schema_sha256=_OBSERVATION_SCHEMA,
    )
    second_manifests, second_assignments = prepare_action_collection(
        protocol,
        image_digests=_IMAGES,
        observation_schema_sha256=_OBSERVATION_SCHEMA,
    )

    assert first_manifests == second_manifests
    assert first_assignments == second_assignments
    assert len(first_manifests) == 60
    assert len(first_assignments) == 60
    treatment = [
        manifest
        for manifest in first_manifests
        if manifest.action_case.actions
    ]
    assert len(treatment) == 30
    cells = {
        (
            manifest.action_case.actions[0].action_kind,
            manifest.action_case.worker_replicas,
        )
        for manifest in treatment
    }
    assert cells == {
        (kind, workers)
        for kind in ACTION_KINDS
        for workers in (1, 2, 3)
    }
    assert all(
        sum(
            action.action_kind == kind
            and manifest.action_case.worker_replicas == workers
            for manifest in treatment
            for action in manifest.action_case.actions
        )
        == 2
        for kind, workers in cells
    )

    by_pair: dict[str, list[LabActionCaptureManifest]] = {}
    for manifest in first_manifests:
        by_pair.setdefault(
            manifest.action_case.matched_pair_id, []
        ).append(manifest)
    assert len(by_pair) == 30
    for pair in by_pair.values():
        assert len(pair) == 2
        assert pair[0].request_schedule == pair[1].request_schedule
        assert pair[0].action_case.workload_seed == (
            pair[1].action_case.workload_seed
        )
        assert sum(bool(item.action_case.actions) for item in pair) == 1

    assignment_pairs: dict[str, list[object]] = {}
    for assignment in first_assignments:
        assignment_pairs.setdefault(
            assignment.pair_id, []
        ).append(assignment)
    assert {item.batch for item in first_assignments} == set(range(1, 6))
    assert all(
        {item.order_in_pair for item in pair} == {0, 1}
        and len({item.lane for item in pair}) == 1
        and len({item.batch for item in pair}) == 1
        for pair in assignment_pairs.values()
    )


@pytest.mark.parametrize(
    ("name", "expected_pairs"),
    (("smoke", 6), ("pilot", 30)),
)
def test_frozen_protocols_prepare_with_resolved_identities(
    name: str,
    expected_pairs: int,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    protocol = ActionCollectionProtocol.from_dict(
        json.loads(
            (
                repository
                / "lab"
                / "action_dynamics"
                / f"{name}-protocol.json"
            ).read_text()
        )
    )

    manifests, assignments = prepare_action_collection(
        protocol,
        image_digests=_IMAGES,
        observation_schema_sha256=_OBSERVATION_SCHEMA,
    )

    assert len(manifests) == expected_pairs * 2
    assert len(assignments) == expected_pairs * 2
    assert all(
        len(manifest.action_case.case_id) == 36
        for manifest in manifests
    )
    with pytest.raises(
        ValueError, match="resolved image"
    ):
        prepare_action_collection(protocol)


def test_v3_smoke_freezes_api_resolution_and_enqueue_drain_probe() -> None:
    repository = Path(__file__).resolve().parents[1]
    protocol = ActionCollectionProtocol.from_dict(
        json.loads(
            (
                repository
                / "lab"
                / "action_dynamics"
                / "smoke-protocol-v3.json"
            ).read_text()
        )
    )

    manifests, _ = prepare_action_collection(
        protocol,
        image_digests=_IMAGES,
        observation_schema_sha256=_OBSERVATION_SCHEMA,
    )
    treatments = [
        manifest
        for manifest in manifests
        if manifest.action_case.actions
    ]
    api = next(
        manifest
        for manifest in treatments
        if manifest.action_case.actions[0].action_kind
        == "api_rejection"
    )
    api_action = api.action_case.actions[0]
    enqueue = [
        manifest
        for manifest in treatments
        if manifest.action_case.actions[0].action_kind
        == "redis_enqueue_delay"
    ]

    assert protocol.point_count == 108
    assert api_action.magnitude == 0.25
    assert api_action.duration == 20
    assert api.request_schedule[
        api_action.start_index : api_action.stop_index
    ] == (12,) * 20
    assert len(enqueue) == 2
    assert all(
        manifest.action_case.worker_replicas == 3
        and manifest.action_case.actions[0].magnitude == 20.0
        and manifest.action_case.actions[0].duration == 20
        and manifest.request_schedule[84:92] == (0,) * 8
        and manifest.request_schedule[92:107] == (8,) * 15
        for manifest in enqueue
    )


def test_v4_smoke_freezes_continuous_recovery_and_twin_barrier() -> None:
    repository = Path(__file__).resolve().parents[1]
    protocol = ActionCollectionProtocol.from_dict(
        json.loads(
            (
                repository
                / "lab"
                / "action_dynamics"
                / "smoke-protocol-v4.json"
            ).read_text()
        )
    )
    v3 = ActionCollectionProtocol.from_dict(
        json.loads(
            (
                repository
                / "lab"
                / "action_dynamics"
                / "smoke-protocol-v3.json"
            ).read_text()
        )
    )
    enqueue = protocol.action_library["redis_enqueue_delay"]

    assert protocol.scheduling["twin_wave_barrier"] is True
    assert protocol.gates["controller_key_readback_required"] is True
    assert (
        enqueue["recovery_window_kind"]
        == "post_stop_continuous_workload"
    )
    assert enqueue["recovery_washout_windows"] == 2
    assert enqueue["recovery_window_count"] == 16
    assert (
        enqueue["mechanistic_recovery_feature"]
        == "redis_enqueue_latency_ms"
    )
    assert enqueue["mechanistic_recovery_ratio_max"] == 0.3
    assert {
        cell["workload_seed"] for cell in protocol.design["cells"]
    }.isdisjoint(
        cell["workload_seed"] for cell in v3.design["cells"]
    )
    assert {
        cell["intervention_seed"]
        for cell in protocol.design["cells"]
    }.isdisjoint(
        cell["intervention_seed"] for cell in v3.design["cells"]
    )


def test_v4_enqueue_recovery_excludes_cold_restart_probe(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    protocol = ActionCollectionProtocol.from_dict(
        json.loads(
            (
                repository
                / "lab"
                / "action_dynamics"
                / "smoke-protocol-v4.json"
            ).read_text()
        )
    )
    prepared = tmp_path / "inputs"
    write_prepared_action_collection(
        protocol,
        prepared,
        image_digests=_IMAGES,
        observation_schema_sha256=_OBSERVATION_SCHEMA,
        application_build_context_sha256=_BUILD_CONTEXT_SHA256,
    )
    _, manifests, assignments = load_prepared_action_collection(
        prepared
    )
    captures = tmp_path / "cases"
    _write_qualified_captures(captures, manifests)
    treatment = next(
        manifest
        for manifest in manifests
        if manifest.action_case.actions
        and manifest.action_case.actions[0].action_kind
        == "redis_enqueue_delay"
    )
    metrics_path = (
        captures
        / treatment.action_case.case_id
        / "collector-metrics.jsonl"
    )
    metrics = json.loads(metrics_path.read_text())
    request_latency = next(
        metric
        for metric in metrics["resourceMetrics"][0][
            "scopeMetrics"
        ][0]["metrics"]
        if metric["name"] == "request_latency_ms"
    )
    points = request_latency["gauge"]["dataPoints"]
    for point in points[-8:]:
        point["asDouble"] = 104.0
    metrics_path.write_text(json.dumps(metrics) + "\n")
    attestation = tmp_path / "collection-attestation.json"
    attestation.write_text(
        _pretty(_attestation(prepared, assignments))
    )

    cold_restart_noisy = assess_prepared_action_collection(
        prepared, captures, attestation
    )

    assert cold_restart_noisy["status"] == "qualified"
    enqueue_pair = next(
        pair
        for pair in cold_restart_noisy["pairs"]
        if pair["pair_id"]
        == treatment.action_case.matched_pair_id
    )
    assert enqueue_pair["recovery_window_kind"] == (
        "post_stop_continuous_workload"
    )
    assert enqueue_pair["mechanistic_recovery_passed"] is True

    action = treatment.action_case.actions[0]
    for index in range(action.stop_index + 3, action.stop_index + 19):
        points[index]["asDouble"] = 104.0
    metrics_path.write_text(json.dumps(metrics) + "\n")

    continuous_recovery_bad = assess_prepared_action_collection(
        prepared, captures, attestation
    )

    assert continuous_recovery_bad["status"] == "failed"
    assert continuous_recovery_bad["gates"]["recovery"] is False
    assert treatment.action_case.matched_pair_id in (
        continuous_recovery_bad["failed_pair_ids"]
    )

    for index in range(action.stop_index + 3, action.stop_index + 19):
        points[index]["asDouble"] = 4.0
    direct_enqueue = next(
        metric
        for metric in metrics["resourceMetrics"][0][
            "scopeMetrics"
        ][0]["metrics"]
        if metric["name"] == "redis_enqueue_latency_ms"
    )
    for index in range(action.stop_index + 3, action.stop_index + 19):
        direct_enqueue["gauge"]["dataPoints"][index][
            "asDouble"
        ] = 104.0
    metrics_path.write_text(json.dumps(metrics) + "\n")

    mechanistic_recovery_bad = assess_prepared_action_collection(
        prepared, captures, attestation
    )

    assert mechanistic_recovery_bad["status"] == "failed"
    assert mechanistic_recovery_bad["gates"][
        "enqueue_mechanistic_recovery"
    ] is False
    assert treatment.action_case.matched_pair_id in (
        mechanistic_recovery_bad["failed_pair_ids"]
    )

    for index in range(action.stop_index + 3, action.stop_index + 19):
        direct_enqueue["gauge"]["dataPoints"][index][
            "asDouble"
        ] = 4.0
    metrics_path.write_text(json.dumps(metrics) + "\n")
    points[-1]["asDouble"] = -1.0
    metrics_path.write_text(json.dumps(metrics) + "\n")

    restart_liveness_bad = assess_prepared_action_collection(
        prepared, captures, attestation
    )

    assert restart_liveness_bad["status"] == "failed"
    assert restart_liveness_bad["gates"][
        "enqueue_restart_liveness"
    ] is False
    assert treatment.action_case.matched_pair_id in (
        restart_liveness_bad["failed_pair_ids"]
    )

    points[-1]["asDouble"] = 104.0
    metrics_path.write_text(json.dumps(metrics) + "\n")
    actions_path = (
        captures
        / treatment.action_case.case_id
        / "collector-actions.jsonl"
    )
    action_payload = json.loads(actions_path.read_text())
    records = action_payload["resourceLogs"][0]["scopeLogs"][0][
        "logRecords"
    ]
    closed = next(
        record
        for record in records
        if any(
            attribute["key"] == "quantis.run.phase"
            and attribute["value"].get("stringValue") == "closed"
            for attribute in record["attributes"]
        )
    )
    readback = next(
        attribute
        for attribute in closed["attributes"]
        if attribute["key"]
        == "quantis.run.redis_enqueue_delay_ms"
    )
    readback["value"]["doubleValue"] = 1.0
    actions_path.write_text(json.dumps(action_payload) + "\n")

    stale_controller_key = assess_prepared_action_collection(
        prepared, captures, attestation
    )

    assert stale_controller_key["status"] == "failed"
    assert stale_controller_key["gates"][
        "action_command_coverage"
    ] is False


def test_v3_assessment_requires_api_counts_and_enqueue_drain(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    protocol = ActionCollectionProtocol.from_dict(
        json.loads(
            (
                repository
                / "lab"
                / "action_dynamics"
                / "smoke-protocol-v3.json"
            ).read_text()
        )
    )
    prepared = tmp_path / "inputs"
    write_prepared_action_collection(
        protocol,
        prepared,
        image_digests=_IMAGES,
        observation_schema_sha256=_OBSERVATION_SCHEMA,
        application_build_context_sha256=_BUILD_CONTEXT_SHA256,
    )
    _, manifests, assignments = load_prepared_action_collection(
        prepared
    )
    captures = tmp_path / "cases"
    _write_qualified_captures(captures, manifests)
    attestation = tmp_path / "collection-attestation.json"
    attestation.write_text(
        _pretty(_attestation(prepared, assignments))
    )

    assessment = assess_prepared_action_collection(
        prepared, captures, attestation
    )
    api = next(
        pair
        for pair in assessment["pairs"]
        if pair["action_kind"] == "api_rejection"
    )
    enqueue = [
        pair
        for pair in assessment["pairs"]
        if pair["action_kind"] == "redis_enqueue_delay"
    ]

    assert assessment["status"] == "qualified"
    assert assessment["gates"]["api_count_resolution"] is True
    assert assessment["gates"][
        "request_count_schedule_binding"
    ] is True
    assert assessment["gates"]["count_evidence_consistency"] is True
    assert assessment["gates"]["enqueue_drain_eligibility"] is True
    assert api["treatment_active_requests"] == 240
    assert api["control_active_requests"] == 240
    assert len(enqueue) == 2
    assert all(pair["drain_eligible"] for pair in enqueue)

    api_manifest = next(
        manifest
        for manifest in manifests
        if manifest.action_case.actions
        and manifest.action_case.actions[0].action_kind
        == "api_rejection"
    )
    metrics_path = (
        captures
        / api_manifest.action_case.case_id
        / "collector-metrics.jsonl"
    )
    metrics = json.loads(metrics_path.read_text())
    error_metric = next(
        metric
        for metric in metrics["resourceMetrics"][0][
            "scopeMetrics"
        ][0]["metrics"]
        if metric["name"]
        == "quantis.experiment.error_count"
    )
    error_metric["gauge"]["dataPoints"][-1]["asDouble"] = 0.5
    metrics_path.write_text(json.dumps(metrics) + "\n")

    inconsistent = assess_prepared_action_collection(
        prepared, captures, attestation
    )

    assert inconsistent["status"] == "failed"
    assert inconsistent["gates"]["count_evidence_consistency"] is False


def test_v3_assessment_rejects_counts_that_only_match_in_total(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    protocol = ActionCollectionProtocol.from_dict(
        json.loads(
            (
                repository
                / "lab"
                / "action_dynamics"
                / "smoke-protocol-v3.json"
            ).read_text()
        )
    )
    prepared = tmp_path / "inputs"
    write_prepared_action_collection(
        protocol,
        prepared,
        image_digests=_IMAGES,
        observation_schema_sha256=_OBSERVATION_SCHEMA,
        application_build_context_sha256=_BUILD_CONTEXT_SHA256,
    )
    _, manifests, assignments = load_prepared_action_collection(
        prepared
    )
    captures = tmp_path / "cases"
    _write_qualified_captures(captures, manifests)
    api = next(
        manifest
        for manifest in manifests
        if manifest.action_case.actions
        and manifest.action_case.actions[0].action_kind
        == "api_rejection"
    )
    metrics_path = (
        captures
        / api.action_case.case_id
        / "collector-metrics.jsonl"
    )
    metrics = json.loads(metrics_path.read_text())
    request_metric = next(
        metric
        for metric in metrics["resourceMetrics"][0][
            "scopeMetrics"
        ][0]["metrics"]
        if metric["name"]
        == "quantis.experiment.request_count"
    )
    action = api.action_case.actions[0]
    points = request_metric["gauge"]["dataPoints"]
    points[action.start_index + 1]["asDouble"] = 11.0
    points[action.start_index + 2]["asDouble"] = 13.0
    metrics_path.write_text(json.dumps(metrics) + "\n")
    attestation = tmp_path / "collection-attestation.json"
    attestation.write_text(
        _pretty(_attestation(prepared, assignments))
    )

    assessment = assess_prepared_action_collection(
        prepared, captures, attestation
    )

    assert assessment["status"] == "failed"
    assert assessment["gates"][
        "request_count_schedule_binding"
    ] is False


def test_prepared_collection_round_trip_recomputes_plan(
    tmp_path: Path,
) -> None:
    protocol = _protocol("smoke")
    prepared = tmp_path / "inputs"

    written = write_prepared_action_collection(
        protocol,
        prepared,
        image_digests=_IMAGES,
        observation_schema_sha256=_OBSERVATION_SCHEMA,
        application_build_context_sha256=_BUILD_CONTEXT_SHA256,
    )
    restored_protocol, manifests, assignments = (
        load_prepared_action_collection(prepared)
    )

    assert written["manifest_count"] == 12
    assert all(
        manifest.prepared_plan_sha256 == written["plan_sha256"]
        for manifest in manifests
    )
    assert json.loads((prepared / "plan.json").read_text())[
        "application_build_context_sha256"
    ] == _BUILD_CONTEXT_SHA256
    assert json.loads((prepared / "plan.json").read_text())[
        "kind"
    ] == "action_dynamics_execution_plan"
    assert (prepared / "manifest-index.json").is_file()
    assert restored_protocol == protocol
    assert len(manifests) == 12
    assert len(assignments) == 12
    assert {
        assignment.role for assignment in assignments
    } == {"treatment", "control"}
    with pytest.raises(FileExistsError):
        write_prepared_action_collection(
            protocol,
            prepared,
            image_digests=_IMAGES,
            observation_schema_sha256=_OBSERVATION_SCHEMA,
            application_build_context_sha256=(
                _BUILD_CONTEXT_SHA256
            ),
        )


def test_pilot_preparation_binds_qualified_smoke_evidence(
    tmp_path: Path,
) -> None:
    smoke = tmp_path / "smoke"
    smoke_inputs = smoke / "inputs"
    smoke_protocol = _protocol("smoke")
    write_prepared_action_collection(
        smoke_protocol,
        smoke_inputs,
        image_digests=_IMAGES,
        observation_schema_sha256=_OBSERVATION_SCHEMA,
        application_build_context_sha256=_BUILD_CONTEXT_SHA256,
    )
    _, smoke_manifests, smoke_assignments = (
        load_prepared_action_collection(smoke_inputs)
    )
    smoke_captures = smoke / "cases"
    _write_qualified_captures(
        smoke_captures, smoke_manifests
    )
    smoke_attestation = smoke / "collection-attestation.json"
    smoke_attestation.write_text(
        _pretty(_attestation(smoke_inputs, smoke_assignments))
    )
    write_action_collection_assessment(
        smoke_inputs,
        smoke_captures,
        smoke_attestation,
        smoke,
    )
    pilot_inputs = tmp_path / "pilot" / "inputs"

    written = write_prepared_action_collection(
        _protocol("instrumentation_pilot"),
        pilot_inputs,
        image_digests=_IMAGES,
        observation_schema_sha256=_OBSERVATION_SCHEMA,
        application_build_context_sha256=_BUILD_CONTEXT_SHA256,
        qualifying_smoke_directory=smoke,
    )
    _, manifests, assignments = load_prepared_action_collection(
        pilot_inputs
    )
    plan = json.loads((pilot_inputs / "plan.json").read_text())
    qualification = pilot_inputs / "smoke-qualification.json"

    assert qualification.is_file()
    assert plan["qualifying_smoke_sha256"] == hashlib.sha256(
        qualification.read_bytes()
    ).hexdigest()
    assert all(
        manifest.prepared_plan_sha256 == written["plan_sha256"]
        for manifest in manifests
    )
    assert all(
        manifest.qualifying_smoke_sha256
        == plan["qualifying_smoke_sha256"]
        for manifest in manifests
    )
    pilot_captures = pilot_inputs.parent / "cases"
    _write_qualified_captures(pilot_captures, manifests)
    pilot_attestation = (
        pilot_inputs.parent / "collection-attestation.json"
    )
    pilot_attestation.write_text(
        _pretty(_attestation(pilot_inputs, assignments))
    )
    assessment = assess_prepared_action_collection(
        pilot_inputs, pilot_captures, pilot_attestation
    )
    assert assessment["gates"][
        "qualifying_smoke_binding"
    ] is True
    assert assessment["identity"][
        "qualifying_smoke_sha256"
    ] == plan["qualifying_smoke_sha256"]
    assert assessment["identity"]["plan_sha256"] == written[
        "plan_sha256"
    ]
    write_action_collection_assessment(
        pilot_inputs,
        pilot_captures,
        pilot_attestation,
        pilot_inputs.parent,
    )
    assert plan["qualifying_smoke_sha256"] in (
        pilot_inputs.parent / "report.md"
    ).read_text()
    (smoke / "artifact-manifest.json").write_text(
        _pretty(
            {
                "schema_version": 1,
                "kind": "action_dynamics_artifact_manifest",
                "sha256": {},
            }
        )
    )
    with pytest.raises(ValueError, match="artifact hashes"):
        write_prepared_action_collection(
            _protocol("instrumentation_pilot"),
            tmp_path / "forged-pilot" / "inputs",
            image_digests=_IMAGES,
            observation_schema_sha256=_OBSERVATION_SCHEMA,
            application_build_context_sha256=(
                _BUILD_CONTEXT_SHA256
            ),
            qualifying_smoke_directory=smoke,
        )


def test_development_plan_is_complete_pair_split_factorial() -> None:
    protocol = _development_protocol()

    manifests, assignments = prepare_action_collection(
        protocol,
        image_digests=_IMAGES,
        observation_schema_sha256=_OBSERVATION_SCHEMA,
    )

    assert len(manifests) == 240
    assert len(assignments) == 240
    pair_splits: dict[str, set[str]] = {}
    cell_splits: dict[tuple[str, int], list[str]] = {}
    for manifest in manifests:
        pair_id = manifest.action_case.matched_pair_id
        split = manifest.action_case.split
        pair_splits.setdefault(pair_id, set()).add(split)
        if manifest.action_case.actions:
            action = manifest.action_case.actions[0]
            cell_splits.setdefault(
                (
                    action.action_kind,
                    manifest.action_case.worker_replicas,
                ),
                [],
            ).append(split)

    assert len(pair_splits) == 120
    assert all(len(splits) == 1 for splits in pair_splits.values())
    assert sum(splits == {"training"} for splits in pair_splits.values()) == 90
    assert sum(splits == {"validation"} for splits in pair_splits.values()) == 30
    assert set(cell_splits) == {
        (kind, workers)
        for kind in ACTION_KINDS
        for workers in (1, 2, 3)
    }
    assert all(
        splits.count("training") == 6
        and splits.count("validation") == 2
        for splits in cell_splits.values()
    )


def test_development_binds_qualified_v4_smoke_and_pilot(
    tmp_path: Path,
) -> None:
    smoke, pilot = _qualified_v4_smoke_and_pilot(tmp_path)
    authorization = _qualification_authorization(smoke, pilot)
    frozen_authorization = _development_protocol().analysis[
        "authorization_identity"
    ]
    assert isinstance(frozen_authorization, dict)
    with pytest.raises(ValueError, match="not authorized"):
        write_prepared_action_collection(
            _development_protocol(),
            tmp_path / "lookalike-v4" / "inputs",
            image_digests=_IMAGES,
            observation_schema_sha256=_OBSERVATION_SCHEMA,
            application_build_context_sha256=str(
                frozen_authorization[
                    "application_build_context_sha256"
                ]
            ),
            qualifying_smoke_directory=smoke,
            qualifying_pilot_directory=pilot,
        )
    with pytest.raises(ValueError, match="identity"):
        write_prepared_action_collection(
            _development_protocol(authorization),
            tmp_path / "wrong-stack" / "inputs",
            image_digests=_IMAGES,
            observation_schema_sha256=_OBSERVATION_SCHEMA,
            application_build_context_sha256="c" * 64,
            qualifying_smoke_directory=smoke,
            qualifying_pilot_directory=pilot,
        )
    development = tmp_path / "development"
    inputs = development / "inputs"

    written = write_prepared_action_collection(
        _development_protocol(authorization),
        inputs,
        image_digests=_IMAGES,
        observation_schema_sha256=_OBSERVATION_SCHEMA,
        application_build_context_sha256=_BUILD_CONTEXT_SHA256,
        qualifying_smoke_directory=smoke,
        qualifying_pilot_directory=pilot,
    )
    protocol, manifests, assignments = (
        load_prepared_action_collection(inputs)
    )
    plan = json.loads((inputs / "plan.json").read_text())

    assert protocol.stage == "development"
    assert plan["schema_version"] == 3
    assert _HEX(plan["qualifying_smoke_sha256"])
    assert _HEX(plan["qualifying_pilot_sha256"])
    assert plan["split_summary"] == {
        "split_unit": "matched_pair",
        "training_pair_count": 90,
        "validation_pair_count": 30,
        "training_capture_count": 180,
        "validation_capture_count": 60,
    }
    assert all(
        manifest.schema_version == 3
        and manifest.corpus_role == "development"
        and manifest.prepared_plan_sha256 == written["plan_sha256"]
        and manifest.qualifying_smoke_sha256
        == plan["qualifying_smoke_sha256"]
        for manifest in manifests
    )

    captures = development / "cases"
    _write_qualified_captures(captures, manifests)
    attestation = development / "collection-attestation.json"
    invalid_attestation = _attestation(inputs, assignments)
    invalid_attestation["qualifying_pilot_sha256"] = "d" * 64
    attestation.write_text(_pretty(invalid_attestation))
    with pytest.raises(
        ValueError, match="collection attestation"
    ):
        assess_prepared_action_collection(
            inputs, captures, attestation
        )
    attestation.write_text(
        _pretty(_attestation(inputs, assignments))
    )
    assessment = assess_prepared_action_collection(
        inputs, captures, attestation
    )

    assert assessment["status"] == "qualified"
    assert assessment["decision"] == "freeze_training_corpus"
    assert assessment["gates"]["whole_pair_split"] is True
    assert assessment["gates"]["qualifying_smoke_binding"] is True
    assert assessment["gates"]["qualifying_pilot_binding"] is True
    assert assessment["identity"]["qualifying_pilot_sha256"] == (
        plan["qualifying_pilot_sha256"]
    )


def test_assessment_recomputes_smoke_evidence(
    tmp_path: Path,
) -> None:
    protocol = _protocol("smoke")
    prepared = tmp_path / "inputs"
    write_prepared_action_collection(
        protocol,
        prepared,
        image_digests=_IMAGES,
        observation_schema_sha256=_OBSERVATION_SCHEMA,
        application_build_context_sha256=_BUILD_CONTEXT_SHA256,
    )
    _, manifests, assignments = load_prepared_action_collection(prepared)
    captures = tmp_path / "cases"
    _write_qualified_captures(captures, manifests)
    attestation = _attestation(prepared, assignments)
    attestation_path = tmp_path / "collection-attestation.json"
    attestation_path.write_text(_pretty(attestation))

    assessment = assess_prepared_action_collection(
        prepared,
        captures,
        attestation_path,
    )

    assert assessment["status"] == "qualified"
    assert assessment["decision"] == "advance_to_instrumentation_pilot"
    assert assessment["counts"] == {
        "case_count": 12,
        "pair_count": 6,
        "treatment_count": 6,
        "control_count": 6,
    }
    assert assessment["gates"]["final_plan_binding"] is True
    api_pair = next(
        pair
        for pair in assessment["pairs"]
        if pair["action_kind"] == "api_rejection"
    )
    assert api_pair["effect_statistic"] == (
        "pooled_error_count_rate_difference"
    )
    assert api_pair["treatment_active_requests"] > 0
    assert api_pair["treatment_active_requests"] == (
        api_pair["control_active_requests"]
    )
    assert api_pair["active_effect"] >= 0.4
    assert all(assessment["gates"].values())


def test_assessment_rejects_truth_leak_and_missing_stop(
    tmp_path: Path,
) -> None:
    protocol = _protocol("smoke")
    prepared = tmp_path / "inputs"
    write_prepared_action_collection(
        protocol,
        prepared,
        image_digests=_IMAGES,
        observation_schema_sha256=_OBSERVATION_SCHEMA,
        application_build_context_sha256=_BUILD_CONTEXT_SHA256,
    )
    _, manifests, assignments = load_prepared_action_collection(prepared)
    captures = tmp_path / "cases"
    _write_qualified_captures(captures, manifests)
    treatment = next(
        manifest
        for manifest in manifests
        if manifest.action_case.actions
    )
    directory = captures / treatment.action_case.case_id
    logs = json.loads(
        (directory / "collector-logs.jsonl").read_text()
    )
    logs["resourceLogs"][0]["resource"]["attributes"].append(
        {
            "key": "quantis.action.kind",
            "value": {"stringValue": "worker_pause"},
        }
    )
    (directory / "collector-logs.jsonl").write_text(
        json.dumps(logs) + "\n"
    )
    actions = json.loads(
        (directory / "collector-actions.jsonl").read_text()
    )
    actions["resourceLogs"][0]["scopeLogs"][0]["logRecords"] = (
        actions["resourceLogs"][0]["scopeLogs"][0]["logRecords"][:1]
    )
    (directory / "collector-actions.jsonl").write_text(
        json.dumps(actions) + "\n"
    )
    attestation_path = tmp_path / "collection-attestation.json"
    attestation_path.write_text(
        _pretty(_attestation(prepared, assignments))
    )

    assessment = assess_prepared_action_collection(
        prepared,
        captures,
        attestation_path,
    )

    assert assessment["status"] == "failed"
    assert assessment["gates"]["truth_exclusion"] is False
    assert assessment["gates"]["action_command_coverage"] is False


def test_assessment_rejects_incomplete_metrics_and_broken_trace_chain(
    tmp_path: Path,
) -> None:
    protocol = _protocol("smoke")
    prepared = tmp_path / "inputs"
    write_prepared_action_collection(
        protocol,
        prepared,
        image_digests=_IMAGES,
        observation_schema_sha256=_OBSERVATION_SCHEMA,
        application_build_context_sha256=_BUILD_CONTEXT_SHA256,
    )
    _, manifests, assignments = load_prepared_action_collection(
        prepared
    )
    captures = tmp_path / "cases"
    _write_qualified_captures(captures, manifests)
    first = captures / manifests[0].action_case.case_id
    metrics = json.loads(
        (first / "collector-metrics.jsonl").read_text()
    )
    metrics["resourceMetrics"][0]["scopeMetrics"][0][
        "metrics"
    ].pop()
    (first / "collector-metrics.jsonl").write_text(
        json.dumps(metrics) + "\n"
    )
    traces = json.loads(
        (first / "collector-traces.jsonl").read_text()
    )
    traces["resourceSpans"][0]["scopeSpans"][0]["spans"][3][
        "parentSpanId"
    ] = "ffffffffffffffff"
    (first / "collector-traces.jsonl").write_text(
        json.dumps(traces) + "\n"
    )
    attestation_path = tmp_path / "collection-attestation.json"
    attestation_path.write_text(
        _pretty(_attestation(prepared, assignments))
    )

    assessment = assess_prepared_action_collection(
        prepared, captures, attestation_path
    )

    assert assessment["gates"]["metric_completeness"] is False
    assert (
        assessment["gates"]["complete_trace_coverage"] is False
    )

def _protocol(stage: str) -> ActionCollectionProtocol:
    pair_count = 6 if stage == "smoke" else 30
    design = (
        {
            "design_kind": "fixed_smoke_cells",
            "action_kind_quotas": {
                "worker_pause": 2,
                "postgres_lock": 1,
                "redis_enqueue_delay": 1,
                "redis_dequeue_delay": 1,
                "api_rejection": 1,
            },
            "topology_quotas": {
                "workers-1": 5,
                "workers-2": 1,
            },
            "cells": [
                {
                    "cell_id": f"smoke-{index:03d}",
                    "action_kind": kind,
                    "worker_replicas": 1,
                    "magnitude": (
                        1.0
                        if kind
                        in {"worker_pause", "postgres_lock"}
                        else (
                            0.5
                            if kind == "api_rejection"
                            else 40.0
                        )
                    ),
                    "onset_index": 28 + index,
                    "duration": 8,
                    "workload_seed": 51_000 + index,
                    "intervention_seed": 61_000 + index,
                }
                for index, kind in enumerate(ACTION_KINDS, start=1)
            ]
            + [
                {
                    "cell_id": "smoke-006",
                    "action_kind": "worker_pause",
                    "worker_replicas": 2,
                    "magnitude": 1.0,
                    "onset_index": 38,
                    "duration": 12,
                    "workload_seed": 51_006,
                    "intervention_seed": 61_006,
                }
            ],
        }
        if stage == "smoke"
        else {
            "design_kind": "complete_action_topology_factorial",
            "action_kinds": list(ACTION_KINDS),
            "worker_replica_values": [1, 2, 3],
            "replicates_per_cell": 2,
            "pair_count": 30,
        }
    )
    action_library = {
        "worker_pause": {
            "target_entity": "worker_pool",
            "magnitude_unit": "fraction",
            "severity_values": [1.0],
            "effect_feature": "worker_rate",
            "effect_direction": "decrease",
            "minimum_effect": 1.0,
            "recovery_feature_floor": 1.0,
            "recovery_ratio_max": 0.3,
        },
        "postgres_lock": {
            "target_entity": "worker_writes_postgresql",
            "magnitude_unit": "lock_count",
            "severity_values": [1.0],
            "effect_feature": "db_write_rate",
            "effect_direction": "decrease",
            "minimum_effect": 1.0,
            "recovery_feature_floor": 1.0,
            "recovery_ratio_max": 0.3,
        },
        "redis_enqueue_delay": {
            "target_entity": "api_enqueues_queue",
            "magnitude_unit": "milliseconds",
            "severity_values": [20.0, 40.0, 60.0],
            "effect_feature": "request_latency_ms",
            "effect_direction": "increase",
            "minimum_effect": 10.0,
            "recovery_feature_floor": 10.0,
            "recovery_ratio_max": 0.3,
        },
        "redis_dequeue_delay": {
            "target_entity": "queue_dequeues_to_worker",
            "magnitude_unit": "milliseconds",
            "severity_values": [20.0, 40.0, 60.0],
            "effect_feature": "redis_dequeue_latency_ms",
            "effect_direction": "increase",
            "minimum_effect": 10.0,
            "recovery_feature_floor": 10.0,
            "recovery_ratio_max": 0.3,
        },
        "api_rejection": {
            "target_entity": "api",
            "magnitude_unit": "probability",
            "severity_values": [0.25, 0.5, 0.75],
            "effect_feature": "error_rate",
            "effect_direction": "increase",
            "minimum_effect": 0.15,
            "recovery_feature_floor": 0.15,
            "recovery_ratio_max": 0.3,
            "effect_statistic": (
                "pooled_error_count_rate_difference"
            ),
        },
    }
    return ActionCollectionProtocol.from_dict(
        {
            "schema_version": 1,
            "kind": "action_dynamics_collection_protocol",
            "stage": stage,
            "evidence_boundary": "instrumentation evidence only",
            "generator_seed": 717,
            "collection": {
                "pair_count": pair_count,
                "parallel_jobs": 6,
                "expected_capture_count": pair_count * 2,
                "overwrite": False,
            },
            "trajectory": {
                "point_count": 84,
                "sample_period_seconds": 0.25,
                "onset_index_min": 28,
                "onset_index_max": 39,
                "duration_min": 8,
                "duration_max": 20,
                "minimum_recovery_windows": 24,
            },
            "workload": {
                "minimum_requests_per_window": 6,
                "maximum_requests_per_window": 10,
                "schedule_kind": "seeded_explicit_uniform_integer",
                "twins_share_exact_schedule": True,
                "api_request_queue_size": 128,
            },
            "action_library": action_library,
            "design": design,
            "scheduling": {
                "lane_count": 6,
                "twins_run_sequentially_in_lane": True,
                "fresh_project_between_twins": True,
            },
            "retry_policy": {
                "max_attempts_per_pair": 1,
                "automatic_retry": False,
            },
            "gates": {
                "eligible_event_trace_link_rate_min": 0.95,
                "eligible_completed_checkout_path_rate_min": 0.95,
                "placebo_false_positive_rate_max": 0.1,
                "cross_case_trace_reference_count_max": 0,
            },
            "capture": {
                "require_resolved_image_digests": True,
                "require_observation_schema_sha256": True,
                "case_ids": "opaque_uuid",
                "pair_ids": "opaque_uuid",
                "fresh_compose_project_per_capture": True,
                "shared_named_volumes_allowed": False,
                "host_ports_allowed": False,
            },
            "analysis": {"training_permitted": False},
            "claim": {
                "supported": "instrumentation qualification",
                "excluded": "world model",
            },
        }
    )


def _development_protocol(
    authorization_identity: Optional[dict[str, str]] = None,
) -> ActionCollectionProtocol:
    repository = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (
            repository
            / "lab"
            / "action_dynamics"
            / "development-protocol-v1.json"
        ).read_text()
    )
    if authorization_identity is not None:
        payload["analysis"]["authorization_identity"] = (
            authorization_identity
        )
    return ActionCollectionProtocol.from_dict(payload)


def _qualified_v4_smoke_and_pilot(
    root: Path,
) -> tuple[Path, Path]:
    repository = Path(__file__).resolve().parents[1]
    smoke = root / "qualifying-smoke"
    smoke_protocol = ActionCollectionProtocol.from_dict(
        json.loads(
            (
                repository
                / "lab"
                / "action_dynamics"
                / "smoke-protocol-v4.json"
            ).read_text()
        )
    )
    smoke_inputs = smoke / "inputs"
    write_prepared_action_collection(
        smoke_protocol,
        smoke_inputs,
        image_digests=_IMAGES,
        observation_schema_sha256=_OBSERVATION_SCHEMA,
        application_build_context_sha256=_BUILD_CONTEXT_SHA256,
    )
    _, smoke_manifests, smoke_assignments = (
        load_prepared_action_collection(smoke_inputs)
    )
    _write_qualified_captures(smoke / "cases", smoke_manifests)
    smoke_attestation = smoke / "collection-attestation.json"
    smoke_attestation.write_text(
        _pretty(_attestation(smoke_inputs, smoke_assignments))
    )
    write_action_collection_assessment(
        smoke_inputs,
        smoke / "cases",
        smoke_attestation,
        smoke,
    )

    pilot = root / "qualifying-pilot"
    pilot_protocol = ActionCollectionProtocol.from_dict(
        json.loads(
            (
                repository
                / "lab"
                / "action_dynamics"
                / "pilot-protocol-v4.json"
            ).read_text()
        )
    )
    pilot_inputs = pilot / "inputs"
    write_prepared_action_collection(
        pilot_protocol,
        pilot_inputs,
        image_digests=_IMAGES,
        observation_schema_sha256=_OBSERVATION_SCHEMA,
        application_build_context_sha256=_BUILD_CONTEXT_SHA256,
        qualifying_smoke_directory=smoke,
    )
    _, pilot_manifests, pilot_assignments = (
        load_prepared_action_collection(pilot_inputs)
    )
    _write_qualified_captures(pilot / "cases", pilot_manifests)
    pilot_attestation = pilot / "collection-attestation.json"
    pilot_attestation.write_text(
        _pretty(_attestation(pilot_inputs, pilot_assignments))
    )
    write_action_collection_assessment(
        pilot_inputs,
        pilot / "cases",
        pilot_attestation,
        pilot,
    )
    return smoke, pilot


def _HEX(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= set("0123456789abcdef")
    )


def _qualification_authorization(
    smoke: Path, pilot: Path
) -> dict[str, str]:
    smoke_protocol = json.loads(
        (smoke / "inputs" / "protocol.json").read_text()
    )
    smoke_plan = json.loads(
        (smoke / "inputs" / "plan.json").read_text()
    )
    pilot_protocol = json.loads(
        (pilot / "inputs" / "protocol.json").read_text()
    )
    pilot_plan = json.loads(
        (pilot / "inputs" / "plan.json").read_text()
    )
    return {
        "smoke_protocol_sha256": _canonical_sha256(
            smoke_protocol
        ),
        "smoke_execution_plan_sha256": _canonical_sha256(
            smoke_plan
        ),
        "pilot_protocol_sha256": _canonical_sha256(
            pilot_protocol
        ),
        "pilot_execution_plan_sha256": _canonical_sha256(
            pilot_plan
        ),
        "qualifying_smoke_sha256": hashlib.sha256(
            (
                pilot
                / "inputs"
                / "smoke-qualification.json"
            ).read_bytes()
        ).hexdigest(),
        "application_build_context_sha256": str(
            pilot_plan["application_build_context_sha256"]
        ),
    }


def _write_qualified_captures(
    root: Path,
    manifests: tuple[LabActionCaptureManifest, ...],
) -> None:
    by_pair = {
        manifest.action_case.matched_pair_id: [
            candidate
            for candidate in manifests
            if candidate.action_case.matched_pair_id
            == manifest.action_case.matched_pair_id
        ]
        for manifest in manifests
    }
    for manifest in manifests:
        directory = root / manifest.action_case.case_id
        directory.mkdir(parents=True)
        manifest_path = directory / "capture-manifest.json"
        manifest_path.write_text(_pretty(manifest.to_dict()))
        treatment = bool(manifest.action_case.actions)
        action = next(
            item.action_case.actions[0]
            for item in by_pair[manifest.action_case.matched_pair_id]
            if item.action_case.actions
        )
        values = []
        for index in range(manifest.action_case.point_count):
            value = 4.0
            if treatment and action.start_index < index <= action.stop_index:
                value += (
                    -2.0
                    if action.effect_direction == "decrease"
                    else max(20.0, action.minimum_effect + 1.0)
                )
            values.append(value)
        _write_metrics(directory, manifest, action.effect_feature, values)
        _write_logs_and_traces(directory, manifest)
        _write_actions(directory, manifest)
        (directory / "runner.log").write_text("completed\n")


def _write_metrics(
    directory: Path,
    manifest: LabActionCaptureManifest,
    feature: str,
    values: list[float],
) -> None:
    identity = _identity_attributes(manifest)
    identity.extend(
        [
            {
                "key": "quantis.application.image.id",
                "value": {
                    "stringValue": "sha256:" + "a" * 64
                },
            },
            {
                "key": (
                    "quantis.application.build_context.sha256"
                ),
                "value": {"stringValue": "b" * 64},
            },
        ]
    )

    def metric_value(metric_name: str, index: int) -> float:
        request_count = (
            0
            if index == 0
            else manifest.request_schedule[index - 1]
        )
        error_count = (
            round(request_count * 0.5)
            if feature == "error_rate" and values[index] > 4.0
            else 0
        )
        if metric_name == "quantis.experiment.request_count":
            return float(request_count)
        if metric_name == "quantis.experiment.error_count":
            return float(error_count)
        if metric_name == "request_rate":
            return request_count / manifest.sample_period_seconds
        if metric_name == "error_rate":
            return (
                error_count / request_count
                if request_count
                else 0.0
            )
        if metric_name == feature:
            return values[index]
        if (
            feature == "request_latency_ms"
            and metric_name == "redis_enqueue_latency_ms"
        ):
            return values[index]
        return 0.0

    payload = {
        "resourceMetrics": [
            {
                "resource": {"attributes": identity},
                "scopeMetrics": [
                    {
                        "scope": {"name": "quantis.action-lab"},
                        "metrics": [
                            {
                                "name": metric_name,
                                "gauge": {
                                    "dataPoints": [
                                        {
                                            "timeUnixNano": str(
                                                (index + 1)
                                                * 1_000_000_000
                                            ),
                                            "asDouble": metric_value(
                                                metric_name, index
                                            ),
                                        }
                                        for index, _ in enumerate(values)
                                    ]
                                },
                            }
                            for metric_name in (
                                *ACTION_LAB_FEATURE_NAMES,
                                *_EVIDENCE_METRIC_NAMES,
                                "quantis.experiment.window.closed_unix_nano",
                            )
                        ],
                    }
                ],
            }
        ]
    }
    (directory / "collector-metrics.jsonl").write_text(
        json.dumps(payload) + "\n"
    )


def _write_logs_and_traces(
    directory: Path,
    manifest: LabActionCaptureManifest,
) -> None:
    trace_id = hashlib.sha256(
        manifest.action_case.case_id.encode()
    ).hexdigest()[:32]
    span_ids = [f"{index + 1:016x}" for index in range(6)]
    names = [
        "api.admission",
        "redis.enqueue",
        "queue.residence",
        "redis.dequeue",
        "worker.processing",
        "postgresql.write",
    ]
    entities = [
        "api",
        "api_enqueues_queue",
        "checkout_queue",
        "queue_dequeues_to_worker",
        "worker_pool",
        "worker_writes_postgresql",
    ]
    identity = _identity_attributes(manifest)
    logs = {
        "resourceLogs": [
            {
                "resource": {"attributes": identity},
                "scopeLogs": [
                    {
                        "scope": {"name": "quantis.application"},
                        "logRecords": [
                            {
                                "timeUnixNano": "1",
                                "body": {
                                    "stringValue": "checkout accepted"
                                },
                                "attributes": [
                                    {
                                        "key": "event.name",
                                        "value": {
                                            "stringValue": "checkout.accepted"
                                        },
                                    }
                                ],
                                "traceId": trace_id,
                                "spanId": span_ids[0],
                            },
                            {
                                "timeUnixNano": "8",
                                "body": {"stringValue": "checkout completed"},
                                "attributes": [
                                    {
                                        "key": "event.name",
                                        "value": {
                                            "stringValue": "checkout.completed"
                                        },
                                    }
                                ],
                                "traceId": trace_id,
                                "spanId": span_ids[-1],
                            }
                        ],
                    }
                ],
            }
        ]
    }
    traces = {
        "resourceSpans": [
            {
                "resource": {"attributes": identity},
                "scopeSpans": [
                    {
                        "scope": {"name": "quantis.application"},
                        "spans": [
                            {
                                "traceId": trace_id,
                                "spanId": span_id,
                                **(
                                    {"parentSpanId": span_ids[index - 1]}
                                    if index
                                    else {}
                                ),
                                "name": name,
                                "attributes": [
                                    {
                                        "key": (
                                            "quantis.graph.entity.id"
                                        ),
                                        "value": {
                                            "stringValue": entities[index]
                                        },
                                    }
                                ],
                                "startTimeUnixNano": str(index + 1),
                                "endTimeUnixNano": str(index + 2),
                            }
                            for index, (name, span_id) in enumerate(
                                zip(names, span_ids)
                            )
                        ],
                    }
                ],
            }
        ]
    }
    (directory / "collector-logs.jsonl").write_text(
        json.dumps(logs) + "\n"
    )
    (directory / "collector-traces.jsonl").write_text(
        json.dumps(traces) + "\n"
    )


def _write_actions(
    directory: Path,
    manifest: LabActionCaptureManifest,
) -> None:
    records = []
    for action in manifest.action_case.actions:
        for phase, index in (
            ("start", action.start_index),
            ("stop", action.stop_index),
        ):
            records.append(
                {
                    "timeUnixNano": str(index + 1),
                    "body": {"stringValue": "action command"},
                    "attributes": [
                        {
                            "key": "event.name",
                            "value": {"stringValue": "action.command"},
                        },
                        {
                            "key": "quantis.action.id",
                            "value": {"stringValue": action.action_id},
                        },
                        {
                            "key": "quantis.action.phase",
                            "value": {"stringValue": phase},
                        },
                        {
                            "key": "quantis.action.kind",
                            "value": {
                                "stringValue": action.action_kind
                            },
                        },
                        {
                            "key": "quantis.action.target",
                            "value": {
                                "stringValue": action.target_entity
                            },
                        },
                        {
                            "key": "quantis.action.magnitude",
                            "value": {"doubleValue": action.magnitude},
                        },
                        {
                            "key": "quantis.action.logical_index",
                            "value": {"intValue": str(index)},
                        },
                        {
                            "key": "quantis.action.status",
                            "value": {"stringValue": "applied"},
                        },
                        {
                            "key": (
                                "quantis.controller."
                                "redis_enqueue_delay_ms"
                            ),
                            "value": {
                                "doubleValue": (
                                    action.magnitude
                                    if (
                                        action.action_kind
                                        == "redis_enqueue_delay"
                                        and phase == "start"
                                    )
                                    else 0.0
                                )
                            },
                        },
                        {
                            "key": (
                                "quantis.action.realized_worker_count"
                            ),
                            "value": {
                                "intValue": str(
                                    manifest.action_case.worker_replicas
                                    if action.action_kind
                                    == "worker_pause"
                                    else 0
                                )
                            },
                        },
                        {
                            "key": (
                                "quantis.action.realized_worker_ids"
                            ),
                            "value": {
                                "stringValue": (
                                    ",".join(
                                        f"worker-{worker}"
                                        for worker in range(
                                            manifest.action_case.worker_replicas
                                        )
                                    )
                                    if action.action_kind
                                    == "worker_pause"
                                    else ""
                                )
                            },
                        },
                    ],
                }
            )
    records.append(
        {
            "timeUnixNano": "999",
            "body": {"stringValue": "action run boundary"},
            "attributes": [
                {
                    "key": "event.name",
                    "value": {
                        "stringValue": "action.run.boundary"
                    },
                },
                {
                    "key": "quantis.run.phase",
                    "value": {"stringValue": "closed"},
                },
                {
                    "key": "quantis.run.active_action_count",
                    "value": {"intValue": "0"},
                },
                {
                    "key": "quantis.run.cleanup.status",
                    "value": {"stringValue": "clean"},
                },
                {
                    "key": "quantis.run.redis_enqueue_delay_ms",
                    "value": {"doubleValue": 0.0},
                },
            ],
        }
    )
    payload = {
        "resourceLogs": [
            {
                "resource": {
                    "attributes": _identity_attributes(manifest)
                },
                "scopeLogs": [
                    {
                        "scope": {"name": "quantis.action"},
                        "logRecords": records,
                    }
                ],
            }
        ]
    }
    (directory / "collector-actions.jsonl").write_text(
        json.dumps(payload) + "\n"
    )


def _identity_attributes(
    manifest: LabActionCaptureManifest,
) -> list[dict[str, object]]:
    return [
        {
            "key": "quantis.experiment.case.id",
            "value": {
                "stringValue": manifest.action_case.case_id
            },
        },
        {
            "key": "quantis.experiment.manifest.sha256",
            "value": {
                "stringValue": hashlib.sha256(
                    _pretty(manifest.to_dict()).encode()
                ).hexdigest()
            },
        },
        {
            "key": "quantis.experiment.topology.id",
            "value": {
                "stringValue": manifest.action_case.topology_id
            },
        },
    ]


def _attestation(
    prepared: Path,
    assignments: tuple[object, ...],
) -> dict[str, object]:
    protocol = json.loads((prepared / "protocol.json").read_text())
    plan = json.loads((prepared / "plan.json").read_text())
    raw_assignments = [
        assignment.to_dict()  # type: ignore[attr-defined]
        for assignment in assignments
    ]
    return {
        "schema_version": 1,
        "kind": "action_dynamics_collection_attestation",
        "execution_id": "test-execution",
        "started_unix_nano": 1,
        "completed_unix_nano": 2,
        "parallel_jobs": 6,
        "batch_count": max(
            int(item["batch"]) for item in raw_assignments
        ),
        "case_count": len(raw_assignments),
        "pair_count": len(raw_assignments) // 2,
        "application_image_id": "sha256:" + "a" * 64,
        "application_build_context_sha256": "b" * 64,
        "protocol_sha256": _canonical_sha256(protocol),
        "plan_sha256": _canonical_sha256(plan),
        "qualifying_smoke_sha256": plan.get(
            "qualifying_smoke_sha256"
        ),
        "qualifying_pilot_sha256": plan.get(
            "qualifying_pilot_sha256"
        ),
        "cases": [
            {
                **item,
                "compose_project": f"test-lane-{item['lane']}",
                "manifest_sha256": hashlib.sha256(
                    (
                        prepared
                        / "manifests"
                        / f"{item['case_id']}.json"
                    ).read_bytes()
                ).hexdigest(),
                "started_unix_nano": (
                    1 + int(item["order_in_pair"]) * 2
                ),
                "completed_unix_nano": (
                    2 + int(item["order_in_pair"]) * 2
                ),
            }
            for item in raw_assignments
        ],
    }


def _canonical_sha256(value: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _pretty(value: object) -> str:
    return json.dumps(
        value,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
