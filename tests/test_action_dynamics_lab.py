import hashlib
import json
from pathlib import Path

import pytest

from quantis_core.action_conditioned_dynamics import ACTION_KINDS
from quantis_core.action_dynamics_lab import (
    ActionCollectionProtocol,
    LabActionCaptureManifest,
    assess_prepared_action_collection,
    load_prepared_action_collection,
    prepare_action_collection,
    write_prepared_action_collection,
)

_IMAGES = {
    "application": "sha256:" + "a" * 64,
    "redis": "redis@sha256:" + "b" * 64,
}
_OBSERVATION_SCHEMA = "c" * 64


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
    )
    restored_protocol, manifests, assignments = (
        load_prepared_action_collection(prepared)
    )

    assert written["manifest_count"] == 12
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
    )
    _, manifests, assignments = load_prepared_action_collection(prepared)
    captures = tmp_path / "cases"
    _write_qualified_captures(captures, manifests)
    for manifest in manifests:
        if not manifest.action_case.actions:
            (
                captures
                / manifest.action_case.case_id
                / "collector-actions.jsonl"
            ).unlink()
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
    payload = {
        "resourceMetrics": [
            {
                "resource": {"attributes": identity},
                "scopeMetrics": [
                    {
                        "scope": {"name": "quantis.action-lab"},
                        "metrics": [
                            {
                                "name": feature,
                                "gauge": {
                                    "dataPoints": [
                                        {
                                            "timeUnixNano": str(
                                                (index + 1)
                                                * 1_000_000_000
                                            ),
                                            "asDouble": value,
                                        }
                                        for index, value in enumerate(values)
                                    ]
                                },
                            }
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
                "started_unix_nano": 1,
                "completed_unix_nano": 2,
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
