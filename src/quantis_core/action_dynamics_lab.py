"""Strict preparation and evidence checks for the action-dynamics lab."""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

from .action_conditioned_dynamics import (
    ACTION_KINDS,
    ActionConditionedCaseManifest,
    InterventionAction,
)
from .otlp import AttributeValue, read_otlp_capture
from .otlp_logs import read_otlp_log_capture


_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_HEX_TRACE_ID = re.compile(r"^[0-9a-f]{32}$")
_HEX_SPAN_ID = re.compile(r"^[0-9a-f]{16}$")
_PROTOCOL_KEYS = {
    "schema_version",
    "kind",
    "stage",
    "evidence_boundary",
    "generator_seed",
    "collection",
    "trajectory",
    "workload",
    "action_library",
    "design",
    "scheduling",
    "retry_policy",
    "gates",
    "analysis",
    "claim",
}
_CAPTURE_PROTOCOL_KEY = "capture"
_MANIFEST_KEYS = {
    "schema_version",
    "kind",
    "action_case",
    "sample_period_seconds",
    "request_schedule",
    "api_request_queue_size",
    "image_digests",
    "observation_schema_sha256",
    "protocol_sha256",
    "prepared_plan_sha256",
    "graph_observation_schema_sha256",
    "corpus_role",
}
_ASSIGNMENT_KEYS = {
    "pair_id",
    "case_id",
    "role",
    "lane",
    "batch",
    "order_in_pair",
    "worker_replicas",
}
_SPAN_NAMES = frozenset(
    {
        "api.admission",
        "redis.enqueue",
        "queue.residence",
        "redis.dequeue",
        "worker.processing",
        "postgresql.write",
    }
)
_ACTION_LAB_FEATURE_NAMES = (
    "request_rate",
    "request_latency_ms",
    "error_rate",
    "api_inflight_current",
    "api_inflight_peak",
    "api_concurrency_mean",
    "queue_depth",
    "queue_oldest_age_ms",
    "enqueue_event_age_ms",
    "dequeue_event_age_ms",
    "queue_residence_mean_ms",
    "worker_rate",
    "worker_heartbeat_age_s",
    "worker_active_count",
    "worker_busy_count",
    "worker_busy_age_max_ms",
    "worker_busy_fraction",
    "worker_processing_latency_ms",
    "redis_enqueue_latency_ms",
    "redis_enqueue_error_rate",
    "redis_dequeue_latency_ms",
    "redis_dequeue_error_rate",
    "db_write_rate",
    "postgresql_write_latency_ms",
    "postgresql_write_error_rate",
    "postgresql_write_event_age_ms",
    "postgresql_write_busy_age_max_ms",
)
ACTION_LAB_FEATURE_NAMES = _ACTION_LAB_FEATURE_NAMES
_SPAN_ENTITY_OWNERS = {
    "api.admission": "api",
    "redis.enqueue": "api_enqueues_queue",
    "queue.residence": "checkout_queue",
    "redis.dequeue": "queue_dequeues_to_worker",
    "worker.processing": "worker_pool",
    "postgresql.write": "worker_writes_postgresql",
}
_TRACE_EXEMPT_EVENTS = frozenset(
    {
        "worker.state.idle",
        "worker.state.busy",
        "queue.backlog.low",
        "queue.backlog.elevated",
        "queue.backlog.high",
    }
)
_FORBIDDEN_OBSERVATION_KEYS = (
    "quantis.action.",
    "fault.kind",
    "fault_kind",
    "matched_pair",
    "action_id",
    "action_kind",
    "target_entity",
    "intervention_seed",
    "action_phase",
    "action_magnitude",
)


@dataclass(frozen=True)
class LabActionCaptureManifest:
    """A static lab envelope around the frozen scientific v3 manifest."""

    action_case: ActionConditionedCaseManifest
    sample_period_seconds: float
    request_schedule: Tuple[int, ...]
    api_request_queue_size: int
    image_digests: Mapping[str, str]
    observation_schema_sha256: str
    protocol_sha256: str
    prepared_plan_sha256: str
    graph_observation_schema_sha256: str
    corpus_role: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or isinstance(self.sample_period_seconds, bool)
            or not math.isfinite(self.sample_period_seconds)
            or self.sample_period_seconds <= 0.0
            or len(self.request_schedule)
            != self.action_case.point_count
            or any(
                isinstance(value, bool) or value < 1
                for value in self.request_schedule
            )
            or isinstance(self.api_request_queue_size, bool)
            or self.api_request_queue_size < max(self.request_schedule)
            or not _HEX_SHA256.fullmatch(
                self.observation_schema_sha256
            )
            or not _HEX_SHA256.fullmatch(self.protocol_sha256)
            or not _HEX_SHA256.fullmatch(
                self.prepared_plan_sha256
            )
            or not _HEX_SHA256.fullmatch(
                self.graph_observation_schema_sha256
            )
            or self.graph_observation_schema_sha256
            != self.observation_schema_sha256
            or self.corpus_role
            not in {"smoke", "instrumentation_pilot"}
        ):
            raise ValueError("lab action capture manifest is invalid")
        if (
            len(set(self.image_digests)) != len(self.image_digests)
            or any(
                not name
                or not digest
                or (
                    "@sha256:" not in digest
                    and not digest.startswith("sha256:")
                )
                for name, digest in self.image_digests.items()
            )
        ):
            raise ValueError("lab action image digests are invalid")

    def to_dict(self) -> Dict[str, Any]:
        """Return a canonical JSON-compatible representation."""

        return {
            "schema_version": self.schema_version,
            "kind": "lab_action_capture_manifest",
            "action_case": self.action_case.to_dict(),
            "sample_period_seconds": self.sample_period_seconds,
            "request_schedule": list(self.request_schedule),
            "api_request_queue_size": self.api_request_queue_size,
            "image_digests": dict(sorted(self.image_digests.items())),
            "observation_schema_sha256": (
                self.observation_schema_sha256
            ),
            "protocol_sha256": self.protocol_sha256,
            "prepared_plan_sha256": self.prepared_plan_sha256,
            "graph_observation_schema_sha256": (
                self.graph_observation_schema_sha256
            ),
            "corpus_role": self.corpus_role,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "LabActionCaptureManifest":
        """Restore a manifest without coercing malformed field types."""

        if set(payload) != _MANIFEST_KEYS:
            raise ValueError("lab action manifest schema is invalid")
        raw_schedule = payload["request_schedule"]
        raw_images = payload["image_digests"]
        if (
            payload["schema_version"] != 1
            or payload["kind"] != "lab_action_capture_manifest"
            or not isinstance(payload["action_case"], dict)
            or isinstance(payload["sample_period_seconds"], bool)
            or not isinstance(
                payload["sample_period_seconds"], (int, float)
            )
            or not isinstance(raw_schedule, list)
            or any(not _is_integer(value) for value in raw_schedule)
            or not _is_integer(payload["api_request_queue_size"])
            or not isinstance(raw_images, dict)
            or any(
                not isinstance(name, str)
                or not isinstance(value, str)
                for name, value in raw_images.items()
            )
            or not isinstance(
                payload["observation_schema_sha256"], str
            )
            or not isinstance(payload["protocol_sha256"], str)
            or not isinstance(
                payload["prepared_plan_sha256"], str
            )
            or not isinstance(
                payload["graph_observation_schema_sha256"], str
            )
            or not isinstance(payload["corpus_role"], str)
        ):
            raise ValueError(
                "lab action manifest field types are invalid"
            )
        return cls(
            action_case=ActionConditionedCaseManifest.from_dict(
                payload["action_case"]
            ),
            sample_period_seconds=float(
                payload["sample_period_seconds"]
            ),
            request_schedule=tuple(raw_schedule),
            api_request_queue_size=payload[
                "api_request_queue_size"
            ],
            image_digests={
                name: value for name, value in raw_images.items()
            },
            observation_schema_sha256=payload[
                "observation_schema_sha256"
            ],
            protocol_sha256=payload["protocol_sha256"],
            prepared_plan_sha256=payload[
                "prepared_plan_sha256"
            ],
            graph_observation_schema_sha256=payload[
                "graph_observation_schema_sha256"
            ],
            corpus_role=payload["corpus_role"],
            schema_version=payload["schema_version"],
        )

    def canonical_sha256(self) -> str:
        """Hash the semantic capture manifest."""

        return _canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class ActionCollectionProtocol:
    """Frozen generator, scheduling, evidence, and claim configuration."""

    stage: str
    evidence_boundary: str
    generator_seed: int
    collection: Mapping[str, Any]
    trajectory: Mapping[str, Any]
    workload: Mapping[str, Any]
    action_library: Mapping[str, Any]
    design: Mapping[str, Any]
    scheduling: Mapping[str, Any]
    retry_policy: Mapping[str, Any]
    gates: Mapping[str, Any]
    analysis: Mapping[str, Any]
    claim: Mapping[str, Any]
    capture: Mapping[str, Any]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.stage not in {"smoke", "instrumentation_pilot"}
            or not self.evidence_boundary
            or not _is_integer(self.generator_seed)
        ):
            raise ValueError("action collection protocol is invalid")
        expected_pairs = 6 if self.stage == "smoke" else 30
        if (
            _required_integer(self.collection, "pair_count")
            != expected_pairs
            or _required_integer(
                self.collection, "expected_capture_count"
            )
            != expected_pairs * 2
            or _required_integer(
                self.collection, "parallel_jobs"
            )
            != 6
            or self.collection.get("overwrite") is not False
        ):
            raise ValueError(
                "action collection counts or overwrite policy are invalid"
            )
        point_count = _required_integer(
            self.trajectory, "point_count"
        )
        onset_min = _required_integer(
            self.trajectory, "onset_index_min"
        )
        onset_max = _required_integer(
            self.trajectory, "onset_index_max"
        )
        duration_min = _required_integer(
            self.trajectory, "duration_min"
        )
        duration_max = _required_integer(
            self.trajectory, "duration_max"
        )
        recovery = _required_integer(
            self.trajectory, "minimum_recovery_windows"
        )
        sample_period = _required_number(
            self.trajectory, "sample_period_seconds"
        )
        if (
            point_count != 84
            or sample_period != 0.25
            or not 0 <= onset_min <= onset_max
            or not 1 <= duration_min <= duration_max
            or onset_max + duration_max + recovery > point_count
        ):
            raise ValueError(
                "action collection trajectory is invalid"
            )
        minimum_requests = _required_integer(
            self.workload, "minimum_requests_per_window"
        )
        maximum_requests = _required_integer(
            self.workload, "maximum_requests_per_window"
        )
        if (
            minimum_requests < 1
            or maximum_requests < minimum_requests
            or self.workload.get("schedule_kind")
            != "seeded_explicit_uniform_integer"
            or self.workload.get("twins_share_exact_schedule")
            is not True
        ):
            raise ValueError("action workload protocol is invalid")
        if set(self.action_library) != set(ACTION_KINDS):
            raise ValueError("action library coverage is invalid")
        for kind in ACTION_KINDS:
            _validate_action_configuration(
                kind, _required_mapping(self.action_library, kind)
            )
        if (
            _required_integer(self.scheduling, "lane_count") != 6
            or self.scheduling.get("twins_run_sequentially_in_lane")
            is not True
            or self.scheduling.get("fresh_project_between_twins")
            is not True
            or _required_integer(
                self.retry_policy, "max_attempts_per_pair"
            )
            != 1
            or self.retry_policy.get("automatic_retry") is not False
        ):
            raise ValueError(
                "action scheduling or retry policy is invalid"
            )
        for name in (
            "eligible_event_trace_link_rate_min",
            "eligible_completed_checkout_path_rate_min",
        ):
            value = _required_number(self.gates, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError("action trace gate is invalid")
        _validate_design(self.stage, self.design)
        if self.stage == "smoke":
            for raw in self.design["cells"]:
                if not isinstance(raw, dict):
                    raise ValueError("smoke cell is invalid")
                kind = _required_text(raw, "action_kind")
                config = _required_mapping(
                    self.action_library, kind
                )
                if (
                    _required_number(raw, "magnitude")
                    not in _number_sequence(
                        config, "severity_values"
                    )
                    or not onset_min
                    <= _required_integer(raw, "onset_index")
                    <= onset_max
                    or not duration_min
                    <= _required_integer(raw, "duration")
                    <= duration_max
                    or _required_integer(raw, "onset_index")
                    + _required_integer(raw, "duration")
                    + recovery
                    > point_count
                ):
                    raise ValueError(
                        "smoke cell intervention is outside protocol"
                    )
        _validate_capture_requirements(self.capture)

    @property
    def pair_count(self) -> int:
        return _required_integer(self.collection, "pair_count")

    @property
    def parallel_jobs(self) -> int:
        return _required_integer(
            self.collection, "parallel_jobs"
        )

    @property
    def point_count(self) -> int:
        return _required_integer(self.trajectory, "point_count")

    def to_dict(self) -> Dict[str, Any]:
        """Return the exact frozen protocol representation."""

        payload = {
            "schema_version": self.schema_version,
            "kind": "action_dynamics_collection_protocol",
            "stage": self.stage,
            "evidence_boundary": self.evidence_boundary,
            "generator_seed": self.generator_seed,
            "collection": _json_copy(self.collection),
            "trajectory": _json_copy(self.trajectory),
            "workload": _json_copy(self.workload),
            "action_library": _json_copy(self.action_library),
            "design": _json_copy(self.design),
            "scheduling": _json_copy(self.scheduling),
            "retry_policy": _json_copy(self.retry_policy),
            "gates": _json_copy(self.gates),
            "analysis": _json_copy(self.analysis),
            "claim": _json_copy(self.claim),
        }
        if self.capture:
            payload[_CAPTURE_PROTOCOL_KEY] = _json_copy(self.capture)
        return payload

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "ActionCollectionProtocol":
        """Restore and strictly validate one collection protocol."""

        keys = set(payload)
        if keys not in (
            _PROTOCOL_KEYS,
            _PROTOCOL_KEYS | {_CAPTURE_PROTOCOL_KEY},
        ):
            raise ValueError(
                "action collection protocol schema is invalid"
            )
        mapping_names = (
            "collection",
            "trajectory",
            "workload",
            "action_library",
            "design",
            "scheduling",
            "retry_policy",
            "gates",
            "analysis",
            "claim",
        )
        if (
            payload["schema_version"] != 1
            or payload["kind"]
            != "action_dynamics_collection_protocol"
            or not isinstance(payload["stage"], str)
            or not isinstance(payload["evidence_boundary"], str)
            or not _is_integer(payload["generator_seed"])
            or any(
                not isinstance(payload[name], dict)
                for name in mapping_names
            )
            or (
                _CAPTURE_PROTOCOL_KEY in payload
                and not isinstance(
                    payload[_CAPTURE_PROTOCOL_KEY], dict
                )
            )
        ):
            raise ValueError(
                "action collection protocol field types are invalid"
            )
        return cls(
            stage=payload["stage"],
            evidence_boundary=payload["evidence_boundary"],
            generator_seed=payload["generator_seed"],
            collection=_json_copy(payload["collection"]),
            trajectory=_json_copy(payload["trajectory"]),
            workload=_json_copy(payload["workload"]),
            action_library=_json_copy(payload["action_library"]),
            design=_json_copy(payload["design"]),
            scheduling=_json_copy(payload["scheduling"]),
            retry_policy=_json_copy(payload["retry_policy"]),
            gates=_json_copy(payload["gates"]),
            analysis=_json_copy(payload["analysis"]),
            claim=_json_copy(payload["claim"]),
            capture=_json_copy(
                payload.get(_CAPTURE_PROTOCOL_KEY, {})
            ),
            schema_version=payload["schema_version"],
        )

    def canonical_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class CaptureAssignment:
    """One opaque treatment or control capture in a pair-atomic lane."""

    pair_id: str
    case_id: str
    role: str
    lane: int
    batch: int
    order_in_pair: int
    worker_replicas: int

    def __post_init__(self) -> None:
        if (
            not self.pair_id
            or not self.case_id
            or self.role not in {"treatment", "control"}
            or not 1 <= self.lane <= 6
            or self.batch < 1
            or self.order_in_pair not in {0, 1}
            or self.worker_replicas not in {1, 2, 3}
        ):
            raise ValueError("capture assignment is invalid")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "case_id": self.case_id,
            "role": self.role,
            "lane": self.lane,
            "batch": self.batch,
            "order_in_pair": self.order_in_pair,
            "worker_replicas": self.worker_replicas,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "CaptureAssignment":
        if set(payload) != _ASSIGNMENT_KEYS:
            raise ValueError("capture assignment schema is invalid")
        if (
            not isinstance(payload["pair_id"], str)
            or not isinstance(payload["case_id"], str)
            or not isinstance(payload["role"], str)
            or any(
                not _is_integer(payload[name])
                for name in (
                    "lane",
                    "batch",
                    "order_in_pair",
                    "worker_replicas",
                )
            )
        ):
            raise ValueError("capture assignment types are invalid")
        return cls(
            pair_id=payload["pair_id"],
            case_id=payload["case_id"],
            role=payload["role"],
            lane=payload["lane"],
            batch=payload["batch"],
            order_in_pair=payload["order_in_pair"],
            worker_replicas=payload["worker_replicas"],
        )


def prepare_action_collection(
    protocol: ActionCollectionProtocol,
    *,
    image_digests: Mapping[str, str] | None = None,
    observation_schema_sha256: str | None = None,
) -> Tuple[
    Tuple[LabActionCaptureManifest, ...],
    Tuple[CaptureAssignment, ...],
]:
    """Derive all opaque manifests and pair-atomic assignments."""

    cells = _design_cells(protocol)
    if len(cells) != protocol.pair_count:
        raise ValueError("prepared design does not match pair count")
    manifests = []
    assignments = []
    image_digests, observation_schema, queue_size = (
        _capture_settings(
            protocol,
            image_digests=image_digests,
            observation_schema_sha256=observation_schema_sha256,
        )
    )
    protocol_sha256 = protocol.canonical_sha256()
    prepared_plan_sha256 = _canonical_sha256(
        {
            "schema_version": 1,
            "kind": "action_dynamics_prepared_plan_identity",
            "protocol_sha256": protocol_sha256,
            "generator": "prepare_action_collection:v2",
        }
    )
    for pair_position, (kind, workers, replicate) in enumerate(
        cells
    ):
        pair_id = _opaque_id(
            protocol.generator_seed,
            f"pair:{pair_position}:{kind}:{workers}:{replicate}",
        )
        smoke_cell = (
            _smoke_cell(protocol, replicate)
            if protocol.stage == "smoke"
            else None
        )
        workload_seed = (
            _required_integer(smoke_cell, "workload_seed")
            if smoke_cell is not None
            else _derived_int(
                protocol.generator_seed, f"{pair_id}:workload"
            )
        )
        intervention_seed = (
            _required_integer(smoke_cell, "intervention_seed")
            if smoke_cell is not None
            else _derived_int(
                protocol.generator_seed, f"{pair_id}:intervention"
            )
        )
        request_schedule = _request_schedule(
            protocol, workload_seed
        )
        action_config = _required_mapping(
            protocol.action_library, kind
        )
        if smoke_cell is not None:
            onset = _required_integer(smoke_cell, "onset_index")
            duration = _required_integer(smoke_cell, "duration")
        else:
            onset, duration = _pilot_action_schedule(
                protocol, kind, workers, replicate
            )
        stop = onset + duration
        severity_values = _number_sequence(
            action_config, "severity_values"
        )
        magnitude = (
            _required_number(smoke_cell, "magnitude")
            if smoke_cell is not None
            else severity_values[
                ((workers - 1) + replicate) % len(severity_values)
            ]
        )
        action = InterventionAction(
            action_id=_opaque_id(
                protocol.generator_seed, f"{pair_id}:action"
            ),
            action_kind=kind,
            target_entity=_required_text(
                action_config, "target_entity"
            ),
            start_index=onset,
            stop_index=stop,
            magnitude=magnitude,
            magnitude_unit=_required_text(
                action_config, "magnitude_unit"
            ),
            effect_feature=_required_text(
                action_config, "effect_feature"
            ),
            effect_direction=_required_text(
                action_config, "effect_direction"
            ),
            minimum_effect=_required_number(
                action_config, "minimum_effect"
            ),
            recovery_tolerance=_required_number(
                action_config, "recovery_ratio_max"
            ),
        )
        pair_treatment_first = pair_position % 2 == 0
        lane = pair_position % protocol.parallel_jobs + 1
        batch = pair_position // protocol.parallel_jobs + 1
        for role in ("treatment", "control"):
            case_id = _opaque_id(
                protocol.generator_seed, f"{pair_id}:{role}"
            )
            action_case = ActionConditionedCaseManifest(
                case_id=case_id,
                matched_pair_id=pair_id,
                split="validation",
                point_count=protocol.point_count,
                logical_window_period_nano=int(
                    _required_number(
                        protocol.trajectory,
                        "sample_period_seconds",
                    )
                    * 1_000_000_000
                ),
                topology_id=f"workers-{workers}",
                worker_replicas=workers,
                workload_seed=workload_seed,
                intervention_seed=intervention_seed,
                actions=(action,) if role == "treatment" else (),
            )
            manifests.append(
                LabActionCaptureManifest(
                    action_case=action_case,
                    sample_period_seconds=_required_number(
                        protocol.trajectory,
                        "sample_period_seconds",
                    ),
                    request_schedule=request_schedule,
                    api_request_queue_size=queue_size,
                    image_digests=image_digests,
                    observation_schema_sha256=observation_schema,
                    protocol_sha256=protocol_sha256,
                    prepared_plan_sha256=prepared_plan_sha256,
                    graph_observation_schema_sha256=(
                        observation_schema
                    ),
                    corpus_role=protocol.stage,
                )
            )
            treatment_order = 0 if pair_treatment_first else 1
            order = (
                treatment_order
                if role == "treatment"
                else 1 - treatment_order
            )
            assignments.append(
                CaptureAssignment(
                    pair_id=pair_id,
                    case_id=case_id,
                    role=role,
                    lane=lane,
                    batch=batch,
                    order_in_pair=order,
                    worker_replicas=workers,
                )
            )
    return tuple(manifests), tuple(assignments)


def write_prepared_action_collection(
    protocol: ActionCollectionProtocol,
    output_directory: Path,
    *,
    image_digests: Mapping[str, str] | None = None,
    observation_schema_sha256: str | None = None,
) -> Mapping[str, Any]:
    """Persist the exact protocol, plan, and manifests once."""

    output = Path(output_directory)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite prepared collection: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    manifests_directory = output / "manifests"
    manifests_directory.mkdir()
    manifests, assignments = prepare_action_collection(
        protocol,
        image_digests=image_digests,
        observation_schema_sha256=observation_schema_sha256,
    )
    protocol_payload = protocol.to_dict()
    protocol_path = output / "protocol.json"
    protocol_path.write_text(_pretty_json(protocol_payload))
    manifest_sha256s = {}
    for manifest in manifests:
        case_id = manifest.action_case.case_id
        path = manifests_directory / f"{case_id}.json"
        path.write_text(_pretty_json(manifest.to_dict()))
        manifest_sha256s[case_id] = _file_sha256(path)
    plan = {
        "schema_version": 1,
        "kind": "action_dynamics_collection_plan",
        "protocol_sha256": _canonical_sha256(protocol_payload),
        "manifest_sha256s": dict(sorted(manifest_sha256s.items())),
        "assignments": [
            assignment.to_dict() for assignment in assignments
        ],
    }
    (output / "plan.json").write_text(_pretty_json(plan))
    return {
        "schema_version": 1,
        "kind": "prepared_action_collection",
        "protocol_sha256": protocol.canonical_sha256(),
        "plan_sha256": _canonical_sha256(plan),
        "manifest_count": len(manifests),
        "assignment_count": len(assignments),
    }


def load_prepared_action_collection(
    prepared_directory: Path,
) -> Tuple[
    ActionCollectionProtocol,
    Tuple[LabActionCaptureManifest, ...],
    Tuple[CaptureAssignment, ...],
]:
    """Restore and independently regenerate a prepared collection."""

    prepared = Path(prepared_directory)
    protocol_payload = _read_object(prepared / "protocol.json")
    plan = _read_object(prepared / "plan.json")
    if set(plan) != {
        "schema_version",
        "kind",
        "protocol_sha256",
        "manifest_sha256s",
        "assignments",
    }:
        raise ValueError("prepared collection plan schema is invalid")
    if (
        plan["schema_version"] != 1
        or plan["kind"] != "action_dynamics_collection_plan"
        or plan["protocol_sha256"]
        != _canonical_sha256(protocol_payload)
        or not isinstance(plan["manifest_sha256s"], dict)
        or not isinstance(plan["assignments"], list)
    ):
        raise ValueError("prepared collection plan is invalid")
    protocol = ActionCollectionProtocol.from_dict(protocol_payload)
    assignments = tuple(
        CaptureAssignment.from_dict(raw)
        for raw in plan["assignments"]
        if isinstance(raw, dict)
    )
    if len(assignments) != len(plan["assignments"]):
        raise ValueError("prepared assignments are invalid")
    manifests = []
    manifest_paths = sorted(
        (prepared / "manifests").glob("*.json")
    )
    for path in manifest_paths:
        manifest = LabActionCaptureManifest.from_dict(
            _read_object(path)
        )
        if path.stem != manifest.action_case.case_id:
            raise ValueError("prepared manifest filename drifted")
        expected_sha = plan["manifest_sha256s"].get(path.stem)
        if expected_sha != _file_sha256(path):
            raise ValueError("prepared manifest hash drifted")
        manifests.append(manifest)
    if not manifests:
        raise ValueError("prepared collection contains no manifests")
    expected_manifests, expected_assignments = prepare_action_collection(
        protocol,
        image_digests=manifests[0].image_digests,
        observation_schema_sha256=(
            manifests[0].observation_schema_sha256
        ),
    )
    if (
        tuple(manifests) != tuple(
            sorted(
                expected_manifests,
                key=lambda value: value.action_case.case_id,
            )
        )
        or assignments != expected_assignments
        or set(plan["manifest_sha256s"])
        != {
            manifest.action_case.case_id
            for manifest in expected_manifests
        }
    ):
        raise ValueError(
            "prepared collection differs from deterministic generator"
        )
    return protocol, tuple(manifests), assignments


def assess_prepared_action_collection(
    prepared_directory: Path,
    captures_directory: Path,
    attestation_path: Path,
) -> Mapping[str, Any]:
    """Recompute every Phase-1/2 data-quality gate from raw files."""

    protocol, manifests, assignments = (
        load_prepared_action_collection(prepared_directory)
    )
    manifest_by_case = {
        manifest.action_case.case_id: manifest
        for manifest in manifests
    }
    assignment_by_case = {
        assignment.case_id: assignment
        for assignment in assignments
    }
    attestation = _read_object(attestation_path)
    _validate_attestation(
        attestation,
        Path(prepared_directory),
        assignments,
    )
    captures_root = Path(captures_directory)
    observed_directories = {
        path.name for path in captures_root.iterdir() if path.is_dir()
    }
    if observed_directories != set(manifest_by_case):
        raise ValueError("capture case coverage differs from plan")

    cases: Dict[str, Mapping[str, Any]] = {}
    truth_exclusion = True
    identity_binding = True
    action_coverage = True
    metric_completeness = True
    trace_link_numerator = 0
    trace_link_denominator = 0
    complete_trace_numerator = 0
    complete_trace_denominator = 0
    for case_id in sorted(manifest_by_case):
        manifest = manifest_by_case[case_id]
        assignment = assignment_by_case[case_id]
        capture = _assess_capture(
            captures_root / case_id,
            manifest,
            assignment,
            application_image_id=str(
                attestation["application_image_id"]
            ),
            application_build_context_sha256=str(
                attestation[
                    "application_build_context_sha256"
                ]
            ),
        )
        cases[case_id] = capture
        truth_exclusion = truth_exclusion and bool(
            capture["truth_exclusion"]
        )
        identity_binding = identity_binding and bool(
            capture["identity_binding"]
        )
        action_coverage = action_coverage and bool(
            capture["action_command_coverage"]
        )
        metric_completeness = metric_completeness and bool(
            capture["metric_completeness"]
        )
        trace_link_numerator += int(
            capture["trace_link_numerator"]
        )
        trace_link_denominator += int(
            capture["trace_link_denominator"]
        )
        complete_trace_numerator += int(
            capture["complete_trace_numerator"]
        )
        complete_trace_denominator += int(
            capture["complete_trace_denominator"]
        )

    pair_results = _assess_pairs(
        protocol, manifests, cases
    )
    schedule_alignment = all(
        bool(result["schedule_alignment"])
        for result in pair_results
    )
    raw_effects = all(
        bool(result["raw_effect_passed"])
        for result in pair_results
    )
    recovery = all(
        bool(result["recovery_passed"])
        for result in pair_results
    )
    placebo_false_positive_rate = (
        sum(
            bool(result["placebo_false_positive"])
            for result in pair_results
        )
        / len(pair_results)
    )
    cross_case_trace_references = _cross_case_trace_references(
        cases
    )
    trace_link_coverage = _coverage(
        trace_link_numerator, trace_link_denominator
    )
    complete_trace_coverage = _coverage(
        complete_trace_numerator,
        complete_trace_denominator,
    )
    gates = {
        "capture_count": len(cases)
        == int(protocol.collection["expected_capture_count"]),
        "pair_schedule_alignment": schedule_alignment,
        "identity_and_hash_binding": identity_binding,
        "action_command_coverage": action_coverage,
        "metric_completeness": metric_completeness,
        "truth_exclusion": truth_exclusion,
        "trace_link_coverage": trace_link_coverage
        >= _required_number(
            protocol.gates, "eligible_event_trace_link_rate_min"
        ),
        "complete_trace_coverage": complete_trace_coverage
        >= _required_number(
            protocol.gates,
            "eligible_completed_checkout_path_rate_min",
        ),
        "raw_effects": raw_effects,
        "recovery": recovery,
        "placebo_false_positive_rate": (
            placebo_false_positive_rate
            <= _required_number(
                protocol.gates,
                "placebo_false_positive_rate_max",
            )
        ),
        "cross_case_trace_references": (
            cross_case_trace_references
            <= _required_integer(
                protocol.gates,
                "cross_case_trace_reference_count_max",
            )
        ),
        "lane_isolation": _lane_isolation(
            attestation, protocol.parallel_jobs
        ),
    }
    qualified = all(gates.values())
    return {
        "schema_version": 1,
        "kind": "action_dynamics_data_quality_assessment",
        "stage": protocol.stage,
        "evidence_boundary": protocol.evidence_boundary,
        "status": "qualified" if qualified else "failed",
        "decision": (
            (
                "advance_to_instrumentation_pilot"
                if protocol.stage == "smoke"
                else "freeze_development_generator"
            )
            if qualified
            else "stop_and_repair_instrumentation"
        ),
        "counts": {
            "case_count": len(cases),
            "pair_count": len(pair_results),
            "treatment_count": sum(
                assignment.role == "treatment"
                for assignment in assignments
            ),
            "control_count": sum(
                assignment.role == "control"
                for assignment in assignments
            ),
        },
        "coverage": {
            "trace_link": trace_link_coverage,
            "complete_trace": complete_trace_coverage,
            "placebo_false_positive_rate": (
                placebo_false_positive_rate
            ),
            "cross_case_trace_reference_count": (
                cross_case_trace_references
            ),
            "trace_link_numerator": trace_link_numerator,
            "trace_link_denominator": trace_link_denominator,
            "complete_trace_numerator": (
                complete_trace_numerator
            ),
            "complete_trace_denominator": (
                complete_trace_denominator
            ),
        },
        "pair_counts_by_action": {
            kind: sum(
                result["action_kind"] == kind
                for result in pair_results
            )
            for kind in ACTION_KINDS
        },
        "pair_counts_by_topology": {
            f"workers-{workers}": sum(
                result["worker_replicas"] == workers
                for result in pair_results
            )
            for workers in (1, 2, 3)
        },
        "failed_pair_ids": [
            str(result["pair_id"])
            for result in pair_results
            if not (
                result["raw_effect_passed"]
                and result["recovery_passed"]
                and result["schedule_alignment"]
            )
        ],
        "attrition": {
            "planned_captures": len(assignments),
            "observed_captures": len(cases),
            "missing_captures": len(assignments) - len(cases),
            "automatic_retries": 0,
        },
        "gates": gates,
        "pairs": pair_results,
        "case_file_sha256s": {
            case_id: dict(cases[case_id]["file_sha256s"])
            for case_id in sorted(cases)
        },
        "limitations": [
            "instrumentation and randomized-action data quality only",
            "no model was trained or evaluated",
            "not evidence for a world-model claim",
        ],
    }


def write_action_collection_assessment(
    prepared_directory: Path,
    captures_directory: Path,
    attestation_path: Path,
    output_directory: Path,
) -> Mapping[str, Any]:
    """Write recomputed JSON, report, and a final file-hash manifest."""

    output = Path(output_directory)
    targets = (
        output / "data-quality.json",
        output / "report.md",
        output / "artifact-manifest.json",
    )
    if any(path.exists() for path in targets):
        raise FileExistsError(
            "refusing to overwrite action collection assessment"
        )
    output.mkdir(parents=True, exist_ok=True)
    assessment = assess_prepared_action_collection(
        prepared_directory, captures_directory, attestation_path
    )
    targets[0].write_text(_pretty_json(assessment))
    targets[1].write_text(_assessment_report(assessment))
    sha256s = {
        str(path.relative_to(output)): _file_sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path != targets[2]
    }
    artifact_manifest = {
        "schema_version": 1,
        "kind": "action_dynamics_artifact_manifest",
        "sha256": sha256s,
    }
    targets[2].write_text(_pretty_json(artifact_manifest))
    return assessment


def _design_cells(
    protocol: ActionCollectionProtocol,
) -> Tuple[Tuple[str, int, int], ...]:
    if protocol.stage == "instrumentation_pilot":
        cells = tuple(
            (kind, workers, replicate)
            for kind in ACTION_KINDS
            for workers in (1, 2, 3)
            for replicate in range(2)
        )
        return tuple(
            sorted(
                cells,
                key=lambda cell: _derived_int(
                    protocol.generator_seed,
                    "pilot-permutation:"
                    f"{cell[0]}:{cell[1]}:{cell[2]}",
                ),
            )
        )
    raw_cells = protocol.design["cells"]
    if not isinstance(raw_cells, list):
        raise ValueError("smoke design cells are invalid")
    smoke_cells: list[Tuple[str, int, int]] = []
    for position, raw in enumerate(raw_cells):
        if not isinstance(raw, dict):
            raise ValueError("smoke design cell is invalid")
        kind = _required_text(raw, "action_kind")
        workers = _required_integer(raw, "worker_replicas")
        smoke_cells.append((kind, workers, position))
    return tuple(smoke_cells)


def _request_schedule(
    protocol: ActionCollectionProtocol,
    workload_seed: int,
) -> Tuple[int, ...]:
    minimum = _required_integer(
        protocol.workload, "minimum_requests_per_window"
    )
    maximum = _required_integer(
        protocol.workload, "maximum_requests_per_window"
    )
    return tuple(
        _bounded_derived(
            workload_seed, f"request:{index}", minimum, maximum
        )
        for index in range(protocol.point_count)
    )


def _validate_capture_requirements(
    capture: Mapping[str, Any],
) -> None:
    if (
        capture.get("require_resolved_image_digests") is not True
        or capture.get("require_observation_schema_sha256") is not True
        or capture.get("case_ids") != "opaque_uuid"
        or capture.get("pair_ids") != "opaque_uuid"
        or capture.get("fresh_compose_project_per_capture") is not True
        or capture.get("shared_named_volumes_allowed") is not False
        or capture.get("host_ports_allowed") is not False
    ):
        raise ValueError("capture identity requirements are invalid")


def _capture_settings(
    protocol: ActionCollectionProtocol,
    *,
    image_digests: Mapping[str, str] | None,
    observation_schema_sha256: str | None,
) -> Tuple[Mapping[str, str], str, int]:
    _validate_capture_requirements(protocol.capture)
    raw_images = image_digests
    if (
        raw_images is None
        or not raw_images
        or any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or (
                re.fullmatch(r"sha256:[0-9a-f]{64}", value)
                is None
                and re.search(r"@sha256:[0-9a-f]{64}$", value)
                is None
            )
            for key, value in raw_images.items()
        )
    ):
        raise ValueError(
            "capture requires resolved image SHA-256 digests"
        )
    observation_schema = observation_schema_sha256
    if (
        not isinstance(observation_schema, str)
        or not _HEX_SHA256.fullmatch(observation_schema)
    ):
        raise ValueError("capture observation schema is invalid")
    raw_queue_size = protocol.workload.get(
        "api_request_queue_size", 128
    )
    if not _is_integer(raw_queue_size) or raw_queue_size < 1:
        raise ValueError("capture API queue size is invalid")
    return (
        {str(key): str(value) for key, value in raw_images.items()},
        observation_schema,
        raw_queue_size,
    )


def _smoke_cell(
    protocol: ActionCollectionProtocol, ordinal: int
) -> Mapping[str, Any]:
    raw_cells = protocol.design.get("cells")
    if (
        not isinstance(raw_cells, list)
        or not 0 <= ordinal < len(raw_cells)
        or not isinstance(raw_cells[ordinal], dict)
    ):
        raise ValueError("smoke cell is invalid")
    cell = raw_cells[ordinal]
    if not isinstance(cell, dict):
        raise AssertionError("validated smoke cell changed type")
    return dict(cell)


def _pilot_action_schedule(
    protocol: ActionCollectionProtocol,
    kind: str,
    workers: int,
    replicate: int,
) -> Tuple[int, int]:
    onset_min = _required_integer(
        protocol.trajectory, "onset_index_min"
    )
    onset_max = _required_integer(
        protocol.trajectory, "onset_index_max"
    )
    duration_min = _required_integer(
        protocol.trajectory, "duration_min"
    )
    duration_max = _required_integer(
        protocol.trajectory, "duration_max"
    )
    recovery = _required_integer(
        protocol.trajectory, "minimum_recovery_windows"
    )
    candidates = [
        (onset, duration)
        for onset in range(onset_min, onset_max + 1)
        for duration in range(duration_min, duration_max + 1)
        if onset + duration + recovery <= protocol.point_count
    ]
    ordered = sorted(
        candidates,
        key=lambda value: _derived_int(
            protocol.generator_seed,
            f"schedule:{kind}:{value[0]}:{value[1]}",
        ),
    )
    return ordered[(workers - 1) * 2 + replicate]


def _validate_action_configuration(
    kind: str, config: Mapping[str, Any]
) -> None:
    if (
        kind not in ACTION_KINDS
        or not _required_text(config, "target_entity")
        or not _required_text(config, "magnitude_unit")
        or not _required_text(config, "effect_feature")
        or _required_text(config, "effect_direction")
        not in {"increase", "decrease"}
        or _required_number(config, "minimum_effect") <= 0.0
        or _required_number(config, "recovery_feature_floor")
        <= 0.0
        or not 0.0
        <= _required_number(config, "recovery_ratio_max")
        <= 1.0
        or not _number_sequence(config, "severity_values")
    ):
        raise ValueError(f"invalid action configuration: {kind}")


def _validate_design(stage: str, design: Mapping[str, Any]) -> None:
    if stage == "instrumentation_pilot":
        if (
            design.get("design_kind")
            != "complete_action_topology_factorial"
            or tuple(design.get("action_kinds", ())) != ACTION_KINDS
            or design.get("worker_replica_values") != [1, 2, 3]
            or design.get("replicates_per_cell") != 2
            or design.get("pair_count") != 30
        ):
            raise ValueError("pilot design is not the frozen factorial")
        return
    cells = design.get("cells")
    if (
        design.get("design_kind") != "fixed_smoke_cells"
        or not isinstance(cells, list)
        or len(cells) != 6
    ):
        raise ValueError("smoke design is invalid")
    kinds = [
        raw.get("action_kind")
        for raw in cells
        if isinstance(raw, dict)
    ]
    if set(kinds) != set(ACTION_KINDS):
        raise ValueError("smoke design lacks action coverage")
    cell_ids = set()
    observed_kind_counts: Dict[str, int] = {}
    observed_topology_counts: Dict[str, int] = {}
    for raw in cells:
        if not isinstance(raw, dict) or set(raw) != {
            "cell_id",
            "action_kind",
            "worker_replicas",
            "magnitude",
            "onset_index",
            "duration",
            "workload_seed",
            "intervention_seed",
        }:
            raise ValueError("smoke design cell schema is invalid")
        cell_id = _required_text(raw, "cell_id")
        kind = _required_text(raw, "action_kind")
        workers = _required_integer(raw, "worker_replicas")
        if (
            cell_id in cell_ids
            or kind not in ACTION_KINDS
            or workers not in {1, 2, 3}
        ):
            raise ValueError("smoke design cell identity is invalid")
        cell_ids.add(cell_id)
        observed_kind_counts[kind] = (
            observed_kind_counts.get(kind, 0) + 1
        )
        topology_id = f"workers-{workers}"
        observed_topology_counts[topology_id] = (
            observed_topology_counts.get(topology_id, 0) + 1
        )
        for name in (
            "onset_index",
            "duration",
            "workload_seed",
            "intervention_seed",
        ):
            _required_integer(raw, name)
        _required_number(raw, "magnitude")
    if (
        design.get("action_kind_quotas") != observed_kind_counts
        or design.get("topology_quotas")
        != observed_topology_counts
    ):
        raise ValueError("smoke design quotas differ from cells")


def _validate_attestation(
    attestation: Mapping[str, Any],
    prepared: Path,
    assignments: Sequence[CaptureAssignment],
) -> None:
    protocol = _read_object(prepared / "protocol.json")
    plan = _read_object(prepared / "plan.json")
    if (
        attestation.get("schema_version") != 1
        or attestation.get("kind")
        != "action_dynamics_collection_attestation"
        or attestation.get("protocol_sha256")
        != _canonical_sha256(protocol)
        or attestation.get("plan_sha256")
        != _canonical_sha256(plan)
        or attestation.get("case_count") != len(assignments)
        or attestation.get("pair_count") != len(assignments) // 2
        or attestation.get("parallel_jobs") != 6
        or not isinstance(attestation.get("cases"), list)
        or not isinstance(
            attestation.get("application_image_id"), str
        )
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(attestation.get("application_image_id")),
        )
        is None
        or not isinstance(
            attestation.get(
                "application_build_context_sha256"
            ),
            str,
        )
        or _HEX_SHA256.fullmatch(
            str(
                attestation.get(
                    "application_build_context_sha256"
                )
            )
        )
        is None
    ):
        raise ValueError("collection attestation is invalid")
    attested = {}
    for raw in attestation["cases"]:
        if not isinstance(raw, dict):
            raise ValueError("attested case is invalid")
        case_id = raw.get("case_id")
        if not isinstance(case_id, str) or case_id in attested:
            raise ValueError("attested case identity is invalid")
        attested[case_id] = raw
    expected = {item.case_id: item for item in assignments}
    if set(attested) != set(expected):
        raise ValueError("attested cases differ from plan")
    for case_id, assignment in expected.items():
        raw = attested[case_id]
        if any(
            raw.get(key) != value
            for key, value in assignment.to_dict().items()
        ):
            raise ValueError("attested assignment differs from plan")
        manifest_path = prepared / "manifests" / f"{case_id}.json"
        if raw.get("manifest_sha256") != _file_sha256(
            manifest_path
        ):
            raise ValueError("attested manifest hash differs")


def _assess_capture(
    directory: Path,
    manifest: LabActionCaptureManifest,
    assignment: CaptureAssignment,
    *,
    application_image_id: str,
    application_build_context_sha256: str,
) -> Mapping[str, Any]:
    required = (
        "capture-manifest.json",
        "collector-metrics.jsonl",
        "collector-logs.jsonl",
        "collector-traces.jsonl",
        "runner.log",
    )
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise ValueError(
            f"capture {manifest.action_case.case_id} is missing {missing}"
        )
    capture_manifest_path = directory / "capture-manifest.json"
    restored = LabActionCaptureManifest.from_dict(
        _read_object(capture_manifest_path)
    )
    if restored != manifest:
        raise ValueError("captured manifest differs from prepared input")
    manifest_file_sha = _file_sha256(capture_manifest_path)
    metrics_path = directory / "collector-metrics.jsonl"
    logs_path = directory / "collector-logs.jsonl"
    actions_path = directory / "collector-actions.jsonl"
    traces_path = directory / "collector-traces.jsonl"
    metric_capture = read_otlp_capture(metrics_path)
    log_capture = read_otlp_log_capture(logs_path)
    raw_actions = _read_json_lines(
        actions_path, allow_empty=True
    )
    raw_traces = _read_json_lines(traces_path)
    observation_payloads = [
        *_read_json_lines(metrics_path),
        *_read_json_lines(logs_path),
        *raw_traces,
    ]
    forbidden_values = {
        *ACTION_KINDS,
        manifest.action_case.matched_pair_id,
        *(
            action.action_id
            for action in manifest.action_case.actions
        ),
    }
    truth_exclusion = not any(
        _contains_truth(payload, forbidden_values)
        for payload in observation_payloads
    )
    identity_binding = all(
        _identity_matches(
            payload,
            manifest.action_case.case_id,
            manifest_file_sha,
            manifest.action_case.topology_id,
        )
        for payload in observation_payloads + raw_actions
    ) and _metric_build_identity_matches(
        _read_json_lines(metrics_path),
        application_image_id,
        application_build_context_sha256,
    ) and application_image_id in set(manifest.image_digests.values())
    commands = _action_commands(raw_actions)
    action_command_coverage = _commands_match(
        commands, manifest, assignment
    ) and _cleanup_boundary_matches(raw_actions)
    trace_records = [
        record
        for record in log_capture.records
        if str(record.record_attributes.get("event.name", ""))
        not in _TRACE_EXEMPT_EVENTS
    ]
    trace_link_numerator = sum(
        bool(_HEX_TRACE_ID.fullmatch(record.trace_id))
        and bool(_HEX_SPAN_ID.fullmatch(record.span_id))
        for record in trace_records
    )
    spans = _trace_spans(raw_traces)
    valid_trace_structure = _valid_trace_structure(spans)
    accepted_trace_ids = {
        record.trace_id
        for record in trace_records
        if record.record_attributes.get("event.name")
        == "checkout.accepted"
        and _HEX_TRACE_ID.fullmatch(record.trace_id)
    }
    completed_trace_ids = {
        record.trace_id
        for record in trace_records
        if record.record_attributes.get("event.name")
        == "checkout.completed"
        and _HEX_TRACE_ID.fullmatch(record.trace_id)
    }
    complete_trace_numerator = sum(
        trace_id in completed_trace_ids
        and _has_exact_trace_path(spans, trace_id)
        for trace_id in accepted_trace_ids
    )
    if not valid_trace_structure:
        complete_trace_numerator = 0
    metric_series: Dict[str, Tuple[float, ...]] = {}
    metric_timestamps: Dict[str, Tuple[int, ...]] = {}
    observed_metric_names = {
        point.metric_name for point in metric_capture.points
    }
    allowed_metric_names = set(_ACTION_LAB_FEATURE_NAMES) | {
        "quantis.experiment.window.closed_unix_nano"
    }
    for metric_name in observed_metric_names:
        points = sorted(
            (
                point
                for point in metric_capture.points
                if point.metric_name == metric_name
            ),
            key=lambda point: point.time_unix_nano,
        )
        if (
            len(points) == manifest.action_case.point_count
            and all(point.number_value is not None for point in points)
            and len({point.time_unix_nano for point in points})
            == manifest.action_case.point_count
        ):
            metric_series[metric_name] = tuple(
                float(point.number_value)  # type: ignore[arg-type]
                for point in points
            )
            metric_timestamps[metric_name] = tuple(
                point.time_unix_nano for point in points
            )
    required_timestamps = {
        metric_timestamps.get(name)
        for name in _ACTION_LAB_FEATURE_NAMES
    }
    metric_completeness = (
        observed_metric_names == allowed_metric_names
        and set(_ACTION_LAB_FEATURE_NAMES) <= set(metric_series)
        and len(required_timestamps) == 1
        and None not in required_timestamps
    )
    return {
        "truth_exclusion": truth_exclusion,
        "identity_binding": identity_binding,
        "action_command_coverage": action_command_coverage,
        "metric_completeness": metric_completeness,
        "trace_link_numerator": trace_link_numerator,
        "trace_link_denominator": len(trace_records),
        "complete_trace_numerator": complete_trace_numerator,
        "complete_trace_denominator": len(accepted_trace_ids),
        "trace_ids": tuple(
            sorted(
                {
                    str(span["trace_id"]) for span in spans
                }
                | {
                    record.trace_id
                    for record in trace_records
                    if record.trace_id
                }
            )
        ),
        "metric_series": metric_series,
        "file_sha256s": {
            name: _file_sha256(directory / name)
            for name in (*required, "collector-actions.jsonl")
            if (directory / name).is_file()
        },
    }


def _assess_pairs(
    protocol: ActionCollectionProtocol,
    manifests: Sequence[LabActionCaptureManifest],
    cases: Mapping[str, Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    grouped: Dict[str, list[LabActionCaptureManifest]] = {}
    for manifest in manifests:
        grouped.setdefault(
            manifest.action_case.matched_pair_id, []
        ).append(manifest)
    results: list[Mapping[str, Any]] = []
    for pair_id in sorted(grouped):
        pair = grouped[pair_id]
        treatment = next(
            manifest
            for manifest in pair
            if manifest.action_case.actions
        )
        control = next(
            manifest
            for manifest in pair
            if not manifest.action_case.actions
        )
        action = treatment.action_case.actions[0]
        action_config = _required_mapping(
            protocol.action_library, action.action_kind
        )
        treatment_series = cases[
            treatment.action_case.case_id
        ]["metric_series"]
        control_series = cases[
            control.action_case.case_id
        ]["metric_series"]
        feature = action.effect_feature
        treatment_values = treatment_series.get(feature)
        control_values = control_series.get(feature)
        schedule_alignment = (
            treatment.request_schedule == control.request_schedule
            and treatment.action_case.workload_seed
            == control.action_case.workload_seed
            and treatment.action_case.intervention_seed
            == control.action_case.intervention_seed
            and treatment.action_case.topology_id
            == control.action_case.topology_id
        )
        if treatment_values is None or control_values is None:
            active_effect = float("nan")
            recovery_ratio = float("inf")
            placebo_false_positive = True
        else:
            delta = tuple(
                treatment_value - control_value
                for treatment_value, control_value in zip(
                    treatment_values, control_values
                )
            )
            active_effect = statistics.median(
                delta[action.start_index + 1 : action.stop_index + 1]
            )
            recovery_delta = delta[-8:]
            recovery_ratio = statistics.median(
                abs(value) for value in recovery_delta
            ) / max(
                abs(active_effect),
                _required_number(
                    action_config, "recovery_feature_floor"
                ),
            )
            placebo_start = max(
                0, action.start_index - action.duration
            )
            placebo_effect = statistics.median(
                delta[placebo_start:action.start_index]
            )
            signed_placebo = (
                placebo_effect
                if action.effect_direction == "increase"
                else -placebo_effect
            )
            placebo_false_positive = (
                signed_placebo >= action.minimum_effect
            )
        signed_effect = (
            active_effect
            if action.effect_direction == "increase"
            else -active_effect
        )
        results.append(
            {
                "pair_id": pair_id,
                "action_kind": action.action_kind,
                "target_entity": action.target_entity,
                "worker_replicas": (
                    treatment.action_case.worker_replicas
                ),
                "effect_feature": feature,
                "schedule_alignment": schedule_alignment,
                "active_effect": active_effect,
                "minimum_effect": action.minimum_effect,
                "raw_effect_passed": math.isfinite(signed_effect)
                and signed_effect >= action.minimum_effect,
                "recovery_ratio": recovery_ratio,
                "maximum_recovery_ratio": _required_number(
                    action_config, "recovery_ratio_max"
                ),
                "recovery_passed": math.isfinite(recovery_ratio)
                and recovery_ratio
                <= _required_number(
                    action_config, "recovery_ratio_max"
                ),
                "placebo_false_positive": (
                    placebo_false_positive
                ),
            }
        )
    return results


def _action_commands(
    payloads: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, AttributeValue]]:
    commands = []
    for payload in payloads:
        for resource_logs in payload.get("resourceLogs", []):
            for scope_logs in resource_logs.get("scopeLogs", []):
                scope = scope_logs.get("scope", {})
                if scope.get("name") != "quantis.action":
                    continue
                for record in scope_logs.get("logRecords", []):
                    attributes = _parse_attributes(
                        record.get("attributes", [])
                    )
                    if attributes.get("event.name") == "action.command":
                        commands.append(attributes)
    return commands


def _commands_match(
    commands: Sequence[Mapping[str, AttributeValue]],
    manifest: LabActionCaptureManifest,
    assignment: CaptureAssignment,
) -> bool:
    if assignment.role == "control":
        return not commands and not manifest.action_case.actions
    if len(manifest.action_case.actions) != 1 or len(commands) != 2:
        return False
    action = manifest.action_case.actions[0]
    expected = {
        ("start", action.start_index),
        ("stop", action.stop_index),
    }
    observed = set()
    realized_worker_ids: set[Tuple[str, ...]] = set()
    for command in commands:
        raw_magnitude = command.get("quantis.action.magnitude")
        if (
            command.get("quantis.action.id") != action.action_id
            or command.get("quantis.action.kind")
            != action.action_kind
            or command.get("quantis.action.target")
            != action.target_entity
            or command.get("quantis.action.status") != "applied"
            or isinstance(raw_magnitude, bool)
            or not isinstance(raw_magnitude, (int, float))
            or float(raw_magnitude) != action.magnitude
        ):
            return False
        phase = command.get("quantis.action.phase")
        logical_index = command.get(
            "quantis.action.logical_index"
        )
        if (
            not isinstance(phase, str)
            or not _is_integer(logical_index)
        ):
            return False
        observed.add((phase, logical_index))
        if action.action_kind == "worker_pause":
            raw_ids = command.get(
                "quantis.action.realized_worker_ids"
            )
            raw_count = command.get(
                "quantis.action.realized_worker_count"
            )
            if (
                not isinstance(raw_ids, str)
                or not _is_integer(raw_count)
            ):
                return False
            worker_ids = tuple(
                value for value in raw_ids.split(",") if value
            )
            expected_count = min(
                manifest.action_case.worker_replicas,
                max(
                    1,
                    int(
                        action.magnitude
                        * manifest.action_case.worker_replicas
                        + 0.5
                    ),
                ),
            )
            if (
                len(worker_ids) != raw_count
                or raw_count != expected_count
                or len(set(worker_ids)) != len(worker_ids)
            ):
                return False
            realized_worker_ids.add(worker_ids)
    return observed == expected and (
        action.action_kind != "worker_pause"
        or len(realized_worker_ids) == 1
    )


def _cleanup_boundary_matches(
    payloads: Sequence[Mapping[str, Any]],
) -> bool:
    closed = []
    for payload in payloads:
        for resource_logs in payload.get("resourceLogs", []):
            for scope_logs in resource_logs.get("scopeLogs", []):
                if scope_logs.get("scope", {}).get("name") != "quantis.action":
                    continue
                for record in scope_logs.get("logRecords", []):
                    attributes = _parse_attributes(
                        record.get("attributes", [])
                    )
                    if (
                        attributes.get("event.name")
                        == "action.run.boundary"
                        and attributes.get("quantis.run.phase")
                        == "closed"
                    ):
                        closed.append(attributes)
    return len(closed) == 1 and (
        closed[0].get("quantis.run.active_action_count") == 0
        and closed[0].get("quantis.run.cleanup.status") == "clean"
    )


def _trace_spans(
    payloads: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, str]]:
    spans: list[Mapping[str, str]] = []
    for payload in payloads:
        for resource_spans in payload.get("resourceSpans", []):
            for scope_spans in resource_spans.get("scopeSpans", []):
                for raw in scope_spans.get("spans", []):
                    spans.append(
                        {
                            "trace_id": str(raw.get("traceId", "")),
                            "span_id": str(raw.get("spanId", "")),
                            "parent_span_id": str(
                                raw.get("parentSpanId", "")
                            ),
                            "name": str(raw.get("name", "")),
                            "entity_id": str(
                                _parse_attributes(
                                    raw.get("attributes", [])
                                ).get(
                                    "quantis.graph.entity.id", ""
                                )
                            ),
                            "start_unix_nano": str(
                                raw.get("startTimeUnixNano", "")
                            ),
                            "end_unix_nano": str(
                                raw.get("endTimeUnixNano", "")
                            ),
                        }
                    )
    return spans


def _valid_trace_structure(
    spans: Sequence[Mapping[str, str]],
) -> bool:
    if not spans:
        return False
    by_trace: Dict[str, set[str]] = {}
    for span in spans:
        trace_id = span["trace_id"]
        span_id = span["span_id"]
        if (
            not _HEX_TRACE_ID.fullmatch(trace_id)
            or not _HEX_SPAN_ID.fullmatch(span_id)
            or span_id in by_trace.setdefault(trace_id, set())
        ):
            return False
        by_trace[trace_id].add(span_id)
    return all(
        not span["parent_span_id"]
        or span["parent_span_id"] in by_trace[span["trace_id"]]
        for span in spans
    ) and all(
        span["name"] not in _SPAN_ENTITY_OWNERS
        or span["entity_id"]
        == _SPAN_ENTITY_OWNERS[span["name"]]
        for span in spans
    )


def _has_exact_trace_path(
    spans: Sequence[Mapping[str, str]], trace_id: str
) -> bool:
    ordered_names = (
        "api.admission",
        "redis.enqueue",
        "queue.residence",
        "redis.dequeue",
        "worker.processing",
        "postgresql.write",
    )
    selected = [span for span in spans if span["trace_id"] == trace_id]
    by_name = {
        name: [span for span in selected if span["name"] == name]
        for name in ordered_names
    }
    if any(len(values) != 1 for values in by_name.values()):
        return False
    previous_span_id = ""
    previous_start = -1
    for name in ordered_names:
        span = by_name[name][0]
        try:
            start = int(span["start_unix_nano"])
            end = int(span["end_unix_nano"])
        except ValueError:
            return False
        if (
            span["parent_span_id"] != previous_span_id
            or start < previous_start
            or end < start
        ):
            return False
        previous_span_id = span["span_id"]
        previous_start = start
    return True


def _identity_matches(
    payload: Mapping[str, Any],
    case_id: str,
    manifest_sha256: str,
    topology_id: str,
) -> bool:
    resources = list(_resource_attributes(payload))
    if not resources:
        return False
    expected = {
        "quantis.experiment.case.id": case_id,
        "quantis.experiment.manifest.sha256": manifest_sha256,
        "quantis.experiment.topology.id": topology_id,
    }
    return all(
        all(attributes.get(key) == value for key, value in expected.items())
        for attributes in resources
    )


def _metric_build_identity_matches(
    payloads: Sequence[Mapping[str, Any]],
    application_image_id: str,
    application_build_context_sha256: str,
) -> bool:
    expected = {
        "quantis.application.image.id": application_image_id,
        "quantis.application.build_context.sha256": (
            application_build_context_sha256
        ),
    }
    resources = [
        attributes
        for payload in payloads
        for attributes in _resource_attributes(payload)
    ]
    return bool(resources) and all(
        all(attributes.get(key) == value for key, value in expected.items())
        for attributes in resources
    )


def _resource_attributes(
    payload: Mapping[str, Any],
) -> Iterable[Mapping[str, AttributeValue]]:
    for group_name in (
        "resourceMetrics",
        "resourceLogs",
        "resourceSpans",
    ):
        for group in payload.get(group_name, []):
            yield _parse_attributes(
                group.get("resource", {}).get("attributes", [])
            )


def _contains_truth(
    value: Any, forbidden_values: set[str], key: str = ""
) -> bool:
    lowered_key = key.lower()
    if any(
        forbidden in lowered_key
        for forbidden in _FORBIDDEN_OBSERVATION_KEYS
    ):
        return True
    if isinstance(value, str):
        return value in forbidden_values
    if isinstance(value, list):
        return any(
            _contains_truth(item, forbidden_values, key)
            for item in value
        )
    if isinstance(value, dict):
        return any(
            _contains_truth(item, forbidden_values, str(item_key))
            for item_key, item in value.items()
        )
    return False


def _cross_case_trace_references(
    cases: Mapping[str, Mapping[str, Any]],
) -> int:
    owners: Dict[str, int] = {}
    for case in cases.values():
        raw_ids = case.get("trace_ids")
        if not isinstance(raw_ids, tuple):
            return 1
        for trace_id in set(raw_ids):
            if not isinstance(trace_id, str):
                return 1
            owners[trace_id] = owners.get(trace_id, 0) + 1
    return sum(count - 1 for count in owners.values() if count > 1)


def _lane_isolation(
    attestation: Mapping[str, Any], parallel_jobs: int
) -> bool:
    raw_cases = attestation.get("cases", [])
    if not isinstance(raw_cases, list):
        return False
    project_by_lane: Dict[int, set[str]] = {}
    for raw in raw_cases:
        if not isinstance(raw, dict):
            return False
        lane = raw.get("lane")
        project = raw.get("compose_project")
        if (
            not isinstance(lane, int)
            or isinstance(lane, bool)
            or not isinstance(project, str)
        ):
            return False
        lane_number = int(lane)
        project_by_lane.setdefault(lane_number, set()).add(project)
    return (
        set(project_by_lane) == set(range(1, parallel_jobs + 1))
        and all(len(projects) == 1 for projects in project_by_lane.values())
        and len(
            {
                next(iter(projects))
                for projects in project_by_lane.values()
            }
        )
        == parallel_jobs
    )


def _coverage(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _assessment_report(assessment: Mapping[str, Any]) -> str:
    gates = assessment["gates"]
    gate_lines = "\n".join(
        f"- {name}: {'PASS' if passed else 'FAIL'}"
        for name, passed in gates.items()
    )
    coverage = assessment["coverage"]
    pair_lines = "\n".join(
        "- "
        f"{pair['pair_id']} {pair['action_kind']} "
        f"workers-{pair['worker_replicas']}: "
        f"effect={pair['active_effect']:.6g}, "
        f"effect_pass={pair['raw_effect_passed']}, "
        f"recovery_ratio={pair['recovery_ratio']:.6g}, "
        f"recovery_pass={pair['recovery_passed']}"
        for pair in assessment["pairs"]
    )
    return (
        "# Action-dynamics collection assessment\n\n"
        f"Status: **{assessment['status']}**\n\n"
        f"Decision: `{assessment['decision']}`\n\n"
        "This is instrumentation and randomized-action data-quality "
        "evidence only. No model was trained.\n\n"
        "## Gates\n\n"
        f"{gate_lines}\n\n"
        "## Coverage\n\n"
        f"- trace linked: {coverage['trace_link_numerator']}/"
        f"{coverage['trace_link_denominator']} "
        f"({coverage['trace_link']:.3f})\n"
        f"- complete paths: {coverage['complete_trace_numerator']}/"
        f"{coverage['complete_trace_denominator']} "
        f"({coverage['complete_trace']:.3f})\n\n"
        "## Pair counts and attrition\n\n"
        f"- by action: `{assessment['pair_counts_by_action']}`\n"
        f"- by topology: `{assessment['pair_counts_by_topology']}`\n"
        f"- failed pair ids: `{assessment['failed_pair_ids']}`\n"
        f"- attrition: `{assessment['attrition']}`\n\n"
        "## Effect and recovery\n\n"
        f"{pair_lines}\n"
    )


def _opaque_id(seed: int, label: str) -> str:
    digest = hashlib.sha256(f"{seed}:{label}".encode()).digest()
    return str(uuid.UUID(bytes=digest[:16], version=4))


def _derived_int(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _bounded_derived(
    seed: int, label: str, minimum: int, maximum: int
) -> int:
    return minimum + _derived_int(seed, label) % (
        maximum - minimum + 1
    )


def _read_object(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return dict(value)


def _read_json_lines(
    path: Path, *, allow_empty: bool = False
) -> list[Mapping[str, Any]]:
    if not path.exists():
        if allow_empty:
            return []
        raise ValueError(f"JSONL capture is missing: {path}")
    payloads: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text().splitlines(), start=1
    ):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise ValueError(
                f"JSON line {line_number} is not an object: {path}"
            )
        payloads.append(raw)
    if not payloads and not allow_empty:
        raise ValueError(f"JSONL capture is empty: {path}")
    return payloads


def _parse_attributes(
    raw_attributes: Sequence[Mapping[str, Any]],
) -> Mapping[str, AttributeValue]:
    attributes: Dict[str, AttributeValue] = {}
    for raw in raw_attributes:
        key = raw.get("key")
        value = raw.get("value")
        if (
            not isinstance(key, str)
            or key in attributes
            or not isinstance(value, dict)
        ):
            raise ValueError("OTLP attributes are invalid")
        attributes[key] = _parse_any_value(value)
    return attributes


def _parse_any_value(value: Mapping[str, Any]) -> AttributeValue:
    keys = (
        "stringValue",
        "boolValue",
        "intValue",
        "doubleValue",
        "bytesValue",
    )
    present = [key for key in keys if key in value]
    if len(present) != 1:
        raise ValueError("OTLP AnyValue is invalid")
    key = present[0]
    raw = value[key]
    if key == "stringValue":
        return str(raw)
    if key == "boolValue":
        return bool(raw)
    if key == "intValue":
        return int(raw)
    if key == "doubleValue":
        return float(raw)
    return str(raw).encode()


def _required_mapping(
    mapping: Mapping[str, Any], name: str
) -> Mapping[str, Any]:
    value = mapping.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _required_text(mapping: Mapping[str, Any], name: str) -> str:
    value = mapping.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be nonempty text")
    return value


def _required_integer(
    mapping: Mapping[str, Any], name: str
) -> int:
    value = mapping.get(name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value


def _required_number(
    mapping: Mapping[str, Any], name: str
) -> float:
    value = mapping.get(name)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _number_sequence(
    mapping: Mapping[str, Any], name: str
) -> Tuple[float, ...]:
    raw = mapping.get(name)
    if (
        not isinstance(raw, list)
        or not raw
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0.0
            for value in raw
        )
    ):
        raise ValueError(f"{name} must be positive finite numbers")
    return tuple(float(value) for value in raw)


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _json_copy(value: Mapping[str, Any]) -> Dict[str, Any]:
    copied = json.loads(
        json.dumps(value, allow_nan=False, sort_keys=True)
    )
    if not isinstance(copied, dict):
        raise TypeError("expected object")
    return copied


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _pretty_json(value: Any) -> str:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
