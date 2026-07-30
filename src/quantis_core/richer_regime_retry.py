"""Frozen richer-regime replication contract for local alerting retries."""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from .action_conditioned_dynamics import (
    ACTION_KINDS,
    ActionConditionedCaseManifest,
    InterventionAction,
)


WORKLOAD_FAMILIES = (
    "steady",
    "ramp_or_burst",
    "periodic_or_multiphase",
)
CORPUS_ROLES = ("fit", "selection", "calibration", "evaluation")


@dataclass(frozen=True)
class RicherRegimeRetryProtocol:
    """A strict, versioned campaign design with immutable evidence roles."""

    generator_seed: int
    action_kinds: Tuple[str, ...]
    worker_replica_values: Tuple[int, ...]
    workload_families: Tuple[str, ...]
    replicates_per_cell: int
    role_replicates: Mapping[str, Tuple[int, ...]]
    trajectory: Mapping[str, Any]
    workload: Mapping[str, Any]
    statistical_gate: Mapping[str, Any]
    execution: Mapping[str, Any]
    evidence_boundary: str
    schema_version: int = 1

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "RicherRegimeRetryProtocol":
        """Restore and strictly validate the public protocol document."""

        expected = {
            "schema_version",
            "kind",
            "generator_seed",
            "evidence_boundary",
            "design",
            "trajectory",
            "workload",
            "statistical_gate",
            "execution",
        }
        if (
            set(payload) != expected
            or payload.get("schema_version") != 1
            or payload.get("kind")
            != "richer_regime_retry_protocol"
            or not isinstance(payload.get("generator_seed"), int)
            or isinstance(payload.get("generator_seed"), bool)
            or not isinstance(payload.get("evidence_boundary"), str)
            or not payload["evidence_boundary"]
        ):
            raise ValueError("richer-regime protocol envelope is invalid")
        design = _mapping(payload, "design")
        trajectory = _mapping(payload, "trajectory")
        workload = _mapping(payload, "workload")
        statistical_gate = _mapping(payload, "statistical_gate")
        execution = _mapping(payload, "execution")
        design_keys = {
            "design_kind",
            "action_kinds",
            "worker_replica_values",
            "workload_families",
            "replicates_per_cell",
            "role_replicates",
        }
        if set(design) != design_keys:
            raise ValueError("richer-regime design schema is invalid")
        raw_roles = design["role_replicates"]
        if not isinstance(raw_roles, dict):
            raise ValueError("richer-regime roles are invalid")
        roles = {
            str(role): tuple(_integer_sequence(values))
            for role, values in raw_roles.items()
        }
        protocol = cls(
            generator_seed=int(payload["generator_seed"]),
            action_kinds=tuple(_text_sequence(design["action_kinds"])),
            worker_replica_values=tuple(
                _integer_sequence(design["worker_replica_values"])
            ),
            workload_families=tuple(
                _text_sequence(design["workload_families"])
            ),
            replicates_per_cell=_integer(
                design, "replicates_per_cell"
            ),
            role_replicates=roles,
            trajectory=dict(trajectory),
            workload=dict(workload),
            statistical_gate=dict(statistical_gate),
            execution=dict(execution),
            evidence_boundary=str(payload["evidence_boundary"]),
        )
        protocol._validate()
        return protocol

    def _validate(self) -> None:
        if (
            self.schema_version != 1
            or self.action_kinds != tuple(ACTION_KINDS)
            or self.worker_replica_values != (1, 2, 3)
            or self.workload_families != WORKLOAD_FAMILIES
            or self.replicates_per_cell != 11
            or tuple(self.role_replicates) != CORPUS_ROLES
            or self.role_replicates
            != {
                "fit": (0, 1),
                "selection": (2,),
                "calibration": (3, 4, 5, 6),
                "evaluation": (7, 8, 9, 10),
            }
        ):
            raise ValueError(
                "richer-regime factorial or evidence ownership is invalid"
            )
        assigned = tuple(
            replicate
            for role in CORPUS_ROLES
            for replicate in self.role_replicates[role]
        )
        if sorted(assigned) != list(range(self.replicates_per_cell)):
            raise ValueError("richer-regime replicates are not partitioned")
        if (
            _integer(self.trajectory, "point_count") != 108
            or _number(self.trajectory, "sample_period_seconds") != 0.25
        ):
            raise ValueError("richer-regime trajectory is invalid")
        if set(_mapping(self.workload, "families")) != set(
            WORKLOAD_FAMILIES
        ):
            raise ValueError("richer-regime workload families are invalid")
        if (
            _integer(self.workload, "api_rejection_requests_per_window")
            != 12
            or _integer(self.workload, "drain_phase_start_index") != 84
            or _integer(self.workload, "drain_phase_stop_index") != 92
            or _integer(self.workload, "probe_phase_start_index") != 92
            or _integer(self.workload, "probe_requests_per_window") != 8
            or self.workload.get("twins_share_exact_schedule") is not True
        ):
            raise ValueError("richer-regime action workload is invalid")
        if (
            _number(self.statistical_gate, "confidence") != 0.95
            or _number(
                self.statistical_gate,
                "false_positive_rate_max",
            )
            != 0.05
            or _integer(
                self.statistical_gate,
                "minimum_zero_alarm_controls_per_family",
            )
            != 60
        ):
            raise ValueError("richer-regime statistical gate is invalid")
        if (
            _integer(self.execution, "parallel_jobs") != 6
            or self.execution.get("runtime")
            != "local_docker_compose"
            or self.execution.get("overwrite") is not False
            or self.execution.get("automatic_retry") is not False
            or self.execution.get("pair_atomic") is not True
        ):
            raise ValueError("richer-regime execution contract is invalid")

    def to_dict(self) -> Dict[str, Any]:
        """Return a canonical JSON-compatible protocol representation."""

        return {
            "schema_version": self.schema_version,
            "kind": "richer_regime_retry_protocol",
            "generator_seed": self.generator_seed,
            "evidence_boundary": self.evidence_boundary,
            "design": {
                "design_kind": (
                    "complete_action_topology_workload_factorial"
                ),
                "action_kinds": list(self.action_kinds),
                "worker_replica_values": list(
                    self.worker_replica_values
                ),
                "workload_families": list(self.workload_families),
                "replicates_per_cell": self.replicates_per_cell,
                "role_replicates": {
                    role: list(self.role_replicates[role])
                    for role in CORPUS_ROLES
                },
            },
            "trajectory": dict(self.trajectory),
            "workload": json.loads(
                json.dumps(self.workload, sort_keys=True)
            ),
            "statistical_gate": dict(self.statistical_gate),
            "execution": dict(self.execution),
        }


def build_richer_regime_plan(
    protocol: RicherRegimeRetryProtocol,
) -> Mapping[str, Any]:
    """Materialize every pair, role, seed, and request schedule."""

    protocol._validate()
    pairs = []
    for action_kind in protocol.action_kinds:
        for workers in protocol.worker_replica_values:
            for family in protocol.workload_families:
                for replicate in range(protocol.replicates_per_cell):
                    semantic_key = (
                        f"{action_kind}:{workers}:{family}:{replicate}"
                    )
                    pair_id = str(
                        uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            (
                                "quantis:richer-regime:"
                                f"{protocol.generator_seed}:{semantic_key}"
                            ),
                        )
                    )
                    workload_seed = _derived_integer(
                        protocol.generator_seed,
                        f"{semantic_key}:workload",
                    )
                    schedule = materialize_workload_schedule(
                        protocol,
                        workload_family=family,
                        workload_seed=workload_seed,
                        action_kind=action_kind,
                    )
                    pairs.append(
                        {
                            "pair_id": pair_id,
                            "action_kind": action_kind,
                            "worker_replicas": workers,
                            "workload_family": family,
                            "replicate": replicate,
                            "corpus_role": _role_for_replicate(
                                protocol, replicate
                            ),
                            "workload_seed": workload_seed,
                            "intervention_seed": _derived_integer(
                                protocol.generator_seed,
                                f"{semantic_key}:intervention",
                            ),
                            "request_schedule": list(schedule),
                            "request_schedule_sha256": _sha256(
                                list(schedule)
                            ),
                        }
                    )
    pairs.sort(
        key=lambda pair: _derived_integer(
            protocol.generator_seed,
            f"permutation:{pair['pair_id']}",
        )
    )
    return {
        "schema_version": 1,
        "kind": "richer_regime_retry_plan",
        "protocol_sha256": _sha256(protocol.to_dict()),
        "pair_count": len(pairs),
        "capture_count": len(pairs) * 2,
        "parallel_jobs": _integer(
            protocol.execution, "parallel_jobs"
        ),
        "pairs": pairs,
    }


def validate_richer_regime_plan(
    protocol: RicherRegimeRetryProtocol,
    plan: Mapping[str, Any],
) -> None:
    """Reject any plan that differs from the deterministic generator."""

    if plan != build_richer_regime_plan(protocol):
        raise ValueError(
            "richer-regime plan differs from deterministic generator"
        )


def assess_richer_regime_plan(
    protocol: RicherRegimeRetryProtocol,
    plan: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Assess role isolation and the declared false-positive resolution."""

    validate_richer_regime_plan(protocol, plan)
    raw_pairs = plan["pairs"]
    if not isinstance(raw_pairs, list):
        raise AssertionError("validated plan pairs changed type")
    role_counts = {
        role: sum(pair["corpus_role"] == role for pair in raw_pairs)
        for role in CORPUS_ROLES
    }
    evaluation_by_family = _role_family_counts(
        raw_pairs, "evaluation"
    )
    calibration_by_family = _role_family_counts(
        raw_pairs, "calibration"
    )
    minimum = _integer(
        protocol.statistical_gate,
        "minimum_zero_alarm_controls_per_family",
    )
    confidence = _number(
        protocol.statistical_gate, "confidence"
    )
    false_positive_rate_max = _number(
        protocol.statistical_gate, "false_positive_rate_max"
    )
    expected_cells = (
        len(protocol.action_kinds)
        * len(protocol.worker_replica_values)
        * len(protocol.workload_families)
        * protocol.replicates_per_cell
    )
    gates = {
        "complete_factorial": len(raw_pairs) == expected_cells,
        "deterministic_role_ownership": role_counts
        == {
            "fit": 90,
            "selection": 45,
            "calibration": 180,
            "evaluation": 180,
        },
        "evaluation_family_false_positive_resolution": all(
            count >= minimum
            and zero_event_upper_bound(
                count, confidence=confidence
            )
            < false_positive_rate_max
            for count in evaluation_by_family.values()
        ),
        "calibration_family_false_positive_resolution": all(
            count >= minimum for count in calibration_by_family.values()
        ),
        "pair_identity_unique": len(
            {pair["pair_id"] for pair in raw_pairs}
        )
        == len(raw_pairs),
        "workload_family_coverage": set(evaluation_by_family)
        == set(protocol.workload_families),
    }
    return {
        "schema_version": 1,
        "kind": "richer_regime_retry_plan_assessment",
        "status": "qualified" if all(gates.values()) else "failed",
        "pair_counts_by_role": role_counts,
        "evaluation_controls_by_workload_family": (
            evaluation_by_family
        ),
        "calibration_controls_by_workload_family": (
            calibration_by_family
        ),
        "zero_alarm_upper_bounds_by_workload_family": {
            family: zero_event_upper_bound(
                count, confidence=confidence
            )
            for family, count in evaluation_by_family.items()
        },
        "gates": gates,
    }


def prepare_richer_regime_shard(
    protocol: RicherRegimeRetryProtocol,
    plan: Mapping[str, Any],
    *,
    corpus_role: str,
    workload_family: str,
    action_library: Mapping[str, Any],
    image_digests: Mapping[str, str],
    observation_schema_sha256: str,
    application_build_context_sha256: str,
) -> Mapping[str, Any]:
    """Compile one immutable campaign shard for the Compose collector."""

    validate_richer_regime_plan(protocol, plan)
    if (
        corpus_role not in CORPUS_ROLES
        or workload_family not in WORKLOAD_FAMILIES
        or set(action_library) != set(protocol.action_kinds)
        or not _is_sha256(observation_schema_sha256)
        or not _is_sha256(application_build_context_sha256)
        or not image_digests
        or any(
            not isinstance(name, str)
            or not name
            or not isinstance(digest, str)
            or "sha256:" not in digest
            for name, digest in image_digests.items()
        )
    ):
        raise ValueError("richer-regime shard inputs are invalid")
    selected = [
        dict(pair)
        for pair in plan["pairs"]
        if pair["corpus_role"] == corpus_role
        and pair["workload_family"] == workload_family
    ]
    if not selected:
        raise ValueError("richer-regime shard is empty")
    executor_protocol = {
        "schema_version": 1,
        "kind": "richer_regime_retry_executor_protocol",
        "campaign_protocol_sha256": _sha256(protocol.to_dict()),
        "campaign_plan_sha256": _sha256(plan),
        "corpus_role": corpus_role,
        "workload_family": workload_family,
        "application_build_context_sha256": (
            application_build_context_sha256
        ),
        "collection": {
            "pair_count": len(selected),
            "expected_capture_count": len(selected) * 2,
            "parallel_jobs": _integer(
                protocol.execution, "parallel_jobs"
            ),
            "overwrite": False,
        },
    }
    assignments = []
    manifest_specs = []
    for ordinal, pair in enumerate(selected):
        pair_id = str(pair["pair_id"])
        workers = int(pair["worker_replicas"])
        for twin_role in ("treatment", "control"):
            case_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"quantis:richer-regime:{pair_id}:{twin_role}",
                )
            )
            assignments.append(
                {
                    "pair_id": pair_id,
                    "case_id": case_id,
                    "role": twin_role,
                    "lane": ordinal
                    % _integer(protocol.execution, "parallel_jobs")
                    + 1,
                    "batch": ordinal
                    // _integer(protocol.execution, "parallel_jobs")
                    + 1,
                    "order_in_pair": (
                        (0 if ordinal % 2 == 0 else 1)
                        if twin_role == "treatment"
                        else (1 if ordinal % 2 == 0 else 0)
                    ),
                    "worker_replicas": workers,
                }
            )
            manifest_specs.append((pair, twin_role, case_id))
    executor_plan = {
        "schema_version": 1,
        "kind": "richer_regime_retry_execution_plan",
        "protocol_sha256": _sha256(executor_protocol),
        "campaign_protocol_sha256": _sha256(protocol.to_dict()),
        "campaign_plan_sha256": _sha256(plan),
        "application_build_context_sha256": (
            application_build_context_sha256
        ),
        "corpus_role": corpus_role,
        "workload_family": workload_family,
        "pairs": selected,
        "assignments": assignments,
    }
    executor_plan_sha256 = _sha256(executor_plan)
    manifests = {}
    for pair, twin_role, case_id in manifest_specs:
        action = _intervention_action(
            protocol,
            pair,
            action_library,
        )
        action_case = ActionConditionedCaseManifest(
            case_id=case_id,
            matched_pair_id=str(pair["pair_id"]),
            split=_action_split(corpus_role),
            point_count=_integer(protocol.trajectory, "point_count"),
            logical_window_period_nano=int(
                _number(
                    protocol.trajectory, "sample_period_seconds"
                )
                * 1_000_000_000
            ),
            topology_id=f"workers-{pair['worker_replicas']}",
            worker_replicas=int(pair["worker_replicas"]),
            workload_seed=int(pair["workload_seed"]),
            intervention_seed=int(pair["intervention_seed"]),
            actions=(action,) if twin_role == "treatment" else (),
        )
        manifests[case_id] = {
            "schema_version": 1,
            "kind": "lab_action_capture_manifest",
            "action_case": action_case.to_dict(),
            "sample_period_seconds": _number(
                protocol.trajectory, "sample_period_seconds"
            ),
            "request_schedule": list(pair["request_schedule"]),
            "api_request_queue_size": 128,
            "image_digests": dict(sorted(image_digests.items())),
            "observation_schema_sha256": observation_schema_sha256,
            "protocol_sha256": _sha256(executor_protocol),
            "prepared_plan_sha256": executor_plan_sha256,
            "graph_observation_schema_sha256": (
                observation_schema_sha256
            ),
            "corpus_role": "development",
            "retry_corpus_role": corpus_role,
            "workload_family": workload_family,
            "campaign_protocol_sha256": _sha256(protocol.to_dict()),
            "campaign_plan_sha256": _sha256(plan),
        }
    return {
        "protocol": executor_protocol,
        "plan": executor_plan,
        "manifests": manifests,
        "summary": {
            "schema_version": 1,
            "kind": "prepared_richer_regime_retry_shard",
            "corpus_role": corpus_role,
            "workload_family": workload_family,
            "pair_count": len(selected),
            "capture_count": len(selected) * 2,
            "protocol_sha256": _sha256(executor_protocol),
            "plan_sha256": executor_plan_sha256,
        },
    }


def materialize_workload_schedule(
    protocol: RicherRegimeRetryProtocol,
    *,
    workload_family: str,
    workload_seed: int,
    action_kind: str,
) -> Tuple[int, ...]:
    """Build one explicit schedule shared byte-for-byte by pair twins."""

    if (
        workload_family not in protocol.workload_families
        or action_kind not in protocol.action_kinds
        or not isinstance(workload_seed, int)
        or isinstance(workload_seed, bool)
    ):
        raise ValueError("richer-regime workload request is invalid")
    point_count = _integer(protocol.trajectory, "point_count")
    family_config = _mapping(
        _mapping(protocol.workload, "families"),
        workload_family,
    )
    low = _integer(family_config, "minimum_requests_per_window")
    high = _integer(family_config, "maximum_requests_per_window")
    if workload_family == "steady":
        schedule = [
            _bounded(workload_seed, f"steady:{index}", low, high)
            for index in range(point_count)
        ]
    elif workload_family == "ramp_or_burst":
        burst_start = _integer(family_config, "burst_start_index")
        burst_stop = _integer(family_config, "burst_stop_index")
        burst_extra = _integer(family_config, "burst_extra_requests")
        schedule = []
        for index in range(point_count):
            ramp_extra = (
                (high - low) * index // max(point_count - 1, 1)
            )
            value = _bounded(
                workload_seed,
                f"ramp:{index}",
                low,
                low + max(ramp_extra, 0),
            )
            if burst_start <= index < burst_stop:
                value += burst_extra
            schedule.append(value)
    else:
        phase_windows = _integer(
            family_config, "phase_window_count"
        )
        high_phase_extra = _integer(
            family_config, "high_phase_extra_requests"
        )
        schedule = [
            _bounded(
                workload_seed,
                f"periodic:{index}",
                low,
                high,
            )
            + (
                high_phase_extra
                if (index // phase_windows) % 2 == 1
                else 0
            )
            for index in range(point_count)
        ]
    if action_kind == "api_rejection":
        fixed = _integer(
            protocol.workload,
            "api_rejection_requests_per_window",
        )
        schedule = [fixed] * point_count
    if action_kind == "redis_enqueue_delay":
        drain_start = _integer(
            protocol.workload, "drain_phase_start_index"
        )
        drain_stop = _integer(
            protocol.workload, "drain_phase_stop_index"
        )
        probe_start = _integer(
            protocol.workload, "probe_phase_start_index"
        )
        probe_requests = _integer(
            protocol.workload, "probe_requests_per_window"
        )
        schedule[drain_start:drain_stop] = [0] * (
            drain_stop - drain_start
        )
        schedule[probe_start : point_count - 1] = [
            probe_requests
        ] * (point_count - 1 - probe_start)
    return tuple(schedule)


def zero_event_upper_bound(
    sample_count: int, *, confidence: float
) -> float:
    """Return the exact one-sided binomial upper bound for zero events."""

    if (
        not isinstance(sample_count, int)
        or isinstance(sample_count, bool)
        or sample_count < 1
        or not math.isfinite(confidence)
        or not 0.0 < confidence < 1.0
    ):
        raise ValueError("zero-event bound inputs are invalid")
    return float(
        1.0 - (1.0 - confidence) ** (1.0 / sample_count)
    )


def _role_for_replicate(
    protocol: RicherRegimeRetryProtocol, replicate: int
) -> str:
    for role in CORPUS_ROLES:
        if replicate in protocol.role_replicates[role]:
            return role
    raise ValueError("replicate has no evidence owner")


def _action_split(corpus_role: str) -> str:
    return {
        "fit": "training",
        "selection": "validation",
        "calibration": "validation",
        "evaluation": "confirmation",
    }[corpus_role]


def _intervention_action(
    protocol: RicherRegimeRetryProtocol,
    pair: Mapping[str, Any],
    action_library: Mapping[str, Any],
) -> InterventionAction:
    action_kind = str(pair["action_kind"])
    config = _mapping(action_library, action_kind)
    severities = config.get("severity_values")
    if (
        not isinstance(severities, list)
        or not severities
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            for value in severities
        )
    ):
        raise ValueError("richer-regime action severities are invalid")
    replicate = int(pair["replicate"])
    workers = int(pair["worker_replicas"])
    intervention_seed = int(pair["intervention_seed"])
    start_index = 28 + intervention_seed % 12
    duration = (
        int(config["fixed_duration"])
        if "fixed_duration" in config
        else 8 + (intervention_seed // 17) % 13
    )
    return InterventionAction(
        action_id=str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"quantis:richer-regime:{pair['pair_id']}:action",
            )
        ),
        action_kind=action_kind,
        target_entity=str(config["target_entity"]),
        start_index=start_index,
        stop_index=start_index + duration,
        magnitude=float(
            severities[
                ((workers - 1) + replicate) % len(severities)
            ]
        ),
        magnitude_unit=str(config["magnitude_unit"]),
        effect_feature=str(config["effect_feature"]),
        effect_direction=str(config["effect_direction"]),
        minimum_effect=float(config["minimum_effect"]),
        recovery_tolerance=float(config["recovery_ratio_max"]),
    )


def _role_family_counts(
    pairs: Sequence[Mapping[str, Any]], role: str
) -> Dict[str, int]:
    return {
        family: sum(
            pair["corpus_role"] == role
            and pair["workload_family"] == family
            for pair in pairs
        )
        for family in WORKLOAD_FAMILIES
    }


def _derived_integer(seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _bounded(seed: int, key: str, low: int, high: int) -> int:
    if low < 0 or high < low:
        raise ValueError("workload request bounds are invalid")
    return low + _derived_integer(seed, key) % (high - low + 1)


def _sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _mapping(
    payload: Mapping[str, Any], key: str
) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    return value


def _integer(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


def _number(payload: Mapping[str, Any], key: str) -> float:
    value = payload.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{key} must be finite")
    return float(value)


def _text_sequence(value: Any) -> Sequence[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ValueError("expected a non-empty text sequence")
    return value


def _integer_sequence(value: Any) -> Sequence[int]:
    if (
        not isinstance(value, list)
        or not value
        or any(
            not isinstance(item, int) or isinstance(item, bool)
            for item in value
        )
    ):
        raise ValueError("expected a non-empty integer sequence")
    return value
