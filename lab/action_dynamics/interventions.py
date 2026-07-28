"""Reversible external commands for the action-dynamics lab."""

import hashlib
import json
import math
import time
import urllib.request
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    Optional,
    Protocol,
    Sequence,
)


PAUSED_WORKERS = "quantis:action:paused_workers"
ENQUEUE_DELAY_MS = "quantis:action:enqueue_delay_ms"
DEQUEUE_DELAY_MS = "quantis:action:dequeue_delay_ms"
API_REJECTION = "quantis:action:api_rejection"
WORKER_INSTANCES = "quantis:worker:instances"
DATABASE_ADVISORY_LOCK = 424242
ACTION_TARGETS = {
    "worker_pause": "worker_pool",
    "postgres_lock": "worker_writes_postgresql",
    "redis_enqueue_delay": "api_enqueues_queue",
    "redis_dequeue_delay": "queue_dequeues_to_worker",
    "api_rejection": "api",
}


class RedisControlClient(Protocol):
    """Redis operations used by intervention control."""

    def set(self, key: str, value: object) -> object:
        ...

    def delete(self, *keys: str) -> object:
        ...

    def get(self, key: str) -> Optional[str]:
        ...

    def zrange(
        self, key: str, start: int, stop: int
    ) -> Sequence[str]:
        ...

    def sadd(self, key: str, *values: str) -> object:
        ...


class DatabaseLock(Protocol):
    """Connection that can own the PostgreSQL advisory lock."""

    def execute(
        self, statement: str, parameters: tuple[int, ...]
    ) -> object:
        ...

    def close(self) -> None:
        ...


JsonPoster = Callable[[str, Mapping[str, Any]], None]
DatabaseLockFactory = Callable[[], DatabaseLock]


@dataclass(frozen=True)
class ActionCommandEvidence:
    """One successfully applied external command."""

    command_id: str
    action_id: str
    action_kind: str
    target_entity: str
    phase: str
    magnitude: float
    magnitude_unit: str
    logical_index: int
    affected_state_index: int
    applied_unix_nano: int
    status: str
    realized_worker_ids: tuple[str, ...] = ()

    def to_dict(self) -> Mapping[str, Any]:
        """Return stable command evidence."""

        return {
            "command_id": self.command_id,
            "action_id": self.action_id,
            "action_kind": self.action_kind,
            "target_entity": self.target_entity,
            "phase": self.phase,
            "magnitude": self.magnitude,
            "magnitude_unit": self.magnitude_unit,
            "logical_index": self.logical_index,
            "affected_state_index": self.affected_state_index,
            "applied_unix_nano": self.applied_unix_nano,
            "status": self.status,
            "realized_worker_ids": list(
                self.realized_worker_ids
            ),
        }


@dataclass
class _ActiveAction:
    action: Mapping[str, Any]
    lock: Optional[DatabaseLock] = None
    realized_worker_ids: tuple[str, ...] = ()


class ReversibleInterventionController:
    """Apply the five declared actions and guarantee idempotent cleanup."""

    def __init__(
        self,
        *,
        redis_client: RedisControlClient,
        database_lock_factory: DatabaseLockFactory,
        action_endpoint: str,
        case_id: str,
        manifest_sha256: str,
        topology_id: str,
        intervention_seed: int,
        post_json: Optional[JsonPoster] = None,
    ) -> None:
        if (
            not action_endpoint
            or not case_id
            or not topology_id
            or len(manifest_sha256) != 64
            or isinstance(intervention_seed, bool)
        ):
            raise ValueError("intervention controller identity is invalid")
        self.redis_client = redis_client
        self.database_lock_factory = database_lock_factory
        self.action_endpoint = action_endpoint
        self.case_id = case_id
        self.manifest_sha256 = manifest_sha256
        self.topology_id = topology_id
        self.intervention_seed = intervention_seed
        self._post_json = (
            _post_json if post_json is None else post_json
        )
        self._active: Dict[str, _ActiveAction] = {}
        self._command_ids: set[str] = set()

    def command(
        self,
        action: Mapping[str, Any],
        phase: str,
        logical_index: int,
    ) -> ActionCommandEvidence:
        """Apply one start or stop and emit evidence after success."""

        normalized = _validate_action(action)
        if phase not in {"start", "stop"}:
            raise ValueError("command phase must be start or stop")
        if (
            isinstance(logical_index, bool)
            or logical_index < 0
        ):
            raise ValueError("command logical index must be nonnegative")
        action_id = str(normalized["action_id"])
        command_id = f"{action_id}:{phase}"
        if command_id in self._command_ids:
            raise ValueError(f"duplicate action command: {command_id}")
        if phase == "start":
            if action_id in self._active:
                raise ValueError(f"action is already active: {action_id}")
            started_action = self._start(normalized)
            self._active[action_id] = started_action
            realized_worker_ids = (
                started_action.realized_worker_ids
            )
        else:
            active_action = self._active.get(action_id)
            if active_action is None:
                raise ValueError(f"action is not active: {action_id}")
            realized_worker_ids = (
                active_action.realized_worker_ids
            )
            self._stop(active_action)
            del self._active[action_id]
        evidence = _evidence(
            normalized,
            phase,
            logical_index,
            status="applied",
            realized_worker_ids=realized_worker_ids,
        )
        self._emit(evidence)
        self._command_ids.add(command_id)
        return evidence

    def close(self) -> tuple[ActionCommandEvidence, ...]:
        """Best-effort stop every still-active action exactly once."""

        evidence = []
        for action_id in sorted(tuple(self._active)):
            active = self._active.pop(action_id)
            self._stop(active)
            command_id = f"{action_id}:stop"
            if command_id in self._command_ids:
                continue
            logical_index = int(active.action["stop_index"])
            item = _evidence(
                active.action,
                "stop",
                logical_index,
                status="cleanup",
                realized_worker_ids=(
                    active.realized_worker_ids
                ),
            )
            self._emit(item)
            self._command_ids.add(command_id)
            evidence.append(item)
        self.redis_client.delete(
            PAUSED_WORKERS,
            ENQUEUE_DELAY_MS,
            DEQUEUE_DELAY_MS,
            API_REJECTION,
        )
        return tuple(evidence)

    def emit_run_boundary(self, phase: str) -> None:
        """Ensure every capture, including controls, owns an action file."""

        if phase not in {"started", "closed"}:
            raise ValueError("run boundary phase is invalid")
        timestamp = time.time_ns()
        self._post_record(
            timestamp,
            {
                "event.name": "action.run.boundary",
                "quantis.run.phase": phase,
                "quantis.run.active_action_count": len(
                    self._active
                ),
                "quantis.run.cleanup.status": (
                    "clean" if phase == "closed" else "pending"
                ),
                "quantis.run.redis_enqueue_delay_ms": (
                    self._control_value(ENQUEUE_DELAY_MS)
                ),
            },
            body="action run boundary",
        )

    def _start(
        self, action: Mapping[str, Any]
    ) -> _ActiveAction:
        kind = str(action["action_kind"])
        magnitude = float(action["magnitude"])
        if kind == "worker_pause":
            workers = tuple(
                str(worker)
                for worker in self.redis_client.zrange(
                    WORKER_INSTANCES, 0, -1
                )
            )
            if not workers:
                raise RuntimeError("worker pause found no live workers")
            count = min(
                len(workers),
                max(
                    1,
                    int(magnitude * len(workers) + 0.5),
                ),
            )
            selected = sorted(
                workers,
                key=lambda worker: hashlib.sha256(
                    (
                        f"{self.intervention_seed}:"
                        f"{action['action_id']}:{worker}"
                    ).encode("utf-8")
                ).digest(),
            )[:count]
            self.redis_client.delete(PAUSED_WORKERS)
            self.redis_client.sadd(PAUSED_WORKERS, *selected)
            return _ActiveAction(
                action,
                realized_worker_ids=tuple(selected),
            )
        if kind == "postgres_lock":
            lock = self.database_lock_factory()
            lock.execute(
                "SELECT pg_advisory_lock(%s)",
                (DATABASE_ADVISORY_LOCK,),
            )
            return _ActiveAction(action, lock)
        if kind == "redis_enqueue_delay":
            self.redis_client.set(ENQUEUE_DELAY_MS, repr(magnitude))
            return _ActiveAction(action)
        if kind == "redis_dequeue_delay":
            self.redis_client.set(DEQUEUE_DELAY_MS, repr(magnitude))
            return _ActiveAction(action)
        if kind == "api_rejection":
            self.redis_client.set(
                API_REJECTION,
                json.dumps(
                    {
                        "probability": magnitude,
                        "seed": self.intervention_seed,
                    },
                    separators=(",", ":"),
                ),
            )
            return _ActiveAction(action)
        raise AssertionError("validated action kind is unreachable")

    def _stop(self, active: _ActiveAction) -> None:
        kind = str(active.action["action_kind"])
        if kind == "postgres_lock":
            if active.lock is None:
                raise RuntimeError("postgres action lost lock owner")
            try:
                active.lock.execute(
                    "SELECT pg_advisory_unlock(%s)",
                    (DATABASE_ADVISORY_LOCK,),
                )
            finally:
                active.lock.close()
            return
        key = {
            "worker_pause": PAUSED_WORKERS,
            "redis_enqueue_delay": ENQUEUE_DELAY_MS,
            "redis_dequeue_delay": DEQUEUE_DELAY_MS,
            "api_rejection": API_REJECTION,
        }[kind]
        self.redis_client.delete(key)

    def _emit(self, evidence: ActionCommandEvidence) -> None:
        attributes = {
            "event.name": "action.command",
            "quantis.action.id": evidence.action_id,
            "quantis.action.phase": evidence.phase,
            "quantis.action.kind": evidence.action_kind,
            "quantis.action.target": evidence.target_entity,
            "quantis.action.magnitude": evidence.magnitude,
            "quantis.action.magnitude_unit": (
                evidence.magnitude_unit
            ),
            "quantis.action.logical_index": evidence.logical_index,
            "quantis.action.affected_state_index": (
                evidence.affected_state_index
            ),
            "quantis.action.command_id": evidence.command_id,
            "quantis.action.status": evidence.status,
            "quantis.action.realized_worker_count": len(
                evidence.realized_worker_ids
            ),
            "quantis.action.realized_worker_ids": ",".join(
                evidence.realized_worker_ids
            ),
            "quantis.controller.redis_enqueue_delay_ms": (
                self._control_value(ENQUEUE_DELAY_MS)
            ),
        }
        self._post_record(
            evidence.applied_unix_nano,
            attributes,
            body="action command",
        )

    def _control_value(self, key: str) -> float:
        raw = self.redis_client.get(key)
        if raw is None:
            return 0.0
        value = float(raw)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("controller key readback is invalid")
        return value

    def _post_record(
        self,
        timestamp_unix_nano: int,
        attributes: Mapping[str, Any],
        *,
        body: str,
    ) -> None:
        self._post_json(
            self.action_endpoint,
            {
                "resourceLogs": [
                    {
                        "resource": {
                            "attributes": _attributes(
                                {
                                    "service.name": (
                                        "quantis-action-runner"
                                    ),
                                    "quantis.experiment.case.id": (
                                        self.case_id
                                    ),
                                    "quantis.experiment.manifest.sha256": (
                                        self.manifest_sha256
                                    ),
                                    "quantis.experiment.topology.id": (
                                        self.topology_id
                                    ),
                                }
                            )
                        },
                        "scopeLogs": [
                            {
                                "scope": {
                                    "name": "quantis.action",
                                    "version": "1.0.0",
                                },
                                "logRecords": [
                                    {
                                        "timeUnixNano": str(
                                            timestamp_unix_nano
                                        ),
                                        "observedTimeUnixNano": str(
                                            timestamp_unix_nano
                                        ),
                                        "severityNumber": 9,
                                        "severityText": "INFO",
                                        "body": {
                                            "stringValue": body
                                        },
                                        "attributes": _attributes(
                                            attributes
                                        ),
                                    }
                                ],
                            }
                        ],
                    }
                ]
            },
        )


def _validate_action(
    action: Mapping[str, Any],
) -> Mapping[str, Any]:
    required = {
        "action_id",
        "action_kind",
        "target_entity",
        "start_index",
        "stop_index",
        "magnitude",
        "magnitude_unit",
    }
    if not required <= set(action):
        raise ValueError("runtime action is missing required fields")
    kind = action["action_kind"]
    target = action["target_entity"]
    magnitude = action["magnitude"]
    if (
        not isinstance(action["action_id"], str)
        or not action["action_id"]
        or not isinstance(kind, str)
        or kind not in ACTION_TARGETS
        or target != ACTION_TARGETS[kind]
        or isinstance(magnitude, bool)
        or not isinstance(magnitude, (int, float))
        or not math.isfinite(float(magnitude))
        or float(magnitude) <= 0.0
        or not isinstance(action["magnitude_unit"], str)
        or not action["magnitude_unit"]
    ):
        raise ValueError("runtime action fields are invalid")
    if (
        kind in {"worker_pause", "api_rejection"}
        and float(magnitude) > 1.0
    ):
        raise ValueError(f"{kind} magnitude cannot exceed one")
    if kind == "postgres_lock" and float(magnitude) != 1.0:
        raise ValueError("postgres lock magnitude must be one")
    return action


def _evidence(
    action: Mapping[str, Any],
    phase: str,
    logical_index: int,
    *,
    status: str,
    realized_worker_ids: tuple[str, ...] = (),
) -> ActionCommandEvidence:
    return ActionCommandEvidence(
        command_id=f"{action['action_id']}:{phase}",
        action_id=str(action["action_id"]),
        action_kind=str(action["action_kind"]),
        target_entity=str(action["target_entity"]),
        phase=phase,
        magnitude=float(action["magnitude"]),
        magnitude_unit=str(action["magnitude_unit"]),
        logical_index=logical_index,
        affected_state_index=logical_index + 1,
        applied_unix_nano=time.time_ns(),
        status=status,
        realized_worker_ids=realized_worker_ids,
    )


def _attributes(
    values: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    rendered: list[Mapping[str, Any]] = []
    for key, value in sorted(values.items()):
        if isinstance(value, bool):
            encoded: Mapping[str, Any] = {"boolValue": value}
        elif isinstance(value, int):
            encoded = {"intValue": str(value)}
        elif isinstance(value, float):
            encoded = {"doubleValue": value}
        else:
            encoded = {"stringValue": str(value)}
        rendered.append({"key": key, "value": encoded})
    return rendered


def _post_json(endpoint: str, payload: Mapping[str, Any]) -> None:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(
            payload, separators=(",", ":"), allow_nan=False
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=3) as response:
        if response.status != 200:
            raise RuntimeError(
                f"collector returned HTTP {response.status}"
            )
