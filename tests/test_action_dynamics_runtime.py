import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Optional


REPOSITORY = Path(__file__).resolve().parents[1]
RUNTIME = REPOSITORY / "lab" / "action_dynamics"
if str(RUNTIME) not in sys.path:
    sys.path.insert(0, str(RUNTIME))


def _load(name: str):
    specification = importlib.util.spec_from_file_location(
        name, RUNTIME / f"{name}.py"
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


telemetry = _load("application_telemetry")
interventions = _load("interventions")
capture = _load("run_capture")


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.sets: dict[str, set[str]] = {}
        self.workers = {"worker-a", "worker-b", "worker-c"}

    def set(self, key: str, value: object) -> None:
        self.values[key] = str(value)

    def get(self, key: str) -> Optional[str]:
        return self.values.get(key)

    def delete(self, *keys: str) -> None:
        for key in keys:
            self.values.pop(key, None)
            self.sets.pop(key, None)

    def zrange(self, key: str, start: int, stop: int) -> list[str]:
        del key, start, stop
        return sorted(self.workers)

    def sadd(self, key: str, *values: str) -> None:
        self.sets.setdefault(key, set()).update(values)


class FakeLock:
    def __init__(self) -> None:
        self.statements: list[tuple[str, tuple[int, ...]]] = []
        self.closed = False

    def execute(
        self, statement: str, parameters: tuple[int, ...]
    ) -> None:
        self.statements.append((statement, parameters))

    def close(self) -> None:
        self.closed = True


def _action(
    kind: str, target: str, magnitude: float
) -> Mapping[str, Any]:
    return {
        "action_id": f"action-{kind}",
        "action_kind": kind,
        "target_entity": target,
        "start_index": 3,
        "stop_index": 6,
        "magnitude": magnitude,
        "magnitude_unit": (
            "milliseconds"
            if "delay" in kind
            else "probability"
            if kind == "api_rejection"
            else "worker_fraction"
            if kind == "worker_pause"
            else "boolean"
        ),
    }


def test_observation_telemetry_carries_trace_context_without_truth() -> None:
    posted: list[tuple[str, Mapping[str, Any]]] = []
    client = telemetry.OtlpTelemetryClient(
        logs_endpoint="http://collector:4318/v1/logs",
        traces_endpoint="http://collector:4318/v1/traces",
        post_json=lambda endpoint, payload: posted.append(
            (endpoint, payload)
        ),
    )
    identity = telemetry.ObservationIdentity(
        case_id="opaque-case",
        manifest_sha256="a" * 64,
        topology_id="workers-2",
    )
    root = telemetry.TraceContext.from_traceparent(
        "00-00112233445566778899aabbccddeeff-0123456789abcdef-01"
    )
    child = root.child(span_id="1111111111111111")
    client.emit_logs(
        service_name="quantis-action-api",
        service_instance_id="api-1",
        identity=identity,
        events=(
            telemetry.ApplicationEvent(
                event_name="checkout.accepted",
                body="checkout accepted",
                timestamp_unix_nano=20,
                attributes={
                    "quantis.experiment.origin.window.index": 4
                },
                trace=child,
            ),
        ),
    )
    client.emit_spans(
        service_name="quantis-action-api",
        service_instance_id="api-1",
        identity=identity,
        spans=(
            telemetry.Span(
                name="redis enqueue",
                graph_entity_id="api_enqueues_queue",
                context=child,
                parent_span_id=root.span_id,
                start_unix_nano=10,
                end_unix_nano=20,
                kind=3,
                attributes={
                    "quantis.experiment.origin.window.index": 4
                },
            ),
        ),
    )

    rendered = json.dumps(posted, sort_keys=True)
    assert child.trace_id == root.trace_id
    assert child.to_traceparent() == (
        "00-00112233445566778899aabbccddeeff-"
        "1111111111111111-01"
    )
    assert '"traceId": "00112233445566778899aabbccddeeff"' in rendered
    assert '"spanId": "1111111111111111"' in rendered
    assert "api_enqueues_queue" in rendered
    for forbidden in (
        "action.kind",
        "action.target",
        "action.phase",
        "action.magnitude",
        "fault.kind",
        "matched_pair",
    ):
        assert forbidden not in rendered


def test_intervention_controller_applies_and_reverses_all_actions() -> None:
    redis_client = FakeRedis()
    lock = FakeLock()
    posted: list[tuple[str, Mapping[str, Any]]] = []
    controller = interventions.ReversibleInterventionController(
        redis_client=redis_client,
        database_lock_factory=lambda: lock,
        action_endpoint="http://collector:4319/v1/logs",
        case_id="opaque-case",
        manifest_sha256="b" * 64,
        topology_id="workers-3",
        intervention_seed=19,
        post_json=lambda endpoint, payload: posted.append(
            (endpoint, payload)
        ),
    )
    actions = (
        _action("worker_pause", "worker_pool", 2 / 3),
        _action(
            "postgres_lock", "worker_writes_postgresql", 1.0
        ),
        _action(
            "redis_enqueue_delay", "api_enqueues_queue", 25.0
        ),
        _action(
            "redis_dequeue_delay",
            "queue_dequeues_to_worker",
            25.0,
        ),
        _action("api_rejection", "api", 0.5),
    )
    for action in actions:
        started = controller.command(action, "start", 3)
        stopped = controller.command(action, "stop", 6)
        assert started.status == "applied"
        assert stopped.status == "applied"
    controller.emit_run_boundary("started")
    controller.emit_run_boundary("closed")

    assert len(redis_client.sets.get(
        interventions.PAUSED_WORKERS, set()
    )) == 0
    assert not any(
        key.startswith("quantis:action:") for key in redis_client.values
    )
    assert lock.closed
    assert "pg_advisory_lock" in lock.statements[0][0]
    assert "pg_advisory_unlock" in lock.statements[1][0]
    assert len(posted) == 12
    assert {endpoint for endpoint, _ in posted} == {
        "http://collector:4319/v1/logs"
    }
    action_stream = json.dumps(posted, sort_keys=True)
    assert "quantis.action" in action_stream
    assert "action.command" in action_stream
    assert "action.run.boundary" in action_stream
    assert "action-worker_pause:start" in action_stream
    assert "opaque-case" in action_stream


def test_action_schedule_applies_transition_t_before_state_t_plus_one() -> None:
    action_case = {
        "point_count": 12,
        "actions": [
            _action("api_rejection", "api", 0.5),
        ],
    }

    schedule = capture.scheduled_action_commands(action_case)

    assert [
        (
            item.logical_index,
            item.phase,
            item.affected_state_index,
        )
        for item in schedule
    ] == [(3, "start", 4), (6, "stop", 7)]


def test_runtime_configuration_keeps_capture_streams_isolated() -> None:
    collector = (RUNTIME / "collector.yaml").read_text()
    compose = (RUNTIME / "compose.yaml").read_text()
    dockerfile = (RUNTIME / "Dockerfile").read_text()

    assert "endpoint: 0.0.0.0:4319" in collector
    assert "path: /captures/collector-actions.jsonl" in collector
    assert "path: /captures/collector-traces.jsonl" in collector
    assert "receivers: [otlp/actions]" in collector
    assert "receivers: [otlp]" in collector
    assert "quantis-action-dynamics-app:local" in compose
    assert "4319/v1/logs" in compose
    assert "ports:" not in compose
    assert "fault_matrix/service.py" not in dockerfile
    assert "graph_jepa/service.py" not in dockerfile


def test_build_context_hash_covers_every_executable_image_source() -> None:
    hasher = _load("hash_build_context")
    expected = {
        "action_dynamics/Dockerfile",
        "action_dynamics/application.py",
        "action_dynamics/application_telemetry.py",
        "action_dynamics/interventions.py",
        "action_dynamics/run_capture.py",
        "fault_matrix/requirements.txt",
    }

    assert set(hasher.FILES) == expected
    first = hasher.build_context_sha256()
    second = hasher.build_context_sha256()

    assert first == second
    assert len(first) == 64
    assert set(first) <= set("0123456789abcdef")
    assert first == hashlib.sha256(
        hasher.build_context_bytes()
    ).hexdigest()
