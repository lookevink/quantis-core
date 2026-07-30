import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional

import pytest

from quantis_core.otlp import read_otlp_capture


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
collection = _load("collect_pilot")


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
    assert "quantis.run.redis_enqueue_delay_ms" in action_stream
    assert "action-worker_pause:start" in action_stream
    assert "opaque-case" in action_stream


def test_collection_places_a_barrier_between_twin_waves(
    tmp_path: Path, monkeypatch: Any
) -> None:
    protocol_path = tmp_path / "protocol.json"
    plan_path = tmp_path / "plan.json"
    manifests = tmp_path / "manifests"
    captures = tmp_path / "captures"
    attestation_path = tmp_path / "attestation.json"
    manifests.mkdir()
    assignments = [
        {
            "pair_id": f"pair-{lane}",
            "case_id": f"case-{lane}-{order}",
            "role": "treatment" if order == 0 else "control",
            "lane": lane,
            "batch": 1,
            "order_in_pair": order,
            "worker_replicas": 3,
        }
        for lane in (1, 2)
        for order in (0, 1)
    ]
    protocol_path.write_text(
        json.dumps({"collection": {"parallel_jobs": 6}})
    )
    plan_path.write_text(json.dumps({"assignments": assignments}))
    for assignment in assignments:
        (manifests / f"{assignment['case_id']}.json").write_text(
            "{}"
        )
    completed_first_wave: set[int] = set()

    def fake_collect_case(**kwargs: Any) -> Mapping[str, Any]:
        assignment = kwargs["assignment"]
        order = int(assignment["order_in_pair"])
        lane = int(assignment["lane"])
        started = time.time_ns()
        if order == 0 and lane == 2:
            time.sleep(0.05)
        if order == 1:
            assert completed_first_wave == {1, 2}
        else:
            completed_first_wave.add(lane)
        return {
            **dict(assignment),
            "compose_project": f"test-lane-{lane}",
            "manifest_sha256": "a" * 64,
            "started_unix_nano": started,
            "completed_unix_nano": time.time_ns(),
        }

    monkeypatch.setattr(
        collection, "_collect_case", fake_collect_case
    )
    monkeypatch.setattr(
        collection,
        "_clean_all_lanes",
        lambda *args, **kwargs: None,
    )

    attestation = collection.collect_action_cases(
        protocol_path=protocol_path,
        plan_path=plan_path,
        manifests_directory=manifests,
        captures_directory=captures,
        compose_file=tmp_path / "compose.yaml",
        project_prefix="test",
        application_image_id="sha256:" + "a" * 64,
        application_build_context_sha256="b" * 64,
        parallel_jobs=6,
        attestation_path=attestation_path,
    )

    first = [
        case
        for case in attestation["cases"]
        if case["order_in_pair"] == 0
    ]
    second = [
        case
        for case in attestation["cases"]
        if case["order_in_pair"] == 1
    ]
    assert max(case["completed_unix_nano"] for case in first) <= min(
        case["started_unix_nano"] for case in second
    )


def test_failed_runner_output_is_preserved(
    tmp_path: Path, monkeypatch: Any
) -> None:
    protocol_path = tmp_path / "protocol.json"
    plan_path = tmp_path / "plan.json"
    manifests = tmp_path / "manifests"
    captures = tmp_path / "captures"
    manifests.mkdir()
    protocol_path.write_text(
        json.dumps({"collection": {"parallel_jobs": 6}})
    )
    assignments = [
        {
            "pair_id": "pair",
            "case_id": f"case-{role}",
            "role": role,
            "lane": 1,
            "batch": 1,
            "order_in_pair": order,
            "worker_replicas": 1,
        }
        for order, role in enumerate(("treatment", "control"))
    ]
    plan_path.write_text(json.dumps({"assignments": assignments}))
    for assignment in assignments:
        (manifests / f"{assignment['case_id']}.json").write_text(
            "{}"
        )

    def fake_run(
        command: list[str],
        environment: Mapping[str, str],
        *,
        check: bool = True,
        capture: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        del environment, check
        if capture:
            raise subprocess.CalledProcessError(
                1,
                command,
                output="preserved runner failure\n",
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(collection, "_run", fake_run)
    with pytest.raises(subprocess.CalledProcessError):
        collection.collect_action_cases(
            protocol_path=protocol_path,
            plan_path=plan_path,
            manifests_directory=manifests,
            captures_directory=captures,
            compose_file=tmp_path / "compose.yaml",
            project_prefix="test",
            application_image_id="sha256:" + "a" * 64,
            application_build_context_sha256="b" * 64,
            parallel_jobs=6,
            attestation_path=tmp_path / "attestation.json",
        )

    assert (
        captures / "case-treatment" / "runner.log"
    ).read_text() == "preserved runner failure\n"


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


def test_independently_worked_otlp_fixture_separates_truth_stream() -> None:
    fixture = (
        REPOSITORY
        / "tests"
        / "fixtures"
        / "otlp"
        / "action-dynamics-telemetry.jsonl"
    )
    payloads = [
        json.loads(line) for line in fixture.read_text().splitlines()
    ]

    observations = json.dumps(payloads[:2], sort_keys=True)
    conditioning = json.dumps(payloads[2], sort_keys=True)
    assert "00112233445566778899aabbccddeeff" in observations
    assert "quantis.graph.entity.id" in observations
    assert "quantis.action.kind" not in observations
    assert "quantis.action.kind" in conditioning
    assert "worker_pause" in conditioning


def test_independently_worked_count_fixture_preserves_exact_integers() -> None:
    fixture = (
        REPOSITORY
        / "tests"
        / "fixtures"
        / "otlp"
        / "action-dynamics-count-metrics.jsonl"
    )

    capture_result = read_otlp_capture(fixture)
    values = {
        point.metric_name: point.number_value
        for point in capture_result.points
    }

    assert values == {
        "quantis.experiment.request_count": 12.0,
        "quantis.experiment.error_count": 3.0,
    }
