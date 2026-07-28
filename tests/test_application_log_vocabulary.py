import importlib.util
import hashlib
import json
from pathlib import Path
from types import ModuleType

from quantis_core.otlp_logs import read_otlp_log_capture


def _application_logging() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[1]
        / "lab"
        / "fault_matrix"
        / "application_logging.py"
    )
    spec = importlib.util.spec_from_file_location(
        "quantis_test_application_logging",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bounded_application_state_event_names() -> None:
    logging = _application_logging()

    assert logging.queue_backlog_event_name(0) == "queue.backlog.low"
    assert logging.queue_backlog_event_name(4) == (
        "queue.backlog.elevated"
    )
    assert logging.queue_backlog_event_name(12) == "queue.backlog.high"
    assert logging.database_latency_event_name(1_999) == (
        "database.write.latency.fast"
    )
    assert logging.database_latency_event_name(2_000) == (
        "database.write.latency.normal"
    )
    assert logging.database_latency_event_name(10_000) == (
        "database.write.latency.slow"
    )
    assert logging.checkout_queue_wait_event_name(
        enqueued_unix_nano=1_000_000_000,
        dequeued_unix_nano=1_009_999_000,
    ) is None
    assert logging.checkout_queue_wait_event_name(
        enqueued_unix_nano=1_000_000_000,
        dequeued_unix_nano=1_010_000_000,
    ) == (
        "checkout.queue_wait.elevated"
    )
    assert logging.checkout_queue_wait_event_name(
        enqueued_unix_nano=1_000_000_000,
        dequeued_unix_nano=1_050_000_000,
    ) == (
        "checkout.queue_wait.slow"
    )
    queue_wait_event = logging.checkout_queue_wait_event(
        enqueued_unix_nano=1_000_000_000,
        dequeued_unix_nano=1_050_000_000,
        origin_window_index=7,
    )
    assert queue_wait_event is not None
    assert queue_wait_event.event_name == "checkout.queue_wait.slow"
    assert queue_wait_event.timestamp_unix_nano == 1_050_000_000


def test_dependency_events_are_bounded_and_suppress_routine_success() -> None:
    logging = _application_logging()

    assert logging.dependency_operation_events(
        dependency_name="redis",
        latency_us=499,
        origin_window_index=7,
    ) == ()
    events = logging.dependency_operation_events(
        dependency_name="redis",
        latency_us=2_000,
        origin_window_index=7,
        failed=True,
        retry_count=2,
        pool_wait_us=2_000,
        timestamp_unix_nano=1_234,
    )

    assert tuple(event.event_name for event in events) == (
        "dependency.redis.operation.error",
        "dependency.redis.retry.observed",
        "dependency.redis.pool_wait.elevated",
    )
    assert all(
        event.attributes
        == {
            "dependency.name": "redis",
            "quantis.experiment.origin.window.index": 7,
        }
        for event in events
    )
    assert all(
        event.timestamp_unix_nano == 1_234 for event in events
    )
    assert tuple(
        event.event_name
        for event in logging.dependency_operation_events(
            dependency_name="postgresql",
            latency_us=2_000,
            origin_window_index=7,
        )
    ) == ("dependency.postgresql.latency.elevated",)
    assert tuple(
        event.event_name
        for event in logging.dependency_operation_events(
            dependency_name="redis",
            latency_us=2_000,
            origin_window_index=7,
        )
    ) == ("dependency.redis.latency.slow",)
    assert all(
        "key" not in event.attributes
        and "payload" not in event.attributes
        and "exception" not in event.attributes
        for event in events
    )


def test_dependency_events_reject_unregistered_vocabulary() -> None:
    logging = _application_logging()

    try:
        logging.dependency_operation_events(
            dependency_name="supabase",
            latency_us=10_000,
            origin_window_index=7,
        )
    except ValueError as error:
        assert "unsupported dependency" in str(error)
    else:
        raise AssertionError("unregistered dependencies must be rejected")


def test_manifest_identity_is_available_before_worker_dequeue(
    tmp_path,
) -> None:
    logging = _application_logging()
    manifest = {
        "case_id": "contextual-v2-development-f01-w1-89",
        "fault_kind": "none",
        "topology_id": "workers-1",
        "worker_replicas": 1,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    canonical = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    assert logging.experiment_identity_from_manifest(
        manifest_path
    ) == {
        "case_id": "contextual-v2-development-f01-w1-89",
        "fault_kind": "none",
        "manifest_sha256": hashlib.sha256(canonical).hexdigest(),
        "topology_id": "workers-1",
    }

    compose = (
        Path(__file__).resolve().parents[1]
        / "lab"
        / "fault_matrix"
        / "compose.yaml"
    ).read_text()
    assert (
        "${EXPERIMENT_DIRECTORY:-./experiments}:/experiments:ro"
        in compose
    )
    assert "EXPERIMENT_PATH: ${EXPERIMENT_PATH}" in compose


def test_application_events_share_one_bounded_otlp_batch(monkeypatch) -> None:
    logging = _application_logging()
    captured = {}

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    def urlopen(request, timeout):
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data)
        return Response()

    monkeypatch.setattr(logging.urllib.request, "urlopen", urlopen)
    logging.emit_application_events(
        service_name="worker",
        service_instance_id="worker-1",
        experiment={
            "case_id": "promotion-f01-w1-73",
            "fault_kind": "none",
            "manifest_sha256": "sha",
            "topology_id": "workers-1",
        },
        events=(
            logging.ApplicationEvent(
                event_name="checkout.completed",
                severity_number=9,
                severity_text="INFO",
                body="checkout completed",
                attributes={
                    "quantis.experiment.origin.window.index": 7,
                },
                timestamp_unix_nano=123,
            ),
            logging.ApplicationEvent(
                event_name="database.write.latency.normal",
                severity_number=9,
                severity_text="INFO",
                body="database write latency normal",
                attributes={
                    "quantis.experiment.origin.window.index": 7,
                },
            ),
        ),
    )

    records = captured["payload"]["resourceLogs"][0]["scopeLogs"][0][
        "logRecords"
    ]
    assert len(records) == 2
    assert [
        next(
            attribute["value"]["stringValue"]
            for attribute in record["attributes"]
            if attribute["key"] == "event.name"
        )
        for record in records
    ] == [
        "checkout.completed",
        "database.write.latency.normal",
    ]
    assert records[0]["timeUnixNano"] == "123"
    assert records[0]["observedTimeUnixNano"] != "123"
    assert captured["timeout"] == 3


def test_queue_mutation_and_transition_are_one_atomic_script() -> None:
    logging = _application_logging()

    class AtomicRedis:
        def __init__(self):
            self.queue = []
            self.state = "queue.backlog.low"
            self.calls = []

        def eval(self, script, numkeys, *keys):
            self.calls.append((script, numkeys, keys))
            if "RPUSH" in script:
                self.queue.append(keys[2])
                payload = None
            else:
                payload = (
                    self.queue.pop(0) if self.queue else None
                )
                if payload is None:
                    return [None, None, None, None]
            state = logging.queue_backlog_event_name(len(self.queue))
            transition = None
            if self.state != state:
                self.state = state
                transition = [state, "10", "123"]
            if "RPUSH" in script:
                return transition or [None, None, None]
            return [
                payload,
                *(transition or [None, None, None]),
            ]

    redis_client = AtomicRedis()
    low = logging.enqueue_with_queue_transition(
        redis_client,
        queue_key="queue",
        state_key="state",
        payload="one",
    )
    assert low is None
    assert (
        logging.enqueue_with_queue_transition(
            redis_client,
            queue_key="queue",
            state_key="state",
            payload="two",
        )
        is None
    )
    elevated = logging.enqueue_with_queue_transition(
        redis_client,
        queue_key="queue",
        state_key="state",
        payload="three",
    )
    assert elevated is not None
    assert elevated.event_name == "queue.backlog.elevated"
    assert elevated.timestamp_unix_nano == 10_000_123_000

    payload, lowered = logging.dequeue_with_queue_transition(
        redis_client,
        queue_key="queue",
        state_key="state",
    )
    assert payload == "one"
    assert lowered is not None
    assert lowered.event_name == "queue.backlog.low"
    assert len(redis_client.calls) == 4
    assert all(call[1] == 2 for call in redis_client.calls)
    assert redis_client.calls[0][2] == (
        "queue",
        "state",
        "one",
    )
    enqueue_script = redis_client.calls[0][0]
    dequeue_script = redis_client.calls[-1][0]
    assert "enqueued_unix_nano" in enqueue_script
    assert enqueue_script.index("redis.call('TIME')") < (
        enqueue_script.index("RPUSH")
    )
    assert enqueue_script.index("RPUSH") < enqueue_script.index("LLEN")
    assert dequeue_script.index("LPOP") < dequeue_script.index("LLEN")
    assert "redis.call('TIME')" in enqueue_script
    assert "redis.call('TIME')" in dequeue_script
    assert "redis.call('SET', KEYS[2], state)" in (
        enqueue_script
    )


def test_worked_otlp_fixture_contains_only_bounded_state_events() -> None:
    capture = read_otlp_log_capture(
        Path(__file__).resolve().parent
        / "fixtures"
        / "otlp"
        / "contextual-application-state-logs.jsonl"
    )

    assert tuple(
        record.record_attributes["event.name"]
        for record in capture.records
    ) == (
        "checkout.completed",
        "database.write.latency.fast",
        "worker.state.busy",
        "queue.backlog.low",
        "worker.state.idle",
    )
    assert {
        record.resource_attributes[
            "quantis.experiment.case.id"
        ]
        for record in capture.records
    } == {"contextual-promotion-fixture"}


def test_worked_v2_otlp_fixture_contains_no_dependency_payloads() -> None:
    capture = read_otlp_log_capture(
        Path(__file__).resolve().parent
        / "fixtures"
        / "otlp"
        / "contextual-v2-dependency-logs.jsonl"
    )

    assert tuple(
        record.record_attributes["event.name"]
        for record in capture.records
    ) == (
        "dependency.redis.latency.elevated",
        "dependency.postgresql.latency.slow",
        "checkout.queue_wait.slow",
    )
    assert all(
        set(record.record_attributes)
        <= {
            "dependency.name",
            "event.name",
            "quantis.experiment.origin.window.index",
        }
        for record in capture.records
    )
