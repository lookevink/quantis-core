"""Load qualified real action-dynamics development trajectories."""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .action_conditioned_dynamics import ActionConditionedRun
from .action_dynamics_lab import (
    ACTION_LAB_FEATURE_NAMES,
    LabActionCaptureManifest,
    assess_prepared_action_collection,
)
from .graph_telemetry import (
    DeclaredTelemetryGraph,
    GraphEntity,
    TelemetryBinding,
)
from .otlp import read_otlp_capture
from .otlp_logs import read_otlp_log_capture


__all__ = [
    "ActionDynamicsCorpusIdentity",
    "ActionDynamicsCorpusSummary",
    "CONTROL_FEATURE_NAMES",
    "ENTITY_IDS",
    "EVENT_FEATURE_NAMES",
    "LoadedActionDynamicsCorpus",
    "STATE_FEATURE_NAMES",
    "load_action_dynamics_development_corpus",
]

EVENT_FEATURE_NAMES = (
    "log_event_count",
    "log_error_count",
    "trace_span_count",
    "trace_error_count",
)
CONTROL_FEATURE_NAMES = ("request_demand", "worker_replicas")
STATE_FEATURE_NAMES = (*ACTION_LAB_FEATURE_NAMES, *EVENT_FEATURE_NAMES)
ENTITY_IDS = (
    "api",
    "api_enqueues_queue",
    "checkout_queue",
    "queue_dequeues_to_worker",
    "worker_pool",
    "worker_writes_postgresql",
    "postgresql",
)

_METRIC_OWNERS = {
    "request_rate": "api",
    "request_latency_ms": "api",
    "error_rate": "api",
    "api_inflight_current": "api",
    "api_inflight_peak": "api",
    "api_concurrency_mean": "api",
    "queue_depth": "checkout_queue",
    "queue_oldest_age_ms": "checkout_queue",
    "enqueue_event_age_ms": "checkout_queue",
    "dequeue_event_age_ms": "checkout_queue",
    "queue_residence_mean_ms": "checkout_queue",
    "worker_rate": "worker_pool",
    "worker_heartbeat_age_s": "worker_pool",
    "worker_active_count": "worker_pool",
    "worker_busy_count": "worker_pool",
    "worker_busy_age_max_ms": "worker_pool",
    "worker_busy_fraction": "worker_pool",
    "worker_processing_latency_ms": "worker_pool",
    "redis_enqueue_latency_ms": "api_enqueues_queue",
    "redis_enqueue_error_rate": "api_enqueues_queue",
    "redis_dequeue_latency_ms": "queue_dequeues_to_worker",
    "redis_dequeue_error_rate": "queue_dequeues_to_worker",
    "db_write_rate": "worker_writes_postgresql",
    "postgresql_write_latency_ms": "worker_writes_postgresql",
    "postgresql_write_error_rate": "worker_writes_postgresql",
    "postgresql_write_event_age_ms": "worker_writes_postgresql",
    "postgresql_write_busy_age_max_ms": "worker_writes_postgresql",
}
_LOG_OWNERS = {
    "checkout.accepted": "api",
    "checkout.rejected": "api",
    "checkout.completed": "worker_pool",
}
_LOG_SERVICES = {
    "checkout.accepted": "quantis-action-api",
    "checkout.rejected": "quantis-action-api",
    "checkout.completed": "quantis-action-worker",
}
_SPAN_OWNERS = {
    "api.admission": "api",
    "redis.enqueue": "api_enqueues_queue",
    "queue.residence": "checkout_queue",
    "redis.dequeue": "queue_dequeues_to_worker",
    "worker.processing": "worker_pool",
    "postgresql.write": "worker_writes_postgresql",
}
_ACTION_TARGETS = {
    "worker_pause": "worker_pool",
    "postgres_lock": "worker_writes_postgresql",
    "redis_enqueue_delay": "api_enqueues_queue",
    "redis_dequeue_delay": "queue_dequeues_to_worker",
    "api_rejection": "api",
}


@dataclass(frozen=True)
class ActionDynamicsCorpusSummary:
    """Auditable development-corpus counts and balance."""

    corpus_role: str
    run_count: int
    pair_count: int
    training_run_count: int
    training_pair_count: int
    validation_run_count: int
    validation_pair_count: int
    treatment_run_count: int
    control_run_count: int
    action_pair_counts: Tuple[Tuple[str, int], ...]
    topology_pair_counts: Tuple[Tuple[str, int], ...]
    split_cell_pair_counts: Tuple[Tuple[str, str, int, int], ...]
    zero_padded_entity_ids: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe evidence summary."""

        return {
            "corpus_role": self.corpus_role,
            "run_count": self.run_count,
            "pair_count": self.pair_count,
            "training_run_count": self.training_run_count,
            "training_pair_count": self.training_pair_count,
            "validation_run_count": self.validation_run_count,
            "validation_pair_count": self.validation_pair_count,
            "treatment_run_count": self.treatment_run_count,
            "control_run_count": self.control_run_count,
            "action_pair_counts": dict(self.action_pair_counts),
            "topology_pair_counts": dict(
                self.topology_pair_counts
            ),
            "split_cell_pair_counts": [
                {
                    "split": split,
                    "action_kind": action_kind,
                    "worker_replicas": worker_replicas,
                    "pair_count": pair_count,
                }
                for (
                    split,
                    action_kind,
                    worker_replicas,
                    pair_count,
                ) in self.split_cell_pair_counts
            ],
            "zero_padded_entity_ids": list(
                self.zero_padded_entity_ids
            ),
        }


@dataclass(frozen=True)
class ActionDynamicsCorpusIdentity:
    """Exact evidence and semantic-schema hashes."""

    artifact_manifest_sha256: str
    assessment_sha256: str
    protocol_sha256: str
    plan_sha256: str
    observation_schema_sha256: str
    graph_sha256: str
    semantic_schema_sha256: str
    corpus_sha256: str

    def to_dict(self) -> Dict[str, str]:
        """Return named immutable identities."""

        return {
            "artifact_manifest_sha256": (
                self.artifact_manifest_sha256
            ),
            "assessment_sha256": self.assessment_sha256,
            "protocol_sha256": self.protocol_sha256,
            "plan_sha256": self.plan_sha256,
            "observation_schema_sha256": (
                self.observation_schema_sha256
            ),
            "graph_sha256": self.graph_sha256,
            "semantic_schema_sha256": (
                self.semantic_schema_sha256
            ),
            "corpus_sha256": self.corpus_sha256,
        }


@dataclass(frozen=True)
class LoadedActionDynamicsCorpus:
    """Qualified runs with whole-pair partitions and identities."""

    runs: Tuple[ActionConditionedRun, ...]
    training_pair_ids: Tuple[str, ...]
    validation_pair_ids: Tuple[str, ...]
    summary: ActionDynamicsCorpusSummary
    identity: ActionDynamicsCorpusIdentity

    @property
    def training_runs(self) -> Tuple[ActionConditionedRun, ...]:
        """Return runs admitted for preprocessing and model fitting."""

        return tuple(
            run
            for run in self.runs
            if run.manifest.split == "training"
        )

    @property
    def validation_runs(self) -> Tuple[ActionConditionedRun, ...]:
        """Return frozen development-validation runs."""

        return tuple(
            run
            for run in self.runs
            if run.manifest.split == "validation"
        )


def load_action_dynamics_development_corpus(
    directory: Path,
) -> LoadedActionDynamicsCorpus:
    """Load one qualified, content-addressed development corpus."""

    root = Path(directory)
    protocol_path = root / "inputs" / "protocol.json"
    protocol = _read_object(protocol_path)
    if protocol.get("stage") != "development":
        raise ValueError(
            "action-dynamics model input must be a development corpus"
        )
    artifact_hashes = _validate_artifact_manifest(root)
    assessment_path = root / "data-quality.json"
    recorded_assessment = _read_object(assessment_path)
    recomputed_assessment = assess_prepared_action_collection(
        root / "inputs",
        root / "cases",
        root / "collection-attestation.json",
    )
    if recomputed_assessment != recorded_assessment:
        raise ValueError(
            "recorded development assessment differs from recomputation"
        )
    if (
        recorded_assessment.get("stage") != "development"
        or recorded_assessment.get("status") != "qualified"
        or recorded_assessment.get("decision")
        != "freeze_training_corpus"
    ):
        raise ValueError(
            "action-dynamics development corpus is not qualified"
        )
    graph = _development_graph()
    observation_schema = _read_object(
        root / "observation-schema.json"
    )
    _validate_observation_schema(observation_schema)
    runs = _load_runs(
        root,
        graph,
        _canonical_sha256(observation_schema),
    )
    (
        training_pair_ids,
        validation_pair_ids,
        summary,
    ) = _validate_pair_balance(runs)
    identity = _corpus_identity(
        root,
        artifact_hashes,
        graph,
        summary,
    )
    return LoadedActionDynamicsCorpus(
        runs=runs,
        training_pair_ids=training_pair_ids,
        validation_pair_ids=validation_pair_ids,
        summary=summary,
        identity=identity,
    )


def _load_runs(
    root: Path,
    graph: DeclaredTelemetryGraph,
    observation_schema_sha256: str,
) -> Tuple[ActionConditionedRun, ...]:
    prepared = root / "inputs" / "manifests"
    captures = root / "cases"
    prepared_ids = {
        path.stem for path in prepared.glob("*.json")
    }
    capture_ids = {
        path.name for path in captures.iterdir() if path.is_dir()
    }
    if (
        len(prepared_ids) != 240
        or prepared_ids != capture_ids
    ):
        raise ValueError(
            "development corpus must contain exactly 240 captures"
        )
    runs = []
    for case_id in sorted(prepared_ids):
        prepared_payload = _read_object(
            prepared / f"{case_id}.json"
        )
        capture_path = captures / case_id / "capture-manifest.json"
        capture_payload = _read_object(capture_path)
        if prepared_payload != capture_payload:
            raise ValueError(
                f"captured manifest differs for case {case_id}"
            )
        lab_manifest = LabActionCaptureManifest.from_dict(
            capture_payload
        )
        if (
            lab_manifest.schema_version != 3
            or lab_manifest.corpus_role != "development"
            or lab_manifest.action_case.split
            not in {"training", "validation"}
            or lab_manifest.observation_schema_sha256
            != observation_schema_sha256
            or lab_manifest.graph_observation_schema_sha256
            != observation_schema_sha256
        ):
            raise ValueError(
                f"case {case_id} is not development model input"
            )
        observations = _observations(
            captures / case_id,
            lab_manifest,
            graph,
        )
        request_demand = np.asarray(
            tuple(
                float(value)
                for value in lab_manifest.request_schedule[
                    : lab_manifest.action_case.point_count
                ]
            ),
            dtype=np.float64,
        )
        if request_demand.shape != (
            lab_manifest.action_case.point_count,
        ):
            raise ValueError(
                f"case {case_id} request controls do not align"
            )
        controls = np.column_stack(
            (
                request_demand,
                np.full(
                    lab_manifest.action_case.point_count,
                    float(
                        lab_manifest.action_case.worker_replicas
                    ),
                    dtype=np.float64,
                ),
            )
        )
        runs.append(
            ActionConditionedRun(
                manifest=lab_manifest.action_case,
                graph=graph,
                observations=observations,
                controls=np.asarray(controls, dtype=np.float64),
                state_feature_names=STATE_FEATURE_NAMES,
                control_feature_names=CONTROL_FEATURE_NAMES,
            )
        )
    return tuple(runs)


def _observations(
    directory: Path,
    manifest: LabActionCaptureManifest,
    graph: DeclaredTelemetryGraph,
) -> NDArray[np.float64]:
    point_count = manifest.action_case.point_count
    observations = np.zeros(
        (point_count, len(ENTITY_IDS), len(STATE_FEATURE_NAMES)),
        dtype=np.float64,
    )
    entity_position = {
        entity_id: index
        for index, entity_id in enumerate(graph.entity_ids)
    }
    feature_position = {
        name: index for index, name in enumerate(STATE_FEATURE_NAMES)
    }
    metric_capture = read_otlp_capture(
        directory / "collector-metrics.jsonl"
    )
    by_metric: Dict[str, list[Tuple[int, float]]] = {}
    for point in metric_capture.points:
        if (
            point.metric_name in _METRIC_OWNERS
            and point.number_value is not None
        ):
            by_metric.setdefault(point.metric_name, []).append(
                (
                    point.time_unix_nano,
                    float(point.number_value),
                )
            )
    if set(by_metric) != set(ACTION_LAB_FEATURE_NAMES):
        raise ValueError("development operational metrics are incomplete")
    metric_timestamps: set[Tuple[int, ...]] = set()
    for name in ACTION_LAB_FEATURE_NAMES:
        points = sorted(by_metric[name])
        if (
            len(points) != point_count
            or len({timestamp for timestamp, _ in points})
            != point_count
            or any(
                not math.isfinite(value) or value < 0.0
                for _, value in points
            )
        ):
            raise ValueError(
                f"development metric {name} does not align"
            )
        metric_timestamps.add(
            tuple(timestamp for timestamp, _ in points)
        )
        observations[
            :,
            entity_position[_METRIC_OWNERS[name]],
            feature_position[name],
        ] = np.asarray(
            [value for _, value in points], dtype=np.float64
        )
    if len(metric_timestamps) != 1:
        raise ValueError("development metric windows do not align")

    log_capture = read_otlp_log_capture(
        directory / "collector-logs.jsonl"
    )
    for record in log_capture.records:
        event_name = record.record_attributes.get("event.name")
        logical_index = record.record_attributes.get(
            "quantis.experiment.origin.window.index"
        )
        service_name = record.resource_attributes.get(
            "service.name"
        )
        if (
            not isinstance(event_name, str)
            or event_name not in _LOG_OWNERS
            or service_name != _LOG_SERVICES[event_name]
            or isinstance(logical_index, bool)
            or not isinstance(logical_index, int)
            or not 0 <= logical_index < point_count
            or record.severity_number <= 0
        ):
            raise ValueError(
                "structured development log cannot be graph-windowed"
            )
        owner = entity_position[_LOG_OWNERS[event_name]]
        observations[
            logical_index,
            owner,
            feature_position["log_event_count"],
        ] += 1.0
        if record.severity_number >= 17:
            observations[
                logical_index,
                owner,
                feature_position["log_error_count"],
            ] += 1.0

    for logical_index, owner_id, is_error in _trace_events(
        directory / "collector-traces.jsonl",
        point_count,
    ):
        owner = entity_position[owner_id]
        observations[
            logical_index,
            owner,
            feature_position["trace_span_count"],
        ] += 1.0
        if is_error:
            observations[
                logical_index,
                owner,
                feature_position["trace_error_count"],
            ] += 1.0
    terminal = entity_position["postgresql"]
    if np.any(observations[:, terminal, :] != 0.0):
        raise AssertionError(
            "terminal PostgreSQL node must remain zero-padded"
        )
    return observations


def _trace_events(
    path: Path,
    point_count: int,
) -> Tuple[Tuple[int, str, bool], ...]:
    events = []
    message_count = 0
    for line_number, line in enumerate(
        path.read_text().splitlines(), start=1
    ):
        if not line.strip():
            continue
        message_count += 1
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(
                f"invalid OTLP trace message at line {line_number}"
            )
        for resource_spans in payload.get("resourceSpans", []):
            for scope_spans in resource_spans.get(
                "scopeSpans", []
            ):
                for span in scope_spans.get("spans", []):
                    attributes = _otlp_attributes(
                        span.get("attributes", [])
                    )
                    name = span.get("name")
                    owner = attributes.get(
                        "quantis.graph.entity.id"
                    )
                    logical_index = attributes.get(
                        "quantis.experiment.origin.window.index"
                    )
                    status = span.get("status", {})
                    status_code = (
                        status.get("code", 0)
                        if isinstance(status, dict)
                        else -1
                    )
                    if (
                        not isinstance(name, str)
                        or name not in _SPAN_OWNERS
                        or owner != _SPAN_OWNERS[name]
                        or isinstance(logical_index, bool)
                        or not isinstance(logical_index, int)
                        or not 0 <= logical_index < point_count
                        or isinstance(status_code, bool)
                        or not isinstance(status_code, int)
                        or status_code not in {1, 2}
                    ):
                        raise ValueError(
                            "structured development trace cannot be "
                            "graph-windowed"
                        )
                    events.append(
                        (logical_index, str(owner), status_code == 2)
                    )
    if message_count == 0 or not events:
        raise ValueError("development trace capture is empty")
    return tuple(events)


def _validate_pair_balance(
    runs: Sequence[ActionConditionedRun],
) -> Tuple[
    Tuple[str, ...],
    Tuple[str, ...],
    ActionDynamicsCorpusSummary,
]:
    grouped: Dict[str, list[ActionConditionedRun]] = {}
    for run in runs:
        grouped.setdefault(
            run.manifest.matched_pair_id, []
        ).append(run)
    action_counts: Dict[str, int] = {}
    topology_counts: Dict[str, int] = {}
    cell_counts: Dict[Tuple[str, str, int], int] = {}
    split_pairs: Dict[str, list[str]] = {
        "training": [],
        "validation": [],
    }
    treatment_count = 0
    for pair_id, pair in grouped.items():
        if len(pair) != 2:
            raise ValueError(
                f"development matched pair {pair_id} is incomplete"
            )
        treatment = [run for run in pair if run.manifest.actions]
        control = [run for run in pair if not run.manifest.actions]
        if len(treatment) != 1 or len(control) != 1:
            raise ValueError(
                f"development matched pair {pair_id} lacks twins"
            )
        action_run = treatment[0]
        control_run = control[0]
        if len(action_run.manifest.actions) != 1:
            raise ValueError(
                "development treatment must have exactly one action"
            )
        action = action_run.manifest.actions[0]
        split = action_run.manifest.split
        if (
            split != control_run.manifest.split
            or split not in split_pairs
            or action_run.manifest.topology_id
            != control_run.manifest.topology_id
            or action_run.manifest.worker_replicas
            != control_run.manifest.worker_replicas
            or action_run.manifest.workload_seed
            != control_run.manifest.workload_seed
            or action_run.manifest.intervention_seed
            != control_run.manifest.intervention_seed
            or action.target_entity
            != _ACTION_TARGETS.get(action.action_kind)
            or not np.array_equal(
                action_run.controls, control_run.controls
            )
        ):
            raise ValueError(
                f"development matched pair {pair_id} drifted"
            )
        split_pairs[split].append(pair_id)
        action_counts[action.action_kind] = (
            action_counts.get(action.action_kind, 0) + 1
        )
        topology_counts[action_run.manifest.topology_id] = (
            topology_counts.get(
                action_run.manifest.topology_id, 0
            )
            + 1
        )
        key = (
            split,
            action.action_kind,
            action_run.manifest.worker_replicas,
        )
        cell_counts[key] = cell_counts.get(key, 0) + 1
        treatment_count += 1
    training_pair_ids = tuple(sorted(split_pairs["training"]))
    validation_pair_ids = tuple(sorted(split_pairs["validation"]))
    expected_cells = {
        (split, action_kind, worker_replicas): expected
        for split, expected in (("training", 6), ("validation", 2))
        for action_kind in _ACTION_TARGETS
        for worker_replicas in (1, 2, 3)
    }
    if (
        len(runs) != 240
        or len(grouped) != 120
        or len(training_pair_ids) != 90
        or len(validation_pair_ids) != 30
        or set(training_pair_ids) & set(validation_pair_ids)
        or cell_counts != expected_cells
        or treatment_count != 120
    ):
        raise ValueError(
            "development corpus is not the frozen 90/30 "
            "whole-pair factorial"
        )
    summary = ActionDynamicsCorpusSummary(
        corpus_role="development",
        run_count=len(runs),
        pair_count=len(grouped),
        training_run_count=180,
        training_pair_count=90,
        validation_run_count=60,
        validation_pair_count=30,
        treatment_run_count=treatment_count,
        control_run_count=len(runs) - treatment_count,
        action_pair_counts=tuple(sorted(action_counts.items())),
        topology_pair_counts=tuple(
            sorted(topology_counts.items())
        ),
        split_cell_pair_counts=tuple(
            (
                split,
                action_kind,
                worker_replicas,
                count,
            )
            for (
                split,
                action_kind,
                worker_replicas,
            ), count in sorted(cell_counts.items())
        ),
        zero_padded_entity_ids=("postgresql",),
    )
    return training_pair_ids, validation_pair_ids, summary


def _development_graph() -> DeclaredTelemetryGraph:
    entities = (
        GraphEntity("api", "node", "service"),
        GraphEntity(
            "api_enqueues_queue",
            "edge",
            "enqueue",
            "api",
            "checkout_queue",
        ),
        GraphEntity(
            "checkout_queue", "node", "stateful_resource"
        ),
        GraphEntity(
            "queue_dequeues_to_worker",
            "edge",
            "dequeue",
            "checkout_queue",
            "worker_pool",
        ),
        GraphEntity("worker_pool", "node", "service_pool"),
        GraphEntity(
            "worker_writes_postgresql",
            "edge",
            "database_write",
            "worker_pool",
            "postgresql",
        ),
        GraphEntity("postgresql", "node", "dependency"),
    )
    bindings = tuple(
        TelemetryBinding(f"metric.{name}", owner)
        for name, owner in _METRIC_OWNERS.items()
    )
    return DeclaredTelemetryGraph(
        entities=entities,
        bindings=bindings,
    )


def _validate_observation_schema(
    payload: Mapping[str, Any],
) -> None:
    expected_edges = [
        ["api", "api_enqueues_queue"],
        ["api_enqueues_queue", "checkout_queue"],
        ["checkout_queue", "queue_dequeues_to_worker"],
        ["queue_dequeues_to_worker", "worker_pool"],
        ["worker_pool", "worker_writes_postgresql"],
        ["worker_writes_postgresql", "postgresql"],
    ]
    graph = payload.get("graph")
    if (
        payload.get("kind")
        != "action_dynamics_observation_schema"
        or payload.get("feature_names")
        != list(ACTION_LAB_FEATURE_NAMES)
        or not isinstance(graph, dict)
        or graph.get("entities") != list(ENTITY_IDS)
        or graph.get("directed_edges") != expected_edges
        or payload.get("truth_fields_permitted") != []
    ):
        raise ValueError(
            "development observation schema is incompatible"
        )


def _corpus_identity(
    root: Path,
    artifact_hashes: Mapping[str, str],
    graph: DeclaredTelemetryGraph,
    summary: ActionDynamicsCorpusSummary,
) -> ActionDynamicsCorpusIdentity:
    graph_sha256 = _canonical_sha256(graph.to_dict())
    semantic_schema_sha256 = _canonical_sha256(
        {
            "graph": graph.to_dict(),
            "state_feature_names": list(STATE_FEATURE_NAMES),
            "control_feature_names": list(
                CONTROL_FEATURE_NAMES
            ),
            "metric_owners": dict(sorted(_METRIC_OWNERS.items())),
            "event_features": list(EVENT_FEATURE_NAMES),
            "event_ownership": (
                "per_record_log_or_span_graph_entity"
            ),
        }
    )
    exact = {
        "artifact_manifest_sha256": _file_sha256(
            root / "artifact-manifest.json"
        ),
        "assessment_sha256": _file_sha256(
            root / "data-quality.json"
        ),
        "protocol_sha256": _file_sha256(
            root / "inputs" / "protocol.json"
        ),
        "plan_sha256": _file_sha256(
            root / "inputs" / "plan.json"
        ),
        "observation_schema_sha256": _file_sha256(
            root / "observation-schema.json"
        ),
        "graph_sha256": graph_sha256,
        "semantic_schema_sha256": semantic_schema_sha256,
    }
    corpus_sha256 = _canonical_sha256(
        {
            **exact,
            "artifact_file_sha256s": dict(
                sorted(artifact_hashes.items())
            ),
            "summary": summary.to_dict(),
        }
    )
    return ActionDynamicsCorpusIdentity(
        **exact,
        corpus_sha256=corpus_sha256,
    )


def _validate_artifact_manifest(root: Path) -> Mapping[str, str]:
    path = root / "artifact-manifest.json"
    payload = _read_object(path)
    raw_hashes = payload.get("sha256")
    if (
        set(payload) != {"schema_version", "kind", "sha256"}
        or payload.get("schema_version") != 1
        or payload.get("kind")
        != "action_dynamics_artifact_manifest"
        or not isinstance(raw_hashes, dict)
        or any(
            not isinstance(name, str)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in digest
            )
            for name, digest in (
                raw_hashes.items()
                if isinstance(raw_hashes, dict)
                else ()
            )
        )
    ):
        raise ValueError("development artifact manifest is invalid")
    hashes = {
        str(name): str(digest)
        for name, digest in raw_hashes.items()
    }
    observed = {
        candidate.relative_to(root).as_posix()
        for candidate in root.rglob("*")
        if candidate.is_file() and candidate != path
    }
    if observed != set(hashes):
        raise ValueError(
            "development artifact exact file coverage failed"
        )
    for name, expected in hashes.items():
        candidate = root / name
        if (
            Path(name).is_absolute()
            or ".." in Path(name).parts
            or candidate.is_symlink()
            or _file_sha256(candidate) != expected
        ):
            raise ValueError(
                "development artifact exact file hash failed"
            )
    return dict(sorted(hashes.items()))


def _otlp_attributes(
    raw_attributes: object,
) -> Mapping[str, object]:
    if not isinstance(raw_attributes, list):
        raise ValueError("OTLP trace attributes must be a list")
    attributes: Dict[str, object] = {}
    for raw in raw_attributes:
        if not isinstance(raw, dict):
            raise ValueError("OTLP trace attribute is invalid")
        key = raw.get("key")
        value = raw.get("value")
        if (
            not isinstance(key, str)
            or key in attributes
            or not isinstance(value, dict)
        ):
            raise ValueError("OTLP trace attribute is invalid")
        present = [
            field
            for field in (
                "stringValue",
                "intValue",
                "doubleValue",
                "boolValue",
            )
            if field in value
        ]
        if len(present) != 1:
            raise ValueError("OTLP trace attribute value is invalid")
        field = present[0]
        candidate: object = value[field]
        if field == "stringValue":
            candidate = str(candidate)
        elif field == "intValue":
            if not isinstance(candidate, (str, int)):
                raise ValueError(
                    "OTLP trace integer attribute is invalid"
                )
            candidate = int(candidate)
        elif field == "doubleValue":
            if (
                isinstance(candidate, bool)
                or not isinstance(candidate, (str, int, float))
            ):
                raise ValueError(
                    "OTLP trace double attribute is invalid"
                )
            candidate = float(candidate)
        elif field == "boolValue":
            candidate = bool(candidate)
        attributes[key] = candidate
    return attributes


def _read_object(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
