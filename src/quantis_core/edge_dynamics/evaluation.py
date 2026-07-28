"""Comparable evaluation, calibration, and streaming microbenchmarks."""

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from ..action_conditioned_dynamics import (
    ActionConditionedWindows,
    TrajectoryDistribution,
    persistence_rollout,
)
from .data import (
    PreparedAttributionQueries,
    PreparedEdgeDynamicsData,
)
from .models import EdgeDynamicsModel
from ..graph_telemetry import DeclaredTelemetryGraph


@dataclass(frozen=True)
class EdgeModelScores:
    """Comparable development metrics for one fitted candidate."""

    normalized_mse_overall: float
    normalized_mse_action_overlap: float
    downstream_effect_mse: float
    action_and_target_hit_at_1: float
    no_action_specificity: float
    maximum_absolute_prediction: float
    maximum_horizon_norm_growth: float
    rollout_finite: bool
    parameter_count: int
    serialized_size_bytes: int
    batch_one_latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "normalized_mse_overall": self.normalized_mse_overall,
            "normalized_mse_action_overlap": (
                self.normalized_mse_action_overlap
            ),
            "downstream_effect_mse": self.downstream_effect_mse,
            "action_and_target_hit_at_1": (
                self.action_and_target_hit_at_1
            ),
            "no_action_specificity": self.no_action_specificity,
            "maximum_absolute_prediction": (
                self.maximum_absolute_prediction
            ),
            "maximum_horizon_norm_growth": (
                self.maximum_horizon_norm_growth
            ),
            "rollout_finite": self.rollout_finite,
            "parameter_count": self.parameter_count,
            "serialized_size_bytes": self.serialized_size_bytes,
            "batch_one_latency_ms": self.batch_one_latency_ms,
        }


def forecast_objective(
    model: EdgeDynamicsModel,
    windows: ActionConditionedWindows,
    *,
    state_feature_positions: Sequence[int] = (),
) -> Mapping[str, float]:
    """Return leak-free selection metrics without attribution queries."""

    prediction = model.rollout(
        windows.histories,
        windows.future_controls,
        windows.future_actions,
        windows.graph,
    ).mean
    observed = np.asarray(windows.future_states, dtype=np.float64)
    prediction, observed = _select_features(
        prediction, observed, state_feature_positions
    )
    squared_error = np.square(prediction - observed)
    action_overlap = np.any(
        windows.future_actions[..., 1] > 0.5, axis=2
    )
    if not np.any(action_overlap):
        raise ValueError("forecast objective has no action-overlap windows")
    return {
        "normalized_mse_overall": float(np.mean(squared_error)),
        "normalized_mse_action_overlap": float(
            np.mean(squared_error[action_overlap])
        ),
    }


def score_edge_model(
    model: EdgeDynamicsModel,
    windows: ActionConditionedWindows,
    attribution_queries: PreparedAttributionQueries,
    *,
    state_feature_positions: Sequence[int] = (),
) -> EdgeModelScores:
    """Score one fitted model through the shared rollout seam."""

    distribution = model.rollout(
        windows.histories,
        windows.future_controls,
        windows.future_actions,
        windows.graph,
    )
    prediction = distribution.mean
    observed = np.asarray(windows.future_states, dtype=np.float64)
    selected_prediction, selected_observed = _select_features(
        prediction, observed, state_feature_positions
    )
    squared_error = np.square(selected_prediction - selected_observed)
    action_overlap = np.any(
        windows.future_actions[..., 1] > 0.5, axis=2
    )
    if not np.any(action_overlap):
        raise ValueError("edge evaluation has no action-overlap windows")
    attribution = _score_attribution(model, attribution_queries, windows.graph)
    artifact_bytes = json.dumps(
        model.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    latency = _batch_one_latency(model, windows)
    trajectory_norms = np.linalg.norm(
        prediction.reshape(
            len(prediction), prediction.shape[1], -1
        ),
        axis=2,
    )
    first_norm = np.maximum(trajectory_norms[:, :1], 1e-12)
    return EdgeModelScores(
        normalized_mse_overall=float(np.mean(squared_error)),
        normalized_mse_action_overlap=float(
            np.mean(squared_error[action_overlap])
        ),
        downstream_effect_mse=_downstream_effect_mse(
            prediction=prediction,
            observed=observed,
            windows=windows,
        ),
        action_and_target_hit_at_1=attribution[0],
        no_action_specificity=attribution[1],
        maximum_absolute_prediction=float(
            np.max(np.abs(prediction))
        ),
        maximum_horizon_norm_growth=float(
            np.max(trajectory_norms / first_norm)
        ),
        rollout_finite=bool(np.all(np.isfinite(prediction))),
        parameter_count=model.parameter_count,
        serialized_size_bytes=len(artifact_bytes),
        batch_one_latency_ms=latency,
    )


def persistence_scores(
    windows: ActionConditionedWindows,
) -> Mapping[str, float]:
    """Return the common forecast references for persistence."""

    prediction = persistence_rollout(
        windows.histories, windows.future_states.shape[1]
    ).mean
    observed = np.asarray(windows.future_states, dtype=np.float64)
    action_overlap = np.any(
        windows.future_actions[..., 1] > 0.5, axis=2
    )
    squared_error = np.square(prediction - observed)
    return {
        "normalized_mse_overall": float(np.mean(squared_error)),
        "normalized_mse_action_overlap": float(
            np.mean(squared_error[action_overlap])
        ),
        "downstream_effect_mse": _downstream_effect_mse(
            prediction=prediction,
            observed=observed,
            windows=windows,
        ),
    }


def conformal_sequential_detection(
    *,
    model: EdgeDynamicsModel,
    calibration: ActionConditionedWindows,
    evaluation: ActionConditionedWindows,
    alpha: float = 0.01,
) -> Mapping[str, Any]:
    """Calibrate hidden-action residuals and score sequential alarms."""

    if not 0.0 < alpha < 1.0:
        raise ValueError("conformal alpha must be in (0, 1)")
    calibration_scores = _hidden_action_one_step_scores(model, calibration)
    calibration_controls = _control_trajectory_ids(calibration)
    calibration_mask = np.asarray(
        [
            trajectory_id in calibration_controls
            for trajectory_id in calibration.trajectory_ids
        ],
        dtype=np.bool_,
    )
    reference = calibration_scores[calibration_mask]
    if len(reference) < 2:
        raise ValueError("conformal calibration needs control windows")
    point_threshold = float(
        np.quantile(reference, 1.0 - alpha, method="higher")
    )
    calibration_p = _empirical_upper_p_values(
        reference, calibration_scores
    )
    calibration_evidence = np.maximum(
        0.0, -np.log(calibration_p) - math.log(2.0)
    )
    calibration_maxima = _trajectory_cusum_maxima(
        calibration, calibration_evidence, calibration_controls
    )
    sequential_threshold = float(
        np.quantile(
            np.asarray(tuple(calibration_maxima.values())),
            1.0 - alpha,
            method="higher",
        )
    )
    evaluation_scores = _hidden_action_one_step_scores(model, evaluation)
    evaluation_p = _empirical_upper_p_values(
        reference, evaluation_scores
    )
    evaluation_evidence = np.maximum(
        0.0, -np.log(evaluation_p) - math.log(2.0)
    )
    trajectory_rows = _evaluate_trajectory_alarms(
        windows=evaluation,
        scores=evaluation_scores,
        evidence=evaluation_evidence,
        point_threshold=point_threshold,
        sequential_threshold=sequential_threshold,
    )
    control_rows = [
        row for row in trajectory_rows if not row["is_treatment"]
    ]
    treatment_rows = [
        row for row in trajectory_rows if row["is_treatment"]
    ]
    sequential_detected = [
        row
        for row in treatment_rows
        if row["post_onset_sequential_alarm_transition"] is not None
    ]
    point_detected = [
        row
        for row in treatment_rows
        if row["post_onset_point_alarm_transition"] is not None
    ]
    sequential_delays = [
        int(row["post_onset_sequential_alarm_transition"])
        - int(row["onset_transition"])
        for row in sequential_detected
    ]
    point_delays = [
        int(row["post_onset_point_alarm_transition"])
        - int(row["onset_transition"])
        for row in point_detected
    ]
    return {
        "schema_version": 1,
        "kind": "conformal_sequential_hidden_action_detection",
        "alpha": alpha,
        "calibration_control_window_count": len(reference),
        "point_threshold": point_threshold,
        "sequential_threshold": sequential_threshold,
        "evaluation_control_point_alarm_rate": float(
            np.mean(
                [
                    bool(row["any_point_alarm"])
                    for row in control_rows
                ]
            )
        ),
        "evaluation_control_sequential_false_alarm_rate": float(
            np.mean(
                [
                    bool(row["any_sequential_alarm"])
                    for row in control_rows
                ]
            )
        ),
        "evaluation_treatment_point_detection_rate": float(
            len(point_detected) / len(treatment_rows)
        ),
        "evaluation_treatment_sequential_detection_rate": float(
            len(sequential_detected) / len(treatment_rows)
        ),
        "evaluation_treatment_pre_onset_point_alarm_rate": float(
            np.mean(
                [
                    bool(row["pre_onset_point_alarm"])
                    for row in treatment_rows
                ]
            )
        ),
        "evaluation_treatment_pre_onset_sequential_alarm_rate": float(
            np.mean(
                [
                    bool(row["pre_onset_sequential_alarm"])
                    for row in treatment_rows
                ]
            )
        ),
        "median_point_detection_delay_transitions": (
            float(np.median(point_delays)) if point_delays else None
        ),
        "median_sequential_detection_delay_transitions": (
            float(np.median(sequential_delays))
            if sequential_delays
            else None
        ),
        "trajectory_rows": trajectory_rows,
        "limitations": [
            "open development evaluation, not sealed confirmation",
            "point scores are temporally dependent overlapping windows",
            "the action tensor is hidden from the detector",
        ],
    }


class CountMinSketch:
    """Small deterministic weighted Count-Min Sketch."""

    def __init__(self, *, width: int, depth: int, seed: int = 0) -> None:
        if width < 2 or depth < 1:
            raise ValueError("Count-Min Sketch dimensions are invalid")
        self.width = width
        self.depth = depth
        self.seed = seed
        self._counts = np.zeros((depth, width), dtype=np.float64)

    def update(self, key: str, value: float) -> None:
        if not key or not np.isfinite(value) or value < 0.0:
            raise ValueError("Count-Min Sketch update is invalid")
        for row in range(self.depth):
            self._counts[row, self._column(key, row)] += value

    def estimate(self, key: str) -> float:
        if not key:
            raise ValueError("Count-Min Sketch key cannot be empty")
        return float(
            min(
                self._counts[row, self._column(key, row)]
                for row in range(self.depth)
            )
        )

    @property
    def storage_bytes(self) -> int:
        return int(self._counts.nbytes)

    def columns(self, key: str) -> Tuple[int, ...]:
        """Return the deterministic bucket address for one key."""

        if not key:
            raise ValueError("Count-Min Sketch key cannot be empty")
        return tuple(
            self._column(key, row) for row in range(self.depth)
        )

    def _column(self, key: str, row: int) -> int:
        digest = hashlib.blake2b(
            f"{self.seed}:{row}:{key}".encode("utf-8"),
            digest_size=8,
        ).digest()
        return int.from_bytes(digest, "big") % self.width


def benchmark_event_sketch(
    windows: ActionConditionedWindows,
    compiler_artifact: Mapping[str, Any],
    *,
    width: int = 128,
    depth: int = 4,
) -> Mapping[str, Any]:
    """Sketch current structured event totals and measure collisions."""

    feature_positions = tuple(
        position
        for position, name in enumerate(windows.state_feature_names)
        if name
        in {
            "log_event_count",
            "log_error_count",
            "trace_span_count",
            "trace_error_count",
        }
    )
    if len(feature_positions) != 4:
        raise ValueError("event sketch requires four structured features")
    center, scale = _event_feature_scaling(
        windows, compiler_artifact, feature_positions
    )
    sketch = CountMinSketch(width=width, depth=depth, seed=31)
    exact: MutableMapping[str, float] = {}
    states = np.asarray(windows.histories[:, -1], dtype=np.float64)
    for entity_position, entity_id in enumerate(windows.entity_names):
        for feature_position in feature_positions:
            key = (
                f"{entity_id}:"
                f"{windows.state_feature_names[feature_position]}"
            )
            value = float(
                np.sum(
                    np.maximum(
                        (
                            states[
                                :,
                                entity_position,
                                feature_position,
                            ]
                            * scale[feature_position]
                            + center[feature_position]
                        ),
                        0.0,
                    )
                )
            )
            exact[key] = value
            sketch.update(key, value)
    errors = {
        key: sketch.estimate(key) - value
        for key, value in exact.items()
    }
    return {
        "schema_version": 1,
        "kind": "structured_event_count_min_sketch_benchmark",
        "width": width,
        "depth": depth,
        "key_count": len(exact),
        "storage_bytes": sketch.storage_bytes,
        "exact_dictionary_payload_bytes": sum(
            len(key.encode("utf-8")) + 8 for key in exact
        ),
        "maximum_overestimate": max(errors.values()),
        "mean_overestimate": float(np.mean(tuple(errors.values()))),
        "exact_reconstruction_rate": float(
            np.mean([error == 0.0 for error in errors.values()])
        ),
        "bounded_interpretation": (
            "compiled observed event counts are reconstructed exactly, "
            "but the vocabulary is too small to establish a "
            "high-cardinality advantage"
        ),
    }


def sketch_event_predictor_effect(
    *,
    model: EdgeDynamicsModel,
    windows: ActionConditionedWindows,
    compiler_artifact: Mapping[str, Any],
    width: int = 128,
    depth: int = 4,
) -> Mapping[str, Any]:
    """Round-trip event inputs through the current collision-free address."""

    feature_positions = tuple(
        position
        for position, name in enumerate(windows.state_feature_names)
        if name
        in {
            "log_event_count",
            "log_error_count",
            "trace_span_count",
            "trace_error_count",
        }
    )
    center, scale = _event_feature_scaling(
        windows, compiler_artifact, feature_positions
    )
    keyed_positions = tuple(
        (
            f"{entity_id}:{windows.state_feature_names[position]}",
            entity_position,
            feature_position,
        )
        for entity_position, entity_id in enumerate(windows.entity_names)
        for feature_position, position in enumerate(feature_positions)
    )
    keys = tuple(
        key
        for key, _, _ in keyed_positions
    )
    event_history = np.asarray(
        windows.histories[..., feature_positions],
        dtype=np.float64,
    )
    raw_event_history = np.maximum(
        event_history
        * scale[np.asarray(feature_positions)][None, None, None, :]
        + center[np.asarray(feature_positions)][None, None, None, :],
        0.0,
    )
    decoded = np.full_like(raw_event_history, np.inf)
    sketch = CountMinSketch(width=width, depth=depth, seed=31)
    addresses = {
        key: sketch.columns(key)
        for key in keys
    }
    for row in range(depth):
        buckets = np.zeros(
            (
                len(windows.histories),
                windows.histories.shape[1],
                width,
            ),
            dtype=np.float64,
        )
        for key, entity_position, feature_position in keyed_positions:
            buckets[..., addresses[key][row]] += raw_event_history[
                ..., entity_position, feature_position
            ]
        for key, entity_position, feature_position in keyed_positions:
            np.minimum(
                decoded[..., entity_position, feature_position],
                buckets[..., addresses[key][row]],
                out=decoded[..., entity_position, feature_position],
            )
    reconstructed = (
        decoded
        - center[np.asarray(feature_positions)][None, None, None, :]
    ) / scale[np.asarray(feature_positions)][None, None, None, :]
    reconstructed_histories = np.array(
        windows.histories, copy=True
    )
    reconstructed_histories[..., feature_positions] = reconstructed
    round_tripped = ActionConditionedWindows(
        histories=reconstructed_histories,
        future_states=windows.future_states,
        future_controls=windows.future_controls,
        future_actions=windows.future_actions,
        trajectory_ids=windows.trajectory_ids,
        matched_pair_ids=windows.matched_pair_ids,
        transition_indices=windows.transition_indices,
        entity_names=windows.entity_names,
        state_feature_names=windows.state_feature_names,
        control_feature_names=windows.control_feature_names,
        action_feature_names=windows.action_feature_names,
        graph=windows.graph,
    )
    collision_free = all(
        any(
            sum(
                other_columns[row] == columns[row]
                for other_columns in addresses.values()
            )
            == 1
            for row in range(depth)
        )
        for columns in addresses.values()
    )
    reconstruction_error = float(
        np.max(np.abs(reconstructed - event_history))
    )
    return {
        "collision_free_key_address": collision_free,
        "input_reconstruction_max_abs_error": reconstruction_error,
        "selected_predictor_metrics_after_round_trip": dict(
            forecast_objective(model, round_tripped)
        ),
    }


_BODY_PATTERN = re.compile(rb'"body":\{"stringValue":"([^"]+)"\}')
_VARIABLE_TOKEN = re.compile(
    r"^(?:[-+]?\d+(?:\.\d+)?|"
    r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}|"
    r"[0-9a-f]{16,}|"
    r"(?:\d{1,3}\.){3}\d{1,3})$",
    re.IGNORECASE,
)


def audit_streaming_log_templates(
    corpus_directory: Path,
) -> Mapping[str, Any]:
    """Stream raw OTLP JSONL bodies through a Drain-like tokenizer."""

    started = time.perf_counter()
    template_counts: Dict[str, int] = {}
    message_count = 0
    files = sorted(
        (Path(corpus_directory) / "cases").glob(
            "*/collector-logs.jsonl"
        )
    )
    for path in files:
        with path.open("rb") as handle:
            for line in handle:
                for match in _BODY_PATTERN.finditer(line):
                    body = match.group(1).decode(
                        "utf-8", errors="replace"
                    )
                    template = " ".join(
                        "<*>"
                        if _VARIABLE_TOKEN.match(token)
                        else token
                        for token in body.split()
                    )
                    template_counts[template] = (
                        template_counts.get(template, 0) + 1
                    )
                    message_count += 1
    elapsed = time.perf_counter() - started
    return {
        "schema_version": 1,
        "kind": "streaming_log_template_audit",
        "file_count": len(files),
        "message_count": message_count,
        "template_count": len(template_counts),
        "template_counts": dict(sorted(template_counts.items())),
        "template_payload_bytes": sum(
            len(template.encode("utf-8")) + 8
            for template in template_counts
        ),
        "elapsed_seconds": elapsed,
        "messages_per_second": (
            message_count / elapsed if elapsed > 0.0 else None
        ),
        "bounded_interpretation": (
            "parser plumbing only; three fixed messages cannot establish "
            "natural-language template generalization"
        ),
    }


def _event_feature_scaling(
    windows: ActionConditionedWindows,
    compiler_artifact: Mapping[str, Any],
    feature_positions: Sequence[int],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    state = dict(compiler_artifact["state"])
    center = np.asarray(state["state_center"], dtype=np.float64)
    scale = np.asarray(state["state_scale"], dtype=np.float64)
    if (
        center.shape != (len(windows.state_feature_names),)
        or scale.shape != center.shape
        or any(
            position < 0 or position >= len(center)
            for position in feature_positions
        )
    ):
        raise ValueError("event feature scaling does not align")
    return center, scale


def write_edge_experiment_artifacts(
    *,
    output_directory: Path,
    report: Mapping[str, Any],
    model_artifacts: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Write one immutable adjacent-technique result bundle."""

    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite edge experiments: {output}"
        )
    output.mkdir(parents=True)
    models = output / "models"
    models.mkdir()
    (output / "results.json").write_text(_pretty_json(report))
    (output / "report.md").write_text(_markdown_report(report))
    for name, artifact in model_artifacts.items():
        (models / f"{name}.json").write_text(_pretty_json(artifact))
    hashes = {
        path.relative_to(output).as_posix(): _file_sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "schema_version": 1,
        "kind": "edge_dynamics_development_artifact_manifest",
        "sha256": hashes,
    }
    (output / "artifact-manifest.json").write_text(
        _pretty_json(manifest)
    )
    return manifest


def _score_attribution(
    model: EdgeDynamicsModel,
    queries: PreparedAttributionQueries,
    graph: DeclaredTelemetryGraph,
) -> Tuple[float, float]:
    query_count = len(queries.query_ids)
    candidate_count = len(queries.candidate_ids)
    distribution = model.rollout(
        np.repeat(
            queries.histories[:, None, ...],
            candidate_count,
            axis=1,
        ).reshape(
            query_count * candidate_count,
            *queries.histories.shape[1:],
        ),
        np.repeat(
            queries.future_controls[:, None, ...],
            candidate_count,
            axis=1,
        ).reshape(
            query_count * candidate_count,
            *queries.future_controls.shape[1:],
        ),
        queries.candidate_actions.reshape(
            query_count * candidate_count,
            *queries.candidate_actions.shape[2:],
        ),
        graph,
    )
    observed = np.repeat(
        queries.observed_future[:, None, ...],
        candidate_count,
        axis=1,
    ).reshape(
        query_count * candidate_count,
        *queries.observed_future.shape[1:],
    )
    nll = distribution.negative_log_likelihood(observed).reshape(
        query_count, candidate_count
    )
    winners = np.argmin(nll, axis=1)
    treatment_hits = []
    control_hits = []
    for query_index, raw_winner in enumerate(winners):
        winner = int(raw_winner)
        expected_kind = queries.expected_action_kinds[query_index]
        if not expected_kind:
            control_hits.append(queries.candidate_ids[winner] == "no_action")
        else:
            treatment_hits.append(
                queries.candidate_action_kinds[winner] == expected_kind
                and queries.candidate_target_entities[winner]
                == queries.expected_target_entities[query_index]
            )
    return (
        float(np.mean(treatment_hits)),
        float(np.mean(control_hits)),
    )


def _select_features(
    prediction: NDArray[np.float64],
    observed: NDArray[np.float64],
    positions: Sequence[int],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    if not positions:
        return prediction, observed
    return prediction[..., positions], observed[..., positions]


def _downstream_effect_mse(
    *,
    prediction: NDArray[np.float64],
    observed: NDArray[np.float64],
    windows: ActionConditionedWindows,
) -> float:
    index_by_key = {
        (case_id, int(transition)): index
        for index, (case_id, transition) in enumerate(
            zip(windows.trajectory_ids, windows.transition_indices)
        )
    }
    trajectory_action = _trajectory_action_entities(windows)
    pair_trajectories: Dict[str, list[str]] = {}
    for trajectory_id, pair_id in zip(
        windows.trajectory_ids, windows.matched_pair_ids
    ):
        if trajectory_id not in pair_trajectories.setdefault(pair_id, []):
            pair_trajectories[pair_id].append(trajectory_id)
    squared_errors = []
    for pair_id, trajectories in pair_trajectories.items():
        treatments = [
            value for value in trajectories if value in trajectory_action
        ]
        controls = [
            value for value in trajectories if value not in trajectory_action
        ]
        if len(treatments) != 1 or len(controls) != 1:
            continue
        treatment_id = treatments[0]
        control_id = controls[0]
        target_entity = trajectory_action[treatment_id]
        downstream = _downstream_positions(windows.graph, target_entity)
        if not downstream:
            continue
        treatment_indices = [
            index
            for index, trajectory_id in enumerate(windows.trajectory_ids)
            if trajectory_id == treatment_id
        ]
        for treatment_index in treatment_indices:
            transition = int(
                windows.transition_indices[treatment_index]
            )
            control_index = index_by_key.get((control_id, transition))
            if control_index is None:
                continue
            active = np.any(
                windows.future_actions[treatment_index, ..., 1] > 0.5,
                axis=1,
            )
            if not np.any(active):
                continue
            predicted_effect = (
                prediction[treatment_index]
                - prediction[control_index]
            )
            observed_effect = (
                observed[treatment_index] - observed[control_index]
            )
            squared_errors.append(
                np.square(
                    predicted_effect[active][:, downstream]
                    - observed_effect[active][:, downstream]
                ).reshape(-1)
            )
    if not squared_errors:
        return float("nan")
    return float(np.mean(np.concatenate(squared_errors)))


def _trajectory_action_entities(
    windows: ActionConditionedWindows,
) -> Mapping[str, str]:
    values: Dict[str, str] = {}
    for index, trajectory_id in enumerate(windows.trajectory_ids):
        active = np.argwhere(
            windows.future_actions[index, ..., 1] > 0.5
        )
        if len(active):
            values[trajectory_id] = windows.entity_names[int(active[0, 1])]
    return values


def _downstream_positions(
    graph: DeclaredTelemetryGraph, start_entity: str
) -> Tuple[int, ...]:
    adjacency: Dict[str, list[str]] = {
        entity_id: [] for entity_id in graph.entity_ids
    }
    for entity in graph.entities:
        if entity.kind == "edge":
            assert entity.source is not None
            assert entity.target is not None
            adjacency[entity.source].append(entity.entity_id)
            adjacency[entity.entity_id].append(entity.target)
    discovered = []
    frontier = list(adjacency[start_entity])
    while frontier:
        candidate = frontier.pop(0)
        if candidate in discovered or candidate == start_entity:
            continue
        discovered.append(candidate)
        frontier.extend(adjacency[candidate])
    return tuple(graph.entity_ids.index(value) for value in discovered)


def _batch_one_latency(
    model: EdgeDynamicsModel,
    windows: ActionConditionedWindows,
) -> float:
    history = windows.histories[:1]
    controls = windows.future_controls[:1]
    actions = windows.future_actions[:1]
    model.rollout(history, controls, actions, windows.graph)
    timings = []
    for _ in range(20):
        started = time.perf_counter_ns()
        model.rollout(history, controls, actions, windows.graph)
        timings.append((time.perf_counter_ns() - started) / 1e6)
    return float(np.median(timings))


def _hidden_action_one_step_scores(
    model: EdgeDynamicsModel,
    windows: ActionConditionedWindows,
) -> NDArray[np.float64]:
    hidden_actions = np.zeros_like(windows.future_actions)
    hidden_actions[..., 0] = 1.0
    prediction = model.rollout(
        windows.histories,
        windows.future_controls,
        hidden_actions,
        windows.graph,
    ).mean[:, 0]
    observed = np.asarray(windows.future_states[:, 0], dtype=np.float64)
    return np.asarray(
        np.mean(np.square(prediction - observed), axis=(1, 2)),
        dtype=np.float64,
    )


def _control_trajectory_ids(
    windows: ActionConditionedWindows,
) -> set[str]:
    treatments = set(_trajectory_action_entities(windows))
    return set(windows.trajectory_ids) - treatments


def _empirical_upper_p_values(
    reference: NDArray[np.float64],
    scores: NDArray[np.float64],
) -> NDArray[np.float64]:
    return np.asarray(
        [
            (1.0 + float(np.sum(reference >= score)))
            / (len(reference) + 1.0)
            for score in scores
        ],
        dtype=np.float64,
    )


def _trajectory_cusum_maxima(
    windows: ActionConditionedWindows,
    evidence: NDArray[np.float64],
    trajectory_ids: set[str],
) -> Mapping[str, float]:
    maxima: Dict[str, float] = {}
    for trajectory_id in sorted(trajectory_ids):
        positions = [
            index
            for index, value in enumerate(windows.trajectory_ids)
            if value == trajectory_id
        ]
        positions.sort(
            key=lambda index: int(windows.transition_indices[index])
        )
        cumulative = 0.0
        maximum = 0.0
        for position in positions:
            cumulative = max(0.0, cumulative + evidence[position])
            maximum = max(maximum, cumulative)
        maxima[trajectory_id] = maximum
    return maxima


def _evaluate_trajectory_alarms(
    *,
    windows: ActionConditionedWindows,
    scores: NDArray[np.float64],
    evidence: NDArray[np.float64],
    point_threshold: float,
    sequential_threshold: float,
) -> list[Dict[str, Any]]:
    action_entities = _trajectory_action_entities(windows)
    rows = []
    for trajectory_id in sorted(set(windows.trajectory_ids)):
        positions = [
            index
            for index, value in enumerate(windows.trajectory_ids)
            if value == trajectory_id
        ]
        positions.sort(
            key=lambda index: int(windows.transition_indices[index])
        )
        is_treatment = trajectory_id in action_entities
        onset = None
        if is_treatment:
            active_positions = [
                position
                for position in positions
                if np.any(
                    windows.future_actions[position, 0, :, 1] > 0.5
                )
            ]
            if active_positions:
                onset = int(
                    windows.transition_indices[active_positions[0]]
                )
        cumulative = 0.0
        point_alarms = []
        sequential_alarms = []
        for position in positions:
            cumulative = max(0.0, cumulative + evidence[position])
            if scores[position] > point_threshold:
                point_alarms.append(
                    int(windows.transition_indices[position])
                )
            if cumulative > sequential_threshold:
                sequential_alarms.append(
                    int(windows.transition_indices[position])
                )
        post_onset_point = (
            next(
                (
                    transition
                    for transition in point_alarms
                    if onset is not None and transition >= onset
                ),
                None,
            )
            if onset is not None
            else None
        )
        post_onset_sequential = (
            next(
                (
                    transition
                    for transition in sequential_alarms
                    if onset is not None and transition >= onset
                ),
                None,
            )
            if onset is not None
            else None
        )
        rows.append(
            {
                "trajectory_id": trajectory_id,
                "is_treatment": is_treatment,
                "onset_transition": onset,
                "any_point_alarm": bool(point_alarms),
                "any_sequential_alarm": bool(sequential_alarms),
                "pre_onset_point_alarm": bool(
                    onset is not None
                    and any(
                        transition < onset
                        for transition in point_alarms
                    )
                ),
                "pre_onset_sequential_alarm": bool(
                    onset is not None
                    and any(
                        transition < onset
                        for transition in sequential_alarms
                    )
                ),
                "first_point_alarm_transition": (
                    point_alarms[0] if point_alarms else None
                ),
                "first_sequential_alarm_transition": (
                    sequential_alarms[0]
                    if sequential_alarms
                    else None
                ),
                "post_onset_point_alarm_transition": (
                    post_onset_point
                ),
                "post_onset_sequential_alarm_transition": (
                    post_onset_sequential
                ),
            }
        )
    return rows


def _pretty_json(payload: Mapping[str, Any]) -> str:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def _markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Edge dynamics development v1 result",
        "",
        "Open development evidence only; not sealed confirmation or a "
        "world-model claim.",
        "",
        "## Selected model",
        "",
        f"`{report.get('selected_model', 'none')}`",
        "",
        "## Evaluation scores",
        "",
        "| Model | Action MSE | Overall MSE | Downstream effect MSE | "
        "Hit@1 | No-action specificity | Parameters | Latency ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    raw_scores = report.get("evaluation_scores", {})
    if isinstance(raw_scores, dict):
        for name, raw in raw_scores.items():
            if not isinstance(raw, dict):
                continue
            lines.append(
                f"| {name} | "
                f"{float(raw['normalized_mse_action_overlap']):.4f} | "
                f"{float(raw['normalized_mse_overall']):.4f} | "
                f"{float(raw['downstream_effect_mse']):.4f} | "
                f"{float(raw['action_and_target_hit_at_1']):.3f} | "
                f"{float(raw['no_action_specificity']):.3f} | "
                f"{int(raw['parameter_count'])} | "
                f"{float(raw['batch_one_latency_ms']):.3f} |"
            )
    lines.extend(["", "## Evidence boundary", ""])
    lines.extend(
        [
            "- Existing development evaluation influenced this redesign.",
            "- Log vocabulary has only three fixed application templates.",
            "- A fresh sealed corpus is required for confirmation.",
            "",
        ]
    )
    return "\n".join(lines)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
