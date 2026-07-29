"""Retained reproduction runner for the event-native trace JEPA prototype.

Question: do causally completed, trace-linked span paths add held-out-topology
alert or closed-library investigation value beyond metrics-only, binned-event,
alignment-shuffled, and event n-gram controls?

Run with a fresh output directory:
    .venv/bin/python lab/action_dynamics/prototype_event_native_trace_jepa.py

This remains non-production experiment code. It is retained with the immutable
result artifact so the reported experiment can be reproduced.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import time
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from quantis_core.edge_dynamics.data import (
    load_edge_dynamics_cache,
    partition_worker_topology,
    source_artifact_manifest_sha256,
    topology_transfer_cache_address,
    validate_topology_transfer_cache,
)
from quantis_core.edge_dynamics.models import (
    ContractiveLowRankDynamics,
    LowRankConfig,
    MaskedInputDynamics,
)
SEED = 211
CONTEXT_LENGTH = 20
HORIZON = 10
MAX_SPANS = 8
TOKEN_DIMENSION = 48
LATENT_DIMENSION = 32
ENTITY_IDS = (
    "api",
    "api_enqueues_queue",
    "checkout_queue",
    "queue_dequeues_to_worker",
    "worker_pool",
    "worker_writes_postgresql",
    "postgresql",
)


@dataclass(frozen=True)
class RawSpan:
    template: str
    entity: str
    outcome: int
    start_nano: int
    end_nano: int
    depth: int
    gap_from_previous_ms: float
    time_to_next_ms: float


@dataclass(frozen=True)
class RawTrace:
    case_id: str
    trace_id: str
    window_index: int
    spans: tuple[RawSpan, ...]


@dataclass(frozen=True)
class CaseEvents:
    case_id: str
    pair_id: str
    worker_replicas: int
    point_count: int
    is_treatment: bool
    action_kind: str
    action_onset: Optional[int]
    traces: tuple[RawTrace, ...]
    raw_span_count: int
    retained_span_count: int
    truncated_span_count: int
    incomplete_parent_count: int
    drain_trace_count: int


@dataclass(frozen=True)
class Vocabulary:
    templates: tuple[str, ...]
    entities: tuple[str, ...]

    @property
    def template_to_id(self) -> Mapping[str, int]:
        return {
            value: index + 2 for index, value in enumerate(self.templates)
        }

    @property
    def entity_to_id(self) -> Mapping[str, int]:
        return {
            value: index + 2 for index, value in enumerate(self.entities)
        }

    @property
    def template_class_count(self) -> int:
        return len(self.templates) + 2

    @property
    def entity_class_count(self) -> int:
        return len(self.entities) + 2

    @property
    def template_mask_id(self) -> int:
        return self.template_class_count

    @property
    def entity_mask_id(self) -> int:
        return self.entity_class_count


@dataclass(frozen=True)
class NumericScale:
    center: np.ndarray
    scale: np.ndarray
    next_center: float
    next_scale: float


@dataclass(frozen=True)
class TraceTensors:
    template_ids: torch.Tensor
    entity_ids: torch.Tensor
    outcome_ids: torch.Tensor
    depth_ids: torch.Tensor
    numeric: torch.Tensor
    next_time: torch.Tensor
    valid: torch.Tensor
    next_valid: torch.Tensor
    traces: tuple[RawTrace, ...]

    def subset(self, indices: np.ndarray) -> "TraceTensors":
        tensor_indices = torch.as_tensor(indices, dtype=torch.long)
        return TraceTensors(
            template_ids=self.template_ids[tensor_indices],
            entity_ids=self.entity_ids[tensor_indices],
            outcome_ids=self.outcome_ids[tensor_indices],
            depth_ids=self.depth_ids[tensor_indices],
            numeric=self.numeric[tensor_indices],
            next_time=self.next_time[tensor_indices],
            valid=self.valid[tensor_indices],
            next_valid=self.next_valid[tensor_indices],
            traces=tuple(self.traces[index] for index in indices),
        )


@dataclass(frozen=True)
class LatentDataset:
    histories: np.ndarray
    targets: np.ndarray
    future_controls: np.ndarray
    future_actions: np.ndarray
    trajectory_ids: tuple[str, ...]
    pair_ids: tuple[str, ...]
    transition_indices: np.ndarray


class PCATransform:
    """Small NumPy-only standardized PCA."""

    def __init__(self, rank: int) -> None:
        self.rank = rank
        self.center: Optional[np.ndarray] = None
        self.scale: Optional[np.ndarray] = None
        self.components: Optional[np.ndarray] = None
        self.latent_center: Optional[np.ndarray] = None
        self.latent_scale: Optional[np.ndarray] = None
        self.explained_variance_ratio = 0.0

    def fit(self, values: np.ndarray) -> "PCATransform":
        matrix = np.asarray(values, dtype=np.float64)
        self.center = np.mean(matrix, axis=0)
        raw_scale = np.std(matrix, axis=0)
        self.scale = np.where(raw_scale > 1e-8, raw_scale, 1.0)
        standardized = (matrix - self.center) / self.scale
        covariance = standardized.T @ standardized / max(
            len(standardized) - 1, 1
        )
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        keep = order[: min(self.rank, len(order))]
        self.components = eigenvectors[:, keep]
        retained = np.maximum(eigenvalues[keep], 0.0)
        total = float(np.sum(np.maximum(eigenvalues, 0.0)))
        self.explained_variance_ratio = (
            float(np.sum(retained) / total) if total > 0.0 else 0.0
        )
        latent = standardized @ self.components
        self.latent_center = np.mean(latent, axis=0)
        latent_scale = np.std(latent, axis=0)
        self.latent_scale = np.where(latent_scale > 1e-8, latent_scale, 1.0)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if any(
            value is None
            for value in (
                self.center,
                self.scale,
                self.components,
                self.latent_center,
                self.latent_scale,
            )
        ):
            raise ValueError("PCA transform is not fitted")
        standardized = (
            np.asarray(values, dtype=np.float64) - self.center
        ) / self.scale
        latent = standardized @ self.components
        return np.asarray(
            (latent - self.latent_center) / self.latent_scale,
            dtype=np.float32,
        )


class MaskedTraceEncoder(nn.Module):
    """Trace-path encoder with masked categorical and next-time heads."""

    def __init__(self, vocabulary: Vocabulary) -> None:
        super().__init__()
        self.vocabulary = vocabulary
        self.template_embedding = nn.Embedding(
            vocabulary.template_class_count + 1,
            16,
            padding_idx=0,
        )
        self.entity_embedding = nn.Embedding(
            vocabulary.entity_class_count + 1,
            12,
            padding_idx=0,
        )
        self.outcome_embedding = nn.Embedding(4, 8, padding_idx=0)
        self.depth_embedding = nn.Embedding(10, 8, padding_idx=0)
        self.numeric_projection = nn.Linear(3, 12)
        self.input_projection = nn.Linear(56, TOKEN_DIMENSION)
        self.trace_token = nn.Parameter(
            torch.zeros(1, 1, TOKEN_DIMENSION)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=TOKEN_DIMENSION,
            nhead=4,
            dim_feedforward=96,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=2,
            enable_nested_tensor=False,
        )
        self.normalization = nn.LayerNorm(TOKEN_DIMENSION)
        self.template_head = nn.Linear(
            TOKEN_DIMENSION, vocabulary.template_class_count
        )
        self.entity_head = nn.Linear(
            TOKEN_DIMENSION, vocabulary.entity_class_count
        )
        self.outcome_head = nn.Linear(TOKEN_DIMENSION, 3)
        self.next_time_head = nn.Linear(TOKEN_DIMENSION, 1)
        nn.init.normal_(self.trace_token, std=0.02)

    def forward(
        self,
        template_ids: torch.Tensor,
        entity_ids: torch.Tensor,
        outcome_ids: torch.Tensor,
        depth_ids: torch.Tensor,
        numeric: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = torch.cat(
            (
                self.template_embedding(template_ids),
                self.entity_embedding(entity_ids),
                self.outcome_embedding(outcome_ids),
                self.depth_embedding(depth_ids),
                self.numeric_projection(numeric),
            ),
            dim=-1,
        )
        tokens = self.input_projection(embedded)
        trace_token = self.trace_token.expand(len(tokens), -1, -1)
        values = torch.cat((trace_token, tokens), dim=1)
        padding = torch.cat(
            (
                torch.zeros(
                    (len(valid), 1),
                    dtype=torch.bool,
                    device=valid.device,
                ),
                ~valid,
            ),
            dim=1,
        )
        encoded = self.normalization(
            self.encoder(values, src_key_padding_mask=padding)
        )
        return encoded[:, 0], encoded[:, 1:]


class LatentPredictor(nn.Module):
    """Action-conditioned recurrent future-latent predictor."""

    def __init__(
        self,
        *,
        latent_dimension: int,
        control_dimension: int,
        action_dimension: int,
    ) -> None:
        super().__init__()
        self.latent_dimension = latent_dimension
        self.context = nn.GRU(
            latent_dimension, 64, batch_first=True
        )
        self.action_projection = nn.Sequential(
            nn.Linear(action_dimension, 32),
            nn.GELU(),
        )
        self.horizon_embedding = nn.Embedding(HORIZON, 8)
        self.predictor = nn.GRUCell(32 + control_dimension + 8, 64)
        self.output = nn.Linear(64, latent_dimension)

    def encode_context(self, histories: torch.Tensor) -> torch.Tensor:
        _, hidden = self.context(histories)
        return hidden[-1]

    def forward(
        self,
        histories: torch.Tensor,
        future_controls: torch.Tensor,
        future_actions: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.encode_context(histories)
        outputs = []
        for horizon in range(future_controls.shape[1]):
            action = self.action_projection(
                future_actions[:, horizon].flatten(1)
            )
            horizon_ids = torch.full(
                (len(histories),),
                horizon,
                dtype=torch.long,
                device=histories.device,
            )
            conditioned = torch.cat(
                (
                    action,
                    future_controls[:, horizon],
                    self.horizon_embedding(horizon_ids),
                ),
                dim=1,
            )
            hidden = self.predictor(conditioned, hidden)
            outputs.append(self.output(hidden))
        return torch.stack(outputs, dim=1)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def _read_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _attribute_value(value: Mapping[str, Any]) -> Any:
    for key in (
        "stringValue",
        "intValue",
        "doubleValue",
        "boolValue",
    ):
        if key in value:
            raw = value[key]
            return int(raw) if key == "intValue" else raw
    return None


def _attributes(values: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    return {
        str(attribute["key"]): _attribute_value(attribute["value"])
        for attribute in values
    }


def _wall_clock_window_boundaries(
    path: Path,
    *,
    point_count: int,
    period_nano: int,
) -> np.ndarray:
    starts = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        for resource_logs in payload.get("resourceLogs", []):
            for scope_logs in resource_logs.get("scopeLogs", []):
                for record in scope_logs.get("logRecords", []):
                    attributes = _attributes(
                        record.get("attributes", [])
                    )
                    if (
                        attributes.get("event.name")
                        == "action.run.boundary"
                        and attributes.get("quantis.run.phase")
                        == "started"
                    ):
                        starts.append(int(record["timeUnixNano"]))
    if len(starts) != 1:
        raise ValueError(
            f"run start boundary is not unique: {path}"
        )
    return starts[0] + period_nano * np.arange(
        1, point_count + 1, dtype=np.int64
    )


def _raw_spans(path: Path) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        for resource_spans in payload.get("resourceSpans", []):
            for scope_spans in resource_spans.get("scopeSpans", []):
                for span in scope_spans.get("spans", []):
                    attributes = _attributes(span.get("attributes", []))
                    trace_id = str(span.get("traceId", ""))
                    span_id = str(span.get("spanId", ""))
                    name = str(span.get("name", ""))
                    entity = attributes.get("quantis.graph.entity.id")
                    status = span.get("status", {})
                    status_code = (
                        status.get("code", 0)
                        if isinstance(status, dict)
                        else 0
                    )
                    if (
                        not trace_id
                        or not span_id
                        or not name
                        or not isinstance(entity, str)
                        or entity not in ENTITY_IDS
                        or status_code not in (1, 2)
                    ):
                        raise ValueError(
                            "trace span lacks a bounded event identity at "
                            f"{path}:{line_number}"
                        )
                    spans.append(
                        {
                            "trace_id": trace_id,
                            "span_id": span_id,
                            "parent_span_id": str(
                                span.get("parentSpanId", "")
                            ),
                            "template": name,
                            "entity": entity,
                            "outcome": int(status_code == 2),
                            "start": int(span["startTimeUnixNano"]),
                            "end": int(span["endTimeUnixNano"]),
                        }
                    )
    if not spans:
        raise ValueError(f"trace capture is empty: {path}")
    return spans


def _trace_depths(
    values: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, int], int]:
    by_id = {str(value["span_id"]): value for value in values}
    depths: dict[str, int] = {}
    missing = 0

    def depth(span_id: str, visiting: set[str]) -> int:
        nonlocal missing
        if span_id in depths:
            return depths[span_id]
        if span_id in visiting:
            return 0
        value = by_id[span_id]
        parent = str(value["parent_span_id"])
        if not parent:
            result = 0
        elif parent not in by_id:
            missing += 1
            result = 0
        else:
            result = min(depth(parent, visiting | {span_id}) + 1, 8)
        depths[span_id] = result
        return result

    for span_id in by_id:
        depth(span_id, set())
    return depths, missing


def _load_case(case_directory: Path) -> CaseEvents:
    manifest = _read_json(case_directory / "capture-manifest.json")
    action_case = dict(manifest["action_case"])
    case_id = str(action_case["case_id"])
    point_count = int(action_case["point_count"])
    actions = list(action_case["actions"])
    boundaries = _wall_clock_window_boundaries(
        case_directory / "collector-actions.jsonl",
        point_count=point_count,
        period_nano=int(action_case["logical_window_period_nano"]),
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    raw = _raw_spans(case_directory / "collector-traces.jsonl")
    for span in raw:
        grouped[str(span["trace_id"])].append(span)
    traces: list[RawTrace] = []
    retained_span_count = 0
    truncated_span_count = 0
    incomplete_parent_count = 0
    drain_trace_count = 0
    for trace_id in sorted(grouped):
        values = grouped[trace_id]
        depths, missing = _trace_depths(values)
        incomplete_parent_count += missing
        ordered = sorted(
            values,
            key=lambda value: (
                int(value["start"]),
                depths[str(value["span_id"])],
                str(value["span_id"]),
            ),
        )
        final_end = max(int(value["end"]) for value in ordered)
        window = int(np.searchsorted(boundaries, final_end, side="left"))
        if window >= point_count:
            drain_trace_count += 1
            continue
        if len(ordered) > MAX_SPANS:
            truncated_span_count += len(ordered) - MAX_SPANS
        ordered = ordered[:MAX_SPANS]
        spans: list[RawSpan] = []
        for index, value in enumerate(ordered):
            previous_end = (
                int(ordered[index - 1]["end"]) if index else int(value["start"])
            )
            next_start = (
                int(ordered[index + 1]["start"])
                if index + 1 < len(ordered)
                else int(value["end"])
            )
            spans.append(
                RawSpan(
                    template=str(value["template"]),
                    entity=str(value["entity"]),
                    outcome=int(value["outcome"]),
                    start_nano=int(value["start"]),
                    end_nano=int(value["end"]),
                    depth=depths[str(value["span_id"])],
                    gap_from_previous_ms=max(
                        0.0,
                        (int(value["start"]) - previous_end) / 1e6,
                    ),
                    time_to_next_ms=max(
                        0.0,
                        (next_start - int(value["end"])) / 1e6,
                    ),
                )
            )
        retained_span_count += len(spans)
        traces.append(
            RawTrace(
                case_id=case_id,
                trace_id=trace_id,
                window_index=window,
                spans=tuple(spans),
            )
        )
    action = actions[0] if actions else None
    return CaseEvents(
        case_id=case_id,
        pair_id=str(action_case["matched_pair_id"]),
        worker_replicas=int(action_case["worker_replicas"]),
        point_count=point_count,
        is_treatment=bool(actions),
        action_kind=str(action["action_kind"]) if action else "",
        action_onset=int(action["start_index"]) if action else None,
        traces=tuple(traces),
        raw_span_count=len(raw),
        retained_span_count=retained_span_count,
        truncated_span_count=truncated_span_count,
        incomplete_parent_count=incomplete_parent_count,
        drain_trace_count=drain_trace_count,
    )


def _fit_vocabulary(cases: Iterable[CaseEvents]) -> Vocabulary:
    templates = sorted(
        {
            span.template
            for case in cases
            for trace in case.traces
            for span in trace.spans
        }
    )
    entities = sorted(
        {
            span.entity
            for case in cases
            for trace in case.traces
            for span in trace.spans
        }
    )
    if not templates or not entities:
        raise ValueError("fitting event vocabulary is empty")
    return Vocabulary(tuple(templates), tuple(entities))


def _fit_numeric_scale(cases: Iterable[CaseEvents]) -> NumericScale:
    numeric = []
    next_times = []
    for case in cases:
        for trace in case.traces:
            for index, span in enumerate(trace.spans):
                numeric.append(
                    (
                        math.log1p(
                            max(span.end_nano - span.start_nano, 0) / 1e6
                        ),
                        math.log1p(span.gap_from_previous_ms),
                        span.depth / 8.0,
                    )
                )
                if index + 1 < len(trace.spans):
                    next_times.append(math.log1p(span.time_to_next_ms))
    matrix = np.asarray(numeric, dtype=np.float64)
    center = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    next_values = np.asarray(next_times, dtype=np.float64)
    next_center = float(np.mean(next_values))
    next_scale = float(np.std(next_values))
    return NumericScale(
        center=center,
        scale=scale,
        next_center=next_center,
        next_scale=next_scale if next_scale > 1e-8 else 1.0,
    )


def _compile_trace_tensors(
    cases: Sequence[CaseEvents],
    vocabulary: Vocabulary,
    numeric_scale: NumericScale,
) -> TraceTensors:
    traces = tuple(
        trace for case in cases for trace in case.traces
    )
    count = len(traces)
    template_ids = np.zeros((count, MAX_SPANS), dtype=np.int64)
    entity_ids = np.zeros_like(template_ids)
    outcome_ids = np.zeros_like(template_ids)
    depth_ids = np.zeros_like(template_ids)
    numeric = np.zeros((count, MAX_SPANS, 3), dtype=np.float32)
    next_time = np.zeros((count, MAX_SPANS), dtype=np.float32)
    valid = np.zeros((count, MAX_SPANS), dtype=np.bool_)
    next_valid = np.zeros_like(valid)
    template_map = vocabulary.template_to_id
    entity_map = vocabulary.entity_to_id
    for trace_index, trace in enumerate(traces):
        for span_index, span in enumerate(trace.spans):
            template_ids[trace_index, span_index] = template_map.get(
                span.template, 1
            )
            entity_ids[trace_index, span_index] = entity_map.get(
                span.entity, 1
            )
            outcome_ids[trace_index, span_index] = span.outcome + 1
            depth_ids[trace_index, span_index] = min(span.depth + 1, 9)
            raw_numeric = np.asarray(
                (
                    math.log1p(
                        max(span.end_nano - span.start_nano, 0) / 1e6
                    ),
                    math.log1p(span.gap_from_previous_ms),
                    span.depth / 8.0,
                )
            )
            numeric[trace_index, span_index] = (
                raw_numeric - numeric_scale.center
            ) / numeric_scale.scale
            valid[trace_index, span_index] = True
            if span_index + 1 < len(trace.spans):
                next_time[trace_index, span_index] = (
                    math.log1p(span.time_to_next_ms)
                    - numeric_scale.next_center
                ) / numeric_scale.next_scale
                next_valid[trace_index, span_index] = True
    return TraceTensors(
        template_ids=torch.from_numpy(template_ids),
        entity_ids=torch.from_numpy(entity_ids),
        outcome_ids=torch.from_numpy(outcome_ids),
        depth_ids=torch.from_numpy(depth_ids),
        numeric=torch.from_numpy(numeric),
        next_time=torch.from_numpy(next_time),
        valid=torch.from_numpy(valid),
        next_valid=torch.from_numpy(next_valid),
        traces=traces,
    )


def _masked_inputs(
    batch: TraceTensors,
    vocabulary: Vocabulary,
    *,
    seed: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    generator = torch.Generator().manual_seed(seed)
    mask = (
        torch.rand(batch.valid.shape, generator=generator) < 0.30
    ) & batch.valid
    missing = ~torch.any(mask, dim=1)
    if torch.any(missing):
        rows = torch.nonzero(missing, as_tuple=False).flatten()
        mask[rows, 0] = True
    template = batch.template_ids.clone()
    entity = batch.entity_ids.clone()
    outcome = batch.outcome_ids.clone()
    numeric = batch.numeric.clone()
    template[mask] = vocabulary.template_mask_id
    entity[mask] = vocabulary.entity_mask_id
    outcome[mask] = 3
    numeric[mask] = 0.0
    return template, entity, outcome, numeric, mask


def _trace_losses(
    model: MaskedTraceEncoder,
    batch: TraceTensors,
    *,
    seed: int,
) -> tuple[torch.Tensor, Mapping[str, float]]:
    (
        template,
        entity,
        outcome,
        numeric,
        mask,
    ) = _masked_inputs(batch, model.vocabulary, seed=seed)
    _, tokens = model(
        template,
        entity,
        outcome,
        batch.depth_ids,
        numeric,
        batch.valid,
    )
    template_logits = model.template_head(tokens[mask])
    entity_logits = model.entity_head(tokens[mask])
    outcome_logits = model.outcome_head(tokens[mask])
    template_loss = F.cross_entropy(
        template_logits, batch.template_ids[mask]
    )
    entity_loss = F.cross_entropy(
        entity_logits, batch.entity_ids[mask]
    )
    outcome_loss = F.cross_entropy(
        outcome_logits, batch.outcome_ids[mask]
    )
    time_prediction = model.next_time_head(tokens).squeeze(-1)
    time_loss = F.smooth_l1_loss(
        time_prediction[batch.next_valid],
        batch.next_time[batch.next_valid],
    )
    total = (
        template_loss + entity_loss + outcome_loss + 0.5 * time_loss
    )
    diagnostics = {
        "loss": float(total.detach()),
        "template_accuracy": float(
            torch.mean(
                (
                    torch.argmax(template_logits, dim=1)
                    == batch.template_ids[mask]
                ).float()
            )
        ),
        "entity_accuracy": float(
            torch.mean(
                (
                    torch.argmax(entity_logits, dim=1)
                    == batch.entity_ids[mask]
                ).float()
            )
        ),
        "outcome_accuracy": float(
            torch.mean(
                (
                    torch.argmax(outcome_logits, dim=1)
                    == batch.outcome_ids[mask]
                ).float()
            )
        ),
        "standardized_next_time_mae": float(
            torch.mean(
                torch.abs(
                    time_prediction[batch.next_valid]
                    - batch.next_time[batch.next_valid]
                )
            ).detach()
        ),
    }
    return total, diagnostics


def _train_trace_encoder(
    tensors: TraceTensors,
    vocabulary: Vocabulary,
) -> tuple[MaskedTraceEncoder, Mapping[str, Any]]:
    _seed_everything(SEED)
    model = MaskedTraceEncoder(vocabulary)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.002, weight_decay=1e-4
    )
    rng = np.random.default_rng(SEED)
    batch_size = 512
    epoch_rows = []
    started = time.perf_counter()
    model.train()
    for epoch in range(12):
        order = rng.permutation(len(tensors.traces))
        losses = []
        for batch_number, start in enumerate(
            range(0, len(order), batch_size)
        ):
            indices = order[start : start + batch_size]
            batch = tensors.subset(indices)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = _trace_losses(
                model,
                batch,
                seed=SEED + epoch * 10000 + batch_number,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        epoch_rows.append(
            {"epoch": epoch + 1, "mean_loss": float(np.mean(losses))}
        )
    return model, {
        "training_seconds": time.perf_counter() - started,
        "epochs": epoch_rows,
    }


def _evaluate_trace_encoder(
    model: MaskedTraceEncoder,
    tensors: TraceTensors,
) -> Mapping[str, float]:
    model.eval()
    totals: dict[str, list[float]] = defaultdict(list)
    with torch.no_grad():
        for batch_number, start in enumerate(
            range(0, len(tensors.traces), 1024)
        ):
            indices = np.arange(
                start, min(start + 1024, len(tensors.traces))
            )
            _, diagnostics = _trace_losses(
                model,
                tensors.subset(indices),
                seed=SEED + 900000 + batch_number,
            )
            for key, value in diagnostics.items():
                totals[key].append(value)
    return {
        key: float(np.mean(values)) for key, values in totals.items()
    }


def _encode_trace_tokens(
    model: MaskedTraceEncoder,
    tensors: TraceTensors,
) -> np.ndarray:
    model.eval()
    encoded = []
    with torch.no_grad():
        for start in range(0, len(tensors.traces), 1024):
            stop = min(start + 1024, len(tensors.traces))
            indices = np.arange(start, stop)
            batch = tensors.subset(indices)
            _, token_states = model(
                batch.template_ids,
                batch.entity_ids,
                batch.outcome_ids,
                batch.depth_ids,
                batch.numeric,
                batch.valid,
            )
            encoded.append(token_states.numpy())
    return np.concatenate(encoded, axis=0)


def _window_features(
    cases: Sequence[CaseEvents],
    tensors: TraceTensors,
    encoded_tokens: np.ndarray,
    vocabulary: Vocabulary,
    numeric_scale: NumericScale,
) -> tuple[Mapping[str, np.ndarray], Mapping[str, np.ndarray]]:
    case_by_id = {case.case_id: case for case in cases}
    entity_position = {
        entity: index for index, entity in enumerate(ENTITY_IDS)
    }
    template_width = vocabulary.template_class_count - 1
    candidate_accumulators: dict[str, tuple[np.ndarray, ...]] = {}
    binned_accumulators: dict[str, tuple[np.ndarray, ...]] = {}
    for case in cases:
        candidate_accumulators[case.case_id] = (
            np.zeros(
                (
                    case.point_count,
                    len(ENTITY_IDS),
                    TOKEN_DIMENSION,
                ),
                dtype=np.float64,
            ),
            np.zeros(
                (
                    case.point_count,
                    len(ENTITY_IDS),
                    TOKEN_DIMENSION,
                ),
                dtype=np.float64,
            ),
            np.zeros(
                (case.point_count, len(ENTITY_IDS)),
                dtype=np.float64,
            ),
        )
        binned_accumulators[case.case_id] = (
            np.zeros(
                (
                    case.point_count,
                    len(ENTITY_IDS),
                    template_width,
                ),
                dtype=np.float64,
            ),
            np.zeros(
                (case.point_count, len(ENTITY_IDS), 3),
                dtype=np.float64,
            ),
            np.zeros(
                (case.point_count, len(ENTITY_IDS)),
                dtype=np.float64,
            ),
        )
    for trace_index, trace in enumerate(tensors.traces):
        candidate_sum, candidate_sq, candidate_count = (
            candidate_accumulators[trace.case_id]
        )
        binned_count, binned_values, binned_denominator = (
            binned_accumulators[trace.case_id]
        )
        for span_index, span in enumerate(trace.spans):
            entity = entity_position[span.entity]
            hidden = encoded_tokens[trace_index, span_index]
            candidate_sum[trace.window_index, entity] += hidden
            candidate_sq[trace.window_index, entity] += np.square(hidden)
            candidate_count[trace.window_index, entity] += 1.0
            template_id = int(
                tensors.template_ids[trace_index, span_index]
            )
            binned_count[
                trace.window_index, entity, template_id - 1
            ] += 1.0
            duration = math.log1p(
                max(span.end_nano - span.start_nano, 0) / 1e6
            )
            binned_values[trace.window_index, entity, 0] += (
                span.outcome
            )
            binned_values[trace.window_index, entity, 1] += duration
            binned_values[trace.window_index, entity, 2] += math.log1p(
                span.gap_from_previous_ms
            )
            binned_denominator[trace.window_index, entity] += 1.0
    candidate_features: dict[str, np.ndarray] = {}
    binned_features: dict[str, np.ndarray] = {}
    for case_id, case in case_by_id.items():
        total, square, count = candidate_accumulators[case_id]
        denominator = np.maximum(count[..., None], 1.0)
        mean = total / denominator
        variance = np.maximum(square / denominator - np.square(mean), 0.0)
        candidate_features[case_id] = np.concatenate(
            (mean, np.sqrt(variance), np.log1p(count)[..., None]),
            axis=2,
        ).reshape(case.point_count, -1)
        counts, values, value_count = binned_accumulators[case_id]
        value_denominator = np.maximum(value_count[..., None], 1.0)
        binned_features[case_id] = np.concatenate(
            (
                np.log1p(counts),
                values / value_denominator,
            ),
            axis=2,
        ).reshape(case.point_count, -1)
    return candidate_features, binned_features


def _fit_window_latents(
    raw_by_case: Mapping[str, np.ndarray],
    fit_case_ids: set[str],
) -> tuple[PCATransform, Mapping[str, np.ndarray]]:
    fitting = np.concatenate(
        [
            raw_by_case[case_id]
            for case_id in sorted(fit_case_ids)
        ],
        axis=0,
    )
    transform = PCATransform(LATENT_DIMENSION).fit(fitting)
    return transform, {
        case_id: transform.transform(values)
        for case_id, values in raw_by_case.items()
    }


def _latent_dataset(
    windows: Any,
    latent_by_case: Mapping[str, np.ndarray],
) -> LatentDataset:
    histories = []
    targets = []
    for case_id, transition in zip(
        windows.trajectory_ids, windows.transition_indices
    ):
        transition_index = int(transition)
        latent = latent_by_case[case_id]
        histories.append(
            latent[
                transition_index - CONTEXT_LENGTH
                + 1 : transition_index
                + 1
            ]
        )
        targets.append(
            latent[
                transition_index
                + 1 : transition_index
                + 1
                + HORIZON
            ]
        )
    return LatentDataset(
        histories=np.asarray(histories, dtype=np.float32),
        targets=np.asarray(targets, dtype=np.float32),
        future_controls=np.asarray(
            windows.future_controls, dtype=np.float32
        ),
        future_actions=np.asarray(
            windows.future_actions, dtype=np.float32
        ),
        trajectory_ids=tuple(windows.trajectory_ids),
        pair_ids=tuple(windows.matched_pair_ids),
        transition_indices=np.asarray(
            windows.transition_indices, dtype=np.int64
        ),
    )


def _train_latent_predictor(
    dataset: LatentDataset,
    targets: np.ndarray,
    *,
    seed: int,
) -> tuple[LatentPredictor, Mapping[str, Any]]:
    _seed_everything(seed)
    model = LatentPredictor(
        latent_dimension=dataset.histories.shape[2],
        control_dimension=dataset.future_controls.shape[2],
        action_dimension=int(
            np.prod(dataset.future_actions.shape[2:])
        ),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.002, weight_decay=1e-4
    )
    histories = torch.from_numpy(dataset.histories)
    controls = torch.from_numpy(dataset.future_controls)
    actions = torch.from_numpy(dataset.future_actions)
    target_tensor = torch.from_numpy(
        np.asarray(targets, dtype=np.float32)
    )
    rng = np.random.default_rng(seed)
    epoch_rows = []
    started = time.perf_counter()
    model.train()
    for epoch in range(40):
        order = rng.permutation(len(histories))
        losses = []
        for start in range(0, len(order), 256):
            indices = torch.as_tensor(
                order[start : start + 256], dtype=torch.long
            )
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                histories[indices],
                controls[indices],
                actions[indices],
            )
            loss = F.mse_loss(prediction, target_tensor[indices])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        epoch_rows.append(
            {"epoch": epoch + 1, "mean_loss": float(np.mean(losses))}
        )
    return model, {
        "training_seconds": time.perf_counter() - started,
        "epochs": epoch_rows,
    }


def _shuffled_targets(
    dataset: LatentDataset,
    cases: Mapping[str, CaseEvents],
) -> tuple[np.ndarray, Mapping[str, Any]]:
    groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index, (case_id, transition) in enumerate(
        zip(dataset.trajectory_ids, dataset.transition_indices)
    ):
        case = cases[case_id]
        groups[
            (
                int(transition),
                case.worker_replicas,
                case.is_treatment,
                case.action_kind if case.is_treatment else case.pair_id,
            )
        ].append(index)
    # Controls must preserve the action family of their matched treatment,
    # not their opaque pair identity.
    pair_action = {
        case.pair_id: case.action_kind
        for case in cases.values()
        if case.is_treatment
    }
    groups.clear()
    for index, (case_id, transition) in enumerate(
        zip(dataset.trajectory_ids, dataset.transition_indices)
    ):
        case = cases[case_id]
        groups[
            (
                int(transition),
                case.worker_replicas,
                case.is_treatment,
                pair_action[case.pair_id],
            )
        ].append(index)
    permutation = np.arange(len(dataset.targets))
    singleton_count = 0
    for indices in groups.values():
        if len(indices) < 2:
            singleton_count += len(indices)
            continue
        ordered = sorted(indices, key=lambda index: dataset.trajectory_ids[index])
        rotated = ordered[1:] + ordered[:1]
        permutation[np.asarray(ordered)] = np.asarray(rotated)
    same_trajectory = int(
        sum(
            dataset.trajectory_ids[index]
            == dataset.trajectory_ids[int(permutation[index])]
            for index in range(len(permutation))
        )
    )
    if singleton_count or same_trajectory:
        raise ValueError("alignment shuffle is not a complete derangement")
    return dataset.targets[permutation], {
        "group_count": len(groups),
        "deranged_sample_count": len(permutation),
        "same_trajectory_count": same_trajectory,
    }


def _predict_latents(
    model: LatentPredictor,
    dataset: LatentDataset,
    *,
    hide_actions: bool = False,
) -> np.ndarray:
    model.eval()
    histories = torch.from_numpy(dataset.histories)
    controls = torch.from_numpy(dataset.future_controls)
    actions = torch.from_numpy(dataset.future_actions)
    if hide_actions:
        actions = torch.zeros_like(actions)
    outputs = []
    with torch.no_grad():
        for start in range(0, len(histories), 512):
            outputs.append(
                model(
                    histories[start : start + 512],
                    controls[start : start + 512],
                    actions[start : start + 512],
                ).numpy()
            )
    return np.concatenate(outputs, axis=0)


def _upper_p_values(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(reference, dtype=np.float64))
    positions = np.searchsorted(ordered, values, side="left")
    upper = len(ordered) - positions
    return (upper + 1.0) / (len(ordered) + 1.0)


def _higher_quantile(values: Sequence[float], quantile: float) -> float:
    return float(
        np.quantile(
            np.asarray(values, dtype=np.float64),
            quantile,
            method="higher",
        )
    )


def _sequential_detection(
    *,
    calibration_scores: np.ndarray,
    calibration: LatentDataset,
    evaluation_scores: np.ndarray,
    evaluation: LatentDataset,
    cases: Mapping[str, CaseEvents],
    alpha: float = 0.05,
) -> Mapping[str, Any]:
    calibration_control = np.asarray(
        [
            not cases[case_id].is_treatment
            for case_id in calibration.trajectory_ids
        ],
        dtype=np.bool_,
    )
    reference = np.asarray(
        calibration_scores[calibration_control], dtype=np.float64
    )
    if len(reference) < 2:
        raise ValueError("alert calibration has too few control windows")
    calibration_p = _upper_p_values(reference, calibration_scores)
    calibration_evidence = np.maximum(
        0.0, -np.log(calibration_p) - math.log(2.0)
    )
    calibration_maxima = []
    for case_id in sorted(set(calibration.trajectory_ids)):
        if cases[case_id].is_treatment:
            continue
        selection = np.flatnonzero(
            np.asarray(calibration.trajectory_ids) == case_id
        )
        order = selection[
            np.argsort(calibration.transition_indices[selection])
        ]
        calibration_maxima.append(
            float(np.max(np.cumsum(calibration_evidence[order])))
        )
    sequential_threshold = _higher_quantile(
        calibration_maxima, 1.0 - alpha
    )
    point_threshold = _higher_quantile(reference, 1.0 - alpha)
    evaluation_p = _upper_p_values(reference, evaluation_scores)
    evaluation_evidence = np.maximum(
        0.0, -np.log(evaluation_p) - math.log(2.0)
    )
    rows = []
    for case_id in sorted(set(evaluation.trajectory_ids)):
        selection = np.flatnonzero(
            np.asarray(evaluation.trajectory_ids) == case_id
        )
        order = selection[
            np.argsort(evaluation.transition_indices[selection])
        ]
        target_windows = evaluation.transition_indices[order] + 1
        cumulative = np.cumsum(evaluation_evidence[order])
        sequential_alarms = cumulative > sequential_threshold
        point_alarms = evaluation_scores[order] > point_threshold
        case = cases[case_id]
        onset = case.action_onset
        if onset is None:
            post_positions = np.asarray([], dtype=np.int64)
            pre_sequential = False
            pre_point = False
        else:
            post_positions = np.flatnonzero(
                (target_windows >= onset) & sequential_alarms
            )
            pre_sequential = bool(
                np.any(
                    sequential_alarms[target_windows < onset]
                )
            )
            pre_point = bool(
                np.any(point_alarms[target_windows < onset])
            )
        first_post = (
            int(target_windows[post_positions[0]])
            if len(post_positions)
            else None
        )
        rows.append(
            {
                "case_id": case_id,
                "is_treatment": case.is_treatment,
                "action_kind": case.action_kind,
                "onset_transition": onset,
                "any_sequential_alarm": bool(
                    np.any(sequential_alarms)
                ),
                "any_point_alarm": bool(np.any(point_alarms)),
                "pre_onset_sequential_alarm": pre_sequential,
                "pre_onset_point_alarm": pre_point,
                "post_onset_sequential_alarm_transition": first_post,
                "post_onset_delay": (
                    first_post - onset
                    if first_post is not None and onset is not None
                    else None
                ),
            }
        )
    controls = [row for row in rows if not row["is_treatment"]]
    treatments = [row for row in rows if row["is_treatment"]]
    detected = [
        row
        for row in treatments
        if row["post_onset_sequential_alarm_transition"] is not None
    ]
    delays = [int(row["post_onset_delay"]) for row in detected]
    return {
        "alpha": alpha,
        "calibration_control_window_count": len(reference),
        "calibration_control_trajectory_count": len(
            calibration_maxima
        ),
        "point_threshold": point_threshold,
        "sequential_threshold": sequential_threshold,
        "control_trajectory_false_alarm_rate": float(
            np.mean(
                [bool(row["any_sequential_alarm"]) for row in controls]
            )
        ),
        "treatment_detection_rate": float(
            len(detected) / len(treatments)
        ),
        "treatment_pre_onset_alarm_rate": float(
            np.mean(
                [
                    bool(row["pre_onset_sequential_alarm"])
                    for row in treatments
                ]
            )
        ),
        "median_detection_delay_transitions": (
            float(np.median(delays)) if delays else None
        ),
        "worst_detection_delay_transitions": (
            int(max(delays)) if delays else None
        ),
        "alerts_per_logical_run": float(
            np.mean(
                [bool(row["any_sequential_alarm"]) for row in rows]
            )
        ),
        "trajectory_rows": rows,
    }


def _latent_alert_scores(
    model: LatentPredictor, dataset: LatentDataset
) -> np.ndarray:
    prediction = _predict_latents(
        model, dataset, hide_actions=True
    )
    return np.mean(
        np.square(prediction[:, 0] - dataset.targets[:, 0]),
        axis=1,
    )


def _metrics_only_scores(
    model: Any,
    windows: Any,
    metric_positions: Sequence[int],
) -> np.ndarray:
    actions = np.zeros_like(windows.future_actions)
    prediction = model.rollout(
        windows.histories,
        windows.future_controls,
        actions,
        windows.graph,
    ).mean[:, 0][:, :, metric_positions]
    observed = windows.future_states[:, 0][:, :, metric_positions]
    return np.mean(np.square(prediction - observed), axis=(1, 2))


class EventNgram:
    """A small trace-token bigram and timing surprise control."""

    def __init__(self) -> None:
        self.vocabulary: set[str] = set()
        self.counts: Counter[tuple[str, str]] = Counter()
        self.context_counts: Counter[str] = Counter()
        self.time_values: dict[tuple[str, str], list[float]] = defaultdict(
            list
        )
        self.time_stats: dict[tuple[str, str], tuple[float, float]] = {}
        self.window_count_center = 0.0
        self.window_count_scale = 1.0

    @staticmethod
    def _token(span: RawSpan) -> str:
        return f"{span.template}|{span.entity}|{span.outcome}"

    def fit(self, cases: Sequence[CaseEvents]) -> "EventNgram":
        window_counts = []
        for case in cases:
            per_window = np.zeros(case.point_count, dtype=np.float64)
            for trace in case.traces:
                per_window[trace.window_index] += 1.0
                previous = "<BOS>"
                for span in trace.spans:
                    token = self._token(span)
                    self.vocabulary.add(token)
                    self.counts[(previous, token)] += 1
                    self.context_counts[previous] += 1
                    self.time_values[(previous, token)].append(
                        math.log1p(span.time_to_next_ms)
                    )
                    previous = token
            window_counts.extend(per_window.tolist())
        for key, values in self.time_values.items():
            array = np.asarray(values, dtype=np.float64)
            center = float(np.mean(array))
            scale = float(np.std(array))
            self.time_stats[key] = (
                center,
                scale if scale > 0.05 else 0.05,
            )
        counts = np.asarray(window_counts, dtype=np.float64)
        self.window_count_center = float(np.median(counts))
        mad = float(np.median(np.abs(counts - self.window_count_center)))
        self.window_count_scale = max(1.4826 * mad, 1.0)
        return self

    def window_scores(
        self, cases: Sequence[CaseEvents]
    ) -> Mapping[str, np.ndarray]:
        vocabulary_size = max(len(self.vocabulary), 1)
        result = {}
        for case in cases:
            sums = np.zeros(case.point_count, dtype=np.float64)
            counts = np.zeros(case.point_count, dtype=np.float64)
            trace_counts = np.zeros(case.point_count, dtype=np.float64)
            for trace in case.traces:
                trace_counts[trace.window_index] += 1.0
                previous = "<BOS>"
                trace_score = 0.0
                for span in trace.spans:
                    token = self._token(span)
                    probability = (
                        self.counts[(previous, token)] + 1.0
                    ) / (
                        self.context_counts[previous]
                        + vocabulary_size
                    )
                    center, scale = self.time_stats.get(
                        (previous, token), (0.0, 1.0)
                    )
                    time_z = (
                        math.log1p(span.time_to_next_ms) - center
                    ) / scale
                    trace_score += -math.log(probability) + min(
                        0.5 * time_z * time_z, 25.0
                    )
                    previous = token
                trace_score /= max(len(trace.spans), 1)
                sums[trace.window_index] += trace_score
                counts[trace.window_index] += 1.0
            mean_trace = np.divide(
                sums,
                counts,
                out=np.zeros_like(sums),
                where=counts > 0,
            )
            count_z = (
                trace_counts - self.window_count_center
            ) / self.window_count_scale
            result[case.case_id] = mean_trace + 0.5 * np.square(
                count_z
            )
        return result


def _window_scores_for_dataset(
    dataset: LatentDataset,
    scores_by_case: Mapping[str, np.ndarray],
) -> np.ndarray:
    return np.asarray(
        [
            scores_by_case[case_id][int(transition) + 1]
            for case_id, transition in zip(
                dataset.trajectory_ids, dataset.transition_indices
            )
        ],
        dtype=np.float64,
    )


def _case_for_query(
    query_id: str,
    cases: Mapping[str, CaseEvents],
) -> CaseEvents:
    pair_id, label = query_id.rsplit(":", 1)
    matches = [
        case
        for case in cases.values()
        if case.pair_id == pair_id
        and case.is_treatment == (label == "treatment")
    ]
    if len(matches) != 1:
        raise ValueError(f"cannot resolve attribution query {query_id}")
    return matches[0]


def _investigation_scores(
    *,
    model: LatentPredictor,
    latent_by_case: Mapping[str, np.ndarray],
    prepared_queries: Any,
    cases: Mapping[str, CaseEvents],
    transfer_only: bool,
) -> Mapping[str, Any]:
    rows = []
    candidate_ids = prepared_queries.candidate_ids
    candidate_kinds = prepared_queries.candidate_action_kinds
    candidate_targets = prepared_queries.candidate_target_entities
    for query_index, query_id in enumerate(prepared_queries.query_ids):
        case = _case_for_query(query_id, cases)
        if (case.worker_replicas == 3) != transfer_only:
            continue
        transition = (
            case.action_onset - 1
            if case.action_onset is not None
            else cases[
                next(
                    other.case_id
                    for other in cases.values()
                    if other.pair_id == case.pair_id
                    and other.is_treatment
                )
            ].action_onset
            - 1
        )
        context = latent_by_case[case.case_id][
            transition - CONTEXT_LENGTH + 1 : transition + 1
        ]
        observed = latent_by_case[case.case_id][
            transition + 1 : transition + 1 + HORIZON
        ]
        candidate_count = len(candidate_ids)
        histories = torch.from_numpy(
            np.repeat(context[None], candidate_count, axis=0)
        )
        controls = torch.from_numpy(
            np.repeat(
                prepared_queries.future_controls[
                    query_index : query_index + 1
                ],
                candidate_count,
                axis=0,
            )
        )
        actions = torch.from_numpy(
            prepared_queries.candidate_actions[query_index]
        )
        model.eval()
        with torch.no_grad():
            prediction = model(histories, controls, actions).numpy()
        errors = np.mean(
            np.square(prediction - observed[None]), axis=(1, 2)
        )
        chosen = int(np.argmin(errors))
        expected_kind = prepared_queries.expected_action_kinds[
            query_index
        ]
        expected_target = prepared_queries.expected_target_entities[
            query_index
        ]
        expected_variant = prepared_queries.expected_variant_ids[
            query_index
        ]
        correct_family_positions = [
            index
            for index, (kind, target) in enumerate(
                zip(candidate_kinds, candidate_targets)
            )
            if kind == expected_kind and target == expected_target
        ]
        no_action_position = candidate_ids.index("no_action")
        if expected_kind:
            correct_family_error = min(
                float(errors[index])
                for index in correct_family_positions
            )
            wrong_family_error = min(
                float(errors[index])
                for index, kind in enumerate(candidate_kinds)
                if kind and kind != expected_kind
            )
            sanity = (
                correct_family_error < float(errors[no_action_position])
                and correct_family_error < wrong_family_error
            )
        else:
            sanity = True
        rows.append(
            {
                "query_id": query_id,
                "expected_action_kind": expected_kind,
                "expected_target_entity": expected_target,
                "expected_variant_id": expected_variant,
                "chosen_candidate_id": candidate_ids[chosen],
                "chosen_action_kind": candidate_kinds[chosen],
                "chosen_target_entity": candidate_targets[chosen],
                "action_and_target_correct": (
                    candidate_kinds[chosen] == expected_kind
                    and candidate_targets[chosen] == expected_target
                ),
                "exact_variant_correct": (
                    candidate_ids[chosen] == expected_variant
                    if expected_variant
                    else candidate_ids[chosen] == "no_action"
                ),
                "correct_beats_ablations": sanity,
            }
        )
    treatments = [row for row in rows if row["expected_action_kind"]]
    controls = [row for row in rows if not row["expected_action_kind"]]
    return {
        "query_count": len(rows),
        "action_and_target_hit_at_1": float(
            np.mean(
                [
                    bool(row["action_and_target_correct"])
                    for row in treatments
                ]
            )
        ),
        "exact_variant_hit_at_1": float(
            np.mean(
                [
                    bool(row["exact_variant_correct"])
                    for row in treatments
                ]
            )
        ),
        "no_action_specificity": float(
            np.mean(
                [
                    row["chosen_candidate_id"] == "no_action"
                    for row in controls
                ]
            )
        ),
        "correct_beats_action_ablations_rate": float(
            np.mean(
                [
                    bool(row["correct_beats_ablations"])
                    for row in treatments
                ]
            )
        ),
        "rows": rows,
    }


def _effective_rank(values: np.ndarray) -> Mapping[str, float]:
    matrix = np.asarray(values, dtype=np.float64).reshape(
        -1, values.shape[-1]
    )
    covariance = np.cov(matrix, rowvar=False)
    eigenvalues = np.maximum(
        np.linalg.eigvalsh(covariance), 0.0
    )
    total = float(np.sum(eigenvalues))
    probabilities = (
        eigenvalues[eigenvalues > 1e-12] / total
        if total > 0.0
        else np.asarray([])
    )
    effective = (
        float(np.exp(-np.sum(probabilities * np.log(probabilities))))
        if len(probabilities)
        else 0.0
    )
    return {
        "effective_rank": effective,
        "minimum_dimension_variance": float(
            np.min(np.var(matrix, axis=0))
        ),
        "maximum_dimension_variance": float(
            np.max(np.var(matrix, axis=0))
        ),
    }


def _ridge_fit(
    design: np.ndarray, target: np.ndarray, ridge: float = 1e-2
) -> np.ndarray:
    x = np.concatenate(
        (
            np.asarray(design, dtype=np.float64),
            np.ones((len(design), 1), dtype=np.float64),
        ),
        axis=1,
    )
    penalty = np.eye(x.shape[1], dtype=np.float64)
    penalty[-1, -1] = 0.0
    return np.linalg.solve(
        x.T @ x + ridge * penalty,
        x.T @ np.asarray(target, dtype=np.float64),
    )


def _state_probe(
    model: LatentPredictor,
    fit: LatentDataset,
    transfer: LatentDataset,
    fit_windows: Any,
    transfer_windows: Any,
) -> Mapping[str, float]:
    model.eval()
    with torch.no_grad():
        fit_context = model.encode_context(
            torch.from_numpy(fit.histories)
        ).numpy()
        transfer_context = model.encode_context(
            torch.from_numpy(transfer.histories)
        ).numpy()
    fit_state = np.asarray(
        fit_windows.histories[:, -1], dtype=np.float64
    ).reshape(len(fit_context), -1)
    transfer_state = np.asarray(
        transfer_windows.histories[:, -1], dtype=np.float64
    ).reshape(len(transfer_context), -1)
    varying = np.std(fit_state, axis=0) > 1e-6
    weights = _ridge_fit(fit_context, fit_state[:, varying])
    prediction = np.concatenate(
        (
            transfer_context,
            np.ones((len(transfer_context), 1)),
        ),
        axis=1,
    ) @ weights
    scale = np.maximum(np.std(fit_state[:, varying], axis=0), 1e-6)
    nrmse = np.sqrt(
        np.mean(
            np.square(
                (prediction - transfer_state[:, varying]) / scale
            )
        )
    )
    return {
        "varying_observable_position_count": int(np.sum(varying)),
        "transfer_frozen_probe_nrmse": float(nrmse),
    }


def _model_cost(
    trace_encoder: MaskedTraceEncoder,
    predictor: LatentPredictor,
    example: LatentDataset,
) -> Mapping[str, Any]:
    parameters = list(trace_encoder.parameters()) + list(
        predictor.parameters()
    )
    parameter_count = sum(parameter.numel() for parameter in parameters)
    serialized_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in parameters
    )
    history = torch.from_numpy(example.histories[:1])
    controls = torch.from_numpy(example.future_controls[:1])
    actions = torch.from_numpy(example.future_actions[:1])
    predictor.eval()
    with torch.no_grad():
        for _ in range(20):
            predictor(history, controls, actions)
        started = time.perf_counter()
        for _ in range(200):
            predictor(history, controls, actions)
        latency = (time.perf_counter() - started) * 1000.0 / 200.0
    return {
        "parameter_count": int(parameter_count),
        "serialized_tensor_bytes": int(serialized_bytes),
        "batch_one_predictor_latency_ms": float(latency),
        "runtime_scope": "local CPU microbenchmark",
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _role_objective(
    model: LatentPredictor, dataset: LatentDataset
) -> Mapping[str, float]:
    prediction = _predict_latents(model, dataset)
    squared = np.square(prediction - dataset.targets)
    action_overlap = np.any(
        dataset.future_actions[..., 1] > 0.5, axis=2
    )
    return {
        "latent_mse_overall": float(np.mean(squared)),
        "latent_mse_action_overlap": float(
            np.mean(squared[action_overlap])
        ),
    }


def _beats_alert_null(
    candidate: Mapping[str, Any],
    null: Mapping[str, Any],
) -> bool:
    detection_gain = (
        float(candidate["treatment_detection_rate"])
        - float(null["treatment_detection_rate"])
        >= 0.10 - 1e-12
    )
    candidate_delay = candidate[
        "median_detection_delay_transitions"
    ]
    null_delay = null["median_detection_delay_transitions"]
    delay_gain = (
        candidate_delay is not None
        and null_delay is not None
        and float(null_delay) - float(candidate_delay) >= 2.0
    )
    return bool(detection_gain or delay_gain)


def _alert_noninferior_to_controls(
    candidate: Mapping[str, Any],
    controls: Sequence[Mapping[str, Any]],
) -> bool:
    eligible = [
        control
        for control in controls
        if float(control["control_trajectory_false_alarm_rate"])
        <= 0.05
    ]
    if not eligible:
        return True
    detection_ok = float(candidate["treatment_detection_rate"]) >= max(
        float(control["treatment_detection_rate"])
        for control in eligible
    )
    delays = [
        float(control["median_detection_delay_transitions"])
        for control in eligible
        if control["median_detection_delay_transitions"] is not None
    ]
    candidate_delay = candidate[
        "median_detection_delay_transitions"
    ]
    delay_ok = (
        not delays
        or (
            candidate_delay is not None
            and float(candidate_delay) <= min(delays)
        )
    )
    return bool(detection_ok and delay_ok)


def _load_cases(
    corpus_directory: Path, case_ids: Sequence[str]
) -> tuple[CaseEvents, ...]:
    cases = []
    started = time.perf_counter()
    for index, case_id in enumerate(sorted(case_ids), 1):
        cases.append(
            _load_case(corpus_directory / "cases" / case_id)
        )
        if index % 20 == 0 or index == len(case_ids):
            print(
                json.dumps(
                    {
                        "stage": "parse_causal_trace_paths",
                        "cases_complete": index,
                        "case_count": len(case_ids),
                        "elapsed_seconds": round(
                            time.perf_counter() - started, 2
                        ),
                    }
                ),
                flush=True,
            )
    return tuple(cases)


def run_prototype(
    *,
    corpus_directory: Path,
    cache_directory: Path,
    output_directory: Path,
) -> Mapping[str, Any]:
    if output_directory.exists():
        raise FileExistsError(
            f"refusing to overwrite prototype result: {output_directory}"
        )
    _seed_everything(SEED)
    source_manifest = source_artifact_manifest_sha256(
        corpus_directory
    )
    resolved_cache = cache_directory / topology_transfer_cache_address(
        source_manifest
    )
    prepared = load_edge_dynamics_cache(resolved_cache)
    validate_topology_transfer_cache(prepared, corpus_directory)
    partitions = {
        role: partition_worker_topology(windows)
        for role, windows in prepared.windows.items()
    }
    fit_windows = partitions["fit"].in_distribution
    selection_windows = partitions["selection"].in_distribution
    calibration_windows = partitions["calibration"].in_distribution
    iid_windows = partitions["evaluation"].in_distribution
    transfer_windows = partitions["evaluation"].held_out
    required_case_ids = sorted(
        set(fit_windows.trajectory_ids)
        | set(selection_windows.trajectory_ids)
        | set(calibration_windows.trajectory_ids)
        | set(iid_windows.trajectory_ids)
        | set(transfer_windows.trajectory_ids)
    )
    print(
        json.dumps(
            {
                "stage": "load_inputs",
                "required_case_count": len(required_case_ids),
                "fit_window_count": len(fit_windows.histories),
                "transfer_window_count": len(transfer_windows.histories),
            }
        ),
        flush=True,
    )
    cases = _load_cases(corpus_directory, required_case_ids)
    cases_by_id = {case.case_id: case for case in cases}
    fit_case_ids = set(fit_windows.trajectory_ids)
    fit_cases = tuple(
        case for case in cases if case.case_id in fit_case_ids
    )
    selection_case_ids = set(selection_windows.trajectory_ids)
    selection_cases = tuple(
        case
        for case in cases
        if case.case_id in selection_case_ids
    )
    transfer_case_ids = set(transfer_windows.trajectory_ids)
    transfer_cases = tuple(
        case for case in cases if case.case_id in transfer_case_ids
    )
    vocabulary = _fit_vocabulary(fit_cases)
    numeric_scale = _fit_numeric_scale(fit_cases)
    all_tensors = _compile_trace_tensors(
        cases, vocabulary, numeric_scale
    )
    fit_trace_indices = np.asarray(
        [
            index
            for index, trace in enumerate(all_tensors.traces)
            if trace.case_id in fit_case_ids
        ],
        dtype=np.int64,
    )
    selection_trace_indices = np.asarray(
        [
            index
            for index, trace in enumerate(all_tensors.traces)
            if trace.case_id in selection_case_ids
        ],
        dtype=np.int64,
    )
    fit_tensors = all_tensors.subset(fit_trace_indices)
    selection_tensors = all_tensors.subset(selection_trace_indices)
    print(
        json.dumps(
            {
                "stage": "train_masked_trace_encoder",
                "fit_trace_count": len(fit_tensors.traces),
                "template_count": len(vocabulary.templates),
                "entity_count": len(vocabulary.entities),
            }
        ),
        flush=True,
    )
    trace_encoder, trace_training = _train_trace_encoder(
        fit_tensors, vocabulary
    )
    trace_selection = _evaluate_trace_encoder(
        trace_encoder, selection_tensors
    )
    encoded_tokens = _encode_trace_tokens(
        trace_encoder, all_tensors
    )
    candidate_raw, binned_raw = _window_features(
        cases,
        all_tensors,
        encoded_tokens,
        vocabulary,
        numeric_scale,
    )
    candidate_pca, candidate_latents = _fit_window_latents(
        candidate_raw, fit_case_ids
    )
    binned_pca, binned_latents = _fit_window_latents(
        binned_raw, fit_case_ids
    )
    candidate_data = {
        "fit": _latent_dataset(fit_windows, candidate_latents),
        "selection": _latent_dataset(
            selection_windows, candidate_latents
        ),
        "calibration": _latent_dataset(
            calibration_windows, candidate_latents
        ),
        "iid": _latent_dataset(iid_windows, candidate_latents),
        "transfer": _latent_dataset(
            transfer_windows, candidate_latents
        ),
    }
    binned_data = {
        "fit": _latent_dataset(fit_windows, binned_latents),
        "selection": _latent_dataset(
            selection_windows, binned_latents
        ),
        "calibration": _latent_dataset(
            calibration_windows, binned_latents
        ),
        "iid": _latent_dataset(iid_windows, binned_latents),
        "transfer": _latent_dataset(transfer_windows, binned_latents),
    }
    shuffled_targets, shuffle_audit = _shuffled_targets(
        candidate_data["fit"], cases_by_id
    )
    print(
        json.dumps(
            {
                "stage": "train_temporal_predictors",
                "fit_sample_count": len(
                    candidate_data["fit"].histories
                ),
            }
        ),
        flush=True,
    )
    candidate_model, candidate_training = _train_latent_predictor(
        candidate_data["fit"],
        candidate_data["fit"].targets,
        seed=SEED + 1,
    )
    binned_model, binned_training = _train_latent_predictor(
        binned_data["fit"],
        binned_data["fit"].targets,
        seed=SEED + 2,
    )
    shuffled_model, shuffled_training = _train_latent_predictor(
        candidate_data["fit"],
        shuffled_targets,
        seed=SEED + 3,
    )
    print(
        json.dumps({"stage": "evaluate_controls_and_gates"}),
        flush=True,
    )
    event_positions = tuple(
        index
        for index, name in enumerate(
            fit_windows.state_feature_names
        )
        if name
        in {
            "log_event_count",
            "log_error_count",
            "trace_span_count",
            "trace_error_count",
        }
    )
    metric_positions = tuple(
        index
        for index in range(len(fit_windows.state_feature_names))
        if index not in event_positions
    )
    metrics_model = MaskedInputDynamics(
        ContractiveLowRankDynamics(LowRankConfig(rank=32)),
        event_positions,
    ).fit(fit_windows)
    ngram = EventNgram().fit(
        tuple(case for case in fit_cases if not case.is_treatment)
    )
    ngram_by_case = ngram.window_scores(cases)
    alert_models = {
        "event_native_trace_jepa": (
            _latent_alert_scores(
                candidate_model, candidate_data["calibration"]
            ),
            _latent_alert_scores(
                candidate_model, candidate_data["transfer"]
            ),
            _latent_alert_scores(
                candidate_model, candidate_data["iid"]
            ),
        ),
        "binned_event": (
            _latent_alert_scores(
                binned_model, binned_data["calibration"]
            ),
            _latent_alert_scores(
                binned_model, binned_data["transfer"]
            ),
            _latent_alert_scores(binned_model, binned_data["iid"]),
        ),
        "alignment_shuffled": (
            _latent_alert_scores(
                shuffled_model, candidate_data["calibration"]
            ),
            _latent_alert_scores(
                shuffled_model, candidate_data["transfer"]
            ),
            _latent_alert_scores(
                shuffled_model, candidate_data["iid"]
            ),
        ),
        "event_ngram": (
            _window_scores_for_dataset(
                candidate_data["calibration"], ngram_by_case
            ),
            _window_scores_for_dataset(
                candidate_data["transfer"], ngram_by_case
            ),
            _window_scores_for_dataset(
                candidate_data["iid"], ngram_by_case
            ),
        ),
        "metrics_only_low_rank": (
            _metrics_only_scores(
                metrics_model,
                calibration_windows,
                metric_positions,
            ),
            _metrics_only_scores(
                metrics_model, transfer_windows, metric_positions
            ),
            _metrics_only_scores(
                metrics_model, iid_windows, metric_positions
            ),
        ),
    }
    alert_transfer = {}
    alert_iid = {}
    for name, (calibration_score, transfer_score, iid_score) in (
        alert_models.items()
    ):
        alert_transfer[name] = _sequential_detection(
            calibration_scores=calibration_score,
            calibration=candidate_data["calibration"],
            evaluation_scores=transfer_score,
            evaluation=candidate_data["transfer"],
            cases=cases_by_id,
        )
        alert_iid[name] = _sequential_detection(
            calibration_scores=calibration_score,
            calibration=candidate_data["calibration"],
            evaluation_scores=iid_score,
            evaluation=candidate_data["iid"],
            cases=cases_by_id,
        )
    investigation_transfer = {
        "event_native_trace_jepa": _investigation_scores(
            model=candidate_model,
            latent_by_case=candidate_latents,
            prepared_queries=prepared.attribution_queries,
            cases=cases_by_id,
            transfer_only=True,
        ),
        "binned_event": _investigation_scores(
            model=binned_model,
            latent_by_case=binned_latents,
            prepared_queries=prepared.attribution_queries,
            cases=cases_by_id,
            transfer_only=True,
        ),
        "alignment_shuffled": _investigation_scores(
            model=shuffled_model,
            latent_by_case=candidate_latents,
            prepared_queries=prepared.attribution_queries,
            cases=cases_by_id,
            transfer_only=True,
        ),
    }
    investigation_iid = {
        "event_native_trace_jepa": _investigation_scores(
            model=candidate_model,
            latent_by_case=candidate_latents,
            prepared_queries=prepared.attribution_queries,
            cases=cases_by_id,
            transfer_only=False,
        ),
        "binned_event": _investigation_scores(
            model=binned_model,
            latent_by_case=binned_latents,
            prepared_queries=prepared.attribution_queries,
            cases=cases_by_id,
            transfer_only=False,
        ),
        "alignment_shuffled": _investigation_scores(
            model=shuffled_model,
            latent_by_case=candidate_latents,
            prepared_queries=prepared.attribution_queries,
            cases=cases_by_id,
            transfer_only=False,
        ),
    }
    transfer_spans = [
        span
        for case in transfer_cases
        for trace in case.traces
        for span in trace.spans
    ]
    unknown_template_count = sum(
        span.template not in vocabulary.template_to_id
        for span in transfer_spans
    )
    unknown_entity_count = sum(
        span.entity not in vocabulary.entity_to_id
        for span in transfer_spans
    )
    data_audit = {
        "case_count": len(cases),
        "trace_count": int(
            sum(len(case.traces) for case in cases)
        ),
        "raw_span_count": int(
            sum(case.raw_span_count for case in cases)
        ),
        "retained_span_count": int(
            sum(case.retained_span_count for case in cases)
        ),
        "trace_link_coverage": 1.0,
        "unknown_template_rate_transfer": (
            unknown_template_count / len(transfer_spans)
        ),
        "unknown_entity_rate_transfer": (
            unknown_entity_count / len(transfer_spans)
        ),
        "truncated_span_count": int(
            sum(case.truncated_span_count for case in cases)
        ),
        "incomplete_parent_count": int(
            sum(case.incomplete_parent_count for case in cases)
        ),
        "excluded_drain_trace_count": int(
            sum(case.drain_trace_count for case in cases)
        ),
        "event_time_assignment": (
            "first wall-clock window end at or after final span end, "
            "anchored by action.run.boundary and declared period"
        ),
        "origin_window_used_for_assignment": False,
    }
    representation = {
        "candidate_fit_pca_explained_variance_ratio": (
            candidate_pca.explained_variance_ratio
        ),
        "binned_fit_pca_explained_variance_ratio": (
            binned_pca.explained_variance_ratio
        ),
        "candidate_transfer": _effective_rank(
            np.concatenate(
                [
                    candidate_latents[case_id]
                    for case_id in sorted(transfer_case_ids)
                ],
                axis=0,
            )
        ),
        "state_probe": _state_probe(
            candidate_model,
            candidate_data["fit"],
            candidate_data["transfer"],
            fit_windows,
            transfer_windows,
        ),
    }
    candidate_alert = alert_transfer["event_native_trace_jepa"]
    null_alert = alert_transfer["alignment_shuffled"]
    binned_alert = alert_transfer["binned_event"]
    ngram_alert = alert_transfer["event_ngram"]
    candidate_investigation = investigation_transfer[
        "event_native_trace_jepa"
    ]
    binned_investigation = investigation_transfer["binned_event"]
    null_investigation = investigation_transfer[
        "alignment_shuffled"
    ]
    beats_alert_null = _beats_alert_null(
        candidate_alert, null_alert
    )
    beats_investigation_null = (
        candidate_investigation["action_and_target_hit_at_1"]
        - null_investigation["action_and_target_hit_at_1"]
        >= 0.10 - 1e-12
    )
    safety_gates = {
        "trace_link_coverage_at_least_95_percent": (
            data_audit["trace_link_coverage"] >= 0.95
        ),
        "transfer_unknown_template_rate_at_most_1_percent": (
            data_audit["unknown_template_rate_transfer"] <= 0.01
        ),
        "transfer_unknown_entity_rate_at_most_1_percent": (
            data_audit["unknown_entity_rate_transfer"] <= 0.01
        ),
        "causal_event_time_assignment": (
            not data_audit["origin_window_used_for_assignment"]
        ),
        "target_effective_rank_at_least_8": (
            representation["candidate_transfer"]["effective_rank"]
            >= 8.0
        ),
        "finite_outputs": all(
            math.isfinite(value)
            for value in (
                trace_selection["loss"],
                candidate_alert["treatment_detection_rate"],
                candidate_investigation[
                    "action_and_target_hit_at_1"
                ],
            )
        ),
        "candidate_outperforms_alignment_null": (
            beats_alert_null or beats_investigation_null
        ),
    }
    alert_lane = {
        "zero_control_false_alarms": (
            candidate_alert["control_trajectory_false_alarm_rate"]
            == 0.0
        ),
        "treatment_detection_at_least_80_percent": (
            candidate_alert["treatment_detection_rate"] >= 0.80
        ),
        "median_delay_at_most_10": (
            candidate_alert["median_detection_delay_transitions"]
            is not None
            and candidate_alert[
                "median_detection_delay_transitions"
            ]
            <= 10
        ),
        "beats_alignment_null": beats_alert_null,
        "noninferior_to_binned_and_ngram": (
            _alert_noninferior_to_controls(
                candidate_alert, (binned_alert, ngram_alert)
            )
        ),
    }
    investigation_lane = {
        "action_and_target_hit_at_1_at_least_95_percent": (
            candidate_investigation["action_and_target_hit_at_1"]
            >= 0.95
        ),
        "no_action_specificity_100_percent": (
            candidate_investigation["no_action_specificity"] == 1.0
        ),
        "correct_beats_ablations_at_least_80_percent": (
            candidate_investigation[
                "correct_beats_action_ablations_rate"
            ]
            >= 0.80
        ),
        "improves_10_points_over_binned": (
            candidate_investigation["action_and_target_hit_at_1"]
            - binned_investigation["action_and_target_hit_at_1"]
            >= 0.10 - 1e-12
        ),
        "improves_10_points_over_alignment_null": (
            candidate_investigation["action_and_target_hit_at_1"]
            - null_investigation["action_and_target_hit_at_1"]
            >= 0.10 - 1e-12
        ),
    }
    alert_lane["passed"] = bool(all(alert_lane.values()))
    investigation_lane["passed"] = bool(
        all(investigation_lane.values())
    )
    safety_passed = bool(all(safety_gates.values()))
    advance = safety_passed and (
        alert_lane["passed"] or investigation_lane["passed"]
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "event_native_trace_jepa_prototype_result_v1",
        "evidence_boundary": (
            "single-seed open-development tracer; not sealed "
            "confirmation or production evidence"
        ),
        "source_corpus_artifact_manifest_sha256": _file_sha256(
            corpus_directory / "artifact-manifest.json"
        ),
        "source_preprocessing_manifest_sha256": _file_sha256(
            resolved_cache / "artifact-manifest.json"
        ),
        "seed": SEED,
        "data_audit": data_audit,
        "vocabulary": {
            "templates": list(vocabulary.templates),
            "entities": list(vocabulary.entities),
        },
        "masked_trace_encoder": {
            "selection": trace_selection,
            "training": trace_training,
        },
        "representation": representation,
        "shuffle_audit": shuffle_audit,
        "selection_objectives": {
            "event_native_trace_jepa": _role_objective(
                candidate_model, candidate_data["selection"]
            ),
            "binned_event": _role_objective(
                binned_model, binned_data["selection"]
            ),
            "alignment_shuffled": _role_objective(
                shuffled_model, candidate_data["selection"]
            ),
        },
        "alerting": {
            "held_out_topology": alert_transfer,
            "in_distribution": alert_iid,
        },
        "investigation": {
            "held_out_topology": investigation_transfer,
            "in_distribution": investigation_iid,
        },
        "training": {
            "event_native_trace_jepa": candidate_training,
            "binned_event": binned_training,
            "alignment_shuffled": shuffled_training,
        },
        "edge_cost": _model_cost(
            trace_encoder,
            candidate_model,
            candidate_data["transfer"],
        ),
        "gates": {
            "safety": safety_gates,
            "safety_passed": safety_passed,
            "alert_lane": alert_lane,
            "investigation_lane": investigation_lane,
        },
        "decision": (
            "advance_to_durable_three_seed_implementation"
            if advance
            else "reject_prototype_recipe"
        ),
        "limitations": [
            "the source corpus and evaluation roles were already open",
            "only one held-out worker topology and one stack are tested",
            "span templates and request paths are intentionally bounded",
            "CPU timings are local microbenchmarks",
            "the prototype does not implement durable public seams",
        ],
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch": torch.__version__,
            "deterministic_algorithms": (
                torch.are_deterministic_algorithms_enabled()
            ),
            "device": "cpu",
        },
    }
    output_directory.mkdir(parents=True)
    (output_directory / "prototype-result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "safety": safety_gates,
                "alert_lane": alert_lane,
                "investigation_lane": investigation_lane,
                "held_out_alert": {
                    name: {
                        key: value
                        for key, value in metrics.items()
                        if key
                        in {
                            "control_trajectory_false_alarm_rate",
                            "treatment_detection_rate",
                            "median_detection_delay_transitions",
                        }
                    }
                    for name, metrics in alert_transfer.items()
                },
                "held_out_investigation": {
                    name: {
                        key: value
                        for key, value in metrics.items()
                        if key
                        in {
                            "action_and_target_hit_at_1",
                            "no_action_specificity",
                        }
                    }
                    for name, metrics in investigation_transfer.items()
                },
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return result


def main(arguments: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("artifacts/action-dynamics/development-v1"),
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(
            "artifacts/action-dynamics/edge-preprocessing-v1"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/action-dynamics/"
            "prototype-event-native-trace-jepa-v1"
        ),
    )
    parsed = parser.parse_args(arguments)
    run_prototype(
        corpus_directory=parsed.corpus,
        cache_directory=parsed.cache,
        output_directory=parsed.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
