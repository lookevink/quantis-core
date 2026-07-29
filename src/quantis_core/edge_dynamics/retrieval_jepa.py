"""Episode-predictive JEPA retrieval and assessment primitives.

The public seams in this module keep causal query encoding separate from
action-bearing episode compilation and retrieval assessment.
"""

import base64
import io
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple, cast

import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn

from ..action_conditioned_dynamics import ActionConditionedWindows
from ..graph_telemetry import DeclaredTelemetryGraph
from .complete_lejepa import (
    PairBlockedAnchorSchedule,
    fit_owned_feature_mask,
)


@dataclass(frozen=True)
class RetrievalEpisodes:
    """One treatment and one matched-control query per independent pair."""

    contexts: NDArray[np.float64]
    evidence: NDArray[np.float64]
    episode_ids: Tuple[str, ...]
    pair_ids: Tuple[str, ...]
    trajectory_ids: Tuple[str, ...]
    transition_indices: NDArray[np.int64]
    is_treatment: NDArray[np.bool_]
    action_and_target_labels: Tuple[str, ...]
    evidence_refs: Tuple[str, ...]
    topology_values: NDArray[np.float64]

    def __post_init__(self) -> None:
        count = len(self.episode_ids)
        if (
            self.contexts.ndim != 4
            or self.evidence.ndim != 4
            or len(self.contexts) != count
            or len(self.evidence) != count
            or len(self.pair_ids) != count
            or len(self.trajectory_ids) != count
            or self.transition_indices.shape != (count,)
            or self.is_treatment.shape != (count,)
            or len(self.action_and_target_labels) != count
            or len(self.evidence_refs) != count
            or self.topology_values.shape != (count,)
            or len(set(self.episode_ids)) != count
            or not np.all(np.isfinite(self.contexts))
            or not np.all(np.isfinite(self.evidence))
            or not np.all(np.isfinite(self.topology_values))
        ):
            raise ValueError("retrieval episodes do not align")


def compile_retrieval_episodes(
    windows: ActionConditionedWindows,
) -> RetrievalEpisodes:
    """Compile one causal midpoint query and evidence slice per trajectory."""

    try:
        applicable_position = windows.action_feature_names.index("applicable")
        active_position = windows.action_feature_names.index("phase:active")
        elapsed_position = windows.action_feature_names.index(
            "elapsed_fraction"
        )
    except ValueError as error:
        raise ValueError(
            "retrieval compiler requires applicable, active, and elapsed action fields"
        ) from error
    kind_positions = tuple(
        (position, name.removeprefix("kind:"))
        for position, name in enumerate(windows.action_feature_names)
        if name.startswith("kind:")
    )
    if not kind_positions:
        raise ValueError("retrieval compiler requires action kind fields")
    topology_position = (
        windows.control_feature_names.index("worker_replicas")
        if "worker_replicas" in windows.control_feature_names
        else 0
    )
    pair_array = np.asarray(windows.matched_pair_ids, dtype=str)
    rows_out: List[int] = []
    labels: List[str] = []
    treatments: List[bool] = []
    for pair_id in sorted(set(windows.matched_pair_ids)):
        pair_rows = np.flatnonzero(pair_array == pair_id)
        trajectory_ids = tuple(
            sorted({windows.trajectory_ids[int(row)] for row in pair_rows})
        )
        if len(trajectory_ids) != 2:
            raise ValueError(
                "each retrieval pair must contain exactly two trajectories"
            )
        qualifying_by_trajectory: Dict[
            str, List[Tuple[int, int]]
        ] = {}
        for trajectory_id in trajectory_ids:
            trajectory_rows = [
                int(row)
                for row in pair_rows
                if windows.trajectory_ids[int(row)] == trajectory_id
            ]
            qualifying = []
            for row in trajectory_rows:
                action = windows.future_actions[row, 0]
                positions = np.argwhere(
                    (action[:, applicable_position] > 0.5)
                    & (action[:, active_position] > 0.5)
                    & (action[:, elapsed_position] >= 0.5)
                ).reshape(-1)
                if len(positions):
                    if len(positions) != 1:
                        raise ValueError(
                            "retrieval treatment has multiple target entities"
                        )
                    qualifying.append((row, int(positions[0])))
            qualifying_by_trajectory[trajectory_id] = qualifying
        treatment_ids = [
            trajectory_id
            for trajectory_id, qualifying in qualifying_by_trajectory.items()
            if qualifying
        ]
        if len(treatment_ids) != 1:
            raise ValueError(
                "each retrieval pair must have one unambiguous treatment"
            )
        treatment_id = treatment_ids[0]
        treatment_candidates = qualifying_by_trajectory[treatment_id]
        treatment_row, target_position = min(
            treatment_candidates,
            key=lambda item: (
                int(windows.transition_indices[item[0]]),
                item[0],
            ),
        )
        transition = int(windows.transition_indices[treatment_row])
        action = windows.future_actions[treatment_row, 0, target_position]
        active_kinds = [
            name for position, name in kind_positions if action[position] > 0.5
        ]
        if len(active_kinds) != 1:
            raise ValueError(
                "retrieval treatment must have exactly one action kind"
            )
        control_id = next(
            value for value in trajectory_ids if value != treatment_id
        )
        control_rows = [
            int(row)
            for row in pair_rows
            if windows.trajectory_ids[int(row)] == control_id
            and int(windows.transition_indices[int(row)]) == transition
        ]
        if len(control_rows) != 1:
            raise ValueError(
                "retrieval matched control transition is not unique"
            )
        rows_out.extend((treatment_row, control_rows[0]))
        labels.extend(
            (f"{active_kinds[0]}@{windows.entity_names[target_position]}", "no_action")
        )
        treatments.extend((True, False))
    indices = np.asarray(rows_out, dtype=np.int64)
    episode_ids = tuple(
        (
            f"{windows.trajectory_ids[row]}"
            f"#transition={int(windows.transition_indices[row])}"
        )
        for row in rows_out
    )
    return RetrievalEpisodes(
        contexts=windows.histories[indices].copy(),
        evidence=windows.future_states[indices].copy(),
        episode_ids=episode_ids,
        pair_ids=tuple(windows.matched_pair_ids[row] for row in rows_out),
        trajectory_ids=tuple(
            windows.trajectory_ids[row] for row in rows_out
        ),
        transition_indices=windows.transition_indices[indices].copy(),
        is_treatment=np.asarray(treatments, dtype=np.bool_),
        action_and_target_labels=tuple(labels),
        evidence_refs=tuple(
            f"{episode_id}#evidence=+1:+10" for episode_id in episode_ids
        ),
        topology_values=windows.future_controls[
            indices, 0, topology_position
        ].copy(),
    )


_RETRIEVAL_OBJECTIVES = (
    "episode_predictive_jepa",
    "deranged_target_jepa",
    "cpc_infonce",
    "supervised_retriever",
)


@dataclass(frozen=True)
class EpisodeRetrievalConfig:
    """Frozen training controls for a width-64 episode retriever."""

    objective: str = "episode_predictive_jepa"
    width: int = 64
    block_count: int = 2
    head_count: int = 4
    feedforward_width: int = 128
    steps: int = 400
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    ema: float = 0.996
    temperature: float = 0.07
    expected_pair_count: int = 40
    seed: int = 9019

    def __post_init__(self) -> None:
        if (
            self.objective not in _RETRIEVAL_OBJECTIVES
            or self.width != 64
            or self.block_count != 2
            or self.head_count != 4
            or self.feedforward_width != 128
            or isinstance(self.steps, bool)
            or self.steps < 1
            or not 0.0 < float(self.learning_rate)
            or not 0.0 <= float(self.weight_decay)
            or not 0.0 < float(self.ema) < 1.0
            or not 0.0 < float(self.temperature)
            or isinstance(self.expected_pair_count, bool)
            or self.expected_pair_count < 2
            or isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
        ):
            raise ValueError("episode retrieval configuration is invalid")


@dataclass(frozen=True)
class RetrievalEmbeddingBatch:
    """Finite unit vectors in the frozen retrieval space."""

    vectors: NDArray[np.float64]
    graph_entity_ids: Tuple[str, ...]
    representation_kind: str

    def __post_init__(self) -> None:
        if (
            self.vectors.ndim != 2
            or self.vectors.shape[1] < 1
            or not np.all(np.isfinite(self.vectors))
            or not np.allclose(
                np.linalg.norm(self.vectors, axis=1), 1.0, atol=1e-6
            )
        ):
            raise ValueError("retrieval embedding batch is invalid")


class _TelemetryTokenEncoder(nn.Module):
    def __init__(
        self,
        *,
        feature_count: int,
        entity_count: int,
        config: EpisodeRetrievalConfig,
    ) -> None:
        super().__init__()
        self.feature_projection = nn.Linear(feature_count, config.width)
        self.time_embedding = nn.Embedding(30, config.width)
        self.entity_embedding = nn.Embedding(entity_count, config.width)
        layer = nn.TransformerEncoderLayer(
            d_model=config.width,
            nhead=config.head_count,
            dim_feedforward=config.feedforward_width,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            layer, num_layers=config.block_count
        )
        self.output_norm = nn.LayerNorm(config.width)
        self.entity_count = entity_count

    def forward(
        self, values: torch.Tensor, *, time_offset: int = 0
    ) -> torch.Tensor:
        batch, time_count, entity_count, _ = values.shape
        if entity_count != self.entity_count:
            raise ValueError("retrieval encoder entity count is invalid")
        tokens = self.feature_projection(
            values.reshape(batch, time_count * entity_count, -1)
        )
        times = (
            torch.arange(
                time_offset,
                time_offset + time_count,
                device=values.device,
            )[:, None]
            .expand(time_count, entity_count)
            .reshape(-1)
        )
        entities = (
            torch.arange(entity_count, device=values.device)[None, :]
            .expand(time_count, entity_count)
            .reshape(-1)
        )
        tokens = (
            tokens
            + self.time_embedding(times)[None, :]
            + self.entity_embedding(entities)[None, :]
        )
        return cast(
            torch.Tensor, self.output_norm(self.blocks(tokens))
        )


class _PositionalEvidencePredictor(nn.Module):
    def __init__(
        self, *, entity_count: int, config: EpisodeRetrievalConfig
    ) -> None:
        super().__init__()
        target_count = 10 * entity_count
        self.mask_tokens = nn.Parameter(
            torch.zeros(target_count, config.width)
        )
        self.time_embedding = nn.Embedding(30, config.width)
        self.entity_embedding = nn.Embedding(entity_count, config.width)
        self.context_projection = nn.Linear(config.width, config.width)
        layer = nn.TransformerEncoderLayer(
            d_model=config.width,
            nhead=config.head_count,
            dim_feedforward=config.feedforward_width,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            layer, num_layers=config.block_count
        )
        self.output_norm = nn.LayerNorm(config.width)
        self.entity_count = entity_count

    def forward(self, context_tokens: torch.Tensor) -> torch.Tensor:
        batch = len(context_tokens)
        summary = self.context_projection(context_tokens.mean(dim=1))
        times = (
            torch.arange(20, 30, device=context_tokens.device)[:, None]
            .expand(10, self.entity_count)
            .reshape(-1)
        )
        entities = (
            torch.arange(
                self.entity_count, device=context_tokens.device
            )[None, :]
            .expand(10, self.entity_count)
            .reshape(-1)
        )
        targets = (
            self.mask_tokens
            + self.time_embedding(times)
            + self.entity_embedding(entities)
        )
        sequence = torch.cat(
            (
                summary[:, None, :],
                targets[None, :, :].expand(batch, -1, -1),
            ),
            dim=1,
        )
        return cast(
            torch.Tensor,
            self.output_norm(self.blocks(sequence)[:, 1:]),
        )


class _EpisodeRetrievalNetwork(nn.Module):
    def __init__(
        self,
        *,
        feature_count: int,
        entity_count: int,
        config: EpisodeRetrievalConfig,
    ) -> None:
        super().__init__()
        self.context_encoder = _TelemetryTokenEncoder(
            feature_count=feature_count,
            entity_count=entity_count,
            config=config,
        )
        self.target_encoder = _TelemetryTokenEncoder(
            feature_count=feature_count,
            entity_count=entity_count,
            config=config,
        )
        self.predictor = _PositionalEvidencePredictor(
            entity_count=entity_count, config=config
        )
        self.target_encoder.load_state_dict(
            self.context_encoder.state_dict()
        )
        self.entity_count = entity_count

    def query_tokens(self, contexts: torch.Tensor) -> torch.Tensor:
        return cast(
            torch.Tensor,
            self.predictor(self.context_encoder(contexts)),
        )

    def evidence_tokens(
        self, contexts: torch.Tensor, evidence: torch.Tensor
    ) -> torch.Tensor:
        complete = torch.cat((contexts, evidence), dim=1)
        return cast(
            torch.Tensor,
            self.target_encoder(complete)[
                :, -10 * self.entity_count :
            ],
        )


class EpisodeRetrievalRepresentation:
    """Restorable episode-predictive or matched-control retriever."""

    kind = "episode_retrieval_representation"
    schema_version = 1

    def __init__(self, config: EpisodeRetrievalConfig) -> None:
        self.config = config
        self._graph: Optional[DeclaredTelemetryGraph] = None
        self._feature_names: Optional[Tuple[str, ...]] = None
        self._ownership_mask: Optional[NDArray[np.bool_]] = None
        self._center: Optional[NDArray[np.float64]] = None
        self._scale: Optional[NDArray[np.float64]] = None
        self._network: Optional[_EpisodeRetrievalNetwork] = None
        self._training_losses: Tuple[float, ...] = ()

    @property
    def training_losses(self) -> Tuple[float, ...]:
        self._fitted_values()
        return self._training_losses

    @property
    def inference_parameter_count(self) -> int:
        network = self._fitted_values()[-1]
        return sum(
            parameter.numel()
            for module in (network.context_encoder, network.predictor)
            for parameter in module.parameters()
        )

    @property
    def retained_parameter_count(self) -> int:
        network = self._fitted_values()[-1]
        return sum(parameter.numel() for parameter in network.parameters())

    def fit(
        self, windows: ActionConditionedWindows
    ) -> "EpisodeRetrievalRepresentation":
        """Fit using only the supplied role and the objective's frozen truth."""

        schedule = PairBlockedAnchorSchedule(
            windows, seed=self.config.seed + 101
        )
        if len(schedule.pair_ids) != self.config.expected_pair_count:
            raise ValueError("retrieval fit pair count does not match contract")
        ownership = fit_owned_feature_mask(windows)
        complete = np.concatenate(
            (windows.histories, windows.future_states), axis=1
        )
        center = np.mean(complete, axis=(0, 1))
        scale = np.std(complete, axis=(0, 1))
        scale = np.where(scale > 1e-9, scale, 1.0)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.config.seed)
            network = _EpisodeRetrievalNetwork(
                feature_count=len(windows.state_feature_names),
                entity_count=len(windows.entity_names),
                config=self.config,
            )
        trainable = list(network.context_encoder.parameters()) + list(
            network.predictor.parameters()
        )
        if self.config.objective in ("cpc_infonce", "supervised_retriever"):
            trainable += list(network.target_encoder.parameters())
        else:
            for parameter in network.target_encoder.parameters():
                parameter.requires_grad_(False)
        optimizer = torch.optim.AdamW(
            trainable,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self._graph = windows.graph
        self._feature_names = windows.state_feature_names
        self._ownership_mask = ownership.copy()
        self._center = center
        self._scale = scale
        self._network = network
        losses = []
        supervised_episodes = (
            compile_retrieval_episodes(windows)
            if self.config.objective == "supervised_retriever"
            else None
        )
        for step in range(self.config.steps):
            if supervised_episodes is not None:
                selection = supervised_episodes.is_treatment
                contexts = supervised_episodes.contexts[selection]
                evidence = supervised_episodes.evidence[selection]
                labels = tuple(
                    label
                    for label, selected in zip(
                        supervised_episodes.action_and_target_labels,
                        selection,
                    )
                    if selected
                )
            else:
                batch = schedule.batch(step)
                contexts = windows.histories[batch.indices]
                evidence = windows.future_states[batch.indices]
                labels = ()
            context_tensor = self._tensor(self._normalize(contexts))
            evidence_tensor = self._tensor(self._normalize(evidence))
            optimizer.zero_grad(set_to_none=True)
            predicted = network.query_tokens(context_tensor)
            target = network.evidence_tokens(
                context_tensor, evidence_tensor
            )
            if self.config.objective in (
                "episode_predictive_jepa",
                "deranged_target_jepa",
            ):
                target = target.detach()
                if self.config.objective == "deranged_target_jepa":
                    target = torch.roll(target, shifts=1, dims=0)
                short_start = 5 * len(windows.entity_names)
                loss = 0.5 * torch.mean(torch.abs(predicted - target))
                loss = loss + 0.5 * torch.mean(
                    torch.abs(
                        predicted[:, short_start:]
                        - target[:, short_start:]
                    )
                )
            else:
                query_vectors = torch.nn.functional.normalize(
                    predicted.mean(dim=1), dim=1
                )
                evidence_vectors = torch.nn.functional.normalize(
                    target.mean(dim=1), dim=1
                )
                similarities = (
                    query_vectors @ evidence_vectors.T
                ) / self.config.temperature
                if self.config.objective == "cpc_infonce":
                    expected = torch.arange(
                        len(similarities), device=similarities.device
                    )
                    loss = 0.5 * (
                        torch.nn.functional.cross_entropy(
                            similarities, expected
                        )
                        + torch.nn.functional.cross_entropy(
                            similarities.T, expected
                        )
                    )
                else:
                    loss = _multi_positive_retrieval_loss(
                        similarities, labels
                    )
            if not torch.isfinite(loss):
                raise ValueError("retrieval training loss is non-finite")
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            if self.config.objective in (
                "episode_predictive_jepa",
                "deranged_target_jepa",
            ):
                _ema_update(
                    network.target_encoder,
                    network.context_encoder,
                    momentum=self.config.ema,
                )
            losses.append(float(loss.detach()))
        self._training_losses = tuple(losses)
        return self

    def encode_queries(
        self,
        contexts: NDArray[np.float64],
        graph: DeclaredTelemetryGraph,
    ) -> RetrievalEmbeddingBatch:
        graph_, _, _, _, _, network = self._fitted_values()
        if graph.to_dict() != graph_.to_dict():
            raise ValueError("retrieval query graph does not match fit graph")
        with torch.no_grad():
            tokens = network.query_tokens(
                self._tensor(self._normalize(contexts))
            )
            vectors = torch.nn.functional.normalize(
                tokens.mean(dim=1), dim=1, eps=1e-12
            )
        return RetrievalEmbeddingBatch(
            vectors=vectors.cpu().numpy().astype(np.float64),
            graph_entity_ids=graph_.entity_ids,
            representation_kind=self.config.objective,
        )

    def encode_evidence(
        self,
        contexts: NDArray[np.float64],
        evidence: NDArray[np.float64],
        graph: DeclaredTelemetryGraph,
    ) -> RetrievalEmbeddingBatch:
        graph_, _, _, _, _, network = self._fitted_values()
        if graph.to_dict() != graph_.to_dict():
            raise ValueError("retrieval evidence graph does not match fit graph")
        normalized_contexts = self._normalize(contexts)
        normalized_evidence = self._normalize(evidence)
        if len(normalized_contexts) != len(normalized_evidence):
            raise ValueError("retrieval evidence arrays do not align")
        with torch.no_grad():
            tokens = network.evidence_tokens(
                self._tensor(normalized_contexts),
                self._tensor(normalized_evidence),
            )
            vectors = torch.nn.functional.normalize(
                tokens.mean(dim=1), dim=1, eps=1e-12
            )
        return RetrievalEmbeddingBatch(
            vectors=vectors.cpu().numpy().astype(np.float64),
            graph_entity_ids=graph_.entity_ids,
            representation_kind=self.config.objective,
        )

    def to_dict(self) -> Mapping[str, Any]:
        graph, feature_names, ownership, center, scale, network = (
            self._fitted_values()
        )
        arrays = {
            name: value.detach().cpu().numpy()
            for name, value in network.state_dict().items()
        }
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **arrays)
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "config": asdict(self.config),
            "graph": graph.to_dict(),
            "feature_names": list(feature_names),
            "ownership_mask": ownership.astype(int).tolist(),
            "center": center.tolist(),
            "scale": scale.tolist(),
            "training_losses": list(self._training_losses),
            "state_npz_base64": base64.b64encode(
                buffer.getvalue()
            ).decode("ascii"),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "EpisodeRetrievalRepresentation":
        if (
            payload.get("schema_version") != cls.schema_version
            or payload.get("kind") != cls.kind
        ):
            raise ValueError("unsupported retrieval representation artifact")
        config = EpisodeRetrievalConfig(**dict(payload["config"]))
        model = cls(config)
        model._graph = DeclaredTelemetryGraph.from_dict(dict(payload["graph"]))
        model._feature_names = tuple(
            str(value) for value in payload["feature_names"]
        )
        model._ownership_mask = np.asarray(
            payload["ownership_mask"], dtype=np.bool_
        )
        model._center = np.asarray(payload["center"], dtype=np.float64)
        model._scale = np.asarray(payload["scale"], dtype=np.float64)
        model._training_losses = tuple(
            float(value) for value in payload["training_losses"]
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            network = _EpisodeRetrievalNetwork(
                feature_count=len(model._feature_names),
                entity_count=len(model._graph.entities),
                config=config,
            )
        raw_state = base64.b64decode(str(payload["state_npz_base64"]))
        with np.load(io.BytesIO(raw_state), allow_pickle=False) as arrays:
            state = {
                name: torch.from_numpy(arrays[name].copy())
                for name in arrays.files
            }
        network.load_state_dict(state, strict=True)
        model._network = network
        model._fitted_values()
        return model

    def _normalize(
        self, values: NDArray[np.float64]
    ) -> NDArray[np.float32]:
        graph, feature_names, ownership, center, scale, _ = (
            self._fitted_values()
        )
        array = np.asarray(values, dtype=np.float64)
        if (
            array.ndim != 4
            or array.shape[2:] != (
                len(graph.entities),
                len(feature_names),
            )
            or array.shape[1] not in (10, 20)
            or not np.all(np.isfinite(array))
        ):
            raise ValueError("retrieval telemetry input is invalid")
        normalized = (array - center[None, None]) / scale[None, None]
        normalized = np.where(ownership[None, None], normalized, 0.0)
        return normalized.astype(np.float32)

    @staticmethod
    def _tensor(values: NDArray[np.float32]) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(values))

    def _fitted_values(
        self,
    ) -> Tuple[
        DeclaredTelemetryGraph,
        Tuple[str, ...],
        NDArray[np.bool_],
        NDArray[np.float64],
        NDArray[np.float64],
        _EpisodeRetrievalNetwork,
    ]:
        if (
            self._graph is None
            or self._feature_names is None
            or self._ownership_mask is None
            or self._center is None
            or self._scale is None
            or self._network is None
        ):
            raise RuntimeError("retrieval representation is not fitted")
        return (
            self._graph,
            self._feature_names,
            self._ownership_mask,
            self._center,
            self._scale,
            self._network,
        )


class RawTelemetryRetrievalRepresentation:
    """Fit-standardized raw owned telemetry in a common ten-point space."""

    kind = "raw_telemetry_retrieval_representation"
    schema_version = 1

    def __init__(self) -> None:
        self._graph: Optional[DeclaredTelemetryGraph] = None
        self._feature_names: Optional[Tuple[str, ...]] = None
        self._ownership_mask: Optional[NDArray[np.bool_]] = None
        self._center: Optional[NDArray[np.float64]] = None
        self._scale: Optional[NDArray[np.float64]] = None

    @property
    def ownership_mask(self) -> NDArray[np.bool_]:
        return self._fitted_values()[2].copy()

    def fit(
        self, windows: ActionConditionedWindows
    ) -> "RawTelemetryRetrievalRepresentation":
        ownership = fit_owned_feature_mask(windows)
        complete = np.concatenate(
            (windows.histories, windows.future_states), axis=1
        )
        center = np.mean(complete, axis=(0, 1))
        scale = np.std(complete, axis=(0, 1))
        self._graph = windows.graph
        self._feature_names = windows.state_feature_names
        self._ownership_mask = ownership
        self._center = center
        self._scale = np.where(scale > 1e-9, scale, 1.0)
        return self

    def encode_queries(
        self,
        contexts: NDArray[np.float64],
        graph: DeclaredTelemetryGraph,
    ) -> RetrievalEmbeddingBatch:
        return self._encode(np.asarray(contexts)[:, -10:], graph)

    def encode_evidence(
        self,
        contexts: NDArray[np.float64],
        evidence: NDArray[np.float64],
        graph: DeclaredTelemetryGraph,
    ) -> RetrievalEmbeddingBatch:
        if len(contexts) != len(evidence):
            raise ValueError("raw retrieval evidence arrays do not align")
        return self._encode(evidence, graph)

    def to_dict(self) -> Mapping[str, Any]:
        graph, feature_names, ownership, center, scale = (
            self._fitted_values()
        )
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "graph": graph.to_dict(),
            "feature_names": list(feature_names),
            "ownership_mask": ownership.astype(int).tolist(),
            "center": center.tolist(),
            "scale": scale.tolist(),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "RawTelemetryRetrievalRepresentation":
        if (
            payload.get("schema_version") != cls.schema_version
            or payload.get("kind") != cls.kind
        ):
            raise ValueError("unsupported raw retrieval artifact")
        model = cls()
        model._graph = DeclaredTelemetryGraph.from_dict(dict(payload["graph"]))
        model._feature_names = tuple(
            str(value) for value in payload["feature_names"]
        )
        model._ownership_mask = np.asarray(
            payload["ownership_mask"], dtype=np.bool_
        )
        model._center = np.asarray(payload["center"], dtype=np.float64)
        model._scale = np.asarray(payload["scale"], dtype=np.float64)
        model._fitted_values()
        return model

    def _encode(
        self,
        segments: NDArray[np.float64],
        graph: DeclaredTelemetryGraph,
    ) -> RetrievalEmbeddingBatch:
        values = self._flatten(segments, graph)
        return RetrievalEmbeddingBatch(
            vectors=_l2_normalize(values),
            graph_entity_ids=graph.entity_ids,
            representation_kind="raw_telemetry",
        )

    def _flatten(
        self,
        segments: NDArray[np.float64],
        graph: DeclaredTelemetryGraph,
    ) -> NDArray[np.float64]:
        graph_, feature_names, ownership, center, scale = (
            self._fitted_values()
        )
        values = np.asarray(segments, dtype=np.float64)
        if (
            graph.to_dict() != graph_.to_dict()
            or values.ndim != 4
            or values.shape[1:] != (
                10,
                len(graph_.entities),
                len(feature_names),
            )
            or not np.all(np.isfinite(values))
        ):
            raise ValueError("raw retrieval segment is invalid")
        normalized = (values - center[None, None]) / scale[None, None]
        return cast(
            NDArray[np.float64],
            normalized[:, :, ownership].reshape(len(values), -1),
        )

    def _fitted_values(
        self,
    ) -> Tuple[
        DeclaredTelemetryGraph,
        Tuple[str, ...],
        NDArray[np.bool_],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        if (
            self._graph is None
            or self._feature_names is None
            or self._ownership_mask is None
            or self._center is None
            or self._scale is None
        ):
            raise RuntimeError("raw retrieval representation is not fitted")
        return (
            self._graph,
            self._feature_names,
            self._ownership_mask,
            self._center,
            self._scale,
        )


class PcaRetrievalRepresentation:
    """Deterministic fit-only PCA over the shared raw retrieval space."""

    kind = "pca_retrieval_representation"
    schema_version = 1

    def __init__(self, *, width: int = 64) -> None:
        if isinstance(width, bool) or not isinstance(width, int) or width < 1:
            raise ValueError("retrieval PCA width must be positive")
        self.width = width
        self._raw: Optional[RawTelemetryRetrievalRepresentation] = None
        self._center: Optional[NDArray[np.float64]] = None
        self._components: Optional[NDArray[np.float64]] = None

    def fit(
        self, windows: ActionConditionedWindows
    ) -> "PcaRetrievalRepresentation":
        raw = RawTelemetryRetrievalRepresentation().fit(windows)
        query = raw._flatten(windows.histories[:, -10:], windows.graph)
        evidence = raw._flatten(windows.future_states, windows.graph)
        values = np.concatenate((query, evidence), axis=0)
        center = np.mean(values, axis=0)
        _, _, right = np.linalg.svd(values - center, full_matrices=False)
        component_count = min(self.width, len(right))
        components = right[:component_count].copy()
        _orient_pca_components(components)
        self._raw = raw
        self._center = center
        self._components = components
        return self

    def encode_queries(
        self,
        contexts: NDArray[np.float64],
        graph: DeclaredTelemetryGraph,
    ) -> RetrievalEmbeddingBatch:
        raw, _, _ = self._fitted_values()
        return self._encode(raw._flatten(np.asarray(contexts)[:, -10:], graph), graph)

    def encode_evidence(
        self,
        contexts: NDArray[np.float64],
        evidence: NDArray[np.float64],
        graph: DeclaredTelemetryGraph,
    ) -> RetrievalEmbeddingBatch:
        if len(contexts) != len(evidence):
            raise ValueError("PCA retrieval evidence arrays do not align")
        raw, _, _ = self._fitted_values()
        return self._encode(raw._flatten(evidence, graph), graph)

    def to_dict(self) -> Mapping[str, Any]:
        raw, center, components = self._fitted_values()
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "width": self.width,
            "raw": raw.to_dict(),
            "center": center.tolist(),
            "components": components.tolist(),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "PcaRetrievalRepresentation":
        if (
            payload.get("schema_version") != cls.schema_version
            or payload.get("kind") != cls.kind
        ):
            raise ValueError("unsupported retrieval PCA artifact")
        model = cls(width=int(payload["width"]))
        model._raw = RawTelemetryRetrievalRepresentation.from_dict(
            dict(payload["raw"])
        )
        model._center = np.asarray(payload["center"], dtype=np.float64)
        model._components = np.asarray(
            payload["components"], dtype=np.float64
        ).reshape(-1, len(model._center))
        model._fitted_values()
        return model

    def _encode(
        self,
        raw_values: NDArray[np.float64],
        graph: DeclaredTelemetryGraph,
    ) -> RetrievalEmbeddingBatch:
        _, center, components = self._fitted_values()
        projected = (raw_values - center) @ components.T
        padded = np.zeros((len(projected), self.width), dtype=np.float64)
        padded[:, : projected.shape[1]] = projected
        return RetrievalEmbeddingBatch(
            vectors=_l2_normalize(padded),
            graph_entity_ids=graph.entity_ids,
            representation_kind="pca_64",
        )

    def _fitted_values(
        self,
    ) -> Tuple[
        RawTelemetryRetrievalRepresentation,
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        if (
            self._raw is None
            or self._center is None
            or self._components is None
        ):
            raise RuntimeError("retrieval PCA is not fitted")
        return self._raw, self._center, self._components


class OwnedStateRidgeProbe:
    """Fit-only linear safety probe for the final observed owned state."""

    kind = "owned_state_ridge_probe"
    schema_version = 1

    def __init__(self, *, ridge: float = 1e-3) -> None:
        if (
            isinstance(ridge, bool)
            or not isinstance(ridge, (int, float))
            or float(ridge) <= 0.0
        ):
            raise ValueError("owned-state ridge must be positive")
        self.ridge = float(ridge)
        self._input_center: Optional[NDArray[np.float64]] = None
        self._input_scale: Optional[NDArray[np.float64]] = None
        self._input_mask: Optional[NDArray[np.bool_]] = None
        self._target_center: Optional[NDArray[np.float64]] = None
        self._target_scale: Optional[NDArray[np.float64]] = None
        self._target_varying_mask: Optional[NDArray[np.bool_]] = None
        self._coefficients: Optional[NDArray[np.float64]] = None

    @property
    def target_scale(self) -> NDArray[np.float64]:
        return self._fitted_values()[4].copy()

    @property
    def target_varying_mask(self) -> NDArray[np.bool_]:
        return self._fitted_values()[5].copy()

    def fit(
        self,
        vectors: NDArray[np.float64],
        contexts: NDArray[np.float64],
        ownership_mask: NDArray[np.bool_],
    ) -> "OwnedStateRidgeProbe":
        inputs = np.asarray(vectors, dtype=np.float64)
        histories = np.asarray(contexts, dtype=np.float64)
        ownership = np.asarray(ownership_mask, dtype=np.bool_)
        if (
            inputs.ndim != 2
            or histories.ndim != 4
            or len(inputs) != len(histories)
            or ownership.shape != histories.shape[2:]
            or not np.any(ownership)
            or not np.all(np.isfinite(inputs))
            or not np.all(np.isfinite(histories))
        ):
            raise ValueError("owned-state probe inputs do not align")
        targets = histories[:, -1][:, ownership]
        input_center = np.mean(inputs, axis=0)
        input_scale = np.std(inputs, axis=0)
        input_mask = input_scale > 1e-12
        target_center = np.mean(targets, axis=0)
        raw_target_scale = np.std(targets, axis=0)
        target_varying_mask = raw_target_scale > 1e-12
        target_scale = np.where(target_varying_mask, raw_target_scale, 1.0)
        normalized_input = (
            inputs[:, input_mask] - input_center[input_mask]
        ) / input_scale[input_mask]
        normalized_target = (
            targets - target_center
        ) / target_scale
        gram = normalized_input.T @ normalized_input
        coefficients = np.linalg.solve(
            gram
            + self.ridge * np.eye(gram.shape[0], dtype=np.float64),
            normalized_input.T @ normalized_target,
        )
        self._input_center = input_center
        self._input_scale = np.where(input_scale > 1e-12, input_scale, 1.0)
        self._input_mask = input_mask
        self._target_center = target_center
        self._target_scale = target_scale
        self._target_varying_mask = target_varying_mask
        self._coefficients = coefficients
        return self

    def predict(
        self, vectors: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        (
            input_center,
            input_scale,
            input_mask,
            target_center,
            target_scale,
            _,
            coefficients,
        ) = self._fitted_values()
        values = np.asarray(vectors, dtype=np.float64)
        if (
            values.ndim != 2
            or values.shape[1] != len(input_center)
            or not np.all(np.isfinite(values))
        ):
            raise ValueError("owned-state probe vectors are invalid")
        normalized = (
            values[:, input_mask] - input_center[input_mask]
        ) / input_scale[input_mask]
        return cast(
            NDArray[np.float64],
            (normalized @ coefficients) * target_scale + target_center,
        )

    def to_dict(self) -> Mapping[str, Any]:
        values = self._fitted_values()
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "ridge": self.ridge,
            "input_center": values[0].tolist(),
            "input_scale": values[1].tolist(),
            "input_mask": values[2].astype(int).tolist(),
            "target_center": values[3].tolist(),
            "target_scale": values[4].tolist(),
            "target_varying_mask": values[5].astype(int).tolist(),
            "coefficients": values[6].tolist(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OwnedStateRidgeProbe":
        if (
            payload.get("schema_version") != cls.schema_version
            or payload.get("kind") != cls.kind
        ):
            raise ValueError("unsupported owned-state probe artifact")
        model = cls(ridge=float(payload["ridge"]))
        model._input_center = np.asarray(
            payload["input_center"], dtype=np.float64
        )
        model._input_scale = np.asarray(
            payload["input_scale"], dtype=np.float64
        )
        model._input_mask = np.asarray(
            payload["input_mask"], dtype=np.bool_
        )
        model._target_center = np.asarray(
            payload["target_center"], dtype=np.float64
        )
        model._target_scale = np.asarray(
            payload["target_scale"], dtype=np.float64
        )
        model._target_varying_mask = np.asarray(
            payload["target_varying_mask"], dtype=np.bool_
        )
        model._coefficients = np.asarray(
            payload["coefficients"], dtype=np.float64
        ).reshape(int(np.sum(model._input_mask)), -1)
        model._fitted_values()
        return model

    def _fitted_values(
        self,
    ) -> Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.bool_],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.bool_],
        NDArray[np.float64],
    ]:
        if (
            self._input_center is None
            or self._input_scale is None
            or self._input_mask is None
            or self._target_center is None
            or self._target_scale is None
            or self._target_varying_mask is None
            or self._coefficients is None
        ):
            raise RuntimeError("owned-state probe is not fitted")
        return (
            self._input_center,
            self._input_scale,
            self._input_mask,
            self._target_center,
            self._target_scale,
            self._target_varying_mask,
            self._coefficients,
        )


@dataclass(frozen=True)
class ExactRetrievalResult:
    """Deterministically ranked exact-cosine search evidence."""

    similarities: NDArray[np.float64]
    ranking_indices: NDArray[np.int64]
    class_margins: NDArray[np.float64]
    bank_episode_ids: Tuple[str, ...]
    bank_labels: Tuple[str, ...]
    k: int

    def __post_init__(self) -> None:
        query_count, bank_count = self.similarities.shape
        if (
            self.ranking_indices.shape != (query_count, bank_count)
            or self.class_margins.shape != (query_count,)
            or len(self.bank_episode_ids) != bank_count
            or len(self.bank_labels) != bank_count
            or len(set(self.bank_episode_ids)) != bank_count
            or not 1 <= self.k <= bank_count
            or not np.all(np.isfinite(self.similarities))
            or not np.all(np.isfinite(self.class_margins))
        ):
            raise ValueError("exact retrieval result is invalid")


@dataclass(frozen=True)
class EmpiricalAbstentionPolicy:
    """Calibration-role class-margin threshold without a risk guarantee."""

    threshold: float
    calibration_treatment_accepted_correct_rate: float
    calibration_selective_risk: float
    calibration_control_specificity: float

    def accept(
        self, class_margins: NDArray[np.float64]
    ) -> NDArray[np.bool_]:
        margins = np.asarray(class_margins, dtype=np.float64)
        if margins.ndim != 1 or not np.all(np.isfinite(margins)):
            raise ValueError("retrieval confidence margins are invalid")
        return margins >= self.threshold


def exact_retrieval(
    query_vectors: NDArray[np.float64],
    bank_vectors: NDArray[np.float64],
    *,
    bank_episode_ids: Tuple[str, ...],
    bank_labels: Tuple[str, ...],
    k: int = 3,
) -> ExactRetrievalResult:
    """Rank one immutable bank by exact normalized cosine similarity."""

    queries = np.asarray(query_vectors, dtype=np.float64)
    bank = np.asarray(bank_vectors, dtype=np.float64)
    if (
        queries.ndim != 2
        or bank.ndim != 2
        or queries.shape[1] != bank.shape[1]
        or len(bank) != len(bank_episode_ids)
        or len(bank) != len(bank_labels)
        or not np.all(np.isfinite(queries))
        or not np.all(np.isfinite(bank))
        or not np.allclose(np.linalg.norm(queries, axis=1), 1.0, atol=1e-6)
        or not np.allclose(np.linalg.norm(bank, axis=1), 1.0, atol=1e-6)
    ):
        raise ValueError("exact retrieval vectors do not align")
    similarities = queries @ bank.T
    return _retrieval_from_similarities(
        similarities,
        bank_episode_ids=bank_episode_ids,
        bank_labels=bank_labels,
        k=k,
    )


def fit_empirical_abstention(
    retrieval: ExactRetrievalResult,
    *,
    query_labels: Tuple[str, ...],
    is_treatment: NDArray[np.bool_],
) -> EmpiricalAbstentionPolicy:
    """Fit the frozen zero-control-error empirical threshold."""

    treatment = np.asarray(is_treatment, dtype=np.bool_)
    if (
        len(query_labels) != len(retrieval.similarities)
        or treatment.shape != (len(query_labels),)
        or not np.any(treatment)
        or np.all(treatment)
    ):
        raise ValueError("calibration retrieval truth does not align")
    correct = _topk_correct(retrieval, query_labels)
    abstain_all = float(
        np.nextafter(
            np.max(retrieval.class_margins), float("inf")
        )
    )
    candidates = tuple(
        sorted(set(float(value) for value in retrieval.class_margins))
    ) + (abstain_all,)
    best = None
    for threshold in candidates:
        accepted = retrieval.class_margins >= threshold
        if np.any(accepted & ~treatment):
            continue
        accepted_treatment = accepted & treatment
        accepted_count = int(np.sum(accepted_treatment))
        correct_count = int(np.sum(accepted_treatment & correct))
        risk = (
            1.0 - correct_count / accepted_count
            if accepted_count
            else 0.0
        )
        if risk > 0.10 + 1e-12:
            continue
        accepted_correct_rate = correct_count / int(np.sum(treatment))
        candidate = (
            accepted_correct_rate,
            -risk,
            -threshold,
            threshold,
        )
        if best is None or candidate[:3] > best[:3]:
            best = candidate
    if best is None:
        raise RuntimeError("abstention threshold enumeration failed")
    threshold = float(best[3])
    accepted = retrieval.class_margins >= threshold
    accepted_treatment = accepted & treatment
    accepted_count = int(np.sum(accepted_treatment))
    correct_count = int(np.sum(accepted_treatment & correct))
    return EmpiricalAbstentionPolicy(
        threshold=threshold,
        calibration_treatment_accepted_correct_rate=(
            correct_count / int(np.sum(treatment))
        ),
        calibration_selective_risk=(
            1.0 - correct_count / accepted_count
            if accepted_count
            else 0.0
        ),
        calibration_control_specificity=float(
            np.mean(~accepted[~treatment])
        ),
    )


def assess_retrieval_jepa(
    *,
    gallery_episode_ids: Tuple[str, ...],
    gallery_labels: Tuple[str, ...],
    similarities: Mapping[str, Mapping[str, NDArray[np.float64]]],
    query_labels: Mapping[str, Tuple[str, ...]],
    is_treatment: Mapping[str, NDArray[np.bool_]],
    pair_ids: Mapping[str, Tuple[str, ...]],
    bank_vectors: Mapping[str, NDArray[np.float64]],
    restored_bank_vectors: Mapping[str, NDArray[np.float64]],
    state_truth: NDArray[np.float64],
    state_scale: NDArray[np.float64],
    state_varying_mask: NDArray[np.bool_],
    state_predictions: Mapping[str, NDArray[np.float64]],
    original_query_vectors: Mapping[
        str, Mapping[str, NDArray[np.float64]]
    ],
    restored_query_vectors: Mapping[
        str, Mapping[str, NDArray[np.float64]]
    ],
    protocol_checks: Mapping[str, bool],
    edge_metrics: Mapping[str, Mapping[str, float]],
) -> Mapping[str, Any]:
    """Recompute the frozen retrieval metrics, gates, and bounded decision."""

    model_names = (
        "episode_predictive_jepa",
        "raw_telemetry",
        "pca_64",
        "deranged_target_jepa",
        "cpc_infonce",
        "supervised_retriever",
    )
    required_roles = (
        "calibration",
        "selection_iid",
        "selection_transfer",
        "evaluation_iid",
        "evaluation_transfer",
    )
    if (
        tuple(similarities) != required_roles
        or any(set(similarities[role]) != set(model_names) for role in required_roles)
        or set(bank_vectors) != set(model_names)
        or set(restored_bank_vectors) != set(model_names)
        or set(state_predictions) != set(model_names)
        or tuple(original_query_vectors) != required_roles
        or tuple(restored_query_vectors) != required_roles
        or any(
            set(original_query_vectors[role]) != set(model_names)
            or set(restored_query_vectors[role]) != set(model_names)
            for role in required_roles
        )
        or set(edge_metrics) != set(model_names)
    ):
        raise ValueError("retrieval assessment evidence is incomplete")
    retrieval: Dict[
        str, Dict[str, ExactRetrievalResult]
    ] = {}
    policies: Dict[str, EmpiricalAbstentionPolicy] = {}
    metrics: Dict[
        str, Dict[str, Mapping[str, Any]]
    ] = {}
    for role in required_roles:
        if (
            role not in query_labels
            or role not in is_treatment
            or role not in pair_ids
            or len(query_labels[role]) != len(pair_ids[role])
        ):
            raise ValueError("retrieval assessment role truth is incomplete")
        retrieval[role] = {}
        metrics[role] = {}
        for name in model_names:
            result = _retrieval_from_similarities(
                similarities[role][name],
                bank_episode_ids=gallery_episode_ids,
                bank_labels=gallery_labels,
                k=3,
            )
            retrieval[role][name] = result
            if role == "calibration":
                policies[name] = fit_empirical_abstention(
                    result,
                    query_labels=query_labels[role],
                    is_treatment=is_treatment[role],
                )
    for role in required_roles:
        for name in model_names:
            metrics[role][name] = _retrieval_metrics(
                retrieval[role][name],
                query_labels=query_labels[role],
                is_treatment=is_treatment[role],
                pair_ids=pair_ids[role],
                policy=policies[name],
            )
    truth = np.asarray(state_truth, dtype=np.float64)
    scale = np.asarray(state_scale, dtype=np.float64)
    varying = np.asarray(state_varying_mask, dtype=np.bool_)
    if (
        truth.ndim != 2
        or scale.shape != (truth.shape[1],)
        or varying.shape != scale.shape
        or not np.any(varying)
        or np.any(scale <= 0.0)
        or not np.all(np.isfinite(truth))
        or not np.all(np.isfinite(scale))
    ):
        raise ValueError("retrieval state-probe evidence is invalid")
    state_nrmse = {}
    effective_rank = {}
    restoration_max_abs = {}
    restoration_ranking_parity = {}
    restoration_accept_decision_parity = {}
    stored_similarity_matches_vectors = {}
    all_numeric_finite = True
    for name in model_names:
        prediction = np.asarray(state_predictions[name], dtype=np.float64)
        bank = np.asarray(bank_vectors[name], dtype=np.float64)
        restored_bank = np.asarray(
            restored_bank_vectors[name], dtype=np.float64
        )
        if prediction.shape != truth.shape or bank.shape != restored_bank.shape:
            raise ValueError("retrieval diagnostic arrays do not align")
        state_nrmse[name] = float(
            np.sqrt(
                np.mean(
                    (
                        ((prediction - truth) / scale)[:, varying]
                    )
                    ** 2
                )
            )
        )
        centered = bank - np.mean(bank, axis=0, keepdims=True)
        singular_values = np.linalg.svd(
            centered, compute_uv=False, full_matrices=False
        )
        effective_rank[name] = int(
            np.sum(
                singular_values
                > (
                    singular_values[0] * 1e-6
                    if len(singular_values)
                    else 0.0
                )
            )
        )
        maximum_difference = float(
            np.max(np.abs(bank - restored_bank))
            if bank.size
            else 0.0
        )
        ranking_parity = True
        accept_parity = True
        similarities_match = True
        for role in required_roles:
            original = np.asarray(
                original_query_vectors[role][name], dtype=np.float64
            )
            restored = np.asarray(
                restored_query_vectors[role][name], dtype=np.float64
            )
            if original.shape != restored.shape:
                raise ValueError(
                    "restored retrieval query arrays do not align"
                )
            maximum_difference = max(
                maximum_difference,
                float(
                    np.max(np.abs(original - restored))
                    if original.size
                    else 0.0
                ),
            )
            original_similarity = original @ bank.T
            restored_similarity = restored @ restored_bank.T
            similarities_match = similarities_match and bool(
                np.allclose(
                    similarities[role][name],
                    original_similarity,
                    atol=1e-10,
                    rtol=1e-10,
                )
            )
            restored_result = _retrieval_from_similarities(
                restored_similarity,
                bank_episode_ids=gallery_episode_ids,
                bank_labels=gallery_labels,
                k=3,
            )
            ranking_parity = ranking_parity and bool(
                np.array_equal(
                    retrieval[role][name].ranking_indices,
                    restored_result.ranking_indices,
                )
            )
            accept_parity = accept_parity and bool(
                np.array_equal(
                    policies[name].accept(
                        retrieval[role][name].class_margins
                    ),
                    policies[name].accept(
                        restored_result.class_margins
                    ),
                )
            )
            all_numeric_finite = all_numeric_finite and all(
                np.all(np.isfinite(value))
                for value in (original, restored)
            )
        restoration_max_abs[name] = maximum_difference
        restoration_ranking_parity[name] = ranking_parity
        restoration_accept_decision_parity[name] = accept_parity
        stored_similarity_matches_vectors[name] = similarities_match
        all_numeric_finite = all_numeric_finite and all(
            np.all(np.isfinite(value))
            for value in (prediction, bank, restored_bank)
        )
    candidate = "episode_predictive_jepa"
    non_supervised = ("raw_telemetry", "pca_64", "cpc_infonce")
    selection = metrics["selection_transfer"]
    evaluation = metrics["evaluation_transfer"]
    best_selection = max(
        non_supervised,
        key=lambda name: selection[name]["accepted_correct_rate"],
    )
    best_evaluation = max(
        non_supervised,
        key=lambda name: evaluation[name]["accepted_correct_rate"],
    )
    best_mrr = max(
        non_supervised,
        key=lambda name: evaluation[name][
            "pair_balanced_mean_reciprocal_rank"
        ],
    )
    best_hit_at_1 = max(
        non_supervised,
        key=lambda name: evaluation[name]["hit_at_1"],
    )
    pair_win_fraction = _non_tied_pair_win_fraction(
        retrieval["evaluation_transfer"][candidate],
        retrieval["evaluation_transfer"][best_hit_at_1],
        query_labels["evaluation_transfer"],
        is_treatment["evaluation_transfer"],
    )
    required_protocol_checks = (
        "role_pairs_are_disjoint",
        "query_future_is_excluded",
        "action_and_identifiers_are_excluded",
        "bank_membership_is_equal_and_immutable",
        "episode_counts_match_contract",
    )
    safety_gates = {
        "protocol_checks_pass": all(
            protocol_checks.get(name) is True
            for name in required_protocol_checks
        ),
        "all_numeric_evidence_is_finite": all_numeric_finite,
        "restoration_within_1e_6": (
            restoration_max_abs[candidate] <= 1e-6
            and restoration_ranking_parity[candidate]
            and restoration_accept_decision_parity[candidate]
        ),
        "stored_similarities_match_vectors": (
            stored_similarity_matches_vectors[candidate]
        ),
        "state_nrmse_within_pca_plus_0_10": (
            state_nrmse[candidate] <= state_nrmse["pca_64"] + 0.10
        ),
        "candidate_bank_effective_rank_at_least_8": (
            effective_rank[candidate] >= 8
        ),
        "online_parameters_at_most_500000": (
            edge_metrics[candidate]["online_parameter_count"] <= 500_000
        ),
        "serialized_model_at_most_10_mib": (
            edge_metrics[candidate]["serialized_model_bytes"]
            <= 10 * 1024 * 1024
        ),
        "query_latency_at_most_100_ms": (
            edge_metrics[candidate]["query_latency_median_ms"] <= 100.0
        ),
        "search_latency_at_most_5_ms": (
            edge_metrics[candidate]["search_latency_median_ms"] <= 5.0
        ),
        "sgr_guarantee_feasible_is_false": True,
    }
    value_gates = {
        "selection_hit_at_3_at_least_0_90": (
            selection[candidate]["hit_at_3"] >= 0.90
        ),
        "selection_accepted_correct_rate_at_least_0_80": (
            selection[candidate]["accepted_correct_rate"] >= 0.80
        ),
        "selection_selective_accuracy_at_least_0_90": (
            selection[candidate]["selective_accuracy"] >= 0.90
        ),
        "selection_control_specificity_is_1": (
            selection[candidate]["control_specificity"] == 1.0
        ),
        "selection_beats_deranged_by_0_05": (
            selection[candidate]["accepted_correct_rate"]
            >= selection["deranged_target_jepa"]["accepted_correct_rate"]
            + 0.05
        ),
        "selection_beats_best_non_supervised_by_0_05": (
            selection[candidate]["accepted_correct_rate"]
            >= selection[best_selection]["accepted_correct_rate"] + 0.05
        ),
        "evaluation_hit_at_1_at_least_0_80": (
            evaluation[candidate]["hit_at_1"] >= 0.80
        ),
        "evaluation_hit_at_3_at_least_0_90": (
            evaluation[candidate]["hit_at_3"] >= 0.90
        ),
        "evaluation_accepted_correct_rate_at_least_0_80": (
            evaluation[candidate]["accepted_correct_rate"] >= 0.80
        ),
        "evaluation_selective_accuracy_at_least_0_90": (
            evaluation[candidate]["selective_accuracy"] >= 0.90
        ),
        "evaluation_control_specificity_is_1": (
            evaluation[candidate]["control_specificity"] == 1.0
        ),
        "evaluation_beats_deranged_by_0_05": (
            evaluation[candidate]["accepted_correct_rate"]
            >= evaluation["deranged_target_jepa"]["accepted_correct_rate"]
            + 0.05
        ),
        "evaluation_beats_best_non_supervised_by_0_05": (
            evaluation[candidate]["accepted_correct_rate"]
            >= evaluation[best_evaluation]["accepted_correct_rate"] + 0.05
        ),
        "evaluation_mrr_beats_best_non_supervised_by_0_05": (
            evaluation[candidate][
                "pair_balanced_mean_reciprocal_rank"
            ]
            >= evaluation[best_mrr][
                "pair_balanced_mean_reciprocal_rank"
            ]
            + 0.05
        ),
        "within_0_05_of_supervised": (
            evaluation[candidate]["accepted_correct_rate"] + 0.05
            >= evaluation["supervised_retriever"]["accepted_correct_rate"]
        ),
        "non_tied_pair_win_fraction_at_least_0_60": (
            pair_win_fraction >= 0.60
        ),
    }
    passed = all(safety_gates.values()) and all(value_gates.values())
    return {
        "schema_version": 1,
        "kind": "retrieval_jepa_assessment",
        "sgr_guarantee_feasible": False,
        "policies": {
            name: {
                "threshold": policy.threshold,
                "calibration_treatment_accepted_correct_rate": (
                    policy.calibration_treatment_accepted_correct_rate
                ),
                "calibration_selective_risk": (
                    policy.calibration_selective_risk
                ),
                "calibration_control_specificity": (
                    policy.calibration_control_specificity
                ),
            }
            for name, policy in policies.items()
        },
        "metrics": metrics,
        "state_nrmse": state_nrmse,
        "bank_effective_rank": effective_rank,
        "restoration_max_abs": restoration_max_abs,
        "restoration_ranking_parity": restoration_ranking_parity,
        "restoration_accept_decision_parity": (
            restoration_accept_decision_parity
        ),
        "stored_similarity_matches_vectors": (
            stored_similarity_matches_vectors
        ),
        "best_selection_non_supervised": best_selection,
        "best_evaluation_non_supervised": best_evaluation,
        "best_evaluation_mrr_non_supervised": best_mrr,
        "best_evaluation_hit_at_1_non_supervised": best_hit_at_1,
        "non_tied_pair_win_fraction": pair_win_fraction,
        "safety_gates": safety_gates,
        "value_gates": value_gates,
        "passed": passed,
        "decision": (
            "advance_episode_predictive_retriever_to_fixed_multiseed_robustness"
            if passed
            else "reject_episode_predictive_retrieval_jepa_recipe"
        ),
    }


def _retrieval_from_similarities(
    similarities: NDArray[np.float64],
    *,
    bank_episode_ids: Tuple[str, ...],
    bank_labels: Tuple[str, ...],
    k: int,
) -> ExactRetrievalResult:
    values = np.asarray(similarities, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1] != len(bank_episode_ids)
        or values.shape[1] != len(bank_labels)
        or len(set(bank_episode_ids)) != len(bank_episode_ids)
        or not 1 <= k <= len(bank_episode_ids)
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("retrieval similarity evidence is invalid")
    ranking = np.empty(values.shape, dtype=np.int64)
    margins = np.empty(len(values), dtype=np.float64)
    for query_position, row in enumerate(values):
        order = tuple(
            sorted(
                range(len(row)),
                key=lambda position: (
                    -float(row[position]),
                    bank_episode_ids[position],
                ),
            )
        )
        ranking[query_position] = order
        top = order[0]
        different = [
            float(row[position])
            for position in order[1:]
            if bank_labels[position] != bank_labels[top]
        ]
        margins[query_position] = (
            float(row[top]) - max(different) if different else 0.0
        )
    return ExactRetrievalResult(
        similarities=values.copy(),
        ranking_indices=ranking,
        class_margins=margins,
        bank_episode_ids=bank_episode_ids,
        bank_labels=bank_labels,
        k=k,
    )


def _topk_correct(
    retrieval: ExactRetrievalResult, query_labels: Tuple[str, ...]
) -> NDArray[np.bool_]:
    if len(query_labels) != len(retrieval.ranking_indices):
        raise ValueError("retrieval query labels do not align")
    return np.asarray(
        [
            label != "no_action"
            and any(
                retrieval.bank_labels[int(position)] == label
                for position in retrieval.ranking_indices[row, : retrieval.k]
            )
            for row, label in enumerate(query_labels)
        ],
        dtype=np.bool_,
    )


def _retrieval_metrics(
    retrieval: ExactRetrievalResult,
    *,
    query_labels: Tuple[str, ...],
    is_treatment: NDArray[np.bool_],
    pair_ids: Tuple[str, ...],
    policy: EmpiricalAbstentionPolicy,
) -> Mapping[str, Any]:
    treatment = np.asarray(is_treatment, dtype=np.bool_)
    if (
        treatment.shape != (len(query_labels),)
        or len(pair_ids) != len(query_labels)
    ):
        raise ValueError("retrieval metric truth does not align")
    for pair_id in set(pair_ids):
        positions = [
            position
            for position, value in enumerate(pair_ids)
            if value == pair_id
        ]
        if (
            len(positions) != 2
            or int(np.sum(treatment[positions])) != 1
        ):
            raise ValueError(
                "retrieval assessment requires one treatment/control pair"
            )
    hit1 = []
    hit3 = []
    reciprocal = []
    first_relevant_ranks = []
    per_action_rows: Dict[str, List[Tuple[bool, bool, float]]] = {}
    for row, label in enumerate(query_labels):
        if not treatment[row]:
            continue
        ranked_labels = [
            retrieval.bank_labels[int(position)]
            for position in retrieval.ranking_indices[row]
        ]
        relevant_ranks = [
            position + 1
            for position, ranked_label in enumerate(ranked_labels)
            if ranked_label == label
        ]
        hit1.append(bool(relevant_ranks and relevant_ranks[0] == 1))
        hit3.append(bool(relevant_ranks and relevant_ranks[0] <= 3))
        first_rank = (
            relevant_ranks[0]
            if relevant_ranks
            else len(retrieval.bank_labels) + 1
        )
        first_relevant_ranks.append(first_rank)
        reciprocal_value = (
            1.0 / relevant_ranks[0] if relevant_ranks else 0.0
        )
        reciprocal.append(reciprocal_value)
        per_action_rows.setdefault(label, []).append(
            (hit1[-1], hit3[-1], reciprocal_value)
        )
    correct = _topk_correct(retrieval, query_labels)
    accepted = policy.accept(retrieval.class_margins)
    accepted_treatment = accepted & treatment
    accepted_count = int(np.sum(accepted_treatment))
    correct_accepted = int(np.sum(accepted_treatment & correct))
    risk_coverage_curve = []
    thresholds = tuple(
        sorted(
            set(float(value) for value in retrieval.class_margins)
        )
    ) + (
        float(
            np.nextafter(
                np.max(retrieval.class_margins), float("inf")
            )
        ),
    )
    for threshold in thresholds:
        curve_accepted = retrieval.class_margins >= threshold
        curve_treatment = curve_accepted & treatment
        curve_count = int(np.sum(curve_treatment))
        curve_correct = int(np.sum(curve_treatment & correct))
        risk_coverage_curve.append(
            {
                "threshold": threshold,
                "treatment_coverage": float(
                    np.mean(curve_accepted[treatment])
                ),
                "selective_risk": (
                    1.0 - curve_correct / curve_count
                    if curve_count
                    else 1.0
                ),
                "accepted_correct_rate": (
                    curve_correct / int(np.sum(treatment))
                ),
                "control_specificity": float(
                    np.mean(~curve_accepted[~treatment])
                ),
            }
        )
    return {
        "hit_at_1": float(np.mean(hit1)),
        "hit_at_3": float(np.mean(hit3)),
        "mean_reciprocal_rank": float(np.mean(reciprocal)),
        "pair_balanced_mean_reciprocal_rank": float(
            np.mean(reciprocal)
        ),
        "first_relevant_ranks": first_relevant_ranks,
        "per_action": {
            label: {
                "count": len(rows),
                "hit_at_1": float(
                    np.mean([row[0] for row in rows])
                ),
                "hit_at_3": float(
                    np.mean([row[1] for row in rows])
                ),
                "mean_reciprocal_rank": float(
                    np.mean([row[2] for row in rows])
                ),
            }
            for label, rows in sorted(per_action_rows.items())
        },
        "treatment_coverage": float(np.mean(accepted[treatment])),
        "selective_accuracy": (
            correct_accepted / accepted_count if accepted_count else 0.0
        ),
        "selective_risk": (
            1.0 - correct_accepted / accepted_count
            if accepted_count
            else 1.0
        ),
        "accepted_correct_rate": (
            correct_accepted / int(np.sum(treatment))
        ),
        "control_specificity": float(np.mean(~accepted[~treatment])),
        "risk_coverage_curve": risk_coverage_curve,
    }


def _non_tied_pair_win_fraction(
    candidate: ExactRetrievalResult,
    control: ExactRetrievalResult,
    query_labels: Tuple[str, ...],
    is_treatment: NDArray[np.bool_],
) -> float:
    treatment = np.asarray(is_treatment, dtype=np.bool_)
    candidate_hit = np.asarray(
        [
            candidate.bank_labels[int(candidate.ranking_indices[row, 0])]
            == query_labels[row]
            for row in range(len(query_labels))
        ]
    )
    control_hit = np.asarray(
        [
            control.bank_labels[int(control.ranking_indices[row, 0])]
            == query_labels[row]
            for row in range(len(query_labels))
        ]
    )
    non_tied = treatment & (candidate_hit != control_hit)
    return (
        float(np.mean(candidate_hit[non_tied]))
        if np.any(non_tied)
        else 0.0
    )


def _l2_normalize(values: NDArray[np.float64]) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 1e-12) or not np.all(np.isfinite(norms)):
        raise ValueError("retrieval vector has zero or non-finite norm")
    return cast(NDArray[np.float64], array / norms)


def _orient_pca_components(components: NDArray[np.float64]) -> None:
    for row in components:
        if len(row):
            pivot = int(np.argmax(np.abs(row)))
            if row[pivot] < 0.0:
                row *= -1.0


def _ema_update(
    target: nn.Module, source: nn.Module, *, momentum: float
) -> None:
    with torch.no_grad():
        target_state = target.state_dict()
        source_state = source.state_dict()
        for name, target_value in target_state.items():
            source_value = source_state[name]
            if torch.is_floating_point(target_value):
                target_value.mul_(momentum).add_(
                    source_value, alpha=1.0 - momentum
                )
            else:
                target_value.copy_(source_value)


def _multi_positive_retrieval_loss(
    similarities: torch.Tensor, labels: Tuple[str, ...]
) -> torch.Tensor:
    if similarities.shape != (len(labels), len(labels)):
        raise ValueError("supervised retrieval labels do not align")
    identity = torch.eye(
        len(labels), dtype=torch.bool, device=similarities.device
    )
    positives = torch.tensor(
        [
            [
                left == right and left != "no_action"
                for right in labels
            ]
            for left in labels
        ],
        dtype=torch.bool,
        device=similarities.device,
    ) & ~identity
    if not torch.all(torch.any(positives, dim=1)):
        raise ValueError(
            "supervised retrieval requires another positive for every query"
        )
    allowed = ~identity
    row_loss = -(
        torch.logsumexp(
            similarities.masked_fill(~positives, -torch.inf), dim=1
        )
        - torch.logsumexp(
            similarities.masked_fill(~allowed, -torch.inf), dim=1
        )
    ).mean()
    column_loss = -(
        torch.logsumexp(
            similarities.T.masked_fill(~positives.T, -torch.inf), dim=1
        )
        - torch.logsumexp(
            similarities.T.masked_fill(~allowed.T, -torch.inf), dim=1
        )
    ).mean()
    return 0.5 * (row_loss + column_loss)
