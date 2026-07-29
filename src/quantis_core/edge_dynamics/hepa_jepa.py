"""Clean-room horizon-conditioned event-predictive JEPA primitives.

The public seam intentionally exposes entity-preserving context tokens and a
monotone event CDF. Action truth is accepted only by the offline event and
assessment helpers; it is never an encoder or event-head input.
"""

import copy
import importlib
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from ..action_conditioned_dynamics import ActionConditionedWindows
from ..graph_telemetry import DeclaredTelemetryGraph
_HEPA_OBJECTIVES = (
    "hepa",
    "horizon_deranged",
    "supervised_scratch",
)
HEPA_MODEL_NAMES = (
    "hepa",
    "horizon_deranged",
    "supervised_scratch",
)
HEPA_ASSESSMENT_ROLE_NAMES = (
    "calibration",
    "evaluation_iid",
    "evaluation_transfer",
)


@dataclass(frozen=True)
class HepaConfig:
    """Frozen controls for the edge-sized HEPA tracer."""

    objective: str = "hepa"
    width: int = 64
    block_count: int = 2
    head_count: int = 4
    feedforward_width: int = 128
    alert_horizon: int = 10
    stage1_steps: int = 400
    stage2_steps: int = 300
    checkpoint_interval: int = 50
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    sigreg_alpha: float = 0.1
    sketch_dimension: int = 64
    knot_count: int = 17
    expected_pair_count: int = 40
    seed: int = 12012
    device: str = "cpu"

    def __post_init__(self) -> None:
        if (
            self.objective not in _HEPA_OBJECTIVES
            or self.width < 4
            or self.block_count != 2
            or self.head_count < 1
            or self.width % self.head_count != 0
            or self.feedforward_width < self.width
            or self.alert_horizon < 1
            or self.stage1_steps < 1
            or self.stage2_steps < 1
            or self.checkpoint_interval < 1
            or self.batch_size < 2
            or not 0.0 < float(self.learning_rate)
            or not 0.0 <= float(self.weight_decay)
            or float(self.sigreg_alpha) != 0.1
            or self.sketch_dimension < 1
            or self.knot_count < 2
            or self.expected_pair_count < 2
            or isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.device != "cpu"
        ):
            raise ValueError("HEPA configuration is invalid")


@dataclass(frozen=True)
class HepaEncodedTelemetry:
    """Entity-preserving public context representation."""

    tokens: NDArray[np.float64]
    entity_ids: Tuple[str, ...]
    ownership_mask: NDArray[np.bool_]

    def __post_init__(self) -> None:
        if (
            self.tokens.ndim != 3
            or self.tokens.shape[1] != len(self.entity_ids)
            or self.ownership_mask.ndim != 2
            or self.ownership_mask.shape[0] != len(self.entity_ids)
            or not np.all(np.isfinite(self.tokens))
        ):
            raise ValueError("HEPA encoded telemetry is invalid")


class HepaEntityPcaBaseline:
    """Fitting-only entity-preserving width-matched PCA baseline."""

    kind = "hepa_entity_pca_baseline"
    schema_version = 1

    def __init__(self, *, width: int = 64) -> None:
        if isinstance(width, bool) or not isinstance(width, int) or width < 1:
            raise ValueError("HEPA PCA width must be positive")
        self.width = width
        self._graph: Optional[DeclaredTelemetryGraph] = None
        self._feature_names: Tuple[str, ...] = ()
        self._ownership_mask: Optional[NDArray[np.bool_]] = None
        self._centers: Tuple[NDArray[np.float64], ...] = ()
        self._components: Tuple[NDArray[np.float64], ...] = ()

    def fit(
        self, windows: ActionConditionedWindows
    ) -> "HepaEntityPcaBaseline":
        """Fit independent local PCA maps on fitting histories."""

        ownership = _fit_owned_feature_mask(windows)
        centers = []
        components = []
        for entity in range(len(windows.entity_names)):
            selected = ownership[entity]
            values = windows.histories[
                :, :, entity, :
            ][:, :, selected].reshape(len(windows.histories), -1)
            center = np.mean(values, axis=0)
            _, _, right = np.linalg.svd(
                values - center, full_matrices=False
            )
            count = min(self.width, len(right))
            local = right[:count].copy()
            _orient_components(local)
            centers.append(center)
            components.append(local)
        self._graph = windows.graph
        self._feature_names = windows.state_feature_names
        self._ownership_mask = ownership.copy()
        self._centers = tuple(centers)
        self._components = tuple(components)
        return self

    def encode(
        self,
        histories: NDArray[np.float64],
        graph: DeclaredTelemetryGraph,
    ) -> HepaEncodedTelemetry:
        """Encode contexts into width-matched entity-local PCA tokens."""

        graph_, features, ownership, centers, components = (
            self._fitted_values()
        )
        values = np.asarray(histories, dtype=np.float64)
        if (
            graph.to_dict() != graph_.to_dict()
            or values.ndim != 4
            or values.shape[1:] != (
                20,
                len(graph_.entities),
                len(features),
            )
            or not np.all(np.isfinite(values))
        ):
            raise ValueError("HEPA PCA encoding inputs are invalid")
        tokens = np.zeros(
            (len(values), len(graph_.entities), self.width),
            dtype=np.float64,
        )
        for entity in range(len(graph_.entities)):
            local = values[
                :, :, entity, :
            ][:, :, ownership[entity]].reshape(len(values), -1)
            count = len(components[entity])
            tokens[:, entity, :count] = (
                local - centers[entity]
            ) @ components[entity].T
        return HepaEncodedTelemetry(
            tokens=tokens,
            entity_ids=graph_.entity_ids,
            ownership_mask=ownership.copy(),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return the fitted baseline for artifact identity."""

        graph, features, ownership, centers, components = (
            self._fitted_values()
        )
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "width": self.width,
            "graph": graph.to_dict(),
            "feature_names": list(features),
            "ownership_mask": ownership.astype(int).tolist(),
            "centers": [value.tolist() for value in centers],
            "components": [value.tolist() for value in components],
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "HepaEntityPcaBaseline":
        """Restore a fitted entity-local PCA baseline."""

        if (
            payload.get("schema_version") != cls.schema_version
            or payload.get("kind") != cls.kind
        ):
            raise ValueError("unsupported HEPA PCA baseline")
        width = payload.get("width")
        if (
            isinstance(width, bool)
            or not isinstance(width, int)
            or width < 1
        ):
            raise ValueError("HEPA PCA width is invalid")
        graph = DeclaredTelemetryGraph.from_dict(
            dict(payload["graph"])
        )
        features = tuple(
            str(value) for value in payload["feature_names"]
        )
        ownership = np.asarray(
            payload["ownership_mask"], dtype=np.bool_
        )
        centers = tuple(
            np.asarray(value, dtype=np.float64)
            for value in payload["centers"]
        )
        components = tuple(
            np.asarray(value, dtype=np.float64)
            for value in payload["components"]
        )
        expected = (len(graph.entities), len(features))
        if (
            not features
            or ownership.shape != expected
            or not np.any(ownership)
            or len(centers) != len(graph.entities)
            or len(components) != len(graph.entities)
        ):
            raise ValueError("HEPA PCA schema is invalid")
        for center, local in zip(centers, components):
            if (
                center.ndim != 1
                or local.ndim != 2
                or local.shape[1] != len(center)
                or len(local) > width
                or not np.all(np.isfinite(center))
                or not np.all(np.isfinite(local))
            ):
                raise ValueError("HEPA PCA tensors are invalid")
        result = cls(width=width)
        result._graph = graph
        result._feature_names = features
        result._ownership_mask = ownership
        result._centers = centers
        result._components = components
        result._fitted_values()
        return result

    def _fitted_values(
        self,
    ) -> Tuple[
        DeclaredTelemetryGraph,
        Tuple[str, ...],
        NDArray[np.bool_],
        Tuple[NDArray[np.float64], ...],
        Tuple[NDArray[np.float64], ...],
    ]:
        if (
            self._graph is None
            or not self._feature_names
            or self._ownership_mask is None
            or len(self._centers) != len(self._graph.entities)
            or len(self._components) != len(self._graph.entities)
        ):
            raise ValueError("HEPA PCA baseline is not fitted")
        return (
            self._graph,
            self._feature_names,
            self._ownership_mask,
            self._centers,
            self._components,
        )


@dataclass(frozen=True)
class HepaEventDefinition:
    """Fitting-control definition of an action-blind state-change event."""

    entity_ids: Tuple[str, ...]
    feature_names: Tuple[str, ...]
    ownership_mask: NDArray[np.bool_]
    delta_center: NDArray[np.float64]
    delta_scale: NDArray[np.float64]
    threshold: float
    control_trajectory_count: int
    quantile: float = 0.95

    def __post_init__(self) -> None:
        shape = (len(self.entity_ids), len(self.feature_names))
        if (
            self.ownership_mask.shape != shape
            or self.delta_center.shape != shape
            or self.delta_scale.shape != shape
            or not np.any(self.ownership_mask)
            or np.any(self.delta_scale <= 0.0)
            or not np.all(np.isfinite(self.delta_center))
            or not np.all(np.isfinite(self.delta_scale))
            or not np.isfinite(self.threshold)
            or self.threshold <= 0.0
            or self.control_trajectory_count < 2
            or not 0.0 < self.quantile < 1.0
        ):
            raise ValueError("HEPA event definition is invalid")

    @classmethod
    def fit(
        cls, windows: ActionConditionedWindows
    ) -> "HepaEventDefinition":
        """Fit a robust state-change norm on fitting controls only."""

        ownership = _fit_owned_feature_mask(windows)
        controls = _control_trajectory_ids(windows)
        if len(controls) < 2:
            raise ValueError("HEPA event fitting needs control trajectories")
        trajectories = _trajectory_values(windows)
        deltas = []
        per_trajectory_deltas: Dict[str, NDArray[np.float64]] = {}
        for trajectory_id in sorted(controls):
            _, values = trajectories[trajectory_id]
            trajectory_deltas = np.diff(values, axis=0)
            per_trajectory_deltas[trajectory_id] = trajectory_deltas
            deltas.append(trajectory_deltas)
        combined = np.concatenate(deltas, axis=0)
        center = np.median(combined, axis=0)
        mad = 1.4826 * np.median(
            np.abs(combined - center[None]), axis=0
        )
        standard_deviation = np.std(combined, axis=0)
        scale = np.where(
            mad > 1e-8,
            mad,
            np.where(standard_deviation > 1e-8, standard_deviation, 1.0),
        )
        center = np.where(ownership, center, 0.0)
        scale = np.where(ownership, scale, 1.0)
        maxima = np.asarray(
            [
                float(
                    np.max(
                        _effect_norm(
                            values,
                            center=center,
                            scale=scale,
                            ownership=ownership,
                        )
                    )
                )
                for values in per_trajectory_deltas.values()
            ],
            dtype=np.float64,
        )
        threshold = float(
            np.quantile(maxima, 0.95, method="higher")
        )
        return cls(
            entity_ids=windows.entity_names,
            feature_names=windows.state_feature_names,
            ownership_mask=ownership.copy(),
            delta_center=np.asarray(center, dtype=np.float64),
            delta_scale=np.asarray(scale, dtype=np.float64),
            threshold=threshold,
            control_trajectory_count=len(controls),
        )

    def transition_scores(
        self, windows: ActionConditionedWindows
    ) -> NDArray[np.float64]:
        """Return action-blind scores for every available future transition."""

        self._validate_schema(windows)
        trajectories = _trajectory_values(windows)
        score_maps: Dict[str, Mapping[int, float]] = {}
        for trajectory_id, (point_indices, values) in trajectories.items():
            scores = _effect_norm(
                np.diff(values, axis=0),
                center=self.delta_center,
                scale=self.delta_scale,
                ownership=self.ownership_mask,
            )
            score_maps[trajectory_id] = {
                int(point_indices[position]): float(score)
                for position, score in enumerate(scores)
            }
        horizon = windows.future_states.shape[1]
        result = np.empty(
            (len(windows.histories), horizon), dtype=np.float64
        )
        for row, (trajectory_id, transition) in enumerate(
            zip(windows.trajectory_ids, windows.transition_indices)
        ):
            trajectory_score_map = score_maps[trajectory_id]
            for offset in range(horizon):
                result[row, offset] = trajectory_score_map[
                    int(transition) + offset
                ]
        return result

    def observed_effect_scores(
        self, windows: ActionConditionedWindows
    ) -> NDArray[np.float64]:
        """Score the most recent fully observed transition in each context."""

        self._validate_schema(windows)
        deltas = windows.histories[:, -1] - windows.histories[:, -2]
        return _effect_norm(
            deltas,
            center=self.delta_center,
            scale=self.delta_scale,
            ownership=self.ownership_mask,
        )

    def first_event_transitions(
        self, windows: ActionConditionedWindows
    ) -> Mapping[str, Optional[int]]:
        """Return each trajectory's first crossing transition, if any."""

        self._validate_schema(windows)
        trajectories = _trajectory_values(windows)
        result: Dict[str, Optional[int]] = {}
        for trajectory_id, (point_indices, values) in trajectories.items():
            scores = _effect_norm(
                np.diff(values, axis=0),
                center=self.delta_center,
                scale=self.delta_scale,
                ownership=self.ownership_mask,
            )
            crossings = np.flatnonzero(scores > self.threshold)
            result[trajectory_id] = (
                int(point_indices[int(crossings[0])])
                if len(crossings)
                else None
            )
        return result

    def labels(
        self, windows: ActionConditionedWindows
    ) -> NDArray[np.bool_]:
        """Return cumulative first-event labels at every available horizon."""

        events = self.first_event_transitions(windows)
        horizon = windows.future_states.shape[1]
        labels = np.zeros(
            (len(windows.histories), horizon), dtype=np.bool_
        )
        for row, (trajectory_id, transition) in enumerate(
            zip(windows.trajectory_ids, windows.transition_indices)
        ):
            event = events[trajectory_id]
            if event is None:
                continue
            offsets = np.arange(1, horizon + 1, dtype=np.int64)
            labels[row] = (
                (int(transition) < event)
                & (event <= int(transition) + offsets)
            )
        return labels

    def to_dict(self) -> Dict[str, Any]:
        """Return a restorable, auditable event definition."""

        return {
            "schema_version": 1,
            "kind": "hepa_normalized_effect_event_definition",
            "entity_ids": list(self.entity_ids),
            "feature_names": list(self.feature_names),
            "ownership_mask": self.ownership_mask.astype(int).tolist(),
            "delta_center": self.delta_center.tolist(),
            "delta_scale": self.delta_scale.tolist(),
            "threshold": self.threshold,
            "control_trajectory_count": self.control_trajectory_count,
            "quantile": self.quantile,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "HepaEventDefinition":
        """Restore an event definition with schema validation."""

        if (
            payload.get("schema_version") != 1
            or payload.get("kind")
            != "hepa_normalized_effect_event_definition"
        ):
            raise ValueError("unsupported HEPA event definition")
        return cls(
            entity_ids=tuple(str(value) for value in payload["entity_ids"]),
            feature_names=tuple(
                str(value) for value in payload["feature_names"]
            ),
            ownership_mask=np.asarray(
                payload["ownership_mask"], dtype=np.bool_
            ),
            delta_center=np.asarray(
                payload["delta_center"], dtype=np.float64
            ),
            delta_scale=np.asarray(
                payload["delta_scale"], dtype=np.float64
            ),
            threshold=float(payload["threshold"]),
            control_trajectory_count=int(
                payload["control_trajectory_count"]
            ),
            quantile=float(payload["quantile"]),
        )

    def _validate_schema(self, windows: ActionConditionedWindows) -> None:
        if (
            windows.entity_names != self.entity_ids
            or windows.state_feature_names != self.feature_names
        ):
            raise ValueError("event definition telemetry schema differs")


def survival_cdf(
    hazards: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compose finite per-interval hazards into a monotone event CDF."""

    values = np.asarray(hazards, dtype=np.float64)
    if (
        values.ndim != 2
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise ValueError("survival hazards are invalid")
    return np.asarray(
        1.0 - np.cumprod(1.0 - values, axis=1), dtype=np.float64
    )


def calibrate_probability_surface(
    probabilities: NDArray[np.float64],
    *,
    slope: float,
    intercept: float,
) -> NDArray[np.float64]:
    """Apply one increasing logit map to a complete probability surface."""

    values = np.asarray(probabilities, dtype=np.float64)
    if (
        values.ndim != 2
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 1.0)
        or not np.isfinite(slope)
        or slope <= 0.0
        or not np.isfinite(intercept)
    ):
        raise ValueError("probability calibration inputs are invalid")
    clipped = np.clip(values, 1e-7, 1.0 - 1e-7)
    logits = np.log(clipped) - np.log1p(-clipped)
    calibrated_logits = slope * logits + intercept
    return _stable_sigmoid(calibrated_logits)


class HepaJepaModel:
    """Restorable HEPA, horizon-deranged, or supervised-scratch model."""

    kind = "hepa_jepa_alert_adapter"
    schema_version = 1

    def __init__(self, config: HepaConfig = HepaConfig()) -> None:
        self.config = config
        self._graph: Optional[DeclaredTelemetryGraph] = None
        self._feature_names: Tuple[str, ...] = ()
        self._ownership_mask: Optional[NDArray[np.bool_]] = None
        self._network: Any = None
        self._training_metrics: Tuple[Mapping[str, float], ...] = ()
        self._selection_metrics: Tuple[Mapping[str, float], ...] = ()
        self._stage2_checkpoints: Tuple[
            Tuple[int, Mapping[str, Any]], ...
        ] = ()
        self._selected_step: Optional[int] = None
        self._derangement: Mapping[str, str] = {}
        self._calibration: Optional[Mapping[str, float]] = None

    @property
    def stage1_target_alignment(self) -> str:
        """Return the only treatment/null mechanism difference."""

        return (
            "whole_pair_deranged"
            if self.config.objective == "horizon_deranged"
            else "aligned"
        )

    @property
    def inference_parameter_count(self) -> int:
        """Return all parameters used by encode or CDF inference."""

        _, _, _, network = self._fitted_values()
        return int(sum(parameter.numel() for parameter in network.parameters()))

    @property
    def training_metrics(self) -> Tuple[Mapping[str, float], ...]:
        """Return immutable stage-one and stage-two training metrics."""

        self._fitted_values()
        return self._training_metrics

    @property
    def selection_metrics(self) -> Tuple[Mapping[str, float], ...]:
        """Return stage-two checkpoint scores from the selection role."""

        self._fitted_values()
        return self._selection_metrics

    @property
    def calibration(self) -> Optional[Mapping[str, float]]:
        """Return the fitted calibration map and trajectory threshold."""

        return None if self._calibration is None else dict(self._calibration)

    def fit(
        self,
        fit_windows: ActionConditionedWindows,
        event_definition: HepaEventDefinition,
    ) -> "HepaJepaModel":
        """Fit checkpoint candidates using fitting windows only."""

        torch = _require_torch()
        _seed_torch(torch, self.config.seed)
        if len(set(fit_windows.matched_pair_ids)) != (
            self.config.expected_pair_count
        ):
            raise ValueError("HEPA fitting pair count differs from contract")
        if (
            fit_windows.histories.shape[1] != 20
            or fit_windows.future_states.shape[1]
            != self.config.alert_horizon
        ):
            raise ValueError("HEPA windows differ from the frozen schema")
        event_definition._validate_schema(fit_windows)
        ownership = _fit_owned_feature_mask(fit_windows)
        network = _build_network(
            torch,
            config=self.config,
            entity_count=len(fit_windows.entity_names),
            feature_count=len(fit_windows.state_feature_names),
            ownership_mask=ownership,
        )
        generator = np.random.default_rng(self.config.seed + 1)
        metrics: List[Mapping[str, float]] = []
        deranged_indices = np.arange(
            len(fit_windows.histories), dtype=np.int64
        )
        if self.config.objective == "horizon_deranged":
            deranged_indices, self._derangement = (
                _whole_pair_derangement(
                    fit_windows, seed=self.config.seed + 2
                )
            )
        if self.config.objective != "supervised_scratch":
            optimizer = torch.optim.AdamW(
                network.stage1_parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
            sigreg_generator = torch.Generator(device="cpu").manual_seed(
                self.config.seed + 3
            )
            probabilities = 1.0 / np.arange(
                1, self.config.alert_horizon + 1, dtype=np.float64
            )
            probabilities /= np.sum(probabilities)
            network.train()
            for step in range(self.config.stage1_steps):
                indices = generator.integers(
                    0,
                    len(fit_windows.histories),
                    size=self.config.batch_size,
                )
                horizons = generator.choice(
                    np.arange(1, self.config.alert_horizon + 1),
                    size=self.config.batch_size,
                    p=probabilities,
                ).astype(np.int64)
                target_indices = deranged_indices[indices]
                optimizer.zero_grad(set_to_none=True)
                predicted, target = network.stage1(
                    torch.as_tensor(
                        fit_windows.histories[indices],
                        dtype=torch.float32,
                    ),
                    torch.as_tensor(
                        fit_windows.future_states[target_indices],
                        dtype=torch.float32,
                    ),
                    torch.as_tensor(horizons, dtype=torch.long),
                )
                prediction_loss = torch.nn.functional.l1_loss(
                    predicted, target
                )
                sigreg = _sketched_isotropic_gaussian_regularization(
                    predicted,
                    generator=sigreg_generator,
                    sketch_dimension=self.config.sketch_dimension,
                    knot_count=self.config.knot_count,
                )
                loss = (
                    (1.0 - self.config.sigreg_alpha) * prediction_loss
                    + self.config.sigreg_alpha * sigreg
                )
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("HEPA stage one became non-finite")
                loss.backward()
                optimizer.step()
                metrics.append(
                    {
                        "stage": 1.0,
                        "step": float(step + 1),
                        "loss": float(loss.detach()),
                        "l1": float(prediction_loss.detach()),
                        "sigreg": float(sigreg.detach()),
                    }
                )
        fit_labels = event_definition.labels(fit_windows)
        positive_count = int(np.sum(fit_labels))
        negative_count = int(fit_labels.size - positive_count)
        if positive_count < 1 or negative_count < 1:
            raise ValueError("HEPA event fitting needs both label classes")
        positive_weight = negative_count / float(positive_count)
        frozen_encoder = self.config.objective != "supervised_scratch"
        for parameter in network.encoder.parameters():
            parameter.requires_grad_(not frozen_encoder)
        stage2_parameters = list(network.predictor.parameters()) + list(
            network.hazard.parameters()
        )
        if not frozen_encoder:
            stage2_parameters += list(network.encoder.parameters())
        optimizer = torch.optim.AdamW(
            stage2_parameters,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        checkpoints: List[Tuple[int, Mapping[str, Any]]] = []
        network.train()
        for step in range(self.config.stage2_steps):
            indices = generator.integers(
                0,
                len(fit_windows.histories),
                size=self.config.batch_size,
            )
            optimizer.zero_grad(set_to_none=True)
            cdf = network.event_cdf(
                torch.as_tensor(
                    fit_windows.histories[indices],
                    dtype=torch.float32,
                )
            )
            labels = torch.as_tensor(
                fit_labels[indices], dtype=torch.float32
            )
            weights = torch.where(
                labels > 0.5,
                torch.full_like(labels, positive_weight),
                torch.ones_like(labels),
            )
            loss = torch.mean(
                weights
                * torch.nn.functional.binary_cross_entropy(
                    cdf, labels, reduction="none"
                )
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("HEPA stage two became non-finite")
            loss.backward()
            optimizer.step()
            metrics.append(
                {
                    "stage": 2.0,
                    "step": float(step + 1),
                    "loss": float(loss.detach()),
                    "positive_weight": float(positive_weight),
                }
            )
            if (
                (step + 1) % self.config.checkpoint_interval == 0
                or step + 1 == self.config.stage2_steps
            ):
                checkpoints.append(
                    (step + 1, copy.deepcopy(network.state_dict()))
                )
        if not checkpoints:
            raise RuntimeError("HEPA stage two produced no checkpoints")
        for parameter in network.parameters():
            parameter.requires_grad_(False)
        self._graph = fit_windows.graph
        self._feature_names = fit_windows.state_feature_names
        self._ownership_mask = ownership.copy()
        self._network = network.eval()
        self._training_metrics = tuple(metrics)
        self._selection_metrics = ()
        self._stage2_checkpoints = tuple(checkpoints)
        self._selected_step = None
        return self

    def select(
        self,
        selection_windows: ActionConditionedWindows,
        event_definition: HepaEventDefinition,
    ) -> "HepaJepaModel":
        """Choose one immutable fitting checkpoint using selection only."""

        torch = _require_torch()
        if (
            self._graph is None
            or self._network is None
            or not self._stage2_checkpoints
            or selection_windows.graph.to_dict()
            != self._graph.to_dict()
            or selection_windows.state_feature_names
            != self._feature_names
            or selection_windows.future_states.shape[1]
            != self.config.alert_horizon
        ):
            raise ValueError("HEPA selection inputs are invalid")
        event_definition._validate_schema(selection_windows)
        labels = event_definition.labels(selection_windows)
        metrics: List[Mapping[str, float]] = []
        best_score = float("inf")
        best_step = -1
        best_state: Optional[Mapping[str, Any]] = None
        for step, state in self._stage2_checkpoints:
            self._network.load_state_dict(state)
            score = _network_brier(
                torch,
                self._network,
                selection_windows.histories,
                labels,
                batch_size=max(32, self.config.batch_size),
            )
            metrics.append({"step": float(step), "brier": score})
            if score < best_score - 1e-12:
                best_score = score
                best_step = step
                best_state = state
        if best_state is None or best_step < 1:
            raise RuntimeError("HEPA selection chose no checkpoint")
        self._network.load_state_dict(best_state)
        self._network.eval()
        self._selection_metrics = tuple(metrics)
        self._selected_step = best_step
        self._stage2_checkpoints = ()
        return self

    def fit_calibration(
        self,
        windows: ActionConditionedWindows,
        event_definition: HepaEventDefinition,
    ) -> "HepaJepaModel":
        """Fit monotone probability calibration and a control-max threshold."""

        probabilities = self.predict_event_cdf(
            windows.histories, windows.graph
        )
        labels = event_definition.labels(windows)
        slope, intercept, brier = fit_logit_calibrator(
            probabilities, labels
        )
        calibrated = calibrate_probability_surface(
            probabilities, slope=slope, intercept=intercept
        )
        threshold = trajectory_alert_threshold(
            calibrated,
            windows.trajectory_ids,
            _control_trajectory_ids(windows),
        )
        self._calibration = {
            "slope": slope,
            "intercept": intercept,
            "calibration_brier": brier,
            "alert_threshold": threshold,
        }
        return self

    def encode(
        self,
        histories: NDArray[np.float64],
        graph: DeclaredTelemetryGraph,
    ) -> HepaEncodedTelemetry:
        """Encode causal contexts into ordered entity tokens."""

        torch = _require_torch()
        fitted_graph, feature_names, ownership, network = (
            self._fitted_values()
        )
        values = self._validate_histories(
            histories, graph, fitted_graph, feature_names
        )
        batches = []
        with torch.no_grad():
            for start in range(0, len(values), 256):
                tokens = network.context_tokens(
                    torch.as_tensor(
                        values[start : start + 256],
                        dtype=torch.float32,
                    )
                )
                batches.append(tokens.detach().cpu().numpy())
        return HepaEncodedTelemetry(
            tokens=np.asarray(
                np.concatenate(batches, axis=0), dtype=np.float64
            ),
            entity_ids=fitted_graph.entity_ids,
            ownership_mask=ownership.copy(),
        )

    def predict_event_cdf(
        self,
        histories: NDArray[np.float64],
        graph: DeclaredTelemetryGraph,
    ) -> NDArray[np.float64]:
        """Return the finite, monotone event probability surface."""

        torch = _require_torch()
        fitted_graph, feature_names, _, network = self._fitted_values()
        values = self._validate_histories(
            histories, graph, fitted_graph, feature_names
        )
        batches = []
        with torch.no_grad():
            for start in range(0, len(values), 256):
                probabilities = network.event_cdf(
                    torch.as_tensor(
                        values[start : start + 256],
                        dtype=torch.float32,
                    )
                )
                batches.append(probabilities.detach().cpu().numpy())
        result = np.asarray(
            np.concatenate(batches, axis=0), dtype=np.float64
        )
        _validate_probability_surface(result)
        return result

    def calibrated_event_cdf(
        self,
        histories: NDArray[np.float64],
        graph: DeclaredTelemetryGraph,
    ) -> NDArray[np.float64]:
        """Return the calibrated probability surface."""

        if self._calibration is None:
            raise ValueError("HEPA calibration has not been fitted")
        return calibrate_probability_surface(
            self.predict_event_cdf(histories, graph),
            slope=float(self._calibration["slope"]),
            intercept=float(self._calibration["intercept"]),
        )

    def alert_decisions(
        self,
        histories: NDArray[np.float64],
        graph: DeclaredTelemetryGraph,
    ) -> NDArray[np.bool_]:
        """Return the frozen horizon-max alert decision per context."""

        if self._calibration is None:
            raise ValueError("HEPA calibration has not been fitted")
        probabilities = self.calibrated_event_cdf(histories, graph)
        return probabilities[:, -1] > float(
            self._calibration["alert_threshold"]
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a deterministic restorable model artifact."""

        graph, feature_names, ownership, network = self._fitted_values()
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "config": asdict(self.config),
            "graph": graph.to_dict(),
            "feature_names": list(feature_names),
            "ownership_mask": ownership.astype(int).tolist(),
            "stage1_target_alignment": self.stage1_target_alignment,
            "derangement": dict(self._derangement),
            "training_metrics": [
                dict(value) for value in self._training_metrics
            ],
            "selection_metrics": [
                dict(value) for value in self._selection_metrics
            ],
            "selected_step": self._selected_step,
            "calibration": (
                None
                if self._calibration is None
                else dict(self._calibration)
            ),
            "state_dict": _state_dict_to_payload(network.state_dict()),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HepaJepaModel":
        """Restore a fitted HEPA model and validate its mechanism identity."""

        if (
            payload.get("schema_version") != cls.schema_version
            or payload.get("kind") != cls.kind
        ):
            raise ValueError("unsupported HEPA model artifact")
        torch = _require_torch()
        config = HepaConfig(**dict(payload["config"]))
        graph = DeclaredTelemetryGraph.from_dict(payload["graph"])
        feature_names = tuple(
            str(value) for value in payload["feature_names"]
        )
        ownership = np.asarray(
            payload["ownership_mask"], dtype=np.bool_
        )
        network = _build_network(
            torch,
            config=config,
            entity_count=len(graph.entities),
            feature_count=len(feature_names),
            ownership_mask=ownership,
        )
        network.load_state_dict(
            _state_dict_from_payload(torch, payload["state_dict"]),
            strict=True,
        )
        model = cls(config)
        if (
            payload.get("stage1_target_alignment")
            != model.stage1_target_alignment
        ):
            raise ValueError("HEPA target alignment identity differs")
        model._graph = graph
        model._feature_names = feature_names
        model._ownership_mask = ownership
        model._network = network.eval()
        model._derangement = {
            str(key): str(value)
            for key, value in dict(payload["derangement"]).items()
        }
        model._training_metrics = tuple(
            {
                str(key): float(value)
                for key, value in dict(row).items()
            }
            for row in payload["training_metrics"]
        )
        model._selection_metrics = tuple(
            {
                str(key): float(value)
                for key, value in dict(row).items()
            }
            for row in payload["selection_metrics"]
        )
        raw_selected_step = payload.get("selected_step")
        model._selected_step = (
            int(raw_selected_step)
            if raw_selected_step is not None
            else None
        )
        if model._selected_step is None:
            raise ValueError("restored HEPA model has no selected checkpoint")
        raw_calibration = payload.get("calibration")
        model._calibration = (
            None
            if raw_calibration is None
            else {
                str(key): float(value)
                for key, value in dict(raw_calibration).items()
            }
        )
        for parameter in network.parameters():
            parameter.requires_grad_(False)
        return model

    def _fitted_values(
        self,
    ) -> Tuple[
        DeclaredTelemetryGraph,
        Tuple[str, ...],
        NDArray[np.bool_],
        Any,
    ]:
        if (
            self._graph is None
            or not self._feature_names
            or self._ownership_mask is None
            or self._network is None
            or self._selected_step is None
        ):
            raise ValueError("HEPA model is not fitted")
        return (
            self._graph,
            self._feature_names,
            self._ownership_mask,
            self._network,
        )

    @staticmethod
    def _validate_histories(
        histories: NDArray[np.float64],
        graph: DeclaredTelemetryGraph,
        fitted_graph: DeclaredTelemetryGraph,
        feature_names: Tuple[str, ...],
    ) -> NDArray[np.float64]:
        values = np.asarray(histories, dtype=np.float64)
        if (
            graph.to_dict() != fitted_graph.to_dict()
            or values.ndim != 4
            or values.shape[1:] != (
                20,
                len(fitted_graph.entities),
                len(feature_names),
            )
            or len(values) < 1
            or not np.all(np.isfinite(values))
        ):
            raise ValueError("HEPA encoding histories are invalid")
        return values


class EntityStateRidgeProbe:
    """Fitting-only entity-local probe for observable-state retention."""

    kind = "hepa_entity_state_ridge_probe"
    schema_version = 1

    def __init__(self, *, ridge: float = 1e-3) -> None:
        if not np.isfinite(ridge) or ridge <= 0.0:
            raise ValueError("state probe ridge must be positive")
        self.ridge = float(ridge)
        self._weights: Tuple[NDArray[np.float64], ...] = ()
        self._ownership_mask: Optional[NDArray[np.bool_]] = None
        self._target_scale: Optional[NDArray[np.float64]] = None
        self._target_varying_mask: Optional[NDArray[np.bool_]] = None

    @property
    def target_scale(self) -> NDArray[np.float64]:
        """Return fitting-only target scales."""

        _, scale, _ = self._fitted_values()
        return scale.copy()

    @property
    def target_varying_mask(self) -> NDArray[np.bool_]:
        """Return observed target dimensions that varied during fitting."""

        _, _, varying = self._fitted_values()
        return varying.copy()

    def fit(
        self,
        tokens: NDArray[np.float64],
        current_states: NDArray[np.float64],
        ownership_mask: NDArray[np.bool_],
    ) -> "EntityStateRidgeProbe":
        """Fit one affine ridge map per entity."""

        values = np.asarray(tokens, dtype=np.float64)
        targets = np.asarray(current_states, dtype=np.float64)
        ownership = np.asarray(ownership_mask, dtype=np.bool_)
        if (
            values.ndim != 3
            or targets.ndim != 3
            or values.shape[:2] != targets.shape[:2]
            or ownership.shape != targets.shape[1:]
            or not np.all(np.isfinite(values))
            or not np.all(np.isfinite(targets))
        ):
            raise ValueError("entity state probe inputs are invalid")
        target_scale = np.std(targets, axis=0)
        varying = ownership & (target_scale > 1e-8)
        target_scale = np.where(varying, target_scale, 1.0)
        weights = []
        for entity in range(values.shape[1]):
            design = np.concatenate(
                (
                    values[:, entity],
                    np.ones((len(values), 1), dtype=np.float64),
                ),
                axis=1,
            )
            gram = design.T @ design
            penalty = np.eye(gram.shape[0], dtype=np.float64)
            penalty[-1, -1] = 0.0
            right = design.T @ targets[:, entity]
            weights.append(
                np.linalg.solve(gram + self.ridge * penalty, right)
            )
        self._weights = tuple(weights)
        self._ownership_mask = ownership.copy()
        self._target_scale = target_scale
        self._target_varying_mask = varying
        return self

    def predict(
        self, tokens: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Predict every entity's current observed state."""

        ownership, _, _ = self._fitted_values()
        values = np.asarray(tokens, dtype=np.float64)
        if (
            values.ndim != 3
            or values.shape[1] != len(self._weights)
            or not np.all(np.isfinite(values))
        ):
            raise ValueError("entity state probe tokens are invalid")
        predictions = []
        for entity, weights in enumerate(self._weights):
            design = np.concatenate(
                (
                    values[:, entity],
                    np.ones((len(values), 1), dtype=np.float64),
                ),
                axis=1,
            )
            predictions.append(design @ weights)
        result = np.stack(predictions, axis=1)
        if result.shape[1:] != ownership.shape:
            raise RuntimeError("entity state probe output shape differs")
        return np.asarray(result, dtype=np.float64)

    def to_dict(self) -> Dict[str, Any]:
        """Return a restorable probe artifact."""

        ownership, scale, varying = self._fitted_values()
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "ridge": self.ridge,
            "weights": [value.tolist() for value in self._weights],
            "ownership_mask": ownership.astype(int).tolist(),
            "target_scale": scale.tolist(),
            "target_varying_mask": varying.astype(int).tolist(),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "EntityStateRidgeProbe":
        """Restore a fitted probe."""

        if (
            payload.get("schema_version") != cls.schema_version
            or payload.get("kind") != cls.kind
        ):
            raise ValueError("unsupported entity state probe")
        probe = cls(ridge=float(payload["ridge"]))
        probe._weights = tuple(
            np.asarray(value, dtype=np.float64)
            for value in payload["weights"]
        )
        probe._ownership_mask = np.asarray(
            payload["ownership_mask"], dtype=np.bool_
        )
        probe._target_scale = np.asarray(
            payload["target_scale"], dtype=np.float64
        )
        probe._target_varying_mask = np.asarray(
            payload["target_varying_mask"], dtype=np.bool_
        )
        probe._fitted_values()
        return probe

    def _fitted_values(
        self,
    ) -> Tuple[
        NDArray[np.bool_],
        NDArray[np.float64],
        NDArray[np.bool_],
    ]:
        if (
            not self._weights
            or self._ownership_mask is None
            or self._target_scale is None
            or self._target_varying_mask is None
            or len(self._weights) != len(self._ownership_mask)
            or self._target_scale.shape != self._ownership_mask.shape
            or self._target_varying_mask.shape
            != self._ownership_mask.shape
        ):
            raise ValueError("entity state probe is not fitted")
        return (
            self._ownership_mask,
            self._target_scale,
            self._target_varying_mask,
        )


def assess_hepa_tracer(
    *,
    probability_surfaces: Mapping[
        str, Mapping[str, NDArray[np.float64]]
    ],
    restored_probability_surfaces: Mapping[
        str, Mapping[str, NDArray[np.float64]]
    ],
    stored_calibrated_surfaces: Mapping[
        str, Mapping[str, NDArray[np.float64]]
    ],
    restored_calibrated_surfaces: Mapping[
        str, Mapping[str, NDArray[np.float64]]
    ],
    stored_alert_decisions: Mapping[
        str, Mapping[str, NDArray[np.bool_]]
    ],
    restored_alert_decisions: Mapping[
        str, Mapping[str, NDArray[np.bool_]]
    ],
    stored_model_calibrations: Mapping[str, Mapping[str, float]],
    restored_model_calibrations: Mapping[
        str, Mapping[str, float]
    ],
    labels: Mapping[str, NDArray[np.bool_]],
    trajectory_ids: Mapping[str, Tuple[str, ...]],
    transition_indices: Mapping[str, NDArray[np.int64]],
    trajectory_onsets: Mapping[str, Mapping[str, Optional[int]]],
    candidate_tokens: NDArray[np.float64],
    restored_candidate_tokens: NDArray[np.float64],
    state_truth: NDArray[np.float64],
    state_scale: NDArray[np.float64],
    state_varying_mask: NDArray[np.bool_],
    state_predictions: Mapping[str, NDArray[np.float64]],
    inference_parameter_counts: Mapping[str, int],
    protocol_checks: Mapping[str, bool],
    edge_metrics: Mapping[str, Mapping[str, float]],
    raw_effect_scores: Mapping[str, NDArray[np.float64]],
    event_threshold: float,
) -> Mapping[str, Any]:
    """Recompute the complete HEPA decision from stored numeric evidence."""

    for role in HEPA_ASSESSMENT_ROLE_NAMES:
        if (
            role not in probability_surfaces
            or role not in restored_probability_surfaces
            or role not in stored_calibrated_surfaces
            or role not in restored_calibrated_surfaces
            or role not in labels
            or role not in trajectory_ids
            or role not in transition_indices
            or role not in trajectory_onsets
            or role not in raw_effect_scores
            or role not in stored_alert_decisions
            or role not in restored_alert_decisions
        ):
            raise ValueError(f"HEPA assessment role is missing: {role}")
        count = len(trajectory_ids[role])
        if (
            labels[role].shape[0] != count
            or transition_indices[role].shape != (count,)
            or raw_effect_scores[role].shape != (count,)
        ):
            raise ValueError("HEPA assessment role arrays do not align")
        for model in HEPA_MODEL_NAMES:
            shape = labels[role].shape
            if any(
                values[role][model].shape != shape
                for values in (
                    probability_surfaces,
                    restored_probability_surfaces,
                    stored_calibrated_surfaces,
                    restored_calibrated_surfaces,
                )
            ):
                raise ValueError(
                    "HEPA probability surface shape differs"
                )
            if (
                stored_alert_decisions[role][model].shape
                != (count,)
                or restored_alert_decisions[role][model].shape
                != (count,)
            ):
                raise ValueError("HEPA alert decision shape differs")
    calibration_ids = trajectory_ids["calibration"]
    calibration_controls = tuple(
        trajectory_id
        for trajectory_id, onset in trajectory_onsets[
            "calibration"
        ].items()
        if onset is None
    )
    calibrations: Dict[str, Mapping[str, float]] = {}
    calibrated: Dict[str, Dict[str, NDArray[np.float64]]] = {
        role: {} for role in HEPA_ASSESSMENT_ROLE_NAMES
    }
    restored_calibrated: Dict[
        str, Dict[str, NDArray[np.float64]]
    ] = {role: {} for role in HEPA_ASSESSMENT_ROLE_NAMES}
    surface_metrics: Dict[str, Dict[str, Mapping[str, float]]] = {
        role: {} for role in HEPA_ASSESSMENT_ROLE_NAMES
    }
    alert_metrics: Dict[str, Dict[str, Mapping[str, Any]]] = {
        role: {}
        for role in ("evaluation_iid", "evaluation_transfer")
    }
    restoration_checks: List[bool] = []
    for model in HEPA_MODEL_NAMES:
        if (
            model not in stored_model_calibrations
            or model not in restored_model_calibrations
        ):
            raise ValueError("HEPA stored calibration is missing")
        slope, intercept, calibration_brier = fit_logit_calibrator(
            probability_surfaces["calibration"][model],
            labels["calibration"],
        )
        for role in HEPA_ASSESSMENT_ROLE_NAMES:
            calibrated[role][model] = calibrate_probability_surface(
                probability_surfaces[role][model],
                slope=slope,
                intercept=intercept,
            )
            restored_calibrated[role][model] = (
                calibrate_probability_surface(
                    restored_probability_surfaces[role][model],
                    slope=slope,
                    intercept=intercept,
                )
            )
            surface_metrics[role][model] = _surface_metrics(
                calibrated[role][model], labels[role]
            )
            restoration_checks.extend(
                (
                    np.allclose(
                        probability_surfaces[role][model],
                        restored_probability_surfaces[role][model],
                        atol=1e-6,
                        rtol=0.0,
                    ),
                    np.allclose(
                        calibrated[role][model],
                        stored_calibrated_surfaces[role][model],
                        atol=1e-6,
                        rtol=0.0,
                    ),
                    np.allclose(
                        restored_calibrated[role][model],
                        restored_calibrated_surfaces[role][model],
                        atol=1e-6,
                        rtol=0.0,
                    ),
                )
            )
        threshold = trajectory_alert_threshold(
            calibrated["calibration"][model],
            calibration_ids,
            calibration_controls,
        )
        restored_threshold = trajectory_alert_threshold(
            restored_calibrated["calibration"][model],
            calibration_ids,
            calibration_controls,
        )
        restoration_checks.append(
            abs(threshold - restored_threshold) <= 1e-6
        )
        expected_calibration = {
            "slope": slope,
            "intercept": intercept,
            "calibration_brier": calibration_brier,
            "alert_threshold": threshold,
        }
        for stored in (
            stored_model_calibrations[model],
            restored_model_calibrations[model],
        ):
            restoration_checks.append(
                set(stored) == set(expected_calibration)
                and all(
                    abs(
                        float(stored[key])
                        - float(expected_calibration[key])
                    )
                    <= 1e-6
                    for key in expected_calibration
                )
            )
        calibrations[model] = {
            "slope": slope,
            "intercept": intercept,
            "calibration_brier": calibration_brier,
            "alert_threshold": threshold,
        }
        for role in HEPA_ASSESSMENT_ROLE_NAMES:
            original_decisions = (
                calibrated[role][model][:, -1]
                > float(
                    stored_model_calibrations[model][
                        "alert_threshold"
                    ]
                )
            )
            restored_decisions = (
                restored_calibrated[role][model][:, -1]
                > float(
                    restored_model_calibrations[model][
                        "alert_threshold"
                    ]
                )
            )
            restoration_checks.extend(
                (
                    np.array_equal(
                        original_decisions,
                        stored_alert_decisions[role][model],
                    ),
                    np.array_equal(
                        restored_decisions,
                        restored_alert_decisions[role][model],
                    ),
                    np.array_equal(
                        original_decisions, restored_decisions
                    ),
                )
            )
            if role in alert_metrics:
                alert_metrics[role][model] = (
                    _trajectory_alert_metrics(
                        decisions=original_decisions,
                        trajectory_ids=trajectory_ids[role],
                        transition_indices=transition_indices[role],
                        onsets=trajectory_onsets[role],
                    )
                )
    for role in ("evaluation_iid", "evaluation_transfer"):
        alert_metrics[role]["raw_effect_reference"] = (
            _trajectory_alert_metrics(
                decisions=raw_effect_scores[role] > event_threshold,
                trajectory_ids=trajectory_ids[role],
                transition_indices=transition_indices[role],
                onsets=trajectory_onsets[role],
            )
        )
    all_surfaces = [
        values
        for roles in (
            probability_surfaces,
            restored_probability_surfaces,
            stored_calibrated_surfaces,
            restored_calibrated_surfaces,
        )
        for models in roles.values()
        for values in models.values()
    ]
    cdf_valid = all(_probability_surface_is_valid(value) for value in all_surfaces)
    state = _state_retention_metrics(
        truth=state_truth,
        scale=state_scale,
        varying=state_varying_mask,
        predictions=state_predictions,
    )
    candidate_alert = alert_metrics["evaluation_transfer"]["hepa"]
    null_alert = alert_metrics["evaluation_transfer"][
        "horizon_deranged"
    ]
    candidate_brier = float(
        surface_metrics["evaluation_transfer"]["hepa"]["brier"]
    )
    null_brier = float(
        surface_metrics["evaluation_transfer"][
            "horizon_deranged"
        ]["brier"]
    )
    candidate_edge = dict(edge_metrics["hepa"])
    latency = float(candidate_edge["batch_one_cpu_latency_ms"])
    peak_rss = float(candidate_edge["peak_rss_bytes"])
    serialized_bytes = float(
        candidate_edge["serialized_candidate_sidecars_bytes"]
    )
    delay = candidate_alert["median_post_onset_delay_transitions"]
    gates = {
        "finite_bounded_monotone_restored_cdf": cdf_valid,
        "matched_capacity_only_target_alignment_differs": (
            int(inference_parameter_counts["hepa"])
            == int(inference_parameter_counts["horizon_deranged"])
            and bool(protocol_checks.get("only_target_alignment_differs"))
            and bool(protocol_checks.get("pair_atomic_derangement"))
        ),
        "state_retention_within_1_05_pca_all_entities_reported": (
            float(state["hepa"]["aggregate_nrmse"])
            <= 1.05 * float(state["matched_pca"]["aggregate_nrmse"])
            and bool(state["all_varying_entities_reported"])
        ),
        "calibrated_brier_within_1_05_deranged": (
            candidate_brier <= 1.05 * null_brier
        ),
        "control_trajectory_false_alarms_at_most_5_percent": (
            float(candidate_alert["control_trajectory_false_alarm_rate"])
            <= 0.05
        ),
        "treatment_post_onset_detection_at_least_80_percent": (
            float(candidate_alert["treatment_detection_rate"]) >= 0.80
        ),
        "median_post_onset_delay_at_most_10": (
            delay is not None and float(delay) <= 10.0
        ),
        "detection_improves_10_points_over_deranged": (
            float(null_alert["control_trajectory_false_alarm_rate"])
            <= 0.05
            and float(candidate_alert["treatment_detection_rate"])
            - float(null_alert["treatment_detection_rate"])
            >= 0.10 - 1e-12
        ),
        "edge_budget_and_diagnostics": (
            serialized_bytes <= 16.0 * 1024.0 * 1024.0
            and np.isfinite(latency)
            and latency >= 0.0
            and np.isfinite(peak_rss)
            and peak_rss > 0.0
        ),
        "restoration_reproduces_all_public_outputs": (
            bool(all(restoration_checks))
            and np.allclose(
                candidate_tokens,
                restored_candidate_tokens,
                atol=1e-6,
                rtol=0.0,
            )
        ),
    }
    protocol_passed = bool(all(bool(value) for value in protocol_checks.values()))
    passed = bool(all(gates.values()) and protocol_passed)
    return {
        "schema_version": 1,
        "kind": "hepa_jepa_tracer_assessment",
        "calibrations": calibrations,
        "surface_metrics": surface_metrics,
        "alert_metrics": alert_metrics,
        "state_retention": state,
        "inference_parameter_counts": {
            name: int(value)
            for name, value in inference_parameter_counts.items()
        },
        "edge_metrics": {
            name: {
                key: float(value)
                for key, value in metrics.items()
            }
            for name, metrics in edge_metrics.items()
        },
        "protocol_checks": {
            name: bool(value) for name, value in protocol_checks.items()
        },
        "gates": gates,
        "passed": passed,
        "decision": (
            "advance_hepa_to_fixed_seed_robustness"
            if passed
            else "reject_hepa_telemetry_recipe"
        ),
    }


def fit_logit_calibrator(
    probabilities: NDArray[np.float64],
    labels: NDArray[np.bool_],
) -> Tuple[float, float, float]:
    """Select a frozen increasing logit map by calibration Brier score."""

    values = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.bool_)
    if values.shape != truth.shape:
        raise ValueError("calibration probabilities and labels do not align")
    candidates = []
    for slope in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0):
        for intercept in np.linspace(-4.0, 4.0, 33):
            calibrated = calibrate_probability_surface(
                values, slope=slope, intercept=float(intercept)
            )
            brier = float(
                np.mean(np.square(calibrated - truth.astype(np.float64)))
            )
            candidates.append((brier, slope, float(intercept)))
    brier, slope, intercept = min(
        candidates,
        key=lambda value: (
            value[0],
            abs(value[1] - 1.0),
            abs(value[2]),
            value[1],
            value[2],
        ),
    )
    return slope, intercept, brier


def trajectory_alert_threshold(
    calibrated: NDArray[np.float64],
    trajectory_ids: Sequence[str],
    control_trajectory_ids: Sequence[str],
    *,
    quantile: float = 0.95,
) -> float:
    """Fit a conservative control-trajectory maximum threshold."""

    probabilities = np.asarray(calibrated, dtype=np.float64)
    if (
        probabilities.ndim != 2
        or len(probabilities) != len(trajectory_ids)
        or not 0.0 < quantile < 1.0
    ):
        raise ValueError("trajectory alert calibration inputs are invalid")
    controls = set(control_trajectory_ids)
    maxima = []
    ids = np.asarray(tuple(trajectory_ids), dtype=str)
    for trajectory_id in sorted(controls):
        positions = np.flatnonzero(ids == trajectory_id)
        if len(positions):
            maxima.append(float(np.max(probabilities[positions, -1])))
    if len(maxima) < 2:
        raise ValueError("alert calibration needs control trajectories")
    return float(
        np.quantile(
            np.asarray(maxima, dtype=np.float64),
            quantile,
            method="higher",
        )
    )


def trajectory_action_onsets(
    windows: ActionConditionedWindows,
) -> Mapping[str, Optional[int]]:
    """Return action onset only for offline post-onset assessment."""

    try:
        applicable = windows.action_feature_names.index("applicable")
    except ValueError as error:
        raise ValueError(
            "HEPA assessment needs the applicable action field"
        ) from error
    result: Dict[str, Optional[int]] = {
        trajectory_id: None
        for trajectory_id in set(windows.trajectory_ids)
    }
    for index, trajectory_id in enumerate(windows.trajectory_ids):
        if np.any(
            windows.future_actions[index, 0, :, applicable] > 0.5
        ):
            transition = int(windows.transition_indices[index])
            current = result[trajectory_id]
            result[trajectory_id] = (
                transition
                if current is None
                else min(current, transition)
            )
    return {key: result[key] for key in sorted(result)}


def _surface_metrics(
    probabilities: NDArray[np.float64],
    labels: NDArray[np.bool_],
) -> Mapping[str, float]:
    values = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.bool_)
    if values.shape != truth.shape or not _probability_surface_is_valid(
        values
    ):
        raise ValueError("HEPA surface metric inputs are invalid")
    flat_values = values.reshape(-1)
    flat_truth = truth.astype(np.float64).reshape(-1)
    brier = float(np.mean(np.square(flat_values - flat_truth)))
    bins = np.minimum((flat_values * 10.0).astype(np.int64), 9)
    ece = 0.0
    for index in range(10):
        selected = bins == index
        if np.any(selected):
            ece += float(np.mean(selected)) * abs(
                float(np.mean(flat_values[selected]))
                - float(np.mean(flat_truth[selected]))
            )
    return {
        "brier": brier,
        "ece_10_equal_width_bins": ece,
        "positive_rate": float(np.mean(flat_truth)),
    }


def _trajectory_alert_metrics(
    *,
    decisions: NDArray[np.bool_],
    trajectory_ids: Tuple[str, ...],
    transition_indices: NDArray[np.int64],
    onsets: Mapping[str, Optional[int]],
) -> Mapping[str, Any]:
    alerts = np.asarray(decisions, dtype=np.bool_)
    transitions = np.asarray(transition_indices, dtype=np.int64)
    if (
        alerts.shape != (len(trajectory_ids),)
        or transitions.shape != alerts.shape
    ):
        raise ValueError("HEPA trajectory alert arrays do not align")
    ids = np.asarray(trajectory_ids, dtype=str)
    rows: List[Mapping[str, Any]] = []
    for trajectory_id in sorted(set(trajectory_ids)):
        positions = np.flatnonzero(ids == trajectory_id)
        order = positions[np.argsort(transitions[positions])]
        alarm_transitions = transitions[order][alerts[order]]
        onset = onsets[trajectory_id]
        eligible = (
            alarm_transitions[alarm_transitions >= onset]
            if onset is not None
            else np.asarray([], dtype=np.int64)
        )
        first_post = int(np.min(eligible)) if len(eligible) else None
        rows.append(
            {
                "trajectory_id": trajectory_id,
                "is_treatment": onset is not None,
                "onset_transition": onset,
                "any_alert": bool(len(alarm_transitions)),
                "pre_onset_alert": bool(
                    onset is not None
                    and np.any(alarm_transitions < onset)
                ),
                "first_post_onset_alert_transition": first_post,
                "post_onset_delay_transitions": (
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
        if row["first_post_onset_alert_transition"] is not None
    ]
    delays: List[int] = []
    for row in detected:
        value = row["post_onset_delay_transitions"]
        if value is None:
            raise RuntimeError("detected HEPA alert has no delay")
        delays.append(int(value))
    if not controls or not treatments:
        raise ValueError(
            "HEPA alert assessment needs controls and treatments"
        )
    return {
        "control_trajectory_count": len(controls),
        "treatment_trajectory_count": len(treatments),
        "control_trajectory_false_alarm_rate": float(
            np.mean([bool(row["any_alert"]) for row in controls])
        ),
        "treatment_detection_rate": float(
            len(detected) / len(treatments)
        ),
        "treatment_pre_onset_alert_rate": float(
            np.mean(
                [bool(row["pre_onset_alert"]) for row in treatments]
            )
        ),
        "median_post_onset_delay_transitions": (
            float(np.median(delays)) if delays else None
        ),
        "worst_post_onset_delay_transitions": (
            int(max(delays)) if delays else None
        ),
        "alerts_per_logical_run": float(
            np.mean([bool(row["any_alert"]) for row in rows])
        ),
        "trajectory_rows": rows,
    }


def _state_retention_metrics(
    *,
    truth: NDArray[np.float64],
    scale: NDArray[np.float64],
    varying: NDArray[np.bool_],
    predictions: Mapping[str, NDArray[np.float64]],
) -> Mapping[str, Any]:
    target = np.asarray(truth, dtype=np.float64)
    target_scale = np.asarray(scale, dtype=np.float64)
    varying_mask = np.asarray(varying, dtype=np.bool_)
    if (
        target.ndim != 3
        or target_scale.shape != target.shape[1:]
        or varying_mask.shape != target_scale.shape
        or not np.any(varying_mask)
    ):
        raise ValueError("HEPA state retention inputs are invalid")
    metrics: Dict[str, Any] = {}
    reported_entities = set()
    for name in ("hepa", "matched_pca"):
        prediction = np.asarray(predictions[name], dtype=np.float64)
        if prediction.shape != target.shape:
            raise ValueError("HEPA state prediction shape differs")
        normalized = (prediction - target) / target_scale[None]
        per_entity: Dict[str, Optional[float]] = {}
        for entity in range(target.shape[1]):
            selected = varying_mask[entity]
            if np.any(selected):
                per_entity[str(entity)] = float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                normalized[:, entity, selected]
                            )
                        )
                    )
                )
                reported_entities.add(entity)
            else:
                per_entity[str(entity)] = None
        metrics[name] = {
            "aggregate_nrmse": float(
                np.sqrt(np.mean(np.square(normalized[:, varying_mask])))
            ),
            "per_entity_nrmse": per_entity,
        }
    expected_entities = set(
        int(value)
        for value in np.flatnonzero(np.any(varying_mask, axis=1))
    )
    metrics["all_varying_entities_reported"] = (
        reported_entities == expected_entities
    )
    metrics["varying_entity_count"] = len(expected_entities)
    return metrics


def _probability_surface_is_valid(
    values: NDArray[np.float64],
) -> bool:
    probabilities = np.asarray(values, dtype=np.float64)
    return bool(
        probabilities.ndim == 2
        and np.all(np.isfinite(probabilities))
        and np.all(probabilities >= -1e-7)
        and np.all(probabilities <= 1.0 + 1e-7)
        and np.all(np.diff(probabilities, axis=1) >= -1e-7)
    )


def _build_network(
    torch: Any,
    *,
    config: HepaConfig,
    entity_count: int,
    feature_count: int,
    ownership_mask: NDArray[np.bool_],
) -> Any:
    nn = torch.nn

    class SharedTelemetryEncoder(nn.Module):  # type: ignore[misc,name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.input_projection = nn.Linear(feature_count, config.width)
            self.entity_embedding = nn.Embedding(
                entity_count, config.width
            )
            layer = nn.TransformerEncoderLayer(
                d_model=config.width,
                nhead=config.head_count,
                dim_feedforward=config.feedforward_width,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                layer, num_layers=config.block_count
            )
            self.output_norm = nn.LayerNorm(config.width)
            self.pool_score = nn.Linear(config.width, 1)
            self.register_buffer(
                "ownership",
                torch.as_tensor(ownership_mask, dtype=torch.bool),
            )

        def forward(
            self,
            values: Any,
            *,
            causal: bool,
            valid_times: Optional[Any] = None,
        ) -> Any:
            batch, time, entities, _ = values.shape
            owned = torch.where(
                self.ownership[None, None],
                values,
                torch.zeros_like(values),
            )
            entity_ids = torch.arange(
                entities, device=values.device, dtype=torch.long
            )
            position = _sinusoidal_positions(
                torch, time, config.width, values.device, values.dtype
            )
            tokens = (
                self.input_projection(owned)
                + self.entity_embedding(entity_ids)[None, None]
                + position[None, :, None]
            ).reshape(batch, time * entities, config.width)
            key_padding = None
            if valid_times is not None:
                key_padding = (
                    ~valid_times[:, :, None]
                    .expand(batch, time, entities)
                    .reshape(batch, time * entities)
                )
            mask = None
            if causal:
                token_times = torch.arange(
                    time, device=values.device
                ).repeat_interleave(entities)
                mask = token_times[None, :] > token_times[:, None]
            encoded = self.output_norm(
                self.transformer(
                    tokens,
                    mask=mask,
                    src_key_padding_mask=key_padding,
                )
            )
            return encoded.reshape(
                batch, time, entities, config.width
            )

        def pool(self, tokens: Any, valid_times: Optional[Any] = None) -> Any:
            batch, time, entities, width = tokens.shape
            flattened = tokens.reshape(batch, time * entities, width)
            logits = self.pool_score(flattened).squeeze(-1)
            if valid_times is not None:
                valid = (
                    valid_times[:, :, None]
                    .expand(batch, time, entities)
                    .reshape(batch, time * entities)
                )
                logits = logits.masked_fill(~valid, float("-inf"))
            weights = torch.softmax(logits, dim=1)
            return torch.sum(flattened * weights[..., None], dim=1)

    class Network(nn.Module):  # type: ignore[misc,name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.encoder = SharedTelemetryEncoder()
            self.predictor = nn.Sequential(
                nn.Linear(config.width + 1, config.width),
                nn.GELU(),
                nn.Linear(config.width, config.width),
            )
            self.hazard = nn.Linear(config.width, 1)

        def stage1_parameters(self) -> Any:
            return list(self.encoder.parameters()) + list(
                self.predictor.parameters()
            )

        def context_tokens(self, histories: Any) -> Any:
            return self.encoder(histories, causal=True)[:, -1]

        def _context_summary(self, histories: Any) -> Any:
            tokens = self.context_tokens(histories)
            return self.encoder.pool(tokens[:, None])

        def _predict(self, summary: Any, horizons: Any) -> Any:
            normalized = (
                torch.log(horizons.to(summary.dtype))
                / math.log(float(config.alert_horizon + 1))
            )
            return self.predictor(
                torch.cat((summary, normalized[:, None]), dim=1)
            )

        def stage1(
            self, histories: Any, future_states: Any, horizons: Any
        ) -> Tuple[Any, Any]:
            summary = self._context_summary(histories)
            predicted = self._predict(summary, horizons)
            times = torch.arange(
                config.alert_horizon, device=future_states.device
            )[None]
            valid = times < horizons[:, None]
            target_tokens = self.encoder(
                future_states, causal=False, valid_times=valid
            )
            target = self.encoder.pool(target_tokens, valid)
            return predicted, target

        def event_cdf(self, histories: Any) -> Any:
            summary = self._context_summary(histories)
            batch = len(histories)
            horizons = torch.arange(
                1,
                config.alert_horizon + 1,
                device=histories.device,
                dtype=torch.long,
            )
            repeated_summary = (
                summary[:, None]
                .expand(batch, config.alert_horizon, config.width)
                .reshape(batch * config.alert_horizon, config.width)
            )
            repeated_horizons = horizons[None].expand(
                batch, config.alert_horizon
            ).reshape(-1)
            predictions = self._predict(
                repeated_summary, repeated_horizons
            ).reshape(batch, config.alert_horizon, config.width)
            hazards = torch.sigmoid(self.hazard(predictions).squeeze(-1))
            return 1.0 - torch.cumprod(1.0 - hazards, dim=1)

    return Network()


def _whole_pair_derangement(
    windows: ActionConditionedWindows, *, seed: int
) -> Tuple[NDArray[np.int64], Mapping[str, str]]:
    pair_ids = tuple(sorted(set(windows.matched_pair_ids)))
    if len(pair_ids) < 2:
        raise ValueError("whole-pair derangement needs at least two pairs")
    generator = np.random.default_rng(seed)
    shifts = np.arange(1, len(pair_ids), dtype=np.int64)
    shift = int(generator.choice(shifts))
    mapping = {
        pair_id: pair_ids[(position + shift) % len(pair_ids)]
        for position, pair_id in enumerate(pair_ids)
    }
    pair_array = np.asarray(windows.matched_pair_ids, dtype=str)
    trajectories_by_pair = {
        pair_id: tuple(
            sorted(
                {
                    windows.trajectory_ids[int(row)]
                    for row in np.flatnonzero(pair_array == pair_id)
                }
            )
        )
        for pair_id in pair_ids
    }
    if any(len(values) != 2 for values in trajectories_by_pair.values()):
        raise ValueError("HEPA derangement requires two arms per pair")
    lookup = {
        (
            windows.trajectory_ids[index],
            int(windows.transition_indices[index]),
        ): index
        for index in range(len(windows.histories))
    }
    result = np.empty(len(windows.histories), dtype=np.int64)
    for index, (pair_id, trajectory_id, transition) in enumerate(
        zip(
            windows.matched_pair_ids,
            windows.trajectory_ids,
            windows.transition_indices,
        )
    ):
        arm = trajectories_by_pair[pair_id].index(trajectory_id)
        target_pair = mapping[pair_id]
        target_trajectory = trajectories_by_pair[target_pair][arm]
        try:
            result[index] = lookup[
                (target_trajectory, int(transition))
            ]
        except KeyError as error:
            raise ValueError(
                "HEPA derangement transitions do not align"
            ) from error
    return result, mapping


def _trajectory_values(
    windows: ActionConditionedWindows,
) -> Mapping[str, Tuple[NDArray[np.int64], NDArray[np.float64]]]:
    rows: Dict[str, List[int]] = {}
    for index, trajectory_id in enumerate(windows.trajectory_ids):
        rows.setdefault(trajectory_id, []).append(index)
    result = {}
    for trajectory_id, positions in rows.items():
        point_values: Dict[int, NDArray[np.float64]] = {}
        for row in positions:
            transition = int(windows.transition_indices[row])
            history_start = transition - windows.histories.shape[1] + 1
            for offset, value in enumerate(windows.histories[row]):
                _merge_point(
                    point_values, history_start + offset, value
                )
            for offset, value in enumerate(windows.future_states[row], 1):
                _merge_point(point_values, transition + offset, value)
        point_indices = np.asarray(sorted(point_values), dtype=np.int64)
        if (
            len(point_indices) < 2
            or np.any(np.diff(point_indices) != 1)
        ):
            raise ValueError("HEPA trajectory reconstruction has gaps")
        values = np.stack(
            [point_values[int(index)] for index in point_indices], axis=0
        )
        result[trajectory_id] = (point_indices, values)
    return result


def _merge_point(
    values: Dict[int, NDArray[np.float64]],
    index: int,
    value: NDArray[np.float64],
) -> None:
    existing = values.get(index)
    current = np.asarray(value, dtype=np.float64)
    if existing is not None and not np.allclose(
        existing, current, atol=1e-6
    ):
        raise ValueError("overlapping HEPA trajectory values differ")
    values[index] = current


def _effect_norm(
    deltas: NDArray[np.float64],
    *,
    center: NDArray[np.float64],
    scale: NDArray[np.float64],
    ownership: NDArray[np.bool_],
) -> NDArray[np.float64]:
    standardized = (deltas - center[None]) / scale[None]
    selected = standardized[:, ownership]
    return np.asarray(
        np.sqrt(np.mean(np.square(selected), axis=1)),
        dtype=np.float64,
    )


def _fit_owned_feature_mask(
    windows: ActionConditionedWindows,
) -> NDArray[np.bool_]:
    entity_positions = {
        entity_id: position
        for position, entity_id in enumerate(windows.entity_names)
    }
    feature_positions = {
        name: position
        for position, name in enumerate(windows.state_feature_names)
    }
    mask = np.zeros(
        (len(windows.entity_names), len(windows.state_feature_names)),
        dtype=np.bool_,
    )
    for feature_key, entity_id in windows.graph.binding_map().items():
        feature_name = feature_key.split(".", 1)[-1]
        if (
            entity_id in entity_positions
            and feature_name in feature_positions
        ):
            mask[
                entity_positions[entity_id],
                feature_positions[feature_name],
            ] = True
    mask |= np.ptp(windows.histories, axis=(0, 1)) > 1e-9
    if not np.any(mask):
        raise ValueError("HEPA telemetry schema has no observations")
    return mask


def _orient_components(components: NDArray[np.float64]) -> None:
    for row in components:
        if len(row):
            pivot = int(np.argmax(np.abs(row)))
            if row[pivot] < 0.0:
                row *= -1.0


def _control_trajectory_ids(
    windows: ActionConditionedWindows,
) -> Tuple[str, ...]:
    try:
        applicable = windows.action_feature_names.index("applicable")
    except ValueError as error:
        raise ValueError(
            "HEPA event definition needs the applicable action field"
        ) from error
    treatments = {
        windows.trajectory_ids[index]
        for index in range(len(windows.histories))
        if np.any(windows.future_actions[index, ..., applicable] > 0.5)
    }
    return tuple(
        sorted(set(windows.trajectory_ids) - treatments)
    )


def _network_brier(
    torch: Any,
    network: Any,
    histories: NDArray[np.float64],
    labels: NDArray[np.bool_],
    *,
    batch_size: int,
) -> float:
    was_training = bool(network.training)
    network.eval()
    squared = []
    with torch.no_grad():
        for start in range(0, len(histories), batch_size):
            probabilities = network.event_cdf(
                torch.as_tensor(
                    histories[start : start + batch_size],
                    dtype=torch.float32,
                )
            )
            truth = torch.as_tensor(
                labels[start : start + batch_size],
                dtype=torch.float32,
            )
            squared.append(
                torch.square(probabilities - truth)
                .detach()
                .cpu()
                .numpy()
            )
    if was_training:
        network.train()
    return float(np.mean(np.concatenate(squared, axis=0)))


def _sketched_isotropic_gaussian_regularization(
    embeddings: Any,
    *,
    generator: Any,
    sketch_dimension: int,
    knot_count: int,
) -> Any:
    """Return the pinned positive-half Epps-Pulley SIGReg statistic."""

    if (
        embeddings.ndim < 2
        or embeddings.size(-2) < 1
        or embeddings.size(-1) < 1
        or sketch_dimension < 1
        or knot_count < 2
    ):
        raise ValueError("HEPA SIGReg inputs are invalid")
    torch = _require_torch()
    directions = torch.randn(
        embeddings.size(-1),
        sketch_dimension,
        device="cpu",
        dtype=embeddings.dtype,
        generator=generator,
    ).to(device=embeddings.device, dtype=embeddings.dtype)
    directions = directions / directions.norm(
        p=2, dim=0, keepdim=True
    )
    knots = torch.linspace(
        0.0,
        3.0,
        knot_count,
        device=embeddings.device,
        dtype=embeddings.dtype,
    )
    delta = 3.0 / float(knot_count - 1)
    quadrature = torch.full(
        (knot_count,),
        2.0 * delta,
        device=embeddings.device,
        dtype=embeddings.dtype,
    )
    quadrature[[0, -1]] = delta
    gaussian = torch.exp(-torch.square(knots) / 2.0)
    projected = (embeddings @ directions).unsqueeze(-1) * knots
    error = torch.square(
        projected.cos().mean(dim=-3) - gaussian
    ) + torch.square(projected.sin().mean(dim=-3))
    statistic = (
        error @ (quadrature * gaussian)
    ) * embeddings.size(-2)
    return statistic.mean()


def _sinusoidal_positions(
    torch: Any,
    length: int,
    width: int,
    device: Any,
    dtype: Any,
) -> Any:
    position = torch.arange(
        length, device=device, dtype=dtype
    )[:, None]
    even_width = (width + 1) // 2
    frequency = torch.exp(
        torch.arange(even_width, device=device, dtype=dtype)
        * (-math.log(10000.0) / max(1, even_width - 1))
    )
    angles = position * frequency[None]
    result = torch.zeros(
        (length, width), device=device, dtype=dtype
    )
    result[:, 0::2] = torch.sin(angles[:, : result[:, 0::2].shape[1]])
    result[:, 1::2] = torch.cos(angles[:, : result[:, 1::2].shape[1]])
    return result


def _stable_sigmoid(values: NDArray[np.float64]) -> NDArray[np.float64]:
    result = np.empty_like(values)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _validate_probability_surface(values: NDArray[np.float64]) -> None:
    if (
        values.ndim != 2
        or not np.all(np.isfinite(values))
        or np.any(values < -1e-7)
        or np.any(values > 1.0 + 1e-7)
        or np.any(np.diff(values, axis=1) < -1e-7)
    ):
        raise ValueError("HEPA probability surface is invalid")


def _state_dict_to_payload(
    state_dict: Mapping[str, Any]
) -> Mapping[str, Any]:
    return {
        name: {
            "shape": list(value.shape),
            "values": value.detach().cpu().numpy().tolist(),
        }
        for name, value in state_dict.items()
    }


def _state_dict_from_payload(
    torch: Any, payload: Mapping[str, Any]
) -> Mapping[str, Any]:
    result = {}
    for name, raw in payload.items():
        value = dict(raw)
        array = np.asarray(value["values"])
        shape = tuple(int(item) for item in value["shape"])
        if array.shape != shape:
            raise ValueError("HEPA state tensor shape differs")
        if array.dtype.kind in ("i", "u", "b"):
            tensor = torch.as_tensor(array)
        else:
            tensor = torch.as_tensor(array, dtype=torch.float32)
        result[str(name)] = tensor
    return result


def _seed_torch(torch: Any, seed: int) -> None:
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, int(torch.get_num_threads()))))


def _require_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError as error:
        raise RuntimeError(
            "HEPA fitting requires the optional training dependencies"
        ) from error
