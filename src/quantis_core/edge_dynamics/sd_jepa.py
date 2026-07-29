"""Edge-sized SD-JEPA progression/content representation.

Future state, control, and action tensors are fitting-only inputs. Public
encoding and alert inference accept current histories and the declared graph.
"""

import copy
import importlib
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ..action_conditioned_dynamics import ActionConditionedWindows
from ..graph_telemetry import DeclaredTelemetryGraph
from .complete_lejepa import PairBlockedAnchorSchedule
from .hepa_jepa import (
    HepaEventDefinition,
    calibrate_probability_surface,
    fit_logit_calibrator,
    trajectory_alert_threshold,
)


SD_JEPA_OBJECTIVES = ("sd_jepa", "lewm_unsplit", "a2_full")
SD_JEPA_SCORE_NAMES = (
    "sd_jepa_angle",
    "sd_jepa_z_mse",
    "lewm_unsplit_angle",
    "lewm_unsplit_z_mse",
    "a2_full_angle",
    "a2_full_z_mse",
)


@dataclass(frozen=True)
class SdJepaConfig:
    """Frozen controls for one edge-sized SD-JEPA cell."""

    objective: str = "sd_jepa"
    width: int = 32
    progression_width: int = 2
    hidden_width: int = 64
    history_size: int = 3
    pretrain_steps: int = 300
    checkpoint_interval: int = 50
    learning_rate: float = 5e-4
    weight_decay: float = 1e-3
    sigreg_weight: float = 0.09
    triplet_weight: float = 0.10
    triplet_margin: float = 0.2
    sigreg_sketch_dimension: int = 256
    sigreg_knot_count: int = 17
    expected_pair_count: int = 40
    seed: int = 15015
    device: str = "cpu"

    def __post_init__(self) -> None:
        integers = (
            self.width,
            self.progression_width,
            self.hidden_width,
            self.history_size,
            self.pretrain_steps,
            self.checkpoint_interval,
            self.sigreg_sketch_dimension,
            self.sigreg_knot_count,
            self.expected_pair_count,
        )
        if (
            self.objective not in SD_JEPA_OBJECTIVES
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                or value < 1
                for value in integers
            )
            or self.progression_width != 2
            or self.width <= self.progression_width
            or self.hidden_width < self.width
            or self.history_size != 3
            or self.learning_rate <= 0.0
            or self.weight_decay < 0.0
            or self.sigreg_weight != 0.09
            or self.triplet_weight != 0.10
            or self.triplet_margin != 0.2
            or self.sigreg_knot_count < 2
            or isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.device != "cpu"
        ):
            raise ValueError("SD-JEPA configuration is invalid")


@dataclass(frozen=True)
class SdEncodedTelemetry:
    """Entity-preserving and global temporal SD-JEPA states."""

    entity_tokens: NDArray[np.float64]
    scene_tokens: NDArray[np.float64]
    entity_ids: Tuple[str, ...]
    ownership_mask: NDArray[np.bool_]

    def __post_init__(self) -> None:
        if (
            self.entity_tokens.ndim != 4
            or self.scene_tokens.ndim != 3
            or self.entity_tokens.shape[:2] != self.scene_tokens.shape[:2]
            or self.entity_tokens.shape[2] != len(self.entity_ids)
            or self.entity_tokens.shape[-1] != self.scene_tokens.shape[-1]
            or self.ownership_mask.shape[0] != len(self.entity_ids)
            or not np.all(np.isfinite(self.entity_tokens))
            or not np.all(np.isfinite(self.scene_tokens))
        ):
            raise ValueError("SD-JEPA encoded telemetry is invalid")

    @property
    def content_entity_tokens(self) -> NDArray[np.float64]:
        return self.entity_tokens[..., 2:]

    @property
    def current_content_tokens(self) -> NDArray[np.float64]:
        return self.content_entity_tokens[:, -1]


def cosine_margin_triplet_loss(
    embeddings: Any,
    *,
    margin: float = 0.2,
    negative_shift: int = 1,
    eps: float = 1e-8,
) -> Any:
    """Return the released middle/adjacent/cross-trajectory triplet."""

    if (
        embeddings.ndim != 3
        or embeddings.shape[0] < 2
        or embeddings.shape[1] < 2
        or not 1 <= negative_shift < embeddings.shape[0]
        or margin < 0.0
        or eps <= 0.0
    ):
        raise ValueError("SD-JEPA triplet inputs are invalid")
    torch = _require_torch()
    anchor_index = embeddings.shape[1] // 2
    positive_index = (
        anchor_index + 1
        if anchor_index + 1 < embeddings.shape[1]
        else anchor_index - 1
    )
    anchor = embeddings[:, anchor_index]
    positive = embeddings[:, positive_index]
    negative = torch.roll(anchor, shifts=negative_shift, dims=0)
    positive_similarity = torch.nn.functional.cosine_similarity(
        anchor, positive, dim=-1, eps=eps
    )
    negative_similarity = torch.nn.functional.cosine_similarity(
        anchor, negative, dim=-1, eps=eps
    )
    return torch.nn.functional.relu(
        negative_similarity - positive_similarity + margin
    ).mean()


class SdJepaModel:
    """Restorable canonical A2 or matched SD-JEPA control."""

    kind = "sd_jepa_representation"
    schema_version = 1

    def __init__(self, config: SdJepaConfig = SdJepaConfig()) -> None:
        self.config = config
        self._graph: Optional[DeclaredTelemetryGraph] = None
        self._feature_names: Tuple[str, ...] = ()
        self._ownership_mask: Optional[NDArray[np.bool_]] = None
        self._center: Optional[NDArray[np.float64]] = None
        self._scale: Optional[NDArray[np.float64]] = None
        self._condition_dimension: Optional[int] = None
        self._network: Any = None
        self._checkpoints: Tuple[Tuple[int, Mapping[str, Any]], ...] = ()
        self._training_metrics: Tuple[Mapping[str, float], ...] = ()
        self._selection_metrics: Tuple[Mapping[str, float], ...] = ()
        self._selected_step: Optional[int] = None

    @property
    def training_parameter_count(self) -> int:
        *_, network = self._fitted_values()
        return int(sum(value.numel() for value in network.parameters()))

    @property
    def inference_parameter_count(self) -> int:
        *_, network = self._fitted_values()
        return int(sum(value.numel() for value in network.encoder.parameters()))

    @property
    def selected_step(self) -> Optional[int]:
        return self._selected_step

    @property
    def training_metrics(self) -> Tuple[Mapping[str, float], ...]:
        return tuple(dict(row) for row in self._training_metrics)

    @property
    def selection_metrics(self) -> Tuple[Mapping[str, float], ...]:
        return tuple(dict(row) for row in self._selection_metrics)

    def fit(self, windows: ActionConditionedWindows) -> "SdJepaModel":
        """Fit checkpoint candidates with pair-blocked independent samples."""

        if len(set(windows.matched_pair_ids)) != self.config.expected_pair_count:
            raise ValueError("SD-JEPA fitting pair count differs")
        if windows.histories.shape[1] != 20 or windows.future_states.shape[1] != 10:
            raise ValueError("SD-JEPA requires 20+10 timestep windows")
        torch = _require_torch()
        _seed_torch(torch, self.config.seed)
        ownership = _fit_owned_feature_mask(windows)
        full = np.concatenate((windows.histories, windows.future_states), axis=1)
        center, scale = _fit_normalizer(full, ownership)
        condition_dimension = (
            len(windows.control_feature_names)
            + len(windows.entity_names) * len(windows.action_feature_names)
        )
        network = _build_network(
            torch,
            config=self.config,
            entity_count=len(windows.entity_names),
            feature_count=len(windows.state_feature_names),
            condition_dimension=condition_dimension,
            observed_entities=np.any(ownership, axis=1),
        )
        optimizer = torch.optim.AdamW(
            network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.pretrain_steps
        )
        schedule = PairBlockedAnchorSchedule(windows, seed=self.config.seed + 1)
        sigreg_generator = torch.Generator(device="cpu")
        negative_generator = np.random.default_rng(self.config.seed + 2)
        metrics: List[Mapping[str, float]] = []
        checkpoints = []
        network.train()
        for step in range(self.config.pretrain_steps):
            anchor = schedule.batch(step)
            values, conditions = _training_batch(
                windows, anchor.indices, ownership, center, scale
            )
            sigreg_generator.manual_seed(self.config.seed + 10_000 + step)
            negative_shift = int(
                negative_generator.integers(1, len(anchor.indices))
            )
            optimizer.zero_grad(set_to_none=True)
            losses = _objective_loss(
                torch,
                network,
                torch.as_tensor(values, dtype=torch.float32),
                torch.as_tensor(conditions, dtype=torch.float32),
                config=self.config,
                sigreg_generator=sigreg_generator,
                negative_shift=negative_shift,
            )
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            completed = step + 1
            if (
                completed % self.config.checkpoint_interval == 0
                or completed == self.config.pretrain_steps
            ):
                row = {
                    "step": float(completed),
                    "total": float(losses["total"].detach()),
                    "prediction": float(losses["prediction"].detach()),
                    "sigreg": float(losses["sigreg"].detach()),
                    "triplet": float(losses["triplet"].detach()),
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                }
                if not np.all(np.isfinite(list(row.values()))):
                    raise RuntimeError("SD-JEPA training became non-finite")
                metrics.append(row)
                checkpoints.append(
                    (
                        completed,
                        {
                            name: value.detach().cpu().clone()
                            for name, value in network.state_dict().items()
                        },
                    )
                )
        self._graph = windows.graph
        self._feature_names = windows.state_feature_names
        self._ownership_mask = ownership
        self._center = center
        self._scale = scale
        self._condition_dimension = condition_dimension
        self._network = network
        self._checkpoints = tuple(checkpoints)
        self._training_metrics = tuple(metrics)
        return self

    def select(self, windows: ActionConditionedWindows) -> "SdJepaModel":
        """Select this objective's checkpoint by its own held-role loss."""

        (
            graph,
            features,
            ownership,
            center,
            scale,
            condition_dimension,
            network,
        ) = self._fitted_values()
        if (
            windows.graph.to_dict() != graph.to_dict()
            or windows.state_feature_names != features
            or not self._checkpoints
        ):
            raise ValueError("SD-JEPA selection schema differs")
        torch = _require_torch()
        schedule = PairBlockedAnchorSchedule(windows, seed=self.config.seed + 3)
        evaluation_steps = min(10, len(schedule.transitions))
        rows = []
        best_key: Optional[Tuple[float, int]] = None
        best_state = None
        best_step = None
        for checkpoint_step, state in self._checkpoints:
            network.load_state_dict(state)
            network.eval()
            weighted = {"total": 0.0, "prediction": 0.0, "sigreg": 0.0, "triplet": 0.0}
            sigreg_generator = torch.Generator(device="cpu")
            with torch.no_grad():
                for local_step in range(evaluation_steps):
                    anchor = schedule.batch(local_step)
                    values, conditions = _training_batch(
                        windows, anchor.indices, ownership, center, scale
                    )
                    sigreg_generator.manual_seed(
                        self.config.seed + 20_000 + local_step
                    )
                    losses = _objective_loss(
                        torch,
                        network,
                        torch.as_tensor(values, dtype=torch.float32),
                        torch.as_tensor(conditions, dtype=torch.float32),
                        config=self.config,
                        sigreg_generator=sigreg_generator,
                        negative_shift=1,
                    )
                    for name in weighted:
                        weighted[name] += float(losses[name])
            row = {
                "step": float(checkpoint_step),
                **{
                    name: value / float(evaluation_steps)
                    for name, value in weighted.items()
                },
            }
            if not np.all(np.isfinite(list(row.values()))):
                raise RuntimeError("SD-JEPA selection became non-finite")
            rows.append(row)
            key = (row["total"], checkpoint_step)
            if best_key is None or key < best_key:
                best_key = key
                best_state = copy.deepcopy(state)
                best_step = checkpoint_step
        if best_state is None or best_step is None:
            raise RuntimeError("SD-JEPA selected no checkpoint")
        network.load_state_dict(best_state)
        network.eval()
        for parameter in network.parameters():
            parameter.requires_grad_(False)
        self._selected_step = best_step
        self._selection_metrics = tuple(rows)
        self._checkpoints = ()
        return self

    def encode(
        self,
        histories: NDArray[np.float64],
        graph: DeclaredTelemetryGraph,
    ) -> SdEncodedTelemetry:
        """Encode current histories without any future or action input."""

        torch = _require_torch()
        (
            _,
            _,
            ownership,
            center,
            scale,
            _,
            network,
        ) = self._selected_values()
        values = self._validate_histories(histories, graph)
        normalized = _normalize_states(values, ownership, center, scale)
        entity_parts = []
        scene_parts = []
        with torch.no_grad():
            for start in range(0, len(normalized), 256):
                entity, scene = network.encode(
                    torch.as_tensor(
                        normalized[start : start + 256],
                        dtype=torch.float32,
                    )
                )
                entity_parts.append(entity.cpu().numpy())
                scene_parts.append(scene.cpu().numpy())
        entity_tokens = np.asarray(
            np.concatenate(entity_parts, axis=0), dtype=np.float64
        )
        scene_tokens = np.asarray(
            np.concatenate(scene_parts, axis=0), dtype=np.float64
        )
        return SdEncodedTelemetry(
            entity_tokens=entity_tokens,
            scene_tokens=scene_tokens,
            entity_ids=graph.entity_ids,
            ownership_mask=ownership.copy(),
        )

    def raw_score(
        self,
        histories: NDArray[np.float64],
        graph: DeclaredTelemetryGraph,
        *,
        kind: str,
    ) -> NDArray[np.float64]:
        """Return angular change or zero-condition latent prediction error."""

        if kind not in {"angle", "z_mse"}:
            raise ValueError("SD-JEPA score kind is invalid")
        encoded = self.encode(histories, graph)
        if kind == "angle":
            angles = np.arctan2(
                encoded.scene_tokens[:, -2:, 1],
                encoded.scene_tokens[:, -2:, 0],
            )
            difference = np.arctan2(
                np.sin(angles[:, 1] - angles[:, 0]),
                np.cos(angles[:, 1] - angles[:, 0]),
            )
            return np.asarray(np.abs(difference) / math.pi, dtype=np.float64)
        torch = _require_torch()
        *_, condition_dimension, network = self._selected_values()
        scene = encoded.scene_tokens
        context = scene[:, -self.config.history_size - 1 : -1]
        with torch.no_grad():
            prediction = network.predict(
                torch.as_tensor(context, dtype=torch.float32),
                torch.zeros(
                    (len(context), condition_dimension), dtype=torch.float32
                ),
            ).cpu().numpy()
        error = np.mean(np.square(prediction - scene[:, -1]), axis=1)
        return np.asarray(error / (1.0 + error), dtype=np.float64)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe selected model."""

        (
            graph,
            features,
            ownership,
            center,
            scale,
            condition_dimension,
            network,
        ) = self._selected_values()
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "config": asdict(self.config),
            "graph": graph.to_dict(),
            "feature_names": list(features),
            "ownership_mask": ownership.astype(int).tolist(),
            "center": center.tolist(),
            "scale": scale.tolist(),
            "condition_dimension": condition_dimension,
            "state_dict": _state_dict_to_payload(network.state_dict()),
            "selected_step": self._selected_step,
            "training_metrics": [dict(row) for row in self._training_metrics],
            "selection_metrics": [dict(row) for row in self._selection_metrics],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SdJepaModel":
        """Restore and validate a selected SD-JEPA model."""

        if (
            payload.get("schema_version") != cls.schema_version
            or payload.get("kind") != cls.kind
        ):
            raise ValueError("SD-JEPA model schema is invalid")
        config = SdJepaConfig(**dict(payload["config"]))
        graph = DeclaredTelemetryGraph.from_dict(dict(payload["graph"]))
        features = tuple(str(value) for value in payload["feature_names"])
        ownership = np.asarray(payload["ownership_mask"], dtype=np.bool_)
        center = np.asarray(payload["center"], dtype=np.float64)
        scale = np.asarray(payload["scale"], dtype=np.float64)
        condition_dimension = payload["condition_dimension"]
        expected = (len(graph.entities), len(features))
        if (
            ownership.shape != expected
            or center.shape != expected
            or scale.shape != expected
            or not np.any(ownership)
            or np.any(scale <= 0.0)
            or not np.all(np.isfinite(center))
            or not np.all(np.isfinite(scale))
            or isinstance(condition_dimension, bool)
            or not isinstance(condition_dimension, int)
            or condition_dimension < 1
        ):
            raise ValueError("SD-JEPA fitted schema is invalid")
        torch = _require_torch()
        network = _build_network(
            torch,
            config=config,
            entity_count=len(graph.entities),
            feature_count=len(features),
            condition_dimension=condition_dimension,
            observed_entities=np.any(ownership, axis=1),
        )
        network.load_state_dict(
            _state_dict_from_payload(torch, dict(payload["state_dict"])),
            strict=True,
        )
        selected_step = payload.get("selected_step")
        if (
            isinstance(selected_step, bool)
            or not isinstance(selected_step, int)
            or selected_step < 1
        ):
            raise ValueError("SD-JEPA selected step is invalid")
        for parameter in network.parameters():
            parameter.requires_grad_(False)
        result = cls(config)
        result._graph = graph
        result._feature_names = features
        result._ownership_mask = ownership
        result._center = center
        result._scale = scale
        result._condition_dimension = condition_dimension
        result._network = network.eval()
        result._selected_step = selected_step
        result._training_metrics = _metric_rows(payload.get("training_metrics", ()))
        result._selection_metrics = _metric_rows(payload.get("selection_metrics", ()))
        return result

    def _validate_histories(
        self,
        histories: NDArray[np.float64],
        graph: DeclaredTelemetryGraph,
    ) -> NDArray[np.float64]:
        graph_, features, *_ = self._selected_values()
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
            raise ValueError("SD-JEPA public inputs are invalid")
        return values

    def _fitted_values(self) -> Tuple[Any, ...]:
        if (
            self._graph is None
            or not self._feature_names
            or self._ownership_mask is None
            or self._center is None
            or self._scale is None
            or self._condition_dimension is None
            or self._network is None
        ):
            raise ValueError("SD-JEPA model is not fitted")
        return (
            self._graph,
            self._feature_names,
            self._ownership_mask,
            self._center,
            self._scale,
            self._condition_dimension,
            self._network,
        )

    def _selected_values(self) -> Tuple[Any, ...]:
        values = self._fitted_values()
        if self._selected_step is None:
            raise ValueError("SD-JEPA model is not selected")
        return values


class SdScoreCalibrator:
    """Restorable monotone current-event calibrator for one frozen score."""

    kind = "sd_jepa_score_calibrator"
    schema_version = 1

    def __init__(self, *, score_name: str) -> None:
        if score_name not in SD_JEPA_SCORE_NAMES:
            raise ValueError("SD-JEPA score name is invalid")
        self.score_name = score_name
        self._calibration: Optional[Mapping[str, float]] = None
        self._control_trajectory_ids: Tuple[str, ...] = ()

    @property
    def calibration(self) -> Optional[Mapping[str, float]]:
        return None if self._calibration is None else dict(self._calibration)

    @property
    def score_kind(self) -> str:
        return "angle" if self.score_name.endswith("_angle") else "z_mse"

    def fit(
        self,
        model: SdJepaModel,
        windows: ActionConditionedWindows,
        event_definition: HepaEventDefinition,
    ) -> "SdScoreCalibrator":
        raw = model.raw_score(
            windows.histories, windows.graph, kind=self.score_kind
        )
        labels = (
            event_definition.observed_effect_scores(windows)
            > event_definition.threshold
        )
        slope, intercept, brier = fit_logit_calibrator(
            raw[:, None], labels[:, None]
        )
        calibrated = calibrate_probability_surface(
            raw[:, None], slope=slope, intercept=intercept
        )
        controls = _control_trajectory_ids(windows)
        threshold = trajectory_alert_threshold(
            calibrated, windows.trajectory_ids, controls
        )
        self._control_trajectory_ids = controls
        self._calibration = {
            "slope": slope,
            "intercept": intercept,
            "calibration_brier": brier,
            "alert_threshold": threshold,
        }
        return self

    def calibrated_risk(
        self,
        model: SdJepaModel,
        histories: NDArray[np.float64],
        graph: DeclaredTelemetryGraph,
    ) -> NDArray[np.float64]:
        calibration = self._values()
        raw = model.raw_score(histories, graph, kind=self.score_kind)
        return calibrate_probability_surface(
            raw[:, None],
            slope=calibration["slope"],
            intercept=calibration["intercept"],
        )[:, 0]

    def alert_decisions(
        self,
        model: SdJepaModel,
        histories: NDArray[np.float64],
        graph: DeclaredTelemetryGraph,
    ) -> NDArray[np.bool_]:
        calibration = self._values()
        return np.asarray(
            self.calibrated_risk(model, histories, graph)
            >= calibration["alert_threshold"],
            dtype=np.bool_,
        )

    def to_dict(self) -> Dict[str, Any]:
        calibration = self._values()
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "score_name": self.score_name,
            "control_trajectory_ids": list(self._control_trajectory_ids),
            "calibration": dict(calibration),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SdScoreCalibrator":
        if (
            payload.get("schema_version") != cls.schema_version
            or payload.get("kind") != cls.kind
        ):
            raise ValueError("SD-JEPA calibrator schema is invalid")
        result = cls(score_name=str(payload["score_name"]))
        calibration = {
            str(key): float(value)
            for key, value in dict(payload["calibration"]).items()
        }
        if (
            set(calibration)
            != {
                "slope",
                "intercept",
                "calibration_brier",
                "alert_threshold",
            }
            or calibration["slope"] < 0.0
            or not np.all(np.isfinite(list(calibration.values())))
        ):
            raise ValueError("SD-JEPA calibration is invalid")
        result._control_trajectory_ids = tuple(
            str(value) for value in payload.get("control_trajectory_ids", ())
        )
        result._calibration = calibration
        return result

    def _values(self) -> Mapping[str, float]:
        if self._calibration is None:
            raise ValueError("SD-JEPA score is not calibrated")
        return self._calibration


def _build_network(
    torch: Any,
    *,
    config: SdJepaConfig,
    entity_count: int,
    feature_count: int,
    condition_dimension: int,
    observed_entities: NDArray[np.bool_],
) -> Any:
    nn = torch.nn

    class Encoder(nn.Module):  # type: ignore[misc,name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.input_fc = nn.Linear(feature_count, config.hidden_width)
            self.entity_embedding = nn.Embedding(
                entity_count, config.hidden_width
            )
            self.hidden_fc = nn.Linear(
                config.hidden_width, config.hidden_width
            )
            self.output_fc = nn.Linear(config.hidden_width, config.width)
            self.register_buffer(
                "observed_entities",
                torch.as_tensor(observed_entities, dtype=torch.bool),
            )

        def forward(self, values: Any) -> Tuple[Any, Any]:
            if (
                values.ndim != 4
                or values.shape[2] != entity_count
                or values.shape[3] != feature_count
            ):
                raise ValueError("SD-JEPA encoder tensor is misaligned")
            hidden = self.input_fc(values)
            hidden = hidden + self.entity_embedding.weight[None, None]
            hidden = torch.nn.functional.silu(hidden)
            hidden = hidden + torch.nn.functional.silu(
                self.hidden_fc(hidden)
            )
            entity = self.output_fc(hidden)
            scene = entity[:, :, self.observed_entities].mean(dim=2)
            return entity, scene

    class Network(nn.Module):  # type: ignore[misc,name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.encoder = Encoder()
            predictor_input = (
                config.history_size * config.width + condition_dimension
            )
            self.predictor = nn.Sequential(
                nn.Linear(predictor_input, config.hidden_width),
                nn.SiLU(),
                nn.Linear(config.hidden_width, config.hidden_width),
                nn.SiLU(),
                nn.Linear(config.hidden_width, config.width),
            )

        def encode(self, values: Any) -> Tuple[Any, Any]:
            return self.encoder(values)

        def predict(self, context: Any, condition: Any) -> Any:
            if (
                context.ndim != 3
                or context.shape[1:] != (
                    config.history_size,
                    config.width,
                )
                or condition.ndim != 2
                or condition.shape != (
                    len(context),
                    condition_dimension,
                )
            ):
                raise ValueError("SD-JEPA predictor tensor is misaligned")
            return self.predictor(
                torch.cat((context.flatten(1), condition), dim=1)
            )

    return Network()


def _objective_loss(
    torch: Any,
    network: Any,
    full: Any,
    conditions: Any,
    *,
    config: SdJepaConfig,
    sigreg_generator: Any,
    negative_shift: int,
) -> Mapping[str, Any]:
    _, scene = network.encode(full)
    contexts = []
    targets = []
    for offset in range(10):
        target_index = 20 + offset
        contexts.append(
            scene[
                :,
                target_index - config.history_size : target_index,
            ]
        )
        targets.append(scene[:, target_index])
    stacked_context = torch.stack(contexts, dim=1).flatten(0, 1)
    stacked_target = torch.stack(targets, dim=1).flatten(0, 1)
    prediction = network.predict(
        stacked_context, conditions.flatten(0, 1)
    )
    prediction_loss = torch.nn.functional.mse_loss(
        prediction, stacked_target
    )
    from .action_conditioned_jepa import (
        sketched_isotropic_gaussian_regularization,
    )

    sigreg_input = (
        scene[..., config.progression_width :]
        if config.objective == "sd_jepa"
        else scene
    ).transpose(0, 1)
    sigreg = sketched_isotropic_gaussian_regularization(
        sigreg_input,
        generator=sigreg_generator,
        sketch_dimension=config.sigreg_sketch_dimension,
        knot_count=config.sigreg_knot_count,
    )
    if config.objective == "lewm_unsplit":
        triplet = scene.sum() * 0.0
    else:
        triplet_input = (
            scene[..., : config.progression_width]
            if config.objective == "sd_jepa"
            else scene
        )
        triplet = cosine_margin_triplet_loss(
            triplet_input,
            margin=config.triplet_margin,
            negative_shift=negative_shift,
        )
    total = (
        prediction_loss
        + config.sigreg_weight * sigreg
        + config.triplet_weight * triplet
    )
    return {
        "total": total,
        "prediction": prediction_loss,
        "sigreg": sigreg,
        "triplet": triplet,
    }


def _training_batch(
    windows: ActionConditionedWindows,
    indices: NDArray[np.int64],
    ownership: NDArray[np.bool_],
    center: NDArray[np.float64],
    scale: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    full = np.concatenate(
        (windows.histories[indices], windows.future_states[indices]), axis=1
    )
    conditions = np.concatenate(
        (
            windows.future_controls[indices],
            windows.future_actions[indices].reshape(
                len(indices), windows.future_actions.shape[1], -1
            ),
        ),
        axis=2,
    )
    return _normalize_states(full, ownership, center, scale), np.asarray(
        conditions, dtype=np.float64
    )


def _fit_normalizer(
    full: NDArray[np.float64],
    ownership: NDArray[np.bool_],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    center = np.mean(full, axis=(0, 1))
    scale = np.std(full, axis=(0, 1))
    center = np.where(ownership, center, 0.0)
    scale = np.where(ownership, np.maximum(scale, 1e-6), 1.0)
    return np.asarray(center, dtype=np.float64), np.asarray(
        scale, dtype=np.float64
    )


def _normalize_states(
    values: NDArray[np.float64],
    ownership: NDArray[np.bool_],
    center: NDArray[np.float64],
    scale: NDArray[np.float64],
) -> NDArray[np.float64]:
    normalized = (np.asarray(values, dtype=np.float64) - center) / scale
    return np.asarray(
        normalized * ownership[None, None], dtype=np.float64
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
        if entity_id in entity_positions and feature_name in feature_positions:
            mask[
                entity_positions[entity_id],
                feature_positions[feature_name],
            ] = True
    mask |= np.ptp(windows.histories, axis=(0, 1)) > 1e-9
    if not np.any(mask):
        raise ValueError("SD-JEPA schema has no observations")
    return mask


def _control_trajectory_ids(
    windows: ActionConditionedWindows,
) -> Tuple[str, ...]:
    try:
        applicable = windows.action_feature_names.index("applicable")
    except ValueError as error:
        raise ValueError("SD-JEPA needs applicable action field") from error
    treatments = {
        windows.trajectory_ids[index]
        for index in range(len(windows.histories))
        if np.any(windows.future_actions[index, ..., applicable] > 0.5)
    }
    return tuple(sorted(set(windows.trajectory_ids) - treatments))


def _state_dict_to_payload(
    state_dict: Mapping[str, Any],
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
            raise ValueError("SD-JEPA state tensor shape differs")
        if array.dtype.kind not in ("i", "u", "b") and not np.all(
            np.isfinite(array)
        ):
            raise ValueError("SD-JEPA state tensor is non-finite")
        result[str(name)] = (
            torch.as_tensor(array)
            if array.dtype.kind in ("i", "u", "b")
            else torch.as_tensor(array, dtype=torch.float32)
        )
    return result


def _metric_rows(values: Any) -> Tuple[Mapping[str, float], ...]:
    rows = []
    for raw in values:
        row = {
            str(key): float(value) for key, value in dict(raw).items()
        }
        if not np.all(np.isfinite(list(row.values()))):
            raise ValueError("SD-JEPA metric row is non-finite")
        rows.append(row)
    return tuple(rows)


def _seed_torch(torch: Any, seed: int) -> None:
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def _require_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError as error:
        raise RuntimeError(
            "SD-JEPA fitting requires optional training dependencies"
        ) from error
