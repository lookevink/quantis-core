"""Edge-sized CF-JEPA representation and Gaussian alert adapter.

Future state is available only to the self-supervised fitting objective.
Public encoding and alert inference accept current histories and the declared
telemetry graph.
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
from .hepa_jepa import (
    HepaEventDefinition,
    calibrate_probability_surface,
    fit_logit_calibrator,
    trajectory_alert_threshold,
)


CF_JEPA_OBJECTIVES = ("three_zone", "one_zone", "masked_latent")
CF_JEPA_ALERT_MODEL_NAMES = (
    "cf_jepa_target",
    "cf_jepa_online",
    "one_zone_target",
    "masked_latent_target",
    "matched_pca",
)
CF_JEPA_ASSESSMENT_ROLE_NAMES = (
    "calibration",
    "evaluation_iid",
    "evaluation_transfer",
)


@dataclass(frozen=True)
class CfJepaConfig:
    """Frozen controls for one edge-sized CF-JEPA objective."""

    objective: str = "three_zone"
    width: int = 32
    hidden_width: int = 64
    depth: int = 3
    pretrain_steps: int = 300
    checkpoint_interval: int = 50
    batch_size: int = 64
    crop_count: int = 4
    crop_min: float = 0.6
    crop_max: float = 0.8
    mask_ratio: float = 0.3
    learning_rate: float = 2.25e-4
    weight_decay: float = 1e-5
    ema_base: float = 0.983
    variance_weight: float = 0.081
    covariance_weight: float = 0.076
    invariance_weight: float = 1.101
    gradient_clip_norm: float = 1.0
    expected_pair_count: int = 40
    seed: int = 14014
    device: str = "cpu"

    def __post_init__(self) -> None:
        if (
            self.objective not in CF_JEPA_OBJECTIVES
            or self.width < 4
            or self.hidden_width < self.width
            or self.depth < 1
            or self.pretrain_steps < 1
            or self.checkpoint_interval < 1
            or self.batch_size < 2
            or self.crop_count < 1
            or not 0.0 < self.crop_min < self.crop_max < 1.0
            or not 0.0 < self.mask_ratio < 1.0
            or self.learning_rate <= 0.0
            or self.weight_decay < 0.0
            or not 0.0 < self.ema_base < 1.0
            or self.variance_weight < 0.0
            or self.covariance_weight < 0.0
            or self.invariance_weight < 0.0
            or self.gradient_clip_norm <= 0.0
            or self.expected_pair_count < 2
            or isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.device != "cpu"
        ):
            raise ValueError("CF-JEPA configuration is invalid")


@dataclass(frozen=True)
class CfEncodedTelemetry:
    """Entity-preserving pooled and per-timestep CF-JEPA states."""

    tokens: NDArray[np.float64]
    temporal_tokens: NDArray[np.float64]
    entity_ids: Tuple[str, ...]
    ownership_mask: NDArray[np.bool_]
    route: str

    def __post_init__(self) -> None:
        if (
            self.route not in {"online", "target"}
            or self.tokens.ndim != 3
            or self.temporal_tokens.ndim != 4
            or self.temporal_tokens.shape[:2] != self.tokens.shape[:2]
            or self.temporal_tokens.shape[-1] != self.tokens.shape[-1]
            or self.tokens.shape[1] != len(self.entity_ids)
            or self.ownership_mask.shape[0] != len(self.entity_ids)
            or not np.all(np.isfinite(self.tokens))
            or not np.all(np.isfinite(self.temporal_tokens))
        ):
            raise ValueError("CF-JEPA encoded telemetry is invalid")


def sample_cf_crop(
    length: int,
    *,
    crop_min: float,
    crop_max: float,
    generator: np.random.Generator,
) -> Tuple[int, int]:
    """Sample an official-style crop while reserving three future steps."""

    if (
        isinstance(length, bool)
        or length < 7
        or not 0.0 < crop_min < crop_max < 1.0
    ):
        raise ValueError("CF-JEPA crop controls are invalid")
    crop_length = max(int(length * generator.uniform(crop_min, crop_max)), 4)
    crop_length = min(crop_length, length - 3)
    max_start = max(length - crop_length - 3, 0)
    start = int(generator.integers(0, max_start + 1))
    return start, start + crop_length


def cf_forward_zones(
    crop_end: int, series_length: int
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """Split the complete post-crop suffix into three ordered zones."""

    remaining = series_length - crop_end
    if crop_end < 1 or remaining < 3:
        raise ValueError("CF-JEPA needs three post-crop positions")
    zone_size = remaining // 3
    return (
        (crop_end, crop_end + zone_size),
        (crop_end + zone_size, crop_end + 2 * zone_size),
        (crop_end + 2 * zone_size, series_length),
    )


class CfJepaModel:
    """Restorable online/EMA CF-JEPA representation model."""

    kind = "cf_jepa_representation"
    schema_version = 1

    def __init__(self, config: CfJepaConfig = CfJepaConfig()) -> None:
        self.config = config
        self._graph: Optional[DeclaredTelemetryGraph] = None
        self._feature_names: Tuple[str, ...] = ()
        self._ownership_mask: Optional[NDArray[np.bool_]] = None
        self._center: Optional[NDArray[np.float64]] = None
        self._scale: Optional[NDArray[np.float64]] = None
        self._network: Any = None
        self._checkpoints: Tuple[Tuple[int, Mapping[str, Any]], ...] = ()
        self._training_metrics: Tuple[Mapping[str, float], ...] = ()
        self._selection_metrics: Tuple[Mapping[str, float], ...] = ()
        self._selected_step: Optional[int] = None

    @property
    def training_parameter_count(self) -> int:
        """Return active online-encoder and predictor capacity."""

        *_, network = self._fitted_values()
        return int(
            sum(
                parameter.numel()
                for parameter in network.training_parameters()
            )
        )

    @property
    def inference_parameter_count(self) -> int:
        """Return one deployed encoder's parameter count."""

        *_, network = self._fitted_values()
        return int(
            sum(
                parameter.numel()
                for parameter in network.online_encoder.parameters()
            )
        )

    @property
    def selected_step(self) -> Optional[int]:
        return self._selected_step

    @property
    def training_metrics(self) -> Tuple[Mapping[str, float], ...]:
        return tuple(dict(row) for row in self._training_metrics)

    @property
    def selection_metrics(self) -> Tuple[Mapping[str, float], ...]:
        return tuple(dict(row) for row in self._selection_metrics)

    def fit(self, windows: ActionConditionedWindows) -> "CfJepaModel":
        """Fit checkpoint candidates on fitting histories and futures."""

        pair_count = len(set(windows.matched_pair_ids))
        if pair_count != self.config.expected_pair_count:
            raise ValueError(
                "CF-JEPA fitting pair count does not match the contract"
            )
        if windows.histories.shape[1] != 20 or (
            windows.future_states.shape[1] != 10
        ):
            raise ValueError("CF-JEPA requires 20+10 timestep windows")
        torch = _require_torch()
        _seed_torch(torch, self.config.seed)
        ownership = _fit_owned_feature_mask(windows)
        full = np.concatenate(
            (windows.histories, windows.future_states), axis=1
        )
        center, scale = _fit_normalizer(full, ownership)
        network = _build_network(
            torch,
            config=self.config,
            entity_count=len(windows.entity_names),
            feature_count=len(windows.state_feature_names),
        )
        optimizer = torch.optim.AdamW(
            network.training_parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        generator = np.random.default_rng(self.config.seed + 1)
        metrics: List[Mapping[str, float]] = []
        checkpoints = []
        network.train()
        for step in range(self.config.pretrain_steps):
            indices = generator.integers(
                0,
                len(full),
                size=min(self.config.batch_size, len(full)),
            )
            batch = _normalize_series(
                full[indices], ownership, center, scale
            )
            optimizer.zero_grad(set_to_none=True)
            losses = network.objective_loss(
                torch.as_tensor(batch, dtype=torch.float32),
                progress=(step + 1) / self.config.pretrain_steps,
                generator=generator,
            )
            total = losses["total"]
            if not bool(torch.isfinite(total)):
                raise RuntimeError("CF-JEPA pretraining became non-finite")
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                network.training_parameters(),
                self.config.gradient_clip_norm,
            )
            optimizer.step()
            network.update_target(
                _ema_momentum(
                    self.config.ema_base,
                    (step + 1) / self.config.pretrain_steps,
                )
            )
            metrics.append(
                {
                    "step": float(step + 1),
                    **{
                        name: float(value.detach())
                        for name, value in losses.items()
                    },
                }
            )
            if (
                (step + 1) % self.config.checkpoint_interval == 0
                or step + 1 == self.config.pretrain_steps
            ):
                checkpoints.append(
                    (step + 1, copy.deepcopy(network.state_dict()))
                )
        for parameter in network.parameters():
            parameter.requires_grad_(False)
        self._graph = windows.graph
        self._feature_names = windows.state_feature_names
        self._ownership_mask = ownership.copy()
        self._center = center.copy()
        self._scale = scale.copy()
        self._network = network.eval()
        self._checkpoints = tuple(checkpoints)
        self._training_metrics = tuple(metrics)
        return self

    def select(
        self, windows: ActionConditionedWindows
    ) -> "CfJepaModel":
        """Select a checkpoint by deterministic selection objective."""

        graph, features, ownership, center, scale, network = (
            self._fitted_values()
        )
        if (
            not self._checkpoints
            or windows.graph.to_dict() != graph.to_dict()
            or windows.state_feature_names != features
        ):
            raise ValueError("CF-JEPA selection inputs are invalid")
        full = np.concatenate(
            (windows.histories, windows.future_states), axis=1
        )
        normalized = _normalize_series(
            full, ownership, center, scale
        )
        rows = []
        best_loss = float("inf")
        best_step = -1
        best_state = None
        for step, state in self._checkpoints:
            network.load_state_dict(state)
            loss = _selection_loss(
                network,
                normalized,
                batch_size=max(256, self.config.batch_size),
                seed=self.config.seed + 50,
            )
            rows.append({"step": float(step), "loss": loss})
            if loss < best_loss - 1e-12:
                best_loss = loss
                best_step = step
                best_state = state
        if best_state is None:
            raise RuntimeError("CF-JEPA selected no checkpoint")
        network.load_state_dict(best_state)
        network.eval()
        self._selection_metrics = tuple(rows)
        self._selected_step = best_step
        self._checkpoints = ()
        return self

    def encode(
        self,
        histories: NDArray[np.float64],
        graph: DeclaredTelemetryGraph,
        *,
        route: str = "target",
    ) -> CfEncodedTelemetry:
        """Encode current histories with the selected online or EMA route."""

        if route not in {"online", "target"}:
            raise ValueError("CF-JEPA route must be online or target")
        torch = _require_torch()
        _, _, ownership, center, scale, network = self._selected_values()
        values = self._validate_histories(histories, graph)
        normalized = _normalize_series(
            values, ownership, center, scale
        )
        result = []
        encoder = (
            network.target_encoder
            if route == "target"
            else network.online_encoder
        )
        with torch.no_grad():
            for start in range(0, len(normalized), 256):
                result.append(
                    encoder(
                        torch.as_tensor(
                            normalized[start : start + 256],
                            dtype=torch.float32,
                        )
                    )
                    .cpu()
                    .numpy()
                )
        temporal = np.asarray(
            np.concatenate(result, axis=0), dtype=np.float64
        )
        return CfEncodedTelemetry(
            tokens=np.mean(temporal, axis=2),
            temporal_tokens=temporal,
            entity_ids=graph.entity_ids,
            ownership_mask=ownership.copy(),
            route=route,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe selected model."""

        graph, features, ownership, center, scale, network = (
            self._selected_values()
        )
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "config": asdict(self.config),
            "graph": graph.to_dict(),
            "feature_names": list(features),
            "ownership_mask": ownership.astype(int).tolist(),
            "center": center.tolist(),
            "scale": scale.tolist(),
            "state_dict": _state_dict_to_payload(network.state_dict()),
            "selected_step": self._selected_step,
            "training_metrics": [
                dict(row) for row in self._training_metrics
            ],
            "selection_metrics": [
                dict(row) for row in self._selection_metrics
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CfJepaModel":
        """Restore and validate a selected CF-JEPA model."""

        if (
            payload.get("schema_version") != cls.schema_version
            or payload.get("kind") != cls.kind
        ):
            raise ValueError("CF-JEPA model schema is invalid")
        config = CfJepaConfig(**dict(payload["config"]))
        model = cls(config)
        graph = DeclaredTelemetryGraph.from_dict(
            dict(payload["graph"])
        )
        features = tuple(str(value) for value in payload["feature_names"])
        ownership = np.asarray(
            payload["ownership_mask"], dtype=np.bool_
        )
        center = np.asarray(payload["center"], dtype=np.float64)
        scale = np.asarray(payload["scale"], dtype=np.float64)
        expected_shape = (len(graph.entities), len(features))
        if (
            ownership.shape != expected_shape
            or center.shape != expected_shape
            or scale.shape != expected_shape
            or not np.any(ownership)
            or not np.all(np.isfinite(center))
            or not np.all(np.isfinite(scale))
            or np.any(scale <= 0.0)
        ):
            raise ValueError("CF-JEPA normalizer is invalid")
        torch = _require_torch()
        network = _build_network(
            torch,
            config=config,
            entity_count=len(graph.entities),
            feature_count=len(features),
        )
        state = _state_dict_from_payload(torch, dict(payload["state_dict"]))
        network.load_state_dict(state, strict=True)
        for parameter in network.parameters():
            parameter.requires_grad_(False)
        selected_step = payload.get("selected_step")
        if (
            isinstance(selected_step, bool)
            or not isinstance(selected_step, int)
            or selected_step < 1
        ):
            raise ValueError("CF-JEPA selected step is invalid")
        model._graph = graph
        model._feature_names = features
        model._ownership_mask = ownership
        model._center = center
        model._scale = scale
        model._network = network.eval()
        model._selected_step = selected_step
        model._training_metrics = _metric_rows(
            payload.get("training_metrics", ())
        )
        model._selection_metrics = _metric_rows(
            payload.get("selection_metrics", ())
        )
        return model

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
            raise ValueError("CF-JEPA encoding inputs are invalid")
        return values

    def _fitted_values(
        self,
    ) -> Tuple[
        DeclaredTelemetryGraph,
        Tuple[str, ...],
        NDArray[np.bool_],
        NDArray[np.float64],
        NDArray[np.float64],
        Any,
    ]:
        if (
            self._graph is None
            or not self._feature_names
            or self._ownership_mask is None
            or self._center is None
            or self._scale is None
            or self._network is None
        ):
            raise ValueError("CF-JEPA model is not fitted")
        return (
            self._graph,
            self._feature_names,
            self._ownership_mask,
            self._center,
            self._scale,
            self._network,
        )

    def _selected_values(
        self,
    ) -> Tuple[
        DeclaredTelemetryGraph,
        Tuple[str, ...],
        NDArray[np.bool_],
        NDArray[np.float64],
        NDArray[np.float64],
        Any,
    ]:
        values = self._fitted_values()
        if self._selected_step is None:
            raise ValueError("CF-JEPA model is not selected")
        return values


class CfGaussianAlert:
    """Restorable source-faithful Gaussian representation alert adapter."""

    kind = "cf_jepa_gaussian_alert"
    schema_version = 1

    def __init__(
        self, *, route: str = "target", covariance_ridge: float = 1e-3
    ) -> None:
        if route not in {"online", "target"} or covariance_ridge <= 0.0:
            raise ValueError("CF-JEPA Gaussian alert controls are invalid")
        self.route = route
        self.covariance_ridge = float(covariance_ridge)
        self._means: Optional[NDArray[np.float64]] = None
        self._precisions: Optional[NDArray[np.float64]] = None
        self._entity_ids: Tuple[str, ...] = ()
        self._width: Optional[int] = None
        self._control_trajectory_ids: Tuple[str, ...] = ()
        self._calibration: Optional[Mapping[str, float]] = None

    @property
    def calibration(self) -> Optional[Mapping[str, float]]:
        return (
            None
            if self._calibration is None
            else dict(self._calibration)
        )

    def fit(
        self, model: CfJepaModel, windows: ActionConditionedWindows
    ) -> "CfGaussianAlert":
        """Fit entity-local Gaussians on fitting-control tokens only."""

        control_ids = _control_trajectory_ids(windows)
        selected = np.asarray(
            [
                trajectory_id in set(control_ids)
                for trajectory_id in windows.trajectory_ids
            ],
            dtype=np.bool_,
        )
        if np.sum(selected) < 2:
            raise ValueError("CF-JEPA Gaussian alert needs control windows")
        encoded = model.encode(
            windows.histories[selected],
            windows.graph,
            route=self.route,
        )
        means = np.mean(encoded.tokens, axis=0)
        precisions = np.empty(
            (
                len(encoded.entity_ids),
                encoded.tokens.shape[-1],
                encoded.tokens.shape[-1],
            ),
            dtype=np.float64,
        )
        for entity in range(len(encoded.entity_ids)):
            centered = encoded.tokens[:, entity] - means[entity]
            covariance = (
                centered.T @ centered
            ) / max(1, len(centered) - 1)
            covariance += self.covariance_ridge * np.eye(
                covariance.shape[0]
            )
            precisions[entity] = np.linalg.inv(covariance)
        if not np.all(np.isfinite(precisions)):
            raise RuntimeError("CF-JEPA Gaussian precision is non-finite")
        self._means = means
        self._precisions = precisions
        self._entity_ids = encoded.entity_ids
        self._width = encoded.tokens.shape[-1]
        self._control_trajectory_ids = control_ids
        return self

    def score(
        self,
        model: CfJepaModel,
        histories: NDArray[np.float64],
        graph: DeclaredTelemetryGraph,
    ) -> NDArray[np.float64]:
        """Return bounded monotone Mahalanobis anomaly scores."""

        means, precisions, entity_ids, width = self._fitted_values()
        encoded = model.encode(histories, graph, route=self.route)
        if (
            encoded.entity_ids != entity_ids
            or encoded.tokens.shape[-1] != width
        ):
            raise ValueError("CF-JEPA Gaussian representation differs")
        delta = encoded.tokens - means[None]
        per_entity = np.einsum(
            "bei,eij,bej->be", delta, precisions, delta
        )
        distances = np.sum(per_entity, axis=1) / float(
            len(entity_ids) * width
        )
        if not np.all(np.isfinite(distances)) or np.any(
            distances < -1e-8
        ):
            raise RuntimeError("CF-JEPA Gaussian score is invalid")
        distances = np.maximum(distances, 0.0)
        return np.asarray(
            distances / (1.0 + distances), dtype=np.float64
        )

    def fit_calibration(
        self,
        model: CfJepaModel,
        windows: ActionConditionedWindows,
        event_definition: HepaEventDefinition,
    ) -> "CfGaussianAlert":
        """Fit monotone calibration and strict control-maximum threshold."""

        raw = self.score(model, windows.histories, windows.graph)
        labels = event_definition.labels(windows)[:, -1]
        slope, intercept, brier = fit_logit_calibrator(
            raw[:, None], labels[:, None]
        )
        calibrated = calibrate_probability_surface(
            raw[:, None], slope=slope, intercept=intercept
        )
        control_ids = _control_trajectory_ids(windows)
        threshold = trajectory_alert_threshold(
            calibrated, windows.trajectory_ids, control_ids
        )
        self._calibration = {
            "slope": slope,
            "intercept": intercept,
            "calibration_brier": brier,
            "alert_threshold": threshold,
        }
        return self

    def calibrated_risk(
        self,
        model: CfJepaModel,
        histories: NDArray[np.float64],
        graph: DeclaredTelemetryGraph,
    ) -> NDArray[np.float64]:
        """Return calibrated event probability."""

        calibration = self._calibrated_values()
        raw = self.score(model, histories, graph)
        return calibrate_probability_surface(
            raw[:, None],
            slope=calibration["slope"],
            intercept=calibration["intercept"],
        )[:, 0]

    def alert_decisions(
        self,
        model: CfJepaModel,
        histories: NDArray[np.float64],
        graph: DeclaredTelemetryGraph,
    ) -> NDArray[np.bool_]:
        """Return decisions at the calibration-control threshold."""

        calibration = self._calibrated_values()
        return np.asarray(
            self.calibrated_risk(model, histories, graph)
            >= calibration["alert_threshold"],
            dtype=np.bool_,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return the fitted Gaussian and optional calibration."""

        means, precisions, entity_ids, width = self._fitted_values()
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "route": self.route,
            "covariance_ridge": self.covariance_ridge,
            "entity_ids": list(entity_ids),
            "width": width,
            "means": means.tolist(),
            "precisions": precisions.tolist(),
            "control_trajectory_ids": list(
                self._control_trajectory_ids
            ),
            "calibration": (
                None
                if self._calibration is None
                else dict(self._calibration)
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CfGaussianAlert":
        """Restore and validate a Gaussian alert adapter."""

        if (
            payload.get("schema_version") != cls.schema_version
            or payload.get("kind") != cls.kind
        ):
            raise ValueError("CF-JEPA Gaussian schema is invalid")
        result = cls(
            route=str(payload["route"]),
            covariance_ridge=float(payload["covariance_ridge"]),
        )
        entity_ids = tuple(str(value) for value in payload["entity_ids"])
        width = payload["width"]
        if (
            isinstance(width, bool)
            or not isinstance(width, int)
            or width < 1
            or not entity_ids
        ):
            raise ValueError("CF-JEPA Gaussian dimensions are invalid")
        means = np.asarray(payload["means"], dtype=np.float64)
        precisions = np.asarray(payload["precisions"], dtype=np.float64)
        if (
            means.shape != (len(entity_ids), width)
            or precisions.shape != (len(entity_ids), width, width)
            or not np.all(np.isfinite(means))
            or not np.all(np.isfinite(precisions))
        ):
            raise ValueError("CF-JEPA Gaussian tensors are invalid")
        calibration_payload = payload.get("calibration")
        calibration = (
            None
            if calibration_payload is None
            else {
                str(key): float(value)
                for key, value in dict(calibration_payload).items()
            }
        )
        if calibration is not None and (
            set(calibration)
            != {
                "slope",
                "intercept",
                "calibration_brier",
                "alert_threshold",
            }
            or not np.all(
                np.isfinite(list(calibration.values()))
            )
            or calibration["slope"] < 0.0
        ):
            raise ValueError("CF-JEPA Gaussian calibration is invalid")
        result._means = means
        result._precisions = precisions
        result._entity_ids = entity_ids
        result._width = width
        result._control_trajectory_ids = tuple(
            str(value)
            for value in payload.get("control_trajectory_ids", ())
        )
        result._calibration = calibration
        return result

    def _fitted_values(
        self,
    ) -> Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        Tuple[str, ...],
        int,
    ]:
        if (
            self._means is None
            or self._precisions is None
            or not self._entity_ids
            or self._width is None
        ):
            raise ValueError("CF-JEPA Gaussian alert is not fitted")
        return (
            self._means,
            self._precisions,
            self._entity_ids,
            self._width,
        )

    def _calibrated_values(self) -> Mapping[str, float]:
        self._fitted_values()
        if self._calibration is None:
            raise ValueError("CF-JEPA Gaussian alert is not calibrated")
        return self._calibration


def _build_network(
    torch: Any,
    *,
    config: CfJepaConfig,
    entity_count: int,
    feature_count: int,
) -> Any:
    nn = torch.nn
    functional = torch.nn.functional

    class DepthwiseConvolution(nn.Module):  # type: ignore[misc,name-defined]
        def __init__(
            self, channels: int, kernel_size: int, dilation: int
        ) -> None:
            super().__init__()
            receptive = (kernel_size - 1) * dilation
            self.left = receptive // 2
            self.right = receptive - self.left
            self.convolution = nn.Conv1d(
                channels,
                channels,
                kernel_size,
                dilation=dilation,
                groups=channels,
                bias=False,
            )

        def forward(self, values: Any) -> Any:
            return self.convolution(
                functional.pad(values, (self.left, self.right))
            )

    class Block(nn.Module):  # type: ignore[misc,name-defined]
        def __init__(self, channels: int, dilation: int) -> None:
            super().__init__()
            self.depthwise = nn.ModuleList(
                [
                    DepthwiseConvolution(channels, kernel, dilation)
                    for kernel in (3, 9, 15)
                ]
            )
            self.normalization1 = nn.BatchNorm1d(channels)
            self.pointwise = nn.Conv1d(
                channels, channels, 1, bias=False
            )
            self.normalization2 = nn.BatchNorm1d(channels)

        def forward(self, values: Any) -> Any:
            residual = values
            hidden = sum(layer(values) for layer in self.depthwise)
            hidden = functional.gelu(self.normalization1(hidden))
            hidden = functional.gelu(
                self.normalization2(self.pointwise(hidden))
            )
            return hidden + residual

    class Encoder(nn.Module):  # type: ignore[misc,name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.input_fc = nn.Linear(
                feature_count, config.hidden_width
            )
            self.entity_embedding = nn.Embedding(
                entity_count, config.hidden_width
            )
            self.blocks = nn.ModuleList(
                [
                    Block(config.hidden_width, 2**index)
                    for index in range(config.depth)
                ]
            )
            self.output_fc = nn.Linear(
                config.hidden_width, config.width
            )

        def forward(self, values: Any) -> Any:
            batch, entities, time, features = values.shape
            if entities != entity_count or features != feature_count:
                raise ValueError("CF-JEPA encoder tensor is misaligned")
            hidden = self.input_fc(values)
            identity = self.entity_embedding.weight[
                None, :, None, :
            ]
            hidden = hidden + identity
            hidden = hidden.reshape(
                batch * entities, time, config.hidden_width
            ).transpose(1, 2)
            for block in self.blocks:
                hidden = block(hidden)
            hidden = hidden.transpose(1, 2).reshape(
                batch, entities, time, config.hidden_width
            )
            return self.output_fc(hidden)

    class Network(nn.Module):  # type: ignore[misc,name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.online_encoder = Encoder()
            self.target_encoder = copy.deepcopy(self.online_encoder)
            for parameter in self.target_encoder.parameters():
                parameter.requires_grad_(False)
            if config.objective == "three_zone":
                self.predictors = nn.ModuleList(
                    [nn.Linear(config.width, config.width) for _ in range(3)]
                )
                for predictor in self.predictors:
                    with torch.no_grad():
                        predictor.weight.copy_(
                            torch.eye(config.width)
                            + torch.randn(
                                config.width, config.width
                            )
                            * 0.01
                        )
                        predictor.bias.zero_()
                self.predictor = None
            elif config.objective == "one_zone":
                self.predictors = nn.ModuleList()
                self.predictor = nn.Linear(config.width, config.width)
                with torch.no_grad():
                    self.predictor.weight.copy_(
                        torch.eye(config.width)
                        + torch.randn(config.width, config.width) * 0.01
                    )
                    self.predictor.bias.zero_()
            else:
                self.predictors = nn.ModuleList()
                self.predictor = nn.Sequential(
                    nn.Linear(config.width, config.width),
                    nn.GELU(),
                    nn.Linear(config.width, config.width),
                )

        def training_parameters(self) -> Any:
            parameters = list(self.online_encoder.parameters())
            parameters.extend(self.predictors.parameters())
            if self.predictor is not None:
                parameters.extend(self.predictor.parameters())
            return parameters

        def update_target(self, momentum: float) -> None:
            with torch.no_grad():
                for target, online in zip(
                    self.target_encoder.parameters(),
                    self.online_encoder.parameters(),
                ):
                    target.data.mul_(momentum).add_(
                        online.data, alpha=1.0 - momentum
                    )
                for target, online in zip(
                    self.target_encoder.buffers(),
                    self.online_encoder.buffers(),
                ):
                    if target.dtype.is_floating_point:
                        target.data.mul_(momentum).add_(
                            online.data, alpha=1.0 - momentum
                        )
                    else:
                        target.data.copy_(online.data)

        def objective_loss(
            self,
            full: Any,
            *,
            progress: float,
            generator: np.random.Generator,
        ) -> Mapping[str, Any]:
            if config.objective == "masked_latent":
                return self._masked_loss(full, generator)
            return self._forward_loss(full, progress, generator)

        def _forward_loss(
            self,
            full: Any,
            progress: float,
            generator: np.random.Generator,
        ) -> Mapping[str, Any]:
            with torch.no_grad():
                target_full = self.target_encoder(full)
            horizon_loss = torch.zeros((), device=full.device)
            term_count = 0
            crop_means = []
            crop_temporal = []
            length = full.shape[2]
            for _ in range(config.crop_count):
                start, end = sample_cf_crop(
                    length,
                    crop_min=config.crop_min,
                    crop_max=config.crop_max,
                    generator=generator,
                )
                crop = self.online_encoder(full[:, :, start:end])
                crop_means.append(crop.mean(dim=2))
                crop_temporal.append(crop)
                if config.objective == "three_zone":
                    for index, (zone_start, zone_end) in enumerate(
                        cf_forward_zones(end, length)
                    ):
                        prediction = self.predictors[index](crop)
                        target = target_full[
                            :, :, zone_start:zone_end
                        ]
                        prediction = _pool_temporal(
                            functional, prediction, target.shape[2]
                        )
                        horizon_loss = horizon_loss + functional.l1_loss(
                            functional.normalize(prediction, dim=-1),
                            functional.normalize(target, dim=-1),
                        )
                        term_count += 1
                else:
                    assert self.predictor is not None
                    prediction = self.predictor(crop)
                    target = target_full[:, :, end:]
                    prediction = _pool_temporal(
                        functional, prediction, target.shape[2]
                    )
                    horizon_loss = horizon_loss + functional.l1_loss(
                        functional.normalize(prediction, dim=-1),
                        functional.normalize(target, dim=-1),
                    )
                    term_count += 1
            horizon_loss = horizon_loss / max(1, term_count)
            pooled = torch.cat(crop_means, dim=0).reshape(
                -1, config.width
            )
            variance = _variance_loss(torch, functional, pooled)
            covariance = _covariance_loss(torch, pooled)
            invariance = _invariance_loss(
                torch, functional, crop_temporal
            )
            horizon_weight = (
                max(0.0, 1.0 - progress)
                if config.objective == "three_zone"
                else 1.0
            )
            total = (
                horizon_weight * horizon_loss
                + config.variance_weight * variance
                + config.covariance_weight * covariance
                + config.invariance_weight * invariance
            )
            return {
                "total": total,
                "horizon": horizon_loss,
                "variance": variance,
                "covariance": covariance,
                "invariance": invariance,
                "horizon_weight": torch.as_tensor(
                    horizon_weight, device=full.device
                ),
            }

        def _masked_loss(
            self, full: Any, generator: np.random.Generator
        ) -> Mapping[str, Any]:
            batch, entities, time, _ = full.shape
            raw_mask = generator.random((batch, entities, time))
            mask = torch.as_tensor(
                raw_mask < config.mask_ratio,
                dtype=torch.bool,
                device=full.device,
            )
            if not bool(torch.any(mask)):
                mask[0, 0, 0] = True
            masked = full * (~mask).unsqueeze(-1)
            encoded = self.online_encoder(masked)
            with torch.no_grad():
                target = self.target_encoder(full)
            assert self.predictor is not None
            prediction = self.predictor(encoded)
            horizon_loss = functional.l1_loss(
                prediction[mask], target[mask]
            )
            pooled = encoded.mean(dim=2).reshape(-1, config.width)
            variance = _variance_loss(torch, functional, pooled)
            covariance = _covariance_loss(torch, pooled)
            invariance = torch.zeros((), device=full.device)
            total = (
                horizon_loss
                + config.variance_weight * variance
                + config.covariance_weight * covariance
            )
            return {
                "total": total,
                "horizon": horizon_loss,
                "variance": variance,
                "covariance": covariance,
                "invariance": invariance,
                "horizon_weight": torch.ones((), device=full.device),
            }

    return Network()


def _pool_temporal(functional: Any, values: Any, length: int) -> Any:
    if length < 1:
        raise ValueError("CF-JEPA target cannot be empty")
    if values.shape[2] == length:
        return values
    batch, entities, _, width = values.shape
    flattened = values.reshape(batch * entities, values.shape[2], width)
    pooled = functional.adaptive_avg_pool1d(
        flattened.transpose(1, 2), length
    ).transpose(1, 2)
    return pooled.reshape(batch, entities, length, width)


def _variance_loss(torch: Any, functional: Any, values: Any) -> Any:
    if len(values) < 2:
        return torch.zeros((), device=values.device)
    standard_deviation = torch.sqrt(
        values.var(dim=0, unbiased=True) + 1e-4
    )
    return functional.relu(1.0 - standard_deviation).mean()


def _covariance_loss(torch: Any, values: Any) -> Any:
    if len(values) < 2:
        return torch.zeros((), device=values.device)
    centered = values - values.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / (len(values) - 1)
    diagonal = torch.diagonal(covariance)
    return (
        covariance.square().sum() - diagonal.square().sum()
    ) / values.shape[1]


def _invariance_loss(
    torch: Any, functional: Any, crops: List[Any]
) -> Any:
    if len(crops) < 2:
        return torch.zeros((), device=crops[0].device)
    losses = []
    minimum = min(crop.shape[2] for crop in crops)
    pool_sizes = [
        value for value in (2, 4, 8) if value <= minimum
    ] or [1]
    for pool_size in pool_sizes:
        pooled_crops = []
        for crop in crops:
            batch, entities, time, width = crop.shape
            flattened = crop.reshape(batch * entities, time, width)
            pooled = functional.adaptive_avg_pool1d(
                flattened.transpose(1, 2), pool_size
            ).transpose(1, 2)
            pooled_crops.append(
                pooled.reshape(
                    batch, entities, pool_size, width
                ).mean(dim=2)
            )
        stacked = torch.stack(pooled_crops, dim=0)
        losses.append(
            (stacked - stacked.mean(dim=0, keepdim=True))
            .square()
            .mean()
        )
    return sum(losses) / len(losses)


def _selection_loss(
    network: Any,
    normalized: NDArray[np.float64],
    *,
    batch_size: int,
    seed: int,
) -> float:
    torch = _require_torch()
    generator = np.random.default_rng(seed)
    values = []
    network.eval()
    with torch.no_grad():
        for start in range(0, len(normalized), batch_size):
            batch = normalized[start : start + batch_size]
            losses = network.objective_loss(
                torch.as_tensor(batch, dtype=torch.float32),
                progress=0.5,
                generator=generator,
            )
            values.append(float(losses["total"]) * len(batch))
    return float(sum(values) / len(normalized))


def _ema_momentum(base: float, progress: float) -> float:
    return 1.0 - (1.0 - base) * (
        math.cos(math.pi * min(max(progress, 0.0), 1.0)) + 1.0
    ) / 2.0


def _fit_normalizer(
    full: NDArray[np.float64],
    ownership: NDArray[np.bool_],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    center = np.mean(full, axis=(0, 1))
    scale = np.std(full, axis=(0, 1))
    center = np.where(ownership, center, 0.0)
    scale = np.where(ownership, np.maximum(scale, 1e-6), 1.0)
    return (
        np.asarray(center, dtype=np.float64),
        np.asarray(scale, dtype=np.float64),
    )


def _normalize_series(
    values: NDArray[np.float64],
    ownership: NDArray[np.bool_],
    center: NDArray[np.float64],
    scale: NDArray[np.float64],
) -> NDArray[np.float64]:
    normalized = (np.asarray(values, dtype=np.float64) - center) / scale
    normalized = normalized * ownership[None, None]
    return np.asarray(
        normalized.transpose(0, 2, 1, 3), dtype=np.float64
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
        raise ValueError("CF-JEPA telemetry schema has no observations")
    return mask


def _control_trajectory_ids(
    windows: ActionConditionedWindows,
) -> Tuple[str, ...]:
    try:
        applicable = windows.action_feature_names.index("applicable")
    except ValueError as error:
        raise ValueError(
            "CF-JEPA needs the applicable action field"
        ) from error
    treatments = {
        windows.trajectory_ids[index]
        for index in range(len(windows.histories))
        if np.any(windows.future_actions[index, ..., applicable] > 0.5)
    }
    return tuple(sorted(set(windows.trajectory_ids) - treatments))


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
            raise ValueError("CF-JEPA state tensor shape differs")
        if array.dtype.kind not in ("i", "u", "b") and not np.all(
            np.isfinite(array)
        ):
            raise ValueError("CF-JEPA state tensor is non-finite")
        tensor = (
            torch.as_tensor(array)
            if array.dtype.kind in ("i", "u", "b")
            else torch.as_tensor(array, dtype=torch.float32)
        )
        result[str(name)] = tensor
    return result


def _metric_rows(values: Any) -> Tuple[Mapping[str, float], ...]:
    rows = []
    for raw in values:
        row = {
            str(key): float(value)
            for key, value in dict(raw).items()
        }
        if not np.all(np.isfinite(list(row.values()))):
            raise ValueError("CF-JEPA metric row is non-finite")
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
            "CF-JEPA fitting requires optional training dependencies"
        ) from error
