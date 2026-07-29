"""JEPA-assisted empirical error certificates for immutable raw dynamics."""

import copy
import hashlib
import importlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from ..action_conditioned_dynamics import (
    ActionConditionedWindows,
    TrajectoryDistribution,
)
from ..graph_telemetry import DeclaredTelemetryGraph
from .complete_lejepa import fit_owned_feature_mask
from .models import ContractiveLowRankDynamics


ERROR_CERTIFICATE_OBJECTIVES = (
    "jepa_error_certificate",
    "raw_error_certificate",
    "deranged_jepa_certificate",
)


@dataclass(frozen=True)
class ErrorCertificateJepaConfig:
    """Frozen controls for one equal-capacity certificate cell."""

    objective: str = "jepa_error_certificate"
    width: int = 16
    hidden_width: int = 64
    pretrain_steps: int = 800
    checkpoint_interval: int = 100
    batch_size: int = 128
    learning_rate: float = 5e-4
    weight_decay: float = 1e-3
    ema_decay: float = 0.996
    latent_weight: float = 0.2
    quantile: float = 0.95
    expected_pair_count: int = 40
    seed: int = 25021
    device: str = "cpu"

    def __post_init__(self) -> None:
        integers = (
            self.width,
            self.hidden_width,
            self.pretrain_steps,
            self.checkpoint_interval,
            self.batch_size,
            self.expected_pair_count,
        )
        if (
            self.objective not in ERROR_CERTIFICATE_OBJECTIVES
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
                for value in integers
            )
            or self.learning_rate <= 0.0
            or self.weight_decay < 0.0
            or not 0.0 <= self.ema_decay < 1.0
            or self.latent_weight < 0.0
            or not 0.5 < self.quantile < 1.0
            or isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.device != "cpu"
        ):
            raise ValueError(
                "Error-Certificate-JEPA configuration is invalid"
            )

    @property
    def effective_latent_weight(self) -> float:
        if self.objective == "raw_error_certificate":
            return 0.0
        return self.latent_weight

    @property
    def uses_latent_features(self) -> bool:
        return self.objective != "raw_error_certificate"


@dataclass(frozen=True)
class CertifiedForecast:
    """One unchanged raw distribution and empirical horizon bound."""

    distribution: TrajectoryDistribution
    error_bound: NDArray[np.float64]

    def __post_init__(self) -> None:
        if (
            self.error_bound.shape != self.distribution.mean.shape[:2]
            or not np.all(np.isfinite(self.error_bound))
            or np.any(self.error_bound < 0.0)
        ):
            raise ValueError("certified forecast bound is invalid")


class ErrorCertificateJepa:
    """Restorable learned upper certificate over a frozen raw forecast."""

    kind = "error_certificate_jepa"
    schema_version = 1

    def __init__(
        self,
        config: ErrorCertificateJepaConfig = (
            ErrorCertificateJepaConfig()
        ),
    ) -> None:
        self.config = config
        self._graph: Optional[DeclaredTelemetryGraph] = None
        self._feature_names: Tuple[str, ...] = ()
        self._control_names: Tuple[str, ...] = ()
        self._action_names: Tuple[str, ...] = ()
        self._ownership_mask: Optional[NDArray[np.bool_]] = None
        self._baseline_sha256 = ""
        self._network: Any = None
        self._checkpoints: Tuple[Tuple[int, Mapping[str, Any]], ...] = ()
        self._training_metrics: Tuple[Mapping[str, float], ...] = ()
        self._selection_metrics: Tuple[Mapping[str, float], ...] = ()
        self._selected_step: Optional[int] = None
        self._calibration_adjustment: Optional[float] = None
        self._calibration_control_count = 0

    @property
    def training_parameter_count(self) -> int:
        *_, network = self._fitted_values()
        return int(sum(value.numel() for value in network.parameters()))

    @property
    def inference_parameter_count(self) -> int:
        *_, network = self._fitted_values()
        modules: Sequence[Any] = (
            network.online_encoder,
            network.predictor,
            network.certificate_head,
        )
        return int(
            sum(
                parameter.numel()
                for module in modules
                for parameter in module.parameters()
            )
            + network.horizon_embedding.numel()
        )

    @property
    def baseline_sha256(self) -> str:
        self._fitted_values()
        return self._baseline_sha256

    @property
    def selected_step(self) -> Optional[int]:
        return self._selected_step

    @property
    def calibration_adjustment(self) -> float:
        if self._calibration_adjustment is None:
            raise ValueError("error certificate is not calibrated")
        return self._calibration_adjustment

    @property
    def training_metrics(self) -> Tuple[Mapping[str, float], ...]:
        return tuple(dict(row) for row in self._training_metrics)

    @property
    def selection_metrics(self) -> Tuple[Mapping[str, float], ...]:
        return tuple(dict(row) for row in self._selection_metrics)

    def fit(
        self,
        windows: ActionConditionedWindows,
        baseline: ContractiveLowRankDynamics,
    ) -> "ErrorCertificateJepa":
        """Fit quantile and optional JEPA losses without changing raw."""

        _validate_windows(windows, self.config.expected_pair_count)
        baseline_payload = baseline.to_dict()
        baseline_hash = _artifact_sha256(baseline_payload)
        raw_prediction = baseline.rollout(
            windows.histories,
            windows.future_controls,
            windows.future_actions,
            windows.graph,
        ).mean
        ownership = fit_owned_feature_mask(windows)
        error_target = realized_raw_error(
            raw_prediction, windows.future_states, ownership
        )
        torch = _require_torch()
        _seed_torch(torch, self.config.seed)
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
            horizon=len(windows.future_states[0]),
        )
        optimized = [
            parameter
            for name, parameter in network.named_parameters()
            if not name.startswith("target_encoder.")
        ]
        optimizer = torch.optim.AdamW(
            optimized,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.pretrain_steps
        )
        rng = np.random.default_rng(self.config.seed + 1)
        order = rng.permutation(len(windows.histories)).astype(np.int64)
        metrics: List[Mapping[str, float]] = []
        checkpoints = []
        network.train()
        for step in range(self.config.pretrain_steps):
            start = (step * self.config.batch_size) % len(order)
            positions = (
                np.arange(self.config.batch_size, dtype=np.int64) + start
            ) % len(order)
            indices = order[positions]
            donor = np.roll(indices, 1)
            batch = _training_arrays(
                windows,
                raw_prediction,
                error_target,
                indices=indices,
                donor=donor,
                ownership=ownership,
                derange=(
                    self.config.objective
                    == "deranged_jepa_certificate"
                ),
            )
            optimizer.zero_grad(set_to_none=True)
            losses = _objective_loss(
                torch, network, batch, config=self.config
            )
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(optimized, 1.0)
            optimizer.step()
            _update_target_encoder(
                network, decay=self.config.ema_decay
            )
            scheduler.step()
            completed = step + 1
            if (
                completed % self.config.checkpoint_interval == 0
                or completed == self.config.pretrain_steps
            ):
                row = {
                    "step": float(completed),
                    "total": float(losses["total"].detach()),
                    "pinball": float(losses["pinball"].detach()),
                    "latent": float(losses["latent"].detach()),
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                }
                if not np.all(np.isfinite(list(row.values()))):
                    raise RuntimeError(
                        "Error-Certificate-JEPA training is non-finite"
                    )
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
        if _artifact_sha256(baseline.to_dict()) != baseline_hash:
            raise RuntimeError(
                "error certificate fitting mutated raw baseline"
            )
        self._graph = windows.graph
        self._feature_names = windows.state_feature_names
        self._control_names = windows.control_feature_names
        self._action_names = windows.action_feature_names
        self._ownership_mask = ownership
        self._baseline_sha256 = baseline_hash
        self._network = network
        self._checkpoints = tuple(checkpoints)
        self._training_metrics = tuple(metrics)
        return self

    def select(
        self,
        windows: ActionConditionedWindows,
        baseline: ContractiveLowRankDynamics,
    ) -> "ErrorCertificateJepa":
        """Select checkpoint by true selection-role pinball loss."""

        (
            graph,
            features,
            controls,
            actions,
            ownership,
            _,
            network,
        ) = self._fitted_values()
        _validate_schema(
            windows,
            graph=graph,
            features=features,
            controls=controls,
            actions=actions,
        )
        if _artifact_sha256(baseline.to_dict()) != self._baseline_sha256:
            raise ValueError("error certificate selection baseline differs")
        raw = baseline.rollout(
            windows.histories,
            windows.future_controls,
            windows.future_actions,
            windows.graph,
        ).mean
        target = realized_raw_error(raw, windows.future_states, ownership)
        torch = _require_torch()
        rows = []
        best_key: Optional[Tuple[float, int]] = None
        best_state = None
        best_step = None
        for step, state in self._checkpoints:
            network.load_state_dict(state)
            network.eval()
            prediction = _predict_batches(
                torch,
                network,
                windows.histories,
                windows.future_controls,
                windows.future_actions,
                raw,
                ownership,
                use_latent=self.config.uses_latent_features,
            )
            value = pinball_loss(
                prediction, target, quantile=self.config.quantile
            )
            row = {"step": float(step), "pinball": value}
            rows.append(row)
            key = (value, step)
            if best_key is None or key < best_key:
                best_key = key
                best_state = copy.deepcopy(state)
                best_step = step
        if best_state is None or best_step is None:
            raise RuntimeError("error certificate selected no checkpoint")
        network.load_state_dict(best_state)
        network.eval()
        for parameter in network.parameters():
            parameter.requires_grad_(False)
        self._selected_step = best_step
        self._selection_metrics = tuple(rows)
        self._checkpoints = ()
        return self

    def calibrate(
        self,
        windows: ActionConditionedWindows,
        baseline: ContractiveLowRankDynamics,
    ) -> "ErrorCertificateJepa":
        """Additively conformalize on control-trajectory maxima."""

        if _artifact_sha256(baseline.to_dict()) != self.baseline_sha256:
            raise ValueError(
                "error certificate calibration baseline differs"
            )
        raw = baseline.rollout(
            windows.histories,
            windows.future_controls,
            windows.future_actions,
            windows.graph,
        ).mean
        target = realized_raw_error(
            raw, windows.future_states, self._ownership_mask_value()
        )
        predicted = self.predict_unadjusted(
            windows.histories,
            windows.future_controls,
            windows.future_actions,
            raw,
            windows.graph,
        )
        violations = target - predicted
        control_maxima = _control_trajectory_maxima(
            violations, windows
        )
        quantile_adjustment = float(
            np.quantile(
                control_maxima,
                self.config.quantile,
                method="higher",
            )
        )
        adjustment = max(0.0, quantile_adjustment)
        self._calibration_adjustment = adjustment
        self._calibration_control_count = len(control_maxima)
        return self

    def predict_unadjusted(
        self,
        histories: NDArray[np.float64],
        future_controls: NDArray[np.float64],
        future_actions: NDArray[np.float64],
        raw_prediction: NDArray[np.float64],
        graph: DeclaredTelemetryGraph,
    ) -> NDArray[np.float64]:
        """Predict a non-negative pre-calibration error bound."""

        (
            graph_,
            _,
            controls,
            actions,
            ownership,
            _,
            network,
        ) = self._selected_values()
        history = _validate_public_inputs(
            histories,
            future_controls,
            future_actions,
            raw_prediction,
            graph,
            fitted_graph=graph_,
            feature_count=len(self._feature_names),
            control_count=len(controls),
            action_count=len(actions),
        )
        return _predict_batches(
            _require_torch(),
            network,
            history,
            np.asarray(future_controls, dtype=np.float64),
            np.asarray(future_actions, dtype=np.float64),
            np.asarray(raw_prediction, dtype=np.float64),
            ownership,
            use_latent=self.config.uses_latent_features,
        )

    def predict_bound(
        self,
        histories: NDArray[np.float64],
        future_controls: NDArray[np.float64],
        future_actions: NDArray[np.float64],
        raw_prediction: NDArray[np.float64],
        graph: DeclaredTelemetryGraph,
    ) -> NDArray[np.float64]:
        """Return the calibrated non-negative horizon certificate."""

        return np.asarray(
            self.predict_unadjusted(
                histories,
                future_controls,
                future_actions,
                raw_prediction,
                graph,
            )
            + self.calibration_adjustment,
            dtype=np.float64,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe selected and calibrated certificate."""

        (
            graph,
            features,
            controls,
            actions,
            ownership,
            _,
            network,
        ) = self._selected_values()
        adjustment = self.calibration_adjustment
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "config": asdict(self.config),
            "graph": graph.to_dict(),
            "feature_names": list(features),
            "control_names": list(controls),
            "action_names": list(actions),
            "ownership_mask": ownership.astype(int).tolist(),
            "baseline_sha256": self._baseline_sha256,
            "state_dict": _state_dict_to_payload(network.state_dict()),
            "selected_step": self._selected_step,
            "calibration_adjustment": adjustment,
            "calibration_control_count": (
                self._calibration_control_count
            ),
            "training_metrics": [
                dict(row) for row in self._training_metrics
            ],
            "selection_metrics": [
                dict(row) for row in self._selection_metrics
            ],
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "ErrorCertificateJepa":
        """Restore one selected and calibrated certificate."""

        if (
            payload.get("schema_version") != cls.schema_version
            or payload.get("kind") != cls.kind
        ):
            raise ValueError("error certificate schema is invalid")
        config = ErrorCertificateJepaConfig(**dict(payload["config"]))
        graph = DeclaredTelemetryGraph.from_dict(dict(payload["graph"]))
        features = tuple(str(value) for value in payload["feature_names"])
        controls = tuple(str(value) for value in payload["control_names"])
        actions = tuple(str(value) for value in payload["action_names"])
        ownership = np.asarray(payload["ownership_mask"], dtype=np.bool_)
        baseline_hash = str(payload["baseline_sha256"])
        adjustment = float(payload["calibration_adjustment"])
        calibration_count = payload["calibration_control_count"]
        selected = payload["selected_step"]
        if (
            ownership.shape != (len(graph.entities), len(features))
            or not np.any(ownership)
            or len(baseline_hash) != 64
            or not np.isfinite(adjustment)
            or adjustment < 0.0
            or isinstance(calibration_count, bool)
            or not isinstance(calibration_count, int)
            or calibration_count < 1
            or isinstance(selected, bool)
            or not isinstance(selected, int)
            or selected < 1
        ):
            raise ValueError("error certificate fitted schema is invalid")
        torch = _require_torch()
        network = _build_network(
            torch,
            config=config,
            entity_count=len(graph.entities),
            feature_count=len(features),
            condition_dimension=(
                len(controls) + len(graph.entities) * len(actions)
            ),
            horizon=10,
        )
        network.load_state_dict(
            _state_dict_from_payload(torch, dict(payload["state_dict"])),
            strict=True,
        )
        for parameter in network.parameters():
            parameter.requires_grad_(False)
        result = cls(config)
        result._graph = graph
        result._feature_names = features
        result._control_names = controls
        result._action_names = actions
        result._ownership_mask = ownership
        result._baseline_sha256 = baseline_hash
        result._network = network.eval()
        result._selected_step = selected
        result._calibration_adjustment = adjustment
        result._calibration_control_count = calibration_count
        result._training_metrics = _metric_rows(
            payload.get("training_metrics", ())
        )
        result._selection_metrics = _metric_rows(
            payload.get("selection_metrics", ())
        )
        return result

    def _ownership_mask_value(self) -> NDArray[np.bool_]:
        *_, ownership, _, _ = self._fitted_values()
        return np.asarray(ownership, dtype=np.bool_)

    def _fitted_values(self) -> Tuple[Any, ...]:
        if (
            self._graph is None
            or not self._feature_names
            or not self._control_names
            or not self._action_names
            or self._ownership_mask is None
            or not self._baseline_sha256
            or self._network is None
        ):
            raise ValueError("error certificate is not fitted")
        return (
            self._graph,
            self._feature_names,
            self._control_names,
            self._action_names,
            self._ownership_mask,
            self._baseline_sha256,
            self._network,
        )

    def _selected_values(self) -> Tuple[Any, ...]:
        values = self._fitted_values()
        if self._selected_step is None:
            raise ValueError("error certificate is not selected")
        return values


class CertifiedRawDynamics:
    """Return immutable raw forecasts accompanied by one certificate."""

    kind = "certified_raw_dynamics"
    schema_version = 1

    def __init__(
        self,
        baseline: ContractiveLowRankDynamics,
        certificate: ErrorCertificateJepa,
    ) -> None:
        if _artifact_sha256(baseline.to_dict()) != (
            certificate.baseline_sha256
        ):
            raise ValueError("certified raw baseline identity differs")
        certificate.calibration_adjustment
        self.baseline = baseline
        self.certificate = certificate

    @property
    def parameter_count(self) -> int:
        return (
            self.baseline.parameter_count
            + self.certificate.inference_parameter_count
        )

    def forecast_with_certificate(
        self,
        histories: NDArray[Any],
        future_controls: NDArray[Any],
        future_actions: NDArray[Any],
        graph: DeclaredTelemetryGraph,
    ) -> CertifiedForecast:
        """Return exact raw distribution plus calibrated bound."""

        raw = self.baseline.rollout(
            histories, future_controls, future_actions, graph
        )
        bound = self.certificate.predict_bound(
            np.asarray(histories, dtype=np.float64),
            np.asarray(future_controls, dtype=np.float64),
            np.asarray(future_actions, dtype=np.float64),
            raw.mean,
            graph,
        )
        return CertifiedForecast(distribution=raw, error_bound=bound)

    def rollout(
        self,
        histories: NDArray[Any],
        future_controls: NDArray[Any],
        future_actions: NDArray[Any],
        graph: DeclaredTelemetryGraph,
    ) -> TrajectoryDistribution:
        """Return only the unchanged raw distribution."""

        return self.baseline.rollout(
            histories, future_controls, future_actions, graph
        )

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "baseline": self.baseline.to_dict(),
            "certificate": self.certificate.to_dict(),
            "parameter_count": self.parameter_count,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "CertifiedRawDynamics":
        if (
            payload.get("schema_version") != cls.schema_version
            or payload.get("kind") != cls.kind
            or not isinstance(payload.get("baseline"), Mapping)
            or not isinstance(payload.get("certificate"), Mapping)
        ):
            raise ValueError("certified raw artifact is invalid")
        result = cls(
            ContractiveLowRankDynamics.from_dict(
                dict(payload["baseline"])
            ),
            ErrorCertificateJepa.from_dict(
                dict(payload["certificate"])
            ),
        )
        if payload.get("parameter_count") != result.parameter_count:
            raise ValueError("certified raw parameter count differs")
        return result


def realized_raw_error(
    raw_prediction: NDArray[np.float64],
    observed: NDArray[np.float64],
    ownership: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Return owned-coordinate RMSE per window and horizon."""

    prediction = np.asarray(raw_prediction, dtype=np.float64)
    target = np.asarray(observed, dtype=np.float64)
    if (
        prediction.shape != target.shape
        or prediction.ndim != 4
        or ownership.shape != prediction.shape[2:]
        or not np.any(ownership)
    ):
        raise ValueError("raw error target inputs are invalid")
    return np.asarray(
        np.sqrt(
            np.mean(np.square(prediction - target)[..., ownership], axis=2)
        ),
        dtype=np.float64,
    )


def pinball_loss(
    prediction: NDArray[np.float64],
    target: NDArray[np.float64],
    *,
    quantile: float,
) -> float:
    """Return mean quantile pinball loss."""

    error = np.asarray(target) - np.asarray(prediction)
    return float(
        np.mean(np.maximum(quantile * error, (quantile - 1.0) * error))
    )


def _build_network(
    torch: Any,
    *,
    config: ErrorCertificateJepaConfig,
    entity_count: int,
    feature_count: int,
    condition_dimension: int,
    horizon: int,
) -> Any:
    nn = torch.nn

    class EntityEncoder(nn.Module):  # type: ignore[misc,name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.input = nn.Linear(feature_count, config.hidden_width)
            self.entity_embedding = nn.Embedding(
                entity_count, config.hidden_width
            )
            self.hidden = nn.Linear(
                config.hidden_width, config.hidden_width
            )
            self.output = nn.Linear(config.hidden_width, config.width)

        def forward(self, values: Any) -> Any:
            hidden = self.input(values)
            shape = [1] * (hidden.ndim - 2) + [
                entity_count,
                config.hidden_width,
            ]
            hidden = hidden + self.entity_embedding.weight.reshape(shape)
            hidden = torch.nn.functional.silu(hidden)
            hidden = hidden + torch.nn.functional.silu(self.hidden(hidden))
            return self.output(hidden)

    class Network(nn.Module):  # type: ignore[misc,name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.online_encoder = EntityEncoder()
            self.target_encoder = copy.deepcopy(self.online_encoder)
            for parameter in self.target_encoder.parameters():
                parameter.requires_grad_(False)
            self.horizon_embedding = nn.Parameter(
                torch.empty(horizon, config.width)
            )
            nn.init.normal_(self.horizon_embedding, std=0.02)
            predictor_input = (
                entity_count * config.width
                + condition_dimension
                + config.width
            )
            self.predictor = nn.Sequential(
                nn.Linear(predictor_input, config.hidden_width),
                nn.SiLU(),
                nn.Linear(config.hidden_width, config.hidden_width),
                nn.SiLU(),
                nn.Linear(
                    config.hidden_width, entity_count * config.width
                ),
            )
            head_input = (
                2 * entity_count * feature_count
                + condition_dimension
                + config.width
                + entity_count * config.width
            )
            self.certificate_head = nn.Sequential(
                nn.Linear(head_input, config.hidden_width),
                nn.SiLU(),
                nn.Linear(config.hidden_width, config.hidden_width),
                nn.SiLU(),
                nn.Linear(config.hidden_width, 1),
                nn.Softplus(),
            )

        def predict_latent(self, current: Any, condition: Any) -> Any:
            batch = len(current)
            flat = current.flatten(1)
            repeated = flat[:, None].expand(
                batch, horizon, flat.shape[1]
            )
            horizon_values = self.horizon_embedding[None].expand(
                batch, horizon, config.width
            )
            values = torch.cat(
                (repeated, condition, horizon_values), dim=-1
            )
            return self.predictor(values).reshape(
                batch, horizon, entity_count, config.width
            )

        def certificate(
            self,
            current_raw: Any,
            raw_prediction: Any,
            condition: Any,
            latent: Any,
        ) -> Any:
            batch = len(current_raw)
            current = current_raw.flatten(1)[:, None].expand(
                batch, horizon, entity_count * feature_count
            )
            values = torch.cat(
                (
                    current,
                    raw_prediction.flatten(2),
                    condition,
                    self.horizon_embedding[None].expand(
                        batch, horizon, config.width
                    ),
                    latent.flatten(2),
                ),
                dim=-1,
            )
            return self.certificate_head(values).squeeze(-1)

    return Network()


def _objective_loss(
    torch: Any,
    network: Any,
    batch: Mapping[str, NDArray[np.float64]],
    *,
    config: ErrorCertificateJepaConfig,
) -> Mapping[str, Any]:
    histories = torch.as_tensor(
        batch["histories"], dtype=torch.float32
    )
    future = torch.as_tensor(batch["future"], dtype=torch.float32)
    latent_future = torch.as_tensor(
        batch["latent_future"], dtype=torch.float32
    )
    raw = torch.as_tensor(batch["raw"], dtype=torch.float32)
    condition = torch.as_tensor(
        batch["condition"], dtype=torch.float32
    )
    target_error = torch.as_tensor(
        batch["error_target"], dtype=torch.float32
    )
    current = network.online_encoder(histories[:, -1])
    predicted_latent = network.predict_latent(current, condition)
    head_latent = (
        predicted_latent
        if config.uses_latent_features
        else torch.zeros_like(predicted_latent)
    )
    bound = network.certificate(
        histories[:, -1], raw, condition, head_latent
    )
    error = target_error - bound
    pinball = torch.maximum(
        config.quantile * error, (config.quantile - 1.0) * error
    ).mean()
    with torch.no_grad():
        target_latent = network.target_encoder(latent_future)
    latent = torch.nn.functional.l1_loss(
        predicted_latent, target_latent
    )
    return {
        "total": pinball + config.effective_latent_weight * latent,
        "pinball": pinball,
        "latent": latent,
    }


def _training_arrays(
    windows: ActionConditionedWindows,
    raw_prediction: NDArray[np.float64],
    error_target: NDArray[np.float64],
    *,
    indices: NDArray[np.int64],
    donor: NDArray[np.int64],
    ownership: NDArray[np.bool_],
    derange: bool,
) -> Mapping[str, NDArray[np.float64]]:
    future_indices = donor if derange else indices
    return {
        "histories": np.asarray(
            windows.histories[indices] * ownership[None, None],
            dtype=np.float64,
        ),
        "future": np.asarray(
            windows.future_states[indices] * ownership[None, None],
            dtype=np.float64,
        ),
        "latent_future": np.asarray(
            windows.future_states[future_indices]
            * ownership[None, None],
            dtype=np.float64,
        ),
        "raw": np.asarray(
            raw_prediction[indices] * ownership[None, None],
            dtype=np.float64,
        ),
        "condition": _condition(
            windows.future_controls[indices],
            windows.future_actions[indices],
        ),
        "error_target": np.asarray(
            error_target[indices], dtype=np.float64
        ),
    }


def _predict_batches(
    torch: Any,
    network: Any,
    histories: NDArray[np.float64],
    controls: NDArray[np.float64],
    actions: NDArray[np.float64],
    raw_prediction: NDArray[np.float64],
    ownership: NDArray[np.bool_],
    *,
    use_latent: bool,
) -> NDArray[np.float64]:
    parts = []
    network.eval()
    with torch.no_grad():
        for start in range(0, len(histories), 256):
            stop = start + 256
            history = torch.as_tensor(
                histories[start:stop] * ownership[None, None],
                dtype=torch.float32,
            )
            raw = torch.as_tensor(
                raw_prediction[start:stop] * ownership[None, None],
                dtype=torch.float32,
            )
            condition = torch.as_tensor(
                _condition(controls[start:stop], actions[start:stop]),
                dtype=torch.float32,
            )
            current = network.online_encoder(history[:, -1])
            latent = network.predict_latent(current, condition)
            if not use_latent:
                latent = torch.zeros_like(latent)
            bound = network.certificate(
                history[:, -1], raw, condition, latent
            )
            parts.append(bound.cpu().numpy())
    return np.asarray(np.concatenate(parts), dtype=np.float64)


def _condition(
    controls: NDArray[np.float64],
    actions: NDArray[np.float64],
) -> NDArray[np.float64]:
    return np.asarray(
        np.concatenate(
            (
                controls,
                actions.reshape(len(actions), actions.shape[1], -1),
            ),
            axis=2,
        ),
        dtype=np.float64,
    )


def _control_trajectory_maxima(
    values: NDArray[np.float64],
    windows: ActionConditionedWindows,
) -> NDArray[np.float64]:
    rows = []
    trajectory_array = np.asarray(windows.trajectory_ids)
    for trajectory in sorted(set(windows.trajectory_ids)):
        positions = np.flatnonzero(trajectory_array == trajectory)
        if np.any(windows.future_actions[positions, ..., 1] > 0.5):
            continue
        rows.append(float(np.max(values[positions])))
    if len(rows) < 2:
        raise ValueError(
            "error certificate calibration needs control trajectories"
        )
    return np.asarray(rows, dtype=np.float64)


def _validate_public_inputs(
    histories: NDArray[np.float64],
    controls: NDArray[np.float64],
    actions: NDArray[np.float64],
    raw_prediction: NDArray[np.float64],
    graph: DeclaredTelemetryGraph,
    *,
    fitted_graph: DeclaredTelemetryGraph,
    feature_count: int,
    control_count: int,
    action_count: int,
) -> NDArray[np.float64]:
    history = np.asarray(histories, dtype=np.float64)
    control = np.asarray(controls, dtype=np.float64)
    action = np.asarray(actions, dtype=np.float64)
    raw = np.asarray(raw_prediction, dtype=np.float64)
    horizon = control.shape[1] if control.ndim == 3 else -1
    if (
        graph.to_dict() != fitted_graph.to_dict()
        or history.shape[1:]
        != (20, len(fitted_graph.entities), feature_count)
        or control.shape != (len(history), horizon, control_count)
        or action.shape
        != (
            len(history),
            horizon,
            len(fitted_graph.entities),
            action_count,
        )
        or raw.shape
        != (
            len(history),
            horizon,
            len(fitted_graph.entities),
            feature_count,
        )
        or not all(
            np.all(np.isfinite(value))
            for value in (history, control, action, raw)
        )
    ):
        raise ValueError("error certificate public inputs are invalid")
    return history


def _validate_windows(
    windows: ActionConditionedWindows, expected_pair_count: int
) -> None:
    if (
        len(set(windows.matched_pair_ids)) != expected_pair_count
        or windows.histories.shape[1] != 20
        or windows.future_states.shape[1] != 10
        or windows.action_feature_names[0] != "no_action"
        or "applicable" not in windows.action_feature_names
    ):
        raise ValueError("error certificate fitting windows are invalid")


def _validate_schema(
    windows: ActionConditionedWindows,
    *,
    graph: DeclaredTelemetryGraph,
    features: Tuple[str, ...],
    controls: Tuple[str, ...],
    actions: Tuple[str, ...],
) -> None:
    if (
        windows.graph.to_dict() != graph.to_dict()
        or windows.state_feature_names != features
        or windows.control_feature_names != controls
        or windows.action_feature_names != actions
        or windows.histories.shape[1] != 20
        or windows.future_states.shape[1] != 10
    ):
        raise ValueError("error certificate selection schema differs")


def _update_target_encoder(network: Any, *, decay: float) -> None:
    for target, online in zip(
        network.target_encoder.parameters(),
        network.online_encoder.parameters(),
    ):
        target.data.mul_(decay).add_(online.data, alpha=1.0 - decay)


def _artifact_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


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
            raise ValueError("error certificate state shape differs")
        if array.dtype.kind not in ("i", "u", "b") and not np.all(
            np.isfinite(array)
        ):
            raise ValueError("error certificate state tensor is non-finite")
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
            raise ValueError("error certificate metric row is non-finite")
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
            "Error-Certificate-JEPA fitting requires training dependencies"
        ) from error
