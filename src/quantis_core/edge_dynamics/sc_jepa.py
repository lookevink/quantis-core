"""Edge-sized clean-room SC-JEPA interaction models.

The module implements the four capacity-matched cells frozen in
``docs/specs/sc-jepa-interaction-v1.md``. Future state is used only as a
self-supervised fitting target. Public inference accepts current histories
and the declared telemetry graph.
"""

import copy
import importlib
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple, cast

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


SC_JEPA_CELL_NAMES = (
    "continuous_single",
    "continuous_multi",
    "codebook_single",
    "codebook_multi",
)
SC_JEPA_ASSESSMENT_MODEL_NAMES = (
    *SC_JEPA_CELL_NAMES,
    "raw_low_rank",
)
SC_JEPA_ASSESSMENT_ROLE_NAMES = (
    "calibration",
    "evaluation_iid",
    "evaluation_transfer",
)


@dataclass(frozen=True)
class ScJepaConfig:
    """Frozen controls for one SC-JEPA factorial cell."""

    use_codebook: bool = True
    multi_resolution: bool = True
    width: int = 32
    code_count: int = 32
    patch_count: int = 5
    patch_length: int = 2
    encoder_blocks: int = 2
    predictor_blocks: int = 2
    head_count: int = 4
    feedforward_width: int = 64
    alert_hidden_width: int = 64
    pretrain_steps: int = 300
    alert_steps: int = 200
    checkpoint_interval: int = 50
    alert_checkpoint_interval: int = 25
    batch_size: int = 128
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    ema_decay: float = 0.996
    gradient_clip_norm: float = 0.5
    code_temperature: float = 0.1
    prediction_temperature: float = 0.8
    fine_prediction_weight: float = 1.0
    fine_latent_weight: float = 0.1
    global_prediction_weight: float = 0.5
    prototype_weight: float = 1.0
    commitment_weight: float = 0.25
    sample_entropy_weight: float = 0.005
    batch_entropy_weight: float = 0.01
    reconstruction_start_weight: float = 0.5
    reconstruction_end_weight: float = 0.1
    expected_pair_count: int = 40
    seed: int = 13013
    device: str = "cpu"

    def __post_init__(self) -> None:
        if (
            self.width < 4
            or self.code_count != self.width
            or self.patch_count != 5
            or self.patch_length != 2
            or self.encoder_blocks < 1
            or self.predictor_blocks < 1
            or self.head_count < 1
            or self.width % self.head_count != 0
            or self.feedforward_width < self.width
            or self.alert_hidden_width < 1
            or self.pretrain_steps < 1
            or self.alert_steps < 1
            or self.checkpoint_interval < 1
            or self.alert_checkpoint_interval < 1
            or self.batch_size < 2
            or not 0.0 < self.learning_rate
            or not 0.0 <= self.weight_decay
            or not 0.0 < self.ema_decay < 1.0
            or self.gradient_clip_norm <= 0.0
            or self.code_temperature <= 0.0
            or self.prediction_temperature <= 0.0
            or self.expected_pair_count < 2
            or isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.device != "cpu"
        ):
            raise ValueError("SC-JEPA configuration is invalid")


@dataclass(frozen=True)
class ScJepaViews:
    """Raw fine context/future and coarse-future patch views."""

    context_fine: NDArray[np.float64]
    future_fine: NDArray[np.float64]
    future_coarse: NDArray[np.float64]


@dataclass(frozen=True)
class ScEncodedTelemetry:
    """Public entity-preserving SC-JEPA representation."""

    tokens: NDArray[np.float64]
    patch_values: NDArray[np.float64]
    entity_ids: Tuple[str, ...]
    ownership_mask: NDArray[np.bool_]
    code_probabilities: Optional[NDArray[np.float64]]

    def __post_init__(self) -> None:
        if (
            self.tokens.ndim != 3
            or self.patch_values.ndim != 4
            or self.tokens.shape[:2] != self.patch_values.shape[:2]
            or self.tokens.shape[-1] != self.patch_values.shape[-1]
            or self.tokens.shape[1] != len(self.entity_ids)
            or self.ownership_mask.shape[0] != len(self.entity_ids)
            or not np.all(np.isfinite(self.tokens))
            or not np.all(np.isfinite(self.patch_values))
        ):
            raise ValueError("SC-JEPA encoded telemetry is invalid")
        if (
            self.code_probabilities is not None
            and (
                self.code_probabilities.shape
                != self.patch_values.shape
                or not np.all(np.isfinite(self.code_probabilities))
                or np.any(self.code_probabilities < 0.0)
                or not np.allclose(
                    np.sum(self.code_probabilities, axis=-1),
                    1.0,
                    atol=1e-5,
                    rtol=0.0,
                )
            )
        ):
            raise ValueError("SC-JEPA code probabilities do not align")


def sc_jepa_views(
    histories: NDArray[np.float64],
    future_states: NDArray[np.float64],
) -> ScJepaViews:
    """Return the frozen ten-step, five-patch raw view construction."""

    context = np.asarray(histories, dtype=np.float64)
    future = np.asarray(future_states, dtype=np.float64)
    if (
        context.ndim != 4
        or future.ndim != 4
        or context.shape[0] != future.shape[0]
        or context.shape[2:] != future.shape[2:]
        or context.shape[1] < 10
        or future.shape[1] != 10
        or not np.all(np.isfinite(context))
        or not np.all(np.isfinite(future))
    ):
        raise ValueError("SC-JEPA view inputs are invalid")
    batch, _, entities, features = context.shape
    context_fine = (
        context[:, -10:]
        .transpose(0, 2, 1, 3)
        .reshape(batch, entities, 5, 2, features)
    )
    future_fine = (
        future.transpose(0, 2, 1, 3).reshape(
            batch, entities, 5, 2, features
        )
    )
    future_coarse = (
        future.reshape(batch, 2, 5, entities, features)
        .mean(axis=2)
        .transpose(0, 2, 1, 3)[:, :, None]
    )
    return ScJepaViews(
        context_fine=np.asarray(context_fine, dtype=np.float64),
        future_fine=np.asarray(future_fine, dtype=np.float64),
        future_coarse=np.asarray(future_coarse, dtype=np.float64),
    )


class ScJepaModel:
    """Restorable edge SC-JEPA representation and alert-policy adapter."""

    kind = "sc_jepa_interaction_cell"
    schema_version = 1

    def __init__(self, config: ScJepaConfig = ScJepaConfig()) -> None:
        self.config = config
        self._graph: Optional[DeclaredTelemetryGraph] = None
        self._feature_names: Tuple[str, ...] = ()
        self._ownership_mask: Optional[NDArray[np.bool_]] = None
        self._network: Any = None
        self._pretrain_checkpoints: Tuple[
            Tuple[int, Mapping[str, Any]], ...
        ] = ()
        self._alert_checkpoints: Tuple[
            Tuple[int, Mapping[str, Any]], ...
        ] = ()
        self._training_metrics: Tuple[Mapping[str, float], ...] = ()
        self._selection_metrics: Tuple[Mapping[str, float], ...] = ()
        self._alert_training_metrics: Tuple[
            Mapping[str, float], ...
        ] = ()
        self._alert_selection_metrics: Tuple[
            Mapping[str, float], ...
        ] = ()
        self._selected_pretrain_step: Optional[int] = None
        self._selected_alert_step: Optional[int] = None
        self._calibration: Optional[Mapping[str, float]] = None

    @property
    def cell_name(self) -> str:
        """Return the frozen factorial cell name."""

        return (
            ("codebook" if self.config.use_codebook else "continuous")
            + "_"
            + ("multi" if self.config.multi_resolution else "single")
        )

    @property
    def training_parameter_count(self) -> int:
        """Return online trainable architecture capacity."""

        _, _, _, network = self._fitted_values()
        return int(
            sum(
                parameter.numel()
                for parameter in network.online_parameters()
            )
        )

    @property
    def inference_parameter_count(self) -> int:
        """Return deployed encoder, bottleneck, and risk-head capacity."""

        _, _, _, network = self._fitted_values()
        return int(
            sum(
                parameter.numel()
                for parameter in network.inference_parameters()
            )
        )

    @property
    def calibration(self) -> Optional[Mapping[str, float]]:
        """Return the fitted monotone calibration and alert threshold."""

        return (
            None
            if self._calibration is None
            else dict(self._calibration)
        )

    @property
    def training_metrics(self) -> Tuple[Mapping[str, float], ...]:
        self._fitted_values()
        return self._training_metrics

    @property
    def selection_metrics(self) -> Tuple[Mapping[str, float], ...]:
        self._fitted_values()
        return self._selection_metrics

    @property
    def alert_selection_metrics(
        self,
    ) -> Tuple[Mapping[str, float], ...]:
        self._fitted_values()
        return self._alert_selection_metrics

    def fit(self, windows: ActionConditionedWindows) -> "ScJepaModel":
        """Fit self-supervised checkpoint candidates on fitting windows."""

        torch = _require_torch()
        _seed_torch(torch, self.config.seed)
        if (
            len(set(windows.matched_pair_ids))
            != self.config.expected_pair_count
            or windows.histories.shape[1] != 20
            or windows.future_states.shape[1] != 10
            or len(windows.entity_names) != 7
        ):
            raise ValueError("SC-JEPA fitting windows differ from contract")
        ownership = _fit_owned_feature_mask(windows)
        network = _build_network(
            torch,
            config=self.config,
            entity_count=len(windows.entity_names),
            feature_count=len(windows.state_feature_names),
            ownership_mask=ownership,
        )
        optimizer = torch.optim.AdamW(
            network.pretraining_parameters(),
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
                len(windows.histories),
                size=min(self.config.batch_size, len(windows.histories)),
            )
            optimizer.zero_grad(set_to_none=True)
            losses = network.pretraining_loss(
                torch.as_tensor(
                    windows.histories[indices], dtype=torch.float32
                ),
                torch.as_tensor(
                    windows.future_states[indices],
                    dtype=torch.float32,
                ),
                progress=(step + 1) / self.config.pretrain_steps,
            )
            total = losses["total"]
            if not bool(torch.isfinite(total)):
                raise RuntimeError("SC-JEPA pretraining became non-finite")
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                network.pretraining_parameters(),
                self.config.gradient_clip_norm,
            )
            optimizer.step()
            network.update_targets()
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
        self._network = network.eval()
        self._pretrain_checkpoints = tuple(checkpoints)
        self._training_metrics = tuple(metrics)
        return self

    def select(
        self, selection_windows: ActionConditionedWindows
    ) -> "ScJepaModel":
        """Choose one self-supervised checkpoint on selection only."""

        _, feature_names, _, network = self._fitted_values()
        if (
            not self._pretrain_checkpoints
            or selection_windows.graph.to_dict()
            != self._graph.to_dict()  # type: ignore[union-attr]
            or selection_windows.state_feature_names != feature_names
        ):
            raise ValueError("SC-JEPA selection inputs are invalid")
        scores = []
        best_score = float("inf")
        best_step = -1
        best_state = None
        for step, state in self._pretrain_checkpoints:
            network.load_state_dict(state)
            score = _pretraining_score(
                network,
                selection_windows,
                batch_size=max(32, self.config.batch_size),
            )
            scores.append({"step": float(step), "loss": score})
            if score < best_score - 1e-12:
                best_score = score
                best_step = step
                best_state = state
        if best_state is None:
            raise RuntimeError("SC-JEPA selected no pretraining checkpoint")
        network.load_state_dict(best_state)
        network.eval()
        self._selection_metrics = tuple(scores)
        self._selected_pretrain_step = best_step
        self._pretrain_checkpoints = ()
        return self

    def fit_alert_head(
        self,
        windows: ActionConditionedWindows,
        event_definition: HepaEventDefinition,
    ) -> "ScJepaModel":
        """Fit alert-head checkpoint candidates with the encoder frozen."""

        torch = _require_torch()
        _, _, _, network = self._selected_values()
        labels = event_definition.labels(windows)[:, -1]
        positives = int(np.sum(labels))
        negatives = int(len(labels) - positives)
        if positives < 1 or negatives < 1:
            raise ValueError("SC-JEPA alert fitting needs both classes")
        features = self._risk_features(
            windows.histories, windows.graph
        )
        for parameter in network.risk_head.parameters():
            parameter.requires_grad_(True)
        optimizer = torch.optim.AdamW(
            network.risk_head.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        generator = np.random.default_rng(self.config.seed + 2)
        positive_weight = negatives / float(positives)
        checkpoints = []
        metrics = []
        network.risk_head.train()
        for step in range(self.config.alert_steps):
            indices = generator.integers(
                0,
                len(features),
                size=min(self.config.batch_size, len(features)),
            )
            x = torch.as_tensor(features[indices], dtype=torch.float32)
            truth = torch.as_tensor(
                labels[indices], dtype=torch.float32
            )
            optimizer.zero_grad(set_to_none=True)
            logits = network.risk_head(x).squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                truth,
                pos_weight=torch.as_tensor(
                    positive_weight, dtype=torch.float32
                ),
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("SC-JEPA alert head became non-finite")
            loss.backward()
            optimizer.step()
            metrics.append(
                {
                    "step": float(step + 1),
                    "loss": float(loss.detach()),
                    "positive_weight": positive_weight,
                }
            )
            if (
                (step + 1)
                % self.config.alert_checkpoint_interval
                == 0
                or step + 1 == self.config.alert_steps
            ):
                checkpoints.append(
                    (
                        step + 1,
                        copy.deepcopy(network.risk_head.state_dict()),
                    )
                )
        for parameter in network.risk_head.parameters():
            parameter.requires_grad_(False)
        network.eval()
        self._alert_training_metrics = tuple(metrics)
        self._alert_checkpoints = tuple(checkpoints)
        return self

    def select_alert_head(
        self,
        windows: ActionConditionedWindows,
        event_definition: HepaEventDefinition,
    ) -> "ScJepaModel":
        """Choose one alert-head checkpoint using selection Brier."""

        torch = _require_torch()
        _, _, _, network = self._selected_values()
        if not self._alert_checkpoints:
            raise ValueError("SC-JEPA has no alert checkpoints")
        labels = event_definition.labels(windows)[:, -1]
        features = torch.as_tensor(
            self._risk_features(windows.histories, windows.graph),
            dtype=torch.float32,
        )
        scores = []
        best_score = float("inf")
        best_step = -1
        best_state = None
        for step, state in self._alert_checkpoints:
            network.risk_head.load_state_dict(state)
            with torch.no_grad():
                probabilities = torch.sigmoid(
                    network.risk_head(features).squeeze(-1)
                ).cpu().numpy()
            score = float(
                np.mean(
                    np.square(
                        probabilities - labels.astype(np.float64)
                    )
                )
            )
            scores.append({"step": float(step), "brier": score})
            if score < best_score - 1e-12:
                best_score = score
                best_step = step
                best_state = state
        if best_state is None:
            raise RuntimeError("SC-JEPA selected no alert checkpoint")
        network.risk_head.load_state_dict(best_state)
        network.eval()
        self._alert_selection_metrics = tuple(scores)
        self._selected_alert_step = best_step
        self._alert_checkpoints = ()
        return self

    def fit_calibration(
        self,
        windows: ActionConditionedWindows,
        event_definition: HepaEventDefinition,
    ) -> "ScJepaModel":
        """Fit monotone risk calibration and a control-max threshold."""

        raw = self.predict_risk(windows.histories, windows.graph)
        labels = event_definition.labels(windows)[:, -1]
        slope, intercept, brier = fit_logit_calibrator(
            raw[:, None], labels[:, None]
        )
        calibrated = calibrate_probability_surface(
            raw[:, None], slope=slope, intercept=intercept
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
    ) -> ScEncodedTelemetry:
        """Encode current histories into entity-preserving patch states."""

        torch = _require_torch()
        _, _, ownership, network = self._selected_values()
        values = self._validate_histories(histories, graph)
        with torch.no_grad():
            patch_values, probabilities = network.representation(
                torch.as_tensor(values, dtype=torch.float32)
            )
        patches = np.asarray(
            patch_values.detach().cpu().numpy(), dtype=np.float64
        )
        raw_probabilities = (
            None
            if probabilities is None
            else np.asarray(
                probabilities.detach().cpu().numpy(),
                dtype=np.float64,
            )
        )
        return ScEncodedTelemetry(
            tokens=np.mean(patches, axis=2),
            patch_values=patches,
            entity_ids=graph.entity_ids,
            ownership_mask=ownership.copy(),
            code_probabilities=raw_probabilities,
        )

    def predict_risk(
        self,
        histories: NDArray[np.float64],
        graph: DeclaredTelemetryGraph,
    ) -> NDArray[np.float64]:
        """Return uncalibrated next-window event risk."""

        torch = _require_torch()
        _, _, _, network = self._alert_values()
        features = self._risk_features(histories, graph)
        with torch.no_grad():
            result = torch.sigmoid(
                network.risk_head(
                    torch.as_tensor(features, dtype=torch.float32)
                ).squeeze(-1)
            ).cpu().numpy()
        probabilities = np.asarray(result, dtype=np.float64)
        if (
            probabilities.shape != (len(features),)
            or not np.all(np.isfinite(probabilities))
            or np.any(probabilities < 0.0)
            or np.any(probabilities > 1.0)
        ):
            raise ValueError("SC-JEPA risk output is invalid")
        return probabilities

    def calibrated_risk(
        self,
        histories: NDArray[np.float64],
        graph: DeclaredTelemetryGraph,
    ) -> NDArray[np.float64]:
        """Return calibrated next-window event risk."""

        if self._calibration is None:
            raise ValueError("SC-JEPA calibration has not been fitted")
        raw = self.predict_risk(histories, graph)
        return calibrate_probability_surface(
            raw[:, None],
            slope=float(self._calibration["slope"]),
            intercept=float(self._calibration["intercept"]),
        )[:, 0]

    def alert_decisions(
        self,
        histories: NDArray[np.float64],
        graph: DeclaredTelemetryGraph,
    ) -> NDArray[np.bool_]:
        """Return the frozen public alert-policy decision."""

        if self._calibration is None:
            raise ValueError("SC-JEPA calibration has not been fitted")
        return self.calibrated_risk(histories, graph) > float(
            self._calibration["alert_threshold"]
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a deterministic restorable cell payload."""

        graph, features, ownership, network = self._alert_values()
        if self._calibration is None:
            raise ValueError("SC-JEPA model is not calibrated")
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "cell_name": self.cell_name,
            "config": asdict(self.config),
            "graph": graph.to_dict(),
            "feature_names": list(features),
            "ownership_mask": ownership.astype(int).tolist(),
            "training_metrics": [
                dict(value) for value in self._training_metrics
            ],
            "selection_metrics": [
                dict(value) for value in self._selection_metrics
            ],
            "alert_training_metrics": [
                dict(value) for value in self._alert_training_metrics
            ],
            "alert_selection_metrics": [
                dict(value) for value in self._alert_selection_metrics
            ],
            "selected_pretrain_step": self._selected_pretrain_step,
            "selected_alert_step": self._selected_alert_step,
            "calibration": dict(self._calibration),
            "state_dict": _state_dict_to_payload(network.state_dict()),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ScJepaModel":
        """Restore a fitted cell and validate its mechanism identity."""

        if (
            payload.get("schema_version") != cls.schema_version
            or payload.get("kind") != cls.kind
        ):
            raise ValueError("unsupported SC-JEPA model artifact")
        torch = _require_torch()
        config = ScJepaConfig(**dict(payload["config"]))
        model = cls(config)
        if payload.get("cell_name") != model.cell_name:
            raise ValueError("SC-JEPA cell identity differs")
        graph = DeclaredTelemetryGraph.from_dict(payload["graph"])
        features = tuple(str(value) for value in payload["feature_names"])
        ownership = np.asarray(
            payload["ownership_mask"], dtype=np.bool_
        )
        network = _build_network(
            torch,
            config=config,
            entity_count=len(graph.entities),
            feature_count=len(features),
            ownership_mask=ownership,
        )
        network.load_state_dict(
            _state_dict_from_payload(torch, payload["state_dict"]),
            strict=True,
        )
        model._graph = graph
        model._feature_names = features
        model._ownership_mask = ownership
        model._network = network.eval()
        model._training_metrics = _metric_rows(
            payload["training_metrics"]
        )
        model._selection_metrics = _metric_rows(
            payload["selection_metrics"]
        )
        model._alert_training_metrics = _metric_rows(
            payload["alert_training_metrics"]
        )
        model._alert_selection_metrics = _metric_rows(
            payload["alert_selection_metrics"]
        )
        model._selected_pretrain_step = int(
            payload["selected_pretrain_step"]
        )
        model._selected_alert_step = int(
            payload["selected_alert_step"]
        )
        model._calibration = {
            str(key): float(value)
            for key, value in dict(payload["calibration"]).items()
        }
        for parameter in network.parameters():
            parameter.requires_grad_(False)
        return model

    def _risk_features(
        self,
        histories: NDArray[np.float64],
        graph: DeclaredTelemetryGraph,
    ) -> NDArray[np.float64]:
        encoded = self.encode(histories, graph)
        patches = (
            encoded.code_probabilities
            if encoded.code_probabilities is not None
            else encoded.patch_values
        )
        return np.asarray(
            np.mean(patches, axis=1).reshape(len(patches), -1),
            dtype=np.float64,
        )

    def _validate_histories(
        self,
        histories: NDArray[np.float64],
        graph: DeclaredTelemetryGraph,
    ) -> NDArray[np.float64]:
        fitted_graph, features, _, _ = self._selected_values()
        values = np.asarray(histories, dtype=np.float64)
        if (
            graph.to_dict() != fitted_graph.to_dict()
            or values.ndim != 4
            or values.shape[1:] != (
                20,
                len(fitted_graph.entities),
                len(features),
            )
            or not np.all(np.isfinite(values))
        ):
            raise ValueError("SC-JEPA inference histories are invalid")
        return values

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
        ):
            raise ValueError("SC-JEPA model is not fitted")
        return (
            self._graph,
            self._feature_names,
            self._ownership_mask,
            self._network,
        )

    def _selected_values(
        self,
    ) -> Tuple[
        DeclaredTelemetryGraph,
        Tuple[str, ...],
        NDArray[np.bool_],
        Any,
    ]:
        values = self._fitted_values()
        if self._selected_pretrain_step is None:
            raise ValueError("SC-JEPA pretraining is not selected")
        return values

    def _alert_values(
        self,
    ) -> Tuple[
        DeclaredTelemetryGraph,
        Tuple[str, ...],
        NDArray[np.bool_],
        Any,
    ]:
        values = self._selected_values()
        if self._selected_alert_step is None:
            raise ValueError("SC-JEPA alert head is not selected")
        return values


def assess_sc_jepa_interaction(
    *,
    risks: Mapping[str, Mapping[str, NDArray[np.float64]]],
    restored_risks: Mapping[
        str, Mapping[str, NDArray[np.float64]]
    ],
    stored_calibrated_risks: Mapping[
        str, Mapping[str, NDArray[np.float64]]
    ],
    restored_calibrated_risks: Mapping[
        str, Mapping[str, NDArray[np.float64]]
    ],
    stored_alert_decisions: Mapping[
        str, Mapping[str, NDArray[np.bool_]]
    ],
    restored_alert_decisions: Mapping[
        str, Mapping[str, NDArray[np.bool_]]
    ],
    stored_calibrations: Mapping[str, Mapping[str, float]],
    restored_calibrations: Mapping[str, Mapping[str, float]],
    labels: Mapping[str, NDArray[np.bool_]],
    trajectory_ids: Mapping[str, Tuple[str, ...]],
    transition_indices: Mapping[str, NDArray[np.int64]],
    trajectory_onsets: Mapping[str, Mapping[str, Optional[int]]],
    representation_tokens: Mapping[str, NDArray[np.float64]],
    restored_representation_tokens: Mapping[
        str, NDArray[np.float64]
    ],
    representation_patch_values: Mapping[
        str, NDArray[np.float64]
    ],
    restored_representation_patch_values: Mapping[
        str, NDArray[np.float64]
    ],
    representation_code_probabilities: Mapping[
        str, Optional[NDArray[np.float64]]
    ],
    restored_representation_code_probabilities: Mapping[
        str, Optional[NDArray[np.float64]]
    ],
    state_truth: NDArray[np.float64],
    state_scale: NDArray[np.float64],
    state_varying_mask: NDArray[np.bool_],
    state_predictions: Mapping[str, NDArray[np.float64]],
    training_parameter_counts: Mapping[str, int],
    inference_parameter_counts: Mapping[str, int],
    protocol_checks: Mapping[str, bool],
    edge_metrics: Mapping[str, Mapping[str, float]],
) -> Mapping[str, Any]:
    """Recompute the complete factorial decision from stored raw evidence."""

    metrics: Dict[str, Dict[str, Mapping[str, float]]] = {
        role: {} for role in SC_JEPA_ASSESSMENT_ROLE_NAMES
    }
    alert_metrics: Dict[str, Dict[str, Mapping[str, Any]]] = {
        role: {}
        for role in ("evaluation_iid", "evaluation_transfer")
    }
    calibrations: Dict[str, Mapping[str, float]] = {}
    restoration_checks: List[bool] = []
    calibration_ids = trajectory_ids["calibration"]
    calibration_controls = tuple(
        trajectory_id
        for trajectory_id, onset in trajectory_onsets[
            "calibration"
        ].items()
        if onset is None
    )
    for role in SC_JEPA_ASSESSMENT_ROLE_NAMES:
        count = len(trajectory_ids[role])
        if (
            labels[role].shape != (count,)
            or transition_indices[role].shape != (count,)
        ):
            raise ValueError("SC-JEPA assessment role arrays differ")
        for model in SC_JEPA_ASSESSMENT_MODEL_NAMES:
            for values in (
                risks,
                restored_risks,
                stored_calibrated_risks,
                restored_calibrated_risks,
            ):
                if values[role][model].shape != (count,):
                    raise ValueError("SC-JEPA risk shape differs")
            if (
                stored_alert_decisions[role][model].shape
                != (count,)
                or restored_alert_decisions[role][model].shape
                != (count,)
            ):
                raise ValueError("SC-JEPA decision shape differs")
    for model in SC_JEPA_ASSESSMENT_MODEL_NAMES:
        slope, intercept, calibration_brier = fit_logit_calibrator(
            risks["calibration"][model][:, None],
            labels["calibration"][:, None],
        )
        recomputed = {}
        restored_recomputed = {}
        for role in SC_JEPA_ASSESSMENT_ROLE_NAMES:
            recomputed[role] = calibrate_probability_surface(
                risks[role][model][:, None],
                slope=slope,
                intercept=intercept,
            )[:, 0]
            restored_recomputed[role] = (
                calibrate_probability_surface(
                    restored_risks[role][model][:, None],
                    slope=slope,
                    intercept=intercept,
                )[:, 0]
            )
            metrics[role][model] = _binary_probability_metrics(
                recomputed[role], labels[role]
            )
            restoration_checks.extend(
                (
                    np.allclose(
                        risks[role][model],
                        restored_risks[role][model],
                        atol=1e-6,
                        rtol=0.0,
                    ),
                    np.allclose(
                        recomputed[role],
                        stored_calibrated_risks[role][model],
                        atol=1e-6,
                        rtol=0.0,
                    ),
                    np.allclose(
                        restored_recomputed[role],
                        restored_calibrated_risks[role][model],
                        atol=1e-6,
                        rtol=0.0,
                    ),
                )
            )
        threshold = trajectory_alert_threshold(
            recomputed["calibration"][:, None],
            calibration_ids,
            calibration_controls,
        )
        expected_calibration = {
            "slope": slope,
            "intercept": intercept,
            "calibration_brier": calibration_brier,
            "alert_threshold": threshold,
        }
        for stored in (
            stored_calibrations[model],
            restored_calibrations[model],
        ):
            restoration_checks.append(
                set(stored) == set(expected_calibration)
                and all(
                    abs(float(stored[key]) - float(value)) <= 1e-6
                    for key, value in expected_calibration.items()
                )
            )
        calibrations[model] = expected_calibration
        for role in SC_JEPA_ASSESSMENT_ROLE_NAMES:
            decisions = recomputed[role] > float(
                stored_calibrations[model]["alert_threshold"]
            )
            restored_decisions = restored_recomputed[role] > float(
                restored_calibrations[model]["alert_threshold"]
            )
            restoration_checks.extend(
                (
                    np.array_equal(
                        decisions,
                        stored_alert_decisions[role][model],
                    ),
                    np.array_equal(
                        restored_decisions,
                        restored_alert_decisions[role][model],
                    ),
                    np.array_equal(decisions, restored_decisions),
                )
            )
            if role in alert_metrics:
                alert_metrics[role][model] = _trajectory_alert_metrics(
                    decisions=decisions,
                    trajectory_ids=trajectory_ids[role],
                    transition_indices=transition_indices[role],
                    onsets=trajectory_onsets[role],
                )
    representation_checks: List[Any] = []
    for name in SC_JEPA_CELL_NAMES:
        token_values = representation_tokens[name]
        restored_token_values = restored_representation_tokens[name]
        patch_values = representation_patch_values[name]
        restored_patch_values = restored_representation_patch_values[
            name
        ]
        codes = representation_code_probabilities[name]
        restored_codes = restored_representation_code_probabilities[
            name
        ]
        expects_codes = name.startswith("codebook_")
        representation_checks.extend(
            (
                np.all(np.isfinite(token_values)),
                np.all(np.isfinite(restored_token_values)),
                np.all(np.isfinite(patch_values)),
                np.all(np.isfinite(restored_patch_values)),
                np.allclose(
                    token_values,
                    restored_token_values,
                    atol=1e-6,
                    rtol=0.0,
                ),
                np.allclose(
                    patch_values,
                    restored_patch_values,
                    atol=1e-6,
                    rtol=0.0,
                ),
                (codes is not None) == expects_codes,
                (restored_codes is not None) == expects_codes,
            )
        )
        if codes is not None and restored_codes is not None:
            representation_checks.extend(
                (
                    np.all(np.isfinite(codes)),
                    np.all(np.isfinite(restored_codes)),
                    np.all(codes >= 0.0),
                    np.all(restored_codes >= 0.0),
                    np.allclose(
                        np.sum(codes, axis=-1),
                        1.0,
                        atol=1e-5,
                        rtol=0.0,
                    ),
                    np.allclose(
                        np.sum(restored_codes, axis=-1),
                        1.0,
                        atol=1e-5,
                        rtol=0.0,
                    ),
                    np.allclose(
                        codes,
                        restored_codes,
                        atol=1e-6,
                        rtol=0.0,
                    ),
                )
            )
    candidate_code_probabilities = (
        representation_code_probabilities["codebook_multi"]
    )
    candidate_tokens = representation_tokens["codebook_multi"]
    if candidate_code_probabilities is None:
        raise ValueError("SC-JEPA candidate code probabilities are missing")
    code_usage = _code_usage_metrics(
        candidate_code_probabilities,
        observed_entities=np.any(state_varying_mask, axis=1),
    )
    spectra = _representation_spectrum(candidate_tokens)
    state = _state_retention_metrics(
        truth=state_truth,
        scale=state_scale,
        varying=state_varying_mask,
        predictions=state_predictions,
    )
    transfer_brier = {
        model: float(
            metrics["evaluation_transfer"][model]["brier"]
        )
        for model in SC_JEPA_ASSESSMENT_MODEL_NAMES
    }
    transfer_detection = {
        model: float(
            alert_metrics["evaluation_transfer"][model][
                "treatment_detection_rate"
            ]
        )
        for model in SC_JEPA_ASSESSMENT_MODEL_NAMES
    }
    brier_interaction = (
        transfer_brier["continuous_multi"]
        + transfer_brier["codebook_single"]
        - transfer_brier["continuous_single"]
        - transfer_brier["codebook_multi"]
    )
    detection_interaction = (
        transfer_detection["codebook_multi"]
        - transfer_detection["continuous_multi"]
        - transfer_detection["codebook_single"]
        + transfer_detection["continuous_single"]
    )
    candidate_alert = alert_metrics["evaluation_transfer"][
        "codebook_multi"
    ]
    candidate_edge = edge_metrics["codebook_multi"]
    capacity_match = (
        len(
            {
                int(training_parameter_counts[name])
                for name in SC_JEPA_CELL_NAMES
            }
        )
        == 1
        and len(
            {
                int(inference_parameter_counts[name])
                for name in SC_JEPA_CELL_NAMES
            }
        )
        == 1
    )
    safety_gates = {
        "finite_restored_public_outputs": (
            all(
                np.all(np.isfinite(values))
                for roles in (
                    risks,
                    restored_risks,
                    stored_calibrated_risks,
                    restored_calibrated_risks,
                )
                for models in roles.values()
                for values in models.values()
            )
            and bool(all(restoration_checks))
            and bool(all(representation_checks))
        ),
        "factorial_capacity_matches": capacity_match,
        "codebook_usage_is_noncollapsed": (
            int(code_usage["active_code_count"]) >= 8
            and float(code_usage["marginal_perplexity"]) >= 8.0
            and bool(code_usage["every_entity_uses_multiple_codes"])
        ),
        "state_retention_within_1_05_pca": (
            float(state["codebook_multi"]["aggregate_nrmse"])
            <= 1.05
            * float(state["matched_pca"]["aggregate_nrmse"])
            and bool(state["all_varying_entities_reported"])
        ),
        "edge_budget_and_diagnostics": (
            float(
                candidate_edge[
                    "serialized_candidate_sidecars_bytes"
                ]
            )
            <= 16.0 * 1024.0 * 1024.0
            and np.isfinite(
                float(candidate_edge["batch_one_cpu_latency_ms"])
            )
            and np.isfinite(
                float(candidate_edge["batch_one_cpu_p95_latency_ms"])
            )
            and float(candidate_edge["peak_rss_bytes"]) > 0.0
        ),
        "stored_protocol_is_derived_and_passes": bool(
            all(bool(value) for value in protocol_checks.values())
        ),
    }
    best_reference_brier = min(
        transfer_brier[name]
        for name in (
            "continuous_single",
            "continuous_multi",
            "codebook_single",
            "raw_low_rank",
        )
    )
    predictive_lane_gates = {
        "brier_interaction_at_least_5_percent_cs": (
            brier_interaction
            >= 0.05 * transfer_brier["continuous_single"]
        ),
        "full_brier_beats_all_references_by_5_percent": (
            transfer_brier["codebook_multi"]
            <= 0.95 * best_reference_brier
        ),
    }
    reference_detection = max(
        transfer_detection[name]
        for name in (
            "continuous_multi",
            "codebook_single",
            "raw_low_rank",
        )
    )
    delay = candidate_alert["median_post_onset_delay_transitions"]
    alert_lane_gates = {
        "control_false_alarms_at_most_5_percent": (
            float(
                candidate_alert[
                    "control_trajectory_false_alarm_rate"
                ]
            )
            <= 0.05
        ),
        "treatment_detection_at_least_80_percent": (
            transfer_detection["codebook_multi"] >= 0.80
        ),
        "median_post_onset_delay_at_most_10": (
            delay is not None and float(delay) <= 10.0
        ),
        "detection_interaction_at_least_10_points": (
            detection_interaction >= 0.10 - 1e-12
        ),
        "full_detection_beats_references_by_10_points": (
            transfer_detection["codebook_multi"]
            - reference_detection
            >= 0.10 - 1e-12
        ),
    }
    predictive_passed = bool(all(predictive_lane_gates.values()))
    alert_passed = bool(all(alert_lane_gates.values()))
    eligible_for_advance = bool(
        protocol_checks.get("frozen_interpretable_contract", False)
    )
    passed = bool(
        all(safety_gates.values())
        and (predictive_passed or alert_passed)
        and eligible_for_advance
    )
    return {
        "schema_version": 1,
        "kind": "sc_jepa_interaction_assessment",
        "calibrations": calibrations,
        "risk_metrics": metrics,
        "alert_metrics": alert_metrics,
        "code_usage": code_usage,
        "representation_spectrum": spectra,
        "state_retention": state,
        "training_parameter_counts": {
            name: int(value)
            for name, value in training_parameter_counts.items()
        },
        "inference_parameter_counts": {
            name: int(value)
            for name, value in inference_parameter_counts.items()
        },
        "edge_metrics": {
            name: {
                key: float(value)
                for key, value in values.items()
            }
            for name, values in edge_metrics.items()
        },
        "interactions": {
            "held_transfer_brier": brier_interaction,
            "held_transfer_detection": detection_interaction,
        },
        "protocol_checks": {
            name: bool(value) for name, value in protocol_checks.items()
        },
        "safety_gates": safety_gates,
        "predictive_lane_gates": predictive_lane_gates,
        "alert_lane_gates": alert_lane_gates,
        "predictive_lane_passed": predictive_passed,
        "alert_lane_passed": alert_passed,
        "eligible_for_advance": eligible_for_advance,
        "passed": passed,
        "decision": (
            "advance_sc_jepa_interaction_to_fixed_seed_robustness"
            if passed
            else (
                "non_interpretable_sc_jepa_smoke"
                if not eligible_for_advance
                else "reject_sc_jepa_interaction_recipe"
            )
        ),
    }


def _binary_probability_metrics(
    probabilities: NDArray[np.float64],
    labels: NDArray[np.bool_],
) -> Mapping[str, float]:
    values = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.bool_)
    if (
        values.shape != truth.shape
        or values.ndim != 1
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise ValueError("SC-JEPA probability metric inputs are invalid")
    brier = float(
        np.mean(np.square(values - truth.astype(np.float64)))
    )
    bins = np.minimum((values * 10.0).astype(np.int64), 9)
    ece = 0.0
    for index in range(10):
        selected = bins == index
        if np.any(selected):
            ece += float(np.mean(selected)) * abs(
                float(np.mean(values[selected]))
                - float(np.mean(truth[selected]))
            )
    return {
        "brier": brier,
        "ece_10_equal_width_bins": ece,
        "positive_rate": float(np.mean(truth)),
    }


def _trajectory_alert_metrics(
    *,
    decisions: NDArray[np.bool_],
    trajectory_ids: Tuple[str, ...],
    transition_indices: NDArray[np.int64],
    onsets: Mapping[str, Optional[int]],
) -> Mapping[str, Any]:
    rows = []
    control_alerts = []
    treatment_detections = []
    treatment_pre_onset = []
    delays = []
    ids = np.asarray(trajectory_ids, dtype=str)
    for trajectory_id in sorted(set(trajectory_ids)):
        positions = np.flatnonzero(ids == trajectory_id)
        order = positions[
            np.argsort(transition_indices[positions], kind="stable")
        ]
        local_decisions = decisions[order]
        local_transitions = transition_indices[order]
        onset = onsets[trajectory_id]
        if onset is None:
            any_alert = bool(np.any(local_decisions))
            alert_count = int(np.sum(local_decisions))
            control_alerts.append(any_alert)
            rows.append(
                {
                    "trajectory_id": trajectory_id,
                    "is_treatment": False,
                    "onset_transition": None,
                    "any_alert": any_alert,
                    "alert_count": alert_count,
                    "pre_onset_alert": False,
                    "first_post_onset_alert_transition": None,
                    "post_onset_delay_transitions": None,
                }
            )
            continue
        before = local_transitions < int(onset)
        after = local_transitions >= int(onset)
        pre_alert = bool(np.any(local_decisions & before))
        post_positions = np.flatnonzero(local_decisions & after)
        detected = bool(len(post_positions))
        delay = (
            None
            if not detected
            else int(
                local_transitions[int(post_positions[0])] - int(onset)
            )
        )
        treatment_pre_onset.append(pre_alert)
        treatment_detections.append(detected)
        if delay is not None:
            delays.append(delay)
        rows.append(
            {
                "trajectory_id": trajectory_id,
                "is_treatment": True,
                "onset_transition": int(onset),
                "any_alert": bool(np.any(local_decisions)),
                "alert_count": int(np.sum(local_decisions)),
                "pre_onset_alert": pre_alert,
                "first_post_onset_alert_transition": (
                    None
                    if delay is None
                    else int(onset) + delay
                ),
                "post_onset_delay_transitions": delay,
            }
        )
    return {
        "control_trajectory_count": len(control_alerts),
        "treatment_trajectory_count": len(treatment_detections),
        "control_trajectory_false_alarm_rate": (
            float(np.mean(control_alerts)) if control_alerts else 0.0
        ),
        "treatment_detection_rate": (
            float(np.mean(treatment_detections))
            if treatment_detections
            else 0.0
        ),
        "treatment_pre_onset_alert_rate": (
            float(np.mean(treatment_pre_onset))
            if treatment_pre_onset
            else 0.0
        ),
        "total_alert_count": int(np.sum(decisions)),
        "alerts_per_logical_run": float(
            np.sum(decisions) / max(1, len(set(trajectory_ids)))
        ),
        "median_post_onset_delay_transitions": (
            None if not delays else float(np.median(delays))
        ),
        "worst_post_onset_delay_transitions": (
            None if not delays else int(max(delays))
        ),
        "trajectory_rows": rows,
    }


def _code_usage_metrics(
    probabilities: NDArray[np.float64],
    *,
    observed_entities: NDArray[np.bool_],
) -> Mapping[str, Any]:
    values = np.asarray(probabilities, dtype=np.float64)
    if (
        values.ndim != 4
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or not np.allclose(
            np.sum(values, axis=-1), 1.0, atol=1e-5
        )
    ):
        raise ValueError("SC-JEPA code probabilities are invalid")
    observed = np.asarray(observed_entities, dtype=np.bool_)
    if observed.shape != (values.shape[1],) or not np.any(observed):
        raise ValueError("SC-JEPA observed entity mask is invalid")
    marginal = np.mean(values, axis=(0, 1, 2))
    safe_marginal = np.clip(marginal, 1e-12, 1.0)
    entropy = -float(
        np.sum(safe_marginal * np.log(safe_marginal))
    )
    dominant = np.argmax(values, axis=-1)
    per_entity_counts = {
        str(entity): int(len(np.unique(dominant[:, entity])))
        for entity in range(values.shape[1])
    }
    return {
        "active_code_count": int(np.sum(marginal > 0.005)),
        "dead_code_count": int(np.sum(marginal <= 0.005)),
        "marginal_perplexity": float(np.exp(entropy)),
        "mean_sample_entropy": float(
            np.mean(
                -np.sum(
                    np.clip(values, 1e-12, 1.0)
                    * np.log(np.clip(values, 1e-12, 1.0)),
                    axis=-1,
                )
            )
        ),
        "mean_max_probability": float(
            np.mean(np.max(values, axis=-1))
        ),
        "dominant_code_counts_by_entity": per_entity_counts,
        "every_entity_uses_multiple_codes": bool(
            all(
                per_entity_counts[str(entity)] > 1
                for entity in np.flatnonzero(observed)
            )
        ),
        "marginal": marginal.tolist(),
    }


def _representation_spectrum(
    tokens: NDArray[np.float64],
) -> Mapping[str, Any]:
    values = np.asarray(tokens, dtype=np.float64).reshape(
        -1, tokens.shape[-1]
    )
    centered = values - np.mean(values, axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    variance = np.square(singular)
    total = float(np.sum(variance))
    ratios = (
        np.zeros_like(variance)
        if total <= 1e-12
        else variance / total
    )
    positive = ratios[ratios > 1e-12]
    effective_rank = float(
        np.exp(-np.sum(positive * np.log(positive)))
    )
    return {
        "top_1_variance_ratio": float(np.sum(ratios[:1])),
        "top_5_variance_ratio": float(np.sum(ratios[:5])),
        "top_10_variance_ratio": float(np.sum(ratios[:10])),
        "effective_rank": effective_rank,
        "minimum_dimension_variance": float(
            np.min(np.var(values, axis=0))
        ),
    }


def _state_retention_metrics(
    *,
    truth: NDArray[np.float64],
    scale: NDArray[np.float64],
    varying: NDArray[np.bool_],
    predictions: Mapping[str, NDArray[np.float64]],
) -> Mapping[str, Any]:
    actual = np.asarray(truth, dtype=np.float64)
    denominator = np.asarray(scale, dtype=np.float64)
    mask = np.asarray(varying, dtype=np.bool_)
    result: Dict[str, Any] = {}
    for name, prediction in predictions.items():
        predicted = np.asarray(prediction, dtype=np.float64)
        if predicted.shape != actual.shape:
            raise ValueError("SC-JEPA state prediction shape differs")
        rmse = np.sqrt(np.mean(np.square(predicted - actual), axis=0))
        nrmse = np.divide(
            rmse,
            denominator,
            out=np.full_like(rmse, np.nan),
            where=mask,
        )
        per_entity = {}
        for entity in range(len(nrmse)):
            local = nrmse[entity][mask[entity]]
            per_entity[str(entity)] = (
                None if not len(local) else float(np.mean(local))
            )
        result[name] = {
            "aggregate_nrmse": float(np.mean(nrmse[mask])),
            "per_entity_nrmse": per_entity,
        }
    result["all_varying_entities_reported"] = bool(
        all(
            result[name]["per_entity_nrmse"][str(entity)]
            is not None
            for name in predictions
            for entity in range(mask.shape[0])
            if np.any(mask[entity])
        )
    )
    result["varying_entity_count"] = int(
        np.sum(np.any(mask, axis=1))
    )
    return result


def _build_network(
    torch: Any,
    *,
    config: ScJepaConfig,
    entity_count: int,
    feature_count: int,
    ownership_mask: NDArray[np.bool_],
) -> Any:
    nn = torch.nn

    class PatchEncoder(nn.Module):  # type: ignore[misc,name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.patch_projection = nn.Linear(
                config.patch_length * feature_count, config.width
            )
            self.entity_embedding = nn.Embedding(
                entity_count, config.width
            )
            self.position = nn.Parameter(
                torch.zeros(1, config.patch_count, config.width)
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
                layer, num_layers=config.encoder_blocks
            )
            self.norm = nn.LayerNorm(config.width)
            self.register_buffer(
                "ownership",
                torch.as_tensor(ownership_mask, dtype=torch.bool),
            )

        def forward(self, patches: Any) -> Any:
            batch, entities, patch_count, length, features = (
                patches.shape
            )
            if (
                entities != entity_count
                or length != config.patch_length
                or features != feature_count
                or patch_count > config.patch_count
            ):
                raise ValueError("SC-JEPA patch tensor is invalid")
            owned = torch.where(
                self.ownership[None, :, None, None],
                patches,
                torch.zeros_like(patches),
            )
            values = owned.reshape(
                batch * entities,
                patch_count,
                length * features,
            )
            entity_ids = torch.arange(
                entities, device=patches.device
            )
            entity_tokens = (
                self.entity_embedding(entity_ids)[None]
                .expand(batch, entities, config.width)
                .reshape(batch * entities, 1, config.width)
            )
            tokens = (
                self.patch_projection(values)
                + entity_tokens
                + self.position[:, :patch_count]
            )
            encoded = self.norm(self.transformer(tokens))
            return encoded.reshape(
                batch, entities, patch_count, config.width
            )

    class Bottleneck(nn.Module):  # type: ignore[misc,name-defined]
        def __init__(self) -> None:
            super().__init__()
            if config.use_codebook:
                self.matrix = nn.Parameter(
                    torch.empty(config.code_count, config.width)
                )
            else:
                self.matrix = nn.Parameter(
                    torch.empty(config.width, config.width)
                )
            nn.init.orthogonal_(self.matrix)

        def forward(self, features: Any) -> Tuple[Any, Optional[Any]]:
            if config.use_codebook:
                normalized_features = torch.nn.functional.normalize(
                    features, dim=-1
                )
                prototypes = torch.nn.functional.normalize(
                    self.matrix, dim=-1
                )
                logits = torch.einsum(
                    "...d,kd->...k",
                    normalized_features,
                    prototypes,
                )
                probabilities = torch.softmax(
                    logits / config.code_temperature, dim=-1
                )
                values = torch.einsum(
                    "...k,kd->...d", probabilities, prototypes
                )
                return values, probabilities
            return torch.matmul(features, self.matrix.T), None

    class FinePredictor(nn.Module):  # type: ignore[misc,name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.position = nn.Parameter(
                torch.zeros(1, config.patch_count, config.width)
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
                layer, num_layers=config.predictor_blocks
            )
            self.output = nn.Linear(config.width, config.width)
            self.latent = nn.Linear(config.width, config.width)

        def forward(self, values: Any) -> Tuple[Any, Any]:
            batch, entities, patches, width = values.shape
            encoded = self.transformer(
                values.reshape(batch * entities, patches, width)
                + self.position[:, :patches]
            )
            return (
                self.output(encoded).reshape(
                    batch, entities, patches, width
                ),
                self.latent(encoded).reshape(
                    batch, entities, patches, width
                ),
            )

    class GlobalPredictor(nn.Module):  # type: ignore[misc,name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.position = nn.Parameter(
                torch.zeros(1, config.patch_count, config.width)
            )
            self.query = nn.Parameter(
                torch.zeros(1, 1, config.width)
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
                layer, num_layers=config.predictor_blocks
            )
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=config.width,
                num_heads=config.head_count,
                dropout=0.0,
                batch_first=True,
            )
            self.norm = nn.LayerNorm(config.width)
            self.output = nn.Linear(config.width, config.width)

        def forward(self, values: Any) -> Any:
            batch, entities, patches, width = values.shape
            flattened = values.reshape(
                batch * entities, patches, width
            )
            encoded = self.transformer(
                flattened + self.position[:, :patches]
            )
            query = self.query.expand(batch * entities, -1, -1)
            pooled, _ = self.cross_attention(
                query, encoded, encoded, need_weights=False
            )
            return self.output(self.norm(pooled)).reshape(
                batch, entities, 1, width
            )

    class Network(nn.Module):  # type: ignore[misc,name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.online_encoder = PatchEncoder()
            self.online_bottleneck = Bottleneck()
            self.target_encoder = copy.deepcopy(self.online_encoder)
            self.target_bottleneck = copy.deepcopy(
                self.online_bottleneck
            )
            for parameter in self.target_encoder.parameters():
                parameter.requires_grad_(False)
            for parameter in self.target_bottleneck.parameters():
                parameter.requires_grad_(False)
            self.fine_predictor = FinePredictor()
            self.global_predictor = GlobalPredictor()
            self.decoder = nn.Linear(
                config.width, config.patch_length * feature_count
            )
            self.risk_head = nn.Sequential(
                nn.Linear(
                    config.patch_count * config.width,
                    config.alert_hidden_width,
                ),
                nn.GELU(),
                nn.Linear(config.alert_hidden_width, 1),
            )
            self.register_buffer(
                "ownership",
                torch.as_tensor(ownership_mask, dtype=torch.bool),
            )

        def online_parameters(self) -> Any:
            return (
                list(self.online_encoder.parameters())
                + list(self.online_bottleneck.parameters())
                + list(self.fine_predictor.parameters())
                + list(self.global_predictor.parameters())
                + list(self.decoder.parameters())
                + list(self.risk_head.parameters())
            )

        def pretraining_parameters(self) -> Any:
            return (
                list(self.online_encoder.parameters())
                + list(self.online_bottleneck.parameters())
                + list(self.fine_predictor.parameters())
                + list(self.global_predictor.parameters())
                + list(self.decoder.parameters())
            )

        def inference_parameters(self) -> Any:
            return (
                list(self.online_encoder.parameters())
                + list(self.online_bottleneck.parameters())
                + list(self.risk_head.parameters())
            )

        def update_targets(self) -> None:
            with torch.no_grad():
                for target, online in zip(
                    self.target_encoder.parameters(),
                    self.online_encoder.parameters(),
                ):
                    target.mul_(config.ema_decay).add_(
                        online, alpha=1.0 - config.ema_decay
                    )
                for target, online in zip(
                    self.target_bottleneck.parameters(),
                    self.online_bottleneck.parameters(),
                ):
                    target.mul_(config.ema_decay).add_(
                        online, alpha=1.0 - config.ema_decay
                    )

        def representation(
            self, histories: Any
        ) -> Tuple[Any, Optional[Any]]:
            context_raw = _context_patches(torch, histories)
            context_normalized, _, _ = self._normalize(context_raw)
            features = self.online_encoder(context_normalized)
            return cast(
                Tuple[Any, Optional[Any]],
                self.online_bottleneck(features),
            )

        def pretraining_loss(
            self,
            histories: Any,
            future_states: Any,
            *,
            progress: float,
        ) -> Mapping[str, Any]:
            context_raw = _context_patches(torch, histories)
            future_raw = _future_patches(torch, future_states)
            context_normalized, context_mean, context_scale = (
                self._normalize(context_raw)
            )
            future_normalized, _, _ = self._normalize(future_raw)
            online_features = self.online_encoder(context_normalized)
            online_values, online_probabilities = (
                self.online_bottleneck(online_features)
            )
            with torch.no_grad():
                target_features = self.target_encoder(
                    future_normalized
                )
                target_values, target_probabilities = (
                    self.target_bottleneck(target_features)
                )
                if config.multi_resolution:
                    coarse = _coarse_from_normalized(
                        future_normalized
                    )
                    global_features = self.target_encoder(coarse)
                    global_values, global_probabilities = (
                        self.target_bottleneck(global_features)
                    )
                else:
                    global_values = target_values.mean(
                        dim=2, keepdim=True
                    )
                    global_probabilities = (
                        None
                        if target_probabilities is None
                        else target_probabilities.mean(
                            dim=2, keepdim=True
                        )
                    )
            predictor_input = (
                online_probabilities
                if online_probabilities is not None
                else online_values
            )
            fine_output, fine_latent = self.fine_predictor(
                predictor_input
            )
            global_output = self.global_predictor(predictor_input)
            zero = fine_output.sum() * 0.0
            if config.use_codebook:
                if (
                    online_probabilities is None
                    or target_probabilities is None
                    or global_probabilities is None
                ):
                    raise RuntimeError("SC-JEPA code targets are missing")
                fine_prediction = _soft_target_kl(
                    torch,
                    fine_output,
                    target_probabilities,
                    temperature=config.prediction_temperature,
                )
                fine_latent_loss = torch.nn.functional.mse_loss(
                    fine_latent, target_values
                )
                global_prediction = _soft_target_kl(
                    torch,
                    global_output,
                    global_probabilities,
                    temperature=config.prediction_temperature,
                )
                prototype = torch.nn.functional.mse_loss(
                    online_values, online_features.detach()
                )
                commitment = torch.nn.functional.mse_loss(
                    online_features, online_values.detach()
                )
                safe = torch.clamp(
                    online_probabilities, 1e-7, 1.0
                )
                sample_entropy = -torch.mean(
                    torch.sum(safe * torch.log(safe), dim=-1)
                )
                marginal = safe.mean(dim=(0, 1, 2))
                batch_entropy = -torch.sum(
                    marginal * torch.log(marginal)
                )
                predictive = (
                    config.fine_prediction_weight * fine_prediction
                    + config.fine_latent_weight * fine_latent_loss
                    + config.global_prediction_weight
                    * global_prediction
                )
                code_loss = (
                    config.prototype_weight * prototype
                    + config.commitment_weight * commitment
                    + config.sample_entropy_weight * sample_entropy
                    - config.batch_entropy_weight * batch_entropy
                )
            else:
                fine_prediction = torch.nn.functional.mse_loss(
                    fine_output, target_values
                )
                fine_latent_loss = zero
                global_prediction = torch.nn.functional.mse_loss(
                    global_output, global_values
                )
                prototype = zero
                commitment = zero
                sample_entropy = zero
                batch_entropy = zero
                predictive = (
                    (
                        config.fine_prediction_weight
                        + config.fine_latent_weight
                    )
                    * fine_prediction
                    + config.global_prediction_weight
                    * global_prediction
                )
                code_loss = zero
            reconstructed_normalized = self.decoder(
                online_values
            ).reshape_as(context_raw)
            reconstructed = (
                reconstructed_normalized * context_scale
                + context_mean
            )
            mask = self.ownership[
                None, :, None, None
            ].to(context_raw.dtype)
            squared = torch.square(
                (reconstructed - context_raw) * mask
            )
            denominator = (
                float(len(context_raw))
                * config.patch_count
                * config.patch_length
                * float(torch.sum(self.ownership))
            )
            reconstruction = torch.sum(squared) / denominator
            reconstruction_weight = (
                config.reconstruction_start_weight
                + (
                    config.reconstruction_end_weight
                    - config.reconstruction_start_weight
                )
                * float(progress)
            )
            total = (
                predictive
                + code_loss
                + reconstruction_weight * reconstruction
            )
            return {
                "total": total,
                "fine_prediction": fine_prediction,
                "fine_latent": fine_latent_loss,
                "global_prediction": global_prediction,
                "prototype": prototype,
                "commitment": commitment,
                "sample_entropy": sample_entropy,
                "batch_entropy": batch_entropy,
                "reconstruction": reconstruction,
                "reconstruction_weight": torch.as_tensor(
                    reconstruction_weight,
                    dtype=total.dtype,
                    device=total.device,
                ),
            }

        def _normalize(self, patches: Any) -> Tuple[Any, Any, Any]:
            mask = self.ownership[
                None, :, None, None
            ].to(patches.dtype)
            mean = patches.mean(dim=(2, 3), keepdim=True) * mask
            variance = torch.mean(
                torch.square((patches - mean) * mask),
                dim=(2, 3),
                keepdim=True,
            )
            scale = torch.sqrt(variance + 1e-5)
            scale = torch.where(mask > 0.5, scale, torch.ones_like(scale))
            normalized = (patches - mean) / scale
            return normalized * mask, mean, scale

    return Network()


def _context_patches(torch: Any, histories: Any) -> Any:
    batch, _, entities, features = histories.shape
    return (
        histories[:, -10:]
        .permute(0, 2, 1, 3)
        .reshape(batch, entities, 5, 2, features)
    )


def _future_patches(torch: Any, future_states: Any) -> Any:
    batch, _, entities, features = future_states.shape
    return (
        future_states.permute(0, 2, 1, 3).reshape(
            batch, entities, 5, 2, features
        )
    )


def _coarse_from_normalized(future: Any) -> Any:
    batch, entities, _, _, features = future.shape
    return (
        future.reshape(batch, entities, 2, 5, features)
        .mean(dim=3)
        .unsqueeze(2)
    )


def _soft_target_kl(
    torch: Any,
    logits: Any,
    targets: Any,
    *,
    temperature: float,
) -> Any:
    log_probabilities = torch.nn.functional.log_softmax(
        logits / temperature, dim=-1
    )
    safe_targets = torch.clamp(targets, 1e-7, 1.0)
    return torch.mean(
        torch.sum(
            safe_targets
            * (torch.log(safe_targets) - log_probabilities),
            dim=-1,
        )
    )


def _pretraining_score(
    network: Any,
    windows: ActionConditionedWindows,
    *,
    batch_size: int,
) -> float:
    torch = _require_torch()
    values = []
    network.eval()
    with torch.no_grad():
        for start in range(0, len(windows.histories), batch_size):
            losses = network.pretraining_loss(
                torch.as_tensor(
                    windows.histories[start : start + batch_size],
                    dtype=torch.float32,
                ),
                torch.as_tensor(
                    windows.future_states[start : start + batch_size],
                    dtype=torch.float32,
                ),
                progress=1.0,
            )
            values.append(
                float(losses["total"])
                * len(
                    windows.histories[start : start + batch_size]
                )
            )
    return float(sum(values) / len(windows.histories))


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
        raise ValueError("SC-JEPA telemetry schema has no observations")
    return mask


def _control_trajectory_ids(
    windows: ActionConditionedWindows,
) -> Tuple[str, ...]:
    try:
        applicable = windows.action_feature_names.index("applicable")
    except ValueError as error:
        raise ValueError(
            "SC-JEPA event definition needs the applicable action field"
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
            raise ValueError("SC-JEPA state tensor shape differs")
        if array.dtype.kind not in ("i", "u", "b") and not np.all(
            np.isfinite(array)
        ):
            raise ValueError("SC-JEPA state tensor is non-finite")
        if array.dtype.kind in ("i", "u", "b"):
            tensor = torch.as_tensor(array)
        else:
            tensor = torch.as_tensor(array, dtype=torch.float32)
        result[str(name)] = tensor
    return result


def _metric_rows(values: Any) -> Tuple[Mapping[str, float], ...]:
    return tuple(
        {
            str(key): float(value)
            for key, value in dict(row).items()
        }
        for row in values
    )


def _seed_torch(torch: Any, seed: int) -> None:
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def _require_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError as error:
        raise RuntimeError(
            "SC-JEPA fitting requires the optional training dependencies"
        ) from error
