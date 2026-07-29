"""No-augmentation projected next-latent learning for telemetry."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from ..action_conditioned_dynamics import ActionConditionedWindows
from ..graph_telemetry import DeclaredTelemetryGraph
from .action_conditioned_jepa import (
    sketched_isotropic_gaussian_regularization,
)
from .complete_lejepa import (
    EncodedTelemetry,
    PairBlockedAnchorSchedule,
    fit_owned_feature_mask,
)


LENEPA_OBJECTIVES = (
    "projected_lenepa",
    "unprojected_lenepa",
    "projected_sigreg_only",
)


@dataclass(frozen=True)
class LenepaConfig:
    """Frozen architecture and optimizer controls for one LeNEPA cell."""

    objective: str = "projected_lenepa"
    width: int = 64
    depth: int = 8
    head_count: int = 4
    feedforward_width: int = 256
    projector_hidden_width: int = 1536
    projector_width: int = 64
    steps: int = 1600
    warmup_steps: int = 80
    expected_pair_count: int = 40
    learning_rate: float = 1e-4
    minimum_learning_rate: float = 1e-6
    initial_weight_decay: float = 1e-2
    final_weight_decay: float = 1e-1
    sigreg_weight: float = 20.0
    sketch_dimension: int = 256
    knot_count: int = 17
    seed: int = 24024
    anchor_seed: int = 24025
    sigreg_seed: int = 24026
    preprocessing_protocol: str = (
        "edge-dynamics-fit-only-robust-v1"
    )

    def __post_init__(self) -> None:
        integers = (
            self.width,
            self.depth,
            self.head_count,
            self.feedforward_width,
            self.projector_hidden_width,
            self.projector_width,
            self.steps,
            self.warmup_steps,
            self.expected_pair_count,
            self.sketch_dimension,
            self.knot_count,
        )
        if (
            self.objective not in LENEPA_OBJECTIVES
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
                for value in integers
            )
            or self.width % self.head_count
            or self.projector_width != self.width
            or self.warmup_steps > self.steps
            or self.knot_count < 2
            or not (
                0.0 < self.minimum_learning_rate <= self.learning_rate
            )
            or not (
                0.0
                < self.initial_weight_decay
                <= self.final_weight_decay
            )
            or self.sigreg_weight != 20.0
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in (
                    self.seed,
                    self.anchor_seed,
                    self.sigreg_seed,
                )
            )
            or not self.preprocessing_protocol
        ):
            raise ValueError("LeNEPA configuration is invalid")

    def to_dict(self) -> Dict[str, Any]:
        return dict(asdict(self))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LenepaConfig":
        if set(payload) != set(asdict(cls())):
            raise ValueError("LeNEPA config schema is invalid")
        return cls(**dict(payload))


@dataclass(frozen=True)
class LenepaLoss:
    """Literal LeNEPA scalar loss and reportable components."""

    loss: Any
    prediction_mse: Any
    temporal_sigreg: Any


def lenepa_objective(
    layer_zero: Any,
    layer_final: Any,
    *,
    objective: str,
    projector: Any,
    generator: Any,
    sketch_dimension: int = 256,
    knot_count: int = 17,
    sigreg_weight: float = 20.0,
) -> LenepaLoss:
    """Return projected/unprojected next-token MSE plus temporal SIGReg."""

    if (
        objective not in LENEPA_OBJECTIVES
        or layer_zero.ndim != 3
        or layer_final.shape != layer_zero.shape
        or layer_zero.size(1) < 2
        or sketch_dimension < 1
        or knot_count < 2
        or sigreg_weight != 20.0
    ):
        raise ValueError("LeNEPA objective inputs are invalid")
    if objective == "unprojected_lenepa":
        target_tokens = layer_zero
        predicted_tokens = layer_final
    else:
        target_tokens = _project_sequence(projector, layer_zero)
        predicted_tokens = _project_sequence(projector, layer_final)
    prediction_mse = (
        (
            predicted_tokens[:, :-1]
            - target_tokens[:, 1:]
        )
        .square()
        .mean()
    )
    temporal_sigreg = sketched_isotropic_gaussian_regularization(
        torch_cat((target_tokens, predicted_tokens), dim=0),
        generator=generator,
        sketch_dimension=sketch_dimension,
        knot_count=knot_count,
    )
    prediction_weight = (
        0.0 if objective == "projected_sigreg_only" else 1.0
    )
    return LenepaLoss(
        loss=(
            prediction_weight * prediction_mse
            + sigreg_weight * temporal_sigreg
        ),
        prediction_mse=prediction_mse,
        temporal_sigreg=temporal_sigreg,
    )


@dataclass(frozen=True)
class LenepaDiagnostic:
    """Next-latent tensors and scale-free alignment diagnostics."""

    input_tokens: NDArray[np.float64]
    output_tokens: NDArray[np.float64]
    predicted_tokens: NDArray[np.float64]
    target_tokens: NDArray[np.float64]
    cosine_error: float
    retrieval_hit_at_1: float
    input_sigreg: float
    output_sigreg: float

    def __post_init__(self) -> None:
        scalars = (
            self.cosine_error,
            self.retrieval_hit_at_1,
            self.input_sigreg,
            self.output_sigreg,
        )
        if (
            self.input_tokens.ndim != 3
            or self.output_tokens.shape != self.input_tokens.shape
            or self.predicted_tokens.ndim != 3
            or self.predicted_tokens.shape
            != self.output_tokens[:, :-1].shape
            or self.target_tokens.shape != self.predicted_tokens.shape
            or not np.all(np.isfinite(self.input_tokens))
            or not np.all(np.isfinite(self.output_tokens))
            or not np.all(np.isfinite(self.predicted_tokens))
            or not np.all(np.isfinite(self.target_tokens))
            or not all(np.isfinite(value) for value in scalars)
            or self.cosine_error < 0.0
            or not 0.0 <= self.retrieval_hit_at_1 <= 1.0
        ):
            raise ValueError("LeNEPA diagnostic is invalid")


class LenepaRepresentation:
    """Restorable causal LeNEPA telemetry representation."""

    kind = "lenepa_telemetry_representation"
    schema_version = 1

    def __init__(
        self, config: LenepaConfig = LenepaConfig()
    ) -> None:
        self.config = config
        self._graph: Optional[DeclaredTelemetryGraph] = None
        self._feature_names: Optional[Tuple[str, ...]] = None
        self._ownership_mask: Optional[NDArray[np.bool_]] = None
        self._network: Any = None
        self._projector: Any = None
        self._training_metrics: Tuple[Mapping[str, float], ...] = ()
        self._sigreg_generator_state: Tuple[int, ...] = ()

    @property
    def training_metrics(self) -> Tuple[Mapping[str, float], ...]:
        self._fitted_values()
        return tuple(dict(row) for row in self._training_metrics)

    @property
    def inference_parameter_count(self) -> int:
        *_, network, _ = self._fitted_values()
        return int(
            sum(parameter.numel() for parameter in network.parameters())
        )

    @property
    def training_parameter_count(self) -> int:
        *_, network, projector = self._fitted_values()
        return int(
            sum(
                parameter.numel()
                for module in (network, projector)
                for parameter in module.parameters()
            )
        )

    def fit(
        self, windows: ActionConditionedWindows
    ) -> "LenepaRepresentation":
        """Fit one no-augmentation causal objective on fitting histories."""

        if (
            len(set(windows.matched_pair_ids))
            != self.config.expected_pair_count
            or windows.histories.shape[1] != 20
        ):
            raise ValueError("LeNEPA fitting data differs from its contract")
        torch = _require_torch()
        ownership = fit_owned_feature_mask(windows)
        network, projector = _new_modules(
            windows.graph,
            windows.histories.shape[-1],
            self.config,
        )
        optimizer = torch.optim.AdamW(
            list(network.parameters()) + list(projector.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.initial_weight_decay,
        )
        anchors = PairBlockedAnchorSchedule(
            windows, seed=self.config.anchor_seed
        )
        generator = torch.Generator(device="cpu").manual_seed(
            self.config.sigreg_seed
        )
        metrics = []
        network.train()
        projector.train()
        for step in range(self.config.steps):
            _set_optimizer_controls(
                optimizer, self.config, step
            )
            anchor = anchors.batch(step)
            histories = _owned_histories(
                windows.histories[anchor.indices], ownership
            )
            optimizer.zero_grad(set_to_none=True)
            layer_zero, layer_final, _ = network(
                torch.as_tensor(histories, dtype=torch.float32)
            )
            losses = lenepa_objective(
                layer_zero,
                layer_final,
                objective=self.config.objective,
                projector=projector,
                generator=generator,
                sketch_dimension=self.config.sketch_dimension,
                knot_count=self.config.knot_count,
                sigreg_weight=self.config.sigreg_weight,
            )
            if not bool(torch.isfinite(losses.loss)):
                raise RuntimeError("LeNEPA training became non-finite")
            losses.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(network.parameters()) + list(projector.parameters()),
                1.0,
            )
            optimizer.step()
            metrics.append(
                {
                    "step": float(step + 1),
                    "loss": float(losses.loss.detach()),
                    "prediction_mse": float(
                        losses.prediction_mse.detach()
                    ),
                    "temporal_sigreg": float(
                        losses.temporal_sigreg.detach()
                    ),
                    "learning_rate": float(
                        optimizer.param_groups[0]["lr"]
                    ),
                    "weight_decay": float(
                        optimizer.param_groups[0]["weight_decay"]
                    ),
                    "independent_samples": float(len(histories)),
                }
            )
        self._graph = windows.graph
        self._feature_names = windows.state_feature_names
        self._ownership_mask = ownership.copy()
        self._network = network.eval()
        self._projector = projector.eval()
        self._training_metrics = tuple(metrics)
        self._sigreg_generator_state = tuple(
            int(value) for value in generator.get_state().tolist()
        )
        return self

    def encode(
        self,
        histories: NDArray[np.float64],
        graph: DeclaredTelemetryGraph,
    ) -> EncodedTelemetry:
        """Encode current histories without future or outcome inputs."""

        fitted_graph, feature_names, ownership, network = (
            self._encoder_values()
        )
        source = _validate_histories(
            histories, graph, fitted_graph, feature_names
        )
        torch = _require_torch()
        with torch.no_grad():
            _, layer_final, entity_contributions = network(
                torch.as_tensor(
                    _owned_histories(source, ownership),
                    dtype=torch.float32,
                )
            )
            tokens = network.entity_output(
                layer_final, entity_contributions
            )
        return EncodedTelemetry(
            tokens=tokens.cpu().numpy().astype(np.float64),
            entity_ids=fitted_graph.entity_ids,
            ownership_mask=ownership.copy(),
            observation_mask=ownership.copy(),
            content_sha256=_array_sha256(source),
            graph_sha256=_canonical_sha256(fitted_graph.to_dict()),
            state_schema_sha256=_canonical_sha256(
                {"state_feature_names": list(feature_names)}
            ),
            preprocessing_sha256=hashlib.sha256(
                self.config.preprocessing_protocol.encode()
            ).hexdigest(),
            encoder_sha256=_module_sha256(network),
        )

    def encode_sequence(
        self,
        histories: NDArray[np.float64],
        graph: DeclaredTelemetryGraph,
    ) -> NDArray[np.float64]:
        """Return final-layer time tokens for causal-prefix verification."""

        fitted_graph, feature_names, ownership, network = (
            self._encoder_values()
        )
        source = _validate_histories(
            histories, graph, fitted_graph, feature_names
        )
        torch = _require_torch()
        with torch.no_grad():
            _, layer_final, _ = network(
                torch.as_tensor(
                    _owned_histories(source, ownership),
                    dtype=torch.float32,
                )
            )
        result: NDArray[np.float64] = (
            layer_final.cpu().numpy().astype(np.float64)
        )
        return result

    def diagnose_next_latent(
        self,
        histories: NDArray[np.float64],
        graph: DeclaredTelemetryGraph,
    ) -> LenepaDiagnostic:
        """Return projected or backbone next-token alignment evidence."""

        (
            fitted_graph,
            feature_names,
            ownership,
            network,
            projector,
        ) = self._fitted_values()
        source = _validate_histories(
            histories, graph, fitted_graph, feature_names
        )
        torch = _require_torch()
        with torch.no_grad():
            layer_zero, layer_final, _ = network(
                torch.as_tensor(
                    _owned_histories(source, ownership),
                    dtype=torch.float32,
                )
            )
            if self.config.objective == "unprojected_lenepa":
                target_all = layer_zero
                predicted_all = layer_final
            else:
                target_all = _project_sequence(projector, layer_zero)
                predicted_all = _project_sequence(
                    projector, layer_final
                )
            predicted = predicted_all[:, :-1]
            target = target_all[:, 1:]
            cosine_error = _cosine_error(predicted, target)
            retrieval = _retrieval_hit_at_one(predicted, target)
            input_generator = torch.Generator(
                device="cpu"
            ).manual_seed(self.config.sigreg_seed + 90_000)
            output_generator = torch.Generator(
                device="cpu"
            ).manual_seed(self.config.sigreg_seed + 90_000)
            input_sigreg = sketched_isotropic_gaussian_regularization(
                target_all,
                generator=input_generator,
                sketch_dimension=self.config.sketch_dimension,
                knot_count=self.config.knot_count,
            )
            output_sigreg = sketched_isotropic_gaussian_regularization(
                predicted_all,
                generator=output_generator,
                sketch_dimension=self.config.sketch_dimension,
                knot_count=self.config.knot_count,
            )
        return LenepaDiagnostic(
            input_tokens=target_all.cpu().numpy().astype(np.float64),
            output_tokens=predicted_all.cpu().numpy().astype(np.float64),
            predicted_tokens=predicted.cpu().numpy().astype(np.float64),
            target_tokens=target.cpu().numpy().astype(np.float64),
            cosine_error=float(cosine_error),
            retrieval_hit_at_1=float(retrieval),
            input_sigreg=float(input_sigreg),
            output_sigreg=float(output_sigreg),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize training state and the deployable causal encoder."""

        (
            graph,
            feature_names,
            ownership,
            network,
            projector,
        ) = self._fitted_values()
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "config": self.config.to_dict(),
            "graph": graph.to_dict(),
            "feature_names": list(feature_names),
            "ownership_mask": ownership.astype(int).tolist(),
            "network_state": _module_state(network),
            "projector_state": _module_state(projector),
            "training_metrics": [
                dict(row) for row in self._training_metrics
            ],
            "sigreg_generator_state": list(
                self._sigreg_generator_state
            ),
            "inference_parameter_count": (
                self.inference_parameter_count
            ),
            "training_parameter_count": self.training_parameter_count,
        }

    def to_inference_dict(self) -> Dict[str, Any]:
        """Serialize only state required by the public causal encoder."""

        graph, feature_names, ownership, network = (
            self._encoder_values()
        )
        return {
            "schema_version": self.schema_version,
            "kind": "lenepa_student_inference_bundle",
            "config": self.config.to_dict(),
            "graph": graph.to_dict(),
            "feature_names": list(feature_names),
            "ownership_mask": ownership.astype(int).tolist(),
            "network_state": _module_state(network),
        }

    @classmethod
    def from_inference_dict(
        cls, payload: Mapping[str, Any]
    ) -> "LenepaRepresentation":
        """Restore the exact public causal encoder from a deploy bundle."""

        if (
            payload.get("schema_version") != cls.schema_version
            or payload.get("kind")
            != "lenepa_student_inference_bundle"
            or "projector_state" in payload
        ):
            raise ValueError("unsupported LeNEPA inference bundle")
        config = LenepaConfig.from_dict(dict(payload["config"]))
        graph = DeclaredTelemetryGraph.from_dict(dict(payload["graph"]))
        feature_names = tuple(
            str(value) for value in payload["feature_names"]
        )
        ownership = np.asarray(
            payload["ownership_mask"], dtype=np.bool_
        )
        network, _ = _new_modules(
            graph, len(feature_names), config
        )
        _restore_module(network, dict(payload["network_state"]))
        model = cls(config)
        model._graph = graph
        model._feature_names = feature_names
        model._ownership_mask = ownership
        model._network = network.eval()
        model._projector = None
        return model

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "LenepaRepresentation":
        """Restore one complete LeNEPA training cell."""

        if (
            payload.get("schema_version") != cls.schema_version
            or payload.get("kind") != cls.kind
        ):
            raise ValueError("unsupported LeNEPA artifact")
        config = LenepaConfig.from_dict(dict(payload["config"]))
        graph = DeclaredTelemetryGraph.from_dict(dict(payload["graph"]))
        feature_names = tuple(
            str(value) for value in payload["feature_names"]
        )
        ownership = np.asarray(
            payload["ownership_mask"], dtype=np.bool_
        )
        network, projector = _new_modules(
            graph, len(feature_names), config
        )
        _restore_module(network, dict(payload["network_state"]))
        _restore_module(projector, dict(payload["projector_state"]))
        model = cls(config)
        model._graph = graph
        model._feature_names = feature_names
        model._ownership_mask = ownership
        model._network = network.eval()
        model._projector = projector.eval()
        model._training_metrics = tuple(
            {
                str(key): float(value)
                for key, value in dict(row).items()
            }
            for row in payload["training_metrics"]
        )
        model._sigreg_generator_state = tuple(
            int(value)
            for value in payload["sigreg_generator_state"]
        )
        if (
            model.inference_parameter_count
            != int(payload["inference_parameter_count"])
            or model.training_parameter_count
            != int(payload["training_parameter_count"])
        ):
            raise ValueError("LeNEPA artifact capacity differs")
        return model

    def _fitted_values(
        self,
    ) -> Tuple[
        DeclaredTelemetryGraph,
        Tuple[str, ...],
        NDArray[np.bool_],
        Any,
        Any,
    ]:
        if (
            self._graph is None
            or self._feature_names is None
            or self._ownership_mask is None
            or self._network is None
            or self._projector is None
        ):
            raise RuntimeError("LeNEPA representation is not fitted")
        return (
            self._graph,
            self._feature_names,
            self._ownership_mask,
            self._network,
            self._projector,
        )

    def _encoder_values(
        self,
    ) -> Tuple[
        DeclaredTelemetryGraph,
        Tuple[str, ...],
        NDArray[np.bool_],
        Any,
    ]:
        if (
            self._graph is None
            or self._feature_names is None
            or self._ownership_mask is None
            or self._network is None
        ):
            raise RuntimeError("LeNEPA encoder is not fitted")
        return (
            self._graph,
            self._feature_names,
            self._ownership_mask,
            self._network,
        )


def assess_lenepa_gates(
    *,
    forecast_scores: Mapping[str, Mapping[str, Mapping[str, float]]],
    raw_scores: Mapping[str, Mapping[str, float]],
    mechanism: Mapping[str, Mapping[str, Mapping[str, float]]],
    attribution: Mapping[str, Mapping[str, float]],
    action_sanity: Mapping[str, Mapping[str, float]],
    restoration_max_abs: Mapping[str, float],
    protocol_checks: Mapping[str, bool],
    parameter_counts: Mapping[str, Mapping[str, int]],
    transfer_pair_errors: Mapping[str, Mapping[str, float]],
    deployed_bundle_bytes: int,
    median_latency_ms: float,
) -> Dict[str, Any]:
    """Purely recompute the frozen LeNEPA safety and value gates."""

    candidate = forecast_scores["projected_lenepa"]
    raw_selection = raw_scores["selection"]
    raw_transfer = raw_scores["transfer_evaluation"]
    counts = list(parameter_counts.values())
    capacity_matched = bool(counts) and all(
        value == counts[0] for value in counts[1:]
    )
    required_protocols = (
        "evidence_arrays_are_finite",
        "pair_and_trajectory_roles_are_disjoint",
        "capacity_recomputes",
        "public_inference_is_causal",
        "prefix_invariance_recomputes",
        "anchor_schedule_recomputes",
        "selection_only_ridge_choice_recomputes",
        "selection_safety_status_recomputes",
        "bundle_size_recomputes",
        "latency_recomputes",
        "mechanism_history_coverage_recomputes",
        "diagnostic_shift_consistency_recomputes",
    )
    safety_gates = {
        name: bool(protocol_checks.get(name, False))
        for name in required_protocols
    }
    safety_gates.update(
        {
            "capacity_is_matched": capacity_matched,
            "restoration_within_1e_6": all(
                np.isfinite(value) and value <= 1e-6
                for value in restoration_max_abs.values()
            ),
            "selection_overall_within_1_05_raw": (
                candidate["selection"]["overall_mse"]
                <= 1.05 * raw_selection["overall_mse"]
            ),
            "selection_action_within_1_05_raw": (
                candidate["selection"]["action_overlap_mse"]
                <= 1.05 * raw_selection["action_overlap_mse"]
            ),
            "transfer_overall_within_1_05_raw": (
                candidate["transfer_evaluation"]["overall_mse"]
                <= 1.05 * raw_transfer["overall_mse"]
            ),
            "transfer_action_within_1_05_raw": (
                candidate["transfer_evaluation"][
                    "action_overlap_mse"
                ]
                <= 1.05 * raw_transfer["action_overlap_mse"]
            ),
            "action_and_target_hit_at_1": (
                attribution["projected_lenepa"][
                    "action_and_target_hit_at_1"
                ]
                >= 0.95
            ),
            "no_action_specificity": (
                attribution["projected_lenepa"][
                    "no_action_specificity"
                ]
                == 1.0
            ),
            "correct_action_sanity": (
                action_sanity["projected_lenepa"][
                    "correct_action_beats_both_fraction"
                ]
                >= 0.80
            ),
            "deployed_bundle_within_16_mib": (
                isinstance(deployed_bundle_bytes, int)
                and not isinstance(deployed_bundle_bytes, bool)
                and 0 < deployed_bundle_bytes <= 16 * 1024 * 1024
            ),
            "latency_is_recorded": (
                np.isfinite(median_latency_ms)
                and median_latency_ms > 0.0
            ),
        }
    )
    mechanism_gates = {
        "projected_prediction_advantage": all(
            mechanism["projected_lenepa"][role]["cosine_error"]
            <= 0.90
            * mechanism["unprojected_lenepa"][role]["cosine_error"]
            and mechanism["projected_lenepa"][role][
                "retrieval_hit_at_1"
            ]
            >= mechanism["unprojected_lenepa"][role][
                "retrieval_hit_at_1"
            ]
            + 0.10
            for role in ("selection", "transfer_evaluation")
        )
    }
    controls = (
        "unprojected_lenepa",
        "projected_sigreg_only",
        "matched_pca",
    )
    transfer_controls = {
        name: forecast_scores[name]["transfer_evaluation"][
            "downstream_effect_mse"
        ]
        for name in controls
    }
    best_control = min(
        transfer_controls, key=lambda name: transfer_controls[name]
    )
    candidate_pairs = transfer_pair_errors["projected_lenepa"]
    control_pairs = transfer_pair_errors[best_control]
    common_pairs = sorted(set(candidate_pairs) & set(control_pairs))
    pair_win_fraction = (
        float(
            np.mean(
                [
                    candidate_pairs[pair_id] < control_pairs[pair_id]
                    for pair_id in common_pairs
                ]
            )
        )
        if common_pairs
        else 0.0
    )
    value_gates = {
        "selection_effect_is_best": all(
            candidate["selection"]["downstream_effect_mse"]
            < forecast_scores[name]["selection"][
                "downstream_effect_mse"
            ]
            for name in controls
        ),
        "transfer_effect_improves_best_control_and_raw_by_10_percent": (
            candidate["transfer_evaluation"]["downstream_effect_mse"]
            <= 0.90
            * min(
                transfer_controls[best_control],
                raw_transfer["downstream_effect_mse"],
            )
        ),
        "transfer_pair_win_fraction": pair_win_fraction >= 0.60,
    }
    passed = bool(
        all(safety_gates.values())
        and all(mechanism_gates.values())
        and all(value_gates.values())
    )
    return {
        "schema_version": 1,
        "experiment": "lenepa_telemetry_tracer_v1",
        "safety_gates": safety_gates,
        "mechanism_gates": mechanism_gates,
        "value_gates": value_gates,
        "best_transfer_control": best_control,
        "candidate_pair_win_fraction": pair_win_fraction,
        "passed": passed,
        "decision": (
            "advance_lenepa_to_fixed_seed_robustness"
            if passed
            else "reject_lenepa_telemetry_recipe"
        ),
    }


def _new_modules(
    graph: DeclaredTelemetryGraph,
    feature_count: int,
    config: LenepaConfig,
) -> Tuple[Any, Any]:
    torch = _require_torch()
    state = torch.random.get_rng_state()
    try:
        torch.manual_seed(config.seed)
        network = _build_network(
            torch,
            graph=graph,
            feature_count=feature_count,
            config=config,
        )
        projector = torch.nn.Sequential(
            torch.nn.Linear(
                config.width, config.projector_hidden_width
            ),
            torch.nn.BatchNorm1d(config.projector_hidden_width),
            torch.nn.ReLU(),
            torch.nn.Linear(
                config.projector_hidden_width,
                config.projector_width,
            ),
        )
        return network, projector
    finally:
        torch.random.set_rng_state(state)


def _build_network(
    torch: Any,
    *,
    graph: DeclaredTelemetryGraph,
    feature_count: int,
    config: LenepaConfig,
) -> Any:
    entity_count = len(graph.entities)
    kind_ids = torch.as_tensor(
        [0 if entity.kind == "node" else 1 for entity in graph.entities],
        dtype=torch.long,
    )
    degree_ids = torch.as_tensor(
        [
            min(7, len(graph.neighboring_entity_ids(entity_id)))
            for entity_id in graph.entity_ids
        ],
        dtype=torch.long,
    )

    class _CausalTelemetryEncoder(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.feature_weight = torch.nn.Parameter(
                torch.empty(
                    entity_count,
                    feature_count,
                    config.width,
                )
            )
            self.feature_bias = torch.nn.Parameter(
                torch.zeros(entity_count, config.width)
            )
            torch.nn.init.xavier_uniform_(self.feature_weight)
            self.time_embedding = torch.nn.Embedding(20, config.width)
            self.entity_embedding = torch.nn.Embedding(
                entity_count, config.width
            )
            self.kind_embedding = torch.nn.Embedding(2, config.width)
            self.degree_embedding = torch.nn.Embedding(8, config.width)
            self.blocks = torch.nn.ModuleList(
                [
                    torch.nn.TransformerEncoderLayer(
                        d_model=config.width,
                        nhead=config.head_count,
                        dim_feedforward=config.feedforward_width,
                        dropout=0.0,
                        activation="gelu",
                        batch_first=True,
                        norm_first=True,
                        bias=False,
                    )
                    for _ in range(config.depth)
                ]
            )
            self.final_norm = torch.nn.LayerNorm(
                config.width, eps=1e-6
            )
            self.register_buffer("kind_ids", kind_ids)
            self.register_buffer("degree_ids", degree_ids)

        def forward(self, values: Any) -> Tuple[Any, Any, Any]:
            contributions = torch.einsum(
                "btef,efw->btew", values, self.feature_weight
            ) + self.feature_bias[None, None]
            graph_summary = (
                self.entity_embedding.weight
                + self.kind_embedding(self.kind_ids)
                + self.degree_embedding(self.degree_ids)
            ).mean(dim=0)
            layer_zero = (
                contributions.sum(dim=2)
                + self.time_embedding.weight[None]
                + graph_summary[None, None]
            )
            hidden = layer_zero
            causal_mask = torch.triu(
                torch.ones(
                    20, 20, dtype=torch.bool, device=values.device
                ),
                diagonal=1,
            )
            for block in self.blocks:
                hidden = block(
                    hidden,
                    src_mask=causal_mask,
                    is_causal=True,
                )
            return layer_zero, self.final_norm(hidden), contributions

        def entity_output(
            self, layer_final: Any, contributions: Any
        ) -> Any:
            identity = (
                self.entity_embedding.weight
                + self.kind_embedding(self.kind_ids)
                + self.degree_embedding(self.degree_ids)
            )
            return (
                contributions[:, -1]
                + layer_final[:, -1, None]
                + identity[None]
            )

    return _CausalTelemetryEncoder()


def _project_sequence(projector: Any, values: Any) -> Any:
    batch, time_count, width = values.shape
    return projector(values.reshape(-1, width)).reshape(
        batch, time_count, -1
    )


def torch_cat(values: Sequence[Any], *, dim: int) -> Any:
    """Concatenate torch tensors without importing torch at module import."""

    return _require_torch().cat(tuple(values), dim=dim)


def _owned_histories(
    histories: NDArray[np.float64],
    ownership: NDArray[np.bool_],
) -> NDArray[np.float64]:
    return np.where(
        ownership[None, None],
        np.asarray(histories, dtype=np.float64),
        0.0,
    )


def _validate_histories(
    histories: NDArray[np.float64],
    graph: DeclaredTelemetryGraph,
    fitted_graph: DeclaredTelemetryGraph,
    feature_names: Tuple[str, ...],
) -> NDArray[np.float64]:
    source = np.asarray(histories, dtype=np.float64)
    if (
        graph.to_dict() != fitted_graph.to_dict()
        or source.ndim != 4
        or source.shape[1:]
        != (20, len(fitted_graph.entities), len(feature_names))
        or not np.all(np.isfinite(source))
    ):
        raise ValueError("LeNEPA inference histories are invalid")
    return source


def _cosine_error(predicted: Any, target: Any) -> float:
    torch = _require_torch()
    cosine = torch.nn.functional.cosine_similarity(
        predicted, target, dim=-1, eps=1e-8
    )
    return float((1.0 - cosine).mean())


def _retrieval_hit_at_one(predicted: Any, target: Any) -> float:
    torch = _require_torch()
    predicted_norm = torch.nn.functional.normalize(
        predicted, dim=-1, eps=1e-8
    )
    target_norm = torch.nn.functional.normalize(
        target, dim=-1, eps=1e-8
    )
    hits = []
    expected = torch.arange(
        predicted.size(0), device=predicted.device
    )
    for time_position in range(predicted.size(1)):
        similarity = (
            predicted_norm[:, time_position]
            @ target_norm[:, time_position].transpose(0, 1)
        )
        hits.append(similarity.argmax(dim=1) == expected)
    return float(torch.stack(hits, dim=1).to(torch.float32).mean())


def _set_optimizer_controls(
    optimizer: Any, config: LenepaConfig, step: int
) -> None:
    learning_rate = _cosine_control(
        step,
        config.steps,
        config.warmup_steps,
        config.learning_rate,
        config.minimum_learning_rate,
    )
    progress = (
        1.0
        if config.steps <= 1
        else float(step) / float(config.steps - 1)
    )
    weight_decay = config.initial_weight_decay + 0.5 * (
        config.final_weight_decay - config.initial_weight_decay
    ) * (1.0 - math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
        group["weight_decay"] = weight_decay


def _cosine_control(
    step: int,
    steps: int,
    warmup_steps: int,
    maximum: float,
    minimum: float,
) -> float:
    if step < warmup_steps:
        return maximum * float(step + 1) / float(warmup_steps)
    remaining = steps - warmup_steps
    if remaining <= 1:
        return minimum
    progress = float(step - warmup_steps) / float(remaining - 1)
    return minimum + 0.5 * (maximum - minimum) * (
        1.0 + math.cos(math.pi * progress)
    )


def _module_state(module: Any) -> Dict[str, Any]:
    return {
        str(key): value.detach().cpu().tolist()
        for key, value in module.state_dict().items()
    }


def _restore_module(module: Any, payload: Mapping[str, Any]) -> None:
    torch = _require_torch()
    current = module.state_dict()
    if set(payload) != set(current):
        raise ValueError("LeNEPA module state schema is invalid")
    restored = {
        key: torch.as_tensor(payload[key], dtype=value.dtype)
        for key, value in current.items()
    }
    module.load_state_dict(restored)


def _module_sha256(module: Any) -> str:
    return _canonical_sha256(_module_state(module))


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _array_sha256(values: NDArray[np.float64]) -> str:
    contiguous = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode())
    digest.update(str(contiguous.dtype).encode())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("LeNEPA requires PyTorch") from error
    torch.set_num_threads(1)
    return torch
