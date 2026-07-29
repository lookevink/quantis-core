"""Edge-sized Discrete-JEPA semantic tokenizer.

This module implements the paper-faithful telemetry translation frozen in
``docs/specs/discrete-jepa-telemetry-tracer-v1.md``. Tokenizer fitting sees
current histories and declared graph ownership only. Public inference accepts
current histories and the graph, and returns entity-ordered semantic tokens.
"""

import copy
import hashlib
import importlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Tuple, cast

import numpy as np
from numpy.typing import NDArray

from ..action_conditioned_dynamics import ActionConditionedWindows
from ..graph_telemetry import DeclaredTelemetryGraph
from .complete_lejepa import (
    PairBlockedAnchorSchedule,
    fit_owned_feature_mask,
)


DISCRETE_JEPA_OBJECTIVES = (
    "discrete_complete",
    "continuous_complete",
    "discrete_p2p_only",
)


@dataclass(frozen=True)
class DiscreteJepaConfig:
    """Frozen controls for one Discrete-JEPA telemetry cell."""

    objective: str = "discrete_complete"
    width: int = 64
    depth: int = 2
    head_count: int = 4
    feedforward_width: int = 128
    semantic_token_count: int = 7
    patch_count: int = 5
    patch_length: int = 4
    code_count: int = 64
    steps: int = 800
    warmup_steps: int = 40
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    minimum_learning_rate: float = 1e-6
    target_ema_decay: float = 0.996
    codebook_ema_decay: float = 0.99
    commitment_weight: float = 0.25
    gradient_clip_norm: float = 1.0
    expected_pair_count: int = 40
    seed: int = 25025
    preprocessing_protocol: str = (
        "action_conditioned_jepa_topology_transfer_v1"
    )

    def __post_init__(self) -> None:
        if (
            self.objective not in DISCRETE_JEPA_OBJECTIVES
            or self.width < 4
            or self.depth < 1
            or self.head_count < 1
            or self.width % self.head_count
            or self.feedforward_width < self.width
            or self.semantic_token_count != 7
            or self.patch_count != 5
            or self.patch_length != 4
            or self.code_count < 2
            or self.steps < 1
            or self.warmup_steps < 1
            or self.warmup_steps > self.steps
            or self.learning_rate <= 0.0
            or self.weight_decay < 0.0
            or not 0.0 < self.minimum_learning_rate
            <= self.learning_rate
            or not 0.0 < self.target_ema_decay < 1.0
            or not 0.0 < self.codebook_ema_decay < 1.0
            or self.commitment_weight <= 0.0
            or self.gradient_clip_norm <= 0.0
            or self.expected_pair_count < 2
            or isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not self.preprocessing_protocol
        ):
            raise ValueError("Discrete-JEPA configuration is invalid")

    def to_dict(self) -> Dict[str, Any]:
        return dict(asdict(self))

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "DiscreteJepaConfig":
        if set(payload) != set(asdict(cls())):
            raise ValueError("Discrete-JEPA config schema is invalid")
        return cls(**dict(payload))


class DiscreteMaskSchedule:
    """Deterministic 40%-60% entity-patch mask schedule."""

    def __init__(
        self, *, entity_count: int, patch_count: int, seed: int
    ) -> None:
        if (
            entity_count < 1
            or patch_count != 5
            or isinstance(seed, bool)
            or not isinstance(seed, int)
        ):
            raise ValueError("Discrete-JEPA mask controls are invalid")
        self.entity_count = entity_count
        self.patch_count = patch_count
        self.seed = seed

    def batch(self, *, step: int, batch_size: int) -> NDArray[np.bool_]:
        if (
            isinstance(step, bool)
            or not isinstance(step, int)
            or step < 0
            or batch_size < 1
        ):
            raise ValueError("Discrete-JEPA mask request is invalid")
        result = np.zeros(
            (batch_size, self.entity_count, self.patch_count),
            dtype=np.bool_,
        )
        generator = np.random.default_rng(
            np.random.SeedSequence((self.seed, step))
        )
        for row in range(batch_size):
            for entity in range(self.entity_count):
                count = int(generator.integers(2, 4))
                positions = generator.choice(
                    self.patch_count, size=count, replace=False
                )
                result[row, entity, positions] = True
        return result


@dataclass(frozen=True)
class DiscreteJepaLosses:
    """Complementary objective components."""

    total: Any
    s2p: Any
    p2s: Any
    p2p: Any
    commitment: Any


def discrete_jepa_losses(
    *,
    s2p_prediction: Any,
    p2s_prediction: Any,
    p2p_prediction: Any,
    target_patch: Any,
    target_semantic: Any,
    mask: Any,
    commitment: Any,
    objective: str,
) -> DiscreteJepaLosses:
    """Return the frozen complementary latent objective."""

    if (
        objective not in DISCRETE_JEPA_OBJECTIVES
        or s2p_prediction.shape != target_patch.shape
        or p2p_prediction.shape != target_patch.shape
        or p2s_prediction.shape != target_semantic.shape
        or mask.shape != target_patch.shape[:-1]
        or not bool(mask.any())
    ):
        raise ValueError("Discrete-JEPA loss tensors do not align")
    target_patch = target_patch.detach()
    target_semantic = target_semantic.detach()
    expanded = mask[..., None].expand_as(target_patch)
    s2p = (
        (s2p_prediction - target_patch).square()[expanded].mean()
    )
    p2s = (p2s_prediction - target_semantic).square().mean()
    p2p = (
        (p2p_prediction - target_patch).square()[expanded].mean()
    )
    if objective == "discrete_p2p_only":
        total = p2p + commitment
    else:
        total = s2p + p2s + p2p + commitment
    return DiscreteJepaLosses(
        total=total,
        s2p=s2p,
        p2s=p2s,
        p2p=p2p,
        commitment=commitment,
    )


@dataclass(frozen=True)
class DiscreteEncodedTelemetry:
    """Entity-ordered semantic tokens and optional hard indices."""

    tokens: NDArray[np.float64]
    indices: Optional[NDArray[np.int64]]
    entity_ids: Tuple[str, ...]
    ownership_mask: NDArray[np.bool_]
    content_sha256: str
    graph_sha256: str
    encoder_sha256: str

    def __post_init__(self) -> None:
        if (
            self.tokens.ndim != 3
            or self.tokens.shape[1] != len(self.entity_ids)
            or self.ownership_mask.shape[0] != len(self.entity_ids)
            or not np.all(np.isfinite(self.tokens))
            or (
                self.indices is not None
                and (
                    self.indices.shape != self.tokens.shape[:2]
                    or self.indices.dtype.kind not in ("i", "u")
                )
            )
            or any(
                len(value) != 64
                for value in (
                    self.content_sha256,
                    self.graph_sha256,
                    self.encoder_sha256,
                )
            )
        ):
            raise ValueError("Discrete-JEPA encoded telemetry is invalid")


@dataclass(frozen=True)
class DiscreteJepaDiagnostic:
    """Fixed-mask complementary prediction evidence."""

    s2p_prediction: NDArray[np.float64]
    p2s_prediction: NDArray[np.float64]
    p2p_prediction: NDArray[np.float64]
    target_patch: NDArray[np.float64]
    target_semantic: NDArray[np.float64]
    mask: NDArray[np.bool_]
    indices: Optional[NDArray[np.int64]]
    s2p: float
    p2s: float
    p2p: float

    def __post_init__(self) -> None:
        if (
            self.s2p_prediction.shape != self.target_patch.shape
            or self.p2p_prediction.shape != self.target_patch.shape
            or self.p2s_prediction.shape != self.target_semantic.shape
            or self.mask.shape != self.target_patch.shape[:-1]
            or (
                self.indices is not None
                and self.indices.shape != self.target_semantic.shape[:2]
            )
            or not all(
                np.all(np.isfinite(value))
                for value in (
                    self.s2p_prediction,
                    self.p2s_prediction,
                    self.p2p_prediction,
                    self.target_patch,
                    self.target_semantic,
                )
            )
            or not all(
                np.isfinite(value) and value >= 0.0
                for value in (self.s2p, self.p2s, self.p2p)
            )
        ):
            raise ValueError("Discrete-JEPA diagnostic is invalid")


class DiscreteJepaRepresentation:
    """Restorable hard or continuous semantic telemetry tokenizer."""

    kind = "discrete_jepa_telemetry_representation"
    schema_version = 1

    def __init__(
        self, config: DiscreteJepaConfig = DiscreteJepaConfig()
    ) -> None:
        self.config = config
        self._graph: Optional[DeclaredTelemetryGraph] = None
        self._feature_names: Tuple[str, ...] = ()
        self._ownership_mask: Optional[NDArray[np.bool_]] = None
        self._mean: Optional[NDArray[np.float64]] = None
        self._scale: Optional[NDArray[np.float64]] = None
        self._network: Any = None
        self._training_metrics: Tuple[Mapping[str, float], ...] = ()
        self._inference_only = False

    @property
    def training_metrics(self) -> Tuple[Mapping[str, float], ...]:
        self._training_values()
        return tuple(dict(row) for row in self._training_metrics)

    @property
    def training_parameter_count(self) -> int:
        *_, network = self._training_values()
        return int(
            sum(
                parameter.numel()
                for parameter in network.training_parameters()
            )
        )

    @property
    def inference_parameter_count(self) -> int:
        *_, network = self._encoder_values()
        return int(
            sum(
                parameter.numel()
                for parameter in network.inference_parameters()
            )
        )

    def fit(
        self, windows: ActionConditionedWindows
    ) -> "DiscreteJepaRepresentation":
        """Fit the masked same-history tokenizer on fitting only."""

        if (
            len(set(windows.matched_pair_ids))
            != self.config.expected_pair_count
            or windows.histories.shape[1] != 20
            or len(windows.entity_names)
            != self.config.semantic_token_count
        ):
            raise ValueError(
                "Discrete-JEPA fitting data differs from contract"
            )
        torch = _require_torch()
        ownership = fit_owned_feature_mask(windows)
        mean, scale = _fit_normalization(windows.histories, ownership)
        network = _new_network(
            graph=windows.graph,
            feature_count=len(windows.state_feature_names),
            ownership=ownership,
            config=self.config,
        )
        optimizer = torch.optim.AdamW(
            list(network.training_parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        anchors = PairBlockedAnchorSchedule(
            windows, seed=self.config.seed + 1
        )
        masks = DiscreteMaskSchedule(
            entity_count=len(windows.entity_names),
            patch_count=self.config.patch_count,
            seed=self.config.seed + 2,
        )
        metrics = []
        network.train()
        for step in range(self.config.steps):
            _set_learning_rate(optimizer, self.config, step)
            anchor = anchors.batch(step)
            source = _normalized_patches(
                windows.histories[anchor.indices],
                ownership,
                mean,
                scale,
                self.config,
            )
            mask = masks.batch(
                step=step, batch_size=len(anchor.indices)
            )
            optimizer.zero_grad(set_to_none=True)
            losses, _ = network.pretraining_loss(
                torch.as_tensor(source, dtype=torch.float32),
                torch.as_tensor(mask, dtype=torch.bool),
                update_codebook=True,
            )
            if not bool(torch.isfinite(losses.total)):
                raise RuntimeError(
                    "Discrete-JEPA training became non-finite"
                )
            losses.total.backward()
            torch.nn.utils.clip_grad_norm_(
                list(network.training_parameters()),
                self.config.gradient_clip_norm,
            )
            optimizer.step()
            network.update_target()
            metrics.append(
                {
                    "step": float(step + 1),
                    "loss": float(losses.total.detach()),
                    "s2p": float(losses.s2p.detach()),
                    "p2s": float(losses.p2s.detach()),
                    "p2p": float(losses.p2p.detach()),
                    "commitment": float(
                        losses.commitment.detach()
                    ),
                    "learning_rate": float(
                        optimizer.param_groups[0]["lr"]
                    ),
                    "independent_samples": float(
                        len(anchor.indices)
                    ),
                }
            )
        self._graph = windows.graph
        self._feature_names = windows.state_feature_names
        self._ownership_mask = ownership.copy()
        self._mean = mean
        self._scale = scale
        self._network = network.eval()
        self._training_metrics = tuple(metrics)
        return self

    def encode(
        self,
        histories: NDArray[np.float64],
        graph: DeclaredTelemetryGraph,
    ) -> DiscreteEncodedTelemetry:
        """Encode complete current histories into semantic tokens."""

        (
            fitted_graph,
            feature_names,
            ownership,
            mean,
            scale,
            network,
        ) = self._encoder_values()
        source = _validate_histories(
            histories, graph, fitted_graph, feature_names
        )
        patches = _normalized_patches(
            source, ownership, mean, scale, self.config
        )
        torch = _require_torch()
        with torch.no_grad():
            semantic, _ = network.online_encoder(
                torch.as_tensor(patches, dtype=torch.float32),
                torch.zeros(
                    (
                        len(patches),
                        len(fitted_graph.entity_ids),
                        self.config.patch_count,
                    ),
                    dtype=torch.bool,
                ),
            )
            if self.config.objective == "continuous_complete":
                tokens = semantic
                indices = None
            else:
                tokens, raw_indices = network.quantize(
                    semantic, straight_through=False
                )
                indices = (
                    raw_indices.cpu().numpy().astype(np.int64)
                )
        return DiscreteEncodedTelemetry(
            tokens=tokens.cpu().numpy().astype(np.float64),
            indices=indices,
            entity_ids=fitted_graph.entity_ids,
            ownership_mask=ownership.copy(),
            content_sha256=_array_sha256(source),
            graph_sha256=_canonical_sha256(fitted_graph.to_dict()),
            encoder_sha256=_encoder_sha256(network),
        )

    def diagnose(
        self,
        histories: NDArray[np.float64],
        graph: DeclaredTelemetryGraph,
        *,
        mask_seed: int,
    ) -> DiscreteJepaDiagnostic:
        """Return fixed-mask complementary objective evidence."""

        (
            fitted_graph,
            feature_names,
            ownership,
            mean,
            scale,
            network,
        ) = self._training_values()
        source = _validate_histories(
            histories, graph, fitted_graph, feature_names
        )
        patches = _normalized_patches(
            source, ownership, mean, scale, self.config
        )
        mask = DiscreteMaskSchedule(
            entity_count=len(fitted_graph.entity_ids),
            patch_count=self.config.patch_count,
            seed=mask_seed,
        ).batch(step=0, batch_size=len(source))
        torch = _require_torch()
        with torch.no_grad():
            losses, values = network.pretraining_loss(
                torch.as_tensor(patches, dtype=torch.float32),
                torch.as_tensor(mask, dtype=torch.bool),
                update_codebook=False,
            )
        indices = values["indices"]
        return DiscreteJepaDiagnostic(
            s2p_prediction=_numpy(values["s2p_prediction"]),
            p2s_prediction=_numpy(values["p2s_prediction"]),
            p2p_prediction=_numpy(values["p2p_prediction"]),
            target_patch=_numpy(values["target_patch"]),
            target_semantic=_numpy(values["target_semantic"]),
            mask=mask,
            indices=(
                None
                if indices is None
                else indices.cpu().numpy().astype(np.int64)
            ),
            s2p=float(losses.s2p),
            p2s=float(losses.p2s),
            p2p=float(losses.p2p),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize complete training and inference state."""

        (
            graph,
            feature_names,
            ownership,
            mean,
            scale,
            network,
        ) = self._training_values()
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "config": self.config.to_dict(),
            "graph": graph.to_dict(),
            "feature_names": list(feature_names),
            "ownership_mask": ownership.astype(int).tolist(),
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "network_state": _module_state(network),
            "training_metrics": [
                dict(row) for row in self._training_metrics
            ],
            "training_parameter_count": self.training_parameter_count,
            "inference_parameter_count": self.inference_parameter_count,
        }

    def to_inference_dict(self) -> Dict[str, Any]:
        """Serialize only online encoder and hard-code inference state."""

        (
            graph,
            feature_names,
            ownership,
            mean,
            scale,
            network,
        ) = self._encoder_values()
        return {
            "schema_version": self.schema_version,
            "kind": "discrete_jepa_student_inference_bundle",
            "config": self.config.to_dict(),
            "graph": graph.to_dict(),
            "feature_names": list(feature_names),
            "ownership_mask": ownership.astype(int).tolist(),
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "encoder_state": _module_state(network.online_encoder),
            "codebook": _numpy(network.codebook).tolist(),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "DiscreteJepaRepresentation":
        if (
            payload.get("schema_version") != cls.schema_version
            or payload.get("kind") != cls.kind
        ):
            raise ValueError("unsupported Discrete-JEPA artifact")
        model, network = cls._restore_common(payload)
        _restore_module(network, dict(payload["network_state"]))
        model._training_metrics = tuple(
            {
                str(key): float(value)
                for key, value in dict(row).items()
            }
            for row in payload["training_metrics"]
        )
        model._network = network.eval()
        if (
            model.training_parameter_count
            != int(payload["training_parameter_count"])
            or model.inference_parameter_count
            != int(payload["inference_parameter_count"])
        ):
            raise ValueError(
                "Discrete-JEPA artifact capacity differs"
            )
        return model

    @classmethod
    def from_inference_dict(
        cls, payload: Mapping[str, Any]
    ) -> "DiscreteJepaRepresentation":
        if (
            payload.get("schema_version") != cls.schema_version
            or payload.get("kind")
            != "discrete_jepa_student_inference_bundle"
            or "target_state" in payload
            or "predictor_state" in payload
        ):
            raise ValueError(
                "unsupported Discrete-JEPA inference bundle"
            )
        model, network = cls._restore_common(payload)
        _restore_module(
            network.online_encoder, dict(payload["encoder_state"])
        )
        torch = _require_torch()
        codebook = torch.as_tensor(
            payload["codebook"], dtype=network.codebook.dtype
        )
        if codebook.shape != network.codebook.shape:
            raise ValueError("Discrete-JEPA codebook shape differs")
        network.codebook.copy_(codebook)
        network.target_encoder = None
        network.s2p_predictor = None
        network.p2s_predictor = None
        network.p2p_predictor = None
        model._network = network.eval()
        model._inference_only = True
        return model

    @classmethod
    def _restore_common(
        cls, payload: Mapping[str, Any]
    ) -> Tuple["DiscreteJepaRepresentation", Any]:
        config = DiscreteJepaConfig.from_dict(dict(payload["config"]))
        graph = DeclaredTelemetryGraph.from_dict(dict(payload["graph"]))
        feature_names = tuple(
            str(value) for value in payload["feature_names"]
        )
        ownership = np.asarray(
            payload["ownership_mask"], dtype=np.bool_
        )
        mean = np.asarray(payload["mean"], dtype=np.float64)
        scale = np.asarray(payload["scale"], dtype=np.float64)
        if (
            ownership.shape != mean.shape
            or scale.shape != mean.shape
            or mean.shape
            != (len(graph.entity_ids), len(feature_names))
            or np.any(scale <= 0.0)
        ):
            raise ValueError(
                "Discrete-JEPA normalization schema differs"
            )
        network = _new_network(
            graph=graph,
            feature_count=len(feature_names),
            ownership=ownership,
            config=config,
        )
        model = cls(config)
        model._graph = graph
        model._feature_names = feature_names
        model._ownership_mask = ownership
        model._mean = mean
        model._scale = scale
        return model, network

    def _encoder_values(
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
            or self._mean is None
            or self._scale is None
            or self._network is None
        ):
            raise RuntimeError(
                "Discrete-JEPA representation is not fitted"
            )
        return (
            self._graph,
            self._feature_names,
            self._ownership_mask,
            self._mean,
            self._scale,
            self._network,
        )

    def _training_values(self) -> Tuple[Any, ...]:
        values = self._encoder_values()
        if self._inference_only:
            raise RuntimeError(
                "Discrete-JEPA inference bundle has no training state"
            )
        return values


def assess_discrete_jepa_gates(
    *,
    forecast_scores: Mapping[str, Mapping[str, Mapping[str, float]]],
    raw_scores: Mapping[str, Mapping[str, float]],
    mechanism_gates: Mapping[str, bool],
    attribution: Mapping[str, Mapping[str, float]],
    action_sanity: Mapping[str, Mapping[str, float]],
    restoration_max_abs: Mapping[str, float],
    protocol_checks: Mapping[str, bool],
    parameter_counts: Mapping[str, Mapping[str, int]],
    transfer_pair_errors: Mapping[str, Mapping[str, float]],
    deployed_bundle_bytes: int,
    median_latency_ms: float,
) -> Dict[str, Any]:
    """Purely recompute the frozen Discrete-JEPA decision."""

    candidate_name = "discrete_complete"
    candidate = forecast_scores[candidate_name]
    control_names = (
        "continuous_complete",
        "discrete_p2p_only",
        "matched_pca",
    )
    best_selection_control = min(
        control_names,
        key=lambda name: forecast_scores[name]["selection"][
            "downstream_effect_mse"
        ],
    )
    best_transfer_control = min(
        control_names,
        key=lambda name: forecast_scores[name]["transfer_evaluation"][
            "downstream_effect_mse"
        ],
    )
    counts = list(parameter_counts.values())
    required_protocols = (
        "evidence_arrays_are_finite",
        "pair_and_trajectory_roles_are_disjoint",
        "capacity_recomputes",
        "public_inference_is_causal",
        "anchor_schedule_recomputes",
        "mask_schedule_recomputes",
        "selection_only_ridge_choice_recomputes",
        "selection_safety_status_recomputes",
        "bundle_size_recomputes",
        "latency_recomputes",
    )
    safety_gates = {
        name: bool(protocol_checks.get(name, False))
        for name in required_protocols
    }
    safety_gates.update(
        {
            "capacity_is_matched": bool(counts)
            and all(value == counts[0] for value in counts[1:]),
            "restoration_within_1e_6": all(
                np.isfinite(value) and value <= 1e-6
                for value in restoration_max_abs.values()
            ),
            "selection_overall_within_1_05_raw": (
                candidate["selection"]["overall_mse"]
                <= 1.05 * raw_scores["selection"]["overall_mse"]
            ),
            "selection_action_within_1_05_raw": (
                candidate["selection"]["action_overlap_mse"]
                <= 1.05
                * raw_scores["selection"]["action_overlap_mse"]
            ),
            "transfer_overall_within_1_05_raw": (
                candidate["transfer_evaluation"]["overall_mse"]
                <= 1.05
                * raw_scores["transfer_evaluation"]["overall_mse"]
            ),
            "transfer_action_within_1_05_raw": (
                candidate["transfer_evaluation"][
                    "action_overlap_mse"
                ]
                <= 1.05
                * raw_scores["transfer_evaluation"][
                    "action_overlap_mse"
                ]
            ),
            "action_and_target_hit_at_1": (
                attribution[candidate_name][
                    "action_and_target_hit_at_1"
                ]
                >= 0.95
            ),
            "no_action_specificity": (
                attribution[candidate_name]["no_action_specificity"]
                == 1.0
            ),
            "correct_action_sanity": (
                action_sanity[candidate_name][
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
    candidate_pair = transfer_pair_errors[candidate_name]
    control_pair = transfer_pair_errors[best_transfer_control]
    common_pairs = sorted(set(candidate_pair) & set(control_pair))
    pair_win_fraction = float(
        np.mean(
            [
                candidate_pair[pair] < control_pair[pair]
                for pair in common_pairs
            ]
        )
    )
    value_gates = {
        "selection_effect_is_best": (
            candidate["selection"]["downstream_effect_mse"]
            < forecast_scores[best_selection_control]["selection"][
                "downstream_effect_mse"
            ]
        ),
        "transfer_effect_improves_best_control_and_raw_by_10_percent": (
            candidate["transfer_evaluation"][
                "downstream_effect_mse"
            ]
            <= 0.90
            * min(
                forecast_scores[best_transfer_control][
                    "transfer_evaluation"
                ]["downstream_effect_mse"],
                raw_scores["transfer_evaluation"][
                    "downstream_effect_mse"
                ],
            )
        ),
        "transfer_pair_win_fraction": pair_win_fraction >= 0.60,
    }
    passed = bool(
        all(safety_gates.values())
        and all(bool(value) for value in mechanism_gates.values())
        and all(value_gates.values())
    )
    return {
        "schema_version": 1,
        "experiment": "discrete_jepa_telemetry_tracer_v1",
        "safety_gates": safety_gates,
        "mechanism_gates": {
            str(name): bool(value)
            for name, value in mechanism_gates.items()
        },
        "value_gates": value_gates,
        "best_selection_control": best_selection_control,
        "best_transfer_control": best_transfer_control,
        "transfer_pair_errors": {
            str(name): {
                str(pair): float(error)
                for pair, error in pair_errors.items()
            }
            for name, pair_errors in transfer_pair_errors.items()
        },
        "candidate_pair_win_fraction": pair_win_fraction,
        "passed": passed,
        "decision": (
            "advance_discrete_jepa_recipe"
            if passed
            else "reject_discrete_jepa_recipe"
        ),
    }


def _new_network(
    *,
    graph: DeclaredTelemetryGraph,
    feature_count: int,
    ownership: NDArray[np.bool_],
    config: DiscreteJepaConfig,
) -> Any:
    torch = _require_torch()
    state = torch.random.get_rng_state()
    try:
        torch.manual_seed(config.seed)
        network = _build_network(
            torch,
            entity_count=len(graph.entity_ids),
            feature_count=feature_count,
            ownership=ownership,
            config=config,
        )
    finally:
        torch.random.set_rng_state(state)
    return network


def _build_network(
    torch: Any,
    *,
    entity_count: int,
    feature_count: int,
    ownership: NDArray[np.bool_],
    config: DiscreteJepaConfig,
) -> Any:
    nn = torch.nn

    class Encoder(nn.Module):  # type: ignore[misc,name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.patch_projection = nn.Linear(
                config.patch_length * feature_count,
                config.width,
                bias=False,
            )
            self.semantic_seed = nn.Parameter(
                torch.zeros(1, entity_count, config.width)
            )
            self.entity_embedding = nn.Embedding(
                entity_count, config.width
            )
            self.patch_position = nn.Parameter(
                torch.zeros(1, 1, config.patch_count, config.width)
            )
            self.mask_embedding = nn.Parameter(
                torch.zeros(1, 1, 1, config.width)
            )
            layer = nn.TransformerEncoderLayer(
                d_model=config.width,
                nhead=config.head_count,
                dim_feedforward=config.feedforward_width,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
                bias=False,
            )
            self.transformer = nn.TransformerEncoder(
                layer, num_layers=config.depth
            )
            self.norm = nn.LayerNorm(config.width, eps=1e-6)
            self.register_buffer(
                "ownership",
                torch.as_tensor(ownership, dtype=torch.bool),
            )

        def forward(self, patches: Any, mask: Any) -> Tuple[Any, Any]:
            if (
                patches.ndim != 5
                or patches.shape[1:4]
                != (
                    entity_count,
                    config.patch_count,
                    config.patch_length,
                )
                or patches.shape[-1] != feature_count
                or mask.shape != patches.shape[:3]
            ):
                raise ValueError(
                    "Discrete-JEPA encoder inputs are invalid"
                )
            batch = len(patches)
            owned = torch.where(
                self.ownership[None, :, None, None],
                patches,
                torch.zeros_like(patches),
            )
            visible = torch.where(
                mask[..., None, None],
                torch.zeros_like(owned),
                owned,
            )
            projected = self.patch_projection(
                visible.reshape(
                    batch,
                    entity_count,
                    config.patch_count,
                    config.patch_length * feature_count,
                )
            )
            entity_ids = torch.arange(
                entity_count, device=patches.device
            )
            entity = self.entity_embedding(entity_ids)
            semantic = (
                self.semantic_seed
                + entity[None]
            ).expand(batch, -1, -1)
            patch = (
                projected
                + entity[None, :, None]
                + self.patch_position
                + mask[..., None] * self.mask_embedding
            )
            tokens = torch.cat(
                (semantic, patch.reshape(batch, -1, config.width)),
                dim=1,
            )
            encoded = self.norm(self.transformer(tokens))
            return (
                encoded[:, :entity_count],
                encoded[:, entity_count:].reshape(
                    batch,
                    entity_count,
                    config.patch_count,
                    config.width,
                ),
            )

    class PatchPredictor(nn.Module):  # type: ignore[misc,name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.position = nn.Parameter(
                torch.zeros(1, 1, config.patch_count, config.width)
            )
            self.network = nn.Sequential(
                nn.Linear(config.width, config.feedforward_width),
                nn.GELU(),
                nn.Linear(config.feedforward_width, config.width),
            )

        def forward(self, values: Any) -> Any:
            return self.network(values + self.position)

    class SemanticPredictor(nn.Module):  # type: ignore[misc,name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(config.width, config.feedforward_width),
                nn.GELU(),
                nn.Linear(config.feedforward_width, config.width),
            )

        def forward(self, patch: Any) -> Any:
            local = patch.mean(dim=2)
            global_value = local.mean(dim=1, keepdim=True)
            return self.network(local + global_value)

    class Network(nn.Module):  # type: ignore[misc,name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.online_encoder = Encoder()
            self.target_encoder = copy.deepcopy(self.online_encoder)
            for parameter in self.target_encoder.parameters():
                parameter.requires_grad_(False)
            self.s2p_predictor = PatchPredictor()
            self.p2s_predictor = SemanticPredictor()
            self.p2p_predictor = PatchPredictor()
            codebook = torch.empty(config.code_count, config.width)
            nn.init.orthogonal_(codebook)
            self.register_buffer("codebook", codebook)
            self.register_buffer(
                "ema_count", torch.ones(config.code_count)
            )
            self.register_buffer(
                "ema_sum", codebook.clone()
            )

        def training_parameters(self) -> Any:
            return (
                list(self.online_encoder.parameters())
                + list(self.s2p_predictor.parameters())
                + list(self.p2s_predictor.parameters())
                + list(self.p2p_predictor.parameters())
            )

        def inference_parameters(self) -> Any:
            return list(self.online_encoder.parameters())

        def quantize(
            self, semantic: Any, *, straight_through: bool
        ) -> Tuple[Any, Any]:
            distance = (
                semantic.square().sum(dim=-1, keepdim=True)
                - 2.0
                * torch.einsum(
                    "...d,kd->...k", semantic, self.codebook
                )
                + self.codebook.square().sum(dim=-1)[None, None]
            )
            indices = distance.argmin(dim=-1)
            values = self.codebook[indices]
            if straight_through:
                values = semantic + (values - semantic).detach()
            return values, indices

        def update_codebook(self, values: Any) -> None:
            with torch.no_grad():
                flat = values.detach().reshape(-1, config.width)
                distance = (
                    flat.square().sum(dim=-1, keepdim=True)
                    - 2.0 * flat @ self.codebook.T
                    + self.codebook.square().sum(dim=-1)[None]
                )
                indices = distance.argmin(dim=-1)
                one_hot = torch.nn.functional.one_hot(
                    indices, num_classes=config.code_count
                ).to(flat.dtype)
                count = one_hot.sum(dim=0)
                total = one_hot.T @ flat
                decay = config.codebook_ema_decay
                self.ema_count.mul_(decay).add_(
                    count, alpha=1.0 - decay
                )
                self.ema_sum.mul_(decay).add_(
                    total, alpha=1.0 - decay
                )
                normalized = self.ema_sum / self.ema_count.clamp_min(
                    1e-5
                )[:, None]
                self.codebook.copy_(normalized)

        def update_target(self) -> None:
            with torch.no_grad():
                for target, online in zip(
                    self.target_encoder.parameters(),
                    self.online_encoder.parameters(),
                ):
                    target.mul_(config.target_ema_decay).add_(
                        online, alpha=1.0 - config.target_ema_decay
                    )

        def pretraining_loss(
            self,
            patches: Any,
            mask: Any,
            *,
            update_codebook: bool,
        ) -> Tuple[DiscreteJepaLosses, Mapping[str, Any]]:
            context_semantic, context_patch = self.online_encoder(
                patches, mask
            )
            with torch.no_grad():
                target_semantic, target_patch = self.target_encoder(
                    patches, torch.zeros_like(mask)
                )
            indices = None
            if config.objective == "continuous_complete":
                semantic_input = context_semantic
                commitment = context_semantic.sum() * 0.0
            else:
                semantic_input, indices = self.quantize(
                    context_semantic, straight_through=True
                )
                quantized, _ = self.quantize(
                    context_semantic, straight_through=False
                )
                commitment = (
                    config.commitment_weight
                    * torch.nn.functional.mse_loss(
                        context_semantic, quantized.detach()
                    )
                )
                if update_codebook:
                    self.update_codebook(
                        torch.cat(
                            (context_semantic, target_semantic), dim=0
                        )
                    )
            global_semantic = semantic_input.mean(
                dim=1, keepdim=True
            )
            semantic_for_patch = (
                semantic_input + global_semantic
            )[:, :, None].expand(
                -1, -1, config.patch_count, -1
            )
            s2p_prediction = self.s2p_predictor(
                semantic_for_patch
            )
            p2s_prediction = self.p2s_predictor(context_patch)
            global_patch = context_patch.mean(
                dim=(1, 2), keepdim=True
            )
            p2p_prediction = self.p2p_predictor(
                context_patch + global_patch
            )
            losses = discrete_jepa_losses(
                s2p_prediction=s2p_prediction,
                p2s_prediction=p2s_prediction,
                p2p_prediction=p2p_prediction,
                target_patch=target_patch,
                target_semantic=target_semantic,
                mask=mask,
                commitment=commitment,
                objective=config.objective,
            )
            return losses, {
                "s2p_prediction": s2p_prediction,
                "p2s_prediction": p2s_prediction,
                "p2p_prediction": p2p_prediction,
                "target_patch": target_patch,
                "target_semantic": target_semantic,
                "indices": indices,
            }

    return Network()


def _fit_normalization(
    histories: NDArray[np.float64],
    ownership: NDArray[np.bool_],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    values = np.asarray(histories, dtype=np.float64)
    mean = np.mean(values, axis=(0, 1))
    scale = np.std(values, axis=(0, 1))
    mean = np.where(ownership, mean, 0.0)
    scale = np.where(ownership & (scale > 1e-8), scale, 1.0)
    return mean, scale


def _normalized_patches(
    histories: NDArray[np.float64],
    ownership: NDArray[np.bool_],
    mean: NDArray[np.float64],
    scale: NDArray[np.float64],
    config: DiscreteJepaConfig,
) -> NDArray[np.float64]:
    normalized = np.where(
        ownership[None, None],
        (np.asarray(histories, dtype=np.float64) - mean[None, None])
        / scale[None, None],
        0.0,
    )
    batch, _, entities, features = normalized.shape
    return (
        normalized.reshape(
            batch,
            config.patch_count,
            config.patch_length,
            entities,
            features,
        )
        .transpose(0, 3, 1, 2, 4)
        .copy()
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
        or source.shape[1:] != (
            20,
            len(fitted_graph.entity_ids),
            len(feature_names),
        )
        or not np.all(np.isfinite(source))
    ):
        raise ValueError("Discrete-JEPA inference inputs are invalid")
    return source


def _set_learning_rate(
    optimizer: Any, config: DiscreteJepaConfig, step: int
) -> None:
    if step < config.warmup_steps:
        ratio = float(step + 1) / float(config.warmup_steps)
        value = config.learning_rate * ratio
    else:
        denominator = max(1, config.steps - config.warmup_steps)
        progress = float(step - config.warmup_steps) / denominator
        value = config.minimum_learning_rate + 0.5 * (
            config.learning_rate - config.minimum_learning_rate
        ) * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = value


def _module_state(module: Any) -> Dict[str, Any]:
    return {
        str(name): value.detach().cpu().tolist()
        for name, value in module.state_dict().items()
    }


def _restore_module(module: Any, payload: Mapping[str, Any]) -> None:
    torch = _require_torch()
    reference = module.state_dict()
    if set(payload) != set(reference):
        raise ValueError("Discrete-JEPA module state schema differs")
    restored = {}
    for name, value in reference.items():
        tensor = torch.as_tensor(payload[name], dtype=value.dtype)
        if tensor.shape != value.shape:
            raise ValueError(
                "Discrete-JEPA module state shape differs"
            )
        restored[name] = tensor
    module.load_state_dict(restored, strict=True)


def _encoder_sha256(network: Any) -> str:
    payload = {
        "encoder": _module_state(network.online_encoder),
        "codebook": _numpy(network.codebook).tolist(),
    }
    return _canonical_sha256(payload)


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256(value: NDArray[np.float64]) -> str:
    array = np.ascontiguousarray(value, dtype=np.float64)
    return hashlib.sha256(array.view(np.uint8).tobytes()).hexdigest()


def _numpy(value: Any) -> NDArray[np.float64]:
    return cast(
        NDArray[np.float64],
        value.detach().cpu().numpy().astype(np.float64),
    )


def _require_torch() -> Any:
    return importlib.import_module("torch")
