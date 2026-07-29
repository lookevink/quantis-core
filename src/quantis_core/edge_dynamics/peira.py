"""PEIRA inter-view regressor alignment for entity telemetry histories."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ..action_conditioned_dynamics import ActionConditionedWindows
from ..graph_telemetry import DeclaredTelemetryGraph
from .complete_lejepa import (
    EncodedTelemetry,
    PairBlockedAnchorSchedule,
    TelemetryViewSchedule,
    build_telemetry_backbone,
    fit_owned_feature_mask,
)


_OBJECTIVES = ("aligned_peira", "deranged_peira")


@dataclass(frozen=True)
class PeiraConfig:
    """Frozen architecture, optimizer, and stochastic controls."""

    objective: str = "aligned_peira"
    width: int = 64
    block_count: int = 2
    head_count: int = 4
    feedforward_width: int = 128
    projector_width: int = 256
    steps: int = 1600
    expected_pair_count: int = 40
    regularization: float = 0.1
    eta_initial: float = 0.9
    eta_final: float = 0.5
    learning_rate: float = 5e-4
    weight_decay: float = 5e-2
    warmup_steps: int = 80
    minimum_learning_rate: float = 5e-7
    initialization_seed: int = 26026
    anchor_seed: int = 26126
    view_seed: int = 26226
    derangement_seed: int = 26326
    preprocessing_protocol: str = (
        "action_conditioned_jepa_topology_transfer_v1"
    )

    def __post_init__(self) -> None:
        integer_values = (
            self.width,
            self.block_count,
            self.head_count,
            self.feedforward_width,
            self.projector_width,
            self.steps,
            self.expected_pair_count,
            self.warmup_steps,
        )
        if (
            self.objective not in _OBJECTIVES
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
                for value in integer_values
            )
            or self.width % self.head_count
        ):
            raise ValueError("PEIRA architecture controls are invalid")
        if not (
            0.0 < self.regularization < 1.0
            and 0.0 < self.eta_final <= self.eta_initial <= 1.0
            and self.learning_rate > 0.0
            and self.minimum_learning_rate > 0.0
            and self.minimum_learning_rate <= self.learning_rate
            and self.weight_decay >= 0.0
        ):
            raise ValueError("PEIRA numeric controls are invalid")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (
                self.initialization_seed,
                self.anchor_seed,
                self.view_seed,
                self.derangement_seed,
            )
        ):
            raise ValueError("PEIRA seeds must be integers")
        if not self.preprocessing_protocol:
            raise ValueError("PEIRA preprocessing identity is required")

    def learning_rate_at(self, step: int) -> float:
        """Return the frozen warmup/cosine learning rate."""

        _validate_step(step, self.steps)
        warmup = min(self.warmup_steps, self.steps)
        if step < warmup:
            return self.learning_rate * float(step + 1) / float(warmup)
        remaining = self.steps - warmup
        if remaining <= 1:
            return self.minimum_learning_rate
        progress = float(step - warmup) / float(remaining - 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.minimum_learning_rate + (
            self.learning_rate - self.minimum_learning_rate
        ) * cosine

    def clip_enabled_at(self, step: int) -> bool:
        """Return whether the frozen post-warmup clip is active."""

        _validate_step(step, self.steps)
        return step >= min(self.warmup_steps, self.steps)

    def to_dict(self) -> Dict[str, Any]:
        return dict(asdict(self))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PeiraConfig":
        if set(payload) != set(asdict(cls())):
            raise ValueError("PEIRA config schema is invalid")
        return cls(**dict(payload))


class PeiraSchedule:
    """Deterministic EMA-rate and matched-pair derangement schedule."""

    def __init__(
        self,
        *,
        steps: int,
        eta_initial: float,
        eta_final: float,
        derangement_seed: int,
    ) -> None:
        if (
            isinstance(steps, bool)
            or not isinstance(steps, int)
            or steps < 1
            or not (0.0 < eta_final <= eta_initial <= 1.0)
            or isinstance(derangement_seed, bool)
            or not isinstance(derangement_seed, int)
        ):
            raise ValueError("PEIRA schedule controls are invalid")
        self.steps = steps
        self.eta_initial = eta_initial
        self.eta_final = eta_final
        self.derangement_seed = derangement_seed

    def eta(self, step: int) -> float:
        """Linearly anneal the new-minibatch weight including endpoints."""

        _validate_step(step, self.steps)
        if self.steps == 1:
            return self.eta_initial
        fraction = float(step) / float(self.steps - 1)
        return self.eta_initial + fraction * (
            self.eta_final - self.eta_initial
        )

    def derangement(
        self, step: int, batch_size: int
    ) -> NDArray[np.int64]:
        """Return a seeded cyclic no-fixed-point row rotation."""

        _validate_step(step, self.steps)
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 2
        ):
            raise ValueError("PEIRA derangement needs at least two rows")
        generator = np.random.default_rng(
            np.random.SeedSequence((self.derangement_seed, step))
        )
        shift = int(generator.integers(1, batch_size))
        return np.roll(np.arange(batch_size, dtype=np.int64), shift)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": self.steps,
            "eta_initial": self.eta_initial,
            "eta_final": self.eta_final,
            "derangement_seed": self.derangement_seed,
        }


@dataclass(frozen=True)
class PeiraAuxiliaryStep:
    """One exact stochastic-compositional PEIRA update."""

    loss: Any
    trace_objective: Any
    auxiliary_value: Any
    batch_signal: Any
    batch_noise: Any
    signal: Any
    noise: Any
    predictor: Any
    inverse: Any
    symmetry_error: Any
    solve_residual: Any
    condition_number: Any


def peira_auxiliary_step(
    first: Any,
    second: Any,
    *,
    running_signal: Any,
    running_noise: Any,
    regularization: float,
    eta: float,
) -> PeiraAuxiliaryStep:
    """Update uncentered moments and return the literal PEIRA auxiliary loss."""

    torch = _require_torch()
    if (
        first.ndim != 2
        or second.shape != first.shape
        or first.shape[0] < 2
        or first.shape[1] < 1
        or running_signal.shape != (first.shape[1], first.shape[1])
        or running_noise.shape != running_signal.shape
        or not (0.0 < regularization < 1.0)
        or not (0.0 < eta <= 1.0)
    ):
        raise ValueError("PEIRA auxiliary inputs are invalid")
    width = first.shape[1]
    with torch.no_grad():
        first64 = first.detach().to(dtype=torch.float64)
        second64 = second.detach().to(dtype=torch.float64)
        batch_signal = (
            first64.T @ second64 + second64.T @ first64
        ) / float(first.shape[0])
        batch_noise = (
            first64.T @ first64 + second64.T @ second64
        ) / float(first.shape[0])
        signal = (
            (1.0 - eta) * running_signal.to(dtype=torch.float64)
            + eta * batch_signal
        )
        noise = (
            (1.0 - eta) * running_noise.to(dtype=torch.float64)
            + eta * batch_noise
        )
        identity = torch.eye(
            width, dtype=torch.float64, device=noise.device
        )
        regularized_noise = noise + regularization * identity
        inverse = torch.linalg.solve(regularized_noise, identity)
        predictor = signal @ inverse
        symmetry_error = torch.maximum(
            torch.max(torch.abs(signal - signal.T)),
            torch.max(torch.abs(noise - noise.T)),
        )
        solve_residual = torch.max(
            torch.abs(regularized_noise @ inverse - identity)
        )
        condition_number = torch.linalg.cond(regularized_noise)
    predictor_value = predictor.to(
        dtype=first.dtype, device=first.device
    )
    inverse_value = inverse.to(dtype=first.dtype, device=first.device)
    residual_first = first @ predictor_value.T - second
    residual_second = second @ predictor_value.T - first
    auxiliary = 0.5 * (
        torch.sum(
            first * (residual_first @ inverse_value.T), dim=1
        ).mean()
        + torch.sum(
            second * (residual_second @ inverse_value.T), dim=1
        ).mean()
    )
    scale = 0.5 * regularization * (
        torch.sum(first.square(), dim=1).mean()
        + torch.sum(second.square(), dim=1).mean()
    )
    loss = auxiliary + scale
    with torch.no_grad():
        trace_objective = (
            -0.5 * torch.trace(predictor)
            + 0.5
            * regularization
            * (
                torch.sum(first64.square(), dim=1).mean()
                + torch.sum(second64.square(), dim=1).mean()
            )
        )
    return PeiraAuxiliaryStep(
        loss=loss,
        trace_objective=trace_objective,
        auxiliary_value=auxiliary.detach(),
        batch_signal=batch_signal,
        batch_noise=batch_noise,
        signal=signal,
        noise=noise,
        predictor=predictor,
        inverse=inverse,
        symmetry_error=symmetry_error,
        solve_residual=solve_residual,
        condition_number=condition_number,
    )


class PeiraRepresentation:
    """Shared-encoder PEIRA representation with training-only moment state."""

    kind = "peira_telemetry_representation"
    schema_version = 1

    def __init__(self, config: PeiraConfig = PeiraConfig()) -> None:
        self.config = config
        self._graph: Optional[DeclaredTelemetryGraph] = None
        self._feature_names: Optional[Tuple[str, ...]] = None
        self._ownership_mask: Optional[NDArray[np.bool_]] = None
        self._varying_entity_mask: Optional[NDArray[np.bool_]] = None
        self._network: Any = None
        self._projector: Any = None
        self._running_signal: Optional[NDArray[np.float64]] = None
        self._running_noise: Optional[NDArray[np.float64]] = None
        self._training_evidence: Optional[
            Mapping[str, NDArray[Any]]
        ] = None
        self._inference_only = False

    @property
    def inference_parameter_count(self) -> int:
        _, _, _, network = self._encoder_values()
        return int(
            sum(parameter.numel() for parameter in network.parameters())
        )

    @property
    def training_parameter_count(self) -> int:
        _, _, _, network = self._encoder_values()
        if self._projector is None:
            raise RuntimeError("PEIRA inference bundle has no projector")
        return int(
            sum(parameter.numel() for parameter in network.parameters())
            + sum(
                parameter.numel()
                for parameter in self._projector.parameters()
            )
        )

    @property
    def training_evidence(self) -> Mapping[str, NDArray[Any]]:
        if self._training_evidence is None:
            raise RuntimeError("PEIRA training evidence is unavailable")
        return {
            name: values.copy()
            for name, values in self._training_evidence.items()
        }

    @property
    def final_moments(
        self,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        self._training_values()
        assert self._running_signal is not None
        assert self._running_noise is not None
        return self._running_signal.copy(), self._running_noise.copy()

    @property
    def final_operators(
        self,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return final paper operators P=Sigma Q and Q=(N+lI)^-1."""

        signal, noise = self.final_moments
        inverse = np.linalg.solve(
            noise
            + self.config.regularization
            * np.eye(self.config.width, dtype=np.float64),
            np.eye(self.config.width, dtype=np.float64),
        )
        return signal @ inverse, inverse

    def fit(
        self, windows: ActionConditionedWindows
    ) -> "PeiraRepresentation":
        """Fit the final deterministic CPU PEIRA representation."""

        if len(set(windows.matched_pair_ids)) != self.config.expected_pair_count:
            raise ValueError("PEIRA fit pair count differs from its contract")
        if windows.histories.shape[1] != 20:
            raise ValueError("PEIRA requires 20-point current histories")
        torch = _require_torch()
        ownership = fit_owned_feature_mask(windows)
        varying_entities = np.any(
            (
                np.ptp(windows.histories, axis=(0, 1)) > 1e-9
            )
            & ownership,
            axis=1,
        )
        anchors = PairBlockedAnchorSchedule(
            windows, seed=self.config.anchor_seed
        )
        views = TelemetryViewSchedule(
            graph=windows.graph,
            ownership_mask=ownership,
            varying_entity_mask=varying_entities,
            seed=self.config.view_seed,
        )
        schedule = PeiraSchedule(
            steps=self.config.steps,
            eta_initial=self.config.eta_initial,
            eta_final=self.config.eta_final,
            derangement_seed=self.config.derangement_seed,
        )
        rng_state = torch.random.get_rng_state()
        torch.manual_seed(self.config.initialization_seed)
        network = build_telemetry_backbone(
            torch,
            feature_count=windows.histories.shape[-1],
            graph=windows.graph,
            config=self.config,
        )
        projector = torch.nn.Sequential(
            torch.nn.Linear(
                self.config.width, self.config.projector_width
            ),
            torch.nn.GELU(),
            torch.nn.Linear(
                self.config.projector_width, self.config.width
            ),
        )
        torch.random.set_rng_state(rng_state)
        optimizer = torch.optim.AdamW(
            list(network.parameters()) + list(projector.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        signal = torch.zeros(
            (self.config.width, self.config.width),
            dtype=torch.float64,
        )
        noise = torch.zeros_like(signal)
        matrix_names = (
            "batch_signal",
            "batch_noise",
            "running_signal",
            "running_noise",
        )
        evidence: Dict[str, NDArray[Any]] = {
            name: np.empty(
                (
                    self.config.steps,
                    self.config.width,
                    self.config.width,
                ),
                dtype=np.float64,
            )
            for name in matrix_names
        }
        scalar_names = (
            "eta",
            "learning_rate",
            "clip_enabled",
            "loss",
            "auxiliary_value",
            "trace_objective",
            "trace_predictor",
            "symmetry_error",
            "solve_residual",
            "condition_number",
            "gradient_norm",
        )
        evidence.update(
            {
                name: np.empty(self.config.steps, dtype=np.float64)
                for name in scalar_names
            }
        )
        pair_count = len(anchors.pair_ids)
        evidence.update(
            {
                name: np.empty(
                    (self.config.steps, pair_count), dtype=np.int64
                )
                for name in (
                    "anchor_indices",
                    "anchor_arm_ids",
                    "anchor_transitions",
                    "pairing_indices",
                )
            }
        )
        evidence.update(
            {
                name: np.empty(
                    (
                        self.config.steps,
                        2,
                        20,
                        len(windows.graph.entities),
                    ),
                    dtype=np.bool_,
                )
                for name in ("view_visible", "view_present")
            }
        )
        network.train()
        projector.train()
        positions = np.arange(
            20 * len(windows.graph.entities), dtype=np.int64
        )
        for step in range(self.config.steps):
            learning_rate = self.config.learning_rate_at(step)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            anchor = anchors.batch(step)
            view_batch = views.batch(
                windows.histories[anchor.indices], step=step
            )
            embeddings = []
            for view_position in range(2):
                visible = torch.as_tensor(
                    view_batch.visible_tokens[view_position],
                    dtype=torch.bool,
                )
                hidden = network(
                    torch.as_tensor(
                        view_batch.values[view_position],
                        dtype=torch.float32,
                    ),
                    visible,
                    torch.as_tensor(
                        view_batch.present_tokens[view_position],
                        dtype=torch.bool,
                    ),
                    positions,
                )
                selected = visible.reshape(len(anchor.indices), -1)
                pooled = (
                    hidden
                    * selected.to(hidden.dtype).unsqueeze(-1)
                ).sum(dim=1) / selected.sum(
                    dim=1
                ).clamp_min(1).unsqueeze(-1)
                embeddings.append(projector(pooled))
            first, second = embeddings
            pairing = np.arange(len(anchor.indices), dtype=np.int64)
            if self.config.objective == "deranged_peira":
                pairing = schedule.derangement(
                    step, len(anchor.indices)
                )
                second = second[
                    torch.as_tensor(pairing, dtype=torch.long)
                ]
            eta = schedule.eta(step)
            update = peira_auxiliary_step(
                first,
                second,
                running_signal=signal,
                running_noise=noise,
                regularization=self.config.regularization,
                eta=eta,
            )
            if not bool(torch.isfinite(update.loss)):
                raise RuntimeError("PEIRA training became non-finite")
            optimizer.zero_grad(set_to_none=True)
            update.loss.backward()
            if self.config.clip_enabled_at(step):
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    list(network.parameters())
                    + list(projector.parameters()),
                    1.0,
                )
            else:
                gradient_norm = _gradient_norm(
                    list(network.parameters())
                    + list(projector.parameters())
                )
            optimizer.step()
            signal = update.signal
            noise = update.noise
            update_names = {
                "batch_signal": "batch_signal",
                "batch_noise": "batch_noise",
                "running_signal": "signal",
                "running_noise": "noise",
            }
            for name in matrix_names:
                evidence[name][step] = (
                    getattr(update, update_names[name])
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
            evidence["eta"][step] = eta
            evidence["learning_rate"][step] = learning_rate
            evidence["clip_enabled"][step] = float(
                self.config.clip_enabled_at(step)
            )
            evidence["loss"][step] = float(update.loss.detach())
            evidence["auxiliary_value"][step] = float(
                update.auxiliary_value
            )
            evidence["trace_objective"][step] = float(
                update.trace_objective
            )
            evidence["trace_predictor"][step] = float(
                torch.trace(update.predictor)
            )
            evidence["symmetry_error"][step] = float(
                update.symmetry_error
            )
            evidence["solve_residual"][step] = float(
                update.solve_residual
            )
            evidence["condition_number"][step] = float(
                update.condition_number
            )
            evidence["gradient_norm"][step] = float(gradient_norm)
            evidence["anchor_indices"][step] = anchor.indices
            evidence["anchor_arm_ids"][step] = anchor.arm_ids
            evidence["anchor_transitions"][
                step
            ] = anchor.transition_indices
            evidence["pairing_indices"][step] = pairing
            evidence["view_visible"][step] = (
                view_batch.visible_tokens[:2, 0]
            )
            evidence["view_present"][step] = (
                view_batch.present_tokens[:2, 0]
            )
        self._graph = windows.graph
        self._feature_names = windows.state_feature_names
        self._ownership_mask = ownership.copy()
        self._varying_entity_mask = varying_entities.copy()
        self._network = network.eval()
        self._projector = projector.eval()
        self._running_signal = signal.cpu().numpy().astype(np.float64)
        self._running_noise = noise.cpu().numpy().astype(np.float64)
        self._training_evidence = evidence
        self._inference_only = False
        return self

    def encode(
        self,
        histories: NDArray[np.float64],
        graph: DeclaredTelemetryGraph,
    ) -> EncodedTelemetry:
        """Encode complete current histories as entity-ordered tokens."""

        fitted_graph, feature_names, ownership, network = (
            self._encoder_values()
        )
        source = np.asarray(histories, dtype=np.float64)
        if (
            graph.to_dict() != fitted_graph.to_dict()
            or source.ndim != 4
            or source.shape[1:]
            != (
                20,
                len(fitted_graph.entities),
                len(feature_names),
            )
            or not np.all(np.isfinite(source))
        ):
            raise ValueError("PEIRA encoding input is invalid")
        torch = _require_torch()
        values = np.where(ownership[None, None], source, 0.0)
        visible = np.ones(values.shape[:-1], dtype=np.bool_)
        with torch.no_grad():
            hidden = network(
                torch.as_tensor(values, dtype=torch.float32),
                torch.as_tensor(visible, dtype=torch.bool),
                torch.as_tensor(visible, dtype=torch.bool),
                np.arange(20 * len(fitted_graph.entities)),
            )
            tokens = (
                hidden.reshape(
                    len(source),
                    20,
                    len(fitted_graph.entities),
                    self.config.width,
                )[:, -1]
                .cpu()
                .numpy()
                .astype(np.float64)
            )
        return EncodedTelemetry(
            tokens=tokens,
            entity_ids=fitted_graph.entity_ids,
            ownership_mask=ownership.copy(),
            observation_mask=ownership.copy(),
            content_sha256=self._content_sha256(),
            graph_sha256=_canonical_sha256(fitted_graph.to_dict()),
            state_schema_sha256=_canonical_sha256(
                {"feature_names": list(feature_names)}
            ),
            preprocessing_sha256=_canonical_sha256(
                {"protocol": self.config.preprocessing_protocol}
            ),
            encoder_sha256=self._content_sha256(),
        )

    def diagnose_views(
        self,
        histories: NDArray[np.float64],
        graph: DeclaredTelemetryGraph,
        *,
        step: int,
    ) -> NDArray[np.float64]:
        """Return the two aligned projector embeddings for diagnostics."""

        fitted_graph, _, ownership, network = self._encoder_values()
        if (
            self._projector is None
            or self._varying_entity_mask is None
            or graph.to_dict() != fitted_graph.to_dict()
        ):
            raise ValueError("PEIRA diagnostic state is unavailable")
        source = np.asarray(histories, dtype=np.float64)
        views = TelemetryViewSchedule(
            graph=graph,
            ownership_mask=ownership,
            varying_entity_mask=self._varying_entity_mask,
            seed=self.config.view_seed,
        ).batch(source, step=step)
        torch = _require_torch()
        positions = np.arange(20 * len(graph.entities), dtype=np.int64)
        embeddings = []
        with torch.no_grad():
            for view_position in range(2):
                visible = torch.as_tensor(
                    views.visible_tokens[view_position], dtype=torch.bool
                )
                hidden = network(
                    torch.as_tensor(
                        views.values[view_position], dtype=torch.float32
                    ),
                    visible,
                    torch.as_tensor(
                        views.present_tokens[view_position],
                        dtype=torch.bool,
                    ),
                    positions,
                )
                selected = visible.reshape(len(source), -1)
                pooled = (
                    hidden * selected[..., None].to(hidden.dtype)
                ).sum(dim=1) / selected.sum(
                    dim=1
                ).clamp_min(1).unsqueeze(-1)
                embeddings.append(self._projector(pooled))
        result: NDArray[np.float64] = (
            torch.stack(embeddings)
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the full fitted training and inference state."""

        graph, feature_names, ownership, network = self._training_values()
        assert self._projector is not None
        assert self._running_signal is not None
        assert self._running_noise is not None
        assert self._varying_entity_mask is not None
        predictor, inverse = self.final_operators
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "config": self.config.to_dict(),
            "graph": graph.to_dict(),
            "feature_names": list(feature_names),
            "ownership_mask": ownership.astype(int).tolist(),
            "varying_entity_mask": (
                self._varying_entity_mask.astype(int).tolist()
            ),
            "network_state": _module_state(network),
            "projector_state": _module_state(self._projector),
            "running_signal": self._running_signal.tolist(),
            "running_noise": self._running_noise.tolist(),
            "final_predictor": predictor.tolist(),
            "final_inverse": inverse.tolist(),
            "training_parameter_count": self.training_parameter_count,
            "inference_parameter_count": self.inference_parameter_count,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "PeiraRepresentation":
        """Restore a full PEIRA model without ambient random-state drift."""

        model = cls._restore_encoder(payload)
        torch = _require_torch()
        rng_state = torch.random.get_rng_state()
        projector = torch.nn.Sequential(
            torch.nn.Linear(
                model.config.width, model.config.projector_width
            ),
            torch.nn.GELU(),
            torch.nn.Linear(
                model.config.projector_width, model.config.width
            ),
        )
        torch.random.set_rng_state(rng_state)
        _restore_module(
            projector, dict(payload["projector_state"]), "projector"
        )
        model._projector = projector.eval()
        model._running_signal = np.asarray(
            payload["running_signal"], dtype=np.float64
        )
        model._running_noise = np.asarray(
            payload["running_noise"], dtype=np.float64
        )
        model._varying_entity_mask = np.asarray(
            payload["varying_entity_mask"], dtype=np.bool_
        )
        model._inference_only = False
        predictor, inverse = model.final_operators
        graph, _, _, _ = model._encoder_values()
        if (
            model._varying_entity_mask.shape
            != (len(graph.entities),)
            or not np.allclose(
                predictor,
                np.asarray(payload["final_predictor"], dtype=np.float64),
                rtol=1e-12,
                atol=1e-12,
            )
            or not np.allclose(
                inverse,
                np.asarray(payload["final_inverse"], dtype=np.float64),
                rtol=1e-12,
                atol=1e-12,
            )
            or model.training_parameter_count
            != int(payload["training_parameter_count"])
            or model.inference_parameter_count
            != int(payload["inference_parameter_count"])
        ):
            raise ValueError("PEIRA parameter identity mismatch")
        return model

    def to_inference_dict(self) -> Dict[str, Any]:
        """Serialize only state called by public edge inference."""

        graph, feature_names, ownership, network = self._encoder_values()
        return {
            "schema_version": self.schema_version,
            "kind": self.kind + "_inference",
            "config": self.config.to_dict(),
            "graph": graph.to_dict(),
            "feature_names": list(feature_names),
            "ownership_mask": ownership.astype(int).tolist(),
            "network_state": _module_state(network),
            "inference_parameter_count": self.inference_parameter_count,
        }

    @classmethod
    def from_inference_dict(
        cls, payload: Mapping[str, Any]
    ) -> "PeiraRepresentation":
        """Restore the causal inference-only PEIRA bundle."""

        expected_keys = {
            "schema_version",
            "kind",
            "config",
            "graph",
            "feature_names",
            "ownership_mask",
            "network_state",
            "inference_parameter_count",
        }
        if (
            set(payload) != expected_keys
            or payload.get("schema_version") != cls.schema_version
            or payload.get("kind") != cls.kind + "_inference"
        ):
            raise ValueError("unsupported PEIRA inference artifact")
        adjusted = dict(payload)
        adjusted["kind"] = cls.kind
        model = cls._restore_encoder(adjusted)
        model._inference_only = True
        if model.inference_parameter_count != int(
            payload["inference_parameter_count"]
        ):
            raise ValueError("PEIRA inference capacity mismatch")
        return model

    @classmethod
    def _restore_encoder(
        cls, payload: Mapping[str, Any]
    ) -> "PeiraRepresentation":
        if (
            payload.get("schema_version") != cls.schema_version
            or payload.get("kind") != cls.kind
        ):
            raise ValueError("unsupported PEIRA artifact")
        config = PeiraConfig.from_dict(dict(payload["config"]))
        graph = DeclaredTelemetryGraph.from_dict(dict(payload["graph"]))
        feature_names = tuple(str(value) for value in payload["feature_names"])
        ownership = np.asarray(
            payload["ownership_mask"], dtype=np.bool_
        )
        torch = _require_torch()
        rng_state = torch.random.get_rng_state()
        network = build_telemetry_backbone(
            torch,
            feature_count=len(feature_names),
            graph=graph,
            config=config,
        )
        torch.random.set_rng_state(rng_state)
        _restore_module(network, dict(payload["network_state"]), "network")
        model = cls(config)
        model._graph = graph
        model._feature_names = feature_names
        model._ownership_mask = ownership
        model._network = network.eval()
        return model

    def _content_sha256(self) -> str:
        graph, feature_names, ownership, network = self._encoder_values()
        return _canonical_sha256(
            {
                "config": self.config.to_dict(),
                "graph": graph.to_dict(),
                "feature_names": list(feature_names),
                "ownership_mask": ownership.astype(int).tolist(),
                "network_state": _module_state(network),
            }
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
            raise RuntimeError("PEIRA representation is not fitted")
        return (
            self._graph,
            self._feature_names,
            self._ownership_mask,
            self._network,
        )

    def _training_values(
        self,
    ) -> Tuple[
        DeclaredTelemetryGraph,
        Tuple[str, ...],
        NDArray[np.bool_],
        Any,
    ]:
        values = self._encoder_values()
        if self._inference_only or self._projector is None:
            raise RuntimeError("PEIRA inference bundle has no training state")
        return values


def assess_peira_gates(
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
    """Purely recompute the frozen PEIRA promotion decision."""

    candidate_name = "aligned_peira"
    candidate = forecast_scores[candidate_name]
    control_names = tuple(
        name for name in forecast_scores if name != candidate_name
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
    required_protocols = (
        "evidence_arrays_are_finite",
        "role_contract_recomputes",
        "capacity_recomputes",
        "public_inference_is_causal",
        "all_schedules_recompute",
        "training_moments_recompute",
        "final_operators_recompute",
        "varying_entity_mask_recomputes",
        "copied_source_assessor_recomputes",
        "selection_only_ridge_choice_recomputes",
        "selection_safety_status_recomputes",
        "bundle_size_recomputes",
        "latency_recomputes",
        "copied_prior_controls_match",
    )
    safety = {
        name: bool(protocol_checks.get(name, False))
        for name in required_protocols
    }
    counts = list(parameter_counts.values())
    safety.update(
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
                attribution[candidate_name]["no_action_specificity"] == 1.0
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
    value = {
        "selection_effect_is_best": (
            candidate["selection"]["downstream_effect_mse"]
            < forecast_scores[best_selection_control]["selection"][
                "downstream_effect_mse"
            ]
        ),
        "transfer_effect_improves_best_control_and_raw_by_5_percent": (
            candidate["transfer_evaluation"]["downstream_effect_mse"]
            <= 0.95
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
        all(safety.values())
        and all(bool(value) for value in mechanism_gates.values())
        and all(value.values())
    )
    return {
        "schema_version": 1,
        "experiment": "peira_telemetry_tracer_v1",
        "safety_gates": safety,
        "mechanism_gates": {
            str(name): bool(value)
            for name, value in mechanism_gates.items()
        },
        "value_gates": value,
        "best_selection_control": best_selection_control,
        "best_transfer_control": best_transfer_control,
        "transfer_pair_errors": {
            str(name): {
                str(pair): float(error)
                for pair, error in pair_values.items()
            }
            for name, pair_values in transfer_pair_errors.items()
        },
        "candidate_pair_win_fraction": pair_win_fraction,
        "passed": passed,
        "decision": (
            "advance_peira_recipe" if passed else "reject_peira_recipe"
        ),
    }


def _gradient_norm(parameters: Any) -> Any:
    torch = _require_torch()
    squared = [
        torch.sum(parameter.grad.detach().square())
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not squared:
        return torch.tensor(0.0)
    return torch.sqrt(torch.stack(squared).sum())


def _module_state(module: Any) -> Dict[str, Any]:
    return {
        name: tensor.detach().cpu().tolist()
        for name, tensor in module.state_dict().items()
    }


def _restore_module(
    module: Any, payload: Mapping[str, Any], label: str
) -> None:
    torch = _require_torch()
    expected = module.state_dict()
    if set(payload) != set(expected):
        raise ValueError(f"PEIRA {label} tensor names do not match")
    module.load_state_dict(
        {
            name: torch.as_tensor(payload[name], dtype=tensor.dtype)
            for name, tensor in expected.items()
        }
    )


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _validate_step(step: int, steps: int) -> None:
    if (
        isinstance(step, bool)
        or not isinstance(step, int)
        or step < 0
        or step >= steps
    ):
        raise ValueError("PEIRA step is outside its schedule")


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PEIRA requires the torch extra") from error
    return torch
