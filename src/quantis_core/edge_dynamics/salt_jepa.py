"""Static-teacher asymmetric latent training for telemetry.

The public primitives in this module implement the frozen SALT telemetry
contract.  Training and artifact assessment are added in vertical slices;
the mask schedule is deliberately usable and testable without a model.
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from ..action_conditioned_dynamics import ActionConditionedWindows
from ..graph_telemetry import DeclaredTelemetryGraph
from .complete_lejepa import (
    CompleteLejepaConfig,
    EncodedTelemetry,
    PairBlockedAnchorSchedule,
    build_complete_lejepa_backbone,
    fit_owned_feature_mask,
)


@dataclass(frozen=True)
class SaltMaskedTelemetry:
    """One immutable masked context batch."""

    values: NDArray[np.float64]
    visible_tokens: NDArray[np.bool_]
    target_tokens: NDArray[np.bool_]
    block_rectangles: NDArray[np.int64]
    fill_order: NDArray[np.int64]

    def __post_init__(self) -> None:
        if (
            self.values.ndim != 4
            or self.visible_tokens.shape != self.values.shape[:-1]
            or self.target_tokens.shape != self.visible_tokens.shape
            or self.block_rectangles.shape
            != (len(self.values), 64, 5)
            or self.fill_order.shape != (len(self.values), 126)
            or self.block_rectangles.dtype.kind not in ("i", "u")
            or self.fill_order.dtype.kind not in ("i", "u")
            or not np.array_equal(
                self.target_tokens, ~self.visible_tokens
            )
            or not np.all(np.isfinite(self.values))
        ):
            raise ValueError("SALT masked telemetry is invalid")


class SaltMaskSchedule:
    """Seeded 90% time-by-connected-entity multi-block masks."""

    def __init__(
        self,
        *,
        graph: DeclaredTelemetryGraph,
        ownership_mask: NDArray[np.bool_],
        seed: int,
    ) -> None:
        ownership = np.asarray(ownership_mask, dtype=np.bool_)
        if (
            ownership.ndim != 2
            or ownership.shape[0] != len(graph.entities)
            or isinstance(seed, bool)
            or not isinstance(seed, int)
        ):
            raise ValueError("SALT mask schedule schema is invalid")
        observed = tuple(
            int(value)
            for value in np.flatnonzero(np.any(ownership, axis=1))
        )
        if not observed:
            raise ValueError("SALT masking requires observed entities")
        token_count = 20 * len(graph.entities)
        target_count = int(round(0.90 * token_count))
        if target_count > token_count - len(observed):
            raise ValueError(
                "SALT mask cannot preserve anchor-time entity visibility"
            )
        self.graph = graph
        self.ownership_mask = ownership.copy()
        self.observed_entities = observed
        self.seed = seed
        self.target_count = target_count
        self._entity_blocks = tuple(
            self._connected_block(root) for root in observed
        )

    def batch(
        self, histories: NDArray[np.float64], *, step: int
    ) -> SaltMaskedTelemetry:
        """Return a copied masked batch for one deterministic schedule step."""

        source = np.asarray(histories, dtype=np.float64)
        if (
            source.ndim != 4
            or source.shape[1:] != (
                20,
                len(self.graph.entities),
                self.ownership_mask.shape[1],
            )
            or isinstance(step, bool)
            or not isinstance(step, int)
            or step < 0
            or not np.all(np.isfinite(source))
        ):
            raise ValueError("SALT mask input is invalid")
        target = np.zeros(source.shape[:-1], dtype=np.bool_)
        protected = np.zeros_like(target)
        protected[:, -1, list(self.observed_entities)] = True
        block_rectangles = np.full(
            (len(source), 64, 5), -1, dtype=np.int64
        )
        fill_order = np.full(
            (len(source), self.target_count), -1, dtype=np.int64
        )
        durations = (4, 6, 8, 10, 12)
        for sample_position in range(len(source)):
            generator = np.random.default_rng(
                np.random.SeedSequence(
                    (self.seed, step, sample_position)
                )
            )
            attempts = 0
            block_position = 0
            while (
                int(np.sum(target[sample_position])) < self.target_count
                and attempts < 64
            ):
                attempts += 1
                duration = durations[
                    int(generator.integers(0, len(durations)))
                ]
                start = int(generator.integers(0, 21 - duration))
                block = self._entity_blocks[
                    int(generator.integers(0, len(self._entity_blocks)))
                ]
                proposal = np.zeros_like(target[sample_position])
                proposal[start : start + duration, list(block)] = True
                proposal[protected[sample_position]] = False
                additions = np.flatnonzero(
                    proposal & ~target[sample_position]
                )
                needed = self.target_count - int(
                    np.sum(target[sample_position])
                )
                if len(additions) == 0 or len(additions) > needed:
                    continue
                target[sample_position].flat[additions] = True
                target[sample_position][protected[sample_position]] = False
                block_rectangles[
                    sample_position, block_position
                ] = np.asarray((start, duration, *block), dtype=np.int64)
                block_position += 1
            remaining = self.target_count - int(
                np.sum(target[sample_position])
            )
            fill_position = 0
            while remaining:
                candidates = np.flatnonzero(
                    ~(target[sample_position] | protected[sample_position])
                )
                candidates = np.asarray(
                    [
                        value
                        for value in generator.permutation(candidates)
                        if self._extends_mask(
                            target[sample_position], int(value)
                        )
                    ],
                    dtype=np.int64,
                )
                if not len(candidates):
                    raise RuntimeError(
                        "SALT mask schedule cannot extend a connected block"
                    )
                chosen = int(candidates[0])
                target[sample_position].flat[chosen] = True
                fill_order[sample_position, fill_position] = chosen
                fill_position += 1
                remaining -= 1
            if int(np.sum(target[sample_position])) != self.target_count:
                raise RuntimeError("SALT mask schedule could not fill target")
        visible = ~target
        values = np.where(
            visible[..., None],
            np.where(
                self.ownership_mask[None, None], source, 0.0
            ),
            0.0,
        )
        return SaltMaskedTelemetry(
            values=values,
            visible_tokens=visible,
            target_tokens=target,
            block_rectangles=block_rectangles,
            fill_order=fill_order,
        )

    def _extends_mask(
        self, target: NDArray[np.bool_], flat_position: int
    ) -> bool:
        time_position, entity_position = np.unravel_index(
            flat_position, target.shape
        )
        if time_position > 0 and target[time_position - 1, entity_position]:
            return True
        if (
            time_position + 1 < target.shape[0]
            and target[time_position + 1, entity_position]
        ):
            return True
        return any(
            target[
                time_position,
                self.graph.entity_ids.index(neighbor_id),
            ]
            for neighbor_id in self.graph.neighboring_entity_ids(
                self.graph.entity_ids[entity_position]
            )
        )

    def _connected_block(self, root: int) -> Tuple[int, ...]:
        selected = [root]
        frontier = [
            self.graph.entity_ids.index(entity_id)
            for entity_id in self.graph.neighboring_entity_ids(
                self.graph.entity_ids[root]
            )
        ]
        while len(selected) < 3 and frontier:
            candidate = int(frontier.pop(0))
            if candidate in selected:
                continue
            selected.append(candidate)
            for entity_id in self.graph.neighboring_entity_ids(
                self.graph.entity_ids[candidate]
            ):
                neighbor = self.graph.entity_ids.index(entity_id)
                if neighbor not in selected and neighbor not in frontier:
                    frontier.append(neighbor)
        if len(selected) < 3:
            selected.extend(
                value
                for value in self.observed_entities
                if value not in selected
            )
        if len(selected) < 3:
            raise ValueError(
                "SALT multi-block masking requires three connected entities"
            )
        return tuple(sorted(selected[:3]))


class SaltTargetSchedule:
    """Aligned or no-fixed-point cyclic teacher-target assignment."""

    def __init__(self, alignment: str) -> None:
        if alignment not in ("aligned", "deranged"):
            raise ValueError("unsupported SALT target alignment")
        self.alignment = alignment

    def indices(
        self, pair_ids: Tuple[str, ...], *, step: int
    ) -> NDArray[np.int64]:
        """Return teacher-target row positions for one pair-blocked batch."""

        if (
            len(set(pair_ids)) != len(pair_ids)
            or isinstance(step, bool)
            or not isinstance(step, int)
            or step < 0
        ):
            raise ValueError("SALT target schedule input is invalid")
        count = len(pair_ids)
        if count < 1:
            raise ValueError("SALT target schedule requires pairs")
        positions = np.arange(count, dtype=np.int64)
        if self.alignment == "aligned":
            return positions
        if count < 2:
            raise ValueError("SALT derangement requires at least two pairs")
        shift = 1 + step % (count - 1)
        return np.roll(positions, shift)


@dataclass(frozen=True)
class SaltJepaConfig:
    """Frozen architecture and optimizer controls for SALT telemetry."""

    alignment: str = "aligned"
    width: int = 64
    block_count: int = 2
    head_count: int = 4
    feedforward_width: int = 128
    predictor_width: int = 256
    teacher_steps: int = 320
    student_steps: int = 1280
    expected_pair_count: int = 40
    learning_rate: float = 5e-4
    weight_decay: float = 5e-2
    warmup_steps: int = 80
    minimum_learning_rate: float = 5e-7
    teacher_seed: int = 23023
    decoder_seed: int = 24023
    student_seed: int = 25023
    predictor_seed: int = 26023
    anchor_seed: int = 27023
    mask_seed: int = 28023
    preprocessing_protocol: str = (
        "action_conditioned_jepa_topology_transfer_v1"
    )

    def __post_init__(self) -> None:
        if self.alignment not in ("aligned", "deranged"):
            raise ValueError("unsupported SALT alignment")
        integers = (
            self.width,
            self.block_count,
            self.head_count,
            self.feedforward_width,
            self.predictor_width,
            self.teacher_steps,
            self.student_steps,
            self.expected_pair_count,
            self.warmup_steps,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in integers
        ):
            raise ValueError("SALT integer controls are invalid")
        if self.width % self.head_count:
            raise ValueError("SALT heads must divide width")
        if not (
            self.learning_rate > 0.0
            and self.weight_decay >= 0.0
            and 0.0 < self.minimum_learning_rate <= self.learning_rate
        ):
            raise ValueError("SALT optimizer controls are invalid")
        seeds = (
            self.teacher_seed,
            self.decoder_seed,
            self.student_seed,
            self.predictor_seed,
            self.anchor_seed,
            self.mask_seed,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in seeds
        ):
            raise ValueError("SALT seeds must be integers")
        if not self.preprocessing_protocol:
            raise ValueError("SALT preprocessing identity cannot be empty")

    def to_dict(self) -> Dict[str, Any]:
        return dict(asdict(self))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SaltJepaConfig":
        if set(payload) != set(asdict(cls())):
            raise ValueError("SALT config schema is invalid")
        return cls(**dict(payload))


@dataclass(frozen=True)
class SaltPredictionDiagnostic:
    """Aligned masked-target evidence from a fitted SALT representation."""

    predicted_tokens: NDArray[np.float64]
    target_tokens: NDArray[np.float64]
    target_mask: NDArray[np.bool_]
    l1: float

    def __post_init__(self) -> None:
        if (
            self.predicted_tokens.ndim != 3
            or self.target_tokens.shape != self.predicted_tokens.shape
            or self.target_mask.shape != self.predicted_tokens.shape[:-1]
            or not np.all(np.isfinite(self.predicted_tokens))
            or not np.all(np.isfinite(self.target_tokens))
            or not np.isfinite(self.l1)
            or self.l1 < 0.0
        ):
            raise ValueError("SALT prediction diagnostic is invalid")


class SaltJepaRepresentation:
    """Restorable two-stage static-teacher telemetry representation."""

    kind = "salt_jepa_telemetry_representation"
    schema_version = 1

    def __init__(
        self, config: SaltJepaConfig = SaltJepaConfig()
    ) -> None:
        self.config = config
        self._graph: Optional[DeclaredTelemetryGraph] = None
        self._feature_names: Optional[Tuple[str, ...]] = None
        self._ownership_mask: Optional[NDArray[np.bool_]] = None
        self._teacher: Any = None
        self._decoder: Any = None
        self._student: Any = None
        self._predictor: Any = None
        self._teacher_metrics: Tuple[Mapping[str, float], ...] = ()
        self._student_metrics: Tuple[Mapping[str, float], ...] = ()
        self._teacher_sha256_before_student: Optional[str] = None
        self._teacher_sha256_after_student: Optional[str] = None

    @property
    def teacher_training_metrics(
        self,
    ) -> Tuple[Mapping[str, float], ...]:
        self._fitted_values()
        return self._teacher_metrics

    @property
    def student_training_metrics(
        self,
    ) -> Tuple[Mapping[str, float], ...]:
        self._fitted_values()
        return self._student_metrics

    @property
    def teacher_unchanged_during_student(self) -> bool:
        self._fitted_values()
        return (
            self._teacher_sha256_before_student
            == self._teacher_sha256_after_student
        )

    @property
    def inference_parameter_count(self) -> int:
        _, _, _, _, _, student, _ = self._fitted_values()
        return int(
            sum(parameter.numel() for parameter in student.parameters())
        )

    @property
    def training_only_parameter_count(self) -> int:
        _, _, _, teacher, decoder, _, predictor = self._fitted_values()
        return int(
            sum(
                parameter.numel()
                for module in (teacher, decoder, predictor)
                for parameter in module.parameters()
            )
        )

    def fit(
        self, windows: ActionConditionedWindows
    ) -> "SaltJepaRepresentation":
        """Fit the reconstructive teacher, freeze it, then fit the student."""

        torch = _require_torch()
        if len(set(windows.matched_pair_ids)) != self.config.expected_pair_count:
            raise ValueError("SALT fit pair count differs from its contract")
        if windows.histories.shape[1] != 20:
            raise ValueError("SALT requires 20-point contexts")
        ownership = fit_owned_feature_mask(windows)
        anchors = PairBlockedAnchorSchedule(
            windows, seed=self.config.anchor_seed
        )
        masks = SaltMaskSchedule(
            graph=windows.graph,
            ownership_mask=ownership,
            seed=self.config.mask_seed,
        )
        target_schedule = SaltTargetSchedule(self.config.alignment)
        teacher = _new_backbone(
            windows.graph,
            windows.histories.shape[-1],
            self.config,
            seed=self.config.teacher_seed,
        )
        decoder = _new_decoder(
            self.config, windows.histories.shape[-1]
        )
        teacher_optimizer = torch.optim.AdamW(
            list(teacher.parameters()) + list(decoder.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        positions = np.arange(
            20 * len(windows.graph.entities), dtype=np.int64
        )
        present = torch.ones(
            (
                self.config.expected_pair_count,
                20,
                len(windows.graph.entities),
            ),
            dtype=torch.bool,
        )
        owned_coordinates = torch.as_tensor(
            np.broadcast_to(
                ownership[None, None],
                (
                    self.config.expected_pair_count,
                    20,
                )
                + ownership.shape,
            ).reshape(
                self.config.expected_pair_count,
                20 * len(windows.graph.entities),
                windows.histories.shape[-1],
            ),
            dtype=torch.bool,
        )
        teacher_metrics = []
        teacher.train()
        decoder.train()
        for step in range(self.config.teacher_steps):
            _set_learning_rate(
                teacher_optimizer,
                _learning_rate(
                    self.config, step, self.config.teacher_steps
                ),
            )
            anchor = anchors.batch(step)
            source = windows.histories[anchor.indices]
            masked = masks.batch(source, step=step)
            teacher_optimizer.zero_grad(set_to_none=True)
            hidden = teacher(
                torch.as_tensor(masked.values, dtype=torch.float32),
                torch.as_tensor(
                    masked.visible_tokens, dtype=torch.bool
                ),
                present,
                positions,
            )
            prediction = decoder(hidden)
            raw_target = torch.as_tensor(
                np.where(
                    ownership[None, None], source, 0.0
                ).reshape(
                    len(source),
                    20 * len(windows.graph.entities),
                    source.shape[-1],
                ),
                dtype=torch.float32,
            )
            coordinate_mask = (
                torch.as_tensor(
                    masked.target_tokens.reshape(len(source), -1),
                    dtype=torch.bool,
                )[..., None]
                & owned_coordinates
            )
            loss = (
                (prediction[coordinate_mask] - raw_target[coordinate_mask])
                ** 2
            ).mean()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("SALT teacher training became non-finite")
            loss.backward()
            teacher_optimizer.step()
            teacher_metrics.append(
                {
                    "step": float(step + 1),
                    "learning_rate": float(
                        teacher_optimizer.param_groups[0]["lr"]
                    ),
                    "masked_reconstruction_mse": float(loss.detach()),
                    "independent_samples": float(len(source)),
                }
            )
        teacher.eval()
        decoder.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        teacher_before = _module_sha256(teacher)

        student = _new_backbone(
            windows.graph,
            windows.histories.shape[-1],
            self.config,
            seed=self.config.student_seed,
        )
        predictor = _new_predictor(self.config)
        student_optimizer = torch.optim.AdamW(
            list(student.parameters()) + list(predictor.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        student_metrics = []
        student.train()
        predictor.train()
        for step in range(self.config.student_steps):
            _set_learning_rate(
                student_optimizer,
                _learning_rate(
                    self.config, step, self.config.student_steps
                ),
            )
            schedule_step = self.config.teacher_steps + step
            anchor = anchors.batch(schedule_step)
            source = windows.histories[anchor.indices]
            masked = masks.batch(source, step=schedule_step)
            full_values = np.where(
                ownership[None, None], source, 0.0
            )
            student_optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                teacher_target = teacher(
                    torch.as_tensor(full_values, dtype=torch.float32),
                    present,
                    present,
                    positions,
                )
                target_indices = target_schedule.indices(
                    anchor.pair_ids, step=step
                )
                teacher_target = teacher_target[target_indices]
            student_hidden = student(
                torch.as_tensor(masked.values, dtype=torch.float32),
                torch.as_tensor(
                    masked.visible_tokens, dtype=torch.bool
                ),
                present,
                positions,
            )
            predicted = predictor(student_hidden)
            target_mask = torch.as_tensor(
                masked.target_tokens.reshape(len(source), -1),
                dtype=torch.bool,
            )
            loss = torch.abs(
                predicted[target_mask] - teacher_target[target_mask]
            ).mean()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("SALT student training became non-finite")
            loss.backward()
            student_optimizer.step()
            student_metrics.append(
                {
                    "step": float(step + 1),
                    "learning_rate": float(
                        student_optimizer.param_groups[0]["lr"]
                    ),
                    "masked_latent_l1": float(loss.detach()),
                    "independent_samples": float(len(source)),
                }
            )
        teacher_after = _module_sha256(teacher)
        if teacher_before != teacher_after:
            raise RuntimeError("SALT teacher changed during student fitting")
        self._graph = windows.graph
        self._feature_names = windows.state_feature_names
        self._ownership_mask = ownership.copy()
        self._teacher = teacher.eval()
        self._decoder = decoder.eval()
        self._student = student.eval()
        self._predictor = predictor.eval()
        self._teacher_metrics = tuple(teacher_metrics)
        self._student_metrics = tuple(student_metrics)
        self._teacher_sha256_before_student = teacher_before
        self._teacher_sha256_after_student = teacher_after
        return self

    def encode(
        self,
        histories: NDArray[np.float64],
        graph: DeclaredTelemetryGraph,
    ) -> EncodedTelemetry:
        """Encode complete contexts with the deployed student only."""

        return self._encode_with(histories, graph, teacher=False)

    def encode_teacher(
        self,
        histories: NDArray[np.float64],
        graph: DeclaredTelemetryGraph,
    ) -> EncodedTelemetry:
        """Encode complete contexts with the frozen stage-one teacher."""

        return self._encode_with(histories, graph, teacher=True)

    def diagnose_masked_prediction(
        self,
        histories: NDArray[np.float64],
        graph: DeclaredTelemetryGraph,
        *,
        step: int,
    ) -> SaltPredictionDiagnostic:
        """Evaluate aligned frozen-teacher targets under a declared mask."""

        (
            fitted_graph,
            feature_names,
            ownership,
            teacher,
            _,
            student,
            predictor,
        ) = self._fitted_values()
        source = _validate_histories(
            histories, graph, fitted_graph, feature_names
        )
        masked = SaltMaskSchedule(
            graph=fitted_graph,
            ownership_mask=ownership,
            seed=self.config.mask_seed,
        ).batch(source, step=step)
        torch = _require_torch()
        positions = np.arange(
            20 * len(fitted_graph.entities), dtype=np.int64
        )
        present = torch.ones(
            source.shape[:-1], dtype=torch.bool
        )
        with torch.no_grad():
            target = teacher(
                torch.as_tensor(
                    np.where(
                        ownership[None, None], source, 0.0
                    ),
                    dtype=torch.float32,
                ),
                present,
                present,
                positions,
            )
            hidden = student(
                torch.as_tensor(masked.values, dtype=torch.float32),
                torch.as_tensor(
                    masked.visible_tokens, dtype=torch.bool
                ),
                present,
                positions,
            )
            predicted = predictor(hidden)
        target_mask = masked.target_tokens.reshape(len(source), -1)
        predicted_array = predicted.cpu().numpy().astype(np.float64)
        target_array = target.cpu().numpy().astype(np.float64)
        return SaltPredictionDiagnostic(
            predicted_tokens=predicted_array,
            target_tokens=target_array,
            target_mask=target_mask,
            l1=float(
                np.mean(
                    np.abs(
                        predicted_array[target_mask]
                        - target_array[target_mask]
                    )
                )
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize all training and inference state needed to reproduce."""

        (
            graph,
            feature_names,
            ownership,
            teacher,
            decoder,
            student,
            predictor,
        ) = self._fitted_values()
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "config": self.config.to_dict(),
            "graph": graph.to_dict(),
            "feature_names": list(feature_names),
            "ownership_mask": ownership.astype(int).tolist(),
            "teacher_state": _module_state(teacher),
            "decoder_state": _module_state(decoder),
            "student_state": _module_state(student),
            "predictor_state": _module_state(predictor),
            "teacher_metrics": [
                dict(row) for row in self._teacher_metrics
            ],
            "student_metrics": [
                dict(row) for row in self._student_metrics
            ],
            "teacher_sha256_before_student": (
                self._teacher_sha256_before_student
            ),
            "teacher_sha256_after_student": (
                self._teacher_sha256_after_student
            ),
            "inference_parameter_count": self.inference_parameter_count,
            "training_only_parameter_count": (
                self.training_only_parameter_count
            ),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "SaltJepaRepresentation":
        """Restore a complete teacher/student SALT artifact."""

        if (
            payload.get("schema_version") != cls.schema_version
            or payload.get("kind") != cls.kind
        ):
            raise ValueError("unsupported SALT artifact")
        config = SaltJepaConfig.from_dict(dict(payload["config"]))
        graph = DeclaredTelemetryGraph.from_dict(dict(payload["graph"]))
        feature_names = tuple(
            str(value) for value in payload["feature_names"]
        )
        ownership = np.asarray(
            payload["ownership_mask"], dtype=np.bool_
        )
        model = cls(config)
        teacher = _new_backbone(
            graph, len(feature_names), config, seed=config.teacher_seed
        )
        decoder = _new_decoder(config, len(feature_names))
        student = _new_backbone(
            graph, len(feature_names), config, seed=config.student_seed
        )
        predictor = _new_predictor(config)
        _restore_module(teacher, dict(payload["teacher_state"]))
        _restore_module(decoder, dict(payload["decoder_state"]))
        _restore_module(student, dict(payload["student_state"]))
        _restore_module(predictor, dict(payload["predictor_state"]))
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        model._graph = graph
        model._feature_names = feature_names
        model._ownership_mask = ownership
        model._teacher = teacher.eval()
        model._decoder = decoder.eval()
        model._student = student.eval()
        model._predictor = predictor.eval()
        model._teacher_metrics = tuple(
            {
                str(key): float(value)
                for key, value in dict(row).items()
            }
            for row in payload["teacher_metrics"]
        )
        model._student_metrics = tuple(
            {
                str(key): float(value)
                for key, value in dict(row).items()
            }
            for row in payload["student_metrics"]
        )
        model._teacher_sha256_before_student = str(
            payload["teacher_sha256_before_student"]
        )
        model._teacher_sha256_after_student = str(
            payload["teacher_sha256_after_student"]
        )
        if (
            model.inference_parameter_count
            != int(payload["inference_parameter_count"])
            or model.training_only_parameter_count
            != int(payload["training_only_parameter_count"])
            or not model.teacher_unchanged_during_student
            or _module_sha256(teacher)
            != model._teacher_sha256_after_student
        ):
            raise ValueError("SALT artifact identity mismatch")
        return model

    def _encode_with(
        self,
        histories: NDArray[np.float64],
        graph: DeclaredTelemetryGraph,
        *,
        teacher: bool,
    ) -> EncodedTelemetry:
        (
            fitted_graph,
            feature_names,
            ownership,
            teacher_network,
            _,
            student_network,
            _,
        ) = self._fitted_values()
        source = _validate_histories(
            histories, graph, fitted_graph, feature_names
        )
        network = teacher_network if teacher else student_network
        torch = _require_torch()
        visible = torch.ones(source.shape[:-1], dtype=torch.bool)
        with torch.no_grad():
            hidden = network(
                torch.as_tensor(
                    np.where(
                        ownership[None, None], source, 0.0
                    ),
                    dtype=torch.float32,
                ),
                visible,
                visible,
                np.arange(
                    20 * len(fitted_graph.entities), dtype=np.int64
                ),
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
        module_hash = _module_sha256(network)
        return EncodedTelemetry(
            tokens=tokens,
            entity_ids=fitted_graph.entity_ids,
            ownership_mask=ownership.copy(),
            observation_mask=ownership.copy(),
            content_sha256=module_hash,
            graph_sha256=_canonical_sha256(fitted_graph.to_dict()),
            state_schema_sha256=_canonical_sha256(
                {"feature_names": list(feature_names)}
            ),
            preprocessing_sha256=_canonical_sha256(
                {"protocol": self.config.preprocessing_protocol}
            ),
            encoder_sha256=module_hash,
        )

    def _fitted_values(
        self,
    ) -> Tuple[
        DeclaredTelemetryGraph,
        Tuple[str, ...],
        NDArray[np.bool_],
        Any,
        Any,
        Any,
        Any,
    ]:
        if (
            self._graph is None
            or self._feature_names is None
            or self._ownership_mask is None
            or self._teacher is None
            or self._decoder is None
            or self._student is None
            or self._predictor is None
        ):
            raise RuntimeError("SALT representation is not fitted")
        return (
            self._graph,
            self._feature_names,
            self._ownership_mask,
            self._teacher,
            self._decoder,
            self._student,
            self._predictor,
        )


def assess_salt_jepa_gates(
    *,
    forecast_scores: Mapping[str, Mapping[str, Mapping[str, float]]],
    raw_scores: Mapping[str, Mapping[str, float]],
    masked_latent_l1: Mapping[str, Mapping[str, float]],
    state_probes: Mapping[str, Mapping[str, Mapping[str, Any]]],
    attribution: Mapping[str, Mapping[str, float]],
    action_sanity: Mapping[str, Mapping[str, float]],
    restoration_max_abs: Mapping[str, float],
    protocol_checks: Mapping[str, bool],
    parameter_counts: Mapping[str, Mapping[str, int]],
    teacher_unchanged: Mapping[str, bool],
    transfer_pair_errors: Mapping[str, Mapping[str, float]],
    deployed_bundle_bytes: int,
    median_latency_ms: float,
) -> Dict[str, Any]:
    """Purely recompute every frozen SALT safety and value gate."""

    candidate = forecast_scores["salt_jepa"]
    raw_selection = raw_scores["selection"]
    raw_transfer = raw_scores["transfer_evaluation"]
    candidate_selection = candidate["selection"]
    candidate_transfer = candidate["transfer_evaluation"]
    counts_match = (
        parameter_counts["salt_jepa"]
        == parameter_counts["deranged_salt_jepa"]
    )
    restoration_passed = all(
        np.isfinite(value) and value <= 1e-6
        for value in restoration_max_abs.values()
    )
    safety_gates = {
        "evidence_arrays_are_finite": bool(
            protocol_checks.get("evidence_arrays_are_finite", False)
        ),
        "pair_and_trajectory_roles_are_disjoint": bool(
            protocol_checks.get(
                "pair_and_trajectory_roles_are_disjoint", False
            )
        ),
        "capacity_is_matched": counts_match,
        "teachers_are_unchanged": all(teacher_unchanged.values()),
        "restoration_within_1e_6": restoration_passed,
        "public_inference_is_causal": bool(
            protocol_checks.get("public_inference_is_causal", False)
        ),
        "mask_schedule_is_valid": bool(
            protocol_checks.get("mask_schedule_is_valid", False)
        ),
        "selection_only_ridge_choice_recomputes": bool(
            protocol_checks.get(
                "selection_only_ridge_choice_recomputes", False
            )
        ),
        "selection_safety_status_recomputes": bool(
            protocol_checks.get(
                "selection_safety_status_recomputes", False
            )
        ),
        "capacity_metadata_recomputes": bool(
            protocol_checks.get("capacity_metadata_recomputes", False)
        ),
        "teacher_metadata_recomputes": bool(
            protocol_checks.get("teacher_metadata_recomputes", False)
        ),
        "causality_metadata_recomputes": bool(
            protocol_checks.get("causality_metadata_recomputes", False)
        ),
        "deployed_bundle_metadata_recomputes": bool(
            protocol_checks.get(
                "deployed_bundle_metadata_recomputes", False
            )
        ),
        "latency_metadata_recomputes": bool(
            protocol_checks.get("latency_metadata_recomputes", False)
        ),
        "selection_overall_within_1_05_raw": (
            candidate_selection["overall_mse"]
            <= 1.05 * raw_selection["overall_mse"]
        ),
        "selection_action_within_1_05_raw": (
            candidate_selection["action_overlap_mse"]
            <= 1.05 * raw_selection["action_overlap_mse"]
        ),
        "transfer_overall_within_1_05_raw": (
            candidate_transfer["overall_mse"]
            <= 1.05 * raw_transfer["overall_mse"]
        ),
        "transfer_action_within_1_05_raw": (
            candidate_transfer["action_overlap_mse"]
            <= 1.05 * raw_transfer["action_overlap_mse"]
        ),
        "action_and_target_hit_at_1": (
            attribution["salt_jepa"]["action_and_target_hit_at_1"]
            >= 0.95
        ),
        "no_action_specificity": (
            attribution["salt_jepa"]["no_action_specificity"] == 1.0
        ),
        "correct_action_sanity": (
            action_sanity["salt_jepa"][
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
            np.isfinite(median_latency_ms) and median_latency_ms > 0.0
        ),
    }
    mechanism_gates = {
        "masked_latent_advantage": all(
            masked_latent_l1["salt_jepa"][role]
            <= 0.90 * masked_latent_l1["deranged_salt_jepa"][role]
            for role in ("selection", "transfer_evaluation")
        )
    }
    learned_controls = (
        "deranged_salt_jepa",
        "reconstructive_teacher",
        "matched_pca",
    )
    candidate_pair_errors = transfer_pair_errors["salt_jepa"]
    teacher_pair_errors = transfer_pair_errors["reconstructive_teacher"]
    common_pairs = sorted(
        set(candidate_pair_errors) & set(teacher_pair_errors)
    )
    teacher_win_fraction = (
        float(
            np.mean(
                [
                    candidate_pair_errors[pair_id]
                    < teacher_pair_errors[pair_id]
                    for pair_id in common_pairs
                ]
            )
        )
        if common_pairs
        else 0.0
    )
    candidate_state = state_probes["salt_jepa"][
        "transfer_evaluation"
    ]
    teacher_state = state_probes["reconstructive_teacher"][
        "transfer_evaluation"
    ]
    candidate_entities = candidate_state["entities"]
    teacher_entities = teacher_state["entities"]
    comparable_entities = {
        entity_id
        for entity_id, values in candidate_entities.items()
        if values["nrmse"] is not None
    }
    teacher_comparable_entities = {
        entity_id
        for entity_id, values in teacher_entities.items()
        if values["nrmse"] is not None
    }
    state_not_worse = (
        candidate_state["aggregate_nrmse"]
        <= teacher_state["aggregate_nrmse"]
        and comparable_entities == teacher_comparable_entities
        and all(
            candidate_entities[entity_id]["nrmse"]
            <= teacher_entities[entity_id]["nrmse"]
            for entity_id in comparable_entities
        )
    )
    value_gates = {
        "transfer_effect_beats_teacher_deranged_and_raw_by_10_percent": (
            candidate_transfer["downstream_effect_mse"]
            <= 0.90
            * min(
                forecast_scores["reconstructive_teacher"][
                    "transfer_evaluation"
                ]["downstream_effect_mse"],
                forecast_scores["deranged_salt_jepa"][
                    "transfer_evaluation"
                ]["downstream_effect_mse"],
                raw_transfer["downstream_effect_mse"],
            )
        ),
        "selection_effect_is_best": all(
            candidate_selection["downstream_effect_mse"]
            < forecast_scores[name]["selection"][
                "downstream_effect_mse"
            ]
            for name in learned_controls
        ),
        "transfer_teacher_pair_win_fraction": (
            teacher_win_fraction >= 0.60
        ),
        "state_retention_not_worse_than_teacher": state_not_worse,
    }
    passed = (
        all(safety_gates.values())
        and all(mechanism_gates.values())
        and all(value_gates.values())
    )
    return {
        "schema_version": 2,
        "experiment": "salt_jepa_telemetry_tracer_v2",
        "safety_gates": safety_gates,
        "mechanism_gates": mechanism_gates,
        "value_gates": value_gates,
        "candidate_teacher_pair_win_fraction": teacher_win_fraction,
        "passed": passed,
        "decision": (
            "advance_salt_jepa_to_fixed_seed_robustness"
            if passed
            else "reject_salt_jepa_telemetry_recipe"
        ),
    }


def _new_backbone(
    graph: DeclaredTelemetryGraph,
    feature_count: int,
    config: SaltJepaConfig,
    *,
    seed: int,
) -> Any:
    torch = _require_torch()
    backbone_config = CompleteLejepaConfig(
        width=config.width,
        block_count=config.block_count,
        head_count=config.head_count,
        feedforward_width=config.feedforward_width,
        projector_width=config.predictor_width,
        steps=1,
        expected_pair_count=config.expected_pair_count,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=1,
        minimum_learning_rate=config.minimum_learning_rate,
        preprocessing_protocol=config.preprocessing_protocol,
    )
    state = torch.random.get_rng_state()
    try:
        torch.manual_seed(seed)
        return build_complete_lejepa_backbone(
            feature_count=feature_count,
            graph=graph,
            config=backbone_config,
        )
    finally:
        torch.random.set_rng_state(state)


def _new_decoder(config: SaltJepaConfig, feature_count: int) -> Any:
    torch = _require_torch()
    return _seeded_module(
        config.decoder_seed,
        lambda: torch.nn.Sequential(
            torch.nn.Linear(config.width, config.width),
            torch.nn.GELU(),
            torch.nn.Linear(config.width, feature_count),
        ),
    )


def _new_predictor(config: SaltJepaConfig) -> Any:
    torch = _require_torch()
    return _seeded_module(
        config.predictor_seed,
        lambda: torch.nn.Sequential(
            torch.nn.Linear(config.width, config.predictor_width),
            torch.nn.GELU(),
            torch.nn.Linear(config.predictor_width, config.width),
        ),
    )


def _seeded_module(seed: int, factory: Any) -> Any:
    torch = _require_torch()
    state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    module = factory()
    torch.random.set_rng_state(state)
    return module


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
            len(fitted_graph.entities),
            len(feature_names),
        )
        or not np.all(np.isfinite(source))
    ):
        raise ValueError("SALT encoding input is invalid")
    return source


def _learning_rate(
    config: SaltJepaConfig, step: int, stage_steps: int
) -> float:
    warmup = min(config.warmup_steps, stage_steps)
    if step < warmup:
        return config.learning_rate * float(step + 1) / float(warmup)
    remaining = stage_steps - warmup
    if remaining <= 1:
        return config.minimum_learning_rate
    progress = float(step - warmup) / float(remaining - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.minimum_learning_rate + (
        config.learning_rate - config.minimum_learning_rate
    ) * cosine


def _set_learning_rate(optimizer: Any, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def _module_state(module: Any) -> Dict[str, Any]:
    return {
        str(key): value.detach().cpu().tolist()
        for key, value in module.state_dict().items()
    }


def _restore_module(module: Any, payload: Mapping[str, Any]) -> None:
    torch = _require_torch()
    current = module.state_dict()
    if set(payload) != set(current):
        raise ValueError("SALT module state schema is invalid")
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
        ).encode("utf-8")
    ).hexdigest()


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("SALT-JEPA requires PyTorch") from error
    torch.set_num_threads(1)
    return torch
