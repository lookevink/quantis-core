"""Action-conditioned graph trajectories, rollout, and attribution."""

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .graph_telemetry import DeclaredTelemetryGraph

ACTION_KINDS = (
    "worker_pause",
    "postgres_lock",
    "redis_enqueue_delay",
    "redis_dequeue_delay",
    "api_rejection",
)
SPLITS = ("training", "validation", "confirmation")


@dataclass(frozen=True)
class InterventionAction:
    """One external, reversible command applied to a graph entity."""

    action_id: str
    action_kind: str
    target_entity: str
    start_index: int
    stop_index: int
    magnitude: float
    parameter_schema_version: int = 1
    magnitude_unit: str = "normalized"
    effect_feature: str = "pressure"
    effect_direction: str = "increase"
    minimum_effect: float = 0.1
    recovery_tolerance: float = 0.2

    def __post_init__(self) -> None:
        if not self.action_id:
            raise ValueError("action_id cannot be empty")
        if self.action_kind not in ACTION_KINDS:
            raise ValueError(
                f"unsupported action kind: {self.action_kind}"
            )
        if not self.target_entity:
            raise ValueError("action target cannot be empty")
        if (
            isinstance(self.start_index, bool)
            or isinstance(self.stop_index, bool)
            or self.start_index < 0
            or self.stop_index <= self.start_index
        ):
            raise ValueError("action interval is invalid")
        if (
            isinstance(self.magnitude, bool)
            or not np.isfinite(self.magnitude)
            or not 0.0 < self.magnitude
        ):
            raise ValueError(
                "action magnitude must be finite and positive"
            )
        if (
            self.parameter_schema_version != 1
            or not self.magnitude_unit
            or not self.effect_feature
            or self.effect_direction not in {"increase", "decrease"}
            or isinstance(self.minimum_effect, bool)
            or not np.isfinite(self.minimum_effect)
            or self.minimum_effect <= 0.0
            or isinstance(self.recovery_tolerance, bool)
            or not np.isfinite(self.recovery_tolerance)
            or not 0.0 <= self.recovery_tolerance <= 1.0
        ):
            raise ValueError(
                "action parameter/effect/recovery schema is invalid"
            )

    @property
    def duration(self) -> int:
        """Return the number of active logical windows."""

        return self.stop_index - self.start_index

    def to_dict(self) -> Dict[str, Any]:
        """Return the canonical action representation."""

        return {
            "action_id": self.action_id,
            "action_kind": self.action_kind,
            "target_entity": self.target_entity,
            "start_index": self.start_index,
            "stop_index": self.stop_index,
            "magnitude": self.magnitude,
            "parameter_schema_version": (
                self.parameter_schema_version
            ),
            "magnitude_unit": self.magnitude_unit,
            "effect_feature": self.effect_feature,
            "effect_direction": self.effect_direction,
            "minimum_effect": self.minimum_effect,
            "recovery_tolerance": self.recovery_tolerance,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "InterventionAction":
        """Restore one action from its canonical representation."""

        if set(payload) != {
            "action_id",
            "action_kind",
            "target_entity",
            "start_index",
            "stop_index",
            "magnitude",
            "parameter_schema_version",
            "magnitude_unit",
            "effect_feature",
            "effect_direction",
            "minimum_effect",
            "recovery_tolerance",
        }:
            raise ValueError("action schema is invalid")
        if (
            not isinstance(payload["action_id"], str)
            or not isinstance(payload["action_kind"], str)
            or not isinstance(payload["target_entity"], str)
            or not _is_integer(payload["start_index"])
            or not _is_integer(payload["stop_index"])
            or isinstance(payload["magnitude"], bool)
            or not isinstance(payload["magnitude"], (int, float))
            or not _is_integer(
                payload["parameter_schema_version"]
            )
            or not isinstance(payload["magnitude_unit"], str)
            or not isinstance(payload["effect_feature"], str)
            or not isinstance(payload["effect_direction"], str)
            or isinstance(payload["minimum_effect"], bool)
            or not isinstance(
                payload["minimum_effect"], (int, float)
            )
            or isinstance(payload["recovery_tolerance"], bool)
            or not isinstance(
                payload["recovery_tolerance"], (int, float)
            )
        ):
            raise ValueError("action field types are invalid")
        return cls(
            action_id=payload["action_id"],
            action_kind=payload["action_kind"],
            target_entity=payload["target_entity"],
            start_index=payload["start_index"],
            stop_index=payload["stop_index"],
            magnitude=float(payload["magnitude"]),
            parameter_schema_version=payload[
                "parameter_schema_version"
            ],
            magnitude_unit=payload["magnitude_unit"],
            effect_feature=payload["effect_feature"],
            effect_direction=payload["effect_direction"],
            minimum_effect=float(payload["minimum_effect"]),
            recovery_tolerance=float(
                payload["recovery_tolerance"]
            ),
        )


@dataclass(frozen=True)
class ActionConditionedCaseManifest:
    """One content-addressed action trajectory and matched-pair identity."""

    case_id: str
    matched_pair_id: str
    split: str
    point_count: int
    logical_window_period_nano: int
    topology_id: str
    worker_replicas: int
    workload_seed: int
    intervention_seed: int
    actions: Tuple[InterventionAction, ...] = ()
    schema_version: int = 3

    def __post_init__(self) -> None:
        if self.schema_version != 3:
            raise ValueError("unsupported action manifest schema_version")
        if not self.case_id or not self.matched_pair_id:
            raise ValueError("case and matched-pair ids cannot be empty")
        if self.split not in SPLITS:
            raise ValueError(f"unsupported action split: {self.split}")
        if (
            not _is_integer(self.point_count)
            or self.point_count < 1
            or not _is_integer(self.logical_window_period_nano)
            or self.logical_window_period_nano < 1
        ):
            raise ValueError("trajectory dimensions must be positive")
        if not self.topology_id:
            raise ValueError("topology_id cannot be empty")
        if (
            not _is_integer(self.worker_replicas)
            or self.worker_replicas < 1
        ):
            raise ValueError("worker_replicas must be positive")
        if (
            not _is_integer(self.workload_seed)
            or not _is_integer(self.intervention_seed)
        ):
            raise ValueError("manifest seeds must be integers")
        action_ids = tuple(action.action_id for action in self.actions)
        if len(set(action_ids)) != len(action_ids):
            raise ValueError("action ids must be unique within a trajectory")
        ordered_actions = sorted(
            self.actions, key=lambda action: action.start_index
        )
        for action_index, action in enumerate(ordered_actions):
            if action.stop_index > self.point_count:
                raise ValueError(
                    "action interval is outside the trajectory"
                )
            if (
                action_index
                and action.start_index
                < ordered_actions[action_index - 1].stop_index
            ):
                raise ValueError(
                    "overlapping actions are not supported in Phase 0"
                )

    def to_dict(self) -> Dict[str, Any]:
        """Return the canonical manifest representation."""

        return {
            "schema_version": self.schema_version,
            "kind": "action_conditioned_case_manifest",
            "case_id": self.case_id,
            "matched_pair_id": self.matched_pair_id,
            "split": self.split,
            "point_count": self.point_count,
            "logical_window_period_nano": (
                self.logical_window_period_nano
            ),
            "topology_id": self.topology_id,
            "worker_replicas": self.worker_replicas,
            "workload_seed": self.workload_seed,
            "intervention_seed": self.intervention_seed,
            "actions": [
                action.to_dict()
                for action in sorted(
                    self.actions,
                    key=lambda value: (
                        value.start_index,
                        value.stop_index,
                        value.action_id,
                    ),
                )
            ],
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "ActionConditionedCaseManifest":
        """Restore and validate a canonical manifest."""

        if set(payload) != {
            "schema_version",
            "kind",
            "case_id",
            "matched_pair_id",
            "split",
            "point_count",
            "logical_window_period_nano",
            "topology_id",
            "worker_replicas",
            "workload_seed",
            "intervention_seed",
            "actions",
        }:
            raise ValueError("action manifest schema is invalid")
        if (
            payload.get("schema_version") != 3
            or payload.get("kind")
            != "action_conditioned_case_manifest"
            or not isinstance(payload["case_id"], str)
            or not isinstance(payload["matched_pair_id"], str)
            or not isinstance(payload["split"], str)
            or not _is_integer(payload["point_count"])
            or not _is_integer(
                payload["logical_window_period_nano"]
            )
            or not isinstance(payload["topology_id"], str)
            or not _is_integer(payload["worker_replicas"])
            or not _is_integer(payload["workload_seed"])
            or not _is_integer(payload["intervention_seed"])
            or not isinstance(payload["actions"], list)
        ):
            raise ValueError("action manifest field types are invalid")
        raw_actions = payload["actions"]
        if any(not isinstance(raw, dict) for raw in raw_actions):
            raise ValueError("action manifest actions are invalid")
        return cls(
            case_id=payload["case_id"],
            matched_pair_id=payload["matched_pair_id"],
            split=payload["split"],
            point_count=payload["point_count"],
            logical_window_period_nano=(
                payload["logical_window_period_nano"]
            ),
            topology_id=payload["topology_id"],
            worker_replicas=payload["worker_replicas"],
            workload_seed=payload["workload_seed"],
            intervention_seed=payload["intervention_seed"],
            actions=tuple(
                InterventionAction.from_dict(raw)
                for raw in raw_actions
            ),
            schema_version=payload["schema_version"],
        )

    def canonical_sha256(self) -> str:
        """Hash the exact semantic manifest."""

        encoded = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ActionConditionedRun:
    """One observable trajectory plus separate controls and action truth."""

    manifest: ActionConditionedCaseManifest
    graph: DeclaredTelemetryGraph
    observations: NDArray[np.float64]
    controls: NDArray[np.float64]
    state_feature_names: Tuple[str, ...]
    control_feature_names: Tuple[str, ...]

    def __post_init__(self) -> None:
        point_count = self.manifest.point_count
        entity_count = len(self.graph.entities)
        if self.observations.shape != (
            point_count,
            entity_count,
            len(self.state_feature_names),
        ):
            raise ValueError(
                "run observations do not align with manifest and graph"
            )
        if self.controls.shape != (
            point_count,
            len(self.control_feature_names),
        ):
            raise ValueError("run controls do not align with manifest")
        if (
            not self.state_feature_names
            or len(set(self.state_feature_names))
            != len(self.state_feature_names)
            or not self.control_feature_names
            or len(set(self.control_feature_names))
            != len(self.control_feature_names)
        ):
            raise ValueError("run feature names must be unique and nonempty")
        if not (
            np.all(np.isfinite(self.observations))
            and np.all(np.isfinite(self.controls))
        ):
            raise ValueError("run values must be finite")
        entity_ids = set(self.graph.entity_ids)
        if any(
            action.target_entity not in entity_ids
            for action in self.manifest.actions
        ):
            raise ValueError("action target is not in the declared graph")


def validate_matched_action_pairs(
    runs: Sequence[ActionConditionedRun],
) -> Dict[str, Any]:
    """Validate treatment/control pairing before scientific compilation."""

    if not runs:
        raise ValueError("matched-pair validation requires runs")
    case_ids = tuple(run.manifest.case_id for run in runs)
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("matched pair case ids must be unique")
    grouped: Dict[str, List[ActionConditionedRun]] = {}
    action_ids = [
        action.action_id
        for run in runs
        for action in run.manifest.actions
    ]
    if len(set(action_ids)) != len(action_ids):
        raise ValueError(
            "action ids must be unique across matched captures"
        )
    effect_check_count = 0
    recovery_check_count = 0
    for run in runs:
        grouped.setdefault(
            run.manifest.matched_pair_id, []
        ).append(run)
    for pair_id, pair in grouped.items():
        if len(pair) != 2:
            raise ValueError(
                f"matched pair {pair_id!r} must contain two runs"
            )
        treatment = [run for run in pair if run.manifest.actions]
        controls = [
            run for run in pair if not run.manifest.actions
        ]
        if len(treatment) != 1 or len(controls) != 1:
            raise ValueError(
                f"matched pair {pair_id!r} needs treatment and control"
            )
        action_run = treatment[0]
        control_run = controls[0]
        comparable_manifest_fields = (
            action_run.manifest.split
            == control_run.manifest.split,
            action_run.manifest.point_count
            == control_run.manifest.point_count,
            action_run.manifest.logical_window_period_nano
            == control_run.manifest.logical_window_period_nano,
            action_run.manifest.topology_id
            == control_run.manifest.topology_id,
            action_run.manifest.worker_replicas
            == control_run.manifest.worker_replicas,
            action_run.manifest.workload_seed
            == control_run.manifest.workload_seed,
            action_run.manifest.intervention_seed
            == control_run.manifest.intervention_seed,
            action_run.graph.to_dict()
            == control_run.graph.to_dict(),
            action_run.state_feature_names
            == control_run.state_feature_names,
            action_run.control_feature_names
            == control_run.control_feature_names,
        )
        if not all(comparable_manifest_fields):
            raise ValueError(
                f"matched pair {pair_id!r} metadata does not align"
            )
        if not np.array_equal(
            action_run.controls, control_run.controls
        ):
            raise ValueError(
                f"matched pair {pair_id!r} control schedule drifted"
            )
        for action in action_run.manifest.actions:
            try:
                feature_position = (
                    action_run.state_feature_names.index(
                        action.effect_feature
                    )
                )
            except ValueError as error:
                raise ValueError(
                    f"matched pair {pair_id!r} effect feature "
                    "is not observed"
                ) from error
            entity_position = action_run.graph.entity_ids.index(
                action.target_entity
            )
            delta = (
                action_run.observations[
                    :, entity_position, feature_position
                ]
                - control_run.observations[
                    :, entity_position, feature_position
                ]
            )
            active_effect = float(
                np.mean(
                    delta[
                        action.start_index
                        + 1 : action.stop_index
                        + 1
                    ]
                )
            )
            signed_effect = (
                active_effect
                if action.effect_direction == "increase"
                else -active_effect
            )
            if signed_effect < action.minimum_effect:
                raise ValueError(
                    f"matched pair {pair_id!r} raw effect check failed"
                )
            effect_check_count += 1
            recovery_start = max(
                action.stop_index + 1,
                action_run.manifest.point_count - 3,
            )
            recovery_delta = delta[recovery_start:]
            if (
                not len(recovery_delta)
                or float(np.max(np.abs(recovery_delta)))
                > action.recovery_tolerance
            ):
                raise ValueError(
                    f"matched pair {pair_id!r} recovery check failed"
                )
            recovery_check_count += 1
    return {
        "schema_version": 1,
        "kind": "matched_action_pair_validation",
        "run_count": len(runs),
        "pair_count": len(grouped),
        "effect_check_count": effect_check_count,
        "recovery_check_count": recovery_check_count,
        "splits": sorted(
            {run.manifest.split for run in runs}
        ),
    }


@dataclass(frozen=True)
class ActionConditionedWindows:
    """Normalized histories and aligned future conditioning/targets."""

    histories: NDArray[np.float64]
    future_states: NDArray[np.float64]
    future_controls: NDArray[np.float64]
    future_actions: NDArray[np.float64]
    trajectory_ids: Tuple[str, ...]
    matched_pair_ids: Tuple[str, ...]
    transition_indices: NDArray[np.int64]
    entity_names: Tuple[str, ...]
    state_feature_names: Tuple[str, ...]
    control_feature_names: Tuple[str, ...]
    action_feature_names: Tuple[str, ...]
    graph: DeclaredTelemetryGraph

    def __post_init__(self) -> None:
        sample_count = len(self.histories)
        if (
            self.histories.ndim != 4
            or self.future_states.ndim != 4
            or self.future_controls.ndim != 3
            or self.future_actions.ndim != 4
            or len(self.future_states) != sample_count
            or len(self.future_controls) != sample_count
            or len(self.future_actions) != sample_count
            or len(self.trajectory_ids) != sample_count
            or len(self.matched_pair_ids) != sample_count
            or self.transition_indices.shape != (sample_count,)
        ):
            raise ValueError("action-conditioned windows do not align")
        if not (
            self.future_states.shape[1]
            == self.future_controls.shape[1]
            == self.future_actions.shape[1]
        ):
            raise ValueError(
                "action-conditioned future horizons do not align"
            )
        if (
            self.histories.shape[2]
            != len(self.entity_names)
            or self.entity_names != self.graph.entity_ids
            or self.future_states.shape[2:]
            != self.histories.shape[2:]
            or self.future_controls.shape[2]
            != len(self.control_feature_names)
            or self.future_actions.shape[2:]
            != (
                len(self.entity_names),
                len(self.action_feature_names),
            )
        ):
            raise ValueError("action-conditioned schemas do not align")
        if any(
            not np.all(np.isfinite(values))
            for values in (
                self.histories,
                self.future_states,
                self.future_controls,
                self.future_actions,
            )
        ):
            raise ValueError("action-conditioned windows must be finite")

    @property
    def semantic_schema_sha256(self) -> str:
        """Hash ordered graph, state, control, and action semantics."""

        return _semantic_schema_sha256(
            self.graph,
            self.state_feature_names,
            self.control_feature_names,
            self.action_feature_names,
        )


class ActionTrajectoryCompiler:
    """Fit training-only transforms and align action-driven transitions."""

    kind = "action_trajectory_compiler"
    schema_version = 1

    def __init__(
        self, *, context_length: int, rollout_horizon: int
    ) -> None:
        if (
            isinstance(context_length, bool)
            or context_length < 1
            or isinstance(rollout_horizon, bool)
            or rollout_horizon < 1
        ):
            raise ValueError(
                "context length and rollout horizon must be positive"
            )
        self.context_length = context_length
        self.rollout_horizon = rollout_horizon
        self._state_center: Optional[NDArray[np.float64]] = None
        self._state_scale: Optional[NDArray[np.float64]] = None
        self._control_center: Optional[NDArray[np.float64]] = None
        self._control_scale: Optional[NDArray[np.float64]] = None
        self._action_scales: Optional[NDArray[np.float64]] = None
        self._graph: Optional[DeclaredTelemetryGraph] = None
        self._state_feature_names: Optional[Tuple[str, ...]] = None
        self._control_feature_names: Optional[Tuple[str, ...]] = None
        self._training_pair_ids: Tuple[str, ...] = ()

    @property
    def action_feature_names(self) -> Tuple[str, ...]:
        """Return the fixed action-conditioning vocabulary."""

        return (
            "no_action",
            "applicable",
            *(f"kind:{name}" for name in ACTION_KINDS),
            "phase:start",
            "phase:active",
            "phase:stop",
            "magnitude",
            "elapsed_fraction",
            "remaining_fraction",
        )

    def fit(
        self, runs: Sequence[ActionConditionedRun]
    ) -> "ActionTrajectoryCompiler":
        """Fit feature transforms using training trajectories only."""

        if not runs:
            raise ValueError("compiler fit requires training runs")
        if any(run.manifest.split != "training" for run in runs):
            raise ValueError("compiler fit accepts training runs only")
        self._validate_run_schemas(runs)
        states = np.concatenate(
            [run.observations for run in runs], axis=0
        )
        controls = np.concatenate(
            [run.controls for run in runs], axis=0
        )
        self._state_center = np.mean(states, axis=(0, 1))
        self._state_scale = _nonzero_scale(
            np.std(states, axis=(0, 1))
        )
        self._control_center = np.mean(controls, axis=0)
        self._control_scale = _nonzero_scale(
            np.std(controls, axis=0)
        )
        action_scales = np.ones(len(ACTION_KINDS), dtype=np.float64)
        for kind_index, action_kind in enumerate(ACTION_KINDS):
            magnitudes = [
                action.magnitude
                for run in runs
                for action in run.manifest.actions
                if action.action_kind == action_kind
            ]
            if magnitudes:
                action_scales[kind_index] = max(magnitudes)
        self._action_scales = action_scales
        self._graph = runs[0].graph
        self._state_feature_names = runs[0].state_feature_names
        self._control_feature_names = runs[0].control_feature_names
        self._training_pair_ids = tuple(
            sorted(
                {
                    run.manifest.matched_pair_id
                    for run in runs
                }
            )
        )
        return self

    def transform(
        self, runs: Sequence[ActionConditionedRun]
    ) -> ActionConditionedWindows:
        """Compile windows where action at t predicts observed state t+1."""

        (
            state_center,
            state_scale,
            control_center,
            control_scale,
            action_scales,
            graph,
        ) = self._fitted_values()
        if not runs:
            raise ValueError("compiler transform requires runs")
        self._validate_run_schemas(runs)
        for run in runs:
            if (
                run.manifest.split != "training"
                and run.manifest.matched_pair_id
                in self._training_pair_ids
            ):
                raise ValueError(
                    "matched pair cannot cross training and held-out splits"
                )

        histories = []
        future_states = []
        future_controls = []
        future_actions = []
        trajectory_ids = []
        matched_pair_ids = []
        transition_indices = []
        for run in runs:
            normalized_states = (
                run.observations - state_center
            ) / state_scale
            normalized_controls = (
                run.controls - control_center
            ) / control_scale
            action_values = _compile_action_values(
                run, action_scales
            )
            first_transition = self.context_length - 1
            last_transition = (
                run.manifest.point_count
                - self.rollout_horizon
                - 1
            )
            for transition_index in range(
                first_transition, last_transition + 1
            ):
                history_start = (
                    transition_index - self.context_length + 1
                )
                histories.append(
                    normalized_states[
                        history_start : transition_index + 1
                    ]
                )
                future_states.append(
                    normalized_states[
                        transition_index
                        + 1 : transition_index
                        + 1
                        + self.rollout_horizon
                    ]
                )
                future_controls.append(
                    normalized_controls[
                        transition_index : transition_index
                        + self.rollout_horizon
                    ]
                )
                future_actions.append(
                    action_values[
                        transition_index : transition_index
                        + self.rollout_horizon
                    ]
                )
                trajectory_ids.append(run.manifest.case_id)
                matched_pair_ids.append(
                    run.manifest.matched_pair_id
                )
                transition_indices.append(transition_index)

        return ActionConditionedWindows(
            histories=np.asarray(histories, dtype=np.float64),
            future_states=np.asarray(
                future_states, dtype=np.float64
            ),
            future_controls=np.asarray(
                future_controls, dtype=np.float64
            ),
            future_actions=np.asarray(
                future_actions, dtype=np.float64
            ),
            trajectory_ids=tuple(trajectory_ids),
            matched_pair_ids=tuple(matched_pair_ids),
            transition_indices=np.asarray(
                transition_indices, dtype=np.int64
            ),
            entity_names=graph.entity_ids,
            state_feature_names=(
                self._required_state_feature_names()
            ),
            control_feature_names=(
                self._required_control_feature_names()
            ),
            action_feature_names=self.action_feature_names,
            graph=graph,
        )

    def compile_action_trajectory(
        self,
        *,
        point_count: int,
        actions: Sequence[InterventionAction],
        graph: DeclaredTelemetryGraph,
    ) -> NDArray[np.float64]:
        """Compile a planned candidate without observed future values."""

        (
            _,
            _,
            _,
            _,
            action_scales,
            fitted_graph,
        ) = self._fitted_values()
        if graph.to_dict() != fitted_graph.to_dict():
            raise ValueError(
                "candidate graph does not match compiler graph"
            )
        manifest = ActionConditionedCaseManifest(
            case_id="candidate",
            matched_pair_id="candidate",
            split="confirmation",
            point_count=point_count,
            logical_window_period_nano=1,
            topology_id="candidate",
            worker_replicas=1,
            workload_seed=0,
            intervention_seed=0,
            actions=tuple(actions),
        )
        run = ActionConditionedRun(
            manifest=manifest,
            graph=graph,
            observations=np.zeros(
                (
                    point_count,
                    len(graph.entities),
                    len(self._required_state_feature_names()),
                ),
                dtype=np.float64,
            ),
            controls=np.zeros(
                (
                    point_count,
                    len(self._required_control_feature_names()),
                ),
                dtype=np.float64,
            ),
            state_feature_names=(
                self._required_state_feature_names()
            ),
            control_feature_names=(
                self._required_control_feature_names()
            ),
        )
        return _compile_action_values(run, action_scales)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize fitted transforms and exact semantic identity."""

        (
            state_center,
            state_scale,
            control_center,
            control_scale,
            action_scales,
            graph,
        ) = self._fitted_values()
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "context_length": self.context_length,
            "rollout_horizon": self.rollout_horizon,
            "training_pair_count": len(
                self._training_pair_ids
            ),
            "training_pair_ids": list(self._training_pair_ids),
            "semantic_schema_sha256": _semantic_schema_sha256(
                graph,
                self._required_state_feature_names(),
                self._required_control_feature_names(),
                self.action_feature_names,
            ),
            "semantic_schema": {
                "graph": graph.to_dict(),
                "state_feature_names": list(
                    self._required_state_feature_names()
                ),
                "control_feature_names": list(
                    self._required_control_feature_names()
                ),
                "action_feature_names": list(
                    self.action_feature_names
                ),
            },
            "state": {
                "state_center": state_center.tolist(),
                "state_scale": state_scale.tolist(),
                "control_center": control_center.tolist(),
                "control_scale": control_scale.tolist(),
                "action_scales": action_scales.tolist(),
            },
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "ActionTrajectoryCompiler":
        """Restore fitted transforms after strict schema validation."""

        expected_keys = {
            "schema_version",
            "kind",
            "context_length",
            "rollout_horizon",
            "training_pair_count",
            "training_pair_ids",
            "semantic_schema_sha256",
            "semantic_schema",
            "state",
        }
        if (
            set(payload) != expected_keys
            or payload.get("schema_version") != cls.schema_version
            or payload.get("kind") != cls.kind
            or not _is_integer(payload["context_length"])
            or not _is_integer(payload["rollout_horizon"])
            or not _is_integer(payload["training_pair_count"])
            or not isinstance(payload["training_pair_ids"], list)
            or any(
                not isinstance(pair_id, str)
                for pair_id in payload["training_pair_ids"]
            )
        ):
            raise ValueError(
                "unsupported action compiler artifact"
            )
        semantic = dict(payload["semantic_schema"])
        state = dict(payload["state"])
        if set(semantic) != {
            "graph",
            "state_feature_names",
            "control_feature_names",
            "action_feature_names",
        } or set(state) != {
            "state_center",
            "state_scale",
            "control_center",
            "control_scale",
            "action_scales",
        }:
            raise ValueError(
                "action compiler artifact schema is invalid"
            )
        compiler = cls(
            context_length=payload["context_length"],
            rollout_horizon=payload["rollout_horizon"],
        )
        graph = DeclaredTelemetryGraph.from_dict(
            dict(semantic["graph"])
        )
        for name_list in (
            semantic["state_feature_names"],
            semantic["control_feature_names"],
            semantic["action_feature_names"],
        ):
            if (
                not isinstance(name_list, list)
                or any(
                    not isinstance(name, str)
                    for name in name_list
                )
            ):
                raise ValueError(
                    "action compiler semantic names are invalid"
                )
        state_feature_names = tuple(
            semantic["state_feature_names"]
        )
        control_feature_names = tuple(
            semantic["control_feature_names"]
        )
        action_feature_names = tuple(
            semantic["action_feature_names"]
        )
        semantic_schema_sha256 = payload[
            "semantic_schema_sha256"
        ]
        training_pair_ids = tuple(payload["training_pair_ids"])
        state_center = np.asarray(
            state["state_center"], dtype=np.float64
        )
        state_scale = np.asarray(
            state["state_scale"], dtype=np.float64
        )
        control_center = np.asarray(
            state["control_center"], dtype=np.float64
        )
        control_scale = np.asarray(
            state["control_scale"], dtype=np.float64
        )
        action_scales = np.asarray(
            state["action_scales"], dtype=np.float64
        )
        if (
            not state_feature_names
            or not control_feature_names
            or len(set(state_feature_names))
            != len(state_feature_names)
            or len(set(control_feature_names))
            != len(control_feature_names)
            or action_feature_names
            != compiler.action_feature_names
            or not isinstance(semantic_schema_sha256, str)
            or semantic_schema_sha256
            != _semantic_schema_sha256(
                graph,
                state_feature_names,
                control_feature_names,
                action_feature_names,
            )
            or len(set(training_pair_ids))
            != len(training_pair_ids)
            or len(training_pair_ids)
            != payload["training_pair_count"]
            or state_center.shape
            != (len(state_feature_names),)
            or state_scale.shape != state_center.shape
            or control_center.shape
            != (len(control_feature_names),)
            or control_scale.shape != control_center.shape
            or action_scales.shape != (len(ACTION_KINDS),)
            or any(
                not np.all(np.isfinite(values))
                for values in (
                    state_center,
                    state_scale,
                    control_center,
                    control_scale,
                    action_scales,
                )
            )
            or np.any(state_scale <= 0.0)
            or np.any(control_scale <= 0.0)
            or np.any(action_scales <= 0.0)
        ):
            raise ValueError(
                "action compiler artifact state is invalid"
            )
        compiler._graph = graph
        compiler._state_feature_names = state_feature_names
        compiler._control_feature_names = control_feature_names
        compiler._training_pair_ids = training_pair_ids
        compiler._state_center = state_center
        compiler._state_scale = state_scale
        compiler._control_center = control_center
        compiler._control_scale = control_scale
        compiler._action_scales = action_scales
        return compiler

    def _validate_run_schemas(
        self, runs: Sequence[ActionConditionedRun]
    ) -> None:
        first = runs[0]
        pair_splits: Dict[str, str] = {}
        for run in runs:
            if (
                run.graph.to_dict() != first.graph.to_dict()
                or run.state_feature_names
                != first.state_feature_names
                or run.control_feature_names
                != first.control_feature_names
            ):
                raise ValueError("run semantic schemas do not match")
            pair_id = run.manifest.matched_pair_id
            previous_split = pair_splits.setdefault(
                pair_id, run.manifest.split
            )
            if previous_split != run.manifest.split:
                raise ValueError(
                    "matched pair cannot cross corpus splits"
                )
        if self._graph is not None and (
            first.graph.to_dict() != self._graph.to_dict()
            or first.state_feature_names
            != self._required_state_feature_names()
            or first.control_feature_names
            != self._required_control_feature_names()
        ):
            raise ValueError(
                "run schema is incompatible with fitted compiler"
            )

    def _fitted_values(
        self,
    ) -> Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        DeclaredTelemetryGraph,
    ]:
        if (
            self._state_center is None
            or self._state_scale is None
            or self._control_center is None
            or self._control_scale is None
            or self._action_scales is None
            or self._graph is None
        ):
            raise RuntimeError("action compiler is not fitted")
        return (
            self._state_center,
            self._state_scale,
            self._control_center,
            self._control_scale,
            self._action_scales,
            self._graph,
        )

    def _required_state_feature_names(self) -> Tuple[str, ...]:
        if self._state_feature_names is None:
            raise RuntimeError("action compiler is not fitted")
        return self._state_feature_names

    def _required_control_feature_names(self) -> Tuple[str, ...]:
        if self._control_feature_names is None:
            raise RuntimeError("action compiler is not fitted")
        return self._control_feature_names


def _compile_action_values(
    run: ActionConditionedRun,
    action_scales: NDArray[np.float64],
) -> NDArray[np.float64]:
    entity_positions = {
        entity_id: index
        for index, entity_id in enumerate(run.graph.entity_ids)
    }
    values = np.zeros(
        (
            run.manifest.point_count,
            len(run.graph.entities),
            2 + len(ACTION_KINDS) + 6,
        ),
        dtype=np.float64,
    )
    values[:, :, 0] = 1.0
    for action in run.manifest.actions:
        kind_index = ACTION_KINDS.index(action.action_kind)
        entity_index = entity_positions[action.target_entity]
        duration = float(action.duration)
        for point_index in range(
            action.start_index, action.stop_index
        ):
            values[point_index, entity_index, 0] = 0.0
            values[point_index, entity_index, 1] = 1.0
            values[
                point_index, entity_index, 2 + kind_index
            ] = 1.0
            values[
                point_index,
                entity_index,
                2 + len(ACTION_KINDS)
                + (
                    0
                    if point_index == action.start_index
                    else 1
                ),
            ] = 1.0
            values[
                point_index, entity_index, 5 + len(ACTION_KINDS)
            ] = action.magnitude / action_scales[kind_index]
            values[
                point_index,
                entity_index,
                6 + len(ACTION_KINDS),
            ] = (point_index - action.start_index) / duration
            values[
                point_index,
                entity_index,
                7 + len(ACTION_KINDS),
            ] = (action.stop_index - point_index) / duration
        if action.stop_index < run.manifest.point_count:
            stop_index = action.stop_index
            values[stop_index, entity_index, 0] = 0.0
            values[stop_index, entity_index, 1] = 1.0
            values[
                stop_index, entity_index, 2 + kind_index
            ] = 1.0
            values[
                stop_index,
                entity_index,
                4 + len(ACTION_KINDS),
            ] = 1.0
    return values


def _nonzero_scale(
    values: NDArray[np.float64],
) -> NDArray[np.float64]:
    return np.where(values > 1e-12, values, 1.0)


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _semantic_schema_sha256(
    graph: DeclaredTelemetryGraph,
    state_feature_names: Sequence[str],
    control_feature_names: Sequence[str],
    action_feature_names: Sequence[str],
) -> str:
    payload = {
        "graph": graph.to_dict(),
        "state_feature_names": list(state_feature_names),
        "control_feature_names": list(control_feature_names),
        "action_feature_names": list(action_feature_names),
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class GraphVarxConfig:
    """Deterministic graph-constrained linear rollout choices."""

    ridge: float = 1e-3
    include_actions: bool = True
    variance_floor: float = 1e-4

    def __post_init__(self) -> None:
        if (
            isinstance(self.ridge, bool)
            or not np.isfinite(self.ridge)
            or self.ridge <= 0.0
        ):
            raise ValueError("VARX ridge must be finite and positive")
        if (
            isinstance(self.include_actions, bool) is False
            or isinstance(self.variance_floor, bool)
            or not np.isfinite(self.variance_floor)
            or self.variance_floor <= 0.0
        ):
            raise ValueError("VARX configuration is invalid")

    def to_dict(self) -> Dict[str, Any]:
        """Return serializable configuration."""

        return {
            "ridge": self.ridge,
            "include_actions": self.include_actions,
            "variance_floor": self.variance_floor,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "GraphVarxConfig":
        """Restore configuration."""

        if set(payload) != {
            "ridge",
            "include_actions",
            "variance_floor",
        } or (
            not isinstance(payload["ridge"], (int, float))
            or isinstance(payload["ridge"], bool)
            or not isinstance(payload["include_actions"], bool)
            or not isinstance(
                payload["variance_floor"], (int, float)
            )
            or isinstance(payload["variance_floor"], bool)
        ):
            raise ValueError("graph VARX configuration is invalid")
        return cls(
            ridge=float(payload["ridge"]),
            include_actions=payload["include_actions"],
            variance_floor=float(payload["variance_floor"]),
        )


@dataclass(frozen=True)
class TrajectoryDistribution:
    """Diagonal Gaussian approximation for a multi-step graph rollout."""

    mean: NDArray[np.float64]
    variance: NDArray[np.float64]

    def __post_init__(self) -> None:
        if (
            self.mean.ndim != 4
            or self.variance.shape != self.mean.shape
            or not np.all(np.isfinite(self.mean))
            or not np.all(np.isfinite(self.variance))
            or np.any(self.variance <= 0.0)
        ):
            raise ValueError("trajectory distribution is invalid")

    def negative_log_likelihood(
        self, observed: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return one normalized diagonal-Gaussian NLL per sample."""

        values = np.asarray(observed, dtype=np.float64)
        if values.shape != self.mean.shape:
            raise ValueError(
                "observed trajectory does not match distribution"
            )
        terms = (
            np.square(values - self.mean) / self.variance
            + np.log(self.variance)
            + np.log(2.0 * np.pi)
        )
        return np.asarray(
            0.5 * np.mean(terms, axis=(1, 2, 3)),
            dtype=np.float64,
        )


def persistence_rollout(
    histories: NDArray[np.float64],
    rollout_horizon: int,
) -> TrajectoryDistribution:
    """Repeat the last observed graph state as a baseline rollout."""

    values = np.asarray(histories, dtype=np.float64)
    if (
        values.ndim != 4
        or isinstance(rollout_horizon, bool)
        or rollout_horizon < 1
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("persistence rollout inputs are invalid")
    mean = np.repeat(
        values[:, -1:, :, :], rollout_horizon, axis=1
    )
    variance = np.ones_like(mean)
    return TrajectoryDistribution(mean=mean, variance=variance)


class GraphVarxDynamics:
    """Graph-constrained VARX with local actions and autoregressive rollout."""

    kind = "action_conditioned_graph_varx"
    schema_version = 1

    def __init__(self, config: GraphVarxConfig) -> None:
        self.config = config
        self._graph: Optional[DeclaredTelemetryGraph] = None
        self._state_feature_names: Optional[Tuple[str, ...]] = None
        self._control_feature_names: Optional[Tuple[str, ...]] = None
        self._action_feature_names: Optional[Tuple[str, ...]] = None
        self._source_positions: Optional[Tuple[Tuple[int, ...], ...]] = (
            None
        )
        self._coefficients: Optional[
            Tuple[NDArray[np.float64], ...]
        ] = None
        self._residual_variance: Optional[
            NDArray[np.float64]
        ] = None

    def fit(
        self, windows: ActionConditionedWindows
    ) -> "GraphVarxDynamics":
        """Fit one training-only transition for each graph entity."""

        sample_count = len(windows.histories)
        if sample_count < 2:
            raise ValueError("VARX fit requires at least two samples")
        graph = windows.graph
        feature_count = len(windows.state_feature_names)
        control_count = len(windows.control_feature_names)
        action_count = len(windows.action_feature_names)
        source_positions = tuple(
            _incoming_entity_positions(graph, target_position)
            for target_position in range(len(graph.entities))
        )
        coefficients: List[NDArray[np.float64]] = []
        residual_variance = np.empty(
            (len(graph.entities), feature_count),
            dtype=np.float64,
        )
        for target_position, sources in enumerate(source_positions):
            state_inputs = windows.histories[
                :, -1, sources, :
            ].reshape(sample_count, len(sources) * feature_count)
            parts = [
                state_inputs,
                windows.future_controls[:, 0, :].reshape(
                    sample_count, control_count
                ),
            ]
            if self.config.include_actions:
                parts.append(
                    windows.future_actions[
                        :, 0, target_position, :
                    ].reshape(sample_count, action_count)
                )
            parts.append(np.ones((sample_count, 1), dtype=np.float64))
            design = np.concatenate(parts, axis=1)
            target = windows.future_states[
                :, 0, target_position, :
            ]
            penalty = np.eye(design.shape[1], dtype=np.float64)
            penalty[-1, -1] = 0.0
            matrix = (
                design.T @ design
                + self.config.ridge * penalty
            )
            weights = np.linalg.solve(
                matrix, design.T @ target
            )
            coefficients.append(
                np.asarray(weights, dtype=np.float64)
            )
            residual = target - design @ weights
            residual_variance[target_position] = np.maximum(
                np.mean(np.square(residual), axis=0),
                self.config.variance_floor,
            )

        self._graph = graph
        self._state_feature_names = windows.state_feature_names
        self._control_feature_names = windows.control_feature_names
        self._action_feature_names = windows.action_feature_names
        self._source_positions = source_positions
        self._coefficients = tuple(coefficients)
        self._residual_variance = residual_variance
        return self

    def rollout(
        self,
        histories: NDArray[np.float64],
        future_controls: NDArray[np.float64],
        future_actions: NDArray[np.float64],
        graph: DeclaredTelemetryGraph,
    ) -> TrajectoryDistribution:
        """Roll forward without consuming any observed future state."""

        (
            fitted_graph,
            state_feature_names,
            control_feature_names,
            action_feature_names,
            source_positions,
            coefficients,
            residual_variance,
        ) = self._fitted_values()
        if graph.to_dict() != fitted_graph.to_dict():
            raise ValueError(
                "rollout graph does not match the fitted graph"
            )
        history_values = np.asarray(histories, dtype=np.float64)
        control_values = np.asarray(
            future_controls, dtype=np.float64
        )
        action_values = np.asarray(
            future_actions, dtype=np.float64
        )
        batch_size = len(history_values)
        if (
            history_values.ndim != 4
            or history_values.shape[2:]
            != (
                len(graph.entities),
                len(state_feature_names),
            )
            or control_values.ndim != 3
            or control_values.shape[0] != batch_size
            or control_values.shape[2]
            != len(control_feature_names)
            or action_values.shape != (
                batch_size,
                control_values.shape[1],
                len(graph.entities),
                len(action_feature_names),
            )
        ):
            raise ValueError("rollout tensors do not match model schema")
        if any(
            not np.all(np.isfinite(values))
            for values in (
                history_values,
                control_values,
                action_values,
            )
        ):
            raise ValueError("rollout tensors must be finite")

        horizon = control_values.shape[1]
        mean = np.empty(
            (
                batch_size,
                horizon,
                len(graph.entities),
                len(state_feature_names),
            ),
            dtype=np.float64,
        )
        variance = np.empty_like(mean)
        current = history_values[:, -1].copy()
        for horizon_index in range(horizon):
            next_state = np.empty_like(current)
            for target_position, sources in enumerate(
                source_positions
            ):
                state_inputs = current[:, sources, :].reshape(
                    batch_size,
                    len(sources) * len(state_feature_names),
                )
                parts = [
                    state_inputs,
                    control_values[:, horizon_index, :],
                ]
                if self.config.include_actions:
                    parts.append(
                        action_values[
                            :, horizon_index, target_position, :
                        ]
                    )
                parts.append(
                    np.ones(
                        (batch_size, 1), dtype=np.float64
                    )
                )
                design = np.concatenate(parts, axis=1)
                next_state[:, target_position, :] = (
                    design @ coefficients[target_position]
                )
            if not np.all(np.isfinite(next_state)):
                raise FloatingPointError(
                    "VARX rollout produced non-finite state"
                )
            mean[:, horizon_index] = next_state
            variance[:, horizon_index] = (
                residual_variance * (horizon_index + 1)
            )
            current = next_state
        return TrajectoryDistribution(mean=mean, variance=variance)

    def ensure_compatible(
        self, windows: ActionConditionedWindows
    ) -> None:
        """Reject schema-bearing inputs from another compiler."""

        (
            graph,
            state_feature_names,
            control_feature_names,
            action_feature_names,
            _,
            _,
            _,
        ) = self._fitted_values()
        expected = _semantic_schema_sha256(
            graph,
            state_feature_names,
            control_feature_names,
            action_feature_names,
        )
        if windows.semantic_schema_sha256 != expected:
            raise ValueError(
                "graph VARX input semantic schema is incompatible"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize exact model and semantic identity."""

        (
            graph,
            state_feature_names,
            control_feature_names,
            action_feature_names,
            source_positions,
            coefficients,
            residual_variance,
        ) = self._fitted_values()
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "config": self.config.to_dict(),
            "semantic_schema_sha256": _semantic_schema_sha256(
                graph,
                state_feature_names,
                control_feature_names,
                action_feature_names,
            ),
            "semantic_schema": {
                "graph": graph.to_dict(),
                "state_feature_names": list(
                    state_feature_names
                ),
                "control_feature_names": list(
                    control_feature_names
                ),
                "action_feature_names": list(
                    action_feature_names
                ),
            },
            "state": {
                "source_positions": [
                    list(values) for values in source_positions
                ],
                "coefficients": [
                    values.tolist() for values in coefficients
                ],
                "residual_variance": (
                    residual_variance.tolist()
                ),
            },
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "GraphVarxDynamics":
        """Restore a fitted CPU-portable linear model."""

        if (
            set(payload)
            != {
                "schema_version",
                "kind",
                "config",
                "semantic_schema_sha256",
                "semantic_schema",
                "state",
            }
            or payload.get("schema_version") != cls.schema_version
            or payload.get("kind") != cls.kind
        ):
            raise ValueError("unsupported graph VARX artifact")
        config = GraphVarxConfig.from_dict(
            dict(payload["config"])
        )
        semantic = dict(payload["semantic_schema"])
        state = dict(payload["state"])
        if set(semantic) != {
            "graph",
            "state_feature_names",
            "control_feature_names",
            "action_feature_names",
        } or set(state) != {
            "source_positions",
            "coefficients",
            "residual_variance",
        }:
            raise ValueError("graph VARX artifact schema is invalid")
        graph = DeclaredTelemetryGraph.from_dict(
            dict(semantic["graph"])
        )
        for name_list in (
            semantic["state_feature_names"],
            semantic["control_feature_names"],
            semantic["action_feature_names"],
        ):
            if (
                not isinstance(name_list, list)
                or any(
                    not isinstance(name, str)
                    for name in name_list
                )
            ):
                raise ValueError(
                    "graph VARX semantic names are invalid"
                )
        if (
            not isinstance(state["source_positions"], list)
            or any(
                not isinstance(values, list)
                or any(
                    not _is_integer(position)
                    for position in values
                )
                for values in state["source_positions"]
            )
        ):
            raise ValueError(
                "graph VARX source positions are invalid"
            )
        source_positions = tuple(
            tuple(position for position in values)
            for values in state["source_positions"]
        )
        coefficients = tuple(
            np.asarray(values, dtype=np.float64)
            for values in state["coefficients"]
        )
        residual_variance = np.asarray(
            state["residual_variance"], dtype=np.float64
        )
        entity_count = len(graph.entities)
        state_feature_names = tuple(
            semantic["state_feature_names"]
        )
        control_feature_names = tuple(
            semantic["control_feature_names"]
        )
        action_feature_names = tuple(
            semantic["action_feature_names"]
        )
        expected_action_feature_names = (
            "no_action",
            "applicable",
            *(f"kind:{name}" for name in ACTION_KINDS),
            "phase:start",
            "phase:active",
            "phase:stop",
            "magnitude",
            "elapsed_fraction",
            "remaining_fraction",
        )
        if action_feature_names != expected_action_feature_names:
            raise ValueError(
                "graph VARX action schema is invalid"
            )
        semantic_schema_sha256 = payload[
            "semantic_schema_sha256"
        ]
        if (
            not isinstance(semantic_schema_sha256, str)
            or semantic_schema_sha256
            != _semantic_schema_sha256(
                graph,
                state_feature_names,
                control_feature_names,
                action_feature_names,
            )
        ):
            raise ValueError(
                "graph VARX semantic schema hash is invalid"
            )
        if (
            len(source_positions) != entity_count
            or len(coefficients) != entity_count
            or residual_variance.shape
            != (entity_count, len(state_feature_names))
            or not np.all(np.isfinite(residual_variance))
            or np.any(residual_variance <= 0.0)
        ):
            raise ValueError("graph VARX artifact state is invalid")
        for target_position, (sources, weights) in enumerate(
            zip(source_positions, coefficients)
        ):
            expected_sources = _incoming_entity_positions(
                graph, target_position
            )
            expected_inputs = (
                len(sources) * len(state_feature_names)
                + len(control_feature_names)
                + (
                    len(action_feature_names)
                    if config.include_actions
                    else 0
                )
                + 1
            )
            if (
                sources != expected_sources
                or any(
                    position < 0 or position >= entity_count
                    for position in sources
                )
                or weights.shape
                != (expected_inputs, len(state_feature_names))
                or not np.all(np.isfinite(weights))
            ):
                raise ValueError(
                    "graph VARX topology or coefficients are invalid"
                )
        model = cls(config)
        model._graph = graph
        model._state_feature_names = state_feature_names
        model._control_feature_names = control_feature_names
        model._action_feature_names = action_feature_names
        model._source_positions = source_positions
        model._coefficients = coefficients
        model._residual_variance = residual_variance
        return model

    def _fitted_values(
        self,
    ) -> Tuple[
        DeclaredTelemetryGraph,
        Tuple[str, ...],
        Tuple[str, ...],
        Tuple[str, ...],
        Tuple[Tuple[int, ...], ...],
        Tuple[NDArray[np.float64], ...],
        NDArray[np.float64],
    ]:
        if (
            self._graph is None
            or self._state_feature_names is None
            or self._control_feature_names is None
            or self._action_feature_names is None
            or self._source_positions is None
            or self._coefficients is None
            or self._residual_variance is None
        ):
            raise RuntimeError("graph VARX model is not fitted")
        return (
            self._graph,
            self._state_feature_names,
            self._control_feature_names,
            self._action_feature_names,
            self._source_positions,
            self._coefficients,
            self._residual_variance,
        )


def _incoming_entity_positions(
    graph: DeclaredTelemetryGraph, target_position: int
) -> Tuple[int, ...]:
    target = graph.entities[target_position]
    entity_positions = {
        entity.entity_id: index
        for index, entity in enumerate(graph.entities)
    }
    positions = {target_position}
    if target.kind == "edge":
        assert target.source is not None
        positions.add(entity_positions[target.source])
    else:
        positions.update(
            index
            for index, entity in enumerate(graph.entities)
            if entity.kind == "edge"
            and entity.target == target.entity_id
        )
    return tuple(sorted(positions))


@dataclass(frozen=True)
class RolloutCandidate:
    """One candidate external action trajectory for inverse attribution."""

    candidate_id: str
    future_actions: NDArray[np.float64]

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate id cannot be empty")
        if (
            self.future_actions.ndim != 3
            or not np.all(np.isfinite(self.future_actions))
        ):
            raise ValueError("candidate action trajectory is invalid")


@dataclass(frozen=True)
class AttributionResult:
    """Ranked candidate likelihoods and winning counterfactual effect."""

    candidate_ids: Tuple[str, ...]
    candidate_distribution: TrajectoryDistribution
    ranked_candidate_ids: Tuple[str, ...]
    negative_log_likelihoods: Tuple[float, ...]
    counterfactual_delta: NDArray[np.float64]
    per_entity_effect: NDArray[np.float64]

    def __post_init__(self) -> None:
        if (
            not self.candidate_ids
            or len(set(self.candidate_ids)) != len(self.candidate_ids)
            or self.candidate_distribution.mean.shape[0]
            != len(self.candidate_ids)
            or not self.ranked_candidate_ids
            or len(self.ranked_candidate_ids)
            != len(self.negative_log_likelihoods)
            or len(set(self.ranked_candidate_ids))
            != len(self.ranked_candidate_ids)
            or set(self.ranked_candidate_ids)
            != set(self.candidate_ids)
            or self.counterfactual_delta.ndim != 3
            or self.per_entity_effect.shape
            != (self.counterfactual_delta.shape[1],)
            or any(
                not np.isfinite(value)
                for value in self.negative_log_likelihoods
            )
            or not np.all(
                np.isfinite(self.counterfactual_delta)
            )
            or not np.all(np.isfinite(self.per_entity_effect))
        ):
            raise ValueError("attribution result is invalid")


def rank_action_candidates(
    *,
    model: GraphVarxDynamics,
    history: NDArray[np.float64],
    future_controls: NDArray[np.float64],
    observed_future: NDArray[np.float64],
    candidates: Sequence[RolloutCandidate],
    graph: DeclaredTelemetryGraph,
    no_action_candidate_id: str,
) -> AttributionResult:
    """Rank hidden-action explanations through the rollout seam."""

    if not candidates:
        raise ValueError("attribution requires candidates")
    candidate_ids = tuple(
        candidate.candidate_id for candidate in candidates
    )
    if (
        len(set(candidate_ids)) != len(candidate_ids)
        or no_action_candidate_id not in set(candidate_ids)
    ):
        raise ValueError(
            "candidate ids must be unique and include no action"
        )
    history_values = np.asarray(history, dtype=np.float64)
    control_values = np.asarray(
        future_controls, dtype=np.float64
    )
    observed_values = np.asarray(
        observed_future, dtype=np.float64
    )
    action_values = np.stack(
        [candidate.future_actions for candidate in candidates],
        axis=0,
    )
    candidate_count = len(candidates)
    if (
        history_values.ndim != 3
        or control_values.ndim != 2
        or observed_values.ndim != 3
        or action_values.shape[1] != len(control_values)
        or observed_values.shape[0] != len(control_values)
    ):
        raise ValueError("attribution trajectories do not align")
    distribution = model.rollout(
        np.repeat(
            history_values[np.newaxis, ...],
            candidate_count,
            axis=0,
        ),
        np.repeat(
            control_values[np.newaxis, ...],
            candidate_count,
            axis=0,
        ),
        action_values,
        graph,
    )
    observed_batch = np.repeat(
        observed_values[np.newaxis, ...],
        candidate_count,
        axis=0,
    )
    nll = distribution.negative_log_likelihood(observed_batch)
    ranked_positions = tuple(
        sorted(
            range(candidate_count),
            key=lambda position: (
                float(nll[position]),
                candidate_ids[position],
            ),
        )
    )
    winner_position = ranked_positions[0]
    no_action_position = candidate_ids.index(no_action_candidate_id)
    delta = (
        distribution.mean[winner_position]
        - distribution.mean[no_action_position]
    )
    per_entity_effect = np.sqrt(
        np.mean(np.square(delta), axis=(0, 2))
    )
    return AttributionResult(
        candidate_ids=candidate_ids,
        candidate_distribution=distribution,
        ranked_candidate_ids=tuple(
            candidate_ids[position]
            for position in ranked_positions
        ),
        negative_log_likelihoods=tuple(
            float(nll[position]) for position in ranked_positions
        ),
        counterfactual_delta=np.asarray(
            delta, dtype=np.float64
        ),
        per_entity_effect=np.asarray(
            per_entity_effect, dtype=np.float64
        ),
    )
