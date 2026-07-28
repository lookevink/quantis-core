"""Training and held-out evaluation for real action-dynamics corpora.

The public seam deliberately accepts already-loaded runs.  Corpus collection
and telemetry parsing can therefore evolve without changing the scientific
boundary enforced here: development data only, training-only fitting, and
validation-only forecast and attribution scoring.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .action_conditioned_dynamics import (
    ACTION_KINDS,
    ActionConditionedRun,
    ActionConditionedWindows,
    ActionTrajectoryCompiler,
    GraphVarxConfig,
    GraphVarxDynamics,
    InterventionAction,
    RolloutCandidate,
    TrajectoryDistribution,
    persistence_rollout,
    rank_action_candidates,
)
from .graph_telemetry import DeclaredTelemetryGraph


_DEVELOPMENT_CONTEXT_LENGTH = 20
_DEVELOPMENT_ROLLOUT_HORIZON = 10
_DEVELOPMENT_RIDGE = 1e-3
_DEVELOPMENT_VARIANCE_FLOOR = 1e-4
_NO_ACTION_CANDIDATE_ID = "no_action"
_ACTION_TARGETS = {
    "worker_pause": "worker_pool",
    "postgres_lock": "worker_writes_postgresql",
    "redis_enqueue_delay": "api_enqueues_queue",
    "redis_dequeue_delay": "queue_dequeues_to_worker",
    "api_rejection": "api",
}
_ACTION_SEVERITIES = {
    "worker_pause": (1.0,),
    "postgres_lock": (1.0,),
    "redis_enqueue_delay": (20.0, 40.0, 60.0),
    "redis_dequeue_delay": (20.0, 40.0, 60.0),
    "api_rejection": (0.25, 0.5, 0.75),
}
_ACTION_DURATIONS = {
    "worker_pause": tuple(range(8, 21)),
    "postgres_lock": tuple(range(8, 21)),
    "redis_enqueue_delay": tuple(range(8, 21)),
    "redis_dequeue_delay": tuple(range(8, 21)),
    "api_rejection": (20,),
}


@dataclass(frozen=True)
class AllowedActionVariant:
    """One member of the frozen development-v1 candidate grid."""

    action_kind: str
    target_entity: str
    magnitude: float
    duration: int
    candidate_id: str


def _frozen_action_candidate_grid() -> Tuple[AllowedActionVariant, ...]:
    variants = []
    for action_kind in ACTION_KINDS:
        target = _ACTION_TARGETS[action_kind]
        for magnitude in _ACTION_SEVERITIES[action_kind]:
            for duration in _ACTION_DURATIONS[action_kind]:
                variants.append(
                    AllowedActionVariant(
                        action_kind=action_kind,
                        target_entity=target,
                        magnitude=magnitude,
                        duration=duration,
                        candidate_id=(
                            f"action:{action_kind}@{target}:"
                            f"m={magnitude:g}:d={duration}"
                        ),
                    )
                )
    return tuple(variants)


FROZEN_ACTION_CANDIDATE_GRID = _frozen_action_candidate_grid()


@dataclass(frozen=True)
class RealCorpusRun:
    """One trajectory plus its independently assigned evidence role."""

    run: ActionConditionedRun
    corpus_role: str

    def __post_init__(self) -> None:
        if not self.corpus_role:
            raise ValueError("corpus role cannot be empty")


@dataclass(frozen=True)
class RealCorpusStudyConfig:
    """Preregistered training and rollout choices."""

    context_length: int = _DEVELOPMENT_CONTEXT_LENGTH
    rollout_horizon: int = _DEVELOPMENT_ROLLOUT_HORIZON
    ridge: float = _DEVELOPMENT_RIDGE
    variance_floor: float = _DEVELOPMENT_VARIANCE_FLOOR

    def __post_init__(self) -> None:
        if (
            self.context_length != _DEVELOPMENT_CONTEXT_LENGTH
            or self.rollout_horizon != _DEVELOPMENT_ROLLOUT_HORIZON
            or self.ridge != _DEVELOPMENT_RIDGE
            or self.variance_floor != _DEVELOPMENT_VARIANCE_FLOOR
        ):
            raise ValueError(
                "configuration differs from frozen development-v1"
            )


@dataclass(frozen=True)
class AttributionCandidatePlan:
    """One candidate intervention schedule declared before evaluation."""

    candidate_id: str
    actions: Tuple[InterventionAction, ...]

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate id cannot be empty")


@dataclass(frozen=True)
class AttributionQuery:
    """One preregistered held-out attribution question."""

    query_id: str
    validation_case_id: str
    transition_index: int
    candidates: Tuple[AttributionCandidatePlan, ...]
    no_action_candidate_id: str
    expected_action_kind: Optional[str]
    expected_target_entity: Optional[str]
    expected_variant_candidate_id: Optional[str] = None

    def __post_init__(self) -> None:
        candidate_ids = tuple(
            candidate.candidate_id for candidate in self.candidates
        )
        if (
            not self.query_id
            or not self.validation_case_id
            or isinstance(self.transition_index, bool)
            or self.transition_index < 0
            or not self.candidates
            or len(set(candidate_ids)) != len(candidate_ids)
            or self.no_action_candidate_id not in candidate_ids
            or len(candidate_ids) < 2
            or (self.expected_action_kind is None)
            != (self.expected_target_entity is None)
            or (self.expected_action_kind is None)
            != (self.expected_variant_candidate_id is None)
            or (
                self.expected_action_kind is not None
                and self.expected_action_kind not in ACTION_KINDS
            )
            or (
                self.expected_variant_candidate_id is not None
                and self.expected_variant_candidate_id not in candidate_ids
            )
        ):
            raise ValueError("attribution query is invalid")
        no_action = self.candidates[
            candidate_ids.index(self.no_action_candidate_id)
        ]
        if no_action.actions:
            raise ValueError("no-action candidate must contain no actions")
        if any(
            not candidate.actions
            for candidate in self.candidates
            if candidate.candidate_id != self.no_action_candidate_id
        ):
            raise ValueError("action candidates must contain actions")

    def to_dict(self) -> Dict[str, Any]:
        """Return the immutable query declaration."""

        return {
            "query_id": self.query_id,
            "validation_case_id": self.validation_case_id,
            "transition_index": self.transition_index,
            "candidates": [
                {
                    "candidate_id": candidate.candidate_id,
                    "actions": [
                        action.to_dict()
                        for action in candidate.actions
                    ],
                }
                for candidate in self.candidates
            ],
            "no_action_candidate_id": self.no_action_candidate_id,
            "expected_action_kind": self.expected_action_kind,
            "expected_target_entity": self.expected_target_entity,
            "expected_variant_candidate_id": (
                self.expected_variant_candidate_id
            ),
        }


@dataclass(frozen=True)
class ForecastMetrics:
    """Normalized forecast error on held-out windows."""

    normalized_mse_overall: float
    normalized_mse_action_overlap: float

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.normalized_mse_overall)
            or self.normalized_mse_overall < 0.0
            or not np.isfinite(self.normalized_mse_action_overlap)
            or self.normalized_mse_action_overlap < 0.0
        ):
            raise ValueError("forecast metrics must be finite and nonnegative")

    def to_dict(self) -> Dict[str, float]:
        """Return serializable normalized metrics."""

        return {
            "normalized_mse_overall": self.normalized_mse_overall,
            "normalized_mse_action_overlap": (
                self.normalized_mse_action_overlap
            ),
        }


@dataclass(frozen=True)
class StratifiedForecastPanel:
    """One forecast slice overall and by intervention/topology."""

    normalized_mse: Mapping[str, float]
    by_action_kind: Mapping[str, Mapping[str, float]]
    by_worker_replicas: Mapping[str, Mapping[str, float]]
    selected_state_count: int
    comparison_kind: str

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe panel."""

        return {
            "normalized_mse": dict(self.normalized_mse),
            "by_action_kind": {
                key: dict(values)
                for key, values in self.by_action_kind.items()
            },
            "by_worker_replicas": {
                key: dict(values)
                for key, values in self.by_worker_replicas.items()
            },
            "selected_state_count": self.selected_state_count,
            "comparison_kind": self.comparison_kind,
        }


@dataclass(frozen=True)
class DevelopmentStudyAssessment:
    """Frozen development gates with graph evidence kept independent."""

    action_vs_agnostic_improvement: float
    action_vs_persistence_improvement: float
    graph_vs_dense_downstream_improvement: float
    action_location_hit_at_1: float
    no_action_specificity: float
    action_vs_agnostic_gate: bool
    action_vs_persistence_gate: bool
    graph_vs_dense_downstream_gate: bool
    action_location_gate: bool
    no_action_specificity_gate: bool
    action_conditioning_supported: bool
    attribution_supported: bool
    graph_topology_supported: bool
    graph_claim_blocked: bool
    decision: str

    def to_dict(self) -> Dict[str, Any]:
        """Return the complete gate evidence and bounded decision."""

        return {
            "thresholds": {
                "action_vs_agnostic_improvement_min": 0.10,
                "action_vs_persistence_improvement_min": 0.10,
                "graph_vs_dense_downstream_improvement_min": 0.05,
                "action_location_hit_at_1_min": 0.70,
                "no_action_specificity_min": 0.90,
            },
            "measurements": {
                "action_vs_agnostic_improvement": (
                    self.action_vs_agnostic_improvement
                ),
                "action_vs_persistence_improvement": (
                    self.action_vs_persistence_improvement
                ),
                "graph_vs_dense_downstream_improvement": (
                    self.graph_vs_dense_downstream_improvement
                ),
                "action_location_hit_at_1": (
                    self.action_location_hit_at_1
                ),
                "no_action_specificity": self.no_action_specificity,
            },
            "gates": {
                "action_vs_agnostic": self.action_vs_agnostic_gate,
                "action_vs_persistence": (
                    self.action_vs_persistence_gate
                ),
                "graph_vs_dense_downstream": (
                    self.graph_vs_dense_downstream_gate
                ),
                "action_location_hit_at_1": self.action_location_gate,
                "no_action_specificity": (
                    self.no_action_specificity_gate
                ),
            },
            "action_conditioning_supported": (
                self.action_conditioning_supported
            ),
            "attribution_supported": self.attribution_supported,
            "graph_topology_supported": self.graph_topology_supported,
            "graph_claim_blocked": self.graph_claim_blocked,
            "decision": self.decision,
        }


def assess_development_gates(
    *,
    action_overlap_mse: float,
    action_agnostic_overlap_mse: float,
    persistence_overlap_mse: float,
    graph_downstream_mse: float,
    dense_downstream_mse: float,
    action_location_hit_at_1: float,
    no_action_specificity: float,
) -> DevelopmentStudyAssessment:
    """Apply only the numerical gates frozen in development-v1."""

    values = (
        action_overlap_mse,
        action_agnostic_overlap_mse,
        persistence_overlap_mse,
        graph_downstream_mse,
        dense_downstream_mse,
        action_location_hit_at_1,
        no_action_specificity,
    )
    if any(not np.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("development gate inputs must be nonnegative")
    if (
        action_location_hit_at_1 > 1.0
        or no_action_specificity > 1.0
    ):
        raise ValueError("development attribution rates exceed one")
    action_vs_agnostic = _relative_improvement(
        action_overlap_mse, action_agnostic_overlap_mse
    )
    action_vs_persistence = _relative_improvement(
        action_overlap_mse, persistence_overlap_mse
    )
    graph_vs_dense = _relative_improvement(
        graph_downstream_mse, dense_downstream_mse
    )
    action_agnostic_gate = action_vs_agnostic >= 0.10 - 1e-12
    persistence_gate = action_vs_persistence >= 0.10 - 1e-12
    graph_gate = graph_vs_dense >= 0.05 - 1e-12
    action_location_gate = (
        action_location_hit_at_1 >= 0.70 - 1e-12
    )
    no_action_gate = no_action_specificity >= 0.90 - 1e-12
    action_supported = action_agnostic_gate and persistence_gate
    attribution_supported = action_location_gate and no_action_gate
    if action_supported and attribution_supported and graph_gate:
        decision = "advance_to_sealed_confirmation"
    elif action_supported and attribution_supported:
        decision = "publish_action_conditioning_without_graph_claim"
    elif action_supported:
        decision = "attribution_claim_blocked"
    else:
        decision = "publish_bounded_negative_result"
    return DevelopmentStudyAssessment(
        action_vs_agnostic_improvement=action_vs_agnostic,
        action_vs_persistence_improvement=action_vs_persistence,
        graph_vs_dense_downstream_improvement=graph_vs_dense,
        action_location_hit_at_1=action_location_hit_at_1,
        no_action_specificity=no_action_specificity,
        action_vs_agnostic_gate=action_agnostic_gate,
        action_vs_persistence_gate=persistence_gate,
        graph_vs_dense_downstream_gate=graph_gate,
        action_location_gate=action_location_gate,
        no_action_specificity_gate=no_action_gate,
        action_conditioning_supported=action_supported,
        attribution_supported=attribution_supported,
        graph_topology_supported=graph_gate,
        graph_claim_blocked=not graph_gate,
        decision=decision,
    )


def _relative_improvement(candidate: float, baseline: float) -> float:
    if baseline == 0.0:
        return 0.0 if candidate == 0.0 else -candidate / 1e-12
    return (baseline - candidate) / baseline


@dataclass(frozen=True)
class AttributionQueryResult:
    """Auditable result for one preregistered attribution query."""

    query_id: str
    winning_candidate_id: str
    ranked_candidate_ids: Tuple[str, ...]
    negative_log_likelihoods: Tuple[float, ...]
    action_family_hit_at_1: Optional[bool]
    action_location_hit_at_1: Optional[bool]
    action_and_target_hit_at_1: Optional[bool]
    exact_variant_hit_at_1: Optional[bool]
    action_family_hit_at_3: Optional[bool]
    no_action_hit_at_1: Optional[bool]
    likelihood_margin: float
    worker_replicas: int

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable query result."""

        return {
            "query_id": self.query_id,
            "winning_candidate_id": self.winning_candidate_id,
            "ranked_candidate_ids": list(self.ranked_candidate_ids),
            "negative_log_likelihoods": list(
                self.negative_log_likelihoods
            ),
            "action_family_hit_at_1": self.action_family_hit_at_1,
            "action_location_hit_at_1": self.action_location_hit_at_1,
            "action_and_target_hit_at_1": (
                self.action_and_target_hit_at_1
            ),
            "exact_variant_hit_at_1": self.exact_variant_hit_at_1,
            "action_family_hit_at_3": self.action_family_hit_at_3,
            "no_action_hit_at_1": self.no_action_hit_at_1,
            "likelihood_margin": self.likelihood_margin,
            "worker_replicas": self.worker_replicas,
        }


@dataclass(frozen=True)
class AttributionMetrics:
    """Aggregate attribution and nominal-specificity measurements."""

    query_count: int
    action_query_count: int
    no_action_query_count: int
    action_family_hit_at_1: float
    action_location_hit_at_1: float
    action_and_target_hit_at_1: float
    exact_variant_hit_at_1: float
    action_family_hit_at_3: float
    no_action_specificity: float
    mean_likelihood_margin: float
    by_worker_replicas: Mapping[str, Mapping[str, float]]
    query_results: Tuple[AttributionQueryResult, ...]

    def to_dict(self) -> Dict[str, Any]:
        """Return serializable aggregate and per-query measurements."""

        return {
            "query_count": self.query_count,
            "action_query_count": self.action_query_count,
            "no_action_query_count": self.no_action_query_count,
            "action_family_hit_at_1": self.action_family_hit_at_1,
            "action_location_hit_at_1": self.action_location_hit_at_1,
            "action_and_target_hit_at_1": (
                self.action_and_target_hit_at_1
            ),
            "exact_variant_hit_at_1": self.exact_variant_hit_at_1,
            "action_family_hit_at_3": self.action_family_hit_at_3,
            "no_action_specificity": self.no_action_specificity,
            "mean_likelihood_margin": self.mean_likelihood_margin,
            "by_worker_replicas": {
                key: dict(values)
                for key, values in self.by_worker_replicas.items()
            },
            "queries": [
                result.to_dict() for result in self.query_results
            ],
        }


@dataclass(frozen=True)
class RealCorpusStudyResult:
    """Fitted artifacts plus strictly held-out study measurements."""

    training_run_ids: Tuple[str, ...]
    validation_run_ids: Tuple[str, ...]
    compiler_artifact: Mapping[str, Any]
    model_artifacts: Mapping[str, Mapping[str, Any]]
    forecast_metrics: Mapping[str, ForecastMetrics]
    forecast_panels: Mapping[str, StratifiedForecastPanel]
    attribution: AttributionMetrics
    assessment: DevelopmentStudyAssessment
    query_declaration_sha256: str

    def to_dict(self) -> Dict[str, Any]:
        """Return a portable evidence record."""

        return {
            "schema_version": 1,
            "kind": "real_action_dynamics_study",
            "evaluation_boundary": {
                "corpus_role": "development",
                "fit_run_ids": list(self.training_run_ids),
                "forecast_run_ids": list(self.validation_run_ids),
                "attribution_query_ids": [
                    result.query_id
                    for result in self.attribution.query_results
                ],
                "query_declaration_sha256": (
                    self.query_declaration_sha256
                ),
            },
            "compiler": dict(self.compiler_artifact),
            "models": {
                name: dict(artifact)
                for name, artifact in self.model_artifacts.items()
            },
            "forecast_metrics": {
                name: metric.to_dict()
                for name, metric in self.forecast_metrics.items()
            },
            "forecast_panels": {
                name: panel.to_dict()
                for name, panel in self.forecast_panels.items()
            },
            "attribution": self.attribution.to_dict(),
            "assessment": self.assessment.to_dict(),
            "limitations": [
                "This is open development evidence, not confirmation "
                "evidence or a world-model claim.",
                "Results apply only to the fixed Quantis lab, declared "
                "telemetry graph, and randomized action library.",
                "Normalized MSE is measured in training-fitted feature "
                "coordinates, not raw telemetry units.",
                "Candidate-set attribution ranks preregistered alternatives; "
                "it does not discover arbitrary unseen actions.",
            ],
        }


@dataclass(frozen=True)
class WrittenStudyArtifacts:
    """Paths and identity of one immutable study artifact bundle."""

    output_directory: Path
    manifest_path: Path
    manifest_sha256: str
    artifact_sha256s: Mapping[str, str]


def write_real_corpus_study_artifacts(
    result: RealCorpusStudyResult,
    output_directory: Path,
) -> WrittenStudyArtifacts:
    """Write a non-overwriting, content-addressed JSON artifact bundle."""

    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite study artifacts: {output}"
        )
    if any(
        not name
        or any(
            character not in "abcdefghijklmnopqrstuvwxyz_"
            for character in name
        )
        for name in result.model_artifacts
    ):
        raise ValueError("model artifact name is not path safe")
    output.mkdir(parents=True)
    models = output / "models"
    models.mkdir()
    payloads: Dict[str, Mapping[str, Any]] = {
        "study.json": result.to_dict(),
        "compiler.json": result.compiler_artifact,
        "assessment.json": result.assessment.to_dict(),
    }
    for name, artifact in result.model_artifacts.items():
        payloads[f"models/{name}.json"] = artifact
    artifact_hashes: Dict[str, str] = {}
    for relative_path, payload in payloads.items():
        encoded = _pretty_json_bytes(payload)
        (output / relative_path).write_bytes(encoded)
        artifact_hashes[relative_path] = hashlib.sha256(
            encoded
        ).hexdigest()
    manifest = {
        "schema_version": 1,
        "kind": "real_action_dynamics_artifact_manifest",
        "query_declaration_sha256": (
            result.query_declaration_sha256
        ),
        "artifacts": [
            {
                "path": path,
                "sha256": artifact_hashes[path],
            }
            for path in sorted(artifact_hashes)
        ],
    }
    manifest_bytes = _pretty_json_bytes(manifest)
    manifest_path = output / "artifact-manifest.json"
    manifest_path.write_bytes(manifest_bytes)
    return WrittenStudyArtifacts(
        output_directory=output,
        manifest_path=manifest_path,
        manifest_sha256=hashlib.sha256(
            manifest_bytes
        ).hexdigest(),
        artifact_sha256s=artifact_hashes,
    )


def _pretty_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


class DenseActionVarxDynamics:
    """Dense action-conditioned linear baseline without graph locality."""

    kind = "action_conditioned_dense_varx"
    schema_version = 1

    def __init__(
        self, *, ridge: float = 1e-3, variance_floor: float = 1e-4
    ) -> None:
        if (
            isinstance(ridge, bool)
            or not np.isfinite(ridge)
            or ridge <= 0.0
            or isinstance(variance_floor, bool)
            or not np.isfinite(variance_floor)
            or variance_floor <= 0.0
        ):
            raise ValueError("dense VARX configuration is invalid")
        self.ridge = ridge
        self.variance_floor = variance_floor
        self._graph: Optional[DeclaredTelemetryGraph] = None
        self._state_feature_names: Optional[Tuple[str, ...]] = None
        self._control_feature_names: Optional[Tuple[str, ...]] = None
        self._action_feature_names: Optional[Tuple[str, ...]] = None
        self._coefficients: Optional[NDArray[np.float64]] = None
        self._residual_variance: Optional[NDArray[np.float64]] = None

    def fit(
        self, windows: ActionConditionedWindows
    ) -> "DenseActionVarxDynamics":
        """Fit one global transition from training windows."""

        sample_count = len(windows.histories)
        if sample_count < 2:
            raise ValueError("dense VARX fit requires at least two samples")
        state = windows.histories[:, -1].reshape(sample_count, -1)
        controls = windows.future_controls[:, 0].reshape(
            sample_count, -1
        )
        actions = windows.future_actions[:, 0].reshape(sample_count, -1)
        design = np.concatenate(
            (
                state,
                controls,
                actions,
                np.ones((sample_count, 1), dtype=np.float64),
            ),
            axis=1,
        )
        target = windows.future_states[:, 0].reshape(sample_count, -1)
        penalty = np.eye(design.shape[1], dtype=np.float64)
        penalty[-1, -1] = 0.0
        coefficients = np.linalg.solve(
            design.T @ design + self.ridge * penalty,
            design.T @ target,
        )
        residual = target - design @ coefficients
        self._graph = windows.graph
        self._state_feature_names = windows.state_feature_names
        self._control_feature_names = windows.control_feature_names
        self._action_feature_names = windows.action_feature_names
        self._coefficients = np.asarray(
            coefficients, dtype=np.float64
        )
        self._residual_variance = np.maximum(
            np.mean(np.square(residual), axis=0),
            self.variance_floor,
        ).reshape(
            len(windows.entity_names),
            len(windows.state_feature_names),
        )
        return self

    def rollout(
        self,
        histories: NDArray[np.float64],
        future_controls: NDArray[np.float64],
        future_actions: NDArray[np.float64],
        graph: DeclaredTelemetryGraph,
    ) -> TrajectoryDistribution:
        """Autoregress the dense transition over one requested horizon."""

        (
            fitted_graph,
            state_names,
            control_names,
            action_names,
            coefficients,
            residual_variance,
        ) = self._fitted_values()
        history = np.asarray(histories, dtype=np.float64)
        controls = np.asarray(future_controls, dtype=np.float64)
        actions = np.asarray(future_actions, dtype=np.float64)
        if (
            graph.to_dict() != fitted_graph.to_dict()
            or history.ndim != 4
            or controls.ndim != 3
            or actions.ndim != 4
            or history.shape[0] != controls.shape[0]
            or history.shape[0] != actions.shape[0]
            or controls.shape[1] != actions.shape[1]
            or history.shape[2:]
            != (len(graph.entity_ids), len(state_names))
            or controls.shape[2] != len(control_names)
            or actions.shape[2:]
            != (len(graph.entity_ids), len(action_names))
            or not np.all(np.isfinite(history))
            or not np.all(np.isfinite(controls))
            or not np.all(np.isfinite(actions))
        ):
            raise ValueError("dense VARX rollout inputs are invalid")
        sample_count = history.shape[0]
        horizon = controls.shape[1]
        current = history[:, -1].copy()
        means = np.empty(
            (
                sample_count,
                horizon,
                len(graph.entity_ids),
                len(state_names),
            ),
            dtype=np.float64,
        )
        for step in range(horizon):
            design = np.concatenate(
                (
                    current.reshape(sample_count, -1),
                    controls[:, step],
                    actions[:, step].reshape(sample_count, -1),
                    np.ones((sample_count, 1), dtype=np.float64),
                ),
                axis=1,
            )
            current = (design @ coefficients).reshape(
                sample_count,
                len(graph.entity_ids),
                len(state_names),
            )
            means[:, step] = current
        variances = np.broadcast_to(
            residual_variance,
            means.shape,
        ).copy()
        return TrajectoryDistribution(mean=means, variance=variances)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the fitted dense baseline."""

        (
            graph,
            state_names,
            control_names,
            action_names,
            coefficients,
            residual_variance,
        ) = self._fitted_values()
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "config": {
                "ridge": self.ridge,
                "variance_floor": self.variance_floor,
            },
            "semantic_schema": {
                "graph": graph.to_dict(),
                "state_feature_names": list(state_names),
                "control_feature_names": list(control_names),
                "action_feature_names": list(action_names),
            },
            "state": {
                "coefficients": coefficients.tolist(),
                "residual_variance": residual_variance.tolist(),
            },
        }

    def _fitted_values(
        self,
    ) -> Tuple[
        DeclaredTelemetryGraph,
        Tuple[str, ...],
        Tuple[str, ...],
        Tuple[str, ...],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        if (
            self._graph is None
            or self._state_feature_names is None
            or self._control_feature_names is None
            or self._action_feature_names is None
            or self._coefficients is None
            or self._residual_variance is None
        ):
            raise ValueError("dense VARX is not fitted")
        return (
            self._graph,
            self._state_feature_names,
            self._control_feature_names,
            self._action_feature_names,
            self._coefficients,
            self._residual_variance,
        )


def build_development_validation_queries(
    validation_runs: Sequence[ActionConditionedRun],
    graph: DeclaredTelemetryGraph,
) -> Tuple[AttributionQuery, ...]:
    """Build the frozen treatment/control queries without observations."""

    if len(validation_runs) != 60 or any(
        run.manifest.split != "validation"
        for run in validation_runs
    ):
        raise ValueError(
            "development query builder requires 60 validation runs"
        )
    if any(
        run.graph.to_dict() != graph.to_dict()
        for run in validation_runs
    ):
        raise ValueError("validation query graph does not match")
    pairs: Dict[str, List[ActionConditionedRun]] = {}
    for run in validation_runs:
        pairs.setdefault(run.manifest.matched_pair_id, []).append(run)
    if len(pairs) != 30:
        raise ValueError(
            "development query builder requires 30 validation pairs"
        )
    queries = []
    for pair_id in sorted(pairs):
        pair_runs = pairs[pair_id]
        treatments = [
            run for run in pair_runs if len(run.manifest.actions) == 1
        ]
        controls = [
            run for run in pair_runs if not run.manifest.actions
        ]
        if (
            len(pair_runs) != 2
            or len(treatments) != 1
            or len(controls) != 1
        ):
            raise ValueError(
                "each validation pair needs one treatment and control"
            )
        treatment = treatments[0]
        control = controls[0]
        action = treatment.manifest.actions[0]
        if (
            treatment.manifest.worker_replicas
            != control.manifest.worker_replicas
        ):
            raise ValueError("validation pair topology differs")
        onset = action.start_index
        candidates = _candidate_plans_at_onset(onset)
        true_variant_id = _variant_candidate_id(
            action.action_kind,
            action.target_entity,
            action.magnitude,
            action.duration,
        )
        candidate_ids = {
            candidate.candidate_id for candidate in candidates
        }
        if true_variant_id not in candidate_ids:
            raise ValueError(
                "validation treatment is outside frozen candidate grid"
            )
        for label, run in (
            ("treatment", treatment),
            ("control", control),
        ):
            queries.append(
                AttributionQuery(
                    query_id=f"{pair_id}:{label}",
                    validation_case_id=run.manifest.case_id,
                    transition_index=onset - 1,
                    candidates=candidates,
                    no_action_candidate_id=_NO_ACTION_CANDIDATE_ID,
                    expected_action_kind=(
                        action.action_kind
                        if label == "treatment"
                        else None
                    ),
                    expected_target_entity=(
                        action.target_entity
                        if label == "treatment"
                        else None
                    ),
                    expected_variant_candidate_id=(
                        true_variant_id
                        if label == "treatment"
                        else None
                    ),
                )
            )
    return tuple(queries)


def _candidate_plans_at_onset(
    onset: int,
) -> Tuple[AttributionCandidatePlan, ...]:
    candidates = [
        AttributionCandidatePlan(
            candidate_id=_NO_ACTION_CANDIDATE_ID,
            actions=(),
        )
    ]
    for variant in FROZEN_ACTION_CANDIDATE_GRID:
        candidates.append(
            AttributionCandidatePlan(
                candidate_id=variant.candidate_id,
                actions=(
                    InterventionAction(
                        action_id=variant.candidate_id,
                        action_kind=variant.action_kind,
                        target_entity=variant.target_entity,
                        start_index=onset,
                        stop_index=onset + variant.duration,
                        magnitude=variant.magnitude,
                    ),
                ),
            )
        )
    return tuple(candidates)


def _variant_candidate_id(
    action_kind: str,
    target_entity: str,
    magnitude: float,
    duration: int,
) -> str:
    return (
        f"action:{action_kind}@{target_entity}:"
        f"m={magnitude:g}:d={duration}"
    )


def train_and_evaluate_real_corpus(
    *,
    runs: Sequence[RealCorpusRun],
    graph: DeclaredTelemetryGraph,
    queries: Sequence[AttributionQuery],
    config: RealCorpusStudyConfig,
) -> RealCorpusStudyResult:
    """Fit training-only baselines and score validation-only evidence."""

    if any(run.corpus_role != "development" for run in runs):
        raise ValueError("real-corpus study accepts development corpus only")
    if not runs:
        raise ValueError("real-corpus study requires runs")
    raw_runs = tuple(admitted.run for admitted in runs)
    if any(
        run.graph.to_dict() != graph.to_dict() for run in raw_runs
    ):
        raise ValueError("all runs must use the declared study graph")
    case_ids = tuple(run.manifest.case_id for run in raw_runs)
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("case ids must be unique")
    if any(
        run.manifest.split not in {"training", "validation"}
        for run in raw_runs
    ):
        raise ValueError(
            "real-corpus study accepts training and validation splits only"
        )
    training_runs = tuple(
        run for run in raw_runs if run.manifest.split == "training"
    )
    validation_runs = tuple(
        run for run in raw_runs if run.manifest.split == "validation"
    )
    if not training_runs or not validation_runs:
        raise ValueError(
            "real-corpus study requires training and validation runs"
        )
    if not queries:
        raise ValueError("study requires preregistered queries")
    _validate_queries(queries, validation_runs, graph)

    compiler = ActionTrajectoryCompiler(
        context_length=config.context_length,
        rollout_horizon=config.rollout_horizon,
    ).fit(training_runs)
    training = compiler.transform(training_runs)
    validation = compiler.transform(validation_runs)
    graph_action = GraphVarxDynamics(
        GraphVarxConfig(
            ridge=config.ridge,
            include_actions=True,
            variance_floor=config.variance_floor,
        )
    ).fit(training)
    graph_agnostic = GraphVarxDynamics(
        GraphVarxConfig(
            ridge=config.ridge,
            include_actions=False,
            variance_floor=config.variance_floor,
        )
    ).fit(training)
    dense_action = DenseActionVarxDynamics(
        ridge=config.ridge,
        variance_floor=config.variance_floor,
    ).fit(training)

    predictions = {
        "action_conditioned_graph_varx": graph_action.rollout(
            validation.histories,
            validation.future_controls,
            validation.future_actions,
            graph,
        ),
        "action_agnostic_graph_varx": graph_agnostic.rollout(
            validation.histories,
            validation.future_controls,
            validation.future_actions,
            graph,
        ),
        "action_conditioned_dense_varx": dense_action.rollout(
            validation.histories,
            validation.future_controls,
            validation.future_actions,
            graph,
        ),
        "persistence": persistence_rollout(
            validation.histories,
            config.rollout_horizon,
        ),
    }
    (
        panel_selections,
        action_kinds,
        worker_replicas,
    ) = _panel_selections(
        validation=validation,
        validation_runs=validation_runs,
        graph=graph,
    )
    action_overlap = np.any(
        panel_selections["action_overlap"],
        axis=2,
    )
    if not np.any(action_overlap):
        raise ValueError(
            "validation corpus has no action-overlap forecast steps"
        )
    metrics = {
        name: _forecast_metrics(
            distribution.mean,
            validation.future_states,
            action_overlap,
        )
        for name, distribution in predictions.items()
    }
    prediction_means = {
        name: distribution.mean
        for name, distribution in predictions.items()
    }
    paired_predictions, paired_observed = _paired_effect_trajectories(
        predictions=prediction_means,
        observed=validation.future_states,
        validation=validation,
        validation_runs=validation_runs,
    )
    panels = {
        name: _stratified_forecast_panel(
            predictions=(
                prediction_means
                if name in {
                    "all_forecast_states",
                    "action_overlap",
                }
                else paired_predictions
            ),
            observed=(
                validation.future_states
                if name in {
                    "all_forecast_states",
                    "action_overlap",
                }
                else paired_observed
            ),
            selection=selection,
            action_kinds=action_kinds,
            worker_replicas=worker_replicas,
            comparison_kind=(
                "state_forecast"
                if name in {
                    "all_forecast_states",
                    "action_overlap",
                }
                else "paired_treatment_minus_control"
            ),
        )
        for name, selection in panel_selections.items()
    }
    attribution = _evaluate_queries(
        queries=queries,
        validation_runs=validation_runs,
        validation=validation,
        compiler=compiler,
        model=graph_action,
        graph=graph,
    )
    assessment = assess_development_gates(
        action_overlap_mse=metrics[
            "action_conditioned_graph_varx"
        ].normalized_mse_action_overlap,
        action_agnostic_overlap_mse=metrics[
            "action_agnostic_graph_varx"
        ].normalized_mse_action_overlap,
        persistence_overlap_mse=metrics[
            "persistence"
        ].normalized_mse_action_overlap,
        graph_downstream_mse=panels[
            "downstream_entity_intervention_effect"
        ].normalized_mse["action_conditioned_graph_varx"],
        dense_downstream_mse=panels[
            "downstream_entity_intervention_effect"
        ].normalized_mse["action_conditioned_dense_varx"],
        action_location_hit_at_1=(
            attribution.action_and_target_hit_at_1
        ),
        no_action_specificity=attribution.no_action_specificity,
    )
    model_artifacts: Dict[str, Mapping[str, Any]] = {
        "action_conditioned_graph_varx": graph_action.to_dict(),
        "action_agnostic_graph_varx": graph_agnostic.to_dict(),
        "action_conditioned_dense_varx": dense_action.to_dict(),
    }
    return RealCorpusStudyResult(
        training_run_ids=tuple(
            run.manifest.case_id for run in training_runs
        ),
        validation_run_ids=tuple(
            run.manifest.case_id for run in validation_runs
        ),
        compiler_artifact=compiler.to_dict(),
        model_artifacts=model_artifacts,
        forecast_metrics=metrics,
        forecast_panels=panels,
        attribution=attribution,
        assessment=assessment,
        query_declaration_sha256=_query_declaration_sha256(queries),
    )


def _forecast_metrics(
    prediction: NDArray[np.float64],
    observed: NDArray[np.float64],
    action_overlap: NDArray[np.bool_],
) -> ForecastMetrics:
    squared_error = np.square(prediction - observed)
    return ForecastMetrics(
        normalized_mse_overall=float(np.mean(squared_error)),
        normalized_mse_action_overlap=float(
            np.mean(squared_error[action_overlap])
        ),
    )


def _panel_selections(
    *,
    validation: ActionConditionedWindows,
    validation_runs: Sequence[ActionConditionedRun],
    graph: DeclaredTelemetryGraph,
) -> Tuple[
    Mapping[str, NDArray[np.bool_]],
    Tuple[Optional[str], ...],
    Tuple[Optional[int], ...],
]:
    shape = validation.future_states.shape[:3]
    targeted = np.zeros(shape, dtype=np.bool_)
    downstream = np.zeros(shape, dtype=np.bool_)
    recovery = np.zeros(shape, dtype=np.bool_)
    pair_action_kinds = {
        run.manifest.matched_pair_id: run.manifest.actions[0].action_kind
        for run in validation_runs
        if len(run.manifest.actions) == 1
    }
    run_by_id = {
        run.manifest.case_id: run for run in validation_runs
    }
    action_kinds: List[Optional[str]] = []
    worker_replicas: List[Optional[int]] = []
    for sample, case_id in enumerate(validation.trajectory_ids):
        run = run_by_id[case_id]
        action_kinds.append(
            pair_action_kinds.get(run.manifest.matched_pair_id)
        )
        worker_replicas.append(run.manifest.worker_replicas)
        if len(run.manifest.actions) != 1:
            continue
        action = run.manifest.actions[0]
        target_position = graph.entity_ids.index(
            action.target_entity
        )
        downstream_positions = tuple(
            graph.entity_ids.index(entity_id)
            for entity_id in _downstream_entity_ids(
                graph, action.target_entity
            )
        )
        transition = int(validation.transition_indices[sample])
        for horizon_step in range(shape[1]):
            action_index = transition + horizon_step
            if action.start_index <= action_index < action.stop_index:
                targeted[sample, horizon_step, target_position] = True
                downstream[
                    sample, horizon_step, downstream_positions
                ] = True
            if action_index >= action.stop_index:
                recovery[sample, horizon_step, :] = True
    active_steps = np.any(targeted, axis=2)
    active_entities = np.broadcast_to(
        active_steps[:, :, np.newaxis], shape
    ).copy()
    selections = {
        "all_forecast_states": np.ones(shape, dtype=np.bool_),
        "action_overlap": active_entities,
        "target_entity_intervention_effect": targeted,
        "downstream_entity_intervention_effect": downstream,
        "recovery": recovery,
    }
    if any(not np.any(selection) for selection in selections.values()):
        raise ValueError(
            "validation windows do not identify every forecast panel"
        )
    return selections, tuple(action_kinds), tuple(worker_replicas)


def _downstream_entity_ids(
    graph: DeclaredTelemetryGraph, source_id: str
) -> Tuple[str, ...]:
    adjacency: Dict[str, List[str]] = {
        entity_id: [] for entity_id in graph.entity_ids
    }
    for entity in graph.entities:
        if entity.kind == "edge":
            assert entity.source is not None
            assert entity.target is not None
            adjacency[entity.source].append(entity.entity_id)
            adjacency[entity.entity_id].append(entity.target)
    visited = {source_id}
    frontier = [source_id]
    while frontier:
        current = frontier.pop(0)
        for candidate in adjacency[current]:
            if candidate not in visited:
                visited.add(candidate)
                frontier.append(candidate)
    return tuple(
        entity_id
        for entity_id in graph.entity_ids
        if entity_id in visited and entity_id != source_id
    )


def _stratified_forecast_panel(
    *,
    predictions: Mapping[str, NDArray[np.float64]],
    observed: NDArray[np.float64],
    selection: NDArray[np.bool_],
    action_kinds: Tuple[Optional[str], ...],
    worker_replicas: Tuple[Optional[int], ...],
    comparison_kind: str,
) -> StratifiedForecastPanel:
    def selected_mse(
        values: NDArray[np.float64],
        selected: NDArray[np.bool_],
    ) -> float:
        expanded = np.broadcast_to(
            selected[:, :, :, np.newaxis], observed.shape
        )
        if not np.any(expanded):
            raise ValueError("forecast panel slice is empty")
        return float(np.mean(np.square(values - observed)[expanded]))

    overall = {
        name: selected_mse(values, selection)
        for name, values in predictions.items()
    }
    by_action: Dict[str, Mapping[str, float]] = {}
    for action_kind in ACTION_KINDS:
        sample_mask = np.asarray(
            [value == action_kind for value in action_kinds],
            dtype=np.bool_,
        )
        selected = selection & sample_mask[:, np.newaxis, np.newaxis]
        if np.any(selected):
            by_action[action_kind] = {
                name: selected_mse(values, selected)
                for name, values in predictions.items()
            }
    by_topology: Dict[str, Mapping[str, float]] = {}
    for replica_count in sorted(
        {
            value
            for value in worker_replicas
            if value is not None
        }
    ):
        sample_mask = np.asarray(
            [value == replica_count for value in worker_replicas],
            dtype=np.bool_,
        )
        selected = selection & sample_mask[:, np.newaxis, np.newaxis]
        key = str(replica_count)
        by_topology[key] = {
            name: selected_mse(values, selected)
            for name, values in predictions.items()
        }
    return StratifiedForecastPanel(
        normalized_mse=overall,
        by_action_kind=by_action,
        by_worker_replicas=by_topology,
        selected_state_count=int(np.count_nonzero(selection)),
        comparison_kind=comparison_kind,
    )


def _paired_effect_trajectories(
    *,
    predictions: Mapping[str, NDArray[np.float64]],
    observed: NDArray[np.float64],
    validation: ActionConditionedWindows,
    validation_runs: Sequence[ActionConditionedRun],
) -> Tuple[
    Mapping[str, NDArray[np.float64]],
    NDArray[np.float64],
]:
    run_by_id = {
        run.manifest.case_id: run for run in validation_runs
    }
    controls_by_pair = {
        run.manifest.matched_pair_id: run.manifest.case_id
        for run in validation_runs
        if not run.manifest.actions
    }
    window_position = {
        (case_id, int(validation.transition_indices[index])): index
        for index, case_id in enumerate(validation.trajectory_ids)
    }
    paired_observed = np.zeros_like(observed)
    paired_predictions = {
        name: np.zeros_like(values)
        for name, values in predictions.items()
    }
    for treatment_position, case_id in enumerate(
        validation.trajectory_ids
    ):
        run = run_by_id[case_id]
        if len(run.manifest.actions) != 1:
            continue
        pair_id = run.manifest.matched_pair_id
        control_case_id = controls_by_pair.get(pair_id)
        if control_case_id is None:
            raise ValueError(
                "paired effect panel cannot find matched control"
            )
        transition = int(
            validation.transition_indices[treatment_position]
        )
        control_position = window_position.get(
            (control_case_id, transition)
        )
        if control_position is None:
            raise ValueError(
                "paired effect panel cannot align treatment and control"
            )
        paired_observed[treatment_position] = (
            observed[treatment_position]
            - observed[control_position]
        )
        for name, values in predictions.items():
            paired_predictions[name][treatment_position] = (
                values[treatment_position] - values[control_position]
            )
    return paired_predictions, paired_observed


def _validate_queries(
    queries: Sequence[AttributionQuery],
    validation_runs: Sequence[ActionConditionedRun],
    graph: DeclaredTelemetryGraph,
) -> None:
    query_ids = tuple(query.query_id for query in queries)
    validation_ids = {
        run.manifest.case_id for run in validation_runs
    }
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("attribution query ids must be unique")
    if any(
        query.validation_case_id not in validation_ids
        for query in queries
    ):
        raise ValueError(
            "attribution queries must reference validation runs only"
        )
    if not any(
        query.expected_action_kind is not None for query in queries
    ) or not any(
        query.expected_action_kind is None for query in queries
    ):
        raise ValueError(
            "queries must include action and no-action cases"
        )
    entity_ids = set(graph.entity_ids)
    for query in queries:
        if (
            query.expected_target_entity is not None
            and query.expected_target_entity not in entity_ids
        ):
            raise ValueError("expected query target is outside graph")
        for candidate in query.candidates:
            if any(
                action.target_entity not in entity_ids
                for action in candidate.actions
            ):
                raise ValueError("candidate target is outside graph")
        if query.expected_action_kind is not None and not any(
            _candidate_label(candidate)
            == (
                query.expected_action_kind,
                query.expected_target_entity,
            )
            for candidate in query.candidates
        ):
            raise ValueError(
                "expected action family and target must be a candidate"
            )


def _evaluate_queries(
    *,
    queries: Sequence[AttributionQuery],
    validation_runs: Sequence[ActionConditionedRun],
    validation: ActionConditionedWindows,
    compiler: ActionTrajectoryCompiler,
    model: GraphVarxDynamics,
    graph: DeclaredTelemetryGraph,
) -> AttributionMetrics:
    run_by_id = {
        run.manifest.case_id: run for run in validation_runs
    }
    results: List[AttributionQueryResult] = []
    family_hits: List[bool] = []
    location_hits: List[bool] = []
    joint_hits: List[bool] = []
    exact_hits: List[bool] = []
    family_top_three_hits: List[bool] = []
    no_action_hits: List[bool] = []
    for query in queries:
        matching = np.flatnonzero(
            np.asarray(validation.trajectory_ids)
            == query.validation_case_id
        )
        matching = matching[
            validation.transition_indices[matching]
            == query.transition_index
        ]
        if len(matching) != 1:
            raise ValueError(
                "query transition is not a unique validation window"
            )
        window_index = int(matching[0])
        source_run = run_by_id[query.validation_case_id]
        candidates = tuple(
            RolloutCandidate(
                candidate_id=candidate.candidate_id,
                future_actions=compiler.compile_action_trajectory(
                    point_count=source_run.manifest.point_count,
                    actions=candidate.actions,
                    graph=graph,
                )[
                    query.transition_index : query.transition_index
                    + validation.future_states.shape[1]
                ],
            )
            for candidate in query.candidates
        )
        if any(
            candidate.future_actions.shape[0]
            != validation.future_states.shape[1]
            for candidate in candidates
        ):
            raise ValueError(
                "query candidate does not cover the rollout horizon"
            )
        ranked = rank_action_candidates(
            model=model,
            history=validation.histories[window_index],
            future_controls=validation.future_controls[window_index],
            observed_future=validation.future_states[window_index],
            candidates=candidates,
            graph=graph,
            no_action_candidate_id=query.no_action_candidate_id,
        )
        winner_id = ranked.ranked_candidate_ids[0]
        winner = next(
            candidate
            for candidate in query.candidates
            if candidate.candidate_id == winner_id
        )
        if query.expected_action_kind is None:
            family_hit: Optional[bool] = None
            location_hit: Optional[bool] = None
            joint_hit: Optional[bool] = None
            exact_hit: Optional[bool] = None
            family_top_three_hit: Optional[bool] = None
            nominal_hit = (
                winner_id == query.no_action_candidate_id
            )
            no_action_hit: Optional[bool] = nominal_hit
            no_action_hits.append(nominal_hit)
        else:
            winner_family, winner_target = _candidate_label(winner)
            family_hit = (
                winner_family == query.expected_action_kind
            )
            location_hit = (
                winner_target == query.expected_target_entity
            )
            joint_hit = family_hit and location_hit
            exact_hit = (
                winner_id == query.expected_variant_candidate_id
            )
            candidate_by_id = {
                candidate.candidate_id: candidate
                for candidate in query.candidates
            }
            family_top_three_hit = any(
                _candidate_label(candidate_by_id[candidate_id])[0]
                == query.expected_action_kind
                for candidate_id in ranked.ranked_candidate_ids[:3]
            )
            no_action_hit = None
            family_hits.append(family_hit)
            location_hits.append(location_hit)
            joint_hits.append(joint_hit)
            exact_hits.append(exact_hit)
            family_top_three_hits.append(family_top_three_hit)
        likelihood_margin = float(
            ranked.negative_log_likelihoods[1]
            - ranked.negative_log_likelihoods[0]
        )
        results.append(
            AttributionQueryResult(
                query_id=query.query_id,
                winning_candidate_id=winner_id,
                ranked_candidate_ids=ranked.ranked_candidate_ids,
                negative_log_likelihoods=(
                    ranked.negative_log_likelihoods
                ),
                action_family_hit_at_1=family_hit,
                action_location_hit_at_1=location_hit,
                action_and_target_hit_at_1=joint_hit,
                exact_variant_hit_at_1=exact_hit,
                action_family_hit_at_3=family_top_three_hit,
                no_action_hit_at_1=no_action_hit,
                likelihood_margin=likelihood_margin,
                worker_replicas=(
                    source_run.manifest.worker_replicas
                ),
            )
        )
    by_topology = _attribution_by_topology(results)
    return AttributionMetrics(
        query_count=len(results),
        action_query_count=len(family_hits),
        no_action_query_count=len(no_action_hits),
        action_family_hit_at_1=float(np.mean(family_hits)),
        action_location_hit_at_1=float(np.mean(location_hits)),
        action_and_target_hit_at_1=float(np.mean(joint_hits)),
        exact_variant_hit_at_1=float(np.mean(exact_hits)),
        action_family_hit_at_3=float(
            np.mean(family_top_three_hits)
        ),
        no_action_specificity=float(np.mean(no_action_hits)),
        mean_likelihood_margin=float(
            np.mean(
                [result.likelihood_margin for result in results]
            )
        ),
        by_worker_replicas=by_topology,
        query_results=tuple(results),
    )


def _attribution_by_topology(
    results: Sequence[AttributionQueryResult],
) -> Mapping[str, Mapping[str, float]]:
    panels: Dict[str, Mapping[str, float]] = {}
    for worker_replicas in sorted(
        {result.worker_replicas for result in results}
    ):
        selected = [
            result
            for result in results
            if result.worker_replicas == worker_replicas
        ]
        action = [
            result
            for result in selected
            if result.action_and_target_hit_at_1 is not None
        ]
        nominal = [
            result
            for result in selected
            if result.no_action_hit_at_1 is not None
        ]
        values: Dict[str, float] = {
            "mean_likelihood_margin": float(
                np.mean(
                    [result.likelihood_margin for result in selected]
                )
            )
        }
        if action:
            values.update(
                {
                    "action_and_target_hit_at_1": float(
                        np.mean(
                            [
                                bool(
                                    result.action_and_target_hit_at_1
                                )
                                for result in action
                            ]
                        )
                    ),
                    "exact_variant_hit_at_1": float(
                        np.mean(
                            [
                                bool(
                                    result.exact_variant_hit_at_1
                                )
                                for result in action
                            ]
                        )
                    ),
                    "action_family_hit_at_3": float(
                        np.mean(
                            [
                                bool(
                                    result.action_family_hit_at_3
                                )
                                for result in action
                            ]
                        )
                    ),
                }
            )
        if nominal:
            values["no_action_specificity"] = float(
                np.mean(
                    [
                        bool(result.no_action_hit_at_1)
                        for result in nominal
                    ]
                )
            )
        panels[str(worker_replicas)] = values
    return panels


def _candidate_label(
    candidate: AttributionCandidatePlan,
) -> Tuple[Optional[str], Optional[str]]:
    if not candidate.actions:
        return None, None
    action_kinds = {
        action.action_kind for action in candidate.actions
    }
    targets = {
        action.target_entity for action in candidate.actions
    }
    if len(action_kinds) != 1 or len(targets) != 1:
        return None, None
    return next(iter(action_kinds)), next(iter(targets))


def _query_declaration_sha256(
    queries: Sequence[AttributionQuery],
) -> str:
    payload = [query.to_dict() for query in queries]
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
