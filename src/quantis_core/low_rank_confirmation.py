"""Frozen contract and decision rule for low-rank action-dynamics confirmation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Dict, Mapping, Sequence, Tuple, cast

import numpy as np
from numpy.typing import NDArray

from .action_dynamics_lab import ActionCollectionProtocol
from .graph_telemetry import DeclaredTelemetryGraph


_SEED = 26073042
_DRAWS = 99_999


@dataclass(frozen=True)
class LowRankConfirmationContract:
    """Strict public view of the sealed low-rank confirmation contract."""

    payload: Mapping[str, Any]

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "LowRankConfirmationContract":
        """Restore and validate the exact v1 contract."""

        required = {
            "schema_version",
            "kind",
            "status",
            "base_collection_protocol",
            "generator_seed",
            "evidence_boundary",
            "candidate",
            "collection",
            "decision_gates",
            "execution",
            "claim",
        }
        if (
            set(payload) != required
            or payload.get("schema_version") != 2
            or payload.get("kind")
            != "low_rank_action_dynamics_confirmation_contract"
            or payload.get("status") != "frozen_pre_collection"
            or payload.get("generator_seed") != 26073051
        ):
            raise ValueError("low-rank confirmation contract is invalid")
        contract = cls(
            json.loads(json.dumps(payload, sort_keys=True))
        )
        contract._validate()
        return contract

    def _validate(self) -> None:
        base = _mapping(self.payload, "base_collection_protocol")
        candidate = _mapping(self.payload, "candidate")
        collection = _mapping(self.payload, "collection")
        gates = _mapping(self.payload, "decision_gates")
        execution = _mapping(self.payload, "execution")
        claim = _mapping(self.payload, "claim")
        if (
            base.get("path")
            != "lab/action_dynamics/development-protocol-v1.json"
            or not _is_sha256(base.get("canonical_sha256"))
            or candidate.get("kind")
            != "contractive_low_rank_action_dynamics"
            or candidate.get("rank") != 32
            or candidate.get("maximum_spectral_radius") != 0.98
            or not _is_sha256(candidate.get("model_sha256"))
            or not _is_sha256(
                candidate.get("development_artifact_manifest_sha256")
            )
            or not _is_sha256(
                candidate.get("compiler_metadata_sha256")
            )
            or not _is_sha256(
                candidate.get("compiler_artifact_manifest_sha256")
            )
            or collection.get("pair_count") != 120
            or collection.get("capture_count") != 240
            or collection.get("parallel_jobs") != 6
            or collection.get("automatic_retry") is not False
            or collection.get("overwrite") is not False
            or gates.get("aggregate_action_mse_ratio_max") != 0.75
            or gates.get("per_family_action_mse_ratio_max") != 0.90
            or gates.get("downstream_effect_mse_ratio_max") != 0.80
            or gates.get("paired_sign_flip_p_value_max") != 0.05
            or gates.get("paired_sign_flip_seed") != _SEED
            or gates.get("paired_sign_flip_draws") != _DRAWS
            or gates.get("paired_sign_flip_unit") != "matched_pair"
            or gates.get("paired_sign_flip_tail")
            != "candidate_minus_control_less_than_or_equal"
            or gates.get("spectral_radius_max") != 0.98
            or gates.get("parameter_count_max") != 40_000
            or gates.get("serialized_size_bytes_max") != 1_048_576
            or gates.get("rollout_finite_required") is not True
            or set(execution)
            != {"contract_module", "runner", "independent_assessor"}
            or any(
                not isinstance(value, Mapping)
                or not isinstance(value.get("path"), str)
                or not _is_sha256(value.get("sha256"))
                for value in execution.values()
            )
            or claim.get("pass_decision")
            != "confirm_learnable_action_dynamics"
            or claim.get("failure_decision")
            != "do_not_confirm_learnable_action_dynamics"
        ):
            raise ValueError("low-rank confirmation choices drifted")

    @property
    def base_protocol_sha256(self) -> str:
        return str(
            _mapping(
                self.payload, "base_collection_protocol"
            )["canonical_sha256"]
        )

    def base_protocol(
        self, payload: Mapping[str, Any]
    ) -> ActionCollectionProtocol:
        """Restore the declared transport protocol without confirmation edits."""

        return ActionCollectionProtocol.from_dict(payload)

    def materialize_collection_protocol(
        self,
        base_payload: Mapping[str, Any],
        *,
        execution_source_commit: str,
    ) -> ActionCollectionProtocol:
        """Bind this contract into a fresh collector-compatible protocol."""

        if _sha256(base_payload) != self.base_protocol_sha256:
            raise ValueError("base protocol identity differs from contract")
        if not _is_git_commit(execution_source_commit):
            raise ValueError("execution source commit is invalid")
        base = self.base_protocol(base_payload)
        if base.stage != "development":
            raise ValueError("base protocol is not collector-compatible")
        payload = base.to_dict()
        payload["generator_seed"] = int(self.payload["generator_seed"])
        payload["evidence_boundary"] = str(
            self.payload["evidence_boundary"]
        )
        analysis = dict(cast(Mapping[str, Any], payload["analysis"]))
        analysis.update(
            {
                "authoritative_corpus_role": "sealed_confirmation",
                "model_training_allowed_during_collection": False,
                "model_training_allowed_after_collection": False,
                "all_pairs_have_one_confirmation_role": True,
                "transport_split_labels_are_non_analytic": True,
                "execution_source_commit": execution_source_commit,
            }
        )
        payload["analysis"] = analysis
        claim = dict(cast(Mapping[str, Any], payload["claim"]))
        claim.update(
            {
                "supported": self.payload["claim"]["supported"],
                "excluded": self.payload["claim"]["excluded"],
                "failure_outcome": self.payload["claim"][
                    "failure_outcome"
                ],
                "confirmation_contract": self.to_dict(),
                "execution_source_commit": execution_source_commit,
            }
        )
        payload["claim"] = claim
        return ActionCollectionProtocol.from_dict(payload)

    def to_dict(self) -> Dict[str, Any]:
        restored = json.loads(json.dumps(self.payload, sort_keys=True))
        if not isinstance(restored, dict):
            raise AssertionError("serialized contract changed type")
        return cast(Dict[str, Any], restored)


def paired_sign_flip_p_value(
    candidate_minus_control: NDArray[Any],
    *,
    seed: int,
    draws: int,
) -> float:
    """Return the frozen one-sided pair-blocked Monte Carlo p-value."""

    differences = np.asarray(
        candidate_minus_control, dtype=np.float64
    )
    if (
        differences.ndim != 1
        or not len(differences)
        or not np.all(np.isfinite(differences))
        or seed != _SEED
        or draws != _DRAWS
    ):
        raise ValueError("paired sign-flip inputs differ from contract")
    observed = float(np.mean(differences))
    generator = np.random.default_rng(seed)
    extreme = 0
    remaining = draws
    while remaining:
        count = min(remaining, 2048)
        raw_signs = generator.integers(
            0, 2, size=(count, len(differences)), dtype=np.int8
        )
        signs = raw_signs.astype(np.float64) * 2.0 - 1.0
        randomized = np.mean(signs * differences[None, :], axis=1)
        extreme += int(np.count_nonzero(randomized <= observed))
        remaining -= count
    return float((extreme + 1) / (draws + 1))


def assess_low_rank_confirmation_arrays(
    *,
    pair_ids: Sequence[str],
    action_kind_by_pair: Mapping[str, str],
    candidate_action_mse: NDArray[Any],
    action_masked_action_mse: NDArray[Any],
    persistence_action_mse: NDArray[Any],
    candidate_downstream_effect_mse: float,
    action_masked_downstream_effect_mse: float,
    persistence_downstream_effect_mse: float,
    spectral_radius: float,
    parameter_count: int,
    serialized_size_bytes: int,
    rollout_finite: bool,
    seed: int,
    draws: int,
) -> Mapping[str, Any]:
    """Apply the frozen conjunction to pair-balanced stored scores."""

    ids = tuple(str(value) for value in pair_ids)
    candidate = _loss_vector(candidate_action_mse, len(ids))
    action_masked = _loss_vector(
        action_masked_action_mse, len(ids)
    )
    persistence = _loss_vector(
        persistence_action_mse, len(ids)
    )
    if (
        len(ids) != 120
        or len(set(ids)) != len(ids)
        or set(action_kind_by_pair) != set(ids)
        or any(not action_kind_by_pair[pair_id] for pair_id in ids)
    ):
        raise ValueError("confirmation pair identities do not align")
    family_counts = {
        action_kind: sum(
            action_kind_by_pair[pair_id] == action_kind
            for pair_id in ids
        )
        for action_kind in set(action_kind_by_pair.values())
    }
    if len(family_counts) != 5 or set(family_counts.values()) != {24}:
        raise ValueError("confirmation action-family coverage is incomplete")
    scalars = (
        candidate_downstream_effect_mse,
        action_masked_downstream_effect_mse,
        persistence_downstream_effect_mse,
        spectral_radius,
    )
    if (
        not all(np.isfinite(value) and value >= 0.0 for value in scalars)
        or isinstance(parameter_count, bool)
        or parameter_count < 1
        or isinstance(serialized_size_bytes, bool)
        or serialized_size_bytes < 1
    ):
        raise ValueError("confirmation scalar metrics are invalid")

    aggregate = {
        "candidate_action_mse": float(np.mean(candidate)),
        "action_masked_action_mse": float(np.mean(action_masked)),
        "persistence_action_mse": float(np.mean(persistence)),
        "candidate_to_action_masked_ratio": _ratio(
            float(np.mean(candidate)),
            float(np.mean(action_masked)),
        ),
        "candidate_to_persistence_ratio": _ratio(
            float(np.mean(candidate)), float(np.mean(persistence))
        ),
        "candidate_downstream_effect_mse": float(
            candidate_downstream_effect_mse
        ),
        "action_masked_downstream_effect_mse": float(
            action_masked_downstream_effect_mse
        ),
        "persistence_downstream_effect_mse": float(
            persistence_downstream_effect_mse
        ),
        "candidate_to_action_masked_downstream_ratio": _ratio(
            candidate_downstream_effect_mse,
            action_masked_downstream_effect_mse,
        ),
        "candidate_to_persistence_downstream_ratio": _ratio(
            candidate_downstream_effect_mse,
            persistence_downstream_effect_mse,
        ),
    }
    family_rows: Dict[str, Mapping[str, float]] = {}
    family_passes = []
    for action_kind in sorted(set(action_kind_by_pair.values())):
        mask = np.asarray(
            [
                action_kind_by_pair[pair_id] == action_kind
                for pair_id in ids
            ],
            dtype=np.bool_,
        )
        candidate_mean = float(np.mean(candidate[mask]))
        masked_mean = float(np.mean(action_masked[mask]))
        persistence_mean = float(np.mean(persistence[mask]))
        masked_ratio = _ratio(candidate_mean, masked_mean)
        persistence_ratio = _ratio(candidate_mean, persistence_mean)
        passed = masked_ratio <= 0.90 and persistence_ratio <= 0.90
        family_passes.append(passed)
        family_rows[action_kind] = {
            "pair_count": int(np.count_nonzero(mask)),
            "candidate_action_mse": candidate_mean,
            "action_masked_action_mse": masked_mean,
            "persistence_action_mse": persistence_mean,
            "candidate_to_action_masked_ratio": masked_ratio,
            "candidate_to_persistence_ratio": persistence_ratio,
            "passed": passed,
        }
    p_masked = paired_sign_flip_p_value(
        candidate - action_masked, seed=seed, draws=draws
    )
    p_persistence = paired_sign_flip_p_value(
        candidate - persistence, seed=seed, draws=draws
    )
    gates = {
        "rollout_finite": bool(rollout_finite),
        "aggregate_action_value_over_masked": (
            aggregate["candidate_to_action_masked_ratio"] <= 0.75
        ),
        "aggregate_action_value_over_persistence": (
            aggregate["candidate_to_persistence_ratio"] <= 0.75
        ),
        "downstream_action_value_over_masked": (
            aggregate[
                "candidate_to_action_masked_downstream_ratio"
            ]
            <= 0.80
        ),
        "downstream_action_value_over_persistence": (
            aggregate[
                "candidate_to_persistence_downstream_ratio"
            ]
            <= 0.80
        ),
        "every_action_family_improves": all(family_passes),
        "paired_significance_over_masked": p_masked <= 0.05,
        "paired_significance_over_persistence": (
            p_persistence <= 0.05
        ),
        "contractive_transition": spectral_radius <= 0.98,
        "parameter_envelope": parameter_count <= 40_000,
        "serialized_size_envelope": (
            serialized_size_bytes <= 1_048_576
        ),
    }
    confirmed = all(gates.values())
    return {
        "schema_version": 1,
        "kind": "low_rank_action_dynamics_confirmation_assessment",
        "status": "confirmed" if confirmed else "not_confirmed",
        "decision": (
            "confirm_learnable_action_dynamics"
            if confirmed
            else "do_not_confirm_learnable_action_dynamics"
        ),
        "pair_count": len(ids),
        "aggregate": aggregate,
        "action_families": family_rows,
        "paired_sign_flip": {
            "seed": seed,
            "draws": draws,
            "candidate_vs_action_masked_p_value": p_masked,
            "candidate_vs_persistence_p_value": p_persistence,
        },
        "edge_envelope": {
            "spectral_radius": spectral_radius,
            "parameter_count": parameter_count,
            "serialized_size_bytes": serialized_size_bytes,
        },
        "gates": gates,
    }


def pair_balanced_action_mse(
    *,
    prediction: NDArray[Any],
    observed: NDArray[Any],
    future_actions: NDArray[Any],
    matched_pair_ids: Sequence[str],
) -> Tuple[Tuple[str, ...], NDArray[np.float64]]:
    """Return one action-overlap forecast loss per matched pair."""

    predicted = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(observed, dtype=np.float64)
    actions = np.asarray(future_actions, dtype=np.float64)
    pair_ids = tuple(str(value) for value in matched_pair_ids)
    if (
        predicted.shape != target.shape
        or predicted.ndim != 4
        or actions.ndim != 4
        or predicted.shape[:2] != actions.shape[:2]
        or len(pair_ids) != len(predicted)
        or not np.all(np.isfinite(predicted))
        or not np.all(np.isfinite(target))
        or not np.all(np.isfinite(actions))
    ):
        raise ValueError("stored confirmation arrays do not align")
    active = np.any(actions[..., 1] > 0.5, axis=2)
    squared = np.square(predicted - target)
    unique_pairs = tuple(sorted(set(pair_ids)))
    losses = []
    raw_pair_ids = np.asarray(pair_ids)
    for pair_id in unique_pairs:
        selected = raw_pair_ids == pair_id
        overlap = active[selected]
        if not np.any(overlap):
            raise ValueError("confirmation pair has no action overlap")
        losses.append(float(np.mean(squared[selected][overlap])))
    return unique_pairs, np.asarray(losses, dtype=np.float64)


def downstream_effect_mse(
    *,
    prediction: NDArray[Any],
    observed: NDArray[Any],
    future_actions: NDArray[Any],
    trajectory_ids: Sequence[str],
    matched_pair_ids: Sequence[str],
    transition_indices: NDArray[Any],
    graph: DeclaredTelemetryGraph,
) -> float:
    """Score paired treatment-minus-control effects on downstream entities."""

    predicted = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(observed, dtype=np.float64)
    actions = np.asarray(future_actions, dtype=np.float64)
    trajectories = tuple(str(value) for value in trajectory_ids)
    pairs = tuple(str(value) for value in matched_pair_ids)
    transitions = np.asarray(transition_indices, dtype=np.int64)
    if (
        predicted.shape != target.shape
        or predicted.ndim != 4
        or actions.ndim != 4
        or predicted.shape[:2] != actions.shape[:2]
        or predicted.shape[2] != len(graph.entity_ids)
        or len(trajectories) != len(predicted)
        or len(pairs) != len(predicted)
        or transitions.shape != (len(predicted),)
    ):
        raise ValueError("downstream confirmation arrays do not align")
    index_by_key = {
        (trajectory_id, int(transition)): index
        for index, (trajectory_id, transition) in enumerate(
            zip(trajectories, transitions)
        )
    }
    action_entity_by_trajectory: Dict[str, str] = {}
    for index, trajectory_id in enumerate(trajectories):
        active = np.argwhere(actions[index, ..., 1] > 0.5)
        if len(active):
            action_entity_by_trajectory[trajectory_id] = (
                graph.entity_ids[int(active[0, 1])]
            )
    trajectories_by_pair: Dict[str, list[str]] = {}
    for trajectory_id, pair_id in zip(trajectories, pairs):
        values = trajectories_by_pair.setdefault(pair_id, [])
        if trajectory_id not in values:
            values.append(trajectory_id)
    squared_errors = []
    for pair_trajectories in trajectories_by_pair.values():
        treatments = [
            value
            for value in pair_trajectories
            if value in action_entity_by_trajectory
        ]
        controls = [
            value
            for value in pair_trajectories
            if value not in action_entity_by_trajectory
        ]
        if len(treatments) != 1 or len(controls) != 1:
            raise ValueError("confirmation pair arms do not align")
        treatment_id = treatments[0]
        control_id = controls[0]
        downstream = _downstream_positions(
            graph, action_entity_by_trajectory[treatment_id]
        )
        for treatment_index, trajectory_id in enumerate(trajectories):
            if trajectory_id != treatment_id:
                continue
            control_index = index_by_key.get(
                (control_id, int(transitions[treatment_index]))
            )
            if control_index is None:
                raise ValueError("confirmation pair transitions do not align")
            active = np.any(
                actions[treatment_index, ..., 1] > 0.5, axis=1
            )
            if not np.any(active) or not downstream:
                continue
            predicted_effect = (
                predicted[treatment_index] - predicted[control_index]
            )
            observed_effect = target[treatment_index] - target[control_index]
            squared_errors.append(
                np.square(
                    predicted_effect[active][:, downstream]
                    - observed_effect[active][:, downstream]
                ).reshape(-1)
            )
    if not squared_errors:
        raise ValueError("confirmation has no downstream effect observations")
    return float(np.mean(np.concatenate(squared_errors)))


def _loss_vector(values: NDArray[Any], size: int) -> NDArray[np.float64]:
    restored = np.asarray(values, dtype=np.float64)
    if (
        restored.shape != (size,)
        or not np.all(np.isfinite(restored))
        or np.any(restored < 0.0)
    ):
        raise ValueError("confirmation pair losses are invalid")
    return restored


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return float("inf")
    return float(numerator / denominator)


def _downstream_positions(
    graph: DeclaredTelemetryGraph, start_entity: str
) -> Tuple[int, ...]:
    adjacency: Dict[str, list[str]] = {
        entity_id: [] for entity_id in graph.entity_ids
    }
    for entity in graph.entities:
        if entity.kind == "edge":
            if entity.source is None or entity.target is None:
                raise ValueError("declared graph edge is incomplete")
            adjacency[entity.source].append(entity.entity_id)
            adjacency[entity.entity_id].append(entity.target)
    discovered = []
    frontier = list(adjacency[start_entity])
    while frontier:
        candidate = frontier.pop(0)
        if candidate in discovered or candidate == start_entity:
            continue
        discovered.append(candidate)
        frontier.extend(adjacency[candidate])
    return tuple(graph.entity_ids.index(value) for value in discovered)


def _mapping(
    payload: Mapping[str, Any], name: str
) -> Mapping[str, Any]:
    value = payload.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_git_commit(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
