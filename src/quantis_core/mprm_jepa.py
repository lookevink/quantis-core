"""Frozen evidence contract for the mean-preserving residual mixture tracer."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import uuid
from typing import Any, Dict, Mapping, Sequence, Tuple, cast

import numpy as np
from numpy.typing import NDArray

from .action_conditioned_dynamics import (
    ACTION_KINDS,
    ActionConditionedCaseManifest,
    InterventionAction,
)
from .richer_regime_retry import (
    WORKLOAD_FAMILIES,
    materialize_workload_schedule,
)


_REQUIRED_BINDINGS = (
    "candidate_protocol_sha256",
    "model_freeze_manifest_sha256",
    "action_protocol_sha256",
    "observation_schema_sha256",
    "application_build_context_sha256",
    "application_image_digest",
    "collector_image_digest",
    "attempt_id",
)
_PAIR_GATES = (
    "schedule_alignment",
    "raw_effect_passed",
    "recovery_passed",
    "count_resolution_passed",
    "drain_eligible",
    "restart_probe_live",
    "mechanistic_recovery_passed",
)


@dataclass(frozen=True)
class MprmJepaProtocol:
    """Strict public view of the frozen MPRM-JEPA contract."""

    payload: Mapping[str, Any]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MprmJepaProtocol":
        """Restore the one exact v1 protocol shape."""

        required = {
            "schema_version",
            "kind",
            "status",
            "generator_namespace",
            "generator_seed",
            "evidence_boundary",
            "fit_source",
            "recipe",
            "controls",
            "selection_design",
            "trajectory",
            "workload",
            "selection_gates",
            "edge_envelope",
            "supersedes",
        }
        if (
            set(payload) != required
            or payload.get("schema_version") != 1
            or payload.get("kind")
            != "mean_preserving_residual_mixture_jepa_protocol"
            or payload.get("status") != "frozen_pre_fit_contract"
            or payload.get("generator_namespace")
            != "quantis:mprm-jepa:selection:v1"
            or payload.get("generator_seed") != 26072931
        ):
            raise ValueError("MPRM-JEPA protocol envelope is invalid")
        protocol = cls(
            json.loads(json.dumps(payload, sort_keys=True))
        )
        protocol._validate()
        return protocol

    def _validate(self) -> None:
        design = _mapping(self.payload, "selection_design")
        recipe = _mapping(self.payload, "recipe")
        gates = _mapping(self.payload, "selection_gates")
        envelope = _mapping(self.payload, "edge_envelope")
        if (
            tuple(design.get("action_kinds", ())) != tuple(ACTION_KINDS)
            or tuple(design.get("worker_replica_values", ())) != (1, 2, 3)
            or tuple(design.get("workload_families", ()))
            != WORKLOAD_FAMILIES
            or design.get("replicates_per_cell") != 2
            or design.get("required_pair_count") != 90
            or design.get("required_capture_count") != 180
            or design.get("parallel_jobs") != 6
            or design.get("automatic_retry") is not False
            or design.get("pair_retry") is not False
            or design.get("overwrite") is not False
        ):
            raise ValueError("MPRM-JEPA selection design is invalid")
        expected_recipe = {
            "component_count": 4,
            "state_latent_width": 12,
            "context_width": 16,
            "predictor_width": 128,
            "epochs": 40,
            "batch_size": 256,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "ema_decay": 0.996,
            "latent_weight": 0.2,
            "target_reconstruction_weight": 0.1,
            "context_reconstruction_weight": 0.05,
            "component_variance_floor": 0.0001,
            "mixture_weight_floor": 1e-9,
            "seed": 307,
            "optimizer": "torch.optim.AdamW",
            "target": (
                "observable_residual_from_frozen_raw_rank_32_mean"
            ),
            "residual_centering": (
                "float64_weighted_mean_subtraction"
            ),
            "raw_anchor_rank": 32,
            "raw_variance": (
                "fit_role_residual_variance_only_no_selection_calibration"
            ),
        }
        if dict(recipe) != expected_recipe:
            raise ValueError("MPRM-JEPA recipe is not frozen")
        if (
            gates.get("paired_randomization_draws") != 99999
            or gates.get("paired_randomization_seed") != 26072932
            or gates.get("candidate_log_score_margin") != 0.01
            or gates.get("mean_identity_absolute_tolerance") != 1e-10
            or gates.get("raw_mean_mse_relative_tolerance") != 0.001
            or envelope.get("serialized_bytes_max") != 4194304
            or envelope.get("batch_one_p95_latency_ms_max") != 5.0
            or envelope.get("runtime_cpu") != "Apple M1 Max"
            or envelope.get("runtime_architecture") != "arm64"
            or envelope.get("torch_threads") != 1
        ):
            raise ValueError("MPRM-JEPA gates or edge envelope drifted")

    @property
    def generator_seed(self) -> int:
        return int(self.payload["generator_seed"])

    @property
    def generator_namespace(self) -> str:
        return str(self.payload["generator_namespace"])

    @property
    def action_kinds(self) -> Tuple[str, ...]:
        return tuple(
            str(value)
            for value in _mapping(
                self.payload, "selection_design"
            )["action_kinds"]
        )

    @property
    def worker_replica_values(self) -> Tuple[int, ...]:
        return tuple(
            int(value)
            for value in _mapping(
                self.payload, "selection_design"
            )["worker_replica_values"]
        )

    @property
    def workload_families(self) -> Tuple[str, ...]:
        return tuple(
            str(value)
            for value in _mapping(
                self.payload, "selection_design"
            )["workload_families"]
        )

    @property
    def workload(self) -> Mapping[str, Any]:
        return _mapping(self.payload, "workload")

    @property
    def trajectory(self) -> Mapping[str, Any]:
        return _mapping(self.payload, "trajectory")

    def to_dict(self) -> Dict[str, Any]:
        restored = json.loads(json.dumps(self.payload, sort_keys=True))
        if not isinstance(restored, dict):
            raise AssertionError("serialized MPRM-JEPA protocol changed type")
        return cast(Dict[str, Any], restored)


def canonicalize_mixture_weights(
    weight: NDArray[Any], *, floor: float
) -> NDArray[np.float64]:
    """Apply the shared float64 transport canonicalization exactly once."""

    values = np.asarray(weight, dtype=np.float64)
    if (
        values.ndim != 2
        or not np.all(np.isfinite(values))
        or floor <= 1e-12
        or not np.isfinite(floor)
        or np.any(values < 0.0)
    ):
        raise ValueError("mixture weight canonicalization inputs are invalid")
    if floor * values.shape[1] >= 1.0:
        raise ValueError("mixture weight floor leaves no probability mass")
    raw_total = np.sum(values, axis=1, keepdims=True)
    if np.any(raw_total <= 0.0):
        raise ValueError("mixture weights have no positive mass")
    normalized = floor + (
        1.0 - floor * values.shape[1]
    ) * values / raw_total
    normalized[:, -1] = 1.0 - np.sum(normalized[:, :-1], axis=1)
    if np.any(normalized < floor) or not np.array_equal(
        np.sum(normalized, axis=1), np.ones(len(normalized))
    ):
        raise ValueError("canonical mixture weights violate the floor")
    return np.asarray(normalized, dtype=np.float64)


def mean_preserving_component_means(
    anchor_mean: NDArray[Any],
    residual_mean: NDArray[Any],
    weight: NDArray[Any],
) -> NDArray[np.float64]:
    """Center residual hypotheses so their weighted mean is the anchor."""

    anchor = np.asarray(anchor_mean, dtype=np.float64)
    residual = np.asarray(residual_mean, dtype=np.float64)
    weights = np.asarray(weight, dtype=np.float64)
    if (
        anchor.ndim != 4
        or residual.ndim != 5
        or residual.shape[:1] + residual.shape[2:] != anchor.shape
        or weights.shape != residual.shape[:2]
        or not np.all(np.isfinite(anchor))
        or not np.all(np.isfinite(residual))
        or not np.all(np.isfinite(weights))
    ):
        raise ValueError("mean-preserving residual inputs are invalid")
    expanded = weights[:, :, None, None, None]
    centered = residual - np.sum(expanded * residual, axis=1)[:, None]
    component_mean = anchor[:, None] + centered
    recovered = np.sum(expanded * component_mean, axis=1)
    correction = anchor - recovered
    component_mean[:, -1] += correction / weights[:, -1, None, None, None]
    if np.max(
        np.abs(np.sum(expanded * component_mean, axis=1) - anchor)
    ) > 1e-10:
        raise ValueError("mean-preserving identity exceeds tolerance")
    return np.asarray(component_mean, dtype=np.float64)


def build_mprm_selection_plan(
    protocol: MprmJepaProtocol,
) -> Mapping[str, Any]:
    """Build the deterministic fresh 90-pair selection plan."""

    protocol._validate()
    pairs = []
    for action in protocol.action_kinds:
        for workers in protocol.worker_replica_values:
            for family in protocol.workload_families:
                for replicate in range(2):
                    key = f"{action}:{workers}:{family}:{replicate}"
                    pair_id = str(
                        uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            f"{protocol.generator_namespace}:{key}",
                        )
                    )
                    workload_seed = _derived_integer(
                        protocol.generator_seed, key + ":workload"
                    )
                    pairs.append(
                        {
                            "pair_id": pair_id,
                            "action_kind": action,
                            "worker_replicas": workers,
                            "workload_family": family,
                            "replicate": replicate,
                            "workload_seed": workload_seed,
                            "intervention_seed": _derived_integer(
                                protocol.generator_seed,
                                key + ":intervention",
                            ),
                            "request_schedule": list(
                                materialize_workload_schedule(
                                    cast(Any, protocol),
                                    workload_family=family,
                                    workload_seed=workload_seed,
                                    action_kind=action,
                                )
                            ),
                        }
                    )
    pairs.sort(
        key=lambda pair: _derived_integer(
            protocol.generator_seed, "order:" + str(pair["pair_id"])
        )
    )
    return {
        "schema_version": 1,
        "kind": "mprm_jepa_fresh_selection_plan",
        "protocol_sha256": _sha256(protocol.to_dict()),
        "pair_count": len(pairs),
        "capture_count": len(pairs) * 2,
        "pairs": pairs,
    }


def prepare_mprm_selection_campaign(
    protocol: MprmJepaProtocol,
    plan: Mapping[str, Any],
    *,
    action_library: Mapping[str, Any],
    bindings: Mapping[str, str],
) -> Mapping[str, Any]:
    """Prepare collector-compatible manifests with complete campaign binding."""

    if plan != build_mprm_selection_plan(protocol):
        raise ValueError("MPRM-JEPA selection plan drifted")
    _validate_bindings(bindings)
    if set(action_library) != set(protocol.action_kinds):
        raise ValueError("MPRM-JEPA action library differs")
    campaign_bindings = dict(sorted(bindings.items()))
    protocol_sha = _sha256(protocol.to_dict())
    plan_sha = _sha256(plan)
    executor_protocol = {
        "schema_version": 1,
        "kind": "mprm_jepa_selection_executor_protocol",
        "campaign_bindings": campaign_bindings,
        "campaign_protocol_sha256": protocol_sha,
        "campaign_plan_sha256": plan_sha,
        "collection": {
            "parallel_jobs": 6,
            "pair_count": 90,
            "expected_capture_count": 180,
            "overwrite": False,
            "automatic_retry": False,
        },
    }
    executor_protocol_sha = _sha256(executor_protocol)
    assignments = []
    manifests: Dict[str, Mapping[str, Any]] = {}
    for ordinal, pair in enumerate(plan["pairs"]):
        for role in ("treatment", "control"):
            case_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    (
                        f"{protocol.generator_namespace}:"
                        f"{pair['pair_id']}:{role}"
                    ),
                )
            )
            assignments.append(
                {
                    "pair_id": pair["pair_id"],
                    "case_id": case_id,
                    "role": role,
                    "lane": ordinal % 6 + 1,
                    "batch": ordinal // 6 + 1,
                    "order_in_pair": (
                        int(ordinal % 2 == 1)
                        if role == "treatment"
                        else int(ordinal % 2 == 0)
                    ),
                    "worker_replicas": pair["worker_replicas"],
                }
            )
            action = _intervention_action(
                protocol, pair, action_library
            )
            action_case = ActionConditionedCaseManifest(
                case_id=case_id,
                matched_pair_id=str(pair["pair_id"]),
                split="validation",
                point_count=108,
                logical_window_period_nano=250_000_000,
                topology_id=f"workers-{pair['worker_replicas']}",
                worker_replicas=int(pair["worker_replicas"]),
                workload_seed=int(pair["workload_seed"]),
                intervention_seed=int(pair["intervention_seed"]),
                actions=(action,) if role == "treatment" else (),
            )
            manifests[case_id] = {
                "schema_version": 1,
                "kind": "lab_action_capture_manifest",
                "action_case": action_case.to_dict(),
                "sample_period_seconds": 0.25,
                "request_schedule": list(pair["request_schedule"]),
                "api_request_queue_size": 128,
                "image_digests": {
                    "application": bindings[
                        "application_image_digest"
                    ],
                    "collector": bindings["collector_image_digest"],
                },
                "observation_schema_sha256": bindings[
                    "observation_schema_sha256"
                ],
                "protocol_sha256": executor_protocol_sha,
                "prepared_plan_sha256": "",
                "graph_observation_schema_sha256": bindings[
                    "observation_schema_sha256"
                ],
                "corpus_role": "development",
                "mprm_corpus_role": "selection",
                "workload_family": pair["workload_family"],
                "campaign_bindings": campaign_bindings,
                "campaign_protocol_sha256": protocol_sha,
                "campaign_plan_sha256": plan_sha,
            }
    executor_plan = {
        "schema_version": 1,
        "kind": "mprm_jepa_selection_execution_plan",
        "campaign_bindings": campaign_bindings,
        "protocol_sha256": executor_protocol_sha,
        "campaign_protocol_sha256": protocol_sha,
        "campaign_plan_sha256": plan_sha,
        "assignments": assignments,
    }
    executor_plan_sha = _sha256(executor_plan)
    manifests = {
        case_id: {
            **manifest,
            "prepared_plan_sha256": executor_plan_sha,
        }
        for case_id, manifest in manifests.items()
    }
    return {
        "executor_protocol": executor_protocol,
        "executor_plan": executor_plan,
        "executor_protocol_sha256": executor_protocol_sha,
        "executor_plan_sha256": executor_plan_sha,
        "campaign_bindings": campaign_bindings,
        "manifests": manifests,
        "manifest_sha256": {
            case_id: _sha256(manifest)
            for case_id, manifest in sorted(manifests.items())
        },
    }


def qualify_mprm_selection_campaign(
    protocol: MprmJepaProtocol,
    plan: Mapping[str, Any],
    manifests: Mapping[str, Mapping[str, Any]],
    captures: Mapping[str, Mapping[str, str]],
    attestation: Mapping[str, Any],
    pair_assessments: Mapping[str, Mapping[str, bool]],
    action_library: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Qualify complete stored evidence or reject before model scoring."""

    if plan != build_mprm_selection_plan(protocol):
        raise ValueError("MPRM-JEPA qualification plan drifted")
    if len(manifests) != 180 or set(captures) != set(manifests):
        raise ValueError("MPRM-JEPA capture coverage is incomplete")
    expected_cases: Dict[str, Tuple[Mapping[str, Any], str]] = {}
    for pair in plan["pairs"]:
        for role in ("treatment", "control"):
            case_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    (
                        f"{protocol.generator_namespace}:"
                        f"{pair['pair_id']}:{role}"
                    ),
                )
            )
            expected_cases[case_id] = (pair, role)
    if set(manifests) != set(expected_cases):
        raise ValueError("MPRM-JEPA manifest case identities drifted")
    manifest_bindings = {
        _sha256(dict(manifest["campaign_bindings"]))
        for manifest in manifests.values()
    }
    if len(manifest_bindings) != 1:
        raise ValueError("MPRM-JEPA manifest bindings drifted")
    campaign_bindings = dict(
        next(iter(manifests.values()))["campaign_bindings"]
    )
    _validate_bindings(campaign_bindings)
    expected_manifest_sha = {
        case_id: _sha256(manifest)
        for case_id, manifest in sorted(manifests.items())
    }
    protocol_hashes = {
        manifest.get("protocol_sha256")
        for manifest in manifests.values()
    }
    plan_hashes = {
        manifest.get("prepared_plan_sha256")
        for manifest in manifests.values()
    }
    for case_id, manifest in manifests.items():
        pair, role = expected_cases[case_id]
        action_case = _mapping(manifest, "action_case")
        expected_actions = (
            [_intervention_action(protocol, pair, action_library).to_dict()]
            if role == "treatment"
            else []
        )
        if (
            action_case.get("case_id") != case_id
            or action_case.get("matched_pair_id") != pair["pair_id"]
            or action_case.get("worker_replicas")
            != pair["worker_replicas"]
            or action_case.get("workload_seed")
            != pair["workload_seed"]
            or action_case.get("intervention_seed")
            != pair["intervention_seed"]
            or manifest.get("workload_family")
            != pair["workload_family"]
            or manifest.get("request_schedule")
            != pair["request_schedule"]
            or action_case.get("actions") != expected_actions
            or manifest.get("campaign_protocol_sha256")
            != _sha256(protocol.to_dict())
            or manifest.get("campaign_plan_sha256") != _sha256(plan)
            or captures[case_id].get("capture_manifest_sha256")
            != expected_manifest_sha[case_id]
        ):
            raise ValueError("MPRM-JEPA manifest content drifted")
    if len(protocol_hashes) != 1 or len(plan_hashes) != 1:
        raise ValueError("MPRM-JEPA executor identities drifted")
    if (
        attestation.get("kind")
        != "mprm_jepa_collection_attestation_v1"
        or attestation.get("campaign_bindings") != campaign_bindings
        or attestation.get("case_count") != 180
        or attestation.get("pair_count") != 90
        or attestation.get("manifest_sha256")
        != expected_manifest_sha
        or attestation.get("protocol_sha256")
        != next(iter(protocol_hashes))
        or attestation.get("plan_sha256")
        != next(iter(plan_hashes))
    ):
        raise ValueError("MPRM-JEPA attestation is incomplete or drifted")
    pair_ids = {str(pair["pair_id"]) for pair in plan["pairs"]}
    if set(pair_assessments) != pair_ids or any(
        assessment.get(gate) is not True
        for assessment in pair_assessments.values()
        for gate in _PAIR_GATES
    ):
        raise ValueError("MPRM-JEPA action qualification failed")
    required_capture_keys = {
        "capture_manifest_sha256",
        "runner_log_sha256",
        "metrics_sha256",
        "logs_sha256",
        "traces_sha256",
        "actions_sha256",
    }
    if any(
        set(evidence) != required_capture_keys
        or any(not _is_sha256(value) for value in evidence.values())
        for evidence in captures.values()
    ):
        raise ValueError("MPRM-JEPA capture content identity is invalid")
    source = {
        "protocol_sha256": _sha256(protocol.to_dict()),
        "plan_sha256": _sha256(plan),
        "manifest_sha256": expected_manifest_sha,
        "capture_sha256": {
            case_id: dict(sorted(evidence.items()))
            for case_id, evidence in sorted(captures.items())
        },
        "attestation_sha256": _sha256(attestation),
        "pair_assessment_sha256": _sha256(pair_assessments),
    }
    return {
        "schema_version": 1,
        "kind": "qualified_mprm_jepa_selection_corpus",
        "status": "qualified",
        "pair_count": 90,
        "capture_count": 180,
        "campaign_bindings": campaign_bindings,
        "qualified_corpus_sha256": _sha256(source),
        "source_content_manifest": source,
    }


def paired_randomization_p_value(
    candidate_minus_raw: NDArray[Any], *, seed: int, draws: int
) -> float:
    """Return the frozen one-sided Monte Carlo sign-flip p-value."""

    differences = np.asarray(candidate_minus_raw, dtype=np.float64)
    if (
        differences.ndim != 1
        or not len(differences)
        or not np.all(np.isfinite(differences))
        or seed != 26072932
        or draws != 99999
    ):
        raise ValueError("paired randomization inputs differ from contract")
    observed = float(np.mean(differences))
    generator = np.random.default_rng(seed)
    extreme = 0
    remaining = draws
    while remaining:
        count = min(remaining, 2048)
        signs = generator.integers(
            0, 2, size=(count, len(differences)), dtype=np.int8
        )
        signs = signs * 2 - 1
        statistics = np.mean(signs * differences[None], axis=1)
        extreme += int(np.sum(statistics <= observed))
        remaining -= count
    return (extreme + 1.0) / (draws + 1.0)


def _intervention_action(
    protocol: MprmJepaProtocol,
    pair: Mapping[str, Any],
    action_library: Mapping[str, Any],
) -> InterventionAction:
    config = _mapping(action_library, str(pair["action_kind"]))
    severities = config["severity_values"]
    replicate = int(pair["replicate"])
    workers = int(pair["worker_replicas"])
    intervention_seed = int(pair["intervention_seed"])
    start = 28 + intervention_seed % 12
    duration = (
        int(config["fixed_duration"])
        if "fixed_duration" in config
        else 8 + (intervention_seed // 17) % 13
    )
    return InterventionAction(
        action_id=str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                (
                    f"{protocol.generator_namespace}:"
                    f"{pair['pair_id']}:action"
                ),
            )
        ),
        action_kind=str(pair["action_kind"]),
        target_entity=str(config["target_entity"]),
        start_index=start,
        stop_index=start + duration,
        magnitude=float(
            severities[((workers - 1) + replicate) % len(severities)]
        ),
        magnitude_unit=str(config["magnitude_unit"]),
        effect_feature=str(config["effect_feature"]),
        effect_direction=str(config["effect_direction"]),
        minimum_effect=float(config["minimum_effect"]),
        recovery_tolerance=float(config["recovery_ratio_max"]),
    )


def _validate_bindings(bindings: Mapping[str, str]) -> None:
    if set(bindings) != set(_REQUIRED_BINDINGS):
        raise ValueError("MPRM-JEPA campaign bindings are incomplete")
    for name in _REQUIRED_BINDINGS:
        value = bindings[name]
        if name == "attempt_id":
            if (
                value != "mprm-jepa-selection-v1-attempt-001"
                and not value.startswith("mprm-jepa-selection-v")
            ):
                raise ValueError("MPRM-JEPA attempt identity is invalid")
        elif name.endswith("_digest"):
            if not value.startswith("sha256:") or not _is_sha256(
                value.split(":", 1)[1]
            ):
                raise ValueError("MPRM-JEPA image digest is invalid")
        elif not _is_sha256(value):
            raise ValueError("MPRM-JEPA content binding is invalid")


def _mapping(value: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    result = value.get(name)
    if not isinstance(result, dict):
        raise ValueError(f"MPRM-JEPA {name} must be an object")
    return result


def _derived_integer(seed: int, key: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{seed}:{key}".encode()).digest()[:8], "big"
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
