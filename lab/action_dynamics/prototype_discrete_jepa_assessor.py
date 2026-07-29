#!/usr/bin/env python3
"""Independent stored-evidence assessor for Discrete-JEPA telemetry."""

import argparse
import ast
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from lab.action_dynamics.prototype_lenepa_jepa_assessor import (
        _action_sanity_from_predictions,
        _attribution_scores_from_predictions,
        _downstream_pair_errors,
        _forecast_scores,
        _state_probe,
    )
except ModuleNotFoundError:
    from prototype_lenepa_jepa_assessor import (
        _action_sanity_from_predictions,
        _attribution_scores_from_predictions,
        _downstream_pair_errors,
        _forecast_scores,
        _state_probe,
    )
from quantis_core.action_conditioned_dynamics import (
    ActionConditionedWindows,
)
from quantis_core.edge_dynamics.complete_lejepa import (
    EntityPcaRepresentation,
    PairBlockedAnchorSchedule,
    ReducedRankActionProbe,
)
from quantis_core.edge_dynamics.data import PreparedAttributionQueries
from quantis_core.edge_dynamics.discrete_jepa import (
    DiscreteJepaConfig,
    DiscreteJepaRepresentation,
    DiscreteMaskSchedule,
    assess_discrete_jepa_gates,
)
from quantis_core.graph_telemetry import DeclaredTelemetryGraph


NEURAL_NAMES = (
    "discrete_complete",
    "continuous_complete",
    "discrete_p2p_only",
)
REPRESENTATION_NAMES = NEURAL_NAMES + ("matched_pca",)
EVALUATED_ROLES = (
    "selection",
    "iid_evaluation",
    "transfer_evaluation",
)
FROZEN_SOURCE_CORPUS_SHA256 = (
    "df03af282f48591216251f22f934b97e1555147df9ed6f6c6d1f7684f9644a26"
)
FROZEN_SOURCE_ARTIFACT_MANIFEST_SHA256 = (
    "d02afa33a5977cba69b255cd1a5f31470751b6681da1ecae31b11e86307e65b1"
)
FROZEN_PREPROCESSING_PROTOCOL = (
    "action_conditioned_jepa_topology_transfer_v1"
)


def assess_stored_bundle(directory: Path) -> Mapping[str, Any]:
    """Recompute every frozen gate from retained artifact evidence."""

    root = Path(directory)
    metadata = _read_json(root / "evidence-metadata.json")
    if (
        metadata.get("schema_version") != 1
        or metadata.get("kind")
        != "discrete_jepa_assessment_evidence_v1"
    ):
        raise ValueError("unsupported Discrete-JEPA evidence")
    with np.load(root / "evidence.npz", allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    finite = all(
        np.all(np.isfinite(value))
        for value in arrays.values()
        if value.dtype.kind in ("f", "i", "u")
    )
    graph = DeclaredTelemetryGraph.from_dict(dict(metadata["graph"]))
    ownership = np.asarray(metadata["ownership_mask"], dtype=np.bool_)
    windows = {
        role: _windows_from_evidence(role, metadata, arrays, graph)
        for role in ("fit",) + EVALUATED_ROLES
    }
    forecast_scores = {
        name: {
            role: _forecast_scores(
                arrays[f"prediction__{name}__{role}"],
                windows[role],
            )
            for role in EVALUATED_ROLES
        }
        for name in REPRESENTATION_NAMES
    }
    raw_scores = {
        role: _forecast_scores(
            arrays[f"raw_prediction__{role}"], windows[role]
        )
        for role in EVALUATED_ROLES
    }
    state_probes = {
        name: {
            role: _state_probe(
                arrays[f"representation__{name}__fit"],
                windows["fit"],
                arrays[f"representation__{name}__{role}"],
                windows[role],
                ownership,
            )
            for role in EVALUATED_ROLES
        }
        for name in REPRESENTATION_NAMES
    }
    queries = _queries_from_evidence(metadata, arrays)
    attribution = {
        name: _attribution_scores_from_predictions(
            arrays[f"attribution_prediction__{name}"],
            queries,
            ownership,
        )
        for name in REPRESENTATION_NAMES
    }
    action_sanity = {
        name: _action_sanity_from_predictions(
            {
                variant: arrays[
                    f"action_sanity__{name}__{variant}"
                ]
                for variant in ("correct", "no_action", "shuffled")
            },
            windows["transfer_evaluation"],
            ownership,
        )
        for name in REPRESENTATION_NAMES
    }
    diagnostics = {
        name: {
            role: _diagnostic_metrics(name, role, arrays)
            for role in ("selection", "transfer_evaluation")
        }
        for name in NEURAL_NAMES
    }
    code_usage = {
        name: {
            role: _code_usage(
                arrays[f"indices__{name}__{role}"],
                int(dict(metadata["configs"])[name]["code_count"]),
            )
            for role in ("selection", "transfer_evaluation")
        }
        for name in ("discrete_complete", "discrete_p2p_only")
    }
    transition_accuracy = {
        name: _transition_metrics(
            fit_indices=arrays[f"indices__{name}__fit"],
            fit=windows["fit"],
            evaluation_indices={
                role: arrays[f"indices__{name}__{role}"]
                for role in ("selection", "transfer_evaluation")
            },
            evaluations={
                role: windows[role]
                for role in ("selection", "transfer_evaluation")
            },
            code_count=int(
                dict(metadata["configs"])[name]["code_count"]
            ),
        )
        for name in ("discrete_complete", "discrete_p2p_only")
    }
    varying_entities = np.any(
        (
            np.ptp(windows["fit"].histories, axis=(0, 1))
            > 1e-9
        )
        & ownership,
        axis=1,
    )
    mechanism_gates = _mechanism_gates(
        code_usage,
        diagnostics,
        transition_accuracy,
        varying_entities,
    )
    restoration_max_abs, bundle_replay = (
        _replay_retained_artifacts(
            root,
            {
                role: windows[role]
                for role in ("selection", "transfer_evaluation")
            },
            arrays,
        )
    )
    parameter_counts = _recompute_parameter_counts(root)
    selection_recomputes, safety_status_recomputes = (
        _selection_recomputes(metadata, arrays, windows, raw_scores)
    )
    configs = {
        str(name): dict(value)
        for name, value in dict(metadata["configs"]).items()
    }
    anchor_ok = _anchor_schedule_recomputes(
        root, windows["fit"], configs["discrete_complete"]
    )
    mask_ok = _mask_schedule_recomputes(
        root, configs["discrete_complete"]
    )
    bundle_path = (
        root / "models" / "discrete_complete-inference.json.gz"
    )
    bundle_payload = dict(
        json.loads(gzip.decompress(bundle_path.read_bytes()).decode())
    )
    candidate_payload = _read_json(
        root / "models" / "discrete_complete.json"
    )
    candidate_probe = _read_json(
        root / "models" / "discrete_complete-probe.json"
    )
    candidate = DiscreteJepaRepresentation.from_dict(
        candidate_payload
    )
    expected_bundle = candidate.to_inference_dict()
    expected_bundle["probe"] = candidate_probe
    deployed_bundle_bytes = len(bundle_path.read_bytes())
    bundle_ok = bool(
        bundle_payload == expected_bundle and bundle_replay <= 1e-6
    )
    latency_samples = np.asarray(
        arrays["latency_samples_ms"], dtype=np.float64
    )
    latency = {
        "median_ms": float(np.median(latency_samples)),
        "p95_ms": float(np.quantile(latency_samples, 0.95)),
        "repetitions": int(len(latency_samples)),
    }
    stored_latency = dict(metadata["latency"])
    frozen_controls = all(
        configs.get(name)
        == DiscreteJepaConfig(objective=name).to_dict()
        for name in NEURAL_NAMES
    )
    interpretable = bool(
        metadata.get("interpretable") is True
        and metadata.get("source_corpus_sha256")
        == FROZEN_SOURCE_CORPUS_SHA256
        and metadata.get("source_artifact_manifest_sha256")
        == FROZEN_SOURCE_ARTIFACT_MANIFEST_SHA256
        and metadata.get("preprocessing_protocol")
        == FROZEN_PREPROCESSING_PROTOCOL
        and frozen_controls
        and len(latency_samples) == 100
    )
    protocol_checks = {
        "evidence_arrays_are_finite": finite,
        "pair_and_trajectory_roles_are_disjoint": (
            _role_identifiers_are_disjoint(metadata)
        ),
        "capacity_recomputes": parameter_counts
        == {
            str(name): {
                str(key): int(value)
                for key, value in dict(raw).items()
            }
            for name, raw in dict(
                metadata["parameter_counts"]
            ).items()
        },
        "public_inference_is_causal": _public_inference_is_causal(root),
        "anchor_schedule_recomputes": anchor_ok,
        "mask_schedule_recomputes": mask_ok,
        "selection_only_ridge_choice_recomputes": (
            selection_recomputes
        ),
        "selection_safety_status_recomputes": (
            safety_status_recomputes
        ),
        "bundle_size_recomputes": (
            bundle_ok
            and deployed_bundle_bytes
            == int(metadata["deployed_bundle_bytes"])
        ),
        "latency_recomputes": bool(
            int(stored_latency["repetitions"])
            == latency["repetitions"]
            and latency["repetitions"] > 0
            and (not frozen_controls or latency["repetitions"] == 100)
            and np.isclose(
                float(stored_latency["median_ms"]),
                latency["median_ms"],
                rtol=0.0,
                atol=1e-12,
            )
            and np.isclose(
                float(stored_latency["p95_ms"]),
                latency["p95_ms"],
                rtol=0.0,
                atol=1e-12,
            )
        ),
    }
    transfer_pair_errors = {
        name: _downstream_pair_errors(
            arrays[
                f"prediction__{name}__transfer_evaluation"
            ],
            windows["transfer_evaluation"],
        )
        for name in REPRESENTATION_NAMES
    }
    assessment = dict(
        assess_discrete_jepa_gates(
            forecast_scores=forecast_scores,
            raw_scores=raw_scores,
            mechanism_gates=mechanism_gates,
            attribution=attribution,
            action_sanity=action_sanity,
            restoration_max_abs=restoration_max_abs,
            protocol_checks=protocol_checks,
            parameter_counts=parameter_counts,
            transfer_pair_errors=transfer_pair_errors,
            deployed_bundle_bytes=deployed_bundle_bytes,
            median_latency_ms=latency["median_ms"],
        )
    )
    assessment.update(
        {
            "protocol_checks": protocol_checks,
            "forecast_scores": forecast_scores,
            "raw_scores": raw_scores,
            "state_probes": state_probes,
            "diagnostics": diagnostics,
            "code_usage": code_usage,
            "transition_accuracy": transition_accuracy,
            "attribution": attribution,
            "action_sanity": action_sanity,
            "restoration_max_abs": restoration_max_abs,
            "parameter_counts": parameter_counts,
            "deployed_bundle_bytes": deployed_bundle_bytes,
            "latency": latency,
            "eligible_for_advance": interpretable,
        }
    )
    if not interpretable:
        assessment["provisional_decision"] = assessment["decision"]
        assessment["decision"] = "non_interpretable_discrete_jepa_smoke"
        assessment["passed"] = False
    return assessment


def verify_stored_assessment(directory: Path) -> Mapping[str, Any]:
    """Verify manifest identity and exact stored reassessment."""

    root = Path(directory)
    manifest = _read_json(root / "artifact-manifest.json")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind")
        != "discrete_jepa_artifact_manifest_v1"
    ):
        raise ValueError("unsupported Discrete-JEPA manifest")
    for filename, expected in dict(manifest["sha256"]).items():
        actual = hashlib.sha256((root / filename).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(
                f"Discrete-JEPA artifact identity mismatch: {filename}"
            )
    recorded = _read_json(root / "assessment.json")
    reassessed = assess_stored_bundle(root)
    if _canonical_json(recorded) != _canonical_json(reassessed):
        raise ValueError(
            "stored Discrete-JEPA assessment differs from recomputation"
        )
    return reassessed


def _diagnostic_metrics(
    name: str,
    role: str,
    arrays: Mapping[str, np.ndarray],
) -> Mapping[str, float]:
    mask = arrays[f"diagnostic_mask__{name}__{role}"].astype(bool)
    target_patch = arrays[
        f"diagnostic_target_patch__{name}__{role}"
    ]
    expanded = np.broadcast_to(mask[..., None], target_patch.shape)

    def masked(field: str) -> float:
        prediction = arrays[f"diagnostic_{field}__{name}__{role}"]
        return float(
            np.mean(np.square(prediction - target_patch)[expanded])
        )

    p2s = arrays[f"diagnostic_p2s_prediction__{name}__{role}"]
    target_semantic = arrays[
        f"diagnostic_target_semantic__{name}__{role}"
    ]
    return {
        "s2p": masked("s2p_prediction"),
        "p2s": float(np.mean(np.square(p2s - target_semantic))),
        "p2p": masked("p2p_prediction"),
    }


def _code_usage(
    indices: np.ndarray, code_count: int
) -> Mapping[str, Any]:
    values = np.asarray(indices, dtype=np.int64)
    if (
        values.ndim != 2
        or np.any(values < 0)
        or np.any(values >= code_count)
    ):
        raise ValueError("Discrete-JEPA code indices are invalid")
    count = np.bincount(values.reshape(-1), minlength=code_count)
    probability = count[count > 0] / np.sum(count)
    perplexity = float(
        np.exp(-np.sum(probability * np.log(probability)))
    )
    return {
        "active_codes": int(np.sum(count > 0)),
        "perplexity": perplexity,
        "per_entity_active_codes": [
            int(len(np.unique(values[:, entity])))
            for entity in range(values.shape[1])
        ],
        "counts": count.astype(int).tolist(),
    }


def _transition_metrics(
    *,
    fit_indices: np.ndarray,
    fit: ActionConditionedWindows,
    evaluation_indices: Mapping[str, np.ndarray],
    evaluations: Mapping[str, ActionConditionedWindows],
    code_count: int,
) -> Mapping[str, Mapping[str, Any]]:
    tables, fallback = _fit_transition_tables(
        fit_indices, fit, code_count
    )
    return {
        role: _transition_accuracy(
            values, evaluations[role], tables, fallback
        )
        for role, values in evaluation_indices.items()
    }


def _fit_transition_tables(
    indices: np.ndarray,
    windows: ActionConditionedWindows,
    code_count: int,
) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(indices, dtype=np.int64)
    entity_count = values.shape[1]
    table = np.zeros(
        (entity_count, code_count, code_count), dtype=np.int64
    )
    fallback_count = np.zeros(
        (entity_count, code_count), dtype=np.int64
    )
    for current, following in _next_rows(windows):
        for entity in range(entity_count):
            table[
                entity, values[current, entity], values[following, entity]
            ] += 1
            fallback_count[entity, values[following, entity]] += 1
    return table, fallback_count.argmax(axis=1)


def _transition_accuracy(
    indices: np.ndarray,
    windows: ActionConditionedWindows,
    table: np.ndarray,
    fallback: np.ndarray,
) -> Mapping[str, Any]:
    values = np.asarray(indices, dtype=np.int64)
    hits_by_entity = [
        [] for _ in range(values.shape[1])
    ]
    for current, following in _next_rows(windows):
        for entity in range(values.shape[1]):
            current_code = values[current, entity]
            counts = table[entity, current_code]
            prediction = (
                int(np.argmax(counts))
                if np.any(counts)
                else int(fallback[entity])
            )
            hits_by_entity[entity].append(
                prediction == values[following, entity]
            )
    if not hits_by_entity or not hits_by_entity[0]:
        raise ValueError("Discrete-JEPA transition role has no next rows")
    per_entity = [
        float(np.mean(hits)) for hits in hits_by_entity
    ]
    return {
        "overall": float(np.mean(per_entity)),
        "per_entity": per_entity,
    }


def _next_rows(
    windows: ActionConditionedWindows,
) -> Sequence[Tuple[int, int]]:
    index = {
        (trajectory, int(transition)): row
        for row, (trajectory, transition) in enumerate(
            zip(windows.trajectory_ids, windows.transition_indices)
        )
    }
    rows = []
    for row, (trajectory, transition) in enumerate(
        zip(windows.trajectory_ids, windows.transition_indices)
    ):
        following = index.get((trajectory, int(transition) + 1))
        if following is not None:
            rows.append((row, following))
    return rows


def _mechanism_gates(
    usage: Mapping[str, Mapping[str, Mapping[str, Any]]],
    diagnostics: Mapping[str, Mapping[str, Mapping[str, float]]],
    transitions: Mapping[
        str, Mapping[str, Mapping[str, Any]]
    ],
    varying_entities: np.ndarray,
) -> Mapping[str, bool]:
    roles = ("selection", "transfer_evaluation")
    candidate = "discrete_complete"
    control = "discrete_p2p_only"
    return {
        "noncollapsed_code_usage": all(
            float(usage[candidate][role]["perplexity"]) >= 8.0
            and min(
                int(value)
                for entity, value in enumerate(
                    usage[candidate][role][
                        "per_entity_active_codes"
                    ]
                )
                if varying_entities[entity]
            )
            >= 2
            for role in roles
        ),
        "complementary_heads_are_learned": all(
            diagnostics[candidate][role]["s2p"]
            <= 0.90 * diagnostics[control][role]["s2p"]
            and diagnostics[candidate][role]["p2s"]
            <= 0.90 * diagnostics[control][role]["p2s"]
            for role in roles
        ),
        "p2p_is_preserved": all(
            diagnostics[candidate][role]["p2p"]
            <= 1.05 * diagnostics[control][role]["p2p"]
            for role in roles
        ),
        "next_code_advantage": all(
            float(transitions[candidate][role]["overall"])
            >= float(transitions[control][role]["overall"]) + 0.05
            for role in roles
        ),
    }


def _replay_retained_artifacts(
    root: Path,
    windows: Mapping[str, ActionConditionedWindows],
    arrays: Mapping[str, np.ndarray],
) -> Tuple[Mapping[str, float], float]:
    models: Dict[str, Any] = {
        name: DiscreteJepaRepresentation.from_dict(
            _read_json(root / "models" / f"{name}.json")
        )
        for name in NEURAL_NAMES
    }
    models["matched_pca"] = EntityPcaRepresentation.from_dict(
        _read_json(root / "models" / "matched_pca.json")
    )
    probes = {
        name: ReducedRankActionProbe.from_dict(
            _read_json(root / "models" / f"{name}-probe.json")
        )
        for name in REPRESENTATION_NAMES
    }
    restoration = {}
    for name in REPRESENTATION_NAMES:
        errors = []
        for role, role_windows in windows.items():
            replay = _encode_chunks(
                models[name].encode,
                role_windows.histories,
                role_windows.graph,
            )
            prediction = probes[name].predict(
                replay,
                role_windows.future_controls,
                role_windows.future_actions,
            )
            errors.extend(
                (
                    _max_abs(
                        replay,
                        arrays[
                            f"restoration_original_tokens__{name}__{role}"
                        ],
                    ),
                    _max_abs(
                        replay,
                        arrays[
                            f"restoration_restored_tokens__{name}__{role}"
                        ],
                    ),
                    _max_abs(
                        prediction,
                        arrays[
                            f"restoration_original_probe__{name}__{role}"
                        ],
                    ),
                    _max_abs(
                        prediction,
                        arrays[
                            f"restoration_restored_probe__{name}__{role}"
                        ],
                    ),
                )
            )
            if name in (
                "discrete_complete",
                "discrete_p2p_only",
            ):
                replay_indices = _encode_indices(
                    models[name],
                    role_windows.histories,
                    role_windows.graph,
                )
                errors.extend(
                    (
                        _max_abs(
                            replay_indices,
                            arrays[
                                "restoration_original_indices__"
                                f"{name}__{role}"
                            ],
                        ),
                        _max_abs(
                            replay_indices,
                            arrays[
                                "restoration_restored_indices__"
                                f"{name}__{role}"
                            ],
                        ),
                    )
                )
        restoration[name] = max(errors)
    bundle_path = (
        root / "models" / "discrete_complete-inference.json.gz"
    )
    payload = dict(
        json.loads(gzip.decompress(bundle_path.read_bytes()).decode())
    )
    deployed_model = DiscreteJepaRepresentation.from_inference_dict(
        payload
    )
    deployed_probe = ReducedRankActionProbe.from_dict(
        dict(payload["probe"])
    )
    transfer = windows["transfer_evaluation"]
    deployed_tokens = _encode_chunks(
        deployed_model.encode,
        transfer.histories,
        transfer.graph,
    )
    deployed_prediction = deployed_probe.predict(
        deployed_tokens,
        transfer.future_controls,
        transfer.future_actions,
    )
    full_prediction = probes["discrete_complete"].predict(
        arrays[
            "restoration_original_tokens__"
            "discrete_complete__transfer_evaluation"
        ],
        transfer.future_controls,
        transfer.future_actions,
    )
    bundle_replay = max(
        _max_abs(
            deployed_tokens,
            arrays[
                "restoration_original_tokens__"
                "discrete_complete__transfer_evaluation"
            ],
        ),
        _max_abs(deployed_prediction, full_prediction),
    )
    return restoration, bundle_replay


def _encode_chunks(
    call: Any,
    histories: np.ndarray,
    graph: DeclaredTelemetryGraph,
) -> np.ndarray:
    return np.concatenate(
        [
            call(histories[start : start + 128], graph).tokens
            for start in range(0, len(histories), 128)
        ],
        axis=0,
    )


def _encode_indices(
    model: DiscreteJepaRepresentation,
    histories: np.ndarray,
    graph: DeclaredTelemetryGraph,
) -> np.ndarray:
    values = []
    for start in range(0, len(histories), 128):
        encoded = model.encode(
            histories[start : start + 128], graph
        )
        if encoded.indices is None:
            raise ValueError(
                "Discrete-JEPA hard replay has no indices"
            )
        values.append(encoded.indices)
    return np.concatenate(values, axis=0)


def _recompute_parameter_counts(
    root: Path,
) -> Mapping[str, Mapping[str, int]]:
    result = {}
    for name in NEURAL_NAMES:
        model = DiscreteJepaRepresentation.from_dict(
            _read_json(root / "models" / f"{name}.json")
        )
        result[name] = {
            "training": model.training_parameter_count,
            "inference": model.inference_parameter_count,
        }
    return result


def _anchor_schedule_recomputes(
    root: Path,
    fit: ActionConditionedWindows,
    config: Mapping[str, Any],
) -> bool:
    with np.load(
        root / "anchor-schedule.npz", allow_pickle=False
    ) as stored:
        indices = stored["indices"]
        arms = stored["arm_ids"]
        transitions = stored["transition_indices"]
    schedule = PairBlockedAnchorSchedule(
        fit, seed=int(config["seed"]) + 1
    )
    if len(indices) != int(config["steps"]):
        return False
    return all(
        np.array_equal(indices[step], schedule.batch(step).indices)
        and np.array_equal(arms[step], schedule.batch(step).arm_ids)
        and np.array_equal(
            transitions[step],
            schedule.batch(step).transition_indices,
        )
        for step in range(len(indices))
    )


def _mask_schedule_recomputes(
    root: Path, config: Mapping[str, Any]
) -> bool:
    with np.load(
        root / "mask-schedule.npz", allow_pickle=False
    ) as stored:
        masks = stored["masks"]
    schedule = DiscreteMaskSchedule(
        entity_count=int(config["semantic_token_count"]),
        patch_count=int(config["patch_count"]),
        seed=int(config["seed"]) + 2,
    )
    return len(masks) == int(config["steps"]) and all(
        np.array_equal(
            masks[step],
            schedule.batch(step=step, batch_size=masks.shape[1]),
        )
        for step in range(len(masks))
    )


def _selection_recomputes(
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    windows: Mapping[str, ActionConditionedWindows],
    raw_scores: Mapping[str, Mapping[str, float]],
) -> Tuple[bool, bool]:
    selected = dict(metadata["selected_ridges"])
    stored_failed = {
        str(name): bool(value)
        for name, value in dict(
            metadata["selection_safety_failed"]
        ).items()
    }
    ridges = [float(value) for value in metadata["ridge_values"]]
    recomputed_failed = {}
    recomputed_selected = {}
    for name in REPRESENTATION_NAMES:
        rows = []
        for position, ridge in enumerate(ridges):
            scores = _forecast_scores(
                arrays[f"ridge_prediction__{name}__{position}"],
                windows["selection"],
            )
            rows.append(
                {
                    "ridge": ridge,
                    "raw_safe": (
                        scores["overall_mse"]
                        <= 1.05
                        * raw_scores["selection"]["overall_mse"]
                        and scores["action_overlap_mse"]
                        <= 1.05
                        * raw_scores["selection"][
                            "action_overlap_mse"
                        ]
                    ),
                    **scores,
                }
            )
        eligible = [row for row in rows if row["raw_safe"]]
        recomputed_failed[name] = not bool(eligible)
        chosen = min(
            eligible or rows,
            key=lambda row: (
                row["downstream_effect_mse"],
                row["ridge"],
            ),
        )
        recomputed_selected[name] = float(chosen["ridge"])
    return (
        all(
            np.isclose(
                float(selected[name]),
                recomputed_selected[name],
                rtol=0.0,
                atol=0.0,
            )
            for name in REPRESENTATION_NAMES
        ),
        recomputed_failed == stored_failed,
    )


def _windows_from_evidence(
    role: str,
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    graph: DeclaredTelemetryGraph,
) -> ActionConditionedWindows:
    identity = dict(dict(metadata["roles"])[role])
    return ActionConditionedWindows(
        histories=arrays[f"histories__{role}"],
        future_states=arrays[f"target__{role}"],
        future_controls=arrays[f"controls__{role}"],
        future_actions=arrays[f"actions__{role}"],
        trajectory_ids=tuple(
            str(value) for value in identity["trajectory_ids"]
        ),
        matched_pair_ids=tuple(
            str(value) for value in identity["pair_ids"]
        ),
        transition_indices=np.asarray(
            identity["transition_indices"], dtype=np.int64
        ),
        entity_names=tuple(
            str(value) for value in metadata["entity_names"]
        ),
        state_feature_names=tuple(
            str(value) for value in metadata["state_feature_names"]
        ),
        control_feature_names=tuple(
            str(value) for value in metadata["control_feature_names"]
        ),
        action_feature_names=tuple(
            str(value) for value in metadata["action_feature_names"]
        ),
        graph=graph,
    )


def _queries_from_evidence(
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> PreparedAttributionQueries:
    raw = dict(metadata["queries"])
    return PreparedAttributionQueries(
        query_ids=tuple(str(value) for value in raw["query_ids"]),
        histories=arrays["query_histories"],
        future_controls=arrays["query_future_controls"],
        observed_future=arrays["query_observed_future"],
        candidate_actions=arrays["query_candidate_actions"],
        candidate_ids=tuple(
            str(value) for value in raw["candidate_ids"]
        ),
        candidate_action_kinds=tuple(
            str(value) for value in raw["candidate_action_kinds"]
        ),
        candidate_target_entities=tuple(
            str(value) for value in raw["candidate_target_entities"]
        ),
        expected_action_kinds=tuple(
            str(value) for value in raw["expected_action_kinds"]
        ),
        expected_target_entities=tuple(
            str(value) for value in raw["expected_target_entities"]
        ),
        expected_variant_ids=tuple(
            str(value) for value in raw["expected_variant_ids"]
        ),
    )


def _role_identifiers_are_disjoint(
    metadata: Mapping[str, Any],
) -> bool:
    expected_pair_counts = {
        "fit": 40,
        "selection": 10,
        "calibration": 10,
        "iid_evaluation": 20,
        "transfer_evaluation": 10,
    }
    roles = dict(metadata["roles"])
    role_names = tuple(expected_pair_counts)
    pair_sets = {
        role: set(
            str(value)
            for value in dict(roles[role])["pair_ids"]
        )
        for role in role_names
    }
    if any(
        len(pair_sets[role]) != expected_pair_counts[role]
        for role in role_names
    ):
        return False
    for left_index, left in enumerate(role_names):
        left_raw = dict(roles[left])
        left_trajectories = set(
            str(value) for value in left_raw["trajectory_ids"]
        )
        for right in role_names[left_index + 1 :]:
            right_raw = dict(roles[right])
            if pair_sets[left] & pair_sets[right] or left_trajectories & set(
                str(value) for value in right_raw["trajectory_ids"]
            ):
                return False
    return True


def _public_inference_is_causal(root: Path) -> bool:
    path = (
        root
        / "reproduction-source"
        / "src/quantis_core/edge_dynamics/discrete_jepa.py"
    )
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ClassDef)
            and node.name == "DiscreteJepaRepresentation"
        ):
            for item in node.body:
                if (
                    isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and item.name == "encode"
                ):
                    return [
                        argument.arg for argument in item.args.args
                    ] == ["self", "histories", "graph"]
    return False


def _max_abs(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape:
        return float("inf")
    return float(np.max(np.abs(left.astype(float) - right.astype(float))))


def _read_json(path: Path) -> Mapping[str, Any]:
    return dict(json.loads(Path(path).read_text()))


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def main(arguments: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args(arguments)
    print(
        json.dumps(
            verify_stored_assessment(args.artifact),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
