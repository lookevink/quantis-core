#!/usr/bin/env python3
"""Independent stored-evidence assessor for the LeNEPA tracer."""

import argparse
import ast
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from quantis_core.action_conditioned_dynamics import (
    ActionConditionedWindows,
)
from quantis_core.edge_dynamics.action_conditioned_jepa import (
    sketched_isotropic_gaussian_regularization,
)
from quantis_core.edge_dynamics.complete_lejepa import (
    EntityPcaRepresentation,
    PairBlockedAnchorSchedule,
    ReducedRankActionProbe,
)
from quantis_core.edge_dynamics.data import PreparedAttributionQueries
from quantis_core.edge_dynamics.lenepa_jepa import (
    LenepaConfig,
    LenepaRepresentation,
    assess_lenepa_gates,
)
from quantis_core.graph_telemetry import DeclaredTelemetryGraph


NEURAL_NAMES = (
    "projected_lenepa",
    "unprojected_lenepa",
    "projected_sigreg_only",
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
        or metadata.get("kind") != "lenepa_assessment_evidence_v1"
    ):
        raise ValueError("unsupported LeNEPA evidence")
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
        role: _windows_from_evidence(
            role, metadata, arrays, graph
        )
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
    configs = {
        str(name): dict(value)
        for name, value in dict(metadata["configs"]).items()
    }
    mechanism = {
        name: {
            role: _diagnostic_metrics(
                arrays[f"diagnostic_input_tokens__{name}__{role}"],
                arrays[f"diagnostic_output_tokens__{name}__{role}"],
                arrays[f"diagnostic_predicted_tokens__{name}__{role}"],
                arrays[f"diagnostic_target_tokens__{name}__{role}"],
                configs[name],
            )
            for role in ("selection", "transfer_evaluation")
        }
        for name in NEURAL_NAMES
    }
    diagnostic_shift_consistency = all(
        _max_abs(
            arrays[f"diagnostic_predicted_tokens__{name}__{role}"],
            arrays[f"diagnostic_output_tokens__{name}__{role}"][
                :, :-1
            ],
        )
        <= 1e-6
        and _max_abs(
            arrays[f"diagnostic_target_tokens__{name}__{role}"],
            arrays[f"diagnostic_input_tokens__{name}__{role}"][
                :, 1:
            ],
        )
        <= 1e-6
        for name in NEURAL_NAMES
        for role in ("selection", "transfer_evaluation")
    )
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
    replay = _replay_retained_artifacts(
        root, windows["transfer_evaluation"], arrays
    )
    restoration_max_abs = replay["restoration_max_abs"]
    parameter_counts = _recompute_parameter_counts(root)
    selection_recomputes, safety_status_recomputes = (
        _selection_recomputes(metadata, arrays, windows, raw_scores)
    )
    prefix_max_abs = replay["prefix_max_abs"]
    bundle_path = (
        root / "models" / "projected_lenepa-inference.json.gz"
    )
    bundle_bytes = bundle_path.read_bytes()
    deployed_bundle_bytes = len(bundle_bytes)
    bundle_payload = replay["bundle_payload"]
    candidate_model = _read_json(
        root / "models" / "projected_lenepa.json"
    )
    candidate_probe = _read_json(
        root / "models" / "projected_lenepa-probe.json"
    )
    candidate = LenepaRepresentation.from_dict(candidate_model)
    expected_bundle = candidate.to_inference_dict()
    expected_bundle["probe"] = candidate_probe
    bundle_is_deployable = bool(
        bundle_payload == expected_bundle
        and replay["bundle_replay_max_abs"] <= 1e-6
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
        == LenepaConfig(objective=name).to_dict()
        for name in NEURAL_NAMES
    )
    mechanism_history_coverage = all(
        arrays[f"diagnostic_predicted_tokens__{name}__{role}"].shape[
            0
        ]
        == len(windows[role].histories)
        for name in NEURAL_NAMES
        for role in ("selection", "transfer_evaluation")
    )
    interpretable = _recompute_interpretable(
        metadata=metadata,
        frozen_controls=frozen_controls,
        latency_repetitions=latency["repetitions"],
        mechanism_history_coverage=mechanism_history_coverage,
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
        "prefix_invariance_recomputes": all(
            value <= 1e-6 for value in prefix_max_abs.values()
        ),
        "anchor_schedule_recomputes": _anchor_schedule_recomputes(
            root, windows["fit"], configs["projected_lenepa"]
        ),
        "selection_only_ridge_choice_recomputes": (
            selection_recomputes
        ),
        "selection_safety_status_recomputes": (
            safety_status_recomputes
        ),
        "bundle_size_recomputes": (
            bundle_is_deployable
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
        "mechanism_history_coverage_recomputes": (
            mechanism_history_coverage
        ),
        "diagnostic_shift_consistency_recomputes": (
            diagnostic_shift_consistency
        ),
    }
    assessment = dict(
        assess_lenepa_gates(
            forecast_scores=forecast_scores,
            raw_scores=raw_scores,
            mechanism=mechanism,
            attribution=attribution,
            action_sanity=action_sanity,
            restoration_max_abs=restoration_max_abs,
            protocol_checks=protocol_checks,
            parameter_counts=parameter_counts,
            transfer_pair_errors={
                name: _downstream_pair_errors(
                    arrays[
                        f"prediction__{name}__transfer_evaluation"
                    ],
                    windows["transfer_evaluation"],
                )
                for name in REPRESENTATION_NAMES
            },
            deployed_bundle_bytes=deployed_bundle_bytes,
            median_latency_ms=latency["median_ms"],
        )
    )
    assessment["protocol_checks"] = protocol_checks
    assessment["forecast_scores"] = forecast_scores
    assessment["raw_scores"] = raw_scores
    assessment["state_probes"] = state_probes
    assessment["mechanism"] = mechanism
    assessment["attribution"] = attribution
    assessment["action_sanity"] = action_sanity
    assessment["restoration_max_abs"] = restoration_max_abs
    assessment["prefix_max_abs"] = prefix_max_abs
    assessment["parameter_counts"] = parameter_counts
    assessment["deployed_bundle_bytes"] = deployed_bundle_bytes
    assessment["latency"] = latency
    assessment["eligible_for_advance"] = interpretable
    if not interpretable:
        assessment["provisional_decision"] = assessment["decision"]
        assessment["decision"] = "non_interpretable_lenepa_smoke"
        assessment["passed"] = False
    return assessment


def verify_stored_assessment(directory: Path) -> Mapping[str, Any]:
    """Verify manifest identity and exact stored reassessment."""

    root = Path(directory)
    manifest = _read_json(root / "artifact-manifest.json")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "lenepa_artifact_manifest_v1"
    ):
        raise ValueError("unsupported LeNEPA manifest")
    for filename, expected in dict(manifest["sha256"]).items():
        actual = hashlib.sha256((root / filename).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(
                f"LeNEPA artifact identity mismatch: {filename}"
            )
    recorded = _read_json(root / "assessment.json")
    reassessed = assess_stored_bundle(root)
    if _canonical_json(recorded) != _canonical_json(reassessed):
        raise ValueError(
            "stored LeNEPA assessment differs from recomputation"
        )
    return reassessed


def _recompute_interpretable(
    *,
    metadata: Mapping[str, Any],
    frozen_controls: bool,
    latency_repetitions: int,
    mechanism_history_coverage: bool,
) -> bool:
    return bool(
        metadata.get("interpretable") is True
        and metadata.get("source_corpus_sha256")
        == FROZEN_SOURCE_CORPUS_SHA256
        and metadata.get("source_artifact_manifest_sha256")
        == FROZEN_SOURCE_ARTIFACT_MANIFEST_SHA256
        and metadata.get("preprocessing_protocol")
        == FROZEN_PREPROCESSING_PROTOCOL
        and frozen_controls
        and latency_repetitions == 100
        and mechanism_history_coverage
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
    metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
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


def _forecast_scores(
    prediction: np.ndarray, windows: ActionConditionedWindows
) -> Mapping[str, float]:
    observed = np.asarray(windows.future_states, dtype=np.float64)
    squared = np.square(
        np.asarray(prediction, dtype=np.float64) - observed
    )
    row_mse = np.mean(squared, axis=(1, 2, 3))
    active = np.any(windows.future_actions[..., 1] > 0.5, axis=2)
    action_rows = np.asarray(
        [
            np.mean(squared[index][active[index]])
            if np.any(active[index])
            else np.nan
            for index in range(len(squared))
        ]
    )
    pair_errors = _downstream_pair_errors(prediction, windows)
    return {
        "overall_mse": _pair_balanced(
            row_mse, windows.matched_pair_ids
        ),
        "action_overlap_mse": _pair_balanced(
            action_rows, windows.matched_pair_ids
        ),
        "downstream_effect_mse": float(
            np.mean(tuple(pair_errors.values()))
        ),
    }


def _downstream_pair_errors(
    prediction: np.ndarray, windows: ActionConditionedWindows
) -> Mapping[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64)
    index = {
        (trajectory, int(transition)): position
        for position, (trajectory, transition) in enumerate(
            zip(windows.trajectory_ids, windows.transition_indices)
        )
    }
    trajectories: Dict[str, list[str]] = {}
    treatment_target: Dict[str, int] = {}
    for row, (pair, trajectory) in enumerate(
        zip(windows.matched_pair_ids, windows.trajectory_ids)
    ):
        if trajectory not in trajectories.setdefault(pair, []):
            trajectories[pair].append(trajectory)
        active = np.argwhere(
            windows.future_actions[row, ..., 1] > 0.5
        )
        if len(active):
            treatment_target[trajectory] = int(active[0, 1])
    rows: Dict[str, float] = {}
    for pair, pair_trajectories in trajectories.items():
        treatment = [
            value
            for value in pair_trajectories
            if value in treatment_target
        ]
        control = [
            value
            for value in pair_trajectories
            if value not in treatment_target
        ]
        if len(treatment) != 1 or len(control) != 1:
            continue
        downstream = _downstream_positions(
            windows, treatment_target[treatment[0]]
        )
        errors = []
        for row, trajectory in enumerate(windows.trajectory_ids):
            if trajectory != treatment[0]:
                continue
            active = np.any(
                windows.future_actions[row, ..., 1] > 0.5, axis=1
            )
            other = index.get(
                (control[0], int(windows.transition_indices[row]))
            )
            if other is None or not np.any(active) or not downstream:
                continue
            predicted_effect = prediction[row] - prediction[other]
            observed_effect = (
                windows.future_states[row]
                - windows.future_states[other]
            )
            errors.append(
                np.square(
                    predicted_effect[active][:, downstream]
                    - observed_effect[active][:, downstream]
                ).mean()
            )
        if errors:
            rows[pair] = float(np.mean(errors))
    if not rows:
        raise ValueError("downstream effect assessment has no matched rows")
    return rows


def _downstream_positions(
    windows: ActionConditionedWindows, start: int
) -> Tuple[int, ...]:
    graph = windows.graph
    adjacency = {name: [] for name in graph.entity_ids}
    for entity in graph.entities:
        if entity.kind == "edge":
            adjacency[entity.source].append(entity.entity_id)
            adjacency[entity.entity_id].append(entity.target)
    start_name = graph.entity_ids[start]
    discovered = []
    frontier = list(adjacency[start_name])
    while frontier:
        candidate = frontier.pop(0)
        if candidate in discovered or candidate == start_name:
            continue
        discovered.append(candidate)
        frontier.extend(adjacency[candidate])
    return tuple(
        graph.entity_ids.index(value) for value in discovered
    )


def _pair_balanced(
    values: np.ndarray, pair_ids: Sequence[str]
) -> float:
    rows = []
    pair_array = np.asarray(pair_ids)
    for pair in sorted(set(pair_ids)):
        local = values[pair_array == pair]
        local = local[np.isfinite(local)]
        if len(local):
            rows.append(float(np.mean(local)))
    return float(np.mean(rows))


def _state_probe(
    fit_tokens: np.ndarray,
    fit: ActionConditionedWindows,
    evaluation_tokens: np.ndarray,
    evaluation: ActionConditionedWindows,
    ownership: np.ndarray,
) -> Mapping[str, Any]:
    entities = {}
    normalized_errors = []
    for entity, name in enumerate(fit.entity_names):
        mask = ownership[entity] & (
            np.ptp(fit.histories[:, -1, entity], axis=0) > 1e-9
        )
        if not np.any(mask):
            entities[name] = {"nrmse": None, "feature_count": 0}
            continue
        x = fit_tokens[:, entity]
        x_center = x.mean(axis=0)
        x_scale = x.std(axis=0)
        x_scale[x_scale <= 1e-12] = 1.0
        design = np.column_stack(
            ((x - x_center) / x_scale, np.ones(len(x)))
        )
        penalty = np.eye(design.shape[1])
        penalty[-1, -1] = 0.0
        target = fit.histories[:, -1, entity][:, mask]
        coefficients = np.linalg.solve(
            design.T @ design + 1e-3 * penalty,
            design.T @ target,
        )
        evaluation_design = np.column_stack(
            (
                (evaluation_tokens[:, entity] - x_center) / x_scale,
                np.ones(len(evaluation_tokens)),
            )
        )
        scale = target.std(axis=0)
        scale[scale <= 1e-12] = 1.0
        normalized = np.square(
            (
                evaluation_design @ coefficients
                - evaluation.histories[:, -1, entity][:, mask]
            )
            / scale
        ).reshape(-1)
        normalized_errors.append(normalized)
        entities[name] = {
            "nrmse": float(np.sqrt(np.mean(normalized))),
            "feature_count": int(np.sum(mask)),
        }
    return {
        "aggregate_nrmse": float(
            np.sqrt(np.mean(np.concatenate(normalized_errors)))
        ),
        "entities": entities,
    }


def _attribution_scores_from_predictions(
    predictions: np.ndarray,
    queries: PreparedAttributionQueries,
    ownership: np.ndarray,
) -> Mapping[str, float]:
    treatment_hits = []
    control_hits = []
    for index in range(len(queries.query_ids)):
        error = np.mean(
            np.square(
                predictions[index]
                - queries.observed_future[index][None]
            )[..., ownership],
            axis=(1, 2),
        )
        winner = int(np.argmin(error))
        expected = queries.expected_action_kinds[index]
        if expected:
            treatment_hits.append(
                queries.candidate_action_kinds[winner] == expected
                and queries.candidate_target_entities[winner]
                == queries.expected_target_entities[index]
            )
        else:
            control_hits.append(
                queries.candidate_ids[winner] == "no_action"
            )
    return {
        "action_and_target_hit_at_1": float(
            np.mean(treatment_hits)
        ),
        "no_action_specificity": float(np.mean(control_hits)),
    }


def _action_sanity_from_predictions(
    predictions: Mapping[str, np.ndarray],
    windows: ActionConditionedWindows,
    ownership: np.ndarray,
) -> Mapping[str, float]:
    correct = predictions["correct"]
    absent = predictions["no_action"]
    shuffled = predictions["shuffled"]
    wins = []
    pair_array = np.asarray(windows.matched_pair_ids)
    for pair in sorted(set(windows.matched_pair_ids)):
        rows = np.flatnonzero(pair_array == pair)
        active = np.any(
            windows.future_actions[rows, ..., 1] > 0.5,
            axis=(1, 2),
        )
        rows = rows[active]
        if not len(rows):
            continue
        target = windows.future_states[rows]

        def score(values: np.ndarray) -> float:
            return float(
                np.mean(
                    np.square(values[rows] - target)[..., ownership]
                )
            )

        wins.append(
            score(correct) < score(absent)
            and score(correct) < score(shuffled)
        )
    return {
        "correct_action_beats_both_fraction": float(np.mean(wins))
    }


def _replay_retained_artifacts(
    root: Path,
    transfer: ActionConditionedWindows,
    arrays: Mapping[str, np.ndarray],
) -> Mapping[str, Any]:
    """Replay restoration, causality, and deployment from retained weights."""

    histories = transfer.histories[:8]
    controls = transfer.future_controls[:8]
    actions = transfer.future_actions[:8]
    models: Dict[str, Any] = {
        name: LenepaRepresentation.from_dict(
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
    replay_tokens = {
        name: models[name].encode(histories, transfer.graph).tokens
        for name in REPRESENTATION_NAMES
    }
    restoration_max_abs: Dict[str, float] = {}
    for name in REPRESENTATION_NAMES:
        replay_prediction = probes[name].predict(
            replay_tokens[name], controls, actions
        )
        restoration_max_abs[name] = max(
            _max_abs(
                replay_tokens[name],
                arrays[f"restoration_original__{name}"],
            ),
            _max_abs(
                replay_tokens[name],
                arrays[f"restoration_restored__{name}"],
            ),
            _max_abs(
                replay_prediction,
                arrays[f"restoration_probe_original__{name}"],
            ),
            _max_abs(
                replay_prediction,
                arrays[f"restoration_probe_restored__{name}"],
            ),
        )

    altered = histories.copy()
    altered[:, 10:] += 10_000.0
    prefix_max_abs: Dict[str, float] = {}
    for name in NEURAL_NAMES:
        sequence = models[name].encode_sequence(
            histories, transfer.graph
        )
        altered_sequence = models[name].encode_sequence(
            altered, transfer.graph
        )
        diagnostic = models[name].diagnose_next_latent(
            histories, transfer.graph
        )
        restoration_max_abs[name] = max(
            restoration_max_abs[name],
            _max_abs(
                sequence,
                arrays[f"restoration_sequence_original__{name}"],
            ),
            _max_abs(
                sequence,
                arrays[f"restoration_sequence_restored__{name}"],
            ),
            *[
                max(
                    _max_abs(
                        getattr(diagnostic, field),
                        arrays[
                            "restoration_diagnostic_"
                            f"{field}_original__{name}"
                        ],
                    ),
                    _max_abs(
                        getattr(diagnostic, field),
                        arrays[
                            "restoration_diagnostic_"
                            f"{field}_restored__{name}"
                        ],
                    ),
                )
                for field in (
                    "input_tokens",
                    "output_tokens",
                    "predicted_tokens",
                    "target_tokens",
                )
            ],
        )
        prefix_max_abs[name] = max(
            _max_abs(sequence[:, :10], altered_sequence[:, :10]),
            _max_abs(
                sequence[:, :10],
                arrays[f"prefix_original__{name}"],
            ),
            _max_abs(
                altered_sequence[:, :10],
                arrays[f"prefix_altered__{name}"],
            ),
        )

    bundle_path = (
        root / "models" / "projected_lenepa-inference.json.gz"
    )
    bundle_payload = dict(
        json.loads(gzip.decompress(bundle_path.read_bytes()).decode())
    )
    deployed_model = LenepaRepresentation.from_inference_dict(
        bundle_payload
    )
    deployed_probe = ReducedRankActionProbe.from_dict(
        dict(bundle_payload["probe"])
    )
    deployed_tokens = deployed_model.encode(
        histories, transfer.graph
    ).tokens
    deployed_prediction = deployed_probe.predict(
        deployed_tokens, controls, actions
    )
    full_prediction = probes["projected_lenepa"].predict(
        replay_tokens["projected_lenepa"], controls, actions
    )
    bundle_replay_max_abs = max(
        _max_abs(
            deployed_tokens, replay_tokens["projected_lenepa"]
        ),
        _max_abs(deployed_prediction, full_prediction),
    )
    return {
        "restoration_max_abs": restoration_max_abs,
        "prefix_max_abs": prefix_max_abs,
        "bundle_payload": bundle_payload,
        "bundle_replay_max_abs": bundle_replay_max_abs,
    }


def _diagnostic_metrics(
    input_tokens: np.ndarray,
    output_tokens: np.ndarray,
    predicted: np.ndarray,
    target: np.ndarray,
    config: Mapping[str, Any],
) -> Mapping[str, float]:
    predicted = output_tokens[:, :-1]
    target = input_tokens[:, 1:]
    prediction_norm = np.linalg.norm(predicted, axis=-1)
    target_norm = np.linalg.norm(target, axis=-1)
    cosine = np.sum(predicted * target, axis=-1) / np.maximum(
        prediction_norm * target_norm, 1e-8
    )
    hits = []
    normalized_prediction = predicted / np.maximum(
        prediction_norm[..., None], 1e-8
    )
    normalized_target = target / np.maximum(
        target_norm[..., None], 1e-8
    )
    expected = np.arange(len(predicted))
    for time_position in range(predicted.shape[1]):
        similarity = (
            normalized_prediction[:, time_position]
            @ normalized_target[:, time_position].T
        )
        hits.append(similarity.argmax(axis=1) == expected)
    import torch

    input_generator = torch.Generator(device="cpu").manual_seed(
        int(config["sigreg_seed"]) + 90_000
    )
    output_generator = torch.Generator(device="cpu").manual_seed(
        int(config["sigreg_seed"]) + 90_000
    )
    input_sigreg = sketched_isotropic_gaussian_regularization(
        torch.as_tensor(input_tokens, dtype=torch.float32),
        generator=input_generator,
        sketch_dimension=int(config["sketch_dimension"]),
        knot_count=int(config["knot_count"]),
    )
    output_sigreg = sketched_isotropic_gaussian_regularization(
        torch.as_tensor(output_tokens, dtype=torch.float32),
        generator=output_generator,
        sketch_dimension=int(config["sketch_dimension"]),
        knot_count=int(config["knot_count"]),
    )
    return {
        "cosine_error": float(np.mean(1.0 - cosine)),
        "retrieval_hit_at_1": float(np.mean(np.stack(hits, axis=1))),
        "input_sigreg": float(input_sigreg),
        "output_sigreg": float(output_sigreg),
        "prediction_effective_rank": _effective_rank(predicted),
        "target_effective_rank": _effective_rank(target),
    }


def _effective_rank(values: np.ndarray) -> float:
    matrix = np.asarray(values, dtype=np.float64).reshape(
        -1, values.shape[-1]
    )
    centered = matrix - np.mean(matrix, axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = singular**2
    if not np.any(energy > 0.0):
        return 0.0
    probability = energy / np.sum(energy)
    probability = probability[probability > 0.0]
    return float(np.exp(-np.sum(probability * np.log(probability))))


def _recompute_parameter_counts(
    root: Path,
) -> Dict[str, Dict[str, int]]:
    result = {}
    for name in NEURAL_NAMES:
        payload = _read_json(root / "models" / f"{name}.json")
        network = _state_count(
            dict(payload["network_state"]),
            exclude={"kind_ids", "degree_ids"},
        )
        projector = _state_count(
            dict(payload["projector_state"]),
            exclude={
                key
                for key in dict(payload["projector_state"])
                if key.endswith(
                    ("running_mean", "running_var", "num_batches_tracked")
                )
            },
        )
        result[name] = {
            "inference": network,
            "training": network + projector,
        }
    return result


def _state_count(
    state: Mapping[str, Any], *, exclude: set[str]
) -> int:
    return int(
        sum(
            np.asarray(value).size
            for name, value in state.items()
            if name not in exclude
        )
    )


def _anchor_schedule_recomputes(
    root: Path,
    fit: ActionConditionedWindows,
    config: Mapping[str, Any],
) -> bool:
    with np.load(
        root / "anchor-schedule.npz", allow_pickle=False
    ) as stored:
        indices = stored["indices"]
        arm_ids = stored["arm_ids"]
        transitions = stored["transition_indices"]
        pair_ids = tuple(str(value) for value in stored["pair_ids"])
    schedule = PairBlockedAnchorSchedule(
        fit, seed=int(config["anchor_seed"])
    )
    if pair_ids != schedule.pair_ids:
        return False
    if indices.shape[0] != int(config["steps"]):
        return False
    for step in range(len(indices)):
        batch = schedule.batch(step)
        if (
            not np.array_equal(indices[step], batch.indices)
            or not np.array_equal(arm_ids[step], batch.arm_ids)
            or not np.array_equal(
                transitions[step], batch.transition_indices
            )
        ):
            return False
    return True


def _selection_recomputes(
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    windows: Mapping[str, ActionConditionedWindows],
    raw_scores: Mapping[str, Mapping[str, float]],
) -> tuple[bool, bool]:
    selected = dict(metadata["selected_ridges"])
    stored_failed = {
        str(name): bool(value)
        for name, value in dict(
            metadata["selection_safety_failed"]
        ).items()
    }
    ridges = [float(value) for value in metadata["ridge_values"]]
    recomputed_failed = {}
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
        if float(selected[name]) != float(chosen["ridge"]):
            return False, stored_failed == recomputed_failed
    return True, stored_failed == recomputed_failed


def _role_identifiers_are_disjoint(
    metadata: Mapping[str, Any],
) -> bool:
    expected = {
        "fit": 40,
        "selection": 10,
        "calibration": 10,
        "iid_evaluation": 20,
        "transfer_evaluation": 10,
    }
    roles = dict(metadata["roles"])
    pairs = {
        role: set(str(value) for value in dict(roles[role])["pair_ids"])
        for role in expected
    }
    trajectories = {
        role: set(
            str(value)
            for value in dict(roles[role])["trajectory_ids"]
        )
        for role in expected
    }
    return bool(
        all(len(pairs[role]) == count for role, count in expected.items())
        and all(
            pairs[left].isdisjoint(pairs[right])
            and trajectories[left].isdisjoint(trajectories[right])
            for index, left in enumerate(expected)
            for right in tuple(expected)[index + 1 :]
        )
    )


def _public_inference_is_causal(root: Path) -> bool:
    path = (
        root
        / "reproduction-source"
        / "src"
        / "quantis_core"
        / "edge_dynamics"
        / "lenepa_jepa.py"
    )
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if (
            isinstance(node, ast.ClassDef)
            and node.name == "LenepaRepresentation"
        ):
            methods = {
                child.name: child
                for child in node.body
                if isinstance(
                    child, (ast.FunctionDef, ast.AsyncFunctionDef)
                )
            }
            return bool(
                _method_arguments(methods.get("encode"))
                == ("self", "histories", "graph")
                and _method_arguments(methods.get("encode_sequence"))
                == ("self", "histories", "graph")
                and _method_arguments(
                    methods.get("diagnose_next_latent")
                )
                == ("self", "histories", "graph")
            )
    return False


def _method_arguments(node: Any) -> tuple[str, ...]:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return ()
    if node.args.vararg is not None or node.args.kwarg is not None:
        return ()
    return tuple(
        argument.arg
        for argument in (
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        )
    )


def _max_abs(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.max(
            np.abs(
                np.asarray(left, dtype=np.float64)
                - np.asarray(right, dtype=np.float64)
            )
        )
    )


def _read_json(path: Path) -> Dict[str, Any]:
    return dict(json.loads(path.read_text()))


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def main(arguments: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parsed = parser.parse_args(arguments)
    print(
        json.dumps(
            verify_stored_assessment(parsed.directory),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
