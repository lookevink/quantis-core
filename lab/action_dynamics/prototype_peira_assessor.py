#!/usr/bin/env python3
"""Independent stored-evidence assessor for the PEIRA telemetry tracer."""

import argparse
import ast
import gzip
import hashlib
import json
import subprocess
import sys
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
    CompleteLejepaRepresentation,
    EntityPcaRepresentation,
    PairBlockedAnchorSchedule,
    ReducedRankActionProbe,
    TelemetryViewSchedule,
    fit_owned_feature_mask,
)
from quantis_core.edge_dynamics.data import PreparedAttributionQueries
from quantis_core.edge_dynamics.peira import (
    PeiraConfig,
    PeiraRepresentation,
    PeiraSchedule,
    assess_peira_gates,
)
from quantis_core.graph_telemetry import DeclaredTelemetryGraph


PEIRA_NAMES = ("aligned_peira", "deranged_peira")
REPRESENTATION_NAMES = PEIRA_NAMES + (
    "complete_lejepa",
    "masked_autoencoder",
    "matched_pca",
)
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
FROZEN_PRIOR_MANIFEST_SHA256 = (
    "00639afaee81cd3844e8b60014ed87f886a8e7a7e0b20bc50834416da1deb265"
)
FROZEN_PRIOR_MODEL_SHA256 = {
    "complete_lejepa": (
        "eda9795582f2965ba1091b1dca710bc74ce2098bbc747ddfc0de3a324e39e412"
    ),
    "masked_autoencoder": (
        "4149452dfdf18c7abe651f5cb77788737c7ef9b4bfbbeb3cd14ef69e82f9bad4"
    ),
}
FROZEN_PREPROCESSING_PROTOCOL = (
    "action_conditioned_jepa_topology_transfer_v1"
)


def peira_diagnostics(
    views: np.ndarray,
    backbone_tokens: np.ndarray,
    ownership: np.ndarray,
    varying_entities: np.ndarray,
    *,
    regularization: float,
) -> Mapping[str, Any]:
    """Recompute PEIRA objective, spectra, alignment, and collapse metrics."""

    values = np.asarray(views, dtype=np.float64)
    tokens = np.asarray(backbone_tokens, dtype=np.float64)
    owned = np.asarray(ownership, dtype=np.bool_)
    varying = np.asarray(varying_entities, dtype=np.bool_)
    if (
        values.ndim != 3
        or values.shape[0] != 2
        or tokens.ndim != 3
        or len(tokens) != values.shape[1]
        or tokens.shape[1] != len(owned)
        or varying.shape != (tokens.shape[1],)
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(tokens))
    ):
        raise ValueError("PEIRA diagnostic arrays do not align")
    first, second = values
    count, width = first.shape
    signal = (first.T @ second + second.T @ first) / float(count)
    noise = (first.T @ first + second.T @ second) / float(count)
    regularized = noise + regularization * np.eye(width)
    inverse = np.linalg.solve(regularized, np.eye(width))
    predictor = signal @ inverse
    residual_first = first @ predictor.T - second
    residual_second = second @ predictor.T - first
    auxiliary = 0.5 * (
        np.mean(np.sum(first * (residual_first @ inverse.T), axis=1))
        + np.mean(
            np.sum(second * (residual_second @ inverse.T), axis=1)
        )
    ) + 0.5 * regularization * (
        np.mean(np.sum(first * first, axis=1))
        + np.mean(np.sum(second * second, axis=1))
    )
    objective = -0.5 * np.trace(predictor) + 0.5 * regularization * (
        np.mean(np.sum(first * first, axis=1))
        + np.mean(np.sum(second * second, axis=1))
    )
    eigenvalues, eigenvectors = np.linalg.eigh(signal)
    order = np.argsort(eigenvalues)[::-1][: min(8, width)]
    alignments = []
    for position in order:
        vector = eigenvectors[:, position]
        transformed = noise @ vector
        denominator = np.linalg.norm(vector) * np.linalg.norm(transformed)
        alignments.append(
            0.0
            if denominator <= 1e-15
            else float(vector @ transformed / denominator)
        )
    per_entity_variance = [
        float(np.mean(np.var(tokens[:, entity], axis=0)))
        for entity in range(tokens.shape[1])
    ]
    return {
        "auxiliary_value": float(auxiliary),
        "trace_objective": float(objective),
        "negative_trace_objective": float(-objective),
        "trace_predictor": float(np.trace(predictor)),
        "signal_spectrum": np.linalg.eigvalsh(signal)[::-1].tolist(),
        "noise_spectrum": np.linalg.eigvalsh(noise)[::-1].tolist(),
        "symmetric_predictor_spectrum": np.linalg.eigvalsh(
            0.5 * (predictor + predictor.T)
        )[::-1].tolist(),
        "eigenvector_alignment_top8": float(np.mean(alignments)),
        "projector_effective_rank": _effective_rank(
            values.reshape(-1, width)
        ),
        "backbone_effective_rank": _effective_rank(
            tokens.reshape(len(tokens), -1)
        ),
        "per_entity_variance": per_entity_variance,
        "varying_entity_variance_min": float(
            min(
                value
                for entity, value in enumerate(per_entity_variance)
                if varying[entity]
            )
        ),
        "signal_symmetry_error": float(
            np.max(np.abs(signal - signal.T))
        ),
        "noise_symmetry_error": float(
            np.max(np.abs(noise - noise.T))
        ),
        "solve_residual": float(
            np.max(np.abs(regularized @ inverse - np.eye(width)))
        ),
        "condition_number": float(np.linalg.cond(regularized)),
    }


def assess_stored_bundle(directory: Path) -> Mapping[str, Any]:
    """Recompute every frozen PEIRA gate from retained evidence."""

    root = Path(directory)
    metadata = _read_json(root / "evidence-metadata.json")
    if (
        metadata.get("schema_version") != 1
        or metadata.get("kind") != "peira_assessment_evidence_v1"
    ):
        raise ValueError("unsupported PEIRA evidence")
    graph = DeclaredTelemetryGraph.from_dict(dict(metadata["graph"]))
    with np.load(root / "evidence.npz", allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    finite = all(
        np.all(np.isfinite(value))
        for value in arrays.values()
        if value.dtype != np.bool_
    )
    windows = {
        role: _windows_from_evidence(role, metadata, arrays, graph)
        for role in ("fit",) + EVALUATED_ROLES
    }
    ownership = np.asarray(metadata["ownership_mask"], dtype=np.bool_)
    declared_varying_entities = np.asarray(
        metadata["varying_entity_mask"], dtype=np.bool_
    )
    varying_entities = np.any(
        (
            np.ptp(windows["fit"].histories, axis=(0, 1)) > 1e-9
        )
        & ownership,
        axis=1,
    )
    forecast_scores = {
        name: {
            role: _forecast_scores(
                arrays[f"prediction__{name}__{role}"], windows[role]
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
            role: peira_diagnostics(
                arrays[f"diagnostic_views__{name}__{role}"],
                arrays[f"representation__{name}__{role}"],
                ownership,
                varying_entities,
                regularization=float(
                    dict(metadata["configs"])[name]["regularization"]
                ),
            )
            for role in ("selection", "transfer_evaluation")
        }
        for name in PEIRA_NAMES
    }
    mechanism_gates = _mechanism_gates(diagnostics)
    restoration_max_abs, bundle_replay = _replay_models(
        root, windows, arrays
    )
    transfer_pair_errors = {
        name: _downstream_pair_errors(
            arrays[f"prediction__{name}__transfer_evaluation"],
            windows["transfer_evaluation"],
        )
        for name in REPRESENTATION_NAMES
    }
    parameter_counts = _recompute_parameter_counts(root)
    selection_ok, selection_safety_ok = _selection_recomputes(
        metadata, arrays, windows, raw_scores
    )
    schedule_ok = _schedules_recompute(
        root,
        windows["fit"],
        PeiraConfig.from_dict(
            dict(dict(metadata["configs"])["aligned_peira"])
        ),
        arrays,
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
    bundle = root / "models" / "aligned_peira-inference.json.gz"
    deployed_bundle_bytes = int(bundle.stat().st_size)
    copied_controls_match = _copied_prior_controls_recompute(
        root, metadata
    )
    configs_frozen = all(
        dict(dict(metadata["configs"])[name])
        == PeiraConfig(objective=name).to_dict()
        for name in PEIRA_NAMES
    )
    interpretable = bool(
        metadata.get("interpretable") is True
        and metadata.get("source_corpus_sha256")
        == FROZEN_SOURCE_CORPUS_SHA256
        and metadata.get("source_artifact_manifest_sha256")
        == FROZEN_SOURCE_ARTIFACT_MANIFEST_SHA256
        and metadata.get("prior_control_manifest_sha256")
        == FROZEN_PRIOR_MANIFEST_SHA256
        and metadata.get("preprocessing_protocol")
        == FROZEN_PREPROCESSING_PROTOCOL
        and configs_frozen
        and len(latency_samples) == 100
    )
    protocol_checks = {
        "evidence_arrays_are_finite": finite,
        "role_contract_recomputes": _role_contract_recomputes(metadata),
        "varying_entity_mask_recomputes": np.array_equal(
            declared_varying_entities, varying_entities
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
        "all_schedules_recompute": schedule_ok,
        "training_moments_recompute": (
            _training_moments_recompute(metadata, arrays)
        ),
        "final_operators_recompute": _final_operators_recompute(
            metadata, arrays
        ),
        "selection_only_ridge_choice_recomputes": selection_ok,
        "selection_safety_status_recomputes": selection_safety_ok,
        "bundle_size_recomputes": deployed_bundle_bytes
        == int(metadata["deployed_bundle_bytes"]),
        "latency_recomputes": bool(
            int(stored_latency["repetitions"]) == len(latency_samples)
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
        "copied_prior_controls_match": copied_controls_match,
        "copied_source_assessor_recomputes": (
            _copied_source_assessor_receipt_recomputes(root, metadata)
        ),
    }
    assessment = dict(
        assess_peira_gates(
            forecast_scores=forecast_scores,
            raw_scores=raw_scores,
            mechanism_gates=mechanism_gates,
            attribution=attribution,
            action_sanity=action_sanity,
            restoration_max_abs={
                **restoration_max_abs,
                "deployment_bundle": bundle_replay,
            },
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
        assessment["decision"] = "non_interpretable_peira_smoke"
        assessment["passed"] = False
    return assessment


def verify_stored_assessment(directory: Path) -> Mapping[str, Any]:
    """Verify manifest identity and exact stored PEIRA reassessment."""

    root = Path(directory)
    manifest = _read_json(root / "artifact-manifest.json")
    expected = {
        str(path.relative_to(root)): _file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.name != "artifact-manifest.json"
    }
    if dict(manifest["sha256"]) != expected:
        raise ValueError("PEIRA artifact manifest differs")
    assessment = dict(assess_stored_bundle(root))
    stored = _read_json(root / "assessment.json")
    result = _read_json(root / "result.json")
    if (
        _canonical_json(assessment) != _canonical_json(stored)
        or _canonical_json(assessment)
        != _canonical_json(dict(result["assessment"]))
    ):
        raise ValueError("PEIRA stored assessment differs")
    copied = _run_copied_assessor(root)
    if _canonical_json(copied) != _canonical_json(assessment):
        raise ValueError("copied PEIRA assessor differs")
    return assessment


def _copied_prior_controls_recompute(
    root: Path, metadata: Mapping[str, Any]
) -> bool:
    retained = root / "prior-control-manifest.json"
    if (
        not retained.is_file()
        or _file_sha256(retained) != FROZEN_PRIOR_MANIFEST_SHA256
    ):
        return False
    manifest = _read_json(retained)
    declared = dict(manifest.get("sha256", {}))
    metadata_hashes = dict(metadata.get("prior_model_sha256", {}))
    return all(
        declared.get(f"models/{name}.json")
        == FROZEN_PRIOR_MODEL_SHA256[name]
        and metadata_hashes.get(name) == FROZEN_PRIOR_MODEL_SHA256[name]
        and _file_sha256(root / "models" / f"{name}.json")
        == FROZEN_PRIOR_MODEL_SHA256[name]
        for name in FROZEN_PRIOR_MODEL_SHA256
    )


def _copied_source_assessor_receipt_recomputes(
    root: Path, metadata: Mapping[str, Any]
) -> bool:
    receipt_path = root / "isolated-assessor-receipt.json"
    if not receipt_path.is_file():
        return False
    receipt = _read_json(receipt_path)
    source_hashes = {
        str(name): str(value)
        for name, value in dict(
            metadata.get("source_sha256", {})
        ).items()
    }
    source_snapshot_sha256 = hashlib.sha256(
        _canonical_json(source_hashes).encode()
    ).hexdigest()
    copied_sources_match = bool(source_hashes) and all(
        _file_sha256(root / "reproduction-source" / name) == expected
        for name, expected in source_hashes.items()
    )
    assessor_name = (
        "lab/action_dynamics/prototype_peira_assessor.py"
    )
    preflight_hash = str(
        receipt.get("preflight_assessment_sha256", "")
    )
    return bool(
        receipt.get("schema_version") == 1
        and receipt.get("kind")
        == "peira_isolated_assessor_receipt_v1"
        and receipt.get("returncode") == 0
        and receipt.get("assessor_sha256")
        == source_hashes.get(assessor_name)
        and receipt.get("source_snapshot_sha256")
        == source_snapshot_sha256
        and len(preflight_hash) == 64
        and all(value in "0123456789abcdef" for value in preflight_hash)
        and copied_sources_match
    )


def _mechanism_gates(
    diagnostics: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> Mapping[str, bool]:
    roles = ("selection", "transfer_evaluation")
    candidate = "aligned_peira"
    control = "deranged_peira"
    return {
        "matrix_numerics": all(
            max(
                float(diagnostics[name][role]["signal_symmetry_error"]),
                float(diagnostics[name][role]["noise_symmetry_error"]),
            )
            <= 1e-8
            and float(diagnostics[name][role]["solve_residual"]) <= 1e-8
            and float(diagnostics[name][role]["condition_number"]) <= 1e6
            for name in PEIRA_NAMES
            for role in roles
        ),
        "noncollapsed": all(
            float(
                diagnostics[candidate][role][
                    "projector_effective_rank"
                ]
            )
            >= 8.0
            and float(
                diagnostics[candidate][role][
                    "varying_entity_variance_min"
                ]
            )
            > 0.0
            for role in roles
        ),
        "trace_objective_advantage": all(
            float(
                diagnostics[candidate][role][
                    "negative_trace_objective"
                ]
            )
            > 0.0
            and float(
                diagnostics[control][role][
                    "negative_trace_objective"
                ]
            )
            > 0.0
            and float(
                diagnostics[candidate][role][
                    "negative_trace_objective"
                ]
            )
            >= 1.10
            * float(
                diagnostics[control][role][
                    "negative_trace_objective"
                ]
            )
            for role in roles
        ),
        "eigenvector_alignment_advantage": all(
            float(
                diagnostics[candidate][role][
                    "eigenvector_alignment_top8"
                ]
            )
            >= float(
                diagnostics[control][role][
                    "eigenvector_alignment_top8"
                ]
            )
            + 0.05
            for role in roles
        ),
    }


def _replay_models(
    root: Path,
    windows: Mapping[str, ActionConditionedWindows],
    arrays: Mapping[str, np.ndarray],
) -> Tuple[Mapping[str, float], float]:
    models: Dict[str, Any] = {
        name: PeiraRepresentation.from_dict(
            _read_json(root / "models" / f"{name}.json")
        )
        for name in PEIRA_NAMES
    }
    models.update(
        {
            name: CompleteLejepaRepresentation.from_dict(
                _read_json(root / "models" / f"{name}.json")
            )
            for name in ("complete_lejepa", "masked_autoencoder")
        }
    )
    pca = EntityPcaRepresentation.from_dict(
        _read_json(root / "models" / "matched_pca.json")
    )
    maxima = {}
    for name in REPRESENTATION_NAMES:
        probe = ReducedRankActionProbe.from_dict(
            _read_json(root / "models" / f"{name}-probe.json")
        )
        values = []
        for role in ("selection", "transfer_evaluation"):
            role_windows = windows[role]
            call = pca.encode if name == "matched_pca" else models[name].encode
            replay = _encode_chunks(
                call, role_windows.histories, role_windows.graph
            )
            replay_probe = probe.predict(
                replay,
                role_windows.future_controls,
                role_windows.future_actions,
            )
            values.extend(
                (
                    _max_abs(
                        arrays[
                            f"restoration_original_tokens__{name}__{role}"
                        ],
                        arrays[
                            f"restoration_restored_tokens__{name}__{role}"
                        ],
                    ),
                    _max_abs(
                        replay,
                        arrays[
                            f"restoration_restored_tokens__{name}__{role}"
                        ],
                    ),
                    _max_abs(
                        arrays[
                            f"restoration_original_probe__{name}__{role}"
                        ],
                        arrays[
                            f"restoration_restored_probe__{name}__{role}"
                        ],
                    ),
                    _max_abs(
                        replay_probe,
                        arrays[
                            f"restoration_restored_probe__{name}__{role}"
                        ],
                    ),
                )
            )
        if name in PEIRA_NAMES:
            signal, noise = models[name].final_moments
            predictor, inverse = models[name].final_operators
            values.extend(
                (
                    _max_abs(
                        signal,
                        arrays[f"training__{name}__running_signal"][-1],
                    ),
                    _max_abs(
                        noise,
                        arrays[f"training__{name}__running_noise"][-1],
                    ),
                    _max_abs(
                        predictor, arrays[f"final_predictor__{name}"]
                    ),
                    _max_abs(
                        inverse, arrays[f"final_inverse__{name}"]
                    ),
                )
            )
            for role in ("selection", "transfer_evaluation"):
                role_windows = windows[role]
                diagnostic_replay = _diagnose_chunks(
                    models[name],
                    role_windows.histories,
                    role_windows.graph,
                    step=models[name].config.steps - 1,
                )
                values.append(
                    _max_abs(
                        diagnostic_replay,
                        arrays[f"diagnostic_views__{name}__{role}"],
                    )
                )
        maxima[name] = max(values)
    payload = dict(
        json.loads(
            gzip.decompress(
                (
                    root / "models" / "aligned_peira-inference.json.gz"
                ).read_bytes()
            ).decode()
        )
    )
    if (
        set(payload)
        != {"schema_version", "kind", "representation", "probe"}
        or payload.get("schema_version") != 1
        or payload.get("kind")
        != "peira_forecast_inference_bundle_v1"
    ):
        raise ValueError("unsupported PEIRA deployment bundle")
    deployed_model = PeiraRepresentation.from_inference_dict(
        dict(payload["representation"])
    )
    deployed_probe = ReducedRankActionProbe.from_dict(
        dict(payload["probe"])
    )
    transfer = windows["transfer_evaluation"]
    deployed_tokens = _encode_chunks(
        deployed_model.encode, transfer.histories, transfer.graph
    )
    deployed_prediction = deployed_probe.predict(
        deployed_tokens,
        transfer.future_controls,
        transfer.future_actions,
    )
    bundle_replay = max(
        _max_abs(
            deployed_tokens,
            arrays["representation__aligned_peira__transfer_evaluation"],
        ),
        _max_abs(
            deployed_prediction,
            arrays["prediction__aligned_peira__transfer_evaluation"],
        ),
    )
    return maxima, bundle_replay


def _recompute_parameter_counts(
    root: Path,
) -> Mapping[str, Mapping[str, int]]:
    return {
        name: {
            "training": model.training_parameter_count,
            "inference": model.inference_parameter_count,
        }
        for name in PEIRA_NAMES
        for model in (
            PeiraRepresentation.from_dict(
                _read_json(root / "models" / f"{name}.json")
            ),
        )
    }


def _schedules_recompute(
    root: Path,
    fit: ActionConditionedWindows,
    config: PeiraConfig,
    arrays: Mapping[str, np.ndarray],
) -> bool:
    with np.load(root / "schedule.npz", allow_pickle=False) as stored:
        values = {name: stored[name] for name in stored.files}
    anchors = PairBlockedAnchorSchedule(fit, seed=config.anchor_seed)
    ownership = fit_owned_feature_mask(fit)
    varying_entities = np.any(
        (np.ptp(fit.histories, axis=(0, 1)) > 1e-9) & ownership,
        axis=1,
    )
    views = TelemetryViewSchedule(
        graph=fit.graph,
        ownership_mask=ownership,
        varying_entity_mask=varying_entities,
        seed=config.view_seed,
    )
    schedule = PeiraSchedule(
        steps=config.steps,
        eta_initial=config.eta_initial,
        eta_final=config.eta_final,
        derangement_seed=config.derangement_seed,
    )
    batches = [anchors.batch(step) for step in range(config.steps)]
    view_batches = [
        views.batch(fit.histories[:1], step=step)
        for step in range(config.steps)
    ]
    return bool(
        np.array_equal(
            values["anchor_indices"],
            np.stack([batch.indices for batch in batches]),
        )
        and np.array_equal(
            values["anchor_arm_ids"],
            np.stack([batch.arm_ids for batch in batches]),
        )
        and np.array_equal(
            values["anchor_transitions"],
            np.stack([batch.transition_indices for batch in batches]),
        )
        and np.array_equal(
            values["view_visible"],
            np.stack(
                [batch.visible_tokens[:2, 0] for batch in view_batches]
            ),
        )
        and np.array_equal(
            values["view_present"],
            np.stack(
                [batch.present_tokens[:2, 0] for batch in view_batches]
            ),
        )
        and np.array_equal(
            values["derangements"],
            np.stack(
                [
                    schedule.derangement(step, len(anchors.pair_ids))
                    for step in range(config.steps)
                ]
            ),
        )
        and np.array_equal(
            values["eta"],
            np.asarray(
                [schedule.eta(step) for step in range(config.steps)]
            ),
        )
        and np.array_equal(
            values["learning_rate"],
            np.asarray(
                [
                    config.learning_rate_at(step)
                    for step in range(config.steps)
                ]
            ),
        )
        and np.array_equal(
            values["clip_enabled"],
            np.asarray(
                [
                    config.clip_enabled_at(step)
                    for step in range(config.steps)
                ]
            ),
        )
        and np.array_equal(
            arrays["training__aligned_peira__pairing_indices"],
            np.broadcast_to(
                np.arange(len(anchors.pair_ids), dtype=np.int64),
                values["derangements"].shape,
            ),
        )
        and all(
            np.array_equal(
                arrays[f"training__{name}__anchor_indices"],
                values["anchor_indices"],
            )
            and np.array_equal(
                arrays[f"training__{name}__anchor_arm_ids"],
                values["anchor_arm_ids"],
            )
            and np.array_equal(
                arrays[f"training__{name}__anchor_transitions"],
                values["anchor_transitions"],
            )
            and np.array_equal(
                arrays[f"training__{name}__view_visible"],
                values["view_visible"],
            )
            and np.array_equal(
                arrays[f"training__{name}__view_present"],
                values["view_present"],
            )
            and np.array_equal(
                arrays[f"training__{name}__eta"], values["eta"]
            )
            and np.array_equal(
                arrays[f"training__{name}__learning_rate"],
                values["learning_rate"],
            )
            and np.array_equal(
                arrays[f"training__{name}__clip_enabled"],
                values["clip_enabled"].astype(np.float64),
            )
            for name in PEIRA_NAMES
        )
        and np.array_equal(
            arrays["training__deranged_peira__pairing_indices"],
            values["derangements"],
        )
    )


def _final_operators_recompute(
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> bool:
    for name in PEIRA_NAMES:
        config = PeiraConfig.from_dict(
            dict(dict(metadata["configs"])[name])
        )
        signal = np.asarray(
            arrays[f"training__{name}__running_signal"][-1],
            dtype=np.float64,
        )
        noise = np.asarray(
            arrays[f"training__{name}__running_noise"][-1],
            dtype=np.float64,
        )
        inverse = np.linalg.solve(
            noise
            + config.regularization
            * np.eye(config.width, dtype=np.float64),
            np.eye(config.width, dtype=np.float64),
        )
        predictor = signal @ inverse
        if not (
            np.allclose(
                arrays[f"final_inverse__{name}"],
                inverse,
                rtol=1e-12,
                atol=1e-12,
            )
            and np.allclose(
                arrays[f"final_predictor__{name}"],
                predictor,
                rtol=1e-12,
                atol=1e-12,
            )
        ):
            return False
    return True


def _selection_recomputes(
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    windows: Mapping[str, ActionConditionedWindows],
    raw_scores: Mapping[str, Mapping[str, float]],
) -> Tuple[bool, bool]:
    selected = dict(metadata["selected_ridges"])
    failed = dict(metadata["selection_safety_failed"])
    ridges = [float(value) for value in metadata["ridge_values"]]
    chosen_ok = True
    failed_ok = True
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
                        <= 1.05 * raw_scores["selection"]["overall_mse"]
                        and scores["action_overlap_mse"]
                        <= 1.05
                        * raw_scores["selection"]["action_overlap_mse"]
                    ),
                    **scores,
                }
            )
        eligible = [row for row in rows if row["raw_safe"]]
        chosen = min(
            eligible or rows,
            key=lambda row: (
                row["downstream_effect_mse"],
                row["ridge"],
            ),
        )
        chosen_ok &= float(selected[name]) == float(chosen["ridge"])
        failed_ok &= bool(failed[name]) == (not bool(eligible))
    return bool(chosen_ok), bool(failed_ok)


def _training_moments_recompute(
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> bool:
    for name in PEIRA_NAMES:
        config = PeiraConfig.from_dict(
            dict(dict(metadata["configs"])[name])
        )
        previous_signal = np.zeros(
            (config.width, config.width), dtype=np.float64
        )
        previous_noise = np.zeros_like(previous_signal)
        for step in range(config.steps):
            eta = float(arrays[f"training__{name}__eta"][step])
            batch_signal = np.asarray(
                arrays[f"training__{name}__batch_signal"][step],
                dtype=np.float64,
            )
            batch_noise = np.asarray(
                arrays[f"training__{name}__batch_noise"][step],
                dtype=np.float64,
            )
            signal = (1.0 - eta) * previous_signal + eta * batch_signal
            noise = (1.0 - eta) * previous_noise + eta * batch_noise
            if (
                not np.allclose(
                    signal,
                    arrays[f"training__{name}__running_signal"][step],
                    rtol=0.0,
                    atol=1e-12,
                )
                or not np.allclose(
                    noise,
                    arrays[f"training__{name}__running_noise"][step],
                    rtol=0.0,
                    atol=1e-12,
                )
            ):
                return False
            regularized = noise + config.regularization * np.eye(
                config.width
            )
            inverse = np.linalg.solve(
                regularized, np.eye(config.width)
            )
            predictor = signal @ inverse
            auxiliary = 0.5 * np.trace(
                inverse @ predictor @ batch_noise
                - inverse @ batch_signal
            )
            loss = auxiliary + 0.5 * config.regularization * np.trace(
                batch_noise
            )
            objective = -0.5 * np.trace(
                predictor
            ) + 0.5 * config.regularization * np.trace(batch_noise)
            expected = {
                "auxiliary_value": auxiliary,
                "loss": loss,
                "trace_objective": objective,
                "trace_predictor": np.trace(predictor),
                "symmetry_error": max(
                    np.max(np.abs(signal - signal.T)),
                    np.max(np.abs(noise - noise.T)),
                ),
                "solve_residual": np.max(
                    np.abs(
                        regularized @ inverse - np.eye(config.width)
                    )
                ),
                "condition_number": np.linalg.cond(regularized),
            }
            if any(
                not np.isclose(
                    float(arrays[f"training__{name}__{field}"][step]),
                    float(value),
                    rtol=1e-5,
                    atol=1e-7,
                )
                for field, value in expected.items()
            ):
                return False
            previous_signal = signal
            previous_noise = noise
    return True


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
            str(value) for value in identity["row_trajectory_ids"]
        ),
        matched_pair_ids=tuple(
            str(value) for value in identity["matched_pair_ids"]
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
        candidate_actions=arrays["query_candidate_actions"],
        observed_future=arrays["query_observed_future"],
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


def _role_contract_recomputes(metadata: Mapping[str, Any]) -> bool:
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
        / "reproduction-source/src/quantis_core/edge_dynamics/peira.py"
    )
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ClassDef)
            and node.name == "PeiraRepresentation"
        ):
            for item in node.body:
                if (
                    isinstance(item, ast.FunctionDef)
                    and item.name == "encode"
                ):
                    return [
                        argument.arg for argument in item.args.args
                    ] == ["self", "histories", "graph"]
    return False


def _encode_chunks(
    call: Any, histories: np.ndarray, graph: Any
) -> np.ndarray:
    return np.concatenate(
        [
            call(histories[start : start + 128], graph).tokens
            for start in range(0, len(histories), 128)
        ],
        axis=0,
    )


def _diagnose_chunks(
    model: PeiraRepresentation,
    histories: np.ndarray,
    graph: Any,
    *,
    step: int,
) -> np.ndarray:
    return np.concatenate(
        [
            model.diagnose_views(
                histories[start : start + 128],
                graph,
                step=step,
            )
            for start in range(0, len(histories), 128)
        ],
        axis=1,
    )


def _effective_rank(values: np.ndarray) -> float:
    singular = np.linalg.svd(values, compute_uv=False)
    total = float(np.sum(singular))
    if total <= 1e-15:
        return 0.0
    probabilities = singular[singular > 0.0] / total
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def _max_abs(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape:
        return float("inf")
    return float(
        np.max(np.abs(left.astype(float) - right.astype(float)))
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_json(path: Path) -> Mapping[str, Any]:
    return dict(json.loads(Path(path).read_text()))


def _run_copied_assessor(directory: Path) -> Mapping[str, Any]:
    root = Path(directory).resolve()
    reproduction = root / "reproduction-source"
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            (
                "import runpy,sys;"
                "sys.path[:0]=sys.argv[1:4];"
                "sys.argv=sys.argv[4:];"
                "runpy.run_path(sys.argv[0],run_name='__main__')"
            ),
            str(reproduction / "src"),
            str(reproduction / "lab/action_dynamics"),
            str(Path(np.__file__).resolve().parents[1]),
            str(
                reproduction
                / "lab/action_dynamics/prototype_peira_assessor.py"
            ),
            "--assessment-only",
            str(root),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=root.parent,
    )
    value = json.loads(result.stdout)
    if not isinstance(value, dict):
        raise ValueError("copied PEIRA assessor returned invalid output")
    return value


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def main(arguments: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assessment-only", action="store_true")
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args(arguments)
    assessment = (
        assess_stored_bundle(args.artifact)
        if args.assessment_only
        else verify_stored_assessment(args.artifact)
    )
    print(
        json.dumps(
            assessment,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
