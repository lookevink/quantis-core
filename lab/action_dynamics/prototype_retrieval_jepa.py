#!/usr/bin/env python3
"""Retained runner for the frozen ticket 009 retrieval-JEPA tracer."""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from quantis_core.edge_dynamics.data import (
    load_edge_dynamics_cache,
    partition_worker_topology,
)
from quantis_core.edge_dynamics.retrieval_jepa import (
    EpisodeRetrievalConfig,
    EpisodeRetrievalRepresentation,
    OwnedStateRidgeProbe,
    PcaRetrievalRepresentation,
    RawTelemetryRetrievalRepresentation,
    RetrievalEpisodes,
    assess_retrieval_jepa,
    compile_retrieval_episodes,
)


MODEL_NAMES = (
    "episode_predictive_jepa",
    "raw_telemetry",
    "pca_64",
    "deranged_target_jepa",
    "cpc_infonce",
    "supervised_retriever",
)
ROLE_NAMES = (
    "calibration",
    "selection_iid",
    "selection_transfer",
    "evaluation_iid",
    "evaluation_transfer",
)
FROZEN_CACHE = Path(
    "artifacts/action-dynamics/edge-preprocessing-v1/"
    "eb54271132f88c9a431b01e786ea66279a563776434cca2290e47e6b7ae9b3ff"
)
FROZEN_OUTPUT = Path(
    "artifacts/action-dynamics/prototype-retrieval-jepa-v1"
)
FROZEN_STEPS = 400


def run_experiment(
    *,
    cache_directory: Path,
    output_directory: Path,
    steps: int,
    latency_repetitions: int,
    allow_noninterpretable_smoke: bool,
) -> Path:
    """Run one non-overwriting tracer and atomically publish its bundle."""

    cache = Path(cache_directory).resolve()
    output = Path(output_directory).resolve()
    building = output.with_name(output.name + ".building")
    if output.exists() or building.exists():
        raise FileExistsError(
            "retrieval tracer refuses an existing output or staging directory"
        )
    frozen_cache = (Path.cwd() / FROZEN_CACHE).resolve()
    interpretable = (
        steps == FROZEN_STEPS
        and cache == frozen_cache
        and latency_repetitions == 100
    )
    frozen_output = (Path.cwd() / FROZEN_OUTPUT).resolve()
    if not interpretable and not allow_noninterpretable_smoke:
        raise ValueError(
            "non-frozen runs require --allow-noninterpretable-smoke"
        )
    if not interpretable and output == frozen_output:
        raise ValueError(
            "a non-interpretable smoke run cannot use the frozen result path"
        )
    building.mkdir(parents=True)
    started = time.time()
    try:
        data = load_edge_dynamics_cache(cache)
        partitions = {
            role: partition_worker_topology(windows)
            for role, windows in data.windows.items()
        }
        fit_windows = partitions["fit"].in_distribution
        episode_sets = {
            "fit_probe": compile_retrieval_episodes(fit_windows),
            "selection_iid": compile_retrieval_episodes(
                partitions["selection"].in_distribution
            ),
            "selection_transfer": compile_retrieval_episodes(
                partitions["selection"].held_out
            ),
            "evaluation_iid": compile_retrieval_episodes(
                partitions["evaluation"].in_distribution
            ),
            "evaluation_transfer": compile_retrieval_episodes(
                partitions["evaluation"].held_out
            ),
        }
        episode_sets["calibration"] = _concatenate_episodes(
            (
                compile_retrieval_episodes(
                    partitions["calibration"].in_distribution
                ),
                compile_retrieval_episodes(
                    partitions["calibration"].held_out
                ),
            )
        )
        episode_sets["fit_gallery"] = _subset_episodes(
            episode_sets["fit_probe"],
            episode_sets["fit_probe"].is_treatment,
        )
        models, fit_seconds = _fit_models(fit_windows, steps=steps)
        vectors = _encode_all(models, episode_sets, fit_windows.graph)
        model_payloads = {
            name: model.to_dict() for name, model in models.items()
        }
        restored_models = _restore_models(model_payloads)
        restored_vectors = _encode_all(
            restored_models, episode_sets, fit_windows.graph
        )
        causality_audit = _build_causality_audit(
            models,
            episode_sets["evaluation_transfer"],
            fit_windows.graph,
            original_vectors=vectors["evaluation_transfer"],
        )
        probe_payloads = {}
        state_predictions = {}
        target_scales = {}
        target_varying_masks = {}
        ownership = models["raw_telemetry"].ownership_mask
        state_truth = episode_sets["evaluation_transfer"].contexts[
            :, -1
        ][:, ownership]
        for name in MODEL_NAMES:
            probe = OwnedStateRidgeProbe(ridge=1e-3).fit(
                vectors["fit_probe"][name],
                episode_sets["fit_probe"].contexts,
                ownership,
            )
            probe_payloads[name] = probe.to_dict()
            target_scales[name] = probe.target_scale
            target_varying_masks[name] = probe.target_varying_mask
            state_predictions[name] = probe.predict(
                vectors["evaluation_transfer"][name]
            )
        state_scale = target_scales[MODEL_NAMES[0]]
        state_varying_mask = target_varying_masks[MODEL_NAMES[0]]
        if any(
            not np.array_equal(state_scale, target_scales[name])
            or not np.array_equal(
                state_varying_mask, target_varying_masks[name]
            )
            for name in MODEL_NAMES[1:]
        ):
            raise ValueError("owned-state probe target scales diverged")
        gallery = episode_sets["fit_gallery"]
        similarities = {
            role: {
                name: vectors[role][name]
                @ vectors["fit_gallery"][name].T
                for name in MODEL_NAMES
            }
            for role in ROLE_NAMES
        }
        edge_metrics = _measure_edge_metrics(
            models=models,
            model_payloads=model_payloads,
            graph=fit_windows.graph,
            query_context=episode_sets["evaluation_transfer"].contexts[:1],
            bank_vectors={
                name: vectors["fit_gallery"][name]
                for name in MODEL_NAMES
            },
            repetitions=latency_repetitions,
        )
        protocol_checks = _protocol_checks(
            data, episode_sets, causality_audit
        )
        assessment = assess_retrieval_jepa(
            gallery_episode_ids=gallery.episode_ids,
            gallery_labels=gallery.action_and_target_labels,
            similarities=similarities,
            query_labels={
                role: episode_sets[role].action_and_target_labels
                for role in ROLE_NAMES
            },
            is_treatment={
                role: episode_sets[role].is_treatment
                for role in ROLE_NAMES
            },
            pair_ids={
                role: episode_sets[role].pair_ids
                for role in ROLE_NAMES
            },
            bank_vectors={
                name: vectors["fit_gallery"][name]
                for name in MODEL_NAMES
            },
            restored_bank_vectors={
                name: restored_vectors["fit_gallery"][name]
                for name in MODEL_NAMES
            },
            state_truth=state_truth,
            state_scale=state_scale,
            state_varying_mask=state_varying_mask,
            state_predictions=state_predictions,
            original_query_vectors={
                role: {
                    name: vectors[role][name]
                    for name in MODEL_NAMES
                }
                for role in ROLE_NAMES
            },
            restored_query_vectors={
                role: {
                    name: restored_vectors[role][name]
                    for name in MODEL_NAMES
                }
                for role in ROLE_NAMES
            },
            protocol_checks=protocol_checks,
            edge_metrics=edge_metrics,
        )
        protocol = {
            "schema_version": 1,
            "kind": "retrieval_jepa_tracer_protocol",
            "contract": "retrieval-jepa-evidence-contract-v1",
            "interpretable": interpretable,
            "smoke_only": not interpretable,
            "steps": steps,
            "frozen_steps": FROZEN_STEPS,
            "seed": 9019,
            "top_k": 3,
            "latency_repetitions": latency_repetitions,
            "started_unix_seconds": started,
            "completed_unix_seconds": time.time(),
            "runtime": _runtime_identity(),
        }
        data_identity = {
            "schema_version": 1,
            "kind": "retrieval_jepa_data_identity",
            "cache_directory": str(cache),
            "cache_manifest_sha256": _file_sha256(
                cache / "artifact-manifest.json"
            ),
            "source_corpus_sha256": data.source_corpus_sha256,
            "source_artifact_manifest_sha256": (
                data.source_artifact_manifest_sha256
            ),
            "preprocessing_protocol": data.preprocessing_protocol,
            "semantic_schema_sha256": fit_windows.semantic_schema_sha256,
            "implementation_commit": _git_head(),
            "git_status": _git_status(),
        }
        episode_metadata = _episode_metadata(data, episode_sets)
        retrieval_metadata = {
            "schema_version": 1,
            "kind": "retrieval_jepa_stored_assessment_inputs",
            "model_names": list(MODEL_NAMES),
            "role_names": list(ROLE_NAMES),
            "gallery_episode_ids": list(gallery.episode_ids),
            "gallery_labels": list(
                gallery.action_and_target_labels
            ),
            "query_encoder_inputs": ["contexts", "declared_graph"],
            "forbidden_query_encoder_inputs": [
                "future_states",
                "future_controls",
                "future_actions",
                "action_labels",
                "pair_ids",
                "trajectory_ids",
                "transition_indices",
            ],
            "roles": {
                role: {
                    "query_labels": list(
                        episode_sets[role].action_and_target_labels
                    ),
                    "is_treatment": episode_sets[
                        role
                    ].is_treatment.astype(int).tolist(),
                    "pair_ids": list(episode_sets[role].pair_ids),
                }
                for role in ROLE_NAMES
            },
            "edge_metrics": edge_metrics,
            "fit_seconds": fit_seconds,
            "causality_audit": {
                "original_action_labels": list(
                    causality_audit["original_action_labels"]
                ),
                "counterfactual_action_labels": list(
                    causality_audit["counterfactual_action_labels"]
                ),
                "original_pair_ids": list(
                    causality_audit["original_pair_ids"]
                ),
                "counterfactual_pair_ids": list(
                    causality_audit["counterfactual_pair_ids"]
                ),
            },
        }
        _write_json(building / "protocol.json", protocol)
        _write_json(building / "data-identity.json", data_identity)
        _write_json(building / "episode-metadata.json", episode_metadata)
        _write_episode_arrays(building / "episodes.npz", episode_sets)
        _write_json(
            building / "models.json",
            {
                "schema_version": 1,
                "kind": "retrieval_jepa_fitted_models",
                "representations": model_payloads,
                "state_probes": probe_payloads,
            },
        )
        _write_representation_arrays(
            building / "representations.npz", vectors
        )
        _write_json(
            building / "retrieval-metadata.json", retrieval_metadata
        )
        _write_retrieval_evidence(
            building / "retrieval-evidence.npz",
            similarities=similarities,
            bank_vectors={
                name: vectors["fit_gallery"][name]
                for name in MODEL_NAMES
            },
            restored_bank_vectors={
                name: restored_vectors["fit_gallery"][name]
                for name in MODEL_NAMES
            },
            state_truth=state_truth,
            state_scale=state_scale,
            state_varying_mask=state_varying_mask,
            state_predictions=state_predictions,
            original_query_vectors={
                role: {
                    name: vectors[role][name]
                    for name in MODEL_NAMES
                }
                for role in ROLE_NAMES
            },
            restored_query_vectors={
                role: {
                    name: restored_vectors[role][name]
                    for name in MODEL_NAMES
                }
                for role in ROLE_NAMES
            },
            causality_audit=causality_audit,
        )
        _write_json(building / "assessment.json", assessment)
        (building / "report.md").write_text(
            _render_report(
                assessment,
                interpretable=interpretable,
                steps=steps,
            )
        )
        _copy_reproduction_sources(building)
        _write_manifest(building)
        from prototype_retrieval_jepa_assessor import (
            verify_stored_assessment,
        )

        verify_stored_assessment(building)
        os.replace(building, output)
        return output
    except BaseException as error:
        failure = {
            "schema_version": 1,
            "kind": "retrieval_jepa_staging_failure",
            "error_type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        (building / "failure.json").write_text(_pretty_json(failure))
        raise


def _fit_models(
    fit_windows: Any, *, steps: int
) -> Tuple[Mapping[str, Any], Mapping[str, float]]:
    models: Dict[str, Any] = {}
    durations = {}
    specifications = (
        ("episode_predictive_jepa", "episode_predictive_jepa"),
        ("deranged_target_jepa", "deranged_target_jepa"),
        ("cpc_infonce", "cpc_infonce"),
        ("supervised_retriever", "supervised_retriever"),
    )
    for name, objective in specifications:
        started = time.perf_counter()
        models[name] = EpisodeRetrievalRepresentation(
            EpisodeRetrievalConfig(objective=objective, steps=steps)
        ).fit(fit_windows)
        durations[name] = time.perf_counter() - started
    started = time.perf_counter()
    models["raw_telemetry"] = (
        RawTelemetryRetrievalRepresentation().fit(fit_windows)
    )
    durations["raw_telemetry"] = time.perf_counter() - started
    started = time.perf_counter()
    models["pca_64"] = PcaRetrievalRepresentation(width=64).fit(
        fit_windows
    )
    durations["pca_64"] = time.perf_counter() - started
    return (
        {name: models[name] for name in MODEL_NAMES},
        {name: float(durations[name]) for name in MODEL_NAMES},
    )


def _encode_all(
    models: Mapping[str, Any],
    episodes: Mapping[str, RetrievalEpisodes],
    graph: Any,
) -> Mapping[str, Mapping[str, np.ndarray]]:
    output = {}
    for role, values in episodes.items():
        output[role] = {}
        for name in MODEL_NAMES:
            model = models[name]
            if role == "fit_gallery":
                encoded = model.encode_evidence(
                    values.contexts, values.evidence, graph
                )
            else:
                encoded = model.encode_queries(values.contexts, graph)
            output[role][name] = encoded.vectors
    return output


def _restore_models(
    payloads: Mapping[str, Mapping[str, Any]]
) -> Mapping[str, Any]:
    restored = {}
    for name, payload in payloads.items():
        if name == "raw_telemetry":
            restored[name] = (
                RawTelemetryRetrievalRepresentation.from_dict(payload)
            )
        elif name == "pca_64":
            restored[name] = PcaRetrievalRepresentation.from_dict(payload)
        else:
            restored[name] = EpisodeRetrievalRepresentation.from_dict(
                payload
            )
    return restored


def _measure_edge_metrics(
    *,
    models: Mapping[str, Any],
    model_payloads: Mapping[str, Mapping[str, Any]],
    graph: Any,
    query_context: np.ndarray,
    bank_vectors: Mapping[str, np.ndarray],
    repetitions: int,
) -> Mapping[str, Mapping[str, float]]:
    if repetitions < 1:
        raise ValueError("latency repetitions must be positive")
    output = {}
    for name in MODEL_NAMES:
        model = models[name]
        for _ in range(10):
            query = model.encode_queries(query_context, graph).vectors
            _ = query @ bank_vectors[name].T
        query_times = []
        search_times = []
        for _ in range(repetitions):
            started = time.perf_counter_ns()
            query = model.encode_queries(query_context, graph).vectors
            query_times.append(
                (time.perf_counter_ns() - started) / 1_000_000.0
            )
            started = time.perf_counter_ns()
            scores = query @ bank_vectors[name].T
            _ = int(np.argmax(scores[0]))
            search_times.append(
                (time.perf_counter_ns() - started) / 1_000_000.0
            )
        online_parameters = (
            model.inference_parameter_count
            if isinstance(model, EpisodeRetrievalRepresentation)
            else 0
        )
        retained_parameters = (
            model.retained_parameter_count
            if isinstance(model, EpisodeRetrievalRepresentation)
            else 0
        )
        output[name] = {
            "online_parameter_count": float(online_parameters),
            "retained_parameter_count": float(retained_parameters),
            "serialized_model_bytes": float(
                len(_pretty_json(model_payloads[name]).encode())
            ),
            "query_latency_median_ms": float(
                np.median(query_times)
            ),
            "query_latency_p95_ms": float(
                np.percentile(query_times, 95)
            ),
            "search_latency_median_ms": float(
                np.median(search_times)
            ),
            "search_latency_p95_ms": float(
                np.percentile(search_times, 95)
            ),
            "bank_bytes": float(bank_vectors[name].nbytes),
            "bank_items": float(len(bank_vectors[name])),
            "bank_dimension": float(bank_vectors[name].shape[1]),
        }
    return output


def _protocol_checks(
    data: Any,
    episodes: Mapping[str, RetrievalEpisodes],
    causality_audit: Mapping[str, Any],
) -> Mapping[str, bool]:
    pair_roles = {
        role: set(values)
        for role, values in (
            ("fit", data.roles.fit_pair_ids),
            ("selection", data.roles.selection_pair_ids),
            ("calibration", data.roles.calibration_pair_ids),
            ("evaluation", data.roles.evaluation_pair_ids),
        )
    }
    disjoint = all(
        not (pair_roles[left] & pair_roles[right])
        for position, left in enumerate(pair_roles)
        for right in tuple(pair_roles)[position + 1 :]
    )
    context_equal = np.array_equal(
        causality_audit["original_contexts"],
        causality_audit["counterfactual_contexts"],
    )
    forbidden_changed = (
        not np.array_equal(
            causality_audit["original_evidence"],
            causality_audit["counterfactual_evidence"],
        )
        and not np.array_equal(
            causality_audit["original_topology_values"],
            causality_audit["counterfactual_topology_values"],
        )
        and causality_audit["original_action_labels"]
        != causality_audit["counterfactual_action_labels"]
        and causality_audit["original_pair_ids"]
        != causality_audit["counterfactual_pair_ids"]
    )
    vectors_equal = all(
        np.array_equal(
            causality_audit["original_vectors"][name],
            causality_audit["counterfactual_vectors"][name],
        )
        for name in MODEL_NAMES
    )
    return {
        "role_pairs_are_disjoint": disjoint,
        "query_future_is_excluded": (
            context_equal and forbidden_changed and vectors_equal
        ),
        "action_and_identifiers_are_excluded": (
            context_equal and forbidden_changed and vectors_equal
        ),
        "bank_membership_is_equal_and_immutable": (
            len(episodes["fit_gallery"].episode_ids) == 40
            and list(episodes["fit_gallery"].episode_ids)
            == sorted(episodes["fit_gallery"].episode_ids)
        ),
        "episode_counts_match_contract": {
            name: len(values.episode_ids)
            for name, values in episodes.items()
        }
        == {
            "fit_probe": 80,
            "selection_iid": 20,
            "selection_transfer": 10,
            "evaluation_iid": 40,
            "evaluation_transfer": 20,
            "calibration": 30,
            "fit_gallery": 40,
        },
    }


def _build_causality_audit(
    models: Mapping[str, Any],
    episodes: RetrievalEpisodes,
    graph: Any,
    *,
    original_vectors: Mapping[str, np.ndarray],
) -> Mapping[str, Any]:
    counterfactual_contexts = episodes.contexts.copy()
    counterfactual_evidence = (
        episodes.evidence[:, ::-1].copy() + 123.456
    )
    counterfactual_topology = episodes.topology_values + 123.456
    counterfactual_vectors = {
        name: models[name]
        .encode_queries(counterfactual_contexts, graph)
        .vectors
        for name in MODEL_NAMES
    }
    return {
        "original_contexts": episodes.contexts,
        "counterfactual_contexts": counterfactual_contexts,
        "original_evidence": episodes.evidence,
        "counterfactual_evidence": counterfactual_evidence,
        "original_topology_values": episodes.topology_values,
        "counterfactual_topology_values": counterfactual_topology,
        "original_action_labels": episodes.action_and_target_labels,
        "counterfactual_action_labels": tuple(
            f"counterfactual:{position}"
            for position in range(len(episodes.episode_ids))
        ),
        "original_pair_ids": episodes.pair_ids,
        "counterfactual_pair_ids": tuple(
            f"counterfactual-pair:{position}"
            for position in range(len(episodes.episode_ids))
        ),
        "original_vectors": original_vectors,
        "counterfactual_vectors": counterfactual_vectors,
    }


def _concatenate_episodes(
    groups: Sequence[RetrievalEpisodes],
) -> RetrievalEpisodes:
    first = groups[0]
    return RetrievalEpisodes(
        contexts=np.concatenate([value.contexts for value in groups]),
        evidence=np.concatenate([value.evidence for value in groups]),
        episode_ids=tuple(
            item for value in groups for item in value.episode_ids
        ),
        pair_ids=tuple(
            item for value in groups for item in value.pair_ids
        ),
        trajectory_ids=tuple(
            item for value in groups for item in value.trajectory_ids
        ),
        transition_indices=np.concatenate(
            [value.transition_indices for value in groups]
        ),
        is_treatment=np.concatenate(
            [value.is_treatment for value in groups]
        ),
        action_and_target_labels=tuple(
            item
            for value in groups
            for item in value.action_and_target_labels
        ),
        evidence_refs=tuple(
            item for value in groups for item in value.evidence_refs
        ),
        topology_values=np.concatenate(
            [value.topology_values for value in groups]
        ),
    )


def _subset_episodes(
    episodes: RetrievalEpisodes, selection: np.ndarray
) -> RetrievalEpisodes:
    mask = np.asarray(selection, dtype=np.bool_)
    indices = np.asarray(
        sorted(
            np.flatnonzero(mask),
            key=lambda index: episodes.episode_ids[int(index)],
        ),
        dtype=np.int64,
    )
    return RetrievalEpisodes(
        contexts=episodes.contexts[indices],
        evidence=episodes.evidence[indices],
        episode_ids=tuple(episodes.episode_ids[index] for index in indices),
        pair_ids=tuple(episodes.pair_ids[index] for index in indices),
        trajectory_ids=tuple(
            episodes.trajectory_ids[index] for index in indices
        ),
        transition_indices=episodes.transition_indices[indices],
        is_treatment=episodes.is_treatment[indices],
        action_and_target_labels=tuple(
            episodes.action_and_target_labels[index] for index in indices
        ),
        evidence_refs=tuple(
            episodes.evidence_refs[index] for index in indices
        ),
        topology_values=episodes.topology_values[indices],
    )


def _episode_metadata(
    data: Any, episodes: Mapping[str, RetrievalEpisodes]
) -> Mapping[str, Any]:
    return {
        "schema_version": 1,
        "kind": "retrieval_episode_metadata",
        "episode_counts": {
            name: len(values.episode_ids)
            for name, values in episodes.items()
        },
        "source_role_pair_ids": {
            "fit": list(data.roles.fit_pair_ids),
            "selection": list(data.roles.selection_pair_ids),
            "calibration": list(data.roles.calibration_pair_ids),
            "evaluation": list(data.roles.evaluation_pair_ids),
        },
        "episodes": {
            name: {
                "episode_ids": list(values.episode_ids),
                "pair_ids": list(values.pair_ids),
                "trajectory_ids": list(values.trajectory_ids),
                "transition_indices": values.transition_indices.tolist(),
                "is_treatment": values.is_treatment.astype(int).tolist(),
                "action_and_target_labels": list(
                    values.action_and_target_labels
                ),
                "evidence_refs": list(values.evidence_refs),
                "topology_values": values.topology_values.tolist(),
            }
            for name, values in episodes.items()
        },
    }


def _write_episode_arrays(
    path: Path, episodes: Mapping[str, RetrievalEpisodes]
) -> None:
    arrays = {}
    for name, values in episodes.items():
        arrays[f"contexts__{name}"] = values.contexts
        arrays[f"evidence__{name}"] = values.evidence
        arrays[f"transition_indices__{name}"] = (
            values.transition_indices
        )
        arrays[f"is_treatment__{name}"] = values.is_treatment
        arrays[f"topology_values__{name}"] = values.topology_values
    np.savez_compressed(path, **arrays)


def _write_representation_arrays(
    path: Path,
    vectors: Mapping[str, Mapping[str, np.ndarray]],
) -> None:
    np.savez_compressed(
        path,
        **{
            f"vectors__{role}__{model}": values
            for role, models in vectors.items()
            for model, values in models.items()
        },
    )


def _write_retrieval_evidence(
    path: Path,
    *,
    similarities: Mapping[str, Mapping[str, np.ndarray]],
    bank_vectors: Mapping[str, np.ndarray],
    restored_bank_vectors: Mapping[str, np.ndarray],
    state_truth: np.ndarray,
    state_scale: np.ndarray,
    state_varying_mask: np.ndarray,
    state_predictions: Mapping[str, np.ndarray],
    original_query_vectors: Mapping[
        str, Mapping[str, np.ndarray]
    ],
    restored_query_vectors: Mapping[
        str, Mapping[str, np.ndarray]
    ],
    causality_audit: Mapping[str, Any],
) -> None:
    arrays = {
        f"similarity__{role}__{model}": values
        for role, models in similarities.items()
        for model, values in models.items()
    }
    arrays.update(
        {
            f"bank_vectors__{name}": values
            for name, values in bank_vectors.items()
        }
    )
    arrays.update(
        {
            f"restored_bank_vectors__{name}": values
            for name, values in restored_bank_vectors.items()
        }
    )
    arrays.update(
        {
            f"state_prediction__{name}": values
            for name, values in state_predictions.items()
        }
    )
    arrays.update(
        {
            f"original_query__{role}__{name}": values
            for role, models in original_query_vectors.items()
            for name, values in models.items()
        }
    )
    arrays.update(
        {
            f"restored_query__{role}__{name}": values
            for role, models in restored_query_vectors.items()
            for name, values in models.items()
        }
    )
    arrays["state_truth"] = state_truth
    arrays["state_scale"] = state_scale
    arrays["state_varying_mask"] = state_varying_mask
    arrays["causality_original_contexts"] = causality_audit[
        "original_contexts"
    ]
    arrays["causality_counterfactual_contexts"] = causality_audit[
        "counterfactual_contexts"
    ]
    arrays["causality_original_evidence"] = causality_audit[
        "original_evidence"
    ]
    arrays["causality_counterfactual_evidence"] = causality_audit[
        "counterfactual_evidence"
    ]
    arrays["causality_original_topology_values"] = causality_audit[
        "original_topology_values"
    ]
    arrays["causality_counterfactual_topology_values"] = causality_audit[
        "counterfactual_topology_values"
    ]
    arrays.update(
        {
            f"causality_original_query__{name}": values
            for name, values in causality_audit[
                "original_vectors"
            ].items()
        }
    )
    arrays.update(
        {
            f"causality_counterfactual_query__{name}": values
            for name, values in causality_audit[
                "counterfactual_vectors"
            ].items()
        }
    )
    np.savez_compressed(path, **arrays)


def _copy_reproduction_sources(building: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    sources = (
        Path(__file__).resolve(),
        root
        / "lab/action_dynamics/prototype_retrieval_jepa_assessor.py",
        root / "src/quantis_core/edge_dynamics/retrieval_jepa.py",
        root / "tests/test_retrieval_jepa.py",
        root / "docs/specs/retrieval-jepa-evidence-contract-v1.md",
        root
        / "docs/research/retrieval-jepa-primary-source-notes.md",
    )
    reproduction = building / "reproduction"
    reproduction.mkdir()
    for source in sources:
        shutil.copy2(source, reproduction / source.name)


def _write_manifest(building: Path) -> None:
    files = {
        str(path.relative_to(building)): {
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        for path in sorted(building.rglob("*"))
        if path.is_file() and path.name != "artifact-manifest.json"
    }
    _write_json(
        building / "artifact-manifest.json",
        {
            "schema_version": 1,
            "kind": "retrieval_jepa_artifact_manifest",
            "files": files,
        },
    )


def _render_report(
    assessment: Mapping[str, Any],
    *,
    interpretable: bool,
    steps: int,
) -> str:
    evaluation = assessment["metrics"]["evaluation_transfer"]
    rows = []
    for name in MODEL_NAMES:
        values = evaluation[name]
        rows.append(
            "| "
            + " | ".join(
                (
                    name,
                    f"{values['hit_at_1']:.3f}",
                    f"{values['hit_at_3']:.3f}",
                    f"{values['mean_reciprocal_rank']:.3f}",
                    f"{values['accepted_correct_rate']:.3f}",
                    f"{values['selective_accuracy']:.3f}",
                    f"{values['control_specificity']:.3f}",
                )
            )
            + " |"
        )
    status = (
        "Frozen interpretable tracer"
        if interpretable
        else "NON-INTERPRETABLE SMOKE RUN"
    )
    return "\n".join(
        (
            "# Retrieval-JEPA tracer report",
            "",
            f"Status: **{status}** (`{steps}` optimizer steps).",
            "",
            "| model | hit@1 | hit@3 | MRR | accepted-correct | selective accuracy | control specificity |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            f"Decision: `{assessment['decision']}`.",
            "",
            "Abstention is empirical only. The calibration corpus is too small",
            "for a 10%-risk, 95%-confidence SGR guarantee.",
            "",
        )
    )


def _runtime_identity() -> Mapping[str, Any]:
    import torch

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
    }


def _git_head() -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_status() -> Sequence[str]:
    return subprocess.run(
        ("git", "status", "--short"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(_pretty_json(value))


def _pretty_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=FROZEN_CACHE)
    parser.add_argument("--output", type=Path, default=FROZEN_OUTPUT)
    parser.add_argument("--steps", type=int, default=FROZEN_STEPS)
    parser.add_argument("--latency-repetitions", type=int, default=100)
    parser.add_argument(
        "--allow-noninterpretable-smoke", action="store_true"
    )
    arguments = parser.parse_args()
    result = run_experiment(
        cache_directory=arguments.cache,
        output_directory=arguments.output,
        steps=arguments.steps,
        latency_repetitions=arguments.latency_repetitions,
        allow_noninterpretable_smoke=(
            arguments.allow_noninterpretable_smoke
        ),
    )
    print(result)


if __name__ == "__main__":
    main()
