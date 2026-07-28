"""Train the development-only structured-event temporal graph JEPA."""

import argparse
import gc
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from quantis_core.hybrid_event_features import (
    HybridEventCorpus,
    compile_hybrid_event_corpus,
)
from quantis_core.hybrid_graph_jepa import (
    HybridGraphJepa,
    HybridJepaConfig,
)
from quantis_core.hybrid_frozen_probe import (
    fit_frozen_ridge_future_probe,
    fit_per_entity_pca,
    raw_context_representation,
)
from quantis_core.hybrid_graph_tokens import (
    AlignedEventFeatures,
    HybridGraphTokens,
    MultiMaskConfig,
    compile_hybrid_graph_tokens,
    sample_multi_masks,
)
from quantis_core.hybrid_jepa_development import (
    assess_hybrid_jepa_development,
)
from quantis_core.hybrid_jepa_evaluation import (
    embedding_effective_rank_fraction,
    score_normalized_target_blocks,
)
from quantis_core.observability_graph_corpus import (
    ObservabilityGraphCorpus,
    load_observability_graph_cache,
)
from quantis_core.otlp_logs import (
    OtlpLogCapture,
    read_otlp_log_capture,
)
from quantis_core.structured_events import (
    StructuredEventVocabulary,
)


def run_hybrid_development(
    *,
    cache_index_path: Path,
    cases_directory: Path,
    protocol_path: Path,
    event_cache_path: Path,
    output: Path,
    epochs_override: Optional[int] = None,
    max_seeds: Optional[int] = None,
    device_override: Optional[str] = None,
) -> Mapping[str, Any]:
    """Compile once, train all fixed controls, and write an assessment."""

    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite hybrid development run: {output}"
        )
    staging = output.with_name(f"{output.name}.in-progress")
    if staging.exists():
        raise FileExistsError(
            f"unfinished hybrid development run already exists: {staging}"
        )
    protocol = json.loads(protocol_path.read_text())
    cache_index = json.loads(cache_index_path.read_text())
    cache_directory = (
        cache_index_path.parent / str(cache_index["cache_key"])
    )
    corpus = load_observability_graph_cache(cache_directory)
    started = time.time_ns()
    staging.mkdir(parents=True)
    (staging / "models").mkdir()
    try:
        events, ingestion = _load_or_compile_events(
            corpus,
            cases_directory,
            protocol,
            event_cache_path,
            cache_key=str(cache_index["cache_key"]),
        )
        coarse_factor = int(
            protocol["tokens"]["coarse_pool_size"]
        )
        declared_training = compile_hybrid_graph_tokens(
            corpus.training,
            coarse_factor=coarse_factor,
            aligned_event_features=events.training,
        )
        declared_validation = compile_hybrid_graph_tokens(
            corpus.validation,
            coarse_factor=coarse_factor,
            aligned_event_features=events.validation,
        )
        no_event_training = compile_hybrid_graph_tokens(
            corpus.training,
            coarse_factor=coarse_factor,
        )
        no_event_validation = compile_hybrid_graph_tokens(
            corpus.validation,
            coarse_factor=coarse_factor,
        )
        shuffled_training, shuffled_validation = _shuffled_tokens(
            declared_training,
            declared_validation,
            seed=211,
        )
        seeds = tuple(int(value) for value in protocol["training_seeds"])
        if max_seeds is not None:
            if max_seeds < 1:
                raise ValueError("max_seeds must be positive")
            seeds = seeds[:max_seeds]
        model_values = dict(protocol["model"])
        if epochs_override is not None:
            if epochs_override < 1:
                raise ValueError("epochs_override must be positive")
            model_values["epochs"] = epochs_override
        if device_override is not None:
            model_values["device"] = device_override
        else:
            model_values["device"] = protocol["runtime"]["device"]
        model_values["mask_count"] = int(
            protocol["tokens"]["mask_count"]
        )
        model_values["target_coverage"] = float(
            protocol["tokens"]["mask_coverage"]
        )
        variants = {
            "declared_topology": (
                declared_training,
                declared_validation,
            ),
            "shuffled_topology": (
                shuffled_training,
                shuffled_validation,
            ),
            "no_event_features": (
                no_event_training,
                no_event_validation,
            ),
        }
        operational_feature_count = int(
            corpus.training.contexts.shape[-1]
        )
        operational_mask = np.zeros_like(
            declared_training.feature_mask
        )
        operational_mask[:, :operational_feature_count] = True
        probe_ridge = float(
            protocol["fixed_controls"]["matched_probe_ridge"]
        )
        pca_width = int(
            protocol["fixed_controls"]["matched_pca_width"]
        )
        pca = fit_per_entity_pca(
            declared_training,
            width=pca_width,
        )
        pca_training_representation = pca.transform(
            declared_training
        )
        pca_validation_representation = pca.transform(
            declared_validation
        )
        pca_global_probe = fit_frozen_ridge_future_probe(
            declared_training,
            pca_training_representation,
            mode="one_hop",
            ridge=probe_ridge,
        )
        pca_local_probe = fit_frozen_ridge_future_probe(
            declared_training,
            pca_training_representation,
            mode="entity_local",
            ridge=probe_ridge,
        )
        matched_pca_score = score_normalized_target_blocks(
            declared_training,
            declared_validation,
            pca_global_probe.predict(
                declared_validation,
                pca_validation_representation,
            ),
            channel_mask=operational_mask,
        )
        matched_pca_local_score = score_normalized_target_blocks(
            declared_training,
            declared_validation,
            pca_local_probe.predict(
                declared_validation,
                pca_validation_representation,
            ),
            channel_mask=operational_mask,
        )
        raw_training_representation = (
            raw_context_representation(declared_training)
        )
        raw_validation_representation = (
            raw_context_representation(declared_validation)
        )
        raw_probe = fit_frozen_ridge_future_probe(
            declared_training,
            raw_training_representation,
            mode="one_hop",
            ridge=probe_ridge,
        )
        matched_raw_score = score_normalized_target_blocks(
            declared_training,
            declared_validation,
            raw_probe.predict(
                declared_validation,
                raw_validation_representation,
            ),
            channel_mask=operational_mask,
        )
        results: Dict[str, Any] = {}
        for seed in seeds:
            seed_results: Dict[str, Any] = {}
            seed_directory = staging / "models" / f"seed-{seed}"
            seed_directory.mkdir()
            shared_masks = sample_multi_masks(
                declared_training,
                MultiMaskConfig(
                    mask_count=int(
                        protocol["tokens"]["mask_count"]
                    ),
                    target_coverage=float(
                        protocol["tokens"]["mask_coverage"]
                    ),
                    seed=seed,
                ),
            )
            for variant_name in protocol[
                "development_ablations"
            ]:
                training, validation = variants[str(variant_name)]
                config = HybridJepaConfig(
                    **model_values,
                    seed=seed,
                )
                model_started = time.perf_counter()
                model = HybridGraphJepa(config).fit(
                    training,
                    masks=shared_masks,
                )
                training_prediction = model.predict(training)
                prediction = model.predict(validation)
                probe_training = (
                    replace(
                        training,
                        typed_adjacency=(
                            declared_training.typed_adjacency.copy()
                        ),
                    )
                    if str(variant_name) == "shuffled_topology"
                    else training
                )
                probe_validation = (
                    replace(
                        validation,
                        typed_adjacency=(
                            declared_validation.typed_adjacency.copy()
                        ),
                    )
                    if str(variant_name) == "shuffled_topology"
                    else validation
                )
                variant_operational_mask = np.zeros_like(
                    training.feature_mask
                )
                variant_operational_mask[
                    :, :operational_feature_count
                ] = True
                operational = score_normalized_target_blocks(
                    training,
                    validation,
                    prediction.reconstructed_targets,
                    channel_mask=variant_operational_mask,
                )
                context_recovery = _score_reconstructed_contexts(
                    training,
                    validation,
                    prediction.reconstructed_contexts,
                    variant_operational_mask,
                )
                representation_width = (
                    config.latent_dimension
                )
                training_representation = (
                    training_prediction.validation_embeddings.reshape(
                        len(training.fine_context),
                        len(training.entity_names),
                        representation_width,
                    )
                )
                validation_representation = (
                    prediction.validation_embeddings.reshape(
                        len(validation.fine_context),
                        len(validation.entity_names),
                        representation_width,
                    )
                )
                global_probe = fit_frozen_ridge_future_probe(
                    probe_training,
                    training_representation,
                    mode="one_hop",
                    ridge=probe_ridge,
                )
                local_probe = fit_frozen_ridge_future_probe(
                    probe_training,
                    training_representation,
                    mode="entity_local",
                    ridge=probe_ridge,
                )
                frozen_probe_operational = (
                    score_normalized_target_blocks(
                        probe_training,
                        probe_validation,
                        global_probe.predict(
                            probe_validation,
                            validation_representation,
                        ),
                        channel_mask=variant_operational_mask,
                    )
                )
                local_probe_operational = (
                    score_normalized_target_blocks(
                        probe_training,
                        probe_validation,
                        local_probe.predict(
                            probe_validation,
                            validation_representation,
                        ),
                        channel_mask=variant_operational_mask,
                    )
                )
                joint = score_normalized_target_blocks(
                    training,
                    validation,
                    prediction.reconstructed_targets,
                )
                event = None
                if (
                    training.fine_context.shape[-1]
                    > operational_feature_count
                ):
                    event_mask = np.zeros_like(
                        training.feature_mask
                    )
                    event_mask[
                        :, operational_feature_count:
                    ] = True
                    event = score_normalized_target_blocks(
                        training,
                        validation,
                        prediction.reconstructed_targets,
                        channel_mask=event_mask,
                    )
                latent_l1 = float(
                    np.mean(
                        np.abs(
                            prediction.predicted_latents
                            - prediction.target_latents
                        )
                    )
                )
                future_rank_by_entity_horizon = (
                    _input_dependent_rank_by_entity_horizon(
                        prediction.predicted_latents,
                        training.entity_names,
                        training.horizons,
                        np.any(training.feature_mask, axis=1),
                    )
                )
                context_rank_by_entity = (
                    _context_rank_by_entity(
                        validation_representation,
                        training.entity_names,
                        np.any(training.feature_mask, axis=1),
                    )
                )
                context_effective_rank_fraction = min(
                    context_rank_by_entity.values()
                )
                artifact = model.to_dict()
                artifact_bytes = _canonical_bytes(artifact)
                (
                    seed_directory / f"{variant_name}.json"
                ).write_bytes(artifact_bytes)
                seed_results[str(variant_name)] = {
                    "operational": asdict(operational),
                    "frozen_probe_operational": asdict(
                        frozen_probe_operational
                    ),
                    "local_frozen_probe_operational": asdict(
                        local_probe_operational
                    ),
                    "operational_context_recovery": (
                        context_recovery
                    ),
                    "event": (
                        asdict(event) if event is not None else None
                    ),
                    "joint": asdict(joint),
                    "latent_l1": latent_l1,
                    "effective_rank_fraction": (
                        context_effective_rank_fraction
                    ),
                    "effective_rank_fraction_by_entity": (
                        context_rank_by_entity
                    ),
                    "future_effective_rank_fraction_by_entity_horizon": (
                        future_rank_by_entity_horizon
                    ),
                    "latent_diagnostics": dict(
                        prediction.diagnostics
                    ),
                    "training_losses": list(
                        model.training_losses
                    ),
                    "epoch_metrics": [
                        dict(values)
                        for values in model.epoch_metrics
                    ],
                    "device": model.device,
                    "elapsed_seconds": (
                        time.perf_counter() - model_started
                    ),
                    "model_sha256": hashlib.sha256(
                        artifact_bytes
                    ).hexdigest(),
                }
                del model
                _release_accelerator_cache()
            results[str(seed)] = seed_results
            _write_json(staging / "partial-results.json", results)

        aggregate = _aggregate(results)
        paired_topology_effect = _paired_topology_effect(
            results
        )
        thresholds = protocol["advance_only_if"]
        controls = protocol["fixed_controls"]
        assessment = assess_hybrid_jepa_development(
            declared_topology_normalized_mse=aggregate[
                "declared_topology"
            ]["operational_normalized_mse_mean"],
            shuffled_topology_normalized_mse=aggregate[
                "shuffled_topology"
            ]["operational_normalized_mse_mean"],
            local_context_normalized_mse=aggregate[
                "declared_topology"
            ]["local_probe_operational_normalized_mse_mean"],
            equal_width_pca_normalized_mse=float(
                matched_pca_score.mean_normalized_mse
            ),
            raw_context_normalized_mse=float(
                matched_raw_score.mean_normalized_mse
            ),
            state_reconstruction_normalized_mse=aggregate[
                "declared_topology"
            ]["context_recovery_normalized_mse_mean"],
            effective_rank_fraction=aggregate[
                "declared_topology"
            ]["effective_rank_fraction_minimum"],
            trace_link_coverage=float(
                events.data_quality["trace_link_coverage"]
            ),
            maximum_state_reconstruction_normalized_mse=float(
                thresholds[
                    "maximum_state_reconstruction_normalized_mse"
                ]
            ),
            minimum_effective_rank_fraction=float(
                thresholds[
                    "minimum_effective_rank_fraction"
                ]
            ),
            minimum_topology_relative_improvement=float(
                thresholds[
                    "minimum_topology_relative_improvement"
                ]
            ),
        )
        smoke_run = (
            epochs_override is not None or max_seeds is not None
        )
        report = {
            "schema_version": 1,
            "kind": "hybrid_telemetry_jepa_development_run_v1",
            "status": (
                "smoke_complete"
                if smoke_run
                else assessment["status"]
            ),
            "protocol_sha256": _sha256(protocol_path),
            "corpus_cache_key": cache_index["cache_key"],
            "started_unix_nano": started,
            "completed_unix_nano": time.time_ns(),
            "runtime": {
                "ingestion": ingestion,
                "training_parallelism": (
                    "serialized_on_single_accelerator"
                ),
                "seed_count": len(seeds),
                "epochs": int(model_values["epochs"]),
            },
            "data": {
                "training_shape": list(
                    declared_training.fine_context.shape
                ),
                "validation_shape": list(
                    declared_validation.fine_context.shape
                ),
                "event_feature_count": len(
                    events.training.feature_names
                ),
                "event_data_quality": dict(
                    events.data_quality
                ),
            },
            "fixed_controls": dict(controls),
            "matched_frozen_pca": {
                "width_per_entity": pca_width,
                "one_hop_probe_operational": asdict(
                    matched_pca_score
                ),
                "local_probe_operational": asdict(
                    matched_pca_local_score
                ),
            },
            "matched_raw_hybrid_context": {
                "one_hop_probe_operational": asdict(
                    matched_raw_score
                ),
            },
            "seed_results": results,
            "aggregate": aggregate,
            "paired_topology_effect": paired_topology_effect,
            "assessment": assessment,
            "evidence_boundary": (
                "development-only reuse of an opened nominal corpus; "
                "not confirmation, causal, fault-localization, or "
                "world-model evidence"
            ),
            "next_step": (
                "If every nominal gate passes, preregister and collect "
                "intervention sequences with trace-linked propagation "
                "and multi-step rollout evaluation."
            ),
        }
        _write_json(staging / "assessment.json", report)
        partial = staging / "partial-results.json"
        if partial.exists():
            partial.unlink()
        staging.rename(output)
        return report
    except Exception as error:
        _write_json(
            staging / "failure.json",
            {
                "error_type": type(error).__name__,
                "message": str(error),
                "failed_unix_nano": time.time_ns(),
            },
        )
        raise


def _load_or_compile_events(
    corpus: ObservabilityGraphCorpus,
    cases_directory: Path,
    protocol: Mapping[str, Any],
    event_cache_path: Path,
    *,
    cache_key: str,
) -> Tuple[HybridEventCorpus, Mapping[str, Any]]:
    started = time.perf_counter()
    metadata_path = event_cache_path.with_suffix(".json")
    case_ids = tuple(
        dict.fromkeys(
            corpus.training_case_ids
            + corpus.validation_case_ids
        )
    )
    structured = dict(protocol["structured_events"])
    structured_sha256 = hashlib.sha256(
        _canonical_bytes(structured)
    ).hexdigest()
    workers = int(protocol["runtime"]["log_reader_threads"])
    capture_paths = {
        case_id: (
            cases_directory
            / case_id
            / "collector-logs.jsonl"
        )
        for case_id in case_ids
    }
    with ThreadPoolExecutor(max_workers=workers) as executor:
        capture_sha256 = dict(
            executor.map(
                lambda item: (item[0], _sha256(item[1])),
                capture_paths.items(),
            )
        )
    if (
        event_cache_path.exists()
        and metadata_path.exists()
        and _event_cache_matches(
            metadata_path,
            cache_key=cache_key,
            structured_sha256=structured_sha256,
            capture_sha256=capture_sha256,
        )
    ):
        events = _read_event_cache(
            event_cache_path,
            metadata_path,
            cache_key=cache_key,
            structured_sha256=structured_sha256,
            capture_sha256=capture_sha256,
        )
        return events, {
            "source": "event_cache",
            "elapsed_seconds": time.perf_counter() - started,
        }

    def read(case_id: str) -> Tuple[str, OtlpLogCapture]:
        return (
            case_id,
            read_otlp_log_capture(capture_paths[case_id]),
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        captures = dict(executor.map(read, case_ids))
    events = compile_hybrid_event_corpus(
        corpus,
        captures,
        logical_window_attribute=str(
            structured["logical_window_attribute"]
        ),
        numeric_attribute_names=tuple(
            str(value)
            for value in structured["numeric_attribute_names"]
        ),
        service_to_entity={
            str(key): str(value)
            for key, value in structured[
                "service_to_entity"
            ].items()
        },
        event_entity_overrides={
            str(key): str(value)
            for key, value in structured[
                "event_entity_overrides"
            ].items()
        },
        service_event_entity_overrides={
            str(key): str(value)
            for key, value in structured[
                "service_event_entity_overrides"
            ].items()
        },
    )
    _write_event_cache(
        event_cache_path,
        metadata_path,
        events,
        cache_key=cache_key,
        structured_sha256=structured_sha256,
        capture_sha256=capture_sha256,
    )
    return events, {
        "source": "raw_application_logs",
        "reader_threads": workers,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _write_event_cache(
    path: Path,
    metadata_path: Path,
    events: HybridEventCorpus,
    *,
    cache_key: str,
    structured_sha256: str,
    capture_sha256: Mapping[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        training_contexts=events.training.contexts,
        training_target_blocks=events.training.target_blocks,
        training_observation_mask=(
            events.training.observation_mask
        ),
        validation_contexts=events.validation.contexts,
        validation_target_blocks=events.validation.target_blocks,
        validation_observation_mask=(
            events.validation.observation_mask
        ),
    )
    _write_json(
        metadata_path,
        {
            "schema_version": 1,
            "kind": "hybrid_structured_event_cache_v1",
            "corpus_cache_key": cache_key,
            "structured_event_protocol_sha256": (
                structured_sha256
            ),
            "capture_sha256": dict(
                sorted(capture_sha256.items())
            ),
            "archive_sha256": _sha256(path),
            "feature_names": list(
                events.training.feature_names
            ),
            "vocabulary": events.vocabulary.to_dict(),
            "data_quality": dict(events.data_quality),
        },
    )


def _read_event_cache(
    path: Path,
    metadata_path: Path,
    *,
    cache_key: str,
    structured_sha256: str,
    capture_sha256: Mapping[str, str],
) -> HybridEventCorpus:
    metadata = json.loads(metadata_path.read_text())
    if (
        metadata.get("kind")
        != "hybrid_structured_event_cache_v1"
        or metadata.get("corpus_cache_key") != cache_key
        or metadata.get("structured_event_protocol_sha256")
        != structured_sha256
        or metadata.get("capture_sha256")
        != dict(sorted(capture_sha256.items()))
        or metadata.get("archive_sha256") != _sha256(path)
    ):
        raise ValueError("hybrid event cache identity changed")
    feature_names = tuple(
        str(value) for value in metadata["feature_names"]
    )
    with np.load(path, allow_pickle=False) as archive:
        training = AlignedEventFeatures(
            contexts=np.asarray(
                archive["training_contexts"],
                dtype=np.float64,
            ),
            target_blocks=np.asarray(
                archive["training_target_blocks"],
                dtype=np.float64,
            ),
            observation_mask=np.asarray(
                archive["training_observation_mask"],
                dtype=np.bool_,
            ),
            feature_names=feature_names,
        )
        validation = AlignedEventFeatures(
            contexts=np.asarray(
                archive["validation_contexts"],
                dtype=np.float64,
            ),
            target_blocks=np.asarray(
                archive["validation_target_blocks"],
                dtype=np.float64,
            ),
            observation_mask=np.asarray(
                archive["validation_observation_mask"],
                dtype=np.bool_,
            ),
            feature_names=feature_names,
        )
    return HybridEventCorpus(
        training=training,
        validation=validation,
        vocabulary=StructuredEventVocabulary.from_dict(
            metadata["vocabulary"]
        ),
        data_quality=dict(metadata["data_quality"]),
    )


def _event_cache_matches(
    metadata_path: Path,
    *,
    cache_key: str,
    structured_sha256: str,
    capture_sha256: Mapping[str, str],
) -> bool:
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return bool(
        metadata.get("kind")
        == "hybrid_structured_event_cache_v1"
        and metadata.get("corpus_cache_key") == cache_key
        and metadata.get("structured_event_protocol_sha256")
        == structured_sha256
        and metadata.get("capture_sha256")
        == dict(sorted(capture_sha256.items()))
    )


def _shuffled_tokens(
    training: HybridGraphTokens,
    validation: HybridGraphTokens,
    *,
    seed: int,
) -> Tuple[HybridGraphTokens, HybridGraphTokens]:
    generator = np.random.default_rng(seed)
    permutation = generator.permutation(len(training.entity_ids))
    adjacency = training.typed_adjacency[
        :, permutation, :
    ][:, :, permutation]
    return (
        replace(training, typed_adjacency=adjacency.copy()),
        replace(validation, typed_adjacency=adjacency.copy()),
    )


def _aggregate(
    seed_results: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> Mapping[str, Any]:
    variant_names = tuple(
        next(iter(seed_results.values())).keys()
    )
    aggregate: Dict[str, Any] = {}
    for variant_name in variant_names:
        values = [
            result[variant_name] for result in seed_results.values()
        ]
        aggregate[variant_name] = {
            "operational_normalized_mse_mean": float(
                np.mean(
                    [
                        value["frozen_probe_operational"][
                            "mean_normalized_mse"
                        ]
                        for value in values
                    ]
                )
            ),
            "operational_normalized_mse_std": float(
                np.std(
                    [
                        value["frozen_probe_operational"][
                            "mean_normalized_mse"
                        ]
                        for value in values
                    ]
                )
            ),
            "decoder_operational_normalized_mse_mean": float(
                np.mean(
                    [
                        value["operational"][
                            "mean_normalized_mse"
                        ]
                        for value in values
                    ]
                )
            ),
            "local_probe_operational_normalized_mse_mean": float(
                np.mean(
                    [
                        value["local_frozen_probe_operational"][
                            "mean_normalized_mse"
                        ]
                        for value in values
                    ]
                )
            ),
            "context_recovery_normalized_mse_mean": float(
                np.mean(
                    [
                        value["operational_context_recovery"][
                            "mean_normalized_mse"
                        ]
                        for value in values
                    ]
                )
            ),
            "joint_normalized_mse_mean": float(
                np.mean(
                    [
                        value["joint"]["mean_normalized_mse"]
                        for value in values
                    ]
                )
            ),
            "effective_rank_fraction_minimum": float(
                min(
                    value["effective_rank_fraction"]
                    for value in values
                )
            ),
        }
    return aggregate


def _paired_topology_effect(
    seed_results: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> Mapping[str, Any]:
    relative_improvements = np.asarray(
        [
            (
                values["shuffled_topology"][
                    "frozen_probe_operational"
                ]["mean_normalized_mse"]
                - values["declared_topology"][
                    "frozen_probe_operational"
                ]["mean_normalized_mse"]
            )
            / max(
                values["shuffled_topology"][
                    "frozen_probe_operational"
                ]["mean_normalized_mse"],
                1e-12,
            )
            for values in seed_results.values()
        ],
        dtype=np.float64,
    )
    mean = float(np.mean(relative_improvements))
    if len(relative_improvements) < 2:
        half_width = float("inf")
        confidence_interval: Sequence[Optional[float]] = (
            None,
            None,
        )
    else:
        # Two-sided 95% Student-t critical value for df=2.
        half_width = float(
            4.302652729696142
            * np.std(relative_improvements, ddof=1)
            / np.sqrt(len(relative_improvements))
        )
        confidence_interval = (
            mean - half_width,
            mean + half_width,
        )
    return {
        "relative_improvement_by_seed": (
            relative_improvements.tolist()
        ),
        "mean_relative_improvement": mean,
        "confidence_interval_95": list(confidence_interval),
        "all_seeds_favor_declared": bool(
            np.all(relative_improvements > 0.0)
        ),
    }


def _score_reconstructed_contexts(
    training: HybridGraphTokens,
    evaluation: HybridGraphTokens,
    predictions: NDArray[np.float64],
    channel_mask: NDArray[np.bool_],
) -> Mapping[str, Any]:
    if (
        predictions.shape != evaluation.fine_context.shape
        or channel_mask.shape != training.feature_mask.shape
        or training.feature_names != evaluation.feature_names
        or training.entity_names != evaluation.entity_names
        or not np.all(np.isfinite(predictions))
    ):
        raise ValueError(
            "context reconstruction does not match token schema"
        )
    observed = np.logical_and(
        training.feature_mask, channel_mask
    )
    variance = np.var(training.fine_context, axis=(0, 1))
    mse = np.mean(
        np.square(predictions - evaluation.fine_context),
        axis=(0, 1),
    )
    varying = np.logical_and(observed, variance > 1e-12)
    if not np.any(varying):
        raise ValueError(
            "no varying context channels are available to score"
        )
    normalized = np.zeros_like(variance)
    normalized[varying] = mse[varying] / variance[varying]
    return {
        "mean_normalized_mse": float(
            np.mean(normalized[varying])
        ),
        "entity_normalized_mse": {
            entity_name: (
                float(
                    np.mean(
                        normalized[
                            entity_position,
                            varying[entity_position],
                        ]
                    )
                )
                if np.any(varying[entity_position])
                else None
            )
            for entity_position, entity_name in enumerate(
                training.entity_names
            )
        },
        "scored_channel_count": int(np.count_nonzero(varying)),
    }


def _input_dependent_rank_by_entity_horizon(
    latents: NDArray[np.float64],
    entity_names: Tuple[str, ...],
    horizons: Tuple[int, ...],
    active_entities: NDArray[np.bool_],
) -> Mapping[str, float]:
    if (
        latents.ndim != 4
        or latents.shape[1] != len(horizons)
        or latents.shape[2] != len(entity_names)
        or active_entities.shape != (len(entity_names),)
    ):
        raise ValueError(
            "future latents do not align with entity/horizon schema"
        )
    return {
        f"{entity_name}@h{horizon}": (
            embedding_effective_rank_fraction(
                latents[:, horizon_position, entity_position, :]
            )
        )
        for horizon_position, horizon in enumerate(horizons)
        for entity_position, entity_name in enumerate(entity_names)
        if active_entities[entity_position]
    }


def _context_rank_by_entity(
    representations: NDArray[np.float64],
    entity_names: Tuple[str, ...],
    active_entities: NDArray[np.bool_],
) -> Mapping[str, float]:
    if (
        representations.ndim != 3
        or representations.shape[1] != len(entity_names)
        or active_entities.shape != (len(entity_names),)
    ):
        raise ValueError(
            "context representations do not align with entities"
        )
    return {
        entity_name: embedding_effective_rank_fraction(
            representations[:, entity_position, :]
        )
        for entity_position, entity_name in enumerate(entity_names)
        if active_entities[entity_position]
    }


def _release_accelerator_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except ImportError:
        return


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the hybrid development protocol from the command line."""

    parser = argparse.ArgumentParser()
    root = Path(
        "artifacts/jepa-world-model-v3/"
        "observability-graph-confirmation-v1"
    )
    parser.add_argument(
        "--cache-index",
        type=Path,
        default=root / "graph-cache/cache-index.json",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=root / "cases",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "lab/graph_jepa/"
            "hybrid-telemetry-jepa-development-v1.json"
        ),
    )
    parser.add_argument(
        "--event-cache",
        type=Path,
        default=(
            root.parent
            / "hybrid-telemetry-event-cache-v1/events.npz"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-seeds", type=int)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps"),
    )
    arguments = parser.parse_args(argv)
    report = run_hybrid_development(
        cache_index_path=arguments.cache_index,
        cases_directory=arguments.cases,
        protocol_path=arguments.protocol,
        event_cache_path=arguments.event_cache,
        output=arguments.output,
        epochs_override=arguments.epochs,
        max_seeds=arguments.max_seeds,
        device_override=arguments.device,
    )
    print(json.dumps(report["assessment"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
