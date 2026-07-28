"""Train and assess the preregistered observability graph models."""

import argparse
import concurrent.futures
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from quantis_core.graph_jepa import (
    GraphJepaTrainingConfig,
    LinearGraphJepaWorldModel,
)
from quantis_core.graph_telemetry import GraphStateWindows
from quantis_core.learned_graph_jepa import (
    GraphEmaJepaConfig,
    LearnedGraphJepaWorldModel,
)
from quantis_core.observability_graph_corpus import (
    load_observability_graph_cache,
)


def run_confirmation(
    *,
    cache_index_path: Path,
    corpus_protocol_path: Path,
    training_protocol_path: Path,
    preregistered_git_commit: str,
    output: Path,
) -> Mapping[str, Any]:
    """Run fixed seeds and controls, then evaluate every frozen gate."""

    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite graph training: {output}"
        )
    if (
        re.fullmatch(
            r"[0-9a-f]{40}", preregistered_git_commit
        )
        is None
    ):
        raise ValueError(
            "preregistered git commit must be a full SHA-1"
        )
    cache_index = json.loads(cache_index_path.read_text())
    cache_directory = (
        cache_index_path.parent / cache_index["cache_key"]
    )
    corpus = load_observability_graph_cache(cache_directory)
    corpus_protocol = json.loads(corpus_protocol_path.read_text())
    training_protocol = json.loads(
        training_protocol_path.read_text()
    )
    corpus_protocol_sha256 = _canonical_sha256(corpus_protocol)
    if (
        corpus_protocol_sha256
        != training_protocol["corpus_protocol_sha256"]
    ):
        raise ValueError(
            "training protocol is not bound to corpus protocol"
        )
    output.mkdir(parents=True)
    (output / "models").mkdir()
    started = time.time_ns()
    widths = {
        str(key): int(value)
        for key, value in training_protocol[
            "entity_latent_dimensions"
        ].items()
    }
    optimization = dict(training_protocol["primary_model"])
    shuffled = {
        str(key): tuple(str(item) for item in value)
        for key, value in training_protocol[
            "shuffled_context_entities"
        ].items()
    }
    seed_results: Dict[str, Any] = {}
    primary_artifact_bytes: Optional[bytes] = None
    primary_model: Optional[LearnedGraphJepaWorldModel] = None
    seeds = tuple(
        int(seed) for seed in training_protocol["training_seeds"]
    )
    models_by_seed: Dict[
        int, Mapping[str, LearnedGraphJepaWorldModel]
    ] = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=int(training_protocol["parallel_seed_jobs"])
    ) as executor:
        future_by_seed = {
            executor.submit(
                _fit_seed_models,
                corpus.training,
                widths,
                optimization,
                seed_number,
                shuffled,
            ): seed_number
            for seed_number in seeds
        }
        for future in concurrent.futures.as_completed(
            future_by_seed
        ):
            seed_number = future_by_seed[future]
            models_by_seed[seed_number] = future.result()
    for seed_number in seeds:
        models = models_by_seed[seed_number]
        model_scores = {}
        seed_directory = output / "models" / f"seed-{seed_number}"
        seed_directory.mkdir()
        for name, model in models.items():
            artifact = model.to_dict()
            artifact_bytes = _canonical_bytes(artifact)
            (seed_directory / f"{name}.json").write_bytes(
                artifact_bytes
            )
            prediction = model.predict(corpus.validation)
            model_scores[name] = {
                **_score_decoded(
                    corpus.training,
                    corpus.validation,
                    corpus.validation_case_ids,
                    prediction.decoded_target_blocks,
                ),
                "reconstruction": _score_decoded(
                    corpus.training,
                    corpus.validation,
                    corpus.validation_case_ids,
                    prediction.reconstructed_target_blocks,
                ),
                "training_initial_loss": (
                    model.training_losses[0]
                ),
                "training_final_loss": (
                    model.training_losses[-1]
                ),
                "model_sha256": hashlib.sha256(
                    artifact_bytes
                ).hexdigest(),
            }
        seed_results[str(seed_number)] = model_scores
        if seed_number == int(
            training_protocol["determinism_repeat_seed"]
        ):
            primary_model = models["one_hop"]
            primary_artifact_bytes = _canonical_bytes(
                primary_model.to_dict()
            )
    assert primary_model is not None
    assert primary_artifact_bytes is not None

    repeat_seed = int(
        training_protocol["determinism_repeat_seed"]
    )
    repeat = _fit_learned(
        corpus.training,
        widths,
        optimization,
        repeat_seed,
        "one_hop",
    )
    repeat_bytes = _canonical_bytes(repeat.to_dict())
    (output / "models" / "determinism-repeat.json").write_bytes(
        repeat_bytes
    )
    deterministic = repeat_bytes == primary_artifact_bytes

    fixed = dict(training_protocol["fixed_controls"])
    equal_width = int(fixed["equal_pca_width"])
    pca_widths = {
        entity_id: equal_width
        for position, entity_id in enumerate(
            corpus.training.entity_ids
        )
        if np.any(corpus.training.observation_mask[position])
    }
    pca_model = LinearGraphJepaWorldModel(
        GraphJepaTrainingConfig(
            latent_dimension=equal_width,
            ridge=float(fixed["ridge"]),
            context_scope="one_hop",
            entity_latent_dimensions=pca_widths,
        )
    ).fit(corpus.training)
    pca_artifact = pca_model.to_dict()
    (output / "models" / "equal-width-pca.json").write_bytes(
        _canonical_bytes(pca_artifact)
    )
    pca_prediction = pca_model.predict(corpus.validation)
    controls = {
        "equal_width_pca": _score_decoded(
            corpus.training,
            corpus.validation,
            corpus.validation_case_ids,
            pca_prediction.decoded_target_blocks,
        ),
        "training_mean": _score_decoded(
            corpus.training,
            corpus.validation,
            corpus.validation_case_ids,
            _training_mean_prediction(
                corpus.training, corpus.validation
            ),
        ),
        "persistence": _score_decoded(
            corpus.training,
            corpus.validation,
            corpus.validation_case_ids,
            _persistence_prediction(corpus.validation),
        ),
        "flat_raw_ridge": _score_decoded(
            corpus.training,
            corpus.validation,
            corpus.validation_case_ids,
            _raw_ridge_prediction(
                corpus.training,
                corpus.validation,
                float(fixed["ridge"]),
                one_hop=False,
            ),
        ),
        "one_hop_raw_ridge": _score_decoded(
            corpus.training,
            corpus.validation,
            corpus.validation_case_ids,
            _raw_ridge_prediction(
                corpus.training,
                corpus.validation,
                float(fixed["ridge"]),
                one_hop=True,
            ),
        ),
    }
    aggregate = _aggregate_seed_results(seed_results)
    thresholds = dict(training_protocol["gates"])
    one_hop = aggregate["one_hop"]
    local = aggregate["entity_local"]
    all_entities = aggregate["all_entities"]
    shuffled_score = aggregate["shuffled_one_hop"]
    family_wins = sum(
        one_hop["family_normalized_mse"][family]
        <= local["family_normalized_mse"][family]
        for family in one_hop["family_normalized_mse"]
    )
    seed_wins = sum(
        float(scores["one_hop"]["mean_normalized_mse"])
        < float(
            scores["entity_local"]["mean_normalized_mse"]
        )
        and float(scores["one_hop"]["mean_normalized_mse"])
        < float(
            scores["shuffled_one_hop"][
                "mean_normalized_mse"
            ]
        )
        for scores in seed_results.values()
    )
    reconstruction_error = float(
        one_hop["mean_reconstruction_normalized_mse"]
    )
    raw_one_hop = float(
        controls["one_hop_raw_ridge"]["mean_normalized_mse"]
    )
    training_mean = float(
        controls["training_mean"]["mean_normalized_mse"]
    )
    compression = _compression(corpus.training, widths)
    training_variance = _training_feature_variance(
        corpus.training
    )
    expected_constant = set(
        training_protocol["training_only_pre_validation_audit"][
            "expected_constant_normal_features"
        ]
    )
    observed_constant = {
        key
        for key, variance in training_variance.items()
        if variance <= 1e-12
    }
    critical_entities = tuple(
        str(value)
        for value in training_protocol["critical_entities"]
    )
    critical_maximum = max(
        float(
            controls["one_hop_raw_ridge"][
                "entity_normalized_mse"
            ][entity_id]
        )
        for entity_id in critical_entities
    )
    gates = {
        "raw_flat_beats_training_mean": (
            float(
                controls["flat_raw_ridge"][
                    "mean_normalized_mse"
                ]
            )
            < training_mean
        ),
        "raw_one_hop_beats_training_mean": (
            raw_one_hop < training_mean
        ),
        "raw_one_hop_below_one": (
            raw_one_hop
            < float(
                thresholds[
                    "maximum_raw_one_hop_normalized_mse"
                ]
            )
        ),
        "critical_raw_groups_below_one": (
            critical_maximum
            < float(
                thresholds[
                    "maximum_critical_group_normalized_mse"
                ]
            )
        ),
        "target_representation_reconstructs_state": (
            reconstruction_error
            <= float(
                thresholds[
                    "maximum_reconstruction_normalized_mse"
                ]
            )
        ),
        "active_context_is_compressed": (
            compression["ratio"]
            >= float(
                thresholds[
                    "minimum_context_compression_ratio"
                ]
            )
        ),
        "one_hop_beats_entity_local": (
            float(one_hop["mean_normalized_mse"])
            < float(local["mean_normalized_mse"])
        ),
        "one_hop_retains_all_entity_performance": (
            float(one_hop["mean_normalized_mse"])
            <= float(
                thresholds[
                    "maximum_one_hop_to_all_entity_ratio"
                ]
            )
            * float(all_entities["mean_normalized_mse"])
        ),
        "one_hop_beats_equal_width_pca": (
            float(one_hop["mean_normalized_mse"])
            < float(
                controls["equal_width_pca"][
                    "mean_normalized_mse"
                ]
            )
        ),
        "one_hop_beats_shuffled_topology": (
            float(one_hop["mean_normalized_mse"])
            < float(shuffled_score["mean_normalized_mse"])
        ),
        "validation_family_stability": (
            family_wins
            >= int(
                thresholds[
                    "minimum_no_worse_validation_families"
                ]
            )
        ),
        "seed_stability": (
            seed_wins
            >= int(thresholds["minimum_seed_wins"])
        ),
        "deterministic_primary_repeat": deterministic,
        "expected_nominal_invariants_are_constant": (
            expected_constant <= observed_constant
        ),
        "all_claimed_targets_vary_in_training": (
            observed_constant == expected_constant
        ),
    }
    passed = all(gates.values())
    completed = time.time_ns()
    assessment = {
        "schema_version": 1,
        "kind": "observability_rich_graph_jepa_assessment_v1",
        "status": "supported" if passed else "not_supported",
        "publication_ready": passed,
        "claim": corpus_protocol["claim"],
        "evidence_boundary": (
            "fault-free workload-schedule and one-to-three-worker "
            "transfer inside the fixed Quantis checkout stack"
        ),
        "started_unix_nano": started,
        "completed_unix_nano": completed,
        "corpus_cache_key": cache_index["cache_key"],
        "corpus_protocol_sha256": corpus_protocol_sha256,
        "training_protocol_sha256": _canonical_sha256(
            training_protocol
        ),
        "preregistered_git_commit": preregistered_git_commit,
        "seed_results": seed_results,
        "aggregate_learned_models": aggregate,
        "controls": controls,
        "compression": compression,
        "family_wins": family_wins,
        "seed_wins": seed_wins,
        "critical_raw_group_maximum": critical_maximum,
        "training_feature_variance": training_variance,
        "expected_constant_normal_features": sorted(
            expected_constant
        ),
        "observed_constant_training_features": sorted(
            observed_constant
        ),
        "gates": {
            name: {"passed": value}
            for name, value in gates.items()
        },
        "limitations": corpus_protocol["claim_exclusions"],
    }
    (output / "assessment.json").write_text(
        json.dumps(
            assessment,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    return assessment


def _fit_learned(
    training: GraphStateWindows,
    widths: Mapping[str, int],
    optimization: Mapping[str, Any],
    seed: int,
    scope: str,
    overrides: Optional[Mapping[str, Tuple[str, ...]]] = None,
) -> LearnedGraphJepaWorldModel:
    return LearnedGraphJepaWorldModel(
        GraphEmaJepaConfig(
            entity_latent_dimensions=widths,
            context_scope=scope,
            context_entity_overrides=overrides,
            epochs=int(optimization["epochs"]),
            learning_rate=float(
                optimization["learning_rate"]
            ),
            ema_decay=float(optimization["ema_decay"]),
            weight_decay=float(optimization["weight_decay"]),
            batch_size=int(optimization["batch_size"]),
            seed=seed,
        )
    ).fit(training)


def _fit_seed_models(
    training: GraphStateWindows,
    widths: Mapping[str, int],
    optimization: Mapping[str, Any],
    seed: int,
    shuffled: Mapping[str, Tuple[str, ...]],
) -> Mapping[str, LearnedGraphJepaWorldModel]:
    return {
        "one_hop": _fit_learned(
            training,
            widths,
            optimization,
            seed,
            "one_hop",
        ),
        "entity_local": _fit_learned(
            training,
            widths,
            optimization,
            seed,
            "entity_local",
        ),
        "all_entities": _fit_learned(
            training,
            widths,
            optimization,
            seed,
            "all_entities",
        ),
        "shuffled_one_hop": _fit_learned(
            training,
            widths,
            optimization,
            seed,
            "one_hop",
            shuffled,
        ),
    }


def _score_decoded(
    training: GraphStateWindows,
    validation: GraphStateWindows,
    validation_case_ids: Sequence[str],
    decoded: NDArray[np.float64],
) -> Mapping[str, Any]:
    if decoded.shape != validation.target_blocks.shape:
        raise ValueError("decoded graph targets do not align")
    repeated_cases = np.repeat(
        np.asarray(validation_case_ids, dtype=object),
        len(validation.horizons) * validation.target_block_size,
    )
    features: Dict[str, float] = {}
    entities: Dict[str, list[float]] = {}
    families: Dict[str, list[float]] = {}
    for entity_position, entity_id in enumerate(
        training.entity_ids
    ):
        for slot_position, feature_key in enumerate(
            training.local_feature_keys[entity_position]
        ):
            training_target = training.target_blocks[
                :, :, :, entity_position, slot_position
            ].reshape(-1)
            variance = float(np.var(training_target))
            if variance <= 1e-12:
                continue
            validation_target = validation.target_blocks[
                :, :, :, entity_position, slot_position
            ].reshape(-1)
            squared = np.square(
                decoded[
                    :, :, :, entity_position, slot_position
                ].reshape(-1)
                - validation_target
            )
            error = float(np.mean(squared) / variance)
            features[feature_key] = error
            entities.setdefault(entity_id, []).append(error)
            for case_id in sorted(set(validation_case_ids)):
                family = _family_id(case_id)
                selected = repeated_cases == case_id
                families.setdefault(family, []).append(
                    float(np.mean(squared[selected]) / variance)
                )
    return {
        "mean_normalized_mse": float(
            np.mean(tuple(features.values()))
        ),
        "feature_normalized_mse": features,
        "entity_normalized_mse": {
            key: float(np.mean(value))
            for key, value in entities.items()
        },
        "family_normalized_mse": {
            key: float(np.mean(value))
            for key, value in families.items()
        },
    }


def _aggregate_seed_results(
    seed_results: Mapping[str, Any],
) -> Mapping[str, Any]:
    model_names = tuple(
        next(iter(seed_results.values())).keys()
    )
    aggregate = {}
    for model_name in model_names:
        scores = [
            seed_result[model_name]
            for seed_result in seed_results.values()
        ]
        aggregate[model_name] = {
            "mean_normalized_mse": float(
                np.mean(
                    [
                        score["mean_normalized_mse"]
                        for score in scores
                    ]
                )
            ),
            "mean_reconstruction_normalized_mse": float(
                np.mean(
                    [
                        score["reconstruction"][
                            "mean_normalized_mse"
                        ]
                        for score in scores
                    ]
                )
            ),
            "family_normalized_mse": {
                family: float(
                    np.mean(
                        [
                            score["family_normalized_mse"][
                                family
                            ]
                            for score in scores
                        ]
                    )
                )
                for family in scores[0][
                    "family_normalized_mse"
                ]
            },
        }
    return aggregate


def _training_mean_prediction(
    training: GraphStateWindows,
    validation: GraphStateWindows,
) -> NDArray[np.float64]:
    prediction = np.zeros_like(validation.target_blocks)
    for entity_position, keys in enumerate(
        training.local_feature_keys
    ):
        for slot_position, _ in enumerate(keys):
            prediction[
                :, :, :, entity_position, slot_position
            ] = float(
                np.mean(
                    training.target_blocks[
                        :, :, :, entity_position, slot_position
                    ]
                )
            )
    return prediction


def _persistence_prediction(
    validation: GraphStateWindows,
) -> NDArray[np.float64]:
    last = validation.contexts[:, -1]
    return np.repeat(
        np.repeat(
            last[:, None, None, :, :],
            len(validation.horizons),
            axis=1,
        ),
        validation.target_block_size,
        axis=2,
    )


def _raw_ridge_prediction(
    training: GraphStateWindows,
    validation: GraphStateWindows,
    ridge: float,
    *,
    one_hop: bool,
) -> NDArray[np.float64]:
    prediction = np.zeros_like(validation.target_blocks)
    if not one_hop:
        training_design = _flat_design(training)
        validation_design = _flat_design(validation)
        target = training.target_blocks.reshape(
            len(training.contexts) * len(training.horizons),
            -1,
        )
        result = _ridge_predict(
            training_design,
            target,
            validation_design,
            ridge,
        )
        return result.reshape(validation.target_blocks.shape)
    for position, entity_id in enumerate(training.entity_ids):
        mask = training.observation_mask[position]
        if not np.any(mask):
            continue
        context_ids = tuple(
            candidate
            for candidate in training.graph.neighboring_entity_ids(
                entity_id
            )
            if np.any(
                training.observation_mask[
                    training.entity_ids.index(candidate)
                ]
            )
        )
        train_design = _entity_design(training, context_ids)
        validation_design = _entity_design(
            validation, context_ids
        )
        target = training.target_blocks[
            :, :, :, position, mask
        ].reshape(
            len(training.contexts) * len(training.horizons),
            -1,
        )
        result = _ridge_predict(
            train_design,
            target,
            validation_design,
            ridge,
        ).reshape(
            len(validation.contexts),
            len(validation.horizons),
            validation.target_block_size,
            int(np.count_nonzero(mask)),
        )
        prediction[:, :, :, position, mask] = result
    return prediction


def _flat_design(
    windows: GraphStateWindows,
) -> NDArray[np.float64]:
    active = windows.contexts[
        :, :, windows.observation_mask
    ].reshape(len(windows.contexts), -1)
    context = np.repeat(
        active[:, None, :],
        len(windows.horizons),
        axis=1,
    )
    controls = windows.target_controls.reshape(
        len(windows.contexts), len(windows.horizons), -1
    )
    return np.asarray(
        np.concatenate((context, controls), axis=2).reshape(
            len(windows.contexts) * len(windows.horizons),
            -1,
        ),
        dtype=np.float64,
    )


def _entity_design(
    windows: GraphStateWindows,
    entity_ids: Tuple[str, ...],
) -> NDArray[np.float64]:
    context_parts = []
    for entity_id in entity_ids:
        position = windows.entity_ids.index(entity_id)
        mask = windows.observation_mask[position]
        context_parts.append(
            windows.contexts[:, :, position, mask].reshape(
                len(windows.contexts), -1
            )
        )
    context = np.repeat(
        np.concatenate(context_parts, axis=1)[:, None, :],
        len(windows.horizons),
        axis=1,
    )
    controls = windows.target_controls.reshape(
        len(windows.contexts), len(windows.horizons), -1
    )
    return np.asarray(
        np.concatenate((context, controls), axis=2).reshape(
            len(windows.contexts) * len(windows.horizons),
            -1,
        ),
        dtype=np.float64,
    )


def _ridge_predict(
    training_design: NDArray[np.float64],
    training_target: NDArray[np.float64],
    validation_design: NDArray[np.float64],
    ridge: float,
) -> NDArray[np.float64]:
    location = np.mean(training_design, axis=0)
    scale = np.std(training_design, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    normalized = (training_design - location) / scale
    validation_normalized = (
        validation_design - location
    ) / scale
    design = np.column_stack(
        (np.ones(len(normalized)), normalized)
    )
    validation_with_bias = np.column_stack(
        (
            np.ones(len(validation_normalized)),
            validation_normalized,
        )
    )
    penalty = ridge * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(
        design.T @ design + penalty,
        design.T @ training_target,
    )
    return validation_with_bias @ weights


def _compression(
    windows: GraphStateWindows,
    widths: Mapping[str, int],
) -> Mapping[str, Any]:
    raw = windows.contexts.shape[1] * int(
        np.count_nonzero(windows.observation_mask)
    )
    latent = (
        windows.contexts.shape[1]
        // windows.target_block_size
        * sum(widths.values())
    )
    return {
        "active_raw_context_values": raw,
        "latent_context_values": latent,
        "ratio": raw / latent,
    }


def _training_feature_variance(
    windows: GraphStateWindows,
) -> Mapping[str, float]:
    return {
        feature_key: float(
            np.var(
                windows.target_blocks[
                    :, :, :, entity_position, slot_position
                ]
            )
        )
        for entity_position, feature_keys in enumerate(
            windows.local_feature_keys
        )
        for slot_position, feature_key in enumerate(feature_keys)
    }


def _family_id(case_id: str) -> str:
    return next(
        part for part in case_id.split("-") if part.startswith("f")
    )


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(value).rstrip()).hexdigest()


def main(arguments: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-index", type=Path, required=True)
    parser.add_argument(
        "--corpus-protocol", type=Path, required=True
    )
    parser.add_argument(
        "--training-protocol", type=Path, required=True
    )
    parser.add_argument(
        "--preregistered-git-commit", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parsed = parser.parse_args(arguments)
    assessment = run_confirmation(
        cache_index_path=parsed.cache_index,
        corpus_protocol_path=parsed.corpus_protocol,
        training_protocol_path=parsed.training_protocol,
        preregistered_git_commit=(
            parsed.preregistered_git_commit
        ),
        output=parsed.output,
    )
    print(f"Graph JEPA confirmation: {assessment['status']}")
    print(f"Assessment: {parsed.output / 'assessment.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
