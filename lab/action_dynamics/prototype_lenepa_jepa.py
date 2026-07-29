#!/usr/bin/env python3
"""Retained runner for the frozen LeNEPA telemetry tracer."""

import argparse
import gzip
import hashlib
import json
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

try:
    from lab.action_dynamics.prototype_complete_lejepa import (
        _action_sanity_evidence,
        _attribution_evidence,
        _downstream_pair_errors,
        _forecast_scores,
        _state_probe,
        _transfer_queries,
    )
    from lab.action_dynamics.prototype_lenepa_jepa_assessor import (
        NEURAL_NAMES,
        REPRESENTATION_NAMES,
        assess_stored_bundle,
        verify_stored_assessment,
    )
except ModuleNotFoundError:
    from prototype_complete_lejepa import (
        _action_sanity_evidence,
        _attribution_evidence,
        _downstream_pair_errors,
        _forecast_scores,
        _state_probe,
        _transfer_queries,
    )
    from prototype_lenepa_jepa_assessor import (
        NEURAL_NAMES,
        REPRESENTATION_NAMES,
        assess_stored_bundle,
        verify_stored_assessment,
    )
from quantis_core.action_conditioned_dynamics import (
    ActionConditionedWindows,
)
from quantis_core.edge_dynamics.complete_lejepa import (
    EntityPcaRepresentation,
    PairBlockedAnchorSchedule,
    ReducedRankActionProbe,
    fit_owned_feature_mask,
)
from quantis_core.edge_dynamics.data import (
    PreparedAttributionQueries,
    load_edge_dynamics_cache,
    partition_worker_topology,
)
from quantis_core.edge_dynamics.lenepa_jepa import (
    LenepaConfig,
    LenepaRepresentation,
)
from quantis_core.edge_dynamics.models import (
    ContractiveLowRankDynamics,
    LowRankConfig,
)


FROZEN_CACHE = Path(
    "artifacts/action-dynamics/edge-preprocessing-v1/"
    "eb54271132f88c9a431b01e786ea66279a563776434cca2290e47e6b7ae9b3ff"
)
FROZEN_OUTPUT = Path(
    "artifacts/action-dynamics/prototype-lenepa-jepa-v1"
)
FROZEN_STEPS = 1600
RIDGES = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
IMPLEMENTATION_SOURCE_PATHS = (
    "lab/action_dynamics/prototype_lenepa_jepa.py",
    "lab/action_dynamics/prototype_lenepa_jepa_assessor.py",
    "lab/action_dynamics/prototype_complete_lejepa.py",
    "src/quantis_core/edge_dynamics/lenepa_jepa.py",
    "src/quantis_core/edge_dynamics/complete_lejepa.py",
    "src/quantis_core/edge_dynamics/action_conditioned_jepa.py",
    "src/quantis_core/edge_dynamics/models.py",
    "src/quantis_core/edge_dynamics/data.py",
    "src/quantis_core/action_conditioned_dynamics.py",
    "src/quantis_core/graph_telemetry.py",
    "tests/test_lenepa_jepa.py",
    "docs/research/lenepa-primary-source-notes.md",
    "docs/specs/lenepa-telemetry-tracer-v1.md",
    "docs/wayfinding/jepa-implementation-program/"
    "024-test-lenepa-projection.md",
)


def run_experiment(
    *,
    cache_directory: Path,
    output_directory: Path,
    steps: int,
    latency_repetitions: int,
    allow_noninterpretable_smoke: bool,
) -> Path:
    """Fit, independently assess, and atomically publish LeNEPA."""

    cache = Path(cache_directory).resolve()
    output = Path(output_directory).resolve()
    building = output.with_name(output.name + ".building")
    if output.exists() or building.exists():
        raise FileExistsError("LeNEPA refuses an existing output")
    interpretable = (
        cache == (Path.cwd() / FROZEN_CACHE).resolve()
        and output == (Path.cwd() / FROZEN_OUTPUT).resolve()
        and steps == FROZEN_STEPS
        and latency_repetitions == 100
    )
    if not interpretable and not allow_noninterpretable_smoke:
        raise ValueError(
            "non-frozen LeNEPA runs require explicit smoke permission"
        )
    if not interpretable and output == (Path.cwd() / FROZEN_OUTPUT).resolve():
        raise ValueError("LeNEPA smoke cannot use the frozen output")
    implementation_commit = _git_head()
    source_hashes = _source_identity(require_clean=interpretable)
    building.mkdir(parents=True)
    models_directory = building / "models"
    models_directory.mkdir()
    reproduction_directory = building / "reproduction-source"
    started = time.time()

    prepared = load_edge_dynamics_cache(cache)
    partitions = {
        role: partition_worker_topology(windows)
        for role, windows in prepared.windows.items()
    }
    held_values = {
        value.held_out_normalized_value for value in partitions.values()
    }
    if len(held_values) != 1:
        raise ValueError("LeNEPA held topology differs by role")
    held_value = next(iter(held_values))
    windows_by_role = {
        "fit": partitions["fit"].in_distribution,
        "selection": partitions["selection"].in_distribution,
        "calibration": partitions["calibration"].in_distribution,
        "iid_evaluation": partitions["evaluation"].in_distribution,
        "transfer_evaluation": partitions["evaluation"].held_out,
    }
    fit = windows_by_role["fit"]
    ownership = fit_owned_feature_mask(fit)
    base_config = _config("projected_lenepa", steps)
    _write_anchor_schedule(building, fit, base_config)

    models: Dict[str, LenepaRepresentation] = {}
    training_seconds = {}
    for name in NEURAL_NAMES:
        config = _config(name, steps)
        fit_started = time.perf_counter()
        model = LenepaRepresentation(config).fit(fit)
        training_seconds[name] = time.perf_counter() - fit_started
        models[name] = model
        _write_json(models_directory / f"{name}.json", model.to_dict())
    pca_started = time.perf_counter()
    pca = EntityPcaRepresentation(width=64).fit(fit)
    training_seconds["matched_pca"] = (
        time.perf_counter() - pca_started
    )
    _write_json(models_directory / "matched_pca.json", pca.to_dict())

    encoded = {
        name: {
            role: _encode_representation(
                name, models, pca, windows.histories, windows.graph
            )
            for role, windows in windows_by_role.items()
            if role != "calibration"
        }
        for name in REPRESENTATION_NAMES
    }
    raw = ContractiveLowRankDynamics(
        LowRankConfig(rank=32)
    ).fit(fit)
    _write_json(models_directory / "raw_rank32.json", raw.to_dict())
    raw_predictions = {
        role: raw.rollout(
            windows.histories,
            windows.future_controls,
            windows.future_actions,
            windows.graph,
        ).mean
        for role, windows in windows_by_role.items()
        if role not in ("fit", "calibration")
    }
    raw_scores = {
        role: _forecast_scores(values, windows_by_role[role])
        for role, values in raw_predictions.items()
    }

    probes: Dict[str, ReducedRankActionProbe] = {}
    selected_ridges: Dict[str, float] = {}
    selection_safety_failed: Dict[str, bool] = {}
    ridge_curves: Dict[str, list[Mapping[str, Any]]] = {}
    ridge_predictions: Dict[str, Dict[float, np.ndarray]] = {}
    for name in REPRESENTATION_NAMES:
        curve = []
        fitted = {}
        predictions_by_ridge = {}
        for ridge in RIDGES:
            probe = ReducedRankActionProbe(rank=32, ridge=ridge).fit(
                encoded[name]["fit"],
                fit.future_controls,
                fit.future_actions,
                fit.future_states,
            )
            prediction = probe.predict(
                encoded[name]["selection"],
                windows_by_role["selection"].future_controls,
                windows_by_role["selection"].future_actions,
            )
            scores = _forecast_scores(
                prediction, windows_by_role["selection"]
            )
            raw_safe = (
                scores["overall_mse"]
                <= 1.05 * raw_scores["selection"]["overall_mse"]
                and scores["action_overlap_mse"]
                <= 1.05
                * raw_scores["selection"]["action_overlap_mse"]
            )
            curve.append(
                {"ridge": ridge, "raw_safe": raw_safe, **scores}
            )
            fitted[ridge] = probe
            predictions_by_ridge[ridge] = prediction
        eligible = [row for row in curve if row["raw_safe"]]
        selection_safety_failed[name] = not bool(eligible)
        selected = min(
            eligible or curve,
            key=lambda row: (
                row["downstream_effect_mse"],
                row["ridge"],
            ),
        )
        selected_ridge = float(selected["ridge"])
        probes[name] = fitted[selected_ridge]
        selected_ridges[name] = selected_ridge
        ridge_curves[name] = curve
        ridge_predictions[name] = predictions_by_ridge
        _write_json(
            models_directory / f"{name}-probe.json",
            probes[name].to_dict(),
        )

    predictions = {
        name: {
            role: probes[name].predict(
                encoded[name][role],
                windows.future_controls,
                windows.future_actions,
            )
            for role, windows in windows_by_role.items()
            if role not in ("fit", "calibration")
        }
        for name in REPRESENTATION_NAMES
    }
    forecast_scores = {
        name: {
            role: _forecast_scores(values, windows_by_role[role])
            for role, values in role_values.items()
        }
        for name, role_values in predictions.items()
    }
    state_probes = {
        name: {
            role: _state_probe(
                encoded[name]["fit"],
                fit,
                encoded[name][role],
                windows_by_role[role],
                ownership,
            )
            for role in (
                "selection",
                "iid_evaluation",
                "transfer_evaluation",
            )
        }
        for name in REPRESENTATION_NAMES
    }

    transfer_queries = _transfer_queries(
        prepared.attribution_queries,
        fit.control_feature_names,
        held_value,
    )
    query_tokens = {
        name: _encode_representation(
            name,
            models,
            pca,
            transfer_queries.histories,
            fit.graph,
        )
        for name in REPRESENTATION_NAMES
    }
    attribution = {}
    attribution_predictions = {}
    action_sanity = {}
    action_sanity_predictions = {}
    for name in REPRESENTATION_NAMES:
        attribution[name], attribution_predictions[name] = (
            _attribution_evidence(
                probes[name],
                query_tokens[name],
                transfer_queries,
                ownership,
            )
        )
        action_sanity[name], action_sanity_predictions[name] = (
            _action_sanity_evidence(
                probes[name],
                encoded[name]["transfer_evaluation"],
                windows_by_role["transfer_evaluation"],
                ownership,
            )
        )

    diagnostics = {
        name: {
            role: _diagnostic_for_role(
                models[name], windows_by_role[role]
            )
            for role in ("selection", "transfer_evaluation")
        }
        for name in NEURAL_NAMES
    }
    mechanism = {
        name: {
            role: {
                "cosine_error": diagnostic.cosine_error,
                "retrieval_hit_at_1": diagnostic.retrieval_hit_at_1,
                "input_sigreg": diagnostic.input_sigreg,
                "output_sigreg": diagnostic.output_sigreg,
                "prediction_effective_rank": _effective_rank(
                    diagnostic.predicted_tokens
                ),
                "target_effective_rank": _effective_rank(
                    diagnostic.target_tokens
                ),
            }
            for role, diagnostic in roles.items()
        }
        for name, roles in diagnostics.items()
    }

    restored_models = {
        name: LenepaRepresentation.from_dict(models[name].to_dict())
        for name in NEURAL_NAMES
    }
    restored_pca = EntityPcaRepresentation.from_dict(pca.to_dict())
    transfer = windows_by_role["transfer_evaluation"]
    restoration_original = {}
    restoration_restored = {}
    for name in REPRESENTATION_NAMES:
        restoration_original[name] = encoded[name][
            "transfer_evaluation"
        ][:8]
        if name in NEURAL_NAMES:
            restoration_restored[name] = _encode_chunks(
                restored_models[name].encode,
                transfer.histories[:8],
                transfer.graph,
            )
        else:
            restoration_restored[name] = _encode_chunks(
                restored_pca.encode,
                transfer.histories[:8],
                transfer.graph,
            )
    restoration_probe_original = {}
    restoration_probe_restored = {}
    for name in REPRESENTATION_NAMES:
        original = probes[name].predict(
            encoded[name]["transfer_evaluation"][:8],
            transfer.future_controls[:8],
            transfer.future_actions[:8],
        )
        restored_probe = ReducedRankActionProbe.from_dict(
            probes[name].to_dict()
        )
        replay = restored_probe.predict(
            encoded[name]["transfer_evaluation"][:8],
            transfer.future_controls[:8],
            transfer.future_actions[:8],
        )
        restoration_probe_original[name] = original
        restoration_probe_restored[name] = replay
    restoration_sequence_original = {}
    restoration_sequence_restored = {}
    restoration_diagnostic_original = {}
    restoration_diagnostic_restored = {}
    prefix_original = {}
    prefix_altered = {}
    altered = transfer.histories[:8].copy()
    altered[:, 10:] += 10_000.0
    for name in NEURAL_NAMES:
        restoration_sequence_original[name] = models[
            name
        ].encode_sequence(transfer.histories[:8], transfer.graph)
        restoration_sequence_restored[name] = restored_models[
            name
        ].encode_sequence(transfer.histories[:8], transfer.graph)
        restoration_diagnostic_original[name] = models[
            name
        ].diagnose_next_latent(
            transfer.histories[:8], transfer.graph
        )
        restoration_diagnostic_restored[name] = restored_models[
            name
        ].diagnose_next_latent(
            transfer.histories[:8], transfer.graph
        )
        prefix_original[name] = restoration_sequence_original[name][
            :, :10
        ]
        prefix_altered[name] = models[name].encode_sequence(
            altered, transfer.graph
        )[:, :10]

    bundle_path = (
        models_directory / "projected_lenepa-inference.json.gz"
    )
    candidate_bundle = _write_inference_bundle(
        models["projected_lenepa"],
        probes["projected_lenepa"],
        bundle_path,
    )
    deployed_model, deployed_probe = _load_inference_bundle(bundle_path)
    latency, latency_samples = _latency(
        deployed_model,
        deployed_probe,
        transfer,
        repetitions=latency_repetitions,
    )
    parameter_counts = {
        name: {
            "training": models[name].training_parameter_count,
            "inference": models[name].inference_parameter_count,
        }
        for name in NEURAL_NAMES
    }

    evidence_arrays: Dict[str, np.ndarray] = {}
    for role in (
        "fit",
        "selection",
        "iid_evaluation",
        "transfer_evaluation",
    ):
        windows = windows_by_role[role]
        evidence_arrays[f"histories__{role}"] = (
            windows.histories.astype(np.float32)
        )
        evidence_arrays[f"target__{role}"] = (
            windows.future_states.astype(np.float32)
        )
        evidence_arrays[f"controls__{role}"] = (
            windows.future_controls.astype(np.float32)
        )
        evidence_arrays[f"actions__{role}"] = (
            windows.future_actions.astype(np.float32)
        )
        for name in REPRESENTATION_NAMES:
            evidence_arrays[f"representation__{name}__{role}"] = (
                encoded[name][role].astype(np.float32)
            )
    for role in (
        "selection",
        "iid_evaluation",
        "transfer_evaluation",
    ):
        evidence_arrays[f"raw_prediction__{role}"] = (
            raw_predictions[role].astype(np.float32)
        )
        for name in REPRESENTATION_NAMES:
            evidence_arrays[f"prediction__{name}__{role}"] = (
                predictions[name][role].astype(np.float32)
            )
    for name in REPRESENTATION_NAMES:
        for position, ridge in enumerate(RIDGES):
            evidence_arrays[
                f"ridge_prediction__{name}__{position}"
            ] = ridge_predictions[name][ridge].astype(np.float32)
        evidence_arrays[f"attribution_prediction__{name}"] = (
            attribution_predictions[name].astype(np.float32)
        )
        for variant, values in action_sanity_predictions[name].items():
            evidence_arrays[
                f"action_sanity__{name}__{variant}"
            ] = values.astype(np.float32)
        evidence_arrays[f"restoration_original__{name}"] = (
            restoration_original[name].astype(np.float32)
        )
        evidence_arrays[f"restoration_restored__{name}"] = (
            restoration_restored[name].astype(np.float32)
        )
        evidence_arrays[f"restoration_probe_original__{name}"] = (
            restoration_probe_original[name].astype(np.float32)
        )
        evidence_arrays[f"restoration_probe_restored__{name}"] = (
            restoration_probe_restored[name].astype(np.float32)
        )
    for name in NEURAL_NAMES:
        for role, diagnostic in diagnostics[name].items():
            for field in (
                "input_tokens",
                "output_tokens",
                "predicted_tokens",
                "target_tokens",
            ):
                evidence_arrays[
                    f"diagnostic_{field}__{name}__{role}"
                ] = getattr(diagnostic, field).astype(np.float32)
        evidence_arrays[f"restoration_sequence_original__{name}"] = (
            restoration_sequence_original[name].astype(np.float32)
        )
        evidence_arrays[f"restoration_sequence_restored__{name}"] = (
            restoration_sequence_restored[name].astype(np.float32)
        )
        for field in (
            "input_tokens",
            "output_tokens",
            "predicted_tokens",
            "target_tokens",
        ):
            evidence_arrays[
                f"restoration_diagnostic_{field}_original__{name}"
            ] = getattr(
                restoration_diagnostic_original[name], field
            ).astype(np.float32)
            evidence_arrays[
                f"restoration_diagnostic_{field}_restored__{name}"
            ] = getattr(
                restoration_diagnostic_restored[name], field
            ).astype(np.float32)
        evidence_arrays[f"prefix_original__{name}"] = (
            prefix_original[name].astype(np.float32)
        )
        evidence_arrays[f"prefix_altered__{name}"] = (
            prefix_altered[name].astype(np.float32)
        )
    evidence_arrays["query_histories"] = (
        transfer_queries.histories.astype(np.float32)
    )
    evidence_arrays["query_future_controls"] = (
        transfer_queries.future_controls.astype(np.float32)
    )
    evidence_arrays["query_observed_future"] = (
        transfer_queries.observed_future.astype(np.float32)
    )
    evidence_arrays["query_candidate_actions"] = (
        transfer_queries.candidate_actions.astype(np.float32)
    )
    evidence_arrays["latency_samples_ms"] = latency_samples
    np.savez_compressed(building / "evidence.npz", **evidence_arrays)

    metadata = {
        "schema_version": 1,
        "kind": "lenepa_assessment_evidence_v1",
        "interpretable": interpretable,
        "implementation_commit": implementation_commit,
        "graph": fit.graph.to_dict(),
        "entity_names": list(fit.entity_names),
        "state_feature_names": list(fit.state_feature_names),
        "control_feature_names": list(fit.control_feature_names),
        "action_feature_names": list(fit.action_feature_names),
        "ownership_mask": ownership.astype(int).tolist(),
        "roles": {
            role: _role_metadata(windows)
            for role, windows in windows_by_role.items()
        },
        "queries": _query_metadata(transfer_queries),
        "ridge_values": list(RIDGES),
        "selected_ridges": selected_ridges,
        "selection_safety_failed": selection_safety_failed,
        "parameter_counts": parameter_counts,
        "public_inference_is_causal": True,
        "deployed_bundle_bytes": candidate_bundle,
        "latency": latency,
        "configs": {
            name: models[name].config.to_dict() for name in NEURAL_NAMES
        },
    }
    _write_json(building / "evidence-metadata.json", metadata)
    for source_name in IMPLEMENTATION_SOURCE_PATHS:
        source = Path(source_name)
        destination = reproduction_directory / source_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    assessment = dict(assess_stored_bundle(building))
    _write_json(building / "assessment.json", assessment)
    report = {
        "schema_version": 1,
        "kind": "lenepa_telemetry_tracer_v1",
        "evidence_boundary": (
            "single-seed open-development representation tracer; "
            "not a production alert system or sealed confirmation"
        ),
        "interpretable": interpretable,
        "implementation_commit": implementation_commit,
        "source": {
            "cache_directory": str(cache),
            "source_corpus_sha256": prepared.source_corpus_sha256,
            "source_artifact_manifest_sha256": (
                prepared.source_artifact_manifest_sha256
            ),
            "preprocessing_protocol": prepared.preprocessing_protocol,
            "held_out_worker_topology_normalized": held_value,
            "primary_paper": "https://arxiv.org/abs/2607.00958",
            "official_repository": (
                "https://github.com/langotime/lenepa-milets-2026"
            ),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "started_unix": started,
            "completed_unix": time.time(),
        },
        "configurations": {
            name: models[name].config.to_dict() for name in NEURAL_NAMES
        },
        "training_seconds": training_seconds,
        "training_metrics": {
            name: [
                dict(row) for row in models[name].training_metrics
            ]
            for name in NEURAL_NAMES
        },
        "selected_ridges": selected_ridges,
        "selection_safety_failed": selection_safety_failed,
        "ridge_curves": ridge_curves,
        "forecast_scores": forecast_scores,
        "raw_scores": raw_scores,
        "state_probes": state_probes,
        "attribution": attribution,
        "action_sanity": action_sanity,
        "mechanism": mechanism,
        "transfer_pair_errors": {
            name: _downstream_pair_errors(
                predictions[name]["transfer_evaluation"], transfer
            )
            for name in REPRESENTATION_NAMES
        },
        "parameter_counts": parameter_counts,
        "deployed_bundle_bytes": candidate_bundle,
        "latency": latency,
        "assessment": assessment,
        "source_sha256": source_hashes,
    }
    _write_json(building / "result.json", report)
    (building / "REPORT.md").write_text(_markdown_report(report))
    manifest = {
        "schema_version": 1,
        "kind": "lenepa_artifact_manifest_v1",
        "implementation_commit": implementation_commit,
        "sha256": {
            str(path.relative_to(building)): _file_sha256(path)
            for path in sorted(building.rglob("*"))
            if path.is_file()
        },
    }
    _write_json(building / "artifact-manifest.json", manifest)
    verify_stored_assessment(building)
    building.rename(output)
    verify_stored_assessment(output)
    return output


def _write_anchor_schedule(
    directory: Path,
    fit: ActionConditionedWindows,
    config: LenepaConfig,
) -> None:
    schedule = PairBlockedAnchorSchedule(
        fit, seed=config.anchor_seed
    )
    batches = [schedule.batch(step) for step in range(config.steps)]
    np.savez_compressed(
        directory / "anchor-schedule.npz",
        indices=np.stack([batch.indices for batch in batches]),
        arm_ids=np.stack([batch.arm_ids for batch in batches]),
        transition_indices=np.stack(
            [batch.transition_indices for batch in batches]
        ),
        pair_ids=np.asarray(schedule.pair_ids),
    )


def _config(objective: str, steps: int) -> LenepaConfig:
    return LenepaConfig(
        objective=objective,
        steps=steps,
        warmup_steps=min(80, steps),
    )


def _diagnostic_for_role(
    model: LenepaRepresentation,
    windows: ActionConditionedWindows,
) -> Any:
    return model.diagnose_next_latent(
        windows.histories, windows.graph
    )


def _encode_representation(
    name: str,
    models: Mapping[str, LenepaRepresentation],
    pca: EntityPcaRepresentation,
    histories: np.ndarray,
    graph: Any,
) -> np.ndarray:
    call = pca.encode if name == "matched_pca" else models[name].encode
    return _encode_chunks(call, histories, graph)


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


def _role_metadata(
    windows: ActionConditionedWindows,
) -> Mapping[str, Any]:
    return {
        "trajectory_ids": list(windows.trajectory_ids),
        "pair_ids": list(windows.matched_pair_ids),
        "transition_indices": windows.transition_indices.astype(int).tolist(),
    }


def _query_metadata(
    queries: PreparedAttributionQueries,
) -> Mapping[str, Any]:
    return {
        "query_ids": list(queries.query_ids),
        "candidate_ids": list(queries.candidate_ids),
        "candidate_action_kinds": list(queries.candidate_action_kinds),
        "candidate_target_entities": list(
            queries.candidate_target_entities
        ),
        "expected_action_kinds": list(queries.expected_action_kinds),
        "expected_target_entities": list(
            queries.expected_target_entities
        ),
        "expected_variant_ids": list(queries.expected_variant_ids),
    }


def _write_inference_bundle(
    model: LenepaRepresentation,
    probe: ReducedRankActionProbe,
    path: Path,
) -> int:
    inference = model.to_inference_dict()
    inference["probe"] = probe.to_dict()
    encoded = json.dumps(
        inference,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    path.write_bytes(gzip.compress(encoded, mtime=0))
    return len(path.read_bytes())


def _load_inference_bundle(
    path: Path,
) -> tuple[LenepaRepresentation, ReducedRankActionProbe]:
    payload = dict(
        json.loads(gzip.decompress(Path(path).read_bytes()).decode())
    )
    model = LenepaRepresentation.from_inference_dict(payload)
    probe = ReducedRankActionProbe.from_dict(dict(payload["probe"]))
    return model, probe


def _latency(
    model: LenepaRepresentation,
    probe: ReducedRankActionProbe,
    windows: ActionConditionedWindows,
    *,
    repetitions: int,
) -> tuple[Mapping[str, Any], np.ndarray]:
    if repetitions < 1:
        raise ValueError("LeNEPA latency repetitions must be positive")

    def call() -> None:
        encoded = model.encode(
            windows.histories[:1], windows.graph
        ).tokens
        probe.predict(
            encoded,
            windows.future_controls[:1],
            windows.future_actions[:1],
        )

    call()
    values = []
    for _ in range(repetitions):
        tick = time.perf_counter_ns()
        call()
        values.append((time.perf_counter_ns() - tick) / 1e6)
    samples = np.asarray(values, dtype=np.float64)
    return {
        "median_ms": float(np.median(samples)),
        "p95_ms": float(np.quantile(samples, 0.95)),
        "repetitions": int(repetitions),
    }, samples


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


def _source_identity(*, require_clean: bool) -> Mapping[str, str]:
    if require_clean:
        status = subprocess.run(
            ["git", "status", "--short"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if status.strip():
            raise RuntimeError(
                "frozen LeNEPA run requires a clean implementation commit"
            )
    return {
        name: _file_sha256(Path(name))
        for name in IMPLEMENTATION_SOURCE_PATHS
    }


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def _markdown_report(report: Mapping[str, Any]) -> str:
    assessment = dict(report["assessment"])
    return "\n".join(
        (
            "# LeNEPA telemetry tracer v1",
            "",
            f"- Decision: `{assessment['decision']}`",
            f"- Passed: `{assessment['passed']}`",
            f"- Interpretable: `{report['interpretable']}`",
            f"- Implementation: `{report['implementation_commit']}`",
            "",
            "See `result.json`, `assessment.json`, and `evidence.npz` "
            "for the complete retained evidence.",
            "",
        )
    )


def main(arguments: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=FROZEN_CACHE)
    parser.add_argument("--output", type=Path, default=FROZEN_OUTPUT)
    parser.add_argument("--steps", type=int, default=FROZEN_STEPS)
    parser.add_argument("--latency-repetitions", type=int, default=100)
    parser.add_argument(
        "--allow-noninterpretable-smoke", action="store_true"
    )
    parsed = parser.parse_args(arguments)
    result = run_experiment(
        cache_directory=parsed.cache,
        output_directory=parsed.output,
        steps=parsed.steps,
        latency_repetitions=parsed.latency_repetitions,
        allow_noninterpretable_smoke=(
            parsed.allow_noninterpretable_smoke
        ),
    )
    print(
        json.dumps(
            verify_stored_assessment(result),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
