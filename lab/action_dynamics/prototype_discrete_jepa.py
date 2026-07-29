#!/usr/bin/env python3
"""Retained runner for the frozen Discrete-JEPA telemetry tracer."""

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
        _forecast_scores,
        _transfer_queries,
    )
    from lab.action_dynamics.prototype_discrete_jepa_assessor import (
        NEURAL_NAMES,
        REPRESENTATION_NAMES,
        assess_stored_bundle,
        verify_stored_assessment,
    )
except ModuleNotFoundError:
    from prototype_complete_lejepa import (
        _action_sanity_evidence,
        _attribution_evidence,
        _forecast_scores,
        _transfer_queries,
    )
    from prototype_discrete_jepa_assessor import (
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
from quantis_core.edge_dynamics.discrete_jepa import (
    DiscreteJepaConfig,
    DiscreteJepaRepresentation,
    DiscreteMaskSchedule,
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
    "artifacts/action-dynamics/prototype-discrete-jepa-v1"
)
FROZEN_STEPS = 800
FROZEN_SOURCE_CORPUS_SHA256 = (
    "df03af282f48591216251f22f934b97e1555147df9ed6f6c6d1f7684f9644a26"
)
FROZEN_SOURCE_ARTIFACT_MANIFEST_SHA256 = (
    "d02afa33a5977cba69b255cd1a5f31470751b6681da1ecae31b11e86307e65b1"
)
FROZEN_PREPROCESSING_PROTOCOL = (
    "action_conditioned_jepa_topology_transfer_v1"
)
RIDGES = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
IMPLEMENTATION_SOURCE_PATHS = (
    "lab/action_dynamics/prototype_discrete_jepa.py",
    "lab/action_dynamics/prototype_discrete_jepa_assessor.py",
    "lab/action_dynamics/prototype_lenepa_jepa_assessor.py",
    "lab/action_dynamics/prototype_complete_lejepa.py",
    "src/quantis_core/edge_dynamics/discrete_jepa.py",
    "src/quantis_core/edge_dynamics/lenepa_jepa.py",
    "src/quantis_core/edge_dynamics/complete_lejepa.py",
    "src/quantis_core/edge_dynamics/action_conditioned_jepa.py",
    "src/quantis_core/edge_dynamics/models.py",
    "src/quantis_core/edge_dynamics/data.py",
    "src/quantis_core/action_conditioned_dynamics.py",
    "src/quantis_core/action_dynamics_corpus.py",
    "src/quantis_core/action_dynamics_lab.py",
    "src/quantis_core/action_dynamics_real_corpus.py",
    "src/quantis_core/contextual_multimodal_corpus.py",
    "src/quantis_core/demand_conditioning.py",
    "src/quantis_core/detectors.py",
    "src/quantis_core/fault_matrix.py",
    "src/quantis_core/graph_telemetry.py",
    "src/quantis_core/multimodal_corpus.py",
    "src/quantis_core/otlp.py",
    "src/quantis_core/otlp_logs.py",
    "src/quantis_core/otlp_log_windowing.py",
    "src/quantis_core/otlp_windowing.py",
    "src/quantis_core/telemetry_corpus.py",
    "src/quantis_core/windowing.py",
    "tests/test_discrete_jepa.py",
    "docs/research/discrete-jepa-primary-source-notes.md",
    "docs/specs/discrete-jepa-telemetry-tracer-v1.md",
    "docs/wayfinding/jepa-implementation-program/"
    "025-test-discrete-jepa.md",
)


def run_experiment(
    *,
    cache_directory: Path,
    output_directory: Path,
    steps: int,
    latency_repetitions: int,
    allow_noninterpretable_smoke: bool,
) -> Path:
    """Fit, independently assess, and publish Discrete-JEPA."""

    cache = Path(cache_directory).resolve()
    output = Path(output_directory).resolve()
    building = output.with_name(output.name + ".building")
    if output.exists() or building.exists():
        raise FileExistsError(
            "Discrete-JEPA refuses an existing output"
        )
    interpretable = (
        cache == (Path.cwd() / FROZEN_CACHE).resolve()
        and output == (Path.cwd() / FROZEN_OUTPUT).resolve()
        and steps == FROZEN_STEPS
        and latency_repetitions == 100
    )
    if not interpretable and not allow_noninterpretable_smoke:
        raise ValueError(
            "non-frozen Discrete-JEPA runs require smoke permission"
        )
    if (
        not interpretable
        and output == (Path.cwd() / FROZEN_OUTPUT).resolve()
    ):
        raise ValueError(
            "Discrete-JEPA smoke cannot use the frozen output"
        )
    implementation_commit = _git_head()
    source_hashes = _source_identity(require_clean=interpretable)
    prepared = load_edge_dynamics_cache(cache)
    source_is_frozen = bool(
        prepared.source_corpus_sha256
        == FROZEN_SOURCE_CORPUS_SHA256
        and prepared.source_artifact_manifest_sha256
        == FROZEN_SOURCE_ARTIFACT_MANIFEST_SHA256
        and prepared.preprocessing_protocol
        == FROZEN_PREPROCESSING_PROTOCOL
    )
    if interpretable and not source_is_frozen:
        raise ValueError("frozen Discrete-JEPA source identity differs")
    interpretable = bool(interpretable and source_is_frozen)

    building.mkdir(parents=True)
    models_directory = building / "models"
    models_directory.mkdir()
    reproduction_directory = building / "reproduction-source"
    started = time.time()
    partitions = {
        role: partition_worker_topology(windows)
        for role, windows in prepared.windows.items()
    }
    held_values = {
        value.held_out_normalized_value for value in partitions.values()
    }
    if len(held_values) != 1:
        raise ValueError(
            "Discrete-JEPA held topology differs by role"
        )
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
    base_config = _config("discrete_complete", steps)
    _write_schedules(building, fit, base_config)

    models: Dict[str, DiscreteJepaRepresentation] = {}
    training_seconds = {}
    for name in NEURAL_NAMES:
        config = _config(name, steps)
        tick = time.perf_counter()
        model = DiscreteJepaRepresentation(config).fit(fit)
        training_seconds[name] = time.perf_counter() - tick
        models[name] = model
        _write_json(
            models_directory / f"{name}.json", model.to_dict()
        )
    tick = time.perf_counter()
    pca = EntityPcaRepresentation(width=64).fit(fit)
    training_seconds["matched_pca"] = time.perf_counter() - tick
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
    code_indices = {
        name: {
            role: _encode_indices(
                models[name], windows.histories, windows.graph
            )
            for role, windows in windows_by_role.items()
            if role != "calibration"
        }
        for name in ("discrete_complete", "discrete_p2p_only")
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
    ridge_curves = {}
    ridge_predictions: Dict[str, Dict[float, np.ndarray]] = {}
    for name in REPRESENTATION_NAMES:
        curve = []
        fitted = {}
        predictions_by_ridge = {}
        for ridge in RIDGES:
            probe = ReducedRankActionProbe(
                rank=32, ridge=ridge
            ).fit(
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
    attribution_predictions = {}
    action_sanity_predictions = {}
    for name in REPRESENTATION_NAMES:
        _, attribution_predictions[name] = _attribution_evidence(
            probes[name],
            query_tokens[name],
            transfer_queries,
            ownership,
        )
        _, action_sanity_predictions[name] = _action_sanity_evidence(
            probes[name],
            encoded[name]["transfer_evaluation"],
            windows_by_role["transfer_evaluation"],
            ownership,
        )
    diagnostics = {
        name: {
            role: models[name].diagnose(
                windows_by_role[role].histories,
                windows_by_role[role].graph,
                mask_seed=models[name].config.seed + 90_000,
            )
            for role in ("selection", "transfer_evaluation")
        }
        for name in NEURAL_NAMES
    }

    restored_models = {
        name: DiscreteJepaRepresentation.from_dict(
            models[name].to_dict()
        )
        for name in NEURAL_NAMES
    }
    restored_pca = EntityPcaRepresentation.from_dict(pca.to_dict())
    restoration_original_tokens = {}
    restoration_restored_tokens = {}
    restoration_original_indices = {}
    restoration_restored_indices = {}
    restoration_original_probe = {}
    restoration_restored_probe = {}
    for name in REPRESENTATION_NAMES:
        restoration_original_tokens[name] = {}
        restoration_restored_tokens[name] = {}
        restoration_original_probe[name] = {}
        restoration_restored_probe[name] = {}
        if name in ("discrete_complete", "discrete_p2p_only"):
            restoration_original_indices[name] = {}
            restoration_restored_indices[name] = {}
        restored_probe = ReducedRankActionProbe.from_dict(
            probes[name].to_dict()
        )
        for role in ("selection", "transfer_evaluation"):
            role_windows = windows_by_role[role]
            original = encoded[name][role]
            call = (
                restored_models[name].encode
                if name in NEURAL_NAMES
                else restored_pca.encode
            )
            replay = _encode_chunks(
                call, role_windows.histories, role_windows.graph
            )
            restoration_original_tokens[name][role] = original
            restoration_restored_tokens[name][role] = replay
            restoration_original_probe[name][role] = (
                probes[name].predict(
                    original,
                    role_windows.future_controls,
                    role_windows.future_actions,
                )
            )
            restoration_restored_probe[name][role] = (
                restored_probe.predict(
                    replay,
                    role_windows.future_controls,
                    role_windows.future_actions,
                )
            )
            if name in (
                "discrete_complete",
                "discrete_p2p_only",
            ):
                indices = _encode_indices(
                    restored_models[name],
                    role_windows.histories,
                    role_windows.graph,
                )
                restoration_original_indices[name][role] = (
                    code_indices[name][role]
                )
                restoration_restored_indices[name][role] = indices

    transfer = windows_by_role["transfer_evaluation"]

    bundle_path = (
        models_directory / "discrete_complete-inference.json.gz"
    )
    deployed_bundle_bytes = _write_inference_bundle(
        models["discrete_complete"],
        probes["discrete_complete"],
        bundle_path,
    )
    deployed_model, deployed_probe = _load_inference_bundle(
        bundle_path
    )
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
        for name in ("discrete_complete", "discrete_p2p_only"):
            evidence_arrays[f"indices__{name}__{role}"] = (
                code_indices[name][role].astype(np.int16)
            )
    for role in EVALUATED_ROLES:
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
        for role in ("selection", "transfer_evaluation"):
            evidence_arrays[
                f"restoration_original_tokens__{name}__{role}"
            ] = restoration_original_tokens[name][role].astype(
                np.float32
            )
            evidence_arrays[
                f"restoration_restored_tokens__{name}__{role}"
            ] = restoration_restored_tokens[name][role].astype(
                np.float32
            )
            evidence_arrays[
                f"restoration_original_probe__{name}__{role}"
            ] = restoration_original_probe[name][role].astype(
                np.float32
            )
            evidence_arrays[
                f"restoration_restored_probe__{name}__{role}"
            ] = restoration_restored_probe[name][role].astype(
                np.float32
            )
    for name in restoration_original_indices:
        for role in ("selection", "transfer_evaluation"):
            evidence_arrays[
                f"restoration_original_indices__{name}__{role}"
            ] = restoration_original_indices[name][role].astype(
                np.int16
            )
            evidence_arrays[
                f"restoration_restored_indices__{name}__{role}"
            ] = restoration_restored_indices[name][role].astype(
                np.int16
            )
    for name in NEURAL_NAMES:
        for role, diagnostic in diagnostics[name].items():
            for field in (
                "s2p_prediction",
                "p2s_prediction",
                "p2p_prediction",
                "target_patch",
                "target_semantic",
                "mask",
            ):
                value = getattr(diagnostic, field)
                evidence_arrays[
                    f"diagnostic_{field}__{name}__{role}"
                ] = (
                    value.astype(np.bool_)
                    if field == "mask"
                    else value.astype(np.float32)
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
        "kind": "discrete_jepa_assessment_evidence_v1",
        "interpretable": interpretable,
        "implementation_commit": implementation_commit,
        "source_corpus_sha256": prepared.source_corpus_sha256,
        "source_artifact_manifest_sha256": (
            prepared.source_artifact_manifest_sha256
        ),
        "preprocessing_protocol": prepared.preprocessing_protocol,
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
        "deployed_bundle_bytes": deployed_bundle_bytes,
        "latency": latency,
        "configs": {
            name: models[name].config.to_dict()
            for name in NEURAL_NAMES
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
        "kind": "discrete_jepa_telemetry_tracer_v1",
        "evidence_boundary": (
            "single-seed open-development paper-faithful telemetry "
            "translation; not exact paper reproduction or production alert"
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
            "primary_paper": "https://arxiv.org/abs/2506.14373",
            "official_repository": None,
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "started_unix": started,
            "completed_unix": time.time(),
        },
        "configurations": {
            name: models[name].config.to_dict()
            for name in NEURAL_NAMES
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
        "assessment": assessment,
        "source_sha256": source_hashes,
    }
    _write_json(building / "result.json", report)
    (building / "REPORT.md").write_text(_markdown_report(report))
    manifest = {
        "schema_version": 1,
        "kind": "discrete_jepa_artifact_manifest_v1",
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


EVALUATED_ROLES = (
    "selection",
    "iid_evaluation",
    "transfer_evaluation",
)


def _config(objective: str, steps: int) -> DiscreteJepaConfig:
    return DiscreteJepaConfig(
        objective=objective,
        steps=steps,
        warmup_steps=min(40, steps),
    )


def _write_schedules(
    directory: Path,
    fit: ActionConditionedWindows,
    config: DiscreteJepaConfig,
) -> None:
    anchor = PairBlockedAnchorSchedule(
        fit, seed=config.seed + 1
    )
    batches = [anchor.batch(step) for step in range(config.steps)]
    np.savez_compressed(
        directory / "anchor-schedule.npz",
        indices=np.stack([batch.indices for batch in batches]),
        arm_ids=np.stack([batch.arm_ids for batch in batches]),
        transition_indices=np.stack(
            [batch.transition_indices for batch in batches]
        ),
    )
    masks = DiscreteMaskSchedule(
        entity_count=config.semantic_token_count,
        patch_count=config.patch_count,
        seed=config.seed + 2,
    )
    np.savez_compressed(
        directory / "mask-schedule.npz",
        masks=np.stack(
            [
                masks.batch(
                    step=step, batch_size=len(batches[step].indices)
                )
                for step in range(config.steps)
            ]
        ),
    )


def _encode_representation(
    name: str,
    models: Mapping[str, DiscreteJepaRepresentation],
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


def _encode_indices(
    model: DiscreteJepaRepresentation,
    histories: np.ndarray,
    graph: Any,
) -> np.ndarray:
    values = []
    for start in range(0, len(histories), 128):
        encoded = model.encode(
            histories[start : start + 128], graph
        )
        if encoded.indices is None:
            raise RuntimeError(
                "Discrete-JEPA hard cell produced no indices"
            )
        values.append(encoded.indices)
    return np.concatenate(values, axis=0)


def _write_inference_bundle(
    model: DiscreteJepaRepresentation,
    probe: ReducedRankActionProbe,
    path: Path,
) -> int:
    payload = model.to_inference_dict()
    payload["probe"] = probe.to_dict()
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    path.write_bytes(gzip.compress(encoded, mtime=0))
    return len(path.read_bytes())


def _load_inference_bundle(
    path: Path,
) -> tuple[DiscreteJepaRepresentation, ReducedRankActionProbe]:
    payload = dict(
        json.loads(gzip.decompress(Path(path).read_bytes()).decode())
    )
    return (
        DiscreteJepaRepresentation.from_inference_dict(payload),
        ReducedRankActionProbe.from_dict(dict(payload["probe"])),
    )


def _latency(
    model: DiscreteJepaRepresentation,
    probe: ReducedRankActionProbe,
    windows: ActionConditionedWindows,
    *,
    repetitions: int,
) -> tuple[Mapping[str, Any], np.ndarray]:
    if repetitions < 1:
        raise ValueError(
            "Discrete-JEPA latency repetitions must be positive"
        )

    def call() -> None:
        tokens = model.encode(
            windows.histories[:1], windows.graph
        ).tokens
        probe.predict(
            tokens,
            windows.future_controls[:1],
            windows.future_actions[:1],
        )

    call()
    samples = []
    for _ in range(repetitions):
        tick = time.perf_counter_ns()
        call()
        samples.append((time.perf_counter_ns() - tick) / 1e6)
    values = np.asarray(samples, dtype=np.float64)
    return {
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.quantile(values, 0.95)),
        "repetitions": int(repetitions),
    }, values


def _role_metadata(
    windows: ActionConditionedWindows,
) -> Mapping[str, Any]:
    return {
        "trajectory_ids": list(windows.trajectory_ids),
        "pair_ids": list(windows.matched_pair_ids),
        "transition_indices": (
            windows.transition_indices.astype(int).tolist()
        ),
    }


def _query_metadata(
    queries: PreparedAttributionQueries,
) -> Mapping[str, Any]:
    return {
        "query_ids": list(queries.query_ids),
        "candidate_ids": list(queries.candidate_ids),
        "candidate_action_kinds": list(
            queries.candidate_action_kinds
        ),
        "candidate_target_entities": list(
            queries.candidate_target_entities
        ),
        "expected_action_kinds": list(
            queries.expected_action_kinds
        ),
        "expected_target_entities": list(
            queries.expected_target_entities
        ),
        "expected_variant_ids": list(
            queries.expected_variant_ids
        ),
    }


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
                "frozen Discrete-JEPA run requires clean commit"
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
            value, indent=2, sort_keys=True, allow_nan=False
        )
        + "\n"
    )


def _markdown_report(report: Mapping[str, Any]) -> str:
    assessment = dict(report["assessment"])
    return "\n".join(
        (
            "# Discrete-JEPA telemetry tracer v1",
            "",
            f"- Decision: `{assessment['decision']}`",
            f"- Passed: `{assessment['passed']}`",
            f"- Interpretable: `{report['interpretable']}`",
            f"- Implementation: `{report['implementation_commit']}`",
            "",
            "See `result.json`, `assessment.json`, and `evidence.npz` "
            "for complete retained evidence.",
            "",
        )
    )


def main(arguments: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=FROZEN_CACHE)
    parser.add_argument("--output", type=Path, default=FROZEN_OUTPUT)
    parser.add_argument("--steps", type=int, default=FROZEN_STEPS)
    parser.add_argument(
        "--latency-repetitions", type=int, default=100
    )
    parser.add_argument(
        "--allow-noninterpretable-smoke", action="store_true"
    )
    args = parser.parse_args(arguments)
    result = run_experiment(
        cache_directory=args.cache,
        output_directory=args.output,
        steps=args.steps,
        latency_repetitions=args.latency_repetitions,
        allow_noninterpretable_smoke=(
            args.allow_noninterpretable_smoke
        ),
    )
    print(
        json.dumps(
            verify_stored_assessment(result),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
