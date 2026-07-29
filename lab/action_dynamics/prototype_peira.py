#!/usr/bin/env python3
"""Retained runner for the frozen PEIRA telemetry tracer."""

import argparse
import gzip
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from lab.action_dynamics.prototype_complete_lejepa import (
        _action_sanity_evidence,
        _attribution_evidence,
        _forecast_scores,
        _transfer_queries,
    )
    from lab.action_dynamics.prototype_peira_assessor import (
        EVALUATED_ROLES,
        PEIRA_NAMES,
        REPRESENTATION_NAMES,
        assess_stored_bundle,
        peira_diagnostics,
        verify_stored_assessment,
    )
except ModuleNotFoundError:
    from prototype_complete_lejepa import (
        _action_sanity_evidence,
        _attribution_evidence,
        _forecast_scores,
        _transfer_queries,
    )
    from prototype_peira_assessor import (
        EVALUATED_ROLES,
        PEIRA_NAMES,
        REPRESENTATION_NAMES,
        assess_stored_bundle,
        peira_diagnostics,
        verify_stored_assessment,
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
from quantis_core.edge_dynamics.data import (
    PreparedAttributionQueries,
    load_edge_dynamics_cache,
    partition_worker_topology,
)
from quantis_core.edge_dynamics.models import (
    ContractiveLowRankDynamics,
    LowRankConfig,
)
from quantis_core.edge_dynamics.peira import (
    PeiraConfig,
    PeiraRepresentation,
    PeiraSchedule,
)


FROZEN_CACHE = Path(
    "artifacts/action-dynamics/edge-preprocessing-v1/"
    "eb54271132f88c9a431b01e786ea66279a563776434cca2290e47e6b7ae9b3ff"
)
FROZEN_PRIOR_CONTROL = Path(
    "artifacts/action-dynamics/prototype-complete-lejepa-v1"
)
FROZEN_OUTPUT = Path("artifacts/action-dynamics/prototype-peira-v1")
FROZEN_STEPS = 1600
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
RIDGES = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
IMPLEMENTATION_SOURCE_PATHS = (
    "lab/action_dynamics/prototype_peira.py",
    "lab/action_dynamics/prototype_peira_assessor.py",
    "lab/action_dynamics/prototype_complete_lejepa.py",
    "lab/action_dynamics/prototype_lenepa_jepa_assessor.py",
    "src/quantis_core/edge_dynamics/peira.py",
    "src/quantis_core/edge_dynamics/lenepa_jepa.py",
    "src/quantis_core/edge_dynamics/complete_lejepa.py",
    "src/quantis_core/edge_dynamics/action_conditioned_jepa.py",
    "src/quantis_core/edge_dynamics/models.py",
    "src/quantis_core/edge_dynamics/data.py",
    "src/quantis_core/edge_dynamics/__init__.py",
    "src/quantis_core/__init__.py",
    "src/quantis_core/scenarios.py",
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
    "tests/test_peira.py",
    "docs/research/peira-primary-source-notes.md",
    "docs/specs/peira-telemetry-tracer-v1.md",
    "docs/wayfinding/jepa-implementation-program/026-test-peira.md",
)


def run_experiment(
    *,
    cache_directory: Path,
    prior_control_directory: Path,
    output_directory: Path,
    steps: int,
    latency_repetitions: int,
    allow_noninterpretable_smoke: bool,
) -> Path:
    """Fit, independently assess, and atomically publish PEIRA."""

    cache = Path(cache_directory).resolve()
    prior = Path(prior_control_directory).resolve()
    output = Path(output_directory).resolve()
    building = output.with_name(output.name + ".building")
    if output.exists() or building.exists():
        raise FileExistsError("PEIRA refuses an existing output")
    frozen_paths = (
        cache == (Path.cwd() / FROZEN_CACHE).resolve()
        and prior == (Path.cwd() / FROZEN_PRIOR_CONTROL).resolve()
        and output == (Path.cwd() / FROZEN_OUTPUT).resolve()
        and steps == FROZEN_STEPS
        and latency_repetitions == 100
    )
    if not frozen_paths and not allow_noninterpretable_smoke:
        raise ValueError("non-frozen PEIRA runs require smoke permission")
    if (
        not frozen_paths
        and output == (Path.cwd() / FROZEN_OUTPUT).resolve()
    ):
        raise ValueError("PEIRA smoke cannot use the frozen output")
    prior_manifest_sha256 = _file_sha256(
        prior / "artifact-manifest.json"
    )
    if prior_manifest_sha256 != FROZEN_PRIOR_MANIFEST_SHA256:
        raise ValueError("PEIRA prior-control artifact identity differs")
    prior_manifest = _read_json(prior / "artifact-manifest.json")
    prior_declared_hashes = dict(prior_manifest["sha256"])
    for name in ("complete_lejepa", "masked_autoencoder"):
        relative = f"models/{name}.json"
        declared = str(prior_declared_hashes[relative])
        if (
            declared != FROZEN_PRIOR_MODEL_SHA256[name]
            or _file_sha256(prior / relative) != declared
        ):
            raise ValueError("PEIRA prior-control model is corrupted")
    implementation_commit = _git_head()
    source_hashes = _source_identity(require_clean=frozen_paths)
    prepared = load_edge_dynamics_cache(cache)
    source_is_frozen = bool(
        prepared.source_corpus_sha256 == FROZEN_SOURCE_CORPUS_SHA256
        and prepared.source_artifact_manifest_sha256
        == FROZEN_SOURCE_ARTIFACT_MANIFEST_SHA256
        and prepared.preprocessing_protocol
        == FROZEN_PREPROCESSING_PROTOCOL
    )
    if frozen_paths and not source_is_frozen:
        raise ValueError("frozen PEIRA source identity differs")
    interpretable = bool(frozen_paths and source_is_frozen)

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
        partition.held_out_normalized_value
        for partition in partitions.values()
    }
    if len(held_values) != 1:
        raise ValueError("PEIRA held topology differs by role")
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
    varying_entities = np.any(
        (np.ptp(fit.histories, axis=(0, 1)) > 1e-9) & ownership,
        axis=1,
    )
    base_config = _config("aligned_peira", steps)
    _write_schedules(building, fit, base_config)

    peira_models: Dict[str, PeiraRepresentation] = {}
    training_seconds: Dict[str, float] = {}
    for name in PEIRA_NAMES:
        tick = time.perf_counter()
        model = PeiraRepresentation(_config(name, steps)).fit(fit)
        training_seconds[name] = time.perf_counter() - tick
        peira_models[name] = model
        _write_json(models_directory / f"{name}.json", model.to_dict())
    prior_models = {
        name: CompleteLejepaRepresentation.from_dict(
            _read_json(prior / "models" / f"{name}.json")
        )
        for name in ("complete_lejepa", "masked_autoencoder")
    }
    for name, model in prior_models.items():
        shutil.copy2(
            prior / "models" / f"{name}.json",
            models_directory / f"{name}.json",
        )
    pca = EntityPcaRepresentation(width=64).fit(fit)
    _write_json(models_directory / "matched_pca.json", pca.to_dict())

    models: Dict[str, Any] = {**peira_models, **prior_models}
    encoded = {
        name: {
            role: _encode_chunks(
                pca.encode if name == "matched_pca" else models[name].encode,
                windows.histories,
                windows.graph,
            )
            for role, windows in windows_by_role.items()
            if role != "calibration"
        }
        for name in REPRESENTATION_NAMES
    }
    raw = ContractiveLowRankDynamics(LowRankConfig(rank=32)).fit(fit)
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
        role: _forecast_scores(prediction, windows_by_role[role])
        for role, prediction in raw_predictions.items()
    }

    probes: Dict[str, ReducedRankActionProbe] = {}
    selected_ridges: Dict[str, float] = {}
    selection_safety_failed: Dict[str, bool] = {}
    ridge_curves: Dict[str, Any] = {}
    ridge_predictions: Dict[str, Dict[float, np.ndarray]] = {}
    selection = windows_by_role["selection"]
    for name in REPRESENTATION_NAMES:
        rows = []
        fitted = {}
        by_ridge = {}
        for ridge in RIDGES:
            probe = ReducedRankActionProbe(rank=32, ridge=ridge).fit(
                encoded[name]["fit"],
                fit.future_controls,
                fit.future_actions,
                fit.future_states,
            )
            prediction = probe.predict(
                encoded[name]["selection"],
                selection.future_controls,
                selection.future_actions,
            )
            scores = _forecast_scores(prediction, selection)
            raw_safe = bool(
                scores["overall_mse"]
                <= 1.05 * raw_scores["selection"]["overall_mse"]
                and scores["action_overlap_mse"]
                <= 1.05
                * raw_scores["selection"]["action_overlap_mse"]
            )
            rows.append({"ridge": ridge, "raw_safe": raw_safe, **scores})
            fitted[ridge] = probe
            by_ridge[ridge] = prediction
        eligible = [row for row in rows if row["raw_safe"]]
        chosen = min(
            eligible or rows,
            key=lambda row: (
                row["downstream_effect_mse"],
                row["ridge"],
            ),
        )
        ridge = float(chosen["ridge"])
        probes[name] = fitted[ridge]
        selected_ridges[name] = ridge
        selection_safety_failed[name] = not bool(eligible)
        ridge_curves[name] = rows
        ridge_predictions[name] = by_ridge
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
        name: _encode_chunks(
            pca.encode if name == "matched_pca" else models[name].encode,
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

    diagnostic_views: Dict[str, Dict[str, np.ndarray]] = {
        name: {
            role: _diagnose_chunks(
                peira_models[name],
                windows_by_role[role].histories,
                windows_by_role[role].graph,
                step=steps - 1,
            )
            for role in ("selection", "transfer_evaluation")
        }
        for name in PEIRA_NAMES
    }
    diagnostics = {
        name: {
            role: peira_diagnostics(
                diagnostic_views[name][role],
                encoded[name][role],
                ownership,
                varying_entities,
                regularization=peira_models[name].config.regularization,
            )
            for role in ("selection", "transfer_evaluation")
        }
        for name in PEIRA_NAMES
    }

    restored_models = {
        name: PeiraRepresentation.from_dict(model.to_dict())
        for name, model in peira_models.items()
    }
    restored_prior = {
        name: CompleteLejepaRepresentation.from_dict(model.to_dict())
        for name, model in prior_models.items()
    }
    restored_pca = EntityPcaRepresentation.from_dict(pca.to_dict())
    restoration: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    for name in REPRESENTATION_NAMES:
        restored_probe = ReducedRankActionProbe.from_dict(
            probes[name].to_dict()
        )
        restoration[name] = {}
        for role in ("selection", "transfer_evaluation"):
            role_windows = windows_by_role[role]
            if name in PEIRA_NAMES:
                call = restored_models[name].encode
            elif name in restored_prior:
                call = restored_prior[name].encode
            else:
                call = restored_pca.encode
            replay = _encode_chunks(
                call, role_windows.histories, role_windows.graph
            )
            restoration[name][role] = {
                "original_tokens": encoded[name][role],
                "restored_tokens": replay,
                "original_probe": probes[name].predict(
                    encoded[name][role],
                    role_windows.future_controls,
                    role_windows.future_actions,
                ),
                "restored_probe": restored_probe.predict(
                    replay,
                    role_windows.future_controls,
                    role_windows.future_actions,
                ),
            }

    bundle_path = models_directory / "aligned_peira-inference.json.gz"
    deployed_bundle_bytes = _write_inference_bundle(
        peira_models["aligned_peira"],
        probes["aligned_peira"],
        bundle_path,
    )
    deployed_model, deployed_probe = _load_inference_bundle(bundle_path)
    latency, latency_samples = _latency(
        deployed_model,
        deployed_probe,
        windows_by_role["transfer_evaluation"],
        repetitions=latency_repetitions,
    )
    parameter_counts = {
        name: {
            "training": peira_models[name].training_parameter_count,
            "inference": peira_models[name].inference_parameter_count,
        }
        for name in PEIRA_NAMES
    }

    evidence: Dict[str, np.ndarray] = {}
    for role in ("fit",) + EVALUATED_ROLES:
        windows = windows_by_role[role]
        evidence[f"histories__{role}"] = windows.histories.astype(
            np.float32
        )
        evidence[f"target__{role}"] = windows.future_states.astype(
            np.float32
        )
        evidence[f"controls__{role}"] = windows.future_controls.astype(
            np.float32
        )
        evidence[f"actions__{role}"] = windows.future_actions.astype(
            np.float32
        )
        for name in REPRESENTATION_NAMES:
            evidence[f"representation__{name}__{role}"] = encoded[
                name
            ][role].astype(np.float32)
    for role in EVALUATED_ROLES:
        evidence[f"raw_prediction__{role}"] = raw_predictions[role].astype(
            np.float32
        )
        for name in REPRESENTATION_NAMES:
            evidence[f"prediction__{name}__{role}"] = predictions[name][
                role
            ].astype(np.float32)
    for name in REPRESENTATION_NAMES:
        for position, ridge in enumerate(RIDGES):
            evidence[
                f"ridge_prediction__{name}__{position}"
            ] = ridge_predictions[name][ridge].astype(np.float32)
        evidence[f"attribution_prediction__{name}"] = (
            attribution_predictions[name].astype(np.float32)
        )
        for variant, values in action_sanity_predictions[name].items():
            evidence[f"action_sanity__{name}__{variant}"] = values.astype(
                np.float32
            )
        for role in ("selection", "transfer_evaluation"):
            for field, values in restoration[name][role].items():
                evidence[
                    f"restoration_{field}__{name}__{role}"
                ] = values.astype(np.float32)
    for name in PEIRA_NAMES:
        for field, values in peira_models[name].training_evidence.items():
            evidence[f"training__{name}__{field}"] = values
        predictor, inverse = peira_models[name].final_operators
        evidence[f"final_predictor__{name}"] = predictor
        evidence[f"final_inverse__{name}"] = inverse
        for role, values in diagnostic_views[name].items():
            evidence[f"diagnostic_views__{name}__{role}"] = (
                values.astype(np.float32)
            )
    evidence["query_histories"] = transfer_queries.histories.astype(
        np.float32
    )
    evidence["query_future_controls"] = (
        transfer_queries.future_controls.astype(np.float32)
    )
    evidence["query_observed_future"] = (
        transfer_queries.observed_future.astype(np.float32)
    )
    evidence["query_candidate_actions"] = (
        transfer_queries.candidate_actions.astype(np.float32)
    )
    evidence["latency_samples_ms"] = latency_samples
    np.savez_compressed(building / "evidence.npz", **evidence)

    prior_model_sha256 = dict(FROZEN_PRIOR_MODEL_SHA256)
    metadata = {
        "schema_version": 1,
        "kind": "peira_assessment_evidence_v1",
        "interpretable": interpretable,
        "implementation_commit": implementation_commit,
        "source_corpus_sha256": prepared.source_corpus_sha256,
        "source_artifact_manifest_sha256": (
            prepared.source_artifact_manifest_sha256
        ),
        "preprocessing_protocol": prepared.preprocessing_protocol,
        "prior_control_manifest_sha256": prior_manifest_sha256,
        "prior_model_sha256": prior_model_sha256,
        "source_sha256": source_hashes,
        "graph": fit.graph.to_dict(),
        "entity_names": list(fit.entity_names),
        "state_feature_names": list(fit.state_feature_names),
        "control_feature_names": list(fit.control_feature_names),
        "action_feature_names": list(fit.action_feature_names),
        "ownership_mask": ownership.astype(int).tolist(),
        "varying_entity_mask": varying_entities.astype(int).tolist(),
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
            name: peira_models[name].config.to_dict()
            for name in PEIRA_NAMES
        },
        "diagnostics": diagnostics,
    }
    _write_json(building / "evidence-metadata.json", metadata)
    shutil.copy2(
        prior / "artifact-manifest.json",
        building / "prior-control-manifest.json",
    )
    for source_name in IMPLEMENTATION_SOURCE_PATHS:
        source = Path(source_name)
        destination = reproduction_directory / source_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    preflight = _run_copied_assessor(building)
    source_snapshot_sha256 = hashlib.sha256(
        json.dumps(
            source_hashes,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    _write_json(
        building / "isolated-assessor-receipt.json",
        {
            "schema_version": 1,
            "kind": "peira_isolated_assessor_receipt_v1",
            "returncode": 0,
            "assessor_sha256": source_hashes[
                "lab/action_dynamics/prototype_peira_assessor.py"
            ],
            "source_snapshot_sha256": source_snapshot_sha256,
            "preflight_assessment_sha256": hashlib.sha256(
                json.dumps(
                    preflight,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode()
            ).hexdigest(),
        },
    )
    assessment = dict(assess_stored_bundle(building))
    _write_json(building / "assessment.json", assessment)
    report = {
        "schema_version": 1,
        "kind": "peira_telemetry_tracer_v1",
        "evidence_boundary": (
            "single-seed open-development paper-faithful telemetry "
            "translation; no author code and not a production alert"
        ),
        "interpretable": interpretable,
        "implementation_commit": implementation_commit,
        "source": {
            "cache_directory": str(cache),
            "prior_control_directory": str(prior),
            "source_corpus_sha256": prepared.source_corpus_sha256,
            "source_artifact_manifest_sha256": (
                prepared.source_artifact_manifest_sha256
            ),
            "prior_control_manifest_sha256": prior_manifest_sha256,
            "preprocessing_protocol": prepared.preprocessing_protocol,
            "primary_paper": "https://arxiv.org/abs/2605.17671",
            "official_repository": None,
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "started_unix": started,
            "completed_unix": time.time(),
        },
        "training_seconds": training_seconds,
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
        "kind": "peira_artifact_manifest_v1",
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


def _config(objective: str, steps: int) -> PeiraConfig:
    return PeiraConfig(objective=objective, steps=steps)


def _write_schedules(
    directory: Path,
    fit: ActionConditionedWindows,
    config: PeiraConfig,
) -> None:
    anchors = PairBlockedAnchorSchedule(fit, seed=config.anchor_seed)
    ownership = fit_owned_feature_mask(fit)
    views = TelemetryViewSchedule(
        graph=fit.graph,
        ownership_mask=ownership,
        varying_entity_mask=np.any(
            (np.ptp(fit.histories, axis=(0, 1)) > 1e-9) & ownership,
            axis=1,
        ),
        seed=config.view_seed,
    )
    schedule = PeiraSchedule(
        steps=config.steps,
        eta_initial=config.eta_initial,
        eta_final=config.eta_final,
        derangement_seed=config.derangement_seed,
    )
    anchor_batches = [anchors.batch(step) for step in range(config.steps)]
    view_batches = [
        views.batch(fit.histories[:1], step=step)
        for step in range(config.steps)
    ]
    np.savez_compressed(
        directory / "schedule.npz",
        anchor_indices=np.stack(
            [batch.indices for batch in anchor_batches]
        ),
        anchor_arm_ids=np.stack(
            [batch.arm_ids for batch in anchor_batches]
        ),
        anchor_transitions=np.stack(
            [batch.transition_indices for batch in anchor_batches]
        ),
        view_visible=np.stack(
            [batch.visible_tokens[:2, 0] for batch in view_batches]
        ),
        view_present=np.stack(
            [batch.present_tokens[:2, 0] for batch in view_batches]
        ),
        derangements=np.stack(
            [
                schedule.derangement(step, len(anchors.pair_ids))
                for step in range(config.steps)
            ]
        ),
        eta=np.asarray(
            [schedule.eta(step) for step in range(config.steps)]
        ),
        learning_rate=np.asarray(
            [
                config.learning_rate_at(step)
                for step in range(config.steps)
            ]
        ),
        clip_enabled=np.asarray(
            [config.clip_enabled_at(step) for step in range(config.steps)]
        ),
    )


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
    chunks = [
        model.diagnose_views(
            histories[start : start + 128], graph, step=step
        )
        for start in range(0, len(histories), 128)
    ]
    return np.concatenate(chunks, axis=1)


def _write_inference_bundle(
    model: PeiraRepresentation,
    probe: ReducedRankActionProbe,
    path: Path,
) -> int:
    payload = {
        "schema_version": 1,
        "kind": "peira_forecast_inference_bundle_v1",
        "representation": model.to_inference_dict(),
        "probe": probe.to_dict(),
    }
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
) -> Tuple[PeiraRepresentation, ReducedRankActionProbe]:
    payload = dict(
        json.loads(gzip.decompress(Path(path).read_bytes()).decode())
    )
    if (
        set(payload)
        != {"schema_version", "kind", "representation", "probe"}
        or payload.get("schema_version") != 1
        or payload.get("kind")
        != "peira_forecast_inference_bundle_v1"
    ):
        raise ValueError("unsupported PEIRA deployment bundle")
    return (
        PeiraRepresentation.from_inference_dict(
            dict(payload["representation"])
        ),
        ReducedRankActionProbe.from_dict(dict(payload["probe"])),
    )


def _latency(
    model: PeiraRepresentation,
    probe: ReducedRankActionProbe,
    windows: ActionConditionedWindows,
    *,
    repetitions: int,
) -> Tuple[Mapping[str, Any], np.ndarray]:
    if repetitions < 1:
        raise ValueError("PEIRA latency repetitions must be positive")

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
        "repetitions": repetitions,
    }, values


def _role_metadata(
    windows: ActionConditionedWindows,
) -> Mapping[str, Any]:
    return {
        "pair_ids": sorted(set(windows.matched_pair_ids)),
        "trajectory_ids": sorted(set(windows.trajectory_ids)),
        "matched_pair_ids": list(windows.matched_pair_ids),
        "row_trajectory_ids": list(windows.trajectory_ids),
        "transition_indices": windows.transition_indices.tolist(),
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
        "expected_variant_ids": list(queries.expected_variant_ids),
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
            raise RuntimeError("frozen PEIRA run requires clean commit")
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
            "# PEIRA telemetry tracer v1",
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
    parser.add_argument(
        "--prior-control", type=Path, default=FROZEN_PRIOR_CONTROL
    )
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
        prior_control_directory=args.prior_control,
        output_directory=args.output,
        steps=args.steps,
        latency_repetitions=args.latency_repetitions,
        allow_noninterpretable_smoke=args.allow_noninterpretable_smoke,
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
