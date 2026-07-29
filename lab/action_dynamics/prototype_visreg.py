#!/usr/bin/env python3
"""Retained runner for the frozen VISReg telemetry tracer."""

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
        _state_probe,
    )
    from lab.action_dynamics.prototype_visreg_assessor import (
        EVALUATED_ROLES,
        PRIOR_NAMES,
        REPRESENTATION_NAMES,
        VISREG_NAMES,
        assess_stored_bundle,
        collapse_curve,
        fixed_directions,
        verify_stored_assessment,
        visreg_diagnostics,
    )
except ModuleNotFoundError:
    from prototype_complete_lejepa import (
        _action_sanity_evidence,
        _attribution_evidence,
        _forecast_scores,
        _transfer_queries,
    )
    from prototype_peira_assessor import _state_probe
    from prototype_visreg_assessor import (
        EVALUATED_ROLES,
        PRIOR_NAMES,
        REPRESENTATION_NAMES,
        VISREG_NAMES,
        assess_stored_bundle,
        collapse_curve,
        fixed_directions,
        verify_stored_assessment,
        visreg_diagnostics,
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
from quantis_core.edge_dynamics.visreg import (
    VisregConfig,
    VisregRepresentation,
)


FROZEN_CACHE = Path(
    "artifacts/action-dynamics/edge-preprocessing-v1/"
    "eb54271132f88c9a431b01e786ea66279a563776434cca2290e47e6b7ae9b3ff"
)
FROZEN_PRIOR_CONTROL = Path(
    "artifacts/action-dynamics/prototype-complete-lejepa-v1"
)
FROZEN_OUTPUT = Path("artifacts/action-dynamics/prototype-visreg-v1")
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
    "invariance_only": (
        "cbadbda2c8e4f0357ef135224b827a0d75e7a06f84821dc487df2f995fba4723"
    ),
    "sigreg_only": (
        "3559d948fe0801f1b2a0d816f50e6c0269a9a6209a72fe63eec8ac88e450745e"
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
    "lab/action_dynamics/prototype_visreg.py",
    "lab/action_dynamics/prototype_visreg_assessor.py",
    "lab/action_dynamics/prototype_complete_lejepa.py",
    "lab/action_dynamics/prototype_peira_assessor.py",
    "lab/action_dynamics/prototype_lenepa_jepa_assessor.py",
    "src/quantis_core/edge_dynamics/visreg.py",
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
    "tests/test_visreg.py",
    "docs/research/visreg-primary-source-notes.md",
    "docs/specs/visreg-telemetry-tracer-v1.md",
    "docs/wayfinding/jepa-implementation-program/027-test-visreg.md",
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
    """Fit, independently assess, and atomically publish VISReg."""

    cache = Path(cache_directory).resolve()
    prior = Path(prior_control_directory).resolve()
    output = Path(output_directory).resolve()
    building = output.with_name(output.name + ".building")
    if output.exists() or building.exists():
        raise FileExistsError("VISReg refuses an existing output")
    frozen_paths = (
        cache == (Path.cwd() / FROZEN_CACHE).resolve()
        and prior == (Path.cwd() / FROZEN_PRIOR_CONTROL).resolve()
        and output == (Path.cwd() / FROZEN_OUTPUT).resolve()
        and steps == FROZEN_STEPS
        and latency_repetitions == 100
    )
    if not frozen_paths and not allow_noninterpretable_smoke:
        raise ValueError("non-frozen VISReg runs require smoke permission")
    if (
        not frozen_paths
        and output == (Path.cwd() / FROZEN_OUTPUT).resolve()
    ):
        raise ValueError("VISReg smoke cannot use the frozen output")
    prior_manifest_sha256 = _file_sha256(
        prior / "artifact-manifest.json"
    )
    if prior_manifest_sha256 != FROZEN_PRIOR_MANIFEST_SHA256:
        raise ValueError("VISReg prior-control artifact identity differs")
    prior_manifest = _read_json(prior / "artifact-manifest.json")
    declared = dict(prior_manifest["sha256"])
    for name in PRIOR_NAMES:
        relative = f"models/{name}.json"
        if (
            declared.get(relative) != FROZEN_PRIOR_MODEL_SHA256[name]
            or _file_sha256(prior / relative)
            != FROZEN_PRIOR_MODEL_SHA256[name]
        ):
            raise ValueError("VISReg prior-control model is corrupted")
    implementation_commit = _git_head()
    source_hashes = _source_identity(require_clean=frozen_paths)
    prepared = load_edge_dynamics_cache(cache)
    source_corpus_sha256 = prepared.source_corpus_sha256
    source_artifact_manifest_sha256 = (
        prepared.source_artifact_manifest_sha256
    )
    preprocessing_protocol = prepared.preprocessing_protocol
    attribution_queries = prepared.attribution_queries
    source_is_frozen = bool(
        source_corpus_sha256 == FROZEN_SOURCE_CORPUS_SHA256
        and source_artifact_manifest_sha256
        == FROZEN_SOURCE_ARTIFACT_MANIFEST_SHA256
        and preprocessing_protocol == FROZEN_PREPROCESSING_PROTOCOL
    )
    if frozen_paths and not source_is_frozen:
        raise ValueError("frozen VISReg source identity differs")
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
        raise ValueError("VISReg held topology differs by role")
    held_value = next(iter(held_values))
    windows_by_role = {
        "fit": partitions["fit"].in_distribution,
        "selection": partitions["selection"].in_distribution,
        "calibration": partitions["calibration"].in_distribution,
        "iid_evaluation": partitions["evaluation"].in_distribution,
        "transfer_evaluation": partitions["evaluation"].held_out,
    }
    role_metadata = {
        role: _role_metadata(windows)
        for role, windows in windows_by_role.items()
    }
    windows_by_role.pop("calibration")
    del prepared
    del partitions
    fit = windows_by_role["fit"]
    ownership = fit_owned_feature_mask(fit)
    varying = np.any(
        (np.ptp(fit.histories, axis=(0, 1)) > 1e-9) & ownership,
        axis=1,
    )
    base_config = _config("detached_visreg", steps)
    _write_schedules(building, fit, base_config)

    visreg_models: Dict[str, VisregRepresentation] = {}
    training_seconds: Dict[str, float] = {}
    for name in VISREG_NAMES:
        tick = time.perf_counter()
        model = VisregRepresentation(_config(name, steps)).fit(fit)
        training_seconds[name] = time.perf_counter() - tick
        visreg_models[name] = model
        _write_json(models_directory / f"{name}.json", model.to_dict())
    detached_evidence = visreg_models[
        "detached_visreg"
    ]._training_evidence
    no_detach_evidence = visreg_models[
        "no_detach_visreg"
    ]._training_evidence
    if detached_evidence is None or no_detach_evidence is None:
        raise RuntimeError("VISReg training evidence is missing")
    if not np.array_equal(
        detached_evidence["directions"],
        no_detach_evidence["directions"],
    ):
        raise RuntimeError("VISReg cells consumed different directions")

    prior_models = {
        name: CompleteLejepaRepresentation.from_dict(
            _read_json(prior / "models" / f"{name}.json")
        )
        for name in PRIOR_NAMES
    }
    for name in PRIOR_NAMES:
        shutil.copy2(
            prior / "models" / f"{name}.json",
            models_directory / f"{name}.json",
        )
    pca = EntityPcaRepresentation(width=64).fit(fit)
    _write_json(models_directory / "matched_pca.json", pca.to_dict())
    models: Dict[str, Any] = {**visreg_models, **prior_models}
    encoded = {
        name: {
            role: _encode_chunks(
                pca.encode if name == "matched_pca" else models[name].encode,
                windows.histories,
                windows.graph,
            ).astype(np.float32)
            for role, windows in windows_by_role.items()
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
        attribution_queries,
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
        name: {}
        for name in VISREG_NAMES
    }
    for name in VISREG_NAMES:
        for role in ("selection", "transfer_evaluation"):
            role_windows = windows_by_role[role]
            anchor = PairBlockedAnchorSchedule(
                role_windows,
                seed=visreg_models[name].config.anchor_seed,
            ).batch(steps - 1)
            diagnostic_views[name][role] = visreg_models[
                name
            ].diagnose_views(
                role_windows.histories[anchor.indices],
                role_windows.graph,
                step=steps - 1,
            )
    diagnostic_directions = fixed_directions(
        width=64, projection_count=1024, seed=6509
    )
    collapse_base, collapse_directions = _collapse_inputs()
    diagnostics = {
        name: {
            role: visreg_diagnostics(
                diagnostic_views[name][role],
                encoded[name][role],
                ownership,
                varying,
                diagnostic_directions,
            )
            for role in ("selection", "transfer_evaluation")
        }
        for name in VISREG_NAMES
    }
    curve = collapse_curve(collapse_base, collapse_directions)
    restored_models = {
        name: VisregRepresentation.from_dict(model.to_dict())
        for name, model in visreg_models.items()
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
        for role in EVALUATED_ROLES:
            role_windows = windows_by_role[role]
            if name in VISREG_NAMES:
                original_call = visreg_models[name].encode
                restored_call = restored_models[name].encode
            elif name in restored_prior:
                original_call = prior_models[name].encode
                restored_call = restored_prior[name].encode
            else:
                original_call = pca.encode
                restored_call = restored_pca.encode
            original = _encode_chunks(
                original_call,
                role_windows.histories,
                role_windows.graph,
            )
            replay = _encode_chunks(
                restored_call,
                role_windows.histories,
                role_windows.graph,
            )
            restoration[name][role] = {
                "original_tokens": original,
                "restored_tokens": replay,
                "original_probe": probes[name].predict(
                    original,
                    role_windows.future_controls,
                    role_windows.future_actions,
                ),
                "restored_probe": restored_probe.predict(
                    replay,
                    role_windows.future_controls,
                    role_windows.future_actions,
                ),
            }
    state_probes = {
        name: {
            role: _state_probe(
                encoded[name]["fit"],
                fit,
                restoration[name][role]["original_tokens"],
                windows_by_role[role],
                ownership,
            )
            for role in EVALUATED_ROLES
        }
        for name in REPRESENTATION_NAMES
    }

    fit_mode_anchor = PairBlockedAnchorSchedule(
        fit, seed=base_config.anchor_seed
    ).batch(steps - 1)
    mode_public_tokens = {
        name: visreg_models[name]
        .encode(fit.histories[fit_mode_anchor.indices], fit.graph)
        .tokens
        for name in VISREG_NAMES
    }
    gradients = [
        np.asarray(
            (
                detached_evidence
                if name == "detached_visreg"
                else no_detach_evidence
            )["regularizer_gradient_step0"],
            dtype=np.float32,
        )
        for name in VISREG_NAMES
    ]
    mode_enforcement = {
        "gradient_sha256": [
            hashlib.sha256(value.tobytes(order="C")).hexdigest()
            for value in gradients
        ],
        "gradient_max_abs": float(
            np.max(np.abs(gradients[0] - gradients[1]))
        ),
        "network_sha256": [
            visreg_models[name].network_sha256
            for name in VISREG_NAMES
        ],
        "projector_sha256": [
            visreg_models[name].projector_sha256
            for name in VISREG_NAMES
        ],
        "public_token_max_abs": float(
            np.max(
                np.abs(
                    mode_public_tokens[VISREG_NAMES[0]]
                    - mode_public_tokens[VISREG_NAMES[1]]
                )
            )
        ),
    }

    bundle_path = (
        models_directory / "detached_visreg-inference.json.gz"
    )
    deployed_bundle_bytes = _write_inference_bundle(
        visreg_models["detached_visreg"],
        probes["detached_visreg"],
        bundle_path,
    )
    deployed_model, deployed_probe = _load_inference_bundle(bundle_path)
    deployment_reference_tokens = _encode_chunks(
        visreg_models["detached_visreg"].encode,
        windows_by_role["transfer_evaluation"].histories,
        windows_by_role["transfer_evaluation"].graph,
    )
    deployment_reference_prediction = probes[
        "detached_visreg"
    ].predict(
        deployment_reference_tokens,
        windows_by_role["transfer_evaluation"].future_controls,
        windows_by_role["transfer_evaluation"].future_actions,
    )
    latency, latency_samples = _latency(
        deployed_model,
        deployed_probe,
        windows_by_role["transfer_evaluation"],
        repetitions=latency_repetitions,
    )
    parameter_counts = {
        name: {
            "training": visreg_models[name].training_parameter_count,
            "inference": visreg_models[name].inference_parameter_count,
        }
        for name in VISREG_NAMES
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
            if role == "fit":
                value = encoded[name][role].astype(np.float32)
            else:
                value = restoration[name][role][
                    "original_tokens"
                ].astype(np.float64)
            evidence[f"representation__{name}__{role}"] = value
    for role in EVALUATED_ROLES:
        evidence[f"raw_prediction__{role}"] = raw_predictions[role].astype(
            np.float64
        )
        for name in REPRESENTATION_NAMES:
            evidence[f"prediction__{name}__{role}"] = restoration[name][
                role
            ]["original_probe"].astype(np.float64)
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
        for role in EVALUATED_ROLES:
            for field, values in restoration[name][role].items():
                evidence[
                    f"restoration_{field}__{name}__{role}"
                ] = values.astype(np.float64)
    evidence["training_directions"] = np.asarray(
        detached_evidence["directions"], dtype=np.float32
    )
    for name, values in (
        ("detached_visreg", detached_evidence),
        ("no_detach_visreg", no_detach_evidence),
    ):
        for field, value in values.items():
            if field == "directions":
                continue
            evidence[f"training__{name}__{field}"] = value
        for role, value in diagnostic_views[name].items():
            evidence[
                f"diagnostic_views__{name}__{role}"
            ] = value.astype(np.float32)
        evidence[f"mode_public_tokens__{name}"] = mode_public_tokens[
            name
        ].astype(np.float64)
    evidence["fixed_diagnostic_directions"] = (
        diagnostic_directions.astype(np.float32)
    )
    evidence["collapse_base"] = collapse_base.astype(np.float32)
    evidence["collapse_directions"] = (
        collapse_directions.astype(np.float32)
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
    evidence["deployment_reference_tokens"] = (
        deployment_reference_tokens.astype(np.float64)
    )
    evidence["deployment_reference_prediction"] = (
        deployment_reference_prediction.astype(np.float64)
    )
    np.savez_compressed(building / "evidence.npz", **evidence)

    metadata = {
        "schema_version": 1,
        "kind": "visreg_assessment_evidence_v1",
        "interpretable": interpretable,
        "implementation_commit": implementation_commit,
        "source_corpus_sha256": source_corpus_sha256,
        "source_artifact_manifest_sha256": (
            source_artifact_manifest_sha256
        ),
        "preprocessing_protocol": preprocessing_protocol,
        "prior_control_manifest_sha256": prior_manifest_sha256,
        "prior_model_sha256": dict(FROZEN_PRIOR_MODEL_SHA256),
        "source_sha256": source_hashes,
        "graph": fit.graph.to_dict(),
        "entity_names": list(fit.entity_names),
        "state_feature_names": list(fit.state_feature_names),
        "control_feature_names": list(fit.control_feature_names),
        "action_feature_names": list(fit.action_feature_names),
        "ownership_mask": ownership.astype(int).tolist(),
        "varying_entity_mask": varying.astype(int).tolist(),
        "roles": role_metadata,
        "queries": _query_metadata(transfer_queries),
        "ridge_values": list(RIDGES),
        "selected_ridges": selected_ridges,
        "selection_safety_failed": selection_safety_failed,
        "parameter_counts": parameter_counts,
        "deployed_bundle_bytes": deployed_bundle_bytes,
        "latency": latency,
        "configs": {
            name: visreg_models[name].config.to_dict()
            for name in VISREG_NAMES
        },
        "mode_enforcement": mode_enforcement,
        "diagnostics": diagnostics,
        "collapse_curve": curve,
        "state_probes": state_probes,
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
            "kind": "visreg_isolated_assessor_receipt_v1",
            "returncode": 0,
            "assessor_sha256": source_hashes[
                "lab/action_dynamics/prototype_visreg_assessor.py"
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
        "kind": "visreg_telemetry_tracer_v1",
        "evidence_boundary": (
            "single-seed open-development clean-room telemetry "
            "translation; not a production alert"
        ),
        "interpretable": interpretable,
        "implementation_commit": implementation_commit,
        "source": {
            "cache_directory": str(cache),
            "prior_control_directory": str(prior),
            "source_corpus_sha256": source_corpus_sha256,
            "source_artifact_manifest_sha256": (
                source_artifact_manifest_sha256
            ),
            "prior_control_manifest_sha256": prior_manifest_sha256,
            "preprocessing_protocol": preprocessing_protocol,
            "primary_paper": "https://arxiv.org/abs/2606.02572",
            "official_repository_commit": (
                "https://github.com/HaiyuWu/visreg/tree/"
                "47b1cf4d725b6cbc76dae1394eb46acc2d282fc1"
            ),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": _torch_version(),
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
        "kind": "visreg_artifact_manifest_v1",
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


def _config(objective: str, steps: int) -> VisregConfig:
    return VisregConfig(objective=objective, steps=steps)


def _write_schedules(
    directory: Path,
    fit: ActionConditionedWindows,
    config: VisregConfig,
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
            [batch.visible_tokens[:, 0] for batch in view_batches]
        ),
        view_present=np.stack(
            [batch.present_tokens[:, 0] for batch in view_batches]
        ),
        learning_rate=np.asarray(
            [
                config.learning_rate_at(step)
                for step in range(config.steps)
            ]
        ),
    )


def _collapse_inputs() -> Tuple[np.ndarray, np.ndarray]:
    import torch

    generator = torch.Generator(device="cpu").manual_seed(8509)
    base = torch.randn(
        (8, 40, 64), generator=generator, dtype=torch.float32
    )
    base = base / (
        torch.linalg.vector_norm(base, dim=-1, keepdim=True) + 1e-12
    )
    return (
        base.numpy(),
        fixed_directions(
            width=64, projection_count=256, seed=7509
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


def _write_inference_bundle(
    model: VisregRepresentation,
    probe: ReducedRankActionProbe,
    path: Path,
) -> int:
    payload = {
        "schema_version": 1,
        "kind": "visreg_forecast_inference_bundle_v1",
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
    return path.stat().st_size


def _load_inference_bundle(
    path: Path,
) -> Tuple[VisregRepresentation, ReducedRankActionProbe]:
    payload = dict(
        json.loads(gzip.decompress(Path(path).read_bytes()).decode())
    )
    if (
        set(payload)
        != {"schema_version", "kind", "representation", "probe"}
        or payload.get("schema_version") != 1
        or payload.get("kind")
        != "visreg_forecast_inference_bundle_v1"
    ):
        raise ValueError("unsupported VISReg deployment bundle")
    return (
        VisregRepresentation.from_inference_dict(
            dict(payload["representation"])
        ),
        ReducedRankActionProbe.from_dict(dict(payload["probe"])),
    )


def _latency(
    model: VisregRepresentation,
    probe: ReducedRankActionProbe,
    windows: ActionConditionedWindows,
    *,
    repetitions: int,
) -> Tuple[Mapping[str, Any], np.ndarray]:
    if repetitions < 1:
        raise ValueError("VISReg latency repetitions must be positive")

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
            raise RuntimeError("frozen VISReg run requires clean commit")
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


def _torch_version() -> str:
    import torch

    return str(torch.__version__)


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
                / "lab/action_dynamics/prototype_visreg_assessor.py"
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
        raise ValueError("copied VISReg assessor returned invalid output")
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
            "# VISReg telemetry tracer v1",
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
    options = parser.parse_args(arguments)
    output = run_experiment(
        cache_directory=options.cache,
        prior_control_directory=options.prior_control,
        output_directory=options.output,
        steps=options.steps,
        latency_repetitions=options.latency_repetitions,
        allow_noninterpretable_smoke=(
            options.allow_noninterpretable_smoke
        ),
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
