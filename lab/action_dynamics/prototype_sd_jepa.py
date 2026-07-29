#!/usr/bin/env python3
"""Retained runner for the frozen ticket 015 SD-JEPA alert tracer."""

import argparse
import hashlib
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from lab.action_dynamics.prototype_sd_jepa_assessor import (
    assess_stored_bundle,
)
from quantis_core.edge_dynamics.data import (
    load_edge_dynamics_cache,
    partition_worker_topology,
)
from quantis_core.edge_dynamics.hepa_jepa import (
    EntityStateRidgeProbe,
    HepaEntityPcaBaseline,
    HepaEventDefinition,
    trajectory_action_onsets,
)
from quantis_core.edge_dynamics.sd_jepa import (
    SD_JEPA_OBJECTIVES,
    SD_JEPA_SCORE_NAMES,
    SdJepaConfig,
    SdJepaModel,
    SdScoreCalibrator,
)


FROZEN_CACHE = Path(
    "artifacts/action-dynamics/edge-preprocessing-v1/"
    "eb54271132f88c9a431b01e786ea66279a563776434cca2290e47e6b7ae9b3ff"
)
FROZEN_OUTPUT = Path(
    "artifacts/action-dynamics/prototype-sd-jepa-alert-v1"
)
FROZEN_PRETRAIN_STEPS = 300
IMPLEMENTATION_SOURCE_PATHS = (
    "lab/action_dynamics/prototype_sd_jepa.py",
    "lab/action_dynamics/prototype_sd_jepa_assessor.py",
    "src/quantis_core/edge_dynamics/sd_jepa.py",
    "tests/test_sd_jepa.py",
    "docs/specs/sd-jepa-alert-tracer-v1.md",
    "docs/specs/jepa-experiment-ladder-v1.md",
    "docs/research/sd-jepa-primary-source-notes.md",
    "docs/research/jepa-frontier-technique-audit-2026.md",
    "docs/wayfinding/jepa-implementation-program/"
    "015-test-sd-jepa-alert-tracer.md",
    "src/quantis_core/edge_dynamics/complete_lejepa.py",
    "src/quantis_core/edge_dynamics/hepa_jepa.py",
    "src/quantis_core/edge_dynamics/data.py",
    "src/quantis_core/graph_telemetry.py",
    "src/quantis_core/action_conditioned_dynamics.py",
)


def run_experiment(
    *,
    cache_directory: Path,
    output_directory: Path,
    pretrain_steps: int,
    latency_repetitions: int,
    allow_noninterpretable_smoke: bool,
    expected_pair_count: int = 40,
) -> Path:
    """Run, assess, and atomically publish the non-overwriting tracer."""

    cache = Path(cache_directory).resolve()
    output = Path(output_directory).resolve()
    building = output.with_name(output.name + ".building")
    if output.exists() or building.exists():
        raise FileExistsError(
            "SD-JEPA refuses an existing output or staging directory"
        )
    interpretable = (
        cache == (Path.cwd() / FROZEN_CACHE).resolve()
        and output == (Path.cwd() / FROZEN_OUTPUT).resolve()
        and pretrain_steps == FROZEN_PRETRAIN_STEPS
        and latency_repetitions == 100
        and expected_pair_count == 40
    )
    if not interpretable and not allow_noninterpretable_smoke:
        raise ValueError(
            "non-frozen SD-JEPA runs require "
            "--allow-noninterpretable-smoke"
        )
    if (
        not interpretable
        and output == (Path.cwd() / FROZEN_OUTPUT).resolve()
    ):
        raise ValueError(
            "an SD-JEPA smoke run cannot use the frozen result path"
        )
    implementation_commit = _git_head()
    implementation_sources = implementation_source_identity(
        commit=implementation_commit,
        require_head_match=interpretable,
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
        selection_windows = partitions["selection"].in_distribution
        role_windows = {
            "calibration": partitions[
                "calibration"
            ].in_distribution,
            "evaluation_iid": partitions[
                "evaluation"
            ].in_distribution,
            "evaluation_transfer": partitions[
                "evaluation"
            ].held_out,
        }
        role_identities = {
            "fit": _role_input_identity(fit_windows),
            "selection": _role_input_identity(selection_windows),
            **{
                role: _role_input_identity(windows)
                for role, windows in role_windows.items()
            },
        }
        event_definition = HepaEventDefinition.fit(fit_windows)
        models: Dict[str, SdJepaModel] = {}
        fit_seconds = {}
        checkpoint_directory = building / "objective-checkpoints"
        checkpoint_directory.mkdir(parents=True)
        for objective in SD_JEPA_OBJECTIVES:
            fit_started = time.perf_counter()
            model = SdJepaModel(
                SdJepaConfig(
                    objective=objective,
                    pretrain_steps=pretrain_steps,
                    checkpoint_interval=max(
                        1, min(50, pretrain_steps)
                    ),
                    sigreg_sketch_dimension=(
                        256 if interpretable else 16
                    ),
                    expected_pair_count=expected_pair_count,
                )
            )
            model.fit(fit_windows).select(selection_windows)
            payload = model.to_dict()
            _write_json(
                checkpoint_directory / f"{objective}.json", payload
            )
            restored = SdJepaModel.from_dict(payload)
            original_encoding = model.encode(
                selection_windows.histories[:1],
                selection_windows.graph,
            )
            restored_encoding = restored.encode(
                selection_windows.histories[:1],
                selection_windows.graph,
            )
            if (
                not np.allclose(
                    original_encoding.entity_tokens,
                    restored_encoding.entity_tokens,
                    atol=1e-7,
                    rtol=0.0,
                )
                or not np.allclose(
                    original_encoding.scene_tokens,
                    restored_encoding.scene_tokens,
                    atol=1e-7,
                    rtol=0.0,
                )
            ):
                raise RuntimeError(
                    f"SD-JEPA checkpoint does not restore: {objective}"
                )
            models[objective] = model
            fit_seconds[objective] = (
                time.perf_counter() - fit_started
            )
            _print_progress(
                "fitted",
                {
                    "objective": objective,
                    "seconds": fit_seconds[objective],
                    "selected_step": model.selected_step,
                    "selection": model.selection_metrics,
                },
            )

        score_routes = {
            "sd_jepa_angle": (models["sd_jepa"], "angle"),
            "sd_jepa_z_mse": (models["sd_jepa"], "z_mse"),
            "lewm_unsplit_angle": (
                models["lewm_unsplit"],
                "angle",
            ),
            "lewm_unsplit_z_mse": (
                models["lewm_unsplit"],
                "z_mse",
            ),
            "a2_full_angle": (models["a2_full"], "angle"),
            "a2_full_z_mse": (models["a2_full"], "z_mse"),
        }
        calibrators = {}
        for name in SD_JEPA_SCORE_NAMES:
            model, _ = score_routes[name]
            calibrators[name] = SdScoreCalibrator(
                score_name=name
            ).fit(model, role_windows["calibration"], event_definition)

        model_payloads = {
            name: model.to_dict() for name, model in models.items()
        }
        calibrator_payloads = {
            name: calibrator.to_dict()
            for name, calibrator in calibrators.items()
        }
        restored_models = {
            name: SdJepaModel.from_dict(payload)
            for name, payload in model_payloads.items()
        }
        restored_calibrators = {
            name: SdScoreCalibrator.from_dict(payload)
            for name, payload in calibrator_payloads.items()
        }
        restored_routes = {
            "sd_jepa_angle": (restored_models["sd_jepa"], "angle"),
            "sd_jepa_z_mse": (
                restored_models["sd_jepa"],
                "z_mse",
            ),
            "lewm_unsplit_angle": (
                restored_models["lewm_unsplit"],
                "angle",
            ),
            "lewm_unsplit_z_mse": (
                restored_models["lewm_unsplit"],
                "z_mse",
            ),
            "a2_full_angle": (
                restored_models["a2_full"],
                "angle",
            ),
            "a2_full_z_mse": (
                restored_models["a2_full"],
                "z_mse",
            ),
        }

        scores: Dict[str, Dict[str, NDArray[np.float64]]] = {}
        restored_scores: Dict[
            str, Dict[str, NDArray[np.float64]]
        ] = {}
        calibrated: Dict[str, Dict[str, NDArray[np.float64]]] = {}
        restored_calibrated: Dict[
            str, Dict[str, NDArray[np.float64]]
        ] = {}
        decisions: Dict[str, Dict[str, NDArray[np.bool_]]] = {}
        restored_decisions: Dict[
            str, Dict[str, NDArray[np.bool_]]
        ] = {}
        labels = {}
        for role, windows in role_windows.items():
            scores[role] = {}
            restored_scores[role] = {}
            calibrated[role] = {}
            restored_calibrated[role] = {}
            decisions[role] = {}
            restored_decisions[role] = {}
            labels[role] = (
                event_definition.observed_effect_scores(windows)
                > event_definition.threshold
            )
            for name in SD_JEPA_SCORE_NAMES:
                model, kind = score_routes[name]
                restored_model, restored_kind = restored_routes[name]
                scores[role][name] = model.raw_score(
                    windows.histories, windows.graph, kind=kind
                )
                restored_scores[role][name] = (
                    restored_model.raw_score(
                        windows.histories,
                        windows.graph,
                        kind=restored_kind,
                    )
                )
                calibrated[role][name] = calibrators[
                    name
                ].calibrated_risk(
                    model, windows.histories, windows.graph
                )
                restored_calibrated[role][name] = (
                    restored_calibrators[name].calibrated_risk(
                        restored_model,
                        windows.histories,
                        windows.graph,
                    )
                )
                decisions[role][name] = calibrators[
                    name
                ].alert_decisions(
                    model, windows.histories, windows.graph
                )
                restored_decisions[role][name] = (
                    restored_calibrators[name].alert_decisions(
                        restored_model,
                        windows.histories,
                        windows.graph,
                    )
                )
            _print_progress(
                "evaluated", {"role": role, "rows": len(windows.histories)}
            )

        transfer = role_windows["evaluation_transfer"]
        candidate_transfer = models["sd_jepa"].encode(
            transfer.histories, transfer.graph
        )
        restored_candidate_transfer = restored_models[
            "sd_jepa"
        ].encode(transfer.histories, transfer.graph)
        unsplit_transfer = models["lewm_unsplit"].encode(
            transfer.histories, transfer.graph
        )

        candidate_fit = models["sd_jepa"].encode(
            fit_windows.histories, fit_windows.graph
        )
        pca = HepaEntityPcaBaseline(width=30).fit(fit_windows)
        pca_fit = pca.encode(fit_windows.histories, fit_windows.graph)
        pca_transfer = pca.encode(transfer.histories, transfer.graph)
        current_fit = fit_windows.histories[:, -1]
        current_transfer = transfer.histories[:, -1]
        candidate_probe = EntityStateRidgeProbe().fit(
            candidate_fit.current_content_tokens,
            current_fit,
            candidate_fit.ownership_mask,
        )
        pca_probe = EntityStateRidgeProbe().fit(
            pca_fit.tokens,
            current_fit,
            pca_fit.ownership_mask,
        )
        if (
            not np.array_equal(
                candidate_probe.target_scale, pca_probe.target_scale
            )
            or not np.array_equal(
                candidate_probe.target_varying_mask,
                pca_probe.target_varying_mask,
            )
        ):
            raise RuntimeError("SD-JEPA state probe targets diverged")
        state_predictions = {
            "sd_jepa_content": candidate_probe.predict(
                candidate_transfer.current_content_tokens
            ),
            "matched_pca": pca_probe.predict(pca_transfer.tokens),
        }

        progress_truth = _normalized_progress(transfer)
        progress_features = {
            "sd_jepa_progression": candidate_transfer.scene_tokens[
                :, -1, :2
            ],
            "sd_jepa_content": candidate_transfer.scene_tokens[
                :, -1, 2:
            ],
            "lewm_unsplit_first_two": unsplit_transfer.scene_tokens[
                :, -1, :2
            ],
        }
        latency_samples = _measure_latency(
            score_routes=score_routes,
            calibrators=calibrators,
            histories=transfer.histories[:1],
            graph=transfer.graph,
            repetitions=latency_repetitions,
        )
        peak_rss_bytes = _peak_rss_bytes()

        audit_histories = transfer.histories[:2].copy()
        audit_counterfactual_histories = audit_histories.copy()
        audit_counterfactual_histories[:, -1] += 0.1
        audit_forbidden = np.concatenate(
            (
                transfer.future_states[:2].reshape(2, -1),
                transfer.future_controls[:2].reshape(2, -1),
                transfer.future_actions[:2].reshape(2, -1),
            ),
            axis=1,
        )
        audit_counterfactual_forbidden = audit_forbidden.copy()
        audit_counterfactual_forbidden[:, 0] += 1.0
        audit_counterfactual_forbidden[:, -1] += 1.0
        audit_original_outputs = _audit_outputs(
            models["sd_jepa"],
            calibrators["sd_jepa_angle"],
            audit_histories,
            transfer.graph,
        )
        audit_history_counterfactual_outputs = _audit_outputs(
            models["sd_jepa"],
            calibrators["sd_jepa_angle"],
            audit_counterfactual_histories,
            transfer.graph,
        )
        audit_forbidden_counterfactual_outputs = (
            audit_original_outputs.copy()
        )
        keyword_rejections = np.asarray(
            [
                _rejects_forbidden_keyword(
                    models["sd_jepa"].encode,
                    audit_histories,
                    transfer.graph,
                    keyword,
                    value,
                )
                for keyword, value in (
                    ("future_states", transfer.future_states[:2]),
                    ("future_controls", transfer.future_controls[:2]),
                    ("future_actions", transfer.future_actions[:2]),
                )
            ],
            dtype=np.bool_,
        )

        final_sources = implementation_source_identity(
            commit=implementation_commit,
            require_head_match=interpretable,
        )
        if (
            final_sources != implementation_sources
            or (interpretable and _git_head() != implementation_commit)
        ):
            raise RuntimeError(
                "SD-JEPA implementation identity changed during run"
            )

        _write_json(
            building / "models.json",
            {
                "schema_version": 1,
                "kind": "sd_jepa_models",
                "objectives": model_payloads,
                "matched_pca": pca.to_dict(),
            },
        )
        _write_json(
            building / "calibrators.json",
            {
                "schema_version": 1,
                "kind": "sd_jepa_calibrators",
                "calibrators": calibrator_payloads,
            },
        )
        _write_json(
            building / "state-probes.json",
            {
                "schema_version": 1,
                "kind": "sd_jepa_state_probes",
                "sd_jepa_content": candidate_probe.to_dict(),
                "matched_pca": pca_probe.to_dict(),
            },
        )
        _write_json(
            building / "event-definition.json",
            event_definition.to_dict(),
        )
        protocol = {
            "schema_version": 1,
            "kind": "sd_jepa_alert_protocol",
            "interpretable": interpretable,
            "implementation_commit": implementation_commit,
            "official_source": {
                "repository": "https://github.com/LucasStill/SD-JEPA",
                "revision": (
                    "1cc121065e83220a495808f4c65ef4b0b1915f9f"
                ),
                "paper": "https://arxiv.org/abs/2605.31111",
            },
            "implementation_sources": implementation_sources,
            "pretrain_steps": pretrain_steps,
            "latency_repetitions": latency_repetitions,
            "expected_pair_count": expected_pair_count,
            "roles": {
                role: {
                    "trajectory_ids": list(windows.trajectory_ids),
                    "matched_pair_ids": list(
                        windows.matched_pair_ids
                    ),
                    "control_trajectory_ids": list(
                        _control_trajectory_ids(windows)
                    ),
                    "trajectory_onsets": trajectory_action_onsets(
                        windows
                    ),
                }
                for role, windows in role_windows.items()
            },
            "training_parameter_counts": {
                name: model.training_parameter_count
                for name, model in models.items()
            },
            "inference_parameter_counts": {
                name: model.inference_parameter_count
                for name, model in models.items()
            },
            "fit_seconds": fit_seconds,
            "peak_rss_bytes": peak_rss_bytes,
            "runtime": _runtime_identity(),
            "started_unix_seconds": started,
            "completed_unix_seconds": time.time(),
        }
        _write_json(building / "protocol.json", protocol)
        _write_json(
            building / "data-identity.json",
            {
                "schema_version": 1,
                "kind": "sd_jepa_data_identity",
                "cache_directory": str(cache),
                "source_corpus_sha256": data.source_corpus_sha256,
                "source_artifact_manifest_sha256": (
                    data.source_artifact_manifest_sha256
                ),
                "preprocessing_protocol": data.preprocessing_protocol,
                "roles": role_identities,
            },
        )
        _write_evidence(
            building / "evidence.npz",
            scores=scores,
            restored_scores=restored_scores,
            calibrated=calibrated,
            restored_calibrated=restored_calibrated,
            decisions=decisions,
            restored_decisions=restored_decisions,
            labels=labels,
            role_windows=role_windows,
            candidate_scene=candidate_transfer.scene_tokens,
            candidate_entity=candidate_transfer.entity_tokens,
            restored_candidate_scene=(
                restored_candidate_transfer.scene_tokens
            ),
            restored_candidate_entity=(
                restored_candidate_transfer.entity_tokens
            ),
            progress_truth=progress_truth,
            progress_trajectory_ids=transfer.trajectory_ids,
            progress_features=progress_features,
            state_truth=current_transfer,
            state_scale=candidate_probe.target_scale,
            state_varying_mask=candidate_probe.target_varying_mask,
            state_predictions=state_predictions,
            latency_samples=latency_samples,
            audit_histories=audit_histories,
            audit_counterfactual_histories=(
                audit_counterfactual_histories
            ),
            audit_forbidden=audit_forbidden,
            audit_counterfactual_forbidden=(
                audit_counterfactual_forbidden
            ),
            audit_original_outputs=audit_original_outputs,
            audit_history_counterfactual_outputs=(
                audit_history_counterfactual_outputs
            ),
            audit_forbidden_counterfactual_outputs=(
                audit_forbidden_counterfactual_outputs
            ),
            audit_forbidden_keyword_rejections=keyword_rejections,
        )
        assessment = assess_stored_bundle(building)
        _write_json(building / "assessment.json", assessment)
        (building / "report.md").write_text(
            _render_report(assessment, interpretable=interpretable)
        )
        _copy_reproduction_sources(building)
        write_artifact_manifest(building)
        os.replace(building, output)
        return output
    except BaseException as error:
        _write_json(
            building / "failure.json",
            {
                "schema_version": 1,
                "kind": "sd_jepa_failure",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "failed_unix_seconds": time.time(),
            },
        )
        raise


def _write_evidence(
    path: Path,
    *,
    scores: Mapping[str, Mapping[str, NDArray[np.float64]]],
    restored_scores: Mapping[
        str, Mapping[str, NDArray[np.float64]]
    ],
    calibrated: Mapping[str, Mapping[str, NDArray[np.float64]]],
    restored_calibrated: Mapping[
        str, Mapping[str, NDArray[np.float64]]
    ],
    decisions: Mapping[str, Mapping[str, NDArray[np.bool_]]],
    restored_decisions: Mapping[
        str, Mapping[str, NDArray[np.bool_]]
    ],
    labels: Mapping[str, NDArray[np.bool_]],
    role_windows: Mapping[str, Any],
    candidate_scene: NDArray[np.float64],
    candidate_entity: NDArray[np.float64],
    restored_candidate_scene: NDArray[np.float64],
    restored_candidate_entity: NDArray[np.float64],
    progress_truth: NDArray[np.float64],
    progress_trajectory_ids: Sequence[str],
    progress_features: Mapping[str, NDArray[np.float64]],
    state_truth: NDArray[np.float64],
    state_scale: NDArray[np.float64],
    state_varying_mask: NDArray[np.bool_],
    state_predictions: Mapping[str, NDArray[np.float64]],
    latency_samples: Mapping[str, NDArray[np.float64]],
    audit_histories: NDArray[np.float64],
    audit_counterfactual_histories: NDArray[np.float64],
    audit_forbidden: NDArray[np.float64],
    audit_counterfactual_forbidden: NDArray[np.float64],
    audit_original_outputs: NDArray[np.float64],
    audit_history_counterfactual_outputs: NDArray[np.float64],
    audit_forbidden_counterfactual_outputs: NDArray[np.float64],
    audit_forbidden_keyword_rejections: NDArray[np.bool_],
) -> None:
    arrays: Dict[str, NDArray[Any]] = {}
    for prefix, values in (
        ("score", scores),
        ("restored_score", restored_scores),
        ("calibrated", calibrated),
        ("restored_calibrated", restored_calibrated),
        ("alert_decision", decisions),
        ("restored_alert_decision", restored_decisions),
    ):
        arrays.update(
            {
                f"{prefix}__{role}__{name}": value
                for role, names in values.items()
                for name, value in names.items()
            }
        )
    arrays.update(
        {f"labels__{role}": value for role, value in labels.items()}
    )
    arrays.update(
        {
            f"transition_indices__{role}": windows.transition_indices
            for role, windows in role_windows.items()
        }
    )
    arrays.update(
        {
            "candidate_scene": candidate_scene,
            "candidate_entity": candidate_entity,
            "restored_candidate_scene": restored_candidate_scene,
            "restored_candidate_entity": restored_candidate_entity,
            "progress_truth": progress_truth,
            "progress_trajectory_ids": np.asarray(
                progress_trajectory_ids, dtype=str
            ),
            "state_truth": state_truth,
            "state_scale": state_scale,
            "state_varying_mask": state_varying_mask,
            "audit_histories": audit_histories,
            "audit_counterfactual_histories": (
                audit_counterfactual_histories
            ),
            "audit_forbidden": audit_forbidden,
            "audit_counterfactual_forbidden": (
                audit_counterfactual_forbidden
            ),
            "audit_original_outputs": audit_original_outputs,
            "audit_history_counterfactual_outputs": (
                audit_history_counterfactual_outputs
            ),
            "audit_forbidden_counterfactual_outputs": (
                audit_forbidden_counterfactual_outputs
            ),
            "audit_forbidden_keyword_rejections": (
                audit_forbidden_keyword_rejections
            ),
        }
    )
    arrays.update(
        {
            f"progress_features__{name}": value
            for name, value in progress_features.items()
        }
    )
    arrays.update(
        {
            f"state_prediction__{name}": value
            for name, value in state_predictions.items()
        }
    )
    arrays.update(
        {
            f"latency_samples__{name}": value
            for name, value in latency_samples.items()
        }
    )
    np.savez_compressed(path, **arrays)


def _normalized_progress(windows: Any) -> NDArray[np.float64]:
    ids = np.asarray(windows.trajectory_ids, dtype=str)
    transitions = np.asarray(windows.transition_indices, dtype=np.float64)
    result = np.empty(len(ids), dtype=np.float64)
    for trajectory_id in sorted(set(windows.trajectory_ids)):
        selected = ids == trajectory_id
        local = transitions[selected]
        span = float(np.max(local) - np.min(local))
        result[selected] = (
            0.0 if span <= 0.0 else (local - np.min(local)) / span
        )
    return result


def _measure_latency(
    *,
    score_routes: Mapping[str, Tuple[SdJepaModel, str]],
    calibrators: Mapping[str, SdScoreCalibrator],
    histories: NDArray[np.float64],
    graph: Any,
    repetitions: int,
) -> Mapping[str, NDArray[np.float64]]:
    if repetitions < 1:
        raise ValueError("SD-JEPA latency repetitions must be positive")
    result = {}
    for name in SD_JEPA_SCORE_NAMES:
        model, _ = score_routes[name]
        for _ in range(3):
            calibrators[name].calibrated_risk(
                model, histories, graph
            )
        samples = []
        for _ in range(repetitions):
            started = time.perf_counter_ns()
            calibrators[name].calibrated_risk(
                model, histories, graph
            )
            samples.append(
                (time.perf_counter_ns() - started) / 1_000_000.0
            )
        result[name] = np.asarray(samples, dtype=np.float64)
    return result


def _audit_outputs(
    model: SdJepaModel,
    calibrator: SdScoreCalibrator,
    histories: NDArray[np.float64],
    graph: Any,
) -> NDArray[np.float64]:
    encoded = model.encode(histories, graph)
    risk = calibrator.calibrated_risk(model, histories, graph)
    return np.concatenate(
        (encoded.scene_tokens[:, -1], risk[:, None]), axis=1
    )


def _rejects_forbidden_keyword(
    function: Any,
    histories: NDArray[np.float64],
    graph: Any,
    keyword: str,
    value: NDArray[np.float64],
) -> bool:
    try:
        function(histories, graph, **{keyword: value})
    except TypeError:
        return True
    return False


def _control_trajectory_ids(windows: Any) -> Tuple[str, ...]:
    applicable = windows.action_feature_names.index("applicable")
    treatments = {
        windows.trajectory_ids[index]
        for index in range(len(windows.histories))
        if np.any(windows.future_actions[index, ..., applicable] > 0.5)
    }
    return tuple(sorted(set(windows.trajectory_ids) - treatments))


def implementation_source_identity(
    *, commit: str, require_head_match: bool
) -> Mapping[str, Any]:
    result = {}
    for relative in IMPLEMENTATION_SOURCE_PATHS:
        path = Path(relative)
        local = path.read_bytes()
        head = _git_blob(commit, relative)
        matches = head == local
        if require_head_match and not matches:
            raise RuntimeError(
                f"frozen SD-JEPA source does not match {commit}: "
                f"{relative}"
            )
        result[relative] = {
            "path": relative,
            "scope": "file",
            "bytes": len(local),
            "sha256": hashlib.sha256(local).hexdigest(),
            "matches_head": matches,
            "head_sha256": (
                None
                if head is None
                else hashlib.sha256(head).hexdigest()
            ),
        }
    return result


def _copy_reproduction_sources(building: Path) -> None:
    root = building / "reproduction-sources"
    for relative in IMPLEMENTATION_SOURCE_PATHS:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(relative, destination)


def write_artifact_manifest(directory: Path) -> None:
    files = {
        str(path.relative_to(directory)): {
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "artifact-manifest.json"
    }
    _write_json(
        directory / "artifact-manifest.json",
        {
            "schema_version": 1,
            "kind": "sd_jepa_artifact_manifest",
            "files": files,
        },
    )


def _render_report(
    assessment: Mapping[str, Any], *, interpretable: bool
) -> str:
    localization = assessment["localization"]["evaluation_transfer"]
    risk = assessment["risk_metrics"]["evaluation_transfer"]
    alerts = assessment["alert_metrics"]["evaluation_transfer"]
    rows = []
    for name in SD_JEPA_SCORE_NAMES:
        rows.append(
            "| "
            + " | ".join(
                (
                    name,
                    str(localization[name]["pooled_auroc"]),
                    f"{risk[name]['brier']:.6f}",
                    f"{alerts[name]['control_trajectory_false_alarm_rate']:.3f}",
                    f"{alerts[name]['treatment_detection_rate']:.3f}",
                )
            )
            + " |"
        )
    return "\n".join(
        (
            "# SD-JEPA alert report",
            "",
            (
                "Status: **frozen interpretable tracer**."
                if interpretable
                else "Status: **NON-INTERPRETABLE SMOKE RUN**."
            ),
            "",
            "| score | event AUROC | Brier | control FPR | detection |",
            "|---|---:|---:|---:|---:|",
            *rows,
            "",
            (
                "Progress R2 (candidate / content / A0 first-two): "
                f"`{assessment['progress']['sd_jepa_progression']['pooled_r2']:.6f}` / "
                f"`{assessment['progress']['sd_jepa_content']['pooled_r2']:.6f}` / "
                f"`{assessment['progress']['lewm_unsplit_first_two']['pooled_r2']:.6f}`."
            ),
            f"Decision: `{assessment['decision']}`.",
            "",
        )
    )


def _role_input_identity(windows: Any) -> Mapping[str, Any]:
    fields = {
        "row_count": len(windows.histories),
        "matched_pair_ids": list(windows.matched_pair_ids),
        "trajectory_ids": list(windows.trajectory_ids),
        "transition_indices": [
            int(value) for value in windows.transition_indices
        ],
        "semantic_schema_sha256": windows.semantic_schema_sha256,
        "arrays": {
            "histories": _array_identity(windows.histories),
            "future_states": _array_identity(windows.future_states),
            "future_controls": _array_identity(
                windows.future_controls
            ),
            "future_actions": _array_identity(windows.future_actions),
        },
    }
    return {
        **fields,
        "identity_sha256": hashlib.sha256(
            _canonical_json_bytes(fields)
        ).hexdigest(),
    }


def _array_identity(values: NDArray[Any]) -> Mapping[str, Any]:
    array = np.ascontiguousarray(values)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


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


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _git_head() -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_blob(commit: str, relative: str) -> Optional[bytes]:
    result = subprocess.run(
        ("git", "show", f"{commit}:{relative}"),
        check=False,
        capture_output=True,
    )
    return result.stdout if result.returncode == 0 else None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            value, indent=2, sort_keys=True, allow_nan=False
        )
        + "\n"
    )


def _print_progress(name: str, value: Mapping[str, Any]) -> None:
    print(
        json.dumps(
            {"stage": name, **dict(value)},
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=FROZEN_CACHE)
    parser.add_argument("--output", type=Path, default=FROZEN_OUTPUT)
    parser.add_argument(
        "--pretrain-steps",
        type=int,
        default=FROZEN_PRETRAIN_STEPS,
    )
    parser.add_argument(
        "--latency-repetitions", type=int, default=100
    )
    parser.add_argument(
        "--allow-noninterpretable-smoke", action="store_true"
    )
    arguments = parser.parse_args()
    result = run_experiment(
        cache_directory=arguments.cache,
        output_directory=arguments.output,
        pretrain_steps=arguments.pretrain_steps,
        latency_repetitions=arguments.latency_repetitions,
        allow_noninterpretable_smoke=(
            arguments.allow_noninterpretable_smoke
        ),
    )
    print(result)


if __name__ == "__main__":
    main()
