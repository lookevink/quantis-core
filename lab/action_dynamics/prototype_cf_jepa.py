#!/usr/bin/env python3
"""Retained runner for the frozen ticket 014 CF-JEPA alert tracer."""

import argparse
import ast
import hashlib
import inspect
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

from lab.action_dynamics.prototype_cf_jepa_assessor import (
    assess_stored_bundle,
)
from quantis_core.edge_dynamics.cf_jepa import (
    CF_JEPA_ALERT_MODEL_NAMES,
    CF_JEPA_ASSESSMENT_ROLE_NAMES,
    CF_JEPA_OBJECTIVES,
    CfGaussianAlert,
    CfJepaConfig,
    CfJepaModel,
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


FROZEN_CACHE = Path(
    "artifacts/action-dynamics/edge-preprocessing-v1/"
    "eb54271132f88c9a431b01e786ea66279a563776434cca2290e47e6b7ae9b3ff"
)
FROZEN_OUTPUT = Path(
    "artifacts/action-dynamics/prototype-cf-jepa-alert-v1"
)
FROZEN_PRETRAIN_STEPS = 300
IMPLEMENTATION_SOURCE_PATHS = (
    "lab/action_dynamics/prototype_cf_jepa.py",
    "lab/action_dynamics/prototype_cf_jepa_assessor.py",
    "src/quantis_core/edge_dynamics/cf_jepa.py",
    "tests/test_cf_jepa.py",
    "docs/specs/cf-jepa-alert-tracer-v1.md",
    "docs/specs/jepa-experiment-ladder-v1.md",
    "docs/research/cf-jepa-primary-source-notes.md",
    "docs/research/jepa-frontier-technique-audit-2026.md",
    "docs/wayfinding/jepa-implementation-program/"
    "014-test-cf-jepa-alert-tracer.md",
    "src/quantis_core/edge_dynamics/hepa_jepa.py",
    "src/quantis_core/edge_dynamics/data.py",
    "src/quantis_core/graph_telemetry.py",
)
IMPLEMENTATION_SYMBOL_SOURCES = {
    "src/quantis_core/action_conditioned_dynamics.py": (
        "ActionConditionedWindows",
        "_semantic_schema_sha256",
    ),
}


class _PcaModelAdapter:
    """Give the matched PCA the same alert-adapter encoding call."""

    def __init__(self, model: HepaEntityPcaBaseline) -> None:
        self.model = model

    def encode(
        self,
        histories: NDArray[np.float64],
        graph: Any,
        *,
        route: str,
    ) -> Any:
        if route != "target":
            raise ValueError("matched PCA exposes only the target route")
        return self.model.encode(histories, graph)


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
            "CF-JEPA refuses an existing output or staging directory"
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
            "non-frozen CF-JEPA runs require "
            "--allow-noninterpretable-smoke"
        )
    if (
        not interpretable
        and output == (Path.cwd() / FROZEN_OUTPUT).resolve()
    ):
        raise ValueError(
            "a smoke run cannot use the frozen CF-JEPA result path"
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
        objectives: Dict[str, CfJepaModel] = {}
        fit_seconds = {}
        for objective in CF_JEPA_OBJECTIVES:
            fit_started = time.perf_counter()
            model = CfJepaModel(
                CfJepaConfig(
                    objective=objective,
                    pretrain_steps=pretrain_steps,
                    checkpoint_interval=max(
                        1, min(50, pretrain_steps)
                    ),
                    expected_pair_count=expected_pair_count,
                )
            )
            model.fit(fit_windows).select(selection_windows)
            checkpoint_payload = model.to_dict()
            checkpoint_directory = (
                building / "objective-checkpoints"
            )
            checkpoint_directory.mkdir(parents=True, exist_ok=True)
            _write_json(
                checkpoint_directory / f"{objective}.json",
                checkpoint_payload,
            )
            checkpoint_model = CfJepaModel.from_dict(
                checkpoint_payload
            )
            for route in ("online", "target"):
                original = model.encode(
                    selection_windows.histories[:1],
                    selection_windows.graph,
                    route=route,
                ).temporal_tokens
                restored = checkpoint_model.encode(
                    selection_windows.histories[:1],
                    selection_windows.graph,
                    route=route,
                ).temporal_tokens
                if not np.allclose(
                    original, restored, atol=1e-7, rtol=0.0
                ):
                    raise RuntimeError(
                        "CF-JEPA objective checkpoint does not restore: "
                        f"{objective}/{route}"
                    )
            objectives[objective] = model
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

        pca = HepaEntityPcaBaseline(width=32).fit(fit_windows)
        pca_adapter = _PcaModelAdapter(pca)
        model_routes: Mapping[str, Tuple[Any, str]] = {
            "cf_jepa_target": (objectives["three_zone"], "target"),
            "cf_jepa_online": (objectives["three_zone"], "online"),
            "one_zone_target": (objectives["one_zone"], "target"),
            "masked_latent_target": (
                objectives["masked_latent"],
                "target",
            ),
            "matched_pca": (pca_adapter, "target"),
        }
        alerts: Dict[str, CfGaussianAlert] = {}
        for name, (model, route) in model_routes.items():
            alert = CfGaussianAlert(route=route).fit(
                model, fit_windows
            )
            alert.fit_calibration(
                model, role_windows["calibration"], event_definition
            )
            alerts[name] = alert

        objective_payloads = {
            name: model.to_dict()
            for name, model in objectives.items()
        }
        pca_payload = pca.to_dict()
        alert_payloads = {
            name: alert.to_dict() for name, alert in alerts.items()
        }
        restored_objectives = {
            name: CfJepaModel.from_dict(payload)
            for name, payload in objective_payloads.items()
        }
        restored_pca = HepaEntityPcaBaseline.from_dict(pca_payload)
        restored_pca_adapter = _PcaModelAdapter(restored_pca)
        restored_alerts = {
            name: CfGaussianAlert.from_dict(payload)
            for name, payload in alert_payloads.items()
        }
        restored_routes: Mapping[str, Tuple[Any, str]] = {
            "cf_jepa_target": (
                restored_objectives["three_zone"],
                "target",
            ),
            "cf_jepa_online": (
                restored_objectives["three_zone"],
                "online",
            ),
            "one_zone_target": (
                restored_objectives["one_zone"],
                "target",
            ),
            "masked_latent_target": (
                restored_objectives["masked_latent"],
                "target",
            ),
            "matched_pca": (restored_pca_adapter, "target"),
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
            labels[role] = event_definition.labels(windows)[:, -1]
            for name in CF_JEPA_ALERT_MODEL_NAMES:
                model, _ = model_routes[name]
                restored_model, _ = restored_routes[name]
                scores[role][name] = alerts[name].score(
                    model, windows.histories, windows.graph
                )
                restored_scores[role][name] = restored_alerts[
                    name
                ].score(
                    restored_model,
                    windows.histories,
                    windows.graph,
                )
                calibrated[role][name] = alerts[
                    name
                ].calibrated_risk(
                    model, windows.histories, windows.graph
                )
                restored_calibrated[role][name] = restored_alerts[
                    name
                ].calibrated_risk(
                    restored_model,
                    windows.histories,
                    windows.graph,
                )
                decisions[role][name] = alerts[
                    name
                ].alert_decisions(
                    model, windows.histories, windows.graph
                )
                restored_decisions[role][name] = restored_alerts[
                    name
                ].alert_decisions(
                    restored_model,
                    windows.histories,
                    windows.graph,
                )
            _print_progress(
                "evaluated",
                {"role": role, "rows": len(windows.histories)},
            )

        transfer = role_windows["evaluation_transfer"]
        target_transfer = objectives["three_zone"].encode(
            transfer.histories, transfer.graph, route="target"
        )
        online_transfer = objectives["three_zone"].encode(
            transfer.histories, transfer.graph, route="online"
        )
        restored_target_transfer = restored_objectives[
            "three_zone"
        ].encode(transfer.histories, transfer.graph, route="target")
        restored_online_transfer = restored_objectives[
            "three_zone"
        ].encode(transfer.histories, transfer.graph, route="online")

        candidate_fit = objectives["three_zone"].encode(
            fit_windows.histories, fit_windows.graph, route="target"
        )
        pca_fit = pca.encode(
            fit_windows.histories, fit_windows.graph
        )
        pca_transfer = pca.encode(
            transfer.histories, transfer.graph
        )
        current_fit = fit_windows.histories[:, -1]
        current_transfer = transfer.histories[:, -1]
        candidate_probe = EntityStateRidgeProbe().fit(
            candidate_fit.tokens,
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
                candidate_probe.target_scale,
                pca_probe.target_scale,
            )
            or not np.array_equal(
                candidate_probe.target_varying_mask,
                pca_probe.target_varying_mask,
            )
        ):
            raise RuntimeError("CF-JEPA state probe targets diverged")
        state_predictions = {
            "cf_jepa_target": candidate_probe.predict(
                target_transfer.tokens
            ),
            "matched_pca": pca_probe.predict(pca_transfer.tokens),
        }

        latency_samples = _measure_latency(
            model_routes=model_routes,
            alerts=alerts,
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
            objectives["three_zone"],
            alerts["cf_jepa_target"],
            audit_histories,
            transfer.graph,
        )
        audit_history_counterfactual_outputs = _audit_outputs(
            objectives["three_zone"],
            alerts["cf_jepa_target"],
            audit_counterfactual_histories,
            transfer.graph,
        )
        audit_forbidden_counterfactual_outputs = (
            audit_original_outputs.copy()
        )
        keyword_rejections = np.asarray(
            [
                _rejects_forbidden_keyword(
                    objectives["three_zone"].encode,
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
                "CF-JEPA implementation identity changed during run"
            )

        models_payload = {
            "schema_version": 1,
            "kind": "cf_jepa_models",
            "objectives": objective_payloads,
            "matched_pca": pca_payload,
        }
        alerts_payload = {
            "schema_version": 1,
            "kind": "cf_jepa_alerts",
            "alerts": alert_payloads,
        }
        probes_payload = {
            "schema_version": 1,
            "kind": "cf_jepa_state_probes",
            "cf_jepa_target": candidate_probe.to_dict(),
            "matched_pca": pca_probe.to_dict(),
        }
        protocol = {
            "schema_version": 1,
            "kind": "cf_jepa_alert_protocol",
            "interpretable": interpretable,
            "implementation_commit": implementation_commit,
            "official_source": {
                "repository": "https://github.com/WDSLab/CF-JEPA",
                "revision": (
                    "4968faf731c8c56e89d78d944716e212392eb5a0"
                ),
                "paper": "https://arxiv.org/abs/2606.07031",
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
                for name, model in objectives.items()
            },
            "inference_parameter_counts": {
                name: objectives[
                    (
                        "three_zone"
                        if name
                        in {"cf_jepa_target", "cf_jepa_online"}
                        else (
                            "one_zone"
                            if name == "one_zone_target"
                            else "masked_latent"
                        )
                    )
                ].inference_parameter_count
                for name in (
                    "cf_jepa_target",
                    "cf_jepa_online",
                    "one_zone_target",
                    "masked_latent_target",
                )
            },
            "fit_seconds": fit_seconds,
            "peak_rss_bytes": peak_rss_bytes,
            "runtime": _runtime_identity(),
            "started_unix_seconds": started,
            "completed_unix_seconds": time.time(),
        }
        data_identity = {
            "schema_version": 1,
            "kind": "cf_jepa_data_identity",
            "cache_directory": str(cache),
            "source_corpus_sha256": data.source_corpus_sha256,
            "source_artifact_manifest_sha256": (
                data.source_artifact_manifest_sha256
            ),
            "preprocessing_protocol": data.preprocessing_protocol,
            "roles": role_identities,
        }
        _write_json(building / "models.json", models_payload)
        _write_json(building / "alerts.json", alerts_payload)
        _write_json(building / "state-probes.json", probes_payload)
        _write_json(
            building / "event-definition.json",
            event_definition.to_dict(),
        )
        _write_json(building / "protocol.json", protocol)
        _write_json(building / "data-identity.json", data_identity)
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
            target_transfer=target_transfer.temporal_tokens,
            online_transfer=online_transfer.temporal_tokens,
            restored_target_transfer=(
                restored_target_transfer.temporal_tokens
            ),
            restored_online_transfer=(
                restored_online_transfer.temporal_tokens
            ),
            state_truth=current_transfer,
            state_scale=candidate_probe.target_scale,
            state_varying_mask=(
                candidate_probe.target_varying_mask
            ),
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
                "kind": "cf_jepa_failure",
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
    target_transfer: NDArray[np.float64],
    online_transfer: NDArray[np.float64],
    restored_target_transfer: NDArray[np.float64],
    restored_online_transfer: NDArray[np.float64],
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
                f"{prefix}__{role}__{model}": model_values
                for role, models in values.items()
                for model, model_values in models.items()
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
            "candidate_target_temporal": target_transfer,
            "candidate_online_temporal": online_transfer,
            "restored_candidate_target_temporal": (
                restored_target_transfer
            ),
            "restored_candidate_online_temporal": (
                restored_online_transfer
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
            f"state_prediction__{name}": values
            for name, values in state_predictions.items()
        }
    )
    arrays.update(
        {
            f"latency_samples__{name}": values
            for name, values in latency_samples.items()
        }
    )
    np.savez_compressed(path, **arrays)


def _measure_latency(
    *,
    model_routes: Mapping[str, Tuple[Any, str]],
    alerts: Mapping[str, CfGaussianAlert],
    histories: NDArray[np.float64],
    graph: Any,
    repetitions: int,
) -> Mapping[str, NDArray[np.float64]]:
    if repetitions < 1:
        raise ValueError("CF-JEPA latency repetitions must be positive")
    result = {}
    for name in CF_JEPA_ALERT_MODEL_NAMES:
        model, _ = model_routes[name]
        for _ in range(3):
            alerts[name].score(model, histories, graph)
        samples = []
        for _ in range(repetitions):
            started = time.perf_counter_ns()
            alerts[name].score(model, histories, graph)
            samples.append(
                (time.perf_counter_ns() - started) / 1_000_000.0
            )
        result[name] = np.asarray(samples, dtype=np.float64)
    return result


def _audit_outputs(
    model: CfJepaModel,
    alert: CfGaussianAlert,
    histories: NDArray[np.float64],
    graph: Any,
) -> NDArray[np.float64]:
    encoded = model.encode(histories, graph, route="target")
    score = alert.score(model, histories, graph)
    return np.concatenate(
        (encoded.tokens.reshape(len(histories), -1), score[:, None]),
        axis=1,
    )


def _rejects_forbidden_keyword(
    function: Any,
    histories: NDArray[np.float64],
    graph: Any,
    keyword: str,
    value: NDArray[np.float64],
) -> bool:
    try:
        function(
            histories,
            graph,
            route="target",
            **{keyword: value},
        )
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
    """Hash exact implementation files and selected dirty-file symbols."""

    result: Dict[str, Any] = {}
    for relative in IMPLEMENTATION_SOURCE_PATHS:
        path = Path(relative)
        local = path.read_bytes()
        head = _git_blob(commit, relative)
        matches = head == local
        if require_head_match and not matches:
            raise RuntimeError(
                f"frozen CF-JEPA source does not match {commit}: "
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
    for relative, symbols in IMPLEMENTATION_SYMBOL_SOURCES.items():
        local_file = Path(relative).read_bytes()
        head_file = _git_blob(commit, relative)
        for symbol in symbols:
            local = _source_symbol_bytes(local_file, symbol)
            head = (
                None
                if head_file is None
                else _source_symbol_bytes(head_file, symbol)
            )
            matches = head == local
            if require_head_match and not matches:
                raise RuntimeError(
                    "frozen CF-JEPA dependency symbol does not "
                    f"match {commit}: {relative}::{symbol}"
                )
            result[f"{relative}::{symbol}"] = {
                "path": relative,
                "scope": "symbol",
                "symbol": symbol,
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
    for relative in (
        *IMPLEMENTATION_SOURCE_PATHS,
        *IMPLEMENTATION_SYMBOL_SOURCES,
    ):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(relative, destination)


def write_artifact_manifest(directory: Path) -> None:
    """Write a SHA-256 and size manifest for all artifact files."""

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
            "kind": "cf_jepa_artifact_manifest",
            "files": files,
        },
    )


def _render_report(
    assessment: Mapping[str, Any], *, interpretable: bool
) -> str:
    transfer = assessment["risk_metrics"]["evaluation_transfer"]
    alerts = assessment["alert_metrics"]["evaluation_transfer"]
    rows = []
    for name in CF_JEPA_ALERT_MODEL_NAMES:
        rows.append(
            "| "
            + " | ".join(
                (
                    name,
                    f"{transfer[name]['brier']:.6f}",
                    f"{alerts[name]['control_trajectory_false_alarm_rate']:.3f}",
                    f"{alerts[name]['treatment_detection_rate']:.3f}",
                    str(
                        alerts[name][
                            "median_post_onset_delay_transitions"
                        ]
                    ),
                )
            )
            + " |"
        )
    geometry = assessment["geometry"]
    return "\n".join(
        (
            "# CF-JEPA alert report",
            "",
            (
                "Status: **frozen interpretable tracer**."
                if interpretable
                else "Status: **NON-INTERPRETABLE SMOKE RUN**."
            ),
            "",
            "| model | Brier | control FPR | detection | median delay |",
            "|---|---:|---:|---:|---:|",
            *rows,
            "",
            (
                "Target/online adjacent cosine: "
                f"`{geometry['target_adjacent_cosine_similarity']:.6f}` / "
                f"`{geometry['online_adjacent_cosine_similarity']:.6f}`."
            ),
            (
                "Target/online effective rank: "
                f"`{geometry['target_effective_rank_90']}` / "
                f"`{geometry['online_effective_rank_90']}`."
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
    torch = _require_torch()
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


def _source_symbol_bytes(source: bytes, symbol: str) -> bytes:
    text = source.decode("utf-8")
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(
            node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ) and node.name == symbol:
            segment = ast.get_source_segment(text, node)
            if segment is not None:
                return segment.encode("utf-8")
    raise ValueError(f"CF-JEPA dependency symbol is missing: {symbol}")


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


def _require_torch() -> Any:
    import torch

    return torch


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
