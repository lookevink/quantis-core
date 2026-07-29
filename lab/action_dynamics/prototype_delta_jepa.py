#!/usr/bin/env python3
"""Retained runner for the frozen ticket 016 Delta-JEPA tracer."""

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
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from lab.action_dynamics.prototype_complete_lejepa import (
    _action_sanity_evidence,
    _attribution_evidence,
    _encode_histories,
    _forecast_scores,
    _transfer_queries,
)
from lab.action_dynamics.prototype_delta_jepa_assessor import (
    CELL_NAMES,
    assess_stored_bundle,
)
from quantis_core.edge_dynamics.complete_lejepa import (
    EntityPcaRepresentation,
    PairBlockedAnchorSchedule,
    ReducedRankActionProbe,
    fit_owned_feature_mask,
)
from quantis_core.edge_dynamics.data import (
    load_edge_dynamics_cache,
    partition_worker_topology,
)
from quantis_core.edge_dynamics.delta_jepa import (
    DeltaIntervalDiagnostics,
    DeltaJepaConfig,
    DeltaJepaModel,
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
    "artifacts/action-dynamics/prototype-delta-jepa-v1"
)
FROZEN_PRETRAIN_STEPS = 1600
RIDGES = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
IMPLEMENTATION_SOURCE_PATHS = (
    "lab/action_dynamics/prototype_delta_jepa.py",
    "lab/action_dynamics/prototype_delta_jepa_assessor.py",
    "src/quantis_core/edge_dynamics/delta_jepa.py",
    "tests/test_delta_jepa.py",
    "docs/specs/delta-jepa-action-displacement-tracer-v1.md",
    "docs/research/delta-jepa-primary-source-notes.md",
    "docs/wayfinding/jepa-implementation-program/"
    "016-test-delta-jepa-action-displacement.md",
    "src/quantis_core/edge_dynamics/complete_lejepa.py",
    "src/quantis_core/edge_dynamics/data.py",
    "src/quantis_core/edge_dynamics/models.py",
    "src/quantis_core/action_conditioned_dynamics.py",
    "src/quantis_core/graph_telemetry.py",
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
    """Run, independently assess, and atomically publish one tracer."""

    cache = Path(cache_directory).resolve()
    output = Path(output_directory).resolve()
    building = output.with_name(output.name + ".building")
    if output.exists() or building.exists():
        raise FileExistsError(
            "Delta-JEPA refuses an existing output or staging directory"
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
            "non-frozen Delta-JEPA runs require "
            "--allow-noninterpretable-smoke"
        )
    if not interpretable and output == (Path.cwd() / FROZEN_OUTPUT).resolve():
        raise ValueError(
            "a Delta-JEPA smoke run cannot use the frozen result path"
        )
    implementation_commit = _git_head()
    implementation_sources = implementation_source_identity(
        commit=implementation_commit,
        require_head_match=interpretable,
    )
    building.mkdir(parents=True)
    started = time.time()
    try:
        prepared = load_edge_dynamics_cache(cache)
        partitions = {
            role: partition_worker_topology(windows)
            for role, windows in prepared.windows.items()
        }
        held_out_values = {
            value.held_out_normalized_value for value in partitions.values()
        }
        if len(held_out_values) != 1:
            raise ValueError("Delta-JEPA held topology identity differs")
        held_out_value = next(iter(held_out_values))
        windows_by_role = {
            "fit": partitions["fit"].in_distribution,
            "selection": partitions["selection"].in_distribution,
            "iid_evaluation": partitions["evaluation"].in_distribution,
            "transfer_evaluation": partitions["evaluation"].held_out,
        }
        fit = windows_by_role["fit"]
        ownership = fit_owned_feature_mask(fit)
        transfer_queries = _transfer_queries(
            prepared.attribution_queries,
            fit.control_feature_names,
            held_out_value,
        )
        schedule = PairBlockedAnchorSchedule(
            fit, seed=DeltaJepaConfig().seed + 1
        )
        anchor_batches = [
            schedule.batch(step) for step in range(pretrain_steps)
        ]
        np.savez_compressed(
            building / "anchor-schedule.npz",
            indices=np.stack([batch.indices for batch in anchor_batches]),
            arm_ids=np.stack([batch.arm_ids for batch in anchor_batches]),
            transition_indices=np.stack(
                [batch.transition_indices for batch in anchor_batches]
            ),
            pair_ids=np.asarray(schedule.pair_ids),
        )

        models: Dict[str, DeltaJepaModel] = {}
        training_seconds = {}
        checkpoint_directory = building / "models"
        checkpoint_directory.mkdir()
        for name in CELL_NAMES:
            config = DeltaJepaConfig(
                objective=name,
                pretrain_steps=pretrain_steps,
                checkpoint_interval=max(
                    1, min(200, pretrain_steps)
                ),
                expected_pair_count=expected_pair_count,
            )
            fit_started = time.perf_counter()
            model = DeltaJepaModel(config).fit(fit).select(
                windows_by_role["selection"]
            )
            training_seconds[name] = time.perf_counter() - fit_started
            models[name] = model
            _write_json(
                checkpoint_directory / f"{name}.json", model.to_dict()
            )
            _print_progress(
                "fitted",
                {
                    "objective": name,
                    "seconds": training_seconds[name],
                    "selected_step": model.selected_step,
                    "selection": model.selection_metrics,
                },
            )

        encoded = {
            name: {
                role: _encode_delta_batches(model, windows.histories, fit.graph)
                for role, windows in windows_by_role.items()
            }
            for name, model in models.items()
        }
        pca = EntityPcaRepresentation(width=16).fit(fit)
        pca_encoded = {
            role: _encode_histories(pca, windows.histories, fit.graph)
            for role, windows in windows_by_role.items()
        }
        _write_json(checkpoint_directory / "matched_pca.json", pca.to_dict())

        raw_model = ContractiveLowRankDynamics(
            LowRankConfig(rank=32)
        ).fit(fit)
        _write_json(
            checkpoint_directory / "raw_low_rank.json", raw_model.to_dict()
        )
        raw_predictions = {
            role: raw_model.rollout(
                windows.histories,
                windows.future_controls,
                windows.future_actions,
                windows.graph,
            ).mean
            for role, windows in windows_by_role.items()
            if role != "fit"
        }
        raw_scores = {
            role: _forecast_scores(raw_predictions[role], windows)
            for role, windows in windows_by_role.items()
            if role != "fit"
        }

        probes: Dict[str, ReducedRankActionProbe] = {}
        ridge_predictions: Dict[str, Dict[float, NDArray[np.float64]]] = {}
        selected_ridges = {}
        for name in CELL_NAMES:
            rows = []
            fitted = {}
            ridge_predictions[name] = {}
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
                safe = all(
                    scores[key] <= 1.05 * raw_scores["selection"][key]
                    for key in (
                        "overall_mse",
                        "action_overlap_mse",
                        "downstream_effect_mse",
                    )
                )
                rows.append({"ridge": ridge, "raw_safe": safe, **scores})
                fitted[ridge] = probe
                ridge_predictions[name][ridge] = prediction
            eligible = [row for row in rows if row["raw_safe"]]
            selected = min(
                eligible if eligible else rows,
                key=lambda row: (
                    row["downstream_effect_mse"],
                    row["ridge"],
                ),
            )
            selected_ridges[name] = float(selected["ridge"])
            probes[name] = fitted[selected_ridges[name]]
            _write_json(
                checkpoint_directory / f"{name}-probe.json",
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
                if role != "fit"
            }
            for name in CELL_NAMES
        }
        query_tokens = {
            name: _encode_delta_batches(
                model, transfer_queries.histories, fit.graph
            )
            for name, model in models.items()
        }
        attribution = {}
        attribution_predictions = {}
        action_sanity = {}
        action_sanity_predictions = {}
        for name in CELL_NAMES:
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
                role: model.diagnose_intervals(windows_by_role[role])
                for role in ("fit", "transfer_evaluation")
            }
            for name, model in models.items()
        }
        restored_arrays = {}
        for name in CELL_NAMES:
            restored_model = DeltaJepaModel.from_dict(models[name].to_dict())
            restored_probe = ReducedRankActionProbe.from_dict(
                probes[name].to_dict()
            )
            restored_representation = _encode_delta_batches(
                restored_model,
                windows_by_role["transfer_evaluation"].histories[:8],
                fit.graph,
            )
            restored_decoder = restored_model.diagnose_intervals(
                _window_subset(
                    windows_by_role["transfer_evaluation"],
                    np.arange(
                        min(
                            8,
                            len(
                                windows_by_role[
                                    "transfer_evaluation"
                                ].histories
                            ),
                        )
                    ),
                )
            ).predicted_actions
            restored_prediction = restored_probe.predict(
                restored_representation,
                windows_by_role["transfer_evaluation"].future_controls[:8],
                windows_by_role["transfer_evaluation"].future_actions[:8],
            )
            _, restored_attribution = _attribution_evidence(
                restored_probe,
                _encode_delta_batches(
                    restored_model,
                    transfer_queries.histories,
                    fit.graph,
                ),
                transfer_queries,
                ownership,
            )
            restored_arrays[name] = {
                "representation": restored_representation,
                "decoder": restored_decoder[:16],
                "probe_prediction": restored_prediction,
                "attribution_prediction": restored_attribution,
            }

        latency = {}
        latency_histories = windows_by_role[
            "transfer_evaluation"
        ].histories[:1]
        for name, model in models.items():
            samples = []
            model.encode(latency_histories, fit.graph)
            for _ in range(latency_repetitions):
                latency_started = time.perf_counter_ns()
                model.encode(latency_histories, fit.graph)
                samples.append(
                    (time.perf_counter_ns() - latency_started) / 1e6
                )
            values = np.asarray(samples, dtype=np.float64)
            latency[name] = {
                "mean_ms": float(np.mean(values)),
                "p95_ms": float(np.quantile(values, 0.95)),
                "repetitions": latency_repetitions,
            }

        public_causality = all(
            _rejects_forbidden_keyword(
                models["delta_jepa"].encode,
                fit.histories[:1],
                fit.graph,
                keyword,
                value,
            )
            for keyword, value in (
                ("future_states", fit.future_states[:1]),
                ("future_controls", fit.future_controls[:1]),
                ("future_actions", fit.future_actions[:1]),
            )
        )
        model_bytes = {
            name: len(_canonical_json_bytes(models[name].to_dict()))
            for name in CELL_NAMES
        }
        probe_bytes = {
            name: len(_canonical_json_bytes(probes[name].to_dict()))
            for name in CELL_NAMES
        }
        parameter_counts = {
            name: {
                "training": models[name].training_parameter_count,
                "inference": models[name].inference_parameter_count,
            }
            for name in CELL_NAMES
        }

        evidence: Dict[str, NDArray[Any]] = {}
        for role, windows in windows_by_role.items():
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
            evidence[
                f"representation__matched_pca__{role}"
            ] = pca_encoded[role].astype(np.float32)
            for name in CELL_NAMES:
                evidence[
                    f"representation__{name}__{role}"
                ] = encoded[name][role].astype(np.float32)
        for role in ("selection", "iid_evaluation", "transfer_evaluation"):
            evidence[f"raw_prediction__{role}"] = raw_predictions[
                role
            ].astype(np.float32)
            for name in CELL_NAMES:
                evidence[
                    f"prediction__{name}__{role}"
                ] = predictions[name][role].astype(np.float32)
        for name in CELL_NAMES:
            for ridge, values in ridge_predictions[name].items():
                evidence[
                    f"ridge_prediction__{name}__{ridge:.4g}"
                ] = values.astype(np.float32)
            evidence[f"attribution_prediction__{name}"] = (
                attribution_predictions[name].astype(np.float32)
            )
            for variant, values in action_sanity_predictions[name].items():
                evidence[
                    f"action_sanity__{name}__{variant}"
                ] = values.astype(np.float32)
            for role, diagnostic in diagnostics[name].items():
                _store_diagnostics(
                    evidence,
                    prefix=f"diagnostic__{name}__{role}",
                    diagnostic=diagnostic,
                )
            for field, values in restored_arrays[name].items():
                evidence[
                    f"restored_{field}__{name}"
                ] = np.asarray(values, dtype=np.float32)
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
        np.savez_compressed(building / "evidence.npz", **evidence)

        metadata = {
            "schema_version": 1,
            "kind": "delta_jepa_assessment_evidence",
            "interpretable": interpretable,
            "graph": fit.graph.to_dict(),
            "entity_names": list(fit.entity_names),
            "state_feature_names": list(fit.state_feature_names),
            "control_feature_names": list(fit.control_feature_names),
            "action_feature_names": list(fit.action_feature_names),
            "ownership_mask": ownership.astype(int).tolist(),
            "roles": {
                role: _role_identity(windows)
                for role, windows in windows_by_role.items()
            },
            "pair_counts": {
                role: len(set(windows.matched_pair_ids))
                for role, windows in windows_by_role.items()
            },
            "queries": {
                "query_ids": list(transfer_queries.query_ids),
                "candidate_ids": list(transfer_queries.candidate_ids),
                "candidate_action_kinds": list(
                    transfer_queries.candidate_action_kinds
                ),
                "candidate_target_entities": list(
                    transfer_queries.candidate_target_entities
                ),
                "expected_action_kinds": list(
                    transfer_queries.expected_action_kinds
                ),
                "expected_target_entities": list(
                    transfer_queries.expected_target_entities
                ),
                "expected_variant_ids": list(
                    transfer_queries.expected_variant_ids
                ),
            },
            "ridge_values": list(RIDGES),
            "selected_ridges_runner": selected_ridges,
            "parameter_counts": parameter_counts,
            "candidate_bundle_bytes": (
                model_bytes["delta_jepa"] + probe_bytes["delta_jepa"]
            ),
            "public_causality": public_causality,
        }
        _write_json(building / "evidence-metadata.json", metadata)
        assessment = assess_stored_bundle(building)
        _write_json(building / "assessment.json", assessment)
        report = {
            "schema_version": 1,
            "kind": "delta_jepa_action_displacement_tracer_v1",
            "evidence_boundary": (
                "single-seed open-development action-displacement tracer; "
                "not a production alert system or sealed confirmation"
            ),
            "interpretable": interpretable,
            "source": {
                "cache_directory": str(cache),
                "source_corpus_sha256": prepared.source_corpus_sha256,
                "source_artifact_manifest_sha256": (
                    prepared.source_artifact_manifest_sha256
                ),
                "preprocessing_protocol": prepared.preprocessing_protocol,
                "held_out_worker_topology_normalized": held_out_value,
            },
            "implementation": {
                "commit": implementation_commit,
                "sources": implementation_sources,
                "runtime": _runtime_identity(),
            },
            "configuration": {
                name: models[name].to_dict()["config"]
                for name in CELL_NAMES
            },
            "training_seconds": training_seconds,
            "selected_steps": {
                name: models[name].selected_step for name in CELL_NAMES
            },
            "parameter_counts": parameter_counts,
            "serialized_size_bytes": {
                name: {
                    "model": model_bytes[name],
                    "probe": probe_bytes[name],
                    "bundle": model_bytes[name] + probe_bytes[name],
                }
                for name in CELL_NAMES
            },
            "latency": latency,
            "peak_resident_memory_bytes": _peak_rss_bytes(),
            "elapsed_seconds": time.time() - started,
            "assessment": assessment,
        }
        _write_json(building / "result.json", report)
        (building / "REPORT.md").write_text(_render_report(report))
        _copy_reproduction_sources(building)
        write_artifact_manifest(building)
        building.rename(output)
        return output
    except BaseException as error:
        _write_json(
            building / "FAILURE.json",
            {
                "error_type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise


def _encode_delta_batches(
    model: DeltaJepaModel,
    histories: NDArray[np.float64],
    graph: Any,
) -> NDArray[np.float64]:
    return np.concatenate(
        [
            model.encode(histories[start : start + 256], graph).tokens
            for start in range(0, len(histories), 256)
        ],
        axis=0,
    )


def _store_diagnostics(
    evidence: Dict[str, NDArray[Any]],
    *,
    prefix: str,
    diagnostic: DeltaIntervalDiagnostics,
) -> None:
    pair_names = sorted(set(diagnostic.pair_ids))
    pair_positions = {name: index for index, name in enumerate(pair_names)}
    evidence[f"{prefix}__displacements"] = (
        diagnostic.displacements.astype(np.float32)
    )
    evidence[f"{prefix}__predicted_actions"] = (
        diagnostic.predicted_actions.astype(np.float32)
    )
    evidence[f"{prefix}__target_actions"] = (
        diagnostic.target_actions.astype(np.float32)
    )
    evidence[f"{prefix}__state_changes"] = (
        diagnostic.state_changes.astype(np.float32)
    )
    evidence[f"{prefix}__treatment_mask"] = (
        diagnostic.treatment_mask.astype(np.uint8)
    )
    evidence[f"{prefix}__pair_indices"] = np.asarray(
        [pair_positions[value] for value in diagnostic.pair_ids],
        dtype=np.int64,
    )


def _window_subset(windows: Any, indices: NDArray[np.int64]) -> Any:
    from quantis_core.action_conditioned_dynamics import (
        ActionConditionedWindows,
    )

    positions = np.asarray(indices, dtype=np.int64)
    return ActionConditionedWindows(
        histories=windows.histories[positions],
        future_states=windows.future_states[positions],
        future_controls=windows.future_controls[positions],
        future_actions=windows.future_actions[positions],
        trajectory_ids=tuple(windows.trajectory_ids[index] for index in positions),
        matched_pair_ids=tuple(
            windows.matched_pair_ids[index] for index in positions
        ),
        transition_indices=windows.transition_indices[positions],
        entity_names=windows.entity_names,
        state_feature_names=windows.state_feature_names,
        control_feature_names=windows.control_feature_names,
        action_feature_names=windows.action_feature_names,
        graph=windows.graph,
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


def _role_identity(windows: Any) -> Mapping[str, Any]:
    return {
        "trajectory_ids": list(windows.trajectory_ids),
        "matched_pair_ids": list(windows.matched_pair_ids),
        "transition_indices": [
            int(value) for value in windows.transition_indices
        ],
    }


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
                f"frozen Delta-JEPA source does not match {commit}: "
                f"{relative}"
            )
        result[relative] = {
            "path": relative,
            "bytes": len(local),
            "sha256": hashlib.sha256(local).hexdigest(),
            "matches_head": matches,
            "head_sha256": (
                None if head is None else hashlib.sha256(head).hexdigest()
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
            "kind": "delta_jepa_artifact_manifest",
            "files": files,
        },
    )


def _render_report(report: Mapping[str, Any]) -> str:
    assessment = report["assessment"]
    scores = assessment["forecast_scores"]
    mechanism = assessment["mechanism"]
    lines = [
        "# Delta-JEPA action-displacement tracer v1",
        "",
        (
            "Status: **frozen interpretable tracer**."
            if report["interpretable"]
            else "Status: **NON-INTERPRETABLE SMOKE RUN**."
        ),
        "",
        "| Cell | Selection effect MSE | Transfer effect MSE | "
        "Treatment action MSE | Retrieval |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in CELL_NAMES:
        lines.append(
            f"| {name} | "
            f"{scores[name]['selection']['downstream_effect_mse']:.6f} | "
            f"{scores[name]['transfer_evaluation']['downstream_effect_mse']:.6f} | "
            f"{mechanism[name]['action_reconstruction']['treatment_mse']:.6f} | "
            f"{mechanism[name]['action_sequence_retrieval']['treatment_correct_fraction']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Decision: `{assessment['decision']}`.",
            "",
        ]
    )
    return "\n".join(lines)


def _runtime_identity() -> Mapping[str, Any]:
    import torch

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "pid": os.getpid(),
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


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def main(arguments: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=FROZEN_CACHE)
    parser.add_argument("--output", type=Path, default=FROZEN_OUTPUT)
    parser.add_argument(
        "--pretrain-steps", type=int, default=FROZEN_PRETRAIN_STEPS
    )
    parser.add_argument("--latency-repetitions", type=int, default=100)
    parser.add_argument(
        "--allow-noninterpretable-smoke", action="store_true"
    )
    parsed = parser.parse_args(arguments)
    output = run_experiment(
        cache_directory=parsed.cache,
        output_directory=parsed.output,
        pretrain_steps=parsed.pretrain_steps,
        latency_repetitions=parsed.latency_repetitions,
        allow_noninterpretable_smoke=(
            parsed.allow_noninterpretable_smoke
        ),
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
