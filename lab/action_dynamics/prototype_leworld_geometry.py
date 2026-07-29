#!/usr/bin/env python3
"""Retained runner for the frozen ticket 017 geometry screen."""

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

from lab.action_dynamics.prototype_complete_lejepa import (
    _action_sanity_evidence,
    _attribution_evidence,
    _encode_histories,
    _forecast_scores,
    _transfer_queries,
)
from lab.action_dynamics.prototype_leworld_geometry_assessor import (
    CELL_NAMES,
    EVALUATION_ROLES,
    REGULARIZED_CELLS,
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
from quantis_core.edge_dynamics.leworld_geometry import (
    LeWorldGeometryConfig,
    LeWorldGeometryModel,
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
    "artifacts/action-dynamics/prototype-leworld-geometry-v1"
)
FROZEN_PRETRAIN_STEPS = 800
RIDGES = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
IMPLEMENTATION_SOURCE_PATHS = (
    "lab/action_dynamics/prototype_leworld_geometry.py",
    "lab/action_dynamics/prototype_leworld_geometry_assessor.py",
    "src/quantis_core/edge_dynamics/leworld_geometry.py",
    "tests/test_leworld_geometry.py",
    "docs/specs/leworld-geometry-screen-v1.md",
    "docs/research/leworld-geometry-primary-source-notes.md",
    "docs/wayfinding/jepa-implementation-program/"
    "017-test-leworld-geometry-screen.md",
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
    """Run, independently assess, and atomically publish one screen."""

    cache = Path(cache_directory).resolve()
    output = Path(output_directory).resolve()
    building = output.with_name(output.name + ".building")
    if output.exists() or building.exists():
        raise FileExistsError(
            "LeWorld geometry refuses an existing output or staging path"
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
            "non-frozen geometry runs require "
            "--allow-noninterpretable-smoke"
        )
    if not interpretable and output == (Path.cwd() / FROZEN_OUTPUT).resolve():
        raise ValueError("a smoke run cannot use the frozen result path")
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
            raise ValueError("held topology identity differs across roles")
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
            fit, seed=LeWorldGeometryConfig().seed + 1
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

        models: Dict[str, LeWorldGeometryModel] = {}
        training_seconds = {}
        model_directory = building / "models"
        model_directory.mkdir()
        for name in CELL_NAMES:
            config = LeWorldGeometryConfig(
                objective=name,
                pretrain_steps=pretrain_steps,
                checkpoint_interval=max(1, min(100, pretrain_steps)),
                expected_pair_count=expected_pair_count,
            )
            fit_started = time.perf_counter()
            model = LeWorldGeometryModel(config).fit(fit).select(
                windows_by_role["selection"]
            )
            training_seconds[name] = time.perf_counter() - fit_started
            models[name] = model
            _write_json(model_directory / f"{name}.json", model.to_dict())
            _print_progress(
                "fitted",
                {
                    "objective": name,
                    "seconds": training_seconds[name],
                    "selected_step": model.selected_step,
                },
            )

        encoded: Dict[str, Dict[str, np.ndarray]] = {}
        scene_history: Dict[str, Dict[str, np.ndarray]] = {}
        for name, model in models.items():
            encoded[name] = {}
            scene_history[name] = {}
            for role, windows in windows_by_role.items():
                tokens, scenes = _encode_geometry_batches(
                    model, windows.histories, fit.graph
                )
                encoded[name][role] = tokens
                scene_history[name][role] = scenes

        pca = EntityPcaRepresentation(width=32).fit(fit)
        pca_encoded = {
            role: _encode_histories(pca, windows.histories, fit.graph)
            for role, windows in windows_by_role.items()
        }
        _write_json(model_directory / "matched_pca.json", pca.to_dict())
        raw_model = ContractiveLowRankDynamics(
            LowRankConfig(rank=32)
        ).fit(fit)
        _write_json(
            model_directory / "raw_low_rank.json", raw_model.to_dict()
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
        ridge_predictions: Dict[str, Dict[float, np.ndarray]] = {}
        ridge_rows: Dict[str, Any] = {}
        selected_ridges = {}
        for name in CELL_NAMES:
            rows = []
            fitted = {}
            ridge_predictions[name] = {}
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
                eligible or rows,
                key=lambda row: (
                    row["downstream_effect_mse"],
                    row["ridge"],
                ),
            )
            ridge_rows[name] = rows
            selected_ridges[name] = float(selected["ridge"])
            probes[name] = fitted[selected_ridges[name]]
            _write_json(
                model_directory / f"{name}-probe.json",
                probes[name].to_dict(),
            )

        safe_cells = [
            name
            for name in REGULARIZED_CELLS
            if next(
                row
                for row in ridge_rows[name]
                if row["ridge"] == selected_ridges[name]
            )["raw_safe"]
        ]
        winner_pool = safe_cells or list(REGULARIZED_CELLS)
        screen_winner = min(
            winner_pool,
            key=lambda name: (
                next(
                    row
                    for row in ridge_rows[name]
                    if row["ridge"] == selected_ridges[name]
                )["downstream_effect_mse"],
                name,
            ),
        )
        winner_selection_safe = screen_winner in safe_cells

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
            name: _encode_geometry_batches(
                model, transfer_queries.histories, fit.graph
            )[0]
            for name, model in models.items()
        }
        attribution_predictions = {}
        action_sanity_predictions = {}
        for name in CELL_NAMES:
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

        restored = {}
        transfer = windows_by_role["transfer_evaluation"]
        for name in CELL_NAMES:
            restored_model = LeWorldGeometryModel.from_dict(
                models[name].to_dict()
            )
            restored_probe = ReducedRankActionProbe.from_dict(
                probes[name].to_dict()
            )
            restored_encoded = restored_model.encode(
                transfer.histories[:8], fit.graph
            )
            restored_prediction = restored_probe.predict(
                restored_encoded.tokens,
                transfer.future_controls[:8],
                transfer.future_actions[:8],
            )
            restored_query_tokens = _encode_geometry_batches(
                restored_model, transfer_queries.histories, fit.graph
            )[0]
            _, restored_attribution = _attribution_evidence(
                restored_probe,
                restored_query_tokens,
                transfer_queries,
                ownership,
            )
            restored[name] = {
                "representation": restored_encoded.tokens,
                "scene_history": restored_encoded.scene_history,
                "probe_prediction": restored_prediction,
                "attribution_prediction": restored_attribution,
            }

        latency = {}
        for name, model in models.items():
            samples = []
            model.encode(transfer.histories[:1], fit.graph)
            for _ in range(latency_repetitions):
                latency_started = time.perf_counter_ns()
                model.encode(transfer.histories[:1], fit.graph)
                samples.append(
                    (time.perf_counter_ns() - latency_started) / 1e6
                )
            values = np.asarray(samples)
            latency[name] = {
                "mean_ms": float(np.mean(values)),
                "p95_ms": float(np.quantile(values, 0.95)),
                "repetitions": latency_repetitions,
            }
        public_causality = all(
            _rejects_forbidden_keyword(
                model.encode,
                fit.histories[:1],
                fit.graph,
                keyword,
                value,
            )
            for model in models.values()
            for keyword, value in (
                ("future_states", fit.future_states[:1]),
                ("future_controls", fit.future_controls[:1]),
                ("future_actions", fit.future_actions[:1]),
            )
        )
        parameter_counts = {
            name: {
                "training": models[name].training_parameter_count,
                "inference": models[name].inference_parameter_count,
            }
            for name in CELL_NAMES
        }
        model_bytes = {
            name: len(_canonical_json_bytes(models[name].to_dict()))
            for name in CELL_NAMES
        }
        probe_bytes = {
            name: len(_canonical_json_bytes(probes[name].to_dict()))
            for name in CELL_NAMES
        }
        bundle_bytes = {
            name: model_bytes[name] + probe_bytes[name]
            for name in CELL_NAMES
        }

        evidence: Dict[str, np.ndarray] = {}
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
                evidence[
                    f"scene_history__{name}__{role}"
                ] = scene_history[name][role].astype(np.float32)
        for role in EVALUATION_ROLES:
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
            for field, values in restored[name].items():
                evidence[f"restored_{field}__{name}"] = np.asarray(
                    values, dtype=np.float32
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
        np.savez_compressed(building / "evidence.npz", **evidence)

        metadata = {
            "schema_version": 1,
            "kind": "leworld_geometry_assessment_evidence",
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
            "queries": _query_identity(transfer_queries),
            "ridge_values": list(RIDGES),
            "selected_ridges_runner": selected_ridges,
            "screen_winner_runner": screen_winner,
            "winner_selection_safe_runner": winner_selection_safe,
            "parameter_counts": parameter_counts,
            "bundle_bytes": bundle_bytes,
            "public_causality": public_causality,
        }
        _write_json(building / "evidence-metadata.json", metadata)
        assessment = assess_stored_bundle(building)
        _write_json(building / "assessment.json", assessment)
        report = {
            "schema_version": 1,
            "kind": "leworld_geometry_screen_v1",
            "evidence_boundary": (
                "single-seed open-development bounded geometry screen; "
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
                    "bundle": bundle_bytes[name],
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


def _encode_geometry_batches(
    model: LeWorldGeometryModel, histories: np.ndarray, graph: Any
) -> Tuple[np.ndarray, np.ndarray]:
    tokens = []
    scenes = []
    for start in range(0, len(histories), 256):
        encoded = model.encode(histories[start : start + 256], graph)
        tokens.append(encoded.tokens)
        scenes.append(encoded.scene_history)
    return np.concatenate(tokens), np.concatenate(scenes)


def _rejects_forbidden_keyword(
    function: Any,
    histories: np.ndarray,
    graph: Any,
    keyword: str,
    value: np.ndarray,
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


def _query_identity(queries: Any) -> Mapping[str, Any]:
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


def implementation_source_identity(
    *, commit: str, require_head_match: bool
) -> Mapping[str, Any]:
    result = {}
    for relative in IMPLEMENTATION_SOURCE_PATHS:
        local = Path(relative).read_bytes()
        head = _git_blob(commit, relative)
        matches = head == local
        if require_head_match and not matches:
            raise RuntimeError(
                f"frozen geometry source does not match {commit}: {relative}"
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
            "kind": "leworld_geometry_artifact_manifest",
            "files": files,
        },
    )


def _render_report(report: Mapping[str, Any]) -> str:
    assessment = report["assessment"]
    lines = [
        "# LeWorldModel bounded geometry screen v1",
        "",
        (
            "Status: **frozen interpretable screen**."
            if report["interpretable"]
            else "Status: **NON-INTERPRETABLE SMOKE RUN**."
        ),
        "",
        "| Cell | Selection effect MSE | Transfer effect MSE | "
        "Effective rank | State NRMSE |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in CELL_NAMES:
        selection_effect = assessment["forecast_scores"][name][
            "selection"
        ]["downstream_effect_mse"]
        transfer_effect = assessment["forecast_scores"][name][
            "transfer_evaluation"
        ]["downstream_effect_mse"]
        lines.append(
            f"| {name} | "
            f"{selection_effect:.6f} | "
            f"{transfer_effect:.6f} | "
            f"{assessment['geometry'][name]['effective_rank']:.3f} | "
            f"{assessment['state_probes'][name]['aggregate_nrmse']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"Winner: `{assessment['screen_winner']}`.",
            "",
            f"Decision: `{assessment['decision']}`.",
            "",
            (
                "UR-JEPA prerequisite met: "
                f"`{str(assessment['ur_jepa_prerequisite_met']).lower()}`."
            ),
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
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def _print_progress(name: str, value: Mapping[str, Any]) -> None:
    print(
        json.dumps({"stage": name, **dict(value)}, sort_keys=True),
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
