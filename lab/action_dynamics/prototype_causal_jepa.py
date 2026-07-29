#!/usr/bin/env python3
"""Retained runner for the frozen ticket 018 Causal-JEPA tracer."""

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
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from lab.action_dynamics.prototype_complete_lejepa import (
    _forecast_scores,
    _transfer_queries,
)
from lab.action_dynamics.prototype_causal_jepa_assessor import (
    CELL_NAMES,
    EVALUATION_ROLES,
    assess_stored_bundle,
)
from quantis_core.action_conditioned_dynamics import ActionConditionedWindows
from quantis_core.edge_dynamics.causal_jepa import (
    CausalJepaConfig,
    CausalJepaModel,
    causal_mask_plan,
)
from quantis_core.edge_dynamics.complete_lejepa import (
    PairBlockedAnchorSchedule,
    fit_owned_feature_mask,
)
from quantis_core.edge_dynamics.data import (
    load_edge_dynamics_cache,
    partition_worker_topology,
)
from quantis_core.edge_dynamics.models import (
    ContractiveLowRankDynamics,
    LowRankConfig,
)


FROZEN_CACHE = Path(
    "artifacts/action-dynamics/edge-preprocessing-v1/"
    "eb54271132f88c9a431b01e786ea66279a563776434cca2290e47e6b7ae9b3ff"
)
FROZEN_OUTPUT = Path("artifacts/action-dynamics/prototype-causal-jepa-v1")
FROZEN_PRETRAIN_STEPS = 1200
IMPLEMENTATION_SOURCE_PATHS = (
    "lab/action_dynamics/prototype_causal_jepa.py",
    "lab/action_dynamics/prototype_causal_jepa_assessor.py",
    "src/quantis_core/edge_dynamics/causal_jepa.py",
    "tests/test_causal_jepa.py",
    "docs/specs/causal-jepa-entity-intervention-v1.md",
    "docs/research/causal-jepa-primary-source-notes.md",
    "docs/research/causal-jepa-attempt-1-restore-boundary.md",
    "docs/wayfinding/jepa-implementation-program/"
    "018-test-causal-jepa-entity-intervention.md",
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
            "Causal-JEPA refuses an existing output or staging path"
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
            "non-frozen Causal-JEPA runs require "
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
            raise ValueError("Causal-JEPA held topology differs")
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
        base_config = CausalJepaConfig()
        anchor_schedule = PairBlockedAnchorSchedule(
            fit, seed=base_config.seed + 1
        )
        anchors = [
            anchor_schedule.batch(step) for step in range(pretrain_steps)
        ]
        np.savez_compressed(
            building / "anchor-schedule.npz",
            indices=np.stack([batch.indices for batch in anchors]),
            arm_ids=np.stack([batch.arm_ids for batch in anchors]),
            transition_indices=np.stack(
                [batch.transition_indices for batch in anchors]
            ),
            pair_ids=np.asarray(anchor_schedule.pair_ids),
        )
        np.savez_compressed(
            building / "mask-schedule.npz",
            **{
                name: np.stack(
                    [
                        causal_mask_plan(
                            name,
                            step=step,
                            entity_count=len(fit.entity_names),
                            seed=base_config.seed + 2,
                        )
                        for step in range(pretrain_steps)
                    ]
                )
                for name in CELL_NAMES
            },
        )

        model_directory = building / "models"
        model_directory.mkdir()
        models: Dict[str, CausalJepaModel] = {}
        training_seconds = {}
        for name in CELL_NAMES:
            config = CausalJepaConfig(
                objective=name,
                pretrain_steps=pretrain_steps,
                checkpoint_interval=max(1, min(200, pretrain_steps)),
                expected_pair_count=expected_pair_count,
            )
            fit_started = time.perf_counter()
            model = CausalJepaModel(config).fit(fit).select(
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

        raw_model = ContractiveLowRankDynamics(
            LowRankConfig(rank=32)
        ).fit(fit)
        _write_json(
            model_directory / "raw_low_rank.json", raw_model.to_dict()
        )
        predictions = {
            name: {
                role: _predict_batches(model, windows)
                for role, windows in windows_by_role.items()
                if role != "fit"
            }
            for name, model in models.items()
        }
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
        completion = {
            name: model.complete_masked_histories(
                windows_by_role["transfer_evaluation"]
            ).predictions
            for name, model in models.items()
        }
        attribution_predictions = {
            name: _attribution_predictions(
                model, transfer_queries, fit.graph
            )
            for name, model in models.items()
        }
        action_sanity_predictions = {
            name: _action_sanity_predictions(
                model, windows_by_role["transfer_evaluation"]
            )
            for name, model in models.items()
        }

        transfer = windows_by_role["transfer_evaluation"]
        restored = {}
        for name in CELL_NAMES:
            restored_model = CausalJepaModel.from_dict(
                models[name].to_dict()
            )
            restored[name] = {
                "prediction": restored_model.predict(
                    transfer.histories[:8],
                    transfer.future_controls[:8],
                    transfer.future_actions[:8],
                    fit.graph,
                ),
                "completion": restored_model.complete_masked_histories(
                    transfer
                ).predictions[:8],
                "attribution_prediction": _attribution_predictions(
                    restored_model, transfer_queries, fit.graph
                ),
            }

        latency = {}
        for name, model in models.items():
            model.predict(
                transfer.histories[:1],
                transfer.future_controls[:1],
                transfer.future_actions[:1],
                fit.graph,
            )
            samples = []
            for _ in range(latency_repetitions):
                latency_started = time.perf_counter_ns()
                model.predict(
                    transfer.histories[:1],
                    transfer.future_controls[:1],
                    transfer.future_actions[:1],
                    fit.graph,
                )
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
            _rejects_future_states(model, fit) for model in models.values()
        )
        parameter_counts = {
            name: model.training_parameter_count
            for name, model in models.items()
        }
        model_bytes = {
            name: len(_canonical_json_bytes(model.to_dict()))
            for name, model in models.items()
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
        for role in EVALUATION_ROLES:
            evidence[f"raw_prediction__{role}"] = raw_predictions[
                role
            ].astype(np.float32)
            for name in CELL_NAMES:
                evidence[
                    f"prediction__{name}__{role}"
                ] = predictions[name][role].astype(np.float32)
        for name in CELL_NAMES:
            evidence[f"completion_prediction__{name}"] = completion[
                name
            ].astype(np.float32)
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
            "kind": "causal_jepa_assessment_evidence",
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
            "selected_steps": {
                name: model.selected_step for name, model in models.items()
            },
            "selection_metrics": {
                name: list(model.selection_metrics)
                for name, model in models.items()
            },
            "parameter_counts": parameter_counts,
            "model_bytes": model_bytes,
            "public_causality": public_causality,
        }
        _write_json(building / "evidence-metadata.json", metadata)
        assessment = assess_stored_bundle(building)
        _write_json(building / "assessment.json", assessment)
        report = {
            "schema_version": 1,
            "kind": "causal_jepa_entity_intervention_v1",
            "evidence_boundary": (
                "single-seed open-development observability-intervention "
                "tracer; not causal identification or production evidence"
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
                name: model.to_dict()["config"]
                for name, model in models.items()
            },
            "training_seconds": training_seconds,
            "selected_steps": {
                name: model.selected_step for name, model in models.items()
            },
            "parameter_counts": parameter_counts,
            "serialized_size_bytes": model_bytes,
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


def _predict_batches(
    model: CausalJepaModel, windows: ActionConditionedWindows
) -> np.ndarray:
    return np.concatenate(
        [
            model.predict(
                windows.histories[start : start + 128],
                windows.future_controls[start : start + 128],
                windows.future_actions[start : start + 128],
                windows.graph,
            )
            for start in range(0, len(windows.histories), 128)
        ]
    )


def _attribution_predictions(
    model: CausalJepaModel, queries: Any, graph: Any
) -> np.ndarray:
    results = []
    candidate_count = len(queries.candidate_ids)
    for index in range(len(queries.query_ids)):
        results.append(
            model.predict(
                np.repeat(
                    queries.histories[index : index + 1],
                    candidate_count,
                    axis=0,
                ),
                np.repeat(
                    queries.future_controls[index : index + 1],
                    candidate_count,
                    axis=0,
                ),
                queries.candidate_actions[index],
                graph,
            )
        )
    return np.stack(results)


def _action_sanity_predictions(
    model: CausalJepaModel, windows: ActionConditionedWindows
) -> Mapping[str, np.ndarray]:
    correct = _predict_batches(model, windows)
    no_action = np.zeros_like(windows.future_actions)
    no_action[..., 0] = 1.0
    absent = model.predict(
        windows.histories,
        windows.future_controls,
        no_action,
        windows.graph,
    )
    pair_ids = sorted(set(windows.matched_pair_ids))
    pair_array = np.asarray(windows.matched_pair_ids)
    shuffled_actions = np.zeros_like(windows.future_actions)
    for position, pair in enumerate(pair_ids):
        rows = np.flatnonzero(pair_array == pair)
        donor = pair_ids[(position + 1) % len(pair_ids)]
        donor_rows = np.flatnonzero(pair_array == donor)
        shuffled_actions[rows] = windows.future_actions[
            donor_rows[: len(rows)]
        ]
    shuffled = model.predict(
        windows.histories,
        windows.future_controls,
        shuffled_actions,
        windows.graph,
    )
    return {
        "correct": correct,
        "no_action": absent,
        "shuffled": shuffled,
    }


def _window_subset(
    windows: ActionConditionedWindows, indices: np.ndarray
) -> ActionConditionedWindows:
    positions = np.asarray(indices, dtype=np.int64)
    return ActionConditionedWindows(
        histories=windows.histories[positions],
        future_states=windows.future_states[positions],
        future_controls=windows.future_controls[positions],
        future_actions=windows.future_actions[positions],
        trajectory_ids=tuple(
            windows.trajectory_ids[index] for index in positions
        ),
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


def _rejects_future_states(
    model: CausalJepaModel, fit: ActionConditionedWindows
) -> bool:
    try:
        model.predict(
            fit.histories[:1],
            fit.future_controls[:1],
            fit.future_actions[:1],
            fit.graph,
            future_states=fit.future_states[:1],  # type: ignore[call-arg]
        )
    except TypeError:
        return True
    return False


def _role_identity(windows: ActionConditionedWindows) -> Mapping[str, Any]:
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
                f"frozen Causal-JEPA source differs from {commit}: {relative}"
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
            "kind": "causal_jepa_artifact_manifest",
            "files": files,
        },
    )


def _render_report(report: Mapping[str, Any]) -> str:
    assessment = report["assessment"]
    lines = [
        "# Causal-JEPA entity-intervention tracer v1",
        "",
        (
            "Status: **frozen interpretable tracer**."
            if report["interpretable"]
            else "Status: **NON-INTERPRETABLE SMOKE RUN**."
        ),
        "",
        "| Cell | Selection effect MSE | Transfer effect MSE | "
        "Completion MSE |",
        "|---|---:|---:|---:|",
    ]
    for name in CELL_NAMES:
        selection = assessment["forecast_scores"][name]["selection"][
            "downstream_effect_mse"
        ]
        transfer = assessment["forecast_scores"][name][
            "transfer_evaluation"
        ]["downstream_effect_mse"]
        completion = assessment["completion"][name]["overall_mse"]
        lines.append(
            f"| {name} | {selection:.6f} | {transfer:.6f} | "
            f"{completion:.6f} |"
        )
    lines.extend(
        ["", f"Decision: `{assessment['decision']}`.", ""]
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
