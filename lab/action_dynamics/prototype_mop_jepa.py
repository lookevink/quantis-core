#!/usr/bin/env python3
"""Retained runner for the frozen ticket 019 MoP-JEPA tracer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from lab.action_dynamics.prototype_mop_jepa_assessor import (
    EVALUATION_ROLES,
    MODEL_NAMES,
    assess_stored_bundle,
)
from quantis_core.action_conditioned_dynamics import (
    ActionConditionedWindows,
    MixtureTrajectoryDistribution,
)
from quantis_core.edge_dynamics.data import (
    load_edge_dynamics_cache,
    partition_worker_topology,
)
from quantis_core.edge_dynamics.models import (
    ContractiveLowRankDynamics,
    LowRankConfig,
)
from quantis_core.edge_dynamics.mop_jepa import (
    ContextFreeCodebookConfig,
    ContextFreeTrajectoryCodebook,
    MopJepaConfig,
    MopJepaModel,
)


FROZEN_CACHE = Path(
    "artifacts/action-dynamics/edge-preprocessing-v1/"
    "eb54271132f88c9a431b01e786ea66279a563776434cca2290e47e6b7ae9b3ff"
)
FROZEN_OUTPUT = Path("artifacts/action-dynamics/prototype-mop-jepa-v1")
FROZEN_EPOCHS = 40
FROZEN_CODEBOOK_ITERATIONS = 20
IMPLEMENTATION_SOURCE_PATHS = (
    "lab/action_dynamics/prototype_mop_jepa.py",
    "lab/action_dynamics/prototype_mop_jepa_assessor.py",
    "src/quantis_core/edge_dynamics/mop_jepa.py",
    "tests/test_mop_jepa.py",
    "tests/test_action_conditioned_dynamics.py",
    "tests/test_sd_jepa.py",
    "docs/specs/mop-jepa-hard-assignment-v1.md",
    "docs/research/mop-jepa-primary-source-notes.md",
    "docs/wayfinding/jepa-implementation-program/"
    "019-test-mop-jepa-hard-assignment.md",
    "src/quantis_core/action_conditioned_dynamics.py",
    "src/quantis_core/edge_dynamics/data.py",
    "src/quantis_core/edge_dynamics/models.py",
    "src/quantis_core/graph_telemetry.py",
    "lab/action_dynamics/prototype_complete_lejepa.py",
)


def run_experiment(
    *,
    cache_directory: Path,
    output_directory: Path,
    epochs: int,
    codebook_iterations: int,
    latency_repetitions: int,
    allow_noninterpretable_smoke: bool,
    evaluation_pair_limit: Optional[int] = None,
) -> Path:
    """Run, independently assess, and atomically publish one tracer."""

    cache = Path(cache_directory).resolve()
    output = Path(output_directory).resolve()
    building = output.with_name(output.name + ".building")
    if output.exists() or building.exists():
        raise FileExistsError(
            "MoP-JEPA refuses an existing output or staging path"
        )
    interpretable = (
        cache == (Path.cwd() / FROZEN_CACHE).resolve()
        and output == (Path.cwd() / FROZEN_OUTPUT).resolve()
        and epochs == FROZEN_EPOCHS
        and codebook_iterations == FROZEN_CODEBOOK_ITERATIONS
        and latency_repetitions == 100
        and evaluation_pair_limit is None
    )
    if not interpretable and not allow_noninterpretable_smoke:
        raise ValueError(
            "non-frozen MoP-JEPA runs require "
            "--allow-noninterpretable-smoke"
        )
    if not interpretable and output == (Path.cwd() / FROZEN_OUTPUT).resolve():
        raise ValueError("a smoke run cannot use the frozen result path")
    if evaluation_pair_limit is not None and evaluation_pair_limit < 1:
        raise ValueError("evaluation pair limit must be positive")
    implementation_commit = _git_head()
    implementation_sources = implementation_source_identity(
        commit=implementation_commit,
        require_head_match=interpretable,
    )
    building.mkdir(parents=True)
    started = time.time()
    try:
        _copy_reproduction_sources(
            building,
            implementation_sources,
            commit=implementation_commit if interpretable else None,
        )
        prepared = load_edge_dynamics_cache(cache)
        partitions = {
            role: partition_worker_topology(windows)
            for role, windows in prepared.windows.items()
        }
        held_out_values = {
            value.held_out_normalized_value for value in partitions.values()
        }
        if len(held_out_values) != 1:
            raise ValueError("MoP-JEPA held topology differs")
        held_out_value = next(iter(held_out_values))
        fit = partitions["fit"].in_distribution
        windows_by_role = {
            "calibration": partitions["calibration"].in_distribution,
            "selection": partitions["selection"].in_distribution,
            "transfer_evaluation": partitions["evaluation"].held_out,
        }
        if evaluation_pair_limit is not None:
            windows_by_role = {
                role: _limit_pairs(windows, evaluation_pair_limit)
                for role, windows in windows_by_role.items()
            }

        model_directory = building / "models"
        model_directory.mkdir()
        configs = {
            "mop_jepa": MopJepaConfig(
                objective="mop_jepa", head_count=8, epochs=epochs
            ),
            "dense_jepa": MopJepaConfig(
                objective="dense_jepa", head_count=1, epochs=epochs
            ),
            "supervised_hard_wta": MopJepaConfig(
                objective="supervised_hard_wta",
                head_count=8,
                epochs=epochs,
            ),
        }
        models: Dict[str, Any] = {}
        training_seconds: Dict[str, float] = {}
        model_bytes: Dict[str, int] = {}
        for name, config in configs.items():
            fit_started = time.perf_counter()
            model = MopJepaModel(config).fit(fit).calibrate(
                windows_by_role["calibration"]
            )
            training_seconds[name] = time.perf_counter() - fit_started
            models[name] = model
            model_bytes[name] = model.save(model_directory, name)
            _progress(
                "fitted",
                {
                    "cell": name,
                    "seconds": training_seconds[name],
                    "assignment_count": (
                        model.calibration_assignment_count.tolist()
                    ),
                },
            )

        codebook_started = time.perf_counter()
        codebook = ContextFreeTrajectoryCodebook(
            ContextFreeCodebookConfig(
                component_count=8,
                iterations=codebook_iterations,
            )
        ).fit(fit).calibrate(windows_by_role["calibration"])
        training_seconds["context_free_codebook"] = (
            time.perf_counter() - codebook_started
        )
        models["context_free_codebook"] = codebook
        model_bytes["context_free_codebook"] = codebook.save(
            model_directory, "context_free_codebook"
        )

        raw_started = time.perf_counter()
        raw = ContractiveLowRankDynamics(
            LowRankConfig(rank=32)
        ).fit(fit)
        raw_calibration_windows = windows_by_role["calibration"]
        raw_calibration_prediction = raw.rollout(
            raw_calibration_windows.histories,
            raw_calibration_windows.future_controls,
            raw_calibration_windows.future_actions,
            raw_calibration_windows.graph,
        ).mean
        raw_calibrated_variance = np.maximum(
            np.mean(
                np.square(
                    raw_calibration_prediction
                    - raw_calibration_windows.future_states
                ),
                axis=0,
            ),
            1e-4,
        ).astype(np.float64)
        training_seconds["raw_low_rank"] = (
            time.perf_counter() - raw_started
        )
        models["raw_low_rank"] = raw
        raw_payload = raw.to_dict()
        raw_path = model_directory / "raw_low_rank.json"
        _write_json(raw_path, raw_payload)
        raw_calibration_path = (
            model_directory / "raw_low_rank_calibration.npz"
        )
        np.savez_compressed(
            raw_calibration_path,
            component_variance=raw_calibrated_variance,
        )
        model_bytes["raw_low_rank"] = (
            raw_path.stat().st_size + raw_calibration_path.stat().st_size
        )

        evidence: Dict[str, np.ndarray] = {}
        for role, windows in windows_by_role.items():
            _store_windows(evidence, role, windows)
            for name in MODEL_NAMES:
                distribution = _rollout(
                    models[name],
                    windows,
                    raw_calibrated_variance=raw_calibrated_variance,
                )
                _store_distribution(evidence, name, role, distribution)
                _progress(
                    "predicted",
                    {"cell": name, "role": role, "rows": len(windows.histories)},
                )
        for role in EVALUATION_ROLES:
            windows = windows_by_role[role]
            permutation = np.random.default_rng(
                19019 + (1 if role == "selection" else 2)
            ).permutation(len(windows.histories))
            shuffled = models["mop_jepa"].rollout(
                windows.histories[permutation],
                windows.future_controls[permutation],
                windows.future_actions[permutation],
            )
            _store_distribution(
                evidence, "mop_jepa_shuffled", role, shuffled
            )

        restored_models = {
            "mop_jepa": MopJepaModel.load(model_directory, "mop_jepa"),
            "dense_jepa": MopJepaModel.load(
                model_directory, "dense_jepa"
            ),
            "supervised_hard_wta": MopJepaModel.load(
                model_directory, "supervised_hard_wta"
            ),
            "context_free_codebook": (
                ContextFreeTrajectoryCodebook.load(
                    model_directory, "context_free_codebook"
                )
            ),
            "raw_low_rank": ContractiveLowRankDynamics.from_dict(
                raw_payload
            ),
        }
        with np.load(
            raw_calibration_path, allow_pickle=False
        ) as restored_raw_calibration:
            restored_raw_variance = np.asarray(
                restored_raw_calibration["component_variance"],
                dtype=np.float64,
            )
        selection = windows_by_role["selection"]
        restore_count = min(8, len(selection.histories))
        restore_windows = _subset_rows(
            selection, np.arange(restore_count)
        )
        for name in MODEL_NAMES:
            distribution = _rollout(
                restored_models[name],
                restore_windows,
                raw_calibrated_variance=(
                    restored_raw_variance
                    if name == "raw_low_rank"
                    else raw_calibrated_variance
                ),
            )
            evidence[f"restored_mean__{name}"] = (
                distribution.component_mean.astype(np.float32)
            )
            evidence[f"restored_weight__{name}"] = (
                distribution.weight.astype(np.float32)
            )
            evidence[f"restored_variance__{name}"] = (
                distribution.component_variance[0].astype(np.float32)
            )

        np.savez_compressed(building / "evidence.npz", **evidence)

        latency = {
            name: _latency(
                models[name],
                windows_by_role["selection"],
                latency_repetitions,
                raw_calibrated_variance,
            )
            for name in MODEL_NAMES
        }
        parameter_counts = {
            "mop_jepa": models["mop_jepa"].training_parameter_count,
            "dense_jepa": models["dense_jepa"].training_parameter_count,
            "supervised_hard_wta": (
                models["supervised_hard_wta"].training_parameter_count
            ),
            "context_free_codebook": codebook.parameter_count,
            "raw_low_rank": raw.parameter_count,
        }
        public_causality = all(
            _rejects_future_states(models[name], fit)
            for name in (
                "mop_jepa",
                "dense_jepa",
                "supervised_hard_wta",
                "context_free_codebook",
            )
        )
        metadata = {
            "schema_version": 1,
            "kind": "mop_jepa_assessment_evidence",
            "interpretable": interpretable,
            "graph": fit.graph.to_dict(),
            "entity_names": list(fit.entity_names),
            "state_feature_names": list(fit.state_feature_names),
            "control_feature_names": list(fit.control_feature_names),
            "action_feature_names": list(fit.action_feature_names),
            "roles": {
                role: _role_identity(windows)
                for role, windows in windows_by_role.items()
            },
            "pair_counts": {
                role: len(set(windows.matched_pair_ids))
                for role, windows in windows_by_role.items()
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
            "kind": "mop_jepa_hard_assignment_v1",
            "evidence_boundary": (
                "single-seed open-development hard-assignment tracer; "
                "realized-transition precision is not an environment oracle"
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
                **{
                    name: {
                        key: value
                        for key, value in vars(config).items()
                    }
                    for name, config in configs.items()
                },
                "context_free_codebook": {
                    key: value
                    for key, value in vars(codebook.config).items()
                },
                "raw_low_rank": raw_payload["config"],
            },
            "training_seconds": training_seconds,
            "parameter_counts": parameter_counts,
            "serialized_size_bytes": model_bytes,
            "latency": latency,
            "peak_resident_memory_bytes": _peak_rss_bytes(),
            "elapsed_seconds": time.time() - started,
            "assessment": assessment,
        }
        _write_json(building / "result.json", report)
        (building / "REPORT.md").write_text(_render_report(report))
        if interpretable:
            if _git_head() != implementation_commit:
                raise RuntimeError(
                    "repository HEAD changed during frozen MoP-JEPA run"
                )
            implementation_source_identity(
                commit=implementation_commit,
                require_head_match=True,
            )
        _verify_reproduction_sources(building, implementation_sources)
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


def _rollout(
    model: Any,
    windows: ActionConditionedWindows,
    *,
    raw_calibrated_variance: Optional[np.ndarray] = None,
) -> MixtureTrajectoryDistribution:
    if isinstance(model, ContractiveLowRankDynamics):
        distribution = model.rollout(
            windows.histories,
            windows.future_controls,
            windows.future_actions,
            windows.graph,
        )
        if raw_calibrated_variance is None:
            raise ValueError("raw calibrated variance is required")
        return MixtureTrajectoryDistribution(
            component_mean=distribution.mean[:, None],
            component_variance=np.broadcast_to(
                raw_calibrated_variance[None, None],
                (len(windows.histories), 1)
                + raw_calibrated_variance.shape,
            ).copy(),
            weight=np.ones((len(windows.histories), 1)),
        )
    return model.rollout(
        windows.histories,
        windows.future_controls,
        windows.future_actions,
    )


def _store_distribution(
    evidence: Dict[str, np.ndarray],
    name: str,
    role: str,
    distribution: MixtureTrajectoryDistribution,
) -> None:
    reference = distribution.component_variance[0]
    if not np.allclose(
        distribution.component_variance,
        reference[None],
        atol=1e-12,
        rtol=1e-12,
    ):
        raise ValueError(f"{name} variance is not context-independent")
    evidence[f"mean__{name}__{role}"] = (
        distribution.component_mean.astype(np.float32)
    )
    evidence[f"weight__{name}__{role}"] = (
        distribution.weight.astype(np.float32)
    )
    key = f"variance__{name}"
    observed = reference.astype(np.float32)
    if key in evidence and not np.array_equal(evidence[key], observed):
        raise ValueError(f"{name} variance differs across roles")
    evidence[key] = observed


def _store_windows(
    evidence: Dict[str, np.ndarray],
    role: str,
    windows: ActionConditionedWindows,
) -> None:
    evidence[f"histories__{role}"] = windows.histories.astype(np.float32)
    evidence[f"target__{role}"] = windows.future_states.astype(np.float32)
    evidence[f"controls__{role}"] = (
        windows.future_controls.astype(np.float32)
    )
    evidence[f"actions__{role}"] = (
        windows.future_actions.astype(np.float32)
    )


def _limit_pairs(
    windows: ActionConditionedWindows, count: int
) -> ActionConditionedWindows:
    chosen = set(sorted(set(windows.matched_pair_ids))[:count])
    indices = np.asarray(
        [
            index
            for index, pair in enumerate(windows.matched_pair_ids)
            if pair in chosen
        ],
        dtype=np.int64,
    )
    return _subset_rows(windows, indices)


def _subset_rows(
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


def _latency(
    model: Any,
    windows: ActionConditionedWindows,
    repetitions: int,
    raw_calibrated_variance: np.ndarray,
) -> Mapping[str, float]:
    sample = _subset_rows(windows, np.asarray([0]))
    _rollout(
        model,
        sample,
        raw_calibrated_variance=raw_calibrated_variance,
    )
    values = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        _rollout(
            model,
            sample,
            raw_calibrated_variance=raw_calibrated_variance,
        )
        values.append((time.perf_counter_ns() - started) / 1e6)
    array = np.asarray(values)
    return {
        "mean_ms": float(np.mean(array)),
        "p95_ms": float(np.quantile(array, 0.95)),
        "repetitions": repetitions,
    }


def _rejects_future_states(
    model: Any, fit: ActionConditionedWindows
) -> bool:
    try:
        model.rollout(
            fit.histories[:1],
            fit.future_controls[:1],
            fit.future_actions[:1],
            future_states=fit.future_states[:1],
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
                f"frozen MoP-JEPA source differs from {commit}: {relative}"
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


def _copy_reproduction_sources(
    building: Path,
    identities: Mapping[str, Any],
    *,
    commit: Optional[str],
) -> None:
    root = building / "reproduction-sources"
    for relative in IMPLEMENTATION_SOURCE_PATHS:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        content = (
            Path(relative).read_bytes()
            if commit is None
            else _git_blob(commit, relative)
        )
        if content is None:
            raise RuntimeError(
                f"committed MoP-JEPA source is missing: {relative}"
            )
        destination.write_bytes(content)
        if hashlib.sha256(content).hexdigest() != str(
            identities[relative]["sha256"]
        ):
            raise RuntimeError(
                f"MoP-JEPA source changed during snapshot: {relative}"
            )


def _verify_reproduction_sources(
    building: Path, identities: Mapping[str, Any]
) -> None:
    root = building / "reproduction-sources"
    for relative in IMPLEMENTATION_SOURCE_PATHS:
        if _file_sha256(root / relative) != str(
            identities[relative]["sha256"]
        ):
            raise RuntimeError(
                f"MoP-JEPA reproduction snapshot differs: {relative}"
            )


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
            "kind": "mop_jepa_artifact_manifest",
            "files": files,
        },
    )


def _render_report(report: Mapping[str, Any]) -> str:
    assessment = report["assessment"]
    lines = [
        "# MoP-JEPA hard-assignment tracer v1",
        "",
        (
            "Status: **frozen interpretable tracer**."
            if report["interpretable"]
            else "Status: **NON-INTERPRETABLE SMOKE RUN**."
        ),
        "",
        "| Cell | Selection NLL | Point MSE | Oracle MSE | "
        "Gated precision |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in MODEL_NAMES:
        row = assessment["metrics"]["selection"][name]
        lines.append(
            f"| {name} | {row['trajectory_nll']:.6f} | "
            f"{row['point_overall_mse']:.6f} | "
            f"{row['oracle_mse']:.6f} | "
            f"{row['gated_realized_transition_precision']:.2%} |"
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


def _progress(stage: str, values: Mapping[str, Any]) -> None:
    print(
        json.dumps({"stage": stage, **dict(values)}, sort_keys=True),
        flush=True,
    )


def main(arguments: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=FROZEN_CACHE)
    parser.add_argument("--output", type=Path, default=FROZEN_OUTPUT)
    parser.add_argument("--epochs", type=int, default=FROZEN_EPOCHS)
    parser.add_argument(
        "--codebook-iterations",
        type=int,
        default=FROZEN_CODEBOOK_ITERATIONS,
    )
    parser.add_argument("--latency-repetitions", type=int, default=100)
    parser.add_argument("--evaluation-pair-limit", type=int)
    parser.add_argument(
        "--allow-noninterpretable-smoke", action="store_true"
    )
    parsed = parser.parse_args(arguments)
    output = run_experiment(
        cache_directory=parsed.cache,
        output_directory=parsed.output,
        epochs=parsed.epochs,
        codebook_iterations=parsed.codebook_iterations,
        latency_repetitions=parsed.latency_repetitions,
        allow_noninterpretable_smoke=(
            parsed.allow_noninterpretable_smoke
        ),
        evaluation_pair_limit=parsed.evaluation_pair_limit,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
