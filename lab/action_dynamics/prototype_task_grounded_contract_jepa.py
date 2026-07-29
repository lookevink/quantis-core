#!/usr/bin/env python3
"""Retained runner for task-grounded Contract-JEPA."""

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from lab.action_dynamics import (
    prototype_pair_effect_jepa_assessor as shared,
)
from lab.action_dynamics.prototype_task_grounded_contract_jepa_assessor import (
    CELL_NAMES,
    assess_stored_bundle,
)
from quantis_core.action_conditioned_dynamics import (
    ActionConditionedWindows,
)
from quantis_core.edge_dynamics.complete_lejepa import (
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
from quantis_core.edge_dynamics.task_grounded_contract_jepa import (
    TaskGroundedContractConfig,
    TaskGroundedContractDynamics,
    TaskGroundedContractJepa,
)


FROZEN_CACHE = Path(
    "artifacts/action-dynamics/edge-preprocessing-v1/"
    "eb54271132f88c9a431b01e786ea66279a563776434cca2290e47e6b7ae9b3ff"
)
FROZEN_OUTPUT = Path(
    "artifacts/action-dynamics/prototype-task-grounded-contract-jepa-v2"
)
FROZEN_PRETRAIN_STEPS = 800
GAIN_CANDIDATES = (0.0, 0.25, 0.5, 0.75, 1.0)
IMPLEMENTATION_SOURCE_PATHS = (
    "lab/action_dynamics/prototype_task_grounded_contract_jepa.py",
    "lab/action_dynamics/"
    "prototype_task_grounded_contract_jepa_assessor.py",
    "lab/action_dynamics/prototype_pair_effect_jepa_assessor.py",
    "src/quantis_core/edge_dynamics/task_grounded_contract_jepa.py",
    "tests/test_task_grounded_contract_jepa.py",
    "docs/specs/task-grounded-contract-jepa-tracer-v1.md",
    "docs/wayfinding/jepa-implementation-program/"
    "021-test-task-grounded-contract-jepa.md",
    "src/quantis_core/edge_dynamics/models.py",
    "src/quantis_core/edge_dynamics/data.py",
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
    """Run, independently assess, and publish one Contract-JEPA tracer."""

    cache = Path(cache_directory).resolve()
    output = Path(output_directory).resolve()
    building = output.with_name(output.name + ".building")
    if output.exists() or building.exists():
        raise FileExistsError("Contract-JEPA refuses an existing output")
    interpretable = (
        cache == (Path.cwd() / FROZEN_CACHE).resolve()
        and output == (Path.cwd() / FROZEN_OUTPUT).resolve()
        and pretrain_steps == FROZEN_PRETRAIN_STEPS
        and latency_repetitions == 100
        and expected_pair_count == 40
    )
    if not interpretable and not allow_noninterpretable_smoke:
        raise ValueError(
            "non-frozen Contract-JEPA runs require smoke permission"
        )
    if not interpretable and output == (Path.cwd() / FROZEN_OUTPUT).resolve():
        raise ValueError("Contract-JEPA smoke cannot use frozen output")
    commit = _git_head()
    sources = _source_identity(commit, require_clean=interpretable)
    building.mkdir(parents=True)
    started = time.time()
    try:
        prepared = load_edge_dynamics_cache(cache)
        partitions = {
            role: partition_worker_topology(windows)
            for role, windows in prepared.windows.items()
        }
        held = {
            value.held_out_normalized_value
            for value in partitions.values()
        }
        if len(held) != 1:
            raise ValueError(
                "Contract-JEPA held topology identity differs"
            )
        roles = {
            "fit": partitions["fit"].in_distribution,
            "selection": partitions["selection"].in_distribution,
            "calibration": partitions["calibration"].in_distribution,
            "iid_evaluation": partitions["evaluation"].in_distribution,
            "transfer_evaluation": partitions["evaluation"].held_out,
        }
        fit = roles["fit"]
        ownership = fit_owned_feature_mask(fit)
        baseline = ContractiveLowRankDynamics(
            LowRankConfig(rank=32)
        ).fit(fit)
        raw_before = _canonical_json_bytes(baseline.to_dict())
        model_directory = building / "models"
        model_directory.mkdir()
        _write_json(model_directory / "raw.json", baseline.to_dict())
        branches: Dict[str, TaskGroundedContractJepa] = {}
        dynamics: Dict[str, TaskGroundedContractDynamics] = {}
        training_seconds = {}
        for name in CELL_NAMES:
            config = replace(
                TaskGroundedContractConfig(),
                objective=name,
                pretrain_steps=pretrain_steps,
                checkpoint_interval=max(1, min(100, pretrain_steps)),
                expected_pair_count=expected_pair_count,
            )
            tick = time.perf_counter()
            branch = TaskGroundedContractJepa(config).fit(
                fit, baseline
            ).select(roles["selection"], baseline)
            training_seconds[name] = time.perf_counter() - tick
            branches[name] = branch
            dynamics[name] = TaskGroundedContractDynamics(
                baseline, branch
            )
        raw_unchanged = (
            raw_before == _canonical_json_bytes(baseline.to_dict())
        )
        raw_predictions = {
            role: baseline.rollout(
                windows.histories,
                windows.future_controls,
                windows.future_actions,
                windows.graph,
            ).mean
            for role, windows in roles.items()
            if role != "fit"
        }
        gain_curves = {}
        selected_gains = {}
        for name in CELL_NAMES:
            gain, curve = _select_gain(
                dynamics[name],
                roles["selection"],
                raw_predictions["selection"],
            )
            dynamics[name].set_gain(gain)
            selected_gains[name] = gain
            gain_curves[name] = curve
            _write_json(
                model_directory / f"{name}-branch.json",
                branches[name].to_dict(),
            )
            _write_json(
                model_directory / f"{name}.json",
                dynamics[name].to_dict(),
            )

        predictions = {
            name: {
                role: model.rollout(
                    windows.histories,
                    windows.future_controls,
                    windows.future_actions,
                    windows.graph,
                ).mean
                for role, windows in roles.items()
                if role not in ("fit", "calibration")
            }
            for name, model in dynamics.items()
        }
        corrections = {}
        witnesses = {}
        for name, branch in branches.items():
            corrections[name] = {}
            witnesses[name] = {}
            for role, windows in roles.items():
                if role == "fit":
                    continue
                correction, witness = branch.predict_contract(
                    windows.histories,
                    windows.future_controls,
                    windows.future_actions,
                    windows.graph,
                )
                corrections[name][role] = correction
                witnesses[name][role] = witness

        paired = {
            role: _paired_evidence(
                windows,
                ownership,
                {
                    name: (
                        predictions[name][role]
                        if role != "calibration"
                        else dynamics[name].rollout(
                            windows.histories,
                            windows.future_controls,
                            windows.future_actions,
                            windows.graph,
                        ).mean
                    )
                    for name in CELL_NAMES
                },
                witnesses,
                role,
            )
            for role, windows in roles.items()
            if role != "fit"
        }
        queries = _transfer_queries(
            prepared.attribution_queries,
            fit.control_feature_names,
            next(iter(held)),
        )
        query_predictions = {
            name: _query_predictions(model, queries, fit.graph)
            for name, model in dynamics.items()
        }
        sanity = {
            name: _action_sanity_predictions(
                model, roles["transfer_evaluation"]
            )
            for name, model in dynamics.items()
        }

        sample = roles["transfer_evaluation"]
        restoration_evidence: Dict[str, np.ndarray] = {}
        restoration_max = 0.0
        restored_alert_decisions_match = True
        for name in CELL_NAMES:
            restored = TaskGroundedContractDynamics.from_dict(
                dynamics[name].to_dict()
            )
            original = dynamics[name].rollout(
                sample.histories[:8],
                sample.future_controls[:8],
                sample.future_actions[:8],
                sample.graph,
            )
            replay = restored.rollout(
                sample.histories[:8],
                sample.future_controls[:8],
                sample.future_actions[:8],
                sample.graph,
            )
            original_correction, original_witness = branches[
                name
            ].predict_contract(
                sample.histories[:8],
                sample.future_controls[:8],
                sample.future_actions[:8],
                sample.graph,
            )
            replay_correction, replay_witness = (
                restored.branch.predict_contract(
                    sample.histories[:8],
                    sample.future_controls[:8],
                    sample.future_actions[:8],
                    sample.graph,
                )
            )
            original_tokens = branches[name].encode_contract(
                sample.histories[:8], sample.graph
            )
            restored_tokens = restored.branch.encode_contract(
                sample.histories[:8], sample.graph
            )
            cutoff = _calibration_control_cutoff(
                witnesses[name]["calibration"], roles["calibration"]
            )
            original_alerts = original_witness > cutoff
            restored_alerts = replay_witness > cutoff
            restored_alert_decisions_match = (
                restored_alert_decisions_match
                and np.array_equal(original_alerts, restored_alerts)
            )
            for field, original_values, restored_values in (
                ("rollout_mean", original.mean, replay.mean),
                (
                    "rollout_variance",
                    original.variance,
                    replay.variance,
                ),
                (
                    "correction",
                    original_correction,
                    replay_correction,
                ),
                ("witness", original_witness, replay_witness),
                (
                    "learned_tokens",
                    original_tokens.learned_tokens,
                    restored_tokens.learned_tokens,
                ),
                (
                    "raw_current_state",
                    original_tokens.raw_current_state,
                    restored_tokens.raw_current_state,
                ),
            ):
                restoration_evidence[
                    f"restoration_original_{field}__{name}"
                ] = original_values
                restoration_evidence[
                    f"restoration_restored_{field}__{name}"
                ] = restored_values
                restoration_max = max(
                    restoration_max,
                    float(
                        np.max(
                            np.abs(
                                original_values - restored_values
                            )
                        )
                    ),
                )
            restoration_evidence[
                f"restoration_original_alerts__{name}"
            ] = original_alerts
            restoration_evidence[
                f"restoration_restored_alerts__{name}"
            ] = restored_alerts
        zero_model = TaskGroundedContractDynamics(
            baseline, branches["task_grounded_contract_jepa"], gain=0.0
        )
        raw_sample = baseline.rollout(
            sample.histories[:8],
            sample.future_controls[:8],
            sample.future_actions[:8],
            sample.graph,
        )
        zero_sample = zero_model.rollout(
            sample.histories[:8],
            sample.future_controls[:8],
            sample.future_actions[:8],
            sample.graph,
        )
        gain_zero_exact = bool(
            np.array_equal(raw_sample.mean, zero_sample.mean)
            and np.array_equal(raw_sample.variance, zero_sample.variance)
        )
        causality = _rejects_forbidden_inputs(
            branches["task_grounded_contract_jepa"], sample
        )
        latency = {}
        for name, model in dynamics.items():
            call = lambda candidate=model: candidate.rollout(
                sample.histories[:1],
                sample.future_controls[:1],
                sample.future_actions[:1],
                sample.graph,
            )
            call()
            timings = []
            for _ in range(latency_repetitions):
                tick = time.perf_counter_ns()
                call()
                timings.append(
                    (time.perf_counter_ns() - tick) / 1e6
                )
            latency[name] = {
                "median_ms": float(np.median(timings)),
                "p95_ms": float(np.quantile(timings, 0.95)),
                "repetitions": latency_repetitions,
            }

        evidence: Dict[str, np.ndarray] = {}
        for role, windows in roles.items():
            if role == "fit":
                continue
            evidence[f"actions__{role}"] = windows.future_actions.astype(
                np.float32
            )
            for name in CELL_NAMES:
                evidence[f"witness__{name}__{role}"] = witnesses[
                    name
                ][role].astype(np.float32)
            if role == "calibration":
                continue
            evidence[f"target__{role}"] = windows.future_states.astype(
                np.float32
            )
            evidence[f"prediction__raw__{role}"] = raw_predictions[
                role
            ].astype(np.float32)
            evidence[f"paired_target__{role}"] = paired[role][
                "target"
            ].astype(np.float32)
            evidence[f"paired_witness_target__{role}"] = paired[role][
                "witness_target"
            ].astype(np.float32)
            for name in CELL_NAMES:
                evidence[f"prediction__{name}__{role}"] = predictions[
                    name
                ][role].astype(np.float32)
                evidence[f"correction__{name}__{role}"] = corrections[
                    name
                ][role].astype(np.float32)
                evidence[f"paired_prediction__{name}__{role}"] = paired[
                    role
                ][name].astype(np.float32)
                evidence[f"paired_witness__{name}__{role}"] = paired[
                    role
                ][f"witness__{name}"].astype(np.float32)
        for name in CELL_NAMES:
            evidence[f"correction_bound__{name}"] = branches[
                name
            ].correction_bound.astype(np.float32)
        evidence["query_observed_future"] = (
            queries.observed_future.astype(np.float32)
        )
        for name in CELL_NAMES:
            evidence[f"query_prediction__{name}"] = query_predictions[
                name
            ].astype(np.float32)
            for variant, values in sanity[name].items():
                evidence[
                    f"action_sanity__{name}__{variant}"
                ] = values.astype(np.float32)
        evidence.update(
            {
                name: values.astype(
                    np.bool_
                    if values.dtype.kind == "b"
                    else np.float64
                )
                for name, values in restoration_evidence.items()
            }
        )
        np.savez_compressed(building / "evidence.npz", **evidence)

        parameter_counts = {
            name: {
                "training": branches[name].training_parameter_count,
                "inference": branches[name].inference_parameter_count,
            }
            for name in CELL_NAMES
        }
        bundle_bytes = len(
            _canonical_json_bytes(
                dynamics["task_grounded_contract_jepa"].to_dict()
            )
        )
        metadata = {
            "schema_version": 1,
            "kind": "task_grounded_contract_jepa_evidence",
            "interpretable": interpretable,
            "graph": fit.graph.to_dict(),
            "ownership_mask": ownership.astype(int).tolist(),
            "roles": {
                role: {
                    "pair_ids": list(windows.matched_pair_ids),
                    "trajectory_ids": list(windows.trajectory_ids),
                    "transition_indices": (
                        windows.transition_indices.tolist()
                    ),
                }
                for role, windows in roles.items()
                if role != "fit"
            },
            "queries": {
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
            },
            "selected_gains": selected_gains,
            "gain_curves": gain_curves,
            "parameter_counts": parameter_counts,
            "candidate_bundle_bytes": bundle_bytes,
            "raw_hash_unchanged": raw_unchanged,
            "gain_zero_is_exact_raw": gain_zero_exact,
            "public_causality": causality,
            "restoration_max_abs": restoration_max,
            "restored_alert_decisions_match": (
                restored_alert_decisions_match
            ),
            "latency": latency,
        }
        _write_json(building / "evidence-metadata.json", metadata)
        assessment = assess_stored_bundle(building)
        _write_json(building / "assessment.json", assessment)
        report = {
            "schema_version": 1,
            "kind": "task_grounded_contract_jepa_tracer_v1",
            "evidence_boundary": (
                "single-seed open-development task-grounded residual "
                "contract; not production paging or sealed confirmation"
            ),
            "interpretable": interpretable,
            "source": {
                "cache_directory": str(cache),
                "source_corpus_sha256": prepared.source_corpus_sha256,
                "source_artifact_manifest_sha256": (
                    prepared.source_artifact_manifest_sha256
                ),
                "preprocessing_protocol": prepared.preprocessing_protocol,
                "held_out_worker_topology_normalized": next(iter(held)),
            },
            "implementation": {
                "commit": commit,
                "sources": sources,
                "runtime": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "numpy": np.__version__,
                },
            },
            "configuration": {
                name: branches[name].to_dict()["config"]
                for name in CELL_NAMES
            },
            "training_seconds": training_seconds,
            "selected_steps": {
                name: branches[name].selected_step for name in CELL_NAMES
            },
            "selected_gains": selected_gains,
            "parameter_counts": parameter_counts,
            "candidate_bundle_bytes": bundle_bytes,
            "latency": latency,
            "elapsed_seconds": time.time() - started,
            "assessment": assessment,
        }
        _write_json(building / "result.json", report)
        (building / "REPORT.md").write_text(_render_report(report))
        _copy_sources(building)
        _write_manifest(building)
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


def _select_gain(
    model: TaskGroundedContractDynamics,
    windows: ActionConditionedWindows,
    raw_prediction: np.ndarray,
) -> Tuple[float, list[Mapping[str, Any]]]:
    raw_scores = _scores(raw_prediction, windows)
    rows = []
    for gain in GAIN_CANDIDATES:
        model.set_gain(gain)
        prediction = model.rollout(
            windows.histories,
            windows.future_controls,
            windows.future_actions,
            windows.graph,
        ).mean
        scores = _scores(prediction, windows)
        safe = (
            scores["overall_mse"]
            <= 1.05 * raw_scores["overall_mse"]
            and scores["action_overlap_mse"]
            <= 1.05 * raw_scores["action_overlap_mse"]
        )
        rows.append({"gain": gain, "raw_safe": safe, **scores})
    eligible = [row for row in rows if bool(row["raw_safe"])]
    selected = min(
        eligible,
        key=lambda row: (
            float(row["downstream_effect_mse"]),
            float(row["gain"]),
        ),
    )
    model.set_gain(float(selected["gain"]))
    return float(selected["gain"]), rows


def _calibration_control_cutoff(
    scores: np.ndarray,
    windows: ActionConditionedWindows,
) -> float:
    trajectory_ids = np.asarray(windows.trajectory_ids)
    maxima = []
    for trajectory in sorted(set(windows.trajectory_ids)):
        rows = np.flatnonzero(trajectory_ids == trajectory)
        if not np.any(windows.future_actions[rows, ..., 1] > 0.5):
            maxima.append(float(np.max(scores[rows])))
    if len(maxima) < 2:
        raise ValueError(
            "Contract-JEPA alert policy needs control trajectories"
        )
    return float(np.quantile(maxima, 0.95, method="higher"))


def _scores(
    prediction: np.ndarray, windows: ActionConditionedWindows
) -> Mapping[str, float]:
    return shared._forecast_scores(
        prediction,
        windows.future_states,
        windows.future_actions,
        windows.matched_pair_ids,
        windows.trajectory_ids,
        windows.transition_indices,
        windows.graph.to_dict(),
    )


def _paired_evidence(
    windows: ActionConditionedWindows,
    ownership: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    witnesses: Mapping[str, Mapping[str, np.ndarray]],
    role: str,
) -> Mapping[str, np.ndarray]:
    treatment, control = _active_aligned_rows(windows)
    target = (
        windows.future_states[treatment]
        - windows.future_states[control]
    )
    witness_target = np.sqrt(
        np.mean(np.square(target)[..., ownership], axis=2)
    )
    result: Dict[str, np.ndarray] = {
        "target": target,
        "witness_target": witness_target,
    }
    for name in CELL_NAMES:
        result[name] = (
            predictions[name][treatment] - predictions[name][control]
        )
        result[f"witness__{name}"] = witnesses[name][role][treatment]
    return result


def _active_aligned_rows(
    windows: ActionConditionedWindows,
) -> Tuple[np.ndarray, np.ndarray]:
    pair_names = sorted(set(windows.matched_pair_ids))
    treatment_rows = []
    control_rows = []
    for pair in pair_names:
        pair_rows = np.flatnonzero(
            np.asarray(windows.matched_pair_ids) == pair
        )
        trajectories = sorted(
            {windows.trajectory_ids[row] for row in pair_rows}
        )
        treatment = [
            trajectory
            for trajectory in trajectories
            if np.any(
                windows.future_actions[
                    np.asarray(windows.trajectory_ids) == trajectory,
                    ...,
                    1,
                ]
                > 0.5
            )
        ]
        control = [
            trajectory
            for trajectory in trajectories
            if trajectory not in treatment
        ]
        if len(treatment) != 1 or len(control) != 1:
            raise ValueError("Contract-JEPA paired evidence lost an arm")
        treatment_index = {
            int(windows.transition_indices[row]): row
            for row in pair_rows
            if windows.trajectory_ids[row] == treatment[0]
        }
        control_index = {
            int(windows.transition_indices[row]): row
            for row in pair_rows
            if windows.trajectory_ids[row] == control[0]
        }
        for transition in sorted(
            set(treatment_index) & set(control_index)
        ):
            row = treatment_index[transition]
            if np.any(windows.future_actions[row, ..., 1] > 0.5):
                treatment_rows.append(row)
                control_rows.append(control_index[transition])
    return (
        np.asarray(treatment_rows, dtype=np.int64),
        np.asarray(control_rows, dtype=np.int64),
    )


def _query_predictions(
    model: TaskGroundedContractDynamics,
    queries: Any,
    graph: Any,
) -> np.ndarray:
    rows = []
    for index in range(len(queries.query_ids)):
        count = len(queries.candidate_ids)
        rows.append(
            model.rollout(
                np.repeat(
                    queries.histories[index : index + 1],
                    count,
                    axis=0,
                ),
                np.repeat(
                    queries.future_controls[index : index + 1],
                    count,
                    axis=0,
                ),
                queries.candidate_actions[index],
                graph,
            ).mean
        )
    return np.stack(rows)


def _action_sanity_predictions(
    model: TaskGroundedContractDynamics,
    windows: ActionConditionedWindows,
) -> Mapping[str, np.ndarray]:
    correct = model.rollout(
        windows.histories,
        windows.future_controls,
        windows.future_actions,
        windows.graph,
    ).mean
    no_action = np.zeros_like(windows.future_actions)
    no_action[..., 0] = 1.0
    absent = model.rollout(
        windows.histories,
        windows.future_controls,
        no_action,
        windows.graph,
    ).mean
    pair_names = sorted(set(windows.matched_pair_ids))
    pair_array = np.asarray(windows.matched_pair_ids)
    shuffled = np.zeros_like(windows.future_actions)
    for position, pair in enumerate(pair_names):
        rows = np.flatnonzero(pair_array == pair)
        donor = pair_names[(position + 1) % len(pair_names)]
        donor_rows = np.flatnonzero(pair_array == donor)
        shuffled[rows] = windows.future_actions[donor_rows[: len(rows)]]
    shifted = model.rollout(
        windows.histories,
        windows.future_controls,
        shuffled,
        windows.graph,
    ).mean
    return {
        "correct": correct,
        "no_action": absent,
        "shuffled": shifted,
    }


def _transfer_queries(
    queries: Any, control_names: Tuple[str, ...], held_value: float
) -> Any:
    from lab.action_dynamics.prototype_complete_lejepa import (
        _transfer_queries as select_transfer_queries,
    )

    return select_transfer_queries(queries, control_names, held_value)


def _rejects_forbidden_inputs(
    model: TaskGroundedContractJepa,
    windows: ActionConditionedWindows,
) -> bool:
    for keyword, value in (
        ("future_states", windows.future_states[:1]),
        ("pair_ids", windows.matched_pair_ids[:1]),
        ("target_truth", windows.future_states[:1]),
    ):
        try:
            model.predict_contract(
                windows.histories[:1],
                windows.future_controls[:1],
                windows.future_actions[:1],
                windows.graph,
                **{keyword: value},
            )
        except TypeError:
            continue
        return False
    return True


def _source_identity(
    commit: str, *, require_clean: bool
) -> Mapping[str, Any]:
    if require_clean:
        result = subprocess.run(
            [
                "git",
                "diff",
                "--quiet",
                "HEAD",
                "--",
                *IMPLEMENTATION_SOURCE_PATHS,
            ],
            check=False,
        )
        untracked = subprocess.run(
            [
                "git",
                "ls-files",
                "--others",
                "--exclude-standard",
                "--",
                *IMPLEMENTATION_SOURCE_PATHS,
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if result.returncode != 0 or untracked:
            raise ValueError("frozen Contract-JEPA sources must match HEAD")
    return {
        path: {
            "sha256": _file_sha256(Path(path)),
            "git_blob": _git_blob(path, commit),
        }
        for path in IMPLEMENTATION_SOURCE_PATHS
    }


def _copy_sources(directory: Path) -> None:
    root = directory / "reproduction-sources"
    for name in IMPLEMENTATION_SOURCE_PATHS:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(name), target)


def _write_manifest(directory: Path) -> None:
    values = {
        path.relative_to(directory).as_posix(): _file_sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "artifact-manifest.json"
    }
    _write_json(
        directory / "artifact-manifest.json",
        {
            "schema_version": 1,
            "kind": "task_grounded_contract_jepa_manifest",
            "sha256": values,
        },
    )


def _render_report(report: Mapping[str, Any]) -> str:
    assessment = dict(report["assessment"])
    transfer = assessment["roles"]["transfer_evaluation"]["scores"]
    lines = [
        "# Task-grounded Contract-JEPA tracer",
        "",
        f"Decision: **{assessment['decision']}**",
        "",
        "| model | gain | overall MSE | action MSE | downstream effect MSE |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("raw", *CELL_NAMES):
        score = transfer[name]
        gain = (
            0.0
            if name == "raw"
            else assessment["selected_gains"][name]
        )
        lines.append(
            f"| {name} | {gain:.2f} | {score['overall_mse']:.6g} | "
            f"{score['action_overlap_mse']:.6g} | "
            f"{score['downstream_effect_mse']:.6g} |"
        )
    lines.extend(
        [
            "",
            "This is open-development evidence, not production paging or "
            "sealed confirmation.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


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
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_blob(path: str, commit: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", f"{commit}:{path}"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _build_parser() -> argparse.ArgumentParser:
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
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    output = run_experiment(
        cache_directory=args.cache,
        output_directory=args.output,
        pretrain_steps=args.pretrain_steps,
        latency_repetitions=args.latency_repetitions,
        allow_noninterpretable_smoke=args.allow_noninterpretable_smoke,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
