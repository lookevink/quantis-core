#!/usr/bin/env python3
"""Retained runner for the frozen PairEffect-JEPA tracer."""

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

from lab.action_dynamics.prototype_pair_effect_jepa_assessor import (
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
from quantis_core.edge_dynamics.pair_effect_jepa import (
    PairEffectCorrectedDynamics,
    PairEffectJepaConfig,
    PairEffectJepaModel,
)


FROZEN_CACHE = Path(
    "artifacts/action-dynamics/edge-preprocessing-v1/"
    "eb54271132f88c9a431b01e786ea66279a563776434cca2290e47e6b7ae9b3ff"
)
FROZEN_OUTPUT = Path(
    "artifacts/action-dynamics/prototype-pair-effect-jepa-v1"
)
FROZEN_PRETRAIN_STEPS = 800
IMPLEMENTATION_SOURCE_PATHS = (
    "lab/action_dynamics/prototype_pair_effect_jepa.py",
    "lab/action_dynamics/prototype_pair_effect_jepa_assessor.py",
    "src/quantis_core/edge_dynamics/pair_effect_jepa.py",
    "tests/test_pair_effect_jepa.py",
    "docs/specs/pair-effect-jepa-tracer-v1.md",
    "docs/wayfinding/jepa-implementation-program/"
    "020-test-pair-effect-jepa.md",
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
    """Run, assess, and atomically publish one paired-effect tracer."""

    cache = Path(cache_directory).resolve()
    output = Path(output_directory).resolve()
    building = output.with_name(output.name + ".building")
    if output.exists() or building.exists():
        raise FileExistsError(
            "PairEffect-JEPA refuses an existing output"
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
            "non-frozen PairEffect-JEPA runs require smoke permission"
        )
    if not interpretable and output == (Path.cwd() / FROZEN_OUTPUT).resolve():
        raise ValueError(
            "PairEffect-JEPA smoke cannot use the frozen output"
        )
    implementation_commit = _git_head()
    sources = _source_identity(
        implementation_commit, require_clean=interpretable
    )
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
                "PairEffect-JEPA held topology identity differs"
            )
        roles = {
            "fit": partitions["fit"].in_distribution,
            "selection": partitions["selection"].in_distribution,
            "iid_evaluation": partitions["evaluation"].in_distribution,
            "transfer_evaluation": partitions["evaluation"].held_out,
        }
        fit = roles["fit"]
        ownership = fit_owned_feature_mask(fit)
        raw = ContractiveLowRankDynamics(
            LowRankConfig(rank=32)
        ).fit(fit)
        model_directory = building / "models"
        model_directory.mkdir()
        _write_json(model_directory / "raw.json", raw.to_dict())
        cells: Dict[str, PairEffectJepaModel] = {}
        composed: Dict[str, PairEffectCorrectedDynamics] = {}
        training_seconds = {}
        for name in CELL_NAMES:
            config = replace(
                PairEffectJepaConfig(),
                objective=name,
                pretrain_steps=pretrain_steps,
                checkpoint_interval=max(1, min(100, pretrain_steps)),
                expected_pair_count=expected_pair_count,
            )
            fit_started = time.perf_counter()
            model = PairEffectJepaModel(config).fit(fit).select(
                roles["selection"]
            )
            training_seconds[name] = (
                time.perf_counter() - fit_started
            )
            cells[name] = model
            composed[name] = PairEffectCorrectedDynamics(raw, model)
            _write_json(
                model_directory / f"{name}.json", model.to_dict()
            )
            _write_json(
                model_directory / f"{name}-composed.json",
                composed[name].to_dict(),
            )

        raw_predictions = {
            role: raw.rollout(
                windows.histories,
                windows.future_controls,
                windows.future_actions,
                windows.graph,
            ).mean
            for role, windows in roles.items()
            if role != "fit"
        }
        predictions = {
            name: {
                role: dynamics.rollout(
                    windows.histories,
                    windows.future_controls,
                    windows.future_actions,
                    windows.graph,
                ).mean
                for role, windows in roles.items()
                if role != "fit"
            }
            for name, dynamics in composed.items()
        }
        effect_rows = {
            role: _effect_evidence(cells, windows)
            for role, windows in roles.items()
            if role != "fit"
        }
        queries = _transfer_queries(
            prepared.attribution_queries,
            fit.control_feature_names,
            next(iter(held)),
        )
        query_predictions = {
            name: _query_predictions(dynamics, queries, fit.graph)
            for name, dynamics in composed.items()
        }
        sanity = {
            name: _action_sanity_predictions(dynamics, roles[
                "transfer_evaluation"
            ])
            for name, dynamics in composed.items()
        }

        restored_max = 0.0
        sample = roles["transfer_evaluation"]
        for name in CELL_NAMES:
            restored = PairEffectCorrectedDynamics.from_dict(
                composed[name].to_dict()
            )
            original = composed[name].rollout(
                sample.histories[:8],
                sample.future_controls[:8],
                sample.future_actions[:8],
                sample.graph,
            ).mean
            replay = restored.rollout(
                sample.histories[:8],
                sample.future_controls[:8],
                sample.future_actions[:8],
                sample.graph,
            ).mean
            restored_max = max(
                restored_max, float(np.max(np.abs(original - replay)))
            )
        no_action = np.zeros_like(sample.future_actions[:8])
        no_action[..., 0] = 1.0
        zero_effect = cells["pair_effect_jepa"].predict_effect(
            sample.histories[:8],
            sample.future_controls[:8],
            no_action,
            sample.graph,
        )
        public_causality = _rejects_forbidden_inputs(
            cells["pair_effect_jepa"], sample
        )
        latency = {}
        for name, dynamics in composed.items():
            call = lambda model=dynamics: model.rollout(
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
            evidence[f"target__{role}"] = windows.future_states.astype(
                np.float32
            )
            evidence[f"actions__{role}"] = windows.future_actions.astype(
                np.float32
            )
            evidence[f"prediction__raw__{role}"] = raw_predictions[
                role
            ].astype(np.float32)
            evidence[f"effect_target__{role}"] = effect_rows[role][
                "target"
            ].astype(np.float32)
            for name in CELL_NAMES:
                evidence[f"prediction__{name}__{role}"] = predictions[
                    name
                ][role].astype(np.float32)
                evidence[f"effect_prediction__{name}__{role}"] = (
                    effect_rows[role][name].astype(np.float32)
                )
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
        np.savez_compressed(building / "evidence.npz", **evidence)

        parameter_counts = {
            name: {
                "training": cells[name].training_parameter_count,
                "inference": cells[name].inference_parameter_count,
            }
            for name in CELL_NAMES
        }
        bundle_bytes = len(
            _canonical_json_bytes(
                composed["pair_effect_jepa"].to_dict()
            )
        )
        metadata = {
            "schema_version": 1,
            "kind": "pair_effect_jepa_evidence",
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
                    "effect_pair_ids": effect_rows[role][
                        "pair_ids"
                    ].tolist(),
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
            "parameter_counts": parameter_counts,
            "candidate_bundle_bytes": bundle_bytes,
            "restoration_max_abs": restored_max,
            "zero_effect_max_abs": float(np.max(np.abs(zero_effect))),
            "public_causality": public_causality,
            "latency": latency,
        }
        _write_json(building / "evidence-metadata.json", metadata)
        assessment = assess_stored_bundle(building)
        _write_json(building / "assessment.json", assessment)
        report = {
            "schema_version": 1,
            "kind": "pair_effect_jepa_tracer_v1",
            "evidence_boundary": (
                "single-seed open-development paired-effect tracer; "
                "not production paging or sealed confirmation"
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
                "commit": implementation_commit,
                "sources": sources,
                "runtime": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "numpy": np.__version__,
                },
            },
            "configuration": {
                name: cells[name].to_dict()["config"]
                for name in CELL_NAMES
            },
            "training_seconds": training_seconds,
            "selected_steps": {
                name: cells[name].selected_step for name in CELL_NAMES
            },
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


def _effect_evidence(
    cells: Mapping[str, PairEffectJepaModel],
    windows: ActionConditionedWindows,
) -> Mapping[str, np.ndarray]:
    pairs = sorted(set(windows.matched_pair_ids))
    trajectories: Dict[str, Dict[str, list[int]]] = {
        pair: {} for pair in pairs
    }
    for row, (pair, trajectory) in enumerate(
        zip(windows.matched_pair_ids, windows.trajectory_ids)
    ):
        trajectories[pair].setdefault(trajectory, []).append(row)
    treatment_rows = []
    control_rows = []
    pair_ids = []
    for pair in pairs:
        arms = trajectories[pair]
        treatment = [
            name
            for name, rows in arms.items()
            if np.any(windows.future_actions[rows, ..., 1] > 0.5)
        ]
        control = [name for name in arms if name not in treatment]
        if len(treatment) != 1 or len(control) != 1:
            raise ValueError("paired-effect evidence lost an arm")
        treatment_by_transition = {
            int(windows.transition_indices[row]): row
            for row in arms[treatment[0]]
        }
        control_by_transition = {
            int(windows.transition_indices[row]): row
            for row in arms[control[0]]
        }
        for transition in sorted(
            set(treatment_by_transition) & set(control_by_transition)
        ):
            treatment_row = treatment_by_transition[transition]
            if not np.any(
                windows.future_actions[treatment_row, ..., 1] > 0.5
            ):
                continue
            treatment_rows.append(treatment_row)
            control_rows.append(control_by_transition[transition])
            pair_ids.append(pair)
    treatment_index = np.asarray(treatment_rows, dtype=np.int64)
    control_index = np.asarray(control_rows, dtype=np.int64)
    result: Dict[str, np.ndarray] = {
        "target": (
            windows.future_states[treatment_index]
            - windows.future_states[control_index]
        ),
        "pair_ids": np.asarray(pair_ids),
    }
    for name, model in cells.items():
        result[name] = model.predict_effect(
            windows.histories[treatment_index],
            windows.future_controls[treatment_index],
            windows.future_actions[treatment_index],
            windows.graph,
        )
    return result


def _query_predictions(
    model: PairEffectCorrectedDynamics,
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
    model: PairEffectCorrectedDynamics,
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
    shuffled_prediction = model.rollout(
        windows.histories,
        windows.future_controls,
        shuffled,
        windows.graph,
    ).mean
    return {
        "correct": correct,
        "no_action": absent,
        "shuffled": shuffled_prediction,
    }


def _transfer_queries(
    queries: Any, control_names: Tuple[str, ...], held_value: float
) -> Any:
    from lab.action_dynamics.prototype_complete_lejepa import (
        _transfer_queries as select_transfer_queries,
    )

    return select_transfer_queries(queries, control_names, held_value)


def _rejects_forbidden_inputs(
    model: PairEffectJepaModel, windows: ActionConditionedWindows
) -> bool:
    values = {
        "future_states": windows.future_states[:1],
        "pair_ids": windows.matched_pair_ids[:1],
        "target_truth": windows.future_states[:1],
    }
    for keyword, value in values.items():
        try:
            model.predict_effect(
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
            ["git", "diff", "--quiet", "HEAD", "--", *IMPLEMENTATION_SOURCE_PATHS],
            check=False,
        )
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "--", *IMPLEMENTATION_SOURCE_PATHS],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if result.returncode != 0 or untracked:
            raise ValueError(
                "frozen PairEffect-JEPA sources must match HEAD"
            )
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
        source = Path(name)
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


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
            "kind": "pair_effect_jepa_manifest",
            "sha256": values,
        },
    )


def _render_report(report: Mapping[str, Any]) -> str:
    assessment = dict(report["assessment"])
    transfer = assessment["roles"]["transfer_evaluation"]["scores"]
    lines = [
        "# PairEffect-JEPA tracer",
        "",
        f"Decision: **{assessment['decision']}**",
        "",
        "| model | overall MSE | action MSE | downstream effect MSE |",
        "|---|---:|---:|---:|",
    ]
    for name in ("raw", *CELL_NAMES):
        score = transfer[name]
        lines.append(
            f"| {name} | {score['overall_mse']:.6g} | "
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
