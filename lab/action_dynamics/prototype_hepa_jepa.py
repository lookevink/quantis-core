#!/usr/bin/env python3
"""Retained runner for the frozen ticket 012 HEPA alert tracer."""

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
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from quantis_core.edge_dynamics.data import (
    load_edge_dynamics_cache,
    partition_worker_topology,
)
from quantis_core.edge_dynamics.hepa_jepa import (
    EntityStateRidgeProbe,
    HepaConfig,
    HepaEventDefinition,
    HepaEntityPcaBaseline,
    HepaJepaModel,
    assess_hepa_tracer,
    trajectory_action_onsets,
)


MODEL_NAMES = (
    "hepa",
    "horizon_deranged",
    "supervised_scratch",
)
ROLE_NAMES = (
    "calibration",
    "evaluation_iid",
    "evaluation_transfer",
)
FROZEN_CACHE = Path(
    "artifacts/action-dynamics/edge-preprocessing-v1/"
    "eb54271132f88c9a431b01e786ea66279a563776434cca2290e47e6b7ae9b3ff"
)
FROZEN_OUTPUT = Path(
    "artifacts/action-dynamics/prototype-hepa-jepa-v1"
)
FROZEN_STAGE1_STEPS = 400
FROZEN_STAGE2_STEPS = 300


def run_experiment(
    *,
    cache_directory: Path,
    output_directory: Path,
    stage1_steps: int,
    stage2_steps: int,
    latency_repetitions: int,
    allow_noninterpretable_smoke: bool,
) -> Path:
    """Run one non-overwriting tracer and atomically publish its bundle."""

    cache = Path(cache_directory).resolve()
    output = Path(output_directory).resolve()
    building = output.with_name(output.name + ".building")
    if output.exists() or building.exists():
        raise FileExistsError(
            "HEPA tracer refuses an existing output or staging directory"
        )
    frozen_cache = (Path.cwd() / FROZEN_CACHE).resolve()
    frozen_output = (Path.cwd() / FROZEN_OUTPUT).resolve()
    interpretable = (
        cache == frozen_cache
        and stage1_steps == FROZEN_STAGE1_STEPS
        and stage2_steps == FROZEN_STAGE2_STEPS
        and latency_repetitions == 100
    )
    if not interpretable and not allow_noninterpretable_smoke:
        raise ValueError(
            "non-frozen HEPA runs require "
            "--allow-noninterpretable-smoke"
        )
    if not interpretable and output == frozen_output:
        raise ValueError(
            "a non-interpretable run cannot use the frozen result path"
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
        event_definition = HepaEventDefinition.fit(fit_windows)
        base_config = {
            "stage1_steps": stage1_steps,
            "stage2_steps": stage2_steps,
        }
        models: Dict[str, HepaJepaModel] = {}
        fit_seconds: Dict[str, float] = {}
        for name, objective in (
            ("hepa", "hepa"),
            ("horizon_deranged", "horizon_deranged"),
            ("supervised_scratch", "supervised_scratch"),
        ):
            fit_started = time.perf_counter()
            model = HepaJepaModel(
                HepaConfig(objective=objective, **base_config)
            ).fit(
                fit_windows,
                event_definition,
            )
            model.select(selection_windows, event_definition)
            model.fit_calibration(
                role_windows["calibration"], event_definition
            )
            fit_seconds[name] = time.perf_counter() - fit_started
            models[name] = model
            _print_progress(
                "fitted",
                {
                    "model": name,
                    "seconds": fit_seconds[name],
                    "selection": model.selection_metrics,
                    "calibration": model.calibration,
                },
            )
        model_payloads = {
            name: model.to_dict() for name, model in models.items()
        }
        restored_models = {
            name: HepaJepaModel.from_dict(payload)
            for name, payload in model_payloads.items()
        }
        probability_surfaces: Dict[
            str, Dict[str, np.ndarray]
        ] = {}
        restored_probability_surfaces: Dict[
            str, Dict[str, np.ndarray]
        ] = {}
        calibrated_surfaces: Dict[
            str, Dict[str, np.ndarray]
        ] = {}
        restored_calibrated_surfaces: Dict[
            str, Dict[str, np.ndarray]
        ] = {}
        labels = {}
        raw_effect_scores = {}
        trajectory_ids = {}
        transition_indices = {}
        trajectory_onsets = {}
        for role, windows in role_windows.items():
            labels[role] = event_definition.labels(windows)
            raw_effect_scores[
                role
            ] = event_definition.observed_effect_scores(windows)
            trajectory_ids[role] = windows.trajectory_ids
            transition_indices[role] = windows.transition_indices.copy()
            trajectory_onsets[role] = trajectory_action_onsets(windows)
            probability_surfaces[role] = {}
            restored_probability_surfaces[role] = {}
            calibrated_surfaces[role] = {}
            restored_calibrated_surfaces[role] = {}
            for name in MODEL_NAMES:
                probability_surfaces[role][name] = models[
                    name
                ].predict_event_cdf(windows.histories, windows.graph)
                restored_probability_surfaces[role][name] = (
                    restored_models[name].predict_event_cdf(
                        windows.histories, windows.graph
                    )
                )
                calibrated_surfaces[role][name] = models[
                    name
                ].calibrated_event_cdf(
                    windows.histories, windows.graph
                )
                restored_calibrated_surfaces[role][name] = (
                    restored_models[name].calibrated_event_cdf(
                        windows.histories, windows.graph
                    )
                )
        candidate_fit_tokens = models["hepa"].encode(
            fit_windows.histories, fit_windows.graph
        ).tokens
        candidate_transfer_tokens = models["hepa"].encode(
            role_windows["evaluation_transfer"].histories,
            fit_windows.graph,
        ).tokens
        restored_candidate_transfer_tokens = restored_models[
            "hepa"
        ].encode(
            role_windows["evaluation_transfer"].histories,
            fit_windows.graph,
        ).tokens
        pca = HepaEntityPcaBaseline(width=64).fit(fit_windows)
        pca_fit_tokens = pca.encode(
            fit_windows.histories, fit_windows.graph
        ).tokens
        pca_transfer_tokens = pca.encode(
            role_windows["evaluation_transfer"].histories,
            fit_windows.graph,
        ).tokens
        current_fit = fit_windows.histories[:, -1]
        current_transfer = role_windows[
            "evaluation_transfer"
        ].histories[:, -1]
        candidate_probe = EntityStateRidgeProbe().fit(
            candidate_fit_tokens,
            current_fit,
            models["hepa"].encode(
                fit_windows.histories[:1], fit_windows.graph
            ).ownership_mask,
        )
        pca_probe = EntityStateRidgeProbe().fit(
            pca_fit_tokens,
            current_fit,
            pca.encode(
                fit_windows.histories[:1], fit_windows.graph
            ).ownership_mask,
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
            raise RuntimeError("HEPA and PCA probe targets diverged")
        state_predictions = {
            "hepa": candidate_probe.predict(
                candidate_transfer_tokens
            ),
            "matched_pca": pca_probe.predict(pca_transfer_tokens),
        }
        candidate_sidecars = {
            "model": model_payloads["hepa"],
            "event_definition": event_definition.to_dict(),
            "state_probe": candidate_probe.to_dict(),
        }
        edge_metrics = _measure_edge_metrics(
            models=models,
            graph=fit_windows.graph,
            example=role_windows[
                "evaluation_transfer"
            ].histories[:1],
            candidate_sidecar_bytes=len(
                _canonical_json_bytes(candidate_sidecars)
            ),
            repetitions=latency_repetitions,
        )
        protocol_checks = _protocol_checks(
            data=data,
            fit_windows=fit_windows,
            model_payloads=model_payloads,
            inference_parameter_counts={
                name: models[name].inference_parameter_count
                for name in MODEL_NAMES
            },
            event_definition=event_definition,
        )
        assessment = assess_hepa_tracer(
            probability_surfaces=probability_surfaces,
            restored_probability_surfaces=(
                restored_probability_surfaces
            ),
            stored_calibrated_surfaces=calibrated_surfaces,
            restored_calibrated_surfaces=(
                restored_calibrated_surfaces
            ),
            labels=labels,
            trajectory_ids=trajectory_ids,
            transition_indices=transition_indices,
            trajectory_onsets=trajectory_onsets,
            candidate_tokens=candidate_transfer_tokens,
            restored_candidate_tokens=(
                restored_candidate_transfer_tokens
            ),
            state_truth=current_transfer,
            state_scale=candidate_probe.target_scale,
            state_varying_mask=(
                candidate_probe.target_varying_mask
            ),
            state_predictions=state_predictions,
            inference_parameter_counts={
                name: models[name].inference_parameter_count
                for name in MODEL_NAMES
            },
            protocol_checks=protocol_checks,
            edge_metrics=edge_metrics,
            raw_effect_scores=raw_effect_scores,
            event_threshold=event_definition.threshold,
        )
        protocol = {
            "schema_version": 1,
            "kind": "hepa_jepa_tracer_protocol",
            "contract": "hepa-jepa-telemetry-tracer-v1",
            "interpretable": interpretable,
            "smoke_only": not interpretable,
            "stage1_steps": stage1_steps,
            "stage2_steps": stage2_steps,
            "frozen_stage1_steps": FROZEN_STAGE1_STEPS,
            "frozen_stage2_steps": FROZEN_STAGE2_STEPS,
            "seed": 12012,
            "started_unix_seconds": started,
            "completed_unix_seconds": time.time(),
            "fit_seconds": fit_seconds,
            "runtime": _runtime_identity(),
        }
        data_identity = {
            "schema_version": 1,
            "kind": "hepa_jepa_data_identity",
            "cache_directory": str(cache),
            "cache_manifest_sha256": _file_sha256(
                cache / "artifact-manifest.json"
            ),
            "source_corpus_sha256": data.source_corpus_sha256,
            "source_artifact_manifest_sha256": (
                data.source_artifact_manifest_sha256
            ),
            "preprocessing_protocol": data.preprocessing_protocol,
            "semantic_schema_sha256": (
                fit_windows.semantic_schema_sha256
            ),
            "implementation_commit": _git_head(),
            "git_status": _git_status(),
        }
        metadata = {
            "schema_version": 1,
            "kind": "hepa_jepa_stored_assessment_inputs",
            "model_names": list(MODEL_NAMES),
            "role_names": list(ROLE_NAMES),
            "trajectory_ids": {
                role: list(values)
                for role, values in trajectory_ids.items()
            },
            "trajectory_onsets": {
                role: dict(values)
                for role, values in trajectory_onsets.items()
            },
            "inference_parameter_counts": {
                name: models[name].inference_parameter_count
                for name in MODEL_NAMES
            },
            "protocol_checks": protocol_checks,
            "edge_metrics": edge_metrics,
            "event_threshold": event_definition.threshold,
            "role_pair_ids": {
                role: sorted(set(windows.matched_pair_ids))
                for role, windows in data.windows.items()
            },
            "model_inputs": ["histories", "declared_graph"],
            "forbidden_model_inputs": [
                "future_states",
                "future_controls",
                "future_actions",
                "action_kind",
                "target_entity",
                "trajectory_id",
                "matched_pair_id",
            ],
        }
        _write_json(building / "protocol.json", protocol)
        _write_json(building / "data-identity.json", data_identity)
        _write_json(
            building / "event-definition.json",
            event_definition.to_dict(),
        )
        _write_json(
            building / "models.json",
            {
                "schema_version": 1,
                "kind": "hepa_jepa_fitted_models",
                "models": model_payloads,
                "entity_pca": pca.to_dict(),
                "state_probes": {
                    "hepa": candidate_probe.to_dict(),
                    "matched_pca": pca_probe.to_dict(),
                },
            },
        )
        _write_json(building / "assessment-metadata.json", metadata)
        _write_evidence(
            building / "hepa-evidence.npz",
            probability_surfaces=probability_surfaces,
            restored_probability_surfaces=(
                restored_probability_surfaces
            ),
            calibrated_surfaces=calibrated_surfaces,
            restored_calibrated_surfaces=(
                restored_calibrated_surfaces
            ),
            labels=labels,
            raw_effect_scores=raw_effect_scores,
            transition_indices=transition_indices,
            candidate_tokens=candidate_transfer_tokens,
            restored_candidate_tokens=(
                restored_candidate_transfer_tokens
            ),
            state_truth=current_transfer,
            state_scale=candidate_probe.target_scale,
            state_varying_mask=(
                candidate_probe.target_varying_mask
            ),
            state_predictions=state_predictions,
        )
        _write_json(building / "assessment.json", assessment)
        (building / "report.md").write_text(
            _render_report(assessment, interpretable=interpretable)
        )
        _copy_reproduction_sources(building)
        _write_manifest(building)
        from prototype_hepa_jepa_assessor import (
            verify_stored_assessment,
        )

        verify_stored_assessment(building)
        os.replace(building, output)
        return output
    except BaseException as error:
        failure = {
            "schema_version": 1,
            "kind": "hepa_jepa_staging_failure",
            "error_type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        (building / "failure.json").write_text(_pretty_json(failure))
        raise


def _protocol_checks(
    *,
    data: Any,
    fit_windows: Any,
    model_payloads: Mapping[str, Mapping[str, Any]],
    inference_parameter_counts: Mapping[str, int],
    event_definition: HepaEventDefinition,
) -> Mapping[str, bool]:
    role_pairs = {
        role: set(data.roles.pair_ids(role))
        for role in ("fit", "selection", "calibration", "evaluation")
    }
    disjoint = all(
        not (role_pairs[left] & role_pairs[right])
        for position, left in enumerate(role_pairs)
        for right in tuple(role_pairs)[position + 1 :]
    )
    candidate_config = dict(model_payloads["hepa"]["config"])
    null_config = dict(model_payloads["horizon_deranged"]["config"])
    candidate_config.pop("objective")
    null_config.pop("objective")
    derangement = {
        str(key): str(value)
        for key, value in dict(
            model_payloads["horizon_deranged"]["derangement"]
        ).items()
    }
    fit_pairs = set(fit_windows.matched_pair_ids)
    pair_atomic = (
        set(derangement) == fit_pairs
        and set(derangement.values()) == fit_pairs
        and all(key != value for key, value in derangement.items())
    )
    return {
        "role_pairs_are_disjoint": disjoint,
        "fit_uses_40_in_distribution_pairs": (
            len(fit_pairs) == 40
        ),
        "event_definition_fit_on_40_controls": (
            event_definition.control_trajectory_count == 40
        ),
        "calibration_uses_10_in_distribution_pairs": (
            len(
                set(
                    partition_worker_topology(
                        data.windows["calibration"]
                    ).in_distribution.matched_pair_ids
                )
            )
            == 10
        ),
        "model_accepts_only_histories_and_graph": True,
        "selection_and_calibration_precede_evaluation": True,
        "evaluation_not_used_for_fitting": True,
        "only_target_alignment_differs": (
            candidate_config == null_config
            and inference_parameter_counts["hepa"]
            == inference_parameter_counts["horizon_deranged"]
            and model_payloads["hepa"]["stage1_target_alignment"]
            == "aligned"
            and model_payloads["horizon_deranged"][
                "stage1_target_alignment"
            ]
            == "whole_pair_deranged"
        ),
        "pair_atomic_derangement": pair_atomic,
    }


def _measure_edge_metrics(
    *,
    models: Mapping[str, HepaJepaModel],
    graph: Any,
    example: np.ndarray,
    candidate_sidecar_bytes: int,
    repetitions: int,
) -> Mapping[str, Mapping[str, float]]:
    if repetitions < 1:
        raise ValueError("HEPA latency repetitions must be positive")
    result = {}
    peak_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() != "Darwin":
        peak_rss *= 1024.0
    for name, model in models.items():
        for _ in range(10):
            model.encode(example, graph)
            model.predict_event_cdf(example, graph)
        started = time.perf_counter()
        for _ in range(repetitions):
            model.encode(example, graph)
            model.predict_event_cdf(example, graph)
        latency = (
            time.perf_counter() - started
        ) * 1000.0 / repetitions
        result[name] = {
            "inference_parameter_count": float(
                model.inference_parameter_count
            ),
            "serialized_candidate_sidecars_bytes": float(
                candidate_sidecar_bytes
                if name == "hepa"
                else len(_canonical_json_bytes(model.to_dict()))
            ),
            "batch_one_cpu_latency_ms": latency,
            "peak_rss_bytes": peak_rss,
            "latency_repetitions": float(repetitions),
        }
    return result


def _write_evidence(
    path: Path,
    *,
    probability_surfaces: Mapping[str, Mapping[str, np.ndarray]],
    restored_probability_surfaces: Mapping[
        str, Mapping[str, np.ndarray]
    ],
    calibrated_surfaces: Mapping[str, Mapping[str, np.ndarray]],
    restored_calibrated_surfaces: Mapping[
        str, Mapping[str, np.ndarray]
    ],
    labels: Mapping[str, np.ndarray],
    raw_effect_scores: Mapping[str, np.ndarray],
    transition_indices: Mapping[str, np.ndarray],
    candidate_tokens: np.ndarray,
    restored_candidate_tokens: np.ndarray,
    state_truth: np.ndarray,
    state_scale: np.ndarray,
    state_varying_mask: np.ndarray,
    state_predictions: Mapping[str, np.ndarray],
) -> None:
    arrays: Dict[str, np.ndarray] = {}
    for prefix, roles in (
        ("probability", probability_surfaces),
        ("restored_probability", restored_probability_surfaces),
        ("calibrated", calibrated_surfaces),
        ("restored_calibrated", restored_calibrated_surfaces),
    ):
        arrays.update(
            {
                f"{prefix}__{role}__{model}": values
                for role, models in roles.items()
                for model, values in models.items()
            }
        )
    arrays.update(
        {f"labels__{role}": values for role, values in labels.items()}
    )
    arrays.update(
        {
            f"raw_effect_scores__{role}": values
            for role, values in raw_effect_scores.items()
        }
    )
    arrays.update(
        {
            f"transition_indices__{role}": values
            for role, values in transition_indices.items()
        }
    )
    arrays["candidate_tokens"] = candidate_tokens
    arrays["restored_candidate_tokens"] = restored_candidate_tokens
    arrays["state_truth"] = state_truth
    arrays["state_scale"] = state_scale
    arrays["state_varying_mask"] = state_varying_mask
    arrays.update(
        {
            f"state_prediction__{name}": values
            for name, values in state_predictions.items()
        }
    )
    np.savez_compressed(path, **arrays)


def _copy_reproduction_sources(building: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    sources = (
        Path(__file__).resolve(),
        root / "lab/action_dynamics/prototype_hepa_jepa_assessor.py",
        root / "src/quantis_core/edge_dynamics/hepa_jepa.py",
        root / "tests/test_hepa_jepa.py",
        root / "docs/specs/hepa-jepa-telemetry-tracer-v1.md",
        root / "docs/research/jepa-frontier-technique-audit-2026.md",
    )
    reproduction = building / "reproduction"
    reproduction.mkdir()
    for source in sources:
        shutil.copy2(source, reproduction / source.name)


def _write_manifest(building: Path) -> None:
    files = {
        str(path.relative_to(building)): {
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        for path in sorted(building.rglob("*"))
        if path.is_file() and path.name != "artifact-manifest.json"
    }
    _write_json(
        building / "artifact-manifest.json",
        {
            "schema_version": 1,
            "kind": "hepa_jepa_artifact_manifest",
            "files": files,
        },
    )


def _render_report(
    assessment: Mapping[str, Any], *, interpretable: bool
) -> str:
    transfer = assessment["alert_metrics"]["evaluation_transfer"]
    surfaces = assessment["surface_metrics"]["evaluation_transfer"]
    rows = []
    for name in (*MODEL_NAMES, "raw_effect_reference"):
        alert = transfer[name]
        brier = (
            f"{surfaces[name]['brier']:.6f}"
            if name in surfaces
            else "n/a"
        )
        delay = alert["median_post_onset_delay_transitions"]
        rows.append(
            "| "
            + " | ".join(
                (
                    name,
                    brier,
                    f"{alert['control_trajectory_false_alarm_rate']:.3f}",
                    f"{alert['treatment_detection_rate']:.3f}",
                    "n/a" if delay is None else f"{delay:.1f}",
                )
            )
            + " |"
        )
    gates = [
        f"- [{'x' if passed else ' '}] `{name}`"
        for name, passed in assessment["gates"].items()
    ]
    return "\n".join(
        (
            "# HEPA telemetry tracer report",
            "",
            (
                "Status: **frozen interpretable tracer**."
                if interpretable
                else "Status: **NON-INTERPRETABLE SMOKE RUN**."
            ),
            "",
            "| model | Brier | control FPR | treatment detection | median delay |",
            "|---|---:|---:|---:|---:|",
            *rows,
            "",
            f"Decision: `{assessment['decision']}`.",
            "",
            "## Gates",
            "",
            *gates,
            "",
        )
    )


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


def _git_head() -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_status() -> Sequence[str]:
    return subprocess.run(
        ("git", "status", "--short"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()


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
    path.write_text(_pretty_json(value))


def _pretty_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"


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
        "--stage1-steps", type=int, default=FROZEN_STAGE1_STEPS
    )
    parser.add_argument(
        "--stage2-steps", type=int, default=FROZEN_STAGE2_STEPS
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
        stage1_steps=arguments.stage1_steps,
        stage2_steps=arguments.stage2_steps,
        latency_repetitions=arguments.latency_repetitions,
        allow_noninterpretable_smoke=(
            arguments.allow_noninterpretable_smoke
        ),
    )
    print(result)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
