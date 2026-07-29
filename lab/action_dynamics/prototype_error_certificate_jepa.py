#!/usr/bin/env python3
"""Retained runner for the frozen Error-Certificate-JEPA tracer."""

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
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from lab.action_dynamics.prototype_error_certificate_jepa_assessor import (
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
from quantis_core.edge_dynamics.error_certificate_jepa import (
    CertifiedRawDynamics,
    ErrorCertificateJepa,
    ErrorCertificateJepaConfig,
    realized_raw_error,
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
    "artifacts/action-dynamics/prototype-error-certificate-jepa-v3"
)
FROZEN_PRETRAIN_STEPS = 800
IMPLEMENTATION_SOURCE_PATHS = (
    "lab/action_dynamics/prototype_error_certificate_jepa.py",
    "lab/action_dynamics/prototype_error_certificate_jepa_assessor.py",
    "src/quantis_core/edge_dynamics/error_certificate_jepa.py",
    "tests/test_error_certificate_jepa.py",
    "docs/specs/error-certificate-jepa-tracer-v1.md",
    "docs/wayfinding/jepa-implementation-program/"
    "022-test-error-certificate-jepa.md",
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
    """Run, independently assess, and publish one certificate tracer."""

    cache = Path(cache_directory).resolve()
    output = Path(output_directory).resolve()
    building = output.with_name(output.name + ".building")
    if output.exists() or building.exists():
        raise FileExistsError(
            "Error-Certificate-JEPA refuses an existing output"
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
            "non-frozen Error-Certificate-JEPA runs require smoke "
            "permission"
        )
    if not interpretable and output == (
        Path.cwd() / FROZEN_OUTPUT
    ).resolve():
        raise ValueError(
            "Error-Certificate-JEPA smoke cannot use frozen output"
        )
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
                "Error-Certificate-JEPA held topology differs"
            )
        roles = {
            "fit": partitions["fit"].in_distribution,
            "selection": partitions["selection"].in_distribution,
            "calibration": partitions["calibration"].in_distribution,
            "iid_evaluation": partitions[
                "evaluation"
            ].in_distribution,
            "transfer_evaluation": partitions["evaluation"].held_out,
        }
        fit = roles["fit"]
        ownership = fit_owned_feature_mask(fit)
        baseline = ContractiveLowRankDynamics(
            LowRankConfig(rank=32)
        ).fit(fit)
        raw_payload = baseline.to_dict()
        raw_before = _canonical_json_bytes(raw_payload)
        raw_hash = _sha256_bytes(raw_before)
        model_directory = building / "models"
        model_directory.mkdir()
        _write_json(model_directory / "raw.json", raw_payload)

        certificates: Dict[str, ErrorCertificateJepa] = {}
        wrappers: Dict[str, CertifiedRawDynamics] = {}
        training_seconds = {}
        raw_hashes_after_fit = {}
        for name in CELL_NAMES:
            config = replace(
                ErrorCertificateJepaConfig(),
                objective=name,
                pretrain_steps=pretrain_steps,
                checkpoint_interval=max(1, min(100, pretrain_steps)),
                expected_pair_count=expected_pair_count,
            )
            tick = time.perf_counter()
            certificate = (
                ErrorCertificateJepa(config)
                .fit(fit, baseline)
                .select(roles["selection"], baseline)
                .calibrate(roles["calibration"], baseline)
            )
            training_seconds[name] = time.perf_counter() - tick
            certificates[name] = certificate
            wrappers[name] = CertifiedRawDynamics(
                baseline, certificate
            )
            raw_hashes_after_fit[name] = _sha256_bytes(
                _canonical_json_bytes(baseline.to_dict())
            )
            _write_json(
                model_directory / f"{name}-certificate.json",
                certificate.to_dict(),
            )
            _write_json(
                model_directory / f"{name}.json",
                wrappers[name].to_dict(),
            )
        raw_unchanged = all(
            value == raw_hash for value in raw_hashes_after_fit.values()
        )

        raw_distributions = {
            role: baseline.rollout(
                windows.histories,
                windows.future_controls,
                windows.future_actions,
                windows.graph,
            )
            for role, windows in roles.items()
            if role != "fit"
        }
        realized_errors = {
            role: realized_raw_error(
                raw_distributions[role].mean,
                windows.future_states,
                ownership,
            )
            for role, windows in roles.items()
            if role != "fit"
        }
        unadjusted = {
            name: {
                role: model.predict_unadjusted(
                    windows.histories,
                    windows.future_controls,
                    windows.future_actions,
                    raw_distributions[role].mean,
                    windows.graph,
                )
                for role, windows in roles.items()
                if role != "fit"
            }
            for name, model in certificates.items()
        }
        bounds = {
            name: {
                role: model.predict_bound(
                    windows.histories,
                    windows.future_controls,
                    windows.future_actions,
                    raw_distributions[role].mean,
                    windows.graph,
                )
                for role, windows in roles.items()
                if role != "fit"
            }
            for name, model in certificates.items()
        }
        constant_bound = _constant_conformal_bound(
            realized_errors["calibration"],
            roles["calibration"],
        )

        sample = roles["transfer_evaluation"]
        sample_raw = raw_distributions["transfer_evaluation"]
        wrapper_raw = {}
        restoration_evidence: Dict[str, NDArray[Any]] = {}
        all_cells_exact_raw = True
        restoration_max = 0.0
        restored_alert_decisions_match = True
        for name in CELL_NAMES:
            original = wrappers[name].forecast_with_certificate(
                sample.histories,
                sample.future_controls,
                sample.future_actions,
                sample.graph,
            )
            wrapper_raw[name] = original.distribution
            all_cells_exact_raw = (
                all_cells_exact_raw
                and np.array_equal(
                    sample_raw.mean, original.distribution.mean
                )
                and np.array_equal(
                    sample_raw.variance, original.distribution.variance
                )
            )
            restored = CertifiedRawDynamics.from_dict(
                wrappers[name].to_dict()
            ).forecast_with_certificate(
                sample.histories,
                sample.future_controls,
                sample.future_actions,
                sample.graph,
            )
            restoration_max = max(
                restoration_max,
                float(
                    np.max(
                        np.abs(
                            original.error_bound
                            - restored.error_bound
                        )
                    )
                ),
                float(
                    np.max(
                        np.abs(
                            original.distribution.mean
                            - restored.distribution.mean
                        )
                    )
                ),
                float(
                    np.max(
                        np.abs(
                            original.distribution.variance
                            - restored.distribution.variance
                        )
                    )
                ),
            )
            original_alert = (
                realized_errors["transfer_evaluation"]
                > original.error_bound
            )
            restored_alert = (
                realized_errors["transfer_evaluation"]
                > restored.error_bound
            )
            restored_alert_decisions_match = (
                restored_alert_decisions_match
                and np.array_equal(original_alert, restored_alert)
            )
            for field, original_values, restored_values in (
                (
                    "raw_mean",
                    original.distribution.mean,
                    restored.distribution.mean,
                ),
                (
                    "raw_variance",
                    original.distribution.variance,
                    restored.distribution.variance,
                ),
                (
                    "error_bound",
                    original.error_bound,
                    restored.error_bound,
                ),
                ("alerts", original_alert, restored_alert),
            ):
                restoration_evidence[
                    f"restoration_original_{field}__{name}"
                ] = original_values
                restoration_evidence[
                    f"restoration_restored_{field}__{name}"
                ] = restored_values

        public_causality = _rejects_forbidden_inputs(
            certificates["jepa_error_certificate"],
            sample,
            sample_raw.mean,
        )
        candidate = wrappers["jepa_error_certificate"]
        def call() -> Any:
            return candidate.forecast_with_certificate(
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
            timings.append((time.perf_counter_ns() - tick) / 1e6)
        latency = {
            "median_ms": float(np.median(timings)),
            "p95_ms": float(np.quantile(timings, 0.95)),
            "repetitions": latency_repetitions,
        }

        evidence: Dict[str, NDArray[Any]] = {}
        for role, windows in roles.items():
            if role == "fit":
                continue
            evidence[f"actions__{role}"] = (
                windows.future_actions.astype(np.float32)
            )
            evidence[f"realized_error__{role}"] = (
                realized_errors[role].astype(np.float64)
            )
            evidence[f"raw_mean__{role}"] = (
                raw_distributions[role].mean.astype(np.float32)
            )
            evidence[f"raw_variance__{role}"] = (
                raw_distributions[role].variance.astype(np.float32)
            )
            evidence[f"bound__constant_conformal__{role}"] = (
                np.full_like(
                    realized_errors[role],
                    constant_bound,
                    dtype=np.float64,
                )
            )
            for name in CELL_NAMES:
                evidence[f"unadjusted__{name}__{role}"] = (
                    unadjusted[name][role].astype(np.float64)
                )
                evidence[f"bound__{name}__{role}"] = bounds[name][
                    role
                ].astype(np.float64)
        for name in CELL_NAMES:
            evidence[
                f"wrapper_raw_mean__{name}__transfer_evaluation"
            ] = wrapper_raw[name].mean.astype(np.float32)
            evidence[
                f"wrapper_raw_variance__{name}__transfer_evaluation"
            ] = wrapper_raw[name].variance.astype(np.float32)
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
                "training": model.training_parameter_count,
                "inference": model.inference_parameter_count,
            }
            for name, model in certificates.items()
        }
        bundle_bytes = len(
            _canonical_json_bytes(
                wrappers["jepa_error_certificate"].to_dict()
            )
        )
        metadata = {
            "schema_version": 1,
            "kind": "error_certificate_jepa_evidence",
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
            },
            "parameter_counts": parameter_counts,
            "selected_steps": {
                name: model.selected_step
                for name, model in certificates.items()
            },
            "selection_metrics": {
                name: list(model.selection_metrics)
                for name, model in certificates.items()
            },
            "calibration_adjustments": {
                name: model.calibration_adjustment
                for name, model in certificates.items()
            },
            "constant_conformal_bound": constant_bound,
            "raw_sha256_before_fit": raw_hash,
            "raw_sha256_after_each_fit": raw_hashes_after_fit,
            "raw_hash_unchanged": raw_unchanged,
            "all_cells_exact_raw": all_cells_exact_raw,
            "public_causality": public_causality,
            "restoration_max_abs": restoration_max,
            "restored_alert_decisions_match": (
                restored_alert_decisions_match
            ),
            "candidate_bundle_bytes": bundle_bytes,
            "latency": latency,
            "role_use_audit": {
                "fit": "fit",
                "checkpoint_selection": "selection",
                "calibration_adjustment": "calibration",
                "evaluation_roles": [
                    "iid_evaluation",
                    "transfer_evaluation",
                ],
            },
        }
        _write_json(building / "evidence-metadata.json", metadata)
        assessment = assess_stored_bundle(building)
        _write_json(building / "assessment.json", assessment)
        report = {
            "schema_version": 1,
            "kind": "error_certificate_jepa_tracer_v1",
            "evidence_boundary": (
                "single-seed empirical certificate; not a formal "
                "guarantee or sealed production confirmation"
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
                name: model.to_dict()["config"]
                for name, model in certificates.items()
            },
            "training_seconds": training_seconds,
            "selected_steps": metadata["selected_steps"],
            "calibration_adjustments": (
                metadata["calibration_adjustments"]
            ),
            "constant_conformal_bound": constant_bound,
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


def _constant_conformal_bound(
    realized_error: NDArray[Any],
    windows: ActionConditionedWindows,
) -> float:
    trajectory_ids = np.asarray(windows.trajectory_ids)
    maxima = []
    for trajectory in sorted(set(windows.trajectory_ids)):
        rows = np.flatnonzero(trajectory_ids == trajectory)
        if not np.any(windows.future_actions[rows, ..., 1] > 0.5):
            maxima.append(float(np.max(realized_error[rows])))
    if len(maxima) < 2:
        raise ValueError(
            "constant conformal control needs control trajectories"
        )
    return float(np.quantile(maxima, 0.95, method="higher"))


def _rejects_forbidden_inputs(
    model: ErrorCertificateJepa,
    windows: ActionConditionedWindows,
    raw_prediction: NDArray[Any],
) -> bool:
    for keyword, value in (
        ("future_states", windows.future_states[:1]),
        ("pair_ids", windows.matched_pair_ids[:1]),
        ("evaluation_statistics", {"coverage": 1.0}),
    ):
        try:
            model.predict_bound(
                windows.histories[:1],
                windows.future_controls[:1],
                windows.future_actions[:1],
                raw_prediction[:1],
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
            raise ValueError(
                "frozen Error-Certificate-JEPA sources must match HEAD"
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
            "kind": "error_certificate_jepa_manifest",
            "sha256": values,
        },
    )


def _render_report(report: Mapping[str, Any]) -> str:
    assessment = dict(report["assessment"])
    transfer = assessment["roles"]["transfer_evaluation"][
        "certificates"
    ]
    lines = [
        "# Error-Certificate-JEPA tracer",
        "",
        f"Decision: **{assessment['decision']}**",
        "",
        "| certificate | control coverage | simultaneous | mean bound | "
        "treatment detection | delay |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in (*CELL_NAMES, "constant_conformal"):
        score = transfer[name]
        delay = score["median_post_onset_delay_transitions"]
        lines.append(
            f"| {name} | {score['control_point_coverage']:.3f} | "
            f"{score['control_simultaneous_coverage']:.3f} | "
            f"{score['control_mean_bound']:.6g} | "
            f"{score['treatment_trajectory_detection_rate']:.3f} | "
            f"{'-' if delay is None else f'{delay:.3g}'} |"
        )
    lines.extend(
        [
            "",
            "This is an empirical error certificate, not a formal "
            "uncertainty guarantee.",
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


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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
