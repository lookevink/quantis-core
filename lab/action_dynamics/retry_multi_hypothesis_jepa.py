"""Retry multi-hypothesis JEPA on the richer-regime role contract."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import time
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from quantis_core.edge_dynamics.models import (
    ContractiveLowRankDynamics,
    LowRankConfig,
)
from quantis_core.richer_regime_corpus import (
    load_richer_regime_windows,
)

from prototype_multi_hypothesis_jepa import (
    MODEL_NAMES,
    MultiHypothesisJepaPrototype,
    PrototypeConfig,
    _batch_one_latency,
    _capacity_matched_width,
    _raw_mixture,
    _read_distribution,
    _supported_pair_rate,
    _trajectory_balanced_mean,
    _write_distribution,
    _write_role_inputs,
)


def run_retry(
    *,
    fit_campaign: Path,
    selection_campaign: Path,
    output: Path,
    epochs: int = 40,
) -> Mapping[str, Any]:
    """Fit on v1 fit shards and open only v2 selection evidence."""

    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite multi-hypothesis retry: {output}"
        )
    prepared = load_richer_regime_windows(
        {
            "fit": fit_campaign,
            "selection": selection_campaign,
        }
    )
    fit = prepared.windows["fit"]
    selection = prepared.windows["selection"]
    candidate_config = PrototypeConfig(
        component_count=4,
        objective="jepa",
        epochs=epochs,
    )
    one_config = PrototypeConfig(
        component_count=1,
        objective="jepa",
        epochs=epochs,
    )
    supervised_config = PrototypeConfig(
        component_count=4,
        objective="supervised",
        epochs=epochs,
    )
    import torch

    probe = MultiHypothesisJepaPrototype(candidate_config)
    shape = (
        fit.histories.shape[2],
        fit.histories.shape[3],
        fit.future_states.shape[1],
        fit.future_controls.shape[2],
        fit.future_actions.shape[3],
    )
    from prototype_multi_hypothesis_jepa import _build_network

    probe._shape = shape
    probe._network = _build_network(torch, candidate_config, shape)
    matched_width = _capacity_matched_width(
        torch, fit, probe.parameter_count, epochs
    )
    del probe
    capacity_config = PrototypeConfig(
        component_count=1,
        objective="supervised",
        predictor_width=matched_width,
        epochs=epochs,
    )
    models: Dict[str, Any] = {
        "multi_hypothesis_jepa": MultiHypothesisJepaPrototype(
            candidate_config
        ),
        "one_component_jepa": MultiHypothesisJepaPrototype(one_config),
        "capacity_matched_single_gaussian": (
            MultiHypothesisJepaPrototype(capacity_config)
        ),
        "supervised_four_component_mixture": (
            MultiHypothesisJepaPrototype(supervised_config)
        ),
        "raw_low_rank": ContractiveLowRankDynamics(
            LowRankConfig(rank=32)
        ),
    }
    output.mkdir(parents=True)
    (output / "models").mkdir()
    (output / "predictions").mkdir()
    protocol = {
        "schema_version": 1,
        "kind": "richer_regime_multi_hypothesis_jepa_retry_v1",
        "epochs": epochs,
        "fit_campaign": str(fit_campaign),
        "selection_campaign": str(selection_campaign),
        "model_configs": {
            "multi_hypothesis_jepa": asdict(candidate_config),
            "one_component_jepa": asdict(one_config),
            "capacity_matched_single_gaussian": asdict(
                capacity_config
            ),
            "supervised_four_component_mixture": asdict(
                supervised_config
            ),
            "raw_low_rank": asdict(LowRankConfig(rank=32)),
        },
        "safe_null_gates": {
            "candidate_log_score_margin": 0.01,
            "raw_mse_ratio_max": 1.05,
            "supported_pair_rate_min": 0.20
        },
    }
    (output / "protocol.json").write_text(_pretty(protocol))
    training_seconds = {}
    model_evidence = {}
    for name, model in models.items():
        print(f"fitting {name}", flush=True)
        started = time.perf_counter()
        model.fit(fit)
        training_seconds[name] = time.perf_counter() - started
        if isinstance(model, MultiHypothesisJepaPrototype):
            serialized_bytes = model.save(output / "models", name)
            parameter_count = model.parameter_count
            training_metrics = [
                dict(row) for row in model.training_metrics
            ]
        else:
            artifact = model.to_dict()
            path = output / "models" / f"{name}.json"
            path.write_text(_pretty(artifact))
            serialized_bytes = path.stat().st_size
            parameter_count = model.parameter_count
            training_metrics = []
        model_evidence[name] = {
            "parameter_count": parameter_count,
            "serialized_bytes": serialized_bytes,
            "batch_one_latency_ms": _batch_one_latency(
                model, selection
            ),
            "training_seconds": training_seconds[name],
            "training_metrics": training_metrics,
        }
        print(
            f"fitted {name} in {training_seconds[name]:.2f}s",
            flush=True,
        )
    _write_role_inputs(output, "selection", selection)
    for name, model in models.items():
        distribution = (
            model.rollout(
                selection.histories,
                selection.future_controls,
                selection.future_actions,
            )
            if isinstance(model, MultiHypothesisJepaPrototype)
            else _raw_mixture(model, selection)
        )
        _write_distribution(
            output, "selection", name, distribution
        )
    assessment = assess_selection_retry(output)
    result = {
        "schema_version": 1,
        "kind": "richer_regime_multi_hypothesis_jepa_result_v1",
        "evidence_boundary": (
            "fit and selection only; calibration and evaluation remain "
            "uncollected and unopened"
        ),
        "protocol": protocol,
        "data_identity": {
            "source_assessment_sha256s": (
                prepared.source_assessment_sha256s
            ),
            "compiler_artifact_sha256": _sha256(
                prepared.compiler_artifact
            ),
            "fit_window_count": len(fit.histories),
            "selection_window_count": len(selection.histories),
            "fit_pair_count": len(set(fit.matched_pair_ids)),
            "selection_pair_count": len(
                set(selection.matched_pair_ids)
            ),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": str(torch.__version__),
        },
        "models": model_evidence,
        "assessment": assessment,
    }
    (output / "result.json").write_text(_pretty(result))
    (output / "report.md").write_text(_report(result))
    manifest = {
        "schema_version": 1,
        "kind": "richer_regime_multi_hypothesis_artifact_manifest",
        "sha256": {
            path.relative_to(output).as_posix(): _file_sha256(path)
            for path in sorted(output.rglob("*"))
            if path.is_file()
        },
    }
    (output / "artifact-manifest.json").write_text(
        _pretty(manifest)
    )
    return result


def assess_selection_retry(directory: Path) -> Mapping[str, Any]:
    """Recompute the safe-null decision from stored selection arrays."""

    with np.load(
        directory / "predictions" / "selection-inputs.npz",
        allow_pickle=False,
    ) as arrays:
        observed = np.asarray(arrays["observed"], dtype=np.float64)
        action_active = np.asarray(
            arrays["action_active"], dtype=np.bool_
        )
        trajectory_ids = tuple(
            str(value) for value in arrays["trajectory_ids"]
        )
    metrics = {}
    for name in MODEL_NAMES:
        distribution = _read_distribution(
            directory, "selection", name
        )
        nll = distribution.negative_log_likelihood(observed)
        compatible = distribution.as_trajectory_distribution()
        squared = np.square(compatible.mean - observed)
        metrics[name] = {
            "trajectory_balanced_log_score": (
                _trajectory_balanced_mean(nll, trajectory_ids)
            ),
            "normalized_mse_overall": float(np.mean(squared)),
            "normalized_mse_action_overlap": float(
                np.mean(squared[action_active])
            ),
            "supported_pair_rate_action_overlap": (
                _supported_pair_rate(
                    distribution, np.any(action_active, axis=1)
                )
            ),
            "effective_hypothesis_count": float(
                np.mean(
                    np.exp(
                        -np.sum(
                            distribution.weight
                            * np.log(distribution.weight),
                            axis=1,
                        )
                    )
                )
            ),
            "finite": bool(
                np.all(np.isfinite(nll))
                and np.all(np.isfinite(compatible.mean))
                and np.all(np.isfinite(compatible.variance))
            ),
        }
    candidate = metrics["multi_hypothesis_jepa"]
    raw = metrics["raw_low_rank"]
    gates = {
        "log_score_beats_one_component_by_0_01": (
            candidate["trajectory_balanced_log_score"]
            <= metrics["one_component_jepa"][
                "trajectory_balanced_log_score"
            ]
            - 0.01
        ),
        "log_score_beats_supervised_mixture_by_0_01": (
            candidate["trajectory_balanced_log_score"]
            <= metrics["supervised_four_component_mixture"][
                "trajectory_balanced_log_score"
            ]
            - 0.01
        ),
        "overall_mse_within_5_percent_of_raw": (
            candidate["normalized_mse_overall"]
            <= 1.05 * raw["normalized_mse_overall"]
        ),
        "action_overlap_mse_within_5_percent_of_raw": (
            candidate["normalized_mse_action_overlap"]
            <= 1.05 * raw["normalized_mse_action_overlap"]
        ),
        "supported_pair_rate_at_least_20_percent": (
            candidate["supported_pair_rate_action_overlap"] >= 0.20
        ),
        "all_outputs_finite": all(
            row["finite"] for row in metrics.values()
        ),
    }
    passed = all(gates.values())
    return {
        "schema_version": 1,
        "kind": "richer_regime_multi_hypothesis_safe_null",
        "selection_metrics": metrics,
        "gates": gates,
        "safe_null_passed": passed,
        "decision": (
            "collect_calibration_and_evaluation"
            if passed
            else "reject_multi_hypothesis_retry"
        ),
    }


def diagnose_failed_retry(
    *,
    fit_campaign: Path,
    selection_campaign: Path,
    output: Path,
) -> Mapping[str, Any]:
    """Diagnose stored models after a selection distribution failure."""

    if not output.is_dir() or (output / "result.json").exists():
        raise ValueError("stored retry is not an unfinished attempt")
    prepared = load_richer_regime_windows(
        {
            "fit": fit_campaign,
            "selection": selection_campaign,
        }
    )
    selection = prepared.windows["selection"]
    raw_payload = json.loads(
        (output / "models" / "raw_low_rank.json").read_text()
    )
    if not isinstance(raw_payload, dict):
        raise ValueError("stored raw model is invalid")
    raw_model = ContractiveLowRankDynamics.from_dict(raw_payload)
    _write_distribution(
        output,
        "selection",
        "raw_low_rank",
        _raw_mixture(raw_model, selection),
    )
    with np.load(
        output / "predictions" / "selection-inputs.npz",
        allow_pickle=False,
    ) as arrays:
        observed = np.asarray(arrays["observed"], dtype=np.float64)
        action_active = np.asarray(
            arrays["action_active"], dtype=np.bool_
        )
        trajectory_ids = tuple(
            str(value) for value in arrays["trajectory_ids"]
        )
    valid_names = (
        "multi_hypothesis_jepa",
        "one_component_jepa",
        "capacity_matched_single_gaussian",
        "raw_low_rank",
    )
    metrics = {
        name: _stored_metrics(
            output,
            name,
            observed,
            action_active,
            trajectory_ids,
        )
        for name in valid_names
    }
    supervised = MultiHypothesisJepaPrototype.load(
        output / "models",
        "supervised_four_component_mixture",
    )
    invalidity = _raw_distribution_diagnostics(
        supervised, selection
    )
    candidate = metrics["multi_hypothesis_jepa"]
    raw = metrics["raw_low_rank"]
    independent_gates = {
        "candidate_beats_one_component_log_score_by_0_01": (
            candidate["trajectory_balanced_log_score"]
            <= metrics["one_component_jepa"][
                "trajectory_balanced_log_score"
            ]
            - 0.01
        ),
        "candidate_overall_mse_within_5_percent_of_raw": (
            candidate["normalized_mse_overall"]
            <= 1.05 * raw["normalized_mse_overall"]
        ),
        "candidate_action_mse_within_5_percent_of_raw": (
            candidate["normalized_mse_action_overlap"]
            <= 1.05 * raw["normalized_mse_action_overlap"]
        ),
        "candidate_supported_pair_rate_at_least_20_percent": (
            candidate["supported_pair_rate_action_overlap"] >= 0.20
        ),
        "candidate_outputs_finite": bool(candidate["finite"]),
    }
    rejected_independently = not all(independent_gates.values())
    diagnosis = {
        "schema_version": 1,
        "kind": "richer_regime_multi_hypothesis_failure_diagnosis",
        "failed_model": "supervised_four_component_mixture",
        "failed_boundary": "selection distribution validation",
        "invalidity": invalidity,
        "valid_model_selection_metrics": metrics,
        "independent_candidate_gates": independent_gates,
        "decision": (
            "reject_multi_hypothesis_retry_independent_of_invalid_null"
            if rejected_independently
            else "inconclusive_due_to_invalid_supervised_null"
        ),
        "calibration_or_evaluation_opened": False,
    }
    (output / "failure-diagnosis.json").write_text(
        _pretty(diagnosis)
    )
    (output / "report.md").write_text(
        "# Richer-regime multi-hypothesis JEPA retry\n\n"
        f"Decision: `{diagnosis['decision']}`\n\n"
        "The stored supervised four-component null produced an "
        "invalid selection distribution. No model was retrained, and "
        "calibration/evaluation remained unopened.\n"
    )
    manifest = {
        "schema_version": 1,
        "kind": "richer_regime_multi_hypothesis_artifact_manifest",
        "sha256": {
            path.relative_to(output).as_posix(): _file_sha256(path)
            for path in sorted(output.rglob("*"))
            if path.is_file()
            and path.name != "artifact-manifest.json"
        },
    }
    (output / "artifact-manifest.json").write_text(
        _pretty(manifest)
    )
    return diagnosis


def _stored_metrics(
    directory: Path,
    name: str,
    observed: np.ndarray,
    action_active: np.ndarray,
    trajectory_ids: tuple[str, ...],
) -> Mapping[str, Any]:
    distribution = _read_distribution(
        directory, "selection", name
    )
    nll = distribution.negative_log_likelihood(observed)
    compatible = distribution.as_trajectory_distribution()
    squared = np.square(compatible.mean - observed)
    return {
        "trajectory_balanced_log_score": _trajectory_balanced_mean(
            nll, trajectory_ids
        ),
        "normalized_mse_overall": float(np.mean(squared)),
        "normalized_mse_action_overlap": float(
            np.mean(squared[action_active])
        ),
        "supported_pair_rate_action_overlap": (
            _supported_pair_rate(
                distribution, np.any(action_active, axis=1)
            )
        ),
        "finite": bool(
            np.all(np.isfinite(nll))
            and np.all(np.isfinite(compatible.mean))
            and np.all(np.isfinite(compatible.variance))
        ),
    }


def _raw_distribution_diagnostics(
    model: MultiHypothesisJepaPrototype,
    windows: Any,
) -> Mapping[str, Any]:
    import torch

    network, _ = model._fitted()
    minimum_weight = float("inf")
    maximum_weight_sum_error = 0.0
    nonfinite_mean = 0
    nonfinite_variance = 0
    nonfinite_weight = 0
    below_weight_floor = 0
    network.eval()
    with torch.no_grad():
        for start in range(0, len(windows.histories), 256):
            stop = start + 256
            output = network.predict(
                torch.as_tensor(
                    windows.histories[start:stop],
                    dtype=torch.float32,
                ),
                torch.as_tensor(
                    windows.future_controls[start:stop],
                    dtype=torch.float32,
                ),
                torch.as_tensor(
                    windows.future_actions[start:stop],
                    dtype=torch.float32,
                ),
            )
            mean = output["component_mean"].detach().numpy()
            variance = output["component_variance"].detach().numpy()
            weight = output["weight"].detach().numpy()
            nonfinite_mean += int(np.sum(~np.isfinite(mean)))
            nonfinite_variance += int(
                np.sum(~np.isfinite(variance))
            )
            nonfinite_weight += int(np.sum(~np.isfinite(weight)))
            minimum_weight = min(
                minimum_weight, float(np.min(weight))
            )
            below_weight_floor += int(np.sum(weight < 1e-12))
            maximum_weight_sum_error = max(
                maximum_weight_sum_error,
                float(
                    np.max(
                        np.abs(np.sum(weight, axis=1) - 1.0)
                    )
                ),
            )
    return {
        "minimum_weight": minimum_weight,
        "weight_count_below_1e_12": below_weight_floor,
        "maximum_weight_sum_error": maximum_weight_sum_error,
        "nonfinite_mean_count": nonfinite_mean,
        "nonfinite_variance_count": nonfinite_variance,
        "nonfinite_weight_count": nonfinite_weight,
    }


def _report(result: Mapping[str, Any]) -> str:
    assessment = result["assessment"]
    lines = [
        "# Richer-regime multi-hypothesis JEPA retry",
        "",
        f"Decision: `{assessment['decision']}`",
        "",
        "Fit and selection evidence only. Calibration and evaluation "
        "remain unopened unless every safe-null gate passes.",
        "",
        "## Safe-null gates",
        "",
    ]
    lines.extend(
        f"- `{name}`: {'PASS' if passed else 'FAIL'}"
        for name, passed in assessment["gates"].items()
    )
    lines.append("")
    return "\n".join(lines)


def _pretty(payload: Mapping[str, Any]) -> str:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def _sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(arguments: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fit-campaign",
        type=Path,
        default=Path(
            "artifacts/action-dynamics/richer-regime-retry-v1"
        ),
    )
    parser.add_argument(
        "--selection-campaign",
        type=Path,
        default=Path(
            "artifacts/action-dynamics/richer-regime-retry-v2"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/action-dynamics/"
            "richer-regime-multi-hypothesis-jepa-v1"
        ),
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--diagnose-stored", action="store_true")
    parsed = parser.parse_args(arguments)
    if parsed.diagnose_stored:
        diagnosis = diagnose_failed_retry(
            fit_campaign=parsed.fit_campaign,
            selection_campaign=parsed.selection_campaign,
            output=parsed.output,
        )
        print(json.dumps(diagnosis, indent=2, sort_keys=True))
        return 0
    result = run_retry(
        fit_campaign=parsed.fit_campaign,
        selection_campaign=parsed.selection_campaign,
        output=parsed.output,
        epochs=parsed.epochs,
    )
    print(
        json.dumps(
            result["assessment"], indent=2, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
