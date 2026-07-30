"""Independent stored-array assessor for the frozen MPRM-JEPA tracer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from quantis_core.action_conditioned_dynamics import (
    MixtureTrajectoryDistribution,
)
from quantis_core.mprm_jepa import (
    MprmJepaProtocol,
    paired_randomization_p_value,
)


MODEL_NAMES = (
    "raw_rank_32_predictive_core",
    "one_component_anchored_jepa_residual",
    "supervised_four_component_mean_preserving_residual_mixture",
    "capacity_matched_anchored_single_gaussian",
    "unanchored_four_component_jepa_diagnostic",
    "mprm_jepa_candidate",
)


def assess_stored_mprm_selection(
    *,
    protocol_path: Path,
    model_freeze_manifest: Path,
    qualified_corpus: Path,
    predictions_directory: Path,
    expected_model_freeze_sha256: str,
    expected_qualified_corpus_sha256: str,
    expected_prediction_manifest_sha256: str,
) -> Mapping[str, Any]:
    """Recompute the binary decision from externally pinned stored evidence."""

    protocol = MprmJepaProtocol.from_dict(_read_object(protocol_path))
    prediction_manifest_path = (
        predictions_directory / "prediction-manifest.json"
    )
    prediction_manifest = _read_object(prediction_manifest_path)
    prediction_hashes = prediction_manifest.get("sha256")
    if (
        _file_sha256(model_freeze_manifest)
        != expected_model_freeze_sha256
        or _file_sha256(prediction_manifest_path)
        != expected_prediction_manifest_sha256
        or not isinstance(prediction_hashes, dict)
        or prediction_manifest.get("model_freeze_manifest_sha256")
        != expected_model_freeze_sha256
        or prediction_manifest.get("qualified_corpus_sha256")
        != expected_qualified_corpus_sha256
        or any(
            not (predictions_directory / name).is_file()
            or _file_sha256(predictions_directory / name) != expected
            for name, expected in prediction_hashes.items()
        )
    ):
        raise ValueError("MPRM-JEPA frozen input hash differs")
    qualified = _read_object(qualified_corpus)
    bindings = qualified.get("campaign_bindings")
    source_manifest = qualified.get("source_content_manifest")
    if (
        qualified.get("status") != "qualified"
        or qualified.get("qualified_corpus_sha256")
        != expected_qualified_corpus_sha256
        or not isinstance(bindings, dict)
        or bindings.get("model_freeze_manifest_sha256")
        != expected_model_freeze_sha256
        or bindings.get("candidate_protocol_sha256")
        != _file_sha256(protocol_path)
        or not isinstance(source_manifest, dict)
        or _canonical_sha256(source_manifest)
        != expected_qualified_corpus_sha256
    ):
        raise ValueError("MPRM-JEPA qualified corpus identity differs")
    freeze = _read_object(model_freeze_manifest)
    artifact_hashes = freeze.get("artifact_sha256")
    source_hashes = freeze.get("source_sha256")
    if not isinstance(artifact_hashes, dict) or any(
        _file_sha256(model_freeze_manifest.parent / relative) != expected
        for relative, expected in artifact_hashes.items()
    ):
        raise ValueError("MPRM-JEPA frozen model artifact differs")
    repository = protocol_path.resolve().parents[2]
    if not isinstance(source_hashes, dict) or any(
        not (repository / relative).is_file()
        or _file_sha256(repository / relative) != expected
        for relative, expected in source_hashes.items()
    ):
        raise ValueError("MPRM-JEPA frozen implementation source differs")
    runtime_identity = freeze.get("runtime_identity")
    current_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if (
        not isinstance(runtime_identity, dict)
        or runtime_identity.get("source_commit") != current_commit
    ):
        raise ValueError("MPRM-JEPA frozen source commit differs")
    inputs = _read_inputs(predictions_directory)
    metrics: Dict[str, Mapping[str, Any]] = {}
    pair_scores: Dict[str, Mapping[str, float]] = {}
    for name in MODEL_NAMES:
        distribution = _read_distribution(predictions_directory, name)
        nll = distribution.negative_log_likelihood(inputs["observed"])
        compatible = distribution.as_trajectory_distribution()
        squared = np.square(compatible.mean - inputs["observed"])
        per_sample_mse = np.mean(squared, axis=(1, 2, 3))
        active_mask = inputs["action_active"]
        active_sample = np.any(active_mask, axis=(1, 2, 3))
        active_mse = np.asarray(
            [
                float(np.mean(row_squared[row_mask]))
                for row_squared, row_mask in zip(
                    squared[active_sample],
                    active_mask[active_sample],
                )
            ],
            dtype=np.float64,
        )
        metrics[name] = {
            "pair_balanced_log_score": _pair_balanced_mean(
                nll, inputs["trajectory_ids"], inputs["matched_pair_ids"]
            ),
            "overall_mse": _pair_balanced_mean(
                per_sample_mse,
                inputs["trajectory_ids"],
                inputs["matched_pair_ids"],
            ),
            "action_overlap_mse": _group_balanced_mean(
                active_mse,
                tuple(
                    value
                    for value, selected in zip(
                        inputs["trajectory_ids"], active_sample
                    )
                    if selected
                ),
                tuple(
                    value
                    for value, selected in zip(
                        inputs["matched_pair_ids"], active_sample
                    )
                    if selected
                ),
            ),
            "energy_score": _pair_balanced_mean(
                _energy_scores(
                    distribution,
                    inputs["observed"],
                    inputs["sample_ids"],
                ),
                inputs["trajectory_ids"],
                inputs["matched_pair_ids"],
            ),
            "supported_pair_rate": _supported_pair_rate(
                distribution,
                np.any(inputs["action_active"], axis=(1, 2, 3)),
            ),
        }
        pair_scores[name] = _pair_balanced_values(
            nll, inputs["trajectory_ids"], inputs["matched_pair_ids"]
        )
    candidate = metrics["mprm_jepa_candidate"]
    raw = metrics["raw_rank_32_predictive_core"]
    candidate_distribution = _read_distribution(
        predictions_directory, "mprm_jepa_candidate"
    )
    raw_distribution = _read_distribution(
        predictions_directory, "raw_rank_32_predictive_core"
    )
    mean_error = float(
        np.max(
            np.abs(
                candidate_distribution.as_trajectory_distribution().mean
                - raw_distribution.as_trajectory_distribution().mean
            )
        )
    )
    pair_ids = sorted(pair_scores["mprm_jepa_candidate"])
    differences = np.asarray(
        [
            pair_scores["mprm_jepa_candidate"][pair_id]
            - pair_scores["raw_rank_32_predictive_core"][pair_id]
            for pair_id in pair_ids
        ],
        dtype=np.float64,
    )
    randomization_p = paired_randomization_p_value(
        differences, seed=26072932, draws=99999
    )
    model_evidence = freeze.get("model_evidence")
    candidate_evidence = (
        model_evidence.get("mprm_jepa_candidate")
        if isinstance(model_evidence, dict)
        else None
    )
    envelope = protocol.payload["edge_envelope"]
    margin = 0.01
    gates = {
        "beats_raw_by_0_01": candidate["pair_balanced_log_score"]
        <= raw["pair_balanced_log_score"] - margin,
        "beats_one_component_by_0_01": (
            candidate["pair_balanced_log_score"]
            <= metrics["one_component_anchored_jepa_residual"][
                "pair_balanced_log_score"
            ]
            - margin
        ),
        "beats_supervised_mixture_by_0_01": (
            candidate["pair_balanced_log_score"]
            <= metrics[
                "supervised_four_component_mean_preserving_residual_mixture"
            ]["pair_balanced_log_score"]
            - margin
        ),
        "beats_capacity_matched_gaussian_by_0_01": (
            candidate["pair_balanced_log_score"]
            <= metrics["capacity_matched_anchored_single_gaussian"][
                "pair_balanced_log_score"
            ]
            - margin
        ),
        "mean_identity_within_1e_10": mean_error <= 1e-10,
        "overall_mse_within_0_1_percent": candidate["overall_mse"]
        <= raw["overall_mse"] * 1.001,
        "action_mse_within_0_1_percent": candidate[
            "action_overlap_mse"
        ]
        <= raw["action_overlap_mse"] * 1.001,
        "energy_score_noninferior_1_percent": candidate["energy_score"]
        <= raw["energy_score"] * 1.01,
        "supported_pair_rate_at_least_20_percent": candidate[
            "supported_pair_rate"
        ]
        >= 0.20,
        "no_workload_family_regresses_over_0_01": (
            _family_gate(
                pair_scores["mprm_jepa_candidate"],
                pair_scores["raw_rank_32_predictive_core"],
                inputs["pair_workload_families"],
            )
        ),
        "paired_randomization_p_at_most_0_05": randomization_p <= 0.05,
        "all_scores_finite": all(
            np.isfinite(float(value))
            for model in metrics.values()
            for value in model.values()
        ),
        "serialized_size_at_most_4_mib": (
            isinstance(candidate_evidence, dict)
            and candidate_evidence.get("serialized_bytes", float("inf"))
            <= envelope["serialized_bytes_max"]
        ),
        "batch_one_p95_at_most_5_ms": (
            isinstance(candidate_evidence, dict)
            and candidate_evidence.get(
                "batch_one_p95_latency_ms", float("inf")
            )
            <= envelope["batch_one_p95_latency_ms_max"]
        ),
        "fresh_process_prediction_parity": freeze.get(
            "fresh_process_prediction_parity"
        )
        is True,
        "runtime_identity_frozen": isinstance(
            freeze.get("runtime_identity"), dict
        ),
        "no_network_or_accelerator_dependency": (
            envelope["network_dependency"] is False
            and envelope["accelerator_dependency"] is False
        ),
    }
    passed = all(gates.values())
    return {
        "schema_version": 1,
        "kind": "mprm_jepa_selection_assessment",
        "status": "passed" if passed else "failed",
        "decision": (
            "advance_exact_recipe_to_separate_evaluation_proposal"
            if passed
            else "reject_exact_mprm_jepa_recipe"
        ),
        "metrics": metrics,
        "mean_identity_max_absolute_error": mean_error,
        "paired_randomization_p_value": randomization_p,
        "gates": gates,
        "prediction_sha256": {
            path.name: _file_sha256(path)
            for path in sorted(predictions_directory.glob("*.npz"))
        },
        "prediction_manifest_sha256": (
            expected_prediction_manifest_sha256
        ),
        "evidence_boundary": protocol.payload["evidence_boundary"],
    }


def _read_inputs(directory: Path) -> Mapping[str, Any]:
    with np.load(
        directory / "selection-inputs.npz", allow_pickle=False
    ) as arrays:
        return {
            "observed": np.asarray(arrays["observed"], dtype=np.float64),
            "action_active": np.asarray(
                arrays["action_active"], dtype=np.bool_
            ),
            "trajectory_ids": tuple(
                str(value) for value in arrays["trajectory_ids"]
            ),
            "matched_pair_ids": tuple(
                str(value) for value in arrays["matched_pair_ids"]
            ),
            "sample_ids": tuple(
                str(value) for value in arrays["sample_ids"]
            ),
            "pair_workload_families": json.loads(
                str(arrays["pair_workload_families_json"].item())
            ),
        }


def _read_distribution(
    directory: Path, name: str
) -> MixtureTrajectoryDistribution:
    with np.load(
        directory / f"selection-{name}.npz", allow_pickle=False
    ) as arrays:
        return MixtureTrajectoryDistribution(
            component_mean=np.asarray(
                arrays["component_mean"], dtype=np.float64
            ),
            component_variance=np.asarray(
                arrays["component_variance"], dtype=np.float64
            ),
            weight=np.asarray(arrays["weight"], dtype=np.float64),
        )


def _pair_balanced_values(
    values: NDArray[np.float64],
    trajectory_ids: Tuple[str, ...],
    pair_ids: Tuple[str, ...],
) -> Dict[str, float]:
    trajectories: Dict[str, list[float]] = {}
    ownership: Dict[str, str] = {}
    for value, trajectory_id, pair_id in zip(
        values, trajectory_ids, pair_ids
    ):
        if ownership.setdefault(trajectory_id, pair_id) != pair_id:
            raise ValueError("MPRM-JEPA trajectory crosses pairs")
        trajectories.setdefault(trajectory_id, []).append(float(value))
    pairs: Dict[str, list[float]] = {}
    for trajectory_id, rows in trajectories.items():
        pairs.setdefault(ownership[trajectory_id], []).append(
            float(np.mean(rows))
        )
    if not pairs or any(len(rows) != 2 for rows in pairs.values()):
        raise ValueError("MPRM-JEPA pair balancing is incomplete")
    return {
        pair_id: float(np.mean(rows))
        for pair_id, rows in sorted(pairs.items())
    }


def _pair_balanced_mean(
    values: NDArray[np.float64],
    trajectory_ids: Tuple[str, ...],
    pair_ids: Tuple[str, ...],
) -> float:
    return float(
        np.mean(
            list(
                _pair_balanced_values(
                    values, trajectory_ids, pair_ids
                ).values()
            )
        )
    )


def _group_balanced_mean(
    values: NDArray[np.float64],
    trajectory_ids: Tuple[str, ...],
    pair_ids: Tuple[str, ...],
) -> float:
    trajectories: Dict[str, list[float]] = {}
    ownership: Dict[str, str] = {}
    for value, trajectory_id, pair_id in zip(
        values, trajectory_ids, pair_ids
    ):
        if ownership.setdefault(trajectory_id, pair_id) != pair_id:
            raise ValueError("MPRM-JEPA trajectory crosses pairs")
        trajectories.setdefault(trajectory_id, []).append(float(value))
    pairs: Dict[str, list[float]] = {}
    for trajectory_id, rows in trajectories.items():
        pairs.setdefault(ownership[trajectory_id], []).append(
            float(np.mean(rows))
        )
    if not pairs:
        raise ValueError("MPRM-JEPA grouped score is empty")
    return float(
        np.mean([np.mean(rows) for rows in pairs.values()])
    )


def _supported_pair_rate(
    distribution: MixtureTrajectoryDistribution,
    sample_mask: NDArray[np.bool_],
) -> float:
    supported = np.zeros(len(distribution.weight), dtype=np.bool_)
    for left in range(distribution.component_mean.shape[1]):
        for right in range(
            left + 1, distribution.component_mean.shape[1]
        ):
            variance = 0.5 * (
                distribution.component_variance[:, left]
                + distribution.component_variance[:, right]
            )
            separation = np.sqrt(
                np.mean(
                    np.square(
                        distribution.component_mean[:, left]
                        - distribution.component_mean[:, right]
                    )
                    / variance,
                    axis=(1, 2, 3),
                )
            )
            supported |= (
                (distribution.weight[:, left] >= 0.10)
                & (distribution.weight[:, right] >= 0.10)
                & (separation >= 1.0)
            )
    return float(np.mean(supported[sample_mask]))


def _energy_scores(
    distribution: MixtureTrajectoryDistribution,
    observed: NDArray[np.float64],
    sample_ids: Tuple[str, ...],
) -> NDArray[np.float64]:
    scores = []
    draws = 256
    for index, sample_id in enumerate(sample_ids):
        seed = int.from_bytes(
            hashlib.sha256(
                ("multi-hypothesis-energy-v1:" + sample_id).encode()
            ).digest()[:8],
            "big",
        )
        rng = np.random.default_rng(seed)
        cumulative = np.cumsum(distribution.weight[index])
        component = np.searchsorted(
            cumulative, rng.random(draws), side="right"
        )
        paired_component = np.searchsorted(
            cumulative, rng.random(draws), side="right"
        )
        half_standard = rng.normal(
            size=(draws // 2,) + observed[index].shape
        )
        standard = np.concatenate((half_standard, -half_standard))
        half_paired_standard = rng.normal(
            size=(draws // 2,) + observed[index].shape
        )
        paired_standard = np.concatenate(
            (half_paired_standard, -half_paired_standard)
        )
        first = (
            distribution.component_mean[index, component]
            + np.sqrt(
                distribution.component_variance[index, component]
            )
            * standard
        )
        second = (
            distribution.component_mean[index, paired_component]
            + np.sqrt(
                distribution.component_variance[
                    index, paired_component
                ]
            )
            * paired_standard
        )
        scale = np.sqrt(float(observed[index].size))
        scores.append(
            float(
                np.mean(
                    np.linalg.norm(
                        (first - observed[index]).reshape(draws, -1),
                        axis=1,
                    )
                )
                / scale
                - 0.5
                * np.mean(
                    np.linalg.norm(
                        (first - second).reshape(draws, -1), axis=1
                    )
                )
                / scale
            )
        )
    return np.asarray(scores, dtype=np.float64)


def _family_gate(
    candidate: Mapping[str, float],
    raw: Mapping[str, float],
    family_by_pair: Mapping[str, str],
) -> bool:
    for family in (
        "steady",
        "ramp_or_burst",
        "periodic_or_multiphase",
    ):
        pair_ids = sorted(
            pair_id
            for pair_id, value in family_by_pair.items()
            if value == family
        )
        if not pair_ids or (
            np.mean([candidate[pair_id] for pair_id in pair_ids])
            > np.mean([raw[pair_id] for pair_id in pair_ids]) + 0.01
        ):
            return False
    return True


def _read_object(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def main(arguments: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--model-freeze-manifest", type=Path, required=True)
    parser.add_argument("--qualified-corpus", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--expected-model-freeze-sha256", required=True)
    parser.add_argument("--expected-qualified-corpus-sha256", required=True)
    parser.add_argument("--expected-prediction-manifest-sha256", required=True)
    parsed = parser.parse_args(arguments)
    result = assess_stored_mprm_selection(
        protocol_path=parsed.protocol,
        model_freeze_manifest=parsed.model_freeze_manifest,
        qualified_corpus=parsed.qualified_corpus,
        predictions_directory=parsed.predictions,
        expected_model_freeze_sha256=(
            parsed.expected_model_freeze_sha256
        ),
        expected_qualified_corpus_sha256=(
            parsed.expected_qualified_corpus_sha256
        ),
        expected_prediction_manifest_sha256=(
            parsed.expected_prediction_manifest_sha256
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
