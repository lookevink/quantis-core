"""Retained assessor for the corrected v2 multi-hypothesis decision.

The assessor content-binds the immutable v1 numeric sidecars and writes its
correction to a fresh output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from quantis_core.action_conditioned_dynamics import (
    MixtureTrajectoryDistribution,
)


MODEL_NAMES = (
    "multi_hypothesis_jepa",
    "one_component_jepa",
    "capacity_matched_single_gaussian",
    "supervised_four_component_mixture",
    "raw_low_rank",
)
SINGLE_COMPONENT_CONTROLS = (
    "one_component_jepa",
    "capacity_matched_single_gaussian",
    "raw_low_rank",
)
EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "1a464d6182b4f0abd6987496453ef5f9ef403d9ab62779ffa87e7511184528f8"
)
EXPECTED_SOURCE_RESULT_SHA256 = (
    "295ac75bbff1f85f3cb72833b11e6543fb082a5e027e9f20f814ff529a6c1760"
)
EXPECTED_CORPUS_SHA256 = (
    "df03af282f48591216251f22f934b97e1555147df9ed6f6c6d1f7684f9644a26"
)
EXPECTED_PREPROCESSING_MANIFEST_SHA256 = (
    "d02afa33a5977cba69b255cd1a5f31470751b6681da1ecae31b11e86307e65b1"
)
EXPECTED_PREPROCESSING_CACHE_ADDRESS = (
    "eb54271132f88c9a431b01e786ea66279a563776434cca2290e47e6b7ae9b3ff"
)


def correct_prototype_assessment(
    source_directory: Path,
    output_directory: Path,
) -> Mapping[str, Any]:
    """Write one immutable corrected assessment over v1 sidecars."""

    if output_directory.exists():
        raise FileExistsError(
            f"refusing to overwrite corrected result: {output_directory}"
        )
    source_identity = _verify_source(source_directory)
    source_result = _read_object(
        source_directory / "prototype-result.json"
    )
    if not all(
        bool(row["restoration_parity"])
        for row in dict(source_result["models"]).values()
    ):
        raise ValueError("source model restoration parity failed")
    assessment = assess_source_arrays(source_directory)
    output_directory.mkdir(parents=True)
    protocol = {
        "schema_version": 2,
        "kind": "multi_hypothesis_jepa_prototype_v2_correction",
        "source_protocol": (
            "docs/specs/multi-hypothesis-jepa-prototype-v1.md"
        ),
        "correction_protocol": (
            "docs/specs/multi-hypothesis-jepa-prototype-v2.md"
        ),
        "source_seed": 307,
        "model_refit": False,
        "selection_uses_evaluation_roles": False,
        "safe_null_fail_fast": True,
    }
    (output_directory / "protocol.json").write_text(
        _pretty_json(protocol)
    )
    (output_directory / "source-identity.json").write_text(
        _pretty_json(source_identity)
    )
    (output_directory / "assessment.json").write_text(
        _pretty_json(assessment)
    )
    (output_directory / "report.md").write_text(
        _report_markdown(assessment, source_identity)
    )
    manifest = {
        "schema_version": 1,
        "kind": "multi_hypothesis_jepa_prototype_v2_manifest",
        "sha256": {
            path.name: _file_sha256(path)
            for path in sorted(output_directory.iterdir())
            if path.is_file()
        },
    }
    (output_directory / "artifact-manifest.json").write_text(
        _pretty_json(manifest)
    )
    return assessment


def assess_source_arrays(
    source_directory: Path,
) -> Mapping[str, Any]:
    """Purely recompute selection without evaluation-role influence."""

    role_metrics: Dict[str, Dict[str, Any]] = {}
    for role in ("selection", "transfer"):
        inputs = _read_role_inputs(source_directory, role)
        metrics = {}
        for name in MODEL_NAMES:
            distribution = _read_distribution(
                source_directory, role, name
            )
            nll = distribution.negative_log_likelihood(
                inputs["observed"]
            )
            compatible = distribution.as_trajectory_distribution()
            squared = np.square(
                compatible.mean - inputs["observed"]
            )
            metrics[name] = {
                "pair_balanced_log_score": _pair_balanced_mean(
                    nll,
                    inputs["trajectory_ids"],
                    inputs["matched_pair_ids"],
                ),
                "normalized_mse_overall": float(np.mean(squared)),
                "normalized_mse_action_overlap": float(
                    np.mean(squared[inputs["action_active"]])
                ),
                "supported_pair_rate_action_overlap": (
                    _supported_pair_rate(
                        distribution,
                        np.any(inputs["action_active"], axis=1),
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
        role_metrics[role] = metrics

    selection = role_metrics["selection"]
    candidate = selection["multi_hypothesis_jepa"]
    raw = selection["raw_low_rank"]
    gates = {
        "selection_log_score_beats_one_component_jepa_by_0_01": (
            candidate["pair_balanced_log_score"]
            <= selection["one_component_jepa"][
                "pair_balanced_log_score"
            ]
            - 0.01
        ),
        "selection_log_score_beats_supervised_mixture_by_0_01": (
            candidate["pair_balanced_log_score"]
            <= selection["supervised_four_component_mixture"][
                "pair_balanced_log_score"
            ]
            - 0.01
        ),
        "selection_overall_mse_within_5_percent_of_raw": (
            candidate["normalized_mse_overall"]
            <= 1.05 * raw["normalized_mse_overall"]
        ),
        "selection_action_overlap_mse_within_5_percent_of_raw": (
            candidate["normalized_mse_action_overlap"]
            <= 1.05 * raw["normalized_mse_action_overlap"]
        ),
        "selection_supported_pair_rate_at_least_20_percent": (
            candidate["supported_pair_rate_action_overlap"] >= 0.20
        ),
        "selection_outputs_finite": all(
            row["finite"] for row in selection.values()
        ),
    }
    passed = all(gates.values())
    selected_safe_null = min(
        SINGLE_COMPONENT_CONTROLS,
        key=lambda name: selection[name]["pair_balanced_log_score"],
    )
    return {
        "schema_version": 2,
        "kind": (
            "multi_hypothesis_jepa_corrected_safe_null_assessment_v2"
        ),
        "source_v1_disposition": (
            "invalid: evaluation-role finiteness entered selection and "
            "the promised assessment was incomplete"
        ),
        "selection_metrics": selection,
        "diagnostic_transfer_metrics": role_metrics["transfer"],
        "transfer_influenced_selection": False,
        "gates": gates,
        "safe_null_passed": passed,
        "selected_safe_null": selected_safe_null,
        "decision": (
            "continue_to_calibration_and_full_value_assessment"
            if passed
            else "reject_recipe_at_safe_null_selection"
        ),
        "calibration_and_value_lanes_reached": passed,
        "bounded_interpretation": (
            (
                "The four-component recipe passed safe-null selection and "
                "requires calibration and full value-lane assessment before "
                "any promotion decision."
            )
            if passed
            else (
                "The four-component recipe cannot advance because it failed "
                "selection-role proper-score and point-prediction safety "
                "gates. Alert and investigation value were not assessed."
            )
        ),
    }


def _verify_source(source_directory: Path) -> Mapping[str, Any]:
    manifest_path = source_directory / "artifact-manifest.json"
    result_path = source_directory / "prototype-result.json"
    manifest_sha256 = _file_sha256(manifest_path)
    result_sha256 = _file_sha256(result_path)
    if (
        manifest_sha256 != EXPECTED_SOURCE_MANIFEST_SHA256
        or result_sha256 != EXPECTED_SOURCE_RESULT_SHA256
    ):
        raise ValueError("corrected assessment source identity differs")
    manifest = _read_object(manifest_path)
    recorded = manifest.get("sha256")
    if not isinstance(recorded, dict):
        raise ValueError("source artifact manifest is invalid")
    for relative, expected in recorded.items():
        path = source_directory / str(relative)
        if (
            not isinstance(expected, str)
            or _file_sha256(path) != expected
        ):
            raise ValueError("source artifact content identity mismatch")
    protocol = _read_object(source_directory / "protocol.json")
    data_identity = _read_object(source_directory / "data-identity.json")
    model_configs = protocol.get("model_configs")
    if (
        protocol.get("schema_version") != 1
        or protocol.get("kind")
        != "multi_hypothesis_jepa_prototype_v1"
        or protocol.get("seed") != 307
        or protocol.get("epochs") != 40
        or protocol.get("scoring_contract")
        != "docs/specs/multi-hypothesis-jepa-scoring-contract-v1.md"
        or not isinstance(model_configs, dict)
        or set(model_configs) != set(MODEL_NAMES)
        or any(
            config.get("seed") != 307
            or config.get("epochs") != 40
            for name, config in model_configs.items()
            if name != "raw_low_rank"
        )
        or model_configs["multi_hypothesis_jepa"].get(
            "component_count"
        )
        != 4
        or model_configs["multi_hypothesis_jepa"].get("objective")
        != "jepa"
        or model_configs["one_component_jepa"].get(
            "component_count"
        )
        != 1
        or model_configs["one_component_jepa"].get("objective")
        != "jepa"
        or model_configs["supervised_four_component_mixture"].get(
            "component_count"
        )
        != 4
        or model_configs["supervised_four_component_mixture"].get(
            "objective"
        )
        != "supervised"
        or model_configs["raw_low_rank"].get("rank") != 32
    ):
        raise ValueError("source prototype protocol identity differs")
    if (
        data_identity.get("schema_version") != 1
        or data_identity.get("source_corpus_sha256")
        != EXPECTED_CORPUS_SHA256
        or data_identity.get("source_artifact_manifest_sha256")
        != EXPECTED_PREPROCESSING_MANIFEST_SHA256
        or data_identity.get("preprocessing_cache_address")
        != EXPECTED_PREPROCESSING_CACHE_ADDRESS
        or data_identity.get("preprocessing_protocol")
        != "action_conditioned_jepa_topology_transfer_v1"
        or data_identity.get("window_counts")
        != {"fit": 6320, "selection": 1580, "transfer": 1580}
        or data_identity.get("pair_counts")
        != {"fit": 40, "selection": 10, "transfer": 10}
    ):
        raise ValueError("source prototype data identity differs")
    return {
        "schema_version": 1,
        "source_directory": source_directory.as_posix(),
        "source_manifest_sha256": manifest_sha256,
        "source_result_sha256": result_sha256,
        "source_corpus_sha256": data_identity[
            "source_corpus_sha256"
        ],
        "source_artifact_manifest_sha256": data_identity[
            "source_artifact_manifest_sha256"
        ],
        "source_preprocessing_cache_address": data_identity[
            "preprocessing_cache_address"
        ],
    }


def _read_role_inputs(
    source_directory: Path,
    role: str,
) -> Mapping[str, Any]:
    with np.load(
        source_directory / "predictions" / f"{role}-inputs.npz",
        allow_pickle=False,
    ) as arrays:
        return {
            "observed": np.asarray(
                arrays["observed"], dtype=np.float64
            ),
            "action_active": np.asarray(
                arrays["action_active"], dtype=np.bool_
            ),
            "trajectory_ids": tuple(
                str(value) for value in arrays["trajectory_ids"]
            ),
            "matched_pair_ids": tuple(
                str(value) for value in arrays["matched_pair_ids"]
            ),
        }


def _read_distribution(
    source_directory: Path,
    role: str,
    name: str,
) -> MixtureTrajectoryDistribution:
    with np.load(
        source_directory
        / "predictions"
        / f"{role}-{name}.npz",
        allow_pickle=False,
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


def _pair_balanced_mean(
    values: NDArray[np.float64],
    trajectory_ids: Tuple[str, ...],
    pair_ids: Tuple[str, ...],
) -> float:
    if not (
        len(values) == len(trajectory_ids) == len(pair_ids)
    ):
        raise ValueError("pair-balanced score inputs do not align")
    by_trajectory: Dict[str, list[float]] = {}
    trajectory_pair: Dict[str, str] = {}
    for value, trajectory_id, pair_id in zip(
        values, trajectory_ids, pair_ids
    ):
        existing = trajectory_pair.setdefault(trajectory_id, pair_id)
        if existing != pair_id:
            raise ValueError("trajectory crosses matched pairs")
        by_trajectory.setdefault(trajectory_id, []).append(float(value))
    by_pair: Dict[str, list[float]] = {}
    for trajectory_id, rows in by_trajectory.items():
        by_pair.setdefault(
            trajectory_pair[trajectory_id], []
        ).append(float(np.mean(rows)))
    if not by_pair or any(len(rows) != 2 for rows in by_pair.values()):
        raise ValueError("pair-balanced score requires complete pairs")
    return float(
        np.mean(
            [
                np.mean(by_pair[pair_id])
                for pair_id in sorted(by_pair)
            ]
        )
    )


def _supported_pair_rate(
    distribution: MixtureTrajectoryDistribution,
    sample_mask: NDArray[np.bool_],
) -> float:
    if distribution.component_mean.shape[1] < 2:
        return 0.0
    supported = np.zeros(len(distribution.weight), dtype=np.bool_)
    for left in range(distribution.component_mean.shape[1]):
        for right in range(left + 1, distribution.component_mean.shape[1]):
            variance = 0.5 * (
                distribution.component_variance[:, left]
                + distribution.component_variance[:, right]
            )
            distance = np.sqrt(
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
                & (distance >= 1.0)
            )
    return float(np.mean(supported[sample_mask]))


def _report_markdown(
    assessment: Mapping[str, Any],
    source_identity: Mapping[str, Any],
) -> str:
    selection = assessment["selection_metrics"]
    candidate = selection["multi_hypothesis_jepa"]
    raw = selection["raw_low_rank"]
    supervised = selection["supervised_four_component_mixture"]
    one = selection["one_component_jepa"]
    passed = bool(assessment["safe_null_passed"])
    lines = [
        "# Multi-hypothesis trajectory JEPA prototype v2 result",
        "",
        "## Outcome",
        "",
        (
            "**Continue to calibration and full value-lane assessment.** "
            "The corrected candidate passed safe-null selection."
            if passed
            else (
                "**Reject this recipe at safe-null selection.** The v1 "
                "decision artifact is invalid; this corrected assessment "
                "uses only selection data and explicitly stops before "
                "calibration and value lanes."
            )
        ),
        "",
        "## Selection evidence",
        "",
        (
            "- Candidate pair-balanced log score: "
            f"`{candidate['pair_balanced_log_score']:.6f}`."
        ),
        (
            "- One-component JEPA log score: "
            f"`{one['pair_balanced_log_score']:.6f}`."
        ),
        (
            "- Supervised four-component log score: "
            f"`{supervised['pair_balanced_log_score']:.6f}`."
        ),
        (
            "- Candidate overall MSE versus raw: "
            f"`{candidate['normalized_mse_overall']:.6f}` versus "
            f"`{raw['normalized_mse_overall']:.6f}`."
        ),
        (
            "- Candidate action-overlap MSE versus raw: "
            f"`{candidate['normalized_mse_action_overlap']:.6f}` versus "
            f"`{raw['normalized_mse_action_overlap']:.6f}`."
        ),
        (
            "- Supported-pair rate: "
            f"`{candidate['supported_pair_rate_action_overlap']:.2%}`."
        ),
        "",
        "## Gates",
        "",
    ]
    for name, passed in assessment["gates"].items():
        lines.append(f"- `{name}`: {'PASS' if passed else 'FAIL'}")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            (
                (
                    "The candidate remains eligible for calibration and "
                    "full value-lane assessment."
                )
                if passed
                else (
                    "Selected safe null: "
                    f"`{assessment['selected_safe_null']}`. Alert and "
                    "investigation value were not assessed because the "
                    "candidate failed the prerequisite selection gates."
                )
            ),
            "",
            (
                "Source v1 manifest SHA-256: "
                f"`{source_identity['source_manifest_sha256']}`."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _read_object(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _pretty_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main(arguments: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(
            "artifacts/action-dynamics/"
            "prototype-multi-hypothesis-jepa-v1"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/action-dynamics/"
            "prototype-multi-hypothesis-jepa-v2"
        ),
    )
    parsed = parser.parse_args(arguments)
    assessment = correct_prototype_assessment(
        parsed.source, parsed.output
    )
    print(json.dumps(assessment, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
