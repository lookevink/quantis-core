"""Independent stored-array assessor for the richer-regime JEPA retry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from audit_richer_regime_validity import verify_validity_audit


_VALID_MODELS = (
    "multi_hypothesis_jepa",
    "one_component_jepa",
    "capacity_matched_single_gaussian",
    "raw_low_rank",
)
_REPORTED_ARTIFACT_MANIFEST_SHA256 = (
    "084a5d45d85f310358bacafbf5fba1736aa0649815ff1f21d133ec88f17f42b3"
)
_REPORTED_RECIPE_PROTOCOL_SHA256 = (
    "ee6a84fb79bb4ba87b07062218740815d2b593734a3d7c15a7bac020932e220c"
)
_REPORTED_VALIDITY_AUDIT_MANIFEST_SHA256 = (
    "ce8cef807ba1feb557465b2868e8f9a5e955994211800f192fe424c021671c80"
)


def verify_stored_retry(
    directory: Path,
    *,
    expected_manifest_sha256: str = (
        _REPORTED_ARTIFACT_MANIFEST_SHA256
    ),
    expected_protocol_sha256: str = _REPORTED_RECIPE_PROTOCOL_SHA256,
    validity_audit: Path = Path(
        "artifacts/action-dynamics/"
        "richer-regime-retry-v1-validity-audit-v4"
    ),
    expected_validity_audit_manifest_sha256: str = (
        _REPORTED_VALIDITY_AUDIT_MANIFEST_SHA256
    ),
) -> Mapping[str, Any]:
    """Verify hashes and recompute rejection from prediction sidecars."""

    root = Path(directory)
    if (
        _file_sha256(root / "artifact-manifest.json")
        != expected_manifest_sha256
        or _file_sha256(root / "protocol.json")
        != expected_protocol_sha256
    ):
        raise ValueError("retry artifact or recipe identity differs")
    manifest = _read_object(root / "artifact-manifest.json")
    raw_hashes = manifest.get("sha256")
    if (
        manifest.get("kind")
        != "richer_regime_multi_hypothesis_artifact_manifest"
        or not isinstance(raw_hashes, dict)
    ):
        raise ValueError("retry artifact manifest is invalid")
    expected_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "artifact-manifest.json"
    }
    if set(raw_hashes) != expected_paths or any(
        raw_hashes[name] != _file_sha256(root / name)
        for name in expected_paths
    ):
        raise ValueError("retry artifact hashes differ")
    diagnosis = _read_object(root / "failure-diagnosis.json")
    validity = verify_validity_audit(
        validity_audit,
        expected_manifest_sha256=(
            expected_validity_audit_manifest_sha256
        ),
    )
    with np.load(
        root / "predictions" / "selection-inputs.npz",
        allow_pickle=False,
    ) as arrays:
        observed = np.asarray(arrays["observed"], dtype=np.float64)
        action_active = np.asarray(
            arrays["action_active"], dtype=np.bool_
        )
        trajectory_ids = tuple(
            str(value) for value in arrays["trajectory_ids"]
        )
    if (
        observed.ndim != 4
        or action_active.shape != observed.shape[:2]
        or len(trajectory_ids) != len(observed)
    ):
        raise ValueError("stored selection inputs are invalid")
    metrics = {
        name: _model_metrics(
            root,
            name,
            observed,
            action_active,
            trajectory_ids,
        )
        for name in _VALID_MODELS
    }
    candidate = metrics["multi_hypothesis_jepa"]
    one = metrics["one_component_jepa"]
    raw = metrics["raw_low_rank"]
    gates = {
        "candidate_beats_one_component_log_score_by_0_01": (
            candidate["trajectory_balanced_log_score"]
            <= one["trajectory_balanced_log_score"] - 0.01
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
    decision = (
        "reject_multi_hypothesis_retry_independent_of_invalid_null"
        if not all(gates.values())
        else "inconclusive_due_to_invalid_supervised_null"
    )
    if (
        diagnosis.get("decision") != decision
        or diagnosis.get("independent_candidate_gates") != gates
        or not _metrics_close(
            diagnosis.get("valid_model_selection_metrics"), metrics
        )
    ):
        raise ValueError(
            "stored retry diagnosis differs from independent recomputation"
        )
    return {
        "schema_version": 1,
        "kind": (
            "richer_regime_multi_hypothesis_independent_assessment"
        ),
        "verified": True,
        "artifact_manifest_sha256": expected_manifest_sha256,
        "recipe_protocol_sha256": expected_protocol_sha256,
        "stored_numeric_decision": decision,
        "scientific_status": validity["scientific_status"],
        "stored_model_decision_admissible": validity[
            "stored_model_decision_admissible"
        ],
        "validity_audit_manifest_sha256": (
            expected_validity_audit_manifest_sha256
        ),
        "gates": gates,
        "metrics": metrics,
    }


def _model_metrics(
    root: Path,
    name: str,
    observed: NDArray[np.float64],
    action_active: NDArray[np.bool_],
    trajectory_ids: Tuple[str, ...],
) -> Mapping[str, Any]:
    with np.load(
        root / "predictions" / f"selection-{name}.npz",
        allow_pickle=False,
    ) as arrays:
        component_mean = np.asarray(
            arrays["component_mean"], dtype=np.float64
        )
        component_variance = np.asarray(
            arrays["component_variance"], dtype=np.float64
        )
        weight = np.asarray(arrays["weight"], dtype=np.float64)
    if (
        component_mean.ndim != 5
        or component_variance.shape != component_mean.shape
        or weight.shape != component_mean.shape[:2]
        or observed.shape != component_mean.shape[:1] + component_mean.shape[2:]
        or np.any(component_variance <= 0.0)
        or np.any(weight <= 0.0)
        or not np.allclose(
            np.sum(weight, axis=1),
            1.0,
            rtol=0.0,
            atol=1e-6,
        )
    ):
        raise ValueError(f"stored distribution is invalid: {name}")
    weight = weight / np.sum(weight, axis=1, keepdims=True)
    expanded_observed = observed[:, None]
    component_terms = -0.5 * (
        np.square(expanded_observed - component_mean)
        / component_variance
        + np.log(component_variance)
        + np.log(2.0 * np.pi)
    )
    component_log_density = np.sum(
        component_terms, axis=(2, 3, 4)
    )
    coordinate_count = int(np.prod(observed.shape[1:]))
    negative_log_likelihood = -_logsumexp(
        np.log(weight) + component_log_density, axis=1
    ) / coordinate_count
    expanded_weight = weight[:, :, None, None, None]
    mean = np.sum(expanded_weight * component_mean, axis=1)
    second_moment = np.sum(
        expanded_weight
        * (component_variance + np.square(component_mean)),
        axis=1,
    )
    variance = np.maximum(
        second_moment - np.square(mean),
        np.finfo(np.float64).tiny,
    )
    squared = np.square(mean - observed)
    finite = bool(
        np.all(np.isfinite(negative_log_likelihood))
        and np.all(np.isfinite(mean))
        and np.all(np.isfinite(variance))
    )
    return {
        "trajectory_balanced_log_score": _balanced_mean(
            negative_log_likelihood, trajectory_ids
        ),
        "normalized_mse_overall": float(np.mean(squared)),
        "normalized_mse_action_overlap": float(
            np.mean(squared[action_active])
        ),
        "supported_pair_rate_action_overlap": _supported_rate(
            component_mean,
            component_variance,
            weight,
            np.any(action_active, axis=1),
        ),
        "finite": finite,
    }


def _supported_rate(
    mean: NDArray[np.float64],
    variance: NDArray[np.float64],
    weight: NDArray[np.float64],
    sample_mask: NDArray[np.bool_],
) -> float:
    if mean.shape[1] < 2:
        return 0.0
    supported = np.zeros(len(mean), dtype=np.bool_)
    for left in range(mean.shape[1]):
        for right in range(left + 1, mean.shape[1]):
            pooled = 0.5 * (
                variance[:, left] + variance[:, right]
            )
            distance = np.sqrt(
                np.mean(
                    np.square(mean[:, left] - mean[:, right]) / pooled,
                    axis=(1, 2, 3),
                )
            )
            supported |= (
                (weight[:, left] >= 0.10)
                & (weight[:, right] >= 0.10)
                & (distance >= 1.0)
            )
    return float(np.mean(supported[sample_mask]))


def _balanced_mean(
    values: NDArray[np.float64],
    trajectory_ids: Tuple[str, ...],
) -> float:
    grouped: Dict[str, list[float]] = {}
    for value, trajectory_id in zip(values, trajectory_ids):
        grouped.setdefault(trajectory_id, []).append(float(value))
    return float(
        np.mean(
            [
                np.mean(grouped[trajectory_id])
                for trajectory_id in sorted(grouped)
            ]
        )
    )


def _logsumexp(
    values: NDArray[np.float64], *, axis: int
) -> NDArray[np.float64]:
    maximum = np.max(values, axis=axis, keepdims=True)
    return np.asarray(
        np.squeeze(
            maximum
            + np.log(
                np.sum(
                    np.exp(values - maximum),
                    axis=axis,
                    keepdims=True,
                )
            ),
            axis=axis,
        ),
        dtype=np.float64,
    )


def _metrics_close(raw: Any, expected: Mapping[str, Any]) -> bool:
    if not isinstance(raw, dict) or set(raw) != set(expected):
        return False
    for name, expected_row in expected.items():
        raw_row = raw.get(name)
        if not isinstance(raw_row, dict) or set(raw_row) != set(
            expected_row
        ):
            return False
        for key, expected_value in expected_row.items():
            raw_value = raw_row.get(key)
            if isinstance(expected_value, bool):
                if raw_value is not expected_value:
                    return False
            elif not isinstance(raw_value, (int, float)) or not np.isclose(
                float(raw_value),
                float(expected_value),
                atol=1e-12,
                rtol=1e-12,
            ):
                return False
    return True


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(arguments: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path(
            "artifacts/action-dynamics/"
            "richer-regime-multi-hypothesis-jepa-v1"
        ),
    )
    parser.add_argument(
        "--validity-audit",
        type=Path,
        default=Path(
            "artifacts/action-dynamics/"
            "richer-regime-retry-v1-validity-audit-v4"
        ),
    )
    parser.add_argument(
        "--expected-manifest-sha256",
        default=_REPORTED_ARTIFACT_MANIFEST_SHA256,
    )
    parser.add_argument(
        "--expected-protocol-sha256",
        default=_REPORTED_RECIPE_PROTOCOL_SHA256,
    )
    parser.add_argument(
        "--expected-validity-audit-manifest-sha256",
        default=_REPORTED_VALIDITY_AUDIT_MANIFEST_SHA256,
    )
    parsed = parser.parse_args(arguments)
    assessment = verify_stored_retry(
        parsed.artifact,
        expected_manifest_sha256=parsed.expected_manifest_sha256,
        expected_protocol_sha256=parsed.expected_protocol_sha256,
        validity_audit=parsed.validity_audit,
        expected_validity_audit_manifest_sha256=(
            parsed.expected_validity_audit_manifest_sha256
        ),
    )
    print(json.dumps(assessment, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
