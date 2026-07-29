#!/usr/bin/env python3
"""Recompute ticket 012 from stored arrays without fitted models."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from quantis_core.edge_dynamics.hepa_jepa import (
    HEPA_ASSESSMENT_ROLE_NAMES,
    HEPA_MODEL_NAMES,
    assess_hepa_tracer,
)

MODEL_NAMES = HEPA_MODEL_NAMES
ROLE_NAMES = HEPA_ASSESSMENT_ROLE_NAMES


def assess_stored_bundle(
    directory: Path, *, verify_manifest: bool = True
) -> Mapping[str, Any]:
    """Verify a completed bundle and recompute its complete assessment."""

    root = Path(directory)
    if verify_manifest:
        _verify_manifest(root)
    metadata = _read_object(root / "assessment-metadata.json")
    model_bundle = _read_object(root / "models.json")
    event_definition = _read_object(root / "event-definition.json")
    if (
        tuple(metadata["model_names"]) != MODEL_NAMES
        or tuple(metadata["role_names"]) != ROLE_NAMES
    ):
        raise ValueError("HEPA assessment names differ from contract")
    with np.load(
        root / "hepa-evidence.npz", allow_pickle=False
    ) as arrays:
        probability_surfaces = _read_surfaces(
            arrays, prefix="probability"
        )
        restored_probability_surfaces = _read_surfaces(
            arrays, prefix="restored_probability"
        )
        calibrated_surfaces = _read_surfaces(
            arrays, prefix="calibrated"
        )
        restored_calibrated_surfaces = _read_surfaces(
            arrays, prefix="restored_calibrated"
        )
        alert_decisions = _read_decisions(
            arrays, prefix="alert_decision"
        )
        restored_alert_decisions = _read_decisions(
            arrays, prefix="restored_alert_decision"
        )
        labels = {
            role: arrays[f"labels__{role}"].astype(np.bool_)
            for role in ROLE_NAMES
        }
        raw_effect_scores = {
            role: arrays[f"raw_effect_scores__{role}"]
            for role in ROLE_NAMES
        }
        transition_indices = {
            role: arrays[f"transition_indices__{role}"].astype(
                np.int64
            )
            for role in ROLE_NAMES
        }
        candidate_tokens = arrays["candidate_tokens"]
        restored_candidate_tokens = arrays[
            "restored_candidate_tokens"
        ]
        state_truth = arrays["state_truth"]
        state_scale = arrays["state_scale"]
        state_varying_mask = arrays[
            "state_varying_mask"
        ].astype(np.bool_)
        state_predictions = {
            name: arrays[f"state_prediction__{name}"]
            for name in ("hepa", "matched_pca")
        }
        latency_samples = {
            name: arrays[f"latency_samples__{name}"].copy()
            for name in MODEL_NAMES
        }
        causality_evidence = {
            name: arrays[name].copy()
            for name in (
                "audit_histories",
                "audit_counterfactual_histories",
                "audit_forbidden",
                "audit_counterfactual_forbidden",
                "audit_original_outputs",
                "audit_counterfactual_outputs",
            )
        }
    trajectory_ids = {
        role: tuple(
            str(value)
            for value in metadata["trajectory_ids"][role]
        )
        for role in ROLE_NAMES
    }
    trajectory_onsets = {
        role: {
            str(key): (
                None if value is None else int(value)
            )
            for key, value in dict(
                metadata["trajectory_onsets"][role]
            ).items()
        }
        for role in ROLE_NAMES
    }
    models = {
        str(name): dict(payload)
        for name, payload in dict(model_bundle["models"]).items()
    }
    restored_models = {
        str(name): dict(payload)
        for name, payload in dict(
            model_bundle["restored_models"]
        ).items()
    }
    inference_parameter_counts = {
        name: _inference_parameter_count(models[name])
        for name in MODEL_NAMES
    }
    protocol_checks = _derive_protocol_checks(
        metadata=metadata,
        models=models,
        restored_models=restored_models,
        event_definition=event_definition,
        inference_parameter_counts=inference_parameter_counts,
        causality_evidence=causality_evidence,
    )
    edge_metrics = _derive_edge_metrics(
        metadata=metadata,
        model_bundle=model_bundle,
        event_definition=event_definition,
        inference_parameter_counts=inference_parameter_counts,
        latency_samples=latency_samples,
    )
    return assess_hepa_tracer(
        probability_surfaces=probability_surfaces,
        restored_probability_surfaces=(
            restored_probability_surfaces
        ),
        stored_calibrated_surfaces=calibrated_surfaces,
        restored_calibrated_surfaces=(
            restored_calibrated_surfaces
        ),
        stored_alert_decisions=alert_decisions,
        restored_alert_decisions=restored_alert_decisions,
        stored_model_calibrations=_read_model_calibrations(models),
        restored_model_calibrations=_read_model_calibrations(
            restored_models
        ),
        labels=labels,
        trajectory_ids=trajectory_ids,
        transition_indices=transition_indices,
        trajectory_onsets=trajectory_onsets,
        candidate_tokens=candidate_tokens,
        restored_candidate_tokens=restored_candidate_tokens,
        state_truth=state_truth,
        state_scale=state_scale,
        state_varying_mask=state_varying_mask,
        state_predictions=state_predictions,
        inference_parameter_counts=inference_parameter_counts,
        protocol_checks=protocol_checks,
        edge_metrics=edge_metrics,
        raw_effect_scores=raw_effect_scores,
        event_threshold=float(event_definition["threshold"]),
    )


def verify_stored_assessment(directory: Path) -> Mapping[str, Any]:
    """Require byte-identical canonical stored and recomputed assessment."""

    root = Path(directory)
    recomputed = assess_stored_bundle(root)
    expected = _pretty_json(recomputed)
    actual = (root / "assessment.json").read_text()
    if actual != expected:
        raise ValueError("stored HEPA assessment does not recompute")
    return recomputed


def _read_surfaces(
    arrays: Any, *, prefix: str
) -> Mapping[str, Mapping[str, np.ndarray]]:
    return {
        role: {
            model: arrays[f"{prefix}__{role}__{model}"]
            for model in MODEL_NAMES
        }
        for role in ROLE_NAMES
    }


def _read_decisions(
    arrays: Any, *, prefix: str
) -> Mapping[str, Mapping[str, np.ndarray]]:
    return {
        role: {
            model: arrays[
                f"{prefix}__{role}__{model}"
            ].astype(np.bool_)
            for model in MODEL_NAMES
        }
        for role in ROLE_NAMES
    }


def _read_model_calibrations(
    models: Mapping[str, Mapping[str, Any]]
) -> Mapping[str, Mapping[str, float]]:
    result = {}
    expected = {
        "slope",
        "intercept",
        "calibration_brier",
        "alert_threshold",
    }
    for name in MODEL_NAMES:
        raw = models[name].get("calibration")
        if not isinstance(raw, dict) or set(raw) != expected:
            raise ValueError("HEPA serialized calibration is invalid")
        result[name] = {
            str(key): float(value) for key, value in raw.items()
        }
    return result


def _inference_parameter_count(model: Mapping[str, Any]) -> int:
    state_dict = dict(model["state_dict"])
    count = 0
    for name, raw_value in state_dict.items():
        value = dict(raw_value)
        shape = tuple(int(item) for item in value["shape"])
        if np.asarray(value["values"]).shape != shape:
            raise ValueError("HEPA serialized state tensor shape differs")
        if str(name) != "encoder.ownership":
            count += int(np.prod(shape, dtype=np.int64))
    return count


def _derive_protocol_checks(
    *,
    metadata: Mapping[str, Any],
    models: Mapping[str, Mapping[str, Any]],
    restored_models: Mapping[str, Mapping[str, Any]],
    event_definition: Mapping[str, Any],
    inference_parameter_counts: Mapping[str, int],
    causality_evidence: Mapping[str, np.ndarray],
) -> Mapping[str, bool]:
    source = {
        str(role): set(str(value) for value in values)
        for role, values in dict(
            metadata["source_role_pair_ids"]
        ).items()
    }
    used = {
        str(role): set(str(value) for value in values)
        for role, values in dict(metadata["used_pair_ids"]).items()
    }
    source_roles = ("fit", "selection", "calibration", "evaluation")
    source_names_valid = set(source) == set(source_roles)
    used_names_valid = set(used) == {
        "fit",
        "selection",
        "calibration",
        "evaluation_iid",
        "evaluation_transfer",
    }
    disjoint = source_names_valid and all(
        not (source[left] & source[right])
        for position, left in enumerate(source_roles)
        for right in source_roles[position + 1 :]
    )
    used_roles_match = (
        used_names_valid
        and used["fit"] <= source["fit"]
        and used["selection"] <= source["selection"]
        and used["calibration"] <= source["calibration"]
        and used["evaluation_iid"] <= source["evaluation"]
        and used["evaluation_transfer"] <= source["evaluation"]
        and not (
            used["evaluation_iid"]
            & used["evaluation_transfer"]
        )
    )
    candidate_config = dict(models["hepa"]["config"])
    null_config = dict(models["horizon_deranged"]["config"])
    candidate_objective = candidate_config.pop("objective", None)
    null_objective = null_config.pop("objective", None)
    fit_pairs = used.get("fit", set())
    derangement = {
        str(key): str(value)
        for key, value in dict(
            models["horizon_deranged"]["derangement"]
        ).items()
    }
    pair_atomic = (
        set(derangement) == fit_pairs
        and set(derangement.values()) == fit_pairs
        and all(key != value for key, value in derangement.items())
    )
    phases = {
        str(key): float(value)
        for key, value in dict(metadata["phases"]).items()
    }
    phase_names = (
        "fitting_started_unix_seconds",
        "calibration_completed_unix_seconds",
        "evaluation_started_unix_seconds",
        "evaluation_completed_unix_seconds",
    )
    phases_are_ordered = (
        set(phases) == set(phase_names)
        and all(np.isfinite(phases[name]) for name in phase_names)
        and all(
            phases[left] <= phases[right]
            for left, right in zip(phase_names, phase_names[1:])
        )
    )
    expected_inputs = ["histories", "declared_graph"]
    expected_forbidden = [
        "future_states",
        "future_controls",
        "future_actions",
        "action_kind",
        "target_entity",
        "trajectory_id",
        "matched_pair_id",
    ]
    causal = (
        np.array_equal(
            causality_evidence["audit_histories"],
            causality_evidence[
                "audit_counterfactual_histories"
            ],
        )
        and not np.array_equal(
            causality_evidence["audit_forbidden"],
            causality_evidence[
                "audit_counterfactual_forbidden"
            ],
        )
        and np.allclose(
            causality_evidence["audit_original_outputs"],
            causality_evidence[
                "audit_counterfactual_outputs"
            ],
            atol=0.0,
            rtol=0.0,
        )
    )
    restored_match = all(
        models[name] == restored_models[name] for name in MODEL_NAMES
    )
    return {
        "role_pairs_are_disjoint": disjoint,
        "fit_uses_40_in_distribution_pairs": (
            used_roles_match and len(used["fit"]) == 40
        ),
        "selection_uses_10_in_distribution_pairs": (
            used_roles_match and len(used["selection"]) == 10
        ),
        "event_definition_fit_on_40_controls": (
            int(event_definition["control_trajectory_count"]) == 40
        ),
        "calibration_uses_10_in_distribution_pairs": (
            used_roles_match and len(used["calibration"]) == 10
        ),
        "evaluation_uses_20_iid_and_10_transfer_pairs": (
            used_roles_match
            and len(used["evaluation_iid"]) == 20
            and len(used["evaluation_transfer"]) == 10
        ),
        "model_accepts_only_histories_and_graph": (
            metadata["model_inputs"] == expected_inputs
            and metadata["forbidden_model_inputs"]
            == expected_forbidden
            and causal
        ),
        "selection_and_calibration_precede_evaluation": (
            phases_are_ordered
        ),
        "evaluation_not_used_for_fitting": (
            used_roles_match
            and not (
                used["fit"]
                & (
                    used["evaluation_iid"]
                    | used["evaluation_transfer"]
                )
            )
        ),
        "only_target_alignment_differs": (
            candidate_objective == "hepa"
            and null_objective == "horizon_deranged"
            and candidate_config == null_config
            and inference_parameter_counts["hepa"]
            == inference_parameter_counts["horizon_deranged"]
            and models["hepa"]["stage1_target_alignment"] == "aligned"
            and models["horizon_deranged"][
                "stage1_target_alignment"
            ]
            == "whole_pair_deranged"
        ),
        "pair_atomic_derangement": pair_atomic,
        "restored_model_payloads_match": restored_match,
    }


def _derive_edge_metrics(
    *,
    metadata: Mapping[str, Any],
    model_bundle: Mapping[str, Any],
    event_definition: Mapping[str, Any],
    inference_parameter_counts: Mapping[str, int],
    latency_samples: Mapping[str, np.ndarray],
) -> Mapping[str, Mapping[str, float]]:
    peak_rss_bytes = float(metadata["peak_rss_bytes"])
    models = dict(model_bundle["models"])
    probes = dict(model_bundle["state_probes"])
    result = {}
    for name in MODEL_NAMES:
        samples = np.asarray(
            latency_samples[name], dtype=np.float64
        )
        if samples.ndim != 1 or len(samples) < 1:
            raise ValueError("HEPA latency evidence is invalid")
        serialized = (
            {
                "model": models[name],
                "event_definition": event_definition,
                "state_probe": probes["hepa"],
            }
            if name == "hepa"
            else {"model": models[name]}
        )
        result[name] = {
            "inference_parameter_count": float(
                inference_parameter_counts[name]
            ),
            "serialized_candidate_sidecars_bytes": float(
                len(_canonical_json_bytes(serialized))
            ),
            "batch_one_cpu_latency_ms": float(np.mean(samples)),
            "peak_rss_bytes": peak_rss_bytes,
            "latency_repetitions": float(len(samples)),
        }
    return result


def _verify_manifest(root: Path) -> None:
    manifest = _read_object(root / "artifact-manifest.json")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "hepa_jepa_artifact_manifest"
    ):
        raise ValueError("unsupported HEPA artifact manifest")
    recorded = dict(manifest["files"])
    expected_files = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and path.name != "artifact-manifest.json"
    }
    if set(recorded) != expected_files:
        raise ValueError("HEPA artifact manifest file set mismatch")
    for relative, identity in recorded.items():
        path = root / relative
        if (
            int(identity["bytes"]) != path.stat().st_size
            or str(identity["sha256"]) != _file_sha256(path)
        ):
            raise ValueError("HEPA artifact content identity mismatch")


def _read_object(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pretty_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    arguments = parser.parse_args()
    assessment = verify_stored_assessment(arguments.directory)
    print(_pretty_json(assessment), end="")


if __name__ == "__main__":
    main()
