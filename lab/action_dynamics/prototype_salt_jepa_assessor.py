#!/usr/bin/env python3
"""Independent stored-evidence assessor for the SALT-JEPA tracer."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from lab.action_dynamics.prototype_complete_lejepa import (
    _action_sanity_from_predictions,
    _attribution_scores_from_predictions,
    _downstream_pair_errors,
    _forecast_scores,
    _state_probe,
)
from quantis_core.action_conditioned_dynamics import (
    ActionConditionedWindows,
)
from quantis_core.edge_dynamics.data import PreparedAttributionQueries
from quantis_core.edge_dynamics.salt_jepa import assess_salt_jepa_gates
from quantis_core.graph_telemetry import DeclaredTelemetryGraph


REPRESENTATION_NAMES = (
    "salt_jepa",
    "deranged_salt_jepa",
    "reconstructive_teacher",
    "matched_pca",
)
SALT_NAMES = ("salt_jepa", "deranged_salt_jepa")
EVALUATED_ROLES = (
    "selection",
    "iid_evaluation",
    "transfer_evaluation",
)
ALL_ROLES = (
    "fit",
    "selection",
    "calibration",
    "iid_evaluation",
    "transfer_evaluation",
)


def assess_stored_bundle(directory: Path) -> Mapping[str, Any]:
    """Recompute every frozen gate without loading a fitted model."""

    root = Path(directory)
    metadata = _read_json(root / "evidence-metadata.json")
    if (
        metadata.get("schema_version") != 1
        or metadata.get("kind") != "salt_jepa_assessment_evidence"
    ):
        raise ValueError("unsupported SALT evidence")
    with np.load(root / "evidence.npz", allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    finite = all(
        np.all(np.isfinite(value))
        for value in arrays.values()
        if value.dtype.kind in ("f", "i", "u")
    )
    graph = DeclaredTelemetryGraph.from_dict(dict(metadata["graph"]))
    ownership = np.asarray(metadata["ownership_mask"], dtype=np.bool_)
    windows = {
        role: _windows_from_evidence(
            role, metadata, arrays, graph
        )
        for role in ("fit",) + EVALUATED_ROLES
    }
    forecast_scores = {
        name: {
            role: _forecast_scores(
                arrays[f"prediction__{name}__{role}"],
                windows[role],
            )
            for role in EVALUATED_ROLES
        }
        for name in REPRESENTATION_NAMES
    }
    raw_scores = {
        role: _forecast_scores(
            arrays[f"raw_prediction__{role}"], windows[role]
        )
        for role in EVALUATED_ROLES
    }
    state_probes = {
        name: {
            role: _state_probe(
                arrays[f"representation__{name}__fit"],
                windows["fit"],
                arrays[f"representation__{name}__{role}"],
                windows[role],
                ownership,
            )
            for role in EVALUATED_ROLES
        }
        for name in REPRESENTATION_NAMES
    }
    masked_latent_l1 = {
        name: {
            role: _masked_l1(
                arrays[f"diagnostic_predicted__{name}__{role}"],
                arrays[f"diagnostic_target__{name}__{role}"],
                arrays[f"diagnostic_mask__{name}__{role}"],
            )
            for role in ("selection", "transfer_evaluation")
        }
        for name in SALT_NAMES
    }
    query_metadata = dict(metadata["queries"])
    queries = PreparedAttributionQueries(
        query_ids=tuple(
            str(value) for value in query_metadata["query_ids"]
        ),
        histories=arrays["query_histories"],
        future_controls=arrays["query_future_controls"],
        observed_future=arrays["query_observed_future"],
        candidate_actions=arrays["query_candidate_actions"],
        candidate_ids=tuple(
            str(value) for value in query_metadata["candidate_ids"]
        ),
        candidate_action_kinds=tuple(
            str(value)
            for value in query_metadata["candidate_action_kinds"]
        ),
        candidate_target_entities=tuple(
            str(value)
            for value in query_metadata["candidate_target_entities"]
        ),
        expected_action_kinds=tuple(
            str(value)
            for value in query_metadata["expected_action_kinds"]
        ),
        expected_target_entities=tuple(
            str(value)
            for value in query_metadata["expected_target_entities"]
        ),
        expected_variant_ids=tuple(
            str(value)
            for value in query_metadata["expected_variant_ids"]
        ),
    )
    attribution = {
        name: _attribution_scores_from_predictions(
            arrays[f"attribution_prediction__{name}"],
            queries,
            ownership,
        )
        for name in REPRESENTATION_NAMES
    }
    action_sanity = {
        name: _action_sanity_from_predictions(
            {
                variant: arrays[
                    f"action_sanity__{name}__{variant}"
                ]
                for variant in ("correct", "no_action", "shuffled")
            },
            windows["transfer_evaluation"],
            ownership,
        )
        for name in REPRESENTATION_NAMES
    }
    restoration_max_abs = {
        name: float(
            np.max(
                np.abs(
                    arrays[f"restoration_original__{name}"]
                    - arrays[f"restoration_restored__{name}"]
                )
            )
        )
        for name in REPRESENTATION_NAMES
    }
    for name in REPRESENTATION_NAMES:
        restoration_max_abs[name] = max(
            restoration_max_abs[name],
            float(
                np.max(
                    np.abs(
                        arrays[
                            f"restoration_probe_original__{name}"
                        ]
                        - arrays[
                            f"restoration_probe_restored__{name}"
                        ]
                    )
                )
            ),
        )
    for name in SALT_NAMES:
        restoration_max_abs[name] = max(
            restoration_max_abs[name],
            float(
                np.max(
                    np.abs(
                        arrays[
                            f"restoration_diagnostic_original__{name}"
                        ]
                        - arrays[
                            f"restoration_diagnostic_restored__{name}"
                        ]
                    )
                )
            ),
        )
    protocol_checks = {
        "evidence_arrays_are_finite": finite,
        "pair_and_trajectory_roles_are_disjoint": (
            _role_identifiers_are_disjoint(metadata)
        ),
        "public_inference_is_causal": bool(
            metadata["public_inference_is_causal"]
        ),
        "mask_schedule_is_valid": _mask_schedule_is_valid(
            root, ownership
        ),
        "selection_only_ridge_choice_recomputes": (
            _selection_recomputes(metadata, arrays, windows, raw_scores)
        ),
    }
    assessment = dict(
        assess_salt_jepa_gates(
            forecast_scores=forecast_scores,
            raw_scores=raw_scores,
            masked_latent_l1=masked_latent_l1,
            state_probes=state_probes,
            attribution=attribution,
            action_sanity=action_sanity,
            restoration_max_abs=restoration_max_abs,
            protocol_checks=protocol_checks,
            parameter_counts={
                str(key): {
                    str(inner): int(value)
                    for inner, value in dict(raw).items()
                }
                for key, raw in dict(
                    metadata["parameter_counts"]
                ).items()
            },
            teacher_unchanged={
                str(key): bool(value)
                for key, value in dict(
                    metadata["teacher_unchanged"]
                ).items()
            },
            transfer_pair_errors={
                name: _downstream_pair_errors(
                    arrays[
                        f"prediction__{name}__transfer_evaluation"
                    ],
                    windows["transfer_evaluation"],
                )
                for name in REPRESENTATION_NAMES
            },
            deployed_bundle_bytes=int(
                metadata["deployed_bundle_bytes"]
            ),
            median_latency_ms=float(
                dict(metadata["latency"])["median_ms"]
            ),
        )
    )
    assessment["protocol_checks"] = protocol_checks
    assessment["masked_latent_l1"] = masked_latent_l1
    assessment["restoration_max_abs"] = restoration_max_abs
    assessment["forecast_scores"] = forecast_scores
    assessment["raw_scores"] = raw_scores
    assessment["state_probes"] = state_probes
    assessment["attribution"] = attribution
    assessment["action_sanity"] = action_sanity
    interpretable = bool(metadata["interpretable"])
    assessment["eligible_for_advance"] = interpretable
    if not interpretable:
        assessment["provisional_decision"] = assessment["decision"]
        assessment["decision"] = "non_interpretable_salt_jepa_smoke"
        assessment["passed"] = False
    return assessment


def verify_stored_assessment(directory: Path) -> Mapping[str, Any]:
    """Verify the manifest and exact stored independent assessment."""

    root = Path(directory)
    manifest = _read_json(root / "artifact-manifest.json")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "salt_jepa_artifact_manifest"
    ):
        raise ValueError("unsupported SALT manifest")
    for filename, expected in dict(manifest["sha256"]).items():
        actual = hashlib.sha256((root / filename).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"SALT artifact identity mismatch: {filename}")
    recorded = _read_json(root / "assessment.json")
    reassessed = assess_stored_bundle(root)
    if _canonical_json(recorded) != _canonical_json(reassessed):
        raise ValueError("stored SALT assessment differs from recomputation")
    return reassessed


def _windows_from_evidence(
    role: str,
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    graph: DeclaredTelemetryGraph,
) -> ActionConditionedWindows:
    identity = dict(dict(metadata["roles"])[role])
    return ActionConditionedWindows(
        histories=arrays[f"histories__{role}"],
        future_states=arrays[f"target__{role}"],
        future_controls=arrays[f"controls__{role}"],
        future_actions=arrays[f"actions__{role}"],
        trajectory_ids=tuple(
            str(value) for value in identity["trajectory_ids"]
        ),
        matched_pair_ids=tuple(
            str(value) for value in identity["pair_ids"]
        ),
        transition_indices=np.asarray(
            identity["transition_indices"], dtype=np.int64
        ),
        entity_names=tuple(
            str(value) for value in metadata["entity_names"]
        ),
        state_feature_names=tuple(
            str(value) for value in metadata["state_feature_names"]
        ),
        control_feature_names=tuple(
            str(value) for value in metadata["control_feature_names"]
        ),
        action_feature_names=tuple(
            str(value) for value in metadata["action_feature_names"]
        ),
        graph=graph,
    )


def _masked_l1(
    predicted: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> float:
    target_mask = np.asarray(mask, dtype=np.bool_)
    if (
        predicted.shape != target.shape
        or target_mask.shape != predicted.shape[:-1]
        or not np.any(target_mask)
    ):
        raise ValueError("SALT diagnostic evidence does not align")
    return float(np.mean(np.abs(predicted[target_mask] - target[target_mask])))


def _role_identifiers_are_disjoint(metadata: Mapping[str, Any]) -> bool:
    expected = {
        "fit": 40,
        "selection": 10,
        "calibration": 10,
        "iid_evaluation": 20,
        "transfer_evaluation": 10,
    }
    roles = dict(metadata["roles"])
    pairs = {
        role: set(str(value) for value in dict(roles[role])["pair_ids"])
        for role in expected
    }
    trajectories = {
        role: set(
            str(value)
            for value in dict(roles[role])["trajectory_ids"]
        )
        for role in expected
    }
    return (
        all(len(pairs[role]) == count for role, count in expected.items())
        and all(
            pairs[left].isdisjoint(pairs[right])
            and trajectories[left].isdisjoint(trajectories[right])
            for index, left in enumerate(expected)
            for right in tuple(expected)[index + 1 :]
        )
    )


def _mask_schedule_is_valid(
    root: Path, ownership: np.ndarray
) -> bool:
    with np.load(root / "mask-schedule.npz", allow_pickle=False) as stored:
        visible = stored["visible_tokens"].astype(np.bool_)
        target = stored["target_tokens"].astype(np.bool_)
        aligned = stored["aligned_target_indices"]
        deranged = stored["deranged_target_indices"]
    observed = np.flatnonzero(np.any(ownership, axis=1))
    pair_count = visible.shape[1]
    expected = np.arange(pair_count)
    return bool(
        visible.shape == target.shape
        and visible.ndim == 4
        and visible.shape[2:] == (20, 7)
        and np.array_equal(target, ~visible)
        and np.all(np.sum(target, axis=(2, 3)) == 126)
        and np.all(visible[:, :, -1, observed])
        and aligned.shape == deranged.shape
        and aligned.shape[1] == pair_count
        and np.all(aligned == expected[None])
        and np.all(
            np.sort(deranged, axis=1) == expected[None]
        )
        and np.all(deranged != expected[None])
    )


def _selection_recomputes(
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    windows: Mapping[str, ActionConditionedWindows],
    raw_scores: Mapping[str, Mapping[str, float]],
) -> bool:
    selected = dict(metadata["selected_ridges"])
    ridges = [float(value) for value in metadata["ridge_values"]]
    for name in REPRESENTATION_NAMES:
        rows = []
        for position, ridge in enumerate(ridges):
            scores = _forecast_scores(
                arrays[
                    f"ridge_prediction__{name}__{position}"
                ],
                windows["selection"],
            )
            rows.append(
                {
                    "ridge": ridge,
                    "raw_safe": (
                        scores["overall_mse"]
                        <= 1.05
                        * raw_scores["selection"]["overall_mse"]
                        and scores["action_overlap_mse"]
                        <= 1.05
                        * raw_scores["selection"][
                            "action_overlap_mse"
                        ]
                    ),
                    **scores,
                }
            )
        eligible = [row for row in rows if row["raw_safe"]]
        chosen = min(
            eligible or rows,
            key=lambda row: (
                row["downstream_effect_mse"],
                row["ridge"],
            ),
        )
        if float(selected[name]) != float(chosen["ridge"]):
            return False
    return True


def _read_json(path: Path) -> Dict[str, Any]:
    return dict(json.loads(path.read_text()))


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def main(arguments: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parsed = parser.parse_args(arguments)
    print(
        json.dumps(
            verify_stored_assessment(parsed.directory),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
