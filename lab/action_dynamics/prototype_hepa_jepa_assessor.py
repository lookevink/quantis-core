#!/usr/bin/env python3
"""Recompute ticket 012 from stored arrays without fitted models."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from quantis_core.edge_dynamics.hepa_jepa import assess_hepa_tracer


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


def assess_stored_bundle(directory: Path) -> Mapping[str, Any]:
    """Verify a completed bundle and recompute its complete assessment."""

    root = Path(directory)
    _verify_manifest(root)
    metadata = _read_object(root / "assessment-metadata.json")
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
    return assess_hepa_tracer(
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
        candidate_tokens=candidate_tokens,
        restored_candidate_tokens=restored_candidate_tokens,
        state_truth=state_truth,
        state_scale=state_scale,
        state_varying_mask=state_varying_mask,
        state_predictions=state_predictions,
        inference_parameter_counts={
            str(key): int(value)
            for key, value in dict(
                metadata["inference_parameter_counts"]
            ).items()
        },
        protocol_checks={
            str(key): bool(value)
            for key, value in dict(
                metadata["protocol_checks"]
            ).items()
        },
        edge_metrics={
            str(model): {
                str(key): float(value)
                for key, value in dict(metrics).items()
            }
            for model, metrics in dict(
                metadata["edge_metrics"]
            ).items()
        },
        raw_effect_scores=raw_effect_scores,
        event_threshold=float(metadata["event_threshold"]),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    arguments = parser.parse_args()
    assessment = verify_stored_assessment(arguments.directory)
    print(_pretty_json(assessment), end="")


if __name__ == "__main__":
    main()
