#!/usr/bin/env python3
"""Recompute ticket 009 from stored arrays without loading fitted models."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from quantis_core.edge_dynamics.retrieval_jepa import assess_retrieval_jepa


MODEL_NAMES = (
    "episode_predictive_jepa",
    "raw_telemetry",
    "pca_64",
    "deranged_target_jepa",
    "cpc_infonce",
    "supervised_retriever",
)
ROLE_NAMES = (
    "calibration",
    "selection_iid",
    "selection_transfer",
    "evaluation_iid",
    "evaluation_transfer",
)


def assess_stored_bundle(directory: Path) -> Mapping[str, Any]:
    """Verify a completed bundle and recompute its complete assessment."""

    root = Path(directory)
    _verify_manifest(root)
    metadata = _read_object(root / "retrieval-metadata.json")
    episode_metadata = _read_object(root / "episode-metadata.json")
    with np.load(
        root / "retrieval-evidence.npz", allow_pickle=False
    ) as arrays:
        similarities = {
            role: {
                model: arrays[f"similarity__{role}__{model}"]
                for model in MODEL_NAMES
            }
            for role in ROLE_NAMES
        }
        bank_vectors = {
            model: arrays[f"bank_vectors__{model}"]
            for model in MODEL_NAMES
        }
        restored_bank_vectors = {
            model: arrays[f"restored_bank_vectors__{model}"]
            for model in MODEL_NAMES
        }
        state_predictions = {
            model: arrays[f"state_prediction__{model}"]
            for model in MODEL_NAMES
        }
        original_query_vectors = {
            role: {
                model: arrays[f"original_query__{role}__{model}"]
                for model in MODEL_NAMES
            }
            for role in ROLE_NAMES
        }
        restored_query_vectors = {
            role: {
                model: arrays[f"restored_query__{role}__{model}"]
                for model in MODEL_NAMES
            }
            for role in ROLE_NAMES
        }
        state_truth = arrays["state_truth"]
        state_scale = arrays["state_scale"]
        state_varying_mask = arrays["state_varying_mask"]
        causality_audit = {
            "original_contexts": arrays[
                "causality_original_contexts"
            ],
            "counterfactual_contexts": arrays[
                "causality_counterfactual_contexts"
            ],
            "original_evidence": arrays[
                "causality_original_evidence"
            ],
            "counterfactual_evidence": arrays[
                "causality_counterfactual_evidence"
            ],
            "original_topology_values": arrays[
                "causality_original_topology_values"
            ],
            "counterfactual_topology_values": arrays[
                "causality_counterfactual_topology_values"
            ],
            "original_vectors": {
                model: arrays[f"causality_original_query__{model}"]
                for model in MODEL_NAMES
            },
            "counterfactual_vectors": {
                model: arrays[
                    f"causality_counterfactual_query__{model}"
                ]
                for model in MODEL_NAMES
            },
        }
    query_labels = {
        role: tuple(
            str(value)
            for value in metadata["roles"][role]["query_labels"]
        )
        for role in ROLE_NAMES
    }
    is_treatment = {
        role: np.asarray(
            metadata["roles"][role]["is_treatment"], dtype=np.bool_
        )
        for role in ROLE_NAMES
    }
    pair_ids = {
        role: tuple(
            str(value)
            for value in metadata["roles"][role]["pair_ids"]
        )
        for role in ROLE_NAMES
    }
    protocol_checks = _derive_protocol_checks(
        metadata, episode_metadata, similarities, causality_audit
    )
    return assess_retrieval_jepa(
        gallery_episode_ids=tuple(
            str(value) for value in metadata["gallery_episode_ids"]
        ),
        gallery_labels=tuple(
            str(value) for value in metadata["gallery_labels"]
        ),
        similarities=similarities,
        query_labels=query_labels,
        is_treatment=is_treatment,
        pair_ids=pair_ids,
        bank_vectors=bank_vectors,
        restored_bank_vectors=restored_bank_vectors,
        state_truth=state_truth,
        state_scale=state_scale,
        state_varying_mask=state_varying_mask,
        state_predictions=state_predictions,
        original_query_vectors=original_query_vectors,
        restored_query_vectors=restored_query_vectors,
        protocol_checks=protocol_checks,
        edge_metrics={
            str(model): {
                str(key): float(value)
                for key, value in dict(metrics).items()
            }
            for model, metrics in dict(
                metadata["edge_metrics"]
            ).items()
        },
    )


def verify_stored_assessment(directory: Path) -> Mapping[str, Any]:
    """Require byte-identical canonical stored and recomputed assessment."""

    root = Path(directory)
    recomputed = assess_stored_bundle(root)
    expected = _pretty_json(recomputed)
    actual = (root / "assessment.json").read_text()
    if actual != expected:
        raise ValueError("stored retrieval assessment does not recompute")
    return recomputed


def _derive_protocol_checks(
    metadata: Mapping[str, Any],
    episode_metadata: Mapping[str, Any],
    similarities: Mapping[str, Mapping[str, np.ndarray]],
    causality_audit: Mapping[str, Any],
) -> Mapping[str, bool]:
    pair_roles = {
        role: set(str(value) for value in values)
        for role, values in dict(
            episode_metadata["source_role_pair_ids"]
        ).items()
    }
    disjoint = True
    names = tuple(pair_roles)
    for left_position, left in enumerate(names):
        for right in names[left_position + 1 :]:
            disjoint = disjoint and not (
                pair_roles[left] & pair_roles[right]
            )
    gallery_count = len(metadata["gallery_episode_ids"])
    equal_bank = (
        gallery_count == 40
        and len(set(metadata["gallery_episode_ids"])) == gallery_count
        and list(metadata["gallery_episode_ids"])
        == sorted(metadata["gallery_episode_ids"])
        and all(
            values.shape[1] == gallery_count
            for role in similarities.values()
            for values in role.values()
        )
    )
    expected_counts = {
        "fit_probe": 80,
        "fit_gallery": 40,
        "calibration": 30,
        "selection_iid": 20,
        "selection_transfer": 10,
        "evaluation_iid": 40,
        "evaluation_transfer": 20,
    }
    actual_counts = {
        str(name): int(value)
        for name, value in dict(
            episode_metadata["episode_counts"]
        ).items()
    }
    raw_causality_metadata = dict(metadata["causality_audit"])
    context_equal = np.array_equal(
        causality_audit["original_contexts"],
        causality_audit["counterfactual_contexts"],
    )
    forbidden_changed = (
        not np.array_equal(
            causality_audit["original_evidence"],
            causality_audit["counterfactual_evidence"],
        )
        and not np.array_equal(
            causality_audit["original_topology_values"],
            causality_audit["counterfactual_topology_values"],
        )
        and raw_causality_metadata["original_action_labels"]
        != raw_causality_metadata["counterfactual_action_labels"]
        and raw_causality_metadata["original_pair_ids"]
        != raw_causality_metadata["counterfactual_pair_ids"]
    )
    vectors_equal = all(
        np.array_equal(
            causality_audit["original_vectors"][model],
            causality_audit["counterfactual_vectors"][model],
        )
        for model in MODEL_NAMES
    )
    return {
        "role_pairs_are_disjoint": disjoint,
        "query_future_is_excluded": (
            metadata.get("query_encoder_inputs")
            == ["contexts", "declared_graph"]
            and context_equal
            and forbidden_changed
            and vectors_equal
        ),
        "action_and_identifiers_are_excluded": (
            metadata.get("forbidden_query_encoder_inputs")
            == [
                "future_states",
                "future_controls",
                "future_actions",
                "action_labels",
                "pair_ids",
                "trajectory_ids",
                "transition_indices",
            ]
            and context_equal
            and forbidden_changed
            and vectors_equal
        ),
        "bank_membership_is_equal_and_immutable": equal_bank,
        "episode_counts_match_contract": actual_counts == expected_counts,
    }


def _verify_manifest(root: Path) -> None:
    manifest = _read_object(root / "artifact-manifest.json")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind")
        != "retrieval_jepa_artifact_manifest"
    ):
        raise ValueError("unsupported retrieval artifact manifest")
    recorded = dict(manifest["files"])
    expected_files = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and path.name != "artifact-manifest.json"
    }
    if set(recorded) != expected_files:
        raise ValueError("retrieval artifact manifest file set mismatch")
    for relative, identity in recorded.items():
        path = root / relative
        if (
            int(identity["bytes"]) != path.stat().st_size
            or str(identity["sha256"]) != _file_sha256(path)
        ):
            raise ValueError("retrieval artifact content identity mismatch")


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
