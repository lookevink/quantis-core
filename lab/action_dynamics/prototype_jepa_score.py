#!/usr/bin/env python3
"""Retained runner for the frozen exact JEPA-SCORE edge screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, cast

import numpy as np
from numpy.typing import NDArray

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
from quantis_core.edge_dynamics.jepa_score import (
    ExactJepaScorer,
    JepaScoreBatch,
    assess_jepa_score_gates,
)


FROZEN_CACHE = Path(
    "artifacts/action-dynamics/edge-preprocessing-v1/"
    "eb54271132f88c9a431b01e786ea66279a563776434cca2290e47e6b7ae9b3ff"
)
FROZEN_PRIOR = Path(
    "artifacts/action-dynamics/prototype-complete-lejepa-v1"
)
FROZEN_OUTPUT = Path(
    "artifacts/action-dynamics/prototype-jepa-score-v1"
)
FROZEN_SOURCE_CORPUS_SHA256 = (
    "df03af282f48591216251f22f934b97e1555147df9ed6f6c6d1f7684f9644a26"
)
FROZEN_SOURCE_ARTIFACT_MANIFEST_SHA256 = (
    "d02afa33a5977cba69b255cd1a5f31470751b6681da1ecae31b11e86307e65b1"
)
FROZEN_CACHE_MANIFEST_SHA256 = (
    "525bd7e68b47336fad8eb0c39c0d93b0e99a7a80c0682119be3626d6066a3fa8"
)
CACHE_FILE_SHA256 = {
    "attribution-queries.npz": (
        "d649d238511da59e2f69aa9dc21c9f6a5513c13168f74cffd3e2129daf3c5e64"
    ),
    "calibration.npz": (
        "9885f67751801b60479972e2d04f18dba7b31d3723e5991bbd94b332facaf9fb"
    ),
    "evaluation.npz": (
        "cd861d41bbce2f660b921b654cac4061a5642df1e8781c71d5dbff5ac772b706"
    ),
    "fit.npz": (
        "b481893f59cbd75c19a445c78b2c61e6d052ba8c70324993b552aec9a052a160"
    ),
    "metadata.json": (
        "816cbff2642eb41ea0cf2565074f76d736ede7f365dc3ca0200587b52e0ee6f5"
    ),
    "selection.npz": (
        "dd12288ec3cf650c250bab4e36be4530c4b60f513842fe26cf319513e3977622"
    ),
}
FROZEN_PRIOR_MANIFEST_SHA256 = (
    "00639afaee81cd3844e8b60014ed87f886a8e7a7e0b20bc50834416da1deb265"
)
FROZEN_PREPROCESSING_PROTOCOL = (
    "action_conditioned_jepa_topology_transfer_v1"
)
MODEL_SHA256 = {
    "complete_lejepa": (
        "eda9795582f2965ba1091b1dca710bc74ce2098bbc747ddfc0de3a324e39e412"
    ),
    "sigreg_only": (
        "3559d948fe0801f1b2a0d816f50e6c0269a9a6209a72fe63eec8ac88e450745e"
    ),
    "invariance_only": (
        "cbadbda2c8e4f0357ef135224b827a0d75e7a06f84821dc487df2f995fba4723"
    ),
}
CELL_NAMES = tuple(MODEL_SHA256)
ANCHORS = (19, 39, 59, 79, 97)
SCORED_ROLES = (
    "selection",
    "calibration",
    "iid_evaluation",
    "transfer_evaluation",
)
EXPECTED_ROLE_COUNTS = {
    "fit": (40, 80),
    "selection": (10, 20),
    "calibration": (10, 20),
    "iid_evaluation": (20, 40),
    "transfer_evaluation": (10, 20),
}
SNAPSHOT_FILES = (
    "lab/action_dynamics/prototype_jepa_score.py",
    "lab/action_dynamics/prototype_jepa_score_assessor.py",
    "lab/action_dynamics/prototype_jepa_score_latency.py",
    "tests/test_jepa_score.py",
    "docs/research/jepa-score-primary-source-notes.md",
    "docs/specs/jepa-score-edge-screen-v1.md",
    "docs/wayfinding/jepa-implementation-program/028-test-jepa-score.md",
)


def run_experiment(
    *,
    cache_directory: Path,
    prior_directory: Path,
    output_directory: Path,
    allow_noninterpretable_smoke: bool = False,
) -> Path:
    """Score, independently assess, and atomically publish the screen."""

    started_unix = time.time()
    cache = cache_directory.resolve()
    prior = prior_directory.resolve()
    output = output_directory.resolve()
    building = output.with_name(output.name + ".building")
    if output.exists() or building.exists():
        raise FileExistsError("JEPA-SCORE refuses an existing output")
    frozen_paths = (
        cache == (Path.cwd() / FROZEN_CACHE).resolve()
        and prior == (Path.cwd() / FROZEN_PRIOR).resolve()
        and output == (Path.cwd() / FROZEN_OUTPUT).resolve()
    )
    if not frozen_paths and not allow_noninterpretable_smoke:
        raise ValueError(
            "non-frozen JEPA-SCORE runs require smoke permission"
        )
    if (
        not frozen_paths
        and output == (Path.cwd() / FROZEN_OUTPUT).resolve()
    ):
        raise ValueError(
            "JEPA-SCORE smoke cannot use the frozen output"
        )
    if frozen_paths and _git_status():
        raise RuntimeError(
            "frozen JEPA-SCORE run requires a clean committed tree"
        )
    _verify_cache_manifest(cache)
    prior_manifest_hash = _file_sha256(
        prior / "artifact-manifest.json"
    )
    if prior_manifest_hash != FROZEN_PRIOR_MANIFEST_SHA256:
        raise ValueError("JEPA-SCORE prior manifest identity differs")
    prior_manifest = _read_json(prior / "artifact-manifest.json")
    declared_prior = dict(prior_manifest["sha256"])
    raw_models: Dict[str, bytes] = {}
    for name, expected in MODEL_SHA256.items():
        relative = f"models/{name}.json"
        path = prior / relative
        raw_models[name] = path.read_bytes()
        if (
            _file_sha256(path) != expected
            or declared_prior.get(relative) != expected
        ):
            raise ValueError(
                f"JEPA-SCORE source model identity differs: {name}"
            )

    prepared = load_edge_dynamics_cache(cache)
    source_is_frozen = bool(
        prepared.source_corpus_sha256
        == FROZEN_SOURCE_CORPUS_SHA256
        and prepared.source_artifact_manifest_sha256
        == FROZEN_SOURCE_ARTIFACT_MANIFEST_SHA256
        and prepared.preprocessing_protocol
        == FROZEN_PREPROCESSING_PROTOCOL
    )
    if frozen_paths and not source_is_frozen:
        raise ValueError("JEPA-SCORE cache identity differs")
    partitions = {
        role: partition_worker_topology(windows)
        for role, windows in prepared.windows.items()
    }
    windows_by_role = {
        "fit": partitions["fit"].in_distribution,
        "selection": partitions["selection"].in_distribution,
        "calibration": partitions["calibration"].in_distribution,
        "iid_evaluation": partitions["evaluation"].in_distribution,
        "transfer_evaluation": partitions["evaluation"].held_out,
    }
    original_by_role = {
        "fit": prepared.windows["fit"],
        "selection": prepared.windows["selection"],
        "calibration": prepared.windows["calibration"],
        "iid_evaluation": prepared.windows["evaluation"],
        "transfer_evaluation": prepared.windows["evaluation"],
    }
    _validate_role_structure(windows_by_role)

    building.mkdir(parents=True)
    models_directory = building / "models"
    models_directory.mkdir()
    for name, raw in raw_models.items():
        (models_directory / f"{name}.json").write_bytes(raw)
    scorer_by_cell = {
        name: ExactJepaScorer.from_model_json_bytes(raw_models[name])
        for name in CELL_NAMES
    }
    primary_bundle_path = building / "primary-scorer.json"
    primary_bundle_path.write_bytes(
        _canonical_json_bytes(
            scorer_by_cell["complete_lejepa"].to_dict()
        )
    )
    bundle_bytes = primary_bundle_path.stat().st_size

    (
        sampled_histories,
        sample_metadata,
        role_slices,
    ) = _sample_fixed_anchors(
        windows_by_role=windows_by_role,
        original_by_role=original_by_role,
    )
    all_scores = {
        name: _score_batches(
            scorer, sampled_histories, windows_by_role["fit"].graph
        )
        for name, scorer in scorer_by_cell.items()
    }
    _validate_treatment_balance(windows_by_role)
    batch_parity, parity_evidence = _batch_parity(
        scorer_by_cell=scorer_by_cell,
        histories=sampled_histories,
        sample_metadata=sample_metadata,
        graph=windows_by_role["fit"].graph,
    )

    (
        raw_center,
        raw_scale,
        raw_ownership,
        raw_control_ids,
        raw_fit_delta_count,
    ) = _fit_raw_comparator(windows_by_role["fit"])
    raw_scores = _raw_scores(
        sampled_histories,
        center=raw_center,
        scale=raw_scale,
        ownership=raw_ownership,
    )
    labels = {
        role: _trajectory_labels(windows)
        for role, windows in windows_by_role.items()
    }
    for index, (role, trajectory_id) in enumerate(
        zip(
            sample_metadata["role"],
            sample_metadata["trajectory_id"],
        )
    ):
        treatment, onset = labels[str(role)][str(trajectory_id)]
        sample_metadata["treatment"][index] = treatment
        sample_metadata["onset"][index] = (
            -1 if onset is None else onset
        )

    calibration_slice = role_slices["calibration"]
    calibration_control = ~sample_metadata["treatment"][
        calibration_slice
    ]
    primary_anomaly = all_scores[
        "complete_lejepa"
    ].anomaly_score
    candidate_threshold = _control_max_threshold(
        primary_anomaly[calibration_slice],
        sample_metadata["trajectory_id"][calibration_slice],
        calibration_control,
    )
    raw_threshold = _control_max_threshold(
        raw_scores[calibration_slice],
        sample_metadata["trajectory_id"][calibration_slice],
        calibration_control,
    )
    candidate_decisions = primary_anomaly > candidate_threshold
    raw_decisions = raw_scores > raw_threshold
    candidate_metrics = {
        role: _alert_metrics(
            decisions=candidate_decisions[role_slices[role]],
            trajectory_ids=sample_metadata["trajectory_id"][
                role_slices[role]
            ],
            transitions=sample_metadata["transition"][
                role_slices[role]
            ],
            labels=labels[role],
        )
        for role in ("iid_evaluation", "transfer_evaluation")
    }
    raw_metrics = {
        role: _alert_metrics(
            decisions=raw_decisions[role_slices[role]],
            trajectory_ids=sample_metadata["trajectory_id"][
                role_slices[role]
            ],
            transitions=sample_metadata["transition"][
                role_slices[role]
            ],
            labels=labels[role],
        )
        for role in ("iid_evaluation", "transfer_evaluation")
    }
    selection_pair_win_fraction = _selection_pair_win_fraction(
        anomaly=primary_anomaly[role_slices["selection"]],
        trajectory_ids=sample_metadata["trajectory_id"][
            role_slices["selection"]
        ],
        pair_ids=sample_metadata["pair_id"][role_slices["selection"]],
        transitions=sample_metadata["transition"][
            role_slices["selection"]
        ],
        labels=labels["selection"],
    )

    _write_latency_inputs(
        building / "latency-inputs.npz",
        histories=sampled_histories,
        metadata=sample_metadata,
        selection_slice=role_slices["selection"],
    )
    latency = _run_latency_worker(
        bundle=primary_bundle_path,
        inputs=building / "latency-inputs.npz",
    )

    evidence_arrays: Dict[str, Any] = {
        "sample_histories": sampled_histories,
        "sample_role": sample_metadata["role"],
        "sample_trajectory_ids": sample_metadata["trajectory_id"],
        "sample_pair_ids": sample_metadata["pair_id"],
        "sample_transitions": sample_metadata["transition"],
        "sample_source_row_indices": sample_metadata["source_row_index"],
        "sample_treatment": sample_metadata["treatment"],
        "sample_onset": sample_metadata["onset"],
        "raw_scores": raw_scores,
        "candidate_decisions": candidate_decisions,
        "raw_decisions": raw_decisions,
        "raw_center": raw_center,
        "raw_scale": raw_scale,
        "raw_ownership": raw_ownership,
        **parity_evidence,
    }
    for name, scored in all_scores.items():
        evidence_arrays.update(
            {
                f"{name}_jepa_score": scored.jepa_score,
                f"{name}_anomaly_score": scored.anomaly_score,
                f"{name}_singular_values": scored.singular_values,
                f"{name}_clipped_count": scored.clipped_count,
                f"{name}_projector_embeddings": (
                    scored.projector_embeddings
                ),
                f"{name}_unowned_jacobian_max_abs": (
                    scored.unowned_jacobian_max_abs
                ),
            }
        )
    receipt_metadata = _receipt_arrays(
        windows_by_role=windows_by_role,
        original_by_role=original_by_role,
        evidence_arrays=evidence_arrays,
    )
    np.savez_compressed(building / "evidence.npz", **evidence_arrays)

    score_diagnostics = {
        name: _score_diagnostics(
            scored,
            metadata=sample_metadata,
            role_slices=role_slices,
        )
        for name, scored in all_scores.items()
    }
    source_snapshot_hashes = _snapshot_sources(building)
    protocol_checks = {
        "source_identities_recompute": True,
        "role_contract_recomputes": True,
        "fixed_anchors_recompute": True,
        "action_blind_sampling_recomputes": True,
        "model_restoration_recomputes": True,
        "exact_score_recomputes": True,
        "batch_and_literal_parity_recompute": bool(
            all(batch_parity.values())
        ),
        "latency_contract_recomputes": _latency_contract_valid(
            latency=latency,
            histories=sampled_histories,
            metadata=sample_metadata,
            selection_slice=role_slices["selection"],
        ),
        "evidence_arrays_are_finite": bool(
            _arrays_are_finite(evidence_arrays)
        ),
        "calibration_isolation_recomputes": True,
        "alert_metrics_recompute": True,
        "evaluation_has_no_selection_authority": True,
        "source_snapshots_and_manifest_verify": True,
    }
    assessment = assess_jepa_score_gates(
        interpretable=bool(frozen_paths and source_is_frozen),
        protocol_checks=protocol_checks,
        candidate_metrics=candidate_metrics,
        raw_metrics=raw_metrics,
        selection_pair_win_fraction=selection_pair_win_fraction,
        median_latency_ms=float(latency["median_ms"]),
        p95_latency_ms=float(latency["p95_ms_higher"]),
        bundle_bytes=bundle_bytes,
        parameter_count=scorer_by_cell[
            "complete_lejepa"
        ].parameter_count,
    )
    result = {
        "schema_version": 1,
        "kind": "jepa_score_edge_screen_v1",
        "evidence_boundary": (
            "single-seed open-development exact single-transform "
            "Monte Carlo density screen; not production or sealed evidence"
        ),
        "interpretable": bool(frozen_paths and source_is_frozen),
        "implementation_commit": _git_head(),
        "source": {
            "cache_directory": str(cache),
            "prior_directory": str(prior),
            "source_corpus_sha256": prepared.source_corpus_sha256,
            "source_artifact_manifest_sha256": (
                prepared.source_artifact_manifest_sha256
            ),
            "cache_manifest_sha256": FROZEN_CACHE_MANIFEST_SHA256,
            "cache_file_sha256": dict(CACHE_FILE_SHA256),
            "preprocessing_protocol": prepared.preprocessing_protocol,
            "prior_manifest_sha256": prior_manifest_hash,
            "source_model_file_sha256": dict(MODEL_SHA256),
            "source_model_payload_sha256": {
                name: scorer.source_model_payload_sha256
                for name, scorer in scorer_by_cell.items()
            },
            "primary_paper": "https://arxiv.org/abs/2510.05949",
        },
        "fixed_contract": {
            "anchors": list(ANCHORS),
            "cells": list(CELL_NAMES),
            "roles": list(SCORED_ROLES),
            "role_slices": {
                role: [value.start, value.stop]
                for role, value in role_slices.items()
            },
            "role_counts": {
                role: {
                    "pairs": len(set(windows.matched_pair_ids)),
                    "trajectories": len(set(windows.trajectory_ids)),
                    "windows": len(windows.histories),
                }
                for role, windows in windows_by_role.items()
            },
            "raw_fit_control_trajectory_ids": list(raw_control_ids),
            "raw_definition": {
                "kind": "terminal_fit_control_delta_rms_v1",
                "delta_count": raw_fit_delta_count,
                "center": raw_center.tolist(),
                "scale": raw_scale.tolist(),
                "ownership_mask": raw_ownership.astype(int).tolist(),
            },
            "receipt_metadata": receipt_metadata,
        },
        "thresholds": {
            "candidate": candidate_threshold,
            "raw": raw_threshold,
            "rule": "strict_greater_than",
        },
        "candidate_metrics": candidate_metrics,
        "raw_metrics": raw_metrics,
        "selection_pair_win_fraction": selection_pair_win_fraction,
        "batch_parity": batch_parity,
        "score_diagnostics": score_diagnostics,
        "latency": latency,
        "bundle": {
            "path": "primary-scorer.json",
            "bytes": bundle_bytes,
            "parameter_count": scorer_by_cell[
                "complete_lejepa"
            ].parameter_count,
            "sha256": _file_sha256(primary_bundle_path),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "started_unix": started_unix,
        },
        "source_snapshot_sha256": source_snapshot_hashes,
        "protocol_checks": protocol_checks,
        "assessment": assessment,
    }
    _write_json(building / "result.json", result)
    (building / "REPORT.md").write_text(_markdown_report(result))
    manifest = {
        "schema_version": 1,
        "kind": "jepa_score_artifact_manifest_v1",
        "sha256": {
            str(path.relative_to(building)): _file_sha256(path)
            for path in sorted(building.rglob("*"))
            if path.is_file()
            and path.name != "artifact-manifest.json"
        },
    }
    _write_json(building / "artifact-manifest.json", manifest)
    assessed = _run_copied_assessor(
        artifact=building, cache=cache, prior=prior
    )
    if assessed != assessment:
        raise ValueError(
            "isolated JEPA-SCORE assessment differs from stored result"
        )
    building.rename(output)
    final = _run_copied_assessor(
        artifact=output, cache=cache, prior=prior
    )
    if final != assessment:
        raise ValueError("published JEPA-SCORE assessment drifted")
    return output


def _sample_fixed_anchors(
    *,
    windows_by_role: Mapping[str, ActionConditionedWindows],
    original_by_role: Mapping[str, ActionConditionedWindows],
) -> Tuple[NDArray[np.float64], Dict[str, Any], Mapping[str, slice]]:
    histories: list[NDArray[np.float64]] = []
    role_values: list[str] = []
    trajectory_values: list[str] = []
    pair_values: list[str] = []
    transition_values: list[int] = []
    source_rows: list[int] = []
    role_slices: Dict[str, slice] = {}
    for role in SCORED_ROLES:
        start = len(histories)
        windows = windows_by_role[role]
        original = original_by_role[role]
        source_lookup = {
            (trajectory_id, int(transition)): index
            for index, (trajectory_id, transition) in enumerate(
                zip(
                    original.trajectory_ids,
                    original.transition_indices,
                )
            )
        }
        lookup = {
            (trajectory_id, int(transition)): index
            for index, (trajectory_id, transition) in enumerate(
                zip(windows.trajectory_ids, windows.transition_indices)
            )
        }
        for trajectory_id in sorted(set(windows.trajectory_ids)):
            for transition in ANCHORS:
                key = (trajectory_id, transition)
                if key not in lookup or key not in source_lookup:
                    raise ValueError(
                        "JEPA-SCORE fixed anchor is unavailable"
                    )
                index = lookup[key]
                histories.append(windows.histories[index])
                role_values.append(role)
                trajectory_values.append(trajectory_id)
                pair_values.append(windows.matched_pair_ids[index])
                transition_values.append(transition)
                source_rows.append(source_lookup[key])
        role_slices[role] = slice(start, len(histories))
    count = len(histories)
    return (
        np.asarray(histories, dtype=np.float64),
        {
            "role": np.asarray(role_values),
            "trajectory_id": np.asarray(trajectory_values),
            "pair_id": np.asarray(pair_values),
            "transition": np.asarray(
                transition_values, dtype=np.int64
            ),
            "source_row_index": np.asarray(
                source_rows, dtype=np.int64
            ),
            "treatment": np.zeros(count, dtype=np.bool_),
            "onset": np.full(count, -1, dtype=np.int64),
        },
        role_slices,
    )


def _score_batches(
    scorer: ExactJepaScorer,
    histories: NDArray[np.float64],
    graph: Any,
    *,
    batch_size: int = 10,
) -> JepaScoreBatch:
    batches = [
        scorer.score(histories[start : start + batch_size], graph)
        for start in range(0, len(histories), batch_size)
    ]
    return JepaScoreBatch(
        jepa_score=np.concatenate(
            [value.jepa_score for value in batches]
        ),
        anomaly_score=np.concatenate(
            [value.anomaly_score for value in batches]
        ),
        singular_values=np.concatenate(
            [value.singular_values for value in batches]
        ),
        clipped_count=np.concatenate(
            [value.clipped_count for value in batches]
        ),
        projector_embeddings=np.concatenate(
            [value.projector_embeddings for value in batches]
        ),
        unowned_jacobian_max_abs=np.concatenate(
            [value.unowned_jacobian_max_abs for value in batches]
        ),
    )


def _batch_parity(
    *,
    scorer_by_cell: Mapping[str, ExactJepaScorer],
    histories: NDArray[np.float64],
    sample_metadata: Mapping[str, Any],
    graph: Any,
) -> Tuple[Mapping[str, bool], Mapping[str, Any]]:
    positions = np.flatnonzero(
        (sample_metadata["role"] == "selection")
        & (sample_metadata["transition"] == 39)
    )[:3]
    if len(positions) != 3:
        raise ValueError("JEPA-SCORE parity batch is unavailable")
    result: Dict[str, bool] = {}
    evidence: Dict[str, Any] = {
        "parity_positions": positions.astype(np.int64)
    }
    for name, scorer in scorer_by_cell.items():
        together = scorer.score(histories[positions], graph)
        separate = [
            scorer.score(histories[index : index + 1], graph)
            for index in positions
        ]
        single_scores = np.concatenate(
            [value.jepa_score for value in separate]
        )
        single_singular = np.concatenate(
            [value.singular_values for value in separate]
        )
        evidence[f"{name}_parity_batch_scores"] = (
            together.jepa_score
        )
        evidence[f"{name}_parity_batch_singular_values"] = (
            together.singular_values
        )
        evidence[f"{name}_parity_single_scores"] = single_scores
        evidence[f"{name}_parity_single_singular_values"] = (
            single_singular
        )
        result[name] = bool(
            np.allclose(
                together.jepa_score,
                single_scores,
                atol=1e-3,
                rtol=0.0,
            )
            and np.allclose(
                together.singular_values,
                single_singular,
                atol=2e-5,
                rtol=0.0,
            )
        )
    return result, evidence


def _trajectory_labels(
    windows: ActionConditionedWindows,
) -> Mapping[str, Tuple[bool, Optional[int]]]:
    applicable = windows.action_feature_names.index("applicable")
    result: Dict[str, Tuple[bool, Optional[int]]] = {}
    for trajectory_id in sorted(set(windows.trajectory_ids)):
        positions = np.flatnonzero(
            np.asarray(windows.trajectory_ids) == trajectory_id
        )
        onsets: list[int] = []
        for position in positions:
            active = np.flatnonzero(
                np.any(
                    windows.future_actions[
                        position, :, :, applicable
                    ]
                    > 0.5,
                    axis=1,
                )
            )
            onsets.extend(
                int(windows.transition_indices[position]) + int(offset)
                for offset in active
            )
        result[trajectory_id] = (
            bool(onsets),
            min(onsets) if onsets else None,
        )
    return result


def _raw_scores(
    histories: NDArray[np.float64],
    *,
    center: NDArray[np.float64],
    scale: NDArray[np.float64],
    ownership: NDArray[np.bool_],
) -> NDArray[np.float64]:
    delta = histories[:, -1] - histories[:, -2]
    standardized = (delta - center[None]) / scale[None]
    return cast(
        NDArray[np.float64],
        np.sqrt(
            np.mean(np.square(standardized[:, ownership]), axis=1)
        ).astype(np.float64),
    )


def _fit_raw_comparator(
    windows: ActionConditionedWindows,
) -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.bool_],
    Tuple[str, ...],
    int,
]:
    labels = _trajectory_labels(windows)
    controls = tuple(
        trajectory_id
        for trajectory_id in sorted(labels)
        if not labels[trajectory_id][0]
    )
    deltas = []
    for trajectory_id in controls:
        positions = np.flatnonzero(
            np.asarray(windows.trajectory_ids) == trajectory_id
        )
        order = positions[
            np.argsort(windows.transition_indices[positions])
        ]
        if (
            len(order) != 79
            or not np.array_equal(
                windows.transition_indices[order],
                np.arange(19, 98, dtype=np.int64),
            )
        ):
            raise ValueError(
                "JEPA-SCORE raw fit-control rows do not align"
            )
        deltas.append(
            windows.histories[order, -1]
            - windows.histories[order, -2]
        )
    combined = np.concatenate(deltas, axis=0)
    if len(controls) != 40 or len(combined) != 3160:
        raise ValueError("JEPA-SCORE raw fit population differs")
    ownership = fit_owned_feature_mask(windows)
    center = np.median(combined, axis=0)
    mad = 1.4826 * np.median(
        np.abs(combined - center[None]), axis=0
    )
    standard_deviation = np.std(combined, axis=0)
    scale = np.where(
        mad > 1e-8,
        mad,
        np.where(
            standard_deviation > 1e-8,
            standard_deviation,
            1.0,
        ),
    )
    center = np.where(ownership, center, 0.0)
    scale = np.where(ownership, scale, 1.0)
    return (
        np.asarray(center, dtype=np.float64),
        np.asarray(scale, dtype=np.float64),
        ownership,
        controls,
        len(combined),
    )


def _control_max_threshold(
    scores: NDArray[np.float64],
    trajectory_ids: NDArray[np.str_],
    control_rows: NDArray[np.bool_],
) -> float:
    values = [
        float(np.max(scores[(trajectory_ids == trajectory_id)]))
        for trajectory_id in sorted(
            set(str(value) for value in trajectory_ids[control_rows])
        )
    ]
    if len(values) != 10:
        raise ValueError(
            "JEPA-SCORE calibration needs ten control trajectories"
        )
    return float(
        np.quantile(np.asarray(values), 0.95, method="higher")
    )


def _alert_metrics(
    *,
    decisions: NDArray[np.bool_],
    trajectory_ids: NDArray[np.str_],
    transitions: NDArray[np.int64],
    labels: Mapping[str, Tuple[bool, Optional[int]]],
) -> Mapping[str, Any]:
    rows: list[Dict[str, Any]] = []
    for trajectory_id in sorted(set(str(value) for value in trajectory_ids)):
        selected = trajectory_ids == trajectory_id
        alerts = transitions[selected][decisions[selected]]
        treatment, onset = labels[trajectory_id]
        if treatment and onset is None:
            raise ValueError("treatment trajectory has no onset")
        post = (
            alerts[alerts >= onset]
            if onset is not None
            else np.asarray([], dtype=np.int64)
        )
        first = int(np.min(post)) if len(post) else None
        rows.append(
            {
                "trajectory_id": trajectory_id,
                "is_treatment": treatment,
                "onset": onset,
                "any_alert": bool(len(alerts)),
                "pre_onset_alert": bool(
                    onset is not None and np.any(alerts < onset)
                ),
                "first_post_onset_alert_transition": first,
                "post_onset_delay_transitions": (
                    None
                    if first is None or onset is None
                    else first - onset
                ),
            }
        )
    controls = [row for row in rows if not row["is_treatment"]]
    treatments = [row for row in rows if row["is_treatment"]]
    detected = [
        row
        for row in treatments
        if row["first_post_onset_alert_transition"] is not None
    ]
    delays: list[int] = []
    for row in detected:
        delay = row["post_onset_delay_transitions"]
        if not isinstance(delay, int):
            raise ValueError("detected JEPA-SCORE alert has no delay")
        delays.append(delay)
    return {
        "control_trajectory_count": len(controls),
        "treatment_trajectory_count": len(treatments),
        "control_trajectory_false_alarm_rate": float(
            np.mean([bool(row["any_alert"]) for row in controls])
        ),
        "treatment_detection_rate": float(
            len(detected) / len(treatments)
        ),
        "treatment_pre_onset_alert_rate": float(
            np.mean(
                [
                    bool(row["pre_onset_alert"])
                    for row in treatments
                ]
            )
        ),
        "median_post_onset_delay_transitions": (
            float(np.median(delays)) if delays else None
        ),
        "trajectory_rows": rows,
    }


def _selection_pair_win_fraction(
    *,
    anomaly: NDArray[np.float64],
    trajectory_ids: NDArray[np.str_],
    pair_ids: NDArray[np.str_],
    transitions: NDArray[np.int64],
    labels: Mapping[str, Tuple[bool, Optional[int]]],
) -> float:
    wins = []
    for pair_id in sorted(set(str(value) for value in pair_ids)):
        selected = (pair_ids == pair_id) & (transitions == 39)
        positions = np.flatnonzero(selected)
        if len(positions) != 2:
            raise ValueError("selection pair anchor does not align")
        treatment = [
            position
            for position in positions
            if labels[str(trajectory_ids[position])][0]
        ]
        control = [
            position
            for position in positions
            if not labels[str(trajectory_ids[position])][0]
        ]
        if len(treatment) != 1 or len(control) != 1:
            raise ValueError("selection pair labels do not align")
        wins.append(anomaly[treatment[0]] > anomaly[control[0]])
    return float(np.mean(wins))


def _score_diagnostics(
    scored: JepaScoreBatch,
    *,
    metadata: Mapping[str, Any],
    role_slices: Mapping[str, slice],
) -> Mapping[str, Any]:
    embeddings = scored.projector_embeddings
    centered = embeddings - embeddings.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    probabilities = singular / max(float(np.sum(singular)), 1e-12)
    nonzero = probabilities[probabilities > 0.0]
    covariance = np.cov(embeddings, rowvar=False)
    off_diagonal = covariance[
        ~np.eye(covariance.shape[0], dtype=np.bool_)
    ]
    distributions = {}
    for role, role_slice in role_slices.items():
        for arm_name, treatment in (
            ("control", False),
            ("treatment", True),
        ):
            selected = (
                metadata["treatment"][role_slice] == treatment
            )
            values = scored.jepa_score[role_slice][selected]
            distributions[f"{role}_{arm_name}"] = {
                "count": int(len(values)),
                "minimum": float(np.min(values)),
                "median": float(np.median(values)),
                "maximum": float(np.max(values)),
                "mean": float(np.mean(values)),
            }
    return {
        "effective_rank": float(
            np.exp(-np.sum(nonzero * np.log(nonzero)))
        ),
        "mean_abs_marginal_mean": float(
            np.mean(np.abs(embeddings.mean(axis=0)))
        ),
        "mean_marginal_variance": float(
            np.mean(np.var(embeddings, axis=0))
        ),
        "mean_abs_off_diagonal_covariance": float(
            np.mean(np.abs(off_diagonal))
        ),
        "singular_value_clipping_count": int(
            np.sum(scored.clipped_count)
        ),
        "unowned_jacobian_max_abs": float(
            np.max(scored.unowned_jacobian_max_abs)
        ),
        "score_distributions": distributions,
    }


def _receipt_arrays(
    *,
    windows_by_role: Mapping[str, ActionConditionedWindows],
    original_by_role: Mapping[str, ActionConditionedWindows],
    evidence_arrays: Dict[str, Any],
) -> Mapping[str, Any]:
    result = {}
    for role, windows in windows_by_role.items():
        applicable = windows.action_feature_names.index("applicable")
        original = original_by_role[role]
        source_lookup = {
            (trajectory_id, int(transition)): index
            for index, (trajectory_id, transition) in enumerate(
                zip(
                    original.trajectory_ids,
                    original.transition_indices,
                )
            )
        }
        source_rows = np.asarray(
            [
                source_lookup[(trajectory_id, int(transition))]
                for trajectory_id, transition in zip(
                    windows.trajectory_ids, windows.transition_indices
                )
            ],
            dtype=np.int64,
        )
        prefix = f"receipt_{role}"
        evidence_arrays[f"{prefix}_trajectory_ids"] = np.asarray(
            windows.trajectory_ids
        )
        evidence_arrays[f"{prefix}_pair_ids"] = np.asarray(
            windows.matched_pair_ids
        )
        evidence_arrays[f"{prefix}_transitions"] = (
            windows.transition_indices
        )
        evidence_arrays[f"{prefix}_source_row_indices"] = source_rows
        evidence_arrays[f"{prefix}_applicable"] = np.any(
            windows.future_actions[:, :, :, applicable] > 0.5,
            axis=2,
        )
        result[role] = {
            "rows": len(windows.histories),
            "source_row_sha256": _array_sha256(source_rows),
            "applicable_sha256": _array_sha256(
                evidence_arrays[f"{prefix}_applicable"]
            ),
        }
    return result


def _write_latency_inputs(
    path: Path,
    *,
    histories: NDArray[np.float64],
    metadata: Mapping[str, Any],
    selection_slice: slice,
) -> None:
    selection_positions = np.arange(
        selection_slice.start, selection_slice.stop
    )
    transition_19 = selection_positions[
        metadata["transition"][selection_slice] == 19
    ]
    transition_39 = selection_positions[
        metadata["transition"][selection_slice] == 39
    ]
    if len(transition_19) != 20 or len(transition_39) != 20:
        raise ValueError("JEPA-SCORE latency sample rotation differs")
    np.savez_compressed(
        path,
        warmup_history=histories[transition_19[:1]],
        warmup_trajectory_id=metadata["trajectory_id"][
            transition_19[:1]
        ],
        measurement_histories=histories[transition_39],
        measurement_trajectory_ids=metadata["trajectory_id"][
            transition_39
        ],
        measurement_transitions=metadata["transition"][
            transition_39
        ],
    )


def _run_latency_worker(
    *, bundle: Path, inputs: Path
) -> Mapping[str, Any]:
    environment = dict(os.environ)
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "lab/action_dynamics/prototype_jepa_score_latency.py",
            "--bundle",
            str(bundle),
            "--inputs",
            str(inputs),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    value = json.loads(result.stdout)
    if not isinstance(value, dict):
        raise ValueError("JEPA-SCORE latency receipt is invalid")
    return dict(value)


def _latency_contract_valid(
    *,
    latency: Mapping[str, Any],
    histories: NDArray[np.float64],
    metadata: Mapping[str, Any],
    selection_slice: slice,
) -> bool:
    positions = np.arange(selection_slice.start, selection_slice.stop)
    warmup = positions[
        metadata["transition"][selection_slice] == 19
    ]
    measurements = positions[
        metadata["transition"][selection_slice] == 39
    ]
    samples = np.asarray(latency.get("samples_ms"), dtype=np.float64)
    return bool(
        len(warmup) == 20
        and len(measurements) == 20
        and histories[warmup[:1]].shape == (1, 20, 7, 31)
        and len(samples) == 20
        and np.all(np.isfinite(samples))
        and latency.get("measurement_trajectory_ids")
        == list(metadata["trajectory_id"][measurements])
        and latency.get("measurement_transitions") == [39] * 20
        and latency.get("warmup_count") == 1
        and latency.get("measurement_count") == 20
        and latency.get("timer") == "time.perf_counter_ns"
        and latency.get("model_load_excluded") is True
        and latency.get("torch_intraop_threads") == 1
        and latency.get("torch_interop_threads") == 1
        and latency.get("omp_num_threads") == "1"
        and latency.get("mkl_num_threads") == "1"
        and float(np.median(samples)) == latency.get("median_ms")
        and float(np.quantile(samples, 0.95, method="higher"))
        == latency.get("p95_ms_higher")
    )


def _snapshot_sources(directory: Path) -> Mapping[str, str]:
    destination = directory / "reproduction-source"
    shutil.copytree(
        Path("src/quantis_core"),
        destination / "src/quantis_core",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for name in SNAPSHOT_FILES:
        source = Path(name)
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    hashes = {
        str(path.relative_to(destination)): _file_sha256(path)
        for path in sorted(destination.rglob("*"))
        if path.is_file()
    }
    _write_json(destination / "source-sha256.json", hashes)
    return hashes


def _run_copied_assessor(
    *, artifact: Path, cache: Path, prior: Path
) -> Mapping[str, Any]:
    reproduction = artifact / "reproduction-source"
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            (
                "import runpy,sys;"
                "sys.path[:0]=sys.argv[1:4];"
                "sys.argv=sys.argv[4:];"
                "runpy.run_path(sys.argv[0],run_name='__main__')"
            ),
            str(reproduction / "src"),
            str(reproduction / "lab/action_dynamics"),
            str(Path(np.__file__).resolve().parents[1]),
            str(
                reproduction
                / "lab/action_dynamics/"
                "prototype_jepa_score_assessor.py"
            ),
            "--artifact",
            str(artifact),
            "--cache",
            str(cache),
            "--prior",
            str(prior),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=artifact.parent,
    )
    value = json.loads(result.stdout)
    if not isinstance(value, dict):
        raise ValueError("copied JEPA-SCORE assessor returned invalid output")
    return dict(value)


def _validate_role_structure(
    windows_by_role: Mapping[str, ActionConditionedWindows],
) -> None:
    pair_sets = {}
    trajectory_sets = {}
    for role, windows in windows_by_role.items():
        pairs = set(windows.matched_pair_ids)
        trajectories = set(windows.trajectory_ids)
        expected_pairs, expected_trajectories = EXPECTED_ROLE_COUNTS[role]
        if (
            len(pairs) != expected_pairs
            or len(trajectories) != expected_trajectories
        ):
            raise ValueError(f"JEPA-SCORE {role} count differs")
        pair_sets[role] = pairs
        trajectory_sets[role] = trajectories
    for left_index, left in enumerate(windows_by_role):
        for right in tuple(windows_by_role)[left_index + 1 :]:
            if pair_sets[left] & pair_sets[right]:
                raise ValueError("JEPA-SCORE pair roles overlap")
            if trajectory_sets[left] & trajectory_sets[right]:
                raise ValueError("JEPA-SCORE trajectory roles overlap")


def _validate_treatment_balance(
    windows_by_role: Mapping[str, ActionConditionedWindows],
) -> None:
    for windows in windows_by_role.values():
        labels = _trajectory_labels(windows)
        for pair_id in set(windows.matched_pair_ids):
            pair_trajectories = {
                trajectory_id
                for trajectory_id, candidate_pair in zip(
                    windows.trajectory_ids, windows.matched_pair_ids
                )
                if candidate_pair == pair_id
            }
            if (
                len(pair_trajectories) != 2
                or sorted(
                    labels[value][0] for value in pair_trajectories
                )
                != [False, True]
            ):
                raise ValueError(
                    "JEPA-SCORE pair treatment/control balance differs"
                )


def _verify_cache_manifest(cache: Path) -> None:
    path = cache / "artifact-manifest.json"
    if _file_sha256(path) != FROZEN_CACHE_MANIFEST_SHA256:
        raise ValueError("JEPA-SCORE cache manifest identity differs")
    manifest = _read_json(path)
    declared = dict(manifest.get("sha256", {}))
    if declared != CACHE_FILE_SHA256:
        raise ValueError("JEPA-SCORE cache manifest file set differs")
    for name, expected in CACHE_FILE_SHA256.items():
        if _file_sha256(cache / name) != expected:
            raise ValueError(f"JEPA-SCORE cache file differs: {name}")


def _arrays_are_finite(arrays: Mapping[str, Any]) -> bool:
    for value in arrays.values():
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.number) and not np.all(
            np.isfinite(array)
        ):
            return False
    return True


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(
        str(array.dtype).encode()
        + str(array.shape).encode()
        + array.tobytes()
    ).hexdigest()


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> Mapping[str, Any]:
    return dict(json.loads(path.read_text()))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_status() -> str:
    return subprocess.run(
        ["git", "status", "--short"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _markdown_report(result: Mapping[str, Any]) -> str:
    assessment = dict(result["assessment"])
    latency = dict(result["latency"])
    candidate = dict(result["candidate_metrics"])
    transfer = dict(candidate["transfer_evaluation"])
    return "\n".join(
        (
            "# Exact JEPA-SCORE edge screen v1",
            "",
            f"- Decision: `{assessment['decision']}`",
            f"- Passed: `{assessment['passed']}`",
            f"- Interpretable: `{result['interpretable']}`",
            "- Exact latency median / p95: "
            f"`{latency['median_ms']:.3f} / "
            f"{latency['p95_ms_higher']:.3f} ms`",
            "- Transfer detection / control false alarms: "
            f"`{transfer['treatment_detection_rate']:.3f} / "
            f"{transfer['control_trajectory_false_alarm_rate']:.3f}`",
            "",
            "See `result.json` and `evidence.npz` for retained evidence.",
            "",
        )
    )


def main(arguments: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=FROZEN_CACHE)
    parser.add_argument("--prior", type=Path, default=FROZEN_PRIOR)
    parser.add_argument("--output", type=Path, default=FROZEN_OUTPUT)
    parser.add_argument(
        "--allow-noninterpretable-smoke", action="store_true"
    )
    options = parser.parse_args(arguments)
    output = run_experiment(
        cache_directory=options.cache,
        prior_directory=options.prior,
        output_directory=options.output,
        allow_noninterpretable_smoke=(
            options.allow_noninterpretable_smoke
        ),
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
