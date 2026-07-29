#!/usr/bin/env python3
"""Independent literal assessor for the exact JEPA-SCORE artifact.

This module deliberately does not import or call the production scorer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, cast

import numpy as np
from numpy.typing import NDArray

from quantis_core.action_conditioned_dynamics import (
    ActionConditionedWindows,
)
from quantis_core.edge_dynamics.complete_lejepa import (
    CompleteLejepaRepresentation,
    TelemetryViewSchedule,
)
from quantis_core.edge_dynamics.data import (
    load_edge_dynamics_cache,
    partition_worker_topology,
)


CACHE_MANIFEST_SHA256 = (
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
PRIOR_MANIFEST_SHA256 = (
    "00639afaee81cd3844e8b60014ed87f886a8e7a7e0b20bc50834416da1deb265"
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
CELLS = tuple(MODEL_SHA256)
ROLES = (
    "selection",
    "calibration",
    "iid_evaluation",
    "transfer_evaluation",
)
ANCHORS = (19, 39, 59, 79, 97)
EPSILON = 1e-6


def assess_artifact(
    *, artifact: Path, cache: Path, prior: Path
) -> Mapping[str, Any]:
    """Recompute every conclusion-bearing gate from retained evidence."""

    artifact = artifact.resolve()
    cache = cache.resolve()
    prior = prior.resolve()
    result = _read_json(artifact / "result.json")
    evidence = np.load(artifact / "evidence.npz", allow_pickle=False)
    bundle = _read_json(artifact / "primary-scorer.json")
    manifest_valid = _verify_artifact_manifest(artifact)
    snapshot_valid = _verify_source_snapshot(artifact)
    snapshot_declaration = _read_json(
        artifact / "reproduction-source" / "source-sha256.json"
    )
    snapshot_report_matches = (
        result.get("source_snapshot_sha256") == snapshot_declaration
    )
    cache_valid = _verify_cache(cache)
    prior_valid = _verify_prior(artifact, prior)

    prepared = load_edge_dynamics_cache(cache)
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
    labels = {
        role: _trajectory_labels(windows)
        for role, windows in windows_by_role.items()
    }
    role_contract = _role_contract_recomputes(windows_by_role)
    sampling_checks = _sampling_recomputes(
        evidence=evidence,
        windows_by_role=windows_by_role,
        original_by_role=original_by_role,
        labels=labels,
    )
    receipts_recompute = _receipts_recompute(
        evidence=evidence,
        windows_by_role=windows_by_role,
        original_by_role=original_by_role,
        result=result,
    )
    raw_center, raw_scale, raw_ownership, raw_controls = (
        _fit_raw_comparator(windows_by_role["fit"])
    )
    raw_scores = _raw_scores(
        np.asarray(evidence["sample_histories"], dtype=np.float64),
        center=raw_center,
        scale=raw_scale,
        ownership=raw_ownership,
    )
    raw_recomputes = bool(
        np.array_equal(
            raw_ownership, np.asarray(evidence["raw_ownership"])
        )
        and np.allclose(
            raw_center,
            np.asarray(evidence["raw_center"]),
            atol=0.0,
            rtol=0.0,
        )
        and np.allclose(
            raw_scale,
            np.asarray(evidence["raw_scale"]),
            atol=0.0,
            rtol=0.0,
        )
        and np.allclose(
            raw_scores,
            np.asarray(evidence["raw_scores"]),
            atol=1e-12,
            rtol=0.0,
        )
        and list(raw_controls)
        == result["fixed_contract"][
            "raw_fit_control_trajectory_ids"
        ]
        and _canonical_json(
            result["fixed_contract"]["raw_definition"]
        )
        == _canonical_json(
            {
                "kind": "terminal_fit_control_delta_rms_v1",
                "delta_count": 3160,
                "center": raw_center.tolist(),
                "scale": raw_scale.tolist(),
                "ownership_mask": raw_ownership.astype(int).tolist(),
            }
        )
    )

    source_payloads = {
        name: _read_json(artifact / "models" / f"{name}.json")
        for name in CELLS
    }
    models = {
        name: _restore_without_rng_drift(payload)
        for name, payload in source_payloads.items()
    }
    model_ownership_matches = all(
        np.array_equal(
            model._fitted_values()[2], raw_ownership
        )
        for model in models.values()
    )
    strict_model = _strict_model_payload(
        source_payloads["complete_lejepa"]
    )
    visible = np.asarray(bundle["visible_tokens"], dtype=np.bool_)
    present = np.asarray(bundle["present_tokens"], dtype=np.bool_)
    primary = models["complete_lejepa"]
    graph, _, ownership, _ = primary._fitted_values()
    regenerated = TelemetryViewSchedule(
        graph=graph,
        ownership_mask=ownership,
        varying_entity_mask=np.any(ownership, axis=1),
        seed=primary.config.view_seed,
    ).batch(
        np.zeros((1, 20, 7, 31), dtype=np.float64),
        step=1600,
    )
    bundle_identity = bool(
        bundle.get("epsilon") == EPSILON
        and bundle.get("source_model_file_sha256")
        == MODEL_SHA256["complete_lejepa"]
        and bundle.get("source_model_payload_sha256")
        == _canonical_sha256(source_payloads["complete_lejepa"])
        and strict_model == bundle.get("strict_model_payload")
        and bundle.get("strict_model_payload_sha256")
        == _canonical_sha256(strict_model)
        and np.array_equal(
            visible, regenerated.visible_tokens[0, 0]
        )
        and np.array_equal(
            present, regenerated.present_tokens[0, 0]
        )
        and bundle.get("view_sha256")
        == _view_sha256(visible, present)
    )
    exact_recomputes, literal_parity = _literal_score_checks(
        evidence=evidence,
        models=models,
        graph=graph,
        visible=visible,
        present=present,
    )

    sample_roles = np.asarray(evidence["sample_role"]).astype(str)
    sample_trajectory_ids = np.asarray(
        evidence["sample_trajectory_ids"]
    ).astype(str)
    sample_pair_ids = np.asarray(evidence["sample_pair_ids"]).astype(str)
    sample_transitions = np.asarray(
        evidence["sample_transitions"], dtype=np.int64
    )
    primary_anomaly = np.asarray(
        evidence["complete_lejepa_anomaly_score"],
        dtype=np.float64,
    )
    calibration = sample_roles == "calibration"
    calibration_controls = calibration & ~np.asarray(
        evidence["sample_treatment"], dtype=np.bool_
    )
    candidate_threshold = _control_max_threshold(
        primary_anomaly[calibration],
        sample_trajectory_ids[calibration],
        calibration_controls[calibration],
    )
    raw_threshold = _control_max_threshold(
        raw_scores[calibration],
        sample_trajectory_ids[calibration],
        calibration_controls[calibration],
    )
    candidate_decisions = primary_anomaly > candidate_threshold
    raw_decisions = raw_scores > raw_threshold
    candidate_metrics = {}
    raw_metrics = {}
    for role in ("iid_evaluation", "transfer_evaluation"):
        selected = sample_roles == role
        candidate_metrics[role] = _alert_metrics(
            decisions=candidate_decisions[selected],
            trajectory_ids=sample_trajectory_ids[selected],
            transitions=sample_transitions[selected],
            labels=labels[role],
        )
        raw_metrics[role] = _alert_metrics(
            decisions=raw_decisions[selected],
            trajectory_ids=sample_trajectory_ids[selected],
            transitions=sample_transitions[selected],
            labels=labels[role],
        )
    selection = sample_roles == "selection"
    selection_pair_win_fraction = _selection_pair_win_fraction(
        anomaly=primary_anomaly[selection],
        trajectory_ids=sample_trajectory_ids[selection],
        pair_ids=sample_pair_ids[selection],
        transitions=sample_transitions[selection],
        labels=labels["selection"],
    )
    thresholds_recompute = bool(
        candidate_threshold == result["thresholds"]["candidate"]
        and raw_threshold == result["thresholds"]["raw"]
        and np.array_equal(
            candidate_decisions,
            np.asarray(evidence["candidate_decisions"]),
        )
        and np.array_equal(
            raw_decisions, np.asarray(evidence["raw_decisions"])
        )
    )
    metrics_recompute = bool(
        _canonical_json(candidate_metrics)
        == _canonical_json(result["candidate_metrics"])
        and _canonical_json(raw_metrics)
        == _canonical_json(result["raw_metrics"])
        and selection_pair_win_fraction
        == result["selection_pair_win_fraction"]
    )
    diagnostics_recompute = bool(
        _canonical_json(
            {
                name: _score_diagnostics(
                    evidence=evidence,
                    name=name,
                    roles=sample_roles,
                    treatments=np.asarray(
                        evidence["sample_treatment"], dtype=np.bool_
                    ),
                )
                for name in CELLS
            }
        )
        == _canonical_json(result["score_diagnostics"])
    )
    latency = dict(result["latency"])
    latency_samples = np.asarray(
        latency["samples_ms"], dtype=np.float64
    )
    selection_positions = np.flatnonzero(
        sample_roles == "selection"
    )
    selection_19 = selection_positions[
        sample_transitions[selection_positions] == 19
    ]
    selection_39 = selection_positions[
        sample_transitions[selection_positions] == 39
    ]
    with np.load(
        artifact / "latency-inputs.npz", allow_pickle=False
    ) as latency_inputs:
        latency_input_valid = bool(
            len(selection_19) == 20
            and len(selection_39) == 20
            and np.array_equal(
                latency_inputs["warmup_history"],
                np.asarray(evidence["sample_histories"])[
                    selection_19[:1]
                ],
            )
            and np.array_equal(
                latency_inputs["measurement_histories"],
                np.asarray(evidence["sample_histories"])[
                    selection_39
                ],
            )
            and tuple(
                str(value)
                for value in latency_inputs[
                    "measurement_trajectory_ids"
                ]
            )
            == tuple(sample_trajectory_ids[selection_39])
            and np.array_equal(
                latency_inputs["measurement_transitions"],
                sample_transitions[selection_39],
            )
        )
    latency_recomputes = bool(
        latency_input_valid
        and len(latency_samples) == 20
        and np.all(np.isfinite(latency_samples))
        and float(np.median(latency_samples))
        == latency["median_ms"]
        and float(
            np.quantile(
                latency_samples, 0.95, method="higher"
            )
        )
        == latency["p95_ms_higher"]
        and latency["torch_intraop_threads"] == 1
        and latency["torch_interop_threads"] == 1
        and latency["omp_num_threads"] == "1"
        and latency["mkl_num_threads"] == "1"
        and latency["measurement_count"] == 20
        and latency["measurement_transitions"] == [39] * 20
        and latency["measurement_trajectory_ids"]
        == sorted(latency["measurement_trajectory_ids"])
        and latency["measurement_trajectory_ids"]
        == list(sample_trajectory_ids[selection_39])
        and latency["warmup_count"] == 1
        and latency["timer"] == "time.perf_counter_ns"
        and latency["model_load_excluded"] is True
        and latency["absolute_peak_rss_bytes"]
        >= latency["baseline_rss_bytes"]
        and latency["incremental_peak_rss_bytes"]
        == max(
            0,
            latency["absolute_peak_rss_bytes"]
            - latency["baseline_rss_bytes"],
        )
    )
    bundle_recomputes = bool(
        artifact.joinpath("primary-scorer.json").stat().st_size
        == result["bundle"]["bytes"]
        and _file_sha256(artifact / "primary-scorer.json")
        == result["bundle"]["sha256"]
        and result["bundle"]["parameter_count"] == 116_848
    )
    arrays_finite = _evidence_arrays_valid(evidence)
    source_identities = bool(
        cache_valid
        and prior_valid
        and result["source"]["source_corpus_sha256"]
        == "df03af282f48591216251f22f934b97e1555147df9ed6f6c6d1f7684f9644a26"
        and result["source"]["source_artifact_manifest_sha256"]
        == "d02afa33a5977cba69b255cd1a5f31470751b6681da1ecae31b11e86307e65b1"
        and result["source"]["preprocessing_protocol"]
        == "action_conditioned_jepa_topology_transfer_v1"
        and result["source"]["cache_manifest_sha256"]
        == CACHE_MANIFEST_SHA256
        and result["source"]["cache_file_sha256"]
        == CACHE_FILE_SHA256
        and result["source"]["prior_manifest_sha256"]
        == PRIOR_MANIFEST_SHA256
        and result["source"]["source_model_file_sha256"]
        == MODEL_SHA256
        and result["source"]["source_model_payload_sha256"]
        == {
            name: _canonical_sha256(payload)
            for name, payload in source_payloads.items()
        }
    )
    protocol_checks = {
        "source_identities_recompute": source_identities,
        "role_contract_recomputes": role_contract,
        "fixed_anchors_recompute": sampling_checks["fixed_anchors"],
        "action_blind_sampling_recomputes": bool(
            sampling_checks["source_rows"]
            and receipts_recompute
            and raw_recomputes
            and model_ownership_matches
        ),
        "model_restoration_recomputes": bool(
            bundle_identity and bundle_recomputes
        ),
        "exact_score_recomputes": bool(
            exact_recomputes and diagnostics_recompute
        ),
        "batch_and_literal_parity_recompute": bool(
            literal_parity
            and all(bool(value) for value in result["batch_parity"].values())
        ),
        "latency_contract_recomputes": latency_recomputes,
        "evidence_arrays_are_finite": arrays_finite,
        "calibration_isolation_recomputes": thresholds_recompute,
        "alert_metrics_recompute": metrics_recompute,
        "evaluation_has_no_selection_authority": bool(
            result["fixed_contract"]["cells"] == list(CELLS)
            and result["fixed_contract"]["anchors"] == list(ANCHORS)
        ),
        "source_snapshots_and_manifest_verify": bool(
            snapshot_valid
            and snapshot_report_matches
            and manifest_valid
        ),
    }
    assessment = _assess_gates(
        interpretable=result.get("interpretable") is True,
        protocol_checks=protocol_checks,
        candidate_metrics=candidate_metrics,
        raw_metrics=raw_metrics,
        selection_pair_win_fraction=selection_pair_win_fraction,
        median_latency_ms=float(latency["median_ms"]),
        p95_latency_ms=float(latency["p95_ms_higher"]),
        bundle_bytes=int(result["bundle"]["bytes"]),
        parameter_count=int(result["bundle"]["parameter_count"]),
    )
    evidence.close()
    return assessment


def _literal_score_checks(
    *,
    evidence: Any,
    models: Mapping[str, CompleteLejepaRepresentation],
    graph: Any,
    visible: NDArray[np.bool_],
    present: NDArray[np.bool_],
) -> Tuple[bool, bool]:
    histories = np.asarray(evidence["sample_histories"], dtype=np.float64)
    roles = np.asarray(evidence["sample_role"]).astype(str)
    trajectories = np.asarray(
        evidence["sample_trajectory_ids"]
    ).astype(str)
    transitions = np.asarray(
        evidence["sample_transitions"], dtype=np.int64
    )
    exact = True
    parity = True
    for name, model in models.items():
        stored_scores = np.asarray(
            evidence[f"{name}_jepa_score"], dtype=np.float64
        )
        stored_singular = np.asarray(
            evidence[f"{name}_singular_values"], dtype=np.float64
        )
        stored_embeddings = np.asarray(
            evidence[f"{name}_projector_embeddings"],
            dtype=np.float64,
        )
        for role in ROLES:
            candidates = np.flatnonzero(
                (roles == role) & (transitions == 19)
            )
            position = int(
                candidates[np.argmin(trajectories[candidates])]
            )
            score, singular, unowned, embeddings = _literal_score(
                model=model,
                histories=histories[position : position + 1],
                graph=graph,
                visible=visible,
                present=present,
            )
            exact = bool(
                exact
                and np.allclose(
                    score,
                    stored_scores[position : position + 1],
                    atol=1e-3,
                    rtol=0.0,
                )
                and np.allclose(
                    singular,
                    stored_singular[position : position + 1],
                    atol=2e-5,
                    rtol=0.0,
                )
                and np.allclose(
                    embeddings,
                    stored_embeddings[position : position + 1],
                    atol=2e-6,
                    rtol=0.0,
                )
                and unowned == 0.0
            )
        candidates = np.flatnonzero(
            (roles == "selection") & (transitions == 39)
        )
        selected = candidates[
            np.argsort(trajectories[candidates])
        ][:3]
        if not np.array_equal(
            selected,
            np.asarray(evidence["parity_positions"], dtype=np.int64),
        ):
            parity = False
        (
            together_score,
            together_singular,
            unowned,
            together_embeddings,
        ) = _literal_score(
            model=model,
            histories=histories[selected],
            graph=graph,
            visible=visible,
            present=present,
        )
        separate = [
            _literal_score(
                model=model,
                histories=histories[index : index + 1],
                graph=graph,
                visible=visible,
                present=present,
            )
            for index in selected
        ]
        separate_score = np.concatenate(
            [value[0] for value in separate]
        )
        separate_singular = np.concatenate(
            [value[1] for value in separate]
        )
        separate_embeddings = np.concatenate(
            [value[3] for value in separate]
        )
        parity = bool(
            parity
            and np.allclose(
                together_score,
                separate_score,
                atol=1e-3,
                rtol=0.0,
            )
            and np.allclose(
                together_singular,
                separate_singular,
                atol=2e-5,
                rtol=0.0,
            )
            and np.allclose(
                together_score,
                evidence[f"{name}_parity_batch_scores"],
                atol=2e-4,
                rtol=0.0,
            )
            and np.allclose(
                together_singular,
                evidence[f"{name}_parity_batch_singular_values"],
                atol=2e-5,
                rtol=0.0,
            )
            and np.allclose(
                separate_score,
                evidence[f"{name}_parity_single_scores"],
                atol=2e-4,
                rtol=0.0,
            )
            and np.allclose(
                separate_singular,
                evidence[f"{name}_parity_single_singular_values"],
                atol=2e-5,
                rtol=0.0,
            )
            and np.allclose(
                together_score,
                stored_scores[selected],
                atol=1e-3,
                rtol=0.0,
            )
            and np.allclose(
                together_singular,
                stored_singular[selected],
                atol=2e-5,
                rtol=0.0,
            )
            and np.allclose(
                together_embeddings,
                separate_embeddings,
                atol=2e-6,
                rtol=0.0,
            )
            and np.allclose(
                together_embeddings,
                stored_embeddings[selected],
                atol=2e-6,
                rtol=0.0,
            )
            and unowned == 0.0
        )
    return exact, parity


def _literal_score(
    *,
    model: CompleteLejepaRepresentation,
    histories: NDArray[np.float64],
    graph: Any,
    visible: NDArray[np.bool_],
    present: NDArray[np.bool_],
) -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    float,
    NDArray[np.float64],
]:
    import torch

    fitted_graph, _, ownership, network = model._fitted_values()
    if graph.to_dict() != fitted_graph.to_dict():
        raise ValueError("literal JEPA-SCORE graph differs")
    projector = getattr(model, "_projector")
    for parameter in tuple(network.parameters()) + tuple(
        projector.parameters()
    ):
        parameter.requires_grad_(False)
    visible_tensor = torch.as_tensor(visible, dtype=torch.bool)
    present_tensor = torch.as_tensor(present, dtype=torch.bool)
    ownership_tensor = torch.as_tensor(
        ownership, dtype=torch.float32
    )
    positions = np.arange(140, dtype=np.int64)

    def forward(inputs: Any) -> Any:
        values = inputs * ownership_tensor[None, None]
        batch_visible = visible_tensor[None].expand(
            len(values), -1, -1
        )
        batch_present = present_tensor[None].expand(
            len(values), -1, -1
        )
        hidden = network(
            values, batch_visible, batch_present, positions
        )
        selected = batch_visible.reshape(len(values), -1)
        pooled = (
            hidden * selected.to(hidden.dtype).unsqueeze(-1)
        ).sum(dim=1) / selected.sum(dim=1).clamp_min(1).unsqueeze(-1)
        return projector(pooled)

    inputs = torch.as_tensor(
        histories, dtype=torch.float32
    ).requires_grad_(True)
    with torch.no_grad():
        embeddings = forward(inputs).detach()
    jacobian = torch.autograd.functional.jacobian(  # type: ignore[no-untyped-call]
        lambda values: forward(values).sum(0),
        inputs,
        create_graph=False,
        strict=False,
        vectorize=False,
    )
    matrices = jacobian.flatten(2).permute(1, 0, 2)
    singular = torch.linalg.svdvals(matrices)
    scores = singular.clamp_min(1e-6).log().sum(1)
    unowned = jacobian[:, :, :, ~torch.as_tensor(ownership)].abs()
    return (
        scores.detach().numpy().astype(np.float64),
        singular.detach().numpy().astype(np.float64),
        float(unowned.max()),
        embeddings.numpy().astype(np.float64),
    )


def _sampling_recomputes(
    *,
    evidence: Any,
    windows_by_role: Mapping[str, ActionConditionedWindows],
    original_by_role: Mapping[str, ActionConditionedWindows],
    labels: Mapping[str, Mapping[str, Tuple[bool, Optional[int]]]],
) -> Mapping[str, bool]:
    histories = np.asarray(evidence["sample_histories"])
    roles = np.asarray(evidence["sample_role"]).astype(str)
    trajectories = np.asarray(
        evidence["sample_trajectory_ids"]
    ).astype(str)
    pairs = np.asarray(evidence["sample_pair_ids"]).astype(str)
    transitions = np.asarray(
        evidence["sample_transitions"], dtype=np.int64
    )
    source_rows = np.asarray(
        evidence["sample_source_row_indices"], dtype=np.int64
    )
    treatments = np.asarray(evidence["sample_treatment"], dtype=np.bool_)
    onsets = np.asarray(evidence["sample_onset"], dtype=np.int64)
    fixed_anchors = True
    source_valid = True
    for role in ROLES:
        selected = np.flatnonzero(roles == role)
        windows = windows_by_role[role]
        original = original_by_role[role]
        expected_trajectories = sorted(set(windows.trajectory_ids))
        for trajectory_id in expected_trajectories:
            positions = selected[trajectories[selected] == trajectory_id]
            fixed_anchors = bool(
                fixed_anchors
                and len(positions) == 5
                and tuple(transitions[positions]) == ANCHORS
            )
        for position in selected:
            source = int(source_rows[position])
            treatment, onset = labels[role][trajectories[position]]
            source_valid = bool(
                source_valid
                and original.trajectory_ids[source]
                == trajectories[position]
                and original.matched_pair_ids[source] == pairs[position]
                and int(original.transition_indices[source])
                == int(transitions[position])
                and np.array_equal(
                    original.histories[source], histories[position]
                )
                and treatments[position] == treatment
                and onsets[position]
                == (-1 if onset is None else onset)
            )
    return {
        "fixed_anchors": fixed_anchors,
        "source_rows": source_valid,
    }


def _receipts_recompute(
    *,
    evidence: Any,
    windows_by_role: Mapping[str, ActionConditionedWindows],
    original_by_role: Mapping[str, ActionConditionedWindows],
    result: Mapping[str, Any],
) -> bool:
    metadata = result["fixed_contract"]["receipt_metadata"]
    for role, windows in windows_by_role.items():
        original = original_by_role[role]
        applicable_position = windows.action_feature_names.index(
            "applicable"
        )
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
        applicable = np.any(
            windows.future_actions[
                :, :, :, applicable_position
            ]
            > 0.5,
            axis=2,
        )
        prefix = f"receipt_{role}"
        if not (
            np.array_equal(
                evidence[f"{prefix}_trajectory_ids"],
                np.asarray(windows.trajectory_ids),
            )
            and np.array_equal(
                evidence[f"{prefix}_pair_ids"],
                np.asarray(windows.matched_pair_ids),
            )
            and np.array_equal(
                evidence[f"{prefix}_transitions"],
                windows.transition_indices,
            )
            and np.array_equal(
                evidence[f"{prefix}_source_row_indices"], source_rows
            )
            and np.array_equal(
                evidence[f"{prefix}_applicable"], applicable
            )
            and metadata[role]["rows"] == len(windows.histories)
            and metadata[role]["source_row_sha256"]
            == _array_sha256(source_rows)
            and metadata[role]["applicable_sha256"]
            == _array_sha256(applicable)
        ):
            return False
    return True


def _fit_raw_comparator(
    windows: ActionConditionedWindows,
) -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.bool_],
    Tuple[str, ...],
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
        if not np.array_equal(
            windows.transition_indices[order], np.arange(19, 98)
        ):
            raise ValueError("raw fit-control rows differ")
        deltas.append(
            windows.histories[order, -1]
            - windows.histories[order, -2]
        )
    combined = np.concatenate(deltas)
    if len(combined) != 3160:
        raise ValueError("raw fit population differs")
    ownership = _literal_owned_feature_mask(windows)
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
    return (
        np.where(ownership, center, 0.0),
        np.where(ownership, scale, 1.0),
        ownership,
        controls,
    )


def _literal_owned_feature_mask(
    windows: ActionConditionedWindows,
) -> NDArray[np.bool_]:
    entity_positions = {
        entity_id: position
        for position, entity_id in enumerate(windows.entity_names)
    }
    feature_positions = {
        name: position
        for position, name in enumerate(windows.state_feature_names)
    }
    mask = np.zeros(
        (len(windows.entity_names), len(windows.state_feature_names)),
        dtype=np.bool_,
    )
    for feature_key, entity_id in windows.graph.binding_map().items():
        feature_name = feature_key.split(".", 1)[-1]
        if (
            entity_id in entity_positions
            and feature_name in feature_positions
        ):
            mask[
                entity_positions[entity_id],
                feature_positions[feature_name],
            ] = True
    mask |= np.ptp(windows.histories, axis=(0, 1)) > 1e-9
    if not np.any(mask):
        raise ValueError("literal ownership has no observations")
    return mask


def _raw_scores(
    histories: NDArray[np.float64],
    *,
    center: NDArray[np.float64],
    scale: NDArray[np.float64],
    ownership: NDArray[np.bool_],
) -> NDArray[np.float64]:
    standardized = (
        histories[:, -1] - histories[:, -2] - center[None]
    ) / scale[None]
    return cast(
        NDArray[np.float64],
        np.sqrt(
            np.mean(np.square(standardized[:, ownership]), axis=1)
        ),
    )


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
            offsets = np.flatnonzero(
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
                for offset in offsets
            )
        result[trajectory_id] = (
            bool(onsets),
            min(onsets) if onsets else None,
        )
    return result


def _control_max_threshold(
    scores: NDArray[np.float64],
    trajectory_ids: NDArray[np.str_],
    control_rows: NDArray[np.bool_],
) -> float:
    values = [
        float(np.max(scores[trajectory_ids == trajectory_id]))
        for trajectory_id in sorted(
            set(str(value) for value in trajectory_ids[control_rows])
        )
    ]
    if len(values) != 10:
        raise ValueError("calibration control count differs")
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
        positions = np.flatnonzero(
            (pair_ids == pair_id) & (transitions == 39)
        )
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
        if (
            len(positions) != 2
            or len(treatment) != 1
            or len(control) != 1
        ):
            raise ValueError("selection pair labels do not align")
        wins.append(anomaly[treatment[0]] > anomaly[control[0]])
    return float(np.mean(wins))


def _assess_gates(
    *,
    interpretable: bool,
    protocol_checks: Mapping[str, bool],
    candidate_metrics: Mapping[str, Mapping[str, Any]],
    raw_metrics: Mapping[str, Mapping[str, Any]],
    selection_pair_win_fraction: float,
    median_latency_ms: float,
    p95_latency_ms: float,
    bundle_bytes: int,
    parameter_count: int,
) -> Mapping[str, Any]:
    protocol = {name: bool(value) for name, value in protocol_checks.items()}
    iid = candidate_metrics["iid_evaluation"]
    transfer = candidate_metrics["transfer_evaluation"]
    raw_transfer = raw_metrics["transfer_evaluation"]
    edge = {
        "median_latency_at_most_100_ms": (
            0.0 < median_latency_ms <= 100.0
        ),
        "p95_latency_at_most_125_ms": (
            0.0 < p95_latency_ms <= 125.0
        ),
        "bundle_at_most_8_mib": 0 < bundle_bytes <= 8 * 1024 * 1024,
        "parameters_at_most_120000": (
            0 < parameter_count <= 120_000
        ),
        "iid_control_false_alarm_at_most_0_05": (
            iid["control_trajectory_false_alarm_rate"] <= 0.05
        ),
        "transfer_control_false_alarm_at_most_0_05": (
            transfer["control_trajectory_false_alarm_rate"] <= 0.05
        ),
        "iid_pre_onset_alert_at_most_0_05": (
            iid["treatment_pre_onset_alert_rate"] <= 0.05
        ),
        "transfer_pre_onset_alert_at_most_0_05": (
            transfer["treatment_pre_onset_alert_rate"] <= 0.05
        ),
    }
    no_worse, material = _pareto(transfer, raw_transfer)
    value = {
        "selection_pair_win_fraction_at_least_0_60": (
            selection_pair_win_fraction >= 0.60
        ),
        "iid_detection_at_least_0_80": (
            iid["treatment_detection_rate"] >= 0.80
        ),
        "transfer_detection_at_least_0_80": (
            transfer["treatment_detection_rate"] >= 0.80
        ),
        "transfer_pareto_no_worse_than_raw": no_worse,
        "transfer_materially_improves_raw": material,
    }
    protocol_passed = all(protocol.values())
    edge_passed = all(edge.values())
    value_passed = all(value.values())
    scientific_gates_passed = (
        protocol_passed and edge_passed and value_passed
    )
    passed = interpretable and scientific_gates_passed
    return {
        "protocol_gates": protocol,
        "edge_safety_gates": edge,
        "value_gates": value,
        "protocol_passed": protocol_passed,
        "edge_safety_passed": edge_passed,
        "value_passed": value_passed,
        "interpretable": interpretable,
        "scientific_gates_passed": scientific_gates_passed,
        "passed": passed,
        "selection_pair_win_fraction": selection_pair_win_fraction,
        "decision": (
            "non_interpretable_jepa_score_smoke"
            if not interpretable
            else (
                "advance_exact_jepa_score_to_fixed_seed_robustness"
                if passed
                else "reject_exact_jepa_score_edge_alert_recipe"
            )
        ),
    }


def _pareto(
    candidate: Mapping[str, Any], raw: Mapping[str, Any]
) -> Tuple[bool, bool]:
    candidate_false_alarm = float(
        candidate["control_trajectory_false_alarm_rate"]
    )
    raw_false_alarm = float(raw["control_trajectory_false_alarm_rate"])
    candidate_detection = float(candidate["treatment_detection_rate"])
    raw_detection = float(raw["treatment_detection_rate"])
    candidate_delay = candidate["median_post_onset_delay_transitions"]
    raw_delay = raw["median_post_onset_delay_transitions"]
    if candidate_delay is None and raw_delay is None:
        delay_no_worse, delay_improvement = True, False
    elif candidate_delay is None:
        delay_no_worse, delay_improvement = False, False
    elif raw_delay is None:
        delay_no_worse, delay_improvement = True, False
    else:
        delay_no_worse = float(candidate_delay) <= float(raw_delay)
        delay_improvement = (
            float(raw_delay) - float(candidate_delay) >= 20.0
        )
    no_worse = (
        candidate_false_alarm <= raw_false_alarm
        and candidate_detection >= raw_detection
        and delay_no_worse
    )
    material = no_worse and (
        raw_false_alarm - candidate_false_alarm >= 0.05
        or candidate_detection - raw_detection >= 0.05
        or delay_improvement
    )
    return no_worse, material


def _role_contract_recomputes(
    windows: Mapping[str, ActionConditionedWindows],
) -> bool:
    expected = {
        "fit": (40, 80),
        "selection": (10, 20),
        "calibration": (10, 20),
        "iid_evaluation": (20, 40),
        "transfer_evaluation": (10, 20),
    }
    pair_sets = []
    trajectory_sets = []
    for role, (pairs, trajectories) in expected.items():
        current = windows[role]
        pair_set = set(current.matched_pair_ids)
        trajectory_set = set(current.trajectory_ids)
        if len(pair_set) != pairs or len(trajectory_set) != trajectories:
            return False
        labels = _trajectory_labels(current)
        for pair_id in pair_set:
            pair_trajectories = {
                trajectory_id
                for trajectory_id, candidate_pair in zip(
                    current.trajectory_ids,
                    current.matched_pair_ids,
                )
                if candidate_pair == pair_id
            }
            if (
                len(pair_trajectories) != 2
                or sorted(
                    labels[trajectory_id][0]
                    for trajectory_id in pair_trajectories
                )
                != [False, True]
            ):
                return False
        pair_sets.append(pair_set)
        trajectory_sets.append(trajectory_set)
    return all(
        not pair_sets[left] & pair_sets[right]
        and not trajectory_sets[left] & trajectory_sets[right]
        for left in range(len(pair_sets))
        for right in range(left + 1, len(pair_sets))
    )


def _evidence_arrays_valid(evidence: Any) -> bool:
    for name in evidence.files:
        array = np.asarray(evidence[name])
        if np.issubdtype(array.dtype, np.number) and not np.all(
            np.isfinite(array)
        ):
            return False
    row_count = len(evidence["sample_histories"])
    for name in CELLS:
        singular = np.asarray(
            evidence[f"{name}_singular_values"], dtype=np.float64
        )
        score = np.asarray(
            evidence[f"{name}_jepa_score"], dtype=np.float64
        )
        anomaly = np.asarray(
            evidence[f"{name}_anomaly_score"], dtype=np.float64
        )
        clipped = np.asarray(
            evidence[f"{name}_clipped_count"], dtype=np.int64
        )
        embeddings = np.asarray(
            evidence[f"{name}_projector_embeddings"],
            dtype=np.float64,
        )
        literal = (
            np.log(
                np.maximum(
                    singular.astype(np.float32),
                    np.float32(EPSILON),
                )
            )
            .sum(axis=1, dtype=np.float32)
            .astype(np.float64)
        )
        if not (
            singular.shape == (row_count, 64)
            and score.shape == (row_count,)
            and anomaly.shape == (row_count,)
            and clipped.shape == (row_count,)
            and embeddings.shape == (row_count, 64)
            and np.all(singular >= 0.0)
            and np.allclose(score, literal, atol=2e-4, rtol=0.0)
            and np.array_equal(anomaly, -score)
            and np.array_equal(
                clipped, np.sum(singular < EPSILON, axis=1)
            )
            and np.all(
                np.asarray(
                    evidence[f"{name}_unowned_jacobian_max_abs"]
                )
                == 0.0
            )
        ):
            return False
    return True


def _score_diagnostics(
    *,
    evidence: Any,
    name: str,
    roles: NDArray[np.str_],
    treatments: NDArray[np.bool_],
) -> Mapping[str, Any]:
    embeddings = np.asarray(
        evidence[f"{name}_projector_embeddings"],
        dtype=np.float64,
    )
    scores = np.asarray(
        evidence[f"{name}_jepa_score"], dtype=np.float64
    )
    centered = embeddings - embeddings.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    probabilities = singular / max(float(np.sum(singular)), 1e-12)
    nonzero = probabilities[probabilities > 0.0]
    covariance = np.cov(embeddings, rowvar=False)
    off_diagonal = covariance[
        ~np.eye(covariance.shape[0], dtype=np.bool_)
    ]
    distributions = {}
    for role in ROLES:
        for arm_name, treatment in (
            ("control", False),
            ("treatment", True),
        ):
            selected = (roles == role) & (treatments == treatment)
            values = scores[selected]
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
            np.sum(evidence[f"{name}_clipped_count"])
        ),
        "unowned_jacobian_max_abs": float(
            np.max(evidence[f"{name}_unowned_jacobian_max_abs"])
        ),
        "score_distributions": distributions,
    }


def _verify_cache(cache: Path) -> bool:
    if _file_sha256(
        cache / "artifact-manifest.json"
    ) != CACHE_MANIFEST_SHA256:
        return False
    manifest = _read_json(cache / "artifact-manifest.json")
    if dict(manifest.get("sha256", {})) != CACHE_FILE_SHA256:
        return False
    return all(
        _file_sha256(cache / name) == expected
        for name, expected in CACHE_FILE_SHA256.items()
    )


def _verify_prior(artifact: Path, prior: Path) -> bool:
    if _file_sha256(
        prior / "artifact-manifest.json"
    ) != PRIOR_MANIFEST_SHA256:
        return False
    return all(
        _file_sha256(prior / "models" / f"{name}.json") == expected
        and _file_sha256(artifact / "models" / f"{name}.json")
        == expected
        for name, expected in MODEL_SHA256.items()
    )


def _verify_artifact_manifest(artifact: Path) -> bool:
    manifest = _read_json(artifact / "artifact-manifest.json")
    actual = {
        str(path.relative_to(artifact)): _file_sha256(path)
        for path in sorted(artifact.rglob("*"))
        if path.is_file() and path.name != "artifact-manifest.json"
    }
    return dict(manifest.get("sha256", {})) == actual


def _verify_source_snapshot(artifact: Path) -> bool:
    root = artifact / "reproduction-source"
    declared = _read_json(root / "source-sha256.json")
    actual = {
        str(path.relative_to(root)): _file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "source-sha256.json"
    }
    return dict(declared) == actual


def _strict_model_payload(
    payload: Mapping[str, Any]
) -> Mapping[str, Any]:
    config = dict(payload["config"])
    return {
        "graph": payload["graph"],
        "feature_names": payload["feature_names"],
        "ownership_mask": payload["ownership_mask"],
        "network_state": payload["network_state"],
        "projector_state": payload["projector_state"],
        "inference_config": {
            key: config[key]
            for key in (
                "width",
                "block_count",
                "head_count",
                "feedforward_width",
                "projector_width",
                "preprocessing_protocol",
                "view_seed",
            )
        },
    }


def _restore_without_rng_drift(
    payload: Mapping[str, Any]
) -> CompleteLejepaRepresentation:
    import torch

    state = torch.random.get_rng_state()
    try:
        return CompleteLejepaRepresentation.from_dict(payload)
    finally:
        torch.random.set_rng_state(state)


def _view_sha256(
    visible: NDArray[np.bool_], present: NDArray[np.bool_]
) -> str:
    return _canonical_sha256(
        {
            "visible_tokens": visible.astype(int).tolist(),
            "present_tokens": present.astype(int).tolist(),
        }
    )


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(
        str(array.dtype).encode()
        + str(array.shape).encode()
        + array.tobytes()
    ).hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main(arguments: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    options = parser.parse_args(arguments)
    assessment = assess_artifact(
        artifact=options.artifact,
        cache=options.cache,
        prior=options.prior,
    )
    print(_canonical_json(assessment))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
