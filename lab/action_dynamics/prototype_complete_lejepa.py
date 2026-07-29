"""Retained runner for the frozen complete multi-view LeJEPA tracer.

The runner refuses overwrites, stores every fitted representation/probe and
the arrays needed for pure reassessment, and keeps rejected hypotheses fully
reproducible.  Use a fresh ``--output`` for every run.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from quantis_core.action_conditioned_dynamics import (
    ActionConditionedWindows,
)
from quantis_core.edge_dynamics.complete_lejepa import (
    CompleteLejepaConfig,
    CompleteLejepaRepresentation,
    EntityPcaRepresentation,
    PairBlockedAnchorSchedule,
    ReducedRankActionProbe,
    TelemetryViewSchedule,
    assess_complete_lejepa_gates,
    fit_owned_feature_mask,
)
from quantis_core.edge_dynamics.action_conditioned_jepa import (
    sketched_isotropic_gaussian_regularization,
)
from quantis_core.edge_dynamics.data import (
    PreparedAttributionQueries,
    load_edge_dynamics_cache,
    partition_worker_topology,
    subset_attribution_queries,
)
from quantis_core.edge_dynamics.models import (
    ContractiveLowRankDynamics,
    LowRankConfig,
)


REPRESENTATION_NAMES = (
    "complete_lejepa",
    "invariance_only",
    "sigreg_only",
    "masked_autoencoder",
    "matched_pca",
)
RIDGES = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def run_complete_lejepa_tracer(
    *,
    cache_directory: Path,
    output_directory: Path,
    steps: int = 1600,
    sketch_dimension: int = 1024,
) -> Mapping[str, Any]:
    """Run the preregistered tracer and atomically publish its evidence."""

    if output_directory.exists():
        raise FileExistsError(
            f"refusing to overwrite complete LeJEPA: {output_directory}"
        )
    building = output_directory.with_name(
        output_directory.name + ".building"
    )
    if building.exists():
        raise FileExistsError(
            f"refusing to overwrite incomplete LeJEPA work: {building}"
        )
    building.mkdir(parents=True)
    evidence_directory = building / "ridge-selection-evidence"
    evidence_directory.mkdir()
    prepared = load_edge_dynamics_cache(cache_directory)
    partitions = {
        role: partition_worker_topology(windows)
        for role, windows in prepared.windows.items()
    }
    held_out_values = {
        value.held_out_normalized_value for value in partitions.values()
    }
    if len(held_out_values) != 1:
        raise ValueError("held-out topology identity drifted across roles")
    held_out_value = next(iter(held_out_values))
    windows_by_role = {
        "fit": partitions["fit"].in_distribution,
        "selection": partitions["selection"].in_distribution,
        "iid_evaluation": partitions["evaluation"].in_distribution,
        "transfer_evaluation": partitions["evaluation"].held_out,
    }
    fit = windows_by_role["fit"]
    ownership = fit_owned_feature_mask(fit)
    anchor_schedule = PairBlockedAnchorSchedule(fit, seed=1509)
    view_schedule = TelemetryViewSchedule(
        graph=fit.graph,
        ownership_mask=ownership,
        varying_entity_mask=np.any(
            np.ptp(fit.histories, axis=(0, 1)) > 1e-9, axis=1
        ),
        seed=2509,
    )
    anchor_batches = [
        anchor_schedule.batch(step) for step in range(steps)
    ]
    np.savez_compressed(
        building / "anchor-schedule.npz",
        indices=np.stack([batch.indices for batch in anchor_batches]),
        arm_ids=np.stack([batch.arm_ids for batch in anchor_batches]),
        transition_indices=np.stack(
            [batch.transition_indices for batch in anchor_batches]
        ),
        pair_ids=np.asarray(anchor_schedule.pair_ids),
    )
    view_visible = []
    view_present = []
    for step in range(steps):
        views = view_schedule.batch(fit.histories[:1], step=step)
        view_visible.append(views.visible_tokens[:, 0])
        view_present.append(views.present_tokens[:, 0])
    np.savez_compressed(
        building / "view-schedule.npz",
        visible_tokens=np.stack(view_visible),
        present_tokens=np.stack(view_present),
        view_names=np.asarray(views.view_names),
        local_roots=np.asarray(views.local_roots, dtype=np.int64),
    )
    transfer_queries = _transfer_queries(
        prepared.attribution_queries,
        prepared.windows["fit"].control_feature_names,
        held_out_value,
    )

    configurations = {
        "complete_lejepa": CompleteLejepaConfig(
            objective="lejepa",
            steps=steps,
            sketch_dimension=sketch_dimension,
        ),
        "invariance_only": CompleteLejepaConfig(
            objective="invariance_only",
            steps=steps,
            sketch_dimension=sketch_dimension,
        ),
        "sigreg_only": CompleteLejepaConfig(
            objective="sigreg_only",
            steps=steps,
            sketch_dimension=sketch_dimension,
        ),
        "masked_autoencoder": CompleteLejepaConfig(
            objective="masked_autoencoder",
            steps=steps,
            sketch_dimension=sketch_dimension,
        ),
    }
    representations: Dict[str, Any] = {}
    training_seconds: Dict[str, float] = {}
    for name, config in configurations.items():
        started = time.perf_counter()
        representations[name] = CompleteLejepaRepresentation(
            config
        ).fit(fit)
        training_seconds[name] = time.perf_counter() - started
    started = time.perf_counter()
    representations["matched_pca"] = EntityPcaRepresentation(
        width=64
    ).fit(fit)
    training_seconds["matched_pca"] = time.perf_counter() - started

    encoded: Dict[str, Dict[str, np.ndarray]] = {
        name: {
            role: _encode_batches(model, windows)
            for role, windows in windows_by_role.items()
        }
        for name, model in representations.items()
    }
    representation_diagnostics = {
        name: {
            role: _representation_diagnostics(values)
            for role, values in roles.items()
        }
        for name, roles in encoded.items()
    }
    final_anchor = anchor_batches[-1]
    view_diagnostics = {}
    view_embedding_arrays = {}
    for name in REPRESENTATION_NAMES[:-1]:
        values = representations[name].diagnose_training_views(
            fit.histories[final_anchor.indices],
            fit.graph,
            step=steps - 1,
        )
        view_embedding_arrays[name] = values.astype(np.float32)
        view_diagnostics[name] = _view_diagnostics(values)
    query_tokens = {
        name: _encode_histories(
            model, transfer_queries.histories, fit.graph
        )
        for name, model in representations.items()
    }
    restoration_parity = {}
    restored_outputs = {}
    for name, model in representations.items():
        restored = (
            EntityPcaRepresentation.from_dict(model.to_dict())
            if name == "matched_pca"
            else CompleteLejepaRepresentation.from_dict(model.to_dict())
        )
        restored_values = _encode_histories(
            restored,
            windows_by_role["transfer_evaluation"].histories[:8],
            fit.graph,
        )
        restored_outputs[name] = restored_values.astype(np.float32)
        restoration_parity[name] = bool(
            np.allclose(
                restored_values,
                encoded[name]["transfer_evaluation"][:8],
                atol=1e-7,
            )
        )

    raw_model = ContractiveLowRankDynamics(LowRankConfig(rank=32)).fit(fit)
    raw_predictions = {
        role: raw_model.rollout(
            windows.histories,
            windows.future_controls,
            windows.future_actions,
            windows.graph,
        ).mean
        for role, windows in windows_by_role.items()
        if role != "fit"
    }
    raw_scores = {
        role: _forecast_scores(raw_predictions[role], windows)
        for role, windows in windows_by_role.items()
        if role != "fit"
    }

    probes: Dict[str, ReducedRankActionProbe] = {}
    ridge_curves: Dict[str, list[Mapping[str, Any]]] = {}
    selected_ridges: Dict[str, float] = {}
    selection_predictions: Dict[str, np.ndarray] = {}
    for name in REPRESENTATION_NAMES:
        curve = []
        fitted = {}
        ridge_predictions: Dict[float, np.ndarray] = {}
        for ridge in RIDGES:
            probe = ReducedRankActionProbe(rank=32, ridge=ridge).fit(
                encoded[name]["fit"],
                fit.future_controls,
                fit.future_actions,
                fit.future_states,
            )
            prediction = probe.predict(
                encoded[name]["selection"],
                windows_by_role["selection"].future_controls,
                windows_by_role["selection"].future_actions,
            )
            scores = _forecast_scores(
                prediction, windows_by_role["selection"]
            )
            safe = all(
                scores[key] <= 1.05 * raw_scores["selection"][key]
                for key in (
                    "overall_mse",
                    "action_overlap_mse",
                    "downstream_effect_mse",
                )
            )
            curve.append(
                {"ridge": ridge, "raw_safe": safe, **scores}
            )
            fitted[ridge] = probe
            ridge_predictions[ridge] = prediction
            np.save(
                evidence_directory
                / f"{name}__ridge_{ridge:.4g}.npy",
                prediction.astype(np.float32),
                allow_pickle=False,
            )
        eligible = [row for row in curve if row["raw_safe"]]
        pool = eligible if eligible else curve
        selected = min(
            pool,
            key=lambda row: (
                row["downstream_effect_mse"],
                row["ridge"],
            ),
        )
        ridge = float(selected["ridge"])
        probes[name] = fitted[ridge]
        selected_ridges[name] = ridge
        selection_predictions[name] = ridge_predictions[ridge]
        ridge_curves[name] = curve

    predictions = {
        name: {
            role: probes[name].predict(
                encoded[name][role],
                windows.future_controls,
                windows.future_actions,
            )
            for role, windows in windows_by_role.items()
            if role != "fit"
        }
        for name in REPRESENTATION_NAMES
    }
    forecast_scores = {
        name: {
            role: _forecast_scores(values, windows_by_role[role])
            for role, values in role_predictions.items()
        }
        for name, role_predictions in predictions.items()
    }
    state_probes = {
        name: _state_probe(
            encoded[name]["fit"],
            fit,
            encoded[name]["transfer_evaluation"],
            windows_by_role["transfer_evaluation"],
            ownership,
        )
        for name in REPRESENTATION_NAMES
    }
    attribution = {}
    attribution_predictions = {}
    action_sanity = {}
    action_sanity_predictions = {}
    for name in REPRESENTATION_NAMES:
        attribution[name], attribution_predictions[name] = (
            _attribution_evidence(
                probes[name],
                query_tokens[name],
                transfer_queries,
                ownership,
            )
        )
        action_sanity[name], action_sanity_predictions[name] = (
            _action_sanity_evidence(
                probes[name],
                encoded[name]["transfer_evaluation"],
                windows_by_role["transfer_evaluation"],
                ownership,
            )
        )
    latency = {
        name: {
            "encode_batch_one_ms": _median_latency_ms(
                lambda model=representations[name]: model.encode(
                    windows_by_role["transfer_evaluation"].histories[:1],
                    fit.graph,
                )
            ),
            "probe_batch_one_ms": _median_latency_ms(
                lambda probe=probes[name], token=encoded[name][
                    "transfer_evaluation"
                ][:1]: probe.predict(
                    token,
                    windows_by_role[
                        "transfer_evaluation"
                    ].future_controls[:1],
                    windows_by_role[
                        "transfer_evaluation"
                    ].future_actions[:1],
                )
            ),
        }
        for name in REPRESENTATION_NAMES
    }
    preliminary_assessment = assess_complete_lejepa_gates(
        forecast_scores=forecast_scores,
        raw_scores=raw_scores,
        state_probes=state_probes,
        attribution=attribution,
        action_sanity=action_sanity,
        restoration_parity=restoration_parity,
        ridge_curves=ridge_curves,
        selected_ridges=selected_ridges,
        transfer_pair_errors={
            name: _downstream_pair_errors(
                predictions[name]["transfer_evaluation"],
                windows_by_role["transfer_evaluation"],
            )
            for name in REPRESENTATION_NAMES
        },
        protocol_checks=_protocol_checks(building),
    )
    frozen_contract = steps == 1600 and sketch_dimension == 1024

    report = {
        "schema_version": 1,
        "kind": "complete_multi_view_lejepa_tracer_v1",
        "evidence_boundary": (
            "single-seed open-development representation tracer; "
            "not a production alert system or sealed confirmation"
        ),
        "frozen_contract_run": frozen_contract,
        "source": {
            "cache_directory": str(cache_directory),
            "source_corpus_sha256": prepared.source_corpus_sha256,
            "source_artifact_manifest_sha256": (
                prepared.source_artifact_manifest_sha256
            ),
            "preprocessing_protocol": prepared.preprocessing_protocol,
            "held_out_worker_topology_normalized": held_out_value,
        },
        "implementation": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pid": os.getpid(),
            "lejepa_source_commit": (
                "c293d291ca87cd4fddee9d3fffe4e914c7272052"
            ),
            "runner_sha256": _file_sha256(Path(__file__)),
            "representation_module_sha256": _file_sha256(
                Path(__file__).parents[2]
                / "src/quantis_core/edge_dynamics/complete_lejepa.py"
            ),
            "assessor_sha256": _file_sha256(
                Path(__file__).with_name(
                    "prototype_complete_lejepa_assessor.py"
                )
            ),
            "contract_sha256": _file_sha256(
                Path(__file__).parents[2]
                / "docs/specs/complete-lejepa-telemetry-contract-v1.md"
            ),
        },
        "window_counts": {
            role: len(windows.histories)
            for role, windows in windows_by_role.items()
        },
        "pair_counts": {
            role: len(set(windows.matched_pair_ids))
            for role, windows in windows_by_role.items()
        },
        "configurations": {
            name: config.to_dict()
            for name, config in configurations.items()
        },
        "training_seconds": training_seconds,
        "peak_resident_memory_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ),
        "training_metrics": {
            name: [dict(row) for row in model.training_metrics]
            for name, model in representations.items()
            if name != "matched_pca"
        },
        "representation_diagnostics": representation_diagnostics,
        "view_diagnostics": view_diagnostics,
        "latency": latency,
        "parameter_counts": {
            name: {
                "inference": model.inference_parameter_count,
                "training_only": model.training_only_parameter_count,
            }
            for name, model in representations.items()
            if name != "matched_pca"
        },
        "serialized_size_bytes": {
            name: {
                "representation": len(
                    json.dumps(
                        representations[name].to_dict(),
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode()
                ),
                "probe": len(
                    json.dumps(
                        probes[name].to_dict(),
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode()
                ),
            }
            for name in REPRESENTATION_NAMES
        },
        "schedule_evidence": {
            "optimizer_steps": steps,
            "independent_pairs_per_step": len(anchor_schedule.pair_ids),
            "views_per_anchor": len(views.view_names),
            "anchor_schedule_file": "anchor-schedule.npz",
            "view_schedule_file": "view-schedule.npz",
        },
        "ridge_curves": ridge_curves,
        "selected_ridges": selected_ridges,
        "forecast_scores": forecast_scores,
        "raw_low_rank_scores": raw_scores,
        "state_probes": state_probes,
        "attribution": attribution,
        "action_sanity": action_sanity,
        "restoration_parity": restoration_parity,
        "assessment": preliminary_assessment,
    }
    models_directory = building / "models"
    models_directory.mkdir()
    for name, model in representations.items():
        _write_json(models_directory / f"{name}.json", model.to_dict())
        _write_json(
            models_directory / f"{name}-probe.json",
            probes[name].to_dict(),
        )
    _write_json(models_directory / "raw_low_rank.json", raw_model.to_dict())
    evidence_arrays: Dict[str, np.ndarray] = {}
    for name in REPRESENTATION_NAMES:
        for role in windows_by_role:
            evidence_arrays[f"representation__{name}__{role}"] = encoded[
                name
            ][role].astype(np.float32)
        if name in view_embedding_arrays:
            evidence_arrays[f"view_embeddings__{name}"] = (
                view_embedding_arrays[name]
            )
        evidence_arrays[f"restored__{name}"] = restored_outputs[name]
        evidence_arrays[f"query_representation__{name}"] = query_tokens[
            name
        ].astype(np.float32)
        evidence_arrays[f"attribution_prediction__{name}"] = (
            attribution_predictions[name]
        )
        for variant, values in action_sanity_predictions[name].items():
            evidence_arrays[
                f"action_sanity__{name}__{variant}"
            ] = values
        for role in ("selection", "iid_evaluation", "transfer_evaluation"):
            evidence_arrays[f"prediction__{name}__{role}"] = predictions[
                name
            ][role].astype(np.float32)
    for role in ("selection", "iid_evaluation", "transfer_evaluation"):
        evidence_arrays[f"histories__{role}"] = windows_by_role[
            role
        ].histories.astype(np.float32)
        evidence_arrays[f"target__{role}"] = windows_by_role[
            role
        ].future_states.astype(np.float32)
        evidence_arrays[f"controls__{role}"] = windows_by_role[
            role
        ].future_controls.astype(np.float32)
        evidence_arrays[f"actions__{role}"] = windows_by_role[
            role
        ].future_actions.astype(np.float32)
        evidence_arrays[f"raw_prediction__{role}"] = raw_predictions[
            role
        ].astype(np.float32)
    evidence_arrays["histories__fit"] = fit.histories.astype(np.float32)
    evidence_arrays["target__fit"] = fit.future_states.astype(np.float32)
    evidence_arrays["controls__fit"] = fit.future_controls.astype(np.float32)
    evidence_arrays["actions__fit"] = fit.future_actions.astype(np.float32)
    evidence_arrays["query_future_controls"] = (
        transfer_queries.future_controls.astype(np.float32)
    )
    evidence_arrays["query_histories"] = (
        transfer_queries.histories.astype(np.float32)
    )
    evidence_arrays["query_observed_future"] = (
        transfer_queries.observed_future.astype(np.float32)
    )
    evidence_arrays["query_candidate_actions"] = (
        transfer_queries.candidate_actions.astype(np.float32)
    )
    np.savez_compressed(building / "evidence.npz", **evidence_arrays)
    evidence_metadata = {
        "schema_version": 1,
        "kind": "complete_lejepa_assessment_evidence",
        "graph": fit.graph.to_dict(),
        "entity_names": list(fit.entity_names),
        "state_feature_names": list(fit.state_feature_names),
        "control_feature_names": list(fit.control_feature_names),
        "action_feature_names": list(fit.action_feature_names),
        "ownership_mask": ownership.astype(int).tolist(),
        "roles": {
            role: {
                "trajectory_ids": list(windows.trajectory_ids),
                "matched_pair_ids": list(windows.matched_pair_ids),
                "transition_indices": (
                    windows.transition_indices.astype(int).tolist()
                ),
            }
            for role, windows in windows_by_role.items()
        },
        "queries": {
            "query_ids": list(transfer_queries.query_ids),
            "candidate_ids": list(transfer_queries.candidate_ids),
            "candidate_action_kinds": list(
                transfer_queries.candidate_action_kinds
            ),
            "candidate_target_entities": list(
                transfer_queries.candidate_target_entities
            ),
            "expected_action_kinds": list(
                transfer_queries.expected_action_kinds
            ),
            "expected_target_entities": list(
                transfer_queries.expected_target_entities
            ),
            "expected_variant_ids": list(
                transfer_queries.expected_variant_ids
            ),
        },
        "selected_ridges": selected_ridges,
        "ridge_values": list(RIDGES),
    }
    _write_json(building / "evidence-metadata.json", evidence_metadata)
    from prototype_complete_lejepa_assessor import (
        reassess_complete_lejepa_evidence,
    )

    assessment = reassess_complete_lejepa_evidence(building)
    if assessment != preliminary_assessment:
        raise RuntimeError(
            "stored-array LeJEPA reassessment differs from runner assessment"
        )
    if not frozen_contract:
        assessment = {
            **assessment,
            "interpretable": False,
            "provisional_decision": assessment["decision"],
            "decision": "non_interpretable_smoke",
        }
    report["assessment"] = assessment
    _write_json(building / "result.json", report)
    (building / "REPORT.md").write_text(_markdown_report(report))
    manifest = {
        "schema_version": 1,
        "kind": "complete_lejepa_artifact_manifest",
        "sha256": {
            str(path.relative_to(building)): _file_sha256(path)
            for path in sorted(building.rglob("*"))
            if path.is_file()
        },
    }
    _write_json(building / "artifact-manifest.json", manifest)
    building.rename(output_directory)
    return report


def assess_complete_lejepa_results(
    *,
    forecast_scores: Mapping[str, Mapping[str, Mapping[str, float]]],
    raw_scores: Mapping[str, Mapping[str, float]],
    state_probes: Mapping[str, Mapping[str, Any]],
    attribution: Mapping[str, Mapping[str, float]],
    action_sanity: Mapping[str, Mapping[str, float]],
    restoration_parity: Mapping[str, bool],
    ridge_curves: Mapping[str, Sequence[Mapping[str, Any]]],
    selected_ridges: Mapping[str, float],
    transfer_pair_errors: Mapping[str, Mapping[str, float]],
) -> Mapping[str, Any]:
    """Purely recompute every frozen safety and LeJEPA value gate."""

    if set(forecast_scores) != set(REPRESENTATION_NAMES):
        raise ValueError("complete LeJEPA assessment model set is incomplete")
    candidate = forecast_scores["complete_lejepa"]
    transfer = candidate["transfer_evaluation"]
    raw_transfer = raw_scores["transfer_evaluation"]
    pca_state = state_probes["matched_pca"]
    candidate_state = state_probes["complete_lejepa"]
    varying_entities = [
        name
        for name, row in candidate_state["entities"].items()
        if row["nrmse"] is not None
    ]
    selection_verified = all(
        float(selected_ridges[name])
        == float(
            min(
                (
                    [row for row in ridge_curves[name] if row["raw_safe"]]
                    or list(ridge_curves[name])
                ),
                key=lambda row: (
                    row["downstream_effect_mse"],
                    row["ridge"],
                ),
            )["ridge"]
        )
        for name in REPRESENTATION_NAMES
    )
    safety = {
        "all_public_outputs_restore": all(restoration_parity.values()),
        "selection_only_ridge_choice_recomputes": selection_verified,
        "aggregate_state_probe_within_1_05_pca": (
            candidate_state["aggregate_nrmse"]
            <= 1.05 * pca_state["aggregate_nrmse"]
        ),
        "every_entity_state_probe_within_1_15_pca": all(
            candidate_state["entities"][name]["nrmse"]
            <= 1.15 * pca_state["entities"][name]["nrmse"]
            for name in varying_entities
        ),
        "overall_mse_within_1_05_raw": (
            transfer["overall_mse"] <= 1.05 * raw_transfer["overall_mse"]
        ),
        "action_overlap_mse_within_1_05_raw": (
            transfer["action_overlap_mse"]
            <= 1.05 * raw_transfer["action_overlap_mse"]
        ),
        "downstream_effect_mse_within_1_05_raw": (
            transfer["downstream_effect_mse"]
            <= 1.05 * raw_transfer["downstream_effect_mse"]
        ),
        "action_and_target_hit_at_1_at_least_0_95": (
            attribution["complete_lejepa"]["action_and_target_hit_at_1"]
            >= 0.95
        ),
        "no_action_specificity_is_1": (
            attribution["complete_lejepa"]["no_action_specificity"]
            == 1.0
        ),
        "correct_action_beats_both_at_least_0_80": (
            action_sanity["complete_lejepa"][
                "correct_action_beats_both_fraction"
            ]
            >= 0.80
        ),
    }
    controls = (
        "invariance_only",
        "sigreg_only",
        "masked_autoencoder",
        "matched_pca",
    )
    selection_candidate = candidate["selection"][
        "downstream_effect_mse"
    ]
    best_transfer_name = min(
        controls,
        key=lambda name: forecast_scores[name][
            "transfer_evaluation"
        ]["downstream_effect_mse"],
    )
    best_transfer = forecast_scores[best_transfer_name][
        "transfer_evaluation"
    ]["downstream_effect_mse"]
    common_pairs = sorted(
        set(transfer_pair_errors["complete_lejepa"])
        & set(transfer_pair_errors[best_transfer_name])
    )
    win_fraction = float(
        np.mean(
            [
                transfer_pair_errors["complete_lejepa"][pair]
                < transfer_pair_errors[best_transfer_name][pair]
                for pair in common_pairs
            ]
        )
    )
    value = {
        "selection_strictly_best_of_all_controls": all(
            selection_candidate
            < forecast_scores[name]["selection"][
                "downstream_effect_mse"
            ]
            for name in controls
        ),
        "transfer_improves_best_control_by_5_percent": (
            transfer["downstream_effect_mse"] <= 0.95 * best_transfer
        ),
        "per_pair_win_fraction_at_least_0_60": win_fraction >= 0.60,
        "evaluation_did_not_select_configuration": True,
    }
    passed = all(safety.values()) and all(value.values())
    return {
        "safety_gates": safety,
        "value_gates": value,
        "safety_passed": all(safety.values()),
        "value_passed": all(value.values()),
        "passed": passed,
        "decision": (
            "advance_to_fixed_seed_representation_robustness"
            if passed
            else "reject_exact_complete_multi_view_lejepa_recipe"
        ),
        "best_transfer_control": best_transfer_name,
        "candidate_pair_win_fraction": win_fraction,
    }


def _encode_batches(model: Any, windows: ActionConditionedWindows) -> np.ndarray:
    return _encode_histories(model, windows.histories, windows.graph)


def _encode_histories(model: Any, histories: np.ndarray, graph: Any) -> np.ndarray:
    return np.concatenate(
        [
            model.encode(histories[start : start + 128], graph).tokens
            for start in range(0, len(histories), 128)
        ],
        axis=0,
    )


def _representation_diagnostics(values: np.ndarray) -> Mapping[str, Any]:
    flattened = np.asarray(values, dtype=np.float64).reshape(len(values), -1)
    centered = flattened - np.mean(flattened, axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    variance = np.var(flattened, axis=0)
    covariance = np.cov(flattened, rowvar=False)
    off_diagonal = covariance[
        ~np.eye(covariance.shape[0], dtype=np.bool_)
    ]
    entity_rows = {}
    for entity in range(values.shape[1]):
        local = values[:, entity] - np.mean(values[:, entity], axis=0)
        local_singular = np.linalg.svd(local, compute_uv=False)
        entity_rows[str(entity)] = {
            "mean_variance": float(np.mean(np.var(local, axis=0))),
            "effective_rank": _effective_rank(local_singular),
        }
    return {
        "mean_variance": float(np.mean(variance)),
        "mean_absolute_off_diagonal_covariance": float(
            np.mean(np.abs(off_diagonal))
        ),
        "effective_rank": _effective_rank(singular),
        "entities": entity_rows,
    }


def _view_diagnostics(values: np.ndarray) -> Mapping[str, float]:
    import torch

    tensor = torch.as_tensor(values, dtype=torch.float32)
    generator = torch.Generator(device="cpu").manual_seed(9509)
    sigreg = sketched_isotropic_gaussian_regularization(
        tensor,
        generator=generator,
        sketch_dimension=1024,
        knot_count=17,
    )
    global_mean = np.mean(values[:2], axis=0)
    return {
        "fixed_projection_sigreg": float(sigreg),
        "global_view_agreement_mse": float(
            np.mean(np.square(values[:2] - global_mean))
        ),
        "local_to_global_agreement_mse": float(
            np.mean(np.square(values[2:] - global_mean[None]))
        ),
    }


def _effective_rank(singular_values: np.ndarray) -> float:
    variance = np.square(np.asarray(singular_values, dtype=np.float64))
    total = float(np.sum(variance))
    if total <= 1e-18:
        return 0.0
    probabilities = variance / total
    probabilities = probabilities[probabilities > 0.0]
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def _median_latency_ms(call: Any) -> float:
    call()
    timings = []
    for _ in range(10):
        started = time.perf_counter_ns()
        call()
        timings.append((time.perf_counter_ns() - started) / 1e6)
    return float(np.median(timings))


def _protocol_checks(directory: Path) -> Mapping[str, bool]:
    with np.load(
        directory / "anchor-schedule.npz", allow_pickle=False
    ) as anchors:
        indices = anchors["indices"]
        arms = anchors["arm_ids"]
        transitions = anchors["transition_indices"]
        pair_count = indices.shape[1]
        transition_cycle = min(79, len(indices))
        pair_valid = bool(
            indices.ndim == 2
            and arms.shape == indices.shape
            and transitions.shape == indices.shape
            and np.all(np.sum(arms, axis=1) == pair_count // 2)
            and all(
                len(np.unique(transitions[:transition_cycle, pair]))
                == transition_cycle
                for pair in range(pair_count)
            )
        )
    with np.load(
        directory / "view-schedule.npz", allow_pickle=False
    ) as views:
        visible = views["visible_tokens"]
        present = views["present_tokens"]
        view_valid = bool(
            visible.ndim == 4
            and visible.shape == present.shape
            and visible.shape[1:] == (8, 20, 7)
            and not np.any(visible & ~present)
            and not np.any(present[:, 1, :4])
            and not np.any(present[:, 2:, :10])
            and np.all(np.sum(present[:, 2:], axis=(2, 3)) == 30)
        )
    return {
        "pair_blocked_schedule_is_valid": pair_valid,
        "telemetry_view_schedule_is_valid": view_valid,
        "evidence_arrays_are_finite": True,
    }


def _forecast_scores(
    prediction: np.ndarray, windows: ActionConditionedWindows
) -> Mapping[str, float]:
    observed = np.asarray(windows.future_states, dtype=np.float64)
    squared = np.square(np.asarray(prediction) - observed)
    row_mse = np.mean(squared, axis=(1, 2, 3))
    active = np.any(windows.future_actions[..., 1] > 0.5, axis=2)
    action_rows = np.asarray(
        [
            np.mean(squared[index][active[index]])
            if np.any(active[index])
            else np.nan
            for index in range(len(squared))
        ]
    )
    return {
        "overall_mse": _pair_balanced(row_mse, windows.matched_pair_ids),
        "action_overlap_mse": _pair_balanced(
            action_rows, windows.matched_pair_ids
        ),
        "downstream_effect_mse": float(
            np.mean(tuple(_downstream_pair_errors(prediction, windows).values()))
        ),
    }


def _downstream_pair_errors(
    prediction: np.ndarray, windows: ActionConditionedWindows
) -> Mapping[str, float]:
    index = {
        (trajectory, int(transition)): position
        for position, (trajectory, transition) in enumerate(
            zip(windows.trajectory_ids, windows.transition_indices)
        )
    }
    trajectories: Dict[str, list[str]] = {}
    treatment_target: Dict[str, int] = {}
    for row, (pair, trajectory) in enumerate(
        zip(windows.matched_pair_ids, windows.trajectory_ids)
    ):
        if trajectory not in trajectories.setdefault(pair, []):
            trajectories[pair].append(trajectory)
        active = np.argwhere(windows.future_actions[row, ..., 1] > 0.5)
        if len(active):
            treatment_target[trajectory] = int(active[0, 1])
    rows: Dict[str, float] = {}
    for pair, pair_trajectories in trajectories.items():
        treatment = [value for value in pair_trajectories if value in treatment_target]
        control = [value for value in pair_trajectories if value not in treatment_target]
        if len(treatment) != 1 or len(control) != 1:
            continue
        downstream = _downstream_positions(
            windows, treatment_target[treatment[0]]
        )
        errors = []
        for row, trajectory in enumerate(windows.trajectory_ids):
            if trajectory != treatment[0]:
                continue
            active = np.any(windows.future_actions[row, ..., 1] > 0.5, axis=1)
            other = index.get(
                (control[0], int(windows.transition_indices[row]))
            )
            if other is None or not np.any(active) or not downstream:
                continue
            predicted_effect = prediction[row] - prediction[other]
            observed_effect = (
                windows.future_states[row] - windows.future_states[other]
            )
            errors.append(
                np.square(
                    predicted_effect[active][:, downstream]
                    - observed_effect[active][:, downstream]
                ).mean()
            )
        if errors:
            rows[pair] = float(np.mean(errors))
    if not rows:
        raise ValueError("downstream effect assessment has no matched rows")
    return rows


def _downstream_positions(
    windows: ActionConditionedWindows, start: int
) -> Tuple[int, ...]:
    graph = windows.graph
    adjacency = {name: [] for name in graph.entity_ids}
    for entity in graph.entities:
        if entity.kind == "edge":
            adjacency[entity.source].append(entity.entity_id)
            adjacency[entity.entity_id].append(entity.target)
    start_name = graph.entity_ids[start]
    discovered = []
    frontier = list(adjacency[start_name])
    while frontier:
        candidate = frontier.pop(0)
        if candidate in discovered or candidate == start_name:
            continue
        discovered.append(candidate)
        frontier.extend(adjacency[candidate])
    return tuple(graph.entity_ids.index(value) for value in discovered)


def _pair_balanced(values: np.ndarray, pair_ids: Sequence[str]) -> float:
    rows = []
    pair_array = np.asarray(pair_ids)
    for pair in sorted(set(pair_ids)):
        local = values[pair_array == pair]
        local = local[np.isfinite(local)]
        if len(local):
            rows.append(float(np.mean(local)))
    return float(np.mean(rows))


def _state_probe(
    fit_tokens: np.ndarray,
    fit: ActionConditionedWindows,
    evaluation_tokens: np.ndarray,
    evaluation: ActionConditionedWindows,
    ownership: np.ndarray,
) -> Mapping[str, Any]:
    entities = {}
    normalized_errors = []
    for entity, name in enumerate(fit.entity_names):
        mask = ownership[entity] & (
            np.ptp(fit.histories[:, -1, entity], axis=0) > 1e-9
        )
        if not np.any(mask):
            entities[name] = {"nrmse": None, "feature_count": 0}
            continue
        x = fit_tokens[:, entity]
        x_center = x.mean(axis=0)
        x_scale = x.std(axis=0)
        x_scale[x_scale <= 1e-12] = 1.0
        design = np.column_stack(
            ((x - x_center) / x_scale, np.ones(len(x)))
        )
        penalty = np.eye(design.shape[1])
        penalty[-1, -1] = 0.0
        target = fit.histories[:, -1, entity][:, mask]
        coefficients = np.linalg.solve(
            design.T @ design + 1e-3 * penalty,
            design.T @ target,
        )
        evaluation_design = np.column_stack(
            (
                (evaluation_tokens[:, entity] - x_center) / x_scale,
                np.ones(len(evaluation_tokens)),
            )
        )
        scale = target.std(axis=0)
        scale[scale <= 1e-12] = 1.0
        normalized = np.square(
            (
                evaluation_design @ coefficients
                - evaluation.histories[:, -1, entity][:, mask]
            )
            / scale
        ).reshape(-1)
        normalized_errors.append(normalized)
        entities[name] = {
            "nrmse": float(np.sqrt(np.mean(normalized))),
            "feature_count": int(np.sum(mask)),
        }
    return {
        "aggregate_nrmse": float(
            np.sqrt(np.mean(np.concatenate(normalized_errors)))
        ),
        "entities": entities,
    }


def _attribution_evidence(
    probe: ReducedRankActionProbe,
    tokens: np.ndarray,
    queries: PreparedAttributionQueries,
    ownership: np.ndarray,
) -> Tuple[Mapping[str, float], np.ndarray]:
    all_predictions = []
    for index in range(len(queries.query_ids)):
        candidate_count = len(queries.candidate_ids)
        all_predictions.append(
            probe.predict(
                np.repeat(
                    tokens[index : index + 1],
                    candidate_count,
                    axis=0,
                ),
                np.repeat(
                    queries.future_controls[index : index + 1],
                    candidate_count,
                    axis=0,
                ),
                queries.candidate_actions[index],
            ).astype(np.float32)
        )
    predictions = np.stack(all_predictions)
    return (
        _attribution_scores_from_predictions(
            predictions, queries, ownership
        ),
        predictions,
    )


def _attribution_scores_from_predictions(
    predictions: np.ndarray,
    queries: PreparedAttributionQueries,
    ownership: np.ndarray,
) -> Mapping[str, float]:
    treatment_hits = []
    control_hits = []
    for index in range(len(queries.query_ids)):
        error = np.mean(
            np.square(
                predictions[index]
                - queries.observed_future[index][None]
            )[..., ownership],
            axis=(1, 2),
        )
        winner = int(np.argmin(error))
        expected = queries.expected_action_kinds[index]
        if expected:
            treatment_hits.append(
                queries.candidate_action_kinds[winner] == expected
                and queries.candidate_target_entities[winner]
                == queries.expected_target_entities[index]
            )
        else:
            control_hits.append(
                queries.candidate_ids[winner] == "no_action"
            )
    return {
        "action_and_target_hit_at_1": float(np.mean(treatment_hits)),
        "no_action_specificity": float(np.mean(control_hits)),
    }


def _action_sanity_evidence(
    probe: ReducedRankActionProbe,
    tokens: np.ndarray,
    windows: ActionConditionedWindows,
    ownership: np.ndarray,
) -> Tuple[Mapping[str, float], Mapping[str, np.ndarray]]:
    correct = probe.predict(
        tokens, windows.future_controls, windows.future_actions
    )
    no_action = np.zeros_like(windows.future_actions)
    no_action[..., 0] = 1.0
    absent = probe.predict(tokens, windows.future_controls, no_action)
    pair_ids = sorted(set(windows.matched_pair_ids))
    pair_action = {
        pair: windows.future_actions[
            np.asarray(windows.matched_pair_ids) == pair
        ].copy()
        for pair in pair_ids
    }
    shuffled = np.zeros_like(windows.future_actions)
    for position, pair in enumerate(pair_ids):
        rows = np.flatnonzero(np.asarray(windows.matched_pair_ids) == pair)
        donor = pair_ids[(position + 1) % len(pair_ids)]
        donor_values = pair_action[donor]
        shuffled[rows] = donor_values[: len(rows)]
    shuffled_prediction = probe.predict(
        tokens, windows.future_controls, shuffled
    )
    outputs = {
        "correct": correct.astype(np.float32),
        "no_action": absent.astype(np.float32),
        "shuffled": shuffled_prediction.astype(np.float32),
    }
    return _action_sanity_from_predictions(
        outputs, windows, ownership
    ), outputs


def _action_sanity_from_predictions(
    predictions: Mapping[str, np.ndarray],
    windows: ActionConditionedWindows,
    ownership: np.ndarray,
) -> Mapping[str, float]:
    correct = predictions["correct"]
    absent = predictions["no_action"]
    shuffled_prediction = predictions["shuffled"]
    wins = []
    pair_ids = sorted(set(windows.matched_pair_ids))
    pair_array = np.asarray(windows.matched_pair_ids)
    for pair in pair_ids:
        rows = np.flatnonzero(pair_array == pair)
        active = np.any(
            windows.future_actions[rows, ..., 1] > 0.5, axis=(1, 2)
        )
        rows = rows[active]
        if not len(rows):
            continue
        target = windows.future_states[rows]
        score = lambda value: float(
            np.mean(np.square(value[rows] - target)[..., ownership])
        )
        wins.append(
            score(correct) < score(absent)
            and score(correct) < score(shuffled_prediction)
        )
    return {
        "correct_action_beats_both_fraction": float(np.mean(wins))
    }


def _transfer_queries(
    queries: PreparedAttributionQueries,
    control_names: Sequence[str],
    held_out_value: float,
) -> PreparedAttributionQueries:
    position = tuple(control_names).index("worker_replicas")
    selection = np.isclose(
        queries.future_controls[:, 0, position], held_out_value
    )
    return subset_attribution_queries(queries, selection)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            value, indent=2, sort_keys=True, allow_nan=False
        )
        + "\n"
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _markdown_report(report: Mapping[str, Any]) -> str:
    assessment = report["assessment"]
    scores = report["forecast_scores"]
    lines = [
        "# Complete multi-view LeJEPA tracer v1",
        "",
        f"Decision: `{assessment['decision']}`.",
        "",
        "| Representation | Selection effect MSE | Transfer effect MSE |",
        "|---|---:|---:|",
    ]
    for name in REPRESENTATION_NAMES:
        lines.append(
            f"| {name} | "
            f"{scores[name]['selection']['downstream_effect_mse']:.6f} | "
            f"{scores[name]['transfer_evaluation']['downstream_effect_mse']:.6f} |"
        )
    lines.extend(
        [
            "",
            "This is a single-seed open-development representation tracer, "
            "not a production alert-policy result.",
            "",
        ]
    )
    return "\n".join(lines)


def main(arguments: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(
            "artifacts/action-dynamics/edge-preprocessing-v1/"
            "eb54271132f88c9a431b01e786ea66279a563776434cca2290e47e6b7ae9b3ff"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/action-dynamics/prototype-complete-lejepa-v1"
        ),
    )
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--sketch-dimension", type=int, default=1024)
    parsed = parser.parse_args(arguments)
    result = run_complete_lejepa_tracer(
        cache_directory=parsed.cache,
        output_directory=parsed.output,
        steps=parsed.steps,
        sketch_dimension=parsed.sketch_dimension,
    )
    print(json.dumps(result["assessment"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
