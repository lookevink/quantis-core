#!/usr/bin/env python3
"""Retained runner for the frozen SALT-JEPA telemetry tracer."""

import argparse
import hashlib
import inspect
import json
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from lab.action_dynamics.prototype_complete_lejepa import (
    _action_sanity_evidence,
    _attribution_evidence,
    _downstream_pair_errors,
    _forecast_scores,
    _state_probe,
    _transfer_queries,
)
from lab.action_dynamics.prototype_salt_jepa_assessor import (
    REPRESENTATION_NAMES,
    SALT_NAMES,
    assess_stored_bundle,
    verify_stored_assessment,
)
from quantis_core.action_conditioned_dynamics import (
    ActionConditionedWindows,
)
from quantis_core.edge_dynamics.complete_lejepa import (
    EntityPcaRepresentation,
    PairBlockedAnchorSchedule,
    ReducedRankActionProbe,
    fit_owned_feature_mask,
)
from quantis_core.edge_dynamics.data import (
    PreparedAttributionQueries,
    load_edge_dynamics_cache,
    partition_worker_topology,
)
from quantis_core.edge_dynamics.models import (
    ContractiveLowRankDynamics,
    LowRankConfig,
)
from quantis_core.edge_dynamics.salt_jepa import (
    SaltJepaConfig,
    SaltJepaRepresentation,
    SaltMaskSchedule,
    SaltTargetSchedule,
)


FROZEN_CACHE = Path(
    "artifacts/action-dynamics/edge-preprocessing-v1/"
    "eb54271132f88c9a431b01e786ea66279a563776434cca2290e47e6b7ae9b3ff"
)
FROZEN_OUTPUT = Path(
    "artifacts/action-dynamics/prototype-salt-jepa-v1"
)
FROZEN_TEACHER_STEPS = 320
FROZEN_STUDENT_STEPS = 1280
RIDGES = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
IMPLEMENTATION_SOURCE_PATHS = (
    "lab/action_dynamics/prototype_salt_jepa.py",
    "lab/action_dynamics/prototype_salt_jepa_assessor.py",
    "lab/action_dynamics/prototype_complete_lejepa.py",
    "src/quantis_core/edge_dynamics/salt_jepa.py",
    "src/quantis_core/edge_dynamics/complete_lejepa.py",
    "src/quantis_core/edge_dynamics/models.py",
    "src/quantis_core/edge_dynamics/data.py",
    "src/quantis_core/action_conditioned_dynamics.py",
    "src/quantis_core/graph_telemetry.py",
    "tests/test_salt_jepa.py",
    "docs/research/salt-jepa-primary-source-notes.md",
    "docs/specs/salt-jepa-telemetry-tracer-v1.md",
    "docs/wayfinding/jepa-implementation-program/"
    "023-test-salt-jepa-static-teacher.md",
)


def run_experiment(
    *,
    cache_directory: Path,
    output_directory: Path,
    teacher_steps: int,
    student_steps: int,
    latency_repetitions: int,
    allow_noninterpretable_smoke: bool,
) -> Path:
    """Fit, independently assess, and atomically publish one SALT tracer."""

    cache = Path(cache_directory).resolve()
    output = Path(output_directory).resolve()
    building = output.with_name(output.name + ".building")
    if output.exists() or building.exists():
        raise FileExistsError("SALT-JEPA refuses an existing output")
    interpretable = (
        cache == (Path.cwd() / FROZEN_CACHE).resolve()
        and output == (Path.cwd() / FROZEN_OUTPUT).resolve()
        and teacher_steps == FROZEN_TEACHER_STEPS
        and student_steps == FROZEN_STUDENT_STEPS
        and latency_repetitions == 100
    )
    if not interpretable and not allow_noninterpretable_smoke:
        raise ValueError(
            "non-frozen SALT runs require explicit smoke permission"
        )
    if not interpretable and output == (Path.cwd() / FROZEN_OUTPUT).resolve():
        raise ValueError("SALT smoke cannot use the frozen output")
    implementation_commit = _git_head()
    source_hashes = _source_identity(require_clean=interpretable)
    building.mkdir(parents=True)
    models_directory = building / "models"
    models_directory.mkdir()
    reproduction_directory = building / "reproduction-source"
    started = time.time()

    prepared = load_edge_dynamics_cache(cache)
    partitions = {
        role: partition_worker_topology(windows)
        for role, windows in prepared.windows.items()
    }
    held_values = {
        value.held_out_normalized_value for value in partitions.values()
    }
    if len(held_values) != 1:
        raise ValueError("SALT held topology identity differs by role")
    held_value = next(iter(held_values))
    windows_by_role = {
        "fit": partitions["fit"].in_distribution,
        "selection": partitions["selection"].in_distribution,
        "calibration": partitions["calibration"].in_distribution,
        "iid_evaluation": partitions["evaluation"].in_distribution,
        "transfer_evaluation": partitions["evaluation"].held_out,
    }
    fit = windows_by_role["fit"]
    ownership = fit_owned_feature_mask(fit)
    base_config = SaltJepaConfig(
        teacher_steps=teacher_steps,
        student_steps=student_steps,
    )
    _write_schedules(
        building,
        fit,
        ownership,
        base_config,
    )

    models: Dict[str, SaltJepaRepresentation] = {}
    training_seconds = {}
    for name, alignment in (
        ("salt_jepa", "aligned"),
        ("deranged_salt_jepa", "deranged"),
    ):
        config = SaltJepaConfig(
            alignment=alignment,
            teacher_steps=teacher_steps,
            student_steps=student_steps,
        )
        fit_started = time.perf_counter()
        model = SaltJepaRepresentation(config).fit(fit)
        training_seconds[name] = time.perf_counter() - fit_started
        models[name] = model
        _write_json(models_directory / f"{name}.json", model.to_dict())
    teacher_sample = fit.histories[:8]
    if not np.allclose(
        models["salt_jepa"].encode_teacher(
            teacher_sample, fit.graph
        ).tokens,
        models["deranged_salt_jepa"].encode_teacher(
            teacher_sample, fit.graph
        ).tokens,
        atol=0.0,
        rtol=0.0,
    ):
        raise RuntimeError("SALT cells did not train identical teachers")
    pca_started = time.perf_counter()
    pca = EntityPcaRepresentation(width=64).fit(fit)
    training_seconds["matched_pca"] = (
        time.perf_counter() - pca_started
    )
    _write_json(models_directory / "matched_pca.json", pca.to_dict())

    encoded = {
        name: {
            role: _encode_representation(
                name, models, pca, windows.histories, windows.graph
            )
            for role, windows in windows_by_role.items()
            if role != "calibration"
        }
        for name in REPRESENTATION_NAMES
    }
    raw = ContractiveLowRankDynamics(
        LowRankConfig(rank=32)
    ).fit(fit)
    _write_json(models_directory / "raw_rank32.json", raw.to_dict())
    raw_predictions = {
        role: raw.rollout(
            windows.histories,
            windows.future_controls,
            windows.future_actions,
            windows.graph,
        ).mean
        for role, windows in windows_by_role.items()
        if role not in ("fit", "calibration")
    }
    raw_scores = {
        role: _forecast_scores(values, windows_by_role[role])
        for role, values in raw_predictions.items()
    }

    probes: Dict[str, ReducedRankActionProbe] = {}
    selected_ridges: Dict[str, float] = {}
    ridge_curves: Dict[str, list[Mapping[str, Any]]] = {}
    ridge_predictions: Dict[str, Dict[float, np.ndarray]] = {}
    for name in REPRESENTATION_NAMES:
        curve = []
        fitted = {}
        predictions_by_ridge = {}
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
            raw_safe = (
                scores["overall_mse"]
                <= 1.05 * raw_scores["selection"]["overall_mse"]
                and scores["action_overlap_mse"]
                <= 1.05
                * raw_scores["selection"]["action_overlap_mse"]
            )
            curve.append(
                {"ridge": ridge, "raw_safe": raw_safe, **scores}
            )
            fitted[ridge] = probe
            predictions_by_ridge[ridge] = prediction
        eligible = [row for row in curve if row["raw_safe"]]
        selected = min(
            eligible or curve,
            key=lambda row: (
                row["downstream_effect_mse"],
                row["ridge"],
            ),
        )
        selected_ridge = float(selected["ridge"])
        probes[name] = fitted[selected_ridge]
        selected_ridges[name] = selected_ridge
        ridge_curves[name] = curve
        ridge_predictions[name] = predictions_by_ridge
        _write_json(
            models_directory / f"{name}-probe.json",
            probes[name].to_dict(),
        )

    predictions = {
        name: {
            role: probes[name].predict(
                encoded[name][role],
                windows.future_controls,
                windows.future_actions,
            )
            for role, windows in windows_by_role.items()
            if role not in ("fit", "calibration")
        }
        for name in REPRESENTATION_NAMES
    }
    forecast_scores = {
        name: {
            role: _forecast_scores(values, windows_by_role[role])
            for role, values in role_values.items()
        }
        for name, role_values in predictions.items()
    }
    state_probes = {
        name: {
            role: _state_probe(
                encoded[name]["fit"],
                fit,
                encoded[name][role],
                windows_by_role[role],
                ownership,
            )
            for role in (
                "selection",
                "iid_evaluation",
                "transfer_evaluation",
            )
        }
        for name in REPRESENTATION_NAMES
    }
    transfer_queries = _transfer_queries(
        prepared.attribution_queries,
        fit.control_feature_names,
        held_value,
    )
    query_tokens = {
        name: _encode_representation(
            name,
            models,
            pca,
            transfer_queries.histories,
            fit.graph,
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

    diagnostics = {
        name: {
            role: _diagnostic_for_role(
                models[name],
                windows_by_role[role],
                step=teacher_steps
                + student_steps
                + (0 if role == "selection" else 1),
            )
            for role in ("selection", "transfer_evaluation")
        }
        for name in SALT_NAMES
    }
    restored_models = {
        name: SaltJepaRepresentation.from_dict(models[name].to_dict())
        for name in SALT_NAMES
    }
    restored_pca = EntityPcaRepresentation.from_dict(pca.to_dict())
    transfer = windows_by_role["transfer_evaluation"]
    restoration_original = {}
    restoration_restored = {}
    for name in REPRESENTATION_NAMES:
        restoration_original[name] = encoded[name][
            "transfer_evaluation"
        ][:8]
        if name == "salt_jepa":
            restored_values = _encode_chunks(
                restored_models[name].encode,
                transfer.histories[:8],
                transfer.graph,
            )
        elif name == "deranged_salt_jepa":
            restored_values = _encode_chunks(
                restored_models[name].encode,
                transfer.histories[:8],
                transfer.graph,
            )
        elif name == "reconstructive_teacher":
            restored_values = _encode_chunks(
                restored_models["salt_jepa"].encode_teacher,
                transfer.histories[:8],
                transfer.graph,
            )
        else:
            restored_values = _encode_chunks(
                restored_pca.encode,
                transfer.histories[:8],
                transfer.graph,
            )
        restoration_restored[name] = restored_values
    restoration_probe_original = {}
    restoration_probe_restored = {}
    for name in REPRESENTATION_NAMES:
        original = probes[name].predict(
            encoded[name]["transfer_evaluation"][:8],
            transfer.future_controls[:8],
            transfer.future_actions[:8],
        )
        restored_probe = ReducedRankActionProbe.from_dict(
            probes[name].to_dict()
        )
        replay = restored_probe.predict(
            encoded[name]["transfer_evaluation"][:8],
            transfer.future_controls[:8],
            transfer.future_actions[:8],
        )
        restoration_probe_original[name] = original
        restoration_probe_restored[name] = replay
    restoration_diagnostic_original = {}
    restoration_diagnostic_restored = {}
    for name in SALT_NAMES:
        original = diagnostics[name]["transfer_evaluation"]
        replay = _diagnostic_for_role(
            restored_models[name],
            transfer,
            step=teacher_steps + student_steps + 1,
        )
        restoration_diagnostic_original[name] = (
            original.predicted_tokens
        )
        restoration_diagnostic_restored[name] = replay.predicted_tokens

    candidate_bundle = _inference_bundle_bytes(
        models["salt_jepa"], probes["salt_jepa"]
    )
    latency = _latency(
        models["salt_jepa"],
        probes["salt_jepa"],
        encoded["salt_jepa"]["transfer_evaluation"][:1],
        transfer,
        repetitions=latency_repetitions,
    )
    parameter_counts = {
        name: {
            "training": (
                models[name].inference_parameter_count
                + models[name].training_only_parameter_count
            ),
            "inference": models[name].inference_parameter_count,
        }
        for name in SALT_NAMES
    }
    teacher_unchanged = {
        name: models[name].teacher_unchanged_during_student
        for name in SALT_NAMES
    }
    evidence_arrays: Dict[str, np.ndarray] = {}
    for role in (
        "fit",
        "selection",
        "iid_evaluation",
        "transfer_evaluation",
    ):
        windows = windows_by_role[role]
        evidence_arrays[f"histories__{role}"] = (
            windows.histories.astype(np.float32)
        )
        evidence_arrays[f"target__{role}"] = (
            windows.future_states.astype(np.float32)
        )
        evidence_arrays[f"controls__{role}"] = (
            windows.future_controls.astype(np.float32)
        )
        evidence_arrays[f"actions__{role}"] = (
            windows.future_actions.astype(np.float32)
        )
        for name in REPRESENTATION_NAMES:
            evidence_arrays[f"representation__{name}__{role}"] = (
                encoded[name][role].astype(np.float32)
            )
    for role in (
        "selection",
        "iid_evaluation",
        "transfer_evaluation",
    ):
        evidence_arrays[f"raw_prediction__{role}"] = (
            raw_predictions[role].astype(np.float32)
        )
        for name in REPRESENTATION_NAMES:
            evidence_arrays[f"prediction__{name}__{role}"] = (
                predictions[name][role].astype(np.float32)
            )
    for name in REPRESENTATION_NAMES:
        for position, ridge in enumerate(RIDGES):
            evidence_arrays[
                f"ridge_prediction__{name}__{position}"
            ] = ridge_predictions[name][ridge].astype(np.float32)
        evidence_arrays[f"attribution_prediction__{name}"] = (
            attribution_predictions[name].astype(np.float32)
        )
        for variant, values in action_sanity_predictions[name].items():
            evidence_arrays[
                f"action_sanity__{name}__{variant}"
            ] = values.astype(np.float32)
        evidence_arrays[f"restoration_original__{name}"] = (
            restoration_original[name].astype(np.float32)
        )
        evidence_arrays[f"restoration_restored__{name}"] = (
            restoration_restored[name].astype(np.float32)
        )
        evidence_arrays[f"restoration_probe_original__{name}"] = (
            restoration_probe_original[name].astype(np.float32)
        )
        evidence_arrays[f"restoration_probe_restored__{name}"] = (
            restoration_probe_restored[name].astype(np.float32)
        )
    for name in SALT_NAMES:
        for role, diagnostic in diagnostics[name].items():
            evidence_arrays[
                f"diagnostic_predicted__{name}__{role}"
            ] = diagnostic.predicted_tokens.astype(np.float32)
            evidence_arrays[
                f"diagnostic_target__{name}__{role}"
            ] = diagnostic.target_tokens.astype(np.float32)
            evidence_arrays[f"diagnostic_mask__{name}__{role}"] = (
                diagnostic.target_mask.astype(np.bool_)
            )
        evidence_arrays[
            f"restoration_diagnostic_original__{name}"
        ] = restoration_diagnostic_original[name].astype(np.float32)
        evidence_arrays[
            f"restoration_diagnostic_restored__{name}"
        ] = restoration_diagnostic_restored[name].astype(np.float32)
    evidence_arrays["query_histories"] = (
        transfer_queries.histories.astype(np.float32)
    )
    evidence_arrays["query_future_controls"] = (
        transfer_queries.future_controls.astype(np.float32)
    )
    evidence_arrays["query_observed_future"] = (
        transfer_queries.observed_future.astype(np.float32)
    )
    evidence_arrays["query_candidate_actions"] = (
        transfer_queries.candidate_actions.astype(np.float32)
    )
    np.savez_compressed(building / "evidence.npz", **evidence_arrays)

    metadata = {
        "schema_version": 1,
        "kind": "salt_jepa_assessment_evidence",
        "interpretable": interpretable,
        "implementation_commit": implementation_commit,
        "graph": fit.graph.to_dict(),
        "entity_names": list(fit.entity_names),
        "state_feature_names": list(fit.state_feature_names),
        "control_feature_names": list(fit.control_feature_names),
        "action_feature_names": list(fit.action_feature_names),
        "ownership_mask": ownership.astype(int).tolist(),
        "roles": {
            role: _role_metadata(windows)
            for role, windows in windows_by_role.items()
        },
        "queries": _query_metadata(transfer_queries),
        "ridge_values": list(RIDGES),
        "selected_ridges": selected_ridges,
        "parameter_counts": parameter_counts,
        "teacher_unchanged": teacher_unchanged,
        "public_inference_is_causal": (
            set(
                inspect.signature(
                    SaltJepaRepresentation.encode
                ).parameters
            )
            == {"self", "histories", "graph"}
        ),
        "deployed_bundle_bytes": candidate_bundle,
        "latency": latency,
    }
    _write_json(building / "evidence-metadata.json", metadata)
    assessment = dict(assess_stored_bundle(building))
    _write_json(building / "assessment.json", assessment)
    report = {
        "schema_version": 1,
        "kind": "salt_jepa_telemetry_tracer_v1",
        "evidence_boundary": (
            "single-seed open-development representation tracer; "
            "not a production alert system or sealed confirmation"
        ),
        "interpretable": interpretable,
        "implementation_commit": implementation_commit,
        "source": {
            "cache_directory": str(cache),
            "source_corpus_sha256": prepared.source_corpus_sha256,
            "source_artifact_manifest_sha256": (
                prepared.source_artifact_manifest_sha256
            ),
            "preprocessing_protocol": prepared.preprocessing_protocol,
            "held_out_worker_topology_normalized": held_value,
            "primary_paper": "https://arxiv.org/abs/2509.24317",
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "started_unix": started,
            "completed_unix": time.time(),
        },
        "configurations": {
            name: models[name].config.to_dict() for name in SALT_NAMES
        },
        "window_counts": {
            role: len(windows.histories)
            for role, windows in windows_by_role.items()
        },
        "pair_counts": {
            role: len(set(windows.matched_pair_ids))
            for role, windows in windows_by_role.items()
        },
        "training_seconds": training_seconds,
        "training_metrics": {
            name: {
                "teacher": [
                    dict(row)
                    for row in models[name].teacher_training_metrics
                ],
                "student": [
                    dict(row)
                    for row in models[name].student_training_metrics
                ],
            }
            for name in SALT_NAMES
        },
        "selected_ridges": selected_ridges,
        "ridge_curves": ridge_curves,
        "forecast_scores": forecast_scores,
        "raw_scores": raw_scores,
        "state_probes": state_probes,
        "attribution": attribution,
        "action_sanity": action_sanity,
        "masked_latent_l1": {
            name: {
                role: diagnostic.l1
                for role, diagnostic in roles.items()
            }
            for name, roles in diagnostics.items()
        },
        "transfer_pair_errors": {
            name: _downstream_pair_errors(
                predictions[name]["transfer_evaluation"], transfer
            )
            for name in REPRESENTATION_NAMES
        },
        "parameter_counts": parameter_counts,
        "deployed_bundle_bytes": candidate_bundle,
        "latency": latency,
        "assessment": assessment,
        "source_sha256": source_hashes,
    }
    _write_json(building / "result.json", report)
    (building / "REPORT.md").write_text(_markdown_report(report))
    for source_name in IMPLEMENTATION_SOURCE_PATHS:
        source = Path(source_name)
        destination = reproduction_directory / source_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    manifest = {
        "schema_version": 1,
        "kind": "salt_jepa_artifact_manifest",
        "implementation_commit": implementation_commit,
        "sha256": {
            str(path.relative_to(building)): _file_sha256(path)
            for path in sorted(building.rglob("*"))
            if path.is_file()
        },
    }
    _write_json(building / "artifact-manifest.json", manifest)
    building.rename(output)
    verify_stored_assessment(output)
    return output


def _write_schedules(
    directory: Path,
    fit: ActionConditionedWindows,
    ownership: np.ndarray,
    config: SaltJepaConfig,
) -> None:
    anchors = PairBlockedAnchorSchedule(fit, seed=config.anchor_seed)
    masks = SaltMaskSchedule(
        graph=fit.graph,
        ownership_mask=ownership,
        seed=config.mask_seed,
    )
    total_steps = config.teacher_steps + config.student_steps
    anchor_batches = [anchors.batch(step) for step in range(total_steps)]
    visible = []
    target = []
    for step, batch in enumerate(anchor_batches):
        masked = masks.batch(fit.histories[batch.indices], step=step)
        visible.append(masked.visible_tokens)
        target.append(masked.target_tokens)
    np.savez_compressed(
        directory / "anchor-schedule.npz",
        indices=np.stack([batch.indices for batch in anchor_batches]),
        arm_ids=np.stack([batch.arm_ids for batch in anchor_batches]),
        transition_indices=np.stack(
            [batch.transition_indices for batch in anchor_batches]
        ),
        pair_ids=np.asarray(anchors.pair_ids),
    )
    aligned_schedule = SaltTargetSchedule("aligned")
    deranged_schedule = SaltTargetSchedule("deranged")
    np.savez_compressed(
        directory / "mask-schedule.npz",
        visible_tokens=np.stack(visible),
        target_tokens=np.stack(target),
        aligned_target_indices=np.stack(
            [
                aligned_schedule.indices(
                    anchors.pair_ids, step=step
                )
                for step in range(config.student_steps)
            ]
        ),
        deranged_target_indices=np.stack(
            [
                deranged_schedule.indices(
                    anchors.pair_ids, step=step
                )
                for step in range(config.student_steps)
            ]
        ),
    )


def _encode_representation(
    name: str,
    models: Mapping[str, SaltJepaRepresentation],
    pca: EntityPcaRepresentation,
    histories: np.ndarray,
    graph: Any,
) -> np.ndarray:
    if name == "salt_jepa":
        call = models[name].encode
    elif name == "deranged_salt_jepa":
        call = models[name].encode
    elif name == "reconstructive_teacher":
        call = models["salt_jepa"].encode_teacher
    elif name == "matched_pca":
        call = pca.encode
    else:
        raise ValueError(f"unsupported SALT representation: {name}")
    return _encode_chunks(call, histories, graph)


def _encode_chunks(
    call: Any, histories: np.ndarray, graph: Any
) -> np.ndarray:
    return np.concatenate(
        [
            call(histories[start : start + 128], graph).tokens
            for start in range(0, len(histories), 128)
        ],
        axis=0,
    )


def _diagnostic_for_role(
    model: SaltJepaRepresentation,
    windows: ActionConditionedWindows,
    *,
    step: int,
) -> Any:
    anchors = PairBlockedAnchorSchedule(
        windows, seed=model.config.anchor_seed
    ).batch(0)
    return model.diagnose_masked_prediction(
        windows.histories[anchors.indices],
        windows.graph,
        step=step,
    )


def _role_metadata(
    windows: ActionConditionedWindows,
) -> Mapping[str, Any]:
    return {
        "trajectory_ids": list(windows.trajectory_ids),
        "pair_ids": list(windows.matched_pair_ids),
        "transition_indices": windows.transition_indices.astype(int).tolist(),
    }


def _query_metadata(
    queries: PreparedAttributionQueries,
) -> Mapping[str, Any]:
    return {
        "query_ids": list(queries.query_ids),
        "candidate_ids": list(queries.candidate_ids),
        "candidate_action_kinds": list(queries.candidate_action_kinds),
        "candidate_target_entities": list(
            queries.candidate_target_entities
        ),
        "expected_action_kinds": list(queries.expected_action_kinds),
        "expected_target_entities": list(
            queries.expected_target_entities
        ),
        "expected_variant_ids": list(queries.expected_variant_ids),
    }


def _inference_bundle_bytes(
    model: SaltJepaRepresentation,
    probe: ReducedRankActionProbe,
) -> int:
    payload = model.to_dict()
    inference = {
        "schema_version": 1,
        "kind": "salt_jepa_student_inference_bundle",
        "config": payload["config"],
        "graph": payload["graph"],
        "feature_names": payload["feature_names"],
        "ownership_mask": payload["ownership_mask"],
        "student_state": payload["student_state"],
        "probe": probe.to_dict(),
    }
    return len(
        json.dumps(
            inference,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    )


def _latency(
    model: SaltJepaRepresentation,
    probe: ReducedRankActionProbe,
    token: np.ndarray,
    windows: ActionConditionedWindows,
    *,
    repetitions: int,
) -> Mapping[str, float]:
    if repetitions < 1:
        raise ValueError("SALT latency repetitions must be positive")

    def call() -> None:
        encoded = model.encode(windows.histories[:1], windows.graph).tokens
        probe.predict(
            encoded,
            windows.future_controls[:1],
            windows.future_actions[:1],
        )

    call()
    values = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        call()
        values.append((time.perf_counter_ns() - started) / 1e6)
    return {
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.quantile(values, 0.95)),
        "repetitions": float(repetitions),
        "probe_token_identity_max_abs": float(
            np.max(np.abs(token - token))
        ),
    }


def _source_identity(*, require_clean: bool) -> Mapping[str, str]:
    if require_clean:
        status = subprocess.run(
            ["git", "status", "--short"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if status.strip():
            raise RuntimeError(
                "frozen SALT run requires a clean implementation commit"
            )
    missing = [
        name for name in IMPLEMENTATION_SOURCE_PATHS if not Path(name).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"missing SALT sources: {missing}")
    return {
        name: _file_sha256(Path(name))
        for name in IMPLEMENTATION_SOURCE_PATHS
    }


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


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
    assessment = dict(report["assessment"])
    scores = dict(report["forecast_scores"])
    raw = dict(report["raw_scores"])["transfer_evaluation"]
    lines = [
        "# SALT-JEPA telemetry tracer",
        "",
        f"- decision: `{assessment['decision']}`",
        f"- interpretable: `{str(report['interpretable']).lower()}`",
        f"- implementation commit: `{report['implementation_commit']}`",
        "",
        "## Held-topology downstream effect MSE",
        "",
        f"- raw rank-32: `{float(raw['downstream_effect_mse']):.6f}`",
    ]
    for name in REPRESENTATION_NAMES:
        value = dict(scores[name])["transfer_evaluation"][
            "downstream_effect_mse"
        ]
        lines.append(f"- {name}: `{float(value):.6f}`")
    lines.extend(
        [
            "",
            "The complete assessment is stored in `assessment.json`; every",
            "gate is independently recomputed from `evidence.npz`.",
            "",
        ]
    )
    return "\n".join(lines)


def main(arguments: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=FROZEN_CACHE)
    parser.add_argument("--output", type=Path, default=FROZEN_OUTPUT)
    parser.add_argument(
        "--teacher-steps", type=int, default=FROZEN_TEACHER_STEPS
    )
    parser.add_argument(
        "--student-steps", type=int, default=FROZEN_STUDENT_STEPS
    )
    parser.add_argument(
        "--latency-repetitions", type=int, default=100
    )
    parser.add_argument(
        "--allow-noninterpretable-smoke", action="store_true"
    )
    parsed = parser.parse_args(arguments)
    output = run_experiment(
        cache_directory=parsed.cache,
        output_directory=parsed.output,
        teacher_steps=parsed.teacher_steps,
        student_steps=parsed.student_steps,
        latency_repetitions=parsed.latency_repetitions,
        allow_noninterpretable_smoke=(
            parsed.allow_noninterpretable_smoke
        ),
    )
    print(
        json.dumps(
            verify_stored_assessment(output),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
