#!/usr/bin/env python3
"""Independent stored-evidence assessor for the VISReg telemetry tracer."""

import argparse
import ast
import gzip
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from lab.action_dynamics.prototype_peira_assessor import (
        _action_sanity_from_predictions,
        _attribution_scores_from_predictions,
        _canonical_json,
        _downstream_pair_errors,
        _effective_rank,
        _file_sha256,
        _forecast_scores,
        _max_abs,
        _queries_from_evidence,
        _read_json,
        _role_contract_recomputes,
        _state_probe,
        _windows_from_evidence,
    )
except ModuleNotFoundError:
    from prototype_peira_assessor import (
        _action_sanity_from_predictions,
        _attribution_scores_from_predictions,
        _canonical_json,
        _downstream_pair_errors,
        _effective_rank,
        _file_sha256,
        _forecast_scores,
        _max_abs,
        _queries_from_evidence,
        _read_json,
        _role_contract_recomputes,
        _state_probe,
        _windows_from_evidence,
    )
from quantis_core.action_conditioned_dynamics import (
    ActionConditionedWindows,
)
from quantis_core.edge_dynamics.complete_lejepa import (
    CompleteLejepaRepresentation,
    EntityPcaRepresentation,
    PairBlockedAnchorSchedule,
    ReducedRankActionProbe,
    TelemetryViewSchedule,
    fit_owned_feature_mask,
)
from quantis_core.edge_dynamics.visreg import (
    VisregConfig,
    VisregDirectionSchedule,
    VisregRepresentation,
    assess_visreg_gates,
)
from quantis_core.graph_telemetry import DeclaredTelemetryGraph


VISREG_NAMES = ("detached_visreg", "no_detach_visreg")
PRIOR_NAMES = (
    "complete_lejepa",
    "invariance_only",
    "sigreg_only",
    "masked_autoencoder",
)
REPRESENTATION_NAMES = VISREG_NAMES + PRIOR_NAMES + ("matched_pca",)
EVALUATED_ROLES = (
    "selection",
    "iid_evaluation",
    "transfer_evaluation",
)
FROZEN_SOURCE_CORPUS_SHA256 = (
    "df03af282f48591216251f22f934b97e1555147df9ed6f6c6d1f7684f9644a26"
)
FROZEN_SOURCE_ARTIFACT_MANIFEST_SHA256 = (
    "d02afa33a5977cba69b255cd1a5f31470751b6681da1ecae31b11e86307e65b1"
)
FROZEN_PRIOR_MANIFEST_SHA256 = (
    "00639afaee81cd3844e8b60014ed87f886a8e7a7e0b20bc50834416da1deb265"
)
FROZEN_PRIOR_MODEL_SHA256 = {
    "complete_lejepa": (
        "eda9795582f2965ba1091b1dca710bc74ce2098bbc747ddfc0de3a324e39e412"
    ),
    "invariance_only": (
        "cbadbda2c8e4f0357ef135224b827a0d75e7a06f84821dc487df2f995fba4723"
    ),
    "sigreg_only": (
        "3559d948fe0801f1b2a0d816f50e6c0269a9a6209a72fe63eec8ac88e450745e"
    ),
    "masked_autoencoder": (
        "4149452dfdf18c7abe651f5cb77788737c7ef9b4bfbbeb3cd14ef69e82f9bad4"
    ),
}
FROZEN_PREPROCESSING_PROTOCOL = (
    "action_conditioned_jepa_topology_transfer_v1"
)


@dataclass(frozen=True)
class _LiteralVisregLoss:
    regularization: Any
    scale: Any
    shape: Any
    center: Any
    quantiles: Any
    sorted_projections: Any


def _literal_visreg_loss(
    embeddings: Any,
    directions: Any,
    *,
    detach_shape: bool,
) -> _LiteralVisregLoss:
    """Independent literal implementation of the frozen equations."""

    import math
    import torch

    means = embeddings.mean(dim=1)
    centered = embeddings - means[:, None]
    standard_deviations = (
        torch.linalg.vector_norm(centered, dim=1)
        / math.sqrt(float(embeddings.shape[1]))
    ).clamp_min(1e-6)
    denominator = (
        standard_deviations.detach()
        if detach_shape
        else standard_deviations
    )
    normalized = centered / denominator[:, None]
    sorted_projections = torch.sort(
        normalized @ directions, dim=1
    ).values
    indices = torch.arange(
        1,
        embeddings.shape[1] + 1,
        dtype=torch.float32,
        device="cpu",
    )
    quantiles = math.sqrt(2.0) * torch.erfinv(
        2.0 * indices / float(embeddings.shape[1] + 1) - 1.0
    )
    shape = (
        sorted_projections - quantiles[None, :, None]
    ).square().mean()
    scale = (standard_deviations - 1.0).square().mean()
    center = means.square().mean()
    return _LiteralVisregLoss(
        regularization=scale + shape + center,
        scale=scale,
        shape=shape,
        center=center,
        quantiles=quantiles,
        sorted_projections=sorted_projections,
    )


def fixed_directions(
    *, width: int, projection_count: int, seed: int
) -> np.ndarray:
    """Return one explicit normalized CPU float32 direction draw."""

    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    raw = torch.randn(
        (width, projection_count),
        generator=generator,
        dtype=torch.float32,
    )
    return (
        raw / torch.linalg.vector_norm(raw, dim=0, keepdim=True)
    ).numpy()


def visreg_diagnostics(
    views: np.ndarray,
    backbone_tokens: np.ndarray,
    ownership: np.ndarray,
    varying_entities: np.ndarray,
    directions: np.ndarray,
) -> Mapping[str, Any]:
    """Recompute fixed-direction shape, moments, covariance, and rank."""

    import torch

    values = np.asarray(views, dtype=np.float32)
    tokens = np.asarray(backbone_tokens, dtype=np.float64)
    owned = np.asarray(ownership, dtype=np.bool_)
    varying = np.asarray(varying_entities, dtype=np.bool_)
    matrix = np.asarray(directions, dtype=np.float32)
    if (
        values.ndim != 3
        or values.shape[0] != 8
        or tokens.ndim != 3
        or tokens.shape[1] != len(owned)
        or varying.shape != (tokens.shape[1],)
        or matrix.shape[0] != values.shape[2]
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(tokens))
        or not np.all(np.isfinite(matrix))
    ):
        raise ValueError("VISReg diagnostic arrays do not align")
    result = _literal_visreg_loss(
        torch.as_tensor(values, dtype=torch.float32),
        torch.as_tensor(matrix, dtype=torch.float32),
        detach_shape=True,
    )
    values64 = values.astype(np.float64)
    means = np.mean(values64, axis=1)
    centered = values64 - means[:, None]
    standard_deviations = np.maximum(
        np.linalg.norm(centered, axis=1)
        / np.sqrt(float(values.shape[1])),
        1e-6,
    )
    covariance = np.stack(
        [
            centered[view].T
            @ centered[view]
            / float(values.shape[1])
            for view in range(values.shape[0])
        ]
    )
    per_entity_variance = [
        float(np.mean(np.var(tokens[:, entity], axis=0)))
        for entity in range(tokens.shape[1])
    ]
    return {
        "fixed_shape_loss": float(result.shape.detach()),
        "fixed_scale_loss": float(result.scale.detach()),
        "fixed_center_loss": float(result.center.detach()),
        "means": means.tolist(),
        "population_standard_deviations": (
            standard_deviations.tolist()
        ),
        "population_covariance": covariance.tolist(),
        "projector_effective_rank": _effective_rank(
            values64.reshape(-1, values.shape[-1])
        ),
        "backbone_effective_rank": _effective_rank(
            tokens.reshape(len(tokens), -1)
        ),
        "per_entity_variance": per_entity_variance,
        "varying_entity_variance_min": float(
            min(
                value
                for entity, value in enumerate(per_entity_variance)
                if varying[entity]
            )
        ),
        "direction_norm_max_error": float(
            np.max(np.abs(np.linalg.norm(matrix, axis=0) - 1.0))
        ),
    }


def collapse_curve(
    base: np.ndarray, directions: np.ndarray
) -> Mapping[str, Any]:
    """Recompute the frozen radial VISReg/SIGReg gradient comparison."""

    import torch

    source = torch.as_tensor(
        np.asarray(base, dtype=np.float32), dtype=torch.float32
    )
    matrix = torch.as_tensor(
        np.asarray(directions, dtype=np.float32), dtype=torch.float32
    )
    rows = []
    for radius in (1.0, 0.1, 0.01, 0.001, 0.0001):
        detached_input = (
            radius * source
        ).clone().detach().requires_grad_(True)
        detached = _literal_visreg_loss(
            detached_input, matrix, detach_shape=True
        )
        detached_regularization_gradient = torch.autograd.grad(
            detached.regularization,
            detached_input,
            retain_graph=True,
        )[0]
        detached_shape_gradient = torch.autograd.grad(
            detached.shape, detached_input
        )[0]

        attached_input = (
            radius * source
        ).clone().detach().requires_grad_(True)
        attached = _literal_visreg_loss(
            attached_input, matrix, detach_shape=False
        )
        attached_shape_gradient = torch.autograd.grad(
            attached.shape, attached_input
        )[0]

        sigreg_input = (
            radius * source
        ).clone().detach().requires_grad_(True)
        sigreg = _sigreg_with_directions(sigreg_input, matrix)
        sigreg_gradient = torch.autograd.grad(
            sigreg, sigreg_input
        )[0]
        rows.append(
            {
                "radius": radius,
                "detached_regularization_gradient": (
                    _mean_row_gradient(
                        detached_regularization_gradient
                    )
                ),
                "sigreg_gradient": _mean_row_gradient(sigreg_gradient),
                "detached_shape_gradient": _mean_row_gradient(
                    detached_shape_gradient
                ),
                "no_detach_shape_gradient": _mean_row_gradient(
                    attached_shape_gradient
                ),
                "detached_regularization": float(
                    detached.regularization.detach()
                ),
                "sigreg": float(sigreg.detach()),
            }
        )
    return {"rows": rows}


def assess_stored_bundle(directory: Path) -> Mapping[str, Any]:
    """Recompute every frozen VISReg gate from retained evidence."""

    root = Path(directory)
    metadata = _read_json(root / "evidence-metadata.json")
    if (
        metadata.get("schema_version") != 1
        or metadata.get("kind") != "visreg_assessment_evidence_v1"
    ):
        raise ValueError("unsupported VISReg evidence")
    graph = DeclaredTelemetryGraph.from_dict(dict(metadata["graph"]))
    with np.load(root / "evidence.npz", allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    finite = all(
        np.all(np.isfinite(value))
        for value in arrays.values()
        if value.dtype != np.bool_
    )
    windows = {
        role: _windows_from_evidence(role, metadata, arrays, graph)
        for role in ("fit",) + EVALUATED_ROLES
    }
    ownership = np.asarray(metadata["ownership_mask"], dtype=np.bool_)
    declared_varying = np.asarray(
        metadata["varying_entity_mask"], dtype=np.bool_
    )
    varying = np.any(
        (
            np.ptp(windows["fit"].histories, axis=(0, 1)) > 1e-9
        )
        & ownership,
        axis=1,
    )
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
    queries = _queries_from_evidence(metadata, arrays)
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
    diagnostics = {
        name: {
            role: visreg_diagnostics(
                arrays[f"diagnostic_views__{name}__{role}"],
                arrays[f"representation__{name}__{role}"],
                ownership,
                varying,
                arrays["fixed_diagnostic_directions"],
            )
            for role in ("selection", "transfer_evaluation")
        }
        for name in VISREG_NAMES
    }
    curve = collapse_curve(
        arrays["collapse_base"], arrays["collapse_directions"]
    )
    mechanism_gates = _mechanism_gates(diagnostics, curve)
    restoration_max_abs, bundle_replay = _replay_models(
        root, windows, arrays
    )
    transfer_pair_errors = {
        name: _downstream_pair_errors(
            arrays[f"prediction__{name}__transfer_evaluation"],
            windows["transfer_evaluation"],
        )
        for name in REPRESENTATION_NAMES
    }
    parameter_counts = _recompute_parameter_counts(root)
    selection_ok, selection_safety_ok = _selection_recomputes(
        metadata, arrays, windows, raw_scores
    )
    base_config = VisregConfig.from_dict(
        dict(dict(metadata["configs"])["detached_visreg"])
    )
    schedule_ok = _schedules_recompute(
        root, windows["fit"], base_config, arrays
    )
    objective_ok = _objectives_recompute(metadata, arrays)
    mode_ok = _mode_enforcement_recomputes(
        root, metadata, arrays, windows["fit"]
    )
    diagnostic_rng_ok = _diagnostic_rng_recomputes(arrays)
    latency_samples = np.asarray(
        arrays["latency_samples_ms"], dtype=np.float64
    )
    latency = {
        "median_ms": float(np.median(latency_samples)),
        "p95_ms": float(np.quantile(latency_samples, 0.95)),
        "repetitions": int(len(latency_samples)),
    }
    stored_latency = dict(metadata["latency"])
    bundle = (
        root / "models" / "detached_visreg-inference.json.gz"
    )
    deployed_bundle_bytes = int(bundle.stat().st_size)
    stored_state_probes = dict(metadata["state_probes"])
    state_probe_ok = (
        _canonical_json(state_probes)
        == _canonical_json(stored_state_probes)
    )
    diagnostic_ok = bool(
        _canonical_json(diagnostics)
        == _canonical_json(dict(metadata["diagnostics"]))
        and _canonical_json(curve)
        == _canonical_json(dict(metadata["collapse_curve"]))
    )
    protocol_checks = {
        "evidence_arrays_are_finite": finite,
        "role_contract_recomputes": _role_contract_recomputes(metadata),
        "all_schedules_recompute": bool(
            schedule_ok
            and diagnostic_rng_ok
            and np.array_equal(declared_varying, varying)
        ),
        "objective_recomputes": bool(objective_ok and diagnostic_ok),
        "mode_enforcement_recomputes": mode_ok,
        "capacity_recomputes": parameter_counts
        == {
            str(name): {
                str(key): int(value)
                for key, value in dict(raw).items()
            }
            for name, raw in dict(
                metadata["parameter_counts"]
            ).items()
        },
        "public_inference_is_causal": _public_inference_is_causal(root),
        "copied_source_assessor_recomputes": (
            _copied_source_assessor_receipt_recomputes(root, metadata)
        ),
        "copied_prior_controls_match": (
            _copied_prior_controls_recompute(root, metadata)
        ),
        "selection_only_ridge_choice_recomputes": selection_ok,
        "selection_safety_status_recomputes": selection_safety_ok,
        "bundle_size_recomputes": deployed_bundle_bytes
        == int(metadata["deployed_bundle_bytes"]),
        "latency_recomputes": bool(
            int(stored_latency["repetitions"]) == len(latency_samples)
            and np.isclose(
                float(stored_latency["median_ms"]),
                latency["median_ms"],
                rtol=0.0,
                atol=1e-12,
            )
            and np.isclose(
                float(stored_latency["p95_ms"]),
                latency["p95_ms"],
                rtol=0.0,
                atol=1e-12,
            )
        ),
        "state_probe_recomputes": state_probe_ok,
    }
    varying_ids = tuple(
        graph.entity_ids[index]
        for index in np.flatnonzero(varying)
    )
    state_probe_gate = {
        name: {
            "aggregate_nrmse": state_probes[name][
                "transfer_evaluation"
            ]["aggregate_nrmse"],
            "entity_nrmse": {
                entity_id: state_probes[name][
                    "transfer_evaluation"
                ]["entities"][entity_id]["nrmse"]
                for entity_id in varying_ids
            },
        }
        for name in ("detached_visreg", "matched_pca")
    }
    assessment = dict(
        assess_visreg_gates(
            forecast_scores=forecast_scores,
            raw_scores=raw_scores,
            mechanism_gates=mechanism_gates,
            attribution=attribution,
            action_sanity=action_sanity,
            restoration_max_abs={
                **restoration_max_abs,
                "deployment_bundle": bundle_replay,
            },
            protocol_checks=protocol_checks,
            parameter_counts=parameter_counts,
            transfer_pair_errors=transfer_pair_errors,
            state_probe=state_probe_gate,
            varying_entity_ids=varying_ids,
            deployed_bundle_bytes=deployed_bundle_bytes,
            median_latency_ms=latency["median_ms"],
        )
    )
    configs_frozen = all(
        dict(dict(metadata["configs"])[name])
        == VisregConfig(objective=name).to_dict()
        for name in VISREG_NAMES
    )
    interpretable = bool(
        metadata.get("interpretable") is True
        and metadata.get("source_corpus_sha256")
        == FROZEN_SOURCE_CORPUS_SHA256
        and metadata.get("source_artifact_manifest_sha256")
        == FROZEN_SOURCE_ARTIFACT_MANIFEST_SHA256
        and metadata.get("prior_control_manifest_sha256")
        == FROZEN_PRIOR_MANIFEST_SHA256
        and metadata.get("preprocessing_protocol")
        == FROZEN_PREPROCESSING_PROTOCOL
        and configs_frozen
        and len(latency_samples) == 100
    )
    assessment.update(
        {
            "protocol_checks": protocol_checks,
            "forecast_scores": forecast_scores,
            "raw_scores": raw_scores,
            "state_probes": state_probes,
            "diagnostics": diagnostics,
            "collapse_curve": curve,
            "attribution": attribution,
            "action_sanity": action_sanity,
            "restoration_max_abs": restoration_max_abs,
            "parameter_counts": parameter_counts,
            "deployed_bundle_bytes": deployed_bundle_bytes,
            "latency": latency,
            "eligible_for_advance": interpretable,
        }
    )
    if not interpretable:
        assessment["provisional_decision"] = assessment["decision"]
        assessment["decision"] = "non_interpretable_visreg_smoke"
        assessment["passed"] = False
    return assessment


def verify_stored_assessment(directory: Path) -> Mapping[str, Any]:
    """Verify manifest identity and exact copied-source reassessment."""

    root = Path(directory)
    manifest = _read_json(root / "artifact-manifest.json")
    expected = {
        str(path.relative_to(root)): _file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.name != "artifact-manifest.json"
    }
    if dict(manifest["sha256"]) != expected:
        raise ValueError("VISReg artifact manifest differs")
    assessment = dict(assess_stored_bundle(root))
    stored = _read_json(root / "assessment.json")
    result = _read_json(root / "result.json")
    runtime = dict(result.get("runtime", {}))
    if (
        _canonical_json(assessment) != _canonical_json(stored)
        or _canonical_json(assessment)
        != _canonical_json(dict(result["assessment"]))
        or not isinstance(runtime.get("torch"), str)
        or not runtime["torch"]
    ):
        raise ValueError("VISReg stored assessment differs")
    copied = _run_copied_assessor(root)
    if _canonical_json(copied) != _canonical_json(assessment):
        raise ValueError("copied VISReg assessor differs")
    return assessment


def _mechanism_gates(
    diagnostics: Mapping[str, Mapping[str, Mapping[str, Any]]],
    curve: Mapping[str, Any],
) -> Mapping[str, bool]:
    candidate = diagnostics["detached_visreg"]
    small = [
        dict(row)
        for row in curve["rows"]
        if float(dict(row)["radius"]) <= 0.01
    ]
    return {
        "exact_math_and_rng": all(
            float(
                diagnostics[name][role][
                    "direction_norm_max_error"
                ]
            )
            <= 1e-6
            for name in VISREG_NAMES
            for role in ("selection", "transfer_evaluation")
        ),
        "candidate_noncollapsed": all(
            float(candidate[role]["projector_effective_rank"]) >= 8.0
            and float(
                candidate[role]["varying_entity_variance_min"]
            )
            > 0.0
            for role in ("selection", "transfer_evaluation")
        ),
        "collapse_gradient_beats_sigreg": all(
            np.isfinite(
                float(row["detached_regularization_gradient"])
            )
            and float(row["detached_regularization_gradient"])
            > float(row["sigreg_gradient"])
            for row in small
        ),
        "detached_shape_beats_no_detach": all(
            np.isfinite(float(row["detached_shape_gradient"]))
            and float(row["detached_shape_gradient"])
            > float(row["no_detach_shape_gradient"])
            for row in small
        ),
    }


def _objectives_recompute(
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> bool:
    import torch

    directions = arrays["training_directions"]
    for name in VISREG_NAMES:
        config = VisregConfig.from_dict(
            dict(dict(metadata["configs"])[name])
        )
        detach = name == "detached_visreg"
        for step in range(config.steps):
            embeddings = torch.as_tensor(
                arrays[f"training__{name}__embeddings"][step],
                dtype=torch.float32,
            ).requires_grad_(step == 0)
            matrix = torch.as_tensor(
                directions[step], dtype=torch.float32
            )
            result = _literal_visreg_loss(
                embeddings, matrix, detach_shape=detach
            )
            invariance = (
                embeddings - embeddings[:2].mean(dim=0)[None]
            ).square().mean()
            total = (
                config.prediction_weight * invariance
                + config.regularization_weight
                * result.regularization
            )
            expected = {
                "loss": total,
                "invariance": invariance,
                "regularization": result.regularization,
                "scale": result.scale,
                "shape": result.shape,
                "center": result.center,
            }
            quantiles = (
                result.quantiles.detach().numpy().astype(np.float32)
            )
            sorted_projections = (
                result.sorted_projections.detach()
                .numpy()
                .astype(np.float32)
            )
            if any(
                not np.isclose(
                    float(value.detach()),
                    float(
                        arrays[f"training__{name}__{field}"][step]
                    ),
                    rtol=1e-6,
                    atol=1e-6,
                )
                for field, value in expected.items()
            ):
                return False
            if not (
                np.array_equal(
                    quantiles,
                    arrays[
                        f"training__{name}__gaussian_quantiles"
                    ],
                )
                and np.array_equal(
                    np.frombuffer(
                        hashlib.sha256(
                            quantiles.tobytes(order="C")
                        ).digest(),
                        dtype=np.uint8,
                    ),
                    arrays[
                        f"training__{name}__gaussian_quantile_sha256"
                    ],
                )
                and np.array_equal(
                    np.frombuffer(
                        hashlib.sha256(
                            sorted_projections.tobytes(order="C")
                        ).digest(),
                        dtype=np.uint8,
                    ),
                    arrays[
                        f"training__{name}__sorted_projection_sha256"
                    ][step],
                )
                and np.isclose(
                    np.mean(
                        sorted_projections, dtype=np.float64
                    ),
                    arrays[
                        f"training__{name}__sorted_projection_mean"
                    ][step],
                    rtol=0.0,
                    atol=0.0,
                )
                and np.isclose(
                    np.std(sorted_projections, dtype=np.float64),
                    arrays[
                        f"training__{name}__sorted_projection_std"
                    ][step],
                    rtol=0.0,
                    atol=0.0,
                )
                and float(np.min(sorted_projections))
                == float(
                    arrays[
                        f"training__{name}__sorted_projection_min"
                    ][step]
                )
                and float(np.max(sorted_projections))
                == float(
                    arrays[
                        f"training__{name}__sorted_projection_max"
                    ][step]
                )
            ):
                return False
            if step == 0:
                gradient = torch.autograd.grad(
                    result.regularization, embeddings
                )[0].detach().numpy()
                if not np.allclose(
                    gradient,
                    arrays[
                        f"training__{name}__regularizer_gradient_step0"
                    ],
                    rtol=1e-6,
                    atol=1e-6,
                ):
                    return False
    return True


def _schedules_recompute(
    root: Path,
    fit: ActionConditionedWindows,
    config: VisregConfig,
    arrays: Mapping[str, np.ndarray],
) -> bool:
    with np.load(root / "schedule.npz", allow_pickle=False) as stored:
        values = {name: stored[name] for name in stored.files}
    anchors = PairBlockedAnchorSchedule(fit, seed=config.anchor_seed)
    ownership = fit_owned_feature_mask(fit)
    varying = np.any(
        (np.ptp(fit.histories, axis=(0, 1)) > 1e-9) & ownership,
        axis=1,
    )
    views = TelemetryViewSchedule(
        graph=fit.graph,
        ownership_mask=ownership,
        varying_entity_mask=varying,
        seed=config.view_seed,
    )
    batches = [anchors.batch(step) for step in range(config.steps)]
    view_batches = [
        views.batch(fit.histories[:1], step=step)
        for step in range(config.steps)
    ]
    direction_schedule = VisregDirectionSchedule(
        width=config.width,
        projection_count=config.projection_count,
        seed=config.direction_seed,
    )
    directions = []
    hashes = []
    for _ in range(config.steps):
        matrix = direction_schedule.draw().numpy()
        directions.append(matrix)
        hashes.append(
            np.frombuffer(
                hashlib.sha256(matrix.tobytes(order="C")).digest(),
                dtype=np.uint8,
            )
        )
    directions_array = np.stack(directions)
    return bool(
        np.array_equal(
            values["anchor_indices"],
            np.stack([batch.indices for batch in batches]),
        )
        and np.array_equal(
            values["anchor_arm_ids"],
            np.stack([batch.arm_ids for batch in batches]),
        )
        and np.array_equal(
            values["anchor_transitions"],
            np.stack([batch.transition_indices for batch in batches]),
        )
        and np.array_equal(
            values["view_visible"],
            np.stack(
                [batch.visible_tokens[:, 0] for batch in view_batches]
            ),
        )
        and np.array_equal(
            values["view_present"],
            np.stack(
                [batch.present_tokens[:, 0] for batch in view_batches]
            ),
        )
        and np.array_equal(
            values["learning_rate"],
            np.asarray(
                [
                    config.learning_rate_at(step)
                    for step in range(config.steps)
                ]
            ),
        )
        and np.array_equal(
            arrays["training_directions"], directions_array
        )
        and all(
            np.array_equal(
                arrays[f"training__{name}__anchor_indices"],
                values["anchor_indices"],
            )
            and np.array_equal(
                arrays[f"training__{name}__anchor_arm_ids"],
                values["anchor_arm_ids"],
            )
            and np.array_equal(
                arrays[f"training__{name}__anchor_transitions"],
                values["anchor_transitions"],
            )
            and np.array_equal(
                arrays[f"training__{name}__view_visible"],
                values["view_visible"],
            )
            and np.array_equal(
                arrays[f"training__{name}__view_present"],
                values["view_present"],
            )
            and np.array_equal(
                arrays[f"training__{name}__learning_rate"],
                values["learning_rate"],
            )
            and np.array_equal(
                arrays[f"training__{name}__direction_sha256"],
                np.stack(hashes),
            )
            for name in VISREG_NAMES
        )
        and all(
            np.array_equal(
                np.asarray(
                    _read_json(
                        root / "models" / f"{name}.json"
                    )["direction_initial_state"],
                    dtype=np.uint8,
                ),
                direction_schedule.initial_state,
            )
            and np.array_equal(
                np.asarray(
                    _read_json(
                        root / "models" / f"{name}.json"
                    )["direction_final_state"],
                    dtype=np.uint8,
                ),
                direction_schedule.final_state,
            )
            and int(
                _read_json(
                    root / "models" / f"{name}.json"
                )["direction_draw_count"]
            )
            == config.steps
            for name in VISREG_NAMES
        )
    )


def _diagnostic_rng_recomputes(
    arrays: Mapping[str, np.ndarray],
) -> bool:
    """Rebuild every non-training diagnostic draw from its frozen seed."""

    import torch

    generator = torch.Generator(device="cpu").manual_seed(8509)
    base = torch.randn(
        (8, 40, 64), generator=generator, dtype=torch.float32
    )
    base = base / (
        torch.linalg.vector_norm(base, dim=-1, keepdim=True) + 1e-12
    )
    return bool(
        np.array_equal(
            arrays["fixed_diagnostic_directions"],
            fixed_directions(
                width=64, projection_count=1024, seed=6509
            ),
        )
        and np.array_equal(
            arrays["collapse_directions"],
            fixed_directions(
                width=64, projection_count=256, seed=7509
            ),
        )
        and np.array_equal(arrays["collapse_base"], base.numpy())
    )


def _mode_enforcement_recomputes(
    root: Path,
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    fit: ActionConditionedWindows,
) -> bool:
    gradients = [
        arrays[f"training__{name}__regularizer_gradient_step0"]
        for name in VISREG_NAMES
    ]
    gradient_hashes = [
        hashlib.sha256(
            np.asarray(value, dtype=np.float32).tobytes(order="C")
        ).hexdigest()
        for value in gradients
    ]
    models = {
        name: VisregRepresentation.from_dict(
            _read_json(root / "models" / f"{name}.json")
        )
        for name in VISREG_NAMES
    }
    stored = dict(metadata["mode_enforcement"])
    anchor = PairBlockedAnchorSchedule(
        fit, seed=models[VISREG_NAMES[0]].config.anchor_seed
    ).batch(models[VISREG_NAMES[0]].config.steps - 1)
    replay_tokens = {
        name: models[name]
        .encode(fit.histories[anchor.indices], fit.graph)
        .tokens
        for name in VISREG_NAMES
    }
    token_difference = _max_abs(
        replay_tokens["detached_visreg"],
        replay_tokens["no_detach_visreg"],
    )
    return bool(
        np.array_equal(
            arrays["training__detached_visreg__embeddings"][0],
            arrays["training__no_detach_visreg__embeddings"][0],
        )
        and gradient_hashes[0] != gradient_hashes[1]
        and _max_abs(gradients[0], gradients[1]) > 1e-7
        and models[VISREG_NAMES[0]].network_sha256
        != models[VISREG_NAMES[1]].network_sha256
        and models[VISREG_NAMES[0]].projector_sha256
        != models[VISREG_NAMES[1]].projector_sha256
        and all(
            models[name].config.to_dict()
            == dict(dict(metadata["configs"])[name])
            for name in VISREG_NAMES
        )
        and all(
            _max_abs(
                replay_tokens[name],
                arrays[f"mode_public_tokens__{name}"],
            )
            <= 1e-6
            for name in VISREG_NAMES
        )
        and token_difference > 1e-6
        and stored.get("network_sha256")
        == [models[name].network_sha256 for name in VISREG_NAMES]
        and stored.get("projector_sha256")
        == [models[name].projector_sha256 for name in VISREG_NAMES]
        and stored.get("gradient_sha256") == gradient_hashes
        and np.isclose(
            float(stored["gradient_max_abs"]),
            _max_abs(gradients[0], gradients[1]),
            rtol=0.0,
            atol=0.0,
        )
        and np.isclose(
            float(stored["public_token_max_abs"]),
            token_difference,
            rtol=0.0,
            atol=0.0,
        )
    )


def _replay_models(
    root: Path,
    windows: Mapping[str, ActionConditionedWindows],
    arrays: Mapping[str, np.ndarray],
) -> Tuple[Mapping[str, float], float]:
    models: Dict[str, Any] = {
        name: VisregRepresentation.from_dict(
            _read_json(root / "models" / f"{name}.json")
        )
        for name in VISREG_NAMES
    }
    models.update(
        {
            name: CompleteLejepaRepresentation.from_dict(
                _read_json(root / "models" / f"{name}.json")
            )
            for name in PRIOR_NAMES
        }
    )
    pca = EntityPcaRepresentation.from_dict(
        _read_json(root / "models" / "matched_pca.json")
    )
    maxima = {}
    for name in REPRESENTATION_NAMES:
        probe = ReducedRankActionProbe.from_dict(
            _read_json(root / "models" / f"{name}-probe.json")
        )
        values = []
        for role in EVALUATED_ROLES:
            role_windows = windows[role]
            call = pca.encode if name == "matched_pca" else models[name].encode
            replay = _encode_chunks(
                call, role_windows.histories, role_windows.graph
            )
            replay_probe = probe.predict(
                replay,
                role_windows.future_controls,
                role_windows.future_actions,
            )
            values.extend(
                (
                    _max_abs(
                        arrays[
                            f"restoration_original_tokens__{name}__{role}"
                        ],
                        arrays[
                            f"restoration_restored_tokens__{name}__{role}"
                        ],
                    ),
                    _max_abs(
                        replay,
                        arrays[
                            f"restoration_restored_tokens__{name}__{role}"
                        ],
                    ),
                    _max_abs(
                        arrays[
                            f"restoration_original_probe__{name}__{role}"
                        ],
                        arrays[
                            f"restoration_restored_probe__{name}__{role}"
                        ],
                    ),
                    _max_abs(
                        replay_probe,
                        arrays[
                            f"restoration_restored_probe__{name}__{role}"
                        ],
                    ),
                    _max_abs(
                        replay,
                        arrays[
                            f"representation__{name}__{role}"
                        ],
                    ),
                    _max_abs(
                        replay_probe,
                        arrays[f"prediction__{name}__{role}"],
                    ),
                )
            )
        if name in VISREG_NAMES:
            for role in ("selection", "transfer_evaluation"):
                role_windows = windows[role]
                anchor = PairBlockedAnchorSchedule(
                    role_windows,
                    seed=models[name].config.anchor_seed,
                ).batch(models[name].config.steps - 1)
                diagnostic_replay = models[name].diagnose_views(
                    role_windows.histories[anchor.indices],
                    role_windows.graph,
                    step=models[name].config.steps - 1,
                )
                values.append(
                    _max_abs(
                        diagnostic_replay,
                        arrays[
                            f"diagnostic_views__{name}__{role}"
                        ],
                    )
                )
        maxima[name] = max(values)
    payload = dict(
        json.loads(
            gzip.decompress(
                (
                    root
                    / "models"
                    / "detached_visreg-inference.json.gz"
                ).read_bytes()
            ).decode()
        )
    )
    if (
        set(payload)
        != {"schema_version", "kind", "representation", "probe"}
        or payload.get("schema_version") != 1
        or payload.get("kind")
        != "visreg_forecast_inference_bundle_v1"
    ):
        raise ValueError("unsupported VISReg deployment bundle")
    deployed_model = VisregRepresentation.from_inference_dict(
        dict(payload["representation"])
    )
    deployed_probe = ReducedRankActionProbe.from_dict(
        dict(payload["probe"])
    )
    transfer = windows["transfer_evaluation"]
    deployed_tokens = _encode_chunks(
        deployed_model.encode, transfer.histories, transfer.graph
    )
    deployed_prediction = deployed_probe.predict(
        deployed_tokens,
        transfer.future_controls,
        transfer.future_actions,
    )
    bundle_replay = max(
        _max_abs(
            deployed_tokens,
            arrays["deployment_reference_tokens"],
        ),
        _max_abs(
            deployed_prediction,
            arrays["deployment_reference_prediction"],
        ),
    )
    return maxima, bundle_replay


def _recompute_parameter_counts(
    root: Path,
) -> Mapping[str, Mapping[str, int]]:
    result = {}
    for name in VISREG_NAMES:
        model = VisregRepresentation.from_dict(
            _read_json(root / "models" / f"{name}.json")
        )
        result[name] = {
            "training": model.training_parameter_count,
            "inference": model.inference_parameter_count,
        }
    return result


def _selection_recomputes(
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    windows: Mapping[str, ActionConditionedWindows],
    raw_scores: Mapping[str, Mapping[str, float]],
) -> Tuple[bool, bool]:
    selected = dict(metadata["selected_ridges"])
    failed = dict(metadata["selection_safety_failed"])
    ridges = [float(value) for value in metadata["ridge_values"]]
    chosen_ok = True
    failed_ok = True
    for name in REPRESENTATION_NAMES:
        rows = []
        for position, ridge in enumerate(ridges):
            scores = _forecast_scores(
                arrays[f"ridge_prediction__{name}__{position}"],
                windows["selection"],
            )
            rows.append(
                {
                    "ridge": ridge,
                    "raw_safe": (
                        scores["overall_mse"]
                        <= 1.05 * raw_scores["selection"]["overall_mse"]
                        and scores["action_overlap_mse"]
                        <= 1.05
                        * raw_scores["selection"]["action_overlap_mse"]
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
        chosen_ok &= float(selected[name]) == float(chosen["ridge"])
        failed_ok &= bool(failed[name]) == (not bool(eligible))
    return bool(chosen_ok), bool(failed_ok)


def _copied_prior_controls_recompute(
    root: Path, metadata: Mapping[str, Any]
) -> bool:
    retained = root / "prior-control-manifest.json"
    if (
        not retained.is_file()
        or _file_sha256(retained) != FROZEN_PRIOR_MANIFEST_SHA256
    ):
        return False
    manifest = _read_json(retained)
    declared = dict(manifest.get("sha256", {}))
    metadata_hashes = dict(metadata.get("prior_model_sha256", {}))
    return all(
        declared.get(f"models/{name}.json")
        == FROZEN_PRIOR_MODEL_SHA256[name]
        and metadata_hashes.get(name) == FROZEN_PRIOR_MODEL_SHA256[name]
        and _file_sha256(root / "models" / f"{name}.json")
        == FROZEN_PRIOR_MODEL_SHA256[name]
        for name in PRIOR_NAMES
    )


def _copied_source_assessor_receipt_recomputes(
    root: Path, metadata: Mapping[str, Any]
) -> bool:
    path = root / "isolated-assessor-receipt.json"
    if not path.is_file():
        return False
    receipt = _read_json(path)
    hashes = {
        str(name): str(value)
        for name, value in dict(
            metadata.get("source_sha256", {})
        ).items()
    }
    snapshot = hashlib.sha256(
        _canonical_json(hashes).encode()
    ).hexdigest()
    assessor = "lab/action_dynamics/prototype_visreg_assessor.py"
    return bool(
        hashes
        and receipt.get("schema_version") == 1
        and receipt.get("kind")
        == "visreg_isolated_assessor_receipt_v1"
        and receipt.get("returncode") == 0
        and receipt.get("assessor_sha256") == hashes.get(assessor)
        and receipt.get("source_snapshot_sha256") == snapshot
        and all(
            _file_sha256(root / "reproduction-source" / name)
            == expected
            for name, expected in hashes.items()
        )
    )


def _public_inference_is_causal(root: Path) -> bool:
    path = (
        root
        / "reproduction-source/src/quantis_core/edge_dynamics/visreg.py"
    )
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ClassDef)
            and node.name == "VisregRepresentation"
        ):
            for item in node.body:
                if (
                    isinstance(item, ast.FunctionDef)
                    and item.name == "encode"
                ):
                    return [
                        argument.arg for argument in item.args.args
                    ] == ["self", "histories", "graph"]
    return False


def _sigreg_with_directions(
    embeddings: Any, directions: Any
) -> Any:
    import torch

    knots = torch.linspace(
        0.0, 3.0, 17, dtype=embeddings.dtype
    )
    delta = 3.0 / 16.0
    quadrature = torch.full(
        (17,), 2.0 * delta, dtype=embeddings.dtype
    )
    quadrature[[0, -1]] = delta
    gaussian = torch.exp(-torch.square(knots) / 2.0)
    projected = (embeddings @ directions).unsqueeze(-1) * knots
    error = torch.square(
        projected.cos().mean(dim=-3) - gaussian
    ) + torch.square(projected.sin().mean(dim=-3))
    return ((error @ (quadrature * gaussian)) * embeddings.size(-2)).mean()


def _mean_row_gradient(gradient: Any) -> float:
    import torch

    return float(
        torch.linalg.vector_norm(
            gradient.reshape(-1, gradient.shape[-1]), dim=1
        )
        .mean()
        .detach()
    )


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


def _run_copied_assessor(directory: Path) -> Mapping[str, Any]:
    root = Path(directory).resolve()
    reproduction = root / "reproduction-source"
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
                / "lab/action_dynamics/prototype_visreg_assessor.py"
            ),
            "--assessment-only",
            str(root),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=root.parent,
    )
    value = json.loads(result.stdout)
    if not isinstance(value, dict):
        raise ValueError("copied VISReg assessor returned invalid output")
    return value


def main(arguments: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--assessment-only", action="store_true"
    )
    parser.add_argument("directory", type=Path)
    options = parser.parse_args(arguments)
    assessment = (
        assess_stored_bundle(options.directory)
        if options.assessment_only
        else verify_stored_assessment(options.directory)
    )
    print(_canonical_json(assessment))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
